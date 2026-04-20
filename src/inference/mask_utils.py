"""
BurnSight mask probability map extraction / threshold calibration
"""
import numpy as np
import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from tensorflow.keras import layers, ops
from tensorflow.keras.layers import (
    Masking, Layer, PReLU, Add, Activation, Lambda, Input, Concatenate,
    concatenate, MaxPooling2D, MaxPooling3D, AveragePooling2D,
    GlobalAveragePooling2D, GlobalMaxPooling2D, Conv2DTranspose,
    UpSampling2D, UpSampling3D, TimeDistributed, Dense, Conv3D, Conv2D,
    ConvLSTM2D, Flatten, Reshape, RepeatVector, Multiply,
    BatchNormalization, LayerNormalization, LeakyReLU, ReLU, Dropout,
    UnitNormalization, SpatialDropout2D, SpatialDropout3D
)
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import backend as K
from pathlib import Path
import time, pathlib, json, glob
import cv2
import re
import random
import h5py
import hashlib
import math
import imageio.v2 as imageio
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.applications import VGG16, VGG19, ResNet50
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.initializers import Initializer, HeNormal, GlorotUniform, RandomNormal
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import pairwise_distances_argmin
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.morphology import (
    dilation, remove_small_objects, remove_small_holes,
    closing, opening, square, footprint_rectangle
)
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from joblib import Memory
from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.utils import to_categorical
from multiprocessing import Pool
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mnv2_pi
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess

from src.config import (
    SEED, IMG_SIZE, IMG_H, IMG_W, BATCH, MAX_T, SEQ_LEN, STRIDE, SEQLEN,
    VAL_RATIO, LATENT_DIM, CTX_K, DELTA_MIN, DELTA_MAX, TAU_NCE, TAU_ANTI,
    PATCH_K, TOPK_RATIO, LR_STAGE1, LR_STAGE2, K_MIN, K_MAX, K_VAL,
    NUM_CLASSES, WOUND_IDX, ESCHAR_IDX, HEALED_IDX, EXCLUDE_IDX,
    LESION_IDXS, CHANGE_IDXS, STABLE_IDXS, POLICY_VER
)

from src.utils.image_utils import *

def collect_val_probs_and_gts(
    mask_model,
    seg_val_dataset,
    wound_idx=1,
    preprocess=None,
    assume_prob="auto",
    max_batches=None,

    # --- policy knobs ---
    include_empty_gt=False,
    invert_action="debug",
    invert_eps=1e-12,

    # --- debug knobs ---
    debug_dir="dbg_inverted",
    debug_thr=0.03,
    debug_max_save=50,
    print_every=1,
):
    """
    Returns:
      probs_list: list of (H,W) float32 in [0,1]
      gts_list  : list of (H,W) uint8 in {0,1}
    """
    probs_list, gts_list = [], []
    n_batches, n_saved, global_idx = 0, 0, 0

    for batch in seg_val_dataset:
        meta = None
        img_paths = None
        mask_paths = None

        # --- unpack ---
        if isinstance(batch, (tuple, list)):
            if len(batch) == 2:
                imgs, gts = batch
            elif len(batch) == 3:
                imgs, gts, meta = batch
            elif len(batch) >= 4:
                imgs, gts, img_paths, mask_paths = batch[0], batch[1], batch[2], batch[3]
            else:
                raise ValueError("seg_val_dataset must yield (imgs,gts) or (imgs,gts,meta) or (imgs,gts,img_paths,mask_paths).")
        else:
            raise ValueError("seg_val_dataset must yield a tuple/list.")

        # --- Fix to numpy (Tensor-safe) ---
        imgs_np = imgs.numpy() if tf.is_tensor(imgs) else imgs
        gts_np  = gts.numpy()  if tf.is_tensor(gts)  else gts

        imgs_np = np.asarray(imgs_np)
        gts_np  = np.asarray(gts_np)

        # --- Apply input preprocessing (unify model input scale) ---
        # Preprocessing is the same as that used in support/change:
        # Assumed to return the form/scale directly usable as mask_model input.
        if preprocess is not None:
            imgs_in = preprocess(imgs_np)
        else:
            imgs_in = imgs_np

        # --- model prediction ---
        preds = mask_model.predict(imgs_in, verbose=0)

        # --- Unify probability tensor (apply assume_prob policy) ---
        # Interpret probabilities with the same policy as support/change
        prob_all = get_prob_all_batch(preds, assume=assume_prob)   # (B,H,W,C)
        if tf.is_tensor(prob_all):
            prob_all = prob_all.numpy()
        prob_all = np.asarray(prob_all, dtype=np.float32)

        pw_b = prob_all[..., int(wound_idx)]    # (B,H,W)

        # --- Extract GT wound binary ---
        gt_b = _extract_gt_wound_binary(gts_np, wound_idx=wound_idx)  # Expected shape: (B,H,W)
        gt_b = gt_b.numpy() if tf.is_tensor(gt_b) else gt_b
        gt_b = np.asarray(gt_b)

        # --- Debug image (01 scale) ---
        # save_seg_debug_panel expects [0,1] images,
        # Here, 'original imgs_np' is adjusted to [0,1] for saving purposes.
        imgs01_np = imgs_np.astype(np.float32)
        # Correct based on range since original may be in [-1,1]
        if imgs01_np.min() < -0.01 or imgs01_np.max() > 1.01:
            imgs01_np = (imgs01_np + 1.0) / 2.0
        imgs01_np = np.clip(imgs01_np, 0.0, 1.0)

        B = pw_b.shape[0]

        for i in range(B):
            pwi = pw_b[i].astype(np.float32)
            gti = gt_b[i].astype(np.uint8)

            pos_n = int((gti == 1).sum())

            # ---- policy: handle empty GT ----
            if pos_n == 0 and (not include_empty_gt):
                global_idx += 1
                continue

            # ---- sample id ----
            sample_id = None
            if meta is not None:
                try:
                    m = meta[i]
                    if isinstance(m, (bytes, np.bytes_)):
                        m = m.decode("utf-8", errors="ignore")
                    sample_id = str(m)
                except Exception:
                    sample_id = None
            if sample_id is None:
                if img_paths is not None:
                    sample_id = Path(_to_str(img_paths[i])).stem
                else:
                    sample_id = f"b{n_batches:03d}_i{i:02d}_g{global_idx:05d}"

            # ---- inverted check ----
            gt_mask = (gti == 1)
            if gt_mask.sum() == 0:
                inverted = False
            else:
                mean_pos = float(pwi[gt_mask].mean())
                mean_neg = float(pwi[~gt_mask].mean())
                inverted = (mean_pos < mean_neg - invert_eps)

            # ---- policy: handle inverted case ----
            if inverted and invert_action in ("debug", "skip"):
                print("🚨 inverted separation: GT1 mean < GT0 mean:", sample_id)

                if (img_paths is not None) and (mask_paths is not None):
                    log_inverted(sample_id, _to_str(img_paths[i]), _to_str(mask_paths[i]))

                if invert_action == "debug" and (n_saved < debug_max_save):
                    img_np = np.clip(imgs01_np[i], 0.0, 1.0).astype(np.float32)
                    saved_path = save_seg_debug_panel(
                        out_dir=debug_dir,
                        sample_id=sample_id,
                        img01=img_np,
                        gt_w=gti,
                        pw=pwi,
                        thr=debug_thr
                    )
                    print("  saved:", saved_path)
                    n_saved += 1

                if invert_action == "skip":
                    global_idx += 1
                    continue

            # ---- collect ----
            probs_list.append(pwi)
            gts_list.append(gti)
            global_idx += 1

        n_batches += 1
        if (max_batches is not None) and (n_batches >= max_batches):
            break

    print(f"[debug] saved {n_saved} inverted panels to: {debug_dir}")
    return probs_list, gts_list

def collect_val_sum_probs_and_gts(
    mask_model,
    seg_val_calib,
    IDX_SUM,
    preprocess=None,
    assume_prob="auto",
    include_empty_gt=False,
    max_batches=None,
):
    probs_list, gts_list = [], []
    n_batches = 0

    for batch in seg_val_calib:
        if isinstance(batch, (tuple, list)):
            x, gts = batch[:2]
        else:
            x, gts = batch["image"], batch["mask"]

        x_np   = to_np(x,   np.float32)
        gts_np = to_np(gts, np.float32)

        x_in = preprocess(x_np) if preprocess is not None else x_np

        pred = mask_model.predict(x_in, verbose=0)
        prob_all = get_prob_all_batch(pred, assume=assume_prob)
        prob_all = to_np(prob_all, np.float32)

        # sum prob
        prob_sum = get_prob_sum_from_prob_all(
            prob_all, IDX_SUM,
            exclude_idx=EXCLUDE_IDX, exclude_hard_thr=0.5
        )  # (B,H,W)

        # gt sum (binary)  -- gts_np is already np.float32
        gt_sum = np.zeros(gts_np.shape[:-1], np.float32)
        for k in IDX_SUM:
            gt_sum += gts_np[..., int(k)]
        gt_sum = (gt_sum > 0.5).astype(np.float32)

        if not include_empty_gt:
            keep = gt_sum.reshape(gt_sum.shape[0], -1).sum(axis=1) > 0
            prob_sum = prob_sum[keep]
            gt_sum   = gt_sum[keep]

        for b in range(prob_sum.shape[0]):
            probs_list.append(prob_sum[b].reshape(-1))
            gts_list.append(gt_sum[b].reshape(-1))

        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break

    return probs_list, gts_list

def collect_val_change_probs_and_gts(
    mask_model,
    seg_val_calib,
    CHANGE_IDXS,
    preprocess=None,
    assume_prob="auto",
    include_empty_gt=False,
    max_batches=None,
):
    probs_list, gts_list = [], []

    for bi, batch in enumerate(seg_val_calib):
        if max_batches is not None and bi >= max_batches:
            break

        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            imgs, gts = batch[:2]
        else:
            raise ValueError("seg_val_calib batch must provide (img, gt)")

        imgs_np = to_np(imgs, np.float32)
        gts_np  = to_np(gts,  np.float32)

        imgs_in = preprocess(imgs_np) if preprocess is not None else imgs_np

        pred = mask_model.predict(imgs_in, verbose=0)
        prob_all = get_prob_all_batch(pred, assume=assume_prob)
        prob_all = to_np(prob_all, np.float32)

        # CHANGE prob
        C_prob = np.zeros(prob_all.shape[:-1], dtype=np.float32)
        for idx in CHANGE_IDXS:
            C_prob += prob_all[..., int(idx)]
        C_prob = np.clip(C_prob, 0.0, 1.0)

        # CHANGE GT
        C_gt = np.zeros(gts_np.shape[:-1], dtype=np.float32)
        for idx in CHANGE_IDXS:
            C_gt += gts_np[..., int(idx)]
        C_gt = (C_gt > 0.5).astype(np.float32)

        for b in range(C_prob.shape[0]):
            if not include_empty_gt and C_gt[b].sum() == 0:
                continue
            probs_list.append(C_prob[b])
            gts_list.append(C_gt[b])

    if len(probs_list) == 0:
        raise RuntimeError("No valid CHANGE samples collected")

    return probs_list, gts_list

def dice_binary(pred01, gt01):
    inter = np.sum((pred01 == 1) & (gt01 == 1))
    denom = np.sum(pred01 == 1) + np.sum(gt01 == 1)
    return (2.0 * inter / (denom + 1e-8)) if denom > 0 else 1.0  # Convention: 1 when both empty


def calibrate_thr_policy_aware(probs_list, gts_list,
                              thr_grid=None,
                              fg_min=0.005, fg_max=0.60,
                              use_pos_only=True):
    if thr_grid is None:
        thr_grid = np.linspace(0.001, 0.99, 200)  # Do not include 0 (prevents masking the root cause)

    best_thr, best_score = None, -1e9

    for thr in thr_grid:
        scores = []
        for pw, gt in zip(probs_list, gts_list):
            gt01 = (gt >= 0.5).astype(np.uint8)
            if use_pos_only and gt01.sum() == 0:
                continue  # pos-only policy

            pred01 = (pw >= thr).astype(np.uint8)

            fg = pred01.mean()
            if fg < fg_min or fg > fg_max:
                continue  # guardrail: exclude masks that are too small or too large

            scores.append(dice_binary(pred01, gt01))

        if len(scores) == 0:
            continue

        score = float(np.mean(scores))
        if score > best_score:
            best_score, best_thr = score, float(thr)

    return best_thr, best_score

POLICY_VER = 2

def run_calibration_changefirst_set_once(
    mask_model,
    seg_val_calib,

    # --- index definitions ---
    wound_idx=1,
    support_idxs=(1,2,3),     # lesion/support union (e.g., wound+eschar+healed)
    change_idxs=(1,2),        # change union (e.g., wound+eschar)

    # --- io ---
    calib_path=Path("CALIB_THR_CHANGEFIRST_SET.npz"),
    force=False,

    # --- common ---
    preprocess=None,
    assume_prob="softmax",
    anchor_thr_dict=None,     # {"wound":..., "support":..., "change":...} optional

    # --- calibrate_thr_stable args ---
    trim_ratio=0.10,
    min_samples=20,
    max_batches=None,

    # --- clamp (optional) ---
    thr_support_ceil=0.95,
    thr_support_floor=None,   # Optional: pass THR_CHANGE as floor (only when needed)
):
    """
    returns:
      thr_dict = {"wound": THR_WOUND, "support": THR_SUPPORT, "change": THR_CHANGE}
      stats_dict = {"wound": stats_w, "support": stats_s, "change": stats_c}

    NOTE:
      - change-first inclusion is enforced in 'mask generation' via m_wound = m_change & (...).
      - Here, only the threshold is stably computed.
    """

    # ---------------- load (if exists) ----------------
    if calib_path.exists() and (not force):
        d = np.load(calib_path, allow_pickle=True)
        if str(d.get("mode", "")) == "stable_changefirst_set":
            thr_dict = d["thr_dict"].item()
            stats_dict = d["stats_dict"].item()
            print("[calib-changefirst] loaded:",
                  {k: float(v) for k, v in thr_dict.items()})
            return thr_dict, stats_dict

    # ---------------- helper: run one target ----------------
    def _run_one(target: str):
        # target in {"wound","support","change"}
        if target == "wound":
            probs_list, gts_list = collect_val_probs_and_gts(
                mask_model=mask_model,
                seg_val_dataset=seg_val_calib,
                wound_idx=wound_idx,
                preprocess=preprocess,
                assume_prob=assume_prob,
                include_empty_gt=False,
                invert_action="debug",
                max_batches=max_batches
            )

        elif target == "support":
            probs_list, gts_list = collect_val_sum_probs_and_gts(
                mask_model=mask_model,
                seg_val_calib=seg_val_calib,
                IDX_SUM=support_idxs,
                preprocess=preprocess,
                assume_prob=assume_prob,
                include_empty_gt=False,
                max_batches=max_batches,
            )

        elif target == "change":
            probs_list, gts_list = collect_val_change_probs_and_gts(
                mask_model=mask_model,
                seg_val_calib=seg_val_calib,
                CHANGE_IDXS=change_idxs,
                preprocess=preprocess,
                assume_prob=assume_prob,
                include_empty_gt=False,
                max_batches=max_batches,
            )

        else:
            raise ValueError(f"unknown target: {target}")

        anchor_thr = None
        if anchor_thr_dict is not None and target in anchor_thr_dict:
            anchor_thr = float(anchor_thr_dict[target])

        thr, stats = calibrate_thr_stable(
            probs_list,
            gts_list,
            anchor_thr=anchor_thr,
            trim_ratio=trim_ratio,
            min_samples=min_samples
        )
        return float(thr), stats

    # ---------------- run three calibrations ----------------
    print("[calib] start wound")
    thr_wound, stats_wound   = _run_one("wound")
    print("[calib] done wound")

    print("[calib] start support")
    thr_support_raw, stats_support = _run_one("support")
    print("[calib] done support")

    print("[calib] start change")
    thr_change, stats_change = _run_one("change")
    print("[calib] done change")

    # ---------------- clamp support (optional) ----------------
    thr_support = float(thr_support_raw)
    if thr_support_floor is not None:
        thr_support = max(thr_support, float(thr_support_floor))
    thr_support = min(thr_support, float(thr_support_ceil))

    # attach clamp info
    stats_support = dict(stats_support)
    stats_support.update({
        "thr_raw": float(thr_support_raw),
        "thr_clamped": float(thr_support),
        "thr_floor": None if thr_support_floor is None else float(thr_support_floor),
        "thr_ceil": float(thr_support_ceil),
    })

    thr_dict = {
        "wound": float(thr_wound),
        "support": float(thr_support),
        "change": float(thr_change),
    }
    stats_dict = {
        "wound": stats_wound,
        "support": stats_support,
        "change": stats_change,
    }

    # ---------------- save ----------------
    np.savez(
        calib_path,
        mode="stable_changefirst_set",
        thr_dict=np.array(thr_dict, dtype=object),
        stats_dict=np.array(stats_dict, dtype=object),
        meta=np.array({
            "wound_idx": int(wound_idx),
            "support_idxs": tuple(int(x) for x in support_idxs),
            "change_idxs": tuple(int(x) for x in change_idxs),
            "assume_prob": str(assume_prob),
        }, dtype=object),
        ts=np.array(time.time()),
    )

    print("[calib-changefirst] saved:",
          {k: float(v) for k, v in thr_dict.items()})
    return thr_dict, stats_dict

WOUND_IDX = 1
ESCHAR_IDX = 2
HEALED_IDX = 3
EXCLUDE_IDX = 4
LESION_IDXS = (WOUND_IDX, HEALED_IDX, ESCHAR_IDX)
CHANGE_IDXS  = (WOUND_IDX, ESCHAR_IDX)
STABLE_IDXS  = (HEALED_IDX,)


def get_wound_prob_batch(pred_batch, wound_idx=1, assume="auto"):
    """
    pred_batch: (B,H,W,C) from model.predict
    returns: (B,H,W,1) wound probability in [0,1]
    assume: "auto" | "softmax" | "sigmoid"
    """
    x = np.asarray(pred_batch).astype(np.float32)
    if x.ndim != 4:
        raise ValueError(f"pred_batch ndim must be 4, got {x.ndim}")

    B, H, W, C = x.shape
    if C > 1 and not (0 <= wound_idx < C):
        raise ValueError(f"wound_idx out of range: wound_idx={wound_idx}, C={C}")

    if C == 1:
        # Usually either sigmoid probability or logits (single channel)
        if assume == "sigmoid":
            pw = x[..., 0]
        else:
            # auto: estimate sigmoid/logits based on value range
            # If probability, should be near [0,1]
            mn, mx = float(x.min()), float(x.max())
            if mn >= -1e-3 and mx <= 1.0 + 1e-3:
                pw = x[..., 0]
            else:
                # logits -> sigmoid
                pw = 1.0 / (1.0 + np.exp(-x[..., 0]))
        return pw[..., None].clip(0.0, 1.0)

    # C > 1
    if assume == "softmax":
        # Assume x is already a probability
        prob = x
    elif assume == "sigmoid":
        # For multi-label sigmoid, each channel should be an independent probability
        prob = x.clip(0.0, 1.0)
    else:
        # auto: treat as softmax probability if sum==1, else apply softmax to logits
        s = x[0, 0, 0, :].sum()
        if 0.95 <= s <= 1.05 and x.min() >= -1e-3 and x.max() <= 1.0 + 1e-3:
            prob = x
        else:
            # logits -> softmax
            z = x - x.max(axis=-1, keepdims=True)
            ez = np.exp(z)
            prob = ez / (ez.sum(axis=-1, keepdims=True) + 1e-8)

    pw = prob[..., wound_idx]
    return pw[..., None].clip(0.0, 1.0)

def get_prob_all_batch(pred_batch, assume="auto", auto_samples=64, eps=1e-6):
    """
    pred_batch: (B,H,W,C) from model.predict
    returns:    (B,H,W,C) probability tensor in [0,1]
    assume: "auto" | "softmax_logits" | "softmax_prob" | "sigmoid_logits" | "sigmoid_prob"
    """
    x = np.asarray(pred_batch, dtype=np.float32)
    if x.ndim != 4:
        raise ValueError(f"pred_batch ndim must be 4, got {x.ndim}")
    B, H, W, C = x.shape

    # ---- C==1 ----
    if C == 1:
        if assume in ["sigmoid_prob"]:
            return np.clip(x, 0.0, 1.0)
        if assume in ["sigmoid_logits"]:
            return 1.0 / (1.0 + np.exp(-x))

        # auto for C==1
        mn, mx = float(x.min()), float(x.max())
        if (mn >= -1e-3) and (mx <= 1.0 + 1e-3):
            return np.clip(x, 0.0, 1.0)
        return 1.0 / (1.0 + np.exp(-x))

    # ---- C>1 ----
    def _softmax(z):
        z = z - z.max(axis=-1, keepdims=True)
        ez = np.exp(z)
        return ez / (ez.sum(axis=-1, keepdims=True) + eps)

    if assume == "softmax_prob":
        return np.clip(x, 0.0, 1.0)

    if assume == "softmax_logits":
        return np.clip(_softmax(x), 0.0, 1.0)

    if assume == "sigmoid_prob":
        return np.clip(x, 0.0, 1.0)

    if assume == "sigmoid_logits":
        return np.clip(1.0 / (1.0 + np.exp(-x)), 0.0, 1.0)

    # ---- auto robust ----
    # 1) range check
    in_01 = (x.min() >= -1e-3) and (x.max() <= 1.0 + 1e-3)

    # 2) channel-sum check on sampled pixels (not just [0,0])
    # sample up to auto_samples pixels per batch element
    # deterministic sampling: fixed grid stride
    ys = np.linspace(0, H - 1, int(np.sqrt(auto_samples)), dtype=int)
    xs = np.linspace(0, W - 1, int(np.sqrt(auto_samples)), dtype=int)
    ss = []
    for y in ys:
        for x0 in xs:
            ss.append(x[0, y, x0, :].sum())
    s_mean = float(np.mean(ss))
    sum1 = (0.95 <= s_mean <= 1.05)

    if in_01 and sum1:
        return np.clip(x, 0.0, 1.0)

    return np.clip(_softmax(x), 0.0, 1.0)

def get_lesion_prob_from_prob_all(prob_all, lesion_idxs=LESION_IDXS, exclude_idx=EXCLUDE_IDX, exclude_mode="hard", exclude_hard_thr=0.5, clip=True):
    prob_all = np.asarray(prob_all, np.float32)
    if prob_all.ndim < 3:
        raise ValueError(f"prob_all ndim too small: {prob_all.ndim}, shape={prob_all.shape}")

    C = prob_all.shape[-1]

    # If already 1 channel, treat as lesion
    if C == 1:
        p = prob_all[..., 0]
        return np.clip(p, 0.0, 1.0).astype(np.float32) if clip else p.astype(np.float32)

    # If multi-channel, sum over lesion_idxs
    p = np.zeros(prob_all.shape[:-1], dtype=np.float32)
    for k in lesion_idxs:
        k = int(k)
        if 0 <= k < C:
            p += prob_all[..., k]
    p = np.clip(p, 0.0, 1.0)

    # Apply exclude (optional)
    if exclude_idx is not None:
        ex = prob_all[..., int(exclude_idx)]
        if exclude_mode == "soft":
            p *= (1.0 - np.clip(ex, 0.0, 1.0))
        elif exclude_mode == "hard":
            p *= (ex < float(exclude_hard_thr)).astype(np.float32)
        elif exclude_mode is None:
            pass
        else:
            raise ValueError("exclude_mode must be 'hard', 'soft', or None")

    return np.clip(p, 0.0, 1.0).astype(np.float32) if clip else p.astype(np.float32)

def get_prob_sum_from_prob_all(prob_all, idxs, exclude_idx=None, exclude_hard_thr=0.5):
    """
    prob_all: (H,W,C) or (B,H,W,C) or (T,H,W,C); any leading axis is acceptable
    idxs: tuple of class indices to sum
    """
    prob_all = np.asarray(prob_all, dtype=np.float32)

    p = 0.0
    C = prob_all.shape[-1]

    for k in idxs:
        if 0 <= k < C:
            p = p + prob_all[..., k]

    p = np.clip(p, 0.0, 1.0)

    # exclude: remove pixels with strong probability of background/dressing/other
    if exclude_idx is not None and 0 <= exclude_idx < C:
        ex = (prob_all[..., exclude_idx].astype(np.float32) >= float(exclude_hard_thr)).astype(np.float32)
        p = p * (1.0 - ex)

    return p.astype(np.float32)

def diagnose_channel_mapping(mask_model, seg_val_calib, num_classes, max_batches=10):
    # Per GT channel: how large is the mean probability of each pred channel in the GT=1 region
    # -> wound channel tends to have the highest mean in its own GT region
    acc = np.zeros((num_classes, num_classes), dtype=np.float64)  # [gt_c, pred_c]
    cnt = np.zeros((num_classes,), dtype=np.int64)

    b = 0
    for imgs_m11, gts in seg_val_calib:
        imgs01 = (imgs_m11.numpy() + 1.0) / 2.0
        pred = mask_model.predict(imgs01, verbose=0)  # (B,H,W,Cpred)
        pred = np.asarray(pred)
        Cpred = pred.shape[-1]
        if Cpred != num_classes:
            print("WARNING: Cpred != num_classes", Cpred, num_classes)

        B = pred.shape[0]
        gts_np = gts.numpy()  # (B,H,W,Cgt)

        for i in range(B):
            for gt_c in range(num_classes):
                gt_mask = gts_np[i, ..., gt_c] >= 0.5
                if gt_mask.sum() == 0:
                    continue
                cnt[gt_c] += 1
                for pred_c in range(min(Cpred, num_classes)):
                    acc[gt_c, pred_c] += float(pred[i, ..., pred_c][gt_mask].mean())

        b += 1
        if max_batches is not None and b >= max_batches:
            break

    avg = acc / (cnt[:, None] + 1e-9)
    print("counts per GT channel:", cnt)
    print("avg pred prob inside each GT channel region (rows=GT, cols=Pred):\n", avg)
    print("argmax pred channel per GT channel:", avg.argmax(axis=1))
    return avg
