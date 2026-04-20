"""
BurnSight postprocessing — soft mask, allow/protect, alpha-K scheduling
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

from src.inference.mask_utils import *

def match_meanstd_seq01(seq01, mean=(0.5,0.5,0.5), std=(0.25,0.25,0.25)):
    """
    seq01: (1,T,H,W,3) or (T,H,W,3), values in [0,1]
    Standardize each frame (per time step t) using its own mean/std,
    then match to target mean/std and return in [0,1].
    """
    x = np.clip(seq01.astype(np.float32), 0, 1)
    add_batch = False
    if x.ndim == 5:  # (1,T,H,W,3) -> (T,H,W,3)
        x = x[0]
        add_batch = True

    # Per-frame mean/std: averaged over axes (H,W)
    mu = x.mean(axis=(1,2), keepdims=True)                           # (T,1,1,3)
    sd = x.std(axis=(1,2), keepdims=True) + 1e-6                     # (T,1,1,3)

    mt = np.array(mean, np.float32)[None, None, None, :]             # (1,1,1,3)
    st = np.array(std,  np.float32)[None, None, None, :]             # (1,1,1,3)

    y = np.clip((x - mu) / sd * st + mt, 0, 1)                       # (T,H,W,3)

    if add_batch:                                                    # Restore to (1,T,H,W,3)
        y = y[None, ...]
    return y

def preprocess_infer_seq(x_1thw3):
    x01 = match_meanstd_seq01(x_1thw3)
    return x01*2.0 - 1.0

avg = diagnose_channel_mapping(mask_model=mask, seg_val_calib=seg_val_calib, num_classes=NUM_CLASSES, max_batches=20)

thr_dict, stats_dict = run_calibration_changefirst_set_once(
    mask_model=mask,
    seg_val_calib=seg_val_calib,
    wound_idx=WOUND_IDX,
    support_idxs=LESION_IDXS,   # (WOUND, HEALED, ESCHAR) etc.
    change_idxs=CHANGE_IDXS,    # (WOUND, ESCHAR)
    preprocess=preprocess_infer_seq,
    assume_prob="softmax",
    thr_support_floor=None,     # Optionally pass float(THR_CHANGE) as floor
)
THR_WOUND   = thr_dict["wound"]
THR_SUPPORT = thr_dict["support"]
THR_CHANGE  = thr_dict["change"]

stats_wound = stats_dict["wound"]
stats_change = stats_dict["change"]
stats_sup = stats_dict["support"]

print("stats_wound =", stats_wound)
print("stats_change =", stats_change)
print("stats_sup =", stats_sup)

batch_imgs, batch_gts = next(iter(seg_train))
g = batch_gts.numpy()  # (B,H,W,5)

w_cov = g[..., WOUND_IDX].reshape(g.shape[0], -1).mean(axis=1)  # wound pixel ratio
print("wound coverage per sample:", np.round(w_cov, 4))

# Viewing per-class coverage together gives a clearer picture
cov_all = g.reshape(g.shape[0], -1, g.shape[-1]).mean(axis=1)  # (B,C)
print("class coverage per sample (B,C):\n", np.round(cov_all, 4))

imgs_m11 = batch_imgs.numpy()
pred     = mask.predict(imgs_m11, verbose=0)

prob_batch = get_wound_prob_batch(pred, wound_idx=WOUND_IDX)   # (B,64,64,1)

gts_np = batch_gts.numpy().astype(np.float32)  # (B,H,W,5)

gt_wound_batch = gts_np[..., WOUND_IDX:WOUND_IDX+1]
gt_lesion_batch = (gts_np[..., WOUND_IDX] + gts_np[..., HEALED_IDX] + gts_np[..., ESCHAR_IDX])[..., None]
gt_lesion_batch = (gt_lesion_batch > 0.5).astype(np.float32)  # binary lesion GT (soft sum also acceptable)

prob_all = get_prob_all_batch(pred, assume="auto")
prob_lesion_batch = np.clip(
    prob_all[..., WOUND_IDX] + prob_all[..., HEALED_IDX] + prob_all[..., ESCHAR_IDX],
    0.0, 1.0
)[..., None]


def make_soft_gate(p_soft, mode="raw", thr=None, tau=0.10):
    """
    p_soft: (B,H,W,1) or (H,W,1) or (H,W)
    returns same shape with values in [0,1]
    """
    p = p_soft[...,0] if p_soft.ndim >= 3 else p_soft
    p = np.clip(p, 0.0, 1.0).astype(np.float32)

    if mode == "raw":
        m = p
    elif mode == "soft_thr":
        if thr is None:
            raise ValueError("soft_thr requires thr")
        z = (p - float(thr)) / float(tau)
        m = 1.0 / (1.0 + np.exp(-z))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return m[..., None]

def overlay_soft(img01, m_soft, alpha_max=0.45, color=(1.0,0.0,0.0), gamma=1.0):
    m = m_soft[...,0] if m_soft.ndim==3 else m_soft
    m = np.clip(m, 0, 1).astype(np.float32)
    if gamma != 1.0:
        m = np.power(m, gamma)
    overlay = img01.copy()
    a = (alpha_max * m)[..., None]  # (H,W,1)
    overlay = (1.0 - a) * overlay + a * np.array(color)[None,None,:]
    return overlay

def build_soft_lesion_anchor(prob_wound,
                             prob_healed=None,
                             prob_eschar=None,
                             prob_exclude=None):
    """
    All inputs: (H,W) or (H,W,1), [0,1]
    Returns: (H,W) soft lesion anchor in [0,1]
    """
    L = np.clip(prob_wound, 0, 1)

    if prob_healed is not None:
        L = L + np.clip(prob_healed, 0, 1)

    if prob_eschar is not None:
        L = L + np.clip(prob_eschar, 0, 1)

    # Remove exclude
    if prob_exclude is not None:
        L = L * (1.0 - np.clip(prob_exclude, 0, 1))

    return np.clip(L, 0, 1).astype(np.float32)

# --- VISUAL SANITY CHECK ONLY (not used in main pipeline) ---
N = min(5, prob_batch.shape[0])
for i in range(N):
    gt_wound    = gt_wound_batch[i, ..., 0]
    gt_lesion   = gt_lesion_batch[i, ..., 0]

    pw = prob_all[i, ..., WOUND_IDX]
    ph = prob_all[i, ..., HEALED_IDX]
    pe = prob_all[i, ..., ESCHAR_IDX]
    px = prob_all[i, ..., EXCLUDE_IDX]

    p_lesion = np.clip(pw + ph + pe, 0.0, 1.0)

    exp_area_lesion = float(p_lesion.mean())
    bin_area_wound  = float((pw >= THR_WOUND).mean())
    print(f"[i={i}] exp_area_lesion={exp_area_lesion:.3f}, bin_area_wound@THR={bin_area_wound:.3f}")

    # For gating/analysis: lesion soft gate (raw without THR dependency is recommended)
    gate_lesion = make_soft_gate(p_lesion[..., None], mode="raw")  # (H,W,1)
    overlay_roi = overlay_soft(imgs_m11[i], gate_lesion, alpha_max=0.45)

    # Reporting anchor: wound at global THR
    overlay_w_anchor, used_thr, used_mask = overlay_mask_on_image(
        imgs_m11[i], pw[..., None], thr=THR_WOUND, method="otsu", alpha=0.40
    )

    plt.figure(figsize=(24,4))
    plt.subplot(1,6,1); plt.imshow(imgs_m11[i]);                 plt.title("Input");           plt.axis("off")
    plt.subplot(1,6,2); plt.imshow(gt_wound, cmap='gray');     plt.title("GT wound");        plt.axis("off")
    plt.subplot(1,6,3); plt.imshow(pw, cmap='gray');   plt.title("Prob wound");      plt.axis("off")
    plt.subplot(1,6,4); plt.imshow(gt_lesion, cmap='gray');    plt.title("GT lesion");       plt.axis("off")
    plt.subplot(1,6,5); plt.imshow(p_lesion, cmap='gray');  plt.title("Prob lesion");     plt.axis("off")
    plt.subplot(1,6,6); plt.imshow(overlay_roi);               plt.title("ROI gate (soft lesion)"); plt.axis("off")
    plt.show()

    # Render anchor overlay as a separate figure if needed
    plt.figure(figsize=(6,6))
    plt.imshow(overlay_w_anchor); plt.title(f"Anchor wound THR={THR_WOUND:.2f}"); plt.axis("off")
    plt.show()


def make_fixed_msoft_from_seq(
    example_sequence01, mask_model, preprocess=None,
    THR=0.55,
    mode="mixed", tau=0.30, ksize=1, ema_alpha=0.5,
    floor=0.03, ceil=0.95,
    temp_T=0.9, use_tta=True,
    t_strategy="clamp", delta=0.10, lam=0.20,
    use_elbow=False,
    thr_scope="sequence", thr_reduce="median",
    # QC
    qc_area_min=0.002, qc_area_max=0.60,
    qc_center_radius_frac=0.22, qc_center_frac_min=0.20,
    qc_cc_max=3,
    assume="auto",
    exclude_hard_thr=0.5,
):
    """
    Purpose: build and fix the ROI gate (m_soft) only once

    return:
      m_soft: (1,T,H,W,1) in [0,1]  (final gate with EMA applied)
      m_bin : (1,T,H,W,1) {0,1}     (binary ROI based on threshold)
      info  : dict (t_used_seq, which_seq, reasons_seq, etc.)
    """

    # ---------------- helpers ----------------
    def _temp_scale_prob(p, T=1.0, eps=1e-6):
        p = np.asarray(p, dtype=np.float32)
        p = np.clip(p, eps, 1.0 - eps)
        if T is None or abs(T - 1.0) < 1e-6:
            return p
        logit = np.log(p / (1.0 - p)) / float(T)
        return 1.0 / (1.0 + np.exp(-logit))

    def _robust_center_t(p, t_abs_min=0.02, t_abs_max=0.90,
                         min_fg_frac=0.002, q_backup=0.98):
        p = np.clip(p.astype(np.float32), 0, 1)
        hist, bins = np.histogram(p, bins=256, range=(0, 1))
        if hist.sum() == 0:
            t = 0.1
        else:
            w1 = np.cumsum(hist)
            w2 = np.cumsum(hist[::-1])[::-1]
            bc = (bins[:-1] + bins[1:]) / 2.0
            m1 = np.cumsum(hist * bc) / (w1 + 1e-8)
            m2 = (np.cumsum((hist * bc)[::-1]) / (w2[::-1] + 1e-8))[::-1]
            var = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:])**2
            if var.size == 0:
                t = 0.1
            else:
                k = int(np.argmax(var))
                t = float((bins[k] + bins[k+1]) / 2.0)

        if (not np.isfinite(t)) or ((p >= t).mean() < min_fg_frac):
            t = float(np.quantile(p, q_backup))
        return float(np.clip(t, t_abs_min, t_abs_max))

    def _elbow_thr_area(p, t_min=0.02, t_max=0.70, n=69):
        thrs = np.linspace(t_min, t_max, n, dtype=np.float32)
        areas = np.array([(p >= t).mean() for t in thrs], dtype=np.float32)
        x = (thrs - thrs[0]) / (thrs[-1] - thrs[0] + 1e-8)
        y = (areas - areas[-1]) / (areas[0] - areas[-1] + 1e-8)
        dist = np.abs(y - (1 - x)) / np.sqrt(2.0)
        return float(thrs[int(np.argmax(dist))])

    def _clamp_or_blend(t_pf, global_thr):
        if t_strategy == "clamp":
            return float(np.clip(t_pf, global_thr - delta, global_thr + delta))
        elif t_strategy == "blend":
            return float((1.0 - lam) * t_pf + lam * global_thr)
        else:
            return float(t_pf)

    def _qc_reasons(mask_bin_u8):
        H, W = mask_bin_u8.shape
        area_frac = float(mask_bin_u8.mean())
        reasons = []
        if area_frac < qc_area_min:
            reasons.append(f"too_small({area_frac:.4f})")
        if area_frac > qc_area_max:
            reasons.append(f"too_large({area_frac:.4f})")

        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin_u8, connectivity=8)
        comps = max(0, n - 1)
        if comps == 0:
            reasons.append("no_component")
        if comps > qc_cc_max:
            reasons.append(f"too_many_cc({comps})")

        if mask_bin_u8.sum() > 0:
            cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
            r = qc_center_radius_frac * min(H, W)
            yy, xx = np.mgrid[0:H, 0:W]
            disk = ((yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2)
            center_hit = float(mask_bin_u8[disk].sum()) / (float(mask_bin_u8.sum()) + 1e-6)
            if center_hit < qc_center_frac_min:
                reasons.append(f"off_center({center_hit:.3f})")

        return reasons

    def _pick_thr_global_then_fallback(p_scaled_hw):
        t0 = float(THR)
        m0 = (p_scaled_hw >= t0).astype(np.uint8)
        reasons = _qc_reasons(m0)
        if len(reasons) == 0:
            return t0, "global", []
        t_pf = _elbow_thr_area(p_scaled_hw) if use_elbow else _robust_center_t(p_scaled_hw)
        t_used = _clamp_or_blend(t_pf, THR)
        return float(t_used), "dynamic", reasons

    def _seg_predict_with_optional_tta(x01_1thw3):
        xin = x01_1thw3 if preprocess is None else preprocess(x01_1thw3)
        B0, T0, H0, W0, _ = xin.shape
        BT0 = B0 * T0

        p0 = mask_model.predict(xin.reshape(BT0, H0, W0, 3), verbose=0)

        if use_tta:
            xin_f = xin[:, :, :, ::-1, :]
            p1 = mask_model.predict(xin_f.reshape(BT0, H0, W0, 3), verbose=0)
            p1 = p1.reshape(B0, T0, H0, W0, -1)[:, :, :, ::-1, :].reshape(BT0, H0, W0, -1)
            p_raw = (p0 + p1) * 0.5
        else:
            p_raw = p0

        # User pipeline
        prob_all = get_prob_all_batch(p_raw, assume=assume)
        p_w = prob_all[..., WOUND_IDX].astype(np.float32)
        p_e = prob_all[..., ESCHAR_IDX].astype(np.float32)
        p_h = prob_all[..., HEALED_IDX].astype(np.float32)

        # Lesion candidates: max (reduces saturation/clip issues)
        lesion = np.maximum.reduce([p_w, p_e, p_h])  # (BT,H,W)

        # exclude hard
        if EXCLUDE_IDX is not None:
            ex = prob_all[..., int(EXCLUDE_IDX)].astype(np.float32)
            lesion *= (ex < float(exclude_hard_thr)).astype(np.float32)

        # ---- Convert to single channel: (BT,H,W,1) ----
        lesion = lesion[..., None]  # (BT,H,W,1)

        # ---- Convert to single channel: (BT,H,W,1) ----
        if lesion.ndim == 2:
            lesion = lesion[None, ..., None]
        elif lesion.ndim == 3:
            lesion = lesion[..., None]
        elif lesion.ndim == 4 and lesion.shape[-1] == 1:
            pass
        else:
            raise ValueError(f"S unexpected shape: {lesion.shape}")

        return lesion.astype(np.float32)  # (BT,H,W,1)

    # ---------------- (0) input ----------------
    assert example_sequence01.ndim == 5 and example_sequence01.shape[0] == 1
    B, T, H, W, _ = example_sequence01.shape
    BT = B * T

    # ---------------- (1) seg predict ----------------
    prob_bt = _seg_predict_with_optional_tta(example_sequence01)  # (BT,H,W,1)

    # ---------------- (1.1) Temperature scaling (once) ----------------
    p_bt_scaled = _temp_scale_prob(prob_bt[..., 0], T=temp_T).astype(np.float32)  # (BT,H,W)

    # ---------------- (1.2) Determine threshold ----------------
    t_used_seq = float(THR)
    which_seq, reasons_seq = "global", []

    if thr_scope == "sequence":
        if thr_reduce == "median":
            p_repr = np.median(p_bt_scaled, axis=0)
        elif thr_reduce == "mean":
            p_repr = np.mean(p_bt_scaled, axis=0)
        elif thr_reduce == "max":
            p_repr = np.max(p_bt_scaled, axis=0)
        elif thr_reduce == "p90":
            p_repr = np.quantile(p_bt_scaled, 0.90, axis=0)
        else:
            raise ValueError(f"unknown thr_reduce={thr_reduce}")

        t_used_seq, which_seq, reasons_seq = _pick_thr_global_then_fallback(p_repr)

    print(f"[thr] scope={thr_scope} reduce={thr_reduce} t_used_seq={t_used_seq:.4f} which={which_seq} reasons={reasons_seq}")

    # ---------------- (2) Build gate ----------------
    m_soft_bt = np.empty((BT, H, W, 1), dtype=np.float32)
    m_bin_bt  = np.empty((BT, H, W, 1), dtype=np.float32)

    for i in range(BT):
        p = p_bt_scaled[i]  # (H,W)

        # Select threshold
        if thr_scope == "sequence":
            t_used = t_used_seq
        else:
            t_used, which_i, reasons_i = _pick_thr_global_then_fallback(p)

        m_bin = (p >= t_used).astype(np.float32)

        if mode == "hard":
            s = m_bin
        elif mode == "soft":
            s = np.clip(p, floor, ceil).astype(np.float32)
        else:  # mixed
            z = (p - t_used) / max(float(tau), 1e-6)
            s = 1.0 / (1.0 + np.exp(-z))
            s = np.clip(s, floor, ceil).astype(np.float32)

        if ksize and ksize > 1:
            s = tf.nn.max_pool2d(
                tf.constant(s[None, ..., None]),
                ksize=ksize, strides=1, padding="SAME"
            ).numpy()[0, ..., 0].astype(np.float32)

        m_soft_bt[i, ..., 0] = s
        m_bin_bt[i,  ..., 0] = m_bin

    # ---------------- (3) reshape + EMA ----------------
    m_soft = m_soft_bt.reshape(B, T, H, W, 1)
    m_bin  = m_bin_bt.reshape(B, T, H, W, 1)

    if ema_alpha is not None:
        a = float(ema_alpha)
        for t in range(1, T):
            m_soft[:, t] = a * m_soft[:, t-1] + (1.0 - a) * m_soft[:, t]

    info = {
        "t_used_seq": float(t_used_seq),
        "which_seq": which_seq,
        "reasons_seq": reasons_seq,
        "roi_frac_mean": float(m_bin.mean()),
    }
    return m_soft.astype(np.float32), m_bin.astype(np.float32), info

def build_Xk_with_fixed_msoft(example_sequence01, m_soft, K_VAL, K_MAX):
    """
    m_soft: (1,T,H,W,1) fixed ROI
    """
    rgb_m11 = example_sequence01 * 2.0 - 1.0
    k_norm  = float(K_VAL) / float(K_MAX)
    k_map   = np.full(m_soft.shape, k_norm, np.float32)

    Xk = np.concatenate([rgb_m11, k_map, m_soft], axis=-1)
    return Xk.astype(np.float32)

sorted_image_files = get_sorted_day_images(original_test_dir)
day6_image_path = sorted_image_files[5]

day6_img = cv2.imread(day6_image_path)
day6_img = cv2.resize(day6_img, (64, 64)) / 255.0
target_image = tf.convert_to_tensor(day6_img, dtype=tf.float32)

print("target image - min:", tf.reduce_min(target_image).numpy(),
      "max:", tf.reduce_max(target_image).numpy())
mean_creator = np.mean(target_image)
std_creator = np.std(target_image)

print(f"target image - Mean: {mean_creator:.4f}, Std: {std_creator:.4f}")

gray = tf.image.rgb_to_grayscale(target_image)
contrast = tf.math.reduce_std(gray).numpy()
print("Contrast:", contrast)

hsv = tf.image.rgb_to_hsv(target_image)
saturation = tf.reduce_mean(hsv[..., 1]).numpy()
print("Saturation:", saturation)

# Generate image sequence in DayN order
example_sequence = []
for image_path in sorted_image_files:
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)         # BGR→RGB conversion
    img = cv2.resize(img, (64, 64)).astype(np.float32) / 255.0
    example_sequence.append(img)

# Convert to numpy array (automatically shaped as (T,64,64,3))
example_sequence = np.asarray(example_sequence, dtype=np.float32)

# Add batch dimension → (1,T,64,64,3)
example_sequence = example_sequence[None, ...]
print("example_sequence shape:", example_sequence.shape)

T_needed = SEQLEN  # sequence_length used during training
if example_sequence.shape[1] > T_needed:
    example_sequence = example_sequence[:, -T_needed:]
elif example_sequence.shape[1] < T_needed:
    pad = T_needed - example_sequence.shape[1]
    pad_frames = np.repeat(example_sequence[:, :1], pad, axis=1)  # Pad by repeating the first frame
    example_sequence = np.concatenate([pad_frames, example_sequence], axis=1)

print("example_sequence shape:", example_sequence.shape)

m_soft, m_bin, info = make_fixed_msoft_from_seq(
    example_sequence, mask_model,
    preprocess=preprocess_infer_seq,
    THR=THR_WOUND,
    mode="mixed", tau=0.30, ema_alpha=0.4,
    temp_T=0.9, use_tta=True,
    thr_scope="sequence", thr_reduce="median",
)

Xk_base = build_Xk_with_fixed_msoft(example_sequence, m_soft, K_VAL=1, K_MAX=5)
print("Xk_base shape:", Xk_base.shape)

def sanity_check_Xk(Xk):
    rgb, kseq, mseq = Xk[..., :3], Xk[..., 3:4], Xk[..., 4:5]
    tf.print("[chk] RGB min/max:", tf.reduce_min(rgb), tf.reduce_max(rgb))
    tf.print("[chk] k mean per-t:", tf.reduce_mean(kseq, axis=[0,2,3,4]))
    tf.print("[chk] m mean per-t:", tf.reduce_mean(mseq, axis=[0,2,3,4]))
    tf.print("[chk] m cov>0.5 per-t:",
             tf.reduce_mean(tf.cast(mseq>0.5, tf.float32), axis=[0,2,3,4]))

sanity_check_Xk(Xk_base)

def gate_lesion_union_support_amplitude_pred(
    last01, pred01,
    prob_last_lesion, prob_pred_lesion,
    thr_anchor, tau=0.10,
    support_mode="softthr",
    amplitude_mode="raw",
    prob_last_exclude=None,
    prob_pred_exclude=None,
    exclude_thr=0.5,
    m_floor=0.00,
    union_kind="sum_clip",
    gamma_amp=0.75,
    support_power=1.5,
):
    # (1) sanitize
    last01 = np.clip(np.asarray(last01, np.float32), 0.0, 1.0)
    pred01 = np.clip(np.asarray(pred01, np.float32), 0.0, 1.0)
    L_last = np.clip(np.asarray(prob_last_lesion, np.float32), 0.0, 1.0)
    L_pred = np.clip(np.asarray(prob_pred_lesion, np.float32), 0.0, 1.0)

    # (2) support maps
    if support_mode == "raw":
        S_last, S_pred = L_last, L_pred
    elif support_mode == "softthr":
        S_last = 1.0 / (1.0 + np.exp(-(L_last - float(thr_anchor)) / float(tau)))
        S_pred = 1.0 / (1.0 + np.exp(-(L_pred - float(thr_anchor)) / float(tau)))
    else:
        raise ValueError("support_mode must be 'softthr' or 'raw'")

    # (3) exclude on support
    if prob_last_exclude is not None:
        ex_last = (np.asarray(prob_last_exclude, np.float32) >= float(exclude_thr)).astype(np.float32)
        S_last *= (1.0 - ex_last)
    if prob_pred_exclude is not None:
        ex_pred = (np.asarray(prob_pred_exclude, np.float32) >= float(exclude_thr)).astype(np.float32)
        S_pred *= (1.0 - ex_pred)

    # (4) union support
    if union_kind == "sum_clip":
        S = np.clip(S_last + S_pred, 0.0, 1.0)
    elif union_kind == "or_prob":
        S = 1.0 - (1.0 - S_last) * (1.0 - S_pred)
        S = np.clip(S, 0.0, 1.0)
    elif union_kind == "and_prob":
        S = np.clip(S_last * S_pred, 0.0, 1.0)
    else:
        raise ValueError("union_kind must be 'sum_clip' or 'or_prob'")

    if support_power is not None and support_power != 1.0:
        S = np.power(np.clip(S, 0.0, 1.0), float(support_power)).astype(np.float32)

    # (5) amplitude
    if amplitude_mode == "raw":
        A = L_pred
    elif amplitude_mode == "softthr":
        A = 1.0 / (1.0 + np.exp(-(L_pred - float(thr_anchor)) / float(tau)))
    elif amplitude_mode == "consensus":
        A = np.sqrt(np.clip(L_last * L_pred, 0.0, 1.0))
    else:
        raise ValueError("amplitude_mode must be 'raw' or 'softthr'")

    if gamma_amp is not None and gamma_amp != 1.0:
        A = np.power(np.clip(A, 0.0, 1.0), float(gamma_amp)).astype(np.float32)

    # (6) exclude on amplitude
    if prob_pred_exclude is not None:
        ex_pred = (np.asarray(prob_pred_exclude, np.float32) >= float(exclude_thr)).astype(np.float32)
        A *= (1.0 - ex_pred)

    # (7) gate
    M = S * A

    # (8) floor
    if m_floor and m_floor > 0:
        M = np.maximum(M, float(m_floor)) * (S > 0.05).astype(np.float32)

    # Remove out-blending here
    return M, S, A

# ---------- Basic transform ----------

def calibrate_thr_by_iou(m_soft_pred, bin_last, best_thr,
                         delta=0.05, step=0.01):
    """
    m_soft_pred : (H,W) or (H,W,1) soft mask of predicted image [0,1]
    bin_last    : (H,W) last-frame binary mask (0/1)
    best_thr    : global Dice-calibrated threshold from train/val
    delta       : search range (best_thr ± delta)
    """
    m = m_soft_pred[...,0] if m_soft_pred.ndim == 3 else m_soft_pred
    m = np.clip(m, 0, 1).astype(np.float32)
    H, W = m.shape

    low  = max(0.0, best_thr - delta)
    high = min(1.0, best_thr + delta)
    thrs = np.arange(low, high + 1e-6, step)

    def iou_score(bin_pred, bin_ref):
        inter = np.logical_and(bin_pred==1, bin_ref==1).sum()
        union = np.logical_or(bin_pred==1, bin_ref==1).sum() + 1e-6
        return inter / union

    best_t   = best_thr
    best_iou = -1.0

    for t in thrs:
        bin_pred = (m >= t).astype(np.uint8)
        iou = iou_score(bin_pred, bin_last)
        if iou > best_iou:
            best_iou = iou
            best_t   = float(t)

    return best_t, best_iou

np.save("cache_m_soft.npy", m_soft)
np.save("cache_m_bin.npy",  m_bin)
np.save("cache_m_info.npy", info)

print("✅ m_soft cached.")

m_soft_fixed = np.load("cache_m_soft.npy")

def make_masks_changefirst(prob_all, THR_CHANGE, THR_WOUND=None, healed_margin=0.02):
    p = np.asarray(prob_all, np.float32)

    if p.ndim == 4:
        # Enforcement policy 1) Raise error (recommended: catch mistakes immediately)
        raise ValueError(f"prob_all must be (H,W,C), got {p.shape} (did you forget [0]?)")
        # Enforcement policy 2) Automatically select the first sample (replace below if preferred)
        # p = p[0]

    if p.ndim != 3:
        raise ValueError(f"prob_all must be 3D (H,W,C), got {p.shape}")

    p_w = p[..., WOUND_IDX]
    p_e = p[..., ESCHAR_IDX]
    p_h = p[..., HEALED_IDX]

    # healed lock
    m_healed = (p_h >= np.maximum(p_w, p_e) + healed_margin)

    # change-first
    p_chg  = np.maximum(p_w, p_e)
    m_chg  = (p_chg >= float(THR_CHANGE)) & (~m_healed)

    # split inside change -> wound ⊆ change guaranteed
    if THR_WOUND is None:
        m_wound  = m_chg & (p_w >= p_e)
    else:
        m_wound  = m_chg & (p_w >= p_e) & (p_w >= float(THR_WOUND))

    m_eschar = m_chg & (~m_wound)

    return m_chg, m_wound, m_eschar, m_healed

rgb  = Xk_base[..., :3]
kseq = Xk_base[..., 3:4]
mseq = Xk_base[..., 4:5]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def g_from_K(K, s=5.7, p=2.0, clip=(1.0, 8.0)):
    """
    K in [0,1] -> g in [1, 1+s] (safely bounded by clip)
    """
    K = float(np.clip(K, 0.0, 1.0))
    g = 1.0 + float(s) * (K ** float(p))
    if clip is not None:
        g = float(np.clip(g, float(clip[0]), float(clip[1])))
    return g

def alpha_min_from_K(K, base=0.06, top=0.18, p=2.0, clip=(0.02, 0.40)):
    """
    K=0 -> base, K=1 -> top
    Recommend starting top at 0.16~0.20 (0.28 risks flat/collapse)
    """
    K = float(np.clip(K, 0.0, 1.0))
    a = base + (top - base) * (K ** float(p))
    if clip is not None:
        a = float(np.clip(a, float(clip[0]), float(clip[1])))
    return a

# ============================================================
# make_alpha_K: patched version reflecting steps (1~3) as-is
#  - (1) Keep quantile gate
#  - (2) K is reflected only in alpha progression strength (K^gamma)
#  - (3) alpha_min ensures floor only at the center via g_chg^q
# ============================================================
def make_alpha_K(
    prob_all,
    K,
    # --- indices ---
    wound_idx,
    eschar_idx,
    healed_idx=None,          # Not used directly in alpha (extend if needed)
    exclude_idx=None,

    # --- auto-gate via quantiles on p_chg ---
    q_wound_gate=0.80,
    q_change_gate=0.95,

    # --- sigmoid widths ---
    s_open=0.03,
    s_chg=0.05,

    # --- alpha shaping ---
    gamma=1.5,                 # K^gamma (K is reflected only here in progression)
    alpha_min_base=0.06,        # alpha_min_from_K base
    alpha_min_top=0.18,         # alpha_min_from_K top
    alpha_min_p=2.0,            # curvature
    alpha_min_q=3.0,            # g_chg coupling exponent (recommend 2~4)
    gain=1.0,

    # --- optional soft suppression ---
    exclude_soft_strength=1.0,
    clip01=True,

    debug=False,
):
    """
    prob_all: (H,W,C) softmax prob recommended
    Returns:
      - debug=False: alpha (H,W,1)
      - debug=True : (alpha, dbg)
    """
    p = np.asarray(prob_all, dtype=np.float32)
    if p.ndim == 4:
        raise ValueError(f"prob_all must be (H,W,C). Got {p.shape}. Did you forget [0]?")
    if p.ndim != 3:
        raise ValueError(f"prob_all must be 3D (H,W,C). Got {p.shape}.")

    Kc = float(np.clip(K, 0.0, 1.0))

    pw = p[..., int(wound_idx)]
    pe = p[..., int(eschar_idx)]
    p_chg = np.maximum(pw, pe)  # change-likelihood base

    # (1) quantile thresholds
    thr_wound_gate  = float(np.quantile(p_chg, float(q_wound_gate)))
    thr_change_gate = float(np.quantile(p_chg, float(q_change_gate)))
    if thr_change_gate < thr_wound_gate:
        thr_change_gate = thr_wound_gate

    g_open = sigmoid((p_chg - thr_wound_gate) / float(s_open))
    g_chg  = sigmoid((p_chg - thr_change_gate) / float(s_chg))

    # (2) progression: K only here
    alpha_raw = (Kc ** float(gamma)) * g_chg  # (H,W)

    # (3) alpha_min: "center-only" floor via g_chg coupling
    alpha_min_eff = alpha_min_from_K(
        Kc,
        base=float(alpha_min_base),
        top=float(alpha_min_top),
        p=float(alpha_min_p),
        clip=(0.02, 0.40),
    )
    alpha_floor = float(alpha_min_eff) * (g_chg ** float(alpha_min_q))  # Survives only at the center

    alpha = np.maximum(alpha_raw, alpha_floor)

    # gain + open candidate weighting
    alpha *= float(gain)
    alpha *= g_open

    # optional soft exclude
    if exclude_idx is not None and float(exclude_soft_strength) > 0:
        ex = p[..., int(exclude_idx)]
        alpha *= (1.0 - np.clip(ex, 0.0, 1.0) * float(exclude_soft_strength))

    if clip01:
        alpha = np.clip(alpha, 0.0, 1.0)

    alpha = alpha[..., None]  # (H,W,1)

    if not debug:
        return alpha

    dbg = {
        "K": float(Kc),
        "p_chg_quantiles": np.quantile(p_chg, [0, .5, float(q_wound_gate), float(q_change_gate), .99]).astype(np.float32),
        "thr_wound_gate": float(thr_wound_gate),
        "thr_change_gate": float(thr_change_gate),
        "g_open_mean": float(g_open.mean()),
        "g_open_frac_gt_0.5": float((g_open > 0.5).mean()),
        "g_chg_mean": float(g_chg.mean()),
        "g_chg_frac_gt_0.5": float((g_chg > 0.5).mean()),
        "alpha_min_eff": float(alpha_min_eff),
        "alpha_floor_mean": float(alpha_floor.mean()),
        "alpha_mean": float(alpha.mean()),
        "alpha_frac_gt_0.5": float((alpha[..., 0] > 0.5).mean()),
    }
    return alpha, dbg

def _w_focus_from_prob_all_argmax(
    prob_all_pred,         # (H,W,C)
    WOUND_IDX,
    HEALED_IDX,
    ESCHAR_IDX,
    lesion_mode="argmax",  # "argmax" only (thr-free)
    margin=0.03,           # e.g., 0.05 (optional)
):
    """
    Returns:
      lesion_mask: (H,W) bool  - whether each pixel is classified as one of 3 lesion types (w/h/e)
      w_focus:     (H,W) bool  - argmax==wound inside lesion (+ optional margin)
      arg:         (H,W) int   - argmax label map
      confident:   (H,W) bool or None
    """
    P = prob_all_pred.astype(np.float32)
    arg = np.argmax(P, axis=-1)  # (H,W)

    lesion_idxs = [WOUND_IDX, HEALED_IDX, ESCHAR_IDX]
    lesion_mask = np.isin(arg, lesion_idxs)

    w_focus = (arg == WOUND_IDX) & lesion_mask

    confident = None
    if margin is not None and margin > 0:
        top1 = np.max(P, axis=-1)
        top2 = np.partition(P, -2, axis=-1)[..., -2]
        confident = (top1 - top2) >= float(margin)
        w_focus = w_focus & confident

    return lesion_mask, w_focus, arg, confident

eps = 1e-8

def as_bool(m, thr=0.5):
    m = np.asarray(m)
    if m.dtype == np.bool_:
        return m
    return (m > thr)

def lerp(a, b, t):
    return a * (1.0 - t) + b * t

# ============================================================
# postprocess_allow_protect: full patch inserting steps (1~3)
#  - allow/protect/uncertain decision logic remains unchanged
#  - K is reflected only in the allow blending alpha
#  - w_focus gate preserved + relax fallback preserved
# ============================================================
def postprocess_allow_protect(
    pred01, last01,
    prob_all_last, prob_all_pred,
    m_soft_last_lesion, m_soft_pred_lesion,
    THR_SUPPORT, THR_CHANGE,
    CHANGE_IDXS, STABLE_IDXS,
    # --- indices for w_focus ---
    WOUND_IDX, HEALED_IDX, ESCHAR_IDX,
    W_FOCUS_MARGIN=0.03,          # Used only when building w_focus (no hard restriction)
    W_FOCUS_TO_UNCERTAIN=False,   # Enable only when needed
    # ---- option knobs ----
    pSTB_out=0.50,
    pSTB_in_strict=0.80,
    pSTB_in=0.65,
    pCHG_low=0.10,
    # ---- alpha/K ----
    alpha01=None,
    K=None,
    # ---- K->alpha saturation ----
    K_s=5.7,
    K_p=2.0,
    g_clip=(1.0, 8.0),

    # ---- w_focus as SOFT weight knobs ----
    w_focus_mode="mix",           # "argmax" | "wrel" | "mix"
    wrel_tau=0.25,                # w_rel sigmoid center (0.20~0.35)
    wrel_s=0.08,                  # w_rel sigmoid slope (0.05~0.12)
    wmix=0.70,                    # mix weight: 0.7*wrel + 0.3*argmax
    w_focus_floor=0.15,           # Floor to prevent dying to 0 (0.0~0.2)
    w_focus_clip=(0.0, 1.0),

    # ==========================================================
    # Scenario control (Tx redefined as optimistic vs conservative)
    # ==========================================================
    tx_norm=0.5,                  # 0=conservative, 1=optimistic
    tx_strength=1.0,              # 0: same as before, 1: default effect
    # (conservative, optimistic) : alpha scale
    tx_alpha_scale=(0.75, 1.25),
    # (conservative, optimistic) : K->alpha g scale
    tx_g_scale=(0.85, 1.20),
    # (conservative_shift, optimistic_shift) added to THR_CHANGE
    tx_thr_change_shift=(+0.05, -0.02),
    # (conservative_shift, optimistic_shift) added to pSTB_out (background protect)
    tx_pSTB_out_shift=(+0.05, +0.10),
    # (conservative_shift, optimistic_shift) added to pCHG_low (stability-based protect in ROI)
    tx_pCHG_low_shift=(+0.05, -0.02),
):
    """
    Scenario interpretation:
      tx_norm=0.0 -> conservative (more skeptical, more stable-keep, less blending)
      tx_norm=1.0 -> optimistic   (allows more change where justified, but protects background stronger)
    Notes:
      - We DO NOT hard-gate ROI topology aggressively. We only shift thresholds mildly and scale alpha.
      - This keeps behavior stable and avoids 'mask flicker' across trials.
    """
    dbg = {}

    # -------------------------
    # Scenario interpolation
    # -------------------------
    t = float(np.clip(tx_norm, 0.0, 1.0))
    t = float(np.clip(t * float(tx_strength), 0.0, 1.0))

    thr_change_eff = float(THR_CHANGE) + lerp(float(tx_thr_change_shift[0]), float(tx_thr_change_shift[1]), t)
    thr_change_eff = float(np.clip(thr_change_eff, 0.0, 1.0))

    pSTB_out_eff = float(pSTB_out) + lerp(float(tx_pSTB_out_shift[0]), float(tx_pSTB_out_shift[1]), t)
    pSTB_out_eff = float(np.clip(pSTB_out_eff, 0.0, 1.0))

    pCHG_low_eff = float(pCHG_low) + lerp(float(tx_pCHG_low_shift[0]), float(tx_pCHG_low_shift[1]), t)
    pCHG_low_eff = float(np.clip(pCHG_low_eff, 0.0, 1.0))

    alpha_scale = lerp(float(tx_alpha_scale[0]), float(tx_alpha_scale[1]), t)
    g_scale = lerp(float(tx_g_scale[0]), float(tx_g_scale[1]), t)

    # ROI union (kept stable; do NOT scenario-shift THR_SUPPORT here unless you have strong reason)
    roi = ((m_soft_last_lesion >= float(THR_SUPPORT)) |
           (m_soft_pred_lesion >= float(THR_SUPPORT)))

    # probs
    pCHG_last = np.max(prob_all_last[..., list(CHANGE_IDXS)].astype(np.float32), axis=-1)
    pCHG_pred = np.max(prob_all_pred[..., list(CHANGE_IDXS)].astype(np.float32), axis=-1)
    pSTB_last = np.max(prob_all_last[..., list(STABLE_IDXS)].astype(np.float32), axis=-1)

    # ---- protect ----
    protect = (~roi) & (pSTB_last >= float(pSTB_out_eff))
    protect |= roi & (pSTB_last >= float(pSTB_in_strict))
    protect |= roi & (pSTB_last >= float(pSTB_in)) & (np.maximum(pCHG_last, pCHG_pred) <= float(pCHG_low_eff))

    # ---- allow (base) ----
    # NOTE: allow topology changes only mildly via thr_change_eff
    pCHG_strong = (pCHG_pred >= float(thr_change_eff) * 1.5)
    allow = roi & ((pCHG_last >= float(thr_change_eff)) | (pCHG_pred >= float(thr_change_eff)))
    protect = protect & (~pCHG_strong)
    allow = allow & (~protect)

    # ---- uncertain (base) ----
    uncertain = roi & (~allow) & (~protect)

    # ==========================================================
    # w_focus: no HARD restriction -> create SOFT weight only
    # ==========================================================
    lesion_mask, w_focus_arg, arg_pred, confident = _w_focus_from_prob_all_argmax(
        prob_all_pred,
        WOUND_IDX=WOUND_IDX,
        HEALED_IDX=HEALED_IDX,
        ESCHAR_IDX=ESCHAR_IDX,
        margin=W_FOCUS_MARGIN,
    )
    w_focus_arg = w_focus_arg.astype(np.float32)

    # (b) w_rel-based soft weight
    P = prob_all_pred.astype(np.float32)
    pw = P[..., int(WOUND_IDX)]
    ph = P[..., int(HEALED_IDX)]
    pe = P[..., int(ESCHAR_IDX)]
    w_rel = pw / (pw + ph + pe + 1e-6)  # 0..1
    w_focus_rel = sigmoid((w_rel - float(wrel_tau)) / float(wrel_s))  # 0..1

    # (c) combine
    if w_focus_mode == "argmax":
        w_focus_w = w_focus_arg
    elif w_focus_mode == "wrel":
        w_focus_w = w_focus_rel
    else:  # "mix"
        w_focus_w = float(wmix) * w_focus_rel + (1.0 - float(wmix)) * w_focus_arg

    # Meaningful only inside ROI
    w_focus_w = w_focus_w * roi.astype(np.float32)

    # Meaningful only in allow/uncertain (protect is fixed to last)
    w_focus_w = np.clip(w_focus_w, float(w_focus_clip[0]), float(w_focus_clip[1]))
    if w_focus_floor is not None and float(w_focus_floor) > 0:
        w_focus_w = np.maximum(w_focus_w, float(w_focus_floor) * roi.astype(np.float32))

    # Reorganize uncertain
    if W_FOCUS_TO_UNCERTAIN:
        uncertain = roi & (~protect) & (~allow)
    else:
        uncertain = roi & (~allow) & (~protect)

    # ==========================================================
    # Generate output: protect keeps last, allow uses alpha-blend
    # ==========================================================
    out = last01.copy()

    if alpha01 is None:
        # Without alpha, replace as before
        out[allow] = pred01[allow]
        g_used = None
        a_mean = None
        a_frac = None
        a_eff_mean = None
        a_scale_used = None
    else:
        a = np.asarray(alpha01, np.float32)
        if a.ndim == 3:
            a = a[..., 0]
        a = np.clip(a, 0.0, 1.0)

        # K saturation (enhance visibility within 0..1)
        g_used = None
        if K is not None:
            g_used = g_from_K(K, s=K_s, p=K_p, clip=g_clip)
            # scenario scaling of g (conservative -> smaller g, optimistic -> larger g)
            g_used = float(g_used) * float(g_scale)
            a = 1.0 - np.power((1.0 - a), float(g_used))

        # Multiply w_focus SOFT weight into alpha only (do not reduce allow)
        a_eff = a * w_focus_w  # (H,W)

        # ✅ scenario scaling of alpha blending (conservative -> less blend, optimistic -> more blend)
        a_scale_used = float(alpha_scale)
        a_eff = np.clip(a_eff * a_scale_used, 0.0, 1.0)

        # blend on allow
        out[allow] = (last01[allow] + (pred01[allow] - last01[allow]) * a_eff[allow, None])

        a_mean = float(a.mean())
        a_frac = float((a > 0.5).mean())
        a_eff_mean = float(a_eff.mean())

    dbg.update({
        "roi_frac": float(roi.mean()),
        "allow_frac": float(allow.mean()),
        "protect_frac": float(protect.mean()),
        "uncertain_frac": float(uncertain.mean()),

        "THR_SUPPORT": float(THR_SUPPORT),
        "THR_CHANGE": float(THR_CHANGE),
        "thr_change_eff": float(thr_change_eff),

        "pSTB_out": float(pSTB_out),
        "pSTB_out_eff": float(pSTB_out_eff),
        "pSTB_in_strict": float(pSTB_in_strict),
        "pSTB_in": float(pSTB_in),
        "pCHG_low": float(pCHG_low),
        "pCHG_low_eff": float(pCHG_low_eff),

        # --- focus diagnostics ---
        "lesion_argmax_frac": float(lesion_mask.mean()),
        "w_focus_arg_frac": float(w_focus_arg.mean()),
        "w_rel_mean_roi": float(w_rel[roi].mean()) if roi.any() else np.nan,
        "w_focus_w_mean": float(w_focus_w.mean()),
        "w_focus_mode": str(w_focus_mode),
        "w_focus_margin": None if W_FOCUS_MARGIN is None else float(W_FOCUS_MARGIN),

        # --- alpha / K diagnostics ---
        "alpha_mean": None if a_mean is None else float(a_mean),
        "alpha_frac_gt_0.5": None if a_frac is None else float(a_frac),
        "alpha_eff_mean": None if a_eff_mean is None else float(a_eff_mean),
        "g_used": None if g_used is None else float(g_used),

        # --- scenario diagnostics ---
        "tx_norm": float(tx_norm),
        "tx_strength": float(tx_strength),
        "alpha_scale_used": None if alpha01 is None else float(a_scale_used),
        "g_scale": None if K is None else float(g_scale),
    })
    if confident is not None:
        dbg["w_focus_confident_frac"] = float(confident.mean())

    return out, (roi, allow, protect, uncertain), dbg


def schedule_postproc_params(K: float):
    # K does not change the gate here (preserve temporal distance semantics)
    return dict(
        pSTB_out=0.50,
        pSTB_in_strict=0.85,
        pSTB_in=0.65,
        pCHG_low=0.15,   # Recommend fixing to the value that worked well at K=0.2
    )

def diag_wound_focus_panels(
    pred01_final,                 # (H,W,3) in [0,1]: final image (visualization base)
    prob_all_pred,                # (H,W,C) softmax
    roi=None,                     # (H,W) bool
    prob_all_anchor=None,         # (H,W,C) softmax (optional: anchor prior)
    WOUND_IDX=1, HEALED_IDX=3, ESCHAR_IDX=2,
    margin=0.03,                  # argmax w_focus margin
    wrel_thr=0.30,                # Relaxed w_focus: w_rel threshold (tune 0.25~0.40)
    lam_anchor=0.35,              # Anchor prior blending strength (0.2~0.5)
    title="diag",
):
    P = prob_all_pred.astype(np.float32)
    pw = P[..., int(WOUND_IDX)]
    ph = P[..., int(HEALED_IDX)]
    pe = P[..., int(ESCHAR_IDX)]

    # 0) roi
    if roi is None:
        roi = np.ones(pw.shape, dtype=bool)
    else:
        roi = roi.astype(bool)

    # 1) argmax map
    arg = np.argmax(P, axis=-1)

    # 2) Lesion mask (based on argmax)
    lesion_mask = np.isin(arg, [int(WOUND_IDX), int(HEALED_IDX), int(ESCHAR_IDX)])

    # 3) argmax w_focus + margin
    #    (top1-top2) margin
    top1 = np.max(P, axis=-1)
    top2 = np.partition(P, -2, axis=-1)[..., -2]
    confident = (top1 - top2) >= float(margin) if (margin is not None and margin > 0) else np.ones_like(pw, bool)

    w_focus_arg = (arg == int(WOUND_IDX)) & lesion_mask & confident & roi

    # 4) Relaxed w_focus: relative woundiness inside lesion
    w_rel = pw / (pw + ph + pe + 1e-6)
    w_focus_wrel = roi & (w_rel >= float(wrel_thr))

    # 5) Anchor prior blending (optional)
    pw_mix = None
    w_rel_mix = None
    w_focus_mix = None
    if prob_all_anchor is not None:
        A = prob_all_anchor.astype(np.float32)
        pw_last = A[..., int(WOUND_IDX)]
        pw_mix = np.maximum(pw, float(lam_anchor) * pw_last)
        w_rel_mix = pw_mix / (pw_mix + ph + pe + 1e-6)
        w_focus_mix = roi & (w_rel_mix >= float(wrel_thr))

    # 6) ROI overlay helper
    def overlay_roi_on_gray(gray, roi_bool, alpha=0.25):
        # gray: (H,W) in [0,1]
        g = np.clip(gray, 0, 1)
        rgb = np.stack([g, g, g], axis=-1)
        # Slightly brighten (whiten) the ROI region to show its location
        rgb[roi_bool] = np.clip(rgb[roi_bool] * (1 - alpha) + alpha * 1.0, 0, 1)
        return rgb

    # 7) For argmap visualization (highlight w/h/e only)
    #    0:other, 1:wound, 2:healed, 3:eschar
    arg_vis = np.zeros_like(arg, dtype=np.int32)
    arg_vis[arg == int(WOUND_IDX)] = 1
    arg_vis[arg == int(HEALED_IDX)] = 2
    arg_vis[arg == int(ESCHAR_IDX)] = 3

    # --- plot ---
    plt.figure(figsize=(24, 10))
    k = 1

    def ax_im(idx, img, ttl, cmap=None, vmin=None, vmax=None):
        plt.subplot(3, 4, idx)
        if img.ndim == 2:
            plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            plt.imshow(img)
        plt.title(ttl)
        plt.axis("off")

    ax_im(1, pred01_final, f"{title}: pred01_final")
    ax_im(2, overlay_roi_on_gray(pw, roi), "pw (ROI overlay)", cmap=None)
    ax_im(3, overlay_roi_on_gray(ph, roi), "ph (ROI overlay)", cmap=None)
    ax_im(4, overlay_roi_on_gray(pe, roi), "pe (ROI overlay)", cmap=None)

    ax_im(5, w_rel, f"w_rel = pw/(pw+ph+pe) (thr={wrel_thr})", cmap="gray", vmin=0, vmax=1)
    ax_im(6, confident.astype(np.float32), f"confident (top1-top2 >= {margin})", cmap="gray", vmin=0, vmax=1)
    ax_im(7, arg_vis, "argmax label (0=other,1=W,2=H,3=E)", cmap="tab10", vmin=0, vmax=9)
    ax_im(8, (lesion_mask & roi).astype(np.float32), "lesion_mask(arg in W/H/E) & ROI", cmap="gray", vmin=0, vmax=1)

    ax_im(9, w_focus_arg.astype(np.float32), "w_focus_arg (argmax+w_margin) & ROI", cmap="gray", vmin=0, vmax=1)
    ax_im(10, w_focus_wrel.astype(np.float32), f"w_focus_wrel (w_rel>= {wrel_thr}) & ROI", cmap="gray", vmin=0, vmax=1)

    if prob_all_anchor is not None:
        ax_im(11, pw_mix, f"pw_mix = max(pw, {lam_anchor}*pw_last)", cmap="gray", vmin=0, vmax=1)
        ax_im(12, w_focus_mix.astype(np.float32), f"w_focus_mix (w_rel_mix>= {wrel_thr})", cmap="gray", vmin=0, vmax=1)
    else:
        ax_im(11, np.zeros_like(pw), "pw_mix (anchor not provided)", cmap="gray", vmin=0, vmax=1)
        ax_im(12, np.zeros_like(pw), "w_focus_mix (anchor not provided)", cmap="gray", vmin=0, vmax=1)

    plt.tight_layout()
    plt.show()

    # --- Numeric summary (statistics inside ROI) ---
    roi_idx = roi
    def mean_in(x): return float(np.mean(x[roi_idx])) if roi_idx.any() else float("nan")

    print(f"[{title}] ROI frac:", float(np.mean(roi_idx)))
    print(f"[{title}] mean pw/ph/pe in ROI:", mean_in(pw), mean_in(ph), mean_in(pe))
    print(f"[{title}] mean w_rel in ROI:", mean_in(w_rel))
    print(f"[{title}] w_focus_arg frac:", float(np.mean(w_focus_arg)))
    print(f"[{title}] w_focus_wrel frac:", float(np.mean(w_focus_wrel)))
    if prob_all_anchor is not None:
        print(f"[{title}] w_focus_mix frac:", float(np.mean(w_focus_mix)))

