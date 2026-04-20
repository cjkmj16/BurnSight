"""
BurnSight evaluation utilities — K-sweep, Lab analysis, delta report
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

def _to01(x):
    x = np.asarray(x, dtype=np.float32)
    return np.clip(x, 0.0, 1.0)

def rgb01_to_lab(img01):
    """img01: [H,W,3] in [0,1] → LAB (float32)"""
    x8 = (_to01(img01) * 255).astype(np.uint8)
    return cv2.cvtColor(x8, cv2.COLOR_RGB2LAB).astype(np.float32)

def make_roi_from_anchor_and_pred_prob(
    m_soft_anchor_lesion_fixed,   # (H,W) float in [0,1], K-independent
    m_soft_pred_lesion,           # (H,W) float in [0,1], K-dependent
    thr_support: float,
    union_kind: str = "and_prob", # "anchor_only" | "and_prob" | "or_prob"
):
    A = np.asarray(m_soft_anchor_lesion_fixed, np.float32)
    P = np.asarray(m_soft_pred_lesion, np.float32)

    if union_kind == "anchor_only":
        roi = (A >= float(thr_support))
    elif union_kind == "and_prob":
        # Most recommended: fix ROI shape/position with anchor,
        # and use only the region where pred strongly agrees inside it
        roi = (A >= float(thr_support)) & (P >= float(thr_support))
    elif union_kind == "or_prob":
        roi = (A >= float(thr_support)) | (P >= float(thr_support))
    else:
        raise ValueError(f"Unknown union_kind={union_kind}")

    return roi

# ---------- Δa*, ΔE statistics ----------

def lab_delta_stats(last01, pred01, M01, on_none="fallback"):
    """
    on_none: "raise" | "fallback"
    """
    if M01 is None:
        if on_none == "raise":
            raise ValueError(
                "M01 is None. ROI mask generation failed upstream. "
                "Check lesion_from_prob_all_fn / seg outputs / thr_scope logic."
            )
        # fallback: ROI=full image, OUT=0
        w = np.ones(last01.shape[:2], dtype=np.float32)
        out = np.zeros_like(w)
    else:
        if getattr(M01, "ndim", None) is None:
            M01 = np.asarray(M01)
        w = M01[..., 0] if M01.ndim == 3 else M01
        w = np.clip(w.astype(np.float32), 0, 1)
        out = 1.0 - w

    Lab_l = rgb01_to_lab(last01)
    Lab_p = rgb01_to_lab(pred01)
    d     = Lab_p - Lab_l
    dE    = np.sqrt(np.sum(d**2, axis=-1))

    def wmean(a, ww):
        s = (a * ww).sum()
        z = ww.sum() + 1e-6
        return float(s / z)

    return {
        "dL_roi": wmean(d[...,0], w),   "dL_out": wmean(d[...,0], out),
        "da_roi": wmean(d[...,1], w),   "da_out": wmean(d[...,1], out),
        "dE_roi": wmean(dE, w),         "dE_out": wmean(dE, out),
        "da_map": d[...,1],             "dE_map": dE
    }


# ---------- Panel plot ----------
def delta_stats_in_roi(last01, pred01, M01):
    """
    last01, pred01: (H,W,3) in [0,1]
    M01: (H,W,1) or (H,W) mask in [0,1] or bool
    return: st (dict) from lab_delta_stats
    """
    if M01 is None:
        raise ValueError("M01 must not be None")

    M = np.asarray(M01)
    if M.ndim == 2:
        M = M[..., None]  # (H,W,1)

    # Guard against empty ROI (if needed)
    roi_frac = float(np.mean(M[..., 0] > 0.5)) if M.dtype != np.bool_ else float(np.mean(M[..., 0]))
    if roi_frac < 1e-6:
        # Handle NaN here if lab_delta_stats cannot process empty ROI
        # (This block can be removed if already handled)
        st = lab_delta_stats(last01, pred01, M)  # Can be further optimized if da_map/dE_map are not needed
        st["roi_frac"] = roi_frac
        return st

    st = lab_delta_stats(last01, pred01, M)
    st["roi_frac"] = roi_frac
    return st

def plot_delta_panel(last01, pred01, M01, title="Δ panel", draw=True):
    # Unify all computation here
    st = delta_stats_in_roi(last01, pred01, M01)

    if not draw:
        return st

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(last01); axs[0].set_title("last"); axs[0].axis('off')
    axs[1].imshow(pred01); axs[1].set_title("pred"); axs[1].axis('off')
    im2 = axs[2].imshow(st["da_map"], cmap='bwr'); axs[2].set_title("Δa* (redness)"); axs[2].axis('off')
    im3 = axs[3].imshow(st["dE_map"], cmap='magma'); axs[3].set_title("ΔE"); axs[3].axis('off')
    plt.colorbar(im2, ax=axs[2], fraction=0.046)
    plt.colorbar(im3, ax=axs[3], fraction=0.046)
    fig.suptitle(
        f'{title} | Δa*_roi={st["da_roi"]:.3f}, Δa*_out={st["da_out"]:.3f}, ΔE_roi={st["dE_roi"]:.3f}',
        y=1.02
    )
    plt.tight_layout()
    return st

# ---------- K channel manipulation / prediction ----------
def _set_k_channel(Xk, k):
    Xk2 = Xk.copy()
    Xk2[..., 3:4] = float(k)
    return Xk2

# ---------- K-sweep direction report ----------
def k_sweep_with_fixed_roi(
    creator,
    example_sequence01,
    m_soft_fixed,
    K_vals=(0.2, 0.6, 1.0),
    K_MAX=1.0,
):
    last01 = example_sequence01[0, -1]  # (H,W,3) in [0,1]

    da_list, dE_list = [], []

    for k in K_vals:
        Xk = build_Xk_with_fixed_msoft(
            example_sequence01, m_soft_fixed, k, K_MAX
        )

        y_pred, _ = creator.predict(Xk, verbose=0)
        pred01 = np.clip((y_pred[0] + 1.0) / 2.0, 0, 1)

        st = lab_delta_stats(last01, pred01, m_soft_fixed[0, -1, ..., 0])
        da_list.append(st["da_roi"])
        dE_list.append(st["dE_roi"])

    return {
        "K": np.array(K_vals),
        "da_roi": np.array(da_list),
        "dE_roi": np.array(dE_list),
    }

def plot_k_sweep(rep, expect="improve", title=None):
    """
    expect="improve": K↑ → expect Δa*_roi more positive (redness increase)
    expect="worsen":  K↑ → expect Δa*_roi more negative (redness decrease)
    """
    K, y = rep["K"], rep["da_roi"]
    plt.figure(figsize=(4,3))
    plt.plot(K, y, 'o-'); plt.axhline(0, ls='--')
    trend = "↑ (more positive)" if expect=="improve" else "↓ (more negative)"
    if title is None:
        plt.title(f"Δa*_roi vs K (expect {trend})")
    else:
        plt.title(f"{title} | expect {trend}")

    plt.xlabel("K")
    plt.ylabel("Δa*_roi")
    plt.tight_layout()

def k_sweep_dir_report(
    creator, Xk, mask_model, lesion_from_prob_all_fn,
    THR,
    K_vals=(0.2, 0.6, 1.0),
    union_kind="and_prob",
    # Added (minimum): fixed anchor lesion input
    m_soft_anchor_lesion_fixed=None,
    prob_all_anchor_fixed=None,
    postproc_fn=None,
    postproc_kwargs_fn=None,   # K -> dict(pp knobs)
):
    rep = {
        "K": [],
        # Existing final statistics
        "da_roi": [],
        "dE_roi": [],
        # Added: raw statistics
        "da_roi_raw": [],
        "dE_roi_raw": [],
        # Debug
        "roi_frac": [],
        "allow_frac": [],
        "protect_frac": [],
        "uncertain_frac": [],
    }

    assert m_soft_anchor_lesion_fixed is not None, "Need fixed anchor lesion for stable ROI"
    assert prob_all_anchor_fixed is not None, "Need fixed anchor prob_all_last"
    assert postproc_fn is not None, "Need postprocess_allow_protect (or wrapper)"

    for k in K_vals:
        k = float(k)

        # 1) Creator
        Xk_k = _set_k_channel(Xk, k)
        y_pred, x_last = creator.predict(Xk_k, verbose=0)
        pred01_raw = np.clip((y_pred[0] + 1) / 2, 0, 1).astype(np.float32)
        last01     = np.clip((x_last[0] + 1) / 2, 0, 1).astype(np.float32)

        # 2) Seg on pred
        pred_pred = mask_model.predict((pred01_raw*2-1)[None, ...], verbose=0)
        prob_all_pred = get_prob_all_batch(pred_pred, assume="softmax")[0]
        m_soft_pred_lesion = lesion_from_prob_all_fn(prob_all_pred)

        # 3) ROI (same rule per K, anchor is fixed)
        roi = make_roi_from_anchor_and_pred_prob(
            m_soft_anchor_lesion_fixed=m_soft_anchor_lesion_fixed,
            m_soft_pred_lesion=m_soft_pred_lesion,
            thr_support=float(THR),
            union_kind=union_kind,
        )

        # 4) Build alpha (using your existing function as-is)
        alpha01, _dbg_alpha = make_alpha_K(
            prob_all_pred,
            K=k,
            wound_idx=WOUND_IDX,
            eschar_idx=ESCHAR_IDX,
            exclude_idx=EXCLUDE_IDX,
            debug=True,
        )

        # 5) postprocess (final)
        pp = postproc_kwargs_fn(k) if postproc_kwargs_fn is not None else {}
        pred01_final, (roi_pp, allow, protect, uncertain), dbg = postproc_fn(
            pred01=pred01_raw,
            last01=last01,
            prob_all_last=prob_all_anchor_fixed,                 # fixed
            prob_all_pred=prob_all_pred,
            m_soft_last_lesion=m_soft_anchor_lesion_fixed,       # fixed
            m_soft_pred_lesion=m_soft_pred_lesion,
            THR_SUPPORT=float(THR),
            THR_CHANGE=float(THR_CHANGE),
            CHANGE_IDXS=CHANGE_IDXS,
            STABLE_IDXS=STABLE_IDXS,
            WOUND_IDX=WOUND_IDX,
            HEALED_IDX=HEALED_IDX,
            ESCHAR_IDX=ESCHAR_IDX,
            alpha01=alpha01,
            K=k,
            **pp,
        )

        # 6) Evaluate RAW/FINAL with the same ROI
        #    (roi_pp inside postprocess is for internal logic,
        #     Use the roi built here as the 'evaluation ROI' for consistency)
        st_raw   = delta_stats_in_roi(last01, pred01_raw,  roi)
        st_final = delta_stats_in_roi(last01, pred01_final, roi)

        # 7) Record
        rep["K"].append(k)
        rep["da_roi_raw"].append(st_raw["da_roi"])
        rep["dE_roi_raw"].append(st_raw["dE_roi"])
        rep["da_roi"].append(st_final["da_roi"])
        rep["dE_roi"].append(st_final["dE_roi"])

        rep["roi_frac"].append(float(roi.mean()))
        rep["allow_frac"].append(float(allow.mean()))
        rep["protect_frac"].append(float(protect.mean()))
        rep["uncertain_frac"].append(float(uncertain.mean()))

    # Convert to numpy (match existing style)
    for key in ["K", "da_roi_raw", "dE_roi_raw", "da_roi", "dE_roi",
                "roi_frac", "allow_frac", "protect_frac", "uncertain_frac"]:
        rep[key] = np.array(rep[key], dtype=np.float32)

    return rep


def refiner_delta_report(
    predicted_image_m11,
    final_refined_m11,
    mask_model,
    lesion_from_prob_all_fn,
    THR=0.55,
):
    # --- (0) shape sanitize ---
    predicted_image_m11 = np.asarray(predicted_image_m11, dtype=np.float32)
    final_refined_m11   = np.asarray(final_refined_m11,   dtype=np.float32)

    if predicted_image_m11.ndim == 3:
        predicted_image_m11 = predicted_image_m11[None, ...]
    if final_refined_m11.ndim == 3:
        final_refined_m11 = final_refined_m11[None, ...]

    predicted_image_m11 = predicted_image_m11[..., :3]
    final_refined_m11   = final_refined_m11[..., :3]

    # 1) [0,1]
    pred01 = (predicted_image_m11 + 1.0) * 0.5
    ref01  = (final_refined_m11   + 1.0) * 0.5

    # 2) ROI mask