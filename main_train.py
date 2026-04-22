"""
BurnSight training entry point
— Global execution code ported from the original notebook as-is.
— Paths are automatically resolved by environment in src/config.py.
"""
try:
    from google.colab import drive
    drive.mount('/content/drive')
except ImportError:
    pass  # Local environment

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

  
try:
    import cupy as cp
except ImportError:
    pass
try:
    import albumentations as A
except ImportError:
    pass
try:
    import tensorflow_probability as tfp
except ImportError:
    pass

# ── Import src modules ─────────────────────────────────────
from src.config import *
from src.data.file_utils import *
from src.data.dataset import *
from src.data.augment import *
from src.models.layers import *
from src.models.encoder import *
from src.models.unet import *
from src.models.creator import *
from src.models.refiner import *
from src.losses.creator_losses import *
from src.losses.refiner_losses import *
from src.utils.metrics import *
from src.utils.image_utils import *
from src.utils.debug_utils import *
from src.inference.mask_utils import *
from src.inference.eval_utils import *
from src.inference.postprocess import *
from src.data.file_utils import collect_aug_pairs, collect_pairs_by_pid
import file_utils as fu

fu.aug_pairs   = collect_aug_pairs(config.AUG_DIR)
fu.pid_buckets = collect_pairs_by_pid(config.AUG_DIR)

if not fu.aug_pairs:
    raise RuntimeError("No pairs found. Check AUG_DIR path.")
    
# Below: global execution code from the original notebook (unchanged)
def temp_scale_prob(p, T=0.9):
    p = np.clip(p.astype(np.float32), 1e-6, 1-1e-6)
    logit = np.log(p/(1-p))
    return 1.0/(1.0 + np.exp(-logit/float(T)))

def train_preprocess_infer(x_1thw3, to_size=(64,64), assume_rgb=True):
    """
    Same preprocessing as training (seg_train):
      - Resize: AREA
      - Scale: [0,1] → [-1,1]
      - Color: keep RGB (set assume_rgb=False if input is OpenCV BGR)
    Input:  (1,T,H,W,3) float32/uint8 both accepted
    Output: (1,T,64,64,3) float32 in [-1,1]
    """
    x = np.asarray(x_1thw3)
    assert x.ndim == 5 and x.shape[0] == 1, "expect (1,T,H,W,3)"
    B, T, H, W, C = x.shape
    out = np.empty((B, T, to_size[0], to_size[1], 3), dtype=np.float32)

    for t in range(T):
        img = x[0, t]
        # uint8 → [0,1]
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        # Convert BGR→RGB if needed
        if not assume_rgb:
            img = img[..., ::-1]
        # Resize: AREA (same as training)
        img = cv2.resize(img, to_size[::-1], interpolation=cv2.INTER_AREA)
        # Scale to [-1,1] (same as scale_pair_to_m11 in training)
        img = img * 2.0 - 1.0
        out[0, t] = img
    return out

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

def _diag10(tag, last01, pred01_raw, pred01_constrained, roi01, alpha01=None):
    import numpy as np
    def _rng(x): return (float(np.min(x)), float(np.max(x)))
    def _mean(x): return float(np.mean(x))

    roi = np.asarray(roi01, np.float32)
    if roi.ndim == 3: roi = roi[..., 0]
    roi_bin = (roi > 0.5).astype(np.float32)
    roi_frac = _mean(roi_bin)

    if alpha01 is None:
        a_mean = float("nan")
        a_frac = float("nan")
    else:
        a = np.asarray(alpha01, np.float32)
        if a.ndim == 3: a = a[..., 0]
        a_mean = _mean(a)
        a_frac = _mean(a > 0.5)  # Fraction of 'strongly applied' region (for reference)

    diff_masking = _mean(np.abs(pred01_constrained - pred01_raw))
    diff_to_last = _mean(np.abs(pred01_constrained - last01))

    print(f"[{tag}] (a) last range={_rng(last01)} pred_raw range={_rng(pred01_raw)} pred_cons range={_rng(pred01_constrained)}")
    print(f"[{tag}] (b) ROI frac={roi_frac:.4f} ROI mean={_mean(roi):.4f}  alpha mean={a_mean:.4f} alpha frac(>0.5)={a_frac:.4f}")
    print(f"[{tag}] (b) blend |mean| (cons-raw)={diff_masking:.6f}  (cons-last)={diff_to_last:.6f}")
    print(f"[{tag}] (c) ROI empty? {roi_frac < 1e-6}")
    print(f"[{tag}] note: enforce stats on pred_cons consistently.")
    print(f"[{tag}] note: enforce roi01 fixed across K-sweep (cache/load ok).")
    print(f"[{tag}] note: log roi hash if you cache (np.sum/np.mean/np.var) to ensure identical reload.")
    print(f"[{tag}] note: when roi_frac < eps -> return NaN (do not plot).")
    print(f"[{tag}] note: alpha is temporal progression weight, not confidence.")

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

def id_transform_uint8(img_u8, *, down, blur, contrast, sat, jpeg_q, noise, rng_seed=0):
    """
    img_u8: (H,W,3) uint8
    returns: (H,W,3) uint8
    All steps are deterministic. (rng_seed is kept but not currently used)
    """
    x = img_u8

    H, W = x.shape[:2]

    # 1) Downscale -> upscale (deterministic)
    d = float(down)
    h2 = max(2, int(round(H * d)))
    w2 = max(2, int(round(W * d)))
    x = cv2.resize(x, (w2, h2), interpolation=cv2.INTER_AREA)
    x = cv2.resize(x, (W, H), interpolation=cv2.INTER_LINEAR)

    # 2) Blur (deterministic)
    k = int(blur)
    if k > 0:
        # Kernel must be odd: blur=1 -> 3, blur=2 -> 5, etc.
        ksize = 2 * k + 1
        x = cv2.GaussianBlur(x, (ksize, ksize), sigmaX=0.0)

    # 3) Contrast (deterministic): (x-mean)*c + mean
    c = float(contrast)
    xf = x.astype(np.float32)
    mean = xf.mean(axis=(0,1), keepdims=True)
    xf = (xf - mean) * c + mean
    xf = np.clip(xf, 0, 255)

    # 4) Saturation (deterministic): scale S in HSV
    s = float(sat)
    hsv = cv2.cvtColor(xf.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[...,1] = np.clip(hsv[...,1] * s, 0, 255)
    x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # 5) JPEG artifact (deterministic)
    q = int(jpeg_q)
    q = max(1, min(100, q))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(x, cv2.COLOR_RGB2BGR), encode_param)
    if ok:
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        x = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)

    # 6) Noise (deterministic): fixed-sigma Gaussian noise
    sig = float(noise)  # Interpret as sigma (roughly 0~10) here
    if sig > 0:
        n = np.random.default_rng(rng_seed).normal(0.0, sig, size=x.shape).astype(np.float32)
        xf = x.astype(np.float32) + n
        x = np.clip(xf, 0, 255).astype(np.uint8)

    return x

def build_id_transform_deterministic(p):
    tfms = []

    # 1) Downscale (down->up)
    d = float(p["down"])
    if d < 1.0:
        tfms.append(
            A.Downscale(
                scale_range=(d, d),
                interpolation_pair={"downscale": cv2.INTER_AREA, "upscale": cv2.INTER_LINEAR},
                p=1.0,
            )
        )

    # 2) Blur
    k = int(p["blur"])
    if k > 0:
        ksize = 2*k + 1
        tfms.append(A.GaussianBlur(blur_limit=(ksize, ksize), p=1.0))

    # 3) Contrast (approximate)
    c = float(p["contrast"])
    tfms.append(A.RandomBrightnessContrast(
        brightness_limit=(0.0, 0.0),
        contrast_limit=(c-1.0, c-1.0),
        p=1.0,
    ))

    # 4) Saturation (approximate)
    s = float(p["sat"])
    sat_shift = int(round((s-1.0)*100))
    tfms.append(A.HueSaturationValue(
        hue_shift_limit=(0,0),
        sat_shift_limit=(sat_shift, sat_shift),
        val_shift_limit=(0,0),
        p=1.0
    ))

    # 5) JPEG
    q = int(p["jpeg_q"])
    if q < 100:
        tfms.append(A.ImageCompression(
            compression_type="jpeg",
            quality_range=(q, q),
            p=1.0
        ))

    # 6) Noise (note: std_range is a ratio relative to the max value)
    sig = float(p["noise"])
    if sig > 0:
        std = sig / 255.0  # Use when providing sigma in uint8 scale
        tfms.append(A.GaussNoise(
            std_range=(std, std),
            mean_range=(0.0, 0.0),
            p=1.0
        ))

    return A.Compose(tfms, p=1.0)

def make_tf_id_mapper(tfm_id, *, img_shape=(64,64,3)):
    H, W, C = img_shape

    def _py(img01, msk):
        x = np.clip(img01.astype(np.float32), 0.0, 1.0)
        x_u8 = (x * 255.0 + 0.5).astype(np.uint8)

        y_u8 = tfm_id(image=x_u8)["image"]
        y01  = y_u8.astype(np.float32) / 255.0
        y_m11 = y01 * 2.0 - 1.0

        return y_m11.astype(np.float32), msk.astype(np.float32)

    def _tf(img, msk):
        y_img, y_msk = tf.numpy_function(_py, [img, msk], [tf.float32, tf.float32])
        y_img.set_shape((H, W, C))
        y_msk.set_shape(msk.shape)
        return y_img, y_msk

    return _tf

def make_seg_ds_with_id(pairs, num_classes, tfm_id,
                        batch_size=8, shuffle=True, seed=0,
                        img_shape=(64,64,3),
                        drop_remainder=False, prefetch=4, n_parallel=4):
    base = pairs_to_dataset(pairs, num_classes)  # (img01, msk)

    if shuffle:
        base = base.shuffle(buffer_size=len(pairs), seed=seed, reshuffle_each_iteration=True)

    id_mapper = make_tf_id_mapper(tfm_id, img_shape=img_shape)

    ds = (base
          .map(id_mapper, num_parallel_calls=n_parallel)
          .batch(batch_size, drop_remainder=drop_remainder)
          .prefetch(prefetch))
    return ds

def _to_gray01(img01):
    # img01: (H,W,3) in [0,1]
    r, g, b = img01[...,0], img01[...,1], img01[...,2]
    gray = 0.2989*r + 0.5870*g + 0.1140*b
    return np.clip(gray, 0.0, 1.0).astype(np.float32)

def _sobel_mag(gray01):
    # simple Sobel magnitude (no cv2 dependency)
    g = gray01.astype(np.float32)
    gx = np.zeros_like(g); gy = np.zeros_like(g)
    gx[:,1:-1] = (g[:,2:] - g[:,:-2]) * 0.5
    gy[1:-1,:] = (g[2:,:] - g[:-2,:]) * 0.5
    mag = np.sqrt(gx*gx + gy*gy)
    return mag

def _power_spectrum_features(gray01):
    # Returns (hf_ratio, slope_proxy) from radial spectrum
    g = gray01.astype(np.float32)
    g = g - g.mean()
    F = np.fft.fftshift(np.fft.fft2(g))
    P = (np.abs(F) ** 2).astype(np.float32)

    H, W = P.shape
    cy, cx = H//2, W//2
    yy, xx = np.ogrid[:H, :W]
    rr = np.sqrt((yy-cy)**2 + (xx-cx)**2).astype(np.float32)

    rmax = int(rr.max())
    # radial mean
    radial = np.zeros((rmax+1,), np.float32)
    counts = np.zeros((rmax+1,), np.float32)
    r_int = rr.astype(np.int32)
    np.add.at(radial, r_int, P)
    np.add.at(counts, r_int, 1.0)
    radial = radial / (counts + 1e-6)

    # normalize
    radial = radial / (radial[1:].mean() + 1e-6)

    # hf_ratio: energy in high freq band / mid band (rough)
    r1 = int(0.15*rmax)
    r2 = int(0.35*rmax)
    r3 = int(0.65*rmax)
    r4 = int(0.95*rmax)
    mid = radial[r1:r2].mean() if r2>r1 else radial.mean()
    high = radial[r3:r4].mean() if r4>r3 else radial.mean()
    hf_ratio = float(high / (mid + 1e-6))

    # slope_proxy: log-log slope on [r1..r4]
    xs = np.arange(r1, r4, dtype=np.float32)
    ys = radial[r1:r4].astype(np.float32)
    xs = xs[xs>0]; ys = ys[:len(xs)]
    lx = np.log(xs + 1e-6); ly = np.log(ys + 1e-6)
    # linear fit slope
    A = np.stack([lx, np.ones_like(lx)], axis=1)
    slope, _ = np.linalg.lstsq(A, ly, rcond=None)[0]
    slope = float(slope)
    return hf_ratio, slope

def extract_style_stats(img01):
    """
    img01: (H,W,3) in [0,1]
    returns: dict of scalar stats
    """
    x = np.clip(img01, 0.0, 1.0).astype(np.float32)
    gray = _to_gray01(x)

    # color stats
    mean_rgb = x.reshape(-1,3).mean(axis=0)
    std_rgb  = x.reshape(-1,3).std(axis=0)

    # contrast proxy (gray std)
    gray_std = float(gray.std())

    # edge magnitude stats
    mag = _sobel_mag(gray)
    edge_mean = float(mag.mean())
    edge_p90  = float(np.quantile(mag, 0.90))

    # frequency stats
    hf_ratio, slope = _power_spectrum_features(gray)

    return {
        "mean_r": float(mean_rgb[0]), "mean_g": float(mean_rgb[1]), "mean_b": float(mean_rgb[2]),
        "std_r":  float(std_rgb[0]),  "std_g":  float(std_rgb[1]),  "std_b":  float(std_rgb[2]),
        "gray_std": gray_std,
        "edge_mean": edge_mean,
        "edge_p90": edge_p90,
        "hf_ratio": hf_ratio,
        "spec_slope": slope,
    }

def aggregate_stats(imgs01):
    """
    imgs01: list/array of images (N,H,W,3) in [0,1]
    returns: mean stats dict
    """
    keys = None
    acc = {}
    n = 0
    for im in imgs01:
        s = extract_style_stats(im)
        if keys is None:
            keys = list(s.keys())
            for k in keys: acc[k] = 0.0
        for k in keys:
            acc[k] += float(s[k])
        n += 1
    for k in acc:
        acc[k] /= max(n, 1)
    return acc

def stats_distance(a, b, weights=None):
    """
    L1 distance in normalized space (simple, robust)
    """
    if weights is None:
        weights = {k: 1.0 for k in a.keys()}

    dist = 0.0
    for k in a.keys():
        wa = float(weights.get(k, 1.0))
        # scale normalization: avoid one stat dominating
        denom = abs(float(b[k])) + 1e-3
        dist += wa * abs(float(a[k]) - float(b[k])) / denom
    return float(dist)

DEFAULT_W = {
    "mean_r": 1.5, "mean_g": 1.0, "mean_b": 1.0,
    "std_r":  1.5, "std_g":  1.0, "std_b":  1.0,
    "gray_std": 2.0,
    "edge_mean": 2.0,
    "edge_p90":  2.0,
    "hf_ratio":  2.5,
    "spec_slope": 1.5,
}

def build_id_transform(params):
    """
    params: dict from sampler
    returns: albumentations.Compose
    """
    # down/up: simulate low-frequency / blockiness
    d = params["down"]
    blur = params["blur"]
    c = params["contrast"]
    s = params["sat"]
    jpeg_q = params["jpeg_q"]
    noise = params["noise"]

    tfm = A.Compose([
        # 1) Downscale/UpScale
        A.Downscale(scale_min=d, scale_max=d, interpolation=0, p=1.0),

        # 2) Blur
        A.GaussianBlur(blur_limit=(blur, blur), sigma_limit=(0.1, blur*0.6+0.1), p=1.0) if blur > 0 else A.NoOp(),

        # 3) Contrast/Saturation
        A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=(c-1.0, c-1.0), p=1.0),
        A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=int((s-1.0)*100), val_shift_limit=0, p=1.0),

        # 4) JPEG artifact
        A.ImageCompression(quality_lower=jpeg_q, quality_upper=jpeg_q, p=1.0) if jpeg_q < 100 else A.NoOp(),

        # 5) Mild noise (optional)
        A.GaussNoise(var_limit=(0.0, noise), p=1.0) if noise > 0 else A.NoOp(),
    ])
    return tfm

def apply_id(real01, tfm):
    """
    real01: (H,W,3) float [0,1]
    """
    x = np.clip(real01, 0, 1).astype(np.float32)
    out = tfm(image=(x*255).astype(np.uint8))["image"].astype(np.float32) / 255.0
    return np.clip(out, 0, 1)

def load_image_as_01(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img.numpy()

def collect_real_imgs01_from_pairs(
    pairs,
    num_samples=64,
    seed=0,
):
    """
    pairs: [(img_path, mask_path, pid), ...]
    returns: list of images (H,W,3) in [0,1]
    """
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(pairs), size=min(num_samples, len(pairs)), replace=False)

    imgs01 = []
    for i in idxs:
        img_path, _, _ = pairs[i]

        # Load using the same logic as pairs_to_dataset internally
        img = load_image_as_01(img_path)  # (H,W,3) in [0,1]
        imgs01.append(img.astype(np.float32))

    return imgs01

def sample_params(rng):
    """
    Sampler: if pred_raw tends to appear 'healed-like',
    then downscale/blur/contrast/sat reduction are the main axes.
    """
    # downscale factor: smaller => stronger degradation
    down = rng.uniform(0.35, 0.85)          # 0.35~0.85
    blur = int(rng.integers(0, 5))          # 0~4 (kernel limit)
    contrast = rng.uniform(0.65, 0.95)      # contrast drop
    sat = rng.uniform(0.60, 0.95)           # saturation drop
    jpeg_q = int(rng.integers(55, 101))     # 55~100
    noise = float(rng.uniform(0.0, 8.0))    # var_limit upper (mild)
    return dict(down=down, blur=blur, contrast=contrast, sat=sat, jpeg_q=jpeg_q, noise=noise)

def calibrate_id_to_pred(
    real_imgs01,          # list/array of real images [0,1]
    pred_imgs01,          # list/array of pred images [0,1]
    n_trials=120,         # Number of search trials
    n_eval=24,            # Number of samples for statistical evaluation per trial
    seed=0,
    weights=DEFAULT_W,
    verbose=True,
):
    """
    returns: best_params, best_score, target_stats_pred, best_stats_id
    """
    rng = np.random.default_rng(seed)

    # target stats from pred
    # (If possible, secure pred_imgs01 from various cases/K values)
    target = aggregate_stats(pred_imgs01)

    # subsample pools
    real_pool = list(real_imgs01)
    pred_pool = list(pred_imgs01)

    best_score = 1e18
    best_params = None
    best_stats = None

    for t in range(n_trials):
        params = sample_params(rng)
        tfm = build_id_transform(params)

        # evaluate on n_eval real samples
        idx = rng.choice(len(real_pool), size=min(n_eval, len(real_pool)), replace=False)
        id_imgs = [apply_id(real_pool[i], tfm) for i in idx]

        st = aggregate_stats(id_imgs)
        score = stats_distance(st, target, weights=weights)

        if score < best_score:
            best_score = score
            best_params = params
            best_stats = st
            if verbose:
                print(f"[best@{t}] score={best_score:.4f} params={best_params}")

    return best_params, best_score, target, best_stats

real_imgs01 = collect_real_imgs01_from_pairs(
    train_pairs,
    num_samples=64,
    seed=SEED,
)

pred_cache_base = {}

for k in np.linspace(0.1, 1.0, 10):
    Xk_k = _set_k_channel(Xk_base, float(k))
    y_pred, _ = creator.predict(Xk_k, verbose=0)
    pred01 = np.clip((y_pred[0] + 1)/2, 0, 1).astype(np.float32)

    pred_seg = mask.predict((pred01*2-1)[None, ...], verbose=0)
    prob_all = get_prob_all_batch(pred_seg, assume="softmax")[0]

    pred_cache_base[float(np.round(k,2))] = {
        "pred01_raw": pred01,
        "prob_all_pred": prob_all,
    }

pred_imgs01 = [pred_cache_base[k]["pred01_raw"] for k in sorted(pred_cache_base.keys())]

best_params, best_score, target_pred_stats, best_id_stats = calibrate_id_to_pred(
    real_imgs01=real_imgs01,
    pred_imgs01=pred_imgs01,
    n_trials=160,
    n_eval=32,
    seed=7,
    verbose=True,
)

print("\n=== calibration result ===")
print("best_score:", best_score)
print("best_params:", best_params)
print("target_pred_stats:", target_pred_stats)
print("best_id_stats:", best_id_stats)

tfm_best_id = build_id_transform_deterministic(best_params)

seg_train_id = make_seg_ds_with_id(
    train_pairs, NUM_CLASSES, tfm_best_id,
    batch_size=BATCH, shuffle=True, seed=SEED,
    img_shape=(64,64,3)
)
seg_val_id = make_seg_ds_with_id(
    val_pairs, NUM_CLASSES, tfm_best_id,
    batch_size=BATCH, shuffle=False,
    img_shape=(64,64,3)
)

x, y = next(iter(seg_train_id))
print("x:", x.shape, x.dtype, float(tf.reduce_min(x)), float(tf.reduce_max(x)))
print("y:", y.shape, y.dtype, float(tf.reduce_min(y)), float(tf.reduce_max(y)))

x0, y0 = next(iter(pairs_to_dataset(train_pairs, NUM_CLASSES).batch(BATCH)))
print("orig:", float(tf.reduce_min(x0)), float(tf.reduce_max(x0)))
print("id  :", float(tf.reduce_min(x)),  float(tf.reduce_max(x)))
print("mean|diff|:", float(tf.reduce_mean(tf.abs(tf.cast(x0, tf.float32) - x))))

mask_ft = unet(input_shape=(64,64,3), alpha=1.0, weights='imagenet')
mask_ft.set_weights(mask.get_weights())
mask_ft.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=seg_loss,
    metrics=[soft_dice_metric],
)

hist = mask_ft.fit(
    seg_train_id,
    validation_data=seg_val_id,
    epochs=15,
    callbacks=[ckpt, plateau, early],
    verbose=1,
)

# Anchor fixed (only once)
y_pred0, x_last0 = creator.predict(Xk_base, verbose=0)
last01_anchor = np.clip((x_last0[0] + 1)/2, 0, 1).astype(np.float32)

pred_last0 = mask_ft.predict((last01_anchor*2-1)[None, ...], verbose=0)
prob_all_anchor_fixed = get_prob_all_batch(pred_last0, assume="softmax")[0]
m_soft_anchor_lesion_fixed = get_lesion_prob_from_prob_all(prob_all_anchor_fixed)

THR_LESION_LAST = float(THR_SUPPORT)
roi_anchor_fixed = (m_soft_anchor_lesion_fixed >= THR_LESION_LAST)

def thr_sensitivity(mask_model, img01, thr_list):
    out = []
    pred = mask_model.predict((img01*2-1)[None, ...], verbose=0)
    prob_all = get_prob_all_batch(pred, assume="softmax")[0]
    pw = prob_all[..., int(WOUND_IDX)]

    for thr in thr_list:
        out.append({
            "thr": thr,
            "area": float((pw >= thr).mean())
        })
    return out

def debug_pred_pw(mask_model, pred01_raw, WOUND_IDX, HEALED_IDX=None, ESCHAR_IDX=None):
    pred = mask_model.predict((pred01_raw*2-1)[None, ...], verbose=0)
    prob_all = get_prob_all_batch(pred, assume="softmax")[0]
    pw = prob_all[..., int(WOUND_IDX)]
    out = {"pw_mean": float(pw.mean()), "pw_p99": float(np.quantile(pw, 0.99))}
    if HEALED_IDX is not None:
        ph = prob_all[..., int(HEALED_IDX)]
        out["ph_mean"] = float(ph.mean())
        out["ph_p99"]  = float(np.quantile(ph, 0.99))
    if ESCHAR_IDX is not None:
        pe = prob_all[..., int(ESCHAR_IDX)]
        out["pe_mean"] = float(pe.mean())
        out["pe_p99"]  = float(np.quantile(pe, 0.99))
    return out

pred_cache_ft = {}
K_vals = [0.2, 0.6, 1.0]
for k in K_vals:
    k = float(k)
    # creator output
    Xk_k = _set_k_channel(Xk_base, k)
    y_pred, x_last = creator.predict(Xk_k, verbose=0)
    pred01 = np.clip((y_pred[0] + 1)/2, 0, 1).astype(np.float32)
    last01 = np.clip((x_last[0] + 1)/2, 0, 1).astype(np.float32)

    # Pred segmentation (changes each time)
    pred_pred = mask_ft.predict((pred01*2-1)[None, ...], verbose=0)
    prob_all_pred = get_prob_all_batch(pred_pred, assume="softmax")[0]
    m_soft_pred_lesion = get_lesion_prob_from_prob_all(prob_all_pred)

    alpha01, dbg_alpha = make_alpha_K(
        prob_all_pred,
        K=k,
        wound_idx=WOUND_IDX,
        eschar_idx=ESCHAR_IDX,
        exclude_idx=EXCLUDE_IDX,
        debug=True,
    )

    pp = schedule_postproc_params(float(k))

    # 1) hard region decision
    C = {}
    pred01_con, (roi_c, allow_c, protect_c, uncertain_c), dbg_c = postprocess_allow_protect(
        pred01=pred01,
        last01=last01,
        prob_all_last=prob_all_anchor_fixed,
        prob_all_pred=prob_all_pred,
        m_soft_last_lesion=m_soft_anchor_lesion_fixed,
        m_soft_pred_lesion=m_soft_pred_lesion,
        THR_SUPPORT=THR_SUPPORT,
        THR_CHANGE=THR_CHANGE,
        CHANGE_IDXS=CHANGE_IDXS,
        STABLE_IDXS=STABLE_IDXS,
        WOUND_IDX=WOUND_IDX,
        HEALED_IDX=HEALED_IDX,
        ESCHAR_IDX=ESCHAR_IDX,
        alpha01=alpha01,
        K=k,
        tx_norm=0.0,          # conservative
        **pp,
    )

    # optimistic
    pred01_opt, (roi_o, allow_o, protect_o, uncertain_o), dbg_o = postprocess_allow_protect(
        pred01=pred01,
        last01=last01,
        prob_all_last=prob_all_anchor_fixed,
        prob_all_pred=prob_all_pred,
        m_soft_last_lesion=m_soft_anchor_lesion_fixed,
        m_soft_pred_lesion=m_soft_pred_lesion,
        THR_SUPPORT=THR_SUPPORT,
        THR_CHANGE=THR_CHANGE,
        CHANGE_IDXS=CHANGE_IDXS,
        STABLE_IDXS=STABLE_IDXS,
        WOUND_IDX=WOUND_IDX,
        HEALED_IDX=HEALED_IDX,
        ESCHAR_IDX=ESCHAR_IDX,
        alpha01=alpha01,
        K=k,
        tx_norm=1.0,          # optimistic
        **pp,
    )

    # diagnostics
    dmag = np.mean(np.abs(pred01 - last01), axis=-1)
    cmag_c = np.mean(np.abs(pred01_con - last01), axis=-1)
    cmag_o = np.mean(np.abs(pred01_opt - last01), axis=-1)

    print(f"\n[K={k}] CONS dbg:", dbg_c)
    print("  mean |pred-last| in allow   :", float(dmag[allow_c].mean()) if allow_c.any() else np.nan)
    print("  mean |cons-last| in allow   :", float(cmag_c[allow_c].mean()) if allow_c.any() else np.nan)
    print("  allow/protect/uncertain frac:", float(allow_c.mean()), float(protect_c.mean()), float(uncertain_c.mean()))

    print(f"\n[K={k}] OPT  dbg:", dbg_o)
    print("  mean |pred-last| in allow   :", float(dmag[allow_o].mean()) if allow_o.any() else np.nan)
    print("  mean |opt-last| in allow    :", float(cmag_o[allow_o].mean()) if allow_o.any() else np.nan)
    print("  allow/protect/uncertain frac:", float(allow_o.mean()), float(protect_o.mean()), float(uncertain_o.mean()))

    _diag10(
        tag=f"K={k} CONS",
        last01=last01,
        pred01_raw=pred01,
        pred01_constrained=pred01_con,
        roi01=roi_c.astype(np.float32),
        alpha01=alpha01,
    )

    _diag10(
        tag=f"K={k} OPT",
        last01=last01,
        pred01_raw=pred01,
        pred01_constrained=pred01_opt,
        roi01=roi_o.astype(np.float32),
        alpha01=alpha01,
    )

    # cache everything needed for downstream analysis
    pred_cache_ft[float(k)] = dict(
        pred01_raw=pred01,
        last01=last01,

        # Two scenario outputs
        pred01_con=pred01_con,
        pred01_opt=pred01_opt,

        alpha01=alpha01,
        dbg_alpha=dbg_alpha,

        # Store masks per scenario
        masks_con=dict(roi=roi_c, allow=allow_c, protect=protect_c, uncertain=uncertain_c),
        masks_opt=dict(roi=roi_o, allow=allow_o, protect=protect_o, uncertain=uncertain_o),

        prob_all_pred=prob_all_pred,
        m_soft_pred_lesion=m_soft_pred_lesion,

        dbg_con=dbg_c,
        dbg_opt=dbg_o,
    )

img01_chk = pred_cache_ft[1.0]["pred01_raw"]
thr_list = np.linspace(0.1, 0.9, 17)

thr_curve = thr_sensitivity(mask_ft, img01_chk, thr_list)
print(pd.DataFrame(thr_curve))

# -------------------------
# 2) downstream analysis: pick one k_ref
# -------------------------
k_ref = 1.0
C = pred_cache_ft[float(k_ref)]

scenario = "opt"

last01        = last01_anchor
pred01_raw    = C["pred01_raw"]

if scenario == "con":
    pred01_final = C["pred01_con"]
    masks        = C["masks_con"]
    dbg_use      = C["dbg_con"]
else:
    pred01_final = C["pred01_opt"]
    masks        = C["masks_opt"]
    dbg_use      = C["dbg_opt"]

print("Using scenario:", scenario, "dbg:", dbg_use)

roi = masks["roi"]
allow = masks["allow"]
protect = masks["protect"]
uncertain = masks["uncertain"]
prob_all_pred = C["prob_all_pred"]
m_soft_pred_lesion = C["m_soft_pred_lesion"]

print("[anchor lesion] min/max:", float(m_soft_anchor_lesion_fixed.min()), float(m_soft_anchor_lesion_fixed.max()))
print("[pred lesion@k_ref] min/max:", float(m_soft_pred_lesion.min()), float(m_soft_pred_lesion.max()))

pred_final = mask_ft.predict((pred01_final*2-1)[None, ...], verbose=0)
prob_all_final = get_prob_all_batch(pred_final, assume="softmax")[0]

# overlay on FINAL image
overlay_bin_predicted, used_thr_predicted, _ = overlay_mask_on_image(
    img01=pred01_final,
    prob=prob_all_final[..., int(WOUND_IDX)],
    thr=float(THR_WOUND),
    alpha=0.4
)
plt.figure(figsize=(4,4))
plt.imshow(overlay_bin_predicted)
plt.title(f"Final({scenario}) wound mask @thr={used_thr_predicted:.3f} (k={k_ref})")
plt.axis('off'); plt.show()

# lesion diff (pred - anchor)
diff = np.clip(m_soft_pred_lesion - m_soft_anchor_lesion_fixed, -1, 1)
plt.figure(figsize=(4,4)); plt.imshow(diff, cmap='bwr', vmin=-1, vmax=1)
plt.title(f"lesion_soft_pred - lesion_soft_anchor (k={k_ref})"); plt.axis('off'); plt.show()

diag_wound_focus_panels(
    pred01_final=pred01_final,
    prob_all_pred=C["prob_all_pred"],
    roi=roi,
    prob_all_anchor=prob_all_anchor_fixed,   # To also observe anchor prior blending
    WOUND_IDX=WOUND_IDX, HEALED_IDX=HEALED_IDX, ESCHAR_IDX=ESCHAR_IDX,
    margin=0.03,
    wrel_thr=0.30,
    lam_anchor=0.35,
    title=f"K={k_ref}"
)

def structure_only_np(img01):
    x = tf.convert_to_tensor(img01[None, ...], tf.float32)  # (1,H,W,3)
    gray = tf.image.rgb_to_grayscale(x)                     # (1,H,W,1)
    g = (gray - tf.reduce_mean(gray)) / (tf.math.reduce_std(gray) + 1e-6)
    g = tf.clip_by_value(g * 0.25 + 0.5, 0.0, 1.0)
    g3 = tf.tile(g, [1,1,1,3])[0].numpy()                   # (H,W,3)
    return g3.astype(np.float32)

ex_seq = example_sequence.copy()
ex_seq01 = match_meanstd_seq01(ex_seq)
if ex_seq01.max() > 1.0 or ex_seq01.min() < 0.0:
    ex_seq01 = np.clip((ex_seq01 + 1.0) / 2.0, 0, 1).astype(np.float32)
else:
    ex_seq01 = np.clip(ex_seq01, 0, 1).astype(np.float32)

B, T, H, W, _ = ex_seq01.shape
assert B == 1, "This example assumes a sequence with B=1."
print(f"[example] shape={ex_seq01.shape}")

for t in range(T):
    img01 = ex_seq01[0, t]                              # (H,W,3) in [0,1]
    # (a) Infer probability map using mask model
    img_m11   = img01 * 2.0 - 1.0
    pred_all  = mask.predict(img_m11[None, ...], verbose=0)
    prob_all = get_prob_all_batch(pred_all, assume="softmax")[0]
    prob_lesion = get_lesion_prob_from_prob_all(prob_all)  # (H,W)

    # (b) Binary overlay
    overlay_bin, used_thr, binm = overlay_mask_on_image(
        img01, prob=prob_lesion, thr=THR_SUPPORT, alpha=0.4
    )

    overlay_s = None
    try:
        overlay_s = overlay_soft(img01, m_soft=prob_lesion, alpha_max=0.45, gamma=0.8)
    except NameError:
        pass

    # (d) Visualization
    cols = 3 if overlay_s is not None else 2
    plt.figure(figsize=(4*cols, 4))
    plt.subplot(1, cols, 1); plt.imshow(img01);           plt.title(f"t={t} Input"); plt.axis("off")
    plt.subplot(1, cols, 2); plt.imshow(overlay_bin);     plt.title(f"Binary@thr={THR_WOUND:.2f}"); plt.axis("off")
    if overlay_s is not None:
        plt.subplot(1, cols, 3); plt.imshow(overlay_s);   plt.title("Soft heatmap");  plt.axis("off")
    plt.show()

# agreement on lesion using THR_SUPPORT
agree_bin = ((m_soft_anchor_lesion_fixed >= THR_SUPPORT) &
             (m_soft_pred_lesion >= THR_SUPPORT)).mean()
print("binary lesion agreement@THR_SUPPORT:", float(agree_bin))

# support/amplitude gating uses FINAL image
M_agree, S, A = gate_lesion_union_support_amplitude_pred(
    last01=last01,
    pred01=pred01_final,
    prob_last_lesion=m_soft_anchor_lesion_fixed,
    prob_pred_lesion=m_soft_pred_lesion,
    thr_anchor=THR_SUPPORT,
    tau=0.05,
    support_mode="softthr",
    amplitude_mode="raw",
    union_kind="and_prob",
    gamma_amp=0.65,
    support_power=1.6,
)
print("mean M_agree:", float(M_agree.mean()), "mean S:", float(S.mean()), "mean A:", float(A.mean()))

stats_raw = plot_delta_panel(last01, pred01_raw, roi[..., None], title="RAW", draw=True)
print("raw:", stats_raw)

stats_con = plot_delta_panel(last01, C["pred01_con"], C["masks_con"]["roi"][..., None],
                             title="FINAL_CON", draw=True)
print("con:", stats_con)

stats_opt = plot_delta_panel(last01, C["pred01_opt"], C["masks_opt"]["roi"][..., None],
                             title="FINAL_OPT", draw=True)
print("opt:", stats_opt)

def postproc_kwargs_con(k):
    pp = schedule_postproc_params(float(k))
    pp["tx_norm"] = 0.0
    return pp

def postproc_kwargs_opt(k):
    pp = schedule_postproc_params(float(k))
    pp["tx_norm"] = 1.0
    return pp

rep_con = k_sweep_dir_report(
    creator=creator,
    Xk=Xk_base,
    mask_model=mask_ft,
    lesion_from_prob_all_fn=get_lesion_prob_from_prob_all,
    THR=THR_WOUND,
    K_vals=(0.2, 0.6, 1.0),
    union_kind="and_prob",
    m_soft_anchor_lesion_fixed=m_soft_anchor_lesion_fixed,
    prob_all_anchor_fixed=prob_all_anchor_fixed,
    postproc_fn=postprocess_allow_protect,
    postproc_kwargs_fn=postproc_kwargs_con,
)

rep_opt = k_sweep_dir_report(
    creator=creator,
    Xk=Xk_base,
    mask_model=mask_ft,
    lesion_from_prob_all_fn=get_lesion_prob_from_prob_all,
    THR=THR_WOUND,
    K_vals=(0.2, 0.6, 1.0),
    union_kind="and_prob",
    m_soft_anchor_lesion_fixed=m_soft_anchor_lesion_fixed,
    prob_all_anchor_fixed=prob_all_anchor_fixed,
    postproc_fn=postprocess_allow_protect,
    postproc_kwargs_fn=postproc_kwargs_opt,
)

print("K:", rep_con["K"])

print("[RAW]")
print("  da:", rep_con["da_roi_raw"])
print("  dE:", rep_con["dE_roi_raw"])

print("[FINAL | conservative]")
print("  da:", rep_con["da_roi"])
print("  dE:", rep_con["dE_roi"])

print("[FINAL | optimistic]")
print("  da:", rep_opt["da_roi"])
print("  dE:", rep_opt["dE_roi"])

plot_k_sweep({"K": rep_con["K"], "da_roi": rep_con["da_roi_raw"]}, expect="improve", title="RAW")  # (raw is the same)
plot_k_sweep({"K": rep_con["K"], "da_roi": rep_con["da_roi"]},     expect="improve", title="FINAL_CON")
plot_k_sweep({"K": rep_opt["K"], "da_roi": rep_opt["da_roi"]},     expect="improve", title="FINAL_OPT")
