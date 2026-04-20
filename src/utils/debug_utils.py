"""
BurnSight debug utilities — logging, panel visualization, graph diagnostics
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

from src.inference.eval_utils import *

def save_seg_debug_panel(out_dir, sample_id, img01, gt_w, pw, thr=0.03):
    """
    img01 : (H,W,3) float [0,1]
    gt_w  : (H,W)   {0,1}
    pw    : (H,W)   float [0,1]
    """
    os.makedirs(out_dir, exist_ok=True)

    H, W = gt_w.shape
    overlay = img01.copy()
    mask = (pw >= thr).astype(np.float32)
    # red overlay
    overlay[..., 0] = np.clip(overlay[..., 0] * (1 - 0.35*mask) + 0.35*mask, 0, 1)

    # histogram
    pw_pos = pw[gt_w == 1]
    pw_neg = pw[gt_w == 0]

    fig = plt.figure(figsize=(16, 4))
    ax1 = plt.subplot(1, 5, 1); ax1.imshow(img01); ax1.set_title("Input"); ax1.axis("off")
    ax2 = plt.subplot(1, 5, 2); ax2.imshow(gt_w, cmap="gray"); ax2.set_title("GT wound"); ax2.axis("off")
    ax3 = plt.subplot(1, 5, 3); ax3.imshow(pw, cmap="gray", vmin=0, vmax=1); ax3.set_title("Prob wound"); ax3.axis("off")
    ax4 = plt.subplot(1, 5, 4); ax4.imshow(overlay); ax4.set_title(f"Overlay@{thr:.2f}"); ax4.axis("off")
    ax5 = plt.subplot(1, 5, 5)
    ax5.hist(pw_neg.ravel(), bins=50, alpha=0.6, label="GT=0")
    ax5.hist(pw_pos.ravel(), bins=50, alpha=0.6, label="GT=1")
    ax5.axvline(thr, ls="--")
    ax5.set_title("pw hist")
    ax5.set_xlim(0, 1)
    ax5.legend()

    fig.tight_layout()
    fn = os.path.join(out_dir, f"{_safe_id(sample_id)}.png")
    fig.savefig(fn, dpi=140)
    plt.close(fig)
    return fn


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