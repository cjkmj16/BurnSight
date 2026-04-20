"""
BurnSight Creator training loss function collection
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

from src.models.layers import *

def mean_abs_delta(a, b, roi):
    d = tf.abs(a - b)
    w = tf.cast(roi, d.dtype)
    return tf.reduce_sum(d*w) / (tf.reduce_sum(w)+1e-6)

def change_map_from_history(Xk, mask=None, use_last_pair=True):
    """
    X_full: (B,T,H,W,Cx)  # Cx>=3; observed RGB assumed at X_full[..., :3] ([0,1] scale)
    mask  : (B,H,W,1) or None  # Pass for ROI weighting (e.g., M_next)
    return: (B,H,W,1)  approximate [0,1] range 'change map'
    """
    # 1) Extract RGB only (use as-is if input is [0,1])
    Xrgb = (Xk[..., :3] + 1.0) * 0.5                          # (B,T,H,W,3)

    # 2) Compute frame difference |x_t - x_{t-1}|
    #    -> (B,T-1,H,W,3)
    diff = tf.abs(Xrgb[:, 1:, ...] - Xrgb[:, :-1, ...])

    if use_last_pair:
        diff = diff[:, -1, ...]                                 # (B,H,W,3)
    else:
        diff = tf.reduce_mean(diff, axis=1)                     # (B,H,W,3)

    diff_m = tf.reduce_mean(diff, axis=-1, keepdims=True)       # (B,H,W,1)

    if mask is not None:
        m1 = tf.cast(mask, tf.float32)
        diff_m = diff_m * m1
    return diff_m

def change_scalar_from_history(Xk, mask=None, eps=1e-6, reduce='mean', use_last_pair=True):
    """
    X_full: (B,T,H,W,Cx), mask: (B,H,W,1) or None
    return: (B,)  # 'change intensity' scalar over the sequence
    """
    Xrgb = (Xk[..., :3] + 1.0) * 0.5  # (B,T,H,W,3)
    diff = tf.abs(Xrgb[:, 1:, ...] - Xrgb[:, :-1, ...])  # (B,T-1,H,W,3)

    if use_last_pair:
        diff = diff[:, -1:, ...]  # (B,1,H,W,3)

    if mask is not None:
        # Broadcast mask to 3 channels and time axis
        m1 = tf.cast(mask, tf.float32)                              # (B,H,W,1)
        m3 = tf.repeat(m1, 3, axis=-1)                              # (B,H,W,3)
        mT = tf.broadcast_to(m3[:, tf.newaxis, ...], tf.shape(diff))# (B,T-1,H,W,3)
        num = tf.reduce_sum(diff * mT, axis=[1,2,3,4])              # (B,)
        den = tf.reduce_sum(mT,           axis=[1,2,3,4]) + eps     # (B,)
        val = num / den
    else:
        val = tf.reduce_mean(diff, axis=[1,2,3,4])                  # (B,)

    # Reduce option (use 'none' if per-batch scalar is needed)
    if reduce == 'mean':
        return tf.reduce_mean(val)
    elif reduce == 'sum':
        return tf.reduce_sum(val)
    else:
        return val  # (B,)

def cosine_loss(z_pred, z_goal, eps=1e-6):
    zp = tf.math.l2_normalize(z_pred, axis=-1)
    zg = tf.math.l2_normalize(z_goal, axis=-1)
    return tf.reduce_mean(1.0 - tf.reduce_sum(zp*zg, axis=-1))

def delta_dir_loss(y_pred, x_last, Xk, roi, eps=1e-6):
    """
    y_pred, x_last: [-1,1]
    Xk: (B,T,H,W,5) with RGB in [0,1], K in [0,1], m_seq in [0,1]
    roi: (B,H,W,1) in [0,1]
    """
    # T-2 safety guard
    T = tf.shape(Xk)[1]
    tf.debugging.assert_greater_equal(T, 2, message="delta_dir_loss: need at least 2 frames")

    # Reference direction: RGB difference of the last two frames (kept in [0,1] scale)
    x_t   = Xk[:, -1, :, :, :3]
    x_tm1 = Xk[:, -2, :, :, :3]
    d_ref = (x_t - x_tm1) * tf.cast(roi, tf.float32)

    # Prediction direction: y_pred - x_last (both in [-1,1])
    d_prd = (y_pred - x_last) * tf.cast(roi, tf.float32)

    num = tf.reduce_sum(d_ref * d_prd, axis=[1,2,3])
    den = (tf.sqrt(tf.reduce_sum(d_ref**2, axis=[1,2,3]) + eps) *
           tf.sqrt(tf.reduce_sum(d_prd**2, axis=[1,2,3]) + eps))
    cos = num / (den + eps)
    return tf.reduce_mean(1.0 - cos)

def masked_l1(y_true, y_pred, m_next, eps=1e-6):
    # y_true, y_pred: (B,H,W,3) in [-1,1], m_next: (B,H,W,1) in [0,1]
    diff = tf.abs(y_true - y_pred) * m_next
    return tf.reduce_sum(diff) / (tf.reduce_sum(m_next) + eps)

def pick_y_pred(out):
    # Handle all output forms: single tensor / [y_pred, x_last] / [y_pred, x_last, ...]
    if isinstance(out, (list, tuple)):
        y_pred = out[0]
        x_last = out[1] if len(out) > 1 else None
    else:
        y_pred, x_last = out, None
    return y_pred, x_last


def erode(mask, px=2):
    """Erosion by px pixels."""
    # Approximate erosion via dilation with max_pool then inversion
    # Simply: invert -> dilate -> invert = erosion
    m = tf.clip_by_value(mask, 0., 1.)
    inv = 1.0 - m
    k = 2*px + 1
    dil = tf.nn.max_pool2d(inv, ksize=k, strides=1, padding='SAME')
    ero = 1.0 - dil
    return ero

def laplacian_gray(x):
    k = tf.constant([[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]], tf.float32)
    k = tf.reshape(k,[3,3,1,1])
    g = tf.image.rgb_to_grayscale(x)
    return tf.nn.conv2d(g, k, 1, 'SAME')

def shave_roi(mask, px=2):
    """ROI with boundary pixels removed (erosion)."""
    return erode(mask, px=px)

def soft_weight(mask, gamma=0.8):
    """Weight a 0~1 probability mask smoothly as w = p^gamma."""
    m = tf.clip_by_value(mask, 0., 1.)
    return tf.pow(m, tf.cast(gamma, m.dtype))

def edge_ratio_loss(y_pred, Xk, roi):
    """
    y_pred: [-1,1]
    Xk: (B,T,H,W,5) with RGB in [0,1]
    roi: (B,H,W,1)
    """
    x_t = Xk[:, -1, :, :, :3]               # [0,1]
    Lr  = tf.abs(laplacian_gray(x_t))       # ref edge
    Lp  = tf.abs(laplacian_gray(y_pred))    # pred edge (ok with [-1,1] input: conv is linear)

    # Normalize ref
    mn  = tf.reduce_min(Lr, [1,2,3], keepdims=True)
    mx  = tf.reduce_max(Lr, [1,2,3], keepdims=True)
    Lr  = (Lr - mn) / (mx - mn + 1e-6)

    return tf.reduce_mean(tf.abs(Lp - Lr) * tf.cast(roi, tf.float32))

def tighten_mask(M, ceil=0.6):
    M = tf.nn.avg_pool2d(M, ksize=3, strides=1, padding="SAME")   # Mild erosion
    M = tf.minimum(M, ceil)
    return M

def _first_output(out):
    return out[0] if isinstance(out, (list, tuple)) else out

def psnr_tf(a, b):
    # a,b: both [-1,1] and [0,1] accepted (auto-normalized)
    # Normalize internally to [0,1] and compute PSNR
    a01 = (a + 1.0) * 0.5 if tf.reduce_max(a) <= 1.0 and tf.reduce_min(a) >= -1.0 else a
    b01 = (b + 1.0) * 0.5 if tf.reduce_max(b) <= 1.0 and tf.reduce_min(b) >= -1.0 else b
    mse = tf.reduce_mean(tf.square(a01 - b01))
    return 20.0 * tf.math.log(1.0 / tf.sqrt(mse + 1e-6)) / tf.math.log(10.0)

def roi_delta_ratio(y_pred, last, m_soft):
    y01, last01 = (y_pred + 1.0)*0.5, (last + 1.0)*0.5
    delta = tf.abs(y01 - last01)
    num = tf.reduce_sum(delta * tf.repeat(m_soft, 3, -1), [1,2,3])
    den = tf.reduce_sum(delta, [1,2,3]) + 1e-6
    return tf.reduce_mean(num / den)

def nonroi_identity_test(model, Xk, m_soft, last):
    # Prediction
    y_pred = _first_output(model(Xk, training=False))  # [-1,1]
    # PSNR (tensor scalar)
    p = psnr_tf(y_pred, last)                          # must be a tensor-returning function

    # Mean |Δ| outside ROI (tensor scalar)
    y01, last01 = (y_pred + 1.0) * 0.5, (last + 1.0) * 0.5
    delta = tf.abs(y01 - last01)                       # (B,H,W,3)
    out   = 1.0 - m_soft                               # (B,H,W,1)
    out3  = tf.repeat(out, 3, -1)
    dout  = tf.reduce_sum(delta * out3, [1,2,3]) / (tf.reduce_sum(out3, [1,2,3]) + 1e-6)
    dout_mean = tf.reduce_mean(dout)                   # scalar tensor
    return {"psnr_pred_last": p, "delta_out_mean": dout_mean}

def k_sweep_monotonicity(model, Xk, m_soft, last, K_vals=(0.2,0.6,1.0)):
    energies = []
    K_MAX = 5.0
    last01 = (last + 1.0) * 0.5
    w3 = tf.repeat(m_soft, 3, -1)
    for kv in K_vals:
        Xk_k = tf.concat(
            [Xk[..., :3],
             tf.fill(tf.shape(Xk[..., 3:4]), tf.cast(kv / K_MAX, Xk.dtype)),
             Xk[..., 4:5]],
            axis=-1
        )
        y_pred_k = _first_output(model(Xk_k, training=False))
        y01_k = (y_pred_k + 1.0) * 0.5
        delta_k = tf.abs(y01_k - last01)
        e = tf.reduce_sum(delta_k * w3, [1,2,3]) / (tf.reduce_sum(w3, [1,2,3]) + 1e-6)
        energies.append(tf.reduce_mean(e))             # batch mean, scalar tensor
    energies = tf.stack(energies, axis=0)              # (len(K),)
    mono_pen = tf.reduce_sum(tf.nn.relu(energies[:-1] - energies[1:]))  # scalar tensor
    return {"ROI_|Δ|": energies, "monotonicity_penalty": mono_pen}

def _tighten_soft_mask(m_soft, clip_floor=0.05, clip_ceil=0.95, ksize=3):
    # m_soft: (B,H,W,1) in [0,1]
    m = tf.clip_by_value(m_soft, 0.0, 1.0)
    m = tf.nn.max_pool2d(m, ksize=ksize, strides=1, padding="SAME")
    m = tf.clip_by_value(m, clip_floor, clip_ceil)
    return m

def _specular_protect_weight(last01, thr=0.90, slope=40.0):
    # last01: (B,H,W,3) [0,1]
    y = tf.image.rgb_to_yuv(last01)[..., :1]              # (B,H,W,1) luminance proxy
    # Sigmoid-based 0~1 weight: brighter = more protected (= less change allowed)
    w = tf.math.sigmoid((y - thr) * slope)                # bright -> 1
    return 1.0 - w                                        # protection weight → 'change weight' = 1-w

def postprocess_creator_output(
    y_pred_raw,      # (B,H,W,3) in [-1,1]
    last,            # (B,H,W,3) in [-1,1]
    m_soft,          # (B,H,W,1) in [0,1]  (soft gate of the last frame)
    clip_floor=0.05,
    clip_ceil=0.95,
    ksize=3,
    specular_protect=True,
    sp_thr=0.90,
    sp_slope=40.0,
    deltaE_clamp=0.25,  # e.g., 0.25 (ROI Δ upper bound in [0,1] scale; None to skip)
    delta_mag_min=None,          # e.g., 0.05 ([0,1] scale)
    delta_mag_gain_max=4.0,      # Prevent runaway amplification
    delta_mag_on_roi_only=True,
):
    # 1) Refine soft gate
    M = _tighten_soft_mask(m_soft, clip_floor, clip_ceil, ksize)      # (B,H,W,1)
    M3 = tf.repeat(M, 3, axis=-1)

    # 2) Basic compositing: use y_pred_raw inside ROI, keep last outside
    y_comp = (1.0 - M3) * last + M3 * y_pred_raw                      # [-1,1]

    # 3) Specular protection (suppress changes in highlight regions)
    if specular_protect:
        last01 = (last + 1.0) * 0.5
        w_change = _specular_protect_weight(last01, thr=sp_thr, slope=sp_slope)  # (B,H,W,1)
        w3 = tf.repeat(w_change, 3, axis=-1)
        # Interpolate between last and y_comp → bright areas stay closer to last
        y_comp = w3 * y_comp + (1.0 - w3) * last

    # 3.5) (Optional) Guarantee minimum Δ magnitude: scale up residual inside ROI only
    if delta_mag_min is not None:
        y01    = (y_comp + 1.0) * 0.5
        last01 = (last   + 1.0) * 0.5

        # ROI-weighted mean |Δ|
        mag = _roi_delta_mag(y01, last01, M)  # (B,)
        # gain = delta_mag_min / mag (only when gain > 1)
        gain = delta_mag_min / (mag + 1e-6)
        gain = tf.maximum(gain, 1.0)
        gain = tf.minimum(gain, float(delta_mag_gain_max))
        gain = gain[:, None, None, None]  # (B,1,1,1)

        # residual
        d01 = (y01 - last01)

        if delta_mag_on_roi_only:
            # Scale up inside ROI only (protect background)
            y01 = last01 + d01 * ((1.0 - M3) + M3 * gain)
        else:
            # Global scale-up (not recommended)
            y01 = last01 + d01 * gain

        y_comp = y01 * 2.0 - 1.0

    # 4) (Optional) ΔE clamp: suppress excessive color/tone changes inside ROI (light version)
    if deltaE_clamp is not None:
        # Clamp in [0,1] scale
        y01    = (y_comp + 1.0) * 0.5
        last01 = (last   + 1.0) * 0.5
        d = tf.abs(y01 - last01)                        # L1 approximation
        d_mean = tf.reduce_mean(d, axis=-1, keepdims=True)       # (B,H,W,1)
        # Apply inside ROI only
        limit = deltaE_clamp
        scale = tf.clip_by_value(limit / (d_mean + 1e-6), 0.0, 1.0)
        scale3 = tf.repeat(scale, 3, axis=-1)
        y01 = last01 + (y01 - last01) * ( (1.0 - M3) + M3 * scale3 )
        y_comp = y01 * 2.0 - 1.0

    return y_comp, M, M3

def _roi_delta_mag(y01, last01, M):
    # Mean |Δ| (ROI-weighted)
    w = soft_weight(M, gamma=0.4)               # (B,H,W,1)
    w3 = tf.repeat(w, 3, axis=-1)
    num = tf.reduce_sum(tf.abs(y01 - last01) * w3, axis=[1,2,3])
    den = tf.reduce_sum(w3, axis=[1,2,3]) + 1e-6
    return num / den                            # (B,)

def _bump_K_channel(Xk, dk):
    # Xk: (B,T,H,W,5) = RGB(3)+K(1)+mask(1) in [-1,1] for RGB, [0,1] for K/m
    rgb = Xk[..., :3]
    k   = Xk[..., 3:4]
    m   = Xk[..., 4:5]
    k2  = tf.clip_by_value(k + dk, 0.0, 1.0)
    return tf.concat([rgb, k2, m], axis=-1)

def train_loss_with_kmono(
    self, Xk, y_stage2, y_pred_raw,
    mono_sign=+1.0,        # +1: expect K↑ ⇒ Δ↑;  -1: expect K↑ ⇒ Δ↓
    dk=0.20,               # K sweep width (0~1)
    kmono_margin=0.01,     # Monotonicity margin
    lambda_kmono=0.05,     # Monotonicity penalty weight
    deltaE_clamp=0.25,      # Optional ΔE upper bound for postprocessing (e.g., 0.25)
    extra_losses=None   # Added: dict like {"L_stable": tensor, "L_trans": tensor}
):
    """
    self: CreatorTrainer instance (uses members: anti_copy_loss, leak_penalty, contrast_loss, etc.)
    Xk      : (B,T,H,W,5)
    y_stage2: (B,H,W,7) = [M_next, last, y_next]
    y_pred_raw: (B,H,W,3) in [-1,1] (raw Creator output)
    """
    # --- Unpack ---
    M_next = y_stage2[..., :1]     # [0,1]
    last   = y_stage2[..., 1:4]    # [-1,1]
    y_next = y_stage2[..., 4:7]    # [-1,1]

    B = tf.shape(Xk)[0]
    T = tf.shape(Xk)[1]
    H = tf.shape(Xk)[2]
    W = tf.shape(Xk)[3]

    # --- Compositing / postprocessing (training-time output) ---
    M_wound   = tf.clip_by_value(M_next, 0.0, 1.0)
    M_support = tf.clip_by_value(Xk[:, -1, ..., 4:5], 0.0, 1.0)
    M_change  = tf.clip_by_value(M_wound * M_support, 0.0, 1.0)
    M_stable  = tf.clip_by_value(tf.nn.relu(M_support - M_change), 0.0, 1.0)
    M_loss = tighten_mask(M_change, ceil=0.88)

    y_comp, M_comp, M3 = postprocess_creator_output(
        y_pred_raw, last, M_loss,
        clip_floor=0.05, clip_ceil=0.95, ksize=3,
        specular_protect=True, sp_thr=0.90, sp_slope=40.0,
        deltaE_clamp=deltaE_clamp
    )

    # --- Common preparation ---
    y01    = (y_comp + 1.0) * 0.5
    last01 = (last   + 1.0) * 0.5
    X01    = (Xk[..., :3] + 1.0) * 0.5
    tgt01  = (y_next + 1.0) * 0.5

    # (A) Reconstruction
    w = soft_weight(M_loss, gamma=0.4); w3 = tf.repeat(w, 3, axis=-1)
    L_recon = tf.reduce_sum(tf.abs(y_next - y_comp) * w3) / (tf.reduce_sum(w3) + 1e-6)

    # (B) Feature matching (Stage-1 encoder)
    X_2d  = tf.reshape(X01, [-1, H, W, 3])
    z_all = self.Proj(self.E(X_2d, training=False), training=False)
    z_all = tf.reshape(z_all, [B, T, -1])
    k_ctx = tf.minimum(tf.constant(CTX_K), T)
    ctx   = z_all[:, T - k_ctx:T, :]

    # Current batch K mean
    k_norm = tf.reduce_mean(Xk[..., 3:4], axis=[1,2,3,4])           # (B,)
    z_goal = tf.stop_gradient(self.P([ctx, tf.expand_dims(k_norm, -1)]))
    z_pred = self.Proj(self.E(y01, training=False), training=False)
    feat_loss = tf.reduce_mean(tf.square(z_pred - z_goal))

    # (C) Δ magnitude / map
    C_scalar   = change_scalar_from_history(Xk, M_loss)
    target_mag = self.a0 + self.a1*C_scalar + self.a2*tf.cast(k_norm, C_scalar.dtype)
    pred_mag   = _roi_delta_mag(y01, last01, M_loss)
    L_mag      = tf.reduce_mean(tf.abs(pred_mag - target_mag))

    C_map   = change_map_from_history(Xk, M_loss)
    pred_m  = tf.reduce_mean(tf.abs(y01 - last01), axis=-1, keepdims=True)
    tgt_m   = tf.reduce_mean(tf.abs(tgt01 - last01), axis=-1, keepdims=True)
    L_map   = tf.reduce_mean(
        tf.reduce_sum(tf.abs(pred_m - tgt_m) * w, axis=[1,2,3]) /
        (tf.reduce_sum(w, axis=[1,2,3]) + 1e-6)
    )

    # (D) Anti-copy: based on raw y_pred_raw (lightweight since composited output zeroes out)
    ac_loss = self.anti_copy_loss(
        y_pred_raw, last, M_loss,
        K_norm=k_norm,
        tau_min=0.01,
        tau_max=self.tau_anti,
        gamma=0.4
    )

    # (E) Direction / high-frequency / contrast / leakage
    L_feat_cos = cosine_loss(z_pred, z_goal)
    L_dir      = delta_dir_loss(y_comp, last, Xk, M_loss)
    L_hf       = edge_ratio_loss(y_comp, Xk, M_loss)

    d_pred = tf.abs(y01 - last01)
    L_leak = self.leak_penalty(d_pred, M_support)
    L_ctr  = self.contrast_loss(y01, tgt01, M_loss)

    z_pred_n = tf.math.l2_normalize(z_pred, axis=-1)
    z_goal_n = tf.math.l2_normalize(z_goal, axis=-1)
    cos = tf.reduce_mean(tf.reduce_sum(z_pred_n * z_goal_n, axis=-1))

    # (F) K-monotonicity penalty (2 extra forward passes, no grad)
    # Lower/raise K
    Xk_lo = _bump_K_channel(Xk, -dk)
    Xk_hi = _bump_K_channel(Xk,  +dk)

    def _forward_mag(Xk_var):
        # Raw Creator output
        y_raw = _first_output(self.creator(Xk_var, training=False))
        # Keep compositing consistent with the same M_loss
        y_c, _, _ = postprocess_creator_output(
            y_raw, last, M_loss,
            clip_floor=0.05, clip_ceil=0.95, ksize=3,
            specular_protect=True, sp_thr=0.90, sp_slope=40.0,
            deltaE_clamp=deltaE_clamp
        )
        y01v = (y_c + 1.0) * 0.5
        return _roi_delta_mag(y01v, last01, M_loss)   # (B,)

    with tf.name_scope("k_mono"):
        mag_lo = tf.stop_gradient(_forward_mag(Xk_lo))
        mag_hi = tf.stop_gradient(_forward_mag(Xk_hi))
        # Expectation: mono_sign * (mag_hi - mag_lo) >= margin
        kmono_core = mono_sign * (mag_hi - mag_lo)
        k_mono_pen_vec = tf.nn.relu(kmono_margin - kmono_core)     # (B,)
        k_mono_pen = tf.reduce_mean(k_mono_pen_vec)

    # Total
    total = ( self.lambda_recon * L_recon
            + self.lambda_feat * feat_loss
            + self.lambda_anti * ac_loss
            + self.lambda_mag  * L_mag
            + self.lambda_map  * L_map
            + 0.2 * L_feat_cos + 0.25 * L_dir + 0.35 * L_hf
            + 0.30 * L_leak    + 0.20 * L_ctr
            + lambda_kmono * k_mono_pen )

    # Return logging dictionary
    logs = {
        "loss": total, "recon": L_recon, "feat": feat_loss, "anti": ac_loss,
        "L_mag": L_mag, "L_map": L_map, "L_feat_cos": L_feat_cos,
        "L_dir": L_dir, "L_hf": L_hf, "leak": L_leak, "contrast": L_ctr,
        "k_mono_pen": k_mono_pen, "pred_mag": tf.reduce_mean(pred_mag),
        "cos": cos
    }

    # Accumulate extra_losses
    if extra_losses:
        for name, (weight, loss_val) in extra_losses.items():
            total = total + weight * loss_val
            logs[name] = loss_val

    logs["loss"] = total
    return total, y_comp, logs
