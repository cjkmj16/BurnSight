"""
BurnSight tf.data builder / decoder
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

from src.data.file_utils import *

# ------- 3) Decoder -------
def decode_image_tf(path):
    # path: scalar tf.string
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.AREA)
    img = tf.cast(img, tf.float32) / 255.0   # [0,1]
    img.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
    return img

def decode_mask_npz_tf(path, num_classes):
    # path: scalar tf.string
    def _load(p):
        # p must be bytes or str.
        if isinstance(p, bytes):
            p = p.decode("utf-8")
        with _np.load(p, allow_pickle=False) as data:
            m = data["masks"].astype(_np.float32)  # (H,W,C) or (H,W)
            if m.ndim == 2:                        # (H,W) → (H,W,1)
                m = m[..., None]
        return m

    m = tf.numpy_function(_load, [path], tf.float32)  # (H,W,C?)
    # Fix shape information in the graph
    m.set_shape((None, None, None))
    # Align to desired class axis (e.g., C=num_classes)
    m = tf.image.resize(m, IMG_SIZE, method="nearest")
    m.set_shape((IMG_SIZE[0], IMG_SIZE[1], None))  # C is fixed below
    # If C=num_classes is always guaranteed:
    m = tf.ensure_shape(m, (IMG_SIZE[0], IMG_SIZE[1], num_classes))
    return m

def pairs_to_dataset(pairs, num_classes):
    img_paths = tf.constant([p[0] for p in pairs], dtype=tf.string)
    msk_paths = tf.constant([p[1] for p in pairs], dtype=tf.string)
    ds = tf.data.Dataset.from_tensor_slices((img_paths, msk_paths))
    ds = ds.map(lambda ip, mp: (decode_image_tf(ip), decode_mask_npz_tf(mp, num_classes)),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds

def decode_seq_images_tf(paths_t):
    imgs = tf.map_fn(decode_image_tf, paths_t, fn_output_signature=tf.float32)
    return imgs  # (T,H,W,3)

def decode_seq_masks_tf(paths_t, num_classes):
    msks = tf.map_fn(lambda p: decode_mask_npz_tf(p, num_classes),
                     paths_t, fn_output_signature=tf.float32)
    return msks

def build_seq_dataset(seq_imgs, seq_msks, num_classes, batch_size, training=True, seed=SEED):
    ds = tf.data.Dataset.from_tensor_slices((seq_imgs, seq_msks))

    def load_seq(img_path_vec, msk_path_vec):
        # img_path_vec: (T,) tf.string
        # msk_path_vec: (T,) tf.string

        imgs = tf.map_fn(
            fn=lambda p: decode_image_tf(p),
            elems=img_path_vec,
            fn_output_signature=tf.float32
        )  # -> (T, H, W, 3)

        msks = tf.map_fn(
            fn=lambda p: decode_mask_npz_tf(p, num_classes),
            elems=msk_path_vec,
            fn_output_signature=tf.float32
        )  # -> (T, H, W, num_classes)

        # Fix static shape explicitly
        imgs.set_shape((SEQ_LEN, IMG_SIZE[0], IMG_SIZE[1], 3))
        msks.set_shape((SEQ_LEN, IMG_SIZE[0], IMG_SIZE[1], num_classes))
        return imgs, msks

    if training:
        ds = ds.shuffle(buffer_size=len(seq_imgs), seed=seed, reshuffle_each_iteration=False)

    ds = ds.map(load_seq, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)

    return ds

# ------- 4) Build seg_train / seg_val -------
train_ds = pairs_to_dataset(train_pairs, num_classes=NUM_CLASSES)
val_ds   = pairs_to_dataset(val_pairs,   num_classes=NUM_CLASSES)

# Scale images to [-1,1] (optional)
def scale_pair_to_m11(img, msk):
    return (img*2.0 - 1.0, msk)

seg_train = (pairs_to_dataset(train_pairs, NUM_CLASSES)      # Seed shuffle recommended internally too
             .shuffle(buffer_size=len(train_pairs), seed=SEED, reshuffle_each_iteration=False)
             .map(scale_pair_to_m11, num_parallel_calls=4)
             .batch(8, drop_remainder=False)
             .prefetch(4))

seg_val = (pairs_to_dataset(val_pairs, NUM_CLASSES)
           .shuffle(1, reshuffle_each_iteration=False)
           .map(scale_pair_to_m11, num_parallel_calls=1)
           .batch(8, drop_remainder=False)
           .prefetch(1))

def calibrate_thr_stable(
    probs_list,
    gts_list,
    thr_grid=None,
    anchor_thr=None,
    trim_ratio=0.10,     # Trim top/bottom 10% (optional)
    min_samples=10
):
    """
    Absolutely stable calibration
    """
    if thr_grid is None:
        thr_grid = np.linspace(0.01, 0.99, 400)

    thr_list = []
    dice_list = []

    for pw, gt in zip(probs_list, gts_list):
        if gt.sum() == 0:
            continue  # keep pos-only

        thr_i, d_i = per_sample_best_thr(pw, gt, thr_grid)
        thr_list.append(thr_i)
        dice_list.append(d_i)

    if len(thr_list) < min_samples:
        raise RuntimeError(f"Not enough samples for stable calibration: {len(thr_list)}")

    thr_arr = np.array(thr_list, dtype=np.float32)

    # --- Optional trimming (remove outliers) ---
    if trim_ratio > 0:
        lo = int(len(thr_arr) * trim_ratio)
        hi = len(thr_arr) - lo
        thr_arr = np.sort(thr_arr)[lo:hi]

    thr_med = float(np.median(thr_arr))

    # --- Anchor blend (optional) ---
    if anchor_thr is not None:
        thr_med = float(0.8 * thr_med + 0.2 * anchor_thr)

    stats = {
        "n": len(thr_arr),
        "thr_median": thr_med,
        "thr_mean": float(np.mean(thr_arr)),
        "thr_std": float(np.std(thr_arr)),
        "dice_mean": float(np.mean(dice_list)),
    }

    return thr_med, stats

seg_val_calib = (
    pairs_to_dataset(val_pairs, NUM_CLASSES)
    .map(scale_pair_to_m11, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(8, drop_remainder=False)
)

opt = tf.data.Options()
opt.experimental_deterministic = True
seg_val = seg_val.with_options(opt)

# Note: number of classes can be inferred automatically from the last axis of npz
x_sample, y_sample = next(iter(seg_train.take(1)))
assert x_sample.shape.rank == 4 and y_sample.shape.rank == 4, "U-Net requires (B,H,W,C) shape!"

def list_npz(root):
    files = []
    for r,_,fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(".npz"):
                files.append(os.path.join(r,f))
    return sorted(files)

def load_mask_npz_tf(path, num_classes):
    def _load(p):
        import numpy as _np
        m = _np.load(p.decode("utf-8"), mmap_mode="r")["masks"]  # (H,W,C) one-hot or (H,W) integer
        if m.ndim == 3:
            # one-hot -> float32
            return m.astype(_np.float32)
        else:
            # Convert integer label -> one-hot (can branch to bincount for performance)
            C = num_classes
            oh = _np.zeros((*m.shape, C), dtype=_np.float32)
            for c in range(C):
                oh[..., c] = (m == c).astype(_np.float32)
            return oh
    m = tf.numpy_function(_load, [path], tf.float32)
    m.set_shape([None, None, num_classes])
    return m

def class_pixel_histogram_tfdata(dir_list, num_classes=5, take_ratio=1.0):
    # File list
    files = []
    for d in dir_list:
        files.extend(list_npz(d))
    files = sorted(files)

    # Sampling (optionally use 20% for fast estimation)
    if 0 < take_ratio < 1.0:
        step = max(1, int(1.0 / take_ratio))
        files = files[::step]

    if len(files) == 0:
        return np.zeros(num_classes, dtype=np.int64)

    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.map(lambda p: load_mask_npz_tf(p, num_classes), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda onehot: tf.math.bincount(
                    tf.reshape(tf.argmax(onehot, axis=-1), [-1]),
                    minlength=num_classes, maxlength=num_classes, dtype=tf.int64),
                num_parallel_calls=AUTOTUNE) \
           .batch(256) \
           .prefetch(AUTOTUNE)

    total = ds.reduce(tf.zeros([num_classes], tf.int64), lambda acc, x: acc + tf.reduce_sum(x, axis=0))
    return total.numpy()

# ---------- [3] Cache using file list + mtime hash ----------
def iter_npz_files(dir_list):
    for d in dir_list:
        for r,_,fs in os.walk(d):
            for f in fs:
                if f.lower().endswith(".npz"):
                    yield os.path.join(r,f)

def hash_file_list(paths):
    h = hashlib.sha1()
    for p in sorted(paths):
        h.update(p.encode('utf-8'))
        try:
            st = os.stat(p)
            h.update(str(int(st.st_mtime)).encode('utf-8'))
            h.update(str(st.st_size).encode('utf-8'))
        except OSError:
            pass
    return h.hexdigest()

def cached_histogram_tfdata(dir_list, num_classes=5,
                            cache_path="class_hist_cache.json",
                            take_ratio=1.0):
    files = list(iter_npz_files(dir_list))
    key = f"tfdata::{num_classes}::{take_ratio}::{hash_file_list(files)}"

    db = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                db = json.load(f)
        except Exception:
            db = {}

    if key in db:
        return np.array(db[key], dtype=np.int64)

    # Recompute
    counts = class_pixel_histogram_tfdata(dir_list, num_classes=num_classes, take_ratio=take_ratio)
    db[key] = counts.tolist()
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)
    return counts


def to_stage2(xseq, yseq):
    last_rgb = xseq[:, -1]                                       # (B,64,64,3) [-1,1]
    roi1     = yseq[:, -1, ..., WOUND_IDX:WOUND_IDX+1]           # (B,64,64,1) [0,1]
    y_pack   = tf.concat([roi1, last_rgb], axis=-1)              # (B,64,64,4)
    return xseq, y_pack                                          # xseq: (B,T,64,64,3)

def load_seq(img_paths, msk_paths):
    imgs = tf.map_fn(lambda p: decode_image_tf(p), img_paths, fn_output_signature=tf.float32)
    msks = tf.map_fn(lambda p: decode_mask_npz_tf(p, NUM_CLASSES), msk_paths, fn_output_signature=tf.float32)
    imgs.set_shape((SEQ_LEN, IMG_SIZE[0], IMG_SIZE[1], 3))
    msks.set_shape((SEQ_LEN, IMG_SIZE[0], IMG_SIZE[1], NUM_CLASSES))
    return imgs, msks

def gate_from_prob(prob, best_thr, tau=0.4, ksize=2, floor=0.03, ceil=0.97):
    # prob: (N,H,W,1) in [0,1]  (N = B*T or B)
    p = tf.clip_by_value(prob, 1e-6, 1-1e-6)
    logit_p   = tf.math.log(p) - tf.math.log(1.0 - p)
    logit_thr = tf.math.log(best_thr) - tf.math.log(1.0 - best_thr)
    sharp = tf.sigmoid((logit_p - logit_thr) / tau)            # Reflect best_thr (hard approximation)
    halo  = tf.nn.max_pool2d(sharp, ksize=ksize, strides=1, padding="SAME")
    gate  = tf.clip_by_value(halo, floor, ceil)                # For gating
    m_bin = tf.cast(p > best_thr, tf.float32)                  # For visualization/metrics
    return gate, m_bin

def to_stage2_seq(
    Xseq, Yseq,                         # Xseq: (B,T,H,W,3)[0,1], Yseq: (B,T,H,W,Cy)[0,1 or {0,1}]
    k_norm=0.0, k_tensor=None, wound_idx=1, future_safe=False,
    use_pred_mask=False,                # False: use GT mask / True: use segmentation predicted mask
    pred_seq=None,                      # (B,T_in,H,W,1)[0,1], required when use_pred_mask=True
    best_thr=None, tau=0.5,             # Calibration parameters when using segmentation prediction
    ksize=3, floor=0.03, ceil=0.93
):
    Xseq = tf.convert_to_tensor(Xseq, tf.float32)
    Yseq = tf.convert_to_tensor(Yseq, tf.float32)
    Xseq = tf.clip_by_value(Xseq, 0., 1.)
    Yseq = tf.clip_by_value(Yseq, 0., 1.)

    # Basic check
    tf.debugging.assert_equal(tf.shape(Xseq)[-1], 3,
        message="to_stage2_seq: Xseq must be raw RGB (C=3)")
    Cy = tf.shape(Yseq)[-1]
    tf.debugging.assert_less(tf.cast(wound_idx, tf.int32), Cy)

    B = tf.shape(Xseq)[0]
    T = tf.shape(Xseq)[1]
    H = tf.shape(Xseq)[2]
    W = tf.shape(Xseq)[3]

    # ---- Temporal normalization ----
    T_in = tf.maximum(T - 1, 1)
    def _slice_t(t, x): return x[:, :t, ...] if t > 0 else x[:, :1, ...]
    X_in = _slice_t(T_in, Xseq)              # (B,T_in,H,W,3)
    Y_in = _slice_t(T_in, Yseq)              # (B,T_in,H,W,Cy)

    last_idx = tf.where(T > 1, T - 2, 0)
    next_idx = tf.where(T > 0, T - 1, 0)

    # ---- Labels (last/next) ----
    last_rgb_01 = Xseq[:, last_idx, ...]     # (B,H,W,3) [0,1]
    y_next_01   = Xseq[:, next_idx, ...]     # (B,H,W,3) [0,1]
    last_rgb    = last_rgb_01 * 2.0 - 1.0    # [-1,1]
    y_next      = y_next_01 * 2.0 - 1.0      # [-1,1]
    M_next      = Yseq[:, next_idx, ..., wound_idx:wound_idx+1]  # (B,H,W,1)

    # ---- Build m_seq for gating ----
    if not use_pred_mask:
        # Use GT mask softly
        m_seq = Y_in[..., wound_idx:wound_idx+1]                      # (B,T_in,H,W,1)
        m2d = tf.reshape(m_seq, [B*T_in, H, W, 1])
        m2d = tf.nn.max_pool2d(m2d, ksize=ksize, strides=1, padding="SAME")
        m2d = tf.clip_by_value(m2d, floor, 1.0)
        m_seq = tf.reshape(m2d, [B, T_in, H, W, 1])
    else:
        # Use segmentation model prediction (pred_seq) sharpened at best_thr
        # pred_seq must be (B,T_in,H,W,1) soft probability
        tf.debugging.assert_is_not_none(pred_seq, message="pred_seq is required when use_pred_mask=True.")
        tf.debugging.assert_equal(tf.shape(pred_seq)[:4], tf.stack([B, T_in, H, W]))
        tf.debugging.assert_is_not_none(best_thr, message="best_thr is required when use_pred_mask=True.")

        p2d = tf.reshape(tf.convert_to_tensor(pred_seq, tf.float32), [B*T_in, H, W, 1])
        # Assumes gate_from_prob(prob2d, best_thr, tau, ksize, floor, ceil) is defined
        gate_bt, _ = gate_from_prob(p2d, best_thr=best_thr, tau=tau, ksize=ksize)
        gate_bt = tf.clip_by_value(gate_bt, floor, ceil)
        m_seq = tf.reshape(gate_bt, [B, T_in, H, W, 1])

    # Future frame safety (optional): prevent last frame info leakage
    if future_safe and T_in > 1:
        zero_last = tf.zeros_like(m_seq[:, -1:, ...])
        m_seq = tf.concat([m_seq[:, :-1, ...], zero_last], axis=1)

    # ---- K channel ----
    if k_tensor is not None:
        k_b1 = tf.reshape(tf.cast(k_tensor, tf.float32), [B, 1])
    else:
        k_b1 = tf.fill([B, 1], tf.cast(k_norm, tf.float32))
    k_bt111 = tf.reshape(k_b1, [B, 1, 1, 1, 1])
    k_bt111 = tf.repeat(k_bt111, repeats=T_in, axis=1)
    k_exp   = tf.broadcast_to(k_bt111, [B, T_in, H, W, 1])

    # ---- 5-channel input ----
    rgb_m11 = X_in * 2.0 - 1.0                         # (B,T_in,H,W,3)
    Xk      = tf.concat([rgb_m11, k_exp, m_seq], axis=-1)

    # ---- y_stage2 ----
    y_stage2 = tf.concat([M_next, last_rgb, y_next], axis=-1)  # (B,H,W,7)

    # Safety guard
    tf.debugging.assert_equal(tf.shape(y_stage2)[-1], 7)
    tf.debugging.assert_greater_equal(tf.reduce_min(y_stage2[..., :1]), 0.0)
    tf.debugging.assert_less_equal(tf.reduce_max(y_stage2[..., :1]), 1.0)
    tf.debugging.assert_greater_equal(tf.reduce_min(y_stage2[..., 1:4]), -1.0)
    tf.debugging.assert_less_equal(tf.reduce_max(y_stage2[..., 1:4]),  1.0)
    tf.debugging.assert_greater_equal(tf.reduce_min(y_stage2[..., 4:7]), -1.0)
    tf.debugging.assert_less_equal(tf.reduce_max(y_stage2[..., 4:7]),  1.0)

    return Xk, y_stage2

ds_train_seq = build_seq_dataset(train_seq_imgs, train_seq_msks, NUM_CLASSES, batch_size=16)
ds_val_seq   = build_seq_dataset(val_seq_imgs,   val_seq_msks,   NUM_CLASSES, batch_size=16)

ds2_train = ds_train_seq.map(lambda x,y: to_stage2_seq(x, y, k_norm=K_VAL/float(K_MAX), wound_idx=WOUND_IDX, future_safe=False)
               ).prefetch(tf.data.AUTOTUNE)

ds2_val   = ds_val_seq.map(lambda x,y: to_stage2_seq(x, y, k_norm=K_VAL/float(K_MAX), wound_idx=WOUND_IDX, future_safe=False)
               ).prefetch(tf.data.AUTOTUNE)
