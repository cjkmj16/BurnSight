"""
BurnSight ID transform / style calibration / augmented dataset builder
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
