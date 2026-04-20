"""
BurnSight evaluation metrics / loss functions (segmentation + perceptual)
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

def psnr(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    if y_true.shape[-1] >= 3:
        y_true = y_true[..., :3]
    if y_pred.shape[-1] >= 3:
        y_pred = y_pred[..., :3]

    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def ssim(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Clip to RGB only when channels >= 3
    if y_true.shape[-1] >= 3:
        y_true = y_true[..., :3]
    if y_pred.shape[-1] >= 3:
        y_pred = y_pred[..., :3]

    return tf.image.ssim(y_true, y_pred, max_val=1.0)

def rmse(y_true, y_pred):
    y_true = tf.cast(y_true[..., :-1], tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def mean_abs_delta(a, b, roi):
    d = tf.abs(a - b)
    w = tf.cast(roi, d.dtype)
    return tf.reduce_sum(d*w) / (tf.reduce_sum(w)+1e-6)


def perceptual_loss(y_true, y_pred):
    if len(y_true.shape) == 5:
        y_true_rgb = tf.squeeze(y_true, axis=1)
        # Remove alpha channel only if present
        if y_true_rgb.shape[-1] > 3:
            y_true_rgb = y_true_rgb[..., :3]  # Keep RGB channels only
    else:
        y_true_rgb = y_true

    # Process y_pred the same way
    if len(y_pred.shape) == 5:
        y_pred = tf.squeeze(y_pred, axis=1)
        if y_pred.shape[-1] > 3:
            y_pred = y_pred[..., :3]
    else:
        y_pred = y_pred

    # tanh [-1, 1] → [0, 255]
    y_true_proc = vgg_preprocess((y_true_rgb + 1.0) * 127.5)
    y_pred_proc = vgg_preprocess((y_pred + 1.0) * 127.5)

    # Desired layers
    layer_names = ['block3_conv3', 'block1_conv1', 'block2_conv1']
    weights = [0.5, 0.2, 0.3]

    loss = 0.0
    for name, w in zip(layer_names, weights):
        feature_model = vgg_feature_models[name]
        f_true = feature_model(y_true_proc)
        f_pred = feature_model(y_pred_proc)
        loss += w * tf.reduce_mean(tf.square(f_true - f_pred))
    return tf.cast(loss, tf.float32)

def perceptual_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    if y_true.shape[-1] >= 3:
        y_true_rgb = y_true[..., :3]
    else:
        raise ValueError("y_true must have at least 3 channels for perceptual_metric.")

    if y_pred.shape[-1] >= 3:
        y_pred = y_pred[..., :3]
    else:
        raise ValueError("y_pred must have at least 3 channels fr perceptual_metric.")

    y_true_preprocessed = preprocess_input(y_true_rgb)
    y_pred_preprocessed = preprocess_input(y_pred)

    model = tf.keras.Model(inputs=vgg.input, outputs=vgg.get_layer('block5_conv2').output)

    if len(y_true_rgb.shape) == 3:
        y_true_rgb = tf.expand_dims(y_true_rgb, axis=0)
    if len(y_pred.shape) == 3:
        y_pred = tf.expand_dims(y_pred, axis=0)

    if len(y_true_rgb.shape) == 5:
        y_true_rgb = tf.squeeze(y_true_rgb, axis=1)
    if len(y_pred.shape) == 5:
        y_pred = tf.squeeze(y_pred, axis=1)

    y_true_features = tf.cast(model(y_true_rgb), dtype=tf.float32)
    y_pred_features = tf.cast(model(y_pred), dtype=tf.float32)

    y_true_features = tf.debugging.check_numerics(y_true_features, "y_true_features contains NaN or Inf")
    y_pred_features = tf.debugging.check_numerics(y_pred_features, "y_pred_features contains NaN or Inf")

    return tf.reduce_mean(tf.square(y_true_features - y_pred_features))

def style_loss(y_true, y_pred):
    """
    Function for computing Style Loss.
    """
    # Adjust dimensions
    y_true = tf.reshape(y_true, [-1, 64, 64, 3])
    y_pred = tf.reshape(y_pred, [-1, 64, 64, 3])

    # Convert data type to float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    def gram_matrix(x):
        x = tf.cast(x, tf.float32)  # Cast to float32
        x = tf.linalg.einsum('bijc,bijd->bcd', x, x)
        return x / tf.cast(tf.size(x), tf.float32)

    y_true_features = vgg(y_true)
    y_pred_features = vgg(y_pred)

    # Compute Gram Matrix
    y_true_gram = gram_matrix(y_true_features)
    y_pred_gram = gram_matrix(y_pred_features)

    # Compute style loss
    loss = tf.reduce_mean(tf.square(y_true_gram - y_pred_gram))
    return tf.cast(loss, tf.float32)

def edge_loss(y_true, y_pred):

    # 1) Force to tensor & float32
    y_true = tf.convert_to_tensor(y_true, tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, tf.float32)

    # 2) Normalize to [0,1] (input may be in [-1,1])
    def to_01(x):
        x_min = tf.reduce_min(x)
        x_max = tf.reduce_max(x)
        # Convert to [0,1] if input appears to be in [-1,1]
        return tf.where(x_min < 0.0, (x + 1.0) * 0.5, x)
    y_true = tf.clip_by_value(to_01(y_true), 0.0, 1.0)
    y_pred = tf.clip_by_value(to_01(y_pred), 0.0, 1.0)

    # 3) Ensure 4D: (H,W,C) -> (1,H,W,C), (H,W) -> (1,H,W,1)
    def ensure_4d(x):
        r = tf.rank(x)
        x = tf.cond(tf.equal(r, 2), lambda: tf.expand_dims(tf.expand_dims(x, -1), 0),
                    lambda: tf.cond(tf.equal(r, 3), lambda: tf.expand_dims(x, 0),
                                    lambda: x))
        return x

    y_true4 = ensure_4d(y_true)
    y_pred4 = ensure_4d(y_pred)

    # 4) Match size/channels (if needed)
    th, tw = tf.shape(y_true4)[1], tf.shape(y_true4)[2]
    y_pred4 = tf.image.resize(y_pred4, (th, tw), method="bilinear")

    # If channel counts differ, use first 3 channels or tile
    c_true = tf.shape(y_true4)[-1]
    c_pred = tf.shape(y_pred4)[-1]
    y_pred4 = tf.cond(tf.equal(c_pred, c_true),
                      lambda: y_pred4,
                      lambda: (tf.tile(y_pred4, [1,1,1,c_true]) if tf.equal(c_pred,1) else y_pred4[..., :c_true]))

    # 5) Sobel edges: [B,H,W,C,2] (dy, dx)
    e_true = tf.image.sobel_edges(y_true4)
    e_pred = tf.image.sobel_edges(y_pred4)

    # 6) Edge magnitude (dy/dx is in the last axis=2)
    mag_true = tf.reduce_mean(tf.abs(e_true), axis=-1)  # -> [B,H,W,C]
    mag_pred = tf.reduce_mean(tf.abs(e_pred), axis=-1)

    # 7) L1 difference
    return tf.reduce_mean(tf.abs(mag_true - mag_pred))


def seg_loss(y_t, y_p):
    return 0.5 * weighted_bce_dynamic(y_t, y_p) + 0.5 * dice_loss(y_t, y_p)

mask = unet(input_shape=(64,64,3), weights='imagenet')
mask.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
             loss=seg_loss,
             metrics=[soft_dice_metric])

ckpt   = ModelCheckpoint('unet5_best.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
plateau= ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-5, verbose=1)
early  = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

history = mask.fit(
    seg_train,
    validation_data=seg_val,
    epochs=20,
    batch_size=8,                     # 2 is too small → recommend 8~32
    callbacks=[ckpt, plateau, early],
    verbose=1
)

mask.save('unet5_final.keras')

mask_model = tf.keras.models.load_model("unet5_final.keras", compile=False)

mean_all, std_all = [], []
for imgs, _ in seg_train.take(100):     # Sample 100 batches
    mean_all.append(tf.reduce_mean((imgs+1)/2.0, axis=[0,1,2]))  # [-1,1] → [0,1]
    std_all.append(tf.math.reduce_std((imgs+1)/2.0, axis=[0,1,2]))

mean_all = tf.reduce_mean(tf.stack(mean_all), axis=0)
std_all  = tf.reduce_mean(tf.stack(std_all), axis=0)
print("Dataset mean/std in [0,1] scale:", mean_all.numpy(), std_all.numpy())

def soft_dice_wound(y_true, y_pred, eps=1e-6):
    y_t = tf.cast(y_true[..., WOUND_IDX], tf.float32)
    y_p = tf.cast(y_pred[..., WOUND_IDX], tf.float32)
    inter = tf.reduce_sum(y_t * y_p, axis=[1,2])
    denom = tf.reduce_sum(y_t + y_p, axis=[1,2])
    dice = (2.*inter + eps) / (denom + eps)
    return tf.reduce_mean(dice)

def iou_metric(y_true, y_pred, eps=1e-7, wound_idx=1):
    """
    y_true, y_pred : (B, H, W, C)
    wound_idx      : wound class index (e.g., 1)
    """
    y_t = tf.cast(y_true[..., wound_idx], tf.float32)
    y_p = tf.cast(y_pred[..., wound_idx] > 0.5, tf.float32)
    inter = tf.reduce_sum(y_t * y_p, axis=[1,2])
    union = tf.reduce_sum(y_t + y_p, axis=[1,2]) - inter
    iou = (inter + eps) / (union + eps)
    return tf.reduce_mean(iou)

def ensure_binary01(y, threshold=0.5):
    """
    y: np.ndarray or tf.Tensor, (H,W) or (H,W,1)
    """
    if isinstance(y, tf.Tensor):
        y = tf.cast(y > threshold, tf.uint8)
    else:
        y = (y > threshold).astype(np.uint8)
    return y

def find_threshold_otsu(prob):
    """Auto threshold using Otsu (no GT required)."""
    p = prob[..., 0] if prob.ndim == 3 else prob
    p = np.clip(p, 0.0, 1.0).astype(np.float32)
    hist, bin_edges = np.histogram(p, bins=256, range=(0,1))
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * (bin_edges[:-1] + bin_edges[1:]) / 2) / (weight1 + 1e-8)
    mean2 = (np.cumsum((hist * (bin_edges[:-1] + bin_edges[1:]) / 2)[::-1]) / (weight2[::-1] + 1e-8))[::-1]
    inter_class_var = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:])**2
    idx = np.argmax(inter_class_var)
    thr = float((bin_edges[idx] + bin_edges[idx+1]) / 2.0)
    return thr

def _stats_at_threshold(prob, gt, thr):
    """TP, FP, FN, TN"""
    p = (prob >= thr).astype(np.uint8)
    g = (gt   >= 0.5).astype(np.uint8)
    tp = int(np.sum((p==1) & (g==1)))
    fp = int(np.sum((p==1) & (g==0)))
    fn = int(np.sum((p==0) & (g==1)))
    tn = int(np.sum((p==0) & (g==0)))
    return tp, fp, fn, tn

def find_threshold_dice(prob, gt, thrs=None):
    """Optimize Dice when GT is available."""
    if prob.ndim == 3: prob = prob[..., 0]
    if gt.ndim   == 3: gt   = gt[..., 0]
    prob = np.clip(prob, 0.0, 1.0).astype(np.float32)
    gt   = (gt >= 0.5).astype(np.uint8)
    if thrs is None:
        thrs = np.linspace(0.05, 0.95, 19)
    best_thr, best_dice = 0.5, -1.0
    for t in thrs:
        tp, fp, fn, _ = _stats_at_threshold(prob, gt, t)
        dice = (2*tp) / (2*tp + fp + fn + 1e-8)
        if dice > best_dice:
            best_dice, best_thr = dice, float(t)
    return best_thr, best_dice

def find_threshold_youden(prob, gt, thrs=None):
    """Maximize Youden J (TPR-FPR) when GT is available."""
    if prob.ndim == 3: prob = prob[..., 0]
    if gt.ndim   == 3: gt   = gt[..., 0]
    prob = np.clip(prob, 0.0, 1.0).astype(np.float32)
    gt   = (gt >= 0.5).astype(np.uint8)
    if thrs is None:
        thrs = np.linspace(0.05, 0.95, 19)
    best_thr, best_j = 0.5, -1.0
    P = np.sum(gt==1); N = np.sum(gt==0)
    P = max(P, 1); N = max(N, 1)
    for t in thrs:
        tp, fp, fn, tn = _stats_at_threshold(prob, gt, t)
        tpr = tp / P
        fpr = fp / N
        j = tpr - fpr
        if j > best_j:
            best_j, best_thr = j, float(t)
    return best_thr, best_j

def _dice_macro(pt, gt):
    # pt, gt: (H,W) in {0,1}
    inter = (pt & gt).sum()
    s = pt.sum() + gt.sum()
    return (2*inter) / (s + 1e-8)

def _dice_macro_with_empty(prob, gt, thr, empty_policy="neg_ok"):
    """empty_policy: 'neg_ok' (if GT=0 and pt.sum()==0 → 1.0, else 0.0),
                     'skip'   (exclude the sample),
                     'zero'   (always 0.0)"""
    if gt.sum() == 0:
        if empty_policy == "neg_ok":
            return 1.0 if (prob >= thr).sum() == 0 else 0.0
        elif empty_policy == "skip":
            return None
        else:
            return 0.0
    pt = (prob >= thr).astype(np.uint8)
    return _dice_macro(pt, gt)


def _extract_gt_wound_binary(gts, wound_idx):
    gts_np = gts.numpy() if hasattr(gts, "numpy") else np.asarray(gts)

    if gts_np.ndim == 4:
        # one-hot: (B,H,W,C)
        if not (0 <= wound_idx < gts_np.shape[-1]):
            raise ValueError(f"wound_idx out of range for one-hot GT: {wound_idx}, C={gts_np.shape[-1]}")
        gt_b = gts_np[..., wound_idx]
        gt_b = (gt_b >= 0.5).astype(np.uint8)

    elif gts_np.ndim == 3:
        # label map: (B,H,W) with int labels
        gt_b = (gts_np == wound_idx).astype(np.uint8)

    else:
        raise ValueError(f"Unexpected GT ndim: {gts_np.ndim}, shape={gts_np.shape}")

    return gt_b  # (B,H,W) uint8 {0,1}

def per_sample_best_thr(pw, gt, thr_grid):
        """
        pw: (H,W) prob in [0,1]
        gt: (H,W) {0,1}
        return: thr that maximizes Dice for this sample
        """
        best_thr, best_d = None, -1.0
        gt01 = (gt >= 0.5).astype(np.uint8)

        for thr in thr_grid:
            pred01 = (pw >= thr).astype(np.uint8)
            d = dice_binary(pred01, gt01)
            if d > best_d:
                best_d, best_thr = d, thr

        return float(best_thr), float(best_d)

def to_np(x, dtype=None):
    if tf.is_tensor(x):
        x = x.numpy()
    x = np.asarray(x)
    if dtype is not None:
        x = x.astype(dtype)
    return x
