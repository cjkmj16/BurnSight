"""
BurnSight image utilities — normalization, overlay, filtering, CLAHE, etc.
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

def cast_layer(x):
    return tf.cast(x, tf.float32)

def normalize_images(images):
    return (images / 127.5) - 1

def denormalize_images(images):
    images = (images + 1.0) * 127.5
    return np.clip(images, 0, 255).astype(np.uint8)

def preprocess(image):
    return tf.cast(image, tf.float32) / 127.5 - 1.0

vgg = VGG19(include_top=False, weights='imagenet', input_shape=(64, 64, 3))
vgg.trainable = False

vgg_feature_models = {
    name: tf.keras.Model(inputs=vgg.input, outputs=vgg.get_layer(name).output)
    for name in ['block1_conv1', 'block2_conv1', 'block3_conv3', 'block4_conv1', 'block5_conv2']
}


class FloatCastLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.cast(inputs, tf.float32)

def get_vgg_features_model(layer_name='block3_conv3'):
    vgg = VGG19(include_top=False, weights='imagenet', input_shape=(64, 64, 3))
    vgg.trainable = False
    base_model = Model(inputs=vgg.input, outputs=vgg.get_layer(layer_name).output)

    inputs = vgg.input
    outputs = base_model(inputs)
    outputs = FloatCastLayer()(outputs)

    return Model(inputs, outputs)

def print_graph_operations():
    graph = tf.compat.v1.get_default_graph()
    operations = graph.get_operations()
    for op in operations:
        print(op.name)

print_graph_operations()

def check_for_printv2():
    graph = tf.compat.v1.get_default_graph()
    operations = graph.get_operations()
    found = False
    for op in operations:
        if 'PrintV2' in op.name:
            print(f"Found PrintV2 operation: {op.name}")
            found = True

    if not found:
        print("No PrintV2 operations found.")

def convert_bgr_to_rgb_and_normalize(images):
    """
    Convert BGR image to RGB and normalize.
    :param images: Image array (data in BGR channel order)
    :return: Image converted to RGB and normalized
    """
    rgb_images = []
    for img in images:
        # BGR -> RGB conversion
        img_rgb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)

        # Normalize
        normalized_img = img_rgb.astype(np.float32) / 255.0
        rgb_images.append(normalized_img)

    return np.array(rgb_images)

def imread_gray(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def imread_color(path):
    return cv2.imread(path)  # BGR

def resize01(x, size=(64,64), gray=False):
    if gray:
        x = cv2.resize(x, size, interpolation=cv2.INTER_NEAREST)
        return (x.astype(np.float32) / 255.0)
    else:
        x = cv2.resize(x, size, interpolation=cv2.INTER_AREA)
        return (x.astype(np.float32) / 255.0)

def redness_prior_bgr01(img01):
    """img01: (H,W,3) BGR in [0,1] → redness map [0,1]"""
    b,g,r = img01[...,0], img01[...,1], img01[...,2]
    # Simple redness: R - max(G,B), normalized to 0~1
    raw = r - np.maximum(g, b)
    raw = np.clip(raw, 0, 1)
    # Slight brightness correction (optional): histogram equalization effect
    raw_u8 = (raw*255).astype(np.uint8)
    eq = cv2.equalizeHist(raw_u8).astype(np.float32)/255.0
    # Blend
    return np.clip(0.7*raw + 0.3*eq, 0, 1)

def morph_clean01(m01, open_ks=3, close_ks=5, open_iter=1, close_iter=1):
    u8 = (m01>0).astype(np.uint8)*255
    if open_iter>0:
        u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN,  np.ones((open_ks,open_ks),np.uint8), open_iter)
    if close_iter>0:
        u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, np.ones((close_ks,close_ks),np.uint8), close_iter)
    return (u8>0).astype(np.uint8)

def keep_largest_cc(m01):
    u8 = (m01>0).astype(np.uint8)
    num, lbl, stats, _ = cv2.connectedComponentsWithStats(u8, 8)
    if num <= 1:
        return u8
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + np.argmax(areas)
    return (lbl==idx).astype(np.uint8)


def overlay_mask_on_image(img01, prob, thr="auto", method="otsu",
                          gt=None, alpha=0.4, color=(1.0,0.0,0.0),
                          use="prob"):
    """
    return: overlay (H,W,3), used_thr (float), used_mask (H,W)
    """
    mask_src = gt if (use == "gt" and gt is not None) else prob

    # 2) Determine threshold
    if thr == "auto":
        if method == "dice" and gt is not None:
            thr, _ = find_threshold_dice(prob, gt)
        elif method == "youden" and gt is not None:
            thr, _ = find_threshold_youden(prob, gt)
        else:
            thr = find_threshold_otsu(prob)  # Default: Otsu
    # If thr is a manual number, use as-is

    # 3) Convert to single channel
    if mask_src.ndim == 3:
        mask_src = mask_src[..., 0]

    # 4) Binarize + overlay
    binm = (mask_src >= float(thr)).astype(np.float32)
    overlay = img01.copy()
    overlay[binm == 1] = (1 - alpha) * overlay[binm == 1] + alpha * np.array(color)
    return overlay, float(thr), binm

def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)

def gaussian_kernel(size: int, sigma: float):
    x = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
    x = tf.exp(-0.5 * (x / sigma) ** 2)
    kernel = x / tf.reduce_sum(x)
    return kernel

def gaussian_blur(x, filter_size=3, sigma=1.0):
    channels = x.shape[-1]
    kernel_1d = gaussian_kernel(filter_size, sigma)
    kernel_2d = tf.tensordot(kernel_1d, kernel_1d, axes=0)  # shape: (k, k)
    kernel_2d = kernel_2d[:, :, tf.newaxis, tf.newaxis]     # shape: (k, k, 1, 1)
    kernel_2d = tf.tile(kernel_2d, [1, 1, channels, 1])      # shape: (k, k, C, 1)

    # padding
    pad = filter_size // 2
    x = tf.pad(x, [[0,0], [pad,pad], [pad,pad], [0,0]], mode='REFLECT')

    return tf.nn.depthwise_conv2d(x, kernel_2d, strides=[1,1,1,1], padding='VALID')

def conv_block(x, filters, use_batchnorm=True):
    x = Conv2D(filters, kernel_size=3, strides=2, padding='same')(x)
    if use_batchnorm:
        x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    return x

def deconv_block(x, skip, filters, dropout=False):
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters, kernel_size=3, strides=1, padding='same')(x)
    if dropout:
        x = Dropout(0.1)(x)
    x = LeakyReLU(0.1)(x)
    x = Concatenate()([x, skip])
    return x

CALIB_PATH = pathlib.Path("./mask_thr_calib_v1.npz")
