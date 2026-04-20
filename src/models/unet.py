"""
BurnSight segmentation U-Net (MobileNetV2 backbone)
"""
import numpy as np
import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from tensorflow.keras import layers, ops
from tensorflow.keras.layers import Masking, Layer, PReLU, Add, Activation, Lambda, Input, Concatenate, concatenate, MaxPooling2D, MaxPooling3D, AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Conv2DTranspose, UpSampling2D, UpSampling3D, TimeDistributed, Dense, Conv3D, Conv2D, ConvLSTM2D, Flatten, Reshape, RepeatVector, Multiply, BatchNormalization, LayerNormalization, LeakyReLU, ReLU, Dropout, UnitNormalization, SpatialDropout2D, SpatialDropout3D
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import backend as K
from pathlib import Path
import time, pathlib, json, glob
import cv2
import re
import random
import h5py
import hashlib
import cupy as cp
import math
import albumentations as A
import tensorflow_probability as tfp
import imageio.v2 as imageio
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.applications import VGG16, VGG19, ResNet50
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.initializers import Initializer, HeNormal, GlorotUniform, RandomNormal
from google.colab import drive, files
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import pairwise_distances_argmin
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.morphology import dilation, remove_small_objects, remove_small_holes, closing, opening, square, footprint_rectangle
from collections import defaultdict
from datetime import datetime
from datetime import timedelta
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from joblib import Memory
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from tensorflow.keras.utils import to_categorical
from multiprocessing import Pool
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import precision_recall_curve
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mnv2_pi
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess

from src.config import *
from src.models.layers import *

def conv_bn_relu(x, f, k=3):
    x = layers.Conv2D(f, k, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    return layers.LeakyReLU(0.1)(x)

def up_block(x, skip, f):
    x = layers.UpSampling2D((2,2), interpolation='bilinear')(x)
    x = conv_bn_relu(x, f)
    if skip is not None:
        x = layers.Concatenate()([x, skip])
    x = conv_bn_relu(x, f)
    x = conv_bn_relu(x, f)
    return x

def dice_bin(y_true, y_pred, eps=1e-6):
    y_true_f = y_true.reshape(-1).astype(np.uint8)
    y_pred_f = y_pred.reshape(-1).astype(np.uint8)
    inter = (y_true_f & y_pred_f).sum()
    return (2*inter + eps) / (y_true_f.sum() + y_pred_f.sum() + eps)

def soft_dice_metric(y_true, y_pred, eps=1e-6):
    """
    Multi-class one-hot macro Dice (average Dice per class).
    y_true, y_pred: (B,H,W,C) with softmax for y_pred
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    # Per-class Dice
    inter = tf.reduce_sum(y_true * y_pred, axis=[1,2])             # (B,C)
    denom = tf.reduce_sum(y_true + y_pred, axis=[1,2])             # (B,C)
    dice_c = (2. * inter + eps) / (denom + eps)                    # (B,C)
    return tf.reduce_mean(dice_c)                                  # scalar

def weighted_bce_dynamic(y_true, y_pred, eps=1e-6):
    """
    Dynamic weighted BCE (weight adjusted based on positive class ratio)
    y_true, y_pred : (B,H,W,C) tensor
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), eps, 1. - eps)

    # Positive ratio p: mean over batch
    p = tf.reduce_mean(y_true) + eps           # positives
    pos_weight = (1. - p) / p                  # Higher when more sparse

    # Compute BCE
    loss = -(pos_weight * y_true * tf.math.log(y_pred) +
             (1. - y_true) * tf.math.log(1. - y_pred))

    return tf.reduce_mean(loss)

def dice_loss(y_true, y_pred, eps=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred, axis=[1,2])
    denom = tf.reduce_sum(y_true + y_pred, axis=[1,2])
    dice_c = (2. * inter + eps) / (denom + eps)
    return 1. - tf.reduce_mean(dice_c)

@tf.keras.saving.register_keras_serializable(package="custom")
class MNV2Preprocess(layers.Layer):
    def call(self, x):
        x = tf.cast(x, tf.float32)
        return mnv2_pi(x)   # [0,255] → [-1,1]
    def get_config(self):
        return {}

def unet(input_shape=(64,64,3), alpha=1.0, weights='imagenet'):
    inputs = layers.Input(shape=input_shape)
    in_ch = input_shape[-1]
    if in_ch == 3:
        stem = inputs
    else:
        stem = layers.Conv2D(3, 1, padding='same', use_bias=False, name='stem_proj')(inputs)
        stem = layers.BatchNormalization()(stem)

    # Preprocessing for MobileNetV2 (-1~1)
    stem = layers.Rescaling(255.0)(stem)
    stem = MNV2Preprocess(name="mnv2_preprocess")(stem)

    base = tf.keras.applications.MobileNetV2(
        input_tensor=stem, include_top=False, weights=weights, alpha=alpha
    )

    s1 = base.get_layer('block_1_expand_relu').output
    s2 = base.get_layer('block_3_expand_relu').output
    s3 = base.get_layer('block_6_expand_relu').output
    s4 = base.get_layer('block_13_expand_relu').output
    b  = base.get_layer('block_16_project').output

    def conv_bn_relu(x, f, k=3):
        x = layers.Conv2D(f, k, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        return layers.LeakyReLU(0.1)(x)
    def up_block(x, skip, f):
        x = layers.UpSampling2D((2,2), interpolation='bilinear')(x)
        x = conv_bn_relu(x, f)
        if skip is not None:
            x = layers.Concatenate()([x, skip])
        x = conv_bn_relu(x, f); x = conv_bn_relu(x, f)
        return x

    d1 = up_block(b,  s4, 256)
    d2 = up_block(d1, s3, 128)
    d3 = up_block(d2, s2,  64)
    d4 = up_block(d3, s1,  32)
    d5 = layers.UpSampling2D((2,2), interpolation='bilinear')(d4)
    d5 = conv_bn_relu(d5, 32)
    out = layers.Conv2D(NUM_CLASSES, 1, activation='softmax')(d5)
    return Model(inputs, out)

class MNV2Preprocess(layers.Layer):
    def call(self, x):
        x = tf.cast(x, tf.float32)
        return mnv2_pi(x)   # [0,255] → [-1,1]
    def get_config(self):
        return {}

def unet(input_shape=(64,64,3), alpha=1.0, weights='imagenet'):
    inputs = layers.Input(shape=input_shape)
    in_ch = input_shape[-1]
    if in_ch == 3:
        stem = inputs
    else:
        stem = layers.Conv2D(3, 1, padding='same', use_bias=False, name='stem_proj')(inputs)
        stem = layers.BatchNormalization()(stem)

    # Preprocessing for MobileNetV2 (-1~1)
    stem = layers.Rescaling(255.0)(stem)
    stem = MNV2Preprocess(name="mnv2_preprocess")(stem)

    base = tf.keras.applications.MobileNetV2(
        input_tensor=stem, include_top=False, weights=weights, alpha=alpha
    )

    s1 = base.get_layer('block_1_expand_relu').output
    s2 = base.get_layer('block_3_expand_relu').output
    s3 = base.get_layer('block_6_expand_relu').output
    s4 = base.get_layer('block_13_expand_relu').output
    b  = base.get_layer('block_16_project').output

    def conv_bn_relu(x, f, k=3):
        x = layers.Conv2D(f, k, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        return layers.LeakyReLU(0.1)(x)
    def up_block(x, skip, f):
        x = layers.UpSampling2D((2,2), interpolation='bilinear')(x)
        x = conv_bn_relu(x, f)
        if skip is not None:
            x = layers.Concatenate()([x, skip])
        x = conv_bn_relu(x, f); x = conv_bn_relu(x, f)
        return x

    d1 = up_block(b,  s4, 256)
    d2 = up_block(d1, s3, 128)
    d3 = up_block(d2, s2,  64)
    d4 = up_block(d3, s1,  32)
    d5 = layers.UpSampling2D((2,2), interpolation='bilinear')(d4)
    d5 = conv_bn_relu(d5, 32)
    out = layers.Conv2D(NUM_CLASSES, 1, activation='softmax')(d5)
    return Model(inputs, out)

class MNV2Preprocess(layers.Layer):
    def call(self, x):
        x = tf.cast(x, tf.float32)
        return mnv2_pi(x)   # [0,255] → [-1,1]
    def get_config(self):
        return {}

def unet(input_shape=(64,64,3), alpha=1.0, weights='imagenet'):
    inputs = layers.Input(shape=input_shape)
    in_ch = input_shape[-1]
    if in_ch == 3:
        stem = inputs
    else:
        stem = layers.Conv2D(3, 1, padding='same', use_bias=False, name='stem_proj')(inputs)
        stem = layers.BatchNormalization()(stem)

    # Preprocessing for MobileNetV2 (-1~1)
    stem = layers.Rescaling(255.0)(stem)
    stem = MNV2Preprocess(name="mnv2_preprocess")(stem)

    base = tf.keras.applications.MobileNetV2(
        input_tensor=stem, include_top=False, weights=weights, alpha=alpha
    )

    s1 = base.get_layer('block_1_expand_relu').output
    s2 = base.get_layer('block_3_expand_relu').output
    s3 = base.get_layer('block_6_expand_relu').output
    s4 = base.get_layer('block_13_expand_relu').output
    b  = base.get_layer('block_16_project').output

    def conv_bn_relu(x, f, k=3):
        x = layers.Conv2D(f, k, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        return layers.LeakyReLU(0.1)(x)
    def up_block(x, skip, f):
        x = layers.UpSampling2D((2,2), interpolation='bilinear')(x)
        x = conv_bn_relu(x, f)
        if skip is not None:
            x = layers.Concatenate()([x, skip])
        x = conv_bn_relu(x, f); x = conv_bn_relu(x, f)
        return x

    d1 = up_block(b,  s4, 256)
    d2 = up_block(d1, s3, 128)
    d3 = up_block(d2, s2,  64)
    d4 = up_block(d3, s1,  32)
    d5 = layers.UpSampling2D((2,2), interpolation='bilinear')(d4)
    d5 = conv_bn_relu(d5, 32)
    out = layers.Conv2D(NUM_CLASSES, 1, activation='softmax')(d5)
    return Model(inputs, out)

@tf.keras.saving.register_keras_serializable(package="custom")
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

