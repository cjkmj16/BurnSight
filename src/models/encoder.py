"""
BurnSight Stage-1 encoder / projection head / temporal predictor
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

def extract_patches_2d(x, k=32, s=32):
    # x: (B,H,W,C) -> (B, Np, k*k*C)
    patches = tf.image.extract_patches(
        images=x, sizes=[1,k,k,1], strides=[1,s,s,1], rates=[1,1,1,1], padding='VALID'
    )
    B = tf.shape(x)[0]
    Np = tf.shape(patches)[1] * tf.shape(patches)[2]
    patches = tf.reshape(patches, [B, Np, k*k*tf.shape(x)[-1]])
    return patches

# === Cell 1: Encoder & Projection Head ===
def build_encoder(input_shape=(64,64,3), width=64):
    inp = layers.Input(input_shape)
    x = inp
    for ch in [width, width*2, width*4]:
        x = layers.Conv2D(ch, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x); x = layers.LeakyReLU(0.2)(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    z = layers.Dense(LATENT_DIM, use_bias=False)(x)
    z = L2Normalize(axis=-1, name="z_l2")(z)
    return tf.keras.Model(inp, z, name="E")

def build_proj_head(dim_in=LATENT_DIM, dim_out=LATENT_DIM):
    inp = layers.Input((dim_in,))
    x = layers.Dense(512, activation='relu')(inp)
    x = layers.Dense(dim_out, use_bias=False)(x)
    out = L2Normalize(axis=-1, name="proj_l2")(x)
    return tf.keras.Model(inp, out, name="Proj")

# Instantiate model (H,W,C are automatically inferred from your data)
# Change input_shape=(H,W,C) if needed
E = build_encoder(input_shape=(64,64,3))
Proj = build_proj_head()

# === Cell 2: Temporal Predictor (GRU) ===
def build_temporal_predictor(latent_dim=LATENT_DIM):
    z_in  = layers.Input((None, latent_dim))   # (B,T,D)
    k_in  = layers.Input((1,))                 # (B,1)  k_norm

    k_e   = layers.Dense(128, activation='relu')(k_in)
    k_e   = layers.Dense(latent_dim, activation=None)(k_e)   # (B,D)

    x     = layers.GRU(512, return_sequences=False)(z_in)
    x     = layers.Dense(latent_dim, activation=None)(x)     # (B,D)

    summed = layers.Add()([x, k_e])
    out    = L2Normalize(axis=-1, name="P_l2")(summed)
    return tf.keras.Model([z_in, k_in], out, name="P")

P = build_temporal_predictor()

def info_nce(z_anchor, z_pos, temperature=0.07):
    """
    z_anchor: (B, D) - query
    z_pos:    (B, D) - positive key
    Negatives are composed of other samples in the same batch
    """
    # Normalize
    z_anchor = tf.math.l2_normalize(z_anchor, axis=1)
    z_pos    = tf.math.l2_normalize(z_pos, axis=1)

    # Cosine similarity matrix: (B, B)
    logits = tf.matmul(z_anchor, tf.transpose(z_pos)) / temperature

    # Label: each anchor's positive is at the same index
    labels = tf.range(tf.shape(logits)[0])

    # Cross-entropy loss
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,
        logits=logits
    ))
    return loss
