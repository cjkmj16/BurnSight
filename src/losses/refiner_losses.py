"""
BurnSight Refiner training loss function collection
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

from src.utils.metrics import *

def hf_l1_loss(a_m11, b_m11):
    # Applying highpass in [0,1] range is more stable
    a01 = (a_m11 + 1.0) * 0.5
    b01 = (b_m11 + 1.0) * 0.5
    ha = highpass(a01, 9, 1.5)
    hb = highpass(b01, 9, 1.5)
    return tf.reduce_mean(tf.abs(ha - hb))

def hf_roi_loss(refined_m11, target_m11, M_lesion01):   # New function
    a01 = (refined_m11 + 1.0) * 0.5
    b01 = (target_m11  + 1.0) * 0.5
    ha = highpass(a01, 9, 1.5)
    hb = highpass(b01, 9, 1.5)
    # Match high-frequency patterns only outside lesion region
    w_out = 1.0 - tf.clip_by_value(
        tf.cast(M_lesion01, tf.float32), 0.0, 1.0
    )
    if w_out.shape.rank == 3:           # (B,H,W) → (B,H,W,1)
        w_out = w_out[..., tf.newaxis]
    return tf.reduce_sum(tf.abs(ha - hb) * w_out) / (tf.reduce_sum(w_out) + 1e-6)

def out_lab_drift_loss(base_m11, refined_m11, M01):
    """
    base_m11, refined_m11: (B,H,W,3) in [-1,1]
    M01: (H,W) or (B,H,W) in [0,1]  (soft lesion). OUT = 1-M01
    return: scalar
    """
    if M01 is None:
        return tf.constant(0.0, tf.float32)

    base_m11    = tf.cast(base_m11, tf.float32)
    refined_m11 = tf.cast(refined_m11, tf.float32)

    # M01 shape normalize -> (B,H,W,1)
    M01 = tf.cast(M01, tf.float32)
    if M01.shape.rank == 2:          # (H,W)
        M01 = tf.expand_dims(M01, axis=0)      # (1,H,W)
    if M01.shape.rank == 3:          # (B,H,W) or (1,H,W)
        M01 = tf.expand_dims(M01, axis=-1)     # (B,H,W,1)

    w_out = 1.0 - tf.clip_by_value(M01, 0.0, 1.0)  # (B,H,W,1)

    # [-1,1] -> [0,1]
    base01 = (base_m11 + 1.0) * 0.5
    ref01  = (refined_m11 + 1.0) * 0.5

    Lab_b = rgb01_to_lab_tf(base01)
    Lab_r = rgb01_to_lab_tf(ref01)
    d = Lab_r - Lab_b                                 # (B,H,W,3)
    d2 = tf.reduce_sum(tf.square(d), axis=-1, keepdims=True)  # (B,H,W,1)

    return tf.reduce_mean(d2 * w_out)

def prob_all_to_lesion_soft_tf(prob_all, lesion_idxs=LESION_IDXS, clip=True):
    prob_all = tf.cast(prob_all, tf.float32)
    C = tf.shape(prob_all)[-1]

    if isinstance(lesion_idxs, (list, tuple)):
        idx = tf.constant([int(i) for i in lesion_idxs], dtype=tf.int32)
    else:
        idx = tf.cast(lesion_idxs, tf.int32)

    # Use only indices less than C
    idx = idx[idx < C]

    def _single():
        p = prob_all[..., 0]
        return tf.clip_by_value(p, 0.0, 1.0) if clip else p

    def _multi():
        gathered = tf.gather(prob_all, idx, axis=-1)     # (B,H,W,K)
        p = tf.reduce_sum(gathered, axis=-1)            # (B,H,W)
        return tf.clip_by_value(p, 0.0, 1.0) if clip else p

    return tf.cond(tf.equal(C, 1), _single, _multi)

def prob_all_to_healed_soft_tf(prob_all, healed_idx=HEALED_IDX,
                               exclude_idx=EXCLUDE_IDX, exclude_mode="hard", exclude_hard_thr=0.5,
                               clip=True):
    """
    prob_all: (B,H,W,C)
    return : (B,H,W) healed soft in [0,1]
    """
    prob_all = tf.cast(prob_all, tf.float32)
    C = tf.shape(prob_all)[-1]

    def _single():
        # If C==1, there is no healed concept; setting healed=0 is safer
        return tf.zeros(tf.shape(prob_all)[:-1], tf.float32)

    def _multi():
        p = tf.cond(tf.less(int(healed_idx), C),
                    lambda: prob_all[..., int(healed_idx)],
                    lambda: tf.zeros(tf.shape(prob_all)[:-1], tf.float32))
        p = tf.clip_by_value(p, 0.0, 1.0) if clip else p

        # exclude
        if exclude_idx is not None:
            ex = prob_all[..., int(exclude_idx)]
            ex = tf.clip_by_value(ex, 0.0, 1.0)
            if exclude_mode == "soft":
                p = p * (1.0 - ex)
            elif exclude_mode == "hard":
                p = p * tf.cast(ex < float(exclude_hard_thr), tf.float32)
            elif exclude_mode is None:
                pass
            else:
                raise ValueError("exclude_mode must be 'hard', 'soft', or None")

        return tf.clip_by_value(p, 0.0, 1.0) if clip else p

    return tf.cond(tf.equal(C, 1), _single, _multi)

def lesion_from_prob_all_fn_tf(prob_all, lesion_idxs=LESION_IDXS):
    return prob_all_to_lesion_soft_tf(prob_all, lesion_idxs=lesion_idxs)

def healed_from_prob_all_fn_tf(prob_all, healed_idx=HEALED_IDX):
    return prob_all_to_healed_soft_tf(prob_all, healed_idx=healed_idx)

class ConditionalGAN(tf.keras.Model):
    def __init__(self, generator, discriminator, creator_output, target_image, vgg, ema,
                 seg_model=None, lesion_from_prob_all_fn=None,
                 lesion_idxs=LESION_IDXS,
                 seg_expects="m11"
                 ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.example_sequence = example_sequence
        self.initial_creator_output = tf.convert_to_tensor(creator_output, dtype=tf.float32)
        if len(self.initial_creator_output.shape) == 3:
            self.initial_creator_output = tf.expand_dims(self.initial_creator_output, axis=0)
        self.target_image = tf.convert_to_tensor(target_image, dtype=tf.float32)
        if tf.reduce_max(self.target_image) <= 1.0 and tf.reduce_min(self.target_image) >= 0.0:
            self.target_image = self.target_image * 2.0 - 1.0
        self.predicted_future = self.add_weight(
            name="predicted_future",
            shape=self.initial_creator_output.shape,
            dtype=tf.float32,
            initializer=tf.keras.initializers.Constant(self.initial_creator_output),
            trainable=False,
        )
        self.vgg = vgg
        self.seg_model = seg_model
        self.lesion_from_prob_all_fn = lesion_from_prob_all_fn
        self.lesion_idxs = lesion_idxs
        self.seg_expects = seg_expects
        self.g_losses = []
        self.d_losses = []
        self.ema = float(ema)

    def compile(self, g_optimizer, d_optimizer, g_loss_fn, d_loss_fn, run_eagerly=True):
        super().compile(run_eagerly=run_eagerly)
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.g_loss_fn = g_loss_fn
        self.d_loss_fn = d_loss_fn

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.dummy_loss = lambda y_true, y_pred: tf.constant(0.0, dtype=tf.float32)

    def train_step(self, data):
        target_m11 = tf.expand_dims(self.target_image, axis=0)

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            base_m11 = tf.clip_by_value(self.predicted_future, -1.0, 1.0)

            # ---- (1) Generate M01 from seg (from base only) ----
            seg_in = (base_m11 + 1.0) * 0.5 if self.seg_expects == "01" else base_m11
            prob_all = self.seg_model(seg_in, training=False)

            M_lesion01 = lesion_from_prob_all_fn_tf(prob_all, lesion_idxs=self.lesion_idxs)   # (B,H,W)
            M_healed01 = healed_from_prob_all_fn_tf(prob_all, healed_idx=HEALED_IDX)          # (B,H,W)

            M_lesion01 = tf.stop_gradient(tf.clip_by_value(tf.cast(M_lesion01, tf.float32), 0.0, 1.0))
            M_healed01 = tf.stop_gradient(tf.clip_by_value(tf.cast(M_healed01, tf.float32), 0.0, 1.0))

            M_healed01 = tf.clip_by_value(M_healed01, 0.0, 1.0)
            M_lesion01 = tf.clip_by_value(M_lesion01, 0.0, 1.0)

            M_active01 = tf.clip_by_value(M_lesion01 - M_healed01, 0.0, 1.0)
            M_active01 = tf.clip_by_value(M_active01, 0.0, 1.0)

            # ---- (2) G loss: generate refined/fake_logits here ----
            g_loss, refined_image, fake_logits, delta_pred = self.g_loss_fn(
                base_m11=base_m11,
                generator=self.generator,
                discriminator=self.discriminator,
                creator_cond_m11=base_m11,
                vgg=self.vgg,
                target_m11=target_m11,
                M_lesion01=M_lesion01,
                M_healed01=M_healed01,
                M_active01=M_active01,
                lambda_active=2.0,
                lambda_out=10.0,
                lambda_healed=5.0,
                lambda_hf_roi=1.5,
                healed_mode="lab",
            )

            # ---- (3) D loss ----
            real_logits = self.discriminator([grad_mag(target_m11), grad_mag(base_m11)], training=True)
            d_loss = self.d_loss_fn(real_logits=real_logits, fake_logits=fake_logits)

        # ---- (4) Apply gradients ----
        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        if g_grads is None or all(g is None for g in g_grads):
            raise ValueError("🚨 No gradients for Generator. Check loss graph / generator output.")
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        blended = (1-self.ema) * self.predicted_future + self.ema * refined_image
        self.predicted_future.assign(tf.clip_by_value(blended, -1, 1))

        self.g_losses.append(float(g_loss))
        self.d_losses.append(float(d_loss))
        return {"g_loss": g_loss, "d_loss": d_loss}

    def enable_tf_function(self, enable=True):
        if enable:
            self.train_step = tf.function(self.train_step)
        else:
            self.train_step = self.train_step.__wrapped__

DEBUG_MODE = False

def safe_log(x):
    return tf.math.log(tf.clip_by_value(x, 1e-7, 1.0))

predicted_image_rgb = base_m11[None, ...]

test_dataset = tf.data.Dataset.from_tensor_slices(predicted_image_rgb).batch(1).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.map(lambda x: preprocess(x))

print(f"📌 test_dataset type: {type(test_dataset)}")
print(f"📌 test_dataset element_spec: {test_dataset.element_spec}")
print(f"✅ First run - test_dataset size: {len(list(test_dataset))}")
print(f"✅ Second run - test_dataset size: {len(list(test_dataset))}")

for batch in test_dataset.take(1):
    x = batch.numpy()  # Inspect batch data
    print("Sample batch shape:", x.shape)
print("Input x shape:", x.shape)

real_dataset = test_dataset
predicted_image = tf.convert_to_tensor(base_m11, dtype=tf.float32)
predicted_image = predicted_image.numpy()

generator = generator_model(input_shape=(64, 64, 3))
print("Generator type:", type(generator))
if isinstance(generator, tf.keras.Model):
    print("Generator is a valid Keras Model.")
else:
    print("Generator is NOT a valid Keras Model. It is:", type(generator))

def sharpen_creator_output(x, filter_size=3, sigma=1.0, strength=1.0):
    if len(x.shape) == 3:
        x = tf.expand_dims(x, axis=0)

    blurred = gaussian_blur(x, filter_size=filter_size, sigma=sigma)
    sharpened = x + strength * (x - blurred)
    sharpened = tf.clip_by_value(sharpened, 0.0, 1.0)
    return tf.squeeze(sharpened, axis=0) if x.shape[0] == 1 else sharpened

def sharpen_image(image, strength=1.5):
    """Sharpen image using Laplacian filter."""
    if isinstance(image, tf.Tensor):
        image = image.numpy()
    image_uint8 = (image * 255).astype(np.uint8)

    kernel = np.array([[0, -1, 0],
                       [-1, 5 + strength, -1],
                       [0, -1, 0]], dtype=np.float32)

    # Apply sharpening per channel
    sharpened = np.zeros_like(image_uint8)
    for i in range(3):  # R, G, B
        sharpened[..., i] = cv2.filter2D(image_uint8[..., i], -1, kernel)

    # Convert back to [0,1] float32
    sharpened = np.clip(sharpened.astype(np.float32) / 255.0, 0.0, 1.0)
    return tf.convert_to_tensor(sharpened, dtype=tf.float32)

def laplacian_filter(image):
    if tf.rank(image) == 3:
        image = tf.expand_dims(image, axis=0)  # (1, H, W, C)
        single = True
    else:
        single = False

    lap_kernel = tf.constant([[0., 1., 0.],
                              [1., -4., 1.],
                              [0., 1., 0.]], dtype=tf.float32)
    lap_kernel = tf.reshape(lap_kernel, [3, 3, 1, 1])
    lap_kernel = tf.repeat(lap_kernel, repeats=tf.shape(image)[-1], axis=2)

    filtered = tf.nn.conv2d(image, lap_kernel, strides=[1, 1, 1, 1], padding='SAME')

    if single:
        return tf.squeeze(filtered, axis=0)  # Back to (H, W, C)
    return filtered

def sharpen_loss(predicted_image, y_pred, weight=1.0):
    lap_true = laplacian_filter(predicted_image)
    lap_pred = laplacian_filter(y_pred)
    sharpen_improvement = tf.reduce_mean(tf.abs(lap_pred)) - tf.reduce_mean(tf.abs(lap_true))
    return -sharpen_improvement * weight

def healed_to_target_lab_loss(refined_m11, target_m11, M_healed01, eps=1e-6):
    """
    refined_m11, target_m11: (B,H,W,3) in [-1,1]
    M_healed01: (B,H,W) or (B,H,W,1) or (H,W) in [0,1]  (soft)
    return: scalar
    """
    if M_healed01 is None:
        return tf.constant(0.0, tf.float32)

    refined_m11 = tf.cast(refined_m11, tf.float32)
    target_m11  = tf.cast(target_m11,  tf.float32)

    # mask -> (B,H,W,1)
    m = tf.cast(M_healed01, tf.float32)
    if m.shape.rank == 2:
        m = tf.expand_dims(m, axis=0)       # (1,H,W)
    if m.shape.rank == 3:
        m = tf.expand_dims(m, axis=-1)      # (B,H,W,1)
    m = tf.clip_by_value(m, 0.0, 1.0)

    # [-1,1] -> [0,1]
    ref01 = (refined_m11 + 1.0) * 0.5
    tgt01 = (target_m11  + 1.0) * 0.5

    Lab_r = rgb01_to_lab_tf(ref01)   # (B,H,W,3)
    Lab_t = rgb01_to_lab_tf(tgt01)

    d  = Lab_r - Lab_t
    d2 = tf.reduce_sum(tf.square(d), axis=-1, keepdims=True)  # (B,H,W,1)  (≈ ΔE^2)

    # masked mean
    num = tf.reduce_sum(d2 * m)
    den = tf.reduce_sum(m) + eps
    return num / den

def healed_to_target_l1_loss(refined_m11, target_m11, M_healed01, eps=1e-6):
    if M_healed01 is None:
        return tf.constant(0.0, tf.float32)

    refined_m11 = tf.cast(refined_m11, tf.float32)
    target_m11  = tf.cast(target_m11,  tf.float32)

    m = tf.cast(M_healed01, tf.float32)
    if m.shape.rank == 2:
        m = tf.expand_dims(m, axis=0)
    if m.shape.rank == 3:
        m = tf.expand_dims(m, axis=-1)
    m = tf.clip_by_value(m, 0.0, 1.0)

    diff = tf.abs(refined_m11 - target_m11)  # (B,H,W,3)
    # Broadcast mask to channels
    num = tf.reduce_sum(diff * m)
    den = tf.reduce_sum(m) * 3.0 + eps
    return num / den

def masked_l1(a, b, m01, eps=1e-6):
    # a,b: (B,H,W,3) in [-1,1]
    # m01: (B,H,W) or (B,H,W,1) in [0,1]
    if m01 is None:
        return tf.constant(0.0, tf.float32)
    m = tf.cast(m01, tf.float32)
    if m.shape.rank == 2:
        m = tf.expand_dims(m, 0)
    if m.shape.rank == 3:
        m = tf.expand_dims(m, -1)
    m = tf.clip_by_value(m, 0.0, 1.0)
    num = tf.reduce_sum(tf.abs(a - b) * m)
    den = tf.reduce_sum(m) * 3.0 + eps
    return num / den

def debug_lab_roi_log(gt01, pre01, post01, M01=None, tag="DBG"):
    """
    gt01, pre01, post01 : (H,W,3) in [0,1] expected
    M01 : (H,W) or (H,W,1) in [0,1] or None
    """

    def rng(x):
        return float(x.min()), float(x.max())

    # (a) range check
    print(f"[{tag}] GT   range: {rng(gt01)}")
    print(f"[{tag}] PRE  range: {rng(pre01)}")
    print(f"[{tag}] POST range: {rng(post01)}")

    # (b) ROI status
    if M01 is None:
        print(f"[{tag}] ROI mask: None (FULL image used)")
        roi_frac = 1.0
        M = np.ones(gt01.shape[:2], np.float32)
    else:
        M = M01[...,0] if M01.ndim == 3 else M01
        roi_frac = float(M.mean())
        print(f"[{tag}] ROI mask: present | coverage={roi_frac:.3f}")

    # (c) Lab drift
    Lab_gt   = rgb01_to_lab(gt01)
    Lab_pre  = rgb01_to_lab(pre01)
    Lab_post = rgb01_to_lab(post01)

    for i, ch in enumerate(["L*", "a*", "b*"]):
        d_pre  = (Lab_pre[...,i]  - Lab_gt[...,i]) * M
        d_post = (Lab_post[...,i] - Lab_gt[...,i]) * M
        print(f"[{tag}] Δ{ch}: pre={d_pre.mean():+.3f}, post={d_post.mean():+.3f}")

def apply_clahe(image, alpha=0.5, clip_limit=1.0, tile_grid_size=(8, 8), take="last"):
    """
    Accepts:
      - (H,W,3)
      - (B,H,W,3)
      - (B,T,H,W,3)
    Returns:
      same rank as input, CLAHE applied per image.
    """
    x = image
    if isinstance(x, tf.Tensor):
        x = x.numpy()

    x = np.asarray(x)

    # ---- normalize shape to (N,H,W,3) ----
    orig_rank = x.ndim

    if orig_rank == 3:
        if x.shape[-1] != 3:
            raise ValueError(f"apply_clahe expects last dim=3, got {x.shape}")
        batch = x[None, ...]  # (1,H,W,3)

    elif orig_rank == 4:
        if x.shape[-1] != 3:
            raise ValueError(f"apply_clahe expects last dim=3, got {x.shape}")
        batch = x  # (B,H,W,3)

    elif orig_rank == 5:
        if x.shape[-1] != 3:
            raise ValueError(f"apply_clahe expects last dim=3, got {x.shape}")
        # (B,T,H,W,3) -> (B,H,W,3) by selecting frame
        if take == "last":
            batch = x[:, -1, ...]
        elif take == "first":
            batch = x[:, 0, ...]
        else:
            raise ValueError("take must be 'last' or 'first'")
    else:
        raise ValueError(f"apply_clahe expects rank 3/4/5, got rank={orig_rank}, shape={x.shape}")

    # ---- ensure [0,1] float -> uint8 ----
    batch = np.clip(batch, 0.0, 1.0)
    batch_u8 = (batch * 255.0 + 0.5).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))
    out_u8 = np.empty_like(batch_u8)

    for i in range(batch_u8.shape[0]):
        lab = cv2.cvtColor(batch_u8[i], cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l2 = clahe.apply(l)
        lab2 = cv2.merge([l2, a, b])
        rgb2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

        # alpha blend with original
        out_u8[i] = cv2.addWeighted(rgb2, float(alpha), batch_u8[i], 1.0 - float(alpha), 0)

    out = out_u8.astype(np.float32) / 255.0

    # ---- restore original rank ----
    if orig_rank == 3:
        out = out[0]
    elif orig_rank == 4:
        pass
    elif orig_rank == 5:
        # put back into (B,T,H,W,3) by only replacing chosen frame
        x_out = x.copy()
        if take == "last":
            x_out[:, -1, ...] = out
        else:
            x_out[:, 0, ...] = out
        out = x_out

    return tf.convert_to_tensor(out, dtype=tf.float32)

predicted_image = apply_clahe(predicted_image_rgb, alpha=0.5, clip_limit=1.0)
img3 = predicted_image
if img3.shape.rank == 4:
    img3 = img3[0]  # (H,W,3)

gray = tf.image.rgb_to_grayscale(img3)   # (H,W,1)
contrast = tf.math.reduce_std(gray).numpy()
print("Contrast:", contrast)

creator_image = tf.cast(predicted_image, tf.float32)

predicted_image_color = tf.cast(tf.clip_by_value(img3*255.0, 0.0, 255.0), tf.uint8).numpy()

def match_batch(predicted_image, delta_pred):
    if len(predicted_image.shape) == 3:
        predicted_image = tf.expand_dims(predicted_image, axis=0)

    batch_size = tf.shape(delta_pred)[0]

    predicted_image = tf.repeat(predicted_image, repeats=batch_size, axis=0)

    return predicted_image

skip_gan_training = False

if cos_val >= COS_OK:
    print("🎯 Creator output quality is excellent → Skipping GAN training")
    skip_gan_training = True
else:
    skip_gan_training = False

discriminator = discriminator_model(input_shape=(64, 64, 1))

def gaussian_kernel(ksize=9, sigma=1.5, channels=3):
    import math
    ax = tf.range(-ksize//2 + 1, ksize//2 + 1, dtype=tf.float32)
    xx, yy = tf.meshgrid(ax, ax)
    kernel = tf.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel = kernel / tf.reduce_sum(kernel)
    kernel = kernel[:, :, tf.newaxis, tf.newaxis]
    kernel = tf.tile(kernel, [1, 1, channels, 1])   # depthwise
    return kernel

def lowpass(x, ksize=9, sigma=1.5):
    ch = tf.shape(x)[-1]
    k = gaussian_kernel(ksize, sigma, channels=ch)
    return tf.nn.depthwise_conv2d(x, k, strides=[1,1,1,1], padding="SAME")

def highpass(x, ksize=9, sigma=1.5):
    return x - lowpass(x, ksize, sigma)

def grad_mag(x):
    x = tf.cast(x, tf.float32)

    # Convert from [-1,1] to [0,1] if needed
    x01 = (x + 1.0) * 0.5 if tf.reduce_min(x) < 0 else x
    x01 = tf.clip_by_value(x01, 0.0, 1.0)

    g = tf.image.rgb_to_grayscale(x01)            # (B,H,W,1)
    sob = tf.image.sobel_edges(g)                # (B,H,W,1,2)

    gx = sob[..., 0]                             # (B,H,W,1)  (x-gradient)
    gy = sob[..., 1]                             # (B,H,W,1)  (y-gradient)

    mag = tf.sqrt(gx*gx + gy*gy + 1e-6)          # (B,H,W,1)
    return mag

def generator_loss(
    base_m11,
    generator,
    discriminator,
    creator_cond_m11,
    vgg,
    target_m11,
    M_lesion01=None,
    M_healed01=None,
    M_active01=None,
    lambda_active=2.0,
    lambda_adv=0.003,
    lambda_1=1.5, lambda_2=1.5, lambda_3=0.5,
    lambda_creator=0.02,
    lambda_out=10.0,
    lambda_healed=5.0,
    lambda_hf=3.0,
    lambda_hf_roi=1.5,       # Previously added
    lambda_lpips_out=0.5,    # Added: background region perceptual loss
    healed_mode="lab",
):
    base_m11 = tf.cast(base_m11, tf.float32)
    base_sg  = tf.stop_gradient(base_m11)

    delta_pred  = tf.cast(generator(base_m11, training=True), tf.float32)
    delta_hp    = highpass(delta_pred, ksize=9, sigma=1.5)
    refined_m11 = tf.clip_by_value(base_m11 + delta_hp, -1.0, 1.0)

    fake_logits = discriminator([grad_mag(refined_m11), grad_mag(base_m11)], training=True)
    gan_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(fake_logits), logits=fake_logits
        )
    )

    # ── Inside lesion: preserve structure based on base ──────────────────
    # Prevent refined from deviating too far from Creator prediction
    lpips_value = tf.cast(
        perceptual_metric(base_sg, refined_m11) * lambda_1, tf.float32
    )
    edge  = tf.cast(edge_loss(base_m11, refined_m11) * lambda_2, tf.float32)
    sharp = tf.cast(sharpen_loss(base_m11, refined_m11) * lambda_3, tf.float32)
    creator_loss = tf.reduce_mean(tf.abs(refined_m11 - base_sg)) * lambda_creator

    # ── Background region: mimic HF patterns based on target ─────────────
    if M_lesion01 is not None:
        # HF mimicry outside the lesion (hf_roi_loss added earlier)
        hf_rec = lambda_hf * hf_roi_loss(refined_m11, target_m11, M_lesion01)

        # Background perceptual loss is also based on target
        # (Reduce perceptual difference between refined and target outside M_lesion)
        w_out = tf.clip_by_value(
            1.0 - tf.cast(M_lesion01, tf.float32), 0.0, 1.0
        )[..., tf.newaxis]
        refined_out = refined_m11 * w_out
        target_out  = target_m11  * w_out
        lpips_out   = tf.cast(
            perceptual_metric(target_out, refined_out) * lambda_lpips_out,
            tf.float32
        )
    else:
        hf_rec    = lambda_hf * hf_l1_loss(refined_m11, target_m11)  # fallback
        lpips_out = tf.constant(0.0, tf.float32)

    # ── Lesion region: prevent base deviation + restore healed ─────────────
    out_loss = tf.constant(0.0, tf.float32)
    if M_lesion01 is not None:
        out_loss = tf.cast(
            lambda_out, tf.float32
        ) * out_lab_drift_loss(base_sg, refined_m11, M_lesion01)

    healed_loss = tf.constant(0.0, tf.float32)
    if M_healed01 is not None:
        if healed_mode == "lab":
            healed_loss = tf.cast(lambda_healed, tf.float32) * healed_to_target_lab_loss(
                refined_m11, target_m11, M_healed01
            )
        elif healed_mode == "l1":
            healed_loss = tf.cast(lambda_healed, tf.float32) * healed_to_target_l1_loss(
                refined_m11, target_m11, M_healed01
            )

    active_loss = tf.constant(0.0, tf.float32)
    if M_active01 is not None:
        active_loss = tf.cast(lambda_active, tf.float32) * masked_l1(
            refined_m11, base_sg, M_active01
        )

    total = (lambda_adv * gan_loss
             + hf_rec + lpips_value + lpips_out   # lpips_out added
             + edge + sharp + creator_loss
             + out_loss + active_loss + healed_loss)

    tf.print("G:",
             "gan", gan_loss,
             "lp(base)", lpips_value,
             "lp(out)",  lpips_out,     # Added to logs
             "hf", hf_rec,
             "edge", edge,
             "sharp", sharp,
             "creator", creator_loss,
             "out", out_loss,
             "active", active_loss,
             "healed", healed_loss)

    return total, refined_m11, fake_logits, delta_pred

def discriminator_loss(real_logits, fake_logits):
    real_labels = tf.ones_like(real_logits) * 0.8
    fake_labels = tf.zeros_like(fake_logits) + 0.2

    real_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=real_labels, logits=real_logits
    )
    fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=fake_labels, logits=fake_logits
    )
    return tf.reduce_mean(real_loss + fake_loss)
