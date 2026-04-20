"""
BurnSight Refiner GAN — Generator, Discriminator, ConditionalGAN, GANImageCallback
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
from src.losses.refiner_losses import *

def generator_model(input_shape=(64, 64, 3)):
    creator_input = Input(shape=input_shape)

    e1 = conv_block(creator_input, 64, use_batchnorm=False)
    e2 = conv_block(e1, 128)
    e3 = conv_block(e2, 256)
    e4 = conv_block(e3, 512)
    e5 = conv_block(e4, 512)
    e6 = conv_block(e5, 512)

    d1 = deconv_block(e6, e5, 512, dropout=True)
    d2 = deconv_block(d1, e4, 512, dropout=True)
    d3 = deconv_block(d2, e3, 256)
    d4 = deconv_block(d3, e2, 128)
    d5 = deconv_block(d4, e1, 64)
    d6 = UpSampling2D(size=(2, 2), interpolation='bilinear')(d5)
    d6 = Conv2D(3, kernel_size=3, strides=1, padding='same')(d6)

    delta_pred = Activation('tanh')(d6)

    return Model(creator_input, delta_pred)

def discriminator_model(input_shape=(64, 64, 1)):
    gen_input = Input(shape=input_shape)
    creator_input = Input(shape=input_shape)

    combined = Concatenate(axis=-1)([gen_input, creator_input])

    c1 = Conv2D(64, (4, 4), strides=(2, 2), padding='same', activation=None)(combined)
    c1 = LeakyReLU(0.2)(c1)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.5)(c1)

    c2 = Conv2D(128, (4, 4), strides=(2, 2), padding='same', activation=None)(c1)
    c2 = LeakyReLU(0.2)(c2)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.5)(c2)

    c3 = Conv2D(256, (4, 4), strides=(2, 2), padding='same', activation=None)(c2)
    c3 = LeakyReLU(0.2)(c3)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.5)(c3)

    outputs = Conv2D(1, (4, 4), padding='same', activation=None)(c3)

    return Model(inputs=[gen_input, creator_input], outputs=outputs)

g_optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
d_optimizer = Adam(learning_rate=0.000001, beta_1=0.5)

custom_objects = {
    "LeakyReLU": LeakyReLU,
    "TileCutter1": TileCutter1,
    "TileCutter2": TileCutter2,
    "TimePreservingUpSampling2D": TimePreservingUpSampling2D,
    "psnr": psnr,
    "ssim": ssim,
    "perceptual_metric": perceptual_metric
}

tf.config.run_functions_eagerly(True)

def denormalize_img(img):
    """[-1,1] → [0,1]"""
    return tf.clip_by_value((img + 1.0) / 2.0, 0.0, 1.0)

base_m11 = (pred01_final*2.0 - 1.0)[None, ...].astype(np.float32)

def refiner_delta_report(
    predicted_image_m11,
    final_refined_m11,
    mask_model,
    lesion_from_prob_all_fn,
    THR=0.55,
):
    # --- (0) shape sanitize ---
    predicted_image_m11 = np.asarray(predicted_image_m11, dtype=np.float32)
    final_refined_m11   = np.asarray(final_refined_m11,   dtype=np.float32)

    if predicted_image_m11.ndim == 3:
        predicted_image_m11 = predicted_image_m11[None, ...]
    if final_refined_m11.ndim == 3:
        final_refined_m11 = final_refined_m11[None, ...]

    predicted_image_m11 = predicted_image_m11[..., :3]
    final_refined_m11   = final_refined_m11[..., :3]

    # 1) [0,1]
    pred01 = (predicted_image_m11 + 1.0) * 0.5
    ref01  = (final_refined_m11   + 1.0) * 0.5

    # 2) ROI mask
    p_raw = mask_model.predict(predicted_image_m11, verbose=0)  # Replace with pred01 if needed
    prob_all = get_prob_all_batch(p_raw, assume="auto")
    lesion_prob = lesion_from_prob_all_fn(prob_all)
    lesion_prob = np.asarray(lesion_prob, dtype=np.float32)

    # ---- force (B,H,W,1) ----
    if lesion_prob.ndim == 2:
        lesion_prob = lesion_prob[None, ..., None]
    elif lesion_prob.ndim == 3:
        if lesion_prob.shape[-1] == 1:
            lesion_prob = lesion_prob[None, ...]
        else:
            lesion_prob = lesion_prob[..., None]
    elif lesion_prob.ndim == 4:
        if lesion_prob.shape[-1] != 1:
            lesion_prob = lesion_prob[..., :1]

    lesion_prob = np.clip(lesion_prob, 0.0, 1.0)
    M = (lesion_prob >= float(THR)).astype(np.float32)

    assert M.ndim == 4 and M.shape[-1] == 1, f"M must be (B,H,W,1), got {M.shape}"

    # 3) Δ map
    d_map = np.abs(ref01 - pred01)  # (B,H,W,3)

    # 4) ROI / non-ROI
    M3 = np.repeat(M, 3, axis=-1)  # (B,H,W,3)
    roi_mean = (d_map * M3).sum() / (M3.sum() + 1e-6)
    out_mean = (d_map * (1.0 - M3)).sum() / ((1.0 - M3).sum() + 1e-6)
    roi_frac = float(M.mean())

    return {
        "roi_frac": roi_frac,
        "delta_roi_mean": float(roi_mean),
        "delta_out_mean": float(out_mean),
        "mask_thr": float(THR),
    }

def _srgb_to_linear(x):
    # x in [0,1]
    a = 0.055
    return tf.where(x <= 0.04045, x / 12.92, tf.pow((x + a) / (1.0 + a), 2.4))

def rgb01_to_lab_tf(rgb01):
    """
    rgb01: (B,H,W,3) in [0,1]
    return: Lab (B,H,W,3) float32
    """
    rgb01 = tf.clip_by_value(tf.cast(rgb01, tf.float32), 0.0,  1.0)
    rgb = _srgb_to_linear(rgb01)

    # sRGB -> XYZ (D65)
    M = tf.constant([[0.4124564, 0.3575761, 0.1804375],
                     [0.2126729, 0.7151522, 0.0721750],
                     [0.0193339, 0.1191920, 0.9503041]], dtype=tf.float32)
    xyz = tf.tensordot(rgb, M, axes=[-1, 1])  # (B,H,W,3)

    # Normalize by D65 white
    white = tf.constant([0.95047, 1.00000, 1.08883], dtype=tf.float32)
    xyz_n = xyz / white

    eps = 216.0 / 24389.0   # (6/29)^3
    k   = 24389.0 / 27.0

    def f(t):
        return tf.where(t > eps, tf.pow(t, 1.0/3.0), (k * t + 16.0) / 116.0)

    fxyz = f(xyz_n)
    fx, fy, fz = fxyz[...,0], fxyz[...,1], fxyz[...,2]

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    return tf.stack([L, a, b], axis=-1)

def weighted_mean(x, w, eps=1e-6):
    # x: (B,H,W) or (B,H,W,3), w: (B,H,W,1) or (B,H,W)
    w = tf.cast(w, tf.float32)
    if w.shape.rank == 4 and x.shape.rank == 4:
        # broadcast ok
        num = tf.reduce_sum(x * w)
        den = tf.reduce_sum(w) * tf.cast(tf.shape(x)[-1], tf.float32) + eps
        return num / den
    else:
        num = tf.reduce_sum(x * w)
        den = tf.reduce_sum(w) + eps
        return num / den


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

def to_bhwc3(x):
    """Standardize input to (B,H,W,3) float32.
       - (B,T,H,W,C) -> take last T only as (B,H,W,C)
       - (H,W,C)     -> (1,H,W,C)
       - (B,H,W)     -> (B,H,W,1) then repeat to 3 channels
       - If C==1, repeat to 3 channels; if C>=3, use first 3 channels
    """
    x = tf.convert_to_tensor(x, dtype=tf.float32)

    # (B,T,H,W,C) -> (B,H,W,C)  (assumes T=1; uses last frame if T>1)
    if x.shape.rank == 5:
        x = x[:, -1]  # (B,H,W,C)

    # (H,W,C) -> (1,H,W,C)
    if x.shape.rank == 3:
        x = tf.expand_dims(x, axis=0)

    # (B,H,W) -> (B,H,W,1)
    if x.shape.rank == 4 and x.shape[-1] is None:
        # Guard against symbolic tensors with unknown C (rare)
        pass
    if x.shape.rank == 4 and x.shape[-1] == 0:
        # Defensive handling for abnormal input
        x = tf.expand_dims(x, axis=-1)

    if x.shape.rank == 4:
        c = tf.shape(x)[-1]  # Dynamic channel count
        def take3():   return x[..., :3]
        def rep3():    return tf.repeat(x, 3, axis=-1)
        def keep3():   return x
        def fallback():
            first = x[..., :1]
            return tf.repeat(first, 3, axis=-1)

        # Branch dynamically on C
        x = tf.case([
            (tf.equal(c, 1), rep3),
            (tf.equal(c, 3), keep3),
        ], default=take3, exclusive=True)  # Use first 3 channels if C>=3
    else:
        raise ValueError(f"Unsupported rank {x.shape.rank} for to_bhwc3")

    return x

def to_disc_pair(gen_like, creator_like):
    """Return the two inputs [gen_input, creator_input] expected by the Discriminator.
       Standardize each tensor to (B,H,W,3).
    """
    x_gen = to_bhwc3(gen_like)      # (B,H,W,3)
    x_cre = to_bhwc3(creator_like)  # (B,H,W,3)
    return [x_gen, x_cre]

conditional_gan = ConditionalGAN(
    generator=generator,
    discriminator=discriminator,
    creator_output=base_m11,
    target_image=target_image,
    vgg=vgg,
    ema=0.2,
    seg_model=mask_model,
    lesion_from_prob_all_fn=lesion_from_prob_all_fn_tf,
    lesion_idxs=LESION_IDXS,
    seg_expects="m11",
)

if conditional_gan is None:
    raise ValueError("conditional_gan object is None. The model was not created successfully.")
else:
    print("✅ conditional_gan has been created successfully")
    print("conditional_gan type:", type(conditional_gan))
    print("conditional_gan.generator type:", type(conditional_gan.generator))
    print("conditional_gan.discriminator type:", type(conditional_gan.discriminator))
    print("conditional_gan.predicted_future shape:", conditional_gan.predicted_future.shape)

conditional_gan.enable_tf_function(not DEBUG_MODE)

print(f"📌 generator_loss: {generator_loss}")
print(f"📌 discriminator_loss: {discriminator_loss}")

if not callable(generator_loss):
    raise ValueError("generator_loss is not a function! Verification required!")

if not callable(discriminator_loss):
    raise ValueError("discriminator_loss is not a function! Verification required!")

conditional_gan.compile(g_optimizer=g_optimizer, d_optimizer=d_optimizer,g_loss_fn=generator_loss,d_loss_fn=discriminator_loss, run_eagerly=True)
print(f"📌 ConditionalGAN train_step exists: {'train_step' in dir(conditional_gan)}")
print(f"📌 ConditionalGAN base classes: {conditional_gan.__class__.__bases__}")
print(f"✅ conditional_gan.compile() completed")
print(f"   g_optimizer: {g_optimizer}")
print(f"   d_optimizer: {d_optimizer}")
print(f"   g_loss_fn: {generator_loss}")
print(f"   d_loss_fn: {discriminator_loss}")

check_for_printv2()

class GANImageCallback(tf.keras.callbacks.Callback):
    def __init__(self, conditional_gan, original_test_dir, target_image, real_dataset, display_freq=5, discriminator_threshold=0.05, earlystop_lpips=0.1, patience=50,
                 mask_model=None,
                 lesion_from_prob_all_fn=None,
                 THR_ANCHOR=0.55):
        super().__init__()
        self.conditional_gan = conditional_gan
        self.original_test_dir = original_test_dir
        self.display_freq = display_freq
        self.day_images = {}  # Store images per Day
        self.real_dataset = real_dataset
        self.target_image = target_image
        self.initial_creator_output = tf.identity(
            self.conditional_gan.initial_creator_output
        )
        self._group_test_images_by_day()  # Initialize image grouping
        self.discriminator_threshold = discriminator_threshold
        self.previous_predicted_future = None
        self.best_lpips = float('inf')
        self.earlystop_lpips = earlystop_lpips
        self.patience = patience
        self.mask_model = mask_model
        self.lesion_from_prob_all_fn = lesion_from_prob_all_fn
        self.THR_ANCHOR = THR_WOUND
        self.no_improve_count = 0
        self.lpips_old_list = []
        self.lpips_new_list = []
        self.edge_old_list = []
        self.edge_new_list = []
        self.creator_scores = []
        self.refined_scores = []
        self.dE_pre_list  = []
        self.dE_post_list = []
        self.dE_gain_list = []
        self.dE_epoch_list = []

    def _group_test_images_by_day(self):
        """
        Group test image files in Day order.
        """
        sorted_files = get_sorted_day_images(self.original_test_dir)
        for path in sorted_files:
            day_match = re.search(r'Day(\d+)', os.path.basename(path))
            if day_match:
                day = day_match.group(0)  # Day1, Day2, etc.
                if day not in self.day_images:
                    self.day_images[day] = []
                self.day_images[day].append(path)

    def _load_and_preprocess_image(self, path):
        """
        Load and preprocess images.
        """
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, (64, 64)) / 255.0
            return img
        return None

    def _get_one_batch(self):
        # The structure of (inputs, target, roi, etc.) varies across real_dataset,
        # so we use a common pattern to safely fetch only 1 batch
        return next(iter(self.real_dataset.take(1)))

    def _to_numpy01_hw3(self, x):
        if isinstance(x, (tf.Tensor, tf.Variable)):
            x = x.numpy()
        x = np.asarray(x, dtype=np.float32)

        # Remove batch axis (1,H,W,C) -> (H,W,C)
        if x.ndim == 4 and x.shape[0] == 1:
            x = x[0]

        # Scale [-1,1] -> [0,1]
        if np.min(x) < 0.0:
            x = (x + 1.0) * 0.5
        x = np.clip(x, 0.0, 1.0)

        # Force channels
        if x.ndim == 2:
            x = np.repeat(x[..., None], 3, axis=-1)
        elif x.ndim == 3 and x.shape[-1] == 1:
            x = np.repeat(x, 3, axis=-1)
        elif x.ndim == 3 and x.shape[-1] >= 3:
            x = x[..., :3]
        else:
            raise ValueError(f"_to_numpy01_hw3: bad shape {x.shape}")

        return x

    def _is_last_epoch(self, epoch: int) -> bool:
        total_epochs = int(self.params.get("epochs", 0))
        return (total_epochs > 0) and (epoch == total_epochs - 1)

    def _log_deltaE_deltaa_epoch(self, epoch: int, logs=None, freq: int = 1, print_every: int = 1,
                             use_roi_from_batch: bool = True):
        """
        Every epoch (or every freq epochs):
          pre  = initial_creator_output
          post = current predicted_future
          gt   = target_image
        to compute Δa*_roi / ΔE_roi and accumulate in a list

        freq=1: every epoch, freq=5: every 5 epochs.
        print_every=1: print every time, 5: print every 5 epochs.
        """

        # Cycle condition
        if freq is not None and freq > 1:
            if (epoch % freq) != 0:
                return

        # 0) Tensor -> numpy [0,1], force (H,W,3)
        gt01   = self._to_numpy01_hw3(self.target_image)
        pre01  = self._to_numpy01_hw3(self.initial_creator_output)
        post01 = self._to_numpy01_hw3(self.conditional_gan.predicted_future)

        # Trim batch axis: if (1,H,W,3), take [0]
        if gt01.ndim == 4:   gt01 = gt01[0]
        if pre01.ndim == 4:  pre01 = pre01[0]
        if post01.ndim == 4: post01 = post01[0]

        # 1) Get ROI mask if available, else None
        m0 = None
        if use_roi_from_batch:
            try:
                batch = self._get_one_batch()
                if isinstance(batch, (tuple, list)) and len(batch) >= 3:
                    m = batch[2]
                    m = m.numpy() if isinstance(m, tf.Tensor) else m

                    # (1,H,W,1)/(1,H,W) -> (H,W) or (H,W,1)
                    if m.ndim == 4:
                        m0 = m[0]
                    else:
                        m0 = m
            except Exception:
                m0 = None

        debug_lab_roi_log(
            gt01, pre01, post01,
            M01=m0,          # ROI mask (or None)
            tag=f"E{epoch+1}"
        )

        # 2) Compute statistics
        st_pre  = lab_delta_stats(gt01, pre01,  m0)   # GT vs Creator
        st_post = lab_delta_stats(gt01, post01, m0)   # GT vs Refiner

        da_pre,  dE_pre  = float(st_pre["da_roi"]),  float(st_pre["dE_roi"])
        da_post, dE_post = float(st_post["da_roi"]), float(st_post["dE_roi"])
        dE_gain = dE_pre - dE_post

        # 3) Accumulate and save (record every epoch)
        self.dE_pre_list.append(dE_pre)
        self.dE_post_list.append(dE_post)
        self.dE_gain_list.append(dE_gain)

        # 4) Print (control frequency if desired)
        if print_every is None or print_every <= 1 or (epoch % print_every) == 0:
            print(f"\n🧾 [EPOCH {epoch+1}] Δa*_roi / ΔE_roi (vs GT)")
            print(f"  pre (Creator):  Δa*_roi={da_pre:+.4f},  ΔE_roi={dE_pre:.4f}")
            print(f"  post (Refiner): Δa*_roi={da_post:+.4f}, ΔE_roi={dE_post:.4f}")
            print(f"  gain (ΔE_pre-ΔE_post): {dE_gain:+.4f}\n")

    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n🔄 [Callback] === Start of Epoch {epoch+1} ===", flush=True)
        if epoch == 0:
            self.previous_predicted_future = tf.identity(self.model.predicted_future)
            print("📌 Initial previous_predicted_future has been set.")

    def on_epoch_end(self, epoch, logs=None):
        current_pred = self.conditional_gan.predicted_future

        if epoch == 0:
            # At Epoch 0, record only 'Creator vs Target' baseline
            print("ℹ️ [Epoch 1] Computing baseline LPIPS (Creator vs Target)")

            lpips_initial = perceptual_metric(
                self.target_image,
                self.initial_creator_output
            )
            self.best_lpips = float(tf.reduce_mean(lpips_initial).numpy())

            # Reference for comparison in subsequent epochs: first GAN output
            self.previous_predicted_future = tf.identity(current_pred)
            return

        # From here, runs only when epoch >= 1 (original code kept with minor tweaks)
        new_predicted = current_pred.numpy()

        print(f"✅ [Epoch {epoch+1}] Using blended predicted_future")
        print(f"new_predicted min/max: {np.min(new_predicted)}, {np.max(new_predicted)}")

        lpips_old = perceptual_metric(self.conditional_gan.target_image,
                                      self.previous_predicted_future)
        lpips_new = perceptual_metric(self.conditional_gan.target_image,
                                      new_predicted)
        lpips_improved = (tf.reduce_mean(lpips_old) - tf.reduce_mean(lpips_new)) > 0.02

        edge_old = edge_loss(self.conditional_gan.target_image,
                            self.previous_predicted_future)
        edge_new = edge_loss(self.conditional_gan.target_image,
                            new_predicted)
        edge_improved = (tf.reduce_mean(edge_old) - tf.reduce_mean(edge_new)) > 0.02

        # Creator baseline is always the same regardless of epoch
        x_gen  = grad_mag(self.initial_creator_output)        # (1,H,W,1)
        x_cond = grad_mag(self.previous_predicted_future)     # (1,H,W,1)
        creator_score = tf.reduce_mean(self.conditional_gan.discriminator([x_gen, x_cond], training=False))

        y_gen  = grad_mag(current_pred)                       # (1,H,W,1)
        y_cond = grad_mag(self.previous_predicted_future)     # (1,H,W,1)
        refined_score = tf.reduce_mean(self.conditional_gan.discriminator([y_gen, y_cond], training=False))

        self.lpips_old_list.append(tf.reduce_mean(lpips_old).numpy())
        self.lpips_new_list.append(tf.reduce_mean(lpips_new).numpy())
        self.edge_old_list.append(tf.reduce_mean(edge_old).numpy())
        self.edge_new_list.append(tf.reduce_mean(edge_new).numpy())
        self.creator_scores.append(creator_score.numpy())
        self.refined_scores.append(refined_score.numpy())
        self.dE_epoch_list.append(epoch)

        if epoch < 30:
            disc_improved = refined_score > creator_score + self.discriminator_threshold
        else:
            disc_improved = refined_score > creator_score + 0.02

        lpips_old_m = float(tf.reduce_mean(lpips_old))
        lpips_new_m = float(tf.reduce_mean(lpips_new))
        edge_old_m  = float(tf.reduce_mean(edge_old))
        edge_new_m  = float(tf.reduce_mean(edge_new))
        print(f"Epoch {epoch+1}: LPIPS (old: {lpips_old_m:.4f}, new: {lpips_new_m:.4f})")
        print(f"Epoch {epoch+1}: EDGE  (old: {edge_old_m:.4f}, new: {edge_new_m:.4f})")
        print(f"Epoch {epoch+1}: D(creator): {creator_score:.4f}, D(refined): {refined_score:.4f}")

        if lpips_improved or edge_improved:
            update_condition = True
        else:
            update_condition = False

        if update_condition:
            print("✅ predicted_future updated: conditions met")
            self.previous_predicted_future = tf.identity(self.conditional_gan.predicted_future)
        else:
            print("❌ Update deferred: conditions not met")

            # Reflect early stopping improvement
            lpips_mean = tf.reduce_mean(lpips_new).numpy()
            if lpips_mean < self.best_lpips:
                self.best_lpips = lpips_mean
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1

            if self.no_improve_count >= self.patience and self.best_lpips < self.earlystop_lpips:
                print(f"🛑 Early stopping condition satisfied - best_lpips: {self.best_lpips:.4f}")
                self.model.stop_training = True

            # Reset condition to escape training stagnation (bad best_lpips with no improvement)
            elif self.no_improve_count >= self.patience and self.best_lpips > self.earlystop_lpips:
                print(f"⚠️ Stagnation detected - best_lpips: {self.best_lpips:.4f}")
                print("🔁 Re-initializing predicted_future via random blending with Creator output.")

                creator_output = tf.convert_to_tensor(self.initial_creator_output, dtype=tf.float32)

                # Add noise
                noise = tf.random.normal(shape=creator_output.shape, mean=0.0, stddev=0.05)

                # 🔁 Blending: Creator + Noise
                gen_input = creator_output + noise
                gen_input = tf.clip_by_value(gen_input, -1.0, 1.0)

                # Reinitialize Generator input
                self.conditional_gan.predicted_future.assign(gen_input)
                self.previous_predicted_future = tf.identity(gen_input)

                self.no_improve_count = 0

        if epoch % self.display_freq == 0:
            self.display_day_images_with_prediction(epoch)

        src = tf.convert_to_tensor(self.previous_predicted_future, dtype=tf.float32)
        src = tf.reshape(src, self.model.predicted_future.shape)  # fixed to (1,H,W,3)
        self.model.predicted_future.assign(src)

        # ===================== [INSERT: Refiner k-sweep/delta report] =====================
        if self._is_last_epoch(epoch) and (self.mask_model is not None) and (self.lesion_from_prob_all_fn is not None):
            creator_pred  = tf.convert_to_tensor(self.initial_creator_output, dtype=tf.float32)   # (1,H,W,3) [-1,1]
            final_refined = tf.convert_to_tensor(self.model.predicted_future, dtype=tf.float32)   # (1,H,W,3) [-1,1]

            rep_ref = refiner_delta_report(
                predicted_image_m11=creator_pred.numpy(),
                final_refined_m11=final_refined.numpy(),
                mask_model=self.mask_model,
                lesion_from_prob_all_fn=self.lesion_from_prob_all_fn,
                THR=self.THR_ANCHOR,
            )

            print(f"[RefinerReport][LAST Epoch {epoch+1}] "
                  f"roi_frac={rep_ref['roi_frac']:.3f} "
                  f"Δroi={rep_ref['delta_roi_mean']:.4f} "
                  f"Δout={rep_ref['delta_out_mean']:.4f}")

        # Log ΔE/Δa every epoch
        self._log_deltaE_deltaa_epoch(
            epoch=epoch,
            logs=logs,
            freq=50,
            print_every=50,
            use_roi_from_batch=True
        )
        # =================== [END INSERT: Refiner k-sweep/delta report] ===================

        print(f"✅ [Callback] === End of Epoch {epoch+1} ===", flush=True)

    def on_train_end(self, logs=None):
        total = int(self.params.get("epochs", 0) or 0)
        if total <= 0:
            print("⚠️ No epoch info available; skipping ΔE/Δa logging.")
            return

        last = total - 1
        # Prevent duplication if already logged with freq=50 in on_epoch_end
        if (last % 50) != 0:
            self._log_deltaE_deltaa_epoch(
                epoch=last,
                logs=logs,
                freq=1,
                print_every=1,
                use_roi_from_batch=True
            )

    def display_day_images_with_prediction(self, epoch):
        """
        Arrange images per Day in a single row in Day order, and append Predicted Future as the last column.
        """
        # Sort by Day order (Day1, Day2, ..., DayN)
        sorted_days = sorted(self.day_images.keys(), key=lambda x: int(x.replace("Day", "")))

        # Total image count (Day images + Predicted Future)
        total_images = sum(len(self.day_images[day]) for day in sorted_days) + 1

        # Set overall plot size
        plt.figure(figsize=(4 * total_images, 4))  # Adjust size based on column count

        # Visualize images per Day in a single row
        current_image_index = 1
        for day in sorted_days:
            for path in self.day_images[day]:
                img = self._load_and_preprocess_image(path)
                if img is None:
                    continue
                img_rgb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)

                plt.subplot(1, total_images, current_image_index)
                plt.imshow(img_rgb)
                plt.title(f"{day}")
                plt.axis('off')
                current_image_index += 1

        refined_image = tf.convert_to_tensor(
            self.conditional_gan.predicted_future, dtype=tf.float32
        )
        refined_image = tf.clip_by_value(refined_image, -1.0, 1.0)

        print("✅ Using conditional_gan.predicted_future for visualization.")

        # 2) Convert to numpy + scale to [-1,1] or [0,1] range
        predicted_future_tensor = refined_image
        if isinstance(predicted_future_tensor, (tf.Variable, tf.Tensor)):
            predicted_image_rgb = predicted_future_tensor.numpy()
        else:
            predicted_image_rgb = np.array(predicted_future_tensor)

        print("Min, Max of predicted_image_rgb BEFORE processing:",
              np.min(predicted_image_rgb), np.max(predicted_image_rgb))

        if np.min(predicted_image_rgb) >= 0.0 and np.max(predicted_image_rgb) <= 1.0:
            print("✅ Detected range [0,1]. Scaling to [0,255].")
            predicted_image_rgb = (predicted_image_rgb * 255).astype(np.uint8)

        elif np.min(predicted_image_rgb) >= -1.0 and np.max(predicted_image_rgb) <= 1.0:
            print("⚠️ Detected range [-1,1]. Scaling to [0,255].")
            predicted_image_rgb = ((predicted_image_rgb + 1.0) * 127.5).astype(np.uint8)

        else:
            print("⚠️ Detected extended range. Clipping to [0,1] and scaling.")
            predicted_image_rgb = np.clip(predicted_image_rgb, 0, 1)
            predicted_image_rgb = (predicted_image_rgb * 255).astype(np.uint8)

        print("Min, Max of predicted_image_rgb AFTER processing:",
              np.min(predicted_image_rgb), np.max(predicted_image_rgb))

        disp = ensure_hw3(refined_image)      # (1,H,W,3) float32
        disp = tf.squeeze(disp, axis=0)       # (H,W,3)
        disp01 = tf.clip_by_value((disp + 1.0) / 2.0, 0.0, 1.0)  # [0,1]
        img_np = disp01.numpy()
        plt.subplot(1, total_images, current_image_index)
        plt.imshow(img_np)
        plt.title(f"Predicted Image (Epoch {epoch})")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

check_for_printv2()

image_callback = GANImageCallback(
    conditional_gan=conditional_gan,
    original_test_dir=original_test_dir,
    real_dataset=test_dataset,
    target_image = target_image,
    display_freq=5,
    discriminator_threshold=0.05,
    earlystop_lpips=0.1,
    patience=50,
    mask_model=mask,   # seg model
    lesion_from_prob_all_fn=get_lesion_prob_from_prob_all,
    THR_ANCHOR=THR_WOUND
)

hist = conditional_gan.fit(test_dataset, epochs=701, batch_size=1, callbacks=[reduce_lr, image_callback], verbose=2)

series = [
    getattr(conditional_gan, 'g_losses', []),
    getattr(conditional_gan, 'd_losses', []),
    getattr(image_callback, 'lpips_old_list', []),
    getattr(image_callback, 'lpips_new_list', []),
    getattr(image_callback, 'edge_old_list', []),
    getattr(image_callback, 'edge_new_list', []),
    getattr(image_callback, 'creator_scores', []),
    getattr(image_callback, 'refined_scores', []),
]

valid_lengths = [len(s) for s in series if len(s) > 0]
if not valid_lengths:
    raise RuntimeError("Log to plot is empty. Check the callback or logging setup.")

n = min(valid_lengths)
epochs = range(n)

plt.figure(figsize=(18, 12))

# 1. Generator / Discriminator Loss
plt.subplot(3, 2, 1)
plt.plot(epochs, conditional_gan.g_losses[:n], label='Generator Loss (G)', color='blue')
plt.plot(epochs, conditional_gan.d_losses[:n], label='Discriminator Loss (D)', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('G vs D Loss')
plt.legend()
plt.grid(True)

# 2. LPIPS
plt.subplot(3, 2, 2)
plt.plot(epochs, image_callback.lpips_old_list[:n], label='LPIPS Old', linestyle='--')
plt.plot(epochs, image_callback.lpips_new_list[:n], label='LPIPS New')
plt.xlabel('Epoch')
plt.ylabel('LPIPS')
plt.title('LPIPS Comparison')
plt.legend()
plt.grid(True)

# 3. EDGE
plt.subplot(3, 2, 3)
plt.plot(epochs, image_callback.edge_old_list[:n], label='Edge Old', linestyle='--')
plt.plot(epochs, image_callback.edge_new_list[:n], label='Edge New')
plt.xlabel('Epoch')
plt.ylabel('Edge')
plt.title('Edge Comparison')
plt.legend()
plt.grid(True)

# 4. Discriminator Score
plt.subplot(3, 2, 4)
plt.plot(epochs, image_callback.creator_scores[:n], label='Creator D Score', linestyle='--')
plt.plot(epochs, image_callback.refined_scores[:n], label='Refined D Score')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Discriminator Scores')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

epochs = range(len(image_callback.dE_pre_list))

plt.figure(figsize=(8,3))
plt.plot(epochs, image_callback.dE_pre_list,  label="ΔE pre (Creator)")
plt.plot(epochs, image_callback.dE_post_list, label="ΔE post (Refiner)")
plt.plot(epochs, image_callback.dE_gain_list, label="ΔE gain")
plt.axhline(0, ls='--', color='gray')
plt.xlabel("Epoch")
plt.ylabel("ΔE (ROI)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

tf.config.run_functions_eagerly(False)
