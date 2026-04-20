"""
BurnSight Creator model definition + CreatorTrainer
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
from src.models.encoder import *
from src.losses.creator_losses import *

def creator_model(input_shape=(None, 64, 64, 5), gate_strength=1.0):
    inputs = Input(shape=input_shape, dtype='float32')
    rgb = RGBSlice()(inputs)
    kch = KSlice()(inputs)
    msk = MSlice()(inputs)

    x = Concatenate(axis=-1)([rgb, kch])

    # ---- Soft attention gating from mask ----
    # (Use TimeDistributed and padding='same' to preserve temporal axis)
    g = TimeDistributed(Conv2D(8, (3,3), padding='same', dtype='float32'))(msk)
    g = LeakyReLU(0.2)(g)

    # Match gate channel count to x's channel count (=4)
    g = TimeDistributed(Conv2D(4, (1,1), padding='same', activation='sigmoid', dtype='float32'))(g)
    g = SoftClamp(lo=0.1, hi=0.9, temp=1.0, name="gate_softclamp")(g)
    if gate_strength != 1.0:
        g = GateScale(gate_strength)(g)
    x = layers.Multiply()([x, layers.Add()([OnesLike()(g), g])])

    c1 = ConvLSTM2D(32, (3, 3), padding='same', activation=None, return_sequences=True, dtype='float32')(x)
    p1 = LayerNormalization(dtype='float32')(c1)
    p1 = LeakyReLU(0.2)(p1)
    p1 = TimeDistributed(MaxPooling2D((2, 2), dtype='float32'))(p1)

    c2 = ConvLSTM2D(64, (3, 3), padding='same', activation=None, return_sequences=True, dtype='float32')(p1)
    p2 = LayerNormalization(dtype='float32')(c2)
    p2 = LeakyReLU(0.2)(p2)
    p2 = TimeDistributed(MaxPooling2D((2, 2), dtype='float32'))(p2)

    c3 = ConvLSTM2D(128, (3, 3), padding='same', activation=None, return_sequences=True, dtype='float32')(p2)
    p3 = LayerNormalization(dtype='float32')(c3)
    p3 = LeakyReLU(0.2)(p3)
    p3 = TimeDistributed(MaxPooling2D((2, 2), dtype='float32'))(p3)

    c4 = ConvLSTM2D(256, (3, 3), padding='same', activation=LeakyReLU(0.2), return_sequences=True, dtype='float32')(p3)

    u5 = TimePreservingUpSampling2D(size=(2, 2), method='nearest')(c4)
    u5 = TimeDistributed(Conv2D(128, (3, 3), kernel_initializer=HeNormal(), padding='same', dtype='float32'))(u5)
    u5 = LeakyReLU(0.2)(u5)
    # Use modified TileCutter
    u5_cropped = TileCutter1()([u5, c3])
    c3_cropped = TileCutter2()([u5, c3])
    u5 = concatenate([u5_cropped, c3_cropped], axis=-1)
    c5 = ConvLSTM2D(128, (3, 3), padding='same', activation=LeakyReLU(0.2), return_sequences=True, dtype='float32')(u5)

    u6 = TimePreservingUpSampling2D(size=(2, 2), method='nearest')(c5)
    u6 = TimeDistributed(Conv2D(64, (3, 3), kernel_initializer=HeNormal(), padding='same', dtype='float32'))(u6)
    u6 = LeakyReLU(0.2)(u6)
    c2_processed = TimeDistributed(Conv2D(64, (1, 1), padding='same', dtype='float32'))(c2)
    u6 = concatenate([u6, c2_processed], axis=-1)
    c6 = ConvLSTM2D(64, (3, 3), padding='same', activation=LeakyReLU(0.2), return_sequences=True, dtype='float32')(u6)

    u7 = TimePreservingUpSampling2D(size=(2, 2), method='nearest')(c6)
    u7 = TimeDistributed(Conv2D(32, (3, 3), kernel_initializer=HeNormal(), padding='same', dtype='float32'))(u7)
    u7 = LeakyReLU(0.2)(u7)
    c1_processed = TimeDistributed(Conv2D(32, (1, 1), padding='same', dtype='float32'))(c1)
    u7 = concatenate([u7, c1_processed], axis=-1)
    c7 = ConvLSTM2D(32, (3, 3), padding='same', activation=LeakyReLU(0.2), return_sequences=True, dtype='float32')(u7)
    c7 = TemporalAttentionPooling(temp=0.5)(c7, m=msk)

    delta = Conv2D(3, 3, padding='same', activation='tanh', name='delta_head')(c7)

    m_last = SelectLastTime(name="m_last")(msk)
    m_last = SoftClamp(lo=0.0, hi=0.9, temp=1.0, name="m_last_clamp")(m_last)
    M3 = RepeatChannels3(name="M3")(m_last)

    x_last = XLastPicker(name="x_last_pick")(rgb)
    y_pred = layers.Add(name='compose')([
        x_last,
        layers.Multiply()([M3, delta]),
    ])

    y_pred = SoftClip(limit=1.1, name="softclip")(ScaleBias()(y_pred))
    y_pred = FloatCastLayer()(y_pred)

    model = Model(inputs, [y_pred, x_last], name="creator_with_xlast")
    return model

def pad_repeat_last_tf(X, max_T):
    """
    X: (T,H,W,C) or (B,T,H,W,C)
    Returns: (Xp, tmask)
      - (T,H,W,C) -> (max_T,H,W,C), (max_T,)
      - (B,T,H,W,C) -> (B,max_T,H,W,C), (B,max_T,)
    """
    def _pad_one(seq):
        # seq: (T,H,W,C)
        T = tf.shape(seq)[0]
        pad = tf.maximum(0, max_T - T)
        last = seq[-1:]                         # (1,H,W,C)
        tail = tf.repeat(last, pad, axis=0)     # (pad,H,W,C)
        Xp = tf.concat([seq, tail], axis=0)[:max_T]  # (max_T,H,W,C)
        tmask = tf.concat(
            [tf.ones((T,), tf.bool), tf.zeros((pad,), tf.bool)], axis=0
        )[:max_T]                                # (max_T,)
        return Xp, tmask

    # 1) Branch on static rank if available (works in both eager and graph mode)
    static_rank = X.shape.rank
    if static_rank == 4:
        return _pad_one(X)
    if static_rank == 5:
        Xp, tm = tf.map_fn(
            lambda seq: _pad_one(seq),
            X,
            fn_output_signature=(tf.float32, tf.bool)
        )
        return Xp, tm

    # 2) If static rank is None, use tf.case dynamically
    r = tf.rank(X)
    return tf.case(
        [
            (tf.equal(r, 4), lambda: _pad_one(X)),
            (tf.equal(r, 5), lambda: tf.map_fn(
                lambda seq: _pad_one(seq),
                X,
                fn_output_signature=(tf.float32, tf.bool)
            )),
        ],
        # default: guide with a clean error
        default=lambda: (tf.debugging.assert_fail(
            "pad_repeat_last_tf: X must have rank 4 or 5"
        ),)  # dummy tuple; assert_fail raises an exception so this is unreachable
    )

def reduce_with_tmask(x, tmask, axes):
    # x: (B,T,...) / tmask: (B,T)
    w = tf.cast(tmask, x.dtype)
    while len(tf.shape(w)) < len(tf.shape(x)):
        w = w[..., None]
    num = tf.reduce_sum(x*w, axis=axes)
    den = tf.reduce_sum(w, axis=axes) + 1e-6
    return num/den

# === Cell 5: Stage1 train step ===
@tf.function

class CreatorTrainer(tf.keras.Model):
    def __init__(self, creator, E, Proj, P, seg_model,
                 lambda_feat=1.0, lambda_anti=0.2,
                 tau_anti=0.03, patch_k=16, topk_ratio=0.1,
                 delta_scale=0.2, feather_k=5, name="CreatorTrainer",
                 lambda_mag=0.2, lambda_map=0.2, a0=0.05, a1=1.6, a2=0.12,
                 lambda_recon=2.0, lambda_stable=0.10, lambda_trans=0.05,
                 K_dead=0.10, eps_trans=0.01):                     # Added reconstruction loss weight
        super().__init__(name=name)
        self.creator = creator
        self.E, self.Proj, self.P = E, Proj, P
        self.lambda_feat = lambda_feat
        self.lambda_anti = lambda_anti
        self.tau_anti = tau_anti
        self.patch_k = patch_k
        self.topk_ratio = topk_ratio
        self.l2 = L2Normalize(axis=-1, name="l2norm")
        self.delta_scale = float(delta_scale)
        self.feather_k = int(feather_k)
        self.lambda_mag = float(lambda_mag)
        self.lambda_map = float(lambda_map)
        self.lambda_recon = float(lambda_recon)
        self.lambda_stable = float(lambda_stable)
        self.lambda_trans = float(lambda_trans)
        self.seg_model = seg_model
        self.seg_model.trainable = False
        self.K_dead = float(K_dead)
        self.eps_trans = float(eps_trans)
        self.u0 = self.add_weight(
            name="u0_a0", shape=(), trainable=True,
            initializer=tf.keras.initializers.Constant(np.log(np.exp(a0) - 1.0))
        )
        self.u1 = self.add_weight(
            name="u1_a1m1", shape=(), trainable=True,  # softplus of (a1-1)
            initializer=tf.keras.initializers.Constant(np.log(np.exp(a1 - 1.0) - 1.0))
        )
        self.u2 = self.add_weight(
            name="u2_a2", shape=(), trainable=True,
            initializer=tf.keras.initializers.Constant(np.log(np.exp(a2) - 1.0))
        )

    def leak_penalty(self, d_pred, M):
        out = 1.0 - M
        num = tf.reduce_sum(d_pred * tf.repeat(out, 3, -1), axis=[1,2,3])
        den = tf.reduce_sum(tf.repeat(out, 3, -1), axis=[1,2,3]) + 1e-6
        return tf.reduce_mean(num / den)

    def std_roi(self, x, M):
        w3 = tf.repeat(M, 3, -1)
        mu = tf.reduce_sum(x*w3, [1,2,3]) / (tf.reduce_sum(w3, [1,2,3]) + 1e-6)
        mu = tf.reshape(mu, [-1,1,1,1])
        var = tf.reduce_sum(((x-mu)**2)*w3, [1,2,3]) / (tf.reduce_sum(w3, [1,2,3]) + 1e-6)
        return tf.sqrt(tf.maximum(var, 1e-6))  # (B,)

    def contrast_loss(self, y01, tgt01, M):
        sp = self.std_roi(y01, M)
        st = self.std_roi(tgt01, M)
        return tf.reduce_mean(tf.abs(sp - st))

    @staticmethod
    def _roi_mass(prob_all, idx, M_support):
        p = prob_all[..., idx:idx+1]
        num = tf.reduce_sum(p * M_support, axis=[1,2,3])
        den = tf.reduce_sum(M_support, axis=[1,2,3]) + 1e-6
        return num / den

    @property
    def a0(self):
        return tf.nn.softplus(self.u0)

    @property
    def a1(self):
        return 1.0 + tf.nn.softplus(self.u1)  # Guarantee a1 ≥ 1 (acts as amplifier)

    @property
    def a2(self):
        return tf.nn.softplus(self.u2)

    def call(self, inputs, training=False):
        """
        inputs: Xk (B,T,H,W,5) or a tuple (Xk, y_stage2); only Xk is used
        """
        if isinstance(inputs, (tuple, list)):
            Xk = inputs[0]
        else:
            Xk = inputs
        return self.creator(Xk, training=training)

    def compile(self, optimizer, **kwargs):
        super().compile(optimizer=optimizer, **kwargs)

    @tf.function
    def anti_copy_loss(self, y_pred, last, mask, K_norm,
                      tau_min=0.01, tau_max=0.15,
                      gamma=0.4, shave_px=1):
        """
        K_norm: (B,) in [0,1]  (smaller = nearer future)
        Purpose: smaller K = stricter (stronger copy prohibition), larger K = more lenient
        Current penalty structure: relu(tau - d) => larger tau = stricter
        Therefore tau is designed to decrease as K increases
        """

        # tau(K): K=0 -> tau_max (strict), K=1 -> tau_min (lenient)
        tau = tau_max - K_norm * (tau_max - tau_min)   # (B,)
        tau = tf.reshape(tau, [-1, 1, 1, 1])           # for broadcasting

        core = shave_roi(mask, shave_px)
        w = soft_weight(core, gamma)                   # (B,H,W,1)
        w3 = tf.repeat(w, 3, axis=-1)

        diff = tf.abs(y_pred - last) * w3
        d = tf.reduce_sum(diff, axis=[1,2,3]) / (tf.reduce_sum(w3, axis=[1,2,3]) + 1e-6)  # (B,)

        # Compare as (B,) instead of broadcasted tau
        tau_b = tf.reshape(tau, [-1])
        penalty = tf.nn.relu(tau_b - d)
        return tf.reduce_mean(penalty)

    @tf.function
    def train_step(self, data):
        Xk, y_stage2 = data

        self.E.trainable = False
        self.Proj.trainable = False
        self.P.trainable = False

        with tf.GradientTape() as tape:
            y_pred_raw = _first_output(self.creator(Xk, training=True))

            # Compute L_stable, L_trans based on segmentation
            pred_m11 = (y_pred_raw + 1.0) * 0.5 * 2.0 - 1.0
            last_m11 = (y_stage2[..., 1:4] + 1.0) * 0.5 * 2.0 - 1.0
            prob_pred = self.seg_model(pred_m11, training=False)
            prob_last = self.seg_model(last_m11, training=False)

            M_support = tf.clip_by_value(Xk[:, -1, ..., 4:5], 0.0, 1.0)
            M_wound   = tf.clip_by_value(y_stage2[..., :1], 0.0, 1.0)
            M_change  = tf.clip_by_value(M_wound * M_support, 0.0, 1.0)
            M_stable  = tf.clip_by_value(tf.nn.relu(M_support - M_change), 0.0, 1.0)

            # Stable loss
            w_st = soft_weight(M_stable, gamma=0.8)
            w3s  = tf.repeat(w_st, 3, axis=-1)
            L_stable = tf.reduce_mean(
                tf.reduce_sum(tf.abs(y_pred_raw - y_stage2[..., 1:4]) * w3s, axis=[1,2,3])
                / (tf.reduce_sum(w3s, axis=[1,2,3]) + 1e-6)
            )

            # Transition loss
            L_trans = self._compute_trans_loss(
                prob_pred, prob_last, M_support, Xk
            )

            # Compute main loss including kmono (only once)
            total, y_comp, logs = train_loss_with_kmono(
                self, Xk, y_stage2, y_pred_raw,
                mono_sign=+1.0, dk=0.20,
                kmono_margin=0.01, lambda_kmono=0.05,
                deltaE_clamp=0.25
            )

            # Add L_stable and L_trans here
            total = (total
                    + self.lambda_stable * L_stable
                    + self.lambda_trans  * L_trans)

        grads = tape.gradient(total, self.creator.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.creator.trainable_variables))

        logs["L_stable"] = L_stable
        logs["L_trans"]  = L_trans
        logs["loss"]     = total
        return logs

    @tf.function
    def test_step(self, data):
        # data: (Xk, y_stage2)
        # Xk:      (B,T,H,W,5) = RGB(3)+K(1)+mask_seq(1)
        # y_stage2:(B,H,W,7)  = concat([M_next, last, y_next], -1)
        Xk, y_stage2 = data
        M_next = y_stage2[..., :1]           # (B,H,W,1) in [0,1]
        last   = y_stage2[..., 1:4]          # (B,H,W,3) in [-1,1]
        y_next = y_stage2[..., 4:7]          # (B,H,W,3) in [-1,1]

        B = tf.shape(Xk)[0]
        T = tf.shape(Xk)[1]
        H = tf.shape(Xk)[2]
        W = tf.shape(Xk)[3]

        # (0) Creator forward
        y_pred_raw = _first_output(self.creator(Xk, training=False))

        # ======= [BLOCK #1] Prepare stable/trans computation for validation =======
        M_wound   = tf.clip_by_value(M_next, 0.0, 1.0)
        M_support = tf.clip_by_value(Xk[:, -1, ..., 4:5], 0.0, 1.0)
        M_change  = tf.clip_by_value(M_wound * M_support, 0.0, 1.0)
        M_stable  = tf.clip_by_value(tf.nn.relu(M_support - M_change), 0.0, 1.0)

        # --- L_stable (raw vs last) ---
        w_st = soft_weight(M_stable, gamma=0.8)
        w3s  = tf.repeat(w_st, 3, axis=-1)
        num_st = tf.reduce_sum(tf.abs(y_pred_raw - last) * w3s, axis=[1,2,3])
        den_st = tf.reduce_sum(w3s, axis=[1,2,3]) + 1e-6
        L_stable_val = tf.reduce_mean(num_st / den_st)

        # --- L_trans (seg forward, logging only during validation) ---
        pred01_for_seg = (y_pred_raw + 1.0) * 0.5
        last01_for_seg = (last       + 1.0) * 0.5
        pred_m11 = pred01_for_seg * 2.0 - 1.0
        last_m11 = last01_for_seg * 2.0 - 1.0

        prob_pred = self.seg_model(pred_m11, training=False)
        prob_last = self.seg_model(last_m11, training=False)

        m_w_pred = self._roi_mass(prob_pred, WOUND_IDX,  M_support)
        m_h_pred = self._roi_mass(prob_pred, HEALED_IDX, M_support)
        m_w_last = self._roi_mass(prob_last, WOUND_IDX,  M_support)
        m_h_last = self._roi_mass(prob_last, HEALED_IDX, M_support)

        k_norm = tf.reduce_mean(Xk[..., 3:4], axis=[1,2,3,4])
        use_trans = tf.cast(tf.abs(k_norm) >= self.K_dead, tf.float32)
        scale = tf.abs(k_norm) * use_trans
        eps = self.eps_trans

        L_pos_w = tf.nn.relu((m_w_pred - m_w_last) - eps)
        L_pos_h = tf.nn.relu((m_h_last - m_h_pred) - eps)
        L_neg_w = tf.nn.relu((m_w_last - m_w_pred) - eps)
        L_neg_h = tf.nn.relu((m_h_pred - m_h_last) - eps)

        is_pos = tf.cast(k_norm > 0.0, tf.float32)
        is_neg = tf.cast(k_norm < 0.0, tf.float32)

        L_trans_per = is_pos * (L_pos_w + L_pos_h) + is_neg * (L_neg_w + L_neg_h)
        L_trans_val = tf.reduce_mean(scale * L_trans_per)
        # ======= [END BLOCK #1] =======

        # (1) Prepare evaluation mask (same policy as training: slight tightening + ceil)
        M_comp = tighten_mask(M_change, ceil=0.88)
        M_comp = tf.clip_by_value(M_comp, 0.15, 0.95)
        M_loss = M_comp
        M3     = tf.repeat(M_comp, 3, axis=-1)

        # Compositing
        y_comp = (1.0 - M3) * last + M3 * y_pred_raw

        ratio = roi_delta_ratio(y_comp, last, M_loss)
        res_id = nonroi_identity_test(self.creator, Xk, M_loss, last)
        res_k  = k_sweep_monotonicity(self.creator, Xk, M_loss, last, K_vals=(0.2,0.6,1.0))

        # (2) Reconstruction (ROI-weighted)
        w1 = soft_weight(M_loss, gamma=0.8)         # [B,H,W,1]
        w3 = tf.repeat(w1, 3, axis=-1)                  # [B,H,W,3]
        num = tf.reduce_sum(tf.abs(y_next - y_comp) * w3, axis=[1,2,3])
        den = tf.reduce_sum(w3, axis=[1,2,3]) + 1e-6
        L_recon = tf.reduce_mean(num / den)

        tf.print("[VAL] M_next min/max/mean =",
                 tf.reduce_min(M_next), tf.reduce_max(M_next), tf.reduce_mean(M_next))
        tf.print("[VAL] tighten-> M_loss min/max/mean =",
                 tf.reduce_min(M_loss), tf.reduce_max(M_loss), tf.reduce_mean(M_loss))
        tf.print("[VAL] sum(w1) =", tf.reduce_sum(w1),
                 " y_stage2_ch =", tf.shape(y_stage2)[-1])

        # (3) Convert to [0,1] and compute Δ
        y01    = (y_comp + 1.0) * 0.5
        X01    = (Xk[..., :3] + 1.0) * 0.5
        last01 = (last   + 1.0) * 0.5
        tgt01  = (y_next + 1.0) * 0.5
        d_pred = tf.abs(y01 - last01)                      # [B,H,W,3]

        # (4) Context embedding (target features)
        X_2d  = tf.reshape(X01, [-1, H, W, 3])
        z_all = self.Proj(self.E(X_2d, training=False), training=False)
        z_all = tf.reshape(z_all, [B, T, -1])

        k_ctx = tf.minimum(tf.constant(CTX_K), T)
        ctx   = z_all[:, T - k_ctx:T, :]

        # (5) Extract K (mean)
        k_norm = tf.reduce_mean(Xk[..., 3:4], axis=[1,2,3,4])   # (B,)
        k_vec  = tf.expand_dims(k_norm, -1)                     # (B,1)
        z_goal = tf.stop_gradient(self.P([ctx, k_vec]))

        # (6) feature matching
        z_pred    = self.Proj(self.E(y01, training=False), training=False)
        feat_loss = tf.reduce_mean(tf.square(z_pred - z_goal))

        # (7) Δ magnitude (scalar) proportionality
        C_scalar   = change_scalar_from_history(Xk, M_change)
        target_mag = self.a0 + self.a1*C_scalar + self.a2*tf.cast(k_norm, C_scalar.dtype)
        num_mag    = tf.reduce_sum(tf.abs(y01 - last01) * w3, axis=[1,2,3])
        den_mag    = tf.reduce_sum(w3, axis=[1,2,3]) + 1e-6
        pred_mag   = num_mag / den_mag
        L_mag      = tf.reduce_mean(tf.abs(pred_mag - target_mag))

        # (8) Δ map (spatial) alignment
        C_map  = change_map_from_history(Xk, M_change)
        pred_m = tf.reduce_mean(tf.abs(y01 - last01), axis=-1, keepdims=True)
        tgt_m  = tf.reduce_mean(tf.abs(tgt01 - last01), axis=-1, keepdims=True)
        L_map  = tf.reduce_mean(
                   tf.reduce_sum(tf.abs(pred_m - tgt_m) * w1, axis=[1,2,3]) /
                   (tf.reduce_sum(w1, axis=[1,2,3]) + 1e-6)
                 )

        # (9) Anti-copy (keep same definition during evaluation)
        k_norm = tf.reduce_mean(Xk[..., 3:4], axis=[1,2,3,4])  # (B,)
        ac_loss = self.anti_copy_loss(
            y_pred_raw, last, M_change,    # Match mask policy used in test_step
            K_norm=k_norm, tau_min=0.01, tau_max=self.tau_anti, gamma=0.4
        )

        # (10) Direction / high-frequency / cosine
        L_feat_cos = cosine_loss(z_pred, z_goal)
        L_dir      = delta_dir_loss(y_comp, last, Xk, M_loss)
        L_hf       = edge_ratio_loss(y_comp, Xk, M_loss)

        # (11) Leakage & contrast (included same as training)
        L_leak = self.leak_penalty(d_pred, M_support)      # |Δ| outside ROI
        L_ctr  = self.contrast_loss(y01, tgt01, M_change)  # ROI contrast

        # (12) Total loss
        total = ( self.lambda_recon * L_recon
                + self.lambda_feat * feat_loss
                + self.lambda_anti * ac_loss
                + self.lambda_mag  * L_mag
                + self.lambda_map  * L_map
                + 0.2 * L_feat_cos + 0.25 * L_dir + 0.35 * L_hf
                + 0.30 * L_leak    + 0.20 * L_ctr )

        # Cosine similarity for monitoring
        z_pred_n = self.l2(z_pred)
        z_goal_n = self.l2(z_goal)
        cos = tf.reduce_mean(tf.reduce_sum(z_pred_n * z_goal_n, axis=-1))

        total, y_comp, logs = train_loss_with_kmono(
            self, Xk, y_stage2, y_pred_raw,
            mono_sign=+1.0, dk=0.20, kmono_margin=0.01,
            lambda_kmono=0.0,         # During validation, penalty is logged only
            deltaE_clamp=None         # Usually off during validation
        )

        # ======= [BLOCK #2] Record to logs only =======
        logs["val_L_stable"] = L_stable_val
        logs["val_L_trans"]  = L_trans_val
        logs["val_mw_pred"]  = tf.reduce_mean(m_w_pred)
        logs["val_mh_pred"]  = tf.reduce_mean(m_h_pred)
        # ======= [END BLOCK #2] =======

        ratio_t = tf.cast(ratio, tf.float32)
        id_psnr_t = tf.cast(res_id["psnr_pred_last"], tf.float32)
        id_dout_t = tf.cast(res_id["delta_out_mean"], tf.float32)   # Note: key name must be consistent
        k_mono_t  = tf.cast(res_k["monotonicity_penalty"], tf.float32)

        return logs


def stage1_train_step(X, tmask, optimizer):
    tf.debugging.assert_rank(X, 5, message="ds1_train must yield (B,T,H,W,C)")
    tf.debugging.assert_rank(tmask, 2, message="tmask must be (B,T)")

    B = tf.shape(X)[0]
    T = tf.shape(X)[1]
    H = tf.shape(X)[2]; W = tf.shape(X)[3]

    # dtype branching is done with Python if
    if tmask.dtype == tf.bool:
        tmask_i32 = tf.cast(tmask, tf.int32)
    else:
        tmask_i32 = tf.cast(tmask > 0.5, tf.int32)

    valid_T_per_sample = tf.reduce_sum(tmask_i32, axis=1)      # (B,)
    valid_T = tf.reduce_min(valid_T_per_sample)                # scalar int32
    valid_T = tf.maximum(valid_T, 1)

    max_dt = tf.minimum(DELTA_MAX, valid_T - 1)
    delta  = tf.random.uniform([], DELTA_MIN, max_dt + 1, dtype=tf.int32)
    k_ctx  = tf.minimum(CTX_K, valid_T - 1)

    t_min = k_ctx - 1
    t_max = valid_T - 1 - delta
    t = tf.cond(
        t_max >= t_min,
        lambda: tf.random.uniform([], t_min, t_max + 1, dtype=tf.int32),
        lambda: t_min,
    )

    with tf.GradientTape() as tape:
        X_2d = tf.reshape(X, [-1, H, W, 3])
        z_all = Proj(E(X_2d, training=True), training=True)  # (B*T,D)
        z_all = tf.reshape(z_all, [B, T, -1])                # (B,T,D)

        t_start = tf.maximum(0, t - (k_ctx - 1))
        t_end = tf.minimum(T, t + 1)
        ctx = z_all[:, t_start:t_end, :]
        t_pos = tf.minimum(t + delta, T - 1)
        z_pos = tf.gather(z_all, t_pos, axis=1, batch_dims=0)

        k_vec = tf.zeros([B, 1], tf.float32)
        z_goal = P([ctx, k_vec], training=True)
        loss = info_nce(z_goal, z_pos, temperature=TAU_NCE)

    vars_ = E.trainable_variables + Proj.trainable_variables + P.trainable_variables
    grads = tape.gradient(loss, vars_)
    optimizer.apply_gradients(zip(grads, vars_))
    return loss, delta, t

creator = creator_model(input_shape=(None, 64, 64, 5), gate_strength=0.8)
creator.summary()
print("base outputs:", [t.name for t in creator.outputs])

def take_X_tmask(X, Y, tmask):
    return X, tmask

train_seq_ds = (ds_train_paths
    .shuffle(len(train_seq_imgs), reshuffle_each_iteration=False)
    .map(load_seq, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH)
    .prefetch(tf.data.AUTOTUNE)
)

val_seq_ds = (ds_val_paths
    .map(load_seq, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH)
    .prefetch(tf.data.AUTOTUNE)
)

stage1_ds = (train_seq_ds                               # (Xseq, Yseq)
    .map(lambda X, Y: pad_repeat_last_tf(X[..., :3], MAX_T),
         num_parallel_calls=tf.data.AUTOTUNE)           # -> (Xp, tmask)
    .prefetch(tf.data.AUTOTUNE)
)

opt_s1 = tf.keras.optimizers.Adam(LR_STAGE1)

E.trainable = True; Proj.trainable = True; P.trainable = True
EPOCHS1 = 20

for epoch in range(EPOCHS1):
    for step, (X_rgb, tmask) in enumerate(stage1_ds):
        loss_s1, delta, t = stage1_train_step(X_rgb, tmask, optimizer=opt_s1)
    print(f"[Warm-up] epoch {epoch+1}/{EPOCHS1}  loss={loss_s1.numpy():.4f}  Δ~{int(delta)}")

def ensure_creator_input(X, k_value=0.2):
    X = tf.convert_to_tensor(X)
    r = tf.rank(X)
    # (B,H,W,3) → (B,1,H,W,3)
    X = tf.cond(tf.equal(r, 4), lambda: tf.expand_dims(X, 1), lambda: X)
    # Add k channel if last channel is 3
    def add_k(x):
        k = tf.ones(tf.concat([tf.shape(x)[:-1], [1]], axis=0), tf.float32) * tf.constant(k_value, tf.float32)
        return tf.concat([x, k], axis=-1)
    X = tf.cond(tf.equal(tf.shape(X)[-1], 3), lambda: add_k(X), lambda: X)
    return X  # (B,T,H,W,4)

def ensure_hw3(x):
    """
    Force-convert input tensor/array to (B,H,W,3) shape.
    - For 5D sequence input, select the last frame (T → -1)
    - Replicate single channel to 3 channels
    - Convert to float32
    """
    x = tf.convert_to_tensor(x)
    r = tf.rank(x)

    # (B,T,H,W,C) → (B,H,W,C)
    x = tf.cond(tf.equal(r, 5),
                lambda: x[:, -1, ...],
                lambda: x)

    # (H,W,C) → (1,H,W,C)
    x = tf.cond(tf.equal(r, 3),
                lambda: tf.expand_dims(x, 0),
                lambda: x)

    # Expand to 3 channels if single-channel
    c = tf.shape(x)[-1]
    x = tf.cond(tf.equal(c, 1),
                lambda: tf.tile(x, [1, 1, 1, 3]),
                lambda: x)

    # Convert to float32
    return tf.cast(x, tf.float32)

def normalize_to_X_M_last(batch):
    if isinstance(batch, (tuple, list)):
        if len(batch) == 3:
            X, M, last = batch
            return X, M, last
        elif len(batch) == 4:
            X, M, last, _ = batch  # ignore tmask
            return X, M, last
        elif len(batch) == 2:
            X, y_pack = batch
            M    = y_pack[..., :1]
            last = y_pack[..., 1:4]
            return X, M, last
    raise ValueError("Dataset structure is not normalized to (X, M, last).")


def to_m11(x):   return tf.cast(x, tf.float32) * 2.0 - 1.0
def to_01(x):    return (tf.cast(x, tf.float32) + 1.0) / 2.0

def quick_metrics_to_last(creator, ds, num_batches=3, use_mask=False, thr=0.5):
    psnr_vals, ssim_vals = [], []
    i = 0
    for Xk, y_stage2 in ds:
        # y_stage2 = concat([M_next(1), last(3), y_next(3)], -1)
        M_next = y_stage2[..., :1]       # [0,1]
        y_next = y_stage2[..., 4:7]      # [-1,1]  target

        out    = creator(Xk, training=False)
        y_pred = _first_output(out)       # [-1,1]

        # [-1,1] → [0,1]
        y_pred01 = (y_pred + 1.0) * 0.5
        y_next01 = (y_next + 1.0) * 0.5

        if not use_mask:
            # Based on full image
            ps = tf.image.psnr(y_next01, y_pred01, max_val=1.0)
            ss = tf.image.ssim(y_next01, y_pred01, max_val=1.0)
        else:
            # Simple ROI(mask)-based version: PSNR via weighted MSE→PSNR, SSIM via full-image SSIM
            m  = tf.cast(M_next > thr, tf.float32)                 # (B,H,W,1)
            m3 = tf.repeat(m, 3, axis=-1)

            # Weighted MSE
            num = tf.reduce_sum(tf.square(y_next01 - y_pred01) * m3, axis=[1,2,3])
            den = tf.reduce_sum(m3, axis=[1,2,3]) * 1.0 + 1e-6
            mse = num / den                                         # (B,)
            ps  = -10.0 * tf.math.log(mse) / tf.math.log(10.0)      # (B,)

            # SSIM (simple substitute): full-image SSIM (replace with bbox crop for precise ROI-SSIM)
            ss  = tf.image.ssim(y_next01, y_pred01, max_val=1.0)

        psnr_vals.append(tf.reduce_mean(ps).numpy())
        ssim_vals.append(tf.reduce_mean(ss).numpy())

        i += 1
        if i >= num_batches:
            break

    print(f"[quick] PSNR-to-next: {np.mean(psnr_vals):.3f} / SSIM-to-next: {np.mean(ssim_vals):.3f}")

ds2_train_for_fit = train_seq_ds.map(
    lambda Xseq, Yseq: to_stage2_seq(Xseq, Yseq, k_norm=0.2),
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

# Assume ds2_train_for_fit has shape (Xk, y_stage2)
for Xk, y_stage2 in ds2_train_for_fit.take(3):
    tf.print("[dbg] Xk shape:", tf.shape(Xk))
    tf.print("[dbg] y_stage2 shape:", tf.shape(y_stage2))

    # Input RGB range
    xr_min = tf.reduce_min(Xk[..., :3]).numpy()
    xr_max = tf.reduce_max(Xk[..., :3]).numpy()
    print(f"[INPUT] Xk RGB range: {xr_min:.4f} ~ {xr_max:.4f}")

    # Prediction
    out = creator(Xk, training=False)
    y_pred = out[0] if isinstance(out, (list, tuple)) else out  # Handle both single and dual output

    yp_min = tf.reduce_min(y_pred).numpy()
    yp_max = tf.reduce_max(y_pred).numpy()
    print(f"[OUTPUT] y_pred range: {yp_min:.4f} ~ {yp_max:.4f}")

    last   = y_stage2[..., 1:4]
    y_next = y_stage2[..., 4:7]
    M_next = y_stage2[..., :1]

    delta_mean = tf.reduce_mean(tf.abs(y_next - last))
    tf.print("[Δ-check] mean|y_next - last| =", delta_mean)

    tf.print("[channel means y_next]", tf.reduce_mean(y_next, axis=[0,1,2]))
    tf.print("[channel means last]  ", tf.reduce_mean(last, axis=[0,1,2]))

    tf.print("[mask stats] mean/min/max:",
             tf.reduce_mean(M_next),
             tf.reduce_min(M_next),
             tf.reduce_max(M_next))

ds2_val_for_fit = val_seq_ds.map(
    lambda Xseq, Yseq: to_stage2_seq(Xseq, Yseq, k_norm=0.2),
    num_parallel_calls=tf.data.AUTOTUNE
).prefetch(tf.data.AUTOTUNE)

quick_metrics_to_last(creator, ds2_train_for_fit, num_batches=3)

trainer = CreatorTrainer(
    creator, E, Proj, P, seg_model=mask_model,
    lambda_feat=1.0,      # Initial value; overwritten by the callback above
    lambda_anti=0.2,
    lambda_mag=0.6,
    lambda_map=0.3,
    lambda_recon=1.2,
    lambda_stable = 0.1,
    lambda_trans=0.05,
    delta_scale=0.2, feather_k=3, tau_anti=0.1
)

trainer.compile(optimizer=tf.keras.optimizers.Adam(1e-4))

creator_history = trainer.fit(
    ds2_train_for_fit,
    validation_data=ds2_val_for_fit,
    epochs=10,
    callbacks=[model_checkpoint, early_stopping, reduce_lr]
)

creator.save("/content/creator_model.keras", include_optimizer=False)
creator = tf.keras.models.load_model(
    "/content/creator_model.keras",
    custom_objects={
        "TileCutter1": TileCutter1, "TileCutter2": TileCutter2,
        "TimePreservingUpSampling2D": TimePreservingUpSampling2D,
        "LeakyReLU": tf.keras.layers.LeakyReLU, "FloatCastLayer": FloatCastLayer,
        "RGBSlice": RGBSlice, "KSlice": KSlice, "MSlice": MSlice, "SoftClamp": SoftClamp,
        "GateScale": GateScale, "OnesLike": OnesLike, "XLastPicker": XLastPicker, "ScaledTanh": ScaledTanh, "ScaleBias": ScaleBias,
        "SoftClip": SoftClip, "TanhWithTemp": TanhWithTemp, "TemporalAttentionPooling": TemporalAttentionPooling,
        "DebugPrint": DebugPrint, "SelectLastTime": SelectLastTime, "RepeatChannels3": RepeatChannels3
    },
    compile=False
)
print("outputs:", [t.name for t in creator.outputs])  # Expected: ['y_pred', 'x_last']

COS_OK = 0.80

hist = creator_history.history
if "val_cos" in hist:
    cos_val = hist["val_cos"][-1]
elif "cos" in hist:
    cos_val = hist["cos"][-1]
else:
    raise KeyError("'cos' or 'val_cos' not found in History. Check that CreatorTrainer returns 'cos'.")

print(f"✅ Creator output COS: {cos_val:.4f}")

