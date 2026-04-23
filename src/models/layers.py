"""
BurnSight custom Keras layer collection.
All layers are ported from the original notebook without modification.
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

class L2Normalize(layers.Layer):
    def __init__(self, axis=-1, epsilon=1e-7, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon
    def call(self, x):
        return tf.nn.l2_normalize(x, axis=self.axis, epsilon=self.epsilon)
    def get_config(self):
        return {**super().get_config(), "axis": self.axis, "epsilon": self.epsilon}

class SoftDilate(layers.Layer):
    def __init__(self, k=5, **kwargs):
        super().__init__(**kwargs)
        self.k = int(k)
    def call(self, x):
        return tf.nn.max_pool(x, ksize=[1,self.k,self.k,1], strides=[1,1,1,1], padding="SAME")
    def get_config(self):
        return {**super().get_config(), "k": self.k}

def extract_patches_2d(x, k=32, s=32):
    # x: (B,H,W,C) -> (B, Np, k*k*C)
    patches = tf.image.extract_patches(
        images=x, sizes=[1,k,k,1], strides=[1,s,s,1], rates=[1,1,1,1], padding='VALID'
    )
    B = tf.shape(x)[0]
    Np = tf.shape(patches)[1] * tf.shape(patches)[2]
    patches = tf.reshape(patches, [B, Np, k*k*tf.shape(x)[-1]])
    return patches


class TileCutter1(tf.keras.layers.Layer):
    def call(self, inputs):
        tensor1, tensor2 = inputs
        min_len = tf.minimum(tf.shape(tensor1)[1], tf.shape(tensor2)[1])
        return tensor1[:, :min_len]

class TileCutter2(tf.keras.layers.Layer):
    def call(self, inputs):
        tensor1, tensor2 = inputs
        min_len = tf.minimum(tf.shape(tensor1)[1], tf.shape(tensor2)[1])
        return tensor2[:, :min_len]

class SequenceLengthLayer(Layer):
    def call(self, inputs):
        return tf.shape(inputs)[1]

class TimePreservingUpSampling2D(Layer):
    def __init__(self, size=(2, 2), method='bilinear', **kwargs):
        super(TimePreservingUpSampling2D, self).__init__(**kwargs)
        self.size = size
        self.method = method

    def call(self, x):
        # x: (batch, time, height, width, channels)
        # flatten batch and time to iterate over (batch*time, h, w, c)
        shape = tf.shape(x)
        batch_size, time_steps = shape[0], shape[1]
        reshaped = tf.reshape(x, (-1, shape[2], shape[3], shape[4]))  # (batch*time, h, w, c)

        # resize each frame
        resized = tf.image.resize(reshaped, size=[
            tf.cast(shape[2] * self.size[0], tf.int32),
            tf.cast(shape[3] * self.size[1], tf.int32)
        ], method=self.method)

        # reshape back
        up_height = tf.shape(resized)[1]
        up_width = tf.shape(resized)[2]
        channels = tf.shape(resized)[3]
        output = tf.reshape(resized, (batch_size, time_steps, up_height, up_width, channels))
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            'size': self.size,
            'method': self.method,
        })
        return config

class RGBSlice(tf.keras.layers.Layer):
    def call(self, x): return x[..., :3]
    def get_config(self): return {}

class KSlice(tf.keras.layers.Layer):
    def call(self, x): return x[..., 3:4]
    def get_config(self): return {}

class MSlice(tf.keras.layers.Layer):
    def call(self, x):
        # RGB: 0:3, K:3:4, Mask:4:5  (assumes 5-channel input)
        return x[..., 4:5]
    def get_config(self):
        return {}

class FloatCastLayer(tf.keras.layers.Layer):
    def call(self, x):
        return tf.cast(x, tf.float32)
    def get_config(self):
        return super().get_config()
        
class ScaleBias(tf.keras.layers.Layer):
    def build(self, input_shape):
        c = input_shape[-1]
        self.alpha = self.add_weight(name="alpha", shape=(c,), initializer="ones",  trainable=True)
        self.beta  = self.add_weight(name="beta",  shape=(c,), initializer="zeros", trainable=True)
    def call(self, x): return x * self.alpha + self.beta

class SoftClip(layers.Layer):
    def __init__(self, limit=1.1, **kw):
        super().__init__(**kw); self.limit=float(limit)
    def call(self, x):
        return self.limit * ops.tanh(x / self.limit)

class GatedSkip(tf.keras.layers.Layer):
    def __init__(self, channels, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.channels = int(channels)
        self.dropout_rate = float(dropout_rate)
        self.proj = tf.keras.layers.Conv2D(self.channels, 1, padding='same')
        self.gate = tf.keras.layers.Conv2D(self.channels, 1, padding='same')
        self.drop = tf.keras.layers.Dropout(self.dropout_rate)

    def build(self, input_shape):
        # input_shape: (B,H,W,Cin)
        self.proj.build(input_shape)
        self.gate.build(input_shape)
        self.drop.build(input_shape)
        self.built = True  # Important!

    def call(self, inputs, training=None, **kwargs):
        x = inputs
        x_proj = self.proj(x)
        g = tf.nn.sigmoid(self.gate(x_proj))
        out = x_proj * g
        out = self.drop(out, training=training)
        return out

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.channels,)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"channels": self.channels, "dropout_rate": self.dropout_rate})
        return cfg

class TanhWithTemp(tf.keras.layers.Layer):
    def build(self, _):
        self.temp = self.add_weight(
            name='temp',
            shape=(),
            initializer=tf.keras.initializers.Constant(1.5)
        )

    def call(self, x):
        return tf.tanh(x * self.temp)

class ScaledTanh(tf.keras.layers.Layer):
    def __init__(self, init_alpha=1.8, init_beta=0.0, per_channel=True, **kw):
        super().__init__(**kw)
        self.init_alpha = float(init_alpha)
        self.init_beta  = float(init_beta)
        self.per_channel = per_channel
    def build(self, input_shape):
        ch = input_shape[-1] if self.per_channel else 1
        self.u_alpha = self.add_weight(name='u_alpha', shape=(ch,),
                                       initializer=tf.keras.initializers.Constant(np.log(np.exp(self.init_alpha)-1)),
                                       trainable=True)
        self.beta    = self.add_weight(name='beta',    shape=(ch,),
                                       initializer=tf.keras.initializers.Zeros(),
                                       trainable=True)
    def call(self, x):
        # α = softplus(u) ensures positivity and prevents excessive shrinkage
        alpha = tf.nn.softplus(self.u_alpha) + 1e-6  # shape=(C,)
        y = tf.tanh(x * alpha) + self.beta
        return y

class GateScale(tf.keras.layers.Layer):
    def __init__(self, s=1.0, **kwargs):
        super().__init__(**kwargs); self.s = float(s)
    def call(self, x): return x * self.s
    def get_config(self): return {"s": self.s, **super().get_config()}

class OnesLike(tf.keras.layers.Layer):
    def call(self, x): return tf.ones_like(x)

class XLastPicker(tf.keras.layers.Layer):
    def call(self, x):
        # x: (B,T,H,W,C)
        return x[:, -1, ...]

class DebugPrint(layers.Layer):
    def __init__(self, tag, **kw):
        super().__init__(**kw); self.tag = tag
    def call(self, x):
        tf.print(self.tag,
                 "min/max/mean/std=",
                 tf.reduce_min(x), tf.reduce_max(x),
                 tf.reduce_mean(x), tf.math.reduce_std(x))
        return x

class TemporalAttentionPooling(layers.Layer):
    def __init__(self, temp=0.6, eps=1e-3, **kw):
        super().__init__(**kw)
        self.fc = layers.Dense(1)
        self.temp = float(temp)
        self.eps = float(eps)

    def call(self, x, m=None, training=None):
        # x: (B,T,H,W,C)
        # 1) Spatial average → (B,T,C)
        attn = ops.mean(x, axis=(2, 3))                          # (B,T,C)

        # 2) Compute logits → (B,T,1)
        logits = self.fc(attn)                                   # (B,T,1)
        if self.temp != 1.0:
            logits = logits / self.temp

        # 3) Weight by mask (reflect frame quality)
        if m is not None:
            # m: (B,T,H,W,1) → (B,T)
            mmean = ops.mean(m, axis=(2, 3, 4))                  # (B,T)
            mmean = ops.expand_dims(mmean, axis=-1)              # (B,T,1)
            logits = logits + ops.log(mmean + self.eps)

        # 4) softmax over time → (B,T,1)
        a = ops.softmax(logits, axis=1)                          # (B,T,1)

        # 5) Expand to (B,T,1,1,1) and multiply with x → (B,T,H,W,C)
        a = ops.expand_dims(a, axis=2)                           # (B,T,1,1)
        a = ops.expand_dims(a, axis=3)                           # (B,T,1,1,1)

        # 6) Weighted sum over time axis → (B,H,W,C)
        y = ops.sum(x * a, axis=1)                               # (B,H,W,C)
        return y

class SoftClamp(layers.Layer):
    def __init__(self, lo=0.1, hi=0.9, temp=1.0, **kw):
        super().__init__(**kw)
        assert 0.0 <= lo < hi <= 1.0
        self.lo=float(lo); self.hi=float(hi); self.temp=float(temp)
    def call(self, x):
        mid  = (self.lo + self.hi) * 0.5
        half = (self.hi - self.lo) * 0.5
        z = (x - mid) / (half * self.temp + 1e-6)
        # Must use keras.ops for Functional graph safety
        return mid + half * ops.tanh(z)

class SelectLastTime(layers.Layer):
    """Input: (B,T,H,W,C) → Output: (B,H,W,C)  (last time step)"""
    def call(self, x):
        # x[:, -1] is safe in Keras graph mode (inside layer)
        return x[:, -1]

    def compute_output_shape(self, input_shape):
        # input_shape: (B,T,H,W,C)
        return (input_shape[0], input_shape[2], input_shape[3], input_shape[4])

class RepeatChannels3(layers.Layer):
    """Input: (B,H,W,1) → Output: (B,H,W,3)  (channel repeat)"""
    def call(self, x):
        return tf.repeat(x, repeats=3, axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], 3)
