"""
BurnSight file discovery / path parsing / pair collection
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
from src.config import *

def log_inverted(tag, img_path, mask_path):
    with open(DBG_LIST, "a") as f:
        f.write(f"{tag}\t{img_path}\t{mask_path}\n")

def _to_str(x):
    x = x.numpy() if hasattr(x, "numpy") else x
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8", "ignore")
    return str(x)

def extract_day_and_date(filename):
    """Extract Day number and date from filename."""
    day_match = re.search(r'(Day\d+)', filename)
    date_match = re.search(r'\d{8}', filename)

    day_part = day_match.group(1) if day_match else None
    date_part = date_match.group(0) if date_match else None

    if date_part:
        date = datetime.strptime(date_part, '%Y%m%d')
    else:
        date = None

    return day_part, date

def get_sorted_day_images(folder):
    """
    Return image files in a folder sorted in DayN order.
    """
    files = []
    for root, _, filenames in os.walk(folder):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                files.append(os.path.join(root, filename))

    # Sort by DayN and date
    return sorted(files, key=lambda x: (extract_day_and_date(x)[0], extract_day_and_date(x)[1] or datetime.min))

def is_image(fname):
    return fname.lower().endswith(('.png','.jpg','.jpeg'))

def list_patients(root):
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root,d))])

PAT_RE = re.compile(r"(pat\d+)", re.IGNORECASE)

DAY_RE   = re.compile(r"[ _-]day(\d+)", re.IGNORECASE)
DATE2_RE = re.compile(r"_(\d{8})[-_](\d{6})")  # _YYYYMMDD-HHMMSS
DATE1_RE = re.compile(r"_(\d{8})")             # _YYYYMMDD

def extract_time_key(fname: str, fpath: str) -> float:
    name = os.path.basename(fname).lower()
    m = DAY_RE.search(name)
    if m:                         # Day number is the most reliable
        return float(int(m.group(1)))
    m = DATE2_RE.search(name)
    if m:
        ymd, hms = int(m.group(1)), int(m.group(2))
        return float(ymd * 1_000_000 + hms)
    m = DATE1_RE.search(name)
    if m:
        return float(int(m.group(1)))
    try:
        return os.path.getmtime(fpath)   # Last resort: file modification time
    except Exception:
        return 0.0

def get_base_pid(path):
    """
    Examples:
      /.../pat12_aug03/...  → pat12
      /.../Pat07/...        → pat07
      /.../PAT9_AUG10/...   → pat9
    """
    path_norm = path.replace("\\", "/").lower()
    m = PAT_RE.search(path_norm)
    if not m:
        return "unknown"
    return m.group(1)

def collect_aug_pairs():
    pairs = []
    for root, _, files in os.walk(aug_dir):
        pngs = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        for f in pngs:
            base = os.path.splitext(f)[0]
            img = os.path.join(root, f)
            npz = os.path.join(root, base + "_mask.npz")
            if os.path.exists(npz):
                pid = get_base_pid(root)
                pairs.append((img, npz, pid))
    return sorted(pairs)

def collect_pairs_by_pid(root_dir: str):
    """
    return: buckets[pid] = list of (time_key, img_path, mask_path)
    - img: *.png / *.jpg / *.jpeg
    - mask: same base + '_mask.npz'
    """
    buckets = defaultdict(list)
    for pid in sorted(os.listdir(root_dir)):
        pid_dir = os.path.join(root_dir, pid)
        if not os.path.isdir(pid_dir):
            continue

        files = sorted(os.listdir(pid_dir))
        imgs  = [f for f in files if f.lower().endswith((".png",".jpg",".jpeg"))]

        for imgf in imgs:
            base, _ = os.path.splitext(imgf)
            img_path  = os.path.join(pid_dir, imgf)
            mask_path = os.path.join(pid_dir, base + "_mask.npz")
            if os.path.exists(mask_path):
                tk = extract_time_key(imgf, img_path)
                buckets[pid].append((tk, img_path, mask_path))
    # Sort chronologically
    for pid in buckets:
        buckets[pid].sort(key=lambda x: x[0])
    return buckets

def filter_out_bad_npz(train_pairs, bad_npz_set):
    out = []
    for img_p, npz_p, pid in train_pairs:
        if npz_p in bad_npz_set:
            continue
        # Exclude zero-byte or very small files
        if (not os.path.exists(npz_p)) or os.path.getsize(npz_p) < 128:
            continue
        out.append((img_p, npz_p, pid))
    return out

def build_sequences(buckets, pid_list, T=6, S=1):
    seq_imgs, seq_msks = [], []
    for pid in pid_list:
        triples = buckets[pid]              # (t, img_path, msk_path)
        n = len(triples)
        if n < T:
            continue
        for start in range(0, n - T + 1, S):
            window = triples[start:start + T]
            imgs = [w[1] for w in window]   # String paths only
            msks = [w[2] for w in window]   # String paths only
            seq_imgs.append(imgs)           # list[str] of length T
            seq_msks.append(msks)           # list[str] of length T
    return seq_imgs, seq_msks

aug_pairs = collect_aug_pairs(config.AUG_DIR)
pid_buckets = collect_pairs_by_pid(config.AUG_DIR)

if not aug_pairs:
    import warnings
    warnings.warn("No pairs found in AUG_DIR — running in synthetic/inference mode.")
    train_pairs = val_pairs = []
    train_pids = val_pids = []
    train_seq_imgs = train_seq_msks = []
    val_seq_imgs = val_seq_msks = []
else:
    rng = np.random.RandomState(SEED)

    all_pids_aug = sorted({pid for _,_,pid in aug_pairs})
    n_val_aug = max(1, int(len(all_pids_aug)*VAL_RATIO))
    val_pids_aug = set(rng.choice(all_pids_aug, size=n_val_aug, replace=False))
    train_pids_aug = [p for p in all_pids_aug if p not in val_pids_aug]

    train_pairs = [(i,m,p) for (i,m,p) in aug_pairs if p in train_pids_aug]
    val_pairs   = [(i,m,p) for (i,m,p) in aug_pairs if p in val_pids_aug]

    all_pids = sorted([p for p in pid_buckets.keys() if len(pid_buckets[p]) >= SEQ_LEN])
    n_val = max(1, int(len(all_pids)*VAL_RATIO))
    val_pids = set(rng.choice(all_pids, size=n_val, replace=False))
    train_pids = [p for p in all_pids if p not in val_pids]

    train_seq_imgs, train_seq_msks = build_sequences(pid_buckets, train_pids, T=SEQ_LEN, S=STRIDE)
    val_seq_imgs,   val_seq_msks   = build_sequences(pid_buckets, val_pids,   T=SEQ_LEN, S=STRIDE)

    print(f"train seqs: {len(train_seq_imgs)}  val seqs: {len(val_seq_imgs)}")
    
print(f"train seqs: {len(train_seq_imgs)}  val seqs: {len(val_seq_imgs)}")
