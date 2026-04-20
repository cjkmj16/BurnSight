# 🔥 BurnSight

**BurnSight** is a deep learning framework for wound healing trajectory prediction from sequential burn wound images.  
It segments wound regions (wound / eschar / healed) and synthesizes future-state wound images conditioned on a healing-progress parameter **K**.

---

## Architecture Overview

```
Sequential wound images (T frames)
        │
        ▼
┌─────────────────┐
│  Segmentation   │  MobileNetV2-backbone U-Net
│  (mask model)   │  → wound / eschar / healed / exclude / background
└────────┬────────┘
         │  m_soft ROI gate
         ▼
┌─────────────────┐
│    Creator      │  ConvLSTM2D encoder + U-Net decoder
│  (K-conditioned)│  Input: (RGB, K-channel, ROI mask) × T frames
│                 │  Output: predicted next-state wound image
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Postprocessor  │  allow / protect / uncertain region decision
│                 │  α-blending with K-aware amplitude scheduling
└─────────────────┘
         │
         ▼
  Predicted wound image + Lab color change statistics (Δa*, ΔE)
```

**K parameter** (0 → 1): Controls predicted healing progression.  
- `K = 0` : current state (no change)  
- `K = 1` : maximum predicted healing

---

## Quick Start — Google Colab

Click the badge below to open the setup notebook directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/BurnSight_Colab.ipynb)

> **Note:** Replace `YOUR_USERNAME/YOUR_REPO` with your actual GitHub path after uploading.

---

## Repository Structure

```
BurnSight_src/
├── main_train.py               # Full training + inference entry point
├── src/
│   ├── config.py               # All hyperparameters & path configuration
│   ├── data/
│   │   ├── file_utils.py       # File discovery, patient/day parsing, pair collection
│   │   ├── dataset.py          # tf.data pipeline, sequence builder, augmentation
│   │   └── augment.py          # ID transform calibration, style matching
│   ├── models/
│   │   ├── layers.py           # Custom Keras layers
│   │   ├── unet.py             # MobileNetV2 segmentation U-Net
│   │   ├── encoder.py          # Stage-1 temporal encoder / projection head
│   │   ├── creator.py          # Creator model + CreatorTrainer
│   │   └── refiner.py          # Refiner GAN (generator + discriminator)
│   ├── losses/
│   │   ├── creator_losses.py   # Perceptual, NCE, cosine losses for Creator
│   │   └── refiner_losses.py   # GAN losses for Refiner
│   ├── inference/
│   │   ├── mask_utils.py       # ROI gate construction, threshold calibration
│   │   ├── eval_utils.py       # K-sweep, Lab delta analysis
│   │   └── postprocess.py      # allow/protect/uncertain + α-blending
│   └── utils/
│       ├── metrics.py          # Dice, SSIM, PSNR, soft-dice
│       ├── image_utils.py      # Normalization, overlay, CLAHE
│       └── debug_utils.py      # Visualization panels, logging
```

---

## Data Format

BurnSight expects wound image sequences organized **per patient**:

```
wd_aug/
└── pat01/
    ├── Day1_20230101.png
    ├── Day1_20230101_mask.npz   ← companion mask file
    ├── Day3_20230103.png
    ├── Day3_20230103_mask.npz
    └── ...
```

**Mask `.npz` format:** each file contains a `masks` key with shape `(H, W, 5)` — one-hot encoded across 5 classes:

| Index | Class       |
|-------|-------------|
| 0     | Background  |
| 1     | Wound       |
| 2     | Eschar      |
| 3     | Healed      |
| 4     | Exclude     |

**Filename convention:** filenames must include `DayN` (e.g., `Day7`) or a date string (`YYYYMMDD`) so the pipeline can sort frames chronologically.

---

## Data Access

Training data is sourced from the **NIA (National Information Society Agency) 
Open Dataset** for medical image AI development.

> ⚠️ **Patient data is not included in this repository.  
> This codebase contains only model architecture and training pipeline.**

We provide two paths depending on your use case:

### Option A — Quick demo with synthetic sample data (no approval needed)
The Colab notebook includes a **Step 4-A** cell that auto-generates a small synthetic sequence so you can run the full inference pipeline immediately without any real data.

### Option B — Access to anonymized sample data (small subset)
A small anonymized sample dataset (≤ 10 patient sequences, IRB-reviewed) is available for research purposes.  
To request access, please email **[your-email@institution.edu]** with:
- Your name and affiliation
- Brief description of intended use

You will receive a time-limited Google Drive link after review (typically within 3 business days).

### Option C — Access to full training data
The full dataset is available only to collaborating institutions under a formal data sharing agreement.  
Please contact us at **[your-email@institution.edu]** for details.

---

## Setup: Google Drive Folder Structure

Once you have obtained data access (Option B or C above), organize your Drive as follows:

```
MyDrive/
├── BurnSight_sample/       ← sample data received via access request
│   ├── pat01/
│   │   ├── Day1_20230101.png
│   │   ├── Day1_20230101_mask.npz
│   │   └── ...
│   └── pat02/ ...
├── mask_model.keras        ← pretrained segmentation model (see below)
├── creator.keras           ← pretrained creator model (see below)
└── cache_dir/              ← auto-created by the notebook
```

Update the Drive paths in the Colab notebook's **Step 4** config cell to match your layout.

---

## Configuration (`src/config.py`)

Key parameters you may want to adjust:

| Parameter      | Default | Description                              |
|----------------|---------|------------------------------------------|
| `SEQLEN`       | 6       | Number of input frames per sequence      |
| `IMG_SIZE`     | (64,64) | Image resolution                         |
| `NUM_CLASSES`  | 5       | Segmentation classes                     |
| `WOUND_IDX`    | 1       | Class index for wound                    |
| `K_MIN/K_MAX`  | 0 / 5   | K parameter range                        |
| `BATCH`        | 8       | Batch size                               |
| `VAL_RATIO`    | 0.10    | Validation split ratio                   |
| `SEED`         | 42      | Global random seed                       |

---

## Pretrained Models

Pretrained model weights (`.keras`) are available via Google Drive:

| Model         | Description                          | Link |
|---------------|--------------------------------------|------|
| `mask_model`  | Segmentation U-Net (MobileNetV2)     | [Download](#) |
| `creator`     | K-conditioned wound image synthesizer| [Download](#) |

> Replace `[Download](#)` links with your actual Google Drive sharing URLs.

Place downloaded `.keras` files in your Drive and update the model load paths in the Colab notebook's **Step 3: Load Models** cell.

---

## Requirements

All dependencies are installed automatically in the Colab notebook. For local use:

```bash
pip install tensorflow==2.15 \
            opencv-python \
            scikit-learn \
            scikit-image \
            scipy \
            matplotlib \
            pandas \
            imageio \
            h5py \
            joblib \
            albumentations
```

Optional (GPU acceleration):
```bash
pip install cupy-cuda12x tensorflow-probability
```

---

## Citation

If you use BurnSight in your research, please cite:

```bibtex
@misc{burnsight2025,
  title   = {BurnSight: K-conditioned Wound Healing Trajectory Prediction},
  year    = {2025},
  url     = {https://github.com/YOUR_USERNAME/YOUR_REPO}
}
```

---

## License

This project is released for research use. See `LICENSE` for details.
