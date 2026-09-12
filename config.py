"""
config.py — Central hyperparameter configuration.
Edit ONLY this file; never hardcode values in train.py or dataset.py.
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(ROOT_DIR, "data")
HR_DIR         = os.path.join(DATA_DIR, "hr")        # high-res .tif patches
SEG_DIR        = os.path.join(DATA_DIR, "seg")       # segmentation label patches
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")

# ─── Image / Band Settings ───────────────────────────────────────────────────
NUM_BANDS    = 4          # Sentinel-2 bands loaded: B2, B3, B4, B8
LR_SIZE      = 256        # spatial size of LR input patch
HR_SIZE      = 1024       # spatial size of HR target  (4× upscale)
SCALE_FACTOR = 4          # super-resolution upscale factor

# ─── Segmentation ────────────────────────────────────────────────────────────
NUM_CLASSES  = 6          # road, building, water, cropland, vegetation, other
CLASS_NAMES  = ["road", "building", "water", "cropland", "vegetation", "other"]

# ─── Model Architecture ──────────────────────────────────────────────────────
SR_FEAT_CHANNELS  = 96   # intermediate feature channels tapped from SR backbone
UNET_OUT_CHANNELS = 64   # UNet decoder output channels (fed to KAN head)
BACKBONE_MID_CH   = 64   # internal channels in the SR CNN backbone

# ─── Degradation (LR synthesis) ──────────────────────────────────────────────
BLUR_SIGMA_MIN  = 0.5
BLUR_SIGMA_MAX  = 2.0
NOISE_STD_MAX   = 15.0   # gaussian noise σ, in [0, NOISE_STD_MAX]

# ─── Training ────────────────────────────────────────────────────────────────
BATCH_SIZE   = 4
NUM_EPOCHS   = 100
LEARNING_RATE = 1e-4
NUM_WORKERS   = 2        # DataLoader workers

# Loss weights
W_L1   = 1.0
W_SSIM = 0.1
W_CE   = 1.0

# ─── Logging / Checkpointing ─────────────────────────────────────────────────
SAVE_EVERY   = 10        # save checkpoint every N epochs
LOG_EVERY    = 1         # log metrics every N epochs

# ─── Inference ───────────────────────────────────────────────────────────────
PATCH_SIZE   = 256
PATCH_STRIDE = 192       # overlap = PATCH_SIZE - PATCH_STRIDE = 64 px

# ─── Streamlit Demo ──────────────────────────────────────────────────────────
DEMO_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best.pth")
