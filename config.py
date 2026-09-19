"""
config.py — Central hyperparameter configuration.
Edit ONLY this file; never hardcode values in train.py or dataset.py.
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(ROOT_DIR, "data")
HR_DIR         = "/kaggle/working/hr_tiles"          # EuroSAT .jpg flat folder on Kaggle
SEG_DIR        = os.path.join(DATA_DIR, "seg")       # segmentation label patches
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")

# ─── Dataset mode ────────────────────────────────────────────────────────────
USE_DUMMY = False   # True  → DummyDataset (random tensors, no files needed)
                    # False → TiffDataset  from HR_DIR (EuroSAT .jpg on Kaggle)

# ─── Image / Band Settings ───────────────────────────────────────────────────
NUM_BANDS    = 4          # Sentinel-2 bands; EuroSAT RGB (3 ch) padded to 4
PATCH_SIZE   = 64         # EuroSAT native size is 64×64 — use tiles as HR target
LR_SIZE      = 16         # = PATCH_SIZE // 4  (16×16 LR input to the model)
HR_SIZE      = PATCH_SIZE # SR target = original patch size  (64×64)
SCALE_FACTOR = 4          # super-resolution upscale: LR_SIZE × 4 = HR_SIZE

# ─── Segmentation ────────────────────────────────────────────────────────────
NUM_CLASSES  = 6          # road, building, water, cropland, vegetation, other
CLASS_NAMES  = ["road", "building", "water", "cropland", "vegetation", "other"]

# ─── Model Architecture ──────────────────────────────────────────────────────
SR_FEAT_CHANNELS  = 96   # intermediate feature channels tapped from SR backbone
UNET_OUT_CHANNELS = 64   # UNet decoder output channels (fed to seg head)
BACKBONE_MID_CH   = 64   # internal channels in the SR CNN backbone

# ── Backbone options ──────────────────────────────────────────────────────────
USE_SPECTRAL_STEM     = True   # depthwise-sep stem with LayerNorm before backbone
USE_STRIDE_DOWNSAMPLE = True   # stride-2 conv: LR_SIZE → LR_SIZE//2 before body
                                # 16× memory reduction vs full-res attention

# Derived: feature map spatial size fed into UNet decoder
FEAT_SPATIAL = LR_SIZE // 2 if USE_STRIDE_DOWNSAMPLE else LR_SIZE   # 8 or 16

# Derived: total upscale the Segmentation_Upsampler must do
SEG_UPSCALE  = SCALE_FACTOR * (2 if USE_STRIDE_DOWNSAMPLE else 1)   # 8 or 4

# ─── Segmentation head ───────────────────────────────────────────────────────
HEAD_TYPE = "conv"    # "conv" (stable baseline) | "wavkan" (experimental KAN head)

# ─── Degradation (LR synthesis) ──────────────────────────────────────────────
BLUR_SIGMA_MIN  = 0.5
BLUR_SIGMA_MAX  = 2.0
NOISE_STD_MAX   = 15.0   # gaussian noise σ in [0, NOISE_STD_MAX]

# ─── Training ────────────────────────────────────────────────────────────────
BATCH_SIZE    = 8
NUM_EPOCHS    = 10
LEARNING_RATE = 1e-4
NUM_WORKERS   = 2        # DataLoader workers

# Loss weights
W_L1   = 1.0
W_SSIM = 0.1
W_CE   = 1.0

# ── Perceptual loss (VGG, ramped from 0 over PERCEPTUAL_RAMP_STEPS) ──────────
PERCEPTUAL_LOSS_WEIGHT = 0.0    # starting weight (ramp target = 0.01)
PERCEPTUAL_RAMP_STEPS  = 10000  # linearly ramp to 0.01 over this many steps
                                 # set to 0 to disable perceptual loss entirely

# ─── Logging / Checkpointing ─────────────────────────────────────────────────
SAVE_EVERY   = 10        # save checkpoint every N epochs
LOG_EVERY    = 1         # log metrics every N epochs

# ─── Inference ───────────────────────────────────────────────────────────────
PATCH_STRIDE = 48        # overlap stride for inference tiling

# ─── Streamlit Demo ──────────────────────────────────────────────────────────
DEMO_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best.pth")
