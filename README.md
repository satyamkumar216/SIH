# Satellite Image Super-Resolution & Land-Cover Segmentation

A PyTorch deep-learning pipeline for **4× super-resolution** and **6-class land-cover segmentation** of Sentinel-2 multispectral imagery.

---

## Architecture

```
Input (B, 4, 256, 256)
       │
       ▼
┌──────────────────┐
│  Module 1        │  SR_Backbone (lightweight CNN)
│  SR_Backbone     │  · 8 residual blocks
│                  │  · pixel-shuffle 4× upsampler
└──────┬───────────┘
       │ sr_image (B, 4, 1024, 1024)    ← Output 1
       │ features (B, 96,  256,  256)   ← tapped before upsampling
       ▼
┌──────────────────┐
│  Module 2        │  UNet_Decoder
│  UNet_Decoder    │  · encoder: 256→128→64→32 spatial
│                  │  · decoder: bilinear up + skip concat
└──────┬───────────┘
       │ (B, 64, 256, 256)
       ▼
┌──────────────────┐
│  Module 3        │  WavKAN_Head
│  WavKAN_Head     │  · pixel-wise KAN (efficient-kan)
│                  │  · falls back to 1×1 conv if KAN unavailable
└──────┬───────────┘
       │ (B, 6, 256, 256)
       ▼
┌──────────────────┐
│  Module 4        │  Segmentation_Upsampler
│  Seg_Upsampler   │  · conv + PixelShuffle 4× (learnable)
│                  │
└──────┬───────────┘
       │ seg_logits (B, 6, 1024, 1024)  ← Output 2
```

---

## File Structure

```
satellite_sr/
├── models/
│   ├── sr_backbone.py          # Module 1
│   ├── unet_decoder.py         # Module 2
│   ├── wavkan_head.py          # Module 3
│   └── segmentation_upsampler.py  # Module 4
├── model.py                    # Combines all 4 modules
├── dataset.py                  # TiffDataset + DummyDataset
├── train.py                    # Training loop
├── inference.py                # Patch-based prediction + stitching
├── metrics.py                  # PSNR, SSIM, mIoU
├── app.py                      # Streamlit demo
├── config.py                   # All hyperparameters (edit this!)
├── quick_test.py               # Sanity check (no GPU or data needed)
├── kaggle_setup.sh             # One-command dependency installer
├── kaggle_notebook.ipynb       # Ready-to-run Kaggle notebook
├── requirements.txt
└── README.md
```

---

## Setup

### Local / Linux / macOS

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify the pipeline (no GPU / real data needed)
cd satellite_sr
python quick_test.py
```

### Kaggle / Colab

```bash
bash kaggle_setup.sh
python quick_test.py
```

---

## Training

```bash
cd satellite_sr
python train.py
```

All hyperparameters live in `config.py` — edit that file, not `train.py`.

| Config key      | Default | Meaning                              |
|-----------------|---------|--------------------------------------|
| `NUM_BANDS`     | 4       | Sentinel-2 bands (B2, B3, B4, B8)   |
| `LR_SIZE`       | 256     | LR input patch size (px)             |
| `HR_SIZE`       | 1024    | HR target size (px)                  |
| `NUM_CLASSES`   | 6       | Segmentation classes                 |
| `BATCH_SIZE`    | 4       | Mini-batch size                      |
| `NUM_EPOCHS`    | 100     | Total training epochs                |
| `LEARNING_RATE` | 1e-4    | Adam learning rate                   |
| `SAVE_EVERY`    | 10      | Checkpoint interval (epochs)         |

If `data/hr/` does not exist, training automatically switches to **DummyDataset** (random tensors) so you can verify the pipeline immediately.

---

## Data

### Using DummyDataset (no real data needed)
Just run `python train.py` — if `data/hr/` is missing it falls back to DummyDataset automatically.

### Using Real Images

1. Place HR `.tif` patches in `data/hr/`
2. (Optional) Place segmentation labels as `.npy` files in `data/seg/` with matching filenames

### Recommended small dataset for quick testing

| Dataset | Size | Notes |
|---------|------|-------|
| **[DeepGlobe Land Cover](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset)** | ~3 GB | 6-class seg labels, RGB |
| **[EuroSAT](https://github.com/phelber/EuroSAT)** | ~90 MB | 13 Sentinel-2 bands, 64×64 patches |
| **[LoveDA](https://github.com/Junjue-Wang/LoveDA)** | ~1 GB | Urban/rural, semantic segmentation |

For a **minimal quick test without training**, the Kaggle [EuroSAT RGB](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset) dataset (< 100 MB, `.jpg`) works out of the box.

---

## Inference

```bash
python inference.py --input path/to/image.tif [--checkpoint checkpoints/best.pth]
```

Outputs (saved in the same directory as `--input` by default):
- `<name>_sr.tif`   — 4× super-resolved image
- `<name>_seg.png`  — colourised segmentation map

---

## Demo App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.  
Upload any `.tif`, `.png`, or `.jpg` — the app shows input, SR output, and segmentation side by side.

If no trained checkpoint is found it shows a clear warning instead of crashing.

---

## Segmentation Classes

| ID | Class       | Colour  |
|----|-------------|---------|
| 0  | Road        | Gray    |
| 1  | Building    | Red     |
| 2  | Water       | Blue    |
| 3  | Cropland    | Yellow  |
| 4  | Vegetation  | Green   |
| 5  | Other       | White   |

---

## Loss Function

```
total_loss = 1.0 × L1(sr_image, hr_target)
           + 0.1 × (1 − SSIM(sr_image, hr_target))
           + 1.0 × CrossEntropy(seg_logits, seg_labels)
```

---

## Swapping in HAT / SwinIR

The `SR_Backbone` class is designed to be swappable.  In `model.py`, replace:

```python
from models.sr_backbone import SR_Backbone
```

with your HAT/SwinIR module that has the same signature:

```python
forward(x) → (sr_image, features)
```

No other files need to change.

---

## License

MIT
