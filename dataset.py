"""
dataset.py — Data loading, degradation pipeline, and DummyDataset.

TiffDataset
-----------
Reads HR .tif patches with rasterio (falls back to PIL for png/jpg).
Synthesizes LR input via:
  1. Random Gaussian blur (sigma ∈ [0.5, 2.0])
  2. 4× downscale
  3. Gaussian noise  (std ∈ [0, 15])
  4. 4× upscale back to 256×256
Returns (lr_patch, hr_patch, seg_label_patch).
seg_label_patch is zeros when no label file is found.

DummyDataset
------------
Returns random tensors of the correct shapes — no files needed.
"""

import os
import math
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import rasterio  # type: ignore
    _RASTERIO = True
except ImportError:
    _RASTERIO = False
    warnings.warn("rasterio not installed; .tif files will be read with PIL.", stacklevel=2)

try:
    from PIL import Image
    _PIL = True
except ImportError:
    _PIL = False


def _gaussian_blur_numpy(img: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur channel-wise via separable filter (pure NumPy)."""
    from scipy.ndimage import gaussian_filter
    if img.ndim == 2:
        return gaussian_filter(img, sigma=sigma)
    return np.stack([gaussian_filter(img[c], sigma=sigma) for c in range(img.shape[0])])


try:
    from scipy.ndimage import gaussian_filter as _gf
    _SCIPY = True
except ImportError:
    _SCIPY = False


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Blur (C, H, W) float32 array."""
    if _SCIPY:
        return np.stack([_gf(img[c], sigma=sigma) for c in range(img.shape[0])])
    # Fallback: use cv2 if available
    try:
        import cv2
        ks = max(3, 2 * math.ceil(3 * sigma) + 1) | 1  # ensure odd
        return np.stack([cv2.GaussianBlur(img[c], (ks, ks), sigma) for c in range(img.shape[0])])
    except ImportError:
        # Last resort: no blur (just return as-is)
        return img


def _load_tif(path: str, num_bands: int) -> np.ndarray:
    """Load .tif file → (C, H, W) float32 in [0, 1]."""
    if _RASTERIO:
        with rasterio.open(path) as src:
            data = src.read()  # (bands, H, W)
    elif _PIL:
        img = Image.open(path)
        data = np.array(img)
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        else:
            data = data.transpose(2, 0, 1)
    else:
        raise RuntimeError("Neither rasterio nor PIL is available to read image files.")

    # Normalise to [0, 1]
    data = data.astype(np.float32)
    d_max = data.max()
    if d_max > 0:
        data /= d_max

    # Harmonise band count
    c = data.shape[0]
    if c >= num_bands:
        data = data[:num_bands]
    else:
        # Pad by repeating last band
        data = np.concatenate(
            [data, np.tile(data[-1:], (num_bands - c, 1, 1))], axis=0
        )
    return data  # (num_bands, H, W)


def _load_image(path: str, num_bands: int) -> np.ndarray:
    """Dispatch loader based on file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        try:
            return _load_tif(path, num_bands)
        except Exception:
            pass  # fall through to PIL
    if _PIL:
        img = Image.open(path).convert("RGB")
        data = np.array(img).astype(np.float32) / 255.0
        data = data.transpose(2, 0, 1)   # (3, H, W)
        c = data.shape[0]
        if c >= num_bands:
            data = data[:num_bands]
        else:
            data = np.concatenate([data, np.tile(data[-1:], (num_bands - c, 1, 1))], axis=0)
        return data
    raise RuntimeError(f"Cannot load image: {path}")


def _degrade(hr: np.ndarray, sigma: float, noise_std: float, scale: int = 4) -> np.ndarray:
    """
    Synthesise a low-resolution patch from a high-res one.

    Steps:
      1. Gaussian blur
      2. Downsample 4×
      3. Gaussian noise
      4. Upsample back to original size (bicubic via PIL or nearest)
    """
    _, H, W = hr.shape
    # 1. Blur
    blurred = _blur(hr, sigma)
    # 2. Downsample
    lh, lw = H // scale, W // scale
    # Use PIL for bicubic if available
    if _PIL:
        lr_list = []
        for c in range(blurred.shape[0]):
            ch = (blurred[c] * 255).clip(0, 255).astype(np.uint8)
            pil_ch = Image.fromarray(ch).resize((lw, lh), Image.BICUBIC)
            lr_list.append(np.array(pil_ch).astype(np.float32) / 255.0)
        lr = np.stack(lr_list)
    else:
        lr = blurred[:, ::scale, ::scale]
    # 3. Noise
    if noise_std > 0:
        noise = np.random.randn(*lr.shape).astype(np.float32) * (noise_std / 255.0)
        lr = (lr + noise).clip(0, 1)
    # 4. Upsample back to H×W
    if _PIL:
        up_list = []
        for c in range(lr.shape[0]):
            ch = (lr[c] * 255).clip(0, 255).astype(np.uint8)
            pil_ch = Image.fromarray(ch).resize((W, H), Image.BICUBIC)
            up_list.append(np.array(pil_ch).astype(np.float32) / 255.0)
        lr_up = np.stack(up_list)
    else:
        lr_up = np.repeat(np.repeat(lr, scale, axis=1), scale, axis=2)
    return lr_up  # (C, H, W)  same spatial size as hr


class TiffDataset(Dataset):
    """
    Dataset that reads high-resolution image patches and synthesises
    low-resolution inputs on the fly.

    Directory structure expected:
        hr_dir/   *.tif   (or *.png / *.jpg)
        seg_dir/  *.npy   (integer label arrays, optional)

    Parameters
    ----------
    hr_dir     : directory of HR image files
    seg_dir    : directory of segmentation label .npy files (optional)
    num_bands  : number of spectral bands to load
    patch_size : spatial size of HR patch (model output size)
    scale      : downscale factor for LR synthesis
    blur_sigma_range : (min, max) for random Gaussian blur sigma
    noise_std_max    : maximum Gaussian noise std (in 0-255 range)
    num_classes: used when creating dummy seg labels
    """

    def __init__(
        self,
        hr_dir: str,
        seg_dir: str = "",
        num_bands: int = 4,
        patch_size: int = 1024,
        scale: int = 4,
        blur_sigma_range: tuple = (0.5, 2.0),
        noise_std_max: float = 15.0,
        num_classes: int = 6,
    ):
        self.hr_dir    = hr_dir
        self.seg_dir   = seg_dir
        self.num_bands = num_bands
        self.patch_size = patch_size
        self.lr_size   = patch_size // scale
        self.scale     = scale
        self.blur_sigma_range = blur_sigma_range
        self.noise_std_max    = noise_std_max
        self.num_classes = num_classes

        exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
        self.hr_files = sorted(
            f for f in os.listdir(hr_dir)
            if os.path.splitext(f)[1].lower() in exts
        )
        if len(self.hr_files) == 0:
            raise RuntimeError(f"No image files found in {hr_dir}")

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx: int):
        fname = self.hr_files[idx]
        hr_path = os.path.join(self.hr_dir, fname)
        hr = _load_image(hr_path, self.num_bands)   # (C, H, W) float32

        # Crop / resize to patch_size × patch_size
        _, H, W = hr.shape
        if H != self.patch_size or W != self.patch_size:
            if _PIL:
                hr_list = []
                for c in range(hr.shape[0]):
                    ch = (hr[c] * 255).clip(0, 255).astype(np.uint8)
                    pil_ch = Image.fromarray(ch).resize(
                        (self.patch_size, self.patch_size), Image.BICUBIC
                    )
                    hr_list.append(np.array(pil_ch).astype(np.float32) / 255.0)
                hr = np.stack(hr_list)
            else:
                # Simple crop / pad
                hr = hr[:, :self.patch_size, :self.patch_size]

        # Synthesise LR input
        sigma     = np.random.uniform(*self.blur_sigma_range)
        noise_std = np.random.uniform(0, self.noise_std_max)
        lr = _degrade(hr, sigma=sigma, noise_std=noise_std, scale=self.scale)
        # lr is now (C, 1024, 1024) — resize to (C, 256, 256)
        if _PIL:
            lr_list = []
            for c in range(lr.shape[0]):
                ch = (lr[c] * 255).clip(0, 255).astype(np.uint8)
                pil_ch = Image.fromarray(ch).resize((self.lr_size, self.lr_size), Image.BICUBIC)
                lr_list.append(np.array(pil_ch).astype(np.float32) / 255.0)
            lr = np.stack(lr_list)
        else:
            step = self.scale
            lr = lr[:, ::step, ::step]

        # Load segmentation label if available
        base = os.path.splitext(fname)[0]
        seg_path = os.path.join(self.seg_dir, base + ".npy") if self.seg_dir else ""
        if seg_path and os.path.isfile(seg_path):
            seg = torch.from_numpy(np.load(seg_path).astype(np.int64))  # (H, W)
        else:
            seg = torch.zeros(self.patch_size, self.patch_size, dtype=torch.long)

        lr_t  = torch.from_numpy(lr)            # (C, 256, 256)
        hr_t  = torch.from_numpy(hr)            # (C, 1024, 1024)
        return lr_t, hr_t, seg


# ─── DummyDataset ─────────────────────────────────────────────────────────────

class DummyDataset(Dataset):
    """
    Returns random tensors with the correct shapes.
    Use this for testing the pipeline without any real files.

    Parameters
    ----------
    length      : number of samples to "pretend" to have
    num_bands   : spectral bands (C)
    lr_size     : LR input spatial size (H = W = lr_size)
    hr_size     : HR target spatial size (H = W = hr_size)
    num_classes : segmentation classes
    """

    def __init__(
        self,
        length: int = 32,
        num_bands: int = 4,
        lr_size: int = 256,
        hr_size: int = 1024,
        num_classes: int = 6,
    ):
        self.length      = length
        self.num_bands   = num_bands
        self.lr_size     = lr_size
        self.hr_size     = hr_size
        self.num_classes = num_classes

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        lr  = torch.rand(self.num_bands, self.lr_size, self.lr_size)
        hr  = torch.rand(self.num_bands, self.hr_size, self.hr_size)
        seg = torch.randint(0, self.num_classes, (self.hr_size, self.hr_size))
        return lr, hr, seg
