"""
inference.py — Patch-based inference with Gaussian-weighted stitching.

Usage:
    python inference.py --input path/to/image.tif [--checkpoint checkpoints/best.pth]

Outputs:
    <input>_sr.tif      — super-resolved image
    <input>_seg.png     — colourised segmentation map
"""

import os
import sys
import argparse
import math
import warnings
import numpy as np
import torch
import torch.nn.functional as F

import config as cfg
from model import build_model

# ── Colour map for segmentation ───────────────────────────────────────────────
# road=gray, building=red, water=blue, crop=yellow, vegetation=green, other=white
SEG_COLORS = np.array(
    [
        [128, 128, 128],  # road
        [255,   0,   0],  # building
        [  0,   0, 255],  # water
        [255, 255,   0],  # cropland
        [  0, 255,   0],  # vegetation
        [255, 255, 255],  # other
    ],
    dtype=np.uint8,
)

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import rasterio
    from rasterio.transform import from_bounds
    _RASTERIO = True
except ImportError:
    _RASTERIO = False

try:
    from PIL import Image
    _PIL = True
except ImportError:
    _PIL = False


# ─── Gaussian weight kernel ───────────────────────────────────────────────────

def _gaussian_kernel_2d(size: int, sigma: float) -> np.ndarray:
    ax  = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    k   = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return (k / k.max()).astype(np.float32)


# ─── Image I/O ────────────────────────────────────────────────────────────────

def load_image(path: str, num_bands: int = 4) -> np.ndarray:
    """Load image → (C, H, W) float32 [0, 1]."""
    ext = os.path.splitext(path)[1].lower()
    if _RASTERIO and ext in (".tif", ".tiff"):
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)
            profile = src.profile
        d_max = data.max()
        if d_max > 0:
            data /= d_max
        c = data.shape[0]
        if c >= num_bands:
            data = data[:num_bands]
        else:
            data = np.concatenate([data, np.tile(data[-1:], (num_bands - c, 1, 1))], axis=0)
        return data
    if _PIL:
        img  = Image.open(path).convert("RGB")
        data = np.array(img).astype(np.float32) / 255.0
        data = data.transpose(2, 0, 1)
        c = data.shape[0]
        if c >= num_bands:
            data = data[:num_bands]
        else:
            data = np.concatenate([data, np.tile(data[-1:], (num_bands - c, 1, 1))], axis=0)
        return data
    raise RuntimeError(f"Cannot load: {path}")


def save_tif(array: np.ndarray, path: str, ref_path: str = ""):
    """Save (C, H, W) float32 array as a GeoTiff (or PNG fallback)."""
    if _RASTERIO:
        c, h, w = array.shape
        kwargs = {
            "driver": "GTiff",
            "height": h,
            "width": w,
            "count": c,
            "dtype": "float32",
        }
        if ref_path and os.path.isfile(ref_path):
            try:
                with rasterio.open(ref_path) as ref:
                    kwargs.update(
                        crs=ref.crs,
                        transform=ref.transform,
                    )
            except Exception:
                pass
        with rasterio.open(path, "w", **kwargs) as dst:
            dst.write(array)
        print(f"Saved SR image → {path}")
    elif _PIL:
        png_path = os.path.splitext(path)[0] + "_sr.png"
        vis = (array[:3] * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
        Image.fromarray(vis).save(png_path)
        print(f"Saved SR image (PNG fallback) → {png_path}")
    else:
        warnings.warn("Cannot save output — install rasterio or PIL.")


def save_seg_png(seg_map: np.ndarray, path: str):
    """Colourize and save segmentation map as PNG."""
    rgb = SEG_COLORS[seg_map.clip(0, len(SEG_COLORS) - 1)]   # (H, W, 3)
    if _PIL:
        Image.fromarray(rgb).save(path)
        print(f"Saved seg map → {path}")
    else:
        warnings.warn("PIL not available — cannot save segmentation PNG.")


# ─── Patch-based inference ───────────────────────────────────────────────────

def infer_image(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    patch_size: int = 256,
    stride: int = 192,
    scale: int = 4,
    num_classes: int = 6,
    num_bands: int = 4,
) -> tuple:
    """
    Run the model on a large image using overlapping patches, then
    stitch results with Gaussian-weighted blending.

    Parameters
    ----------
    image : (C, H, W) float32 [0, 1]

    Returns
    -------
    sr_image : (C, H*scale, W*scale) float32
    seg_map  : (H*scale, W*scale)    int64 class indices
    """
    model.eval()
    C, H, W = image.shape

    # Pad image so it divides evenly into patches
    pad_h = (patch_size - (H - patch_size) % stride) % stride
    pad_w = (patch_size - (W - patch_size) % stride) % stride
    padded = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    _, pH, pW = padded.shape

    # Output accumulators (at scale resolution)
    sr_acc   = np.zeros((C,           pH * scale, pW * scale), dtype=np.float32)
    seg_acc  = np.zeros((num_classes, pH * scale, pW * scale), dtype=np.float32)
    wt_acc   = np.zeros((1,           pH * scale, pW * scale), dtype=np.float32)

    kernel_lr = _gaussian_kernel_2d(patch_size, sigma=patch_size / 6.0)
    kernel_hr = _gaussian_kernel_2d(patch_size * scale, sigma=patch_size * scale / 6.0)

    # Collect patch positions
    ys = list(range(0, pH - patch_size + 1, stride))
    xs = list(range(0, pW - patch_size + 1, stride))

    with torch.no_grad():
        for y in ys:
            for x in xs:
                patch = padded[:, y:y + patch_size, x:x + patch_size]
                t = torch.from_numpy(patch).unsqueeze(0).to(device)  # (1, C, ps, ps)
                sr_patch, seg_patch = model(t)

                sr_np  = sr_patch[0].cpu().numpy()      # (C, ps*s, ps*s)
                seg_np = seg_patch[0].cpu().numpy()     # (nc, ps*s, ps*s)

                hy, hx = y * scale, x * scale
                hps = patch_size * scale

                sr_acc[:, hy:hy + hps, hx:hx + hps]  += sr_np  * kernel_hr[np.newaxis]
                seg_acc[:, hy:hy + hps, hx:hx + hps] += seg_np * kernel_hr[np.newaxis]
                wt_acc[:, hy:hy + hps, hx:hx + hps]  += kernel_hr[np.newaxis]

    wt_acc = wt_acc.clip(min=1e-6)
    sr_out  = (sr_acc  / wt_acc)[:, :H * scale, :W * scale]
    seg_out = (seg_acc / wt_acc)[:, :H * scale, :W * scale]
    seg_map = seg_out.argmax(axis=0).astype(np.int64)

    return sr_out, seg_map


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Satellite SR+Seg inference")
    parser.add_argument("--input",      required=True,  help="Input image path (.tif/.png/.jpg)")
    parser.add_argument("--checkpoint", default=cfg.DEMO_CHECKPOINT, help="Model checkpoint .pth")
    parser.add_argument("--out_dir",    default=".",    help="Output directory")
    parser.add_argument("--patch",      type=int, default=cfg.PATCH_SIZE,   help="Patch size")
    parser.add_argument("--stride",     type=int, default=cfg.PATCH_STRIDE, help="Patch stride")
    args = parser.parse_args()

    # Device
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
        else torch.device("cpu")
    )
    print(f"[device] {device}")

    # Model
    model = build_model(cfg).to(device)
    if os.path.isfile(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state)
        print(f"[ckpt] Loaded {args.checkpoint}")
    else:
        print(f"[warn] No checkpoint found at {args.checkpoint}; using random weights.")

    # Load image
    image = load_image(args.input, num_bands=cfg.NUM_BANDS)
    print(f"[input] {image.shape}  ({args.input})")

    # Infer
    sr_image, seg_map = infer_image(
        model, image, device,
        patch_size=args.patch,
        stride=args.stride,
        scale=cfg.SCALE_FACTOR,
        num_classes=cfg.NUM_CLASSES,
        num_bands=cfg.NUM_BANDS,
    )
    print(f"[output] SR {sr_image.shape}, seg {seg_map.shape}")

    # Save
    base = os.path.splitext(os.path.basename(args.input))[0]
    sr_path  = os.path.join(args.out_dir, base + "_sr.tif")
    seg_path = os.path.join(args.out_dir, base + "_seg.png")
    os.makedirs(args.out_dir, exist_ok=True)
    save_tif(sr_image.clip(0, 1), sr_path, ref_path=args.input)
    save_seg_png(seg_map, seg_path)


if __name__ == "__main__":
    main()
