"""
metrics.py — PSNR, SSIM, and mIoU metric implementations.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


# ─── PSNR ────────────────────────────────────────────────────────────────────

def psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
) -> float:
    """
    Peak Signal-to-Noise Ratio (dB).

    Parameters
    ----------
    pred   : predicted image tensor (B, C, H, W) or (C, H, W)
    target : ground-truth image tensor, same shape
    max_val: maximum pixel value (1.0 for normalised floats)

    Returns
    -------
    Scalar PSNR value (average over batch).
    """
    with torch.no_grad():
        mse = F.mse_loss(pred.float(), target.float(), reduction="mean")
        if mse == 0:
            return float("inf")
        return 10.0 * torch.log10(torch.tensor(max_val ** 2) / mse).item()


# ─── SSIM ────────────────────────────────────────────────────────────────────

def _gaussian_kernel(kernel_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.outer(g)


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    kernel_size: int = 11,
    sigma: float = 1.5,
    C1: float = (0.01 ** 2),
    C2: float = (0.03 ** 2),
) -> torch.Tensor:
    """
    Structural Similarity Index (SSIM), averaged over batch & channels.

    Parameters
    ----------
    pred, target : (B, C, H, W) float tensors in [0, 1]

    Returns
    -------
    Scalar SSIM value (torch.Tensor, so it is differentiable and can be
    used directly in a loss: loss += 0.1 * (1 - ssim(sr, hr))
    """
    B, C, H, W = pred.shape
    kernel = _gaussian_kernel(kernel_size, sigma).to(pred.device)
    # Expand kernel for group-wise convolution over C channels
    kernel = kernel.expand(C, 1, kernel_size, kernel_size)
    pad = kernel_size // 2

    def _conv(img: torch.Tensor) -> torch.Tensor:
        return F.conv2d(img, kernel, padding=pad, groups=C)

    mu_x  = _conv(pred)
    mu_y  = _conv(target)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2  = _conv(pred * pred)   - mu_x2
    sigma_y2  = _conv(target * target) - mu_y2
    sigma_xy  = _conv(pred * target) - mu_xy

    numerator   = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)

    ssim_map = numerator / denominator.clamp_min(1e-8)
    return ssim_map.mean()


# ─── mIoU ────────────────────────────────────────────────────────────────────

class MeanIoU:
    """
    Streaming mean Intersection-over-Union for multi-class segmentation.

    Usage
    -----
    metric = MeanIoU(num_classes=6)
    for pred_logits, labels in loader:
        metric.update(pred_logits, labels)
    print(metric.compute())
    metric.reset()
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.long
        )

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Parameters
        ----------
        logits : (B, num_classes, H, W)
        labels : (B, H, W) integer class indices in [0, num_classes)
        """
        preds = logits.argmax(dim=1).view(-1)    # (B*H*W,)
        labels_flat = labels.view(-1)            # (B*H*W,)
        # Mask out invalid labels (e.g., 255 used for "ignore")
        mask = (labels_flat >= 0) & (labels_flat < self.num_classes)
        preds = preds[mask]
        labels_flat = labels_flat[mask]

        idx = labels_flat * self.num_classes + preds
        counts = torch.bincount(idx.cpu(), minlength=self.num_classes ** 2)
        self.confusion += counts.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict:
        """Returns {'miou': float, 'per_class_iou': list[float]}."""
        conf  = self.confusion.float()
        inter = conf.diagonal()
        union = conf.sum(0) + conf.sum(1) - inter
        iou_per_class = (inter / union.clamp_min(1)).tolist()
        miou = float(np.nanmean([v for v in iou_per_class if v > 0]))
        return {"miou": miou, "per_class_iou": iou_per_class}
