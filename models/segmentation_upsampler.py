"""
models/segmentation_upsampler.py
─────────────────────────────────
MODULE 4 — Segmentation_Upsampler.

Learned 4× upsampling of the segmentation logit map from
(B, num_classes, 256, 256) → (B, num_classes, 1024, 1024).

Uses a single Conv2d + PixelShuffle (sub-pixel convolution) for
fully learnable, trainable upsampling — no bilinear/F.interpolate.

Layer breakdown
───────────────
Conv2d(num_classes → num_classes * 16, 3×3, padding=1)
    ↓  expands channels to hold 4×4 = 16 sub-pixels per output pixel
PixelShuffle(upscale_factor=4)
    ↓  rearranges to (B, num_classes, H*4, W*4)
Conv2d(num_classes → num_classes, 3×3, padding=1)   (refinement)
    ↓  final learned refinement at full resolution
"""

import torch
import torch.nn as nn


class Segmentation_Upsampler(nn.Module):
    """
    Trainable 4× upsampler for class logits.

    Parameters
    ----------
    num_classes  : number of segmentation classes (in AND out channels)
    upscale      : upscale factor (must be 4)
    """

    def __init__(self, num_classes: int = 6, upscale: int = 4):
        super().__init__()
        assert upscale == 4, "Segmentation_Upsampler currently only supports upscale=4"

        # Expand channels for pixel-shuffle: need num_classes * upscale² channels
        expand_ch = num_classes * (upscale ** 2)   # num_classes * 16

        self.expand_conv = nn.Conv2d(
            num_classes, expand_ch, kernel_size=3, padding=1, bias=True
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=upscale)
        # After pixel-shuffle: (B, num_classes, H*4, W*4)

        # Lightweight refinement convolution at full resolution
        self.refine = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_classes, num_classes, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x      : (B, num_classes, H,    W)
        returns: (B, num_classes, H*4, W*4)
        """
        x = self.expand_conv(x)     # (B, num_classes*16, H, W)
        x = self.pixel_shuffle(x)   # (B, num_classes,    H*4, W*4)
        x = self.refine(x)          # (B, num_classes,    H*4, W*4)
        return x
