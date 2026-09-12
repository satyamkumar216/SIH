"""
models/sr_backbone.py
─────────────────────
MODULE 1 — SR_Backbone (simple CNN version).

Architecture is intentionally simple so the full pipeline can be
verified end-to-end quickly.  The class exposes exactly the same
interface that a heavier HAT/SwinIR backbone would use, so swapping
is a one-line change in model.py.

Inputs
------
x : (B, C, H, W)          low-resolution patch

Outputs
-------
sr_image  : (B, C, H*scale, W*scale)   reconstructed image
features  : (B, feat_ch, H, W)         intermediate feature map
                                        tapped BEFORE upsampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Simple residual block with two 3×3 convolutions."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class SR_Backbone(nn.Module):
    """
    Lightweight CNN super-resolution backbone.

    Parameters
    ----------
    in_channels  : number of input spectral bands (e.g. 4 for Sentinel-2)
    mid_channels : internal feature width
    feat_channels: channels of the intermediate feature map that is
                   returned for downstream segmentation (default 96)
    scale_factor : super-resolution upscale factor (default 4)
    num_blocks   : number of residual blocks
    """

    def __init__(
        self,
        in_channels: int = 4,
        mid_channels: int = 64,
        feat_channels: int = 96,
        scale_factor: int = 4,
        num_blocks: int = 8,
    ):
        super().__init__()
        self.scale_factor = scale_factor

        # ── Head: project input bands → mid-level feature space ──────────────
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        # ── Body: stack of residual blocks ────────────────────────────────────
        self.body = nn.Sequential(*[ResBlock(mid_channels) for _ in range(num_blocks)])

        # ── Transition to feat_channels (this is the "tap" output) ────────────
        self.feat_proj = nn.Conv2d(mid_channels, feat_channels, 1, bias=True)

        # ── Upsampler: sub-pixel convolution ──────────────────────────────────
        # For 4× we do two 2× pixel-shuffle stages.
        assert scale_factor in (2, 4, 8), "scale_factor must be 2, 4 or 8"
        upsampler_layers = []
        temp_ch = feat_channels
        remaining = scale_factor
        while remaining > 1:
            upsampler_layers += [
                nn.Conv2d(temp_ch, temp_ch * 4, 3, padding=1, bias=True),
                nn.PixelShuffle(2),        # ch→ch, spatial ×2
                nn.ReLU(inplace=True),
            ]
            remaining //= 2
        self.upsampler = nn.Sequential(*upsampler_layers)

        # ── Tail: project back to input channels ──────────────────────────────
        self.tail = nn.Conv2d(feat_channels, in_channels, 3, padding=1, bias=True)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (B, C, H, W)

        Returns
        -------
        sr_image : (B, C, H*scale, W*scale)
        features : (B, feat_channels, H, W)
        """
        head_out = self.head(x)                   # (B, mid, H, W)
        body_out = self.body(head_out)             # (B, mid, H, W)
        body_out = body_out + head_out             # global residual

        features = self.feat_proj(body_out)        # (B, feat_ch, H, W)  ← tap

        up = self.upsampler(features)              # (B, feat_ch, H*s, W*s)
        sr_image = self.tail(up)                   # (B, C,       H*s, W*s)

        return sr_image, features
