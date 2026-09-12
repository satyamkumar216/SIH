"""
models/unet_decoder.py
───────────────────────
MODULE 2 — UNet_Decoder (UPDATED).

Takes the intermediate feature map from the SR backbone and produces
dense per-pixel feature vectors **at the original LR resolution**.
Resolution stays at 256 throughout; decoder output is (B, 64, 256, 256).

Architecture
────────────
Encoder (downsampling):
  (B, 96, 256, 256)  →  skip1: (B, 128, 256, 256)
                      →  skip2: (B, 256, 128, 128)
                      →  skip3: (B, 512,  64,  64)
  bottleneck         →        (B, 512,  32,  32)

Decoder (upsampling with bilinear + skip concat):
  (B, 512, 32, 32)   → (B, 512,  64,  64)  [+skip3 → 1024 ch → 256 ch]
                      → (B, 256, 128, 128)  [+skip2 →  512 ch → 128 ch]
                      → (B, 128, 256, 256)  [+skip1 →  256 ch →  64 ch]
  output             → (B,  64, 256, 256)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet_Decoder(nn.Module):
    """
    U-Net style encoder-decoder operating on (B, in_channels, 256, 256).

    Parameters
    ----------
    in_channels  : channels of the input feature map (96 from SR backbone)
    out_channels : channels of the output feature map (64)
    """

    def __init__(self, in_channels: int = 96, out_channels: int = 64):
        super().__init__()

        # ── Encoder ─────────────────────────────────────────────────────────
        # Level 0 — no downsampling, produces skip1
        self.enc0 = _conv_bn_relu(in_channels, 128)   # (B, 128, 256, 256)

        # Level 1 — 2× downsample, produces skip2
        self.pool1 = nn.MaxPool2d(2)
        self.enc1 = _conv_bn_relu(128, 256)            # (B, 256, 128, 128)

        # Level 2 — 2× downsample, produces skip3
        self.pool2 = nn.MaxPool2d(2)
        self.enc2 = _conv_bn_relu(256, 512)            # (B, 512,  64,  64)

        # Level 3 — 2× downsample, bottleneck
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = _conv_bn_relu(512, 512)      # (B, 512,  32,  32)

        # ── Decoder ─────────────────────────────────────────────────────────
        # Up3: 32→64, concat skip3 (512) → 1024 → 256
        self.up3    = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3   = _conv_bn_relu(512 + 512, 256)   # (B, 256, 64, 64)

        # Up2: 64→128, concat skip2 (256) → 512 → 128
        self.up2    = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2   = _conv_bn_relu(256 + 256, 128)   # (B, 128, 128, 128)

        # Up1: 128→256, concat skip1 (128) → 256 → 64
        self.up1    = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1   = _conv_bn_relu(128 + 128, out_channels)  # (B, 64, 256, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, in_channels, 256, 256)

        Returns
        -------
        out : (B, out_channels, 256, 256)
        """
        # Encoder
        skip1 = self.enc0(x)                          # (B, 128, 256, 256)
        skip2 = self.enc1(self.pool1(skip1))           # (B, 256, 128, 128)
        skip3 = self.enc2(self.pool2(skip2))           # (B, 512,  64,  64)
        bot   = self.bottleneck(self.pool3(skip3))     # (B, 512,  32,  32)

        # Decoder
        d3 = self.dec3(torch.cat([self.up3(bot), skip3], dim=1))   # (B, 256, 64, 64)
        d2 = self.dec2(torch.cat([self.up2(d3),  skip2], dim=1))   # (B, 128,128,128)
        d1 = self.dec1(torch.cat([self.up1(d2),  skip1], dim=1))   # (B, 64, 256,256)

        return d1
