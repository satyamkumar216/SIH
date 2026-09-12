"""
model.py — Full satellite SR + segmentation model.

Combines:
  Module 1: SR_Backbone       (B, C, 256, 256) → sr_image + features
  Module 2: UNet_Decoder      features (B, 96, 256, 256) → (B, 64, 256, 256)
  Module 3: WavKAN_Head       (B, 64, 256, 256) → (B, num_classes, 256, 256)
  Module 4: Segmentation_Upsampler  (B, num_classes, 256, 256)
                                   → (B, num_classes, 1024, 1024)

Forward pass:
  sr_image, features  = sr_backbone(x)
  unet_feats          = unet_decoder(features)
  seg_logits_256      = wavkan_head(unet_feats)
  seg_logits          = seg_upsampler(seg_logits_256)
  return sr_image, seg_logits
"""

import torch
import torch.nn as nn

from models.sr_backbone import SR_Backbone
from models.unet_decoder import UNet_Decoder
from models.wavkan_head import WavKAN_Head
from models.segmentation_upsampler import Segmentation_Upsampler


class SatelliteSRSeg(nn.Module):
    """
    Joint super-resolution and land-cover segmentation model.

    Parameters
    ----------
    num_bands      : spectral bands in the input patch
    num_classes    : segmentation classes
    scale_factor   : SR upscale factor (4 → 256→1024)
    sr_feat_ch     : intermediate feature channels from SR backbone (96)
    unet_out_ch    : UNet decoder output channels (64)
    sr_mid_ch      : internal channels of the SR backbone CNN (64)
    sr_num_blocks  : number of residual blocks in the SR backbone (8)
    kan_grid_size  : KAN grid size (ignored if KAN not available)
    """

    def __init__(
        self,
        num_bands: int = 4,
        num_classes: int = 6,
        scale_factor: int = 4,
        sr_feat_ch: int = 96,
        unet_out_ch: int = 64,
        sr_mid_ch: int = 64,
        sr_num_blocks: int = 8,
        kan_grid_size: int = 5,
    ):
        super().__init__()
        self.sr_backbone = SR_Backbone(
            in_channels=num_bands,
            mid_channels=sr_mid_ch,
            feat_channels=sr_feat_ch,
            scale_factor=scale_factor,
            num_blocks=sr_num_blocks,
        )
        self.unet_decoder = UNet_Decoder(
            in_channels=sr_feat_ch,
            out_channels=unet_out_ch,
        )
        self.wavkan_head = WavKAN_Head(
            in_features=unet_out_ch,
            num_classes=num_classes,
            grid_size=kan_grid_size,
        )
        self.seg_upsampler = Segmentation_Upsampler(
            num_classes=num_classes,
            upscale=scale_factor,
        )

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (B, num_bands, 256, 256)   LR input patch

        Returns
        -------
        sr_image   : (B, num_bands, 1024, 1024)
        seg_logits : (B, num_classes, 1024, 1024)
        """
        sr_image, features = self.sr_backbone(x)          # SR + feature tap
        unet_feats          = self.unet_decoder(features)  # dense seg features
        seg_logits_256      = self.wavkan_head(unet_feats) # logits @ 256
        seg_logits          = self.seg_upsampler(seg_logits_256)  # logits @ 1024
        return sr_image, seg_logits


def build_model(cfg) -> SatelliteSRSeg:
    """Convenience factory that reads from a config module/namespace."""
    return SatelliteSRSeg(
        num_bands=cfg.NUM_BANDS,
        num_classes=cfg.NUM_CLASSES,
        scale_factor=cfg.SCALE_FACTOR,
        sr_feat_ch=cfg.SR_FEAT_CHANNELS,
        unet_out_ch=cfg.UNET_OUT_CHANNELS,
        sr_mid_ch=cfg.BACKBONE_MID_CH,
    )
