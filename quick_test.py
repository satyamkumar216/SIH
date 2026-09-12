"""
quick_test.py — End-to-end pipeline sanity check.

Runs the FULL model pipeline using DummyDataset + random tensors.
No GPU or real data required.

Checks:
  1. Model instantiation
  2. Forward pass shapes (SR image, seg logits)
  3. Loss computation
  4. PSNR, SSIM, mIoU metrics
  5. UNet_Decoder standalone shape
  6. Segmentation_Upsampler standalone shape

Run with:
    python quick_test.py
"""

import sys
import traceback
import torch
import torch.nn.functional as F

import config as cfg
from model import build_model, SatelliteSRSeg
from models.sr_backbone import SR_Backbone
from models.unet_decoder import UNet_Decoder
from models.wavkan_head import WavKAN_Head
from models.segmentation_upsampler import Segmentation_Upsampler
from dataset import DummyDataset
from metrics import psnr, ssim, MeanIoU

PASS = "✅"
FAIL = "❌"
SEP  = "─" * 60


def check(cond: bool, msg: str):
    if cond:
        print(f"  {PASS} {msg}")
    else:
        print(f"  {FAIL} {msg}")
        return False
    return True


def section(title: str):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def main():
    all_ok = True
    device = torch.device("cpu")   # CPU only for quick test

    # ── 1. Config sanity ─────────────────────────────────────────────────────
    section("1 — Config sanity")
    all_ok &= check(cfg.NUM_BANDS    == 4,    f"NUM_BANDS = {cfg.NUM_BANDS}")
    all_ok &= check(cfg.LR_SIZE      == 256,  f"LR_SIZE   = {cfg.LR_SIZE}")
    all_ok &= check(cfg.HR_SIZE      == 1024, f"HR_SIZE   = {cfg.HR_SIZE}")
    all_ok &= check(cfg.SCALE_FACTOR == 4,    f"SCALE     = {cfg.SCALE_FACTOR}")
    all_ok &= check(cfg.NUM_CLASSES  == 6,    f"CLASSES   = {cfg.NUM_CLASSES}")

    # ── 2. Module shapes ─────────────────────────────────────────────────────
    section("2 — Individual module forward shapes")

    B, C, H, W = 2, cfg.NUM_BANDS, cfg.LR_SIZE, cfg.LR_SIZE
    x = torch.randn(B, C, H, W)

    # SR backbone
    bb = SR_Backbone(
        in_channels=C,
        mid_channels=cfg.BACKBONE_MID_CH,
        feat_channels=cfg.SR_FEAT_CHANNELS,
        scale_factor=cfg.SCALE_FACTOR,
    )
    sr, feats = bb(x)
    print(f"  SR_Backbone input  : {list(x.shape)}")
    print(f"  SR_Backbone sr_out : {list(sr.shape)}")
    print(f"  SR_Backbone feats  : {list(feats.shape)}")
    all_ok &= check(
        list(sr.shape) == [B, C, H * cfg.SCALE_FACTOR, W * cfg.SCALE_FACTOR],
        f"SR output shape == [{B}, {C}, {H*4}, {W*4}]"
    )
    all_ok &= check(
        list(feats.shape) == [B, cfg.SR_FEAT_CHANNELS, H, W],
        f"Feature tap shape == [{B}, {cfg.SR_FEAT_CHANNELS}, {H}, {W}]"
    )

    # UNet decoder
    ud = UNet_Decoder(in_channels=cfg.SR_FEAT_CHANNELS, out_channels=cfg.UNET_OUT_CHANNELS)
    unet_out = ud(feats)
    print(f"  UNet_Decoder input : {list(feats.shape)}")
    print(f"  UNet_Decoder output: {list(unet_out.shape)}")
    all_ok &= check(
        list(unet_out.shape) == [B, cfg.UNET_OUT_CHANNELS, H, W],
        f"UNet output shape == [{B}, {cfg.UNET_OUT_CHANNELS}, {H}, {W}]"
    )

    # WavKAN head
    head = WavKAN_Head(in_features=cfg.UNET_OUT_CHANNELS, num_classes=cfg.NUM_CLASSES)
    print(f"  WavKAN_Head using KAN: {head.using_kan}")
    logits_256 = head(unet_out)
    print(f"  WavKAN_Head input  : {list(unet_out.shape)}")
    print(f"  WavKAN_Head output : {list(logits_256.shape)}")
    all_ok &= check(
        list(logits_256.shape) == [B, cfg.NUM_CLASSES, H, W],
        f"WavKAN output shape == [{B}, {cfg.NUM_CLASSES}, {H}, {W}]"
    )

    # Segmentation upsampler
    upsampler = Segmentation_Upsampler(num_classes=cfg.NUM_CLASSES, upscale=cfg.SCALE_FACTOR)
    logits_hr = upsampler(logits_256)
    print(f"  Seg_Upsampler input : {list(logits_256.shape)}")
    print(f"  Seg_Upsampler output: {list(logits_hr.shape)}")
    all_ok &= check(
        list(logits_hr.shape) == [B, cfg.NUM_CLASSES, H * 4, W * 4],
        f"Seg upsampler output shape == [{B}, {cfg.NUM_CLASSES}, {H*4}, {W*4}]"
    )

    # ── 3. Full model forward ─────────────────────────────────────────────────
    section("3 — Full model forward pass")
    model = build_model(cfg).to(device)
    x_lr  = torch.randn(B, C, H, W)
    sr_out, seg_out = model(x_lr)
    print(f"  Input              : {list(x_lr.shape)}")
    print(f"  SR output          : {list(sr_out.shape)}")
    print(f"  Seg logits         : {list(seg_out.shape)}")
    all_ok &= check(
        list(sr_out.shape)  == [B, C, cfg.HR_SIZE, cfg.HR_SIZE],
        f"SR output  shape == [{B}, {C}, {cfg.HR_SIZE}, {cfg.HR_SIZE}]"
    )
    all_ok &= check(
        list(seg_out.shape) == [B, cfg.NUM_CLASSES, cfg.HR_SIZE, cfg.HR_SIZE],
        f"Seg logits shape == [{B}, {cfg.NUM_CLASSES}, {cfg.HR_SIZE}, {cfg.HR_SIZE}]"
    )

    # ── 4. Loss computation ───────────────────────────────────────────────────
    section("4 — Loss computation")
    hr_target  = torch.rand(B, C, cfg.HR_SIZE, cfg.HR_SIZE)
    seg_labels = torch.randint(0, cfg.NUM_CLASSES, (B, cfg.HR_SIZE, cfg.HR_SIZE))

    l1_v    = F.l1_loss(sr_out, hr_target)
    ssim_v  = ssim(sr_out.clamp(0, 1), hr_target.clamp(0, 1))
    ce_v    = F.cross_entropy(seg_out, seg_labels)
    total   = cfg.W_L1 * l1_v + cfg.W_SSIM * (1 - ssim_v) + cfg.W_CE * ce_v

    print(f"  L1 loss      : {l1_v.item():.4f}")
    print(f"  SSIM value   : {ssim_v.item():.4f}")
    print(f"  CE loss      : {ce_v.item():.4f}")
    print(f"  Total loss   : {total.item():.4f}")

    all_ok &= check(torch.isfinite(total), "Total loss is finite")
    all_ok &= check(total.requires_grad,   "Total loss is differentiable")

    # ── 5. Backward pass ─────────────────────────────────────────────────────
    section("5 — Backward pass")
    total.backward()
    params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"  Parameters with gradients: {params_with_grad}")
    all_ok &= check(params_with_grad > 0, "Gradients flow through model")

    # ── 6. Metrics ───────────────────────────────────────────────────────────
    section("6 — Metrics")
    psnr_val = psnr(sr_out.detach(), hr_target)
    ssim_val = ssim(sr_out.detach().clamp(0, 1), hr_target.clamp(0, 1))
    miou_metric = MeanIoU(num_classes=cfg.NUM_CLASSES)
    miou_metric.update(seg_out.detach().cpu(), seg_labels.cpu())
    iou = miou_metric.compute()
    print(f"  PSNR  : {psnr_val:.2f} dB")
    print(f"  SSIM  : {ssim_val.item():.4f}")
    print(f"  mIoU  : {iou['miou']:.4f}")
    all_ok &= check(isinstance(psnr_val, float),    "PSNR returns float")
    all_ok &= check(0 <= ssim_val.item() <= 1,      "SSIM in [0, 1]")
    all_ok &= check(0 <= iou["miou"] <= 1,          "mIoU in [0, 1]")

    # ── 7. DummyDataset ──────────────────────────────────────────────────────
    section("7 — DummyDataset")
    from torch.utils.data import DataLoader
    ds = DummyDataset(
        length=4,
        num_bands=cfg.NUM_BANDS,
        lr_size=cfg.LR_SIZE,
        hr_size=cfg.HR_SIZE,
        num_classes=cfg.NUM_CLASSES,
    )
    loader = DataLoader(ds, batch_size=2)
    lr_b, hr_b, seg_b = next(iter(loader))
    print(f"  DummyDataset lr  : {list(lr_b.shape)}")
    print(f"  DummyDataset hr  : {list(hr_b.shape)}")
    print(f"  DummyDataset seg : {list(seg_b.shape)}")
    all_ok &= check(list(lr_b.shape)  == [2, cfg.NUM_BANDS, cfg.LR_SIZE,  cfg.LR_SIZE],  "LR batch shape")
    all_ok &= check(list(hr_b.shape)  == [2, cfg.NUM_BANDS, cfg.HR_SIZE,  cfg.HR_SIZE],  "HR batch shape")
    all_ok &= check(list(seg_b.shape) == [2, cfg.HR_SIZE, cfg.HR_SIZE],                   "Seg batch shape")

    # ── 8. Parameter count ───────────────────────────────────────────────────
    section("8 — Parameter count")
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params     : {total_params:,}")
    print(f"  Trainable params : {train_params:,}")

    # ── Summary ──────────────────────────────────────────────────────────────
    section("SUMMARY")
    if all_ok:
        print(f"  {PASS} All checks passed!  Pipeline is ready.\n")
    else:
        print(f"  {FAIL} Some checks FAILED — see above.\n")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
