# ── Kaggle / Colab path bootstrap ────────────────────────────────────────────
# Ensures the repo root is on sys.path when running from a notebook cell.
# Has no effect when running locally (chdir to the same dir is a no-op).
import os, sys
os.chdir("/kaggle/working/SIH")
sys.path.insert(0, "/kaggle/working/SIH")
# ─────────────────────────────────────────────────────────────────────────────

"""
train.py — Training loop for SatelliteSRSeg.

All hyperparameters are read from config.py.
Supports GPU (CUDA / MPS) with automatic CPU fallback.

Loss function:
  total = W_L1 × L1(sr, hr)
        + W_SSIM × (1 − SSIM(sr, hr))
        + W_CE × CrossEntropy(seg, labels)
        + perc_weight(step) × VGG_Perceptual(sr[:3], hr[:3])

Perceptual loss weight is linearly ramped from 0 → 0.01 over
PERCEPTUAL_RAMP_STEPS optimizer steps (set to 0 to disable).
"""


import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── local imports ─────────────────────────────────────────────────────────────
import config as cfg
from model import build_model
from dataset import TiffDataset, FolderDataset, DummyDataset
from metrics import psnr, ssim, MeanIoU


# ─── Device selection ─────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[device] CUDA — {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("[device] Apple MPS")
    else:
        dev = torch.device("cpu")
        print("[device] CPU (no GPU found)")
    return dev


# ─── VGG Perceptual Loss ──────────────────────────────────────────────────────

class VGGPerceptualLoss(nn.Module):
    """
    VGG16 feature-level loss using ReLU3_3 activations.

    Only the FIRST 3 BANDS are used (B, G, R ~ Sentinel bands 2,3,4)
    because VGG was trained on 3-channel RGB — Band 8 (NIR) is ignored.

    The ramp schedule starts weight at 0 and linearly increases to 0.01
    over PERCEPTUAL_RAMP_STEPS optimizer steps.  Starting at 0 lets the
    model first learn the broad pixel-level structure via L1/SSIM before
    being pushed toward high-frequency texture by the perceptual loss.
    """

    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        try:
            vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        except Exception:
            # Older torchvision fallback
            vgg = vgg16(pretrained=True)
        # features[0:16] = up to relu3_3
        self.feature_extractor = nn.Sequential(
            *list(vgg.features.children())[:16]
        )
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target : (B, C, H, W) in [0, 1], only first 3 channels used
        """
        pred_rgb   = pred[:, :3].clamp(0, 1)
        target_rgb = target[:, :3].clamp(0, 1)
        return F.l1_loss(
            self.feature_extractor(pred_rgb),
            self.feature_extractor(target_rgb),
        )


def _perc_weight(global_step: int) -> float:
    """Linear ramp: 0 → 0.01 over PERCEPTUAL_RAMP_STEPS steps."""
    if cfg.PERCEPTUAL_RAMP_STEPS <= 0:
        return 0.0
    return min(0.01, 0.01 * (global_step / cfg.PERCEPTUAL_RAMP_STEPS))


# ─── Loss function ────────────────────────────────────────────────────────────

def compute_loss(
    sr_image: torch.Tensor,
    hr_target: torch.Tensor,
    seg_logits: torch.Tensor,
    seg_labels: torch.Tensor,
    perc_loss_fn,
    perc_weight: float,
) -> tuple:
    """
    Returns (total_loss, l1_val, ssim_val, ce_val, perc_val).
    """
    l1_v   = F.l1_loss(sr_image, hr_target)
    ssim_v = ssim(sr_image.clamp(0, 1), hr_target.clamp(0, 1))
    ce_v   = F.cross_entropy(seg_logits, seg_labels)

    if perc_weight > 0 and perc_loss_fn is not None:
        perc_v = perc_loss_fn(sr_image, hr_target)
    else:
        perc_v = torch.zeros(1, device=sr_image.device)

    total = (
        cfg.W_L1   * l1_v
        + cfg.W_SSIM * (1.0 - ssim_v)
        + cfg.W_CE   * ce_v
        + perc_weight * perc_v
    )
    return total, l1_v.item(), ssim_v.item(), ce_v.item(), perc_v.item()


# ─── Dataset / DataLoader ─────────────────────────────────────────────────────

def build_loaders(use_dummy: bool = None):
    # use_dummy=None → read from config; explicit True/False overrides
    dataset_type = getattr(cfg, "DATASET_TYPE", "tiff").lower()

    if dataset_type == "dummy":
        print("[dataset] DummyDataset — random tensors, no files.")
        train_ds = DummyDataset(
            length=64,
            num_bands=cfg.NUM_BANDS,
            lr_size=cfg.LR_SIZE,
            hr_size=cfg.HR_SIZE,
            num_classes=cfg.NUM_CLASSES,
        )
        val_ds = DummyDataset(
            length=16,
            num_bands=cfg.NUM_BANDS,
            lr_size=cfg.LR_SIZE,
            hr_size=cfg.HR_SIZE,
            num_classes=cfg.NUM_CLASSES,
        )

    elif dataset_type == "folder":
        print(f"[dataset] FolderDataset from {cfg.HR_DIR}")
        full_ds = FolderDataset(
            hr_dir=cfg.HR_DIR,
            num_bands=cfg.NUM_BANDS,
            patch_size=cfg.HR_SIZE,
            scale=cfg.SCALE_FACTOR,
            blur_sigma_range=(cfg.BLUR_SIGMA_MIN, cfg.BLUR_SIGMA_MAX),
            noise_std_max=cfg.NOISE_STD_MAX,
        )
        # 90/10 train/val split
        n_val    = max(1, int(0.1 * len(full_ds)))
        n_train  = len(full_ds) - n_val
        train_ds, val_ds = torch.utils.data.random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

    else:  # "tiff"
        print(f"[dataset] TiffDataset from {cfg.HR_DIR}")
        train_ds = TiffDataset(
            hr_dir=cfg.HR_DIR,
            seg_dir=cfg.SEG_DIR,
            num_bands=cfg.NUM_BANDS,
            patch_size=cfg.HR_SIZE,
            scale=cfg.SCALE_FACTOR,
            blur_sigma_range=(cfg.BLUR_SIGMA_MIN, cfg.BLUR_SIGMA_MAX),
            noise_std_max=cfg.NOISE_STD_MAX,
            num_classes=cfg.NUM_CLASSES,
        )
        val_ds = train_ds   # replace with a separate val split when available

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader, val_loader


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch: int, metrics: dict, tag: str = ""):
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    name = f"epoch_{epoch:04d}{tag}.pth"
    path = os.path.join(cfg.CHECKPOINT_DIR, name)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )
    print(f"  ✓ checkpoint saved → {path}")
    return path


def load_checkpoint(model, optimizer, path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    start_step = ckpt.get("global_step", 0)
    print(f"[ckpt] Resumed from epoch {ckpt['epoch']}  →  {path}")
    return ckpt["epoch"], start_step


# ─── Training / Validation step ──────────────────────────────────────────────

def train_one_epoch(
    model, loader, optimizer, device,
    perc_loss_fn, global_step: int,
) -> tuple:
    model.train()
    total_sum = l1_sum = ssim_sum = ce_sum = perc_sum = psnr_sum = 0.0
    n = 0

    for lr, hr, seg in loader:
        lr  = lr.to(device, non_blocking=True)
        hr  = hr.to(device, non_blocking=True)
        seg = seg.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        sr_image, seg_logits = model(lr)

        pw = _perc_weight(global_step)
        loss, l1, ssim_v, ce, perc_v = compute_loss(
            sr_image, hr, seg_logits, seg, perc_loss_fn, pw
        )
        loss.backward()
        optimizer.step()
        global_step += 1

        bs = lr.size(0)
        total_sum += loss.item() * bs
        l1_sum    += l1 * bs
        ssim_sum  += ssim_v * bs
        ce_sum    += ce * bs
        perc_sum  += perc_v * bs
        psnr_sum  += psnr(sr_image.detach(), hr.detach()) * bs
        n         += bs

    metrics = {
        "loss": total_sum / n,
        "l1":   l1_sum   / n,
        "ssim": ssim_sum / n,
        "ce":   ce_sum   / n,
        "perc": perc_sum / n,
        "psnr": psnr_sum / n,
    }
    return metrics, global_step


@torch.no_grad()
def validate(model, loader, device, perc_loss_fn, global_step: int) -> dict:
    model.eval()
    miou_metric = MeanIoU(num_classes=cfg.NUM_CLASSES)
    total_sum = psnr_sum = ssim_sum = 0.0
    n = 0

    for lr, hr, seg in loader:
        lr  = lr.to(device, non_blocking=True)
        hr  = hr.to(device, non_blocking=True)
        seg = seg.to(device, non_blocking=True)

        sr_image, seg_logits = model(lr)
        pw = _perc_weight(global_step)
        loss, _, _, _, _ = compute_loss(sr_image, hr, seg_logits, seg, perc_loss_fn, pw)
        miou_metric.update(seg_logits.cpu(), seg.cpu())

        bs = lr.size(0)
        total_sum += loss.item() * bs
        psnr_sum  += psnr(sr_image, hr) * bs
        ssim_sum  += ssim(sr_image.clamp(0, 1), hr.clamp(0, 1)).item() * bs
        n         += bs

    iou_res = miou_metric.compute()
    return {
        "loss": total_sum / n,
        "psnr": psnr_sum  / n,
        "ssim": ssim_sum  / n,
        "miou": iou_res["miou"],
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = get_device()
    model  = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    # Optional: build perceptual loss (skipped if ramp steps = 0)
    perc_loss_fn = None
    if cfg.PERCEPTUAL_RAMP_STEPS > 0:
        try:
            perc_loss_fn = VGGPerceptualLoss().to(device)
            print("[loss] Perceptual (VGG) loss enabled — ramp over "
                  f"{cfg.PERCEPTUAL_RAMP_STEPS} steps.")
        except Exception as e:
            print(f"[warn] Could not load VGG perceptual loss ({e}); skipping.")

    # Optional: resume from checkpoint
    start_epoch  = 0
    global_step  = 0
    resume_ckpt  = os.path.join(cfg.CHECKPOINT_DIR, "latest.pth")
    if os.path.isfile(resume_ckpt):
        start_epoch, global_step = load_checkpoint(
            model, optimizer, resume_ckpt, device
        )

    train_loader, val_loader = build_loaders()
    best_val_psnr = 0.0

    for epoch in range(start_epoch + 1, cfg.NUM_EPOCHS + 1):
        t0 = time.time()
        train_m, global_step = train_one_epoch(
            model, train_loader, optimizer, device,
            perc_loss_fn, global_step,
        )
        dt = time.time() - t0

        if epoch % cfg.LOG_EVERY == 0:
            val_m = validate(model, val_loader, device, perc_loss_fn, global_step)
            pw    = _perc_weight(global_step)
            print(
                f"Epoch {epoch:4d}/{cfg.NUM_EPOCHS} | "
                f"train loss {train_m['loss']:.4f} | "
                f"val PSNR {val_m['psnr']:.2f} dB | "
                f"val SSIM {val_m['ssim']:.4f} | "
                f"val mIoU {val_m['miou']:.4f} | "
                f"perc_w {pw:.5f} | "
                f"{dt:.1f}s"
            )
            if val_m["psnr"] > best_val_psnr:
                best_val_psnr = val_m["psnr"]
                best_path = os.path.join(cfg.CHECKPOINT_DIR, "best.pth")
                torch.save(
                    {"epoch": epoch, "model_state": model.state_dict(),
                     "global_step": global_step, "metrics": val_m},
                    best_path,
                )
                print(f"  ★ new best ({val_m['psnr']:.2f} dB) → {best_path}")
        else:
            print(
                f"Epoch {epoch:4d}/{cfg.NUM_EPOCHS} | "
                f"train loss {train_m['loss']:.4f} | "
                f"train PSNR {train_m['psnr']:.2f} dB | "
                f"{dt:.1f}s"
            )

        if epoch % cfg.SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, epoch, train_m)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "global_step": global_step,
                    "metrics": train_m,
                },
                os.path.join(cfg.CHECKPOINT_DIR, "latest.pth"),
            )

    print("Training complete.")


if __name__ == "__main__":
    main()
