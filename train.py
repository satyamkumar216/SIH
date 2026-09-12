"""
train.py — Training loop for SatelliteSRSeg.

All hyperparameters are read from config.py.
Supports GPU (CUDA / MPS) with automatic CPU fallback.
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── local imports ─────────────────────────────────────────────────────────────
import config as cfg
from model import build_model
from dataset import TiffDataset, DummyDataset
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


# ─── Loss function ────────────────────────────────────────────────────────────

def compute_loss(
    sr_image: torch.Tensor,
    hr_target: torch.Tensor,
    seg_logits: torch.Tensor,
    seg_labels: torch.Tensor,
) -> torch.Tensor:
    """
    total_loss = W_L1 * L1(sr, hr)
               + W_SSIM * (1 - SSIM(sr, hr))
               + W_CE * CrossEntropy(seg_logits, seg_labels)
    """
    l1   = F.l1_loss(sr_image, hr_target)
    ssim_val = ssim(sr_image.clamp(0, 1), hr_target.clamp(0, 1))
    ce   = F.cross_entropy(seg_logits, seg_labels)

    total = cfg.W_L1 * l1 + cfg.W_SSIM * (1.0 - ssim_val) + cfg.W_CE * ce
    return total, l1.item(), ssim_val.item(), ce.item()


# ─── Dataset / DataLoader ─────────────────────────────────────────────────────

def build_loaders(use_dummy: bool = False):
    if use_dummy or not os.path.isdir(cfg.HR_DIR):
        print("[dataset] Using DummyDataset (no real data found).")
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
    else:
        print(f"[dataset] Loading TiffDataset from {cfg.HR_DIR}")
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
        val_ds = train_ds   # replace with a separate val split if you have one

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
    print(f"[ckpt] Resumed from epoch {ckpt['epoch']}  →  {path}")
    return ckpt["epoch"]


# ─── Training / Validation step ──────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device) -> dict:
    model.train()
    total_loss = l1_sum = ssim_sum = ce_sum = 0.0
    psnr_sum = 0.0
    n = 0

    for lr, hr, seg in loader:
        lr  = lr.to(device, non_blocking=True)
        hr  = hr.to(device, non_blocking=True)
        seg = seg.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        sr_image, seg_logits = model(lr)
        loss, l1, ssim_v, ce = compute_loss(sr_image, hr, seg_logits, seg)
        loss.backward()
        optimizer.step()

        bs = lr.size(0)
        total_loss += loss.item() * bs
        l1_sum     += l1 * bs
        ssim_sum   += ssim_v * bs
        ce_sum     += ce * bs
        psnr_sum   += psnr(sr_image.detach(), hr.detach()) * bs
        n          += bs

    return {
        "loss": total_loss / n,
        "l1":   l1_sum   / n,
        "ssim": ssim_sum / n,
        "ce":   ce_sum   / n,
        "psnr": psnr_sum / n,
    }


@torch.no_grad()
def validate(model, loader, device) -> dict:
    model.eval()
    miou_metric = MeanIoU(num_classes=cfg.NUM_CLASSES)
    total_loss = psnr_sum = ssim_sum = 0.0
    n = 0

    for lr, hr, seg in loader:
        lr  = lr.to(device, non_blocking=True)
        hr  = hr.to(device, non_blocking=True)
        seg = seg.to(device, non_blocking=True)

        sr_image, seg_logits = model(lr)
        loss, _, _, _ = compute_loss(sr_image, hr, seg_logits, seg)
        miou_metric.update(seg_logits.cpu(), seg.cpu())

        bs = lr.size(0)
        total_loss += loss.item() * bs
        psnr_sum   += psnr(sr_image, hr) * bs
        ssim_sum   += ssim(sr_image.clamp(0, 1), hr.clamp(0, 1)).item() * bs
        n          += bs

    iou_res = miou_metric.compute()
    return {
        "loss": total_loss / n,
        "psnr": psnr_sum   / n,
        "ssim": ssim_sum   / n,
        "miou": iou_res["miou"],
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = get_device()
    model  = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    # Optional: resume from checkpoint
    start_epoch = 0
    resume_ckpt = os.path.join(cfg.CHECKPOINT_DIR, "latest.pth")
    if os.path.isfile(resume_ckpt):
        start_epoch = load_checkpoint(model, optimizer, resume_ckpt, device)

    train_loader, val_loader = build_loaders()

    best_val_psnr = 0.0

    for epoch in range(start_epoch + 1, cfg.NUM_EPOCHS + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, optimizer, device)
        dt = time.time() - t0

        if epoch % cfg.LOG_EVERY == 0:
            val_m = validate(model, val_loader, device)
            print(
                f"Epoch {epoch:4d}/{cfg.NUM_EPOCHS} | "
                f"train loss {train_m['loss']:.4f} | "
                f"val PSNR {val_m['psnr']:.2f} dB | "
                f"val SSIM {val_m['ssim']:.4f} | "
                f"val mIoU {val_m['miou']:.4f} | "
                f"{dt:.1f}s"
            )

            # Save best checkpoint
            if val_m["psnr"] > best_val_psnr:
                best_val_psnr = val_m["psnr"]
                best_path = os.path.join(cfg.CHECKPOINT_DIR, "best.pth")
                torch.save(
                    {"epoch": epoch, "model_state": model.state_dict(), "metrics": val_m},
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
            # Also keep a rolling "latest"
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "optimizer_state": optimizer.state_dict(), "metrics": train_m},
                os.path.join(cfg.CHECKPOINT_DIR, "latest.pth"),
            )

    print("Training complete.")


if __name__ == "__main__":
    main()
