#!/usr/bin/env python3
"""Oracle-decoder diagnostic: localizes whether the VAE or the diffusion
denoiser is the bottleneck (per Phase-1 council recommendation).

Three tests, ranked by decisional value:

1. **Encode-decode roundtrip with road-pixel-masked IoU**: encode GT one-hot
   road, decode without diffusion, compare per-class IoU only on
   road-bearing pixels (excluding background which dominates whole-image IoU).
   If IoU >= 0.90, VAE is fine and any sample-quality issue is downstream.

2. **Noise-injection oracle**: encode GT, add `σ·ε` for σ ∈ {0.1, 0.3, 0.5, 1.0},
   decode. Tests latent error-correction margin under noise — DDPM samples
   live exactly in this perturbed regime. Collapse at small σ means the latent
   has zero margin and DDPM cannot work on it regardless of denoiser.

3. **Latent statistics**: per-channel mean/std of `mu` over the train set.
   For SDXL VAE these tell us whether `scaling_factor` is calibrated or way
   off — a known cause of DDPM noise-schedule miscalibration.

Usage:
    python -m model.diag_vae_oracle --vae-type sdxl --out-dir samples/oracle_sdxl
    python -m model.diag_vae_oracle --vae-type custom --vae checkpoints/vae_categorical/vae_epoch_050.pth --out-dir samples/oracle_custom
"""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_pipeline.dataset import RoadLayoutDataset
from model.train_diffusion import onehot_to_rgb
from model.vae import RoadVAE
from model.vae_sdxl import RoadVAESDXL


def per_class_iou_road_pixels(pred_idx, gt_idx, n_classes=5):
    """Per-class IoU computed ONLY over pixels where GT is non-background.
    Whole-image IoU is dominated by ~95% bg agreement and is misleading."""
    road_mask = gt_idx > 0
    out = {}
    for c in range(n_classes):
        if c == 0:
            inter = ((pred_idx == 0) & (gt_idx == 0)).sum()
            union = ((pred_idx == 0) | (gt_idx == 0)).sum()
            out[c] = float(inter) / max(float(union), 1)
        else:
            # Restrict to GT-road pixels for fair per-class evaluation
            p = pred_idx[road_mask] == c
            g = gt_idx[road_mask] == c
            inter = (p & g).sum()
            union = (p | g).sum()
            out[c] = float(inter) / max(float(union), 1)
    return out


@torch.no_grad()
def latent_stats(vae, loader, max_batches=20, device="cpu"):
    """Per-channel empirical mean/std of mu over the train set."""
    means, sq_means, n = None, None, 0
    for i, (cond, road) in enumerate(loader):
        if i >= max_batches:
            break
        road = road.to(device)
        mu, _ = vae.encode(road)
        B = mu.shape[0]
        n += B
        cur_mean = mu.mean(dim=(0, 2, 3))
        cur_sq = mu.pow(2).mean(dim=(0, 2, 3))
        if means is None:
            means, sq_means = cur_mean * B, cur_sq * B
        else:
            means += cur_mean * B
            sq_means += cur_sq * B
    means = (means / n).cpu()
    sq_means = (sq_means / n).cpu()
    stds = (sq_means - means.pow(2)).clamp(min=1e-8).sqrt()
    return means, stds, n


@torch.no_grad()
def oracle_roundtrip(vae, loader, max_batches=10, device="cpu"):
    """Encode -> decode (no diffusion). Per-class road-pixel IoU."""
    ious_acc = {c: [] for c in range(5)}
    for i, (cond, road) in enumerate(loader):
        if i >= max_batches:
            break
        road = road.to(device)
        mu, logvar = vae.encode(road)
        out = vae.decode(mu)
        pred_idx = out.argmax(dim=1).cpu().numpy()
        gt_idx = road.argmax(dim=1).cpu().numpy()
        for b in range(road.shape[0]):
            ious = per_class_iou_road_pixels(pred_idx[b], gt_idx[b])
            for c in range(5):
                ious_acc[c].append(ious[c])
    return {c: float(np.mean(ious_acc[c])) if ious_acc[c] else 0.0 for c in range(5)}


@torch.no_grad()
def noise_injection(vae, loader, sigmas, max_batches=4, device="cpu"):
    """Encode GT, add σ·ε in latent, decode. Per-class IoU per σ."""
    out = {sigma: {c: [] for c in range(5)} for sigma in sigmas}
    for i, (cond, road) in enumerate(loader):
        if i >= max_batches:
            break
        road = road.to(device)
        mu, _ = vae.encode(road)
        for sigma in sigmas:
            z = mu + sigma * torch.randn_like(mu)
            dec = vae.decode(z)
            pred_idx = dec.argmax(dim=1).cpu().numpy()
            gt_idx = road.argmax(dim=1).cpu().numpy()
            for b in range(road.shape[0]):
                ious = per_class_iou_road_pixels(pred_idx[b], gt_idx[b])
                for c in range(5):
                    out[sigma][c].append(ious[c])
    return {
        sigma: {c: float(np.mean(out[sigma][c])) if out[sigma][c] else 0.0 for c in range(5)}
        for sigma in sigmas
    }


@torch.no_grad()
def save_visual_grid(vae, loader, sigmas, out_path, n_tiles=4, device="cpu"):
    """Per-row: GT | recon | recon+0.1ε | recon+0.3ε | recon+0.5ε | recon+1.0ε"""
    sigma_cols = [0.0] + list(sigmas)
    n_cols = 1 + len(sigma_cols)
    fig, axes = plt.subplots(n_tiles, n_cols, figsize=(2.5 * n_cols, 2.5 * n_tiles))
    if n_tiles == 1:
        axes = [axes]
    for cond, road in loader:
        road = road.to(device)
        for i in range(min(n_tiles, road.shape[0])):
            r = road[i:i + 1]
            mu, _ = vae.encode(r)
            gt_idx = r.argmax(dim=1)[0].cpu().numpy()
            axes[i][0].imshow(onehot_to_rgb(gt_idx))
            axes[i][0].set_title("GT" if i == 0 else "")
            axes[i][0].axis("off")
            for j, sigma in enumerate(sigma_cols):
                z = mu + sigma * torch.randn_like(mu)
                dec = vae.decode(z)
                idx = dec.argmax(dim=1)[0].cpu().numpy()
                axes[i][1 + j].imshow(onehot_to_rgb(idx))
                axes[i][1 + j].set_title(
                    f"σ={sigma:.1f}" if i == 0 else "", fontsize=9)
                axes[i][1 + j].axis("off")
        break
    plt.suptitle(f"VAE oracle decoder + noise injection", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae-type", choices=["custom", "sdxl"], default="sdxl")
    p.add_argument("--vae", default=None)
    p.add_argument("--data", default="data/")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--calibrate-sdxl", action="store_true",
                   help="Run SDXL per-channel calibration before tests.")
    p.add_argument("--max-batches", type=int, default=10)
    p.add_argument("--n-tiles-vis", type=int, default=4)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.vae_type == "sdxl":
        vae = RoadVAESDXL().to(device)
    else:
        if not args.vae:
            p.error("--vae required for --vae-type=custom")
        vae = RoadVAE().to(device)
        vae.load_state_dict(torch.load(args.vae, map_location=device)["model"])
    vae.eval()

    train_dirs = [d for d in sorted(glob.glob(os.path.join(args.data, "*")))
                  if "irving_tx" not in d]
    val_dirs = [d for d in sorted(glob.glob(os.path.join(args.data, "*"))) if "irving_tx" in d]
    train_ds = RoadLayoutDataset(train_dirs, augment=False)
    val_ds = RoadLayoutDataset(val_dirs, augment=False)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)

    if args.calibrate_sdxl and args.vae_type == "sdxl":
        print("Calibrating SDXL VAE per-channel scaling on train set...")
        vae.calibrate(train_loader, max_batches=64)

    print("\n=== 1. Latent statistics (train set) ===")
    means, stds, n = latent_stats(vae, train_loader, max_batches=args.max_batches, device=device)
    print(f"  N = {n} samples")
    print(f"  per-channel mean: {[f'{m:+.4f}' for m in means.tolist()]}")
    print(f"  per-channel std:  {[f'{s:.4f}' for s in stds.tolist()]}")
    target_std = 1.0
    if args.vae_type == "sdxl":
        print(f"  (target after calibration: ≈1.0 per channel)")
    elif (stds - target_std).abs().max() > 0.3:
        print(f"  ⚠ stds deviate from target {target_std:.1f}; DDPM noise schedule may be miscalibrated")

    print("\n=== 2. Encode-decode roundtrip IoU (val set, road-pixel-masked) ===")
    ious = oracle_roundtrip(vae, val_loader, max_batches=args.max_batches, device=device)
    classes = ["bg", "residential", "tertiary", "primary", "motorway"]
    for c in range(5):
        flag = "✅" if ious[c] >= 0.85 else ("⚠" if ious[c] >= 0.5 else "❌")
        print(f"  {classes[c]:>12s}: {ious[c]:.3f} {flag}")
    avg_road = np.mean([ious[c] for c in [1, 2, 3, 4]])
    print(f"  avg over road classes: {avg_road:.3f}")
    if avg_road >= 0.85:
        print("  → VAE is FINE for this task; bottleneck is the diffusion denoiser")
    elif avg_road >= 0.5:
        print("  → VAE is marginal; retraining or scale-fix may help")
    else:
        print("  → VAE is the bottleneck; cannot be fixed by denoiser changes")

    print("\n=== 3. Noise-injection oracle ===")
    sigmas = [0.1, 0.3, 0.5, 1.0]
    nz = noise_injection(vae, val_loader, sigmas,
                         max_batches=min(args.max_batches, 4), device=device)
    print(f"  {'σ':>5s} | {'bg':>6s} | {'res':>6s} | {'ter':>6s} | {'pri':>6s} | {'mot':>6s} | {'avg-road':>8s}")
    for sigma in sigmas:
        avg = np.mean([nz[sigma][c] for c in [1, 2, 3, 4]])
        print(f"  {sigma:>5.1f} | "
              f"{nz[sigma][0]:>6.3f} | {nz[sigma][1]:>6.3f} | {nz[sigma][2]:>6.3f} | "
              f"{nz[sigma][3]:>6.3f} | {nz[sigma][4]:>6.3f} | {avg:>8.3f}")
    if nz[0.3][1] < 0.3:
        print("  ⚠ latent has near-zero error margin: any DDPM denoiser will struggle")

    print("\n=== 4. Visual grid ===")
    out_png = os.path.join(args.out_dir, f"oracle_{args.vae_type}.png")
    save_visual_grid(vae, val_loader, sigmas, out_png,
                     n_tiles=args.n_tiles_vis, device=device)
    print(f"  Saved: {out_png}")


if __name__ == "__main__":
    main()
