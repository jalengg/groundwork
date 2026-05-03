#!/usr/bin/env python3
"""
Train the Diffusion U-Net (Stage 2). Requires trained VAE checkpoint.
Usage: python model/train_diffusion.py --vae checkpoints/vae/vae_epoch_050.pth --data data/ --output checkpoints/diffusion/
"""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_pipeline.dataset import RoadLayoutDataset
from model.diffusion import DDPM, compute_class_weight_latent
from model.unet import DiffusionUNet
from model.vae import RoadVAE


ROAD_COLORS = {
    0: [0.15, 0.15, 0.15], 1: [0.9, 0.9, 0.9], 2: [0.6, 0.8, 0.4],
    3: [0.9, 0.6, 0.2], 4: [0.9, 0.2, 0.2],
}


def onehot_to_rgb(idx):
    rgb = np.zeros((*idx.shape, 3))
    for c, col in ROAD_COLORS.items():
        rgb[idx == c] = col
    return rgb


COND_LAYOUT = [
    ("Elev", "terrain", 0),
    ("Water", "Blues", 1),
    ("Residential", "Oranges", 2),
    ("Commercial", "Purples", 3),
    ("Industrial", "Greys", 4),
    ("Parkland", "Greens", 5),
    ("Agricultural", "YlOrBr", 6),
]


def save_progress_samples(vae, net, ddpm, val_ds, device, epoch, out_dir, n=4):
    """Generate n samples from val set; include each cond channel + GT + Pred."""
    net.eval()
    n_cols = len(COND_LAYOUT) + 2  # cond channels + GT + Pred
    fig, axes = plt.subplots(n, n_cols, figsize=(2.2 * n_cols, 2.4 * n))
    if n == 1:
        axes = [axes]

    for i in range(min(n, len(val_ds))):
        cond_t, road_t = val_ds[i]
        cond = cond_t.unsqueeze(0).to(device)
        with torch.no_grad():
            z = ddpm.sample_ddim(net, cond, n_steps=50, guidance_scale=3.0)
            road_pred = vae.decode(z)[0]

        # Cond channels
        for col, (title, cmap, ch) in enumerate(COND_LAYOUT):
            arr = cond_t[ch].numpy() if ch < cond_t.shape[0] else np.zeros_like(cond_t[0].numpy())
            axes[i][col].imshow(arr, cmap=cmap, vmin=0, vmax=1 if ch > 0 else None)
            if i == 0:
                axes[i][col].set_title(title, fontsize=9)
            axes[i][col].axis("off")
        # GT
        axes[i][n_cols - 2].imshow(onehot_to_rgb(road_t.argmax(0).numpy()))
        if i == 0:
            axes[i][n_cols - 2].set_title("GT", fontsize=9)
        axes[i][n_cols - 2].axis("off")
        # Pred
        axes[i][n_cols - 1].imshow(onehot_to_rgb(road_pred.argmax(0).cpu().numpy()))
        if i == 0:
            axes[i][n_cols - 1].set_title("Pred", fontsize=9)
        axes[i][n_cols - 1].axis("off")

    plt.suptitle(f"Epoch {epoch}", fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, f"samples_epoch_{epoch:03d}.png")
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    net.train()
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae", required=True)
    parser.add_argument("--data", default="data/")
    parser.add_argument("--output", default="checkpoints/diffusion/")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--cfg-prob", type=float, default=0.1, help="P of dropping conditioning during training (0.5 = CaRoLS spec)")
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--sample-every", type=int, default=25)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--class-weights",
        type=str,
        default=None,
        help="Comma-separated 5 floats e.g. '1.0,1.2,1.4,1.4,1.4' (DRoLaS Eq. 9). "
        "If set, applies class-weighted denoising loss in latent space.",
    )
    parser.add_argument(
        "--local-module",
        choices=["lde", "load"],
        default="lde",
        help="CDB local-cond integration: 'lde' (CaRoLS concat-fusion, default) or "
        "'load' (DRoLaS SFT/FiLM affine modulation, +9 FID per DRoLaS Table 2).",
    )
    args = parser.parse_args()
    class_weights = None
    if args.class_weights:
        class_weights = torch.tensor([float(x) for x in args.class_weights.split(",")])
        assert class_weights.numel() == 5, "expected 5 weights for 5 road classes"
        print(f"  class_weights: {class_weights.tolist()}")

    os.makedirs(args.output, exist_ok=True)
    samples_dir = os.path.join(args.output, "progress_samples")
    os.makedirs(samples_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== Config ===")
    print(f"  cfg_prob: {args.cfg_prob}")
    print(f"  local_module: {args.local_module}")
    print(f"  lr: {args.lr}, batch: {args.batch}, epochs: {args.epochs}")
    print(f"  output: {args.output}")
    print(f"==============")

    # Load frozen VAE encoder
    vae = RoadVAE().to(device)
    vae.load_state_dict(torch.load(args.vae, map_location=device)["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    all_dirs = sorted(glob.glob(os.path.join(args.data, "*")))
    train_dirs = [d for d in all_dirs if "irving_tx" not in d]
    val_dirs = [d for d in all_dirs if "irving_tx" in d]

    train_ds = RoadLayoutDataset(train_dirs, augment=True)
    val_ds = RoadLayoutDataset(val_dirs, augment=False)
    print(f"Train: {len(train_ds)} tiles, Val: {len(val_ds)} tiles")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    net = DiffusionUNet(latent_channels=4, cond_channels=7, local_module=args.local_module).to(device)
    ddpm = DDPM(T=1000)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    print(f"=== Training start ===")
    for epoch in range(start_epoch, args.epochs):
        net.train()
        train_loss = 0.0
        grad_norm_sum = 0.0
        n_batches = 0
        for cond, road in tqdm(train_dl, desc=f"Epoch {epoch + 1}"):
            cond, road = cond.to(device), road.to(device)
            with torch.no_grad():
                mu, logvar = vae.encode(road)
                x0 = vae.reparameterize(mu, logvar)
                weight_latent = (
                    compute_class_weight_latent(vae, road, class_weights)
                    if class_weights is not None
                    else None
                )
            loss = ddpm.training_loss(net, x0, cond, cfg_prob=args.cfg_prob,
                                      weight_latent=weight_latent)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            grad_norm_sum += grad_norm.item()
            n_batches += 1

        avg_train = train_loss / n_batches
        avg_grad = grad_norm_sum / n_batches
        print(f"Epoch {epoch + 1}: train_loss={avg_train:.6f}, grad_norm={avg_grad:.4f}")

        # Validation loss
        if (epoch + 1) % args.val_every == 0:
            net.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for cond, road in val_dl:
                    cond, road = cond.to(device), road.to(device)
                    mu, logvar = vae.encode(road)
                    x0 = vae.reparameterize(mu, logvar)
                    weight_latent = (
                        compute_class_weight_latent(vae, road, class_weights)
                        if class_weights is not None
                        else None
                    )
                    loss = ddpm.training_loss(net, x0, cond, cfg_prob=args.cfg_prob,
                                      weight_latent=weight_latent)
                    val_loss += loss.item()
                    n_val += 1
            print(f"  val_loss={val_loss / n_val:.6f}")
            net.train()

        # Sample image generation
        if (epoch + 1) % args.sample_every == 0:
            path = save_progress_samples(vae, net, ddpm, val_ds, device, epoch + 1, samples_dir)
            print(f"  Saved sample → {path}")

        if (epoch + 1) % 10 == 0:
            path = os.path.join(args.output, f"diffusion_epoch_{epoch + 1:03d}.pth")
            torch.save(
                {"epoch": epoch, "model": net.state_dict(), "optimizer": optimizer.state_dict()},
                path,
            )
            print(f"  Saved {path}")


if __name__ == "__main__":
    main()
