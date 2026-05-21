#!/usr/bin/env python3
"""VAE-GAN training. Adds patch discriminator + adversarial + feature-matching
losses to the v2 VAE recipe — the missing 'sharpness pressure' that StyleGAN
has and our prior VAE recipes don't.

Why: oracle decoder showed v1/v2/v3/FSQ all hit ~0.67 road-pixel IoU, but
visually preserve roads. The IoU ceiling is sub-pixel boundary blur from MSE
mean-seeking loss. Adversarial loss makes the discriminator say 'real roads
have sharp lines, blurry outputs are fake' — forces the VAE to commit.

LDM (Rombach 2022) and SDXL (Podell 2023) both use this recipe. We dropped
it because LPIPS is photo-trained and we thought our 5-class palette data
didn't need it. Discriminator feature matching gives us a perceptual signal
trained on our domain instead.

Loss schedule (LDM-style):
  - Epochs 0..gan_warmup-1: pure reconstruction (CE + Dice + KL)
  - Epochs gan_warmup+: add adversarial + feature-matching, with adaptive
    weighting (balances grad norms of rec vs adv on last-layer weight)

Usage:
    python -m model.train_vae_gan \\
        --data data/ --output checkpoints/vae_gan/ --epochs 80 \\
        --base-ch 96 --latent-channels 4 --gan-warmup 10
"""
import argparse
import glob
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_pipeline.dataset import RoadLayoutDataset
from model.discriminator import PatchDiscriminator, onehot_logits_to_rgb
from model.vae_loss import (
    adaptive_d_weight,
    feature_matching_loss,
    hinge_d_loss,
    hinge_g_loss,
    vae_loss_v2,
)
from model.vae_v2 import RoadVAEv2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/")
    p.add_argument("--output", default="checkpoints/vae_gan/")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--d-lr", type=float, default=2e-5)
    p.add_argument("--base-ch", type=int, default=96)
    p.add_argument("--latent-channels", type=int, default=4)
    p.add_argument("--kl-weight", type=float, default=1e-3)
    p.add_argument("--fm-weight", type=float, default=1.0,
                   help="Feature-matching weight (perceptual signal from D).")
    p.add_argument("--gan-warmup", type=int, default=10,
                   help="Epochs of pure reconstruction before adv loss kicks in.")
    p.add_argument("--resume", default=None)
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_dirs = sorted(glob.glob(os.path.join(args.data, "*")))
    train_dirs = [d for d in all_dirs if "irving_tx" not in d]
    val_dirs = [d for d in all_dirs if "irving_tx" in d]
    train_ds = RoadLayoutDataset(train_dirs, augment=True)
    val_ds = RoadLayoutDataset(val_dirs, augment=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    vae = RoadVAEv2(base_ch=args.base_ch, latent_channels=args.latent_channels).to(device)
    disc = PatchDiscriminator(in_channels=3, base_ch=64, n_layers=4).to(device)
    n_vae = sum(p.numel() for p in vae.parameters())
    n_disc = sum(p.numel() for p in disc.parameters())
    print(f"VAE: {n_vae:,} params  |  Discriminator: {n_disc:,} params")
    print(f"KL weight={args.kl_weight}, FM weight={args.fm_weight}, GAN warmup={args.gan_warmup}")

    opt_vae = torch.optim.Adam(vae.parameters(), lr=args.lr, betas=(0.5, 0.9))
    opt_disc = torch.optim.Adam(disc.parameters(), lr=args.d_lr, betas=(0.5, 0.9))
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        vae.load_state_dict(ckpt["vae"])
        disc.load_state_dict(ckpt["disc"])
        opt_vae.load_state_dict(ckpt["opt_vae"])
        opt_disc.load_state_dict(ckpt["opt_disc"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    # The VAE's last conv layer — used by adaptive_d_weight to balance grads
    last_layer_weight = vae.decoder.out_proj.weight

    for epoch in range(start_epoch, args.epochs):
        vae.train()
        disc.train()
        use_gan = epoch >= args.gan_warmup

        rec_sum, d_sum, g_sum, fm_sum, dw_sum, n = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        for cond, road in tqdm(train_dl, desc=f"Epoch {epoch + 1}{' (GAN)' if use_gan else ''}"):
            road = road.to(device)
            recon, mu, logvar = vae(road)
            rec_loss, comps = vae_loss_v2(recon, road, mu, logvar, kl_weight=args.kl_weight)

            if use_gan:
                # --- Generator (VAE) step: rec + adaptive-weighted (adv + FM) ---
                real_rgb = onehot_logits_to_rgb(road, soft=False)
                fake_rgb = onehot_logits_to_rgb(recon, soft=True)
                fake_logits = disc(fake_rgb)
                g_loss = hinge_g_loss(fake_logits)
                real_feats = disc.extract_features(real_rgb.detach())
                fake_feats = disc.extract_features(fake_rgb)
                fm_loss = feature_matching_loss(real_feats, fake_feats)
                try:
                    d_weight = adaptive_d_weight(rec_loss, g_loss, last_layer_weight)
                except RuntimeError:
                    d_weight = torch.tensor(0.0, device=device)
                vae_total = rec_loss + d_weight * g_loss + args.fm_weight * fm_loss

                opt_vae.zero_grad()
                vae_total.backward()
                opt_vae.step()

                # --- Discriminator step ---
                with torch.no_grad():
                    recon_d, _, _ = vae(road)
                real_rgb_d = onehot_logits_to_rgb(road, soft=False)
                fake_rgb_d = onehot_logits_to_rgb(recon_d, soft=True)
                real_logits = disc(real_rgb_d)
                fake_logits_d = disc(fake_rgb_d)
                d_loss = hinge_d_loss(real_logits, fake_logits_d)

                opt_disc.zero_grad()
                d_loss.backward()
                opt_disc.step()

                rec_sum += rec_loss.item()
                d_sum += d_loss.item()
                g_sum += g_loss.item()
                fm_sum += fm_loss.item()
                dw_sum += float(d_weight)
            else:
                # Pure reconstruction warmup
                opt_vae.zero_grad()
                rec_loss.backward()
                opt_vae.step()
                rec_sum += rec_loss.item()

            n += 1

        avg_rec = rec_sum / n
        if use_gan:
            print(f"Epoch {epoch + 1}: rec={avg_rec:.4f} d={d_sum/n:.4f} "
                  f"g={g_sum/n:.4f} fm={fm_sum/n:.4f} d_w={dw_sum/n:.3f}")
        else:
            print(f"Epoch {epoch + 1}: rec={avg_rec:.4f} (no GAN yet)")

        # Validation: just reconstruction (don't run discriminator)
        vae.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for cond, road in val_dl:
                road = road.to(device)
                recon, mu, logvar = vae(road)
                vl, _ = vae_loss_v2(recon, road, mu, logvar, kl_weight=args.kl_weight)
                val_loss += vl.item()
                n_val += 1
        print(f"  val rec={val_loss / max(n_val, 1):.4f}")

        if (epoch + 1) % 5 == 0:
            path = os.path.join(args.output, f"vae_epoch_{epoch + 1:03d}.pth")
            torch.save({
                "epoch": epoch,
                "model": vae.state_dict(),                       # for diag compat
                "vae": vae.state_dict(),
                "disc": disc.state_dict(),
                "opt_vae": opt_vae.state_dict(),
                "opt_disc": opt_disc.state_dict(),
            }, path)
            print(f"  Saved {path}")


if __name__ == "__main__":
    main()
