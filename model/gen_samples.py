#!/usr/bin/env python3
"""Generate one comparison PNG (all cond channels + GT + Pred) from a trained
diffusion checkpoint, using the latest save_progress_samples layout.

Usage:
    python -m model.gen_samples --vae checkpoints/vae_categorical/vae_epoch_050.pth \
        --diffusion checkpoints/diff_categorical/diffusion_epoch_150.pth \
        --out-dir samples/diff_categorical
"""
import argparse
import glob
import os

import torch

from data_pipeline.dataset import RoadLayoutDataset
from model.diffusion import DDPM
from model.train_diffusion import save_progress_samples
from model.unet import DiffusionUNet
from model.vae import RoadVAE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae", required=True)
    parser.add_argument("--diffusion", required=True)
    parser.add_argument("--data", default="data/")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--cond-channels", type=int, default=7)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = RoadVAE().to(device)
    vae.load_state_dict(torch.load(args.vae, map_location=device)["model"])
    vae.eval()

    net = DiffusionUNet(latent_channels=4, cond_channels=args.cond_channels).to(device)
    ckpt = torch.load(args.diffusion, map_location=device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    epoch = ckpt.get("epoch", 0) + 1

    val_dirs = [d for d in sorted(glob.glob(os.path.join(args.data, "*"))) if "irving_tx" in d]
    val_ds = RoadLayoutDataset(val_dirs, augment=False)
    print(f"VAE:       {args.vae}")
    print(f"Diffusion: {args.diffusion}  (epoch {epoch})")
    print(f"Val tiles: {len(val_ds)}, n={args.n}")

    ddpm = DDPM(T=1000)
    path = save_progress_samples(vae, net, ddpm, val_ds, device, epoch, args.out_dir, n=args.n)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
