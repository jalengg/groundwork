#!/usr/bin/env python3
"""
Train the Diffusion U-Net (Stage 2). Requires trained VAE checkpoint.
Usage: python model/train_diffusion.py --vae checkpoints/vae/vae_epoch_050.pth --data data/ --output checkpoints/diffusion/
"""
import argparse
import glob
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_pipeline.dataset import RoadLayoutDataset
from model.diffusion import DDPM
from model.unet import DiffusionUNet
from model.vae import RoadVAE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae", required=True)
    parser.add_argument("--data", default="data/")
    parser.add_argument("--output", default="checkpoints/diffusion/")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load frozen VAE encoder
    vae = RoadVAE().to(device)
    vae.load_state_dict(torch.load(args.vae, map_location=device)["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    all_dirs = sorted(glob.glob(os.path.join(args.data, "*")))
    train_dirs = [d for d in all_dirs if "irving_tx" not in d]
    val_dirs = [d for d in all_dirs if "irving_tx" in d]
    train_dl = DataLoader(
        RoadLayoutDataset(train_dirs, augment=True),
        batch_size=args.batch, shuffle=True, num_workers=2,
    )
    val_dl = DataLoader(
        RoadLayoutDataset(val_dirs, augment=False),
        batch_size=args.batch, shuffle=False, num_workers=2,
    )

    net = DiffusionUNet(latent_channels=4, cond_channels=4).to(device)
    ddpm = DDPM(T=1000)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1

    for epoch in range(start_epoch, args.epochs):
        net.train()
        train_loss = 0.0
        for cond, road in tqdm(train_dl, desc=f"Epoch {epoch + 1}"):
            cond, road = cond.to(device), road.to(device)
            with torch.no_grad():
                mu, logvar = vae.encode(road)
                x0 = vae.reparameterize(mu, logvar)
            loss = ddpm.training_loss(net, x0, cond, cfg_prob=0.5)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch + 1}: train_loss={train_loss / len(train_dl):.6f}")

        if (epoch + 1) % 10 == 0:
            path = os.path.join(args.output, f"diffusion_epoch_{epoch + 1:03d}.pth")
            torch.save(
                {"epoch": epoch, "model": net.state_dict(), "optimizer": optimizer.state_dict()},
                path,
            )
            print(f"  Saved {path}")


if __name__ == "__main__":
    main()
