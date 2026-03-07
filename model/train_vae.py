#!/usr/bin/env python3
"""
Train the Road VAE (Stage 1).
Usage: python model/train_vae.py --data data/ --output checkpoints/vae/ --epochs 50
"""
import argparse
import glob
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_pipeline.dataset import RoadLayoutDataset
from model.vae import RoadVAE
from model.vae_loss import vae_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/")
    parser.add_argument("--output", default="checkpoints/vae/")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_city_dirs = sorted(glob.glob(os.path.join(args.data, "*")))
    train_dirs = [d for d in all_city_dirs if "irving_tx" not in d]
    val_dirs = [d for d in all_city_dirs if "irving_tx" in d]

    train_ds = RoadLayoutDataset(train_dirs, augment=True)
    val_ds = RoadLayoutDataset(val_dirs, augment=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    model = RoadVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        for cond, road in tqdm(train_dl, desc=f"Epoch {epoch + 1} train"):
            road = road.to(device)
            recon, mu, logvar = model(road)
            loss = vae_loss(recon, road, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for cond, road in val_dl:
                road = road.to(device)
                recon, mu, logvar = model(road)
                val_loss += vae_loss(recon, road, mu, logvar).item()

        print(
            f"Epoch {epoch + 1}: train={train_loss / len(train_dl):.4f}"
            f"  val={val_loss / max(len(val_dl), 1):.4f}"
        )

        if (epoch + 1) % 5 == 0:
            path = os.path.join(args.output, f"vae_epoch_{epoch + 1:03d}.pth")
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                path,
            )
            print(f"  Saved {path}")


if __name__ == "__main__":
    main()
