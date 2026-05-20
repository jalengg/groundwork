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
from model.vae_fsq import RoadVAEFSQ
from model.vae_v2 import RoadVAEv2
from model.vae_loss import vae_loss, vae_loss_v2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/")
    parser.add_argument("--output", default="checkpoints/vae/")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--version", choices=["v1", "v2", "fsq"], default="v1",
                        help="v1: 5M focal+KL. v2: 14M CE+Dice+KL. fsq: discrete-latent FSQ-quantized.")
    parser.add_argument("--fsq-levels", default="8,5,5,5",
                        help="fsq only: comma-separated per-dim level counts, e.g. '8,5,5,5' for 1000 codes.")
    parser.add_argument("--base-ch", type=int, default=96,
                        help="v2 only: base channel count.")
    parser.add_argument("--latent-channels", type=int, default=4,
                        help="v2 only: latent channel count (4=SDXL, 16=SD3/Flux).")
    parser.add_argument("--kl-weight", type=float, default=None,
                        help="Override default KL weight (v1 default 1e-4, v2 default 1e-3).")
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

    if args.version == "v1":
        model = RoadVAE().to(device)
        kl_w = args.kl_weight if args.kl_weight is not None else 1e-4
    elif args.version == "fsq":
        levels = tuple(int(x) for x in args.fsq_levels.split(","))
        model = RoadVAEFSQ(base_ch=args.base_ch, levels=levels).to(device)
        kl_w = 0.0                                                # FSQ has no KL term
    else:
        model = RoadVAEv2(base_ch=args.base_ch, latent_channels=args.latent_channels).to(device)
        kl_w = args.kl_weight if args.kl_weight is not None else 1e-3
    n_params = sum(p.numel() for p in model.parameters())
    if args.version == "fsq":
        levels = tuple(int(x) for x in args.fsq_levels.split(","))
        n_codes = 1
        for l in levels:
            n_codes *= l
        print(f"VAE fsq: {n_params:,} params, levels={levels} ({n_codes} codes/position), KL=disabled")
    else:
        print(f"VAE {args.version}: {n_params:,} params, KL weight {kl_w:.0e}, latent_channels={args.latent_channels}")
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
        comp_sum = {"ce": 0.0, "dice": 0.0, "kl": 0.0}
        for cond, road in tqdm(train_dl, desc=f"Epoch {epoch + 1} train"):
            road = road.to(device)
            recon, mu, logvar = model(road)
            if args.version == "v1":
                loss = vae_loss(recon, road, mu, logvar, kl_weight=kl_w)
            else:
                # v2 + fsq both use CE + Dice + (optional) KL
                loss, comps = vae_loss_v2(recon, road, mu, logvar, kl_weight=kl_w)
                for k in comp_sum:
                    comp_sum[k] += comps[k]
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
                if args.version == "v1":
                    vl = vae_loss(recon, road, mu, logvar, kl_weight=kl_w)
                else:
                    vl, _ = vae_loss_v2(recon, road, mu, logvar, kl_weight=kl_w)
                val_loss += vl.item()
        msg_suffix = "v2" if args.version != "v1" else "v1"

        avg_train = train_loss / len(train_dl)
        avg_val = val_loss / max(len(val_dl), 1)
        msg = f"Epoch {epoch + 1}: train={avg_train:.4f}  val={avg_val:.4f}"
        if args.version == "v2":
            n = len(train_dl)
            msg += f"  ce={comp_sum['ce']/n:.4f} dice={comp_sum['dice']/n:.4f} kl={comp_sum['kl']/n:.4f}"
        print(msg)

        if (epoch + 1) % 5 == 0:
            path = os.path.join(args.output, f"vae_epoch_{epoch + 1:03d}.pth")
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                path,
            )
            print(f"  Saved {path}")


if __name__ == "__main__":
    main()
