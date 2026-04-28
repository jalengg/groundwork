#!/usr/bin/env python3
"""
Run DDIM inference and visualize predicted vs ground-truth road maps.
Usage:
    python model/sample_diffusion.py \
        --vae checkpoints/vae/vae_epoch_050.pth \
        --diffusion checkpoints/diffusion/diffusion_epoch_200.pth \
        --data data/irving_tx \
        --n 8 \
        --output samples/
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from model.diffusion import DDPM
from model.unet import DiffusionUNet
from model.vae import RoadVAE

ROAD_COLORS = {
    0: [0.15, 0.15, 0.15],   # background — dark gray
    1: [0.9, 0.9, 0.9],      # residential — light gray
    2: [0.6, 0.8, 0.4],      # tertiary — green
    3: [0.9, 0.6, 0.2],      # primary/secondary — orange
    4: [0.9, 0.2, 0.2],      # motorway/trunk — red
}


def onehot_to_rgb(road):
    """(5, H, W) float logits → (H, W, 3) RGB."""
    idx = road.argmax(0).cpu().numpy()
    h, w = idx.shape
    rgb = np.zeros((h, w, 3))
    for cls, color in ROAD_COLORS.items():
        mask = idx == cls
        rgb[mask] = color
    return rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae", default="checkpoints/vae/vae_epoch_050.pth")
    parser.add_argument("--diffusion", default="checkpoints/diffusion/diffusion_epoch_200.pth")
    parser.add_argument("--data", default="data/irving_tx")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--output", default="samples/")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=3.0)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models
    vae = RoadVAE().to(device)
    vae.load_state_dict(torch.load(args.vae, map_location=device)["model"])
    vae.eval()

    net = DiffusionUNet(latent_channels=4, cond_channels=7).to(device)
    ckpt = torch.load(args.diffusion, map_location=device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    print(f"Loaded diffusion checkpoint (epoch {ckpt['epoch'] + 1})")

    ddpm = DDPM(T=1000)

    # Pick samples
    cond_files = sorted(f for f in os.listdir(args.data) if f.startswith("cond_") and f.endswith(".npy"))[:args.n]

    fig, axes = plt.subplots(args.n, 3, figsize=(12, 4 * args.n))
    if args.n == 1:
        axes = [axes]

    for i, cf in enumerate(cond_files):
        idx = cf.replace("cond_", "").replace(".npy", "")
        cond_full = np.load(os.path.join(args.data, cf)).astype(np.float32)
        cond_np = cond_full  # New format: (7, H, W) — elev, water, landuse one-hot
        road_np = np.load(os.path.join(args.data, f"road_{idx}.npy")).astype(np.float32)

        cond = torch.from_numpy(cond_np).unsqueeze(0).to(device)
        road_gt = torch.from_numpy(road_np).unsqueeze(0).to(device)

        with torch.no_grad():
            z = ddpm.sample_ddim(net, cond, n_steps=args.steps, guidance_scale=args.guidance)
            road_pred = vae.decode(z)

        # Column 0: conditioning (elevation channel)
        axes[i][0].imshow(cond_full[0], cmap="terrain")
        axes[i][0].set_title(f"Elevation (tile {idx})")
        axes[i][0].axis("off")

        # Column 1: ground truth roads
        axes[i][1].imshow(onehot_to_rgb(road_gt[0]))
        axes[i][1].set_title("Ground Truth")
        axes[i][1].axis("off")

        # Column 2: predicted roads
        axes[i][2].imshow(onehot_to_rgb(road_pred[0]))
        axes[i][2].set_title("Predicted")
        axes[i][2].axis("off")

        print(f"  [{i+1}/{args.n}] tile {idx} done")

    plt.tight_layout()
    out_path = os.path.join(args.output, "samples.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
