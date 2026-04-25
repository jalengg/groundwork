#!/usr/bin/env python3
"""Inference with channel 3 (existing roads) zeroed out — the fair eval."""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from model.diffusion import DDPM
from model.unet import DiffusionUNet
from model.vae import RoadVAE

ROAD_COLORS = {
    0: [0.15, 0.15, 0.15], 1: [0.9, 0.9, 0.9], 2: [0.6, 0.8, 0.4],
    3: [0.9, 0.6, 0.2], 4: [0.9, 0.2, 0.2],
}


def onehot_to_rgb(road):
    idx = road.argmax(0).cpu().numpy()
    rgb = np.zeros((*idx.shape, 3))
    for c, col in ROAD_COLORS.items():
        rgb[idx == c] = col
    return rgb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae", default="checkpoints/vae/vae_epoch_050.pth")
    p.add_argument("--diffusion", default="checkpoints/diffusion/diffusion_epoch_400.pth")
    p.add_argument("--data", default="data/irving_tx")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--output", default="samples/no_roads_cond")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=3.0)
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda")

    vae = RoadVAE().to(device)
    vae.load_state_dict(torch.load(args.vae, map_location=device)["model"])
    vae.eval()

    net = DiffusionUNet(latent_channels=4, cond_channels=4).to(device)
    net.load_state_dict(torch.load(args.diffusion, map_location=device)["model"])
    net.eval()

    ddpm = DDPM(T=1000)
    cond_files = sorted(f for f in os.listdir(args.data) if f.startswith("cond_") and f.endswith(".npy"))[:args.n]

    fig, axes = plt.subplots(args.n, 4, figsize=(16, 4 * args.n))
    if args.n == 1:
        axes = [axes]

    for i, cf in enumerate(cond_files):
        idx = cf.replace("cond_", "").replace(".npy", "")
        cond_np = np.load(os.path.join(args.data, cf)).astype(np.float32)
        road_np = np.load(os.path.join(args.data, f"road_{idx}.npy")).astype(np.float32)

        # Zero out channel 3 (existing roads)
        cond_no_roads = cond_np.copy()
        cond_no_roads[3] = 0.0

        cond = torch.from_numpy(cond_no_roads).unsqueeze(0).to(device)
        road_gt = torch.from_numpy(road_np).unsqueeze(0).to(device)

        with torch.no_grad():
            z = ddpm.sample_ddim(net, cond, n_steps=args.steps, guidance_scale=args.guidance)
            road_pred = vae.decode(z)

        axes[i][0].imshow(cond_np[0], cmap="terrain"); axes[i][0].set_title(f"Elevation ({idx})"); axes[i][0].axis("off")
        axes[i][1].imshow(cond_np[3], cmap="gray"); axes[i][1].set_title("Roads cond (ZEROED for inference)"); axes[i][1].axis("off")
        axes[i][2].imshow(onehot_to_rgb(road_gt[0])); axes[i][2].set_title("Ground Truth"); axes[i][2].axis("off")
        axes[i][3].imshow(onehot_to_rgb(road_pred[0])); axes[i][3].set_title("Predicted (no road hint)"); axes[i][3].axis("off")
        print(f"  [{i+1}/{args.n}] {idx} done")

    plt.tight_layout()
    out_path = os.path.join(args.output, "samples.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
