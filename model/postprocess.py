#!/usr/bin/env python3
"""
CaRoLS-style post-processing: dilate → skeletonize → prune small components.
Takes raw diffusion output and produces cleaned connected road graphs.

Usage:
    python model/postprocess.py \
        --diffusion checkpoints/diffusion/diffusion_epoch_300.pth \
        --n 8 --output samples/postprocessed
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.morphology import binary_dilation, disk, skeletonize, remove_small_objects

from model.diffusion import DDPM
from model.unet import DiffusionUNet
from model.vae import RoadVAE

ROAD_COLORS = {
    0: [0.15, 0.15, 0.15], 1: [0.9, 0.9, 0.9], 2: [0.6, 0.8, 0.4],
    3: [0.9, 0.6, 0.2], 4: [0.9, 0.2, 0.2],
}


def onehot_to_rgb(idx):
    """(H, W) int array → (H, W, 3) RGB."""
    rgb = np.zeros((*idx.shape, 3))
    for c, col in ROAD_COLORS.items():
        rgb[idx == c] = col
    return rgb


def postprocess_layout(logits, dilate_radius=2, min_component_px=20, road_half_width=1):
    """
    logits: (5, H, W) raw diffusion output
    Returns: (H, W) cleaned class-index array
    """
    pred = logits.argmax(0).cpu().numpy()  # (H, W) in {0..4}
    H, W = pred.shape
    selem = disk(dilate_radius)

    # Process per-level, highest priority first (so we can overwrite)
    cleaned = np.zeros((H, W), dtype=np.int32)  # background=0

    for cls in [1, 2, 3, 4]:
        mask = pred == cls
        if not mask.any():
            continue
        # Dilate to bridge small gaps
        dilated = binary_dilation(mask, footprint=selem)
        # Skeletonize to centerline
        skel = skeletonize(dilated)
        # Remove small connected components (noise)
        cleaned_skel = remove_small_objects(skel, min_size=min_component_px, connectivity=2)
        # Re-dilate the skeleton slightly so roads are visible (not 1px thin)
        if road_half_width > 0:
            rendered = binary_dilation(cleaned_skel, footprint=disk(road_half_width))
        else:
            rendered = cleaned_skel
        # Overwrite (higher classes take priority)
        cleaned[rendered] = cls

    return cleaned


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae", default="checkpoints/vae/vae_epoch_050.pth")
    p.add_argument("--diffusion", default="checkpoints/diffusion/diffusion_epoch_300.pth")
    p.add_argument("--data", default="data/irving_tx")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--output", default="samples/postprocessed")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance", type=float, default=3.0)
    p.add_argument("--dilate", type=int, default=2)
    p.add_argument("--min-comp", type=int, default=20)
    p.add_argument("--road-w", type=int, default=1)
    args = p.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda")

    vae = RoadVAE().to(device)
    vae.load_state_dict(torch.load(args.vae, map_location=device)["model"])
    vae.eval()

    net = DiffusionUNet(latent_channels=4, cond_channels=7).to(device)
    net.load_state_dict(torch.load(args.diffusion, map_location=device)["model"])
    net.eval()

    ddpm = DDPM(T=1000)
    cond_files = sorted(f for f in os.listdir(args.data) if f.startswith("cond_") and f.endswith(".npy"))[:args.n]

    fig, axes = plt.subplots(args.n, 4, figsize=(16, 4 * args.n))
    if args.n == 1:
        axes = [axes]

    for i, cf in enumerate(cond_files):
        idx = cf.replace("cond_", "").replace(".npy", "")
        cond_full = np.load(os.path.join(args.data, cf)).astype(np.float32)
        road_np = np.load(os.path.join(args.data, f"road_{idx}.npy")).astype(np.float32)

        cond = torch.from_numpy(cond_full).unsqueeze(0).to(device)

        with torch.no_grad():
            z = ddpm.sample_ddim(net, cond, n_steps=args.steps, guidance_scale=args.guidance)
            road_pred = vae.decode(z)[0]  # (5, H, W)

        raw_idx = road_pred.argmax(0).cpu().numpy()
        gt_idx = torch.from_numpy(road_np).argmax(0).numpy()
        cleaned_idx = postprocess_layout(road_pred, args.dilate, args.min_comp, args.road_w)

        axes[i][0].imshow(cond_full[0], cmap="terrain"); axes[i][0].set_title(f"Elevation ({idx})"); axes[i][0].axis("off")
        axes[i][1].imshow(onehot_to_rgb(gt_idx)); axes[i][1].set_title("Ground Truth"); axes[i][1].axis("off")
        axes[i][2].imshow(onehot_to_rgb(raw_idx)); axes[i][2].set_title("Raw Predicted"); axes[i][2].axis("off")
        axes[i][3].imshow(onehot_to_rgb(cleaned_idx)); axes[i][3].set_title("Post-processed"); axes[i][3].axis("off")
        print(f"  [{i+1}/{args.n}] {idx} done")

    plt.tight_layout()
    out_path = os.path.join(args.output, "samples.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
