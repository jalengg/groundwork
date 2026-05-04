#!/usr/bin/env python3
"""Diagnose whether the diffusion model is actually using its conditioning.

For one fixed noise seed and one fixed elevation+water layer, we run inference
under several conditioning setups:
  1. Real cond (whatever's in the val tile)
  2. Zeroed cond (unconditional — CFG-drop case)
  3. All residential
  4. All commercial
  5. All industrial
  6. All parkland

If predictions 1-6 are visually identical, the model has collapsed and is
ignoring the landuse channels. If they differ in road density / structure,
conditioning is being used.

Usage:
    python -m model.diag_conditioning \
        --vae checkpoints/vae_categorical/vae_epoch_050.pth \
        --diffusion checkpoints/diff_classwt_a100/diffusion_epoch_200.pth \
        --tile-idx 0 \
        --out-dir samples/diag_cond
"""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_pipeline.dataset import RoadLayoutDataset
from model.diffusion import DDPM
from model.postprocess import vectorize_layout
from model.train_diffusion import onehot_to_rgb
from model.unet import DiffusionUNet
from model.vae import RoadVAE


# Channel layout matches data_pipeline.osm_layers.LANDUSE_CATEGORIES
# 0=elev, 1=water, 2=residential, 3=commercial, 4=industrial, 5=parkland, 6=agricultural
N_LANDUSE = 5
LANDUSE_NAMES = ["residential", "commercial", "industrial", "parkland", "agricultural"]


def make_synthetic_cond(real_cond, only_class):
    """Take the real elev+water from the tile, replace landuse with all-1s in one class."""
    cond = real_cond.clone()
    cond[2:7] = 0
    if only_class is not None:
        cond[2 + only_class] = 1.0
    return cond


def run_inference(net, vae, ddpm, cond, device, seed=42, w=3.0, postprocess=False):
    """Sample with a fixed noise seed for fair comparison.

    If postprocess=True, run the full CaRoLS Sec 3.3 vectorization pipeline
    (skeleton -> graph -> bridge disjoint CCs -> DP-simplify -> re-render)
    before returning the class-index map. Paper computes CI/TC only on this
    representation; res-vs-com pixel-disagreement at the raw-raster level
    over-states class similarity because both classes have the same noise
    statistics before topology is recovered.
    """
    torch.manual_seed(seed)
    cond_b = cond.unsqueeze(0).to(device)
    with torch.no_grad():
        z = ddpm.sample_ddim(net, cond_b, n_steps=50, guidance_scale=w)
        road = vae.decode(z)[0]
    if not postprocess:
        return road.argmax(0).cpu().numpy()
    vec_idx, _ = vectorize_layout(road)
    return vec_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae", required=True)
    parser.add_argument("--diffusion", required=True)
    parser.add_argument("--data", default="data/")
    parser.add_argument("--tile-idx", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-module", choices=["lde", "load"], default="lde",
                        help="Must match the trained checkpoint's CDB local-module variant.")
    parser.add_argument("--postprocess", action="store_true",
                        help="Apply CaRoLS Sec 3.3 vectorization before computing the diag matrix.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = RoadVAE().to(device)
    vae.load_state_dict(torch.load(args.vae, map_location=device)["model"])
    vae.eval()

    net = DiffusionUNet(latent_channels=4, cond_channels=7, local_module=args.local_module).to(device)
    net.load_state_dict(torch.load(args.diffusion, map_location=device)["model"])
    net.eval()

    val_dirs = [d for d in sorted(glob.glob(os.path.join(args.data, "*"))) if "irving_tx" in d]
    val_ds = RoadLayoutDataset(val_dirs, augment=False)
    cond_t, road_t = val_ds[args.tile_idx]

    ddpm = DDPM(T=1000)

    # Run inference under various cond setups
    setups = [
        ("Real cond", cond_t),
        ("Zero cond (uncond)", torch.zeros_like(cond_t)),
    ]
    for i, name in enumerate(LANDUSE_NAMES):
        setups.append((f"All {name}", make_synthetic_cond(cond_t, i)))
    setups.append(("All landuse=0 (kept elev+water)", make_synthetic_cond(cond_t, None)))

    n_setups = len(setups)
    fig, axes = plt.subplots(2, n_setups, figsize=(3 * n_setups, 6))

    # Top row: GT (just first cell), then for each setup show conditioning summary
    for col, (name, cond) in enumerate(setups):
        # Visualize landuse one-hot as combined RGB
        landuse = cond[2:7].numpy()  # (5, H, W)
        # Per-class colors
        landuse_colors = np.array([
            [1.0, 0.6, 0.4],   # residential - orange
            [0.6, 0.4, 0.85],  # commercial - purple
            [0.5, 0.5, 0.5],   # industrial - grey
            [0.4, 0.85, 0.5],  # parkland - green
            [0.95, 0.85, 0.5], # agricultural - yellow
        ])
        H, W = landuse.shape[1:]
        rgb = np.ones((H, W, 3))  # white background
        for c in range(5):
            mask = landuse[c] > 0.5
            rgb[mask] = landuse_colors[c]
        axes[0][col].imshow(rgb)
        axes[0][col].set_title(name, fontsize=9)
        axes[0][col].axis("off")

    # Bottom row: predictions
    for col, (name, cond) in enumerate(setups):
        pred = run_inference(net, vae, ddpm, cond, device, seed=args.seed, w=args.guidance, postprocess=args.postprocess)
        axes[1][col].imshow(onehot_to_rgb(pred))
        axes[1][col].set_title(f"Pred (seed={args.seed})", fontsize=9)
        axes[1][col].axis("off")
        print(f"  {name}: pred class shares = "
              f"{np.bincount(pred.flatten(), minlength=5) / pred.size}")

    title_extra = " [POSTPROCESSED]" if args.postprocess else ""
    plt.suptitle(f"Conditioning diagnostic — tile {args.tile_idx}, w={args.guidance}{title_extra}", fontsize=11)
    plt.tight_layout()
    suffix = "_postproc" if args.postprocess else ""
    out_path = os.path.join(args.out_dir, f"diag_tile{args.tile_idx}{suffix}.png")
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # Quantitative: pairwise pixel-disagreement between predictions
    print("\n=== Pairwise pixel disagreement (% pixels differing) ===")
    preds = []
    for name, cond in setups:
        preds.append((name, run_inference(net, vae, ddpm, cond, device, seed=args.seed, w=args.guidance, postprocess=args.postprocess)))
    print(f"{'':<35}" + "".join(f"{n[:14]:>16}" for n, _ in preds))
    for i, (ni, pi) in enumerate(preds):
        row = f"{ni[:35]:<35}"
        for j, (nj, pj) in enumerate(preds):
            disagreement = (pi != pj).mean() * 100
            row += f"{disagreement:>15.1f}%"
        print(row)


if __name__ == "__main__":
    main()
