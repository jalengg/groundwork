#!/usr/bin/env python3
"""
Visualize training tiles as PNG images for spot-checking.

Usage:
    python data_pipeline/visualize_tiles.py --data data/arlington_tx --out viz/ --sample 20
    python data_pipeline/visualize_tiles.py --data data/ --out viz/ --sample 10  # across all cities
"""
import argparse
import glob
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Road channel colors (matches vlm_eval.py)
_ROAD_COLORS = {
    0: [0.05, 0.05, 0.05],   # background: near-black
    1: [0.5,  0.5,  0.5 ],   # residential: gray
    2: [1.0,  0.65, 0.0 ],   # tertiary: orange
    3: [1.0,  0.2,  0.2 ],   # primary/secondary: red
    4: [1.0,  1.0,  1.0 ],   # motorway/trunk: white
}
_ROAD_LABELS = ["background", "residential", "tertiary", "primary/secondary", "motorway/trunk"]
_COND_LABELS = ["elevation", "land use", "water/no-build", "existing roads"]
_COND_CMAPS  = ["terrain",   "RdYlGn",    "Blues",         "gray"]


def road_to_rgb(road: np.ndarray) -> np.ndarray:
    """(5, H, W) one-hot → (H, W, 3) RGB float."""
    argmax = road.argmax(axis=0)
    rgb = np.zeros((*argmax.shape, 3), dtype=np.float32)
    for ch, color in _ROAD_COLORS.items():
        rgb[argmax == ch] = color
    return rgb


def visualize_tile(cond_path: str, road_path: str, out_path: str):
    cond = np.load(cond_path)  # (4, H, W)
    road = np.load(road_path)  # (5, H, W)

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle(os.path.basename(os.path.dirname(cond_path)) + "  " +
                 os.path.basename(cond_path).replace("cond_", "").replace(".npy", ""),
                 fontsize=11)

    # 4 conditioning channels
    for i in range(4):
        ax = axes[i]
        im = ax.imshow(cond[i], cmap=_COND_CMAPS[i], vmin=0, vmax=1)
        ax.set_title(_COND_LABELS[i], fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Road output
    ax = axes[4]
    ax.imshow(road_to_rgb(road))
    ax.set_title("road output", fontsize=9)
    ax.axis("off")

    # Road legend
    from matplotlib.patches import Patch
    legend = [Patch(color=c, label=_ROAD_LABELS[i]) for i, c in _ROAD_COLORS.items()]
    ax.legend(handles=legend, loc="lower right", fontsize=6, framealpha=0.7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   required=True, help="City dir (data/arlington_tx) or root (data/)")
    parser.add_argument("--out",    default="viz",  help="Output directory for PNGs")
    parser.add_argument("--sample", type=int, default=20, help="Number of random tiles to visualize")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Find all cond_*.npy files under --data (handles both single city and root dir)
    cond_files = sorted(glob.glob(os.path.join(args.data, "**", "cond_*.npy"), recursive=True))
    if not cond_files:
        print(f"No cond_*.npy files found under {args.data}")
        return

    random.seed(args.seed)
    sample = random.sample(cond_files, min(args.sample, len(cond_files)))

    print(f"Found {len(cond_files)} tiles, visualizing {len(sample)}...")
    for cond_path in sample:
        road_path = cond_path.replace("cond_", "road_")
        if not os.path.exists(road_path):
            print(f"  Missing road file for {cond_path}, skipping")
            continue
        city = os.path.basename(os.path.dirname(cond_path))
        idx  = os.path.basename(cond_path).replace("cond_", "").replace(".npy", "")
        out_path = os.path.join(args.out, f"{city}_{idx}.png")
        visualize_tile(cond_path, road_path, out_path)
        print(f"  {out_path}")

    print(f"\nDone. Open {args.out}/ to inspect.")


if __name__ == "__main__":
    main()
