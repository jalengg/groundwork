#!/usr/bin/env python3
"""Diagnose whether the VAE's latent space preserves residential-vs-commercial
pattern distinctions, or whether it averages them away (a ceiling on what any
downstream diffusion model can possibly recover).

Method:
  1. Scan all training tiles, find the most residential-pure and most
     commercial-pure tiles by landuse share.
  2. Encode each road tensor through the VAE → deterministic mu latent.
  3. Compare:
       a. Latent L2 distance vs. random tile-pair baseline
       b. Decoded reconstruction quality (per-class IoU vs GT)
       c. Side-by-side visual: GT road, decoded road
  4. Report.

Verdict:
  - If reconstructions of res and com tiles look identical / lose pattern
    character → VAE is the spatial-conditioning ceiling. No diffusion fix can
    recover what's already gone.
  - If reconstructions preserve pattern character → ceiling lives in the
    diffusion U-Net's conditioning pathway.
"""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_pipeline.dataset import RoadLayoutDataset
from model.train_diffusion import onehot_to_rgb
from model.vae import RoadVAE


def find_class_pure_tiles(ds, class_idx, top_k=3):
    """Return indices of the top_k tiles with the highest share of `class_idx`
    in their landuse channels (cond[2..6] is residential, commercial, industrial,
    parkland, agricultural)."""
    shares = []
    for i in range(len(ds)):
        cond, _ = ds[i]
        landuse = cond[2:7].numpy()  # (5, H, W) one-hot
        share = landuse[class_idx].mean()  # fraction of pixels in this class
        shares.append((share, i))
    shares.sort(reverse=True)
    return [i for _, i in shares[:top_k]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae", required=True)
    parser.add_argument("--data", default="data/")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-pairs", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = RoadVAE().to(device)
    vae.load_state_dict(torch.load(args.vae, map_location=device)["model"])
    vae.eval()

    # Use ALL training tiles (not val) — we want the cleanest pattern examples
    train_dirs = [d for d in sorted(glob.glob(os.path.join(args.data, "*"))) if "irving_tx" not in d]
    ds = RoadLayoutDataset(train_dirs, augment=False)
    print(f"Scanning {len(ds)} tiles for class-pure examples...")

    res_idxs = find_class_pure_tiles(ds, class_idx=0, top_k=args.n_pairs)  # residential
    com_idxs = find_class_pure_tiles(ds, class_idx=1, top_k=args.n_pairs)  # commercial
    print(f"Top residential tile shares: {[float(ds[i][0][2].mean()) for i in res_idxs]}")
    print(f"Top commercial tile shares:  {[float(ds[i][0][3].mean()) for i in com_idxs]}")

    # Encode all
    def encode_road(road):
        with torch.no_grad():
            mu, _ = vae.encode(road.unsqueeze(0).to(device))
        return mu[0]

    def decode_latent(mu):
        with torch.no_grad():
            logits = vae.decode(mu.unsqueeze(0))
        return logits[0].argmax(0).cpu().numpy()

    res_data = [(i, ds[i][1], encode_road(ds[i][1])) for i in res_idxs]
    com_data = [(i, ds[i][1], encode_road(ds[i][1])) for i in com_idxs]

    # === Quantitative ===
    # Latent L2: residential-vs-residential (within-class), residential-vs-commercial (between-class)
    res_lats = torch.stack([d[2].flatten() for d in res_data])  # (n, D)
    com_lats = torch.stack([d[2].flatten() for d in com_data])

    def pairwise_l2(A, B):
        return torch.cdist(A, B).cpu().numpy()

    within_res = pairwise_l2(res_lats, res_lats)
    within_com = pairwise_l2(com_lats, com_lats)
    between = pairwise_l2(res_lats, com_lats)

    print("\n=== Latent L2 distances (lower = more similar) ===")
    print(f"Within-residential: mean={within_res[np.triu_indices(len(res_lats), 1)].mean():.3f}")
    print(f"Within-commercial:  mean={within_com[np.triu_indices(len(com_lats), 1)].mean():.3f}")
    print(f"Between (res<->com): mean={between.mean():.3f}")
    print(f"Ratio between/within: {between.mean() / max((within_res[np.triu_indices(len(res_lats), 1)].mean() + within_com[np.triu_indices(len(com_lats), 1)].mean()) / 2, 1e-6):.2f}x")
    print("(>>1 = VAE preserves class distinction; ~1 = VAE collapses classes)")

    # Per-class IoU on reconstruction
    def per_class_iou(pred, gt):
        # gt: (5, H, W) one-hot; pred: (H, W) class indices
        gt_cls = gt.argmax(0).numpy()
        ious = []
        for c in range(5):
            inter = ((pred == c) & (gt_cls == c)).sum()
            union = ((pred == c) | (gt_cls == c)).sum()
            ious.append(inter / max(union, 1))
        return ious

    print("\n=== Reconstruction per-class IoU (bg, residential road, secondary, primary, motorway) ===")
    for label, data in [("RES tiles", res_data), ("COM tiles", com_data)]:
        for tile_idx, road, mu in data:
            recon = decode_latent(mu)
            ious = per_class_iou(recon, road)
            print(f"  {label} #{tile_idx}: IoU = [{', '.join(f'{x:.2f}' for x in ious)}]")

    # === Visual ===
    n_total = args.n_pairs * 2
    fig, axes = plt.subplots(3, n_total, figsize=(3 * n_total, 9))
    col = 0
    for label, data in [("RES", res_data), ("COM", com_data)]:
        for tile_idx, road, mu in data:
            recon = decode_latent(mu)
            cond = ds[tile_idx][0]
            # Row 0: landuse-as-RGB
            landuse = cond[2:7].numpy()
            colors = np.array([[1.0,0.6,0.4],[0.6,0.4,0.85],[0.5,0.5,0.5],[0.4,0.85,0.5],[0.95,0.85,0.5]])
            H, W = landuse.shape[1:]
            rgb = np.ones((H, W, 3))
            for c in range(5):
                rgb[landuse[c] > 0.5] = colors[c]
            axes[0][col].imshow(rgb)
            axes[0][col].set_title(f"{label} #{tile_idx} cond", fontsize=9)
            axes[0][col].axis("off")
            # Row 1: GT road
            axes[1][col].imshow(onehot_to_rgb(road.argmax(0).numpy()))
            axes[1][col].set_title("GT road", fontsize=9)
            axes[1][col].axis("off")
            # Row 2: VAE round-trip
            axes[2][col].imshow(onehot_to_rgb(recon))
            axes[2][col].set_title("VAE recon", fontsize=9)
            axes[2][col].axis("off")
            col += 1

    plt.suptitle("VAE pattern preservation: cond / GT / VAE round-trip", fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(args.out_dir, "vae_pattern_diag.png")
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
