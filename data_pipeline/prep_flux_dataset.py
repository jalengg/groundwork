#!/usr/bin/env python3
"""Convert our (cond_*.npy, road_*.npy) tile pairs into the paired-PNG layout
SimpleTuner / diffusers ControlNet trainers expect.

Cond encoding (7 -> 3 channels):
    R = elevation (per-tile-normalized, [0,1])
    G = 0.3*water + 0.5*parkland + 0.2*agricultural
    B = 0.6*residential + 0.8*commercial + 1.0*industrial

Target encoding (5 -> 3 channels): fixed RGB palette per class
    bg/0=black, residential/1=red, tertiary/2=green, primary/3=blue, motorway/4=yellow

All tiles are upscaled to 1024x1024 (Flux native resolution):
    elevation channel: bilinear
    one-hot (water, landuse, road): nearest-neighbor (preserves one-hot)

Outputs at $OUT/{cond,target}/{city}_{tile_id}.png and $OUT/meta/{city}_{tile_id}.txt
(constant prompt per tile so ControlNet does the heavy lifting).

Usage:
    python -m data_pipeline.prep_flux_dataset --src data/ --dst data/flux_cnet
"""
import argparse
import glob
from pathlib import Path

import cv2
import numpy as np


PALETTE = np.array(
    [
        [0, 0, 0],         # 0 bg
        [255, 0, 0],       # 1 residential
        [0, 255, 0],       # 2 tertiary
        [0, 0, 255],       # 3 primary
        [255, 255, 0],     # 4 motorway
    ],
    dtype=np.uint8,
)

PROMPT = (
    "top-down satellite-style raster of a US suburban road network, "
    "high-contrast color-coded road class map, flat color, vector style, "
    "no texture, no shading"
)


def encode_cond(cond, target_size=1024):
    """(7, 512, 512) -> (1024, 1024, 3) uint8 RGB."""
    elev = cond[0]
    elev_n = (elev - elev.min()) / (np.ptp(elev) + 1e-6)
    R = elev_n
    G = 0.3 * cond[1] + 0.5 * cond[5] + 0.2 * cond[6]
    B = 0.6 * cond[2] + 0.8 * cond[3] + 1.0 * cond[4]
    rgb = np.stack([R, G, B], axis=-1).clip(0, 1)
    rgb_u8 = (rgb * 255).astype(np.uint8)
    return cv2.resize(rgb_u8, (target_size, target_size), interpolation=cv2.INTER_LINEAR)


def encode_target(road, target_size=1024):
    """(5, 512, 512) -> (1024, 1024, 3) uint8 RGB."""
    cls = road.argmax(axis=0).astype(np.uint8)
    cls = cv2.resize(cls, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    return PALETTE[cls]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="data/", help="Root data dir with city subdirs.")
    p.add_argument("--dst", default="data/flux_cnet/")
    p.add_argument("--holdout-cities", nargs="*", default=["irving_tx"],
                   help="Cities to skip (held out for eval).")
    p.add_argument("--target-size", type=int, default=1024)
    args = p.parse_args()

    out = Path(args.dst)
    (out / "cond").mkdir(parents=True, exist_ok=True)
    (out / "target").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)

    cond_paths = sorted(glob.glob(f"{args.src}/*/cond_*.npy"))
    n_done = n_skipped = 0
    for cp in cond_paths:
        city = Path(cp).parent.name
        if city in args.holdout_cities:
            n_skipped += 1
            continue
        rid = Path(cp).stem.replace("cond_", "")
        rp = cp.replace("cond_", "road_")
        cond = np.load(cp).astype(np.float32)
        road = np.load(rp).astype(np.float32)
        # cv2 expects BGR; reverse the last axis on write
        cv2.imwrite(str(out / "cond" / f"{city}_{rid}.png"),
                    encode_cond(cond, args.target_size)[:, :, ::-1])
        cv2.imwrite(str(out / "target" / f"{city}_{rid}.png"),
                    encode_target(road, args.target_size)[:, :, ::-1])
        (out / "meta" / f"{city}_{rid}.txt").write_text(PROMPT)
        n_done += 1
        if n_done % 100 == 0:
            print(f"  {n_done} tiles converted")
    print(f"Done. {n_done} tiles written to {out}, {n_skipped} held-out tiles skipped.")


if __name__ == "__main__":
    main()
