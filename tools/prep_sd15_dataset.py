#!/usr/bin/env python3
"""Convert (cond_*.npy, road_*.npy) tile pairs into the paired-PNG layout for
SD 1.5 ControlNet training at 512×512.

Cond encoding (7 → 4-channel RGBA):
    R = elevation (per-tile-normalized, [0,1])
    G = 0.3·water + 0.5·parkland + 0.2·agricultural   (cond[1,5,6])
    B = 0.6·residential + 0.8·commercial + 1.0·industrial  (cond[2,3,4])
    A = road skeleton: 0.25·road[1] + 0.5·road[2] + 0.75·road[3] + 1.0·road[4]
        (encodes existing road hierarchy; 0 = no road)

Target encoding (5 → 3-channel RGB): fixed palette
    bg/0=black, residential/1=red, tertiary/2=green, primary/3=blue,
    motorway/4=yellow

Tiles are already 512×512 — no resize needed.

Usage:
    python tools/prep_sd15_dataset.py \\
        --src data/ \\
        --dst data/sd15_cnet \\
        --style us_suburb \\
        --holdout-cities irving_tx
"""
import argparse
import glob
from pathlib import Path

import cv2
import numpy as np

# Import shared PROMPTS and PALETTE from the flux script.
# Fall back to inline copies if run outside the package.
try:
    from data_pipeline.prep_flux_dataset import PALETTE, PROMPTS
except ImportError:
    PALETTE = np.array(
        [
            [0, 0, 0],       # 0 bg
            [255, 0, 0],     # 1 residential
            [0, 255, 0],     # 2 tertiary
            [0, 0, 255],     # 3 primary
            [255, 255, 0],   # 4 motorway
        ],
        dtype=np.uint8,
    )

    PROMPTS = {
        "us_suburb":        "top-down satellite-style raster of a US suburban road network, high-contrast color-coded road class map, flat color, vector style, no texture, no shading",
        "us_grid":          "top-down satellite-style raster of an American urban grid city road network, high-contrast color-coded road class map, flat color, vector style, no texture, no shading",
        "us_arterial":      "top-down satellite-style raster of an American inner suburban arterial grid road network, high-contrast color-coded road class map, flat color, vector style, no texture, no shading",
        "euro_grid":        "top-down satellite-style raster of a European planned grid city road network, high-contrast color-coded road class map, flat color, vector style, no texture, no shading",
        "medieval_organic": "top-down satellite-style raster of a medieval European road network, high-contrast color-coded road class map, flat color, vector style, no texture, no shading",
        "soviet_microrayon":"top-down satellite-style raster of a Soviet housing estate road network, high-contrast color-coded road class map, flat color, vector style, no texture, no shading",
        "latam_colonial":   "top-down satellite-style raster of a Latin American colonial grid city road network, high-contrast color-coded road class map, flat color, vector style, no texture, no shading",
        "latam_informal":   "top-down satellite-style raster of a Latin American informal settlement road network, high-contrast color-coded road class map, flat color, vector style, no texture, no shading",
        "africa_informal":  "top-down satellite-style raster of a sub-Saharan African informal settlement road network, high-contrast color-coded road class map, flat color, vector style, no texture, no shading",
        "africa_township":  "top-down satellite-style raster of a sub-Saharan African township road network, high-contrast color-coded road class map, flat color, vector style, no texture, no shading",
        "east_asian_dense": "top-down satellite-style raster of a dense East Asian city road network, high-contrast color-coded road class map, flat color, vector style, no texture, no shading",
    }


def _safe_ch(arr, idx):
    """Return channel `idx` of arr, or a zero plane if idx is out of range."""
    if arr.shape[0] > idx:
        return arr[idx]
    return np.zeros(arr.shape[1:], dtype=np.float32)


def encode_cond(cond, road):
    """(C, 512, 512), (5, 512, 512) -> (512, 512, 4) uint8 RGBA.

    cond may have fewer than 7 channels (older pipeline); missing channels
    are treated as zero.
    """
    # R: elevation, per-tile normalized
    elev = cond[0].astype(np.float32)
    R = (elev - elev.min()) / (np.ptp(elev) + 1e-6)

    # G: water + parkland + agricultural
    G = (0.3 * _safe_ch(cond, 1)
         + 0.5 * _safe_ch(cond, 5)
         + 0.2 * _safe_ch(cond, 6))

    # B: residential + commercial + industrial
    B = (0.6 * _safe_ch(cond, 2)
         + 0.8 * _safe_ch(cond, 3)
         + 1.0 * _safe_ch(cond, 4))

    # A: road skeleton — encode hierarchy into a single alpha channel
    A = (0.25 * road[1]
         + 0.50 * road[2]
         + 0.75 * road[3]
         + 1.00 * road[4])

    rgba = np.stack([R, G, B, A], axis=-1).clip(0, 1)
    rgba_u8 = (rgba * 255).astype(np.uint8)
    return rgba_u8  # (512, 512, 4) in RGBA order


def encode_target(road):
    """(5, 512, 512) -> (512, 512, 3) uint8 RGB."""
    cls = road.argmax(axis=0).astype(np.uint8)
    return PALETTE[cls]


def main():
    p = argparse.ArgumentParser(
        description="Prepare SD 1.5 ControlNet dataset at 512×512 with 4-ch RGBA cond."
    )
    p.add_argument("--src", default="data/", help="Root data dir with city subdirs.")
    p.add_argument("--dst", default="data/sd15_cnet/")
    p.add_argument("--holdout-cities", nargs="*", default=["irving_tx"],
                   help="Cities to skip (held out for eval).")
    p.add_argument("--style", default="us_suburb", choices=list(PROMPTS.keys()))
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

        # Cond: RGBA — cv2.imwrite expects BGRA, so reverse R↔B (keep A)
        rgba = encode_cond(cond, road)          # (H, W, 4) RGBA uint8
        bgra = rgba[:, :, [2, 1, 0, 3]]         # BGRA for cv2
        cv2.imwrite(str(out / "cond" / f"{city}_{rid}.png"), bgra)

        # Target: RGB — cv2 expects BGR
        rgb = encode_target(road)               # (H, W, 3) RGB uint8
        cv2.imwrite(str(out / "target" / f"{city}_{rid}.png"), rgb[:, :, ::-1])

        (out / "meta" / f"{city}_{rid}.txt").write_text(PROMPTS[args.style])
        n_done += 1
        if n_done % 100 == 0:
            print(f"  {n_done} tiles converted")

    print(
        f"Done. {n_done} tiles written to {out}, "
        f"{n_skipped} held-out tiles skipped."
    )


if __name__ == "__main__":
    main()
