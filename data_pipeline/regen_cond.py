#!/usr/bin/env python3
"""
Regenerate cond_*.npy AND road_*.npy files in-place using the current
assemble_tile() output. Iterates meta_*.json files for each tile (which
preserve lat/lon/rotation), calls assemble_tile() with cached OSM/SRTM
data, and overwrites both files.

Usage:
    python -m data_pipeline.regen_cond --data data/ [--city arlington_tx]
"""
import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np
import yaml
from tqdm import tqdm

from data_pipeline.tile_assembler import assemble_tile


# Channel count for the new format. Tiles whose cond_*.npy already has this
# many channels are skipped (already regenerated).
NEW_COND_CHANNELS = 7


def _process_tile(task):
    """Worker for multiprocessing pool. Returns (idx, status) where
    status is 'ok', 'skip', or an error message."""
    mf, tile_size, pixel_size, osm_cache, srtm_cache = task
    idx = os.path.basename(mf)[5:9]
    city_dir = os.path.dirname(mf)
    cond_path = os.path.join(city_dir, f"cond_{idx}.npy")
    road_path = os.path.join(city_dir, f"road_{idx}.npy")

    # Skip if cond is already in the new format
    if os.path.exists(cond_path):
        try:
            existing = np.load(cond_path, mmap_mode="r")
            if existing.shape[0] == NEW_COND_CHANNELS:
                return idx, "skip"
        except Exception:
            pass

    with open(mf) as f:
        meta = json.load(f)
    try:
        cond, road = assemble_tile(
            meta["lon"], meta["lat"], meta["rotation_deg"],
            tile_size, pixel_size, osm_cache, srtm_cache,
        )
        np.save(cond_path, cond)
        np.save(road_path, road)
        return idx, "ok"
    except Exception as e:
        return idx, f"FAIL: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data_pipeline/cities.yaml")
    parser.add_argument("--data", default="data")
    parser.add_argument("--city", default=None, help="Process only this city")
    parser.add_argument("--osm-cache", default="osm_cache")
    parser.add_argument("--srtm-cache", default="srtm_cache")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    tile_size = cfg["tile_size_px"]
    pixel_size = cfg["pixel_size_m"]

    cities = sorted(glob.glob(os.path.join(args.data, "*")))
    if args.city:
        cities = [c for c in cities if os.path.basename(c) == args.city]

    for city_dir in cities:
        city = os.path.basename(city_dir)
        meta_files = sorted(glob.glob(os.path.join(city_dir, "meta_*.json")))
        if not meta_files:
            print(f"  {city}: no meta files, skipping")
            continue

        tasks = [(mf, tile_size, pixel_size, args.osm_cache, args.srtm_cache) for mf in meta_files]
        print(f"\n=== {city}: {len(tasks)} tiles, {args.workers} workers ===")

        ok = skipped = failed = 0
        with Pool(args.workers) as pool:
            for idx, status in tqdm(pool.imap_unordered(_process_tile, tasks),
                                    total=len(tasks), desc=city):
                if status == "ok":
                    ok += 1
                elif status == "skip":
                    skipped += 1
                else:
                    failed += 1
                    tqdm.write(f"  {idx}: {status}")
        print(f"  {city}: ok={ok}  skipped={skipped}  failed={failed}")


if __name__ == "__main__":
    main()
