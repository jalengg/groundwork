#!/usr/bin/env python3
"""
Generate training tiles for all cities in cities.yaml.

Usage:
    python cdg.py --config data_pipeline/cities.yaml --output data/ [--city arlington_tx]
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import osmnx as ox
import yaml
from tqdm import tqdm

from data_pipeline.tile_assembler import assemble_tile
from data_pipeline.tile_grid import generate_tile_centers


def main():
    parser = argparse.ArgumentParser(description="Generate Groundwork training tiles")
    parser.add_argument("--config", default="data_pipeline/cities.yaml")
    parser.add_argument("--output", default="data")
    parser.add_argument("--city", default=None, help="Process only this city name")
    parser.add_argument("--osm-cache", default="osm_cache")
    parser.add_argument("--srtm-cache", default="srtm_cache")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tile_size = cfg["tile_size_px"]
    pixel_size = cfg["pixel_size_m"]
    n_tiles = cfg["n_tiles_per_city"]
    jitter = cfg["jitter_fraction"]

    for city in cfg["cities"]:
        if args.city and city["name"] != args.city:
            continue
        print(f"\n=== {city['name']} ({city['split']}) ===")
        out_dir = os.path.join(args.output, city["name"])
        os.makedirs(out_dir, exist_ok=True)

        # Get city bbox from OSMnx geocoder
        place = ox.geocode_to_gdf(city["query"])
        bounds = place.total_bounds  # (minx, miny, maxx, maxy) in WGS84

        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        lat_to_m = 111_000
        lon_to_m = 111_000 * math.cos(math.radians(center_lat))

        # Convert WGS84 bbox to local meters (relative to city center)
        bbox_m = (
            (bounds[0] - center_lon) * lon_to_m,
            (bounds[1] - center_lat) * lat_to_m,
            (bounds[2] - center_lon) * lon_to_m,
            (bounds[3] - center_lat) * lat_to_m,
        )

        tile_size_m = tile_size * pixel_size

        # Loop until we reach n_tiles — each pass uses fresh random centers (augmentation
        # via different jitter+rotation), so small cities can exceed their grid-cell cap.
        while True:
            existing = glob.glob(os.path.join(out_dir, "cond_*.npy"))
            n_existing = len(existing)
            if n_existing >= n_tiles:
                print(f"  {city['name']}: {n_existing} tiles — target reached.")
                break

            # Start writing after the highest existing index to avoid overwriting or infinite
            # loops caused by gaps (failed tiles leave holes in the sequence).
            if existing:
                max_idx = max(int(os.path.basename(f)[5:9]) for f in existing)
                start_idx = max_idx + 1
            else:
                start_idx = 0

            need = n_tiles - n_existing
            centers = generate_tile_centers(bbox_m, tile_size_m, need, jitter)
            print(f"  {city['name']}: {start_idx}/{n_tiles} — generating {len(centers)} more")

            for i, (cx_m, cy_m, rot) in enumerate(tqdm(centers, desc=city["name"])):
                tile_idx = start_idx + i
                cond_path = os.path.join(out_dir, f"cond_{tile_idx:04d}.npy")
                road_path = os.path.join(out_dir, f"road_{tile_idx:04d}.npy")
                meta_path = os.path.join(out_dir, f"meta_{tile_idx:04d}.json")
                if os.path.exists(cond_path):
                    continue

                # Convert local meters back to lat/lon
                lon = center_lon + cx_m / lon_to_m
                lat = center_lat + cy_m / lat_to_m

                try:
                    cond, road = assemble_tile(
                        lon, lat, rot, tile_size, pixel_size,
                        args.osm_cache, args.srtm_cache,
                    )
                    np.save(cond_path, cond)
                    np.save(road_path, road)
                    with open(meta_path, "w") as f:
                        json.dump(
                            {"lat": lat, "lon": lon, "rotation_deg": rot,
                             "city": city["name"], "split": city["split"]},
                            f,
                        )
                except Exception as e:
                    print(f"  Skipping tile {tile_idx}: {e}")


if __name__ == "__main__":
    main()
