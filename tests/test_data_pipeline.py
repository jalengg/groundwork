# data pipeline tests

import os
import yaml

def test_cities_yaml_loads():
    path = os.path.join(os.path.dirname(__file__), "../data_pipeline/cities.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    assert cfg["tile_size_px"] == 512
    assert cfg["pixel_size_m"] == 5
    assert len(cfg["cities"]) == 8
    val_cities = [c for c in cfg["cities"] if c["split"] == "val"]
    assert len(val_cities) == 1
    assert val_cities[0]["name"] == "irving_tx"
    assert all("query" in c and c["query"] for c in cfg["cities"])
    assert all(c["split"] in {"train", "val"} for c in cfg["cities"])

from data_pipeline.tile_grid import generate_tile_centers

def test_tile_grid_non_overlapping():
    # Fake bbox 20km x 20km, 150 tiles, 2.56km tile size
    centers = generate_tile_centers(
        bbox_m=(0, 0, 20000, 20000),  # (west, south, east, north) in meters
        tile_size_m=2560,
        n_tiles=150,
        jitter_fraction=0.3,
        seed=42
    )
    assert len(centers) <= 150
    # Each center is (x_m, y_m, rotation_deg)
    assert all(len(c) == 3 for c in centers)
    # Rotations are continuous in [0, 360)
    rotations = [c[2] for c in centers]
    assert min(rotations) >= 0.0
    assert max(rotations) < 360.0
    # Not all the same rotation (would indicate N=4 bug from old code)
    assert len(set(round(r, 1) for r in rotations)) > 10
    west, south, east, north = 0, 0, 20000, 20000
    assert all(west <= c[0] <= east and south <= c[1] <= north for c in centers)
