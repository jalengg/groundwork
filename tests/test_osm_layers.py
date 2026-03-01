import numpy as np
from data_pipeline.osm_layers import fetch_water_grid, fetch_landuse_grid

def test_water_grid_is_binary():
    arr = fetch_water_grid(
        center_lon=-97.108, center_lat=32.735,
        grid_size_px=64, pixel_size_m=5,
        cache_dir="/tmp/osm_test_cache"
    )
    assert arr.shape == (64, 64)
    assert arr.dtype == np.float32
    assert set(np.unique(arr)).issubset({0.0, 1.0})

def test_landuse_grid_values_in_range():
    arr = fetch_landuse_grid(
        center_lon=-97.108, center_lat=32.735,
        grid_size_px=64, pixel_size_m=5,
        cache_dir="/tmp/osm_test_cache"
    )
    assert arr.shape == (64, 64)
    assert arr.dtype == np.float32
    assert arr.min() >= 0.0 and arr.max() <= 1.0
