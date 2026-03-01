import numpy as np
from data_pipeline.elevation_layer import fetch_elevation_grid

def test_elevation_grid_shape_and_range():
    arr = fetch_elevation_grid(
        center_lon=-97.108, center_lat=32.735,
        grid_size_px=64,
        pixel_size_m=5,
        cache_dir="/tmp/srtm_test_cache"
    )
    assert arr.shape == (64, 64)
    assert arr.dtype == np.float32
    # Values normalized 0-1 within tile
    assert 0.0 <= arr.min() <= arr.max() <= 1.0
    # Not all zeros — there is real elevation data here
    assert arr.max() > 0.0
