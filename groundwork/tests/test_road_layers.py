import numpy as np
from data_pipeline.road_layers import fetch_road_graph, rasterize_roads_binary, rasterize_road_output

def test_road_binary_is_binary():
    G = fetch_road_graph(center_lon=-97.108, center_lat=32.735,
                         grid_size_px=64, pixel_size_m=5,
                         cache_dir="/tmp/osm_test_cache")
    arr = rasterize_roads_binary(G, center_lon=-97.108, center_lat=32.735,
                                  grid_size_px=64, pixel_size_m=5)
    assert arr.shape == (64, 64)
    assert arr.dtype == np.float32
    assert set(np.unique(arr)).issubset({0.0, 1.0})
    assert arr.sum() > 0  # there are roads in Arlington

def test_road_output_is_one_hot():
    G = fetch_road_graph(center_lon=-97.108, center_lat=32.735,
                         grid_size_px=64, pixel_size_m=5,
                         cache_dir="/tmp/osm_test_cache")
    arr = rasterize_road_output(G, center_lon=-97.108, center_lat=32.735,
                                 grid_size_px=64, pixel_size_m=5)
    assert arr.shape == (5, 64, 64)
    assert arr.dtype == np.float32
    # One-hot: each pixel sums to exactly 1 across channels
    assert np.allclose(arr.sum(axis=0), 1.0)
    # Road pixels exist (non-background channels have content)
    assert arr[1:].sum() > 0
    # Background exists too (not entirely road)
    assert arr[0].sum() > 0
