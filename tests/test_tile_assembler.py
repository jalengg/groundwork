import numpy as np
from data_pipeline.tile_assembler import assemble_tile


def test_assemble_tile_shapes():
    cond, road = assemble_tile(
        center_lon=-97.108, center_lat=32.735,
        rotation_deg=45.0,
        tile_size_px=64,
        pixel_size_m=5,
        osm_cache_dir="/tmp/osm_test_cache",
        srtm_cache_dir="/tmp/srtm_test_cache",
    )
    assert cond.shape == (4, 64, 64), f"Expected (4,64,64), got {cond.shape}"
    assert road.shape == (5, 64, 64), f"Expected (5,64,64), got {road.shape}"
    assert cond.dtype == np.float32
    assert road.dtype == np.float32
    # Conditioning channels in [0, 1]
    assert cond.min() >= 0.0 and cond.max() <= 1.0
    # Road output is one-hot
    assert np.allclose(road.sum(axis=0), 1.0)
