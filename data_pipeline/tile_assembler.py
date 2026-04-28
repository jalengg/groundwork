import math

import numpy as np
from scipy.ndimage import rotate as scipy_rotate

from data_pipeline.elevation_layer import fetch_elevation_grid
from data_pipeline.osm_layers import (
    LANDUSE_CATEGORIES,
    fetch_landuse_grid_categorical,
    fetch_water_grid,
)
from data_pipeline.road_layers import (
    fetch_road_graph,
    rasterize_road_output,
)


def _rotate_and_crop(arr: np.ndarray, angle_deg: float, target_size: int) -> np.ndarray:
    """Rotate a (H, W) or (C, H, W) array and center-crop to target_size × target_size."""
    if arr.ndim == 2:
        rotated = scipy_rotate(arr, angle=angle_deg, reshape=False, order=1, cval=0.0)
        h, w = rotated.shape
        top = (h - target_size) // 2
        left = (w - target_size) // 2
        return rotated[top : top + target_size, left : left + target_size]
    return np.stack(
        [_rotate_and_crop(arr[c], angle_deg, target_size) for c in range(arr.shape[0])]
    )


def assemble_tile(
    center_lon: float,
    center_lat: float,
    rotation_deg: float,
    tile_size_px: int,
    pixel_size_m: float,
    osm_cache_dir: str = "osm_cache",
    srtm_cache_dir: str = "srtm_cache",
):
    """
    Assemble one training tile.

    Fetches all layers at an oversized resolution (tile_size_px × √2), rotates
    each by rotation_deg, and center-crops to tile_size_px × tile_size_px so
    that all channels are pixel-perfectly aligned at any rotation angle.

    Returns
    -------
    cond : float32 ndarray, shape (2 + len(LANDUSE_CATEGORIES), tile_size_px, tile_size_px)
        Conditioning channels: [elevation, water, landuse_one_hot...]
        Landuse channel order is determined by LANDUSE_CATEGORIES.
    road : float32 ndarray, shape (5, tile_size_px, tile_size_px)
        One-hot road output: [background, residential, tertiary,
                               primary/secondary, motorway/trunk]
    """
    oversized = int(tile_size_px * math.sqrt(2)) + 4

    elev = fetch_elevation_grid(center_lon, center_lat, oversized, pixel_size_m, srtm_cache_dir)
    water = fetch_water_grid(center_lon, center_lat, oversized, pixel_size_m, osm_cache_dir)
    landuse_oh = fetch_landuse_grid_categorical(center_lon, center_lat, oversized, pixel_size_m, osm_cache_dir)
    G = fetch_road_graph(center_lon, center_lat, oversized, pixel_size_m, osm_cache_dir)
    roads_output = rasterize_road_output(G, center_lon, center_lat, oversized, pixel_size_m)

    # Conditioning: [elev, water, landuse_one_hot...] = 2 + len(LANDUSE_CATEGORIES) channels
    cond_oversized = np.concatenate(
        [elev[None], water[None], landuse_oh], axis=0
    )

    # Rotate and crop all channels together
    cond = _rotate_and_crop(cond_oversized, rotation_deg, tile_size_px)
    road = _rotate_and_crop(roads_output, rotation_deg, tile_size_px)

    # Re-enforce one-hot after rotation (bilinear interpolation softens edges)
    road_argmax = road.argmax(axis=0)
    road = (np.arange(5)[:, None, None] == road_argmax[None]).astype(np.float32)

    cond = np.clip(cond, 0.0, 1.0).astype(np.float32)
    return cond, road
