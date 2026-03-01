import numpy as np
from typing import List, Tuple

def generate_tile_centers(
    bbox_m: Tuple[float, float, float, float],
    tile_size_m: float,
    n_tiles: int,
    jitter_fraction: float = 0.3,
    seed: int = None
) -> List[Tuple[float, float, float]]:
    """
    Returns list of (x_m, y_m, rotation_deg) for non-overlapping tiles.
    bbox_m: (west, south, east, north) in projected meters.
    """
    rng = np.random.default_rng(seed)
    west, south, east, north = bbox_m
    width = east - west
    height = north - south

    # Grid cell size = tile_size_m (no overlap)
    cols = int(width / tile_size_m)
    rows = int(height / tile_size_m)

    centers = []
    for row in range(rows):
        for col in range(cols):
            # Cell center
            cx = west + (col + 0.5) * tile_size_m
            cy = south + (row + 0.5) * tile_size_m
            # Random jitter within +-jitter_fraction of cell size
            jitter = tile_size_m * jitter_fraction
            cx += rng.uniform(-jitter, jitter)
            cy += rng.uniform(-jitter, jitter)
            # Random rotation, continuous [0, 360)
            rotation = rng.uniform(0, 360)
            centers.append((float(cx), float(cy), float(rotation)))

    rng.shuffle(centers)
    return centers[:n_tiles]
