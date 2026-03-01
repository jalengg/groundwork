import gzip
import math
import os
from urllib.request import urlretrieve

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject

_SRTM1_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi/{ns}{lat:02d}/{name}.hgt.gz"


def _srtm_tile(lat: float, lon: float):
    """Returns (tile_name, lat_floor, lon_floor, ns, ew) for the SRTM tile covering (lat, lon)."""
    lat_i = int(math.floor(lat))
    lon_i = int(math.floor(lon))
    ns = "N" if lat_i >= 0 else "S"
    ew = "E" if lon_i >= 0 else "W"
    name = f"{ns}{abs(lat_i):02d}{ew}{abs(lon_i):03d}"
    return name, lat_i, lon_i, ns, ew


def _download_hgt(name: str, lat_i: int, ns: str, cache_dir: str) -> str:
    """Downloads and decompresses an SRTM1 HGT tile. Returns path to .hgt file."""
    hgt_path = os.path.join(cache_dir, f"{name}.hgt")
    if os.path.exists(hgt_path):
        return hgt_path
    gz_path = hgt_path + ".gz"
    url = _SRTM1_URL.format(ns=ns, lat=abs(lat_i), name=name)
    urlretrieve(url, gz_path)
    with gzip.open(gz_path, "rb") as f_in, open(hgt_path, "wb") as f_out:
        f_out.write(f_in.read())
    os.remove(gz_path)
    return hgt_path


def fetch_elevation_grid(
    center_lon: float,
    center_lat: float,
    grid_size_px: int,
    pixel_size_m: float,
    cache_dir: str = "srtm_cache",
) -> np.ndarray:
    """
    Returns float32 array of shape (grid_size_px, grid_size_px), elevation
    values normalized 0-1 within the tile.

    Downloads SRTM1 (30m) HGT files directly from AWS and reprojects via
    rasterio bilinear resampling. The bounding box is oversized by sqrt(2) so
    the tile assembler can rotate and center-crop safely.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Bounding box oversized by sqrt(2) for rotation safety
    half_m = (grid_size_px / 2) * pixel_size_m * math.sqrt(2)
    lat_deg = half_m / 111_000
    lon_deg = half_m / (111_000 * math.cos(math.radians(center_lat)))
    west = center_lon - lon_deg
    east = center_lon + lon_deg
    south = center_lat - lat_deg
    north = center_lat + lat_deg

    name, lat_i, _lon_i, ns, _ew = _srtm_tile(center_lat, center_lon)
    hgt_path = _download_hgt(name, lat_i, ns, cache_dir)

    target = np.zeros((grid_size_px, grid_size_px), dtype=np.float32)
    target_transform = from_bounds(west, south, east, north, grid_size_px, grid_size_px)

    with rasterio.open(hgt_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=target,
            dst_transform=target_transform,
            dst_crs=CRS.from_epsg(4326),
            resampling=Resampling.bilinear,
        )

    vmin, vmax = target.min(), target.max()
    if vmax > vmin:
        target = (target - vmin) / (vmax - vmin)
    else:
        target[:] = 0.0
    return target
