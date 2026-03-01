import numpy as np
import os, math, pickle
import osmnx as ox
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_bounds

LANDUSE_VALUES = {
    "residential": 0.2, "apartments": 0.2,
    "commercial": 0.4, "retail": 0.4,
    "industrial": 0.6, "warehouse": 0.6,
    "park": 0.8, "recreation_ground": 0.8,
    "nature_reserve": 0.8, "forest": 0.8,
    "grass": 0.7, "meadow": 0.7,
    "farmland": 0.5, "farmyard": 0.5,
}
WATER_TAGS = {"natural": ["water", "wetland"], "waterway": True, "landuse": ["reservoir", "basin"]}
LANDUSE_TAGS = {"landuse": True, "leisure": ["park", "recreation_ground", "golf_course"]}


def _bbox_latlon(center_lon, center_lat, grid_size_px, pixel_size_m):
    """Returns (west, south, east, north) in WGS84 degrees, oversized by sqrt(2) for rotation."""
    half_m = (grid_size_px / 2) * pixel_size_m * math.sqrt(2)
    lat_deg = half_m / 111_000
    lon_deg = half_m / (111_000 * math.cos(math.radians(center_lat)))
    return (center_lon - lon_deg, center_lat - lat_deg,
            center_lon + lon_deg, center_lat + lat_deg)


def _cache_path(cache_dir, prefix, center_lon, center_lat, grid_size_px):
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{prefix}_{center_lat:.4f}_{center_lon:.4f}_{grid_size_px}.pkl")


def fetch_water_grid(center_lon, center_lat, grid_size_px, pixel_size_m, cache_dir="osm_cache"):
    west, south, east, north = _bbox_latlon(center_lon, center_lat, grid_size_px, pixel_size_m)
    cache = _cache_path(cache_dir, "water", center_lon, center_lat, grid_size_px)

    if os.path.exists(cache):
        with open(cache, "rb") as f:
            gdf = pickle.load(f)
    else:
        try:
            gdf = ox.features_from_bbox((west, south, east, north), tags=WATER_TAGS)
            gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        except Exception:
            gdf = gpd.GeoDataFrame(geometry=[])
        with open(cache, "wb") as f:
            pickle.dump(gdf, f)

    transform = from_bounds(west, south, east, north, grid_size_px, grid_size_px)
    if len(gdf) == 0:
        return np.zeros((grid_size_px, grid_size_px), dtype=np.float32)

    shapes = [(geom.__geo_interface__, 1.0) for geom in gdf.geometry if geom is not None]
    arr = rasterize(shapes, out_shape=(grid_size_px, grid_size_px),
                    transform=transform, fill=0.0, dtype=np.float32)
    return arr


def fetch_landuse_grid(center_lon, center_lat, grid_size_px, pixel_size_m, cache_dir="osm_cache"):
    west, south, east, north = _bbox_latlon(center_lon, center_lat, grid_size_px, pixel_size_m)
    cache = _cache_path(cache_dir, "landuse", center_lon, center_lat, grid_size_px)

    if os.path.exists(cache):
        with open(cache, "rb") as f:
            gdf = pickle.load(f)
    else:
        try:
            gdf = ox.features_from_bbox((west, south, east, north), tags=LANDUSE_TAGS)
            gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        except Exception:
            gdf = gpd.GeoDataFrame(geometry=[])
        with open(cache, "wb") as f:
            pickle.dump(gdf, f)

    transform = from_bounds(west, south, east, north, grid_size_px, grid_size_px)
    arr = np.zeros((grid_size_px, grid_size_px), dtype=np.float32)
    if len(gdf) == 0:
        return arr

    # Render lowest-priority landuse first; higher values overwrite
    sort_col = "landuse" if "landuse" in gdf.columns else gdf.columns[0]
    for _, row in gdf.sort_values(by=sort_col).iterrows():
        geom = row.geometry
        if geom is None:
            continue
        tag_val = row.get("landuse", row.get("leisure", "other"))
        value = LANDUSE_VALUES.get(str(tag_val), 0.1)
        patch = rasterize([(geom.__geo_interface__, value)],
                          out_shape=(grid_size_px, grid_size_px),
                          transform=transform, fill=0.0, dtype=np.float32)
        arr = np.where(patch > 0, patch, arr)
    return arr
