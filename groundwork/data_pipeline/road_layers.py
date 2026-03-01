import numpy as np
import os, math, pickle
import osmnx as ox
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from scipy.ndimage import binary_dilation

ROAD_CHANNELS = {
    1: ["residential", "unclassified", "living_street", "service"],
    2: ["tertiary", "tertiary_link"],
    3: ["primary", "primary_link", "secondary", "secondary_link"],
    4: ["motorway", "motorway_link", "trunk", "trunk_link"],
}


def _bbox_latlon(center_lon, center_lat, grid_size_px, pixel_size_m):
    half_m = (grid_size_px / 2) * pixel_size_m * math.sqrt(2)
    lat_deg = half_m / 111_000
    lon_deg = half_m / (111_000 * math.cos(math.radians(center_lat)))
    return (center_lon - lon_deg, center_lat - lat_deg,
            center_lon + lon_deg, center_lat + lat_deg)


def fetch_road_graph(center_lon, center_lat, grid_size_px, pixel_size_m, cache_dir="osm_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"roads_{center_lat:.4f}_{center_lon:.4f}_{grid_size_px}.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    west, south, east, north = _bbox_latlon(center_lon, center_lat, grid_size_px, pixel_size_m)
    cf = '["highway"~"motorway|trunk|primary|secondary|tertiary|residential|unclassified|living_street|service|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link"]'
    try:
        G = ox.graph_from_bbox((west, south, east, north), custom_filter=cf)
    except Exception:
        G = None
    with open(cache, "wb") as f:
        pickle.dump(G, f)
    return G


def _get_edge_geometries_by_level(G, west, south, east, north):
    from shapely.geometry import LineString
    if G is None:
        return {ch: [] for ch in range(1, 5)}
    result = {ch: [] for ch in range(1, 5)}
    for u, v, data in G.edges(data=True):
        hw = data.get("highway", "")
        if isinstance(hw, list):
            hw = hw[0]
        geom = data.get("geometry", None)
        if geom is None:
            u_data = G.nodes[u]
            v_data = G.nodes[v]
            geom = LineString([(u_data["x"], u_data["y"]), (v_data["x"], v_data["y"])])
        for ch, types in ROAD_CHANNELS.items():
            if hw in types:
                result[ch].append(geom)
                break
    return result


def rasterize_roads_binary(G, center_lon, center_lat, grid_size_px, pixel_size_m,
                            line_width_px=3):
    west, south, east, north = _bbox_latlon(center_lon, center_lat, grid_size_px, pixel_size_m)
    transform = from_bounds(west, south, east, north, grid_size_px, grid_size_px)
    by_level = _get_edge_geometries_by_level(G, west, south, east, north)
    all_geoms = [g for geoms in by_level.values() for g in geoms]
    if not all_geoms:
        return np.zeros((grid_size_px, grid_size_px), dtype=np.float32)
    shapes = [(g.__geo_interface__, 1.0) for g in all_geoms]
    arr = rasterize(shapes, out_shape=(grid_size_px, grid_size_px),
                    transform=transform, fill=0.0, dtype=np.float32)
    struct = np.ones((line_width_px, line_width_px), dtype=bool)
    arr = binary_dilation(arr.astype(bool), structure=struct).astype(np.float32)
    return arr


def rasterize_road_output(G, center_lon, center_lat, grid_size_px, pixel_size_m,
                           line_width_px=5):
    west, south, east, north = _bbox_latlon(center_lon, center_lat, grid_size_px, pixel_size_m)
    transform = from_bounds(west, south, east, north, grid_size_px, grid_size_px)
    by_level = _get_edge_geometries_by_level(G, west, south, east, north)

    priority = np.zeros((grid_size_px, grid_size_px), dtype=np.int32)
    struct = np.ones((line_width_px, line_width_px), dtype=bool)
    for ch in [1, 2, 3, 4]:
        if not by_level[ch]:
            continue
        shapes = [(g.__geo_interface__, 1) for g in by_level[ch]]
        layer = rasterize(shapes, out_shape=(grid_size_px, grid_size_px),
                          transform=transform, fill=0, dtype=np.int32)
        layer = binary_dilation(layer.astype(bool), structure=struct).astype(np.int32)
        priority = np.where(layer > 0, ch, priority)

    one_hot = (np.arange(5)[:, None, None] == priority[None]).astype(np.float32)
    return one_hot
