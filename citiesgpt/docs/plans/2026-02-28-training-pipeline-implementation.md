# CitiesGPT Training Pipeline — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a data pipeline and two-stage diffusion model (VAE + U-Net with CDB) that generates realistic US suburb road networks conditioned on elevation, land use, water, and existing roads.

**Architecture:** Rasterio-based multi-layer tile generator feeds a CaRoLS-faithful model: a VAE trained on 5-channel one-hot road layouts, followed by a DDPM diffusion U-Net with Condition-aware Decoder Blocks (LDE + GCI attention) trained on 4-channel conditioning images. Evaluation includes FID/KID/CI/TC metrics plus a VLM realism scorer using the Claude API.

**Tech Stack:** Python 3.10, PyTorch 2.x, OSMnx, rasterio, GeoPandas, shapely, elevation (SRTM), scipy, scikit-image, sknw, torchmetrics, anthropic SDK, Google Colab A100.

**Design doc:** `docs/plans/2026-02-28-training-pipeline-design.md`

---

## Part A — Project Setup

### Task 1: Scaffold project structure and requirements

**Files:**
- Create: `citiesgpt/requirements.txt`
- Create: `citiesgpt/.gitignore`
- Create: `citiesgpt/data_pipeline/__init__.py` (empty)
- Create: `citiesgpt/model/__init__.py` (empty)
- Create: `citiesgpt/tests/__init__.py` (empty)
- Create: `citiesgpt/tests/test_data_pipeline.py` (empty for now)
- Create: `citiesgpt/tests/test_model.py` (empty for now)
- Create: `citiesgpt/notebooks/` (empty dir, add `.gitkeep`)

**Step 1:** Create `requirements.txt`:
```
torch>=2.0.0
torchvision
numpy
rasterio>=1.3.0
geopandas
osmnx>=1.6.0
shapely
elevation
scipy
scikit-image
sknw
networkx
rdp
torchmetrics[image]
anthropic
pyyaml
tqdm
matplotlib
pyproj
affine
pytest
```

**Step 2:** Create `.gitignore`:
```
data/
checkpoints/
*.npy
*.tif
__pycache__/
*.pyc
*.egg-info/
.pytest_cache/
srtm_cache/
osm_cache/
```

**Step 3:** Create all empty `__init__.py` files and `.gitkeep`.

**Step 4:** Verify install locally:
```bash
cd citiesgpt
pip install -r requirements.txt
python -c "import rasterio, osmnx, torch; print('OK')"
```
Expected: `OK`

**Step 5:** Commit:
```bash
git add citiesgpt/
git commit -m "chore: scaffold citiesgpt project structure"
```

---

### Task 2: City configuration (`data_pipeline/cities.yaml`)

**Files:**
- Create: `citiesgpt/data_pipeline/cities.yaml`

**Step 1:** Create the config:
```yaml
tile_size_px: 512
pixel_size_m: 5
n_tiles_per_city: 150
jitter_fraction: 0.3
style: us_suburb

cities:
  - name: arlington_tx
    query: "Arlington, Texas, USA"
    split: train
  - name: chandler_az
    query: "Chandler, Arizona, USA"
    split: train
  - name: gilbert_az
    query: "Gilbert, Arizona, USA"
    split: train
  - name: henderson_nv
    query: "Henderson, Nevada, USA"
    split: train
  - name: mesa_az
    query: "Mesa, Arizona, USA"
    split: train
  - name: tempe_az
    query: "Tempe, Arizona, USA"
    split: train
  - name: plano_tx
    query: "Plano, Texas, USA"
    split: train
  - name: irving_tx
    query: "Irving, Texas, USA"
    split: val          # held-out validation city
```

**Step 2:** Write a quick validation test in `tests/test_data_pipeline.py`:
```python
import yaml, os

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
```

**Step 3:** Run test:
```bash
pytest tests/test_data_pipeline.py::test_cities_yaml_loads -v
```
Expected: PASS

**Step 4:** Commit:
```bash
git add data_pipeline/cities.yaml tests/test_data_pipeline.py
git commit -m "data: add city configuration"
```

---

## Part B — Data Pipeline

### Task 3: Tile grid generator (`data_pipeline/tile_grid.py`)

Generates non-overlapping tile center points with random jitter and a random rotation angle per tile. Centers are on a regular grid over the city's bounding box; jitter is added per cell to prevent a perfectly uniform pattern.

**Files:**
- Create: `citiesgpt/data_pipeline/tile_grid.py`
- Modify: `citiesgpt/tests/test_data_pipeline.py`

**Step 1:** Write the failing test:
```python
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
```

**Step 2:** Run test to verify it fails:
```bash
pytest tests/test_data_pipeline.py::test_tile_grid_non_overlapping -v
```
Expected: FAIL with `ModuleNotFoundError`

**Step 3:** Implement `tile_grid.py`:
```python
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
    max_available = cols * rows

    centers = []
    for row in range(rows):
        for col in range(cols):
            # Cell center
            cx = west + (col + 0.5) * tile_size_m
            cy = south + (row + 0.5) * tile_size_m
            # Random jitter within ±jitter_fraction of cell size
            jitter = tile_size_m * jitter_fraction
            cx += rng.uniform(-jitter, jitter)
            cy += rng.uniform(-jitter, jitter)
            # Random rotation, continuous [0, 360)
            rotation = rng.uniform(0, 360)
            centers.append((float(cx), float(cy), float(rotation)))

    rng.shuffle(centers)
    return centers[:n_tiles]
```

**Step 4:** Run test:
```bash
pytest tests/test_data_pipeline.py::test_tile_grid_non_overlapping -v
```
Expected: PASS

**Step 5:** Commit:
```bash
git add data_pipeline/tile_grid.py tests/test_data_pipeline.py
git commit -m "data: add non-overlapping tile grid generator with continuous rotation"
```

---

### Task 4: SRTM elevation layer (`data_pipeline/elevation_layer.py`)

Downloads SRTM 30m elevation for a bounding box, reprojects it onto an oversized axis-aligned grid at 5m/px using rasterio, then returns the array. The oversized grid (724×724) is later rotated and cropped to 512×512 by the assembler.

**Files:**
- Create: `citiesgpt/data_pipeline/elevation_layer.py`
- Modify: `citiesgpt/tests/test_data_pipeline.py`

**Step 1:** Write the failing test:
```python
from data_pipeline.elevation_layer import fetch_elevation_grid

def test_elevation_grid_shape_and_range():
    # Small area in Arlington TX
    arr = fetch_elevation_grid(
        center_lon=-97.108, center_lat=32.735,
        grid_size_px=64,    # use small size for test speed
        pixel_size_m=5,
        cache_dir="/tmp/srtm_test_cache"
    )
    assert arr.shape == (64, 64)
    assert arr.dtype == np.float32
    # Arlington is ~150-200m elevation
    assert 100 < arr.mean() < 300
    # Values should be normalized 0-1
    assert 0.0 <= arr.min() <= arr.max() <= 1.0
```

**Step 2:** Run test:
```bash
pytest tests/test_data_pipeline.py::test_elevation_grid_shape_and_range -v
```
Expected: FAIL

**Step 3:** Implement `elevation_layer.py`:
```python
import numpy as np
import os, math, subprocess
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from affine import Affine

def fetch_elevation_grid(
    center_lon: float, center_lat: float,
    grid_size_px: int, pixel_size_m: float,
    cache_dir: str = "srtm_cache"
) -> np.ndarray:
    """
    Returns float32 array of shape (grid_size_px, grid_size_px),
    elevation values normalized 0-1 within the tile.
    Uses SRTM 30m data via the `elevation` library.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Compute axis-aligned bbox in degrees (oversized by sqrt(2) for rotation safety)
    half_size_m = (grid_size_px / 2) * pixel_size_m * math.sqrt(2)
    # Approximate degrees: 1 deg lat ≈ 111km, 1 deg lon ≈ 111km * cos(lat)
    lat_deg = half_size_m / 111_000
    lon_deg = half_size_m / (111_000 * math.cos(math.radians(center_lat)))
    west  = center_lon - lon_deg
    east  = center_lon + lon_deg
    south = center_lat - lat_deg
    north = center_lat + lat_deg

    # Download SRTM to cache
    out_path = os.path.join(cache_dir, f"{center_lat:.4f}_{center_lon:.4f}_{grid_size_px}.tif")
    if not os.path.exists(out_path):
        import elevation as elev_lib
        elev_lib.clip(bounds=(west, south, east, north), output=out_path, product='SRTM1')
        elev_lib.clean()

    # Reproject/resample to our target grid
    target = np.zeros((grid_size_px, grid_size_px), dtype=np.float32)
    target_transform = from_bounds(west, south, east, north, grid_size_px, grid_size_px)
    with rasterio.open(out_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=target,
            dst_transform=target_transform,
            dst_crs='EPSG:4326',
            resampling=Resampling.bilinear
        )

    # Normalize 0-1 within this tile
    vmin, vmax = target.min(), target.max()
    if vmax > vmin:
        target = (target - vmin) / (vmax - vmin)
    else:
        target[:] = 0.0
    return target
```

**Step 4:** Run test (will download SRTM tile for Arlington on first run, ~2-3 min):
```bash
pytest tests/test_data_pipeline.py::test_elevation_grid_shape_and_range -v -s
```
Expected: PASS

**Step 5:** Commit:
```bash
git add data_pipeline/elevation_layer.py tests/test_data_pipeline.py
git commit -m "data: add SRTM elevation layer fetcher"
```

---

### Task 5: OSM water and land use layers (`data_pipeline/osm_layers.py`)

Fetches water polygons and land use polygons from OSM for a bounding box, rasterizes each onto the same grid, returns two float32 arrays.

**Files:**
- Create: `citiesgpt/data_pipeline/osm_layers.py`
- Modify: `citiesgpt/tests/test_data_pipeline.py`

**Step 1:** Write the failing tests:
```python
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
```

**Step 2:** Run tests to verify they fail.

**Step 3:** Implement `osm_layers.py`:
```python
import numpy as np
import os, math, pickle
import osmnx as ox
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box

# Land use category → float value encoding
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
    """Returns (west, south, east, north) in WGS84 degrees, oversized for rotation."""
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
            gdf = ox.features_from_bbox((north, south, east, west), tags=WATER_TAGS)
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
            gdf = ox.features_from_bbox((north, south, east, west), tags=LANDUSE_TAGS)
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
    for _, row in gdf.sort_values(
        by="landuse" if "landuse" in gdf.columns else gdf.columns[0]
    ).iterrows():
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
```

**Step 4:** Run tests:
```bash
pytest tests/test_data_pipeline.py::test_water_grid_is_binary tests/test_data_pipeline.py::test_landuse_grid_values_in_range -v
```
Expected: both PASS

**Step 5:** Commit:
```bash
git add data_pipeline/osm_layers.py tests/test_data_pipeline.py
git commit -m "data: add OSM water and land use layer rasterizers"
```

---

### Task 6: Road rasterizer — conditioning channel + 5-channel output (`data_pipeline/road_layers.py`)

Two functions from the same OSMnx road graph:
1. `rasterize_roads_binary()` → flat binary channel for the conditioning image (Ch3: existing roads)
2. `rasterize_road_output()` → 5-channel one-hot array (the training target)

**Files:**
- Create: `citiesgpt/data_pipeline/road_layers.py`
- Modify: `citiesgpt/tests/test_data_pipeline.py`

**Step 1:** Write failing tests:
```python
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
    # Background (ch0) dominates — roads are sparse
    assert arr[0].mean() > 0.5
```

**Step 2:** Run tests to verify they fail.

**Step 3:** Implement `road_layers.py`:
```python
import numpy as np
import os, math, pickle
import osmnx as ox
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from scipy.ndimage import binary_dilation

# Road level priority (higher index = higher priority, overwrites lower)
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
        G = ox.graph_from_bbox(north, south, east, west, custom_filter=cf)
    except Exception:
        G = None
    with open(cache, "wb") as f:
        pickle.dump(G, f)
    return G

def _get_edge_geometries_by_level(G, west, south, east, north):
    """Returns dict: channel_idx → list of shapely LineString geometries."""
    import geopandas as gpd
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
    # Dilate to line_width_px
    struct = np.ones((line_width_px, line_width_px), dtype=bool)
    arr = binary_dilation(arr.astype(bool), structure=struct).astype(np.float32)
    return arr

def rasterize_road_output(G, center_lon, center_lat, grid_size_px, pixel_size_m,
                           line_width_px=5):
    west, south, east, north = _bbox_latlon(center_lon, center_lat, grid_size_px, pixel_size_m)
    transform = from_bounds(west, south, east, north, grid_size_px, grid_size_px)
    by_level = _get_edge_geometries_by_level(G, west, south, east, north)

    # Priority map: 0=background, 1-4=road levels
    priority = np.zeros((grid_size_px, grid_size_px), dtype=np.int32)
    struct = np.ones((line_width_px, line_width_px), dtype=bool)
    for ch in [1, 2, 3, 4]:  # render in ascending priority order, higher overwrites
        if not by_level[ch]:
            continue
        shapes = [(g.__geo_interface__, 1) for g in by_level[ch]]
        layer = rasterize(shapes, out_shape=(grid_size_px, grid_size_px),
                          transform=transform, fill=0, dtype=np.int32)
        layer = binary_dilation(layer.astype(bool), structure=struct).astype(np.int32)
        priority = np.where(layer > 0, ch, priority)

    # Convert priority map to one-hot (5, H, W)
    one_hot = (np.arange(5)[:, None, None] == priority[None]).astype(np.float32)
    return one_hot
```

**Step 4:** Run tests:
```bash
pytest tests/test_data_pipeline.py::test_road_binary_is_binary tests/test_data_pipeline.py::test_road_output_is_one_hot -v
```
Expected: both PASS

**Step 5:** Commit:
```bash
git add data_pipeline/road_layers.py tests/test_data_pipeline.py
git commit -m "data: add road rasterizers for conditioning channel and 5-channel output"
```

---

### Task 7: Tile assembler — rotation, crop, save (`data_pipeline/tile_assembler.py`)

Assembles all 4 conditioning channels + 5-channel road output into a single tile. Applies the oversized-grid → rotate → center-crop pattern so all channels are aligned at the same rotation.

**Files:**
- Create: `citiesgpt/data_pipeline/tile_assembler.py`
- Modify: `citiesgpt/tests/test_data_pipeline.py`

**Step 1:** Write failing test:
```python
from data_pipeline.tile_assembler import assemble_tile

def test_assemble_tile_shapes():
    cond, road = assemble_tile(
        center_lon=-97.108, center_lat=32.735,
        rotation_deg=45.0,
        tile_size_px=64,
        pixel_size_m=5,
        osm_cache_dir="/tmp/osm_test_cache",
        srtm_cache_dir="/tmp/srtm_test_cache"
    )
    assert cond.shape == (4, 64, 64), f"Expected (4,64,64), got {cond.shape}"
    assert road.shape == (5, 64, 64), f"Expected (5,64,64), got {road.shape}"
    assert cond.dtype == np.float32
    assert road.dtype == np.float32
    # Conditioning channels in [0, 1]
    assert cond.min() >= 0.0 and cond.max() <= 1.0
    # Road output is one-hot
    assert np.allclose(road.sum(axis=0), 1.0)
```

**Step 2:** Run test to verify it fails.

**Step 3:** Implement `tile_assembler.py`:
```python
import numpy as np
import math
from scipy.ndimage import rotate as scipy_rotate
from data_pipeline.elevation_layer import fetch_elevation_grid
from data_pipeline.osm_layers import fetch_water_grid, fetch_landuse_grid
from data_pipeline.road_layers import fetch_road_graph, rasterize_roads_binary, rasterize_road_output

def _rotate_and_crop(arr, angle_deg, target_size):
    """Rotate a (H, W) or (C, H, W) array and center-crop to target_size."""
    if arr.ndim == 2:
        rotated = scipy_rotate(arr, angle=angle_deg, reshape=False, order=1, cval=0.0)
        h, w = rotated.shape
        top = (h - target_size) // 2
        left = (w - target_size) // 2
        return rotated[top:top+target_size, left:left+target_size]
    else:
        return np.stack([_rotate_and_crop(arr[c], angle_deg, target_size)
                         for c in range(arr.shape[0])])

def assemble_tile(center_lon, center_lat, rotation_deg, tile_size_px, pixel_size_m,
                  osm_cache_dir="osm_cache", srtm_cache_dir="srtm_cache"):
    """
    Returns (cond [4, tile_size_px, tile_size_px], road [5, tile_size_px, tile_size_px]).
    All channels are rasterized oversized then rotated+cropped together.
    """
    # Oversized grid size: needs to contain rotated tile at any angle
    oversized = int(tile_size_px * math.sqrt(2)) + 4

    # Fetch all layers at oversized resolution
    elev = fetch_elevation_grid(center_lon, center_lat, oversized, pixel_size_m, srtm_cache_dir)
    water = fetch_water_grid(center_lon, center_lat, oversized, pixel_size_m, osm_cache_dir)
    landuse = fetch_landuse_grid(center_lon, center_lat, oversized, pixel_size_m, osm_cache_dir)
    G = fetch_road_graph(center_lon, center_lat, oversized, pixel_size_m, osm_cache_dir)
    roads_binary = rasterize_roads_binary(G, center_lon, center_lat, oversized, pixel_size_m)
    roads_output = rasterize_road_output(G, center_lon, center_lat, oversized, pixel_size_m)

    # Stack conditioning channels: (4, oversized, oversized)
    cond_oversized = np.stack([elev, landuse, water, roads_binary], axis=0)

    # Rotate and crop all channels together
    cond = _rotate_and_crop(cond_oversized, rotation_deg, tile_size_px)
    road = _rotate_and_crop(roads_output, rotation_deg, tile_size_px)

    # Re-normalize road to valid one-hot after rotation (interpolation can soften edges)
    road_argmax = road.argmax(axis=0)
    road = (np.arange(5)[:, None, None] == road_argmax[None]).astype(np.float32)

    # Clip conditioning to [0, 1]
    cond = np.clip(cond, 0.0, 1.0)
    return cond.astype(np.float32), road.astype(np.float32)
```

**Step 4:** Run test:
```bash
pytest tests/test_data_pipeline.py::test_assemble_tile_shapes -v
```
Expected: PASS

**Step 5:** Commit:
```bash
git add data_pipeline/tile_assembler.py tests/test_data_pipeline.py
git commit -m "data: add tile assembler with rotation and crop"
```

---

### Task 8: Main pipeline script and PyTorch Dataset (`data_pipeline/cdg.py`, `data_pipeline/dataset.py`)

`cdg.py` is the CLI script that generates all tiles for all cities and saves them as `.npy` files. `dataset.py` is the PyTorch Dataset that loads them with on-the-fly augmentation.

**Files:**
- Create: `citiesgpt/data_pipeline/cdg.py`
- Create: `citiesgpt/data_pipeline/dataset.py`
- Modify: `citiesgpt/tests/test_data_pipeline.py`

**Step 1:** Write failing dataset test:
```python
import os, numpy as np, torch
from data_pipeline.dataset import RoadLayoutDataset

def test_dataset_returns_correct_shapes(tmp_path):
    # Create fake tile files
    for i in range(3):
        np.save(tmp_path / f"cond_{i:04d}.npy", np.zeros((4, 64, 64), dtype=np.float32))
        np.save(tmp_path / f"road_{i:04d}.npy", np.zeros((5, 64, 64), dtype=np.float32))
    ds = RoadLayoutDataset([str(tmp_path)], augment=True)
    assert len(ds) == 3
    cond, road = ds[0]
    assert isinstance(cond, torch.Tensor)
    assert cond.shape == (4, 64, 64)
    assert road.shape == (5, 64, 64)
```

**Step 2:** Run test to verify it fails.

**Step 3:** Implement `dataset.py`:
```python
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class RoadLayoutDataset(Dataset):
    def __init__(self, city_dirs: list, augment: bool = True):
        self.samples = []
        for d in city_dirs:
            cond_files = sorted(f for f in os.listdir(d) if f.startswith("cond_") and f.endswith(".npy"))
            for cf in cond_files:
                idx = cf.replace("cond_", "").replace(".npy", "")
                rf = f"road_{idx}.npy"
                if os.path.exists(os.path.join(d, rf)):
                    self.samples.append((os.path.join(d, cf), os.path.join(d, rf)))
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cond_path, road_path = self.samples[idx]
        cond = np.load(cond_path)   # (4, H, W)
        road = np.load(road_path)   # (5, H, W)

        if self.augment:
            # Random horizontal flip
            if np.random.random() > 0.5:
                cond = np.flip(cond, axis=2).copy()
                road = np.flip(road, axis=2).copy()
            # Random vertical flip
            if np.random.random() > 0.5:
                cond = np.flip(cond, axis=1).copy()
                road = np.flip(road, axis=1).copy()
            # Brightness/contrast jitter on conditioning channels only (not road)
            for ch in range(cond.shape[0]):
                brightness = np.random.uniform(0.9, 1.1)
                contrast   = np.random.uniform(0.9, 1.1)
                mean = cond[ch].mean()
                cond[ch] = np.clip((cond[ch] - mean) * contrast + mean * brightness, 0.0, 1.0)

        return torch.from_numpy(cond), torch.from_numpy(road)
```

**Step 4:** Implement `cdg.py` (CLI):
```python
#!/usr/bin/env python3
"""
Generate training tiles for all cities in cities.yaml.
Usage: python cdg.py --config data_pipeline/cities.yaml --output data/ [--city arlington_tx]
"""
import argparse, yaml, os, json, numpy as np
from tqdm import tqdm
from data_pipeline.tile_grid import generate_tile_centers
from data_pipeline.tile_assembler import assemble_tile
import osmnx as ox

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data_pipeline/cities.yaml")
    parser.add_argument("--output", default="data")
    parser.add_argument("--city", default=None, help="Process only this city name")
    parser.add_argument("--osm-cache", default="osm_cache")
    parser.add_argument("--srtm-cache", default="srtm_cache")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tile_size = cfg["tile_size_px"]
    pixel_size = cfg["pixel_size_m"]
    n_tiles = cfg["n_tiles_per_city"]
    jitter = cfg["jitter_fraction"]

    for city in cfg["cities"]:
        if args.city and city["name"] != args.city:
            continue
        print(f"\n=== {city['name']} ===")
        out_dir = os.path.join(args.output, city["name"])
        os.makedirs(out_dir, exist_ok=True)

        # Get city bbox from OSMnx
        place = ox.geocode_to_gdf(city["query"])
        bounds = place.total_bounds  # (minx, miny, maxx, maxy) in WGS84
        # Project to meters (approximate using center lat)
        import math
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
        lat_to_m = 111_000
        lon_to_m = 111_000 * math.cos(math.radians(center_lat))
        bbox_m = (
            (bounds[0] - center_lon) * lon_to_m,
            (bounds[1] - center_lat) * lat_to_m,
            (bounds[2] - center_lon) * lon_to_m,
            (bounds[3] - center_lat) * lat_to_m,
        )

        tile_size_m = tile_size * pixel_size
        centers = generate_tile_centers(bbox_m, tile_size_m, n_tiles, jitter)

        # Skip already-generated tiles
        existing = {f for f in os.listdir(out_dir) if f.startswith("cond_")}
        start_idx = len(existing)

        for i, (cx_m, cy_m, rot) in enumerate(tqdm(centers, desc=city["name"])):
            tile_idx = start_idx + i
            cond_path = os.path.join(out_dir, f"cond_{tile_idx:04d}.npy")
            road_path = os.path.join(out_dir, f"road_{tile_idx:04d}.npy")
            meta_path = os.path.join(out_dir, f"meta_{tile_idx:04d}.json")
            if os.path.exists(cond_path):
                continue

            # Convert local meters back to lat/lon
            lon = center_lon + cx_m / lon_to_m
            lat = center_lat + cy_m / lat_to_m

            try:
                cond, road = assemble_tile(lon, lat, rot, tile_size, pixel_size,
                                           args.osm_cache, args.srtm_cache)
                np.save(cond_path, cond)
                np.save(road_path, road)
                with open(meta_path, "w") as f:
                    json.dump({"lat": lat, "lon": lon, "rotation_deg": rot,
                               "city": city["name"], "split": city["split"]}, f)
            except Exception as e:
                print(f"  Skipping tile {tile_idx}: {e}")

if __name__ == "__main__":
    main()
```

**Step 5:** Run dataset test:
```bash
pytest tests/test_data_pipeline.py::test_dataset_returns_correct_shapes -v
```
Expected: PASS

**Step 6:** Smoke test the pipeline on 3 tiles from Arlington:
```bash
python data_pipeline/cdg.py --config data_pipeline/cities.yaml --output data/ --city arlington_tx
# Check output after 3 tiles (Ctrl+C after a few)
ls data/arlington_tx/ | head -10
python -c "import numpy as np; c=np.load('data/arlington_tx/cond_0000.npy'); r=np.load('data/arlington_tx/road_0000.npy'); print('cond', c.shape, c.min(), c.max()); print('road', r.shape, r.sum(axis=0).min(), r.sum(axis=0).max())"
```
Expected: `cond (4, 512, 512) 0.0 1.0` and `road (5, 512, 512) 1.0 1.0` (valid one-hot)

**Step 7:** Commit:
```bash
git add data_pipeline/cdg.py data_pipeline/dataset.py tests/test_data_pipeline.py
git commit -m "data: add main pipeline script and PyTorch dataset class"
```

---

## Part C — VAE

### Task 9: VAE architecture (`model/vae.py`)

The VAE learns to compress 5-channel road layouts to a 64×64×4 latent space and reconstruct them. Trained first, then frozen. The diffusion model never sees raw road images.

**Files:**
- Create: `citiesgpt/model/vae.py`
- Create: `citiesgpt/tests/test_model.py`

**Step 1:** Write failing tests:
```python
import torch
from model.vae import RoadVAE

def test_vae_encoder_output_shape():
    model = RoadVAE()
    x = torch.zeros(2, 5, 512, 512)
    mu, logvar = model.encode(x)
    assert mu.shape == (2, 4, 64, 64)
    assert logvar.shape == (2, 4, 64, 64)

def test_vae_decoder_output_shape():
    model = RoadVAE()
    z = torch.zeros(2, 4, 64, 64)
    out = model.decode(z)
    assert out.shape == (2, 5, 512, 512)

def test_vae_forward_returns_three_tensors():
    model = RoadVAE()
    x = torch.zeros(2, 5, 512, 512)
    recon, mu, logvar = model(x)
    assert recon.shape == (2, 5, 512, 512)
```

**Step 2:** Run tests to verify they fail.

**Step 3:** Implement `model/vae.py`:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels, stride=1, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.conv1 = (nn.ConvTranspose2d(channels, channels, 3, stride=stride, padding=1,
                                          output_padding=stride-1)
                      if upsample else
                      nn.Conv2d(channels, channels, 3, stride=stride, padding=1))
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()  # Swish

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + (F.interpolate(x, scale_factor=2) if self.upsample else x))

class VAEEncoder(nn.Module):
    def __init__(self, in_channels=5, latent_channels=4, base_ch=64):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1),
            ResBlock(base_ch, stride=2),       # 512 → 256
            nn.Conv2d(base_ch, base_ch*2, 3, padding=1),
            ResBlock(base_ch*2, stride=2),     # 256 → 128
            nn.Conv2d(base_ch*2, base_ch*4, 3, padding=1),
            ResBlock(base_ch*4, stride=2),     # 128 → 64
        )
        self.to_mu     = nn.Conv2d(base_ch*4, latent_channels, 1)
        self.to_logvar = nn.Conv2d(base_ch*4, latent_channels, 1)

    def forward(self, x):
        h = self.blocks(x)
        return self.to_mu(h), self.to_logvar(h)

class VAEDecoder(nn.Module):
    def __init__(self, latent_channels=4, out_channels=5, base_ch=64):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(latent_channels, base_ch*4, 3, padding=1),
            ResBlock(base_ch*4, stride=2, upsample=True),   # 64 → 128
            nn.Conv2d(base_ch*4, base_ch*2, 3, padding=1),
            ResBlock(base_ch*2, stride=2, upsample=True),   # 128 → 256
            nn.Conv2d(base_ch*2, base_ch, 3, padding=1),
            ResBlock(base_ch, stride=2, upsample=True),     # 256 → 512
            nn.Conv2d(base_ch, out_channels, 3, padding=1),
        )

    def forward(self, z):
        return self.blocks(z)

class RoadVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VAEEncoder()
        self.decoder = VAEDecoder()

    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
```

**Step 4:** Run tests:
```bash
pytest tests/test_model.py -v
```
Expected: all PASS

**Step 5:** Commit:
```bash
git add model/vae.py tests/test_model.py
git commit -m "model: add VAE encoder/decoder architecture"
```

---

### Task 10: VAE loss and training script (`model/vae_loss.py`, `model/train_vae.py`)

**Files:**
- Create: `citiesgpt/model/vae_loss.py`
- Create: `citiesgpt/model/train_vae.py`
- Modify: `citiesgpt/tests/test_model.py`

**Step 1:** Write failing loss test:
```python
from model.vae_loss import vae_loss

def test_vae_loss_returns_scalar():
    recon = torch.randn(2, 5, 64, 64)
    target = torch.zeros(2, 5, 64, 64)
    target[:, 0] = 1.0  # background channel
    mu = torch.zeros(2, 4, 16, 16)
    logvar = torch.zeros(2, 4, 16, 16)
    loss = vae_loss(recon, target, mu, logvar)
    assert loss.shape == ()         # scalar
    assert loss.item() > 0.0
    assert not torch.isnan(loss)
```

**Step 2:** Run test to verify it fails.

**Step 3:** Implement `model/vae_loss.py`:
```python
import torch
import torch.nn.functional as F

def focal_loss(logits, targets, gamma=2.0, alpha=None):
    """
    logits: (B, C, H, W) raw logits
    targets: (B, C, H, W) one-hot float
    alpha: (C,) per-class weight or None
    """
    log_p = F.log_softmax(logits, dim=1)
    p = log_p.exp()
    target_p = (p * targets).sum(dim=1, keepdim=True)  # (B, 1, H, W)
    focal_weight = (1 - target_p) ** gamma
    if alpha is not None:
        alpha_t = (alpha[None, :, None, None] * targets).sum(dim=1, keepdim=True)
        focal_weight = focal_weight * alpha_t
    loss = -(focal_weight * (log_p * targets).sum(dim=1, keepdim=True))
    return loss.mean()

def kl_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def vae_loss(recon_logits, targets, mu, logvar, gamma=2.0, kl_weight=1e-4):
    """Combined focal + KL loss for VAE training."""
    # Compute per-class alpha from target frequencies
    freq = targets.mean(dim=(0, 2, 3)) + 1e-6  # (C,)
    alpha = 1.0 / freq
    alpha = alpha / alpha.sum()
    l_focal = focal_loss(recon_logits, targets, gamma=gamma, alpha=alpha)
    l_kl    = kl_loss(mu, logvar)
    return l_focal + kl_weight * l_kl
```

**Step 4:** Run test:
```bash
pytest tests/test_model.py::test_vae_loss_returns_scalar -v
```
Expected: PASS

**Step 5:** Implement `model/train_vae.py`:
```python
#!/usr/bin/env python3
"""
Train the Road VAE (Stage 1).
Usage: python model/train_vae.py --data data/ --output checkpoints/vae/ --epochs 50
"""
import argparse, os, glob
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_pipeline.dataset import RoadLayoutDataset
from model.vae import RoadVAE
from model.vae_loss import vae_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default="data/")
    parser.add_argument("--output",  default="checkpoints/vae/")
    parser.add_argument("--epochs",  type=int, default=50)
    parser.add_argument("--batch",   type=int, default=4)
    parser.add_argument("--lr",      type=float, default=2e-5)
    parser.add_argument("--resume",  default=None)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset: train cities (exclude irving_tx val city)
    all_city_dirs = sorted(glob.glob(os.path.join(args.data, "*")))
    train_dirs = [d for d in all_city_dirs if "irving_tx" not in d]
    val_dirs   = [d for d in all_city_dirs if "irving_tx" in d]

    train_ds = RoadLayoutDataset(train_dirs, augment=True)
    val_ds   = RoadLayoutDataset(val_dirs,   augment=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=2)

    model = RoadVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for cond, road in tqdm(train_dl, desc=f"Epoch {epoch+1} train"):
            road = road.to(device)
            recon, mu, logvar = model(road)
            loss = vae_loss(recon, road, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- Val ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for cond, road in val_dl:
                road = road.to(device)
                recon, mu, logvar = model(road)
                val_loss += vae_loss(recon, road, mu, logvar).item()

        print(f"Epoch {epoch+1}: train={train_loss/len(train_dl):.4f}  val={val_loss/len(val_dl):.4f}")

        # Checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            path = os.path.join(args.output, f"vae_epoch_{epoch+1:03d}.pth")
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict()}, path)
            print(f"  Saved {path}")

if __name__ == "__main__":
    main()
```

**Step 6:** Smoke test (1 epoch, small batch):
```bash
python model/train_vae.py --data data/ --epochs 1 --batch 2
```
Expected: prints `Epoch 1: train=X.XXXX  val=X.XXXX` without crashing.

**Step 7:** Commit:
```bash
git add model/vae_loss.py model/train_vae.py tests/test_model.py
git commit -m "model: add VAE focal+KL loss and training script"
```

---

## Part D — Diffusion U-Net

### Task 11: Condition-aware Decoder Block (`model/cdb.py`)

**Files:**
- Create: `citiesgpt/model/cdb.py`
- Modify: `citiesgpt/tests/test_model.py`

**Step 1:** Write failing tests:
```python
from model.cdb import ConditionAwareDecoderBlock

def test_cdb_conditional_output_shape():
    block = ConditionAwareDecoderBlock(channels=64, cond_channels=64)
    R_down = torch.zeros(2, 64, 32, 32)
    R_up   = torch.zeros(2, 64, 32, 32)
    R_c    = torch.zeros(2, 64, 32, 32)
    out = block(R_down, R_up, R_c)
    assert out.shape == (2, 64, 32, 32)

def test_cdb_unconditional_zeros_same_shape():
    block = ConditionAwareDecoderBlock(channels=64, cond_channels=64)
    R_down = torch.zeros(2, 64, 32, 32)
    R_up   = torch.zeros(2, 64, 32, 32)
    R_c    = torch.zeros(2, 64, 32, 32)   # zeros = unconditional
    out = block(R_down, R_up, R_c)
    assert out.shape == (2, 64, 32, 32)
```

**Step 2:** Run tests to verify they fail.

**Step 3:** Implement `model/cdb.py`:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalDetailsEnhancement(nn.Module):
    def __init__(self, channels, cond_channels):
        super().__init__()
        self.cond_proj   = nn.Sequential(nn.Conv2d(cond_channels, channels, 1), nn.SiLU())
        self.skip_proj   = nn.Conv2d(channels, channels, 1)
        self.fuse        = nn.Conv2d(channels * 2, channels, 3, padding=1)
        self.up_fuse     = nn.Conv2d(channels * 2, channels, 3, padding=1)
        self.norm        = nn.GroupNorm(8, channels)
        self.act         = nn.SiLU()

    def forward(self, R_down, R_up, R_c):
        is_unconditional = (R_c.abs().sum() == 0)
        if not is_unconditional:
            cond_feat  = self.cond_proj(R_c)
            skip_feat  = self.skip_proj(R_down)
            fused      = self.fuse(torch.cat([cond_feat, skip_feat], dim=1))
            R_down     = R_down + fused
        combined = torch.cat([R_down, R_up], dim=1)
        R_l = self.act(self.norm(self.up_fuse(combined)))
        return R_l

class GlobalContextIntegration(nn.Module):
    def __init__(self, channels, cond_channels):
        super().__init__()
        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(cond_channels, channels, 1)
        self.v_proj = nn.Conv2d(cond_channels, channels, 1)
        self.out    = nn.Linear(channels, channels)
        self.scale  = channels ** -0.5

    def forward(self, R_l, R_c):
        B, C, H, W = R_l.shape
        is_unconditional = (R_c.abs().sum() == 0)
        kv_source = R_l if is_unconditional else R_c

        Q = self.q_proj(R_l).flatten(2).transpose(1, 2)    # (B, HW, C)
        K = self.k_proj(kv_source).flatten(2).transpose(1, 2)
        V = self.v_proj(kv_source).flatten(2).transpose(1, 2)

        attn = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
        S = (attn @ V)                                       # (B, HW, C)
        S = self.out(S).transpose(1, 2).reshape(B, C, H, W)
        return R_l + S

class ConditionAwareDecoderBlock(nn.Module):
    def __init__(self, channels, cond_channels):
        super().__init__()
        self.lde = LocalDetailsEnhancement(channels, cond_channels)
        self.gci = GlobalContextIntegration(channels, cond_channels)

    def forward(self, R_down, R_up, R_c):
        R_l = self.lde(R_down, R_up, R_c)
        R_g = self.gci(R_l, R_c)
        return R_g
```

**Step 4:** Run tests:
```bash
pytest tests/test_model.py::test_cdb_conditional_output_shape tests/test_model.py::test_cdb_unconditional_zeros_same_shape -v
```
Expected: both PASS

**Step 5:** Commit:
```bash
git add model/cdb.py tests/test_model.py
git commit -m "model: add Condition-aware Decoder Block (LDE + GCI)"
```

---

### Task 12: Diffusion U-Net (`model/unet.py`)

**Files:**
- Create: `citiesgpt/model/unet.py`
- Modify: `citiesgpt/tests/test_model.py`

**Step 1:** Write failing tests:
```python
from model.unet import DiffusionUNet

def test_unet_unconditional_output_shape():
    net = DiffusionUNet(latent_channels=4, cond_channels=4)
    noise = torch.zeros(2, 4, 64, 64)
    t     = torch.tensor([100, 500])
    cond  = torch.zeros(2, 4, 512, 512)  # zeros = unconditional
    out   = net(noise, t, cond)
    assert out.shape == (2, 4, 64, 64)

def test_unet_conditional_different_from_unconditional():
    net  = DiffusionUNet(latent_channels=4, cond_channels=4)
    noise = torch.randn(1, 4, 64, 64)
    t     = torch.tensor([500])
    cond  = torch.randn(1, 4, 512, 512)
    out_uncond = net(noise, t, torch.zeros_like(cond))
    out_cond   = net(noise, t, cond)
    assert not torch.allclose(out_uncond, out_cond)
```

**Step 2:** Run tests to verify they fail.

**Step 3:** Implement `model/unet.py`:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model.cdb import ConditionAwareDecoderBlock

def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
    args  = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class TimestepMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim*4), nn.SiLU(), nn.Linear(dim*4, dim))

    def forward(self, t_emb):
        return self.net(t_emb)

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, stride=1, upsample=False):
        super().__init__()
        self.upsample = upsample
        conv_cls = nn.ConvTranspose2d if upsample else nn.Conv2d
        self.conv1 = conv_cls(in_ch, out_ch, 3, stride=stride, padding=1,
                              **({"output_padding": stride-1} if upsample else {}))
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.attn = nn.MultiheadAttention(out_ch, num_heads=4, batch_first=True)
        self.attn_norm = nn.LayerNorm(out_ch)

    def forward(self, x, t_emb):
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.t_proj(t_emb)[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        # Self-attention
        B, C, H, W = h.shape
        flat = h.flatten(2).transpose(1, 2)
        flat, _ = self.attn(flat, flat, flat)
        flat = self.attn_norm(flat)
        h = h + flat.transpose(1, 2).reshape(B, C, H, W)
        # Skip connection
        skip = self.skip(x)
        if self.upsample:
            skip = F.interpolate(skip, scale_factor=2)
        return h + skip

class ConditionEncoder(nn.Module):
    def __init__(self, in_ch=4, base_ch=64):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.enc1 = nn.Sequential(nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1), nn.SiLU())     # 512→256
        self.enc2 = nn.Sequential(nn.Conv2d(base_ch, base_ch*2, 3, stride=2, padding=1), nn.SiLU())   # 256→128
        self.enc3 = nn.Sequential(nn.Conv2d(base_ch*2, base_ch*4, 3, stride=2, padding=1), nn.SiLU()) # 128→64
        self.enc4 = nn.Sequential(nn.Conv2d(base_ch*4, base_ch*4, 3, stride=2, padding=1), nn.SiLU()) # 64→32

    def forward(self, c):
        h0 = self.stem(c)
        h1 = self.enc1(h0)
        h2 = self.enc2(h1)
        h3 = self.enc3(h2)
        h4 = self.enc4(h3)
        return h1, h2, h3, h4  # R_c^0..3 at res 256,128,64,32

class DiffusionUNet(nn.Module):
    def __init__(self, latent_channels=4, cond_channels=4, base_ch=64, t_dim=256):
        super().__init__()
        self.t_dim = t_dim
        self.t_mlp = TimestepMLP(t_dim)
        self.cond_enc = ConditionEncoder(cond_channels, base_ch)

        # Noise encoder (operates on 64×64 latent)
        self.noise_enc1 = UNetBlock(latent_channels, base_ch,   t_dim, stride=2)   # 64→32
        self.noise_enc2 = UNetBlock(base_ch,         base_ch*2, t_dim, stride=2)   # 32→16
        self.noise_enc3 = UNetBlock(base_ch*2,       base_ch*4, t_dim, stride=2)   # 16→8
        self.noise_enc4 = UNetBlock(base_ch*4,       base_ch*4, t_dim, stride=2)   # 8→4
        self.bottleneck = UNetBlock(base_ch*4,       base_ch*4, t_dim)

        # CDB decoders
        self.cdb1 = ConditionAwareDecoderBlock(base_ch*4, base_ch*4)
        self.cdb2 = ConditionAwareDecoderBlock(base_ch*4, base_ch*4)
        self.cdb3 = ConditionAwareDecoderBlock(base_ch*2, base_ch*2)
        self.cdb4 = ConditionAwareDecoderBlock(base_ch,   base_ch)

        self.dec1 = UNetBlock(base_ch*4*2, base_ch*4, t_dim, stride=2, upsample=True)  # 4→8
        self.dec2 = UNetBlock(base_ch*4*2, base_ch*2, t_dim, stride=2, upsample=True)  # 8→16
        self.dec3 = UNetBlock(base_ch*2*2, base_ch,   t_dim, stride=2, upsample=True)  # 16→32
        self.dec4 = UNetBlock(base_ch*2,   base_ch,   t_dim, stride=2, upsample=True)  # 32→64
        self.out  = nn.Conv2d(base_ch, latent_channels, 1)

    def forward(self, x, t, cond):
        t_emb = self.t_mlp(sinusoidal_embedding(t, self.t_dim))
        R_c = self.cond_enc(cond)  # tuple of 4 feature maps

        # Encode noise
        e1 = self.noise_enc1(x,  t_emb)
        e2 = self.noise_enc2(e1, t_emb)
        e3 = self.noise_enc3(e2, t_emb)
        e4 = self.noise_enc4(e3, t_emb)
        b  = self.bottleneck(e4, t_emb)

        # CDB-conditioned decode
        d1 = self.dec1(torch.cat([self.cdb1(e4, F.interpolate(b,  size=e4.shape[-2:]), R_c[3]), e4], 1), t_emb)
        d2 = self.dec2(torch.cat([self.cdb2(e3, F.interpolate(d1, size=e3.shape[-2:]), R_c[2]), e3], 1), t_emb)
        d3 = self.dec3(torch.cat([self.cdb3(e2, F.interpolate(d2, size=e2.shape[-2:]), R_c[1]), e2], 1), t_emb)
        d4 = self.dec4(torch.cat([self.cdb4(e1, F.interpolate(d3, size=e1.shape[-2:]), R_c[0]), e1], 1), t_emb)
        return self.out(d4)
```

**Step 4:** Run tests:
```bash
pytest tests/test_model.py::test_unet_unconditional_output_shape tests/test_model.py::test_unet_conditional_different_from_unconditional -v
```
Expected: both PASS

**Step 5:** Commit:
```bash
git add model/unet.py tests/test_model.py
git commit -m "model: add diffusion U-Net with CDB decoder blocks"
```

---

### Task 13: DDPM training logic (`model/diffusion.py`)

**Files:**
- Create: `citiesgpt/model/diffusion.py`
- Modify: `citiesgpt/tests/test_model.py`

**Step 1:** Write failing tests:
```python
from model.diffusion import DDPM

def test_forward_diffusion_near_gaussian_at_T():
    ddpm = DDPM(T=1000)
    x0 = torch.zeros(4, 4, 16, 16)
    t  = torch.full((4,), 999)
    x_t, eps = ddpm.forward_diffusion(x0, t)
    # At T≈1000, x_t should be approximately standard Gaussian
    assert abs(x_t.std().item() - 1.0) < 0.15

def test_training_loss_is_positive_scalar():
    from model.unet import DiffusionUNet
    ddpm = DDPM(T=1000)
    net  = DiffusionUNet(latent_channels=4, cond_channels=4)
    x0   = torch.randn(2, 4, 16, 16)
    cond = torch.randn(2, 4, 64, 64)
    loss = ddpm.training_loss(net, x0, cond, cfg_prob=0.5)
    assert loss.shape == ()
    assert loss.item() > 0
    assert not torch.isnan(loss)
```

**Step 2:** Run tests to verify they fail.

**Step 3:** Implement `model/diffusion.py`:
```python
import torch
import torch.nn.functional as F
import numpy as np

class DDPM:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register = lambda name, val: setattr(self, name, val)
        self.betas    = betas
        self.alphas   = alphas
        self.alpha_bar = alpha_bar

    def _to_device(self, t_device):
        self.alpha_bar = self.alpha_bar.to(t_device)

    def forward_diffusion(self, x0, t):
        """Add noise to x0 at timestep t. Returns (x_t, eps)."""
        self._to_device(x0.device)
        ab = self.alpha_bar[t].view(-1, 1, 1, 1)
        eps = torch.randn_like(x0)
        x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        return x_t, eps

    def training_loss(self, model, x0, cond, cfg_prob=0.5):
        """DDPM epsilon-prediction MSE loss with classifier-free guidance dropout."""
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device)
        x_t, eps = self.forward_diffusion(x0, t)

        # Classifier-free guidance: zero out conditioning with probability cfg_prob
        mask = (torch.rand(B, device=x0.device) > cfg_prob).float()
        cond_masked = cond * mask[:, None, None, None]

        eps_pred = model(x_t, t, cond_masked)
        return F.mse_loss(eps_pred, eps)

    @torch.no_grad()
    def sample_ddim(self, model, cond, n_steps=50, guidance_scale=3.0,
                    latent_shape=(1, 4, 64, 64)):
        """DDIM sampling with classifier-free guidance."""
        self._to_device(cond.device)
        device = cond.device
        x = torch.randn(latent_shape, device=device)
        step_size = self.T // n_steps
        timesteps = list(range(self.T - 1, -1, -step_size))

        for t_val in timesteps:
            t = torch.full((latent_shape[0],), t_val, device=device, dtype=torch.long)
            eps_cond   = model(x, t, cond)
            eps_uncond = model(x, t, torch.zeros_like(cond))
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            ab  = self.alpha_bar[t_val]
            ab_prev = self.alpha_bar[max(t_val - step_size, 0)]
            x0_pred = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
            x0_pred = x0_pred.clamp(-3, 3)
            x = ab_prev.sqrt() * x0_pred + (1 - ab_prev).sqrt() * eps
        return x
```

**Step 4:** Run tests:
```bash
pytest tests/test_model.py::test_forward_diffusion_near_gaussian_at_T tests/test_model.py::test_training_loss_is_positive_scalar -v
```
Expected: both PASS

**Step 5:** Commit:
```bash
git add model/diffusion.py tests/test_model.py
git commit -m "model: add DDPM forward/reverse process with CFG and DDIM sampling"
```

---

### Task 14: Diffusion training script (`model/train_diffusion.py`)

**Files:**
- Create: `citiesgpt/model/train_diffusion.py`

**Step 1:** Implement `model/train_diffusion.py`:
```python
#!/usr/bin/env python3
"""
Train the Diffusion U-Net (Stage 2). Requires trained VAE checkpoint.
Usage: python model/train_diffusion.py --vae checkpoints/vae/vae_epoch_050.pth --data data/ --output checkpoints/diffusion/
"""
import argparse, os, glob
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_pipeline.dataset import RoadLayoutDataset
from model.vae import RoadVAE
from model.unet import DiffusionUNet
from model.diffusion import DDPM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae",    required=True)
    parser.add_argument("--data",   default="data/")
    parser.add_argument("--output", default="checkpoints/diffusion/")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch",  type=int, default=4)
    parser.add_argument("--lr",     type=float, default=2e-5)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load frozen VAE
    vae = RoadVAE().to(device)
    vae.load_state_dict(torch.load(args.vae, map_location=device)["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # Dataset
    all_dirs   = sorted(glob.glob(os.path.join(args.data, "*")))
    train_dirs = [d for d in all_dirs if "irving_tx" not in d]
    val_dirs   = [d for d in all_dirs if "irving_tx" in d]
    train_dl = DataLoader(RoadLayoutDataset(train_dirs, augment=True),
                          batch_size=args.batch, shuffle=True, num_workers=2)
    val_dl   = DataLoader(RoadLayoutDataset(val_dirs,  augment=False),
                          batch_size=args.batch, shuffle=False, num_workers=2)

    net   = DiffusionUNet(latent_channels=4, cond_channels=4).to(device)
    ddpm  = DDPM(T=1000)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1

    for epoch in range(start_epoch, args.epochs):
        net.train()
        train_loss = 0.0
        for cond, road in tqdm(train_dl, desc=f"Epoch {epoch+1}"):
            cond, road = cond.to(device), road.to(device)
            with torch.no_grad():
                mu, logvar = vae.encode(road)
                x0 = vae.reparameterize(mu, logvar)
            loss = ddpm.training_loss(net, x0, cond, cfg_prob=0.5)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch+1}: train_loss={train_loss/len(train_dl):.6f}")

        if (epoch + 1) % 10 == 0:
            path = os.path.join(args.output, f"diffusion_epoch_{epoch+1:03d}.pth")
            torch.save({"epoch": epoch, "model": net.state_dict(),
                        "optimizer": optimizer.state_dict()}, path)
            print(f"  Saved {path}")

if __name__ == "__main__":
    main()
```

**Step 2:** Smoke test (1 epoch):
```bash
python model/train_diffusion.py --vae checkpoints/vae/vae_epoch_005.pth --data data/ --epochs 1 --batch 2
```
Expected: prints `Epoch 1: train_loss=X.XXXXXX` without crashing.

**Step 3:** Commit:
```bash
git add model/train_diffusion.py
git commit -m "model: add diffusion training script with frozen VAE encoder"
```

---

## Part E — Evaluation

### Task 15: Numeric metrics (`model/eval_metrics.py`)

**Files:**
- Create: `citiesgpt/model/eval_metrics.py`
- Modify: `citiesgpt/tests/test_model.py`

**Step 1:** Write failing test:
```python
from model.eval_metrics import compute_connectivity_index, compute_transport_convenience
import networkx as nx

def test_connectivity_index_on_grid():
    G = nx.grid_2d_graph(3, 3)
    ci = compute_connectivity_index(G)
    # In a 3x3 grid: corner nodes have degree 2, edge nodes 3, center 4
    # avg degree = (4*2 + 4*3 + 1*4) / 9 = 24/9 ≈ 2.67
    # CI = avg_degree / n_nodes × ... actually CI = sum(degrees) / n_nodes
    assert 2.0 < ci < 4.0

def test_transport_convenience_on_known_graph():
    G = nx.path_graph(5)  # straight line: 0-1-2-3-4
    # For nodes 0 and 4: euclidean dist = 4, shortest path = 4 → TC contribution = 1.0
    # Should be close to 1.0 on a straight path
    tc = compute_transport_convenience(G)
    assert 0.5 < tc <= 1.0
```

**Step 2:** Run tests to verify they fail.

**Step 3:** Implement `model/eval_metrics.py`:
```python
import torch
import numpy as np
import networkx as nx
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

def compute_connectivity_index(G: nx.Graph) -> float:
    """CI = average ratio of sum-of-degrees to total number of nodes."""
    if G.number_of_nodes() == 0:
        return 0.0
    degrees = [d for _, d in G.degree()]
    return sum(degrees) / G.number_of_nodes()

def compute_transport_convenience(G: nx.Graph, sample_pairs: int = 200) -> float:
    """
    TC = average(euclidean_dist / shortest_path_dist) over random node pairs.
    Requires node attributes 'x', 'y' or integer node IDs (used as coordinates).
    """
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return 0.0
    rng = np.random.default_rng(42)
    scores = []
    for _ in range(min(sample_pairs, len(nodes) * (len(nodes)-1) // 2)):
        u, v = rng.choice(nodes, size=2, replace=False)
        try:
            sp = nx.shortest_path_length(G, u, v, weight="weight")
        except nx.NetworkXNoPath:
            continue
        if sp == 0:
            continue
        # Get coordinates
        if isinstance(u, tuple):
            ux, uy = u
        else:
            ux, uy = G.nodes[u].get("x", u), G.nodes[u].get("y", 0)
        if isinstance(v, tuple):
            vx, vy = v
        else:
            vx, vy = G.nodes[v].get("x", v), G.nodes[v].get("y", 0)
        euclid = ((ux - vx)**2 + (uy - vy)**2) ** 0.5
        if euclid > 0:
            scores.append(euclid / sp)
    return float(np.mean(scores)) if scores else 0.0

class ImageQualityTracker:
    """Accumulates real/fake samples and computes FID and KID."""
    def __init__(self, device="cpu"):
        self.fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        self.kid = KernelInceptionDistance(subset_size=50, normalize=True).to(device)
        self.device = device

    def update_real(self, images_rgb: torch.Tensor):
        """images_rgb: (B, 3, H, W) float32 in [0, 1]"""
        self.fid.update(images_rgb.to(self.device), real=True)
        self.kid.update(images_rgb.to(self.device), real=True)

    def update_fake(self, images_rgb: torch.Tensor):
        self.fid.update(images_rgb.to(self.device), real=False)
        self.kid.update(images_rgb.to(self.device), real=False)

    def compute(self):
        return {"fid": self.fid.compute().item(), "kid": self.kid.compute()[0].item()}
```

**Step 4:** Run tests:
```bash
pytest tests/test_model.py::test_connectivity_index_on_grid tests/test_model.py::test_transport_convenience_on_known_graph -v
```
Expected: both PASS

**Step 5:** Commit:
```bash
git add model/eval_metrics.py tests/test_model.py
git commit -m "model: add FID/KID/CI/TC evaluation metrics"
```

---

### Task 16: VLM realism scorer (`model/vlm_eval.py`)

Sends a batch of generated road layout images to the Claude API and gets back a 1–10 realism score plus a list of specific issues. Run every 20 epochs, 10 samples.

**Files:**
- Create: `citiesgpt/model/vlm_eval.py`

**Step 1:** Implement `model/vlm_eval.py`:
```python
"""
VLM-based realism evaluation using Claude claude-sonnet-4-6.
Requires ANTHROPIC_API_KEY environment variable.
"""
import base64, io, os
import numpy as np
import torch
import anthropic
from PIL import Image

EVAL_PROMPT = """You are evaluating a machine-learning-generated road network image.
The image shows a top-down view of roads for a US suburban area, rendered as colored lines on a black background.
Different colors represent different road types (white=highway, red=arterial, orange=collector, gray=residential).

Rate this image 1-10 on REALISM:
- 10: Looks like a real US suburb from OpenStreetMap. Roads connect properly, spacing is realistic, hierarchy makes sense.
- 7-9: Mostly realistic with minor issues (slightly off spacing or a few disconnections).
- 4-6: Recognizable road patterns but notable problems (some psychedelic artifacts, unrealistic density, or poor connectivity).
- 1-3: Clearly machine-generated garbage: swirling artifacts, disconnected fragments, implausible patterns.

Respond with ONLY:
SCORE: <integer 1-10>
ISSUES: <one sentence describing the main problems, or "none" if score >= 8>"""

def road_tensor_to_rgb(road: np.ndarray) -> np.ndarray:
    """Convert 5-channel one-hot road array (5, H, W) to RGB (H, W, 3) uint8."""
    COLORS = {
        0: (0,   0,   0),    # background: black
        1: (128, 128, 128),  # residential: gray
        2: (255, 165,   0),  # tertiary: orange
        3: (255,  50,  50),  # primary/secondary: red
        4: (255, 255, 255),  # motorway/trunk: white
    }
    argmax = road.argmax(axis=0)  # (H, W)
    rgb = np.zeros((*argmax.shape, 3), dtype=np.uint8)
    for ch, color in COLORS.items():
        mask = argmax == ch
        rgb[mask] = color
    return rgb

def score_samples(road_arrays: list, model: str = "claude-sonnet-4-6") -> list:
    """
    road_arrays: list of np.ndarray, each (5, H, W) one-hot float32
    Returns list of {"score": int, "issues": str} dicts.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    results = []
    for road in road_arrays:
        rgb = road_tensor_to_rgb(road)
        img = Image.fromarray(rgb).resize((512, 512))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.standard_b64encode(buf.getvalue()).decode()

        response = client.messages.create(
            model=model,
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64",
                                                  "media_type": "image/png",
                                                  "data": b64}},
                    {"type": "text", "text": EVAL_PROMPT}
                ]
            }]
        )
        text = response.content[0].text.strip()
        score_line  = next((l for l in text.splitlines() if l.startswith("SCORE:")), "SCORE: 0")
        issues_line = next((l for l in text.splitlines() if l.startswith("ISSUES:")), "ISSUES: parse error")
        try:
            score = int(score_line.split(":")[1].strip())
        except ValueError:
            score = 0
        issues = issues_line.split(":", 1)[1].strip()
        results.append({"score": score, "issues": issues})
    return results
```

**Step 2:** Test with a dummy black image (no real model needed):
```python
# Quick manual test — don't add to pytest (requires API key + network)
# python -c "
# import numpy as np, os
# os.environ['ANTHROPIC_API_KEY'] = 'your_key'
# from model.vlm_eval import score_samples
# dummy = np.zeros((5, 64, 64), dtype=np.float32); dummy[0] = 1.0
# print(score_samples([dummy]))
# "
```

**Step 3:** Commit:
```bash
git add model/vlm_eval.py
git commit -m "model: add VLM realism scorer using Claude API"
```

---

## Part F — Colab Notebooks

### Task 17: VAE training notebook (`notebooks/train_vae.ipynb`)

**Files:**
- Create: `citiesgpt/notebooks/train_vae.ipynb`

The notebook should have these cells in order:

**Cell 1 — Mount Drive + install deps:**
```python
from google.colab import drive
drive.mount('/content/drive')
%cd '/content/drive/MyDrive/citiesgpt'
!pip install -r requirements.txt -q
```

**Cell 2 — Generate data (run once):**
```python
# Run data pipeline for all cities. Takes ~2-3 hours total.
# Skip cities already complete by checking data/ directory.
!python data_pipeline/cdg.py --config data_pipeline/cities.yaml --output data/
```

**Cell 3 — Train VAE:**
```python
!python model/train_vae.py \
    --data data/ \
    --output checkpoints/vae/ \
    --epochs 50 \
    --batch 4
```

**Cell 4 — Visualize reconstruction (sanity check):**
```python
import torch, numpy as np, matplotlib.pyplot as plt
from model.vae import RoadVAE
from data_pipeline.dataset import RoadLayoutDataset

model = RoadVAE()
ckpt = torch.load('checkpoints/vae/vae_epoch_050.pth', map_location='cpu')
model.load_state_dict(ckpt['model'])
model.eval()

ds = RoadLayoutDataset(['data/irving_tx'], augment=False)
_, road = ds[0]
with torch.no_grad():
    recon, _, _ = model(road.unsqueeze(0))
recon_argmax = recon[0].argmax(0).numpy()
road_argmax  = road.argmax(0).numpy()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(road_argmax,  cmap='tab10', vmin=0, vmax=4); axes[0].set_title('Original')
axes[1].imshow(recon_argmax, cmap='tab10', vmin=0, vmax=4); axes[1].set_title('Reconstructed')
plt.show()
# Target: visually similar, SSIM > 0.70
```

**Step 1:** Create the notebook file with these 4 cells.
**Step 2:** Commit:
```bash
git add notebooks/train_vae.ipynb
git commit -m "notebooks: add VAE training Colab notebook"
```

---

### Task 18: Diffusion training notebook (`notebooks/train_diffusion.ipynb`)

**Files:**
- Create: `citiesgpt/notebooks/train_diffusion.ipynb`

**Cell 1 — Mount + install:**
Same as VAE notebook Cell 1.

**Cell 2 — Train diffusion:**
```python
!python model/train_diffusion.py \
    --vae checkpoints/vae/vae_epoch_050.pth \
    --data data/ \
    --output checkpoints/diffusion/ \
    --epochs 200 \
    --batch 4
```

**Cell 3 — Generate samples + evaluate (run every ~20 epochs manually):**
```python
import torch, numpy as np, os
from model.vae import RoadVAE
from model.unet import DiffusionUNet
from model.diffusion import DDPM
from model.vlm_eval import score_samples

EPOCH = 100  # change this
device = torch.device('cuda')

vae = RoadVAE().to(device)
vae.load_state_dict(torch.load('checkpoints/vae/vae_epoch_050.pth')['model'])
vae.eval()

net = DiffusionUNet().to(device)
net.load_state_dict(torch.load(f'checkpoints/diffusion/diffusion_epoch_{EPOCH:03d}.pth')['model'])
net.eval()

ddpm = DDPM()
# Generate 10 unconditional samples (cond=zeros)
samples = []
for _ in range(10):
    with torch.no_grad():
        cond = torch.zeros(1, 4, 512, 512, device=device)
        latent = ddpm.sample_ddim(net, cond, n_steps=50)
        road = vae.decode(latent)
        road_argmax = road[0].argmax(0).cpu().numpy()
        one_hot = (np.arange(5)[:, None, None] == road_argmax[None]).astype(np.float32)
        samples.append(one_hot)

# VLM scoring
import os; os.environ['ANTHROPIC_API_KEY'] = 'YOUR_KEY_HERE'
scores = score_samples(samples)
avg = sum(s['score'] for s in scores) / len(scores)
print(f"Epoch {EPOCH} — Avg VLM realism score: {avg:.1f}/10")
for i, s in enumerate(scores):
    print(f"  Sample {i+1}: {s['score']}/10 — {s['issues']}")
```

**Cell 4 — Numeric metrics:**
```python
# Vectorize samples and compute CI / TC
from vectorize.postprocess import vectorize_road_layout
from model.eval_metrics import compute_connectivity_index, compute_transport_convenience

ci_scores, tc_scores = [], []
for sample in samples:
    G = vectorize_road_layout(sample)
    if G and G.number_of_nodes() > 5:
        ci_scores.append(compute_connectivity_index(G))
        tc_scores.append(compute_transport_convenience(G))

print(f"CI: {np.mean(ci_scores):.3f} (target > 1.8)")
print(f"TC: {np.mean(tc_scores):.3f} (target > 0.6)")
# CaRoLS benchmark: CI=1.948, TC=0.668
```

**Step 1:** Create the notebook with these 4 cells.
**Step 2:** Commit:
```bash
git add notebooks/train_diffusion.ipynb
git commit -m "notebooks: add diffusion training and evaluation Colab notebook"
```

---

## Summary of Milestones

| # | Milestone | Done when |
|---|-----------|-----------|
| M1 | Data pipeline complete | `data/` contains ~1200 tile pairs across 8 cities; shapes verified |
| M2 | VAE trained | Reconstruction SSIM > 0.70 on Irving TX validation set |
| M3 | Diffusion trained | CI > 1.8, TC > 0.6, VLM avg score > 5/10 |
| M4 | Evaluation suite live | FID/KID/CI/TC + VLM scores logged per checkpoint |

## CaRoLS Benchmarks (targets to beat)

| Metric | CaRoLS unconditional | Our target |
|--------|---------------------|-----------|
| KID | 0.0331 | < 0.05 |
| FID | 32.2 | < 50 |
| CI  | 1.948 | > 1.8 |
| TC  | 0.668 | > 0.6 |
| VLM score | N/A | > 6/10 |
