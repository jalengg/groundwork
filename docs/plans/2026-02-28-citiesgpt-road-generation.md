# Groundwork Road Generation — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Cities Skylines 1 mod that lets a player brush an area on the map and have a machine-learning model generate a realistic, multi-level road network that is automatically placed in-game.

**Architecture:** A CaRoLS-inspired two-stage diffusion pipeline (custom VAE + U-Net with CDB conditioning) generates a 5-channel one-hot road layout image from a conditioning map (terrain + land use). A Python inference server wraps the model. A C# Cities Skylines mod extracts the conditioning data, calls the server, vectorizes the result, and places roads via the game's NetManager API.

**Tech Stack:** Python 3.10, PyTorch 2.x, OSMnx, scikit-image, sknw, NetworkX, Flask; C# / .NET Framework 3.5, Harmony 2 (CitiesHarmony), Cities Skylines 1 Managed DLLs; Google Colab A100 for training; local RTX 3090 for inference.

---

## Overview of Phases

```
Phase 1 — Project structure & tooling
Phase 2 — Data pipeline (multi-channel, multi-city, arbitrary rotation)
Phase 3 — VAE: encode/decode 5-channel road layouts
Phase 4 — Diffusion U-Net with Condition-aware Decoder Block (CDB)
Phase 5 — Training orchestration (Colab, checkpoints, evaluation)
Phase 6 — Vectorization post-processing pipeline
Phase 7 — Inference server (Flask, local or Colab-tunnelled)
Phase 8 — Cities Skylines C# mod (brush tool + HTTP client + road placement)
Phase 9 — Integration & style conditioning
```

---

## Phase 1 — Project Structure & Tooling

### Task 1.1: Repository layout

**Files to create:**
```
groundwork/
  data/                      ← generated training images (gitignored)
  docs/plans/                ← this file lives here
  model/
    vae.py                   ← VAE architecture
    unet.py                  ← Diffusion U-Net
    cdb.py                   ← Condition-aware Decoder Block (LDE + GCI)
    diffusion.py             ← DDPM forward/reverse process, loss
    train_vae.py             ← VAE training script
    train_diffusion.py       ← Diffusion training script
  data_pipeline/
    cdg.py                   ← Road image generation (replaces cdg.ipynb)
    cities.yaml              ← City list + bounding boxes
    augment.py               ← Rotation, flipping, conditioning image construction
  vectorize/
    postprocess.py           ← argmax → skeleton → graph → RDP
    export.py                ← NetworkX graph → JSON for mod to consume
  server/
    app.py                   ← Flask inference server
    inference.py             ← Load model, run generation, return vectors
  mod/                       ← C# Visual Studio project
    Groundwork.csproj
    src/
      Mod.cs                 ← IUserMod entry point
      Loading.cs             ← ILoadingExtension
      BrushTool.cs           ← ToolBase subclass
      HeightmapExtractor.cs  ← TerrainManager + zone data → conditioning image
      RoadClient.cs          ← HTTP client to Python server
      RoadPlacer.cs          ← NetManager.CreateNode/CreateSegment calls
      Patcher.cs             ← Harmony patch setup (isolated)
  notebooks/
    train_vae.ipynb          ← Colab wrapper for train_vae.py
    train_diffusion.ipynb    ← Colab wrapper for train_diffusion.py
  requirements.txt
  .gitignore
```

**Step 1:** Create the directory tree above (empty files).
**Step 2:** Add a `requirements.txt` with: `torch`, `torchvision`, `osmnx`, `networkx`, `scikit-image`, `sknw`, `rdp`, `flask`, `pillow`, `numpy`, `pyyaml`, `tqdm`, `matplotlib`.
**Step 3:** Add `.gitignore` entries for `data/`, `*.pkl`, `*.pth`, `__pycache__/`, `*.pyc`, `*.zip`.
**Step 4:** Commit: `chore: scaffold project structure`.

---

## Phase 2 — Data Pipeline

### Background

The 2023 attempt used a single city (Arlington TX), only 4 rotation angles (0°/90°/180°/270°), 1-channel grayscale, and `edge_linewidth=2`. Root cause of "psychedelic" artifacts: roads became sub-pixel at the 64px StyleGAN base resolution. Fix: native 512px, multi-channel one-hot, arbitrary rotation, 8 cities.

### Task 2.1: City list configuration (`data_pipeline/cities.yaml`)

Define 8 US cities covering different road morphologies. Each entry needs:
- `name`: human-readable
- `osm_query`: OSMnx place query string (e.g. `"Arlington, Texas, USA"`)
- `style_label`: one of `us_suburb`, `us_grid`, `organic` — used as conditioning metadata
- `n_tiles`: number of non-overlapping tiles to generate (~150 per city)
- `tile_dist_m`: radius in meters for bbox (use 1280 for 512px @ 5m/px)

Suggested cities:
| City | Style |
|------|-------|
| Arlington TX | us_suburb |
| Chandler AZ | us_suburb |
| Gilbert AZ | us_suburb |
| Henderson NV | us_suburb |
| Chicago IL (south side) | us_grid |
| Phoenix AZ (downtown) | us_grid |
| Houston TX (Midtown) | organic |
| San Antonio TX (near downtown) | organic |

**Step 1:** Write `cities.yaml`.
**Step 2:** Commit: `data: add city list config`.

---

### Task 2.2: Road image generation (`data_pipeline/cdg.py`)

**What this script does:**
1. Loads `cities.yaml`
2. For each city, downloads the OSMnx drivable road graph
3. Randomly samples `n_tiles` non-overlapping center points (use a grid with jitter, not pure random, to guarantee non-overlap)
4. For each tile, applies a random rotation angle `θ ∈ [0°, 360°)` (continuous, not discrete)
5. Renders a **5-channel one-hot PNG** at 512×512 pixels, 5m/pixel resolution:
   - Channel 0: background (non-road)
   - Channel 1: residential / unclassified
   - Channel 2: tertiary
   - Channel 3: primary / secondary
   - Channel 4: motorway / trunk
   - Rule: if a pixel falls under multiple road levels, assign the highest-priority channel
6. Also renders a **conditioning image** at 512×512 (3-channel RGB):
   - Channel R: elevation (normalized 0–255 from SRTM or a flat placeholder for now)
   - Channel G: land use category (simplified: residential=50, commercial=150, park=200, water=10, other=100)
   - Channel B: population density (placeholder uniform value for now; can be added later)
7. Saves both as `data/{city_name}/road_{i:04d}.png` and `data/{city_name}/cond_{i:04d}.png`
8. Saves a `data/{city_name}/metadata.json` with per-tile: lat, lon, rotation, style_label

**Key parameters:**
- `tile_dist_m = 1280` — gives 2560m × 2560m ≈ 512px at 5m/px
- `edge_linewidth` — render each road channel as a rasterized line with **5px width** (not 2px) to avoid sub-pixel sparsity in the VAE
- Road color per channel: use solid fills (white=1, black=0) per channel independently; do not flatten to grayscale
- Rotation: rotate the bbox, not the image — use OSMnx's `bbox_from_point` with a rotated affine transform, OR render at 1.5× size then rotate+crop to 512

**Step 1:** Write `cdg.py` with the above spec.
**Step 2:** Test on a single city (Arlington) for 5 tiles: `python data_pipeline/cdg.py --city Arlington --n 5`.
**Step 3:** Visually inspect output PNGs — confirm 5 channels present, roads are at least 5px wide, no aliasing artifacts.
**Step 4:** Run full pipeline for all 8 cities.
**Step 5:** Commit: `data: add multi-channel multi-city road image generator`.

---

### Task 2.3: Augmentation (`data_pipeline/augment.py`)

At training time, apply additional augmentation on-the-fly (not baked into disk):
- Random horizontal flip
- Random vertical flip
- Random 90° rotation (on top of the continuous rotation already baked in)
- Random brightness/contrast jitter on the conditioning image only (not the road layout)
- **Do NOT apply color jitter to road layout channels** — they are one-hot binary

Write a PyTorch `Dataset` subclass that:
1. Loads `road_{i}.png` + `cond_{i}.png` pairs
2. Applies joint augmentation (same random transform to both)
3. Returns `(road_tensor: float32 [5, 512, 512], cond_tensor: float32 [3, 512, 512], style_label: int)`

**Step 1:** Write `augment.py` with `RoadLayoutDataset`.
**Step 2:** Write a quick sanity check: iterate 10 batches, confirm shapes and value ranges.
**Step 3:** Commit: `data: add dataset class with augmentation`.

---

## Phase 3 — VAE

### Background

CaRoLS uses a custom VAE trained specifically on 5-channel road layout images using Focal loss (to handle class imbalance — roads are sparse) + KL divergence. The VAE is trained first and then frozen. The diffusion model learns to generate latents that this VAE can decode.

### Task 3.1: VAE architecture (`model/vae.py`)

**Encoder:**
- Input: `[B, 5, 512, 512]`
- 3 encoder blocks, each: `3×3 strided conv (stride=2)` → GroupNorm → Swish
- Output: `[B, 8, 64, 64]` (mean + log-var, split to `[B, 4, 64, 64]` each)

**Decoder:**
- Input: `[B, 4, 64, 64]` (sampled latent)
- 3 decoder blocks, each: `3×3 transposed conv (stride=2)` → GroupNorm → Swish
- Final: `3×3 conv` → output `[B, 5, 512, 512]` (raw logits, not softmax)

**Loss:**
- Focal loss (α from grid-search, γ=2) on per-pixel channel classification (one-hot target)
- KL divergence term weighted by 0.0001 (CaRoLS paper Eq. 1)
- Total: `L_VAE = 0.0001 × L_KL + L_Focal`

**Step 1:** Write `model/vae.py` with `VAEEncoder`, `VAEDecoder`, `RoadVAE`, and `vae_loss()`.
**Step 2:** Smoke test: instantiate model, forward pass with random tensor, confirm output shape `[2, 5, 512, 512]`.
**Step 3:** Commit: `model: add VAE architecture`.

---

### Task 3.2: VAE training script (`model/train_vae.py`)

**What it does:**
1. Loads `RoadLayoutDataset` for training cities (Sydney+Melbourne equivalent → your 8 cities)
2. Trains for ~50 epochs with Adam, lr=0.00002, batch=4
3. Saves checkpoint every 5 epochs to `checkpoints/vae_epoch_{n}.pth`
4. Logs train loss to stdout and optionally to a CSV

**Key detail:** The VAE training does **not** use the conditioning image. It only sees road layout images. It learns the distribution of road layouts.

**Step 1:** Write `train_vae.py`.
**Step 2:** Run for 2 epochs locally (CPU or GPU) to confirm no crashes.
**Step 3:** Write `notebooks/train_vae.ipynb` as a thin Colab wrapper that mounts Google Drive, syncs data, runs `train_vae.py`.
**Step 4:** Commit: `model: add VAE training script + Colab notebook`.

---

## Phase 4 — Diffusion U-Net with CDB

### Task 4.1: Condition-aware Decoder Block (`model/cdb.py`)

The CDB replaces standard U-Net decoder blocks. Each CDB receives:
- `R_down`: downsampled encoder feature `[B, C, H, W]`
- `R_up`: upsampled feature from previous decoder block `[B, C, H, W]`
- `R_c`: intermediate conditioning representation `[B, C_c, H, W]` (or zeros in unconditional mode)

**LDE (Local Details Enhancement):**
1. `R_c` → 1×1 Conv + Swish
2. `R_down` → 1×1 Conv (project to same channels as `R_c`)
3. Channel concat → 3×3 Conv (fuse, reduce channels back to C) — **skip if `R_c` is zeros**
4. Residual add to `R_down`
5. Concat with `R_up` → 3×3 Conv → Swish + GroupNorm → `R_l`

**GCI (Global Context Integration):**
1. `Q` = 1×1 Conv on `R_l`, reshape to `[B, H×W, D]`
2. `K`, `V` = 1×1 Conv on `R_c`, reshape similarly — **if unconditional, use `R_l` for K and V too (self-attention)**
3. `S = Softmax(QK^T / √D) · V`
4. FC layer → reshape to `[B, C, H, W]`
5. Residual add to `R_l` → `R_g`

**Output:** `R_g`

**Step 1:** Write `model/cdb.py` with `LocalDetailsEnhancement`, `GlobalContextIntegration`, `ConditionAwareDecoderBlock`.
**Step 2:** Unit test both LDE and GCI with and without conditioning (zeros).
**Step 3:** Commit: `model: add Condition-aware Decoder Block`.

---

### Task 4.2: Diffusion U-Net (`model/unet.py`)

**Architecture:**
- Input: `[B, 4, 64, 64]` noisy latent + timestep embedding
- Optional conditioning image `C ∈ [B, 3, 512, 512]` → processed by `3×3 Conv + ReLU + BN` → 4 encoder blocks → intermediate representations `{R_c^0, R_c^1, R_c^2, R_c^3}` each at progressively lower resolution
- **4 encoder blocks**: 3×3 strided conv → GroupNorm → Swish → self-attention module
- **Bottleneck**
- **4 decoder blocks** using CDB (replace standard skip connections)
- Timestep embedded via sinusoidal + 2-layer MLP, added to each block

**Unconditional mode:** pass `{R_c^i}` as zero tensors → CDB skips the 3×3 fusion step in LDE, GCI becomes self-attention.

**Step 1:** Write `model/unet.py`.
**Step 2:** Smoke test: forward pass with random noisy latent + random conditioning image + random timestep.
**Step 3:** Commit: `model: add diffusion U-Net with CDB`.

---

### Task 4.3: DDPM training logic (`model/diffusion.py`)

Standard DDPM (Ho et al. 2020) with:
- Linear noise schedule, T=1000 steps
- Noise prediction objective (ε-prediction)
- Loss: MSE between predicted noise and actual noise (Eq. 6 from CaRoLS paper)
- **Classifier-free guidance**: during training, randomly zero out conditioning image with probability ρ=0.5. This enables both conditional and unconditional inference from one model.
- At inference: guidance scale `w` (try w=3.0): `ε_guided = ε_uncond + w × (ε_cond − ε_uncond)`

Write:
- `forward_diffusion(x0, t)` → noisy `x_t`, noise `ε`
- `training_loss(model, x0, cond_image)` → scalar loss
- `sample(model, cond_image, guidance_scale, n_steps)` → latent `R`

**Step 1:** Write `model/diffusion.py`.
**Step 2:** Unit test `forward_diffusion`: confirm `x_T` is approximately Gaussian.
**Step 3:** Commit: `model: add DDPM forward/reverse process`.

---

## Phase 5 — Training Orchestration

### Task 5.1: Diffusion training script (`model/train_diffusion.py`)

**What it does:**
1. Loads pretrained frozen VAE from checkpoint
2. Loads `RoadLayoutDataset`
3. For each batch: encode road layout with VAE encoder → `x0`; run `training_loss`; backprop through U-Net only
4. Adam, lr=0.00002, batch=4
5. Checkpoint every 5 epochs
6. Logs FID/KID on a small validation split every 10 epochs (use a helper that generates 64 samples, vectorizes them, and computes image metrics against held-out tiles)

**Step 1:** Write `train_diffusion.py`.
**Step 2:** Write `notebooks/train_diffusion.ipynb` as Colab wrapper.
**Step 3:** Commit: `model: add diffusion training script + Colab notebook`.

---

### Task 5.2: Evaluation metrics helper

Write `model/eval.py` with:
- `compute_fid(real_paths, generated_tensors)` — using `torchmetrics.image.fid.FrechetInceptionDistance`
- `compute_connectivity_index(vectorized_graph)` — avg(sum of node degrees / total nodes) (CI metric from CaRoLS)
- `compute_transportation_convenience(vectorized_graph)` — avg(Euclidean dist / shortest path dist) (TC metric)

These are the same metrics used in the paper so results are comparable.

**Step 1:** Write `model/eval.py`.
**Step 2:** Smoke test with a random graph.
**Step 3:** Commit: `model: add evaluation metrics`.

---

## Phase 6 — Vectorization Post-processing

### Task 6.1: Raster-to-vector pipeline (`vectorize/postprocess.py`)

**Input:** Raw model output `L ∈ [B, 5, 512, 512]` (logits)

**Steps (matching CaRoLS Section 3.3):**

1. **Semantic discretization:** `argmax` over channel dim → `S ∈ [512, 512]` integer label map (0=background, 1=residential, 2=tertiary, 3=primary/secondary, 4=motorway)
2. **Morphological buffering:** binary dilation with radius 1–2px on each road mask (channels 1–4). Bridges sub-pixel gaps, removes isolated pixels.
3. **Skeleton extraction:** `skimage.morphology.medial_axis` per road mask. Any pixel whose degree ≠ 2 in the skeleton becomes a graph node.
4. **Graph construction:** use `sknw.build_sknw()` on combined skeleton. Each edge inherits road level from majority label of pixels it covers.
5. **Topology cleaning:**
   - Discard connected components shorter than 20px total path length
   - For remaining disjoint components: connect to nearest node of main graph by Euclidean shortest link
6. **Node merging:** merge node pairs with Euclidean distance < 2px
7. **Geometric simplification:** Douglas-Peucker via `rdp` library, tolerance=1–2px per edge polyline

**Output:** NetworkX `Graph` with node attributes `(x_px, y_px)` and edge attributes `(road_level, geometry_polyline)`.

**Step 1:** Write `postprocess.py` implementing the above 7-step pipeline.
**Step 2:** Test on a real model output or on a synthetic hand-drawn 512×512 road image.
**Step 3:** Visually confirm the vectorized output by plotting the graph over the raster image.
**Step 4:** Commit: `vectorize: add raster-to-vector post-processing pipeline`.

---

### Task 6.2: Pixel-to-world coordinate transform (`vectorize/export.py`)

**Input:** NetworkX graph (pixel coords) + tile metadata (lat, lon center, rotation, meters/pixel=5)

**What it does:**
1. Computes affine transform: `pixel (x, y) → (lat, lon)` accounting for rotation and scale
2. Converts every node to `(lat, lon)` + `world_x, world_y` in game units (Cities Skylines uses 1 unit = 8m on a 17.28km × 17.28km map — 2160 units per axis)
3. Outputs a JSON structure:

```json
{
  "nodes": [{"id": 0, "x": 1024.5, "z": 512.3}, ...],
  "edges": [
    {"from": 0, "to": 1, "road_level": 2, "waypoints": [[1024.5, 512.3], ...]},
    ...
  ]
}
```

This is the exact format the C# mod will consume.

**Step 1:** Write `export.py`.
**Step 2:** Verify coordinate transform is invertible (round-trip test).
**Step 3:** Commit: `vectorize: add coordinate transform and JSON export`.

---

## Phase 7 — Inference Server

### Task 7.1: Flask inference server (`server/app.py` + `server/inference.py`)

**`inference.py`:**
- Loads VAE decoder and diffusion U-Net from checkpoint on startup (to GPU if available)
- `generate(cond_image_array, guidance_scale=3.0, n_steps=50)`:
  1. Encode conditioning image → tensor
  2. Run DDPM reverse process → latent R
  3. Decode with VAE decoder → road layout L
  4. Run vectorization pipeline → NetworkX graph
  5. Run export → JSON dict

**`app.py`:** Flask server with one endpoint:
- `POST /generate`
  - Request body: `{ "cond_image_b64": "<base64 PNG>", "guidance_scale": 3.0, "style": "us_suburb" }`
  - Response: JSON road network (nodes + edges)
- `GET /health` — returns `{"status": "ok"}`

**Note on latency:** DDPM with 50 steps on RTX 3090 ≈ 3–8 seconds. Acceptable for a "generate and confirm" UX. If too slow, reduce to 20 steps or add DDIM sampling later.

**Step 1:** Write `server/inference.py`.
**Step 2:** Write `server/app.py`.
**Step 3:** Test with a dummy conditioning image: `curl -X POST localhost:5000/generate -d '...'`.
**Step 4:** Commit: `server: add Flask inference server`.

---

### Task 7.2: Colab tunnel option

For users without a local GPU: add a `--tunnel` flag to `app.py` that uses `flask-ngrok` or `pyngrok` to expose the server over HTTPS. The mod then points to the ngrok URL instead of `localhost`.

Write a `notebooks/inference_server.ipynb` that:
1. Installs dependencies
2. Loads model from Google Drive
3. Starts the Flask server with ngrok tunnel
4. Prints the public URL (user pastes into mod settings)

**Step 1:** Write `notebooks/inference_server.ipynb`.
**Step 2:** Commit: `server: add Colab inference notebook with ngrok tunnel`.

---

## Phase 8 — Cities Skylines C# Mod

### Background

CS1 runs on Unity 5 with Mono / .NET Framework 3.5. Mods are DLLs loaded at runtime. The game's API lives in `Assembly-CSharp.dll`, `ICities.dll`, and `ColossalManaged.dll` at:
```
C:\Program Files (x86)\Steam\steamapps\common\Cities_Skylines\Cities_Data\Managed\
```

All road-placement calls must happen on the **simulation thread** via `SimulationManager.instance.AddAction(() => { ... })`.

Harmony patching requires the **CitiesHarmony** Workshop mod (ID 2040656402) to be subscribed. Reference isolation: keep all HarmonyLib calls in `Patcher.cs`.

### Task 8.1: Visual Studio project setup (`mod/Groundwork.csproj`)

**References to add:**
- `Assembly-CSharp.dll`
- `ICities.dll`
- `ColossalManaged.dll`
- `UnityEngine.dll`
- `UnityEngine.UI.dll`
- `0Harmony.dll` (from CitiesHarmony mod folder)

**Build output:** copy DLL to `%LOCALAPPDATA%\Colab\Cities_Skylines\Addons\Mods\Groundwork\`

**Step 1:** Create the `.csproj` file with the above references.
**Step 2:** Create `Mod.cs` with a minimal `IUserMod` implementation (name + description only) and verify it loads in-game without errors.
**Step 3:** Commit: `mod: scaffold C# project and verify IUserMod loads`.

---

### Task 8.2: Brush tool (`mod/src/BrushTool.cs`)

Subclass `ToolBase`. The tool needs to:

1. **Activate/deactivate:** called when player selects/deselects the tool from the toolbar
2. **RayCast** each frame to find the terrain point under the cursor
3. **Render:** draw a circle overlay on the terrain showing brush radius (`RenderManager.instance.OverlayEffect.DrawCircle(...)`)
4. **On left-click:** capture the center world position and radius, then trigger generation (see Task 8.4)
5. **Radius control:** mouse scroll wheel adjusts radius (min 64u, max 512u — roughly 0.5–4km)
6. **Cursor:** set `ToolCursor` to a custom circular cursor or the default

**Step 1:** Write `BrushTool.cs`.
**Step 2:** Register the tool in `Loading.cs` via `ToolsModifierControl.toolController.SetTool<BrushTool>()`.
**Step 3:** Add a toolbar button (use `UITabstrip` or a simple `UIButton` on the main toolbar).
**Step 4:** Test in-game: confirm circle overlay renders and radius changes with scroll.
**Step 5:** Commit: `mod: add brush tool with terrain overlay`.

---

### Task 8.3: Heightmap + zone extractor (`mod/src/HeightmapExtractor.cs`)

**What it produces:** a 512×512 PNG conditioning image (matching the Python server's expected input) from a given world-space bounding box.

**Channels:**
- **Red (elevation):** sample `TerrainManager.instance.SampleRawHeightSmooth(worldPos)` on a 512×512 grid over the bbox. Normalize to 0–255.
- **Green (land use):** iterate `ZoneManager` blocks over the bbox. Map zone types to values: `Residential=50`, `Commercial=100`, `Office=120`, `Industrial=150`, `Park=200`, `None=0`. Rasterize to 512×512.
- **Blue (water/terrain mask):** sample `TerrainManager.instance.SampleRawHeightSmoothWithWater(worldPos)` — if water depth > 0.5m, set pixel to 10 (water marker). Otherwise 128.

**Output:** `byte[]` PNG (encoded with Unity's `Texture2D.EncodeToPNG()`), which is then base64-encoded for the HTTP request.

**Step 1:** Write `HeightmapExtractor.cs` with `ExtractConditioningImage(Vector3 center, float radius)`.
**Step 2:** Test: save the PNG to `%TEMP%\groundwork_cond.png` and inspect it visually.
**Step 3:** Commit: `mod: add heightmap + zone conditioning image extractor`.

---

### Task 8.4: HTTP client (`mod/src/RoadClient.cs`)

**What it does:**
1. Sends a `POST /generate` to the inference server with base64-encoded conditioning image + style label
2. Parses the JSON response (nodes + edges)
3. Returns a `RoadNetwork` object (simple C# data class: `List<Node>`, `List<Edge>`)

**Use `System.Net.HttpWebRequest`** (no external libraries — .NET 3.5).
**Run on a background `Thread`** to avoid blocking the game's main thread.
**Post result back** to the main thread via `ConcurrentQueue<RoadNetwork>` polled in `ThreadingExtensionBase.OnUpdate()`.

**Server URL:** configurable via a settings panel (default: `http://localhost:5000`). Store in a static `ModSettings` class using `SavedString` (CS1's built-in settings persistence).

**Step 1:** Write `RoadClient.cs` with async HTTP call on background thread.
**Step 2:** Write `ModSettings.cs` with server URL persistence.
**Step 3:** Test: call the server from the mod, confirm the JSON arrives and parses correctly (log to `Debug.Log`).
**Step 4:** Commit: `mod: add async HTTP client`.

---

### Task 8.5: Road placer (`mod/src/RoadPlacer.cs`)

**Input:** `RoadNetwork` (nodes + edges with road levels and waypoints in world coords)

**Road level → NetInfo prefab mapping:**
| Road level | `PrefabCollection<NetInfo>.FindLoaded()` key |
|------------|----------------------------------------------|
| 4 (motorway) | `"Highway"` |
| 3 (primary/secondary) | `"Large Road"` |
| 2 (tertiary) | `"Medium Road"` |
| 1 (residential) | `"Basic Road"` |

**Algorithm:**
1. For each node: `NetManager.instance.CreateNode(out ushort nodeId, ref rand, netInfo, position, buildIndex)` — **must run on simulation thread via `AddAction`**
2. For each edge: iterate waypoints; create intermediate nodes for curves; call `NetManager.instance.CreateSegment(out ushort segId, ref rand, netInfo, startNode, endNode, startDir, endDir, buildIndex, modifiedIndex, invert)`
3. Snap start/end nodes to existing road network nodes within 8u radius to avoid floating dangling ends
4. **Water body check:** before placing any segment, check all waypoint world positions with `TerrainManager.SampleRawHeightSmoothWithWater` — skip segments whose majority of waypoints are over water (depth > 0.5m)

**Step 1:** Write `RoadPlacer.cs`.
**Step 2:** Test with a hardcoded simple cross-shaped network (2 segments) to confirm NetManager calls work.
**Step 3:** Test with a server-generated network from a flat rural area.
**Step 4:** Commit: `mod: add road placer with water body exclusion`.

---

### Task 8.6: Wire everything together (`mod/src/Loading.cs`, `mod/src/BrushTool.cs`)

**Full interaction flow:**

```
1. Player activates Groundwork brush tool from toolbar
2. Player adjusts radius with scroll wheel
3. Player selects road style (US Suburb / US Grid / Organic) from a simple dropdown panel
4. Player left-clicks on map
5. BrushTool calls HeightmapExtractor.ExtractConditioningImage(center, radius)
6. BrushTool calls RoadClient.GenerateAsync(condImage, style) on background thread
7. UI shows a "Generating..." spinner overlay
8. On response: BrushTool receives RoadNetwork
9. BrushTool shows a ghost preview (draw the edges as overlay lines)
10. Player confirms (right-click = cancel, left-click again = confirm)
11. On confirm: BrushTool calls RoadPlacer.PlaceRoads(roadNetwork) on simulation thread
12. UI spinner disappears
```

**Step 1:** Implement the full flow in `BrushTool.cs` using a simple state machine (`Idle → Waiting → Preview → Placing`).
**Step 2:** Add a minimal UI panel (style selector dropdown + server URL input) using CS1's `UIPanel` / `UIDropDown`.
**Step 3:** End-to-end test: brush an area, confirm roads appear in-game.
**Step 4:** Commit: `mod: wire full brush → generate → preview → place flow`.

---

## Phase 9 — Integration & Style Conditioning

### Task 9.1: Style conditioning via CFG guidance

The diffusion model is already trained with classifier-free guidance. At inference time, encode the style label as part of the conditioning:

- **Option A (simple):** pass style as a text string to the server; server uses it to select a subset of training data for guidance (e.g., only US suburb tiles in the guidance direction)
- **Option B (proper):** add a style embedding (learned 64-dim vector, one per style class) concatenated to the timestep embedding during training. This requires a retraining pass.

**Recommendation:** Start with Option A for the first end-to-end working version. Add Option B in a later training iteration once the pipeline is proven.

**Step 1:** Implement Option A in `server/inference.py`: accept `style` parameter, filter conditioning image normalization per style (e.g., adjust expected road density).
**Step 2:** Commit: `model: add style hint via server-side CFG guidance`.

---

### Task 9.2: Inpainting mode (boundary conditioning)

To support the "fill within existing city" use case (CaRoLS Fig. 7), add an inpainting mode:

1. Before generating, `HeightmapExtractor` also extracts the **existing road network** in the brushed area as a binary road mask (query `NetManager` for all existing segments in the bbox, rasterize to 512×512)
2. The mask is passed as an additional channel of the conditioning image (4th channel: existing roads)
3. During diffusion inference, use DDPM inpainting: known pixels (existing roads at the boundary) are preserved; only the interior is denoised

This maps to CaRoLS Fig. 7's "local optimization" scenario and is essential for placing new neighborhoods that connect to existing roads.

**Step 1:** Add existing road rasterization to `HeightmapExtractor.cs`.
**Step 2:** Add inpainting logic to `server/inference.py` (boundary masking during reverse diffusion).
**Step 3:** Test: brush an area adjacent to existing roads; confirm generated roads connect at the boundary.
**Step 4:** Commit: `mod+server: add inpainting mode for boundary-coherent generation`.

---

## Testing Strategy

### Data pipeline tests
- Confirm 5-channel output sums to 1 per pixel (valid one-hot)
- Confirm tile non-overlap: no two tiles from the same city share >10% pixel overlap
- Confirm arbitrary rotation: histogram of rotation angles is roughly uniform

### Model tests
- VAE reconstruction: encode then decode a real road image; confirm structural similarity (SSIM > 0.7)
- Diffusion: generated latents should have similar statistics to encoded real latents
- Connectivity index of generated road layouts should be > 1.8 (CaRoLS benchmark)

### Mod tests
- `RoadPlacer`: test with a minimal cross (2 segments), confirm node IDs are valid
- `HeightmapExtractor`: verify conditioning image has non-trivial elevation variance on hilly terrain
- `RoadClient`: test with server down → should fail gracefully with user notification (not crash)
- End-to-end: brush a 256u radius circle on flat suburban terrain; confirm ≥10 road segments placed

---

## Milestones

| # | Milestone | Deliverable |
|---|-----------|-------------|
| M1 | Data pipeline complete | 1,200+ 5-channel road tiles from 8 cities on disk |
| M2 | VAE trained | Reconstruction SSIM > 0.7 on held-out Brisbane-equivalent tiles |
| M3 | Diffusion trained | Generated layouts CI > 1.8, FID < 40 (unconditional) |
| M4 | Server running | `POST /generate` returns valid JSON road network in < 10s |
| M5 | Mod brush works | Brush renders overlay, conditioning image exported correctly |
| M6 | End-to-end works | Brush → generate → roads appear in-game |
| M7 | Style conditioning | US suburb vs. US grid visually distinguishable |
| M8 | Inpainting works | Generated roads connect to existing road network at boundary |

---

## Known Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| VAE doesn't reconstruct sparse roads well | Increase Focal loss weight; increase line width to 7px |
| Diffusion generates disconnected roads | Post-processing topology cleaning (Phase 6 Task 6.1 step 5); adjust guidance scale |
| NetManager calls crash game | Always use `SimulationManager.AddAction`; test with minimal network first |
| Inference too slow for good UX | Reduce to 20 DDIM steps; show preview after 5 steps |
| Roads cross water bodies | Water mask check in `RoadPlacer` (Phase 8 Task 8.5) |
| Single-city training bias | 8 cities in data pipeline; verify style diversity in M1 |
| OSMnx download rate limits | Cache downloaded graphs locally; add `--cache` flag to `cdg.py` |
