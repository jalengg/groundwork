# Groundwork — Session Context
*Last updated: 2026-03-07. Keep this file current after each session.*

---

## High-Level Objective

**Groundwork** is a Cities Skylines 1 Steam Workshop mod that uses a trained ML diffusion model to generate realistic road networks for a selected tile. The user selects an area, the mod reads terrain data from the game, passes it to the model, and the model outputs a road layout which the mod then builds in-game.

**Current focus: training pipeline only.** The mod integration comes after a working trained model exists.

**Architecture being reproduced:** CaRoLS ("Condition-adaptive multi-level Road Layout Synthesis", Feng et al., *Computers & Graphics* 2025). PDF at `/mnt/c/Users/jalen/Downloads/1-s2.0-S0097849325002924-main.pdf`.

---

## Project Layout

```
groundwork/                    ← project root (inside /mnt/c/Users/jalen/)
├── .venv/                     ← Python venv; use .venv/bin/pytest, .venv/bin/python
├── data_pipeline/
│   ├── cities.yaml            ← 8 US suburb city configs (7 train, Irving TX = val)
│   ├── tile_grid.py           ← generate_tile_centers() — non-overlapping grid with jitter
│   ├── elevation_layer.py     ← fetch_elevation_grid() — SRTM HGT direct download
│   ├── osm_layers.py          ← fetch_water_grid(), fetch_landuse_grid() — OSMnx + rasterio
│   ├── road_layers.py         ← fetch_road_graph(), rasterize_roads_binary(), rasterize_road_output()
│   ├── tile_assembler.py      ← assemble_tile() — oversized → rotate → crop pattern
│   ├── cdg.py                 ← CLI: python cdg.py --config ... --output data/ [--city name]
│   └── dataset.py             ← RoadLayoutDataset (loads cond_NNNN.npy + road_NNNN.npy)
├── model/
│   ├── vae.py                 ← RoadVAE (VAEEncoder + VAEDecoder + ResBlock) ✅
│   ├── vae_loss.py            ← focal_loss + kl_loss + vae_loss() ← NEXT: Task 10
│   ├── train_vae.py           ← VAE training script ← NEXT: Task 10
│   ├── cdb.py                 ← ConditionAwareDecoderBlock (LDE + GCI) ← Task 11
│   ├── unet.py                ← DiffusionUNet ← Task 12
│   ├── diffusion.py           ← DDPM class (forward_diffusion, training_loss, sample_ddim) ← Task 13
│   ├── train_diffusion.py     ← diffusion training script ← Task 14
│   ├── eval_metrics.py        ← compute_connectivity_index, compute_transport_convenience, ImageQualityTracker ✅
│   └── vlm_eval.py            ← score_samples() — Claude API realism scorer ✅
├── tests/
│   ├── test_data_pipeline.py  ← cities.yaml + tile_grid tests
│   ├── test_elevation_layer.py
│   ├── test_osm_layers.py
│   ├── test_road_layers.py
│   ├── test_tile_assembler.py
│   ├── test_dataset.py
│   ├── test_vae.py            ✅
│   └── test_eval_metrics.py   ✅
├── notebooks/                 ← Colab notebooks (Tasks 17-18, not yet written)
└── docs/plans/
    ├── 2026-02-28-training-pipeline-design.md    ← approved design doc
    └── 2026-02-28-training-pipeline-implementation.md  ← 18-task plan with full code
```

---

## Task Status

| # | Task | Status | File(s) | Commit |
|---|------|--------|---------|--------|
| 1 | Scaffold project + requirements | ✅ | requirements.txt, .gitignore, __init__.py files | 48f0261 |
| 2 | City configuration (cities.yaml) | ✅ | data_pipeline/cities.yaml | 30784c8 |
| 3 | Tile grid generator | ✅ | data_pipeline/tile_grid.py | 3e4f5a9 |
| 4 | SRTM elevation layer | ✅ | data_pipeline/elevation_layer.py | 965e270 |
| 5 | OSM water + land use layers | ✅ | data_pipeline/osm_layers.py | 253e74d |
| 6 | Road rasterizer | ✅ | data_pipeline/road_layers.py | 6a1e14a |
| 7 | Tile assembler | ✅ | data_pipeline/tile_assembler.py | 0174aeb |
| 8 | Pipeline CLI + PyTorch Dataset | ✅ | data_pipeline/cdg.py, dataset.py | ff046bb |
| 9 | VAE architecture | ✅ | model/vae.py | 449983b |
| 10 | VAE loss + training script | ✅ | model/vae_loss.py, model/train_vae.py | 3e0a71e |
| 11 | CDB block | ✅ | model/cdb.py | e487b94 |
| 12 | Diffusion U-Net | ✅ | model/unet.py | e487b94 |
| 13 | DDPM training logic | ✅ | model/diffusion.py | c5de85f |
| 14 | Diffusion training script | ✅ | model/train_diffusion.py | c5de85f |
| 15 | Numeric eval metrics (FID/KID/CI/TC) | ✅ | model/eval_metrics.py | a0ee6ed |
| 16 | VLM realism scorer | ✅ | model/vlm_eval.py | dc94e31 |
| 17 | VAE Colab notebook | ✅ | notebooks/train_vae.ipynb | 4e21d56 |
| 18 | Diffusion Colab notebook | ✅ | notebooks/train_diffusion.ipynb | 4e21d56 |

**All 18 tasks complete. Training pipeline fully implemented.**

---

## Architecture Details

### Conditioning (input to diffusion model)
4-channel 512×512 image:
| Ch | Name | Source |
|----|------|--------|
| 0 | Elevation | SRTM 30m, normalized 0-1 within tile |
| 1 | Land use | OSM landuse polygons, float encoding (residential=0.2, commercial=0.4, industrial=0.6, park=0.8) |
| 2 | Water + no-build | OSM water polygons, binary 0/1 |
| 3 | Existing roads | OSMnx road graph, binary rasterized at 3px width |

### Road output (training target)
5-channel one-hot 512×512:
| Ch | Road type |
|----|-----------|
| 0 | Background |
| 1 | Residential / unclassified / living_street |
| 2 | Tertiary |
| 3 | Primary / secondary |
| 4 | Motorway / trunk |

Priority rule: Ch4 > Ch3 > Ch2 > Ch1 > Ch0 (higher overwrites lower).

### Stage 1 — VAE (`model/vae.py`)
```
Encoder: [B, 5, 512, 512]
  → Conv2d(5, 64, 3)
  → ResBlock(64, stride=2)          # 512→256, shortcut: avg_pool2d(x, 2)
  → Conv2d(64, 128, 3)
  → ResBlock(128, stride=2)         # 256→128
  → Conv2d(128, 256, 3)
  → ResBlock(256, stride=2)         # 128→64
  → Conv2d(256, 4, 1) → μ
  → Conv2d(256, 4, 1) → logvar
  → reparameterize → z [B, 4, 64, 64]

Decoder: [B, 4, 64, 64]
  → Conv2d(4, 256, 3)
  → ResBlock(256, stride=2, upsample=True)   # 64→128, shortcut: interpolate(x, 2)
  → Conv2d(256, 128, 3)
  → ResBlock(128, stride=2, upsample=True)   # 128→256
  → Conv2d(128, 64, 3)
  → ResBlock(64, stride=2, upsample=True)    # 256→512
  → Conv2d(64, 5, 3)
  → [B, 5, 512, 512] logits
```
ResBlock uses GroupNorm(8) + SiLU. **Key bug fixed:** downsampling shortcut uses `F.avg_pool2d(x, stride)`, not raw `x` (spatial mismatch).

Training: Adam lr=2e-5, batch=4, 50 epochs, checkpoint every 5 epochs. Done when val SSIM > 0.70.
Loss: `L = focal(γ=2, α=1/freq) + 0.0001 × KL`

### Stage 2 — Diffusion U-Net (`model/unet.py`, not yet implemented)
```
ConditionEncoder:  [B, 4, 512, 512] → 4 feature maps at res 256, 128, 64, 32

Noise encoder (on 64×64 latent):
  UNetBlock(4→64, stride=2)    # 64→32
  UNetBlock(64→128, stride=2)  # 32→16
  UNetBlock(128→256, stride=2) # 16→8
  UNetBlock(256→256, stride=2) # 8→4
  bottleneck UNetBlock(256→256)

CDB decoder (4 blocks, each: CDB then UNetBlock upsample):
  CDB1(256,256) + UNetBlock(512→256, upsample)  # 4→8
  CDB2(256,256) + UNetBlock(512→128, upsample)  # 8→16
  CDB3(128,128) + UNetBlock(256→64, upsample)   # 16→32
  CDB4(64,64)   + UNetBlock(128→64, upsample)   # 32→64

Conv2d(64, 4, 1) → ε̂ [B, 4, 64, 64]
```
Timestep embedding: sinusoidal → MLP(256→1024→256).
Training: Adam lr=2e-5, batch=4, 200 epochs. CFG dropout ρ=0.5. Inference: DDIM T=50, w=3.0.

### CDB Block (`model/cdb.py`, not yet implemented)
```
LDE (Local Details Enhancement):
  - If conditional: cond_proj(R_c) + skip_proj(R_down) → fuse → add to R_down
  - cat([R_down, R_up]) → up_fuse → GroupNorm → SiLU → R_l

GCI (Global Context Integration):
  - Q from R_l, K/V from R_c (conditional) or R_l (unconditional)
  - Scaled dot-product attention → out_proj → residual add → R_g

CDB: R_g = GCI(LDE(R_down, R_up, R_c), R_c)
```

### DDPM (`model/diffusion.py`, not yet implemented)
- Linear beta schedule: β₁=1e-4 → β_T=0.02, T=1000
- Forward: x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε
- Loss: MSE(ε_pred, ε) with CFG: zero out cond with prob ρ=0.5
- Sampling: DDIM with n_steps=50, guidance_scale=3.0

---

## Critical Implementation Notes

### OSMnx 2.x API change (FIXED in committed code)
Old: `ox.graph_from_bbox(north, south, east, west, ...)`
New: `ox.graph_from_bbox((west, south, east, north), ...)`
Same for `features_from_bbox`. Both fixed in `osm_layers.py` and `road_layers.py`.

### Elevation library broken (FIXED)
`elevation` library requires `gdalbuildvrt` (GDAL CLI), not available. Replaced with direct SRTM1 HGT download from AWS (`https://s3.amazonaws.com/elevation-tiles-prod/skadi/N{lat:02d}/{name}.hgt.gz`), decompressed with gzip, read with rasterio. See `elevation_layer.py::_download_hgt()`.

### VAE ResBlock shortcut (FIXED in committed code)
Downsampling ResBlock: shortcut must use `F.avg_pool2d(x, stride)`, NOT raw `x`.
Upsampling ResBlock: shortcut uses `F.interpolate(x, scale_factor=2)`.

### Test file convention
Each module has its own test file (e.g., `test_vae.py`, not `test_model.py`) to allow parallel subagent implementation without git conflicts.

### Background subagents can't use Bash
When launching background subagents, they hit permission prompts for Bash and stall. Solution: write files via subagents but run tests and commits directly from the main session.

### Oversized grid pattern
All layer fetchers (`_bbox_latlon`) compute a bbox oversized by √2 so after rotating by any angle and center-cropping, the tile is fully covered. The assembler fetches all layers at `oversized = int(tile_size_px * √2) + 4` pixel resolution.

### Running tests
```bash
cd /mnt/c/Users/jalen/groundwork
.venv/bin/pytest tests/test_vae.py -v          # single file
.venv/bin/pytest tests/ -v                     # all tests
```

---

## Evaluation Targets (CaRoLS paper, unconditional mode)
| Metric | CaRoLS | What it measures |
|--------|--------|-----------------|
| KID | 0.0331 | Distribution quality (unbiased) |
| FID | 32.2 | Distribution quality |
| CI | 1.948 | Connectivity Index (avg node degree) |
| TC | 0.668 | Transport Convenience (euclidean/path ratio) |

VLM scorer: Claude API, prompt rates 1-10, runs every 20 training epochs on 10 samples.

---

## Cities
| City | Split | OSMnx query |
|------|-------|-------------|
| arlington_tx | train | "Arlington, Texas, USA" |
| chandler_az | train | "Chandler, Arizona, USA" |
| gilbert_az | train | "Gilbert, Arizona, USA" |
| henderson_nv | train | "Henderson, Nevada, USA" |
| mesa_az | train | "Mesa, Arizona, USA" |
| tempe_az | train | "Tempe, Arizona, USA" |
| plano_tx | train | "Plano, Texas, USA" |
| irving_tx | **val** | "Irving, Texas, USA" |

Tile spec: 512×512px at 5m/px = 2.56km × 2.56km. ~150 tiles/city → ~1200 total.

---

## Training Flow (once all code is written)
1. Generate tiles: `python data_pipeline/cdg.py --config data_pipeline/cities.yaml --output data/`
2. Train VAE: `python model/train_vae.py --data data/ --output checkpoints/vae/ --epochs 50`
   - Target: val SSIM > 0.70
3. Train diffusion: `python model/train_diffusion.py --vae checkpoints/vae/vae_epoch_050.pth --data data/ --output checkpoints/diffusion/ --epochs 200`
   - Target: CI > 1.8, visual coherence
4. Evaluate: FID/KID/CI/TC every 10 epochs, VLM score every 20 epochs

All training intended for Google Colab A100 (40GB VRAM). Notebooks in `notebooks/`.
