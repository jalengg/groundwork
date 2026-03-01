# CitiesGPT Training Pipeline — Design Document

**Date:** 2026-02-28
**Scope:** Data pipeline + model training only (no mod integration)
**Goal:** Reproduce CaRoLS using conditioning layers extractable from Cities Skylines 1, trained on real-world US suburb OSM data

---

## 1. What We Are Building

A two-stage diffusion model that takes a 4-channel conditioning image (elevation, land use, water/no-build, existing roads) and generates a realistic 5-channel road layout image for a 2.56km × 2.56km tile. Trained entirely on real-world OpenStreetMap data. At inference time inside CS1, the same 4 channels are extracted from the game instead.

The model is a faithful reproduction of the CaRoLS architecture (Feng et al., Computers & Graphics 2025), with three modifications:
- Population density channel dropped (not reliably available from CS1 early-game)
- Existing roads added as a 4th conditioning channel (enables connectivity to player's existing network)
- Training data is US suburbs rather than Australian metropolitan areas

---

## 2. Conditioning Channels

4-channel conditioning image, 512×512px, all layers share one affine transform per tile.

| Ch | Name | Training source | CS1 source |
|----|------|----------------|------------|
| 0 | Elevation | SRTM 30m via `elevation` Python library | `TerrainManager.SampleRawHeightSmooth()` |
| 1 | Land use | OSM `landuse=*` + `leisure=*` polygons | `ZoneManager` zone types |
| 2 | Water + no-build | OSM `natural=water`, `waterway=*` polygons | `TerrainManager` water depth + user brush |
| 3 | Existing roads | OSMnx drive network, all highway types | `NetManager` segment rasterization |

**Land use encoding:** residential=0.2, commercial=0.4, industrial=0.6, park/recreation=0.8, other=0.1, no data=0.0

**Water channel:** binary 0/1. At CS1 inference time, the user's "do not build here" brush adds to this channel — treated identically to water (model already learned to avoid it).

**Existing roads:** rasterized at 3px width, binary 0/1.

---

## 3. Road Output Format

5-channel one-hot image, 512×512px. Same format as CaRoLS.

| Ch | Road level | OSM highway types |
|----|-----------|-------------------|
| 0 | Background | (no road) |
| 1 | Residential | `residential`, `unclassified`, `living_street` |
| 2 | Tertiary | `tertiary`, `tertiary_link` |
| 3 | Primary / Secondary | `primary`, `secondary`, `*_link` |
| 4 | Motorway / Trunk | `motorway`, `trunk`, `*_link` |

Priority rule: if a pixel falls under multiple road levels, assign the highest-priority channel (Ch4 > Ch3 > Ch2 > Ch1 > Ch0).

---

## 4. Data Pipeline

**Tile spec:** 512×512px at 5m/px = 2.56km × 2.56km per tile. All layers rasterized onto the same affine grid using `rasterio.features.rasterize()`. Stored as GeoTIFF pairs per tile.

**Rasterization approach (Approach B):** GeoPandas + rasterio. All OSM geometries projected to a common UTM CRS before rasterization. Guarantees pixel-perfect alignment across all channels.

**Tile sampling:** Grid-based with random jitter (not pure random) to guarantee non-overlap. Grid cell = 2.56km. Jitter up to ±30% per axis. Rotation: continuous uniform θ ∈ [0°, 360°) baked into the affine transform.

**Cities:** 8 US suburb cities, ~150 non-overlapping tiles each → ~1,200 tiles total. All labelled `us_suburb` (style conditioning to be added in a future iteration).

| City | OSMnx query |
|------|-------------|
| Arlington, TX | `Arlington, Texas, USA` |
| Chandler, AZ | `Chandler, Arizona, USA` |
| Gilbert, AZ | `Gilbert, Arizona, USA` |
| Henderson, NV | `Henderson, Nevada, USA` |
| Mesa, AZ | `Mesa, Arizona, USA` |
| Tempe, AZ | `Tempe, Arizona, USA` |
| Plano, TX | `Plano, Texas, USA` |
| Irving, TX | `Irving, Texas, USA` ← validation only |

**SRTM elevation:** `elevation` Python library. Caches downloaded SRTM tiles locally (each 1°×1° tile covers most cities). Bilinear interpolated to 5m/px grid.

**OSM data:** fetched via OSMnx per tile bbox. Road graphs cached locally after first download to avoid repeated API calls.

---

## 5. Model Architecture

### Stage 1 — VAE

Trained on road layout images only. Frozen after training. The diffusion model works entirely in this VAE's latent space.

```
Encoder:  [B, 5, 512, 512]
          → 3× (3×3 strided conv stride=2, GroupNorm, Swish)
          → [B, 8, 64, 64]  →  split μ, σ  →  sample z
          → [B, 4, 64, 64]

Decoder:  [B, 4, 64, 64]
          → 3× (3×3 transposed conv stride=2, GroupNorm, Swish)
          → 3×3 conv
          → [B, 5, 512, 512]  (logits)
```

Loss: `L_VAE = 0.0001 × L_KL + L_Focal`
- Focal loss γ=2, α per channel by inverse pixel frequency
- KL divergence keeps latent space Gaussian

### Stage 2 — Diffusion U-Net with CDB

Learns to generate latents `R ∈ ℝ^{64×64×4}` conditioned on the 4-channel conditioning image.

```
C [B, 4, 512, 512]  →  3×3 Conv + ReLU + BN
                    →  4 encoder blocks
                    →  {R_c^0, R_c^1, R_c^2, R_c^3}

τ [B, 4, 64, 64]  +  timestep embedding
                  →  4 encoder blocks (3×3 strided conv + GroupNorm + Swish + self-attention)
                  →  bottleneck
                  →  4 CDB decoder blocks  ←  each fed R_c^i
                  →  ε̂ [B, 4, 64, 64]
```

**Condition-aware Decoder Block (CDB):**
- LDE (Local Details Enhancement): fuses conditioning features with encoder skip features via 1×1 conv + Swish + channel concat + 3×3 conv. Skipped in unconditional mode.
- GCI (Global Context Integration): cross-attention (Q from road features, K/V from conditioning). Falls back to self-attention in unconditional mode.

**Loss:** DDPM ε-prediction MSE

**Classifier-free guidance:** ρ=0.5 (zero out conditioning with 50% probability during training). Guidance scale w=3.0 at inference.

**Inference:** T=50 DDIM steps (~3–6s on RTX 3090).

---

## 6. Training Configuration

**Hardware:** Google Colab A100 (40GB VRAM)

### Stage 1 — VAE
- Train/val split: 7 cities train / Irving TX val (~1050/150 tiles)
- Epochs: 50 | Batch: 4 | LR: 2e-5 | Optimizer: Adam (β1=0.9, β2=0.999)
- Checkpoint: every 5 epochs to Google Drive
- **Done when:** validation SSIM > 0.70

### Stage 2 — Diffusion
- Load frozen VAE encoder; pre-encode all tiles to latents; cache to disk
- Epochs: 200 | Batch: 4 | LR: 2e-5 | Optimizer: Adam
- Checkpoint: every 10 epochs to Google Drive
- **Done when:** visual inspection coherent AND CI > 1.8

### Data Augmentation (on-the-fly)
- Random horizontal + vertical flip (joint across all channels)
- Brightness/contrast jitter on conditioning channels only
- No additional rotation (baked into tile generation)

---

## 7. Evaluation Suite

| Metric | Type | Frequency | What it catches |
|--------|------|-----------|----------------|
| FID | Numeric | Every 10 epochs | Distribution-level image quality |
| KID | Numeric | Every 10 epochs | Distribution-level image quality (unbiased) |
| CI (Connectivity Index) | Numeric | Every 10 epochs | Topological soundness of road graph |
| TC (Transportation Convenience) | Numeric | Every 10 epochs | Travel efficiency of generated network |
| **VLM Realism Score** | **Qualitative 1–10** | **Every 20 epochs, 10 samples** | **Psychedelic artifacts, visual slop** |

**VLM evaluation:** Send generated samples to Claude API with structured prompt:
> *"Rate this generated road network image 1–10 on realism. A 10 looks like a real US suburb from OpenStreetMap. A 1 has disconnected fragments, swirling artifacts, or implausible intersections. List the specific issues you see."*

**CaRoLS benchmarks to beat (unconditional mode):** KID=0.0331, FID=32.2, CI=1.948, TC=0.668

---

## 8. Key Risks

| Risk | Mitigation |
|------|-----------|
| VAE doesn't reconstruct sparse roads | Increase Focal loss α; widen road rasterization to 5px |
| SRTM API rate limits | Cache SRTM tiles locally after first fetch; batch by city |
| OSM data gaps (missing landuse polygons) | Default missing landuse to 0.0; log coverage % per city |
| Diffusion generates disconnected roads | Post-processing topology cleaning (separate pipeline) |
| Colab session timeout during long training | Checkpoint every 10 epochs; auto-resume from latest checkpoint |
