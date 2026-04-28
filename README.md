# Groundwork

A research prototype for generating realistic US suburban road networks using a CaRoLS-style two-stage diffusion model. Conditioning on terrain (elevation, water) and **categorical land use** (residential, commercial, industrial, parkland, agricultural), the model generates road maps that can later be consumed by a Cities Skylines 1 mod.

> **Status: research prototype.** The ML pipeline trains and produces road-shaped outputs. Known limitations: residential street channel is noisy/scattered, and the in-game mod integration is not yet built. See [`docs/findings.md`](docs/findings.md) for what we tried and what didn't work.

## Architecture

Reproduces (loosely) [CaRoLS: Condition-adaptive multi-level road layout synthesis](https://www.sciencedirect.com/science/article/pii/S0097849325002924) (Feng et al., *Computers & Graphics* 2025).

Two stages:

1. **VAE** (frozen after training) — compresses a `(5, 512, 512)` road map into a `(4, 64, 64)` latent and decodes it back. 5 channels = one-hot road class (background, residential, tertiary, primary/secondary, motorway/trunk). Per-class line widths during rasterization (residential `2px` to motorway `5px`) match real-world relative road widths. Trained with focal loss + KL using paper-specified per-class alphas `[0.1, 0.3, 0.6, 1.0, 2.0]`.
2. **Conditional diffusion U-Net** — generates latent codes from Gaussian noise, optionally conditioned on a `(7, 512, 512)` image with **categorical landuse one-hot**: `[elevation, water, residential, commercial, industrial, parkland, agricultural]`. DDIM sampling with classifier-free guidance. Optionally trained with class-weighted denoising loss (DRoLaS Eq. 9).

```
conditioning (7×512×512)   [elev, water, 5 landuse classes]
    ↓
DDIM 50-step denoising (CFG, w=3)
    ↓
latent z₀ (4×64×64)
    ↓
VAE decoder
    ↓
road logits (5×512×512) → argmax
    ↓
post-processing (dilate → skeletonize → prune)
    ↓
final road map
```

## Repo layout

```
data_pipeline/   OSM + SRTM tile generation, PyTorch Dataset
model/           VAE, U-Net, diffusion, training/sampling/post-processing scripts
notebooks/       Colab training notebooks
tests/           Unit tests for each module
slurm_*.sh       UIUC Campus Cluster job scripts
docs/            Architecture notes, findings writeup
samples/         Inference comparison PNGs (committed)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline

### 1. Generate training tiles

Pulls OSM road graphs, OSM landuse polygons, and SRTM elevation tiles for cities listed in `data_pipeline/cities.yaml`. Tiles are 512×512 at 5m/pixel.

```bash
python data_pipeline/cdg.py --config data_pipeline/cities.yaml --output data/
```

### 2. Train VAE

```bash
python model/train_vae.py --data data/ --output checkpoints/vae/ --epochs 50
```

### 3. Train diffusion U-Net

```bash
python model/train_diffusion.py \
    --vae checkpoints/vae/vae_epoch_050.pth \
    --data data/ --output checkpoints/diffusion/ \
    --epochs 200 --lr 2e-5 --cfg-prob 0.5
```

`--cfg-prob` is the probability of dropping the conditioning during a training step (CaRoLS spec is `0.5`). Diagnostics are saved every 5 epochs (val loss) and every 25 epochs (sample images).

Optional: enable DRoLaS-style **class-weighted denoising loss** with `--class-weights "1.0,1.2,1.4,1.4,1.4"` (background → motorway).

### 3b. Re-encode existing tiles after a pipeline change

If you change the conditioning encoding or road rasterization, regenerate `cond_*.npy` and `road_*.npy` for already-collected tiles without re-fetching OSM/SRTM (uses cache):

```bash
python -m data_pipeline.regen_cond --data data/ --workers 4
# or per-city
python -m data_pipeline.regen_cond --data data/ --city arlington_tx
```

### 4. Inference

```bash
python model/sample_diffusion.py \
    --diffusion checkpoints/diffusion/diffusion_epoch_200.pth \
    --n 8 --output samples/
```

### 5. Post-processing

```bash
python model/postprocess.py --diffusion checkpoints/diffusion/diffusion_epoch_200.pth --n 8
```

Applies dilation → skeletonization → small-component pruning per CaRoLS Section 3.3.

## Cluster (UIUC Campus Cluster)

```bash
# CPU data generation
sbatch slurm_datagen.sh
# Bulk re-encode existing tiles (parallel array job, one task per city)
sbatch slurm_regen_array.sh

# GPU training — full A100 (3-day partition)
CFG_PROB=0.5 EPOCHS=200 sbatch slurm_diffusion.sh
# GPU training — H100 MIG slice (8h partition, faster turnaround)
CFG_PROB=0.5 EPOCHS=150 \
  VAE_CKPT=checkpoints/vae/vae_epoch_050.pth \
  OUT_DIR=checkpoints/diffusion \
  EXTRA_ARGS='--class-weights 1.0,1.2,1.4,1.4,1.4' \
  sbatch slurm_diffusion_express.sh
# VAE retrain on the MIG slice
OUT_DIR=checkpoints/vae EPOCHS=50 sbatch slurm_vae_express.sh
```

## Sample outputs

See [`samples/`](samples/) for example comparison PNGs:
- `vae_recon_check.png` — VAE input vs reconstruction (near-perfect)
- `with_cheat_channel.png` — diffusion with existing-roads conditioning (looks good but is essentially inpainting)
- `terrain_only.png` — diffusion conditioned on terrain only (the fair eval — noisier)
- `postprocessed.png` — post-processing applied to terrain-only output

## What's not here

- The actual Cities Skylines 1 mod (planned)
- Trained checkpoints (large files; could be hosted on Hugging Face)
- The full ~23GB training dataset (small reference in repo, full set generated by `cdg.py`)
- Vector road graph export (post-processing currently outputs raster)

## Citation

If this is useful, cite the original CaRoLS paper:

```
Feng T, Li L, Li W, Li B, Shen J. CaRoLS: Condition-adaptive multi-level road layout synthesis.
Computers & Graphics 133 (2025) 104451.
https://doi.org/10.1016/j.cag.2025.104451
```

## License

MIT — see [LICENSE](LICENSE).
