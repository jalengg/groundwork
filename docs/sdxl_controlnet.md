# Groundwork SDXL ControlNet — US Suburban Road Networks

The final, working approach to generating conditional road networks for the
Cities Skylines mod. This document captures the full method, training recipe,
inference setup, and lessons learned. **If you're picking this up cold, read
this end-to-end before touching anything.**

## TL;DR

We fine-tune a **ControlNet adapter (~1.4 B trainable params) on top of
frozen Stable Diffusion XL (3.5 B)** to map a 3-channel landuse+terrain
raster into a 3-channel color-coded road network image. The ControlNet
adapter is what we own; the foundation model is whatever the user pulls
from HuggingFace at inference time.

| | Value |
|---|---|
| Foundation | `stabilityai/stable-diffusion-xl-base-1.0` (3.5 B params, frozen) |
| Foundation VAE | `madebyollin/sdxl-vae-fp16-fix` (fp16-stable variant of SDXL VAE) |
| Trainable adapter | ControlNet head, ~1.4 B params, copy of SDXL UNet encoder |
| Dataset | 2,443 paired tiles, 1024×1024 RGB, 17 US Sun Belt cities |
| Training | 25,000 steps @ `bs=1 × grad_accum=8`, fp16, ~35 h on A100 80 GB |
| Inference | ~30 s/tile on A100, ~60-90 s on RTX 3060 12 GB |
| Final artifacts | `/scratch/jalenj4/runs/sdxl_cnet_v1/{config.json, diffusion_pytorch_model.safetensors}` (~3 GB) |

## Why this approach (after 3 months of trying others)

We spent ~3 months reproducing **CaRoLS (Feng et al. 2025)**: a small custom
VAE (~5-12 M params) + custom diffusion U-Net (~19 M params) trained from
scratch on road maps. We hit a hard ceiling at **~0.67 road-pixel IoU** on
encode-decode roundtrip and **~46 % res-vs-com pixel disagreement** on
generated samples, with outputs that looked like honeycomb noise rather than
real road networks.

A 4-agent council review (`docs/findings.md` § "Council critique") and
subsequent diagnostics established that the bottleneck was **not** the VAE
or the conditioning architecture in isolation — it was that a 19 M-param
denoiser trained on 2,500 tiles cannot acquire enough spatial-structure
prior to generate coherent linear road features. The MSE noise-prediction
objective is mean-seeking; without enormous data or adversarial pressure,
the model converges to "blur of all possibilities" = honeycomb mesh.

A second council (4 ML-focused agents) recommended the **stand on
foundation models' shoulders** path. We tested it: at training step 1,000
the SDXL ControlNet was already producing coherent continuous road lines —
a quality our small model never reached after 200 epochs. The foundation
prior carries inductive biases for parallel lines, grids, perpendicular
intersections, edge sharpness that took our small model to ~zero progress
to learn from scratch.

## Data pipeline

### Input: paired 7-channel cond + 5-channel road rasters

Our existing pipeline (`data_pipeline/`) produces, per tile (`512×512` @
5 m/px):

- `cond_<id>.npy`: 7-channel `[elevation, water, residential, commercial,
  industrial, parkland, agricultural]` float32
- `road_<id>.npy`: 5-channel one-hot `[bg, residential_street, tertiary,
  primary, motorway]` float32

17 cities, ~2,550 tiles total, `irving_tx` held out (~150 tiles) for
qualitative validation. Production cities: `arlington_tx, beaverton_or,
bellevue_wa, carlsbad_ca, chandler_az, cranberry_township_pa, gilbert_az,
henderson_nv, kissimmee_fl, mesa_az, plano_tx, plymouth_mn, sandy_ut,
sugar_land_tx, tempe_az, virginia_beach_va`.

### Conversion to SDXL-friendly format

`data_pipeline/prep_flux_dataset.py` converts to 3-channel RGB at 1024×1024:

**Cond → 3-channel RGB** (fixed semantic mapping):
- `R = elevation` (per-tile normalized to `[0, 1]`)
- `G = 0.3·water + 0.5·parkland + 0.2·agricultural`
- `B = 0.6·residential + 0.8·commercial + 1.0·industrial`

**Road → 3-channel RGB** (max-separated 5-color palette):
- `bg = (0, 0, 0)`, `residential = (255, 0, 0)`, `tertiary = (0, 255, 0)`,
  `primary = (0, 0, 255)`, `motorway = (255, 255, 0)`

Both upscaled to 1024×1024 (Flux/SDXL native resolution). Bilinear for
elevation, nearest-neighbor for one-hot/categorical channels.

### HuggingFace `imagefolder` layout

The diffusers training script requires a specific layout. `tools/
build_hf_manifest.py` creates it via symlinks (no data duplication):

```
data/flux_cnet_hf/
├── cond/ -> /scratch/jalenj4/groundwork/data/flux_cnet/cond  (symlink)
├── target/ -> /scratch/jalenj4/groundwork/data/flux_cnet/target  (symlink)
└── metadata.jsonl
```

`metadata.jsonl` (one line per sample):
```json
{"file_name": "target/arlington_tx_0000.png",
 "text": "top-down satellite-style raster of a US suburban road network, ...",
 "conditioning_image": "/scratch/jalenj4/groundwork/data/flux_cnet/cond/arlington_tx_0000.png"}
```

**Important**:
- `file_name` must be **relative** (HF imagefolder builder convention)
- `conditioning_image` must be **absolute** — PIL resolves against cwd, not data_dir
- val tiles must live **outside** `flux_cnet_hf/` (a `val/` subdir would be auto-
  detected as a "validation" split and break `load_dataset(split="train")`).
  Ours live at `data/flux_cnet_val/`.

### Training script patches

`third_party/train_controlnet_sdxl.py` is a vendored copy of diffusers
v0.38.0's official ControlNet trainer with **two** hand-applied patches:

**Patch 1** (line 679): cast `conditioning_image` column to HF `Image()`
feature. HF imagefolder auto-decodes only `file_name`; other path columns
stay as strings, breaking `.convert("RGB")` calls.

```python
from datasets import Image as _HFImage
if args.conditioning_image_column in dataset["train"].column_names:
    dataset = dataset.cast_column(args.conditioning_image_column, _HFImage())
```

**Patch 2** (line 1260): cast cached prompt embeddings + UNet added
conditions to `weight_dtype` before passing into ControlNet/UNet. The
script caches text-encoder outputs via `dataset.map`, which serializes
tensors as fp32 in Arrow format. On reload they're fp32 even though the
UNet is fp16, causing `dtype mismatch: float != Half` in cross-attention.

```python
batch["prompt_ids"] = batch["prompt_ids"].to(dtype=weight_dtype)
batch["unet_added_conditions"] = {k: v.to(dtype=weight_dtype)
                                  for k, v in batch["unet_added_conditions"].items()}
```

Without these patches, training fails at step 1.

## Training recipe

Full SLURM script at `slurm_sdxl_cnet.sh`. Key settings:

| Hyperparameter | Value | Why |
|---|---|---|
| `--mixed_precision` | `fp16` | SDXL native; `bf16` hit the same dtype mismatch in cross-attention. fp16 is the well-tested community path. |
| `--train_batch_size` | 1 | bs=2 fits on A100 80GB but slows iteration |
| `--gradient_accumulation_steps` | 8 | effective batch = 8 |
| `--gradient_checkpointing` | on | trades compute for VRAM |
| `--use_8bit_adam` | on | bitsandbytes 8-bit Adam saves ~700 MB |
| `--learning_rate` | `1e-5` | diffusers SDXL ControlNet README default |
| `--lr_scheduler` | `constant_with_warmup` | 500 warmup steps |
| `--max_train_steps` | 25,000 | per-step rate × this = ~35 h on A100 |
| `--checkpointing_steps` | 1,000 | every ~80 min |
| `--checkpoints_total_limit` | 3 | rolling delete; final weights also saved at top level |
| `--validation_steps` | 500 | logged to tensorboard only |
| `--proportion_empty_prompts` | 0.1 | 10% caption dropout for CFG |
| `--dataloader_num_workers` | 4 | |
| Realized step time | ~4.9 s/step | bs=1 × ga=8 = 8 micro-batches per opt step |

### Critical infrastructure

Cluster home (`/u/jalenj4`) had only ~3 GB headroom. **All caches and
checkpoints must live on `/scratch`**:

```bash
export HF_HOME=/scratch/jalenj4/hf                    # SDXL weights (~14 GB)
export HF_DATASETS_CACHE=/scratch/jalenj4/hf_datasets # dataset cache
export TMPDIR=/scratch/jalenj4/tmp
OUT_DIR=/scratch/jalenj4/runs/sdxl_cnet_v1            # checkpoints (~15 GB)
```

Hitting the home quota mid-training will silently corrupt checkpoints.

### Walltime + auto-resume

A100 partition has 18 h walltime. At ~4.9 s/step, one job clears ~13,000
steps. The script auto-resumes if it finds `checkpoint-*` in `OUT_DIR`:

```bash
RESUME=""
if compgen -G "$OUT_DIR/checkpoint-*" > /dev/null; then
    RESUME="--resume_from_checkpoint=latest"
fi
```

We needed 2 sequential `sbatch` invocations to complete 25,000 steps.
First job hit walltime at step 13,292; second job (auto-resumed from
checkpoint-13000) completed the rest.

## Training trajectory (visual evolution)

Captured by sampling intermediate checkpoints. The model migrates from
SDXL's natural-image prior toward our 5-class palette over training:

| Step | Visual style |
|---|---|
| 1,000 | Photorealistic satellite imagery (cyan road lines, photographic textures) |
| 4,000 | More saturated colors, transitioning to map-like |
| 13,000 | Stylized map style with yellow-dominant road grid, some texture noise |
| 18,000 | Cleaner, visible block structure in suburban tiles, multi-class road colors |
| **25,000** | **Multi-class map style with clean continuous road lines, distinct road hierarchy by color, contextual features (water, vegetation)** |

Samples in `samples/sdxl_cnet_e{1000,4000,13000,18000,25000}_multi.png`.

## Inference

`tools/sdxl_cnet_sample.py` is the reference sampler. Loads the trained
ControlNet head, paired with the SDXL base + fp16-fix VAE, generates one
output per cond image in a directory.

```bash
python tools/sdxl_cnet_sample.py \
    --controlnet /scratch/jalenj4/runs/sdxl_cnet_v1 \
    --val-dir data/showcase_cond \
    --out samples/showcase.png
```

Key inference parameters:
- `--steps 30` (DDIM, default in pipeline)
- `--guidance 5.0` (classifier-free guidance scale)
- `--cnet-scale 1.0` (ControlNet conditioning scale)

The prompt is held constant — all class differentiation comes through
the ControlNet input, not text:
```
"top-down satellite-style raster of a US suburban road network,
high-contrast color-coded road class map, flat color, vector style,
no texture, no shading"
```

## Inference latency by GPU

| GPU | VRAM needed | Time per tile @ 30 steps |
|---|---|---|
| A100 80 GB | ~12 GB | ~30 s |
| RTX 4090 | ~12 GB | ~45 s |
| RTX 3080 12 GB | ~12 GB | ~60 s |
| RTX 3060 12 GB | ~12 GB | ~90 s |
| RTX 3060 6 GB | OOM | n/a |

For a Cities Skylines mod with the median CS player on a GPU below
12 GB VRAM, **local inference is not viable for half the audience**.
See `docs/deployment_strategy.md` (TBD) for the cloud-inference +
Patreon-tier-priority queue approach.

## Output → road graph (post-processing)

The generated RGB output needs to be converted back to a road graph
that the CS mod's `NetManager` can consume. Existing pipeline
(`model/postprocess.py`) implements the CaRoLS Sec 3.3 vectorization:

1. RGB → nearest-color → 5-class semantic mask
2. Morphological buffering (dilation r=2)
3. Medial-axis skeletonization → networkx graph (via sknw)
4. Drop components <20 px path length
5. Bridge disjoint components to main graph (KDTree nearest-node)
6. Merge close nodes, Douglas-Peucker simplify
7. Render back to raster or export as graph

Works on the SDXL output without modification (it's still a 3-channel
color-coded map after argmax to nearest palette color).

## Files of record

```
groundwork/
├── data/
│   ├── flux_cnet/        # 7-ch cond + 5-ch road, original npy → encoded PNG pairs
│   ├── flux_cnet_hf/     # HF imagefolder layout (symlinks + metadata.jsonl)
│   ├── flux_cnet_val/    # held-out cond tiles outside dataset dir
│   └── showcase_cond/    # one tile per city for showcase samples
├── data_pipeline/
│   └── prep_flux_dataset.py    # 7-ch → 3-ch + 1024² upscale + manifest
├── third_party/
│   └── train_controlnet_sdxl.py # diffusers v0.38.0 + 2 patches
├── tools/
│   ├── build_hf_manifest.py    # HF imagefolder layout builder
│   └── sdxl_cnet_sample.py     # inference / sampler
├── slurm_sdxl_cnet.sh          # A100 trainer, auto-resume
└── samples/
    └── sdxl_cnet_*.png          # training trajectory + showcase samples
```

Cluster artifacts (will be cleaned periodically; **upload to HF Hub before
they're gone**):
```
/scratch/jalenj4/runs/sdxl_cnet_v1/
├── config.json
├── diffusion_pytorch_model.safetensors    # the final ControlNet (~3 GB)
└── checkpoint-{23000,24000,25000}/         # rolling, will be lost
```

## What we learned along the way

The path from CaRoLS reproduction to this took ~4 months. Critical
lessons captured in commit history and `docs/findings.md`:

1. **Custom small models can't compete with foundation models** for tasks
   needing structured spatial output. We trained 7+ variants of small
   custom VAE+UNet (`vae`, `vae_v2`, `vae_v3_16ch`, `vae_fsq`, `vae_gan`,
   etc.). Every one hit the same ~0.67 IoU ceiling.

2. **Pixel-IoU is a misleading metric.** Visual quality and IoU
   decorrelate. The VAE-GAN run had slightly lower IoU than v3 continuous
   but visibly sharper outputs.

3. **Validation loss in latent space is the most misleading metric of all.**
   `diff_vae_v2` had our lowest val_loss (0.146) but produced 99% bg
   outputs for "commercial" cond — pure mode collapse.

4. **Silent bugs we caught only via 4-agent technical council**:
   - Bilinear interpolation on one-hot landuse channels → label smearing
   - BatchNorm in cond encoder + CFG dropout → eval-time normalization
     divides by zero variance
   - SDXL VAE per-channel scaling factor calibrated on photos doesn't
     match our palette input distribution

5. **The "label noise ceiling" theory was wrong.** Our 0.67 IoU cap
   wasn't intrinsic to OSM rasterization at 5m/px — it was about how
   *any* continuous-Gaussian latent VAE we trained couldn't preserve
   discrete-class structure under decode. FSQ (discrete latent) also hit
   ~0.67. The actual bottleneck was the encoder/decoder architecture
   relative to the model size we trained.

6. **The text-prompt is irrelevant in this setup.** ControlNet with a
   constant per-tile caption doesn't use the text encoder for class
   differentiation — all of it comes from the ControlNet input. The
   ~constant caption may even hurt at inference (no CFG signal).

7. **Foundation models cost compute differently.** We used ~50 GPU-hours
   total across all custom-model experiments over 4 months; SDXL
   ControlNet's *one run* used 35 GPU-hours. The total compute is
   similar — we just spent 4 months learning what 1 training run on
   the right substrate would deliver.

## Next steps

See `docs/cs_mod_integration.md` (TBD) for the Cities Skylines mod
shipping plan. High-level:

1. Upload ControlNet weights to HuggingFace Hub
   (`jalengg/groundwork-sdxl-controlnet-us-suburbs` or similar)
2. Stand up a cloud inference endpoint (Replicate / HF Spaces / custom
   FastAPI on RunPod)
3. Build the CS mod layer that: rasterizes player-painted landuse →
   uploads → polls queue → receives generated road graph → invokes
   `NetManager.CreateSegment`
4. Implement Patreon-tier priority queue
5. Wire up "Chirpy" notification on completion
