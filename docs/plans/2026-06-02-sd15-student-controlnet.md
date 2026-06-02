# Workstream A: SD 1.5 Student ControlNet

*Produced 2026-06-02. Supersedes `docs/prompts/distillation.md` (the research brief
that preceded this plan). This is the authoritative design and implementation plan for
making the road-generation model deployable on consumer hardware.*

---

## Goal

Produce a ControlNet model that:

1. Generates structurally correct US suburban road networks (connected roads,
   recognisable class hierarchy, residential streets that connect to arterials)
2. Accepts a **4-channel conditioning image** — existing 3-ch terrain/landuse + new
   1-ch arterial mask — so generated subdivision streets connect to pre-existing arterials
3. Runs on consumer hardware: RTX 3060 6 GB or better at ≤15 s/tile (primary target);
   M3 MacBook Air/Pro via CoreML (stretch target via BK-SDM fallback)
4. Ships with the game binary at ≤750 MB adapter weight file

**Quality bar (B):** roads are connected, form recognisable grid/organic patterns,
respect the class hierarchy (arterials thicker/more prominent than residential). Exact
teacher-fidelity and fine texture detail are not required.

---

## Why SD 1.5 Instead of SDXL Distillation

SDXL distillation (SSD-1B, Vega, SDXL-Turbo/Lightning) produces *smaller diffusion
models* that still require GPU-class hardware: even a 4-step LCM run of the full SDXL
stack at fp16 takes ~3 min on M3 CPU. Step reduction is a GPU-latency optimisation,
not a CPU-viability fix.

SD 1.5 + ControlNet uses the identical paradigm (diffusion-based generation with
ControlNet conditioning) at ~3× smaller parameter count:

```
                   SDXL (teacher)    SD 1.5 (student)
UNet params        ~2.6 B            ~860 M
ControlNet params  ~1.25 B           ~360 M  (our trained adapter = 5 GB fp32 = 1.25B)
Runtime VRAM       ~12 GB            ~4–5 GB
GPU fit            RTX 3080 12 GB+   RTX 3060 6 GB ✓
Inference (GPU)    30–90 s           5–15 s ✓
Training cost      ~35 A100-hours    ~5–10 A100-hours
```

SDXL's improvements over SD 1.5 are primarily in text adherence and photorealistic
texture. Neither applies to our task: we use a constant prompt and output flat-colour
5-class maps. The quality gap for geometric structured road outputs is expected to be
small and will be validated empirically before committing to full deployment.

---

## Architecture

### Base model

`runwayml/stable-diffusion-v1-5` (frozen). Specifically the UNet
(`scheduler`, `unet`, `vae`, `text_encoder`) from this checkpoint. SD 1.5 operates
natively at **512×512**, which at our 5 m/px resolution = 2.56 km × 2.56 km per
tile — exactly the tile size of the existing training data.

### ControlNet adapter

A ControlNet head initialised from the SD 1.5 UNet encoder. ~360 M trainable
parameters. The conditioning embedding is extended to accept **4 input channels**
instead of the default 3 (see §Input format below).

Diffusers config key to change from default: `conditioning_channels: 4`.

### Resolution

512×512 throughout — no upscaling needed. Our existing tiles are 512×512 at
5 m/px (the 1024×1024 SDXL tiles are upscaled versions; we can use the original
512×512 `.npy` files directly from `data/`).

---

## Input / Output Format

### Conditioning input — 4-channel RGB+A

| Channel | Encoding | Source |
|---------|----------|--------|
| R | Elevation, per-tile normalised `[0,1]` | SRTM; existing pipeline |
| G | `0.3·water + 0.5·parkland + 0.2·agricultural` | OSM landuse; existing pipeline |
| B | `0.6·residential + 0.8·commercial + 1.0·industrial` | OSM landuse; existing pipeline |
| **A (new)** | **Arterial mask: binary {0,1}, 1 where primary or motorway road pixels exist** | Extracted from ground-truth road raster |

The arterial channel is the key addition over the teacher. At training time it is
extracted from the ground-truth road map. At inference time (in-game) it comes from
the player's existing road network rasterised into the tile's coordinate frame.

Saving: PNG with alpha channel (RGBA) or two separate files
(`cond_NNNN.png` + `arterial_NNNN.png`). Either is fine; RGBA is simpler.

### Output — 3-channel RGB road raster

Identical to the SDXL teacher:

| Class | RGB |
|-------|-----|
| Background | `(0, 0, 0)` |
| Residential / living street | `(255, 0, 0)` |
| Tertiary | `(0, 255, 0)` |
| Primary / secondary | `(0, 0, 255)` |
| Motorway / trunk | `(255, 255, 0)` |

---

## Training Data Preparation

### Step 1 — Extract arterial channel from existing `.npy` road files

New script: `tools/extract_arterial_channel.py`

```
input:  data/flux_cnet/road_<city>_NNNN.npy   [5, 512, 512] float32 one-hot
output: data/sd15_cnet/arterial_<city>_NNNN.png  [512, 512] uint8 binary mask

logic: arterial_mask = (road[3] + road[4]) > 0.5  # primary + motorway channels
```

### Step 2 — Prepare 4-ch cond + road target pairs for diffusers training

New script: `tools/prep_sd15_dataset.py` (extend or replace `prep_flux_dataset.py`)

Changes from the SDXL prep script:
- Source resolution: **512×512** (already native; skip the 1024 upscale)
- Cond image: RGBA PNG (R/G/B from existing 3-ch encoding + A from arterial mask)
- Target image: same 3-ch RGB road palette as SDXL
- HF `imagefolder` layout identical to `data/flux_cnet_hf/`; output to
  `data/sd15_cnet_hf/`
- Prompt: identical constant string (text encoder has minimal influence; keep
  consistent for CFG dropout compatibility)

The existing diffusers `train_controlnet.py` (SD 1.5 version, not SDXL) handles
4-channel conditioning via the `conditioning_channels` parameter — no patching needed
beyond the two patches already applied to the SDXL script (dtype cast, HF Image cast).

### Tile count

Existing dataset: 2,443 tiles at 512×512. This is sufficient to start. The SDXL
teacher ran on the same tile count and converged at 25k steps. SD 1.5 is 3× smaller;
expect convergence in 10–20k steps.

**Optional augmentation:** generate additional synthetic pairs from the SDXL teacher
(`tools/sdxl_cnet_sample.py` over the full conditioning set with varied seeds) to
increase diversity. Run only if the base training produces visually insufficient
connectivity. Estimate: 2,443 tiles × 2 seeds = ~4,886 additional pairs ×
30 s/tile on A100 = ~41 additional A100-hours. Defer unless needed.

---

## Training Recipe

### Script

New file: `slurm_sd15_cnet.sh`. Adapt from `slurm_sdxl_cnet.sh` with these changes:

```bash
# Base model — SD 1.5, not SDXL
--pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5

# SD 1.5 ControlNet trainer from diffusers (not the SDXL variant)
train_controlnet.py   # not train_controlnet_sdxl.py

# No SDXL-specific args (no --pretrained_vae_model_name_or_path,
# no --addition_embed_type, no projection_class_embeddings)

# 4-channel conditioning
--conditioning_image_column=conditioning_image   # RGBA PNG with A = arterial
# The ControlNet will be initialised with conditioning_channels=4 automatically
# when the first conditioning image has 4 channels — verify this in diffusers 0.38

# Resolution
--resolution=512   # SD 1.5 native; drop from 1024

# Steps — start at 15,000, evaluate, extend to 25,000 if needed
--max_train_steps=15000
--checkpointing_steps=1000
--validation_steps=500

# Warm start from SD 1.5 ControlNet (not from our SDXL adapter)
# Leave --controlnet_model_name_or_path unset to init from base UNet
```

All other hyperparameters (lr=1e-5, constant_with_warmup, bs=1×ga=8, fp16,
8-bit Adam, gradient checkpointing) carry over unchanged from the SDXL run.

**Note on `conditioning_channels=4`:** diffusers `ControlNetModel.from_unet(unet)`
creates a ControlNet with `conditioning_channels=3` by default. To get 4, either:
- Pass `conditioning_channels=4` in the ControlNetModel config before training, or
- Post-init: replace `controlnet.controlnet_cond_embedding.conv_in` with a new
  `Conv2d(4, ...)` layer (random init for the new channel, copy weights for R/G/B).

The second approach (selective weight transfer) is preferred because it lets the
model warm-start on the 3-ch channels it already understands from SD 1.5 pretraining,
while learning the arterial channel from scratch.

### Compute estimate

~5–10 A100-hours. At 4.9 s/step (SDXL rate) scaled to SD 1.5 (~1.5–2 s/step for the
smaller model), 15k steps ≈ 6–8 hours. Fits in one 18-hour SLURM job.

### Output location

`/scratch/jalenj4/runs/sd15_cnet_v1/`

---

## Evaluation Gate

Run before any further investment in this workstream.

### Visual quality check (primary gate)

Sample 20 tiles from the Irving TX validation set (same held-out city as the SDXL
evaluation). Run both the SDXL teacher and the SD 1.5 student on identical conditioning.
Produce a side-by-side comparison PNG.

Pass criteria (quality bar B):
- [ ] Roads are visually connected (no obvious breaks in residential streets)
- [ ] Arterials (blue/yellow) are present and prominent where the arterial channel indicates
- [ ] Residential streets (red) branch from arterials coherently
- [ ] Block sizes and connectivity patterns recognisably resemble US suburbs

Fail criteria (do not continue without fixing):
- [ ] Mostly background with scattered road pixels (mode collapse)
- [ ] Road classes mixed up or absent
- [ ] No relationship between arterial channel input and generated arterials

### Connectivity metric (secondary gate)

Reuse `model/eval_metrics.py::compute_connectivity_index()` from the existing
pipeline. Target: CI > 1.5 (vs CaRoLS baseline 1.948). Lower bar is acceptable
because we are generating smaller tiles with fewer road classes.

### Escalation path

| Outcome | Next step |
|---------|-----------|
| SD 1.5 passes bar B visually + CI > 1.5 | Proceed to deployment |
| SD 1.5 fails: poor connectivity only | Add Skeleton Recall Loss (see §Loss addendum) |
| SD 1.5 fails: structural quality clearly worse than SDXL | Escalate to BK-SDM or accept SDXL for cloud inference only |
| SD 1.5 passes quality but fails RTX 3060 latency target | Apply LCM step distillation (4-step) to the SD 1.5 model |

---

## Loss Addendum — Skeleton Recall Loss

If connectivity fails the evaluation gate, add **Skeleton Recall Loss** (ECCV 2024,
arXiv:2404.03010) on top of the standard diffusion loss. This adds ~+8% training time
with ~+2% VRAM overhead (vs +88% for clDice).

Implementation: the loss operates on the final denoised prediction `x_0_pred` vs
ground truth `x_0`. It skeletonises both on CPU in a background worker and penalises
missing skeleton pixels. Reference implementation is in the paper's supplementary
code; wrap as `model/skeleton_recall_loss.py`.

Standard diffusion MSE loss remains the primary loss. Skeleton Recall Loss is added
as a weighted auxiliary term: `L_total = L_diffusion + λ·L_skeleton`, λ=0.1 to start.

Do **not** add this upfront — validate the baseline first, add only if connectivity
fails.

---

## BK-SDM Fallback

If SD 1.5 quality is insufficient, or if the deployment target requires M3 MacBook
Air (CPU/Metal, no discrete GPU), escalate to BK-SDM-Tiny.

**What it is:** SD 1.5 with ~30% of UNet blocks removed via architectural pruning +
knowledge distillation. ~400M UNet params. CoreML export available at
`nota-ai/coreml-bk-sdm` on HuggingFace. Reported inference: ~4 s on Apple Silicon
via CoreML.

**How to train the ControlNet:** identical pipeline to the SD 1.5 run. Replace
`runwayml/stable-diffusion-v1-5` with `nota-ai/bk-sdm-v2-tiny` as the frozen base.
The UNet architecture is compatible with diffusers ControlNet training.

**Export to CoreML:** use `coremltools` to convert the diffusion pipeline
(UNet + ControlNet combined forward pass). The `nota-ai` team has already done this
for BK-SDM; the ControlNet forward-pass modification may require a custom export path.
Treat as a separate implementation task if the SD 1.5 path succeeds.

---

## Deployment Format

**Primary:** `diffusion_pytorch_model.safetensors` (the trained ControlNet adapter
only, ~720 MB at fp16). The SD 1.5 base model is pulled from HuggingFace on first
run and cached locally (~2 GB). This is the standard diffusers deployment pattern.

**Bundled option:** convert full pipeline (UNet + ControlNet + VAE + CLIP) to a
single safetensors file. ~4 GB total. Ships with the game installer. No first-run
download. Higher distribution cost but better UX.

**ONNX:** diffusion models are substantially harder to ONNX-export than single-pass
regression models (the iterative loop requires careful handling). Leave for a separate
task if needed for Rust/ONNX-Runtime integration. The primary game integration path
for v1 is a Python sidecar process, not an embedded ONNX model.

**CoreML (BK-SDM path only):** CoreML export produces a `.mlpackage` directory that
runs natively via the Metal GPU on Apple Silicon. The package can be bundled in the
game's app directory.

---

## Implementation Tasks

In order:

| # | Task | Script / file | Notes |
|---|------|---------------|-------|
| 1 | Arterial channel extractor | `tools/extract_arterial_channel.py` | Extract primary+motorway from existing `data/flux_cnet/road_*.npy` → binary mask PNGs |
| 2 | SD 1.5 dataset prep | `tools/prep_sd15_dataset.py` | 4-ch RGBA cond + 3-ch road target, 512×512, HF imagefolder layout at `data/sd15_cnet_hf/` |
| 3 | 4-ch ControlNet init | Init block in `slurm_sd15_cnet.sh` or a `tools/init_sd15_controlnet.py` | Selective weight transfer: copy 3-ch weights, randomly init arterial conv channel |
| 4 | SLURM training script | `slurm_sd15_cnet.sh` | Adapt from `slurm_sdxl_cnet.sh`; use `train_controlnet.py` not SDXL variant |
| 5 | Inference sampler | `tools/sd15_cnet_sample.py` | Adapt from `tools/sdxl_cnet_sample.py`; 4-ch cond input |
| 6 | Evaluation comparison | `tools/eval_compare.py` | Side-by-side PNG of SDXL teacher vs SD 1.5 student on Irving TX val tiles |
| 7 | HF Hub upload | `tools/upload_to_hf.py` (existing) | Target: `jalens-shadow-corp/groundwork-sd15-cnet-us-suburbs` |
| 8 | (conditional) Skeleton Recall Loss | `model/skeleton_recall_loss.py` | Only if evaluation gate fails on connectivity |
| 9 | (conditional) BK-SDM escalation | `slurm_bksdm_cnet.sh` | Only if SD 1.5 quality is insufficient |

---

## Open Questions

| Question | How to resolve |
|----------|---------------|
| Does SD 1.5 produce comparable road structure to SDXL for flat-colour geometric maps? | Task 6 evaluation comparison. Expected answer: yes, gap is small. |
| Does the arterial channel successfully guide residential street connectivity? | Visual inspection of evaluation tiles: do generated residential streets terminate at arterials indicated by the 4th channel? |
| Does conditioning on 4 channels vs 3 require more training steps to converge? | Watch val loss curve; extend to 25k steps if still improving at 15k. |
| What `λ` for Skeleton Recall Loss? | Sweep λ ∈ {0.05, 0.1, 0.2} if activated; pick by CI metric. |
| Is SD 1.5 fast enough on RTX 3060 at 20 steps? | Benchmark after training. Target: ≤15 s at 20 DDIM steps. If slow: apply 4-step LCM. |

---

## Relationship to Other Workstreams

This spec covers **Workstream A (student model)** only.

- **Workstream B (raster → game vectors):** post-processing the student's output into
  a road graph compatible with the citybuilder's petgraph representation. Independent
  of this workstream; can proceed in parallel once the output format (3-ch RGB palette)
  is confirmed stable.

- **Workstream C (game integration):** the citybuilder reads city state, rasterises it
  to the 4-ch conditioning format defined here, calls the student model (via Python
  sidecar or ONNX), and displays the output image. Depends on A and B.

- **Multi-style expansion (`docs/multi_style_plan.md`):** the per-style ControlNets
  planned there will use the SDXL teacher. This student spec is for the deployable
  `us_suburb` style only. If the student approach succeeds, each style ControlNet
  can be distilled using the same pipeline.
