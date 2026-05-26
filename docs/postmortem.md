# Postmortem: Groundwork SDXL ControlNet (4-month journey)

Written 2026-05-25 after the model finally worked. Useful if you ever
need to do this again, or if you're picking up the project cold.

## 1. SDXL training gotchas

The diffusers `train_controlnet_sdxl.py` script is canonical for SDXL
ControlNet training and is well-maintained, **but it has several
non-obvious failure modes that bite within the first 30 seconds of a
training run.** Each of these cost us a SLURM job to discover.

### Disk-quota silent failure

| | |
|---|---|
| Symptom | Training crashes mid-checkpoint with `[Errno 122] Disk quota exceeded`; checkpoint dir half-written; auto-resume then fails on the partial. |
| Cause | `/u/jalenj4` had only ~3 GB headroom. SDXL base ≈ 13 GB. Each checkpoint ≈ 5 GB. HF dataset cache can grow to 50+ GB. |
| Fix | Put **everything** on `/scratch`: `HF_HOME`, `HF_DATASETS_CACHE`, `TMPDIR`, `OUT_DIR`. Verified in `slurm_sdxl_cnet.sh`. |
| Cost | One discovery cycle (~5 min job + ~2h debug). |

### `imagefolder` does not auto-decode `conditioning_image`

| | |
|---|---|
| Symptom | `AttributeError: 'str' object has no attribute 'convert'` at first training step. |
| Cause | HF's `imagefolder` builder casts `file_name` to PIL `Image` automatically, but **only** that column. Other path-string columns (`conditioning_image`) stay as strings, breaking `.convert("RGB")` in `preprocess_train`. |
| Fix | Patch the script to call `dataset.cast_column(args.conditioning_image_column, Image())` right after `load_dataset`. (See patch 1 in `third_party/train_controlnet_sdxl.py`.) |
| Cost | One job + ~30 min reading the script. |

### Relative paths in `conditioning_image` break

| | |
|---|---|
| Symptom | `FileNotFoundError: '/scratch/.../groundwork/data/flux_cnet_hf/target/arlington_tx_0000.png'` (a wrong absolute path) |
| Cause | `PIL.Image.open(path)` resolves the path against the **current working directory**, not the dataset dir. Symlinks created with relative targets resolve from `flux_cnet_hf/cond` which doesn't exist. |
| Fix | Use **absolute paths** in `metadata.jsonl` for `conditioning_image`. `file_name` stays relative (imagefolder convention). Updated `tools/build_hf_manifest.py` accordingly. |
| Cost | One job + ~30 min puzzling at why "OK" smoke tests didn't catch it. (The smoke test ran from the dataset's parent dir; training ran from `~/groundwork`.) |

### `val/` subdir inside dataset = auto-detected "validation" split

| | |
|---|---|
| Symptom | `ValueError: Unknown split "train". Should be one of ['validation']` |
| Cause | HF's imagefolder builder auto-detects subdirs named `train`, `val`, `validation`, `test`, `eval` as splits. Our cherry-picked val tiles in `data/flux_cnet_hf/val/` triggered this. |
| Fix | Put validation cond tiles **outside** the dataset dir. Ours live at `data/flux_cnet_val/`. |
| Cost | One job. |

### `load_dataset(train_data_dir)` ≠ `load_dataset("imagefolder", data_dir=...)`

| | |
|---|---|
| Symptom | `--image_column 'image' not found in dataset columns. Dataset columns are: file_name, text, conditioning_image` |
| Cause | The training script's `get_train_dataset()` has two branches: with `--dataset_name`, it calls `load_dataset(args.dataset_name, data_dir=args.train_data_dir)` (uses imagefolder, auto-decodes `file_name` → `image`). Without `--dataset_name`, it calls `load_dataset(args.train_data_dir)` which treats the dir as a custom dataset script and **doesn't** invoke imagefolder. |
| Fix | Pass `--dataset_name=imagefolder --train_data_dir=...` together. Documented in `slurm_sdxl_cnet.sh`. |
| Cost | One job. |

### Cached prompt embeddings deserialize as fp32, UNet is fp16

| | |
|---|---|
| Symptom | `RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::Half` in cross-attention `to_k(encoder_hidden_states)`. Both bf16 and fp16 hit this. |
| Cause | The script encodes prompts ahead of time and caches via `dataset.map(compute_embeddings_fn)`. `dataset.map` serializes tensors as fp32 in Arrow format. On reload via `collate_fn`'s `torch.tensor(example["prompt_embeds"])`, they're fp32. UNet/ControlNet expect `weight_dtype`. |
| Fix | Patch the training loop to cast `batch["prompt_ids"]` and `batch["unet_added_conditions"]` to `weight_dtype` before the ControlNet/UNet calls. (See patch 2 in `third_party/train_controlnet_sdxl.py`.) |
| Cost | Two jobs (one bf16, one fp16) + ~1h reading the script before realizing the cache was the culprit. |

### bf16 vs fp16 for SDXL

The diffusers README recommends `bf16` for A100. Empirically, bf16 hit
the same dtype mismatch as fp16 (above), but **fp16 is SDXL's native
training dtype** and is what the community uses for ControlNet training.
After applying the dtype-cast patch, fp16 worked stably for 35 hours
with no loss-scaler issues. **Use fp16.**

### `huggingface-cli` is deprecated; use `hf`

| | |
|---|---|
| Symptom | `Warning: huggingface-cli is deprecated and no longer works. Use hf instead.` |
| Cause | The HF Hub team renamed the CLI in late 2025. |
| Fix | `hf upload <repo-id> <local-path> <path-in-repo>` with `--type model --exclude "<pat>"` repeated per pattern. The `--repo-type` flag is now `--type`. |
| Cost | Two trivial misadventures. |

### `huggingface_hub.upload_folder()` from Python API can disconnect

| | |
|---|---|
| Symptom | `httpx.RemoteProtocolError: Server disconnected without sending a response` partway through a 3-5 GB upload. |
| Cause | Either a network glitch on the SLURM compute node, or the SLURM `secondary` partition has outbound HTTPS limitations. The Python API has weaker retry behavior than the CLI. |
| Fix | Use the `hf upload` CLI tool **from the login node**. CLI handles retries automatically. Login nodes have more reliable outbound connectivity. (Don't do this for huge uploads — admins frown — but 5 GB is fine.) |
| Cost | One failed upload + a re-run that took 13 seconds at ~380 MB/s. |

### Step time at 1024² is real; auto-resume is essential

At our hyperparameters (`bs=1 × grad_accum=8 × gradient_checkpointing × 8-bit
Adam`), step time was ~4.9 s on A100 80 GB. 25,000 steps × 4.9 s ≈ 34 h.
The A100 partition has an 18 h walltime cap. **You must build auto-resume
into your SLURM script** or you'll lose half your progress. We did:

```bash
RESUME=""
if compgen -G "$OUT_DIR/checkpoint-*" > /dev/null; then
    RESUME="--resume_from_checkpoint=latest"
fi
```

One full run = 2 sequential `sbatch` submissions.

---

## 2. What we tried and why it failed

### CaRoLS reproduction path (3+ months, ~30 model variants)

| Approach | Result | Why it failed |
|---|---|---|
| Custom VAE v1 (5M, focal+KL, base_ch=64) | 0.678 oracle IoU | Too small to compress road-class structure |
| Custom VAE v2 (12.4M, deeper, Dice loss) | 0.666 oracle IoU | More capacity didn't break the ceiling |
| Custom VAE v3 (16-channel latent, same arch) | 0.683 oracle IoU | Latent shape isn't the bottleneck either |
| SDXL VAE swap (frozen pretrained) | 0.640 oracle IoU | Photo-prior VAE has worse noise robustness; latent-stat mismatch with DDPM schedule |
| FSQ VAE (12.4M, discrete 1000-code latent) | 0.644 oracle IoU | Discrete latent same ceiling — the data-vs-arch bottleneck is real |
| VAE-GAN with PatchGAN discriminator | Visibly sharper but no IoU gain | Right *direction* (adversarial pressure) but still bottlenecked at the diffusion stage |

| Diffusion stage variant (all on 19M-param U-Net) | Res-vs-com pixel disagreement (target >40%) |
|---|---|
| `diff_safe_fix` (GCI K/V fix + LDE i=3) | 33.4% |
| `diff_planA` (zero-init `R_l + R_up`) | 21.2% (regression — Plan A double-injects R_l) |
| `diff_load` (DRoLaS SFT/FiLM) | 25.3% |
| `diff_eq9_planAfix_v2` (LDE + Eq. 9 fix + Plan A fix) | 33% raw → 41% postprocessed (squeaked past target) |
| `diff_phase1_fixes` (GN cond stem + NN-interp + EMA + everything) | 45.7% postprocessed |
| `diff_vae_v2` (phase1 + v2 VAE) | 44.8% but **degenerate commercial collapse** (99% bg) |
| `diff_vae_gan` (phase1 + VAE-GAN) | **66.6%** postprocessed — but visually colored noise, not road structure |

### Root cause of every CaRoLS-path failure

A **19 M-parameter denoiser trained from scratch on 2,500 tiles cannot
acquire enough spatial-structure prior** to generate coherent linear road
features. The MSE noise-prediction objective is mean-seeking; under
uncertainty (and uncertainty is high with so little data), it converges
to "average of all possible road networks" = honeycomb mesh. No amount of:
- VAE capacity bumping
- Latent-shape change
- Conditioning architecture tweaks
- Loss reformulation (focal → CE+Dice, MSE → adversarial)
- Discrete vs continuous latents

...closed the gap. Each fixed *a* problem (e.g. VAE-GAN added adversarial
pressure and class discrimination jumped) but the underlying denoiser was
still too small to produce coherent road structure on this dataset.

### What finally worked

**SDXL ControlNet**, ~1 single training run (35 h), on the same data,
same prompt. The 12 B-parameter foundation model brings inductive biases
for parallel lines, perpendicular intersections, grids, and edge sharpness
that a 19 M-parameter denoiser cannot acquire from 2,500 tiles in any
amount of training time.

The total **GPU-hours** consumed across all the failed CaRoLS variants
combined (~50 hours) was actually higher than the single SDXL ControlNet
run (35 hours). We just spent 4 months learning what 1 training run on
the right substrate would deliver.

---

## 3. Lessons learned and bugs squashed

### Bugs squashed (across the full project, in order discovered)

1. **CFG dropout zeros entire cond tensor → BatchNorm in cond stem
   misbehaves at eval time.** `running_mean → 0, running_var → 0`;
   eval normalization divides by `~ε`. Fix: `BatchNorm2d → GroupNorm(8)`
   throughout the cond stem.
2. **Bilinear interpolation on one-hot landuse channels = silent label
   smearing.** 0.5×residential + 0.5×commercial = non-physical "half-class"
   features. Fix: `mode="nearest"` for categorical channels.
3. **SDXL VAE `scaling_factor=0.13025` calibrated on photos doesn't fit
   our palette inputs.** Per-channel `mu.std()` ranges from 0.5-1.5 even
   after the factor. Fix: compute empirical per-channel shift/scale
   from training set and bake them as buffers (`RoadVAESDXL.calibrate()`).
4. **DRoLaS Eq. 9 weight tensor was using `adaptive_avg_pool2d` instead
   of the VAE encoder.** Paper specifies `Σ w_i · E(m_i)` where E is the
   pretrained encoder. We computed `Σ w_i · pool(m_i)` which is wildly
   different. Fix: `compute_class_weight_latent()` in `model/diffusion.py`.
5. **Plan A double-injected `R_l` into the decoder chain.** Adding
   `R_l + R_up` then feeding the sum forward as both the next R_up
   and the cumulative residual = same signal in twice. Fix: split the
   accumulator from the decoder skip.
6. **VAE without adversarial pressure produces blur under MSE.** This
   is structural to MSE-trained generative models. Fix: add a PatchGAN
   discriminator + feature-matching loss (LDM/SDXL recipe).
7. **SDXL ControlNet: cached prompt embeds deserialize as fp32 from
   Arrow.** UNet is fp16. Cross-attention `to_k` crashes. Fix: cast
   in training loop. (See §1 above.)
8. **SDXL ControlNet: `imagefolder` doesn't auto-decode
   `conditioning_image`.** Fix: explicit `cast_column(..., Image())`.
9. **SDXL ControlNet: relative paths resolve against `cwd`, not
   `data_dir`.** Fix: absolute paths in `metadata.jsonl`.
10. **SDXL ControlNet: `val/` subdir auto-detected as split.** Fix:
    val tiles outside the dataset dir.

### Meta-lessons that cost us months

- **`val_loss` in latent space is a misleading metric.** Different runs
  use different latent statistics; the loss scale shifts. `diff_vae_v2`
  had our lowest val_loss but produced 99% bg outputs for "commercial"
  cond (mode collapse). Always pair val_loss with a downstream task
  metric (we built `diag_conditioning.py` for this).
- **Pixel-IoU is misleading at the road-class level.** Sub-pixel
  boundary drift caps IoU at ~0.67 regardless of model quality.
  Visual inspection of samples is the ground truth — and we should
  have been doing that from the start, instead of optimizing for a
  loss number.
- **"Multiple paradigms hit the same ceiling" is how you identify a
  *fundamental* bottleneck.** v1 (4-ch continuous), v2 (4-ch continuous,
  deeper), v3 (16-ch continuous), FSQ (discrete) all hit ~0.67 IoU.
  That's a data/labeling ceiling, not an architecture ceiling.
- **The right way to argue with yourself is to dispatch 4 independent
  adversarial agents.** Twice during this project we deployed a 4-agent
  council to debate diagnoses. Both times they collectively identified
  bugs and structural problems that we couldn't see alone (BatchNorm
  in cond stem, bilinear-on-one-hot, SDXL VAE scaling).
- **Foundation models are not a research luxury — they're cheaper
  than custom-training the equivalent capability.** Net GPU-hours we
  spent: ~50 across 30+ custom-model runs that didn't work, vs 35 on
  the one SDXL ControlNet that did. Custom models look cheaper per
  run (~1-5 h) but you need 30+ of them.

---

## 4. Implementation decisions for the code

### Why we vendored the diffusers training script

`third_party/train_controlnet_sdxl.py` is a hand-applied patched copy
of `diffusers` v0.38.0's official trainer. We considered:

- **`pip install diffusers` and override at runtime via monkey-patch** —
  fragile, hard to commit reproducibly
- **Maintain a fork of `diffusers`** — too much overhead for two
  one-line patches
- **Wait for upstream PRs to fix these bugs** — they're not bugs from
  upstream's perspective; the imagefolder caveats are documented (in
  pieces) and the dtype cast was added to other trainers but not SDXL's
- **Vendor + apply patches in-tree** ← what we did. The two patches
  are short, clearly marked, and the file is committed for reproducibility.

### All caches on `/scratch`

Cluster home (`/u`) has hard quotas (103 GB block, 500 K files). SDXL
weights + dataset cache + checkpoints would push us over both. **`/scratch`
has effectively unlimited space.** The training script must set:

```bash
export HF_HOME=/scratch/jalenj4/hf
export HF_DATASETS_CACHE=/scratch/jalenj4/hf_datasets
export TMPDIR=/scratch/jalenj4/tmp
OUT_DIR=/scratch/jalenj4/runs/sdxl_cnet_v1
```

Auto-resume relies on `OUT_DIR/checkpoint-*` existing across SLURM job
boundaries. `/scratch` persists across jobs.

### Constant prompt vs varied prompts

We use a single constant caption per tile:
```
"top-down satellite-style raster of a US suburban road network,
high-contrast color-coded road class map, flat color, vector style,
no texture, no shading"
```

Alternatives considered:
- **Per-tile descriptive prompts** ("Phoenix Arizona suburban grid with
  desert terrain") — would add prompt entropy and let the text encoder
  do some work
- **Random captions for CFG** — already handled via `--proportion_empty_prompts=0.1`

Why constant: the **ControlNet is doing all the work**. The text encoder
contributes almost nothing — all class differentiation comes through the
ControlNet input. The constant prompt acts as a stylistic anchor
("flat color, vector style, no texture") that helps push outputs toward
the map aesthetic and away from SDXL's natural-image prior. We tested
this implicitly: the visible style migration from "satellite photo"
(step 1000) to "color-coded map" (step 25000) is the constant prompt
doing exactly its intended job.

### 1024×1024 vs 512×512 training resolution

Flux/SDXL underperform at 512² because their positional encodings are
trained densely at 1024². Community-reported FID regression: 15-25%
at 512² for SDXL. Our dataset was 512², so we upscale to 1024² in
`prep_flux_dataset.py` using:
- Bilinear for elevation (continuous)
- Nearest-neighbor for one-hot/categorical channels

Storage cost is real (2,443 tiles × 1024² × 3 channels × uint8 ≈ 7 GB
on `/scratch`) but worth it for the quality.

### HF imagefolder via symlinks

`tools/build_hf_manifest.py` creates `data/flux_cnet_hf/` with **symlinks**
back to `data/flux_cnet/{cond,target}` plus a `metadata.jsonl`. This
saves ~50 GB vs duplicating the PNG files. The catch: the symlinks
**must use absolute targets** — relative symlinks resolve from
inside `flux_cnet_hf/` and don't find their targets.

### 5-color max-separated palette for output

Output is 5-class road categorical, but SDXL produces 3-channel RGB.
We chose:
- bg = `(0,0,0)`
- residential = `(255,0,0)`
- tertiary = `(0,255,0)`
- primary = `(0,0,255)`
- motorway = `(255,255,0)`

These are RGB cube corners chosen to maximize pairwise distance — making
nearest-color decoding tolerant of SDXL VAE drift at class boundaries.
Lab-space distance would be marginally better but the cube-corner
distances are already huge (255 in at least one axis between any pair).

### 7→3 cond encoding (hand-mapped, not learned)

`prep_flux_dataset.py` maps our 7-channel cond to 3-channel RGB by:
- `R = elevation`
- `G = 0.3·water + 0.5·parkland + 0.2·agricultural`
- `B = 0.6·residential + 0.8·commercial + 1.0·industrial`

Alternatives considered:
- **Learned `1×1 conv(7→3)`** — would have lost the warm-start benefit
  if we ever use a pretrained ControlNet, and 7-to-3 compression is
  inherently lossy
- **Multiple stacked ControlNets** (one per channel group) — 2× VRAM
  for marginal gain

The hand-mapping is lossy (commercial and industrial collapse onto the
same B-channel coefficient gradient) but works in practice because the
ControlNet learns to disambiguate.

### 25,000 training steps

ControlNet training in the wild ranges 15 k - 50 k steps. We picked
25 k because:
- 15 k showed clear "satellite photo" prior still dominating (step 1000-13000
  trajectory)
- 50 k would have needed 3 SLURM jobs instead of 2 and increased
  overfitting risk on our small dataset
- 25 k gave us a clear visible palette-commitment transition by ~18 k
  and clean convergence by 25 k

### `constant_with_warmup` LR schedule

ControlNet training doesn't benefit much from cosine decay over 25 k
steps. Constant LR with 500-step warmup is the diffusers README recipe
and the community standard.

### 8-bit Adam + gradient checkpointing

Both saved VRAM that we didn't strictly need on A100 80 GB, but:
- 8-bit Adam frees ~700 MB; cheap insurance against minor VRAM spikes
- Gradient checkpointing trades ~30% throughput for ~3-5 GB; we kept it
  because we'd hit OOM on first try without it

If migrating to a 24 GB GPU (RTX 3090), both become essential.

### Save final ControlNet at both top-level and `checkpoint-N/`

The training script saves intermediate checkpoints to `OUT_DIR/checkpoint-N/`
but also saves the final ControlNet directly at `OUT_DIR/`. We rely on
the top-level files (`config.json` + `diffusion_pytorch_model.safetensors`)
for the canonical "this is the trained model" path. The `checkpoint-N/`
dirs include optimizer state and can be used for resumption.

### HF Hub upload via CLI from login node

We initially tried the Python `huggingface_hub.upload_folder()` API
from a SLURM compute job. It died mid-upload with `RemoteProtocolError`.
The `hf upload` CLI from a login node worked at ~380 MB/s with no
hand-holding. **Use the CLI** for production uploads.

---

## Total cost accounting

| Item | Time | GPU-hours |
|---|---|---|
| CaRoLS reproduction (30+ small-model variants) | ~3 months | ~50 |
| SDXL ControlNet pivot (council reviews, scoping) | ~2 weeks | ~5 (diagnostics) |
| SDXL ControlNet training | 1.5 days | 35 |
| Validation sampling + showcase | ~1 day | ~2 |
| Documentation + HF Hub | ~1 day | ~0 |
| **Total** | **~4 months** | **~92 GPU-hours** |

GPU-hours is roughly evenly split between custom-model experiments and
the successful SDXL run. The custom-model work was not wasted — it
established what *doesn't* work and built the data pipeline, diagnostic
tooling, and postprocessing that the SDXL ControlNet output flows
through. But if I were starting over: I would have skipped to SDXL
ControlNet at month two.
