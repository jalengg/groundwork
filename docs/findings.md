# What we tried, and what worked

A running log of experiments and what they revealed. Useful if you're trying to reproduce CaRoLS or hit similar problems.

## Architecture / data baseline

- 17 US suburban cities (Arlington TX, Chandler AZ, Mesa AZ, Henderson NV, etc.), 150 tiles each → ~2,550 training tiles. Roughly the same size as CaRoLS's 2,584-tile Australian dataset.
- 512×512 patches at 5m/pixel resolution. Latent space: 64×64×4 (8× spatial reduction). Identical dimensions to CaRoLS.
- Conditioning channels: elevation (SRTM), land use (OSM polygons → scalar 0..1 encoding), water mask (OSM).
- One held-out city for validation: `irving_tx`. **Note:** this is a weak generalization test because it's stylistically similar to the training cities. CaRoLS uses Brisbane vs Sydney+Melbourne — a stronger split.

## Critical bug: the existing-roads conditioning channel was a leak

**Symptom:** Predictions looked like blurry copies of the ground truth. Initially celebrated as "the model learned!"

**Cause:** Channel 3 of the conditioning was a binary rasterization of the OSM road graph — the answer was being passed as input. The model was doing semantic segmentation (color the roads we already know about) rather than generation.

**Fix:** Set `cond_channels=3` and drop channel 3 from the dataset entirely. Retrained from scratch. Outputs got noisier but became an honest measure of what the model actually learned.

## Levers we tested and ruled out

| Lever | Result |
|-------|--------|
| **Latent scaling** | Measured VAE latent std ≈ 1.04 across 100 samples. Stable Diffusion's 0.18215 trick wasn't applicable. |
| **Guidance scale (w)** | Tried 1.0, 1.5, 3.0. No meaningful difference in output quality. Symptom of an undertrained unconditional branch (see CFG below). |
| **EMA proxy (checkpoint averaging)** | Averaged epochs 380/390/400. Slightly cleaner arterials, no impact on residential noise. Suggests training-step noise is not the issue. |
| **More training epochs** | Loss continued to decrease 200→400 (~7%) but visual quality plateaued. Just training longer is not the fix. |

## Levers we believe matter (but haven't fully tested as of writing)

| Lever | Why we think so |
|-------|-----------------|
| **CFG probability `ρ`** | We use `ρ=0.1`; CaRoLS uses `ρ=0.5`. Higher `ρ` means the model trains BOTH conditional and unconditional branches well, which makes guidance amplification at inference (`w · (ε_cond − ε_uncond)`) work as intended rather than amplifying noise. Currently sweeping `ρ ∈ {0.3, 0.5, 0.7}`. |
| **Multi-class landuse encoding** | We encode landuse as a single scalar (residential=0.2, commercial=0.4, …). CaRoLS uses 7 categorical types (commercial, parkland, education, medical, residential, industry, transport, water). Single scalar destroys category separability. |
| **Focal loss weights** | Paper specifies α=`[0.1, 0.3, 0.6, 1.0, 2.0]`. We use α=1/freq. May or may not be equivalent in practice. |
| **Post-processing pipeline** | Paper applies dilation → skeletonization → small-component pruning. They explicitly state this "connects scattered road segments." Implemented as `model/postprocess.py`. Effective when raw output has *some* connected segments; less so when raw output is mostly isolated dots (current state). |

## Architecture-level observations

- **Resolution is not the bottleneck.** Same as CaRoLS in every dimension we checked.
- **VAE quality is not the bottleneck.** Reconstructions are near-perfect (see `samples/vae_recon_check.png`). Diffusion model is the constraint.
- **Conditioning information is fundamentally limited.** Even a perfect model can't fully reconstruct ground truth from terrain alone — there's nothing in elevation/landuse/water that determines whether a residential street runs N-S vs E-W. The goal should shift from "match GT" to "produce plausible network."

## Diagnostics added during investigation

- Validation loss every 5 epochs (was missing — only train loss was logged)
- Sample image generation every 25 epochs (visual progress check during training)
- Gradient norm logging + clip to 1.0 (stability)
- `--cfg-prob` CLI arg for sweeps

## Things we haven't tried

- EMA in the training loop (only the checkpoint-averaging proxy)
- LR schedule (cosine warmup vs flat)
- More DDIM sampling steps (>50)
- Larger U-Net / different architecture
- Vector graph export from post-processed raster
