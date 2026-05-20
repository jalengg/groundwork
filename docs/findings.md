# Groundwork: Findings & Open Questions

A running log of experiments, hypotheses tested, and what we believe matters going forward. Useful if you're trying to reproduce CaRoLS for road-network synthesis, or just want to know what didn't work and why.

---

## Architecture / data baseline

- **Dataset:** 17 US suburban cities (Arlington TX, Chandler AZ, Mesa AZ, Henderson NV, etc.), 150 tiles each → ~2,550 training tiles. Roughly the same size as CaRoLS's 2,584-tile Australian dataset.
- **Tiles:** 512×512 patches at 5 m/pixel resolution. Latent space: 64×64×4 (8× spatial reduction). Identical dimensions to CaRoLS.
- **Conditioning channels (current, 7-ch):** `[elevation, water, residential, commercial, industrial, parkland, agricultural]`. Landuse is **categorical one-hot** per CaRoLS, replacing an earlier scalar 0..1 encoding that collapsed all class info.
- **Road output (5-ch one-hot)** with **per-class line widths** at rasterization: residential `2px` (~10m), tertiary `3px`, primary `4px`, motorway `5px`. Earlier all classes used uniform 5px.
- **Validation split:** held out city `irving_tx`. **Caveat:** stylistically similar to training cities, so this is a weak generalization test (CaRoLS uses Brisbane vs. Sydney+Melbourne — a stronger split).

### Pixel distribution (our data, post-thinning)

| Class | Our share | DRoLaS approx. |
|-------|-----------|----------------|
| Background | 78.4% | ~88–90% |
| Residential | 13.8% | ~6–8% |
| Tertiary | 2.1% | ~1–2% |
| Primary/Secondary | 4.2% | ~1% |
| Motorway/Trunk | 1.5% | ~0.5% |

US Sun Belt suburbs are road-dense vs. Brisbane suburbs in the reference papers. Our class-weighting must compensate accordingly — copying DRoLaS's `[1.0, 1.2, 1.4, 1.4, 1.4]` weights directly assumes their distribution.

---

## What we ruled out empirically

| Lever | Result |
|-------|--------|
| **Existing-roads conditioning channel (channel 3)** | Was a leak — model was doing semantic segmentation, not generation. Removed. |
| **Latent scaling** | Measured VAE latent std ≈ 1.04 across 100 samples. Stable Diffusion's `0.18215` trick wasn't applicable. |
| **Guidance scale `w`** | Tried `w ∈ {1.0, 1.5, 3.0}`. No meaningful difference. Symptom of an undertrained unconditional branch. |
| **EMA proxy (checkpoint averaging)** | Averaged epochs 380/390/400. Slightly cleaner arterials, no impact on residential noise. |
| **More training epochs** | Loss continued to decrease 200→400 (~7%) but visual quality plateaued. |
| **CFG sweep** `cfg_prob ∈ {0.3, 0.5, 0.7}` | `0.5` won numerically (val=0.191) but visually all three were indistinguishable — predictions were dense salt-and-pepper noise with no road topology. Lesson: hyperparameter sweeps don't move quality past a structural ceiling. |
| **Architectural rewrite of CDB toward paper's literal text** | **Catastrophic regression.** Reverted. See section below. |
| **`rot90` augmentation** | Removed in current pipeline. Hypothesis was that randomizing absolute orientation destroys the orientation prior. Trained run pending visual evaluation. |

---

## The CDB Architectural Experiment (2026-04-25)

We attempted to align our `ConditionAwareDecoderBlock` with a literal reading of CaRoLS Sections 3.2 / Fig. 2:

> "each CDB comprises a Local Details Enhancement (LDE) strategy, a Global Context Integration (GCI) strategy, **and a standard decoder block**: LDE and GCI strategies process the input intermediate representation `Z_c^i` to output `Z_l^i` and `Z_g^i` respectively; the standard decoder block produces a upsampled representation `Z_up^i`; and the element-wise addition of `Z_l^i` and `Z_up^i` is regarded as the CDB's output."

Two architectural variants were trained on the cluster. Both failed.

### OLD (working) architecture

```
   x_prev_resized      e_i (skip)            R_c[i] (cond)
        │                  │                       │
        └─────┬────────────┘                       │
              │                                    │
              ▼                                    │
     ┌─────────────────┐                           │
     │ CDB.forward()   │◄──────────────────────────┘
     │  ┌─ LDE: fuses noise+skip+cond ─┐
     │  └─ GCI: cross-attn(R_l, R_c)   ┘
     └────────┬────────┘
              │
              ▼
        cdb_out (at e_i resolution, channels carry cond+skip info)
              │
              ▼
        cat([cdb_out, e_i])  ◄── skip used twice (in CDB AND in concat)
              │
              ▼
     ┌─────────────────┐
     │ dec_i = UNetBlock│
     │ - 3×3 conv      │
     │ - GroupNorm     │
     │ - SiLU          │
     │ - 3×3 conv      │
     │ - self-attn     │
     │ - RESIDUAL: h+skip│  ◄── critical for trainability
     │ - ConvTranspose2d (upsample 2×)│
     └────────┬────────┘
              ▼
        d_i (next level)
```

**Result:** training loss → 0.33 by epoch 5 in earlier runs, plateaued at ~0.05 by epoch 200.

### NEW (broken) architecture

```
        x_prev               e_i (skip)         Z_c^i (cond, R_c[i])
           │                     │                    │
           └─────┐               │                ┌───┴────┐
                 ▼               ▼                ▼        ▼
           ┌──────────────────────┐          ┌──────┐  ┌──────┐
           │ StandardDecoderBlock │          │ LDE  │  │ GCI  │
           │  - fuse(cat([up,skip]))│        │ on Zc│  │ on Zc│
           │  - GN+SiLU+conv      │          └───┬──┘  └───┬──┘
           │  - t-emb             │              │         │
           │  - GN+SiLU           │              │         │
           │  - (v1: NO RESIDUAL) │              │         │
           │  - (v2: residual+attn)│             │         │
           └──────────┬───────────┘              │         │
                      │                          │         │
                      └──────────┬───────────────┴─────────┘
                                 ▼
                         Z_up + Z_l + Z_g
                                 │
                                 ▼
                          (next level)
```

**Result (both v1 and v2):** training loss stuck at **0.808** for 100 epochs across cfg_prob ∈ {0.3, 0.5, 0.7}. Output samples are random color splatter with no road structure. **Regression vs. OLD architecture.**

### What we learned

- **The paper's literal description doesn't carry the noise-prediction signal.** Per the literal text, all three branches operate on `Z_c^i` (conditioning features). The "standard decoder block" supposedly also processes `Z_c^i`, leaving no defined path for the noise stream into the decoder. Either the paper has unstated initialization tricks (e.g., zero-init final convs in LDE/GCI so they start as no-ops) or the text is imprecise.
- **v1's specific failure** was missing the `+ skip` residual that the OLD `UNetBlock` had — a deep decoder without residuals can't backprop reliably (grad_norm decayed steadily from 1.3 → 0.18).
- **v2 added back the residual + self-attn** but still failed at the same loss plateau. So the issue isn't gradient flow; it's *structural*: separating cond integration into parallel Z_l/Z_g branches that only see `Z_c` (not noise+skip) means cond cannot meaningfully shape the noise prediction. The OLD CDB integrates cond into the noise pathway (LDE fuses cond into the skip; GCI cross-attends fused features against cond) — this is the working topology.
- **Reverted both `cdb.py` and `unet.py`** to the OLD architecture. Resumed CFG sweep with the working code at `cfg_prob ∈ {0.3, 0.5, 0.7}` (jobs `8237051/52/53` on `ic-express`).

The takeaway: **paper-fidelity is not always the right north star for empirical work.** The OLD architecture deviates from a literal CaRoLS reading but trains; the literal-paper architecture doesn't.

---

## The honeycomb-vs-grid problem (open)

Even with a model that trains well, predictions look like a **honeycomb texture** — irregular cellular shapes — instead of real US suburban road networks, which have **strong parallel/grid structure**: long parallel residential streets at consistent spacing (~100–200 m), rectilinear blocks, low-entropy orientation distributions within a neighborhood.

Symptom interpretation: the model has learned **local color statistics** (correct road width, plausible branching count) but not **global texture statistics** (parallelism, regular spacing, repeated motif).

### Targeted next-steps (research synthesized 2026-04-25)

Three levers, ordered by effort/likelihood-of-helping:

#### 1. Orientation-field conditioning channel — cheap, high-value

Precompute a **2-channel principal-orientation field** per tile from OSM (or from raster GT via structure tensor + smoothing at ~256 m scale) and inject it as additional conditioning channels alongside elevation/landuse/water:

- Channel 4: `cos(2θ)` of dominant-orientation field
- Channel 5: `sin(2θ)` of dominant-orientation field

This mirrors the **tensor-field representation** used in classical procedural street modeling (Chen et al., SIGGRAPH 2008). Even smoothed/imperfect fields would tell the U-Net "in this tile the dominant axis is θ," collapsing orientation entropy.

**Effort:** ~1 day. **Likelihood:** high — directly addresses the missing inductive bias.

> Reference: [Interactive Procedural Street Modeling, Chen et al. SIGGRAPH 2008](https://www.sci.utah.edu/~chengu/street_sig08/street_sig08.pdf)

#### 2. Drop or align rotation augmentation + add a Fourier loss — cheap, medium-value

Two coupled changes:

- **Augmentation:** `rot90/flips` randomize absolute orientation per tile, *destroying* the prior that "within a 5×5 km region, most subdivisions share an axis." Switch to **flip-only**, or use **aligned rotation** (rotate input + condition together AND feed orientation as a label).
- **Spectral regularizer:** add a [Focal Frequency Loss](https://openaccess.thecvf.com/content/ICCV2021/papers/Jiang_Focal_Frequency_Loss_for_Image_Reconstruction_and_Synthesis_ICCV_2021_paper.pdf) (Jiang et al., ICCV 2021) or simpler magnitude-FFT L1 between predicted-`x_0` (decoded via VAE) and GT. Honeycomb textures show an isotropic ring in `|FFT|`; real grids show **discrete peaks along principal axes**. Frequency-domain loss explicitly penalizes the ring, rewards the peaks.

**Effort:** ~half-day for augmentation change; ~1 day for FFT loss. **Likelihood:** medium. Run the augmentation A/B first since it's a one-line config change.

#### 3. Two-stage: orientation/skeleton diffusion → raster diffusion — expensive, principled

If (1)+(2) only partially fix it, the structurally-correct move is **structure-first generation**, mirroring `VecFusion` ([arXiv 2312.10540](https://arxiv.org/html/2312.10540v2)) and `LDPoly` ([arXiv 2504.20645](https://arxiv.org/html/2504.20645)). Train a tiny first-stage diffusion (or deterministic regressor) that predicts a **low-res orientation field + skeleton heatmap** at `64×64`, then condition the existing latent diffusion on that. Decouples *what-direction* (global, low-frequency, easy) from *paint-the-roads* (local, what current model is good at).

**Effort:** 1–2 weeks. **Likelihood:** high if (1) doesn't suffice — but only attempt after (1) and (2).

### Things to skip

- **Switching U-Net → DiT** ([Peebles & Xie](https://www.wpeebles.com/DiT)): at 64×64 latent the U-Net already has near-global receptive field at the bottleneck. Architectural rewrite just bit us; conditioning-side fixes are far better effort/risk.
- **"Is-grid-like" classifier guidance**: subsumed by lever (1).
- **More training epochs**: validated to not help.

### Calibration

- **Most confident:** lever (1) (orientation channel) will improve residential parallelism.
- **Moderately confident:** removing `rot90` augmentation will help; FFT loss in isolation is less certain.
- **Uncertain:** lever (3) is needed if (1) is implemented well.

---

## Diagnostics added during investigation

- Validation loss every 5 epochs (was missing — only train loss was logged)
- Sample image generation every 25 epochs (visual progress check during training)
- Gradient norm logging + clipping at 1.0 (stability)
- `--cfg-prob` CLI arg for sweeps
- Corrupt-`.npy` scan + cleanup script (5 truncated files found, all in `irving_tx` / `carlsbad_ca` / `plano_tx`)

---

## DRoLaS architectural finding: LoAd ≠ LDE (2026-05-01)

We obtained the DRoLaS paper PDF (`Dong et al. ICMR 2025`) and read it end-to-end. **The LoAd module in DRoLaS is structurally different from the LDE module in CaRoLS** — and the DRoLaS ablation directly identifies LoAd as the single highest-impact architectural lever. Our current `model/cdb.py:LocalDetailsEnhancement` follows CaRoLS's concat-fusion LDE, **not** DRoLaS's affine-modulation LoAd. This is a candidate root cause for the spatial-conditioning failure (`res-vs-com 21%` pixel disagreement).

### LoAd (DRoLaS) — SFT/FiLM-style affine modulation

Per DRoLaS Eq. 3-4:

```
γ_i = conv_γ(f_i),  δ_i = conv_δ(f_i)
ĝ_i = γ_i ⊙ g_i + δ_i + g_i
```

`f_i` is the conditioning feature at level `i`. `g_i` is the noise-stream representation. Modulation is **per-pixel**: each spatial location gets its own scale and bias from the cond. **No spatial blending** in the modulation step — this preserves fine-grained spatial cond differences.

### LDE (CaRoLS, our impl) — concat-fusion

Per `model/cdb.py:30`:

```
fused = Conv3x3(cat([cond_proj, skip_proj]))
R_l = R_down + fused
```

The 3×3 conv has a 3-pixel receptive field, so it **blends features across a 3-pixel neighborhood**. At deeper CDB levels (4×4 latent), each pixel covers ~640 m of real-world space — a 3-pixel blur smears across an entire neighborhood. Plausible mechanism for why our model can't distinguish residential from commercial spatially.

### DRoLaS ablation table (Table 2)

| Setting | Added | FID ↓ | Δ vs prior |
|---|---|---|---|
| 1 | Baseline (concat fusion + standard MSE loss, no refiner) | 33.93 | — |
| 2 | + **LoAd** | 24.75 | **−9.18** ⭐ biggest single lever |
| 3 | + GloP | 24.47 | −0.28 |
| 4 | + class-weighted loss | 19.00 | −5.47 |
| 5 | + connectivity refiner (no disc detector) | 18.42 | −0.58 |
| DRoLaS | + connectivity refiner with disc detector | 17.50 | −0.92 |

**LoAd alone contributes −9 FID. GloP contributes −0.3.** Cross-attention (which we have) is small change relative to affine modulation (which we don't).

### GloP vs GCI — minor differences

- DRoLaS GloP: `Q` from `ĝ_i` (LoAd output), `K, V` from `f_i` directly
- CaRoLS GCI (ours): `Q` from `R_l`, `K, V` from `fuse(R_c, R_l)` (concat-and-fuse)

Same cross-attention math; K/V source differs slightly. Per the ablation, this difference is worth ~0.3 FID — marginal.

### Other DRoLaS findings worth noting

- **Class-weighted denoising loss formula matches our impl exactly**: `L_w = E[||W·(ε-ε_θ)||²]` with `W = Σ w_i E(m_i)` and weights `[1.4, 1.4, 1.4, 1.2, 1.0]`. ✓ already implemented (`model/diffusion.py`).
- **DRoLaS team explicitly states (page 6, line 482)**: *"Due to the unavailability of their source code, we re-implemented these methods to the best of our ability and retrained them on our dataset."* — confirms NO public code for `Birsak 2022`, `Yang 2023`, `Gu 2024`, or any baseline they compare to.
- **ControlNet was tested as a baseline** by DRoLaS and performed terribly (`FID=119` vs DRoLaS's `17.5`, "scattered patches not coherent road segments"). The ControlNet-as-shortcut idea from earlier brainstorming is **ruled out** for this task.
- **DRoLaS landuse channels**: 8 classes (commercial, parkland, educational, medical, residential, industrial, transportation, water) vs our 5 (residential, commercial, industrial, parkland, agricultural). They keep edu/medical/transport as separate channels; we collapsed.
- **DRoLaS dataset spec matches**: 2581 pairs, Sydney/Melbourne/Brisbane, 5m/px, 512×512 patches, 70/30 split. Same as ours (2590, US Sun Belt, 5m/px, 512×512).

### Implication for our work

The field's empirical evidence (DRoLaS Table 2) ranks LoAd *much* higher than the connectivity refiner or even the class-weighted loss as a quality lever. Our architecture has the third-biggest lever (class-weighted) but **lacks the biggest lever (LoAd)**. This is the most well-motivated next change — directly targets the spatial-conditioning failure mechanism, validated by ablation, ~50 lines of code.

### Next experiment

**Replace LDE with LoAd-style affine modulation** in `model/cdb.py`. Plumbed as a flag (`--lde-type {lde,load}`) so the existing `diff_safe_fix` / `diff_planA` checkpoints remain reproducible. Train under the same recipe (`lr=2e-5`, `batch=4`, `ρ=0.1`, `--class-weights`, 200 epochs, `IllinoisComputes-GPU` A100). Evaluate via `model/diag_conditioning.py` against the `>40%` res-vs-com acceptance criterion.

---

## References

- Feng T, Li L, Li W, Li B, Shen J. **CaRoLS: Condition-adaptive multi-level road layout synthesis.** *Computers & Graphics* 133 (2025) 104451. https://doi.org/10.1016/j.cag.2025.104451
- Dong S, Li W, Li B, Li L, Shen J, Feng T. **DRoLaS: Diffusion-Based Coarse-to-Fine Conditional Synthesis of Hierarchical Road Layouts.** ICMR 2025. https://doi.org/10.1145/3731715.3733316
- Wang X, Yu K, Dong C, Loy CC. **Recovering Realistic Texture in Image Super-Resolution by Deep Spatial Feature Transform (SFT).** CVPR 2018. (LoAd inspiration.)
- Chen G, Esch G, Wonka P, Müller P, Zhang E. **Interactive procedural street modeling.** SIGGRAPH 2008.
- Jiang L, Dai B, Wu W, Loy CC. **Focal Frequency Loss for Image Reconstruction and Synthesis.** ICCV 2021.
- Thamizharasan V, Liu Y, et al. **VecFusion: Vector Font Generation with Diffusion.** arXiv:2312.10540.
- **LDPoly: Latent Diffusion for Polygonal Road Outline Extraction.** arXiv:2504.20645.
- Peebles W, Xie S. **Scalable Diffusion Models with Transformers (DiT).** ICCV 2023.
