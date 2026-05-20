# CaRoLS architectural audit v2

Re-verified §3 against the impl. The earlier audit (`carols_spec_audit.md`) is partly invalidated.

## Confirmed correct

VAE shape `512×512×5 ↔ 64×64×4`, 3 enc/dec blocks; Swish in VAE/LDE/noise-enc; GN in VAE/noise-enc; focal `α=[0.1,0.3,0.6,1.0,2.0]`, `γ=2`, KL weight `1e-4`, KL form `½(σ²+μ²−1−lnσ²)`; self-attn after each noise-enc block; cond-enc emits `{R_c^0..3}`; LDE 1×1+Swish/1×1/concat→3×3 + skip-on-uncond + residual to `R_down`; GCI scaled-dot attn with `√D` scale, output FC, residual to `R_l`; DDPM ε-MSE; Adam β=(0.9,0.999), lr=2e-5, batch=4.

## Deviations

| # | Item | Paper | Ours | Impact | Fix |
|---|---|---|---|---|---|
| 1 | **CDB topology** | CDB *contains* std decoder: `LDE→R_l`, `GCI→R_g`, `std_dec(R_g)→R_up`, **out=`R_l+R_up`**; `R_up` feeds next CDB. | `dec_i` is **outside** CDB; CDB returns `R_l+GCI`; `dec_i` takes `concat(CDB_out, e_i)`. **`R_l+R_up` missing**. | Severe | Large — restructure. |
| 2 | **GCI K/V source** (user's bug) | K,V from **fused** repr (cond 1×1+Swish, `R_l` 1×1, concat, 3×3). | K,V from raw `R_c` via 1×1. | Severe | Small. |
| 3 | **LDE `i=3` branch** | `i=3`: skip concat-with-`R_up`, apply Swish+GN on `R_down+fused`. `i<3`: concat `R_up^(i+1)`+3×3, no extra Swish/GN. | Always concat `R_up`+3×3+Swish+GN. Deepest CDB gets upsampled bottleneck as fake `R_up`. | Medium | Branch on `i==3`. |
| 4 | **CDB residual `R^i=R_l+R_up`** | Element-wise add. | Missing (subset of #1). | Severe | Folds into #1. |
| 5 | **Cond-enc stem** | 3×3 + **ReLU** + **BN**. | Bare `Conv2d`. | Low–Med | Trivial. |
| 6 | **Cond-enc blocks** | strided + GN + Swish + 3×3. | `Conv(stride=2)+SiLU`. | Medium | Small. |
| 7 | **Noise stream stem** | "τ → 3×3 conv **and** 4 blocks". | No stem. | Low | Trivial. |
| 8 | **Noise-enc extras** | strided + GN + Swish + 3×3. | + t-emb FiLM, 2nd GN/SiLU, residual skip, post-attn LN. | Low (likely helpful) | Leave. |
| 9 | **VAE channels** | Strided conv handles channel change. | Pre-block `Conv`; `ResBlock` has 2 GN + 2 SiLU + skip + final SiLU. | Low | Note. |
| 10 | **VAE focal form** | Eq. 2 binary. | Multi-class softmax focal (y=t branch only). | Low | Reasonable. |
| 11 | **CFG ρ** | 0.5 | `--cfg-prob` default **0.1**. | Medium | Trivial. |
| 12 | **Inference** | T-step reverse (T=1000). | 50-step DDIM, guidance=3, x0 clamp ±3. | Standard. | Flag. |
| 13 | **Loss option** | Plain MSE. | MSE + opt-in DRoLaS class-weighted MSE. | Off-spec if on. | Flag. |
| 14 | **Cond channels m** | Land use + elev + **pop density**. | Elev + water + 5 land-use; no pop density. | Off-spec. | Out of scope. |

## Ambiguities

- Self-attn heads (we use 4); GCI hidden `D_i` (we use `C_i`); VAE widths `[64,128,256]`; CDB widths matched to noise-enc; diffusion schedule linear `1e-4→0.02`, T=1000.

## TL;DR — top 3 by impact

1. **CDB topology is wrong** (#1, #4): std decoder lives outside CDB; `R^i = R_l^i + R_up^i` never happens. Our decoder is vanilla U-Net with a side-channel CDB, not the spec'd block.
2. **GCI fusion is missing** (#2): K,V from raw `R_c`, not the `R_c⊕R_l` fusion — Q queries features that never saw `R_l`. The bug already found.
3. **LDE has no `i=3` branch** (#3): deepest CDB ingests bilinearly-upsampled bottleneck as fake `R_up`, contaminating low-resolution local-detail enhancement.

Honorable mentions: cond-enc stem missing `ReLU+BN` (#5); cond-enc blocks missing GN + 2nd conv (#6); CFG ρ default `0.1` vs `0.5` (#11).
