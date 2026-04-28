# CaRoLS Specification Audit

> Reference: Feng T, Li L, Li W, Li B, Shen J. CaRoLS: Condition-adaptive multi-level road layout synthesis. *Computers & Graphics* 133 (2025) 104451.

A line-by-line comparison of the paper's spec against our implementation in `groundwork/`. Each item is one architectural decision from the paper.

**Status legend:**
- `OK` matches paper exactly
- `WARN` minor deviation / unclear in paper
- `BAD` deviates from paper
- `?` not yet verified or insufficient info in paper excerpt

---

## Stage 1: Multi-level Layout Reconstruction (VAE)

### Input/output dimensions

| # | Spec | Our impl | Status | File:line |
|---|------|----------|--------|-----------|
| 1.1 | Input image is `R^(512×512×k)` — k road levels, one-hot per pixel | `out_channels=5`, dataset loads `road_*.npy` of shape `(5, H, W)` | `OK` (k=5) | `vae.py:55`, `dataset.py:30` |
| 1.2 | Latent is `R^(64×64×4)` (high-dim continuous) | `latent_channels=4`, three stride-2 downsamples 512→256→128→64 | `OK` | `vae.py:36-47` |
| 1.3 | Decoder reconstructs `A` from `Ã` at full 512² resolution | Decoder produces 5-channel logits at 512² | `OK` | `vae.py:55-65` |
| 1.4 | Each pixel is a **k-dim one-hot vector** (exactly one channel = 1) | Dataset loads pre-computed one-hot tensors; reconstruction trained with focal loss across 5 channels (categorical) | `OK` | `dataset.py:30`, `vae_loss.py:11-19` |

### VAE encoder architecture

Paper: encoder `E` comprises **3 encoder blocks**, each block = `3x3 strided conv (downsample) + GroupNorm + Swish + 3x3 conv`.

| # | Spec | Our impl | Status | File:line |
|---|------|----------|--------|-----------|
| 2.1 | 3 encoder blocks (downsample 512 to 64 in three stride-2 steps) | 3 `ResBlock` instances each with `stride=2`: 512->256, 256->128, 128->64 | `OK` | `vae.py:38-45` |
| 2.2 | Block contains a 3x3 **strided** conv for downsampling | `ResBlock.conv1 = Conv2d(C, C, 3, stride=2, padding=1)` | `OK` | `vae.py:11-17` |
| 2.3 | Followed by GroupNorm | `nn.GroupNorm(8, channels)` (8 groups) | `OK` (8 groups not specified by paper) | `vae.py:18` |
| 2.4 | Followed by Swish activation | `nn.SiLU()` (= Swish) | `OK` | `vae.py:21` |
| 2.5 | Followed by another 3x3 conv | `ResBlock.conv2 = Conv2d(C, C, 3, padding=1)` | `OK` | `vae.py:19` |
| 2.6 | (Implicit) some channel-doubling per block | Channel widths: in_ch -> 64 -> 128 -> 256, achieved by 1x1 / 3x3 channel-mixing convs **between** ResBlocks rather than inside the block | `WARN` Paper does not specify channel counts; the residual block expands channels via separate 3x3 channel-change convs (`vae.py:39, 41, 43`), and `ResBlock` itself is constant-channel. Net topology is equivalent. | `vae.py:38-45` |
| 2.7 | (Implicit) residual / shortcut connection inside block | `ResBlock` adds `avg_pool2d(x, stride)` shortcut to `h` and re-applies SiLU on the sum | `WARN` The paper's prose does not explicitly require a residual connection inside the encoder block — ours adds one. Likely benign, but a deviation from a strict literal reading. | `vae.py:23-32` |
| 2.8 | Output produces `mu` and `logvar` for sampling `Ã` | Two 1x1 convs `to_mu`, `to_logvar` produce the 4-channel statistics | `OK` (paper does not specify how `mu`/`logvar` heads are formed) | `vae.py:46-47` |

### VAE decoder architecture

Paper: decoder `D` is **symmetric / mirror** of `E` with **3 decoder blocks**; each initial 3x3 strided conv is replaced with a **3x3 transposed conv** for upsampling.

| # | Spec | Our impl | Status | File:line |
|---|------|----------|--------|-----------|
| 3.1 | 3 decoder blocks (upsample 64 to 512) | 3 `ResBlock(stride=2, upsample=True)`: 64->128, 128->256, 256->512 | `OK` | `vae.py:57-63` |
| 3.2 | Each block uses a **3x3 transposed conv** for upsampling | `ResBlock` with `upsample=True` uses `nn.ConvTranspose2d(C, C, 3, stride=stride, padding=1, output_padding=stride-1)` | `OK` | `vae.py:11-17` |
| 3.3 | Block then includes GroupNorm + Swish + another 3x3 conv (mirror of encoder) | Same `ResBlock` body as encoder side | `OK` | `vae.py:11-32` |
| 3.4 | Decoder ends at 512x512 with k channels of logits | Final `Conv2d(base_ch, out_channels=5, 3, padding=1)` | `OK` | `vae.py:64` |
| 3.5 | (Implicit) residual / shortcut in upsampling block | `ResBlock` shortcut uses `F.interpolate(x, scale_factor=2)` for the upsample path | `WARN` Same caveat as 2.7 — paper does not require a residual connection in the decoder block. | `vae.py:26-27` |

### VAE loss function

Paper: `Loss_VAE = 0.0001 * Loss_KL + Loss_Focal` (Eq. 1).
`Loss_Focal` per Eq. 2 with `alpha_t` per category, `gamma` adjustable.
`Loss_KL = 0.5 * (sigma^2 + mu^2 - 1 - ln(sigma^2))` (Eq. 3).

| # | Spec | Our impl | Status | File:line |
|---|------|----------|--------|-----------|
| 4.1 | KL weight = `1e-4` | `kl_weight=1e-4` | `OK` | `vae_loss.py:26` |
| 4.2 | KL form `0.5 * (sigma^2 + mu^2 - 1 - ln(sigma^2))` | `kl_loss = -0.5 * mean(1 + logvar - mu^2 - exp(logvar))`, equivalent algebraically | `OK` | `vae_loss.py:22-23` |
| 4.3 | Focal: `-alpha_t (1 - p_t)^gamma log(p_t)` for the true class | Standard categorical focal: `log_p = log_softmax(logits)`; `target_p = sum(p * targets)`; `focal_weight = (1 - target_p)^gamma`; `loss = -focal_weight * sum(log_p * targets)` | `OK` Implements the **first branch** of Eq. 2 (positive class, `c = c_hat`). | `vae_loss.py:5-19` |
| 4.4 | Focal: `-alpha_t * p_t^gamma * log(1 - p_t)` for negative classes | **Not explicitly implemented**. Our `focal_loss` only sums the positive (`c = c_hat`) branch (because `targets` is one-hot and we mask with `targets`). | `WARN` This is a standard interpretation of multi-class focal loss for one-hot targets — the negative-class term in Eq. 2 is implicit through the softmax denominator. Not a real deviation, but the paper's two-branch formulation is unusual; flag for review. | `vae_loss.py:18` |
| 4.5 | `gamma` is "an adjustable factor" | `gamma=2.0` (default) | `WARN` Paper does not give gamma; 2.0 is the standard Lin et al. value. | `vae_loss.py:5, 26` |
| 4.6 | `alpha_t` is "the weight for category t" | `alpha = 1 / freq(c)`, normalized to sum to 1, **computed per batch** from `targets.mean(dim=(0,2,3))` | `WARN` Paper does not say how `alpha_t` is computed. We use inverse class frequency (a common heuristic). Per-batch computation may be noisier than dataset-wide; flag for review. | `vae_loss.py:28-30` |
| 4.7 | Optimizer / LR / batch size | Adam, `lr=2e-5`, `betas=(0.9, 0.999)`, `batch=4`, 50 epochs default | `?` Not in this excerpt. | `train_vae.py:25, 42` |

---

## Stage 2: Condition-adaptive Representation Generation (Diffusion U-Net)

### Forward / reverse diffusion process

Paper (Eqs. 4-6): standard DDPM. `p(x_{t-1} | x_t) = N(mu_theta, sigma_theta)`; `mu_theta = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * eps_theta)`; loss `L = E[||eps - eps_theta(x_t, t, C_tilde)||^2]`. C_tilde is "intermediate representations from C" in conditional mode, **0** in unconditional mode.

| # | Spec | Our impl | Status | File:line |
|---|------|----------|--------|-----------|
| 5.1 | Epsilon-prediction MSE objective | `F.mse_loss(eps_pred, eps)` | `OK` | `diffusion.py:31` |
| 5.2 | Forward diffusion `x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) eps` | Implemented exactly | `OK` | `diffusion.py:15-21` |
| 5.3 | `C_tilde = 0` in unconditional mode | Training does CFG dropout: `cond_masked = cond * mask`, where `mask = 1` w.p. `1-cfg_prob` and 0 otherwise. When dropped, `cond` is **all zeros** (matches "0 in the unconditional mode"). | `OK` | `diffusion.py:28-29` |
| 5.4 | Number of diffusion steps `T` | `T=1000` | `?` Paper silent in this excerpt. Standard DDPM value. | `diffusion.py:6` |
| 5.5 | Beta schedule | Linear, `beta_start=1e-4`, `beta_end=0.02` | `?` Paper silent. Standard DDPM (Ho et al. 2020) values. | `diffusion.py:6-8` |
| 5.6 | Classifier-free guidance dropout probability | `cfg_prob=0.1` (default in `train_diffusion.py`); training arg-help comment says "0.5 = CaRoLS spec" | `?` Paper excerpt does **not** mention CFG. The "0.5 spec" claim in our help text suggests we have outside info from a later section we don't have here. Currently configured to **0.1**, not 0.5. | `train_diffusion.py:69` |
| 5.7 | Sampler at inference | DDIM, **50 steps**, `guidance_scale=3.0`, `x0_pred` clamped to `[-3, 3]` | `?` Paper excerpt does **not** mention DDIM, sampler step count, or guidance scale. | `diffusion.py:33-53`, `train_diffusion.py:46` |

### U-Net encoder architecture

Paper: input noise `z` -> 3x3 conv -> **4 encoder blocks**. Each block = `3x3 strided conv (downsample) + GroupNorm + Swish + 3x3 conv`. **Self-attention module** placed **after each encoder block** for long-range dependencies. All convs in U-Net encoder/decoder are 3x3.

| # | Spec | Our impl | Status | File:line |
|---|------|----------|--------|-----------|
| 6.1 | 4 encoder blocks operating on the noise path | `noise_enc1..noise_enc4` plus a separate `bottleneck` | `WARN` We have **4 strided encoder blocks** (matches) plus a 5th non-strided `bottleneck` block. The paper does not explicitly call out a bottleneck — common practice but a minor extra. | `unet.py:86-90` |
| 6.2 | Initial 3x3 conv before the encoder blocks | **Missing** for the noise path. Paper says "input Gaussian noise z is processed by a 3x3 convolutional layer **and** 4 encoder blocks". Our `noise_enc1` immediately downsamples `z` (4ch -> 64ch with stride 2). There is no pre-encoder stem conv on the noise path. | `BAD` Deviation: paper has a 3x3 stem conv, ours does not. | `unet.py:86` |
| 6.3 | Each encoder block: 3x3 strided conv -> GroupNorm -> Swish -> 3x3 conv | `UNetBlock`: `Conv2d(.., 3, stride=2, padding=1) -> GroupNorm(8) -> SiLU -> Conv2d(.., 3, padding=1) -> GroupNorm(8) -> SiLU` | `WARN` Our block has **two GroupNorm+SiLU** stages (one after each conv); paper prose lists only one GroupNorm and Swish then a second conv. Paper is ambiguous about whether the second conv has its own norm/activation — common practice is yes. | `unet.py:33-46` |
| 6.4 | All convs in U-Net are 3x3 | All convs in `UNetBlock` are 3x3; `to_mu`/`to_logvar` 1x1 are not in the U-Net itself; final output `Conv2d(base_ch, latent_channels, 1)` is **1x1**. | `WARN` Final projection is 1x1 not 3x3. The paper's "all encoder and decoder blocks use 3x3" likely doesn't constrain the output projection, but worth noting. | `unet.py:102` |
| 6.5 | Self-attention **after each encoder block** | Self-attention is applied **inside** every `UNetBlock` (encoder, decoder, **and bottleneck**), not just after encoder blocks | `WARN` Stronger than spec — paper says self-attention only on encoder side; ours puts it on decoder blocks too. Functionally a superset. | `unet.py:40, 47-51` |
| 6.6 | Self-attention head count | `num_heads=4` | `?` Not in excerpt. | `unet.py:40` |
| 6.7 | Channel widths for noise encoder blocks | 4 -> 64 -> 128 -> 256 -> 256 | `?` Not in excerpt. | `unet.py:86-89` |
| 6.8 | Timestep conditioning | Sinusoidal embedding + 2-layer MLP, projected per-block via `nn.Linear(t_dim, out_ch)` and added to feature maps | `?` Paper does not describe how `t` is injected. Standard DDPM practice. | `unet.py:10-22, 37, 45` |
| 6.9 | Latent input shape | `R^(64x64x4)` Gaussian noise | `OK` `latent_shape=(1, 4, 64, 64)` in sampler | `diffusion.py:35`, `unet.py:79` |

### Conditioning image processing

Paper: `C in R^(512×512×k)` -> `3x3 conv + ReLU + BatchNorm` -> `Z_c in R^(512×512×64)`. Then U-Net's encoder blocks transform `Z_c` into `{Z_c^0, Z_c^1, Z_c^2, Z_c^3}` for the four CDBs.

| # | Spec | Our impl | Status | File:line |
|---|------|----------|--------|-----------|
| 7.1 | Conditioning image has **k channels** (same as road layout) | We pass **3 channels** (`cond_channels=3`); dataset drops the existing-roads channel: `cond = np.load(cond_path)[:3]` | `BAD` **Intentional deviation, flagged**: paper uses k-channel conditioning (containing land use, elevation, population density); we explicitly drop channel 3 ("existing roads") to "force terrain-only generation". `cond_channels=3` vs paper's k. | `dataset.py:29`, `train_diffusion.py:104` |
| 7.2 | Initial conditioning layer = `3x3 conv` | `ConditionEncoder.stem = Conv2d(in_ch, 64, 3, padding=1)` | `OK` | `unet.py:63` |
| 7.3 | Initial conditioning activation = **ReLU** | **Missing** the activation entirely on the stem; subsequent blocks use **SiLU**, not ReLU | `BAD` Paper specifies ReLU after the stem conv; ours has no activation on the stem and uses SiLU on encoder layers. | `unet.py:63-67` |
| 7.4 | Initial conditioning normalization = **BatchNorm** | **Missing** — no normalization in `ConditionEncoder` at all (no BN, no GroupNorm) | `BAD` Paper specifies BatchNorm after the conv+ReLU stem. | `unet.py:63-67` |
| 7.5 | Initial `Z_c` shape: `R^(512x512x64)` | `stem` produces `(B, 64, 512, 512)` | `OK` (channel and spatial size match) | `unet.py:63` |
| 7.6 | Encoder blocks transform `Z_c` into `{Z_c^0, Z_c^1, Z_c^2, Z_c^3}` (4 levels) | `enc1..enc4` produce 4 feature maps at 256, 128, 64, 32 px (returned as tuple `(h1, h2, h3, h4)`) | `WARN` We produce 4 levels but the paper's `Z_c^0..Z_c^3` is ambiguous re: which resolutions. Our deepest level is 32px (extra downsample); paper says 4 encoder blocks each with `3x3 strided conv` for the noise path — by symmetry the conditioning encoder would also reach 32px starting from 512. Likely matches. | `unet.py:64-75` |
| 7.7 | Each conditioning encoder block matches the noise encoder block structure (strided 3x3 + GroupNorm + Swish + 3x3) | `enc1..enc4` are just `Conv2d(stride=2) + SiLU` — **no GroupNorm, no second 3x3 conv** | `BAD` The condition encoder is significantly simpler than the paper's spec. The paper states the encoder blocks process `Z_c` — i.e., the same encoder block architecture should apply; ours is a single conv+SiLU per level. | `unet.py:64-67` |

### Condition-aware Decoder Block (CDB)

Paper (Fig. 2 description): `Z_c^i` -> LDE -> `Z_l^i`; `Z_c^i` -> GCI -> `Z_g^i`; standard decoder block produces `Z_up^i` from `Z_c^i` (sic — likely a typo, should be from upstream decoder feature). **Output**: `Z^i = Z_l^i + Z_up^i` (element-wise addition of LDE output and standard decoder output).

| # | Spec | Our impl | Status | File:line |
|---|------|----------|--------|-----------|
| 8.1 | One CDB per decoder level (4 total) | `cdb1..cdb4` (4 instances) | `OK` | `unet.py:93-96` |
| 8.2 | Each CDB takes the corresponding `Z_c^i` | We pass `R_c[3], R_c[2], R_c[1], R_c[0]` to `cdb1..cdb4` respectively (deepest-first) | `OK` | `unet.py:114-117` |
| 8.3 | LDE produces `Z_l^i` from `Z_c^i` | `LocalDetailsEnhancement.forward(R_down, R_up, R_c)` mixes the encoder skip (`R_down`), the upsampled decoder feature (`R_up`), and `R_c` to produce a fused tensor | `WARN` Our LDE takes **three** inputs (skip, upsampled-decoder-feature, condition); paper text says LDE processes `Z_c^i`. Implementation is richer than paper's prose. | `cdb.py:6-27` |
| 8.4 | GCI produces `Z_g^i` from `Z_c^i` | `GlobalContextIntegration` is an attention layer with `Q` from `R_l` and `K, V` from `R_c` (or from `R_l` if unconditional); returns `R_l + S` | `WARN` Paper says GCI processes `Z_c^i`; ours treats `Z_l` as query, `Z_c` as key/value (cross-attention), returning the fusion of the two. This is a sensible "global context integration" but diverges from the literal paper prose. | `cdb.py:30-51` |
| 8.5 | Standard decoder block produces `Z_up^i` from `Z_c^i` | `dec1..dec4` `UNetBlock(stride=2, upsample=True)` are applied **after** the CDB output is concatenated with `e_i` (skip from noise encoder). The "standard decoder block" in our pipeline is therefore `dec_k`, applied to `[CDB_output, noise_encoder_skip]`. | `WARN` Paper's wording is ambiguous (it likely meant from upstream decoder, not `Z_c^i`). Our split makes the CDB a feature-fusion module that feeds into a separate U-Net decoder block. Functionally the components are present but the topology differs. | `unet.py:98-101, 114-117` |
| 8.6 | **CDB output** = `Z_l^i + Z_up^i` (element-wise add) | Our `ConditionAwareDecoderBlock.forward` returns `self.gci(R_l, R_c)` — i.e., **GCI applied to LDE output**. Then this CDB output is **concatenated** (not added) with the noise-encoder skip and passed to the U-Net decoder block (`unet.py:114`). | `BAD` Two deviations: **(a)** Paper's stated CDB output is `Z_l + Z_up`, but ours composes `gci(lde(...), R_c)` — GCI is applied **on top of** LDE, not as a parallel branch. The role of `Z_g^i` (GCI's stated output) is not the parallel branch the paper implies. **(b)** The CDB is followed by `concat + UNetBlock(upsample)` rather than feeding into an additive sum with the standard decoder block's output. | `cdb.py:54-62`, `unet.py:114-117` |
| 8.7 | **What happens to `Z_g^i` (GCI output)** | Paper text in this excerpt does not say. Our implementation makes GCI output **the** CDB output (subsuming LDE), so `Z_g^i` is the final CDB output rather than a separate branch added in. | `?` **Paper is silent in this excerpt** on how `Z_g^i` combines with `Z_l^i` and `Z_up^i`. Worth examining Fig. 2 (which we don't have). Flag as ambiguous. | `cdb.py:60-62` |
| 8.8 | LDE strategy details (architecture) | Cond projection `Conv2d(cond_ch, ch, 1) + SiLU`, skip projection `Conv2d(ch, ch, 1)`, fusion via 3x3 conv, then concat with `R_up` and another 3x3 conv + GroupNorm + SiLU | `?` Paper does not describe LDE internals in this excerpt. | `cdb.py:6-27` |
| 8.9 | GCI strategy details (architecture) | Q from `R_l` via 1x1 conv, K/V from `R_c` (or self-attention if unconditional) via 1x1 convs, scaled dot-product, residual add | `?` Paper does not describe GCI internals in this excerpt. | `cdb.py:30-51` |
| 8.10 | Final output `A_hat` = `Z^0` from the last CDB, decoded by VAE D | Last CDB is `cdb4` at the highest spatial resolution; final `Conv2d(base_ch, 4, 1)` produces the 4-channel latent that is then decoded by `vae.decode` | `OK` (output conv is 1x1; see 6.4) | `unet.py:117, 102`, `train_diffusion.py:47` |

### Inference / sampling

| # | Spec | Our impl | Status | File:line |
|---|------|----------|--------|-----------|
| 9.1 | Sampler | DDIM, 50 steps | `?` not in this excerpt — verify from later sections of the paper | `diffusion.py:34, 40` |
| 9.2 | Classifier-free guidance scale `w` | `guidance_scale=3.0` | `?` not in this excerpt | `diffusion.py:34, 47` |
| 9.3 | CFG dropout probability `rho` during training | `cfg_prob=0.1` (CLI default), help text claims "0.5 = CaRoLS spec" | `?` not in this excerpt — currently set to **0.1**, not 0.5; if 0.5 is the true paper value this is a deviation | `train_diffusion.py:69`, `diffusion.py:23` |
| 9.4 | Unconditional inference uses `C_tilde = 0` | At inference, `eps_uncond = model(x, t, torch.zeros_like(cond))` | `OK` matches Eq. 4-5's "0 in unconditional mode" | `diffusion.py:46` |
| 9.5 | `x0_pred` clamping | Clamps `x0_pred` to `[-3, 3]` | `?` Not in paper. Engineering choice. | `diffusion.py:51` |
| 9.6 | Final decode `A = D(A_hat)` | `vae.decode(z)` then `argmax` over 5 channels for one-hot output | `OK` | `train_diffusion.py:47-50` |

---

## Critical mismatches summary

Ordered by likely impact on output quality:

1. **CDB output composition (8.6)**: Paper says `Z^i = Z_l^i + Z_up^i` (LDE output **added** to the standard-decoder-block output). Ours **composes** GCI on top of LDE (`gci(lde(...), R_c)`) and then **concatenates** with the encoder skip into a `UNetBlock`. The "parallel branch" structure of LDE/GCI/standard-decoder is not preserved. Most consequential architectural mismatch — the entire CDB topology differs from the paper.
2. **Conditioning channels reduced from k=5 to 3 (7.1)**: Dataset drops the "existing roads" conditioning channel. **Intentional deviation** to force terrain-only synthesis. Will produce different conditional behavior than paper.
3. **Conditioning encoder is too shallow (7.4, 7.7)**: The condition stem is missing **BatchNorm** and **ReLU** from the paper spec; the per-level encoder blocks are reduced to a single `Conv2d(stride=2) + SiLU` instead of the full 4-stage encoder block (strided 3x3 + GroupNorm + Swish + 3x3). The conditioning representation is therefore lower-capacity than the paper's design.
4. **No 3x3 stem conv on the noise path (6.2)**: Paper says `z` is processed by `3x3 conv` **before** the 4 encoder blocks. Our `noise_enc1` immediately downsamples. Minor capacity / receptive-field deviation.
5. **Self-attention placed in decoder blocks too (6.5)**: Paper places attention only after encoder blocks. Ours puts attention inside every `UNetBlock` (encoder, decoder, bottleneck). Likely benign / functionally a superset.

---

## Items where the paper is silent or ambiguous

- **Channel widths of U-Net blocks**: Paper does not specify per-block channel counts. Ours: noise path uses `64 -> 128 -> 256 -> 256` (encoder), conditioning path mirrors this.
- **Number of self-attention heads**: Paper does not specify. Ours: `num_heads=4`.
- **Diffusion step count `T`**: Paper says T-step DDPM but excerpt does not give T. Ours: `T=1000` (standard).
- **Beta schedule**: Paper does not specify in this excerpt. Ours: linear `[1e-4, 0.02]` (Ho et al. 2020 default).
- **Sampler choice (DDIM vs DDPM) and step count**: Paper excerpt is silent. Ours: DDIM with 50 steps.
- **Classifier-free guidance**: Paper excerpt does not mention CFG, dropout `rho`, or guidance scale `w`. Ours: `cfg_prob=0.1` (training arg-help comment claims `0.5` is the "CaRoLS spec" — verify against full paper); `guidance_scale=3.0`.
- **Timestep conditioning method**: Paper does not describe how `t` is injected. Ours: sinusoidal embedding -> 2-layer MLP -> per-block linear projection added to feature map.
- **Focal loss `gamma` and `alpha_t` computation**: Paper says `gamma` is "adjustable" and `alpha_t` is "the weight for category t" but does not give values or a recipe. Ours: `gamma=2.0`; `alpha = 1/freq` per **batch** (normalized).
- **Optimizer / LR schedule**: Paper excerpt does not specify. Ours: Adam `lr=2e-5`, `betas=(0.9, 0.999)`, no LR schedule, gradient clipping at 1.0 for diffusion training.
- **GroupNorm group count**: Ours uses 8 groups. Paper does not specify.
- **Whether VAE encoder/decoder blocks include a residual connection**: Paper prose does not require one. Ours adds a residual (`avg_pool`-projected shortcut on encoder side, `interpolate`-projected on decoder side).
- **Behavior of `Z_g^i` (GCI output) in the CDB**: Paper text in this excerpt does not state how `Z_g` combines with `Z_l` and `Z_up`. Refer to Fig. 2 (not available).
- **Resolution at which the conditioning encoder operates**: Paper says 4 levels for the conditioning U-Net encoder; the deepest spatial resolution is unstated. Ours reaches 32x32 (downsampling 4x from 512).
- **`x0_pred` clamping during DDIM**: Engineering detail not in paper. Ours clamps to `[-3, 3]`.
