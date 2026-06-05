You're picking up the Groundwork project at https://github.com/jalengg/groundwork.
Read these files first:

  docs/sdxl_controlnet.md       — the teacher model (SDXL + ControlNet, 12 GB VRAM)
  docs/postmortem.md            — why we ended up here; what smaller models can't do
  tools/sdxl_cnet_sample.py     — reference inference code
  model/postprocess.py          — road-graph vectorization pipeline
  data_pipeline/prep_flux_dataset.py  — how training data is encoded

Existing state:
  - Teacher model: SDXL base (3.5B) + trained ControlNet (~1.4B) on HuggingFace
    Hub (private: jalengg/groundwork-sdxl-cnet-us-suburbs)
  - Inference: ~30s/tile @ 30 DDIM steps on A100, ~10-12 GB VRAM
  - Training data: 2,443 paired tiles (1024×1024 RGB cond → RGB road map)
  - Output: 5-class color-coded road network, decoded via nearest-color palette
  - Eventually: multiple style ControlNets (US grid, European, Latin American,
    etc.) per a separate multi-style expansion stream

The goal: a **student model** that produces smaller, lower-resolution road
network neighborhoods (~256×256 or 512×512) on **consumer laptop hardware**
(no discrete GPU, or integrated graphics, or a laptop RTX 3050 4GB). NOT for
Cities Skylines — this is a standalone tool (think: quick sketch generator,
urban planning prototype, tabletop RPG map maker, indie game asset pipeline).

Constraints:
  - Inference VRAM: **≤4 GB** (must run on integrated graphics or RTX 3050)
  - Inference latency: **<5 seconds** per tile on a 2024 MacBook Air M3 or
    equivalent laptop CPU
  - Model download size: **<500 MB** (ideally <200 MB)
  - Must preserve: road structure coherence, class hierarchy, cond-responsiveness
  - Acceptable to lose: fine texture detail, 1024×1024 resolution, photorealistic
    quality

Your task — research and produce a concrete distillation plan:

1. **Survey distillation approaches for diffusion models (2024-2026 SOTA).**
   Specifically evaluate each for our use case:

   - **LCM-LoRA** (Luo et al. 2023, "Latent Consistency Models"): train a
     LoRA adapter on the teacher's latent trajectory so the student can
     generate in 1-4 steps instead of 30. Keeps the same base model (still
     big) but radically cuts inference time. Does it cut VRAM?

   - **Progressive distillation** (Salimans & Ho 2022): halve the number of
     steps iteratively (32→16→8→4→2→1). Each round trains a student to match
     the teacher's 2-step output in 1 step.

   - **Consistency distillation** (Song et al. 2023): train the student to
     map any point on the ODE trajectory directly to x_0. 1-step inference.

   - **Architecture distillation** (BK-SDM, Segmind, etc.): train a
     *smaller architecture* (e.g., SD 1.5-scale ~860M, or even smaller
     ~100-300M custom U-Net/DiT) to match the teacher's outputs. This is
     what actually shrinks the model size and VRAM.

   - **SDXL-Turbo / SDXL-Lightning** (Sauer 2023, Lin 2024): adversarial
     distillation to 1-4 steps. These are published recipes — can we
     fine-tune their existing distilled checkpoints with our ControlNet?

   - **Tiny diffusion** (custom small DiT/U-Net trained from scratch on
     teacher-generated data): skip the pretrained-model dependency entirely.
     Generate 50k-100k synthetic tiles from the teacher, train a 50-200M
     model on those. This is the "knowledge distillation via synthetic data"
     approach. Pros: tiny model, fast inference, no SDXL dependency at
     inference. Cons: quality ceiling.

   - **Non-diffusion student**: the teacher produces paired (cond, road)
     data. Train a direct **pix2pix / U-Net / ConvNeXt regression model**
     that maps cond → road in a single forward pass. No iterative sampling.
     Inference in <100ms. Quality floor is the question.

   For each, estimate: model size, VRAM, inference latency on M3 MacBook,
   quality vs teacher, training cost.

2. **Recommend a primary path + fallback.** Given the constraints (<4 GB
   VRAM, <5s, <500 MB download), rank the approaches. My prior: "tiny
   diffusion on synthetic data" or "direct pix2pix regression" are the
   only paths that actually hit <500 MB model size. LCM-LoRA / progressive
   distillation keep the base model large. Challenge this if wrong.

3. **Synthetic data generation plan.** If the recommended path uses
   teacher-generated synthetic data:
   - How many tiles? (10k? 50k? 100k?)
   - At what resolution? (256? 512? 1024 downscaled?)
   - How to ensure diversity? (vary cond inputs, vary seeds, vary
     guidance scale, add noise augmentation to conds?)
   - Storage and generation cost estimate (each tile = ~30s on A100)
   - Do we generate from all style ControlNets or just US suburbs for v1?

4. **Student architecture.** If recommending a custom small model:
   - Architecture choice: small U-Net vs small DiT vs ConvNeXt vs
     EfficientViT vs MobileNet-style
   - Input/output: same 3-ch RGB cond → 3-ch RGB road, or switch to
     direct 7-ch → 5-ch (skip the palette encoding since we own both
     ends)?
   - Loss: MSE? Perceptual (LPIPS on road maps)? Adversarial?
     Consistency? Some combination?
   - Resolution: train at 256×256 (smallest useful neighborhood) or
     512×512? Multi-resolution?

5. **Inference runtime.** For the recommended student:
   - Can it run via ONNX Runtime on CPU? (M3 MacBook has no CUDA)
   - CoreML export for Apple Silicon?
   - WebGPU / WebAssembly for browser-based inference?
   - What's the packaging: pip-installable Python library? Electron
     app? Web app? CLI tool?

6. **Quality-vs-size Pareto frontier.** Sketch what you'd expect at each
   model size tier:
   - 50 MB: what's achievable?
   - 200 MB: what's achievable?
   - 500 MB: what's achievable?
   - 2 GB: what's achievable? (this is the "laptop with 8 GB RAM" tier)

7. **Honest failure modes.** What's most likely to not work? E.g.:
   - Distilled model loses road connectivity (the thing we fought for
     4 months on the teacher side)
   - Small models can't do multi-style (need per-style students)
   - CPU inference is too slow even at 256×256
   - Quality drops below "useful" threshold

Output: write findings to `docs/distillation_plan.md`, commit to branch
`jalen/distillation-plan`. Don't start training — produce the plan first.

For context on what "didn't work" at small model sizes: the postmortem
documents our 3-month experience with 5-19M parameter custom models.
They produced honeycomb noise, not road structure. But those were trained
from scratch on 2,500 real tiles. A student trained on 50k+ synthetic
tiles from a good teacher is a fundamentally different regime — the
synthetic data IS the teacher's prior, baked into examples. The student
doesn't need to learn spatial structure from scratch; it just needs to
learn to reproduce the teacher's specific input→output mapping. That's
a much easier task and may succeed at model sizes where from-scratch
training failed. Test this assumption explicitly.
