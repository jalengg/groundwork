#!/usr/bin/env python3
"""Upload the trained ControlNet to HuggingFace Hub.

Why: cluster `/scratch` is periodically cleaned. Without preservation, the
~3 GB of trained ControlNet weights at `/scratch/jalenj4/runs/sdxl_cnet_v1/`
would be lost. HuggingFace Hub is the standard storage for diffusers
models — free for public repos, integrates with `ControlNetModel.from_pretrained()`
out of the box, supports model cards + versioning.

Usage:
    # one-time: get a write token from https://huggingface.co/settings/tokens
    huggingface-cli login --token <YOUR_HF_WRITE_TOKEN>

    # then upload
    python tools/upload_to_hf.py \\
        --src /scratch/jalenj4/runs/sdxl_cnet_v1 \\
        --repo jalengg/groundwork-sdxl-cnet-us-suburbs

After upload, the model is loadable with:
    from diffusers import ControlNetModel
    cnet = ControlNetModel.from_pretrained("jalengg/groundwork-sdxl-cnet-us-suburbs",
                                           torch_dtype=torch.float16)
"""
import argparse
import json
import pathlib

from huggingface_hub import HfApi, create_repo


MODEL_CARD = """\
---
license: openrail++
base_model: stabilityai/stable-diffusion-xl-base-1.0
tags:
  - controlnet
  - sdxl
  - urban-planning
  - road-network-generation
  - cities-skylines
pipeline_tag: image-to-image
---

# Groundwork: SDXL ControlNet for US Suburban Road Networks

ControlNet adapter for [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
that conditions on a 3-channel **landuse + terrain** raster and generates
a color-coded US suburban **road network** map.

Trained for use in a [Cities Skylines](https://www.citiesskylines.com/) Steam
Workshop mod, where players paint landuse zones and the model generates
plausible road networks respecting those zones.

## Conditioning encoding

3-channel RGB input where each channel encodes specific landuse information:

- **R**: terrain elevation, per-tile normalized to `[0, 1]`
- **G**: `0.3·water + 0.5·parkland + 0.2·agricultural` (natural features)
- **B**: `0.6·residential + 0.8·commercial + 1.0·industrial` (built-up density)

## Output encoding

3-channel RGB output, decoded via nearest-color to a 5-class road palette:

| Class | RGB | Meaning |
|---|---|---|
| Background | `(0, 0, 0)` | non-road |
| Residential | `(255, 0, 0)` | small streets |
| Tertiary | `(0, 255, 0)` | minor arterials |
| Primary | `(0, 0, 255)` | major arterials |
| Motorway | `(255, 255, 0)` | highways |

## Usage

```python
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from PIL import Image

controlnet = ControlNetModel.from_pretrained(
    "jalengg/groundwork-sdxl-cnet-us-suburbs",
    torch_dtype=torch.float16,
)
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16,
)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
).to("cuda")

cond = Image.open("your_landuse_raster.png").convert("RGB")  # 1024x1024 RGB
result = pipe(
    prompt="top-down satellite-style raster of a US suburban road network, "
           "high-contrast color-coded road class map, flat color, vector style",
    image=cond,
    num_inference_steps=30,
    guidance_scale=5.0,
    controlnet_conditioning_scale=1.0,
).images[0]
result.save("road_network.png")
```

## Training data

- **2,443 paired tiles** at 1024×1024 RGB
- **17 US Sun Belt cities** (Arlington TX, Chandler AZ, Mesa AZ, Henderson NV,
  Plano TX, etc.); `irving_tx` held out for qualitative validation
- Per-tile coverage: **2.56 km × 2.56 km** at 5 m/pixel
- Source: OpenStreetMap road vectors + SRTM elevation + OSM `landuse=*`
  polygons, rasterized via the project's `data_pipeline/` module

## Training recipe

- Base model: `stabilityai/stable-diffusion-xl-base-1.0` (frozen)
- VAE: `madebyollin/sdxl-vae-fp16-fix` (frozen)
- Optimizer: 8-bit AdamW
- Learning rate: `1e-5`, constant with 500-step warmup
- Effective batch size: 8 (`bs=1 × grad_accum=8`)
- Mixed precision: fp16
- Training steps: 25,000
- Compute: ~35 hours on a single A100 80 GB

## Limitations

- **Trained only on US Sun Belt suburbs** (Texas, Arizona, Nevada, etc.).
  Will not produce European medieval centers, rural villages, Asian
  dense urban, etc. — those styles are out of distribution.
- Output is a **stylized 5-class color-coded map**, not a real road graph.
  Downstream postprocessing (skeletonize → vectorize → snap to grid) is
  required to consume the output as a network in a downstream application.
- Constant prompt was used during training — the text encoder contributes
  little; all class differentiation comes through the ControlNet input.

## License

OpenRAIL++ (inherits from SDXL base license).

## Citation / acknowledgement

Project repository: [github.com/jalengg/groundwork](https://github.com/jalengg/groundwork)

Built on top of:
- Stable Diffusion XL (Stability AI, 2023)
- ControlNet (Zhang & Agrawala, 2023)
- The diffusers library (HuggingFace)
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Local dir with config.json + diffusion_pytorch_model.safetensors")
    p.add_argument("--repo", required=True, help="HF repo id, e.g. jalengg/groundwork-sdxl-cnet-us-suburbs")
    p.add_argument("--private", action="store_true", help="Make repo private (default: public)")
    args = p.parse_args()

    src = pathlib.Path(args.src)
    required = ["config.json", "diffusion_pytorch_model.safetensors"]
    missing = [r for r in required if not (src / r).exists()]
    if missing:
        raise SystemExit(f"Missing required files in {src}: {missing}")

    print(f"Creating repo {args.repo} (private={args.private})...")
    create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)

    # Write model card alongside the weights
    card_path = src / "README.md"
    card_path.write_text(MODEL_CARD)
    print(f"Wrote model card to {card_path}")

    api = HfApi()
    print(f"Uploading {src} to {args.repo}...")
    api.upload_folder(
        folder_path=str(src),
        repo_id=args.repo,
        repo_type="model",
        commit_message="Upload Groundwork SDXL ControlNet v1 (US suburbs, 25k steps)",
        ignore_patterns=["checkpoint-*/**", "logs/**", "*.optimizer.bin", "*.scheduler.bin"],
    )
    print(f"Done. Model is at https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
