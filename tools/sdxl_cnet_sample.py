#!/usr/bin/env python3
"""Quick sampler for SDXL ControlNet checkpoints. Loads the trained
controlnet, runs inference on val cond images, saves a side-by-side
PNG of cond | generated.

Usage:
    python tools/sdxl_cnet_sample.py \\
        --controlnet /scratch/jalenj4/runs/sdxl_cnet_v1/checkpoint-1000/controlnet \\
        --val-dir data/flux_cnet_val \\
        --out samples/sdxl_cnet_e1000.png
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
)
from PIL import Image


PROMPT = (
    "top-down satellite-style raster of a US suburban road network, "
    "high-contrast color-coded road class map, flat color, vector style, "
    "no texture, no shading"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--controlnet", required=True, help="Path to checkpoint-N/controlnet")
    p.add_argument("--base", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--vae", default="madebyollin/sdxl-vae-fp16-fix")
    p.add_argument("--val-dir", default="data/flux_cnet_val")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--cnet-scale", type=float, default=1.0)
    args = p.parse_args()

    device = torch.device("cuda")
    print(f"Loading controlnet from {args.controlnet}")
    controlnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=torch.float16)
    print(f"Loading SDXL base {args.base}")
    vae = AutoencoderKL.from_pretrained(args.vae, torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        args.base, controlnet=controlnet, vae=vae, torch_dtype=torch.float16
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    val_files = sorted(f for f in os.listdir(args.val_dir) if f.endswith(".png"))
    if not val_files:
        raise SystemExit(f"No PNGs in {args.val_dir}")
    print(f"Generating {len(val_files)} samples...")

    fig, axes = plt.subplots(len(val_files), 2, figsize=(8, 4 * len(val_files)))
    if len(val_files) == 1:
        axes = axes.reshape(1, -1)

    for i, f in enumerate(val_files):
        cond = Image.open(os.path.join(args.val_dir, f)).convert("RGB")
        gen = pipe(
            prompt=PROMPT,
            image=cond,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            controlnet_conditioning_scale=args.cnet_scale,
        ).images[0]
        axes[i][0].imshow(cond)
        axes[i][0].set_title(f"cond ({f})")
        axes[i][0].axis("off")
        axes[i][1].imshow(gen)
        axes[i][1].set_title(f"generated")
        axes[i][1].axis("off")
        print(f"  [{i+1}/{len(val_files)}] {f}")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=100, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
