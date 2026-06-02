#!/usr/bin/env python3
"""Initialize a 5-channel ControlNet from SD 1.5 with selective weight transfer.

The ControlNet is initialized from a standard SD 1.5 UNet, which has a 3-channel
conditioning input. We replace the first convolution layer to accept 5 channels:
    Ch 0-2 (RGB): terrain + landuse (copied from pretrained weights)
    Ch 3 (A): road skeleton multi-level (randomly initialized)
    Ch 4 (Z): buildable zone mask (randomly initialized)

This allows the model to warm-start on terrain/landuse understanding from
SD 1.5 pretraining while learning new road and zone channels from scratch.

Usage:
    python tools/init_sd15_controlnet.py \\
        --base-model runwayml/stable-diffusion-v1-5 \\
        --out /scratch/jalenj4/runs/sd15_controlnet_init
"""
import argparse
import torch
from diffusers import ControlNetModel, UNet2DConditionModel


def main():
    p = argparse.ArgumentParser(
        description="Initialize 5-ch SD 1.5 ControlNet with selective weight transfer."
    )
    p.add_argument(
        "--base-model",
        default="runwayml/stable-diffusion-v1-5",
        help="HF model ID for SD 1.5 base (used to load UNet).",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output directory to save initialized ControlNet.",
    )
    args = p.parse_args()

    print(f"Loading SD 1.5 UNet from {args.base_model}...")
    unet = UNet2DConditionModel.from_pretrained(
        args.base_model, subfolder="unet", torch_dtype=torch.float32
    )

    print("Creating ControlNet from UNet...")
    controlnet = ControlNetModel.from_unet(unet)

    # Get the original 3-channel input convolution
    old_conv = controlnet.controlnet_cond_embedding.conv_in
    out_channels = old_conv.out_channels
    kernel_size = old_conv.kernel_size
    padding = old_conv.padding

    print(f"Original conv_in shape: {old_conv.weight.shape}")
    print(f"  in_channels=3, out_channels={out_channels}, kernel={kernel_size}, padding={padding}")

    # Create new 5-channel convolution
    new_conv = torch.nn.Conv2d(
        in_channels=5,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=padding,
    )

    # Copy weights for RGB channels (0-2) from pretrained
    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = old_conv.weight.clone()
        # Randomly initialize new channels (3-4) with small std (0.02, typical for transformers)
        torch.nn.init.normal_(new_conv.weight[:, 3:, :, :], mean=0.0, std=0.02)
        # Copy bias unchanged
        new_conv.bias = old_conv.bias.clone() if old_conv.bias is not None else None

    print(f"New conv_in shape: {new_conv.weight.shape}")
    print(f"  in_channels=5, out_channels={out_channels}, kernel={kernel_size}, padding={padding}")

    # Replace the layer in the ControlNet
    controlnet.controlnet_cond_embedding.conv_in = new_conv

    # Update config to reflect 5-channel conditioning
    controlnet.config.conditioning_channels = 5

    print(f"Saving initialized ControlNet to {args.out}...")
    controlnet.save_pretrained(args.out)

    print("Done. ControlNet saved with:")
    print(f"  - RGB channels (0-2): weights transferred from SD 1.5 UNet")
    print(f"  - Road skeleton (ch 3): randomly initialized (N(0, 0.02))")
    print(f"  - Buildable zone (ch 4): randomly initialized (N(0, 0.02))")


if __name__ == "__main__":
    main()
