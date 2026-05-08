"""SDXL pretrained VAE wrapper exposing the same API as `model.vae.RoadVAE`.

Why: our from-scratch VAE was trained on 2,550 road raster tiles; SDXL's VAE
was trained on millions of natural images and preserves edges, parallel lines,
and fine periodicity that our small VAE compresses away. Same latent shape
(`512×512 → 64×64×4` at f8), so we can drop it in without retraining the
diffusion U-Net.

Encode: 5-ch one-hot road -> 3-ch RGB palette -> SDXL VAE -> 4-ch latent
Decode: 4-ch latent -> SDXL VAE -> 3-ch RGB -> nearest-color -> 5-ch one-hot

The SDXL VAE is frozen; only the palette mapping is "trainable" in the sense
that you could pick different colors. We use maximally-separated RGB corners
to keep the nearest-color decoder margin as wide as possible.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL


# Maximally-separated 5-color palette in RGB cube. Order matches our class
# index: 0=bg (black), 1=residential (red), 2=tertiary (green),
# 3=primary (blue), 4=motorway (yellow).
PALETTE_RGB = torch.tensor([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
], dtype=torch.float32)


def onehot_to_rgb_palette(road_oh, palette=None):
    """(B, 5, H, W) one-hot road -> (B, 3, H, W) RGB in [0,1]."""
    if palette is None:
        palette = PALETTE_RGB.to(road_oh.device)
    # einsum: per-pixel weighted sum of palette colors by class probability
    return torch.einsum("bchw,cd->bdhw", road_oh, palette)


def rgb_to_onehot_palette(rgb, palette=None, return_logits=False):
    """(B, 3, H, W) RGB in [0,1] -> (B, 5, H, W) one-hot via nearest-color
    in plain RGB space. Returns one-hot (or signed logits if requested) so
    downstream code can call .argmax(1) like before."""
    if palette is None:
        palette = PALETTE_RGB.to(rgb.device)
    # rgb: (B, 3, H, W); palette: (5, 3)
    # dist[b,c,h,w] = ||rgb[b,:,h,w] - palette[c,:]||²
    diff = rgb.unsqueeze(1) - palette.view(1, -1, 3, 1, 1)             # (B, 5, 3, H, W)
    dist = diff.pow(2).sum(dim=2)                                       # (B, 5, H, W)
    if return_logits:
        return -dist                                                    # closer = higher logit
    cls = dist.argmin(dim=1)                                            # (B, H, W)
    return F.one_hot(cls, num_classes=5).permute(0, 3, 1, 2).float()


class RoadVAESDXL(nn.Module):
    """Drop-in replacement for `model.vae.RoadVAE` using the SDXL pretrained
    VAE under the hood.

    Same latent shape (4×64×64), same encode/decode/reparameterize API.
    The wrapping palette transformations preserve the (B, 5, H, W) one-hot
    interface the diffusion U-Net's conditioning was trained against."""

    SDXL_SCALING = 0.13025  # SDXL VAE scaling factor (built-in attribute too)

    def __init__(self, model_id="madebyollin/sdxl-vae-fp16-fix", dtype=torch.float32):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(model_id, torch_dtype=dtype)
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.vae.eval()
        self.scaling = self.vae.config.scaling_factor                   # 0.13025 for SDXL

    # ---------- API parity with RoadVAE ----------
    def encode(self, road_oh):
        """5-ch one-hot road -> (mu, logvar) in latent space."""
        rgb = onehot_to_rgb_palette(road_oh)                            # (B, 3, H, W) in [0,1]
        rgb = rgb * 2 - 1                                               # SDXL expects [-1, 1]
        with torch.no_grad():
            posterior = self.vae.encode(rgb).latent_dist
        # Apply SDXL scaling so the latent has unit-ish std (per SDXL convention).
        return posterior.mean * self.scaling, torch.log(posterior.std.pow(2) * self.scaling ** 2 + 1e-12)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        """latent -> 5-ch one-hot road (via nearest-color from RGB output)."""
        with torch.no_grad():
            rgb_out = self.vae.decode(z / self.scaling).sample          # (B, 3, H, W) in [-1, 1]
        rgb_out = (rgb_out + 1) / 2                                      # back to [0, 1]
        rgb_out = rgb_out.clamp(0, 1)
        # Return as logits-like (5-ch) so downstream `.argmax(1)` works without changes.
        return rgb_to_onehot_palette(rgb_out, return_logits=True)

    def forward(self, road_oh):
        mu, logvar = self.encode(road_oh)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
