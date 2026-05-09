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

    Per-channel scale calibration: SDXL's `scaling_factor=0.13025` was
    calibrated on photographic LAION inputs. On our palette-encoded
    inputs (5-color cube corners), the per-channel `mu` distribution is
    different. Calling `calibrate(loader)` once over the training set
    computes empirical per-channel mean+std and stores them as
    `self.shift, self.scale` so the latent fed to the diffusion model
    has near-zero mean and unit std per-channel — required for the
    standard DDPM `α_t` schedule to be correctly calibrated.
    """

    SDXL_SCALING = 0.13025  # legacy default; overridden by .calibrate()

    def __init__(self, model_id="madebyollin/sdxl-vae-fp16-fix", dtype=torch.float32):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(model_id, torch_dtype=dtype)
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.vae.eval()
        # Latent normalization. Default to SDXL's photographic calibration;
        # call .calibrate() to override with our actual training distribution.
        self.register_buffer("shift", torch.zeros(1, 4, 1, 1))
        self.register_buffer("scale", torch.full((1, 4, 1, 1), 1.0 / self.SDXL_SCALING))

    @torch.no_grad()
    def calibrate(self, road_loader, max_batches=None, verbose=True):
        """Estimate per-channel shift/scale from the training distribution.
        Pass a DataLoader yielding (cond, road) tuples (we ignore cond here)."""
        self.eval()
        means, sq_means, n = None, None, 0
        for i, batch in enumerate(road_loader):
            if max_batches is not None and i >= max_batches:
                break
            road = batch[1] if isinstance(batch, (tuple, list)) else batch
            road = road.to(next(self.vae.parameters()).device)
            rgb = onehot_to_rgb_palette(road) * 2 - 1
            mu = self.vae.encode(rgb).latent_dist.mean                  # (B, 4, h, w) raw
            B = mu.shape[0]
            n += B
            cur_mean = mu.mean(dim=(0, 2, 3))                            # (4,)
            cur_sq = mu.pow(2).mean(dim=(0, 2, 3))                       # (4,)
            if means is None:
                means, sq_means = cur_mean * B, cur_sq * B
            else:
                means += cur_mean * B
                sq_means += cur_sq * B
        means /= n
        sq_means /= n
        stds = (sq_means - means.pow(2)).clamp(min=1e-8).sqrt()
        self.shift.copy_(means.view(1, 4, 1, 1))
        self.scale.copy_(stds.view(1, 4, 1, 1))
        if verbose:
            print(f"[VAE calibrate] over {n} samples:")
            print(f"  per-channel mean: {means.tolist()}")
            print(f"  per-channel std:  {stds.tolist()}")
            print(f"  (vs SDXL default scaling 1/{self.SDXL_SCALING:.5f} = {1/self.SDXL_SCALING:.4f})")

    # ---------- API parity with RoadVAE ----------
    def encode(self, road_oh):
        """5-ch one-hot road -> (mu, logvar) in normalized latent space."""
        rgb = onehot_to_rgb_palette(road_oh) * 2 - 1                    # SDXL expects [-1, 1]
        with torch.no_grad():
            posterior = self.vae.encode(rgb).latent_dist
        mu_raw, std_raw = posterior.mean, posterior.std
        mu = (mu_raw - self.shift) / self.scale
        logvar = torch.log((std_raw / self.scale).pow(2) + 1e-12)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        """normalized latent -> 5-ch one-hot road logits."""
        z_unscaled = z * self.scale + self.shift                        # invert encode normalization
        with torch.no_grad():
            rgb_out = self.vae.decode(z_unscaled).sample                # (B, 3, H, W) in [-1, 1]
        rgb_out = ((rgb_out + 1) / 2).clamp(0, 1)
        return rgb_to_onehot_palette(rgb_out, return_logits=True)

    def forward(self, road_oh):
        mu, logvar = self.encode(road_oh)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
