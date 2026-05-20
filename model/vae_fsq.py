"""FSQ-quantized VAE (Mentzer et al. 2023, "Finite Scalar Quantization").

Discrete latent space — directly addresses the categorical-output / continuous-
Gaussian-latent mismatch that v1/v2/v3 continuous VAEs all hit at ~0.68
road-pixel IoU. The hypothesis: a discrete codebook latent matches the
discrete nature of our 5-class one-hot output, so the rec loss can drive
each spatial position to a specific code instead of needing to land near a
specific continuous point in a Gaussian-shaped manifold.

Levels follow council Agent 1's recommendation: `[8, 5, 5, 5]` gives 1000
codes per spatial position with 4-channel latent shape (matches our existing
diffusion U-Net latent dim, drop-in replacement for v2/v3 continuous VAEs).

Mentzer's key insight: no codebook, no commitment loss, no codebook collapse.
Each latent dim is independently quantized to a fixed level set in `[-1, 1]`,
with straight-through gradient. Simpler than VQ-VAE, comparable downstream
quality per the original paper's ImageNet ablations.
"""
import torch
import torch.nn as nn

from model.vae_v2 import VAEDecoderV2, VAEEncoderV2


class FSQ(nn.Module):
    """Per-dim quantization to a fixed level set.

    Latent dim `i` is bounded to `[-1, 1]` via `tanh` then snapped to one of
    `levels[i]` evenly-spaced values. Straight-through estimator passes the
    gradient through the rounding so the encoder can still learn.

    For our default `levels = (8, 5, 5, 5)`:
      - Channel 0: 8 levels = `{-1, -5/7, -3/7, -1/7, 1/7, 3/7, 5/7, 1}`
      - Channels 1-3: 5 levels = `{-1, -0.5, 0, 0.5, 1}`
      - Total codes per position: `8 * 5 * 5 * 5 = 1000`
    """

    def __init__(self, levels=(8, 5, 5, 5)):
        super().__init__()
        self.levels = levels
        self.register_buffer("L", torch.tensor(levels, dtype=torch.float32))

    @property
    def num_codes(self):
        n = 1
        for l in self.levels:
            n *= l
        return n

    def forward(self, z):
        """z: (B, len(levels), H, W). Returns quantized z (same shape)
        with straight-through gradient."""
        z = z.tanh()                                              # (-1, 1)
        L = self.L.view(1, -1, 1, 1)
        # Map [-1, 1] -> [0, L-1] -> rounded int -> back to [-1, 1] grid
        normed = (z + 1) * (L - 1) / 2
        rounded = torch.round(normed)
        z_q = rounded * 2 / (L - 1) - 1
        # Straight-through: forward = z_q, backward = identity through z
        return z + (z_q - z).detach()


class RoadVAEFSQ(nn.Module):
    """API-compatible drop-in for `RoadVAE` / `RoadVAEv2` with FSQ-quantized
    latent. Same encode/decode/reparameterize interface so the diffusion
    training code path is unchanged.

    Note: `reparameterize` returns `mu` unchanged — the latent is already
    deterministic post-FSQ. `logvar` is a placeholder zero tensor."""

    def __init__(self, base_ch=96, levels=(8, 5, 5, 5)):
        super().__init__()
        self.levels = levels
        latent_channels = len(levels)
        self.latent_channels = latent_channels
        self.encoder = VAEEncoderV2(base_ch=base_ch, latent_channels=latent_channels)
        self.decoder = VAEDecoderV2(base_ch=base_ch, latent_channels=latent_channels)
        self.fsq = FSQ(levels=levels)

    @property
    def num_codes(self):
        return self.fsq.num_codes

    def encode(self, x):
        mu_raw, _ = self.encoder(x)
        mu = self.fsq(mu_raw)
        logvar = torch.zeros_like(mu)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        return mu                                                 # deterministic

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, _ = self.encode(x)
        return self.decode(mu), mu, torch.zeros_like(mu)
