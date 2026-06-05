"""PatchGAN discriminator for VAE-GAN training (Isola et al. 2017,
"Image-to-Image Translation with Conditional Adversarial Networks";
adopted by Rombach et al. 2022 LDM and Podell et al. 2023 SDXL for VAE
training).

The discriminator operates on **RGB-rendered output** (palette mapping is
differentiable, so gradient flows back to the VAE through it). Real road
maps are rendered via the same palette so the discriminator can't trivially
learn "softmax fake vs one-hot real" — both inputs are sharp RGB.

Architecture: 4 stride-2 convs with spectral norm + GroupNorm + LReLU,
final 1x1 conv → patch logits. Output is (B, 1, 32, 32) for 512×512 input
— each spatial position predicts realness of a ~70px receptive-field patch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


# Default palette matches model.vae_sdxl.PALETTE_RGB
DEFAULT_PALETTE = torch.tensor([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
], dtype=torch.float32)


def onehot_logits_to_rgb(logits_or_onehot, palette=None, soft=True):
    """(B, 5, H, W) -> (B, 3, H, W) RGB in [0, 1].

    If `soft=True`, applies softmax over class channel first — used during
    training so gradients flow through. If False, treats input as one-hot
    weights directly (for already-hard ground truth).
    """
    if palette is None:
        palette = DEFAULT_PALETTE.to(logits_or_onehot.device)
    if soft:
        weights = F.softmax(logits_or_onehot, dim=1)
    else:
        weights = logits_or_onehot
    return torch.einsum("bchw,cd->bdhw", weights, palette)


class PatchDiscriminator(nn.Module):
    """PatchGAN with 4 stride-2 downsamples + spectral norm + GroupNorm.

    Input: (B, 3, 512, 512) RGB in [0, 1] (we shift to [-1, 1] internally).
    Output: (B, 1, 32, 32) patch real/fake logits.

    Also exposes `extract_features(x)` returning intermediate activations
    for feature-matching loss (a perceptual signal trained on our domain
    without LPIPS's photo bias).
    """

    def __init__(self, in_channels=3, base_ch=64, n_layers=4):
        super().__init__()
        layers = []
        ch = in_channels
        out = base_ch
        # First layer: no norm
        layers.append(nn.Sequential(
            spectral_norm(nn.Conv2d(ch, out, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
        ))
        ch = out
        for i in range(1, n_layers):
            out = min(base_ch * (2 ** i), 512)
            layers.append(nn.Sequential(
                spectral_norm(nn.Conv2d(ch, out, 4, stride=2, padding=1)),
                nn.GroupNorm(8, out),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            ch = out
        self.layers = nn.ModuleList(layers)
        self.head = spectral_norm(nn.Conv2d(ch, 1, 4, padding=1))

    def forward(self, x):
        h = x * 2 - 1                                            # [0,1] -> [-1,1]
        for layer in self.layers:
            h = layer(h)
        return self.head(h)

    def extract_features(self, x):
        """Return list of intermediate activations for feature-matching."""
        h = x * 2 - 1
        feats = []
        for layer in self.layers:
            h = layer(h)
            feats.append(h)
        return feats
