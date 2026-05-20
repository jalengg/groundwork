"""Higher-capacity VAE per council recommendation.

Custom from-scratch v1 VAE (5M params, 1 ResBlock per stage, base_ch=64) hit a
0.68 road-pixel-masked IoU ceiling on val per the oracle decoder diagnostic.
That cap propagates: any diffusion sample we generate has to land near a real
encoded latent and then survive decode — and the decoder loses ~30% of road
structure even on the trained-against distribution.

v2 changes (council Agent 1):
  - **Deeper**: 2 ResBlocks per stage (vs 1)
  - **Wider**: base_ch = 96 (vs 64)
  - **Bottleneck self-attention**: one attention block at the deepest
    resolution, helps long-range road structure
  - Same f8 latent shape (4×64×64) — drop-in for diffusion U-Net
  - Same 5-channel one-hot in/out

Target: ~14M params (3× v1). Training recipe is `vae_loss_v2` (CE + Dice + KL,
KL weight bumped to 1e-3 per council).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Two-conv residual block with GN+SiLU."""
    def __init__(self, in_ch, out_ch, stride=1, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.stride = stride
        conv_cls = nn.ConvTranspose2d if upsample else nn.Conv2d
        extra = {"output_padding": stride - 1} if upsample else {}
        self.conv1 = conv_cls(in_ch, out_ch, 3, stride=stride, padding=1, **extra)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        skip = self.skip(x)
        if self.upsample:
            skip = F.interpolate(skip, scale_factor=2)
        elif self.stride > 1:
            skip = F.avg_pool2d(skip, self.stride)
        return self.act(h + skip)


class SelfAttention2d(nn.Module):
    """Single self-attention block over flattened spatial tokens, used at
    the bottleneck for long-range structure (helps connecting roads across
    the tile)."""
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).flatten(2).transpose(1, 2)
        h, _ = self.attn(h, h, h)
        h = h.transpose(1, 2).reshape(B, C, H, W)
        return x + h


class VAEEncoderV2(nn.Module):
    def __init__(self, in_channels=5, latent_channels=4, base_ch=96):
        super().__init__()
        self.in_proj = nn.Conv2d(in_channels, base_ch, 3, padding=1)
        # 3 stages, each: 2 ResBlocks (one downsamples). Going 512 -> 256 -> 128 -> 64.
        self.stage1 = nn.Sequential(
            ResBlock(base_ch, base_ch, stride=2),                # 512 -> 256
            ResBlock(base_ch, base_ch),
        )
        self.stage2 = nn.Sequential(
            ResBlock(base_ch, base_ch * 2, stride=2),            # 256 -> 128
            ResBlock(base_ch * 2, base_ch * 2),
        )
        self.stage3 = nn.Sequential(
            ResBlock(base_ch * 2, base_ch * 4, stride=2),        # 128 -> 64
            ResBlock(base_ch * 4, base_ch * 4),
        )
        self.attn = SelfAttention2d(base_ch * 4, num_heads=4)
        self.to_mu = nn.Conv2d(base_ch * 4, latent_channels, 1)
        self.to_logvar = nn.Conv2d(base_ch * 4, latent_channels, 1)

    def forward(self, x):
        h = self.in_proj(x)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.attn(h)
        return self.to_mu(h), self.to_logvar(h)


class VAEDecoderV2(nn.Module):
    def __init__(self, latent_channels=4, out_channels=5, base_ch=96):
        super().__init__()
        self.in_proj = nn.Conv2d(latent_channels, base_ch * 4, 3, padding=1)
        self.attn = SelfAttention2d(base_ch * 4, num_heads=4)
        self.stage1 = nn.Sequential(
            ResBlock(base_ch * 4, base_ch * 4),
            ResBlock(base_ch * 4, base_ch * 2, stride=2, upsample=True),  # 64 -> 128
        )
        self.stage2 = nn.Sequential(
            ResBlock(base_ch * 2, base_ch * 2),
            ResBlock(base_ch * 2, base_ch, stride=2, upsample=True),      # 128 -> 256
        )
        self.stage3 = nn.Sequential(
            ResBlock(base_ch, base_ch),
            ResBlock(base_ch, base_ch, stride=2, upsample=True),          # 256 -> 512
        )
        self.out_proj = nn.Conv2d(base_ch, out_channels, 3, padding=1)

    def forward(self, z):
        h = self.in_proj(z)
        h = self.attn(h)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        return self.out_proj(h)


class RoadVAEv2(nn.Module):
    """API-compatible drop-in for `model.vae.RoadVAE` with ~3× the capacity
    and bottleneck attention. (B,5,512,512) -> (B,latent_channels,64,64).

    latent_channels defaults to 4 (matches CaRoLS/SDXL convention). Bump to
    8 or 16 (SD3/Flux convention) for ~2-4× latent capacity — necessary if
    the 4×64×64 information bottleneck is what caps reconstruction IoU.
    """

    def __init__(self, base_ch=96, latent_channels=4):
        super().__init__()
        self.latent_channels = latent_channels
        self.encoder = VAEEncoderV2(base_ch=base_ch, latent_channels=latent_channels)
        self.decoder = VAEDecoderV2(base_ch=base_ch, latent_channels=latent_channels)

    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
