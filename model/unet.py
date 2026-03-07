import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.cdb import ConditionAwareDecoderBlock


def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimestepMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t_emb):
        return self.net(t_emb)


class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, stride=1, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.stride = stride
        conv_cls = nn.ConvTranspose2d if upsample else nn.Conv2d
        extra = {"output_padding": stride - 1} if upsample else {}
        self.conv1 = conv_cls(in_ch, out_ch, 3, stride=stride, padding=1, **extra)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.attn = nn.MultiheadAttention(out_ch, num_heads=4, batch_first=True)
        self.attn_norm = nn.LayerNorm(out_ch)

    def forward(self, x, t_emb):
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.t_proj(t_emb)[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        B, C, H, W = h.shape
        flat = h.flatten(2).transpose(1, 2)
        flat, _ = self.attn(flat, flat, flat)
        flat = self.attn_norm(flat)
        h = h + flat.transpose(1, 2).reshape(B, C, H, W)
        skip = self.skip(x)
        if self.upsample:
            skip = F.interpolate(skip, scale_factor=2)
        elif self.stride > 1:
            skip = F.avg_pool2d(skip, self.stride)
        return h + skip


class ConditionEncoder(nn.Module):
    def __init__(self, in_ch=4, base_ch=64):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.enc1 = nn.Sequential(nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1), nn.SiLU())        # 512→256
        self.enc2 = nn.Sequential(nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1), nn.SiLU())    # 256→128
        self.enc3 = nn.Sequential(nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1), nn.SiLU()) # 128→64
        self.enc4 = nn.Sequential(nn.Conv2d(base_ch * 4, base_ch * 4, 3, stride=2, padding=1), nn.SiLU()) # 64→32

    def forward(self, c):
        h0 = self.stem(c)
        h1 = self.enc1(h0)
        h2 = self.enc2(h1)
        h3 = self.enc3(h2)
        h4 = self.enc4(h3)
        return h1, h2, h3, h4  # R_c at res 256, 128, 64, 32


class DiffusionUNet(nn.Module):
    def __init__(self, latent_channels=4, cond_channels=4, base_ch=64, t_dim=256):
        super().__init__()
        self.t_dim = t_dim
        self.t_mlp = TimestepMLP(t_dim)
        self.cond_enc = ConditionEncoder(cond_channels, base_ch)

        # Noise encoder (on 64×64 latent)
        self.noise_enc1 = UNetBlock(latent_channels, base_ch, t_dim, stride=2)    # 64→32
        self.noise_enc2 = UNetBlock(base_ch, base_ch * 2, t_dim, stride=2)        # 32→16
        self.noise_enc3 = UNetBlock(base_ch * 2, base_ch * 4, t_dim, stride=2)    # 16→8
        self.noise_enc4 = UNetBlock(base_ch * 4, base_ch * 4, t_dim, stride=2)    # 8→4
        self.bottleneck = UNetBlock(base_ch * 4, base_ch * 4, t_dim)

        # CDB decoders
        self.cdb1 = ConditionAwareDecoderBlock(base_ch * 4, base_ch * 4)
        self.cdb2 = ConditionAwareDecoderBlock(base_ch * 4, base_ch * 4)
        self.cdb3 = ConditionAwareDecoderBlock(base_ch * 2, base_ch * 2)
        self.cdb4 = ConditionAwareDecoderBlock(base_ch, base_ch)

        self.dec1 = UNetBlock(base_ch * 4 * 2, base_ch * 4, t_dim, stride=2, upsample=True)  # 4→8
        self.dec2 = UNetBlock(base_ch * 4 * 2, base_ch * 2, t_dim, stride=2, upsample=True)  # 8→16
        self.dec3 = UNetBlock(base_ch * 2 * 2, base_ch, t_dim, stride=2, upsample=True)       # 16→32
        self.dec4 = UNetBlock(base_ch * 2, base_ch, t_dim, stride=2, upsample=True)           # 32→64
        self.out = nn.Conv2d(base_ch, latent_channels, 1)

    def forward(self, x, t, cond):
        t_emb = self.t_mlp(sinusoidal_embedding(t, self.t_dim))
        R_c = self.cond_enc(cond)  # tuple of 4 feature maps

        e1 = self.noise_enc1(x, t_emb)
        e2 = self.noise_enc2(e1, t_emb)
        e3 = self.noise_enc3(e2, t_emb)
        e4 = self.noise_enc4(e3, t_emb)
        b = self.bottleneck(e4, t_emb)

        d1 = self.dec1(torch.cat([self.cdb1(e4, F.interpolate(b, size=e4.shape[-2:]), R_c[3]), e4], 1), t_emb)
        d2 = self.dec2(torch.cat([self.cdb2(e3, F.interpolate(d1, size=e3.shape[-2:]), R_c[2]), e3], 1), t_emb)
        d3 = self.dec3(torch.cat([self.cdb3(e2, F.interpolate(d2, size=e2.shape[-2:]), R_c[1]), e2], 1), t_emb)
        d4 = self.dec4(torch.cat([self.cdb4(e1, F.interpolate(d3, size=e1.shape[-2:]), R_c[0]), e1], 1), t_emb)
        return self.out(d4)
