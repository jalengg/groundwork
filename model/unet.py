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


class _CondEncBlock(nn.Module):
    """Per CaRoLS noise-encoder spec: stride-2 3x3 conv → GroupNorm → Swish → 3x3 conv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.norm = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x):
        return self.conv2(self.act(self.norm(self.conv1(x))))


class ConditionEncoder(nn.Module):
    """Stem: 3×3 conv → BatchNorm → ReLU → R_c ∈ R^(H×W×base_ch).
    Then 4 encoder blocks producing {R_c^0, R_c^1, R_c^2, R_c^3}."""
    def __init__(self, in_ch=7, base_ch=64):
        super().__init__()
        # Paper spec: 3x3 conv + ReLU + BatchNorm. Order from paper: conv, ReLU, BN.
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(base_ch),
        )
        self.enc1 = _CondEncBlock(base_ch, base_ch)              # 512→256
        self.enc2 = _CondEncBlock(base_ch, base_ch * 2)          # 256→128
        self.enc3 = _CondEncBlock(base_ch * 2, base_ch * 4)      # 128→64
        self.enc4 = _CondEncBlock(base_ch * 4, base_ch * 4)      # 64→32

    def forward(self, c):
        h0 = self.stem(c)
        h1 = self.enc1(h0)
        h2 = self.enc2(h1)
        h3 = self.enc3(h2)
        h4 = self.enc4(h3)
        return h1, h2, h3, h4


class DiffusionUNet(nn.Module):
    def __init__(self, latent_channels=4, cond_channels=7, base_ch=64, t_dim=256, local_module="lde"):
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

        # CDBs — local_module='lde' (CaRoLS concat-fusion) or 'load' (DRoLaS SFT). is_deepest=True for cdb1.
        self.cdb1 = ConditionAwareDecoderBlock(base_ch * 4, base_ch * 4, is_deepest=True, local_module=local_module)
        self.cdb2 = ConditionAwareDecoderBlock(base_ch * 4, base_ch * 4, local_module=local_module)
        self.cdb3 = ConditionAwareDecoderBlock(base_ch * 2, base_ch * 2, local_module=local_module)
        self.cdb4 = ConditionAwareDecoderBlock(base_ch, base_ch, local_module=local_module)

        # Standard decoder UNetBlocks (with self-attn + residual + upsample) — outside CDB.
        self.dec1 = UNetBlock(base_ch * 4 * 2, base_ch * 4, t_dim, stride=2, upsample=True)  # 4→8
        self.dec2 = UNetBlock(base_ch * 4 * 2, base_ch * 2, t_dim, stride=2, upsample=True)  # 8→16
        self.dec3 = UNetBlock(base_ch * 2 * 2, base_ch, t_dim, stride=2, upsample=True)       # 16→32
        self.dec4 = UNetBlock(base_ch * 2, base_ch, t_dim, stride=2, upsample=True)           # 32→64

        # Plan A (issue #1): R_l projected to dec_i's output ch/res and added as residual.
        # Zero-init the conv so the new path starts at zero and the model is initially
        # identical to the prior architecture, then learns its way into using R_l.
        self.up_proj1 = self._make_up_proj(base_ch * 4, base_ch * 4)  # res 4→8
        self.up_proj2 = self._make_up_proj(base_ch * 4, base_ch * 2)  # res 8→16
        self.up_proj3 = self._make_up_proj(base_ch * 2, base_ch)      # res 16→32
        self.up_proj4 = self._make_up_proj(base_ch, base_ch)          # res 32→64

        self.out = nn.Conv2d(base_ch, latent_channels, 1)

    @staticmethod
    def _make_up_proj(in_ch, out_ch):
        conv = nn.Conv2d(in_ch, out_ch, 1)
        nn.init.zeros_(conv.weight)
        nn.init.zeros_(conv.bias)
        return nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), conv)

    def forward(self, x, t, cond):
        t_emb = self.t_mlp(sinusoidal_embedding(t, self.t_dim))
        R_c = self.cond_enc(cond)

        e1 = self.noise_enc1(x, t_emb)
        e2 = self.noise_enc2(e1, t_emb)
        e3 = self.noise_enc3(e2, t_emb)
        e4 = self.noise_enc4(e3, t_emb)
        b = self.bottleneck(e4, t_emb)

        # cdb1 deepest: R_up=None (paper i=3 special case).
        # Each level: d_i = dec_i(cat[R_g, e_i], t) + up_proj_i(R_l)  — paper §3.2 R^i = R_up + R_l.
        R_l1, R_g1 = self.cdb1(e4, None, R_c[3])
        d1 = self.dec1(torch.cat([R_g1, e4], 1), t_emb) + self.up_proj1(R_l1)

        R_l2, R_g2 = self.cdb2(e3, F.interpolate(d1, size=e3.shape[-2:]), R_c[2])
        d2 = self.dec2(torch.cat([R_g2, e3], 1), t_emb) + self.up_proj2(R_l2)

        R_l3, R_g3 = self.cdb3(e2, F.interpolate(d2, size=e2.shape[-2:]), R_c[1])
        d3 = self.dec3(torch.cat([R_g3, e2], 1), t_emb) + self.up_proj3(R_l3)

        R_l4, R_g4 = self.cdb4(e1, F.interpolate(d3, size=e1.shape[-2:]), R_c[0])
        d4 = self.dec4(torch.cat([R_g4, e1], 1), t_emb) + self.up_proj4(R_l4)
        return self.out(d4)
