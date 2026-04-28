import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalDetailsEnhancement(nn.Module):
    def __init__(self, channels, cond_channels):
        super().__init__()
        self.cond_proj = nn.Sequential(nn.Conv2d(cond_channels, channels, 1), nn.SiLU())
        self.skip_proj = nn.Conv2d(channels, channels, 1)
        self.fuse = nn.Conv2d(channels * 2, channels, 3, padding=1)
        self.up_fuse = nn.Conv2d(channels * 2, channels, 3, padding=1)
        self.norm = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()

    def forward(self, R_down, R_up, R_c):
        is_unconditional = R_c.abs().sum() == 0
        if not is_unconditional:
            cond_feat = self.cond_proj(R_c)
            # Align R_c spatial size to R_down if they differ (cross-resolution CDB)
            if cond_feat.shape[-2:] != R_down.shape[-2:]:
                cond_feat = F.interpolate(cond_feat, size=R_down.shape[-2:], mode="bilinear", align_corners=False)
            skip_feat = self.skip_proj(R_down)
            fused = self.fuse(torch.cat([cond_feat, skip_feat], dim=1))
            R_down = R_down + fused
        combined = torch.cat([R_down, R_up], dim=1)
        return self.act(self.norm(self.up_fuse(combined)))


class GlobalContextIntegration(nn.Module):
    def __init__(self, channels, cond_channels):
        super().__init__()
        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(cond_channels, channels, 1)
        self.v_proj = nn.Conv2d(cond_channels, channels, 1)
        self.out = nn.Linear(channels, channels)
        self.scale = channels ** -0.5

    def forward(self, R_l, R_c):
        B, C, H, W = R_l.shape
        is_unconditional = R_c.abs().sum() == 0
        kv_source = R_l if is_unconditional else R_c

        Q = self.q_proj(R_l).flatten(2).transpose(1, 2)          # (B, HW, C)
        K = self.k_proj(kv_source).flatten(2).transpose(1, 2)
        V = self.v_proj(kv_source).flatten(2).transpose(1, 2)

        attn = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
        S = attn @ V                                               # (B, HW, C)
        S = self.out(S).transpose(1, 2).reshape(B, C, H, W)
        return R_l + S


class ConditionAwareDecoderBlock(nn.Module):
    def __init__(self, channels, cond_channels):
        super().__init__()
        self.lde = LocalDetailsEnhancement(channels, cond_channels)
        self.gci = GlobalContextIntegration(channels, cond_channels)

    def forward(self, R_down, R_up, R_c):
        R_l = self.lde(R_down, R_up, R_c)
        return self.gci(R_l, R_c)
