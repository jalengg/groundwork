import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalDetailsEnhancement(nn.Module):
    """CaRoLS Section 3.2.1.
    Inputs: R_c^i (cond), R_down^i (noise-stream skip at level i), R_up^(i+1) (prev CDB output, or None at deepest level)
    Output: R_l^i
    """
    def __init__(self, channels, cond_channels, is_deepest=False):
        super().__init__()
        self.is_deepest = is_deepest
        self.cond_proj = nn.Sequential(nn.Conv2d(cond_channels, channels, 1), nn.SiLU())
        self.skip_proj = nn.Conv2d(channels, channels, 1)
        self.fuse = nn.Conv2d(channels * 2, channels, 3, padding=1)
        # Only used when not deepest — concat with R_up^(i+1) and integrate
        if not is_deepest:
            self.up_fuse = nn.Conv2d(channels * 2, channels, 3, padding=1)
        self.norm = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()

    def forward(self, R_down, R_up, R_c):
        is_unconditional = R_c.abs().sum() == 0
        if not is_unconditional:
            cond_feat = self.cond_proj(R_c)
            if cond_feat.shape[-2:] != R_down.shape[-2:]:
                cond_feat = F.interpolate(cond_feat, size=R_down.shape[-2:], mode="bilinear", align_corners=False)
            skip_feat = self.skip_proj(R_down)
            fused = self.fuse(torch.cat([cond_feat, skip_feat], dim=1))
            R_down = R_down + fused

        if self.is_deepest or R_up is None:
            # Per paper: at deepest level (i=3), skip the concat-with-R_up step;
            # apply only Swish + GN.
            return self.act(self.norm(R_down))
        # Non-deepest: concat with previous CDB's R_up (at this level's resolution)
        if R_up.shape[-2:] != R_down.shape[-2:]:
            R_up = F.interpolate(R_up, size=R_down.shape[-2:], mode="nearest")
        combined = torch.cat([R_down, R_up], dim=1)
        return self.act(self.norm(self.up_fuse(combined)))


class GlobalContextIntegration(nn.Module):
    """CaRoLS Section 3.2.2.
    R_c → 1x1+Swish → A;  R_l → 1x1 → B
    fused = 3x3conv(concat([A, B]))    ← K, V come from fused (not just R_c)
    Q from R_l. Residual to R_l.
    """
    def __init__(self, channels, cond_channels):
        super().__init__()
        self.cond_proj = nn.Sequential(nn.Conv2d(cond_channels, channels, 1), nn.SiLU())
        self.local_proj = nn.Conv2d(channels, channels, 1)
        self.fuse = nn.Conv2d(channels * 2, channels, 3, padding=1)
        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        self.out = nn.Linear(channels, channels)
        self.scale = channels ** -0.5

    def forward(self, R_l, R_c):
        Bsz, C, H, W = R_l.shape
        is_unconditional = R_c.abs().sum() == 0
        if is_unconditional:
            # Per paper, fusion step is skipped when R_c is zero — fall back to R_l only
            fused = self.local_proj(R_l)
        else:
            if R_c.shape[-2:] != R_l.shape[-2:]:
                R_c = F.interpolate(R_c, size=R_l.shape[-2:], mode="bilinear", align_corners=False)
            A = self.cond_proj(R_c)
            B_feat = self.local_proj(R_l)
            fused = self.fuse(torch.cat([A, B_feat], dim=1))

        Q = self.q_proj(R_l).flatten(2).transpose(1, 2)          # (B, HW, C)
        K = self.k_proj(fused).flatten(2).transpose(1, 2)
        V = self.v_proj(fused).flatten(2).transpose(1, 2)

        attn = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1)
        S = attn @ V                                              # (B, HW, C)
        S = self.out(S).transpose(1, 2).reshape(Bsz, C, H, W)
        return R_l + S


class LocalAdaptation(nn.Module):
    """DRoLaS Section 3.1, Eq. 3-4 — SFT/FiLM-style affine modulation.

        γ_i = conv_γ(f_i),  δ_i = conv_δ(f_i)
        ĝ_i = γ_i ⊙ g_i + δ_i + g_i

    Per-pixel scale and bias from the conditioning feature, no spatial blending.
    Drop-in replacement for LDE inside the CDB. Same I/O signature as
    LocalDetailsEnhancement so unet.py wiring is unchanged.

    R_up integration (when not deepest) uses an additive 1x1-projected residual,
    keeping the modulation step pure (no concat-fusion mixing).
    """
    def __init__(self, channels, cond_channels, is_deepest=False):
        super().__init__()
        self.is_deepest = is_deepest
        self.gamma_conv = nn.Conv2d(cond_channels, channels, 3, padding=1)
        self.delta_conv = nn.Conv2d(cond_channels, channels, 3, padding=1)
        # Zero-init delta so initial behavior = identity modulation (γ≈0 means
        # ĝ = g — model can learn its way into using cond gradually).
        nn.init.zeros_(self.gamma_conv.weight)
        nn.init.zeros_(self.gamma_conv.bias)
        nn.init.zeros_(self.delta_conv.weight)
        nn.init.zeros_(self.delta_conv.bias)
        if not is_deepest:
            self.up_proj = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()

    def forward(self, R_down, R_up, R_c):
        is_unconditional = R_c.abs().sum() == 0
        if is_unconditional:
            modulated = R_down
        else:
            if R_c.shape[-2:] != R_down.shape[-2:]:
                R_c = F.interpolate(R_c, size=R_down.shape[-2:], mode="bilinear", align_corners=False)
            gamma = self.gamma_conv(R_c)
            delta = self.delta_conv(R_c)
            modulated = gamma * R_down + delta + R_down

        if self.is_deepest or R_up is None:
            return self.act(self.norm(modulated))
        if R_up.shape[-2:] != R_down.shape[-2:]:
            R_up = F.interpolate(R_up, size=R_down.shape[-2:], mode="nearest")
        return self.act(self.norm(modulated + self.up_proj(R_up)))


class ConditionAwareDecoderBlock(nn.Module):
    """LDE → GCI. Returns (R_l, R_g) so the U-Net can wire the paper's
    additive residual R^i = R_up + proj(R_l) (issue #1, Plan A).

    local_module: 'lde' (CaRoLS concat-fusion) or 'load' (DRoLaS SFT/FiLM
    affine modulation). 'load' is the higher-impact lever per DRoLaS Table 2
    (-9 FID alone vs -0.3 for cross-attention).
    """
    def __init__(self, channels, cond_channels, is_deepest=False, local_module="lde"):
        super().__init__()
        if local_module == "lde":
            self.local = LocalDetailsEnhancement(channels, cond_channels, is_deepest=is_deepest)
        elif local_module == "load":
            self.local = LocalAdaptation(channels, cond_channels, is_deepest=is_deepest)
        else:
            raise ValueError(f"Unknown local_module: {local_module!r}; expected 'lde' or 'load'")
        self.gci = GlobalContextIntegration(channels, cond_channels)

    def forward(self, R_down, R_up, R_c):
        R_l = self.local(R_down, R_up, R_c)
        R_g = self.gci(R_l, R_c)
        return R_l, R_g
