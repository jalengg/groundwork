import torch
import torch.nn.functional as F


def focal_loss(logits, targets, gamma=2.0, alpha=None):
    """
    logits:  (B, C, H, W) raw logits
    targets: (B, C, H, W) one-hot float
    alpha:   (C,) per-class weight or None
    """
    log_p = F.log_softmax(logits, dim=1)
    p = log_p.exp()
    target_p = (p * targets).sum(dim=1, keepdim=True)  # (B, 1, H, W)
    focal_weight = (1 - target_p) ** gamma
    if alpha is not None:
        alpha_t = (alpha[None, :, None, None] * targets).sum(dim=1, keepdim=True)
        focal_weight = focal_weight * alpha_t
    loss = -(focal_weight * (log_p * targets).sum(dim=1, keepdim=True))
    return loss.mean()


def kl_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def dice_loss(logits, targets, eps=1e-6):
    """Multi-class soft Dice loss. Drives per-class IoU directly — much
    better than focal/CE for thin sparse classes (residential streets)
    where most pixels are bg.

    logits:  (B, C, H, W) raw logits
    targets: (B, C, H, W) one-hot float
    """
    p = F.softmax(logits, dim=1)
    # per-class intersection and union, summed over spatial dims
    intersection = (p * targets).sum(dim=(0, 2, 3))               # (C,)
    union = p.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))     # (C,)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def ce_loss(logits, targets):
    """Per-pixel cross-entropy. logits (B,C,H,W), targets (B,C,H,W) one-hot."""
    log_p = F.log_softmax(logits, dim=1)
    return -(log_p * targets).sum(dim=1).mean()


_PAPER_ALPHA = torch.tensor([0.1, 0.3, 0.6, 1.0, 2.0])


def vae_loss(recon_logits, targets, mu, logvar, gamma=2.0, kl_weight=1e-4):
    """v1 loss: focal + KL. Paper alpha [0.1, 0.3, 0.6, 1.0, 2.0]."""
    alpha = _PAPER_ALPHA.to(recon_logits.device)
    l_focal = focal_loss(recon_logits, targets, gamma=gamma, alpha=alpha)
    l_kl = kl_loss(mu, logvar)
    return l_focal + kl_weight * l_kl


def vae_loss_v2(recon_logits, targets, mu, logvar,
                ce_weight=1.0, dice_weight=1.0, kl_weight=1e-3):
    """v2 loss per council: CE + Dice + KL with KL bumped 10×.

    Drops focal, adds Dice (drives per-class IoU directly — addresses the
    thin-sparse-class IoU gap that focal alone couldn't close). KL weight
    1e-3 (vs v1's 1e-4) regularizes the latent toward N(0,I), which is
    what the diffusion noise schedule assumes.
    """
    l_ce = ce_loss(recon_logits, targets)
    l_dice = dice_loss(recon_logits, targets)
    l_kl = kl_loss(mu, logvar)
    return ce_weight * l_ce + dice_weight * l_dice + kl_weight * l_kl, {
        "ce": l_ce.item(),
        "dice": l_dice.item(),
        "kl": l_kl.item(),
    }
