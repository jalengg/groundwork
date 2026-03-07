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


def vae_loss(recon_logits, targets, mu, logvar, gamma=2.0, kl_weight=1e-4):
    """Combined focal + KL loss for VAE training."""
    freq = targets.mean(dim=(0, 2, 3)) + 1e-6  # (C,)
    alpha = 1.0 / freq
    alpha = alpha / alpha.sum()
    l_focal = focal_loss(recon_logits, targets, gamma=gamma, alpha=alpha)
    l_kl = kl_loss(mu, logvar)
    return l_focal + kl_weight * l_kl
