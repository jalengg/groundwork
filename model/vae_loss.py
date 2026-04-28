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


_PAPER_ALPHA = torch.tensor([0.1, 0.3, 0.6, 1.0, 2.0])


def vae_loss(recon_logits, targets, mu, logvar, gamma=2.0, kl_weight=1e-4):
    """Combined focal + KL loss for VAE training. Alpha per CaRoLS paper:
    [bg, residential, tertiary, primary, motorway] = [0.1, 0.3, 0.6, 1.0, 2.0]."""
    alpha = _PAPER_ALPHA.to(recon_logits.device)
    l_focal = focal_loss(recon_logits, targets, gamma=gamma, alpha=alpha)
    l_kl = kl_loss(mu, logvar)
    return l_focal + kl_weight * l_kl
