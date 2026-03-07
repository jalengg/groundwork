import torch

from model.vae_loss import vae_loss


def test_vae_loss_returns_scalar():
    recon = torch.randn(2, 5, 64, 64)
    target = torch.zeros(2, 5, 64, 64)
    target[:, 0] = 1.0  # background channel
    mu = torch.zeros(2, 4, 16, 16)
    logvar = torch.zeros(2, 4, 16, 16)
    loss = vae_loss(recon, target, mu, logvar)
    assert loss.shape == ()
    assert loss.item() > 0.0
    assert not torch.isnan(loss)
