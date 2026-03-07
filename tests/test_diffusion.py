import torch

from model.diffusion import DDPM
from model.unet import DiffusionUNet


def test_forward_diffusion_near_gaussian_at_T():
    ddpm = DDPM(T=1000)
    x0 = torch.zeros(4, 4, 16, 16)
    t = torch.full((4,), 999)
    x_t, eps = ddpm.forward_diffusion(x0, t)
    # At T≈1000, x_t should be approximately standard Gaussian
    assert abs(x_t.std().item() - 1.0) < 0.15


def test_training_loss_is_positive_scalar():
    ddpm = DDPM(T=1000)
    net = DiffusionUNet(latent_channels=4, cond_channels=4)
    x0 = torch.randn(2, 4, 16, 16)
    cond = torch.randn(2, 4, 64, 64)
    loss = ddpm.training_loss(net, x0, cond, cfg_prob=0.5)
    assert loss.shape == ()
    assert loss.item() > 0
    assert not torch.isnan(loss)
