import torch

from model.unet import DiffusionUNet


def test_unet_unconditional_output_shape():
    net = DiffusionUNet(latent_channels=4, cond_channels=4)
    noise = torch.zeros(2, 4, 64, 64)
    t = torch.tensor([100, 500])
    cond = torch.zeros(2, 4, 512, 512)  # zeros = unconditional
    out = net(noise, t, cond)
    assert out.shape == (2, 4, 64, 64)


def test_unet_conditional_different_from_unconditional():
    net = DiffusionUNet(latent_channels=4, cond_channels=4)
    noise = torch.randn(1, 4, 64, 64)
    t = torch.tensor([500])
    cond = torch.randn(1, 4, 512, 512)
    out_uncond = net(noise, t, torch.zeros_like(cond))
    out_cond = net(noise, t, cond)
    assert not torch.allclose(out_uncond, out_cond)
