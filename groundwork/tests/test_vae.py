import torch
from model.vae import RoadVAE


def test_vae_encoder_output_shape():
    model = RoadVAE()
    x = torch.zeros(2, 5, 512, 512)
    mu, logvar = model.encode(x)
    assert mu.shape == (2, 4, 64, 64)
    assert logvar.shape == (2, 4, 64, 64)


def test_vae_decoder_output_shape():
    model = RoadVAE()
    z = torch.zeros(2, 4, 64, 64)
    out = model.decode(z)
    assert out.shape == (2, 5, 512, 512)


def test_vae_forward_returns_three_tensors():
    model = RoadVAE()
    x = torch.zeros(2, 5, 512, 512)
    recon, mu, logvar = model(x)
    assert recon.shape == (2, 5, 512, 512)
