import torch

from model.cdb import ConditionAwareDecoderBlock


def test_cdb_conditional_output_shape():
    block = ConditionAwareDecoderBlock(channels=64, cond_channels=64)
    R_down = torch.zeros(2, 64, 32, 32)
    R_up = torch.zeros(2, 64, 32, 32)
    R_c = torch.randn(2, 64, 32, 32)
    out = block(R_down, R_up, R_c)
    assert out.shape == (2, 64, 32, 32)


def test_cdb_unconditional_zeros_same_shape():
    block = ConditionAwareDecoderBlock(channels=64, cond_channels=64)
    R_down = torch.zeros(2, 64, 32, 32)
    R_up = torch.zeros(2, 64, 32, 32)
    R_c = torch.zeros(2, 64, 32, 32)  # zeros = unconditional
    out = block(R_down, R_up, R_c)
    assert out.shape == (2, 64, 32, 32)
