import numpy as np
import torch
from data_pipeline.dataset import RoadLayoutDataset


def test_dataset_returns_correct_shapes(tmp_path):
    # Create fake tile files
    for i in range(3):
        np.save(tmp_path / f"cond_{i:04d}.npy", np.zeros((4, 64, 64), dtype=np.float32))
        np.save(tmp_path / f"road_{i:04d}.npy", np.zeros((5, 64, 64), dtype=np.float32))
    ds = RoadLayoutDataset([str(tmp_path)], augment=True)
    assert len(ds) == 3
    cond, road = ds[0]
    assert isinstance(cond, torch.Tensor)
    assert cond.shape == (4, 64, 64)
    assert road.shape == (5, 64, 64)
