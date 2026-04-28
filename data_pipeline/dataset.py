import os

import numpy as np
import torch
from torch.utils.data import Dataset


class RoadLayoutDataset(Dataset):
    def __init__(self, city_dirs: list, augment: bool = True):
        self.samples = []
        for d in city_dirs:
            if not os.path.isdir(d):
                continue
            cond_files = sorted(
                f for f in os.listdir(d) if f.startswith("cond_") and f.endswith(".npy")
            )
            for cf in cond_files:
                idx = cf.replace("cond_", "").replace(".npy", "")
                rf = f"road_{idx}.npy"
                if os.path.exists(os.path.join(d, rf)):
                    self.samples.append((os.path.join(d, cf), os.path.join(d, rf)))
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cond_path, road_path = self.samples[idx]
        cond = np.load(cond_path)  # (7, H, W): elev, water, residential, commercial, industrial, parkland, agricultural
        road = np.load(road_path)  # (5, H, W)

        if self.augment:
            # Flips only — rot90 destroys absolute orientation prior we want the model to learn
            if np.random.random() > 0.5:
                cond = np.flip(cond, axis=2).copy()
                road = np.flip(road, axis=2).copy()
            if np.random.random() > 0.5:
                cond = np.flip(cond, axis=1).copy()
                road = np.flip(road, axis=1).copy()
            # Brightness/contrast jitter on conditioning channels only
            for ch in range(cond.shape[0]):
                brightness = np.random.uniform(0.9, 1.1)
                contrast = np.random.uniform(0.9, 1.1)
                mean = cond[ch].mean()
                cond[ch] = np.clip(
                    (cond[ch] - mean) * contrast + mean * brightness, 0.0, 1.0
                )

        return torch.from_numpy(cond), torch.from_numpy(road)
