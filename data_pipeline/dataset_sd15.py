import os

import numpy as np
import torch
from torch.utils.data import Dataset

from data_pipeline.barrier_map import sample_inpaint_mask


def sd15_worker_init(worker_id):
    """DataLoader worker_init_fn — re-seeds each worker's RNG so augmentation
    sequences diverge instead of all workers producing identical outputs."""
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed)

# 5-class palette: bg, residential, tertiary, primary, motorway
# Each row is (R, G, B) in [0, 1]
PALETTE_FLOAT = np.array(
    [
        [0.0, 0.0, 0.0],  # 0 — background
        [1.0, 0.0, 0.0],  # 1 — residential
        [0.0, 1.0, 0.0],  # 2 — tertiary
        [0.0, 0.0, 1.0],  # 3 — primary
        [1.0, 1.0, 0.0],  # 4 — motorway
    ],
    dtype=np.float32,
)


class SD15Dataset(Dataset):
    """PyTorch Dataset for SD 1.5 ControlNet training.

    Returns:
        (cond_5ch, target_3ch) — float32 tensors of shapes (5, H, W) and (3, H, W).

    Conditioning channels:
        0 (R): elevation, per-tile normalized to [0, 1]
        1 (G): 0.3·water + 0.5·parkland + 0.2·agricultural  (0 for missing chs)
        2 (B): 0.6·residential + 0.8·commercial + 1.0·industrial  (0 for missing chs)
        3 (A): road skeleton — multi-level blend of classes 1–4 (masked if inpainting)
        4 (Z): buildable zone — cell_mask as float if inpainting, else zeros

    Target: full road encoded as normalized RGB via PALETTE_FLOAT, always the complete road.
    """

    def __init__(
        self,
        city_dirs: list,
        augment: bool = True,
        p_inpaint: float = 0.5,
        min_road_fraction: float = 0.0,
        seed: int = None,
    ):
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
                    if min_road_fraction > 0.0:
                        road = np.load(os.path.join(d, rf)).astype(np.float32)
                        frac = (road[1:].sum(axis=0) > 0).mean()
                        if frac < min_road_fraction:
                            continue
                    self.samples.append((os.path.join(d, cf), os.path.join(d, rf)))

        self.augment = augment
        self.p_inpaint = p_inpaint
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cond_path, road_path = self.samples[idx]
        cond = np.load(cond_path).astype(np.float32)  # (7, H, W) or (4, H, W) for older tiles
        road = np.load(road_path).astype(np.float32)  # (5, H, W)

        n_cond = cond.shape[0]

        # --- Augmentation: flips (same flip applied to both cond and road) ---
        if self.augment:
            if self.rng.random() > 0.5:
                cond = np.flip(cond, axis=2).copy()
                road = np.flip(road, axis=2).copy()
            if self.rng.random() > 0.5:
                cond = np.flip(cond, axis=1).copy()
                road = np.flip(road, axis=1).copy()

        # --- Inpainting augmentation ---
        do_inpaint = self.rng.random() < self.p_inpaint
        if do_inpaint:
            cell_mask, masked_road_skeleton = sample_inpaint_mask(cond, road, rng=self.rng)
            road_a = masked_road_skeleton  # (5, H, W) — local roads zeroed inside cell
        else:
            cell_mask = None
            road_a = road

        # --- Build 5-channel conditioning tensor ---
        H, W = cond.shape[1], cond.shape[2]

        # Ch 0: elevation — per-tile normalized
        elev = cond[0]
        ch0 = (elev - elev.min()) / (elev.max() - elev.min() + 1e-6)

        # Ch 1: 0.3·water + 0.5·parkland + 0.2·agricultural
        # cond indices: 0=elev, 1=water, 2=residential, 3=commercial, 4=industrial, 5=parkland, 6=agricultural
        water       = cond[1] if n_cond > 1 else np.zeros((H, W), dtype=np.float32)
        parkland    = cond[5] if n_cond > 5 else np.zeros((H, W), dtype=np.float32)
        agricultural = cond[6] if n_cond > 6 else np.zeros((H, W), dtype=np.float32)
        ch1 = 0.3 * water + 0.5 * parkland + 0.2 * agricultural

        # Ch 2: 0.6·residential + 0.8·commercial + 1.0·industrial
        residential = cond[2] if n_cond > 2 else np.zeros((H, W), dtype=np.float32)
        commercial  = cond[3] if n_cond > 3 else np.zeros((H, W), dtype=np.float32)
        industrial  = cond[4] if n_cond > 4 else np.zeros((H, W), dtype=np.float32)
        ch2 = 0.6 * residential + 0.8 * commercial + 1.0 * industrial

        # Ch 3: road skeleton multi-level blend — uses road_a (masked or full)
        ch3 = (
            0.25 * road_a[1]
            + 0.50 * road_a[2]
            + 0.75 * road_a[3]
            + 1.00 * road_a[4]
        )

        # Ch 4: buildable zone (inpaint cell mask, or zeros)
        if do_inpaint and cell_mask is not None:
            ch4 = cell_mask.astype(np.float32)
        else:
            ch4 = np.zeros((H, W), dtype=np.float32)

        cond_5ch = np.stack([ch0, ch1, ch2, ch3, ch4], axis=0)  # (5, H, W)

        # --- Build 3-channel target tensor (always full road) ---
        class_map = road.argmax(axis=0)          # (H, W) int in [0, 4]
        target_hwc = PALETTE_FLOAT[class_map]    # (H, W, 3)
        target_3ch = target_hwc.transpose(2, 0, 1)  # (3, H, W)

        return torch.from_numpy(cond_5ch), torch.from_numpy(target_3ch)
