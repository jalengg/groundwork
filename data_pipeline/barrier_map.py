import numpy as np
from scipy.ndimage import label, binary_dilation, generate_binary_structure


def compute_barrier(
    cond: np.ndarray,
    road: np.ndarray,
    arterial_dilation: int = 3,
    landuse_threshold: float = 0.3,
    slope_threshold: float = 0.25,
) -> np.ndarray:
    """Returns (H, W) bool barrier mask."""
    arterial = (road[3] > 0.5) | (road[4] > 0.5)
    if arterial_dilation > 0:
        struct = generate_binary_structure(2, 1)
        arterial = binary_dilation(arterial, structure=struct, iterations=arterial_dilation)

    water = cond[1] > 0.5

    landuse_max = cond[2:7].max(axis=0)
    landuse_grad = _sobel_magnitude(landuse_max)
    landuse_norm = _normalize(landuse_grad)
    landuse_edge = landuse_norm > landuse_threshold

    slope_grad = _sobel_magnitude(cond[0])
    slope_norm = _normalize(slope_grad)
    slope_edge = slope_norm > slope_threshold

    return arterial | water | landuse_edge | slope_edge


def extract_cells(
    barrier: np.ndarray,
    min_frac: float = 0.03,
    max_frac: float = 0.7,
) -> list[np.ndarray]:
    """Returns list of (H, W) bool cell masks."""
    labeled, num_features = label(~barrier)
    total = barrier.size
    cells = []
    for i in range(1, num_features + 1):
        mask = labeled == i
        frac = mask.sum() / total
        if min_frac <= frac <= max_frac:
            cells.append(mask)
    return cells


def sample_inpaint_mask(
    cond: np.ndarray,
    road: np.ndarray,
    rng=None,
    arterial_dilation: int = 3,
    landuse_threshold: float = 0.3,
    slope_threshold: float = 0.25,
    min_frac: float = 0.03,
    max_frac: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Main entry point for dataset augmentation.
    Returns (cell_mask, masked_road_skeleton) where:
    - cell_mask: (H, W) bool — True where model should generate
    - masked_road_skeleton: (5, H, W) float32 — road with classes 1+2 zeroed inside cell
    """
    if rng is None:
        rng = np.random.default_rng()

    barrier = compute_barrier(
        cond,
        road,
        arterial_dilation=arterial_dilation,
        landuse_threshold=landuse_threshold,
        slope_threshold=slope_threshold,
    )
    cells = extract_cells(barrier, min_frac=min_frac, max_frac=max_frac)

    if cells:
        idx = rng.integers(len(cells))
        cell_mask = cells[idx]
    else:
        cell_mask = _random_ellipse_mask(road.shape[1], road.shape[2], rng)

    masked_road = road.copy()
    # zero only local roads (classes 1+2); arterials (3+4) remain as context
    masked_road[1][cell_mask] = 0.0
    masked_road[2][cell_mask] = 0.0

    return cell_mask, masked_road


def _sobel_magnitude(arr: np.ndarray) -> np.ndarray:
    gx = np.gradient(arr, axis=1)
    gy = np.gradient(arr, axis=0)
    return np.sqrt(gx ** 2 + gy ** 2)


def _normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _random_ellipse_mask(H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    # coverage uniform in [0.10, 0.40]; ellipse avoids axis-aligned bias
    target_frac = rng.uniform(0.10, 0.40)
    cx = rng.uniform(0.2, 0.8) * W
    cy = rng.uniform(0.2, 0.8) * H

    area = target_frac * H * W
    aspect = rng.uniform(0.5, 2.0)
    # pi * a * b = area, b = a / aspect
    a = np.sqrt(area * aspect / np.pi)
    b = a / aspect
    angle = rng.uniform(0, np.pi)

    ys, xs = np.mgrid[0:H, 0:W]
    dx = xs - cx
    dy = ys - cy
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    xr = dx * cos_a + dy * sin_a
    yr = -dx * sin_a + dy * cos_a
    return (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
