"""
Sample specific pixel colors at known curve locations to determine
tight color thresholds that cleanly separate blue vs orange.
"""

import numpy as np
from PIL import Image

IMG_PATH = "/mnt/c/Users/jalen/kmgpt_plot.png"
img = Image.open(IMG_PATH).convert("RGB")
pixels = np.array(img)

BBOX = (163, 95, 783, 604)
left, top, right, bottom = BBOX
plot = pixels[top:bottom, left:right]

# Sample pixel colors at specific locations where we KNOW which curve is which.
# These are plot-relative coordinates.

# Blue curve is clearly above orange at months 7-12 (right side of steep section)
# Orange curve is clearly below blue at months 7-12

# At month 8 (plot col ≈ 8/21 * 620 ≈ 236), blue curve is around S=0.30, orange around S=0.19
# At month 10 (plot col ≈ 10/21 * 620 ≈ 295), blue around S=0.22, orange around S=0.11
# At month 2 (plot col ≈ 2/21 * 620 ≈ 59), both curves are around S=0.90-0.94

def sample_region(plot, col_center, row_center, radius=5):
    """Sample pixels in a small region and report RGB stats."""
    r1 = max(0, row_center - radius)
    r2 = min(plot.shape[0], row_center + radius + 1)
    c1 = max(0, col_center - radius)
    c2 = min(plot.shape[1], col_center + radius + 1)

    region = plot[r1:r2, c1:c2].reshape(-1, 3)

    # Filter out white/near-white background
    mask = np.sum(region, axis=1) < 700
    colored = region[mask]

    return colored

print("── Sampling known curve locations ──\n")

# Convert data coords to plot-relative pixel coords
def data_to_plot(t, s):
    col = int(t / 21 * (right - left))
    row = int((1.0 - s) / 1.0 * (bottom - top))
    return col, row

# Blue curve should be here (based on visual inspection):
blue_samples = [
    (8, 0.30, "Blue at t=8, S~0.30"),
    (10, 0.22, "Blue at t=10, S~0.22"),
    (12, 0.12, "Blue at t=12, S~0.12"),
    (3, 0.78, "Blue at t=3, S~0.78"),
]

# Orange curve should be here:
orange_samples = [
    (8, 0.19, "Orange at t=8, S~0.19"),
    (10, 0.11, "Orange at t=10, S~0.11"),
    (6.5, 0.26, "Orange at t=6.5, S~0.26"),
    (4, 0.76, "Orange at t=4, S~0.76"),
]

for t, s, label in blue_samples:
    col, row = data_to_plot(t, s)
    colored = sample_region(plot, col, row, radius=3)
    if len(colored) > 0:
        print(f"{label} (plot col={col}, row={row}):")
        print(f"  Pixels: {len(colored)}")
        print(f"  R: {colored[:,0].min()}-{colored[:,0].max()} (mean {colored[:,0].mean():.0f})")
        print(f"  G: {colored[:,1].min()}-{colored[:,1].max()} (mean {colored[:,1].mean():.0f})")
        print(f"  B: {colored[:,2].min()}-{colored[:,2].max()} (mean {colored[:,2].mean():.0f})")
        # Print each unique pixel
        unique = np.unique(colored, axis=0)
        for px in unique[:5]:
            print(f"    ({px[0]}, {px[1]}, {px[2]})")
    else:
        print(f"{label}: no colored pixels found")
    print()

for t, s, label in orange_samples:
    col, row = data_to_plot(t, s)
    colored = sample_region(plot, col, row, radius=3)
    if len(colored) > 0:
        print(f"{label} (plot col={col}, row={row}):")
        print(f"  Pixels: {len(colored)}")
        print(f"  R: {colored[:,0].min()}-{colored[:,0].max()} (mean {colored[:,0].mean():.0f})")
        print(f"  G: {colored[:,1].min()}-{colored[:,1].max()} (mean {colored[:,1].mean():.0f})")
        print(f"  B: {colored[:,2].min()}-{colored[:,2].max()} (mean {colored[:,2].mean():.0f})")
        unique = np.unique(colored, axis=0)
        for px in unique[:5]:
            print(f"    ({px[0]}, {px[1]}, {px[2]})")
    else:
        print(f"{label}: no colored pixels found")
    print()
