"""
Pin down bbox from axis label positions.
"""

import numpy as np
from PIL import Image

IMG_PATH = "/mnt/c/Users/jalen/kmgpt_plot.png"
img = Image.open(IMG_PATH).convert("RGB")
pixels = np.array(img)
gray = np.mean(pixels, axis=2)

# ── Y-axis labels (already found) ──
# Label centers: 1.0→94.5, 0.8→196.5, 0.6→299.0, 0.4→400.0, 0.2→502.5, 0.0→604.0
# Spacing: ~102 rows per 0.2 units
y_labels = {1.0: 94.5, 0.8: 196.5, 0.6: 299.0, 0.4: 400.0, 0.2: 502.5, 0.0: 604.0}
print("── Y-axis label positions ──")
for val, row in y_labels.items():
    print(f"  y={val:.1f} → row {row:.1f}")

# ── X-axis: scan row 564 (peak label row) for dark clusters ──
print("\n── X-axis brightness at row 564 (cols 120-730) ──")
row_data = gray[564, 120:730]
# Find dark regions (< 180 to catch anti-aliased text)
dark_mask = row_data < 180
dark_positions = np.where(dark_mask)[0] + 120  # offset back to absolute col

# Group into clusters
if len(dark_positions) > 0:
    clusters = []
    current = [dark_positions[0]]
    for i in range(1, len(dark_positions)):
        if dark_positions[i] - dark_positions[i-1] > 4:
            clusters.append(current)
            current = [dark_positions[i]]
        else:
            current.append(dark_positions[i])
    clusters.append(current)

    print(f"  Found {len(clusters)} dark clusters:")
    for i, cl in enumerate(clusters):
        center = np.mean(cl)
        print(f"    {i:2d}: cols {cl[0]}-{cl[-1]}, center={center:.1f}, width={cl[-1]-cl[0]+1}")

# ── X-axis: also try rows 563-565 combined ──
print("\n── X-axis combined rows 562-566 (threshold 180) ──")
combined = np.min(gray[562:567, :], axis=0)  # darkest pixel across these rows
dark_mask = combined[120:730] < 180
dark_positions = np.where(dark_mask)[0] + 120

if len(dark_positions) > 0:
    clusters = []
    current = [dark_positions[0]]
    for i in range(1, len(dark_positions)):
        if dark_positions[i] - dark_positions[i-1] > 4:
            clusters.append(current)
            current = [dark_positions[i]]
        else:
            current.append(dark_positions[i])
    clusters.append(current)

    print(f"  Found {len(clusters)} dark clusters:")
    centers = []
    for i, cl in enumerate(clusters):
        center = np.mean(cl)
        centers.append(center)
        print(f"    {i:2d}: cols {cl[0]}-{cl[-1]}, center={center:.1f}, width={cl[-1]-cl[0]+1}")

    # If we have enough clusters, try to determine spacing
    if len(centers) >= 2:
        # The labels are "0 1 2 3 ... 21" - 22 labels
        # Consecutive label centers should be ~equal spacing
        spacings = [centers[i+1] - centers[i] for i in range(len(centers)-1)]
        print(f"\n  Spacings between cluster centers: {[f'{s:.1f}' for s in spacings]}")
