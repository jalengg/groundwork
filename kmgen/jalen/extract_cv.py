"""
Hybrid LLM+CV extraction pipeline for KM plots.

The LLM analyzed the image and identified:
- Two curves: blue (Trilaciclib) and orange (Placebo)
- A dashed black median line near month 5.9
- White background, anti-aliased lines ~2-3px thick
- Axes: x=0-21 months, y=0.0-1.0 probability

This code:
1. Color-filters to isolate each curve
2. Traces each curve column-by-column to get a 1D survival signal
3. Detects step-downs (sudden vertical drops in the signal)
4. Converts pixel positions to data coordinates via calibrated bbox
5. Draws red circles on the detected step-downs
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# ── Config ─────────────────────────────────────────────────────────
IMG_PATH = Path("/mnt/c/Users/jalen/kmgpt_plot.png")
ITER_ROOT = Path("/mnt/c/Users/jalen/kmgen/iterations")

# Calibrated bbox (from pixel detection)
BBOX = (163, 95, 783, 604)
X_MIN, X_MAX = 0, 21
Y_MIN, Y_MAX = 0.0, 1.0


def px_to_data(col, row):
    """Convert plot-relative pixel (col, row) to data (time, survival)."""
    left, top, right, bottom = BBOX
    t = X_MIN + col / (right - left) * (X_MAX - X_MIN)
    s = Y_MAX - row / (bottom - top) * (Y_MAX - Y_MIN)
    return t, s


def data_to_px_abs(t, s):
    """Convert data (time, survival) to absolute image pixel."""
    left, top, right, bottom = BBOX
    px = left + (t - X_MIN) / (X_MAX - X_MIN) * (right - left)
    py = top + (Y_MAX - s) / (Y_MAX - Y_MIN) * (bottom - top)
    return int(round(px)), int(round(py))


def isolate_curve(plot_rgb, color_name):
    """
    Create a binary mask for a specific curve color.
    Returns mask of shape (h, w).
    """
    r, g, b = plot_rgb[:,:,0].astype(float), plot_rgb[:,:,1].astype(float), plot_rgb[:,:,2].astype(float)

    if color_name == "blue":
        # Sampled blue core: R=97-147, G=120-163, B=145-187
        # Key: B-R > 30, B > G, muted overall
        # Include anti-aliased fringe but keep it tight enough to exclude orange
        mask = (
            (b > r + 30) &        # B clearly dominates R (was +20)
            (b > g + 5) &         # B above G
            (r < 170) &           # R stays low (orange has R>220)
            (b > 130) &           # B reasonably strong
            (r + g + b < 620)     # Not near-white
        )
    elif color_name == "orange":
        # Sampled orange core: R=226-237, G=145-186, B=93-152
        # Key: R > 220, R-B > 60
        mask = (
            (r > 210) &           # R very high (was 140 — way too loose)
            (r > b + 60) &        # R clearly dominates B (was +30)
            (r > g) &             # R above G
            (r + g + b < 660)     # Not near-white
        )
    else:
        raise ValueError(f"Unknown color: {color_name}")

    return mask


def trace_curve(mask, min_col=0, max_col=None):
    """
    Trace a curve through a binary mask using continuous tracking.

    Instead of taking the topmost pixel per column, we track the curve
    by following from the previous position. This prevents jumps to
    stray pixels from text or the other curve.

    For KM curves: the curve only goes DOWN (row increases) or stays flat.
    It never goes back up. We use this constraint.
    """
    h, w = mask.shape
    if max_col is None:
        max_col = w

    cols = []
    rows = []
    prev_row = None

    for col in range(min_col, max_col):
        column = mask[:, col]
        pixel_rows = np.where(column)[0]

        if len(pixel_rows) == 0:
            continue

        if prev_row is None:
            # First column: take topmost pixel (curve starts at S~1.0)
            best_row = pixel_rows[0]
        else:
            # Find the pixel closest to prev_row, but allow downward movement
            # Group pixels into clusters (gap > 5 = different cluster)
            clusters = []
            current = [pixel_rows[0]]
            for i in range(1, len(pixel_rows)):
                if pixel_rows[i] - pixel_rows[i-1] > 5:
                    clusters.append(current)
                    current = [pixel_rows[i]]
                else:
                    current.append(pixel_rows[i])
            clusters.append(current)

            # Pick the cluster whose topmost pixel is closest to prev_row
            # but at or below prev_row (KM curves only go down)
            best_cluster = None
            best_dist = float('inf')
            for cluster in clusters:
                top = cluster[0]
                # Allow small upward jitter (anti-aliasing) but prefer downward
                dist = abs(top - prev_row)
                if dist < best_dist:
                    best_dist = dist
                    best_cluster = cluster

            if best_cluster is None:
                continue

            best_row = best_cluster[0]  # topmost pixel of best cluster

            # Sanity: reject if the jump is too large (> 100px ≈ 0.20 in S)
            # unless it's a genuine step (which should be < 50px typically)
            if abs(best_row - prev_row) > 100:
                continue

        cols.append(col)
        rows.append(best_row)
        prev_row = best_row

    return np.array(cols), np.array(rows)


def detect_stepdowns(cols, rows, min_drop_px=3, max_drop_px=80):
    """
    Detect step-downs in a traced curve.

    A step-down is where the row position suddenly increases (curve drops down).
    KM curves have horizontal segments connected by vertical drops.

    max_drop_px: reject steps larger than this (~0.16 in S-units for our bbox).
    For a KM curve with N patients at risk, the max single-event drop is 1/N.
    Even with multiple simultaneous events, drops > 0.15 are very rare.

    Returns list of (col, row_before, row_after) for each detected step.
    """
    if len(cols) < 2:
        return []

    steps = []

    # Smooth the signal slightly to reduce noise
    # Use a small median filter
    from scipy.ndimage import median_filter
    smoothed = median_filter(rows.astype(float), size=3)

    # Find positions where the row increases significantly
    # (curve drops = row number increases = y decreases)
    diffs = np.diff(smoothed)

    i = 0
    while i < len(diffs):
        if diffs[i] >= min_drop_px:
            # Found start of a drop. Accumulate the full drop.
            drop_start = i
            total_drop = 0
            while i < len(diffs) and diffs[i] > 0:
                total_drop += diffs[i]
                i += 1

            if min_drop_px <= total_drop <= max_drop_px:
                # Record the step: use the column at the drop start,
                # row before = smoothed[drop_start], row after = smoothed[i]
                step_col = cols[drop_start]
                row_before = int(smoothed[drop_start])
                row_after = int(smoothed[min(i, len(smoothed)-1)])
                steps.append((step_col, row_before, row_after))
        else:
            i += 1

    return steps


def run_extraction():
    """Full pipeline: load, filter, trace, detect, annotate."""
    img = Image.open(IMG_PATH).convert("RGB")
    pixels = np.array(img)

    left, top, right, bottom = BBOX
    plot = pixels[top:bottom, left:right]

    # ── Isolate each curve ──
    blue_mask = isolate_curve(plot, "blue")
    orange_mask = isolate_curve(plot, "orange")

    # ── Exclude the dashed median line region ──
    # Dashed line is at plot col ~177 (month 5.9), exclude ±3 cols
    dashed_col = int(5.9 / 21 * (right - left))
    blue_mask[:, max(0,dashed_col-3):dashed_col+4] = False
    orange_mask[:, max(0,dashed_col-3):dashed_col+4] = False

    # ── Exclude legend area (top-right of plot) ──
    # Legend text is roughly in the top 25% vertically, right 50% horizontally
    ph, pw = blue_mask.shape
    legend_top = 0
    legend_bottom = int(ph * 0.30)
    legend_left = int(pw * 0.35)
    blue_mask[legend_top:legend_bottom, legend_left:] = False
    orange_mask[legend_top:legend_bottom, legend_left:] = False

    # ── Trace each curve ──
    blue_cols, blue_rows = trace_curve(blue_mask)
    orange_cols, orange_rows = trace_curve(orange_mask)

    print(f"Blue curve: {len(blue_cols)} column samples")
    print(f"Orange curve: {len(orange_cols)} column samples")

    # ── Detect step-downs ──
    blue_steps = detect_stepdowns(blue_cols, blue_rows, min_drop_px=3)
    orange_steps = detect_stepdowns(orange_cols, orange_rows, min_drop_px=3)

    print(f"Blue step-downs: {len(blue_steps)}")
    print(f"Orange step-downs: {len(orange_steps)}")

    # ── Convert to data coordinates ──
    blue_data = []
    for col, row_before, row_after in blue_steps:
        t, s_before = px_to_data(col, row_before)
        _, s_after = px_to_data(col, row_after)
        blue_data.append({"t": round(t, 2), "s_before": round(s_before, 4), "s_after": round(s_after, 4)})

    orange_data = []
    for col, row_before, row_after in orange_steps:
        t, s_before = px_to_data(col, row_before)
        _, s_after = px_to_data(col, row_after)
        orange_data.append({"t": round(t, 2), "s_before": round(s_before, 4), "s_after": round(s_after, 4)})

    # ── Print results ──
    print("\n── Blue (Trilaciclib) step-downs ──")
    for i, s in enumerate(blue_data, 1):
        print(f"  {i:2d}. t={s['t']:6.2f}  S: {s['s_before']:.4f} → {s['s_after']:.4f}")

    print(f"\n── Orange (Placebo) step-downs ──")
    for i, s in enumerate(orange_data, 1):
        print(f"  {i:2d}. t={s['t']:6.2f}  S: {s['s_before']:.4f} → {s['s_after']:.4f}")

    # ── Create output folder ──
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ITER_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Save debug masks ──
    blue_vis = np.zeros_like(plot)
    blue_vis[blue_mask] = [255, 0, 0]
    Image.fromarray(blue_vis).save(out_dir / "mask_blue.png")

    orange_vis = np.zeros_like(plot)
    orange_vis[orange_mask] = [255, 127, 0]
    Image.fromarray(orange_vis).save(out_dir / "mask_orange.png")

    # ── Draw annotation ──
    draw = ImageDraw.Draw(img)
    r = 7

    for col, row_before, row_after in blue_steps:
        # Circle at the bottom of the drop (the "knee")
        abs_x = left + col
        abs_y = top + row_after
        draw.ellipse([abs_x-r, abs_y-r, abs_x+r, abs_y+r], outline="red", width=2)

    for col, row_before, row_after in orange_steps:
        abs_x = left + col
        abs_y = top + row_after
        draw.ellipse([abs_x-r, abs_y-r, abs_x+r, abs_y+r], outline=(180, 0, 0), width=2)

    img.save(out_dir / "annotation.png")

    # ── Save JSON results ──
    results = {
        "bbox": BBOX,
        "blue_steps": blue_data,
        "orange_steps": orange_data,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    # ── Copy this script into the iteration folder ──
    import shutil
    shutil.copy(__file__, out_dir / "extract_cv.py")

    print(f"\nOutput: {out_dir}")
    print(f"  annotation.png — red circles on detected steps")
    print(f"  mask_blue.png  — isolated blue curve pixels")
    print(f"  mask_orange.png — isolated orange curve pixels")
    print(f"  results.json   — step coordinates")
    print(f"  extract_cv.py  — copy of this script")


if __name__ == "__main__":
    run_extraction()
