"""
Run one full iteration: calibration + annotation into a timestamped folder.

Usage:
    python run_iteration.py
"""

import os
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw

# ── Config ─────────────────────────────────────────────────────────
IMG_PATH = Path("/mnt/c/Users/jalen/kmgpt_plot.png")
ITER_ROOT = Path("/mnt/c/Users/jalen/kmgen/iterations")

# Axis data ranges
X_MIN, X_MAX = 0, 21
Y_MIN, Y_MAX = 0.0, 1.0

# Plot bounding box (left, top, right, bottom) in pixels
BBOX = (163, 95, 783, 604)


def data_to_px(t, s):
    left, top, right, bottom = BBOX
    px = left + (t - X_MIN) / (X_MAX - X_MIN) * (right - left)
    py = top + (Y_MAX - s) / (Y_MAX - Y_MIN) * (bottom - top)
    return int(round(px)), int(round(py))


def draw_cross(draw, cx, cy, size=8, color="lime", width=2):
    draw.line([(cx - size, cy), (cx + size, cy)], fill=color, width=width)
    draw.line([(cx, cy - size), (cx, cy + size)], fill=color, width=width)


# ── Step-down coordinates (visual reads) ───────────────────────────

# BLUE — Trilaciclib (median PFS 5.9 mo)
blue_steps = [
    (0.3,  0.96), (0.7,  0.94), (1.0,  0.91), (1.4,  0.89),
    (1.8,  0.87), (2.1,  0.85), (2.4,  0.83), (2.7,  0.80),
    (3.0,  0.78), (3.3,  0.76), (3.5,  0.74), (3.7,  0.72),
    (3.9,  0.70), (4.1,  0.67), (4.3,  0.65), (4.5,  0.61),
    (4.7,  0.57), (5.0,  0.54), (5.2,  0.50), (5.5,  0.46),
    (5.7,  0.44), (5.9,  0.43), (6.3,  0.41), (6.8,  0.39),
    (7.5,  0.37), (8.5,  0.30), (9.5,  0.26), (10.5, 0.22),
    (11.5, 0.18), (13.0, 0.14), (15.0, 0.11), (17.0, 0.07),
    (18.5, 0.00),
]

# ORANGE — Placebo (median PFS 5.4 mo)
orange_steps = [
    (0.4,  0.98), (0.8,  0.96), (1.5,  0.94), (2.0,  0.91),
    (2.5,  0.89), (2.8,  0.87), (3.2,  0.85), (3.5,  0.83),
    (3.8,  0.80), (4.0,  0.76), (4.2,  0.72), (4.4,  0.68),
    (4.6,  0.65), (4.8,  0.61), (5.0,  0.57), (5.2,  0.53),
    (5.4,  0.50), (5.6,  0.46), (5.8,  0.41), (6.0,  0.35),
    (6.2,  0.30), (6.5,  0.26), (6.8,  0.22), (7.2,  0.20),
    (8.0,  0.19), (9.0,  0.15), (9.5,  0.13), (10.0, 0.11),
    (12.0, 0.09), (17.0, 0.07), (19.0, 0.04), (21.0, 0.00),
]


def make_calibration(out_path: Path):
    img = Image.open(IMG_PATH).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Yellow dots at all gridline intersections
    for t in range(0, 22):
        for s_10 in range(0, 11, 2):
            s = s_10 / 10.0
            px, py = data_to_px(t, s)
            draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill="yellow")

    # Green crosses on axis ticks
    for t in range(0, 22):
        px, py = data_to_px(t, 0.0)
        draw_cross(draw, px, py, color="lime")
    for s_10 in range(0, 11, 2):
        s = s_10 / 10.0
        px, py = data_to_px(0, s)
        draw_cross(draw, px, py, color="lime")

    # Cyan corner crosses
    for t, s in [(0, 0.0), (0, 1.0), (21, 0.0), (21, 1.0)]:
        px, py = data_to_px(t, s)
        draw_cross(draw, px, py, size=12, color="cyan", width=3)

    # Bbox rectangle
    left, top, right, bottom = BBOX
    draw.rectangle([left, top, right, bottom], outline="cyan", width=1)

    img.save(out_path)
    print(f"  Calibration: {out_path}")


def make_annotation(out_path: Path):
    img = Image.open(IMG_PATH).convert("RGB")
    draw = ImageDraw.Draw(img)
    r = 7

    for t, s in blue_steps:
        px, py = data_to_px(t, s)
        draw.ellipse([px - r, py - r, px + r, py + r], outline="red", width=2)

    for t, s in orange_steps:
        px, py = data_to_px(t, s)
        draw.ellipse([px - r, py - r, px + r, py + r], outline=(180, 0, 0), width=2)

    # Legend
    lx, ly = BBOX[2] - 180, BBOX[1] + 10
    draw.ellipse([lx, ly, lx + 12, ly + 12], outline="red", width=2)
    draw.text((lx + 18, ly - 2), "= Trilaciclib steps", fill="red")
    draw.ellipse([lx, ly + 20, lx + 12, ly + 32], outline=(180, 0, 0), width=2)
    draw.text((lx + 18, ly + 18), "= Placebo steps", fill=(180, 0, 0))

    img.save(out_path)
    print(f"  Annotation:  {out_path}")
    print(f"  Blue steps:   {len(blue_steps)}")
    print(f"  Orange steps: {len(orange_steps)}")


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ITER_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Iteration: {out_dir}")
    make_calibration(out_dir / "calibration.png")
    make_annotation(out_dir / "annotation.png")
    print(f"BBOX: {BBOX}")


if __name__ == "__main__":
    main()
