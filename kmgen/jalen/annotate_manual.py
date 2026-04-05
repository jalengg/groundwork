"""
Manual KM coordinate extraction by Claude Code (visual inspection).
Draws red circles on detected step-down points for diagnostic comparison.
"""

from PIL import Image, ImageDraw, ImageFont

# ── Image & plot geometry ──────────────────────────────────────────
IMG_PATH = "/mnt/c/Users/jalen/kmgpt_plot.png"
OUT_PATH = "/mnt/c/Users/jalen/kmgen/iterations/004_2026-03-16_bbox_v2.png"

# Axis data ranges (read from axis labels)
X_MIN, X_MAX = 0, 21   # months
Y_MIN, Y_MAX = 0.0, 1.0  # probability of PFS

# Plot bounding box in pixels (inside the axes).
# Estimated by locating where the axis lines meet at corners.
# Image is 857 x 790.
PLOT_LEFT   = 126
PLOT_TOP    = 78
PLOT_RIGHT  = 714
PLOT_BOTTOM = 536


def data_to_px(t, s):
    """Convert (time, survival) to pixel (x, y)."""
    px = PLOT_LEFT + (t - X_MIN) / (X_MAX - X_MIN) * (PLOT_RIGHT - PLOT_LEFT)
    py = PLOT_TOP + (Y_MAX - s) / (Y_MAX - Y_MIN) * (PLOT_BOTTOM - PLOT_TOP)
    return int(round(px)), int(round(py))


# ── Extracted step-down coordinates ────────────────────────────────
# Each entry: (time, survival_level_after_drop)
# These are read visually from the KM plot.  The circle goes at the
# "knee" of each step (bottom of vertical drop, start of new horizontal).

# BLUE curve — Trilaciclib prior to E/P/A  (median PFS 5.9 mo)
blue_steps = [
    # early steps (months 0–3)
    (0.3,  0.96),
    (0.7,  0.94),
    (1.0,  0.91),
    (1.4,  0.89),
    (1.8,  0.87),
    (2.1,  0.85),
    (2.4,  0.83),
    (2.7,  0.80),
    (3.0,  0.78),
    (3.3,  0.76),
    # mid steps (months 3–6) — steeper section
    (3.5,  0.74),
    (3.7,  0.72),
    (3.9,  0.70),
    (4.1,  0.67),
    (4.3,  0.65),
    (4.5,  0.61),
    (4.7,  0.57),
    (5.0,  0.54),
    (5.2,  0.50),
    (5.5,  0.46),
    (5.7,  0.44),
    (5.9,  0.43),
    # late steps (months 6+) — sparser
    (6.3,  0.41),
    (6.8,  0.39),
    (7.5,  0.37),
    (8.5,  0.30),
    (9.5,  0.26),
    (10.5, 0.22),
    (11.5, 0.18),
    (13.0, 0.14),
    (15.0, 0.11),
    (17.0, 0.07),
    (18.5, 0.00),
]

# ORANGE curve — Placebo prior to E/P/A  (median PFS 5.4 mo)
orange_steps = [
    # early steps (months 0–3)
    (0.4,  0.98),
    (0.8,  0.96),
    (1.5,  0.94),
    (2.0,  0.91),
    (2.5,  0.89),
    (2.8,  0.87),
    (3.2,  0.85),
    (3.5,  0.83),
    # mid steps (months 3–7) — steep section
    (3.8,  0.80),
    (4.0,  0.76),
    (4.2,  0.72),
    (4.4,  0.68),
    (4.6,  0.65),
    (4.8,  0.61),
    (5.0,  0.57),
    (5.2,  0.53),
    (5.4,  0.50),
    (5.6,  0.46),
    (5.8,  0.41),
    (6.0,  0.35),
    (6.2,  0.30),
    (6.5,  0.26),
    (6.8,  0.22),
    (7.2,  0.20),
    # late steps (months 7+)
    (8.0,  0.19),
    (9.0,  0.15),
    (9.5,  0.13),
    (10.0, 0.11),
    (12.0, 0.09),
    (17.0, 0.07),
    (19.0, 0.04),
    (21.0, 0.00),
]


def main():
    img = Image.open(IMG_PATH).convert("RGB")
    draw = ImageDraw.Draw(img)

    r = 7  # circle radius

    # Draw blue-curve detections (red circles, thicker outline)
    for t, s in blue_steps:
        px, py = data_to_px(t, s)
        draw.ellipse([px - r, py - r, px + r, py + r], outline="red", width=2)

    # Draw orange-curve detections (darker red / maroon circles, thinner)
    for t, s in orange_steps:
        px, py = data_to_px(t, s)
        draw.ellipse([px - r, py - r, px + r, py + r], outline=(180, 0, 0), width=2)

    # Add a small legend in the upper-right area
    legend_x, legend_y = PLOT_RIGHT - 180, PLOT_TOP + 10
    draw.ellipse([legend_x, legend_y, legend_x + 12, legend_y + 12], outline="red", width=2)
    draw.text((legend_x + 18, legend_y - 2), "= Trilaciclib steps", fill="red")
    draw.ellipse([legend_x, legend_y + 20, legend_x + 12, legend_y + 32], outline=(180, 0, 0), width=2)
    draw.text((legend_x + 18, legend_y + 18), "= Placebo steps", fill=(180, 0, 0))

    img.save(OUT_PATH)
    print(f"Saved: {OUT_PATH}")
    print(f"Blue (Trilaciclib): {len(blue_steps)} steps detected")
    print(f"Orange (Placebo):   {len(orange_steps)} steps detected")

    # Print coordinates for review
    print("\n── Blue (Trilaciclib) ──")
    for i, (t, s) in enumerate(blue_steps, 1):
        print(f"  {i:2d}. t={t:5.1f}  S={s:.2f}")
    print(f"\n── Orange (Placebo) ──")
    for i, (t, s) in enumerate(orange_steps, 1):
        print(f"  {i:2d}. t={t:5.1f}  S={s:.2f}")


if __name__ == "__main__":
    main()
