"""
Calibration script: draw markers at KNOWN axis positions to verify
the plot bounding box is correct before attempting step detection.

We draw:
- Green crosses at every integer x-axis tick (t=0,1,...,21) along y=0.0
- Green crosses at every y-axis gridline (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) along x=0
- Blue crosses at grid intersections (t=0 at y=1.0, t=21 at y=0.0, etc.)

If the crosses land exactly on the axis ticks/gridlines, the bbox is correct.
"""

from PIL import Image, ImageDraw

IMG_PATH = "/mnt/c/Users/jalen/kmgpt_plot.png"
OUT_PATH = "/mnt/c/Users/jalen/kmgen/iterations/002_2026-03-16_calibration.png"

# ── Adjust these until the calibration markers align ──
X_MIN, X_MAX = 0, 21
Y_MIN, Y_MAX = 0.0, 1.0

# Plot bounding box: (left, top, right, bottom) in pixels
# These define where data point (0, 1.0) and (21, 0.0) map to.
BBOX = (126, 78, 714, 536)


def data_to_px(t, s):
    left, top, right, bottom = BBOX
    px = left + (t - X_MIN) / (X_MAX - X_MIN) * (right - left)
    py = top + (Y_MAX - s) / (Y_MAX - Y_MIN) * (bottom - top)
    return int(round(px)), int(round(py))


def draw_cross(draw, cx, cy, size=8, color="lime", width=2):
    draw.line([(cx - size, cy), (cx + size, cy)], fill=color, width=width)
    draw.line([(cx, cy - size), (cx, cy + size)], fill=color, width=width)


def main():
    img = Image.open(IMG_PATH).convert("RGB")
    draw = ImageDraw.Draw(img)

    # X-axis ticks: green crosses along y=0.0 line
    for t in range(0, 22):
        px, py = data_to_px(t, 0.0)
        draw_cross(draw, px, py, color="lime")

    # Y-axis ticks: green crosses along x=0 line
    for s_10 in range(0, 11, 2):  # 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
        s = s_10 / 10.0
        px, py = data_to_px(0, s)
        draw_cross(draw, px, py, color="lime")

    # Corner markers (should be at exact corners of plot area)
    corners = [(0, 0.0), (0, 1.0), (21, 0.0), (21, 1.0)]
    for t, s in corners:
        px, py = data_to_px(t, s)
        draw_cross(draw, px, py, size=12, color="cyan", width=3)

    # Draw crosses at ALL gridline intersections (y = 0.0, 0.2, ..., 1.0 at each x tick)
    for t in range(0, 22):
        for s_10 in range(0, 11, 2):
            s = s_10 / 10.0
            px, py = data_to_px(t, s)
            # Small yellow dots at grid intersections
            draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill="yellow")

    # Also draw the bbox rectangle itself for visual reference
    left, top, right, bottom = BBOX
    draw.rectangle([left, top, right, bottom], outline="cyan", width=1)

    img.save(OUT_PATH)
    print(f"Saved calibration image: {OUT_PATH}")
    print(f"BBOX: {BBOX}")
    print("Check: green crosses should sit on axis ticks, cyan crosses on plot corners.")


if __name__ == "__main__":
    main()
