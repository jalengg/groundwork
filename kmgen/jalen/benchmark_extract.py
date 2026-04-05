"""
KMGen Benchmark Extraction Pipeline
Extracts survival curves from synthetic KM plots using color filtering + curve tracing.
"""

import json
import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.metrics import compute_score

_REPO = Path(__file__).resolve().parent.parent
SYNTH_DIR = _REPO / 'shared' / 'synthetic'
BENCH_DIR = _REPO / 'benchmark'

# ─── Color detection utilities ───

def rgb_to_hsl(r, g, b):
    """Convert RGB [0-255] to HSL [0-1]."""
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    mx, mn = max(r, g, b), min(r, g, b)
    l = (mx + mn) / 2
    if mx == mn:
        h = s = 0.0
    else:
        d = mx - mn
        s = d / (2 - mx - mn) if l > 0.5 else d / (mx + mn)
        if mx == r:
            h = (g - b) / d + (6 if g < b else 0)
        elif mx == g:
            h = (b - r) / d + 2
        else:
            h = (r - g) / d + 4
        h /= 6
    return h, s, l


def make_color_mask(img_arr, color_spec, exclude_region=None):
    """
    Create a binary mask for pixels matching color_spec.
    color_spec is a dict with either:
      - 'hue_range': (min, max) in [0,1], 'sat_min', 'light_range': (min, max)
      - 'rgb_test': callable(r, g, b) -> bool
    """
    h, w = img_arr.shape[:2]
    mask = np.zeros((h, w), dtype=bool)

    if 'rgb_test' in color_spec:
        test = color_spec['rgb_test']
        for y in range(h):
            for x in range(w):
                r, g, b = img_arr[y, x, :3]
                if test(int(r), int(g), int(b)):
                    mask[y, x] = True
    else:
        hue_min, hue_max = color_spec.get('hue_range', (0, 1))
        sat_min = color_spec.get('sat_min', 0.2)
        light_min, light_max = color_spec.get('light_range', (0.1, 0.8))

        for y in range(h):
            for x in range(w):
                r, g, b = int(img_arr[y, x, 0]), int(img_arr[y, x, 1]), int(img_arr[y, x, 2])
                hue, sat, light = rgb_to_hsl(r, g, b)

                # Handle hue wrapping (for red)
                if hue_min <= hue_max:
                    hue_match = hue_min <= hue <= hue_max
                else:
                    hue_match = hue >= hue_min or hue <= hue_max

                if hue_match and sat >= sat_min and light_min <= light <= light_max:
                    mask[y, x] = True

    if exclude_region is not None:
        x1, y1, x2, y2 = exclude_region
        mask[y1:y2, x1:x2] = False

    return mask


def make_color_mask_vectorized(img_arr, color_spec, exclude_region=None):
    """Vectorized version of make_color_mask for speed."""
    h, w = img_arr.shape[:2]
    r = img_arr[:, :, 0].astype(float)
    g = img_arr[:, :, 1].astype(float)
    b = img_arr[:, :, 2].astype(float)

    if 'rgb_test_vec' in color_spec:
        mask = color_spec['rgb_test_vec'](r, g, b)
    elif 'rgb_test' in color_spec:
        # Fall back to slow version
        return make_color_mask(img_arr, color_spec, exclude_region)
    else:
        # Vectorized HSL
        rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
        mx = np.maximum(np.maximum(rn, gn), bn)
        mn = np.minimum(np.minimum(rn, gn), bn)
        light = (mx + mn) / 2
        d = mx - mn

        # Saturation
        sat = np.where(mx == mn, 0.0,
                       np.where(light > 0.5, d / (2 - mx - mn), d / (mx + mn + 1e-10)))

        # Hue
        hue = np.zeros_like(rn)
        red_max = (mx == rn) & (d > 0)
        grn_max = (mx == gn) & (d > 0) & ~red_max
        blu_max = (mx == bn) & (d > 0) & ~red_max & ~grn_max

        hue[red_max] = ((gn[red_max] - bn[red_max]) / d[red_max]) % 6
        hue[grn_max] = (bn[grn_max] - rn[grn_max]) / d[grn_max] + 2
        hue[blu_max] = (rn[blu_max] - gn[blu_max]) / d[blu_max] + 4
        hue = hue / 6.0
        hue = hue % 1.0

        hue_min, hue_max = color_spec.get('hue_range', (0, 1))
        sat_min = color_spec.get('sat_min', 0.2)
        light_min, light_max = color_spec.get('light_range', (0.1, 0.8))

        if hue_min <= hue_max:
            hue_match = (hue >= hue_min) & (hue <= hue_max)
        else:
            hue_match = (hue >= hue_min) | (hue <= hue_max)

        mask = hue_match & (sat >= sat_min) & (light >= light_min) & (light <= light_max)

    if exclude_region is not None:
        x1, y1, x2, y2 = exclude_region
        mask[y1:y2, x1:x2] = False

    return mask


# ─── Bbox detection ───

def detect_bbox(img_arr):
    """
    Detect the plot bounding box by finding axis lines.
    Looks for long horizontal and vertical dark lines.
    """
    h, w = img_arr.shape[:2]
    gray = np.mean(img_arr[:, :, :3], axis=2)
    dark = gray < 60  # dark pixels

    # Find leftmost vertical axis line: column with many dark pixels
    left = None
    for x in range(w // 4):
        col_count = np.sum(dark[:, x])
        if col_count > h * 0.4:
            left = x
            break

    # Find bottom horizontal axis line: row with many dark pixels
    bottom = None
    for y in range(h - 1, h // 2, -1):
        row_count = np.sum(dark[y, :])
        if row_count > w * 0.4:
            bottom = y
            break

    if left is None or bottom is None:
        # Fallback: use rough percentages
        left = int(w * 0.1)
        bottom = int(h * 0.85)

    # Find top of plot (where y-axis starts): look for topmost axis tick or label
    top = None
    for y in range(h // 10, bottom):
        # Check if there's a dark pixel on the y-axis
        if dark[y, left]:
            top = y
            break
    if top is None:
        top = int(h * 0.05)

    # Find right edge: rightmost point of x-axis
    right = None
    for x in range(w - 1, w // 2, -1):
        if dark[bottom, x]:
            right = x
            break
    if right is None:
        right = int(w * 0.9)

    return (left, top, right, bottom)


def detect_bbox_refined(img_arr):
    """
    Robust bbox detection using TICK MARKS, not frame/spine lines.

    Matplotlib adds ~5% padding beyond the data range, so the frame extends
    past the actual data area. Tick marks sit at the real data positions
    (0.0, 0.2, ..., 1.0 on y-axis), so they give the correct bbox.

    Strategy:
    1. Find the y-axis (leftmost long vertical dark line)
    2. Find tick marks: short horizontal dark segments extending from the y-axis
    3. Use the topmost and bottommost tick marks as the bbox top and bottom
    4. Find the x-axis tick marks similarly for left and right
    """
    h, w = img_arr.shape[:2]
    gray = np.mean(img_arr[:, :, :3], axis=2)
    dark = gray < 80

    # Find axis lines first (frame)
    row_counts = np.sum(dark, axis=1)
    col_counts = np.sum(dark, axis=0)

    # Left axis: leftmost col with many dark pixels
    left_candidates = np.where(col_counts > h * 0.3)[0]
    left_frame = left_candidates[0] if len(left_candidates) > 0 else int(w * 0.1)

    # Bottom axis: lowest row with many dark pixels
    bottom_candidates = np.where(row_counts > w * 0.3)[0]
    bottom_frame = bottom_candidates[-1] if len(bottom_candidates) > 0 else int(h * 0.85)

    # Right: find right end of x-axis frame
    right_frame = int(w * 0.95)
    for x in range(w - 1, left_frame, -1):
        if dark[bottom_frame, x]:
            right_frame = x
            break

    # Top: find top of frame
    top_frame = int(h * 0.05)
    for y in range(bottom_frame - 1, 0, -1):
        if not dark[y, left_frame] and not dark[y-1, left_frame]:
            top_frame = y + 1
            break

    # Now find TICK MARKS on the y-axis
    # Tick marks are short horizontal dark segments (3-10 pixels) extending
    # rightward from the y-axis line. They sit at data positions (0.0, 0.2, ..., 1.0).
    tick_rows = []
    for row in range(top_frame, bottom_frame + 1):
        # Count dark pixels in a short horizontal strip to the right of the y-axis
        tick_region = gray[row, left_frame:left_frame + 15]
        dark_in_tick = np.sum(tick_region < 80)
        if dark_in_tick >= 4:
            tick_rows.append(row)

    # Cluster tick rows (consecutive rows = same tick)
    tick_centers = []
    if tick_rows:
        current_group = [tick_rows[0]]
        for i in range(1, len(tick_rows)):
            if tick_rows[i] - tick_rows[i-1] <= 3:
                current_group.append(tick_rows[i])
            else:
                tick_centers.append(int(np.mean(current_group)))
                current_group = [tick_rows[i]]
        tick_centers.append(int(np.mean(current_group)))

    # Use topmost and bottommost tick marks as bbox top and bottom
    if len(tick_centers) >= 2:
        top = tick_centers[0]
        bottom = tick_centers[-1]
    else:
        top = top_frame
        bottom = bottom_frame

    # Find x-axis tick marks similarly
    tick_cols = []
    for col in range(left_frame, right_frame + 1):
        tick_region = gray[bottom_frame:bottom_frame + 15, col]
        dark_in_tick = np.sum(tick_region < 80)
        if dark_in_tick >= 4:
            tick_cols.append(col)

    tick_col_centers = []
    if tick_cols:
        current_group = [tick_cols[0]]
        for i in range(1, len(tick_cols)):
            if tick_cols[i] - tick_cols[i-1] <= 3:
                current_group.append(tick_cols[i])
            else:
                tick_col_centers.append(int(np.mean(current_group)))
                current_group = [tick_cols[i]]
        tick_col_centers.append(int(np.mean(current_group)))

    if len(tick_col_centers) >= 2:
        left = tick_col_centers[0]
        right = tick_col_centers[-1]
    else:
        left = left_frame
        right = right_frame

    return (left, top, right, bottom)


def detect_bbox_safe(arr, dark_threshold=80):
    """
    Wrapper that tries detect_bbox_refined, then falls back to detect_bbox
    if the result is degenerate (zero-area bbox).
    Also supports a configurable dark_threshold for blurry images.
    """
    # Try refined first with custom threshold
    h, w = arr.shape[:2]
    gray = np.mean(arr[:, :, :3], axis=2)

    # Override the internal dark threshold by pre-processing if needed
    if dark_threshold != 80:
        # Create a version with adjusted contrast for bbox detection
        # Scale so that pixels at dark_threshold map to 80
        scale = 80.0 / max(dark_threshold, 1)
        arr_adj = np.clip(arr.astype(float) * scale, 0, 255).astype(np.uint8)
        bbox = detect_bbox_refined(arr_adj)
    else:
        bbox = detect_bbox_refined(arr)

    left, top, right, bottom = bbox
    # Check for degenerate bbox
    if bottom - top < 10 or right - left < 10:
        # Fall back to simple detection
        bbox = detect_bbox(arr)
        left, top, right, bottom = bbox
        if bottom - top < 10 or right - left < 10:
            # Last resort: use image percentages
            left = int(w * 0.12)
            top = int(h * 0.05)
            right = int(w * 0.92)
            bottom = int(h * 0.85)
            bbox = (left, top, right, bottom)

    return bbox


# ─── Curve tracing ───

def trace_curve(mask, bbox, direction='down'):
    """
    Trace a curve through a binary mask using continuity tracking.
    direction='down' for survival (going down), 'up' for cumulative incidence.
    Returns list of (col, row) pixel positions.

    Uses the TOPMOST pixel of the nearest cluster per column.
    For KM step functions, the survival value IS at the top of the horizontal
    segment (it stays at the higher level until the next event), so topmost
    pixel is the correct choice.
    """
    left, top, right, bottom = bbox
    trace = []
    prev_row = None

    for col in range(left, right + 1):
        # Get all curve pixels in this column within the plot area
        col_pixels = np.where(mask[top:bottom+1, col])[0] + top

        if len(col_pixels) == 0:
            # Gap - carry forward
            if prev_row is not None:
                trace.append((col, prev_row))
            continue

        if prev_row is None:
            # First column: pick topmost pixel (for survival) or bottommost (for CI)
            if direction == 'down':
                best = col_pixels[0]
            else:
                best = col_pixels[-1]
        else:
            # Pick the cluster closest to previous position
            dists = np.abs(col_pixels - prev_row)
            best_idx = np.argmin(dists)
            best = col_pixels[best_idx]

            # For survival curves, reject large upward jumps
            if direction == 'down' and best < prev_row - 5:
                below = col_pixels[col_pixels >= prev_row - 3]
                if len(below) > 0:
                    best = below[0]
                else:
                    best = prev_row
            elif direction == 'up' and best > prev_row + 5:
                above = col_pixels[col_pixels <= prev_row + 3]
                if len(above) > 0:
                    best = above[-1]
                else:
                    best = prev_row

        prev_row = best
        trace.append((col, best))

    return trace


def pixel_to_data(col, row, bbox, axis):
    """Convert pixel (col, row) to data coordinates (t, s).
    col and row can be floats (from centroid calculation)."""
    left, top, right, bottom = bbox
    x_min = axis.get('x_min', 0)
    x_max = axis['x_max']
    y_min = axis.get('y_min', 0)
    y_max = axis.get('y_max', 1.0)

    t = x_min + (float(col) - left) / (right - left) * (x_max - x_min)
    s = y_max - (float(row) - top) / (bottom - top) * (y_max - y_min)

    return t, s


def extract_coordinates_sampled(trace, bbox, axis, sample_interval=None):
    """
    Convert a pixel trace to data coordinates, sampling at regular intervals.
    """
    if not trace:
        return []

    x_max = axis['x_max']
    x_min = axis.get('x_min', 0)

    if sample_interval is None:
        # Sample at roughly 200 points across the curve
        sample_interval = (x_max - x_min) / 200

    coords = []
    last_t = None

    for col, row in trace:
        t, s = pixel_to_data(col, row, bbox, axis)
        t = max(x_min, min(x_max, t))
        s = max(axis.get('y_min', 0), min(axis.get('y_max', 1.0), s))

        if last_t is None or t - last_t >= sample_interval:
            coords.append({'t': round(t, 4), 's': round(s, 6), 'method': 'sample'})
            last_t = t

    # Always include start and end
    if coords and coords[0]['t'] > x_min + 0.01:
        t0, s0 = pixel_to_data(trace[0][0], trace[0][1], bbox, axis)
        coords.insert(0, {'t': round(t0, 4), 's': round(s0, 6), 'method': 'sample'})

    if coords:
        t_last, s_last = pixel_to_data(trace[-1][0], trace[-1][1], bbox, axis)
        if t_last - coords[-1]['t'] > sample_interval * 0.5:
            coords.append({'t': round(t_last, 4), 's': round(s_last, 6), 'method': 'sample'})

    return coords


def extract_steps_from_trace(trace, bbox, axis, min_drop_px=2, direction='down'):
    """
    Detect step-down (or step-up for CI) points from a pixel trace.
    Returns list of coordinate dicts.
    """
    if not trace:
        return []

    coords = []
    # Start point
    t0, s0 = pixel_to_data(trace[0][0], trace[0][1], bbox, axis)
    coords.append({'t': round(t0, 4), 's': round(s0, 6), 'method': 'step'})

    prev_row = trace[0][1]

    for i in range(1, len(trace)):
        col, row = trace[i]

        if direction == 'down':
            drop = row - prev_row  # positive = downward
        else:
            drop = prev_row - row  # positive = upward (for CI)

        if drop >= min_drop_px:
            # This is a step - record the landing position
            t, s = pixel_to_data(col, row, bbox, axis)
            coords.append({'t': round(t, 4), 's': round(s, 6), 'method': 'step'})
            prev_row = row
        elif direction == 'down' and row > prev_row:
            prev_row = row  # small movement down, update
        elif direction == 'up' and row < prev_row:
            prev_row = row

    return coords


def smart_extract(trace, bbox, axis, direction='down'):
    """
    Hybrid extraction: use step detection where steps are clear,
    curve sampling where dense.
    Uses daily granularity as the natural sampling floor.
    """
    if not trace:
        return []

    # Use dense sampling approach (more robust for all curve types)
    x_max = axis['x_max']
    x_min = axis.get('x_min', 0)
    y_max = axis.get('y_max', 1.0)
    y_min = axis.get('y_min', 0.0)

    # Daily granularity: 1 day ≈ 1/30 of a month
    # This is the natural resolution floor — clinical events are recorded by date
    sample_interval = 1.0 / 30.0  # ~1 day in months

    coords = []
    last_t = None
    prev_s = None

    # Force correct starting point: KM curves ALWAYS start at (0, y_max) for
    # survival or (0, y_min) for cumulative incidence. This is a mathematical
    # certainty — no need to detect it from pixels.
    if direction == 'down':
        coords.append({'t': x_min, 's': y_max, 'method': 'step'})
    else:
        coords.append({'t': x_min, 's': y_min, 'method': 'step'})
    last_t = x_min
    prev_s = y_max if direction == 'down' else y_min

    for col, row in trace:
        t, s = pixel_to_data(col, row, bbox, axis)
        t = max(x_min, min(x_max, t))
        s = max(y_min, min(y_max, s))

        if last_t is not None and t - last_t < sample_interval:
            continue

        if t - last_t >= sample_interval:
            # Check if there was a significant change
            if prev_s is not None and abs(s - prev_s) > 0.001:
                coords.append({'t': round(t, 4), 's': round(s, 6), 'method': 'sample'})
                last_t = t
                prev_s = s
            elif t - last_t >= sample_interval * 3:
                # Force a sample even if flat
                coords.append({'t': round(t, 4), 's': round(s, 6), 'method': 'sample'})
                last_t = t
                prev_s = s

    # Always include last point
    if trace:
        t_last, s_last = pixel_to_data(trace[-1][0], trace[-1][1], bbox, axis)
        if not coords or abs(t_last - coords[-1]['t']) > 0.01:
            coords.append({'t': round(t_last, 4), 's': round(s_last, 6), 'method': 'sample'})

    return coords


# ─── Annotation ───

def annotate_image(img_path, extraction, out_path):
    """Draw detected coordinates on the image (original resolution).
    The extraction bbox may be from an upscaled image — detect and rescale."""
    img = Image.open(img_path).convert('RGB')
    img_w, img_h = img.size
    draw = ImageDraw.Draw(img)
    axis = extraction['axis']

    # The bbox in the extraction was computed on the upscaled image.
    # Detect the scale factor by comparing bbox extent to image size.
    ext_bbox = extraction.get('bbox', None)
    if ext_bbox is None or ext_bbox == [0, 0, 0, 0]:
        arr = np.array(img)
        bbox = detect_bbox_refined(arr)
    else:
        # If bbox right > image width, it's from an upscaled image
        if ext_bbox[2] > img_w:
            scale = ext_bbox[2] / (img_w * 0.9)  # rough scale estimate
            scale = round(scale)  # should be 2 or 3
            bbox = [v / scale for v in ext_bbox]
        else:
            bbox = ext_bbox

    left, top, right, bottom = bbox
    x_min = axis.get('x_min', 0)
    x_max = axis['x_max']
    y_min = axis.get('y_min', 0)
    y_max = axis.get('y_max', 1.0)

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']

    for arm_idx, arm in enumerate(extraction['arms']):
        color = colors[arm_idx % len(colors)]
        for coord in arm['coordinates']:
            t, s = coord['t'], coord['s']
            px = left + (t - x_min) / (x_max - x_min) * (right - left)
            py = top + (y_max - s) / (y_max - y_min) * (bottom - top)

            r = 3
            draw.ellipse([px - r, py - r, px + r, py + r], outline=color, width=1)

    img.save(out_path)


# ─── Standard 2-arm extraction (blue solid + red dashed) ───

def extract_standard_2arm(img_path, axis, labels=('Arm A', 'Arm B'), upscale=True):
    """Extract a standard 2-arm KM plot with blue solid + red dashed curves."""
    img = Image.open(img_path).convert('RGB')

    if upscale:
        img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)

    arr = np.array(img)
    bbox = detect_bbox_safe(arr)
    left, top, right, bottom = bbox

    # Mask out legend region (bottom-left quadrant of plot area)
    legend_exclude = (left, int(top + (bottom - top) * 0.65), int(left + (right - left) * 0.35), bottom)

    # Blue color filter (HSL) — widened to handle blur/JPEG artifacts
    blue_spec = {
        'hue_range': (0.50, 0.75),
        'sat_min': 0.15,
        'light_range': (0.15, 0.78)
    }
    blue_mask = make_color_mask_vectorized(arr, blue_spec, exclude_region=legend_exclude)

    # Red color filter (HSL) - red hue wraps around 0
    red_spec = {
        'hue_range': (0.93, 0.10),
        'sat_min': 0.15,
        'light_range': (0.15, 0.78)
    }
    red_mask = make_color_mask_vectorized(arr, red_spec, exclude_region=legend_exclude)

    # Trace curves
    blue_trace = trace_curve(blue_mask, bbox, direction='down')
    red_trace = trace_curve(red_mask, bbox, direction='down')

    # Extract coordinates
    blue_coords = smart_extract(blue_trace, bbox, axis, direction='down')
    red_coords = smart_extract(red_trace, bbox, axis, direction='down')

    extraction = {
        'image': str(img_path),
        'bbox': list(bbox),
        'axis': axis,
        'arms': [
            {'label': labels[0], 'color': 'blue', 'coordinates': blue_coords},
            {'label': labels[1], 'color': 'red', 'coordinates': red_coords},
        ]
    }

    return extraction


# ─── Plot-specific configurations ───

PLOT_CONFIGS = {
    'synthetic_001': {
        'axis': {'x_min': 0, 'x_max': 12, 'y_min': 0.0, 'y_max': 1.0},
        'labels': ('Arm A', 'Arm B'),
        'type': 'standard_2arm',
    },
    'synthetic_002': {
        'axis': {'x_min': 0, 'x_max': 36, 'y_min': 0.0, 'y_max': 1.0},
        'labels': ('Experimental', 'Standard'),
        'type': 'standard_2arm',
    },
    'synthetic_003': {
        'axis': {'x_min': 0, 'x_max': 36, 'y_min': 0.0, 'y_max': 1.0},
        'labels': ('Drug A', 'Placebo'),
        'type': 'standard_2arm',
    },
    'synthetic_004': {
        'axis': {'x_min': 0, 'x_max': 18, 'y_min': 0.0, 'y_max': 1.0},
        'labels': ('Combination', 'Monotherapy'),
        'type': 'standard_2arm',
    },
    'synthetic_005': {
        'axis': {'x_min': 0, 'x_max': 12, 'y_min': 0.0, 'y_max': 1.0},
        'labels': ('Combination', 'Monotherapy'),
        'type': 'standard_2arm',
    },
    'edge_high_survival': {
        'axis': {'x_min': 0, 'x_max': 144, 'y_min': 0.75, 'y_max': 1.0},
        'labels': ('Treatment', 'Control'),
        'type': 'high_survival',
    },
    'edge_cumulative_incidence': {
        'axis': {'x_min': 0, 'x_max': 60, 'y_min': 0.0, 'y_max': 0.5},
        'labels': ('Arm A', 'Arm B', 'Arm C'),
        'type': 'cumulative_incidence',
    },
    'edge_four_arms': {
        'axis': {'x_min': 0, 'x_max': 24, 'y_min': 0, 'y_max': 1.0},
        'labels': ('Arm A', 'Arm B', 'Arm C', 'Arm D'),
        'type': 'four_arms',
    },
    'edge_ci_shading': {
        'axis': {'x_min': 0, 'x_max': 36, 'y_min': 0, 'y_max': 1.0},
        'labels': ('Treatment', 'Control'),
        'type': 'ci_shading',
    },
    'edge_near_flat': {
        'axis': {'x_min': 0, 'x_max': 14, 'y_min': 0, 'y_max': 1.0},
        'labels': ('Low risk', 'High risk'),
        'type': 'standard_2arm',
    },
    'edge_small_dense': {
        'axis': {'x_min': 0, 'x_max': 36, 'y_min': 0, 'y_max': 1.0},
        'labels': ('Treatment', 'Control'),
        'type': 'small_dense',
    },
    'edge_multi_panel': {
        'axis': {
            'panel_a': {'x_min': 0, 'x_max': 36, 'y_min': 0, 'y_max': 1.0},
            'panel_b': {'x_min': 0, 'x_max': 24, 'y_min': 0, 'y_max': 1.0}
        },
        'labels': ('Treatment (OS)', 'Control (OS)', 'Treatment (PFS)', 'Control (PFS)'),
        'type': 'multi_panel',
    },
    # ─── Stress test plots ───
    'stress_blurry': {
        'axis': {'x_min': 0, 'x_max': 24, 'y_min': 0.0, 'y_max': 1.0},
        'labels': ('Treatment', 'Control'),
        'type': 'standard_2arm',
    },
    'stress_jpeg_artifact': {
        'axis': {'x_min': 0, 'x_max': 24, 'y_min': 0.0, 'y_max': 1.0},
        'labels': ('Treatment', 'Control'),
        'type': 'standard_2arm',
    },
    'stress_bw': {
        'axis': {'x_min': 0, 'x_max': 24, 'y_min': 0.0, 'y_max': 1.0},
        'labels': ('Treatment', 'Control'),
        'type': 'bw_2arm',
    },
    'stress_stretched_wide': {
        'axis': {'x_min': 0, 'x_max': 24, 'y_min': 0.0, 'y_max': 1.0},
        'labels': ('Treatment', 'Control'),
        'type': 'standard_2arm',
    },
    'stress_stretched_tall': {
        'axis': {'x_min': 0, 'x_max': 24, 'y_min': 0.0, 'y_max': 1.0},
        'labels': ('Treatment', 'Control'),
        'type': 'standard_2arm',
    },
    'stress_tiny': {
        'axis': {'x_min': 0, 'x_max': 24, 'y_min': 0.0, 'y_max': 1.0},
        'labels': ('Treatment', 'Control'),
        'type': 'tiny_2arm',
    },
    'stress_legend_overlap': {
        'axis': {'x_min': 0, 'x_max': 24, 'y_min': 0.0, 'y_max': 1.0},
        'labels': ('Treatment', 'Control'),
        'type': 'standard_2arm',
    },
    'stress_gridlines': {
        'axis': {'x_min': 0, 'x_max': 24, 'y_min': 0.0, 'y_max': 1.0},
        'labels': ('Treatment', 'Control'),
        'type': 'standard_2arm',
    },
    'stress_three_similar': {
        'axis': {'x_min': 0, 'x_max': 24, 'y_min': 0.0, 'y_max': 1.0},
        'labels': ('Arm A', 'Arm B', 'Arm C'),
        'type': 'three_similar',
    },
    'stress_annotation_heavy': {
        'axis': {'x_min': 0, 'x_max': 24, 'y_min': 0.0, 'y_max': 1.0},
        'labels': ('Treatment', 'Control'),
        'type': 'standard_2arm',
    },
    # ─── Combined stress tests ───
    'stress_combo_tiny_blurry': {
        'axis': {'x_min': 0, 'x_max': 24, 'y_min': 0.0, 'y_max': 1.0},
        'labels': ('Treatment', 'Control'),
        'type': 'tiny_blurry_2arm',
    },
    'stress_combo_bw_overlap': {
        'axis': {'x_min': 0, 'x_max': 36, 'y_min': 0.0, 'y_max': 1.0},
        'labels': ('Arm A', 'Arm B'),
        'type': 'bw_2arm',
    },
    'stress_combo_stretched_dense': {
        'axis': {'x_min': 0, 'x_max': 48, 'y_min': 0.0, 'y_max': 1.0},
        'labels': ('Treatment', 'Control'),
        'type': 'standard_2arm',
    },
    'stress_combo_tiny_bw': {
        'axis': {'x_min': 0, 'x_max': 24, 'y_min': 0.0, 'y_max': 1.0},
        'labels': ('Treatment', 'Control'),
        'type': 'tiny_bw_2arm',
    },
    'stress_combo_jpeg_blurry_dark': {
        'axis': {'x_min': 0, 'x_max': 24, 'y_min': 0.0, 'y_max': 1.0},
        'labels': ('Treatment', 'Control'),
        'type': 'dark_2arm',
    },
    'stress_combo_flat_overlap': {
        'axis': {'x_min': 0, 'x_max': 60, 'y_min': 0.75, 'y_max': 1.0},
        'labels': ('Arm A', 'Arm B'),
        'type': 'high_survival',
    },
    'stress_combo_4arm_tiny': {
        'axis': {'x_min': 0, 'x_max': 36, 'y_min': 0.0, 'y_max': 1.0},
        'labels': ('Arm A', 'Arm B', 'Arm C', 'Arm D'),
        'type': 'tiny_four_arms',
    },
    'stress_combo_grid_annotation_bw': {
        'axis': {'x_min': 0, 'x_max': 24, 'y_min': 0.0, 'y_max': 1.0},
        'labels': ('Treatment', 'Control'),
        'type': 'bw_2arm',
    },
    'stress_diep_like': {
        'axis': {
            'panel_a': {'x_min': 0, 'x_max': 50, 'y_min': 0.0, 'y_max': 0.5},
            'panel_b': {'x_min': 0, 'x_max': 50, 'y_min': 0.0, 'y_max': 0.5},
        },
        'labels': ('DIEP', 'Implant', 'Latissimus', 'TRAM',
                   'Unilateral', 'Bilateral', 'Immediate', 'Delayed'),
        'type': 'diep_multi_panel',
    },
    'stress_multi_panel_clean': {
        'axis': {
            'panel_a': {'x_min': 0, 'x_max': 36, 'y_min': 0, 'y_max': 1.0},
            'panel_b': {'x_min': 0, 'x_max': 24, 'y_min': 0, 'y_max': 1.0},
        },
        'labels': ('Treatment (OS)', 'Control (OS)', 'Treatment (PFS)', 'Control (PFS)'),
        'type': 'multi_panel',
    },
}


def extract_high_survival(img_path, axis, labels):
    """High survival plot with zoomed y-axis (0.75-1.0)."""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
    arr = np.array(img)
    bbox = detect_bbox_refined(arr)
    left, top, right, bottom = bbox

    legend_exclude = (left, int(top + (bottom - top) * 0.6), int(left + (right - left) * 0.35), bottom)

    # Blue and red - both solid lines, with censor tick marks
    blue_spec = {
        'hue_range': (0.5, 0.72),
        'sat_min': 0.25,
        'light_range': (0.2, 0.7)
    }
    red_spec = {
        'hue_range': (0.95, 0.08),
        'sat_min': 0.25,
        'light_range': (0.2, 0.6)
    }

    blue_mask = make_color_mask_vectorized(arr, blue_spec, exclude_region=legend_exclude)
    red_mask = make_color_mask_vectorized(arr, red_spec, exclude_region=legend_exclude)

    blue_trace = trace_curve(blue_mask, bbox, direction='down')
    red_trace = trace_curve(red_mask, bbox, direction='down')

    blue_coords = smart_extract(blue_trace, bbox, axis, direction='down')
    red_coords = smart_extract(red_trace, bbox, axis, direction='down')

    return {
        'image': str(img_path),
        'bbox': list(bbox),
        'axis': axis,
        'arms': [
            {'label': labels[0], 'color': 'blue', 'coordinates': blue_coords},
            {'label': labels[1], 'color': 'red', 'coordinates': red_coords},
        ]
    }


def extract_cumulative_incidence(img_path, axis, labels):
    """Cumulative incidence - 3 arms, curves go UP."""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
    arr = np.array(img)
    bbox = detect_bbox_refined(arr)
    left, top, right, bottom = bbox

    # Legend is top-left
    legend_exclude = (left, top, int(left + (right - left) * 0.25), int(top + (bottom - top) * 0.3))

    # 3 colors: blue, red, green
    blue_spec = {
        'hue_range': (0.55, 0.72),
        'sat_min': 0.3,
        'light_range': (0.15, 0.6)
    }
    red_spec = {
        'hue_range': (0.95, 0.08),
        'sat_min': 0.3,
        'light_range': (0.15, 0.6)
    }
    green_spec = {
        'hue_range': (0.25, 0.45),
        'sat_min': 0.3,
        'light_range': (0.1, 0.55)
    }

    blue_mask = make_color_mask_vectorized(arr, blue_spec, exclude_region=legend_exclude)
    red_mask = make_color_mask_vectorized(arr, red_spec, exclude_region=legend_exclude)
    green_mask = make_color_mask_vectorized(arr, green_spec, exclude_region=legend_exclude)

    # For CI curves, we trace upward (curves go from bottom to top)
    blue_trace = trace_curve(blue_mask, bbox, direction='up')
    red_trace = trace_curve(red_mask, bbox, direction='up')
    green_trace = trace_curve(green_mask, bbox, direction='up')

    # Also need to exclude at-risk table below plot
    # For CI, the s values increase over time
    blue_coords = smart_extract(blue_trace, bbox, axis, direction='up')
    red_coords = smart_extract(red_trace, bbox, axis, direction='up')
    green_coords = smart_extract(green_trace, bbox, axis, direction='up')

    return {
        'image': str(img_path),
        'bbox': list(bbox),
        'axis': axis,
        'arms': [
            {'label': labels[0], 'color': 'blue', 'coordinates': blue_coords},
            {'label': labels[1], 'color': 'red', 'coordinates': red_coords},
            {'label': labels[2], 'color': 'green', 'coordinates': green_coords},
        ]
    }


def extract_four_arms(img_path, axis, labels):
    """Four-arm plot: blue solid, red dashed, green dotted, purple dash-dot."""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
    arr = np.array(img)
    bbox = detect_bbox_refined(arr)
    left, top, right, bottom = bbox

    legend_exclude = (left, int(top + (bottom - top) * 0.55), int(left + (right - left) * 0.25), bottom)

    blue_spec = {
        'hue_range': (0.55, 0.72),
        'sat_min': 0.3,
        'light_range': (0.15, 0.65)
    }
    red_spec = {
        'hue_range': (0.95, 0.08),
        'sat_min': 0.35,
        'light_range': (0.15, 0.6)
    }
    green_spec = {
        'hue_range': (0.25, 0.45),
        'sat_min': 0.3,
        'light_range': (0.1, 0.55)
    }
    purple_spec = {
        'hue_range': (0.72, 0.88),
        'sat_min': 0.2,
        'light_range': (0.1, 0.55)
    }

    blue_mask = make_color_mask_vectorized(arr, blue_spec, exclude_region=legend_exclude)
    red_mask = make_color_mask_vectorized(arr, red_spec, exclude_region=legend_exclude)
    green_mask = make_color_mask_vectorized(arr, green_spec, exclude_region=legend_exclude)
    purple_mask = make_color_mask_vectorized(arr, purple_spec, exclude_region=legend_exclude)

    blue_trace = trace_curve(blue_mask, bbox, direction='down')
    red_trace = trace_curve(red_mask, bbox, direction='down')
    green_trace = trace_curve(green_mask, bbox, direction='down')
    purple_trace = trace_curve(purple_mask, bbox, direction='down')

    blue_coords = smart_extract(blue_trace, bbox, axis, direction='down')
    red_coords = smart_extract(red_trace, bbox, axis, direction='down')
    green_coords = smart_extract(green_trace, bbox, axis, direction='down')
    purple_coords = smart_extract(purple_trace, bbox, axis, direction='down')

    return {
        'image': str(img_path),
        'bbox': list(bbox),
        'axis': axis,
        'arms': [
            {'label': labels[0], 'color': 'blue', 'coordinates': blue_coords},
            {'label': labels[1], 'color': 'red', 'coordinates': red_coords},
            {'label': labels[2], 'color': 'green', 'coordinates': green_coords},
            {'label': labels[3], 'color': 'purple', 'coordinates': purple_coords},
        ]
    }


def extract_ci_shading(img_path, axis, labels):
    """CI shading: need to filter out translucent shading bands, extract only curve lines."""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
    arr = np.array(img)
    bbox = detect_bbox_refined(arr)
    left, top, right, bottom = bbox

    legend_exclude = (left, int(top + (bottom - top) * 0.6), int(left + (right - left) * 0.3), bottom)

    # For CI shading, need tighter saturation/lightness to exclude the translucent bands
    # The actual line is more saturated and darker than the shading
    blue_spec = {
        'hue_range': (0.5, 0.72),
        'sat_min': 0.45,  # Higher saturation threshold to exclude light shading
        'light_range': (0.15, 0.55)  # Darker range to exclude light shading
    }
    red_spec = {
        'hue_range': (0.95, 0.08),
        'sat_min': 0.45,
        'light_range': (0.15, 0.55)
    }

    blue_mask = make_color_mask_vectorized(arr, blue_spec, exclude_region=legend_exclude)
    red_mask = make_color_mask_vectorized(arr, red_spec, exclude_region=legend_exclude)

    blue_trace = trace_curve(blue_mask, bbox, direction='down')
    red_trace = trace_curve(red_mask, bbox, direction='down')

    blue_coords = smart_extract(blue_trace, bbox, axis, direction='down')
    red_coords = smart_extract(red_trace, bbox, axis, direction='down')

    return {
        'image': str(img_path),
        'bbox': list(bbox),
        'axis': axis,
        'arms': [
            {'label': labels[0], 'color': 'blue', 'coordinates': blue_coords},
            {'label': labels[1], 'color': 'red', 'coordinates': red_coords},
        ]
    }


def extract_small_dense(img_path, axis, labels):
    """Small dense image - upscale more aggressively."""
    img = Image.open(img_path).convert('RGB')
    # Upscale 3x for small images
    img = img.resize((img.width * 3, img.height * 3), Image.LANCZOS)
    arr = np.array(img)
    bbox = detect_bbox_refined(arr)
    left, top, right, bottom = bbox

    legend_exclude = (left, int(top + (bottom - top) * 0.6), int(left + (right - left) * 0.4), bottom)

    blue_spec = {
        'hue_range': (0.5, 0.72),
        'sat_min': 0.2,
        'light_range': (0.2, 0.7)
    }
    red_spec = {
        'hue_range': (0.95, 0.1),
        'sat_min': 0.2,
        'light_range': (0.2, 0.65)
    }

    blue_mask = make_color_mask_vectorized(arr, blue_spec, exclude_region=legend_exclude)
    red_mask = make_color_mask_vectorized(arr, red_spec, exclude_region=legend_exclude)

    blue_trace = trace_curve(blue_mask, bbox, direction='down')
    red_trace = trace_curve(red_mask, bbox, direction='down')

    blue_coords = smart_extract(blue_trace, bbox, axis, direction='down')
    red_coords = smart_extract(red_trace, bbox, axis, direction='down')

    return {
        'image': str(img_path),
        'bbox': list(bbox),
        'axis': axis,
        'arms': [
            {'label': labels[0], 'color': 'blue', 'coordinates': blue_coords},
            {'label': labels[1], 'color': 'red', 'coordinates': red_coords},
        ]
    }


def extract_multi_panel(img_path, axis, labels):
    """Multi-panel: split image in half, extract each panel separately."""
    img = Image.open(img_path).convert('RGB')
    w, h = img.size

    # Split into left (panel A) and right (panel B) halves
    mid_x = w // 2
    panel_a_img = img.crop((0, 0, mid_x, h))
    panel_b_img = img.crop((mid_x, 0, w, h))

    results = []
    for panel_img, panel_key, panel_labels in [
        (panel_a_img, 'panel_a', (labels[0], labels[1])),
        (panel_b_img, 'panel_b', (labels[2], labels[3])),
    ]:
        # Upscale 3x since these are small
        panel_img = panel_img.resize((panel_img.width * 3, panel_img.height * 3), Image.LANCZOS)
        arr = np.array(panel_img)
        bbox = detect_bbox_refined(arr)
        left, top, right, bottom = bbox

        legend_exclude = (left, int(top + (bottom - top) * 0.6), int(left + (right - left) * 0.45), bottom)

        panel_axis = axis[panel_key]

        blue_spec = {
            'hue_range': (0.5, 0.72),
            'sat_min': 0.2,
            'light_range': (0.2, 0.7)
        }
        red_spec = {
            'hue_range': (0.95, 0.1),
            'sat_min': 0.2,
            'light_range': (0.2, 0.65)
        }

        blue_mask = make_color_mask_vectorized(arr, blue_spec, exclude_region=legend_exclude)
        red_mask = make_color_mask_vectorized(arr, red_spec, exclude_region=legend_exclude)

        blue_trace = trace_curve(blue_mask, bbox, direction='down')
        red_trace = trace_curve(red_mask, bbox, direction='down')

        blue_coords = smart_extract(blue_trace, bbox, panel_axis, direction='down')
        red_coords = smart_extract(red_trace, bbox, panel_axis, direction='down')

        results.append({'label': panel_labels[0], 'color': 'blue', 'coordinates': blue_coords})
        results.append({'label': panel_labels[1], 'color': 'red', 'coordinates': red_coords})

    # For the combined axis, use panel_a's axis for x_max (the larger one)
    combined_axis = {'x_min': 0, 'x_max': max(axis['panel_a']['x_max'], axis['panel_b']['x_max']),
                     'y_min': 0, 'y_max': 1.0}

    return {
        'image': str(img_path),
        'bbox': [0, 0, 0, 0],  # Not meaningful for multi-panel
        'axis': combined_axis,
        'arms': results
    }


def extract_tiny_2arm(img_path, axis, labels):
    """Tiny image (200x150) — upscale 4x for pixel resolution."""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img.width * 4, img.height * 4), Image.LANCZOS)
    arr = np.array(img)
    bbox = detect_bbox_safe(arr)
    left, top, right, bottom = bbox

    legend_exclude = (left, int(top + (bottom - top) * 0.55), int(left + (right - left) * 0.4), bottom)

    blue_spec = {
        'hue_range': (0.5, 0.72),
        'sat_min': 0.2,
        'light_range': (0.15, 0.7)
    }
    red_spec = {
        'hue_range': (0.95, 0.1),
        'sat_min': 0.2,
        'light_range': (0.15, 0.7)
    }

    blue_mask = make_color_mask_vectorized(arr, blue_spec, exclude_region=legend_exclude)
    red_mask = make_color_mask_vectorized(arr, red_spec, exclude_region=legend_exclude)

    blue_trace = trace_curve(blue_mask, bbox, direction='down')
    red_trace = trace_curve(red_mask, bbox, direction='down')

    blue_coords = smart_extract(blue_trace, bbox, axis, direction='down')
    red_coords = smart_extract(red_trace, bbox, axis, direction='down')

    return {
        'image': str(img_path),
        'bbox': list(bbox),
        'axis': axis,
        'arms': [
            {'label': labels[0], 'color': 'blue', 'coordinates': blue_coords},
            {'label': labels[1], 'color': 'red', 'coordinates': red_coords},
        ]
    }


def extract_bw_2arm(img_path, axis, labels):
    """
    Black-and-white 2-arm plot: both curves are black, distinguished by
    solid vs dashed line style. Strategy:
    1. Create a dark-pixel mask, exclude border lines and small components.
    2. Trace column by column: spatial sort until arms separate, then
       nearest-to-previous to handle crossings.
    3. After tracing, swap arms if needed so arm 0 = solid (more pixels).
    """
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
    arr = np.array(img)
    bbox = detect_bbox_safe(arr)
    left, top, right, bottom = bbox
    plot_h = bottom - top

    # Create dark pixel mask
    gray = np.mean(arr[:, :, :3], axis=2)
    dark_mask = gray < 100

    # Restrict to plot area — use larger margin (15px after 2x upscale)
    # to clear axis border lines which typically extend ~10px into the bbox.
    curve_mask = np.zeros_like(dark_mask)
    margin = 15
    curve_mask[top+margin:bottom-margin, left+margin:right-margin] = \
        dark_mask[top+margin:bottom-margin, left+margin:right-margin]

    # Exclude horizontal border/grid lines: rows where dark pixels span
    # a large fraction of the plot width are axis lines, not curves.
    plot_w = right - left - 2 * margin
    for row in range(top + margin, bottom - margin):
        row_dark = np.sum(curve_mask[row, left+margin:right-margin])
        if row_dark > plot_w * 0.3:
            curve_mask[row, left+margin:right-margin] = False

    # Exclude legend region
    leg_y1 = int(top + (bottom - top) * 0.65)
    leg_x2 = int(left + (right - left) * 0.35)
    curve_mask[leg_y1:bottom, left:leg_x2] = False

    # Bug 3 fix: Filter annotation text using connected components.
    # Small connected components (< threshold) are text/arrows.
    try:
        from scipy import ndimage
        labeled, n_features = ndimage.label(curve_mask)
        if n_features > 0:
            component_sizes = ndimage.sum(curve_mask, labeled, range(1, n_features + 1))
            min_component_size = 200
            for i, size in enumerate(component_sizes):
                if size < min_component_size:
                    curve_mask[labeled == (i + 1)] = False
    except ImportError:
        pass

    MAX_JUMP_PX = 40
    MIN_CONFIDENT_GAP = 8  # min cluster gap to consider arms "separated"

    traces = [[], []]
    prev_positions = [None, None]
    real_pixels = [0, 0]
    arms_established = False  # one-way flag: once arms separate, stay in nearest mode

    for col in range(left + margin, right - margin + 1):
        col_pixels = np.where(curve_mask[top:bottom+1, col])[0] + top
        if len(col_pixels) == 0:
            for arm in range(2):
                if prev_positions[arm] is not None:
                    traces[arm].append((col, prev_positions[arm]))
            continue

        # Cluster pixels (gap > 4px = new cluster)
        clusters = []
        cluster_start = col_pixels[0]
        for i in range(1, len(col_pixels)):
            if col_pixels[i] - col_pixels[i-1] > 4:
                clusters.append(cluster_start)
                cluster_start = col_pixels[i]
        clusters.append(cluster_start)

        if len(clusters) >= 2:
            clusters_sorted = sorted(clusters)
            gap = clusters_sorted[-1] - clusters_sorted[0]

            # Once arms have separated, stay in nearest-to-previous mode
            if not arms_established and all(p is not None for p in prev_positions):
                if abs(prev_positions[0] - prev_positions[1]) >= MIN_CONFIDENT_GAP:
                    arms_established = True

            if arms_established:
                # Bug 2 fix: nearest-to-previous to handle crossings.
                # No MAX_JUMP_PX here: with 2 clusters, both are real curves
                # (text is already filtered by CC analysis). Stale anchors
                # from merged regions must be allowed to catch up.
                c0 = min(clusters, key=lambda c: abs(c - prev_positions[0]))
                remaining = [c for c in clusters if c != c0]
                c1 = min(remaining, key=lambda c: abs(c - prev_positions[1])) if remaining else c0

                real_pixels[0] += 1
                real_pixels[1] += 1

                traces[0].append((col, c0))
                traces[1].append((col, c1))

                # Update anchors when gap is confident
                if gap > MIN_CONFIDENT_GAP:
                    prev_positions[0] = c0
                    prev_positions[1] = c1
            else:
                # Arms not yet separated — use spatial sort (top=arm0)
                prev_positions[0] = clusters_sorted[0]
                prev_positions[1] = clusters_sorted[-1]
                traces[0].append((col, clusters_sorted[0]))
                traces[1].append((col, clusters_sorted[-1]))
                real_pixels[0] += 1
                real_pixels[1] += 1
        elif len(clusters) == 1:
            c = clusters[0]
            if all(p is not None for p in prev_positions):
                d0 = abs(c - prev_positions[0])
                d1 = abs(c - prev_positions[1])

                # Bug 3 fix: skip if jump too large
                if min(d0, d1) > MAX_JUMP_PX:
                    for arm in range(2):
                        traces[arm].append((col, prev_positions[arm]))
                    continue

                if d0 <= d1:
                    prev_positions[0] = c
                    traces[0].append((col, c))
                    traces[1].append((col, prev_positions[1]))
                    real_pixels[0] += 1
                else:
                    prev_positions[1] = c
                    traces[1].append((col, c))
                    traces[0].append((col, prev_positions[0]))
                    real_pixels[1] += 1
            else:
                for arm in range(2):
                    prev_positions[arm] = c
                    traces[arm].append((col, c))

    # Bug 1 fix: Solid line = continuous = higher genuine pixel count.
    # If arm 0 has fewer genuine pixels, swap so arm 0 = solid.
    if real_pixels[0] < real_pixels[1]:
        traces[0], traces[1] = traces[1], traces[0]

    # Extract coordinates
    arm0_coords = smart_extract(traces[0], bbox, axis, direction='down')
    arm1_coords = smart_extract(traces[1], bbox, axis, direction='down')

    return {
        'image': str(img_path),
        'bbox': list(bbox),
        'axis': axis,
        'arms': [
            {'label': labels[0], 'color': 'black_solid', 'coordinates': arm0_coords},
            {'label': labels[1], 'color': 'black_dashed', 'coordinates': arm1_coords},
        ]
    }


def extract_three_similar(img_path, axis, labels):
    """
    Three-arm plot with nearly identical blue colors. Color separation is
    impossible — all three arms use the same RGB. Strategy:
    1. Create a single blue mask for all curves.
    2. For each column, find all blue pixel clusters (vertical groups).
    3. Since the curves don't cross and are vertically separated (Arm A on top,
       Arm C on bottom), assign clusters to arms by vertical position.
    """
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
    arr = np.array(img)
    bbox = detect_bbox_refined(arr)
    left, top, right, bottom = bbox

    # Create blue mask — these are all the same blue ~(38, 89, 166)
    # Use a broad filter to catch anti-aliased edge pixels too
    blue_spec = {
        'rgb_test_vec': lambda r, g, b: (b > 120) & (r < 80) & (g < 130) & (b > g)
    }
    blue_mask = make_color_mask_vectorized(arr, blue_spec)

    # Exclude legend area
    leg_y1 = int(top + (bottom - top) * 0.65)
    leg_x2 = int(left + (right - left) * 0.3)
    blue_mask[leg_y1:bottom, left:leg_x2] = False

    # For each column, find clusters of blue pixels and assign to arms
    # by vertical position (topmost cluster = Arm A, middle = Arm B, bottom = Arm C)
    n_arms = 3
    traces = [[] for _ in range(n_arms)]

    # Track previous positions for continuity
    prev_positions = [None] * n_arms

    for col in range(left, right + 1):
        col_pixels = np.where(blue_mask[top:bottom+1, col])[0] + top
        if len(col_pixels) == 0:
            # Carry forward previous positions
            for arm in range(n_arms):
                if prev_positions[arm] is not None:
                    traces[arm].append((col, prev_positions[arm]))
            continue

        # Cluster the pixels (group consecutive pixels)
        clusters = []
        cluster_start = col_pixels[0]
        for i in range(1, len(col_pixels)):
            if col_pixels[i] - col_pixels[i-1] > 5:  # gap > 5px = new cluster
                cluster_center = (cluster_start + col_pixels[i-1]) / 2.0
                clusters.append(int(cluster_center))
                cluster_start = col_pixels[i]
        cluster_center = (cluster_start + col_pixels[-1]) / 2.0
        clusters.append(int(cluster_center))

        # Assign clusters to arms
        if len(clusters) >= n_arms:
            # Sort by y position (top to bottom = arm 0, 1, 2)
            clusters_sorted = sorted(clusters)
            for arm in range(n_arms):
                pos = clusters_sorted[arm]
                prev_positions[arm] = pos
                traces[arm].append((col, pos))
        elif len(clusters) == 2:
            # Two clusters visible — figure out which arms they belong to
            # by proximity to previous positions
            c_sorted = sorted(clusters)
            if all(p is not None for p in prev_positions):
                # Assign each cluster to nearest previous arm
                assigned = [False] * len(c_sorted)
                for arm in range(n_arms):
                    best_dist = float('inf')
                    best_ci = -1
                    for ci, c in enumerate(c_sorted):
                        if not assigned[ci]:
                            d = abs(c - prev_positions[arm])
                            if d < best_dist:
                                best_dist = d
                                best_ci = ci
                    if best_ci >= 0 and best_dist < 50:
                        assigned[best_ci] = True
                        prev_positions[arm] = c_sorted[best_ci]
                        traces[arm].append((col, c_sorted[best_ci]))
                    else:
                        # Carry forward
                        traces[arm].append((col, prev_positions[arm]))
            else:
                # Early in the curve, just assign top to arm 0, bottom to arm 1
                # (arm 2 may not have started diverging yet)
                for ci, c in enumerate(c_sorted):
                    prev_positions[ci] = c
                    traces[ci].append((col, c))
                if prev_positions[2] is not None:
                    traces[2].append((col, prev_positions[2]))
        elif len(clusters) == 1:
            # Single cluster — likely curves are overlapping, assign to nearest arm
            c = clusters[0]
            if all(p is not None for p in prev_positions):
                best_arm = min(range(n_arms), key=lambda a: abs(c - prev_positions[a]))
                prev_positions[best_arm] = c
                traces[best_arm].append((col, c))
                for arm in range(n_arms):
                    if arm != best_arm and prev_positions[arm] is not None:
                        traces[arm].append((col, prev_positions[arm]))
            else:
                # Very early — curves start at same point
                for arm in range(n_arms):
                    prev_positions[arm] = c
                    traces[arm].append((col, c))

    # Extract coordinates from traces
    arms = []
    for arm in range(n_arms):
        coords = smart_extract(traces[arm], bbox, axis, direction='down')
        arms.append({'label': labels[arm], 'color': 'blue', 'coordinates': coords})

    return {
        'image': str(img_path),
        'bbox': list(bbox),
        'axis': axis,
        'arms': arms
    }


def extract_tiny_blurry_2arm(img_path, axis, labels):
    """Tiny + blurry: 4x upscale with wide color tolerances for washed-out colors."""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img.width * 4, img.height * 4), Image.LANCZOS)
    arr = np.array(img)
    bbox = detect_bbox_safe(arr)
    left, top, right, bottom = bbox

    legend_exclude = (left, int(top + (bottom - top) * 0.55), int(left + (right - left) * 0.4), bottom)

    # Blurry images have very washed-out colors (lightness 0.85-0.95)
    # Use relative channel detection instead of HSL
    blue_spec = {
        'rgb_test_vec': lambda r, g, b: (b > r + 3) & (b > g) & ((r + g + b) < 720) & ((r + g + b) > 200)
    }
    red_spec = {
        'rgb_test_vec': lambda r, g, b: (r > b + 3) & (r > g) & ((r + g + b) < 720) & ((r + g + b) > 200)
    }

    blue_mask = make_color_mask_vectorized(arr, blue_spec, exclude_region=legend_exclude)
    red_mask = make_color_mask_vectorized(arr, red_spec, exclude_region=legend_exclude)

    blue_trace = trace_curve(blue_mask, bbox, direction='down')
    red_trace = trace_curve(red_mask, bbox, direction='down')

    blue_coords = smart_extract(blue_trace, bbox, axis, direction='down')
    red_coords = smart_extract(red_trace, bbox, axis, direction='down')

    return {
        'image': str(img_path),
        'bbox': list(bbox),
        'axis': axis,
        'arms': [
            {'label': labels[0], 'color': 'blue', 'coordinates': blue_coords},
            {'label': labels[1], 'color': 'red', 'coordinates': red_coords},
        ]
    }


def extract_tiny_bw_2arm(img_path, axis, labels):
    """Tiny + BW: 4x upscale, then BW spatial separation."""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img.width * 4, img.height * 4), Image.LANCZOS)
    arr = np.array(img)
    bbox = detect_bbox_safe(arr)
    left, top, right, bottom = bbox

    gray = np.mean(arr[:, :, :3], axis=2)
    dark_mask = gray < 120

    curve_mask = np.zeros_like(dark_mask)
    margin = 6
    curve_mask[top+margin:bottom-margin, left+margin:right-margin] = \
        dark_mask[top+margin:bottom-margin, left+margin:right-margin]

    leg_y1 = int(top + (bottom - top) * 0.55)
    leg_x2 = int(left + (right - left) * 0.45)
    curve_mask[leg_y1:bottom, left:leg_x2] = False

    traces = [[], []]
    prev_positions = [None, None]

    for col in range(left + margin, right - margin + 1):
        col_pixels = np.where(curve_mask[top:bottom+1, col])[0] + top
        if len(col_pixels) == 0:
            for arm in range(2):
                if prev_positions[arm] is not None:
                    traces[arm].append((col, prev_positions[arm]))
            continue

        clusters = []
        cluster_start = col_pixels[0]
        for i in range(1, len(col_pixels)):
            if col_pixels[i] - col_pixels[i-1] > 4:
                clusters.append(cluster_start)
                cluster_start = col_pixels[i]
        clusters.append(cluster_start)

        if len(clusters) >= 2:
            clusters_sorted = sorted(clusters)
            prev_positions[0] = clusters_sorted[0]
            prev_positions[1] = clusters_sorted[-1]
            traces[0].append((col, clusters_sorted[0]))
            traces[1].append((col, clusters_sorted[-1]))
        elif len(clusters) == 1:
            c = clusters[0]
            if all(p is not None for p in prev_positions):
                d0 = abs(c - prev_positions[0])
                d1 = abs(c - prev_positions[1])
                if d0 <= d1:
                    prev_positions[0] = c
                    traces[0].append((col, c))
                    traces[1].append((col, prev_positions[1]))
                else:
                    prev_positions[1] = c
                    traces[1].append((col, c))
                    traces[0].append((col, prev_positions[0]))
            else:
                for arm in range(2):
                    prev_positions[arm] = c
                    traces[arm].append((col, c))

    arm0_coords = smart_extract(traces[0], bbox, axis, direction='down')
    arm1_coords = smart_extract(traces[1], bbox, axis, direction='down')

    return {
        'image': str(img_path),
        'bbox': list(bbox),
        'axis': axis,
        'arms': [
            {'label': labels[0], 'color': 'black_solid', 'coordinates': arm0_coords},
            {'label': labels[1], 'color': 'black_dashed', 'coordinates': arm1_coords},
        ]
    }


def extract_dark_2arm(img_path, axis, labels):
    """Dark/blurry/JPEG degraded image.
    Strategy: enhance contrast uniformly (preserve color ratios), then use
    relative-channel detection (blue channel dominant vs red channel dominant)
    since absolute HSL thresholds fail on dark images."""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
    arr = np.array(img).astype(float)

    # Uniform contrast stretch: scale all channels together to preserve color ratios
    overall_lo = np.percentile(arr, 3)
    overall_hi = np.percentile(arr, 97)
    if overall_hi - overall_lo > 10:
        arr = np.clip((arr - overall_lo) / (overall_hi - overall_lo) * 255, 0, 255)
    arr = arr.astype(np.uint8)

    bbox = detect_bbox_safe(arr)
    left, top, right, bottom = bbox

    # If bbox is degenerate, try with original image
    if bottom - top < 30 or right - left < 30:
        arr_orig = np.array(img.resize((img.width, img.height), Image.LANCZOS))
        arr_orig = np.array(img)
        # Just rescale the upscaled but without contrast
        arr2 = np.array(img.resize((img.width, img.height), Image.LANCZOS))
        arr = np.array(Image.open(img_path).convert('RGB').resize(
            (Image.open(img_path).width * 2, Image.open(img_path).height * 2), Image.LANCZOS))
        bbox = detect_bbox_safe(arr, dark_threshold=50)
        left, top, right, bottom = bbox

        # Re-apply uniform contrast
        arr = arr.astype(float)
        overall_lo = np.percentile(arr, 3)
        overall_hi = np.percentile(arr, 97)
        if overall_hi - overall_lo > 10:
            arr = np.clip((arr - overall_lo) / (overall_hi - overall_lo) * 255, 0, 255)
        arr = arr.astype(np.uint8)

    legend_exclude = (left, int(top + (bottom - top) * 0.55), int(left + (right - left) * 0.4), bottom)

    # Use relative color detection: blue-ish means B > R and B > G;
    # red-ish means R > B and R > G. Use a minimum difference threshold.
    blue_spec = {
        'rgb_test_vec': lambda r, g, b: (b > r + 8) & (b > g) & (b > 80)
    }
    red_spec = {
        'rgb_test_vec': lambda r, g, b: (r > b + 8) & (r > g) & (r > 80)
    }

    blue_mask = make_color_mask_vectorized(arr, blue_spec, exclude_region=legend_exclude)
    red_mask = make_color_mask_vectorized(arr, red_spec, exclude_region=legend_exclude)

    blue_trace = trace_curve(blue_mask, bbox, direction='down')
    red_trace = trace_curve(red_mask, bbox, direction='down')

    blue_coords = smart_extract(blue_trace, bbox, axis, direction='down')
    red_coords = smart_extract(red_trace, bbox, axis, direction='down')

    return {
        'image': str(img_path),
        'bbox': list(bbox),
        'axis': axis,
        'arms': [
            {'label': labels[0], 'color': 'blue', 'coordinates': blue_coords},
            {'label': labels[1], 'color': 'red', 'coordinates': red_coords},
        ]
    }


def extract_tiny_four_arms(img_path, axis, labels):
    """Tiny 4-arm plot: 4x upscale, 4-color extraction."""
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img.width * 4, img.height * 4), Image.LANCZOS)
    arr = np.array(img)
    bbox = detect_bbox_safe(arr)
    left, top, right, bottom = bbox

    legend_exclude = (left, int(top + (bottom - top) * 0.55), int(left + (right - left) * 0.3), bottom)

    blue_spec = {
        'hue_range': (0.55, 0.72),
        'sat_min': 0.2,
        'light_range': (0.15, 0.65)
    }
    red_spec = {
        'hue_range': (0.95, 0.08),
        'sat_min': 0.25,
        'light_range': (0.15, 0.6)
    }
    green_spec = {
        'hue_range': (0.25, 0.45),
        'sat_min': 0.2,
        'light_range': (0.1, 0.55)
    }
    purple_spec = {
        'hue_range': (0.72, 0.88),
        'sat_min': 0.15,
        'light_range': (0.1, 0.55)
    }

    blue_mask = make_color_mask_vectorized(arr, blue_spec, exclude_region=legend_exclude)
    red_mask = make_color_mask_vectorized(arr, red_spec, exclude_region=legend_exclude)
    green_mask = make_color_mask_vectorized(arr, green_spec, exclude_region=legend_exclude)
    purple_mask = make_color_mask_vectorized(arr, purple_spec, exclude_region=legend_exclude)

    blue_trace = trace_curve(blue_mask, bbox, direction='down')
    red_trace = trace_curve(red_mask, bbox, direction='down')
    green_trace = trace_curve(green_mask, bbox, direction='down')
    purple_trace = trace_curve(purple_mask, bbox, direction='down')

    blue_coords = smart_extract(blue_trace, bbox, axis, direction='down')
    red_coords = smart_extract(red_trace, bbox, axis, direction='down')
    green_coords = smart_extract(green_trace, bbox, axis, direction='down')
    purple_coords = smart_extract(purple_trace, bbox, axis, direction='down')

    return {
        'image': str(img_path),
        'bbox': list(bbox),
        'axis': axis,
        'arms': [
            {'label': labels[0], 'color': 'blue', 'coordinates': blue_coords},
            {'label': labels[1], 'color': 'red', 'coordinates': red_coords},
            {'label': labels[2], 'color': 'green', 'coordinates': green_coords},
            {'label': labels[3], 'color': 'purple', 'coordinates': purple_coords},
        ]
    }


def detect_bbox_tiny_panel(arr):
    """
    Bbox detection for very tiny panels where axis lines have too few dark pixels
    for standard detection. Uses a combination of tick detection and percentage fallback.
    """
    h, w = arr.shape[:2]
    gray = np.mean(arr[:, :, :3], axis=2)

    # Find the bottom axis line: the row with the most dark pixels in the lower half
    dark = gray < 100
    row_counts = np.sum(dark, axis=1)
    bottom_half = row_counts[h//2:]
    if np.max(bottom_half) > w * 0.1:
        bottom = h // 2 + np.argmax(bottom_half > w * 0.1)
        # Find the last such row
        candidates = np.where(row_counts[h//2:] > w * 0.1)[0] + h // 2
        if len(candidates) > 0:
            bottom = candidates[-1]
    else:
        bottom = int(h * 0.85)

    # Find left axis: leftmost column with some dark pixels in the upper portion
    col_counts = np.sum(dark[:bottom, :], axis=0)
    left_candidates = np.where(col_counts > h * 0.05)[0]
    if len(left_candidates) > 0:
        left = left_candidates[0]
    else:
        left = int(w * 0.12)

    # Find right edge
    right_candidates = np.where(dark[bottom, left:])[0] + left
    if len(right_candidates) > 0:
        right = right_candidates[-1]
    else:
        right = int(w * 0.92)

    # Find top: look for where curves/data start (non-background pixels above bottom)
    # For CI plots starting at 0, the top of the bbox is the top of the y-axis
    top_candidates = np.where(col_counts[:left+5] > 2)[0]
    if len(top_candidates) > 0:
        # The topmost dark pixel on the y-axis column
        for y in range(0, bottom):
            if dark[y, left]:
                top = y
                break
        else:
            top = int(h * 0.05)
    else:
        top = int(h * 0.05)

    return (left, top, right, bottom)


def extract_diep_multi_panel(img_path, axis, labels):
    """
    DIEP-like: dual panel, 4 cumulative incidence arms per panel.
    Colors: green, blue, red, purple in each panel.
    Curves go UP (cumulative incidence).
    """
    img = Image.open(img_path).convert('RGB')
    w, h = img.size

    mid_x = w // 2
    panel_a_img = img.crop((0, 0, mid_x, h))
    panel_b_img = img.crop((mid_x, 0, w, h))

    color_specs = {
        'green': {
            'hue_range': (0.25, 0.45),
            'sat_min': 0.15,
            'light_range': (0.1, 0.65)
        },
        'blue': {
            'hue_range': (0.55, 0.72),
            'sat_min': 0.15,
            'light_range': (0.15, 0.65)
        },
        'red': {
            'hue_range': (0.95, 0.08),
            'sat_min': 0.20,
            'light_range': (0.15, 0.65)
        },
        'purple': {
            'hue_range': (0.72, 0.88),
            'sat_min': 0.10,
            'light_range': (0.1, 0.65)
        },
    }

    color_order = ['green', 'blue', 'red', 'purple']

    results = []
    for panel_idx, (panel_img, panel_key) in enumerate([
        (panel_a_img, 'panel_a'),
        (panel_b_img, 'panel_b'),
    ]):
        # Upscale 4x since the panels are tiny (~300x250 each)
        panel_img = panel_img.resize((panel_img.width * 4, panel_img.height * 4), Image.LANCZOS)
        arr = np.array(panel_img)

        # Use the tiny-panel bbox detector
        bbox = detect_bbox_tiny_panel(arr)
        left, top, right, bottom = bbox

        # Validate and fall back if degenerate
        if bottom - top < 50 or right - left < 50:
            ph, pw = arr.shape[:2]
            left = int(pw * 0.15)
            top = int(ph * 0.05)
            right = int(pw * 0.90)
            bottom = int(ph * 0.82)
            bbox = (left, top, right, bottom)

        panel_axis = axis[panel_key]

        # Legend in top-left for CI curves (curves start at bottom-left)
        legend_exclude = (left, top, int(left + (right - left) * 0.30), int(top + (bottom - top) * 0.40))

        for ci, cname in enumerate(color_order):
            mask = make_color_mask_vectorized(arr, color_specs[cname], exclude_region=legend_exclude)
            trace = trace_curve(mask, bbox, direction='up')
            coords = smart_extract(trace, bbox, panel_axis, direction='up')

            arm_idx = panel_idx * 4 + ci
            results.append({
                'label': labels[arm_idx],
                'color': cname,
                'coordinates': coords,
            })

    combined_axis = {
        'x_min': 0,
        'x_max': max(axis['panel_a']['x_max'], axis['panel_b']['x_max']),
        'y_min': 0.0,
        'y_max': max(axis['panel_a']['y_max'], axis['panel_b']['y_max']),
    }

    return {
        'image': str(img_path),
        'bbox': [0, 0, 0, 0],
        'axis': combined_axis,
        'arms': results,
    }


def annotate_multi_panel(img_path, extraction, out_path, plot_name=None):
    """Special annotation for multi-panel."""
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    w, h = img.size
    mid_x = w // 2

    colors_list = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']

    # Look up axis config from the plot name, fallback to edge_multi_panel
    if plot_name and plot_name in PLOT_CONFIGS:
        axis_config = PLOT_CONFIGS[plot_name]['axis']
        n_labels = len(PLOT_CONFIGS[plot_name]['labels'])
    else:
        axis_config = PLOT_CONFIGS['edge_multi_panel']['axis']
        n_labels = 4
    arms_per_panel = n_labels // 2

    for arm_idx, arm in enumerate(extraction['arms']):
        color = colors_list[arm_idx % len(colors_list)]

        if arm_idx < arms_per_panel:
            # Panel A
            panel_axis = axis_config['panel_a']
            offset_x = 0
            panel_w = mid_x
        else:
            # Panel B
            panel_axis = axis_config['panel_b']
            offset_x = mid_x
            panel_w = w - mid_x

        for coord in arm['coordinates']:
            t, s = coord['t'], coord['s']
            # Rough pixel mapping (we don't have exact bbox for annotation)
            px = offset_x + int(panel_w * 0.15) + (t - panel_axis['x_min']) / (panel_axis['x_max'] - panel_axis['x_min']) * (panel_w * 0.75)
            py = int(h * 0.08) + (panel_axis['y_max'] - s) / (panel_axis['y_max'] - panel_axis['y_min']) * (h * 0.78)

            r = 2
            draw.ellipse([px - r, py - r, px + r, py + r], outline=color, width=1)

    img.save(out_path)


# ─── Main pipeline ───

def run_extraction(plot_name):
    """Run extraction for a single plot."""
    config = PLOT_CONFIGS[plot_name]
    img_path = SYNTH_DIR / f'{plot_name}.png'
    truth_path = SYNTH_DIR / f'{plot_name}_truth.json'
    out_dir = BENCH_DIR / plot_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'  Extracting: {plot_name}')
    print(f'{"="*60}')

    plot_type = config['type']
    axis = config['axis']
    labels = config['labels']

    if plot_type == 'standard_2arm':
        extraction = extract_standard_2arm(img_path, axis, labels)
    elif plot_type == 'high_survival':
        extraction = extract_high_survival(img_path, axis, labels)
    elif plot_type == 'cumulative_incidence':
        extraction = extract_cumulative_incidence(img_path, axis, labels)
    elif plot_type == 'four_arms':
        extraction = extract_four_arms(img_path, axis, labels)
    elif plot_type == 'ci_shading':
        extraction = extract_ci_shading(img_path, axis, labels)
    elif plot_type == 'small_dense':
        extraction = extract_small_dense(img_path, axis, labels)
    elif plot_type == 'multi_panel':
        extraction = extract_multi_panel(img_path, axis, labels)
    elif plot_type == 'bw_2arm':
        extraction = extract_bw_2arm(img_path, axis, labels)
    elif plot_type == 'three_similar':
        extraction = extract_three_similar(img_path, axis, labels)
    elif plot_type == 'tiny_2arm':
        extraction = extract_tiny_2arm(img_path, axis, labels)
    elif plot_type == 'tiny_blurry_2arm':
        extraction = extract_tiny_blurry_2arm(img_path, axis, labels)
    elif plot_type == 'tiny_bw_2arm':
        extraction = extract_tiny_bw_2arm(img_path, axis, labels)
    elif plot_type == 'dark_2arm':
        extraction = extract_dark_2arm(img_path, axis, labels)
    elif plot_type == 'tiny_four_arms':
        extraction = extract_tiny_four_arms(img_path, axis, labels)
    elif plot_type == 'diep_multi_panel':
        extraction = extract_diep_multi_panel(img_path, axis, labels)
    else:
        raise ValueError(f'Unknown plot type: {plot_type}')

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def deep_convert(obj):
        if isinstance(obj, dict):
            return {k: deep_convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [deep_convert(v) for v in obj]
        return convert_numpy(obj)

    extraction = deep_convert(extraction)

    # Save extraction
    ext_path = out_dir / 'extraction.json'
    with open(ext_path, 'w') as f:
        json.dump(extraction, f, indent=2)
    print(f'  Saved extraction: {ext_path}')

    # Annotate
    ann_path = out_dir / 'annotation.png'
    if plot_type in ('multi_panel', 'diep_multi_panel'):
        annotate_multi_panel(img_path, extraction, ann_path, plot_name=plot_name)
    else:
        annotate_image(img_path, extraction, ann_path)
    print(f'  Saved annotation: {ann_path}')

    # Compute metrics
    with open(truth_path) as f:
        ground_truth = json.load(f)

    # For multi-panel, use per-panel x_max
    if plot_type in ('multi_panel', 'diep_multi_panel'):
        # Compute metrics per arm pair (panel A arms, panel B arms)
        truth_axis_a = ground_truth.get('axis', {}).get('panel_a', {'x_max': 36})
        truth_axis_b = ground_truth.get('axis', {}).get('panel_b', {'x_max': 24})

        # Determine arms-per-panel from config
        n_total_arms = len(config['labels'])
        arms_per_panel = n_total_arms // 2

        # Split into panel pairs and compute separately
        arm_results = []
        for i in range(min(len(extraction['arms']), len(ground_truth['arms']))):
            ext_arm = extraction['arms'][i]
            truth_arm = ground_truth['arms'][i]

            if i < arms_per_panel:
                x_max = truth_axis_a.get('x_max', 36)
            else:
                x_max = truth_axis_b.get('x_max', 24)

            xs_e = np.array([c['t'] for c in ext_arm['coordinates']])
            ys_e = np.array([c['s'] for c in ext_arm['coordinates']])
            xs_t = np.array([c['t'] for c in truth_arm['coordinates']])
            ys_t = np.array([c['s'] for c in truth_arm['coordinates']])

            from shared.metrics import compute_iae as _iae
            iae = _iae(xs_e, ys_e, xs_t, ys_t, x_max)
            arm_results.append({
                'label': ext_arm.get('label', f'Arm {i}'),
                'iae': iae
            })

        mean_iae = np.mean([r['iae'] for r in arm_results])
        metrics = {
            'score': max(0, 1 - mean_iae),
            'iae': float(mean_iae),
            'n_arms': len(arm_results),
            'arms': arm_results,
        }
    else:
        x_max = axis.get('x_max', extraction.get('axis', {}).get('x_max', 21))
        metrics = compute_score(extraction, ground_truth, x_max)

    # Save metrics
    met_path = out_dir / 'metrics.json'
    with open(met_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

    iae = metrics['iae']
    score = metrics.get('score', 1 - iae)
    print(f'  IAE: {iae:.4f}  Score: {score:.4f}')

    for arm in metrics.get('arms', []):
        label = arm.get('label_extracted', arm.get('label', '?'))
        arm_iae = arm.get('iae', '?')
        print(f'    {label}: IAE={arm_iae:.4f}' if isinstance(arm_iae, float) else f'    {label}: IAE={arm_iae}')

    return plot_name, metrics


def main():
    all_plots = list(PLOT_CONFIGS.keys())

    # If specific plot names given as args, only run those
    if len(sys.argv) > 1:
        all_plots = sys.argv[1:]

    summary = {}

    for plot_name in all_plots:
        try:
            name, metrics = run_extraction(plot_name)
            summary[name] = {
                'iae': float(metrics['iae']),
                'score': float(metrics.get('score', 1 - metrics['iae'])),
                'n_arms': metrics.get('n_arms', 0),
            }
        except Exception as e:
            print(f'  ERROR extracting {plot_name}: {e}')
            import traceback
            traceback.print_exc()
            summary[plot_name] = {'iae': None, 'score': None, 'error': str(e)}

    # Save summary
    summary_path = BENCH_DIR / 'summary.json'

    # Compute overall stats
    valid_iaes = [v['iae'] for v in summary.values() if v['iae'] is not None]
    overall = {
        'mean_iae': float(np.mean(valid_iaes)) if valid_iaes else None,
        'median_iae': float(np.median(valid_iaes)) if valid_iaes else None,
        'min_iae': float(np.min(valid_iaes)) if valid_iaes else None,
        'max_iae': float(np.max(valid_iaes)) if valid_iaes else None,
        'n_plots': len(valid_iaes),
    }

    full_summary = {'overall': overall, 'plots': summary}
    with open(summary_path, 'w') as f:
        json.dump(full_summary, f, indent=2)

    print(f'\n{"="*60}')
    print(f'  SUMMARY')
    print(f'{"="*60}')
    print(f'  Mean IAE:   {overall["mean_iae"]:.4f}' if overall['mean_iae'] else '  Mean IAE: N/A')
    print(f'  Median IAE: {overall["median_iae"]:.4f}' if overall['median_iae'] else '  Median IAE: N/A')
    print(f'  Min IAE:    {overall["min_iae"]:.4f}' if overall['min_iae'] else '  Min IAE: N/A')
    print(f'  Max IAE:    {overall["max_iae"]:.4f}' if overall['max_iae'] else '  Max IAE: N/A')
    print(f'  Plots:      {overall["n_plots"]}')
    print()
    for name, data in summary.items():
        iae_str = f'{data["iae"]:.4f}' if data['iae'] is not None else 'ERROR'
        print(f'  {name:35s} IAE={iae_str}')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
