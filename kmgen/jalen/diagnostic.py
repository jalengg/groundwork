"""
KM Curve Extraction Diagnostic Engine V2
Rich root-cause diagnostic signals for self-correction.

Generates per-arm:
  - full_annotated.png: original image with bbox + tick marks
  - arm{i}_mask.png: binary color mask
  - arm{i}_profiles.png: perpendicular profile charts (10 per arm)
  - arm{i}_strategies.png: multi-strategy trace comparison
  - arm{i}_coverage.png: column-wise mask thickness bar chart
  - arm{i}_heatmap.png: residual heatmap
  - arm{i}_strip_*.png: zoomed strips (20 per arm) with centroid+topmost markers
  - diagnostic.json: enhanced schema with strategy agreement + coverage stats
"""

import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jalen.benchmark_extract import (
    make_color_mask_vectorized,
    detect_bbox_safe,
    PLOT_CONFIGS,
    pixel_to_data,
)

_REPO = Path(__file__).resolve().parent.parent
SYNTH_DIR = _REPO / 'shared' / 'synthetic'
BENCH_DIR = _REPO / 'benchmark'

# ─── Color spec lookup ───

COLOR_SPECS = {
    'blue': {
        'rgb_test_vec': lambda r, g, b: (b > 120) & (r < 80) & (b > g),
    },
    'red': {
        'rgb_test_vec': lambda r, g, b: (r > 150) & (b < 100),
    },
    'orange': {
        'rgb_test_vec': lambda r, g, b: (r > 150) & (b < 100),
    },
    'green': {
        'rgb_test_vec': lambda r, g, b: (g > 100) & (r < 100) & (b < 100),
    },
    'purple': {
        'rgb_test_vec': lambda r, g, b: (r > 80) & (b > 80) & (g < 80),
    },
}

BW_COLORS = {'black_solid', 'black_dashed', 'black'}


def _get_color_spec(color_name):
    """Map extraction arm color string to a color_spec dict."""
    if color_name in COLOR_SPECS:
        return COLOR_SPECS[color_name]
    return None


def _make_arm_mask(arr, color_name):
    """Build binary mask for one arm given the image array and color name."""
    if color_name in BW_COLORS or color_name.startswith('black'):
        gray = np.mean(arr[:, :, :3].astype(float), axis=2)
        return gray < 100
    spec = _get_color_spec(color_name)
    if spec is None:
        spec = {
            'hue_range': (0.50, 0.75),
            'sat_min': 0.15,
            'light_range': (0.15, 0.78),
        }
    return make_color_mask_vectorized(arr, spec)


def data_to_pixel(t, s, bbox, axis):
    """Inverse of pixel_to_data: convert data (t, s) to pixel (col, row)."""
    left, top, right, bottom = bbox
    x_min = axis.get('x_min', 0)
    x_max = axis['x_max']
    y_min = axis.get('y_min', 0)
    y_max = axis.get('y_max', 1.0)
    col = left + (t - x_min) / (x_max - x_min) * (right - left)
    row = top + (y_max - s) / (y_max - y_min) * (bottom - top)
    return col, row


def _is_multi_panel(extraction):
    """Check if this extraction is multi-panel (skip for now)."""
    bbox = extraction.get('bbox', [0, 0, 0, 0])
    if bbox == [0, 0, 0, 0]:
        return True
    axis = extraction.get('axis', {})
    if 'panel_a' in axis or 'panel_b' in axis:
        return True
    return False


# ─── Residual heatmap color mapping ───

def _residual_to_rgb(dist):
    """Map pixel distance to color: 0->green, 3->yellow, 6+->red."""
    dist = min(dist, 6.0)
    if dist <= 3.0:
        frac = dist / 3.0
        r = int(255 * frac)
        g = 255
        b = 0
    else:
        frac = (dist - 3.0) / 3.0
        r = 255
        g = int(255 * (1 - frac))
        b = 0
    return (r, g, b)


# ─── Extraction strategies ───

def _strategy_topmost(mask, col):
    """Return topmost (lowest row index) mask pixel in column, or NaN."""
    col_data = mask[:, col]
    rows = np.where(col_data)[0]
    if len(rows) == 0:
        return float('nan')
    return float(rows[0])


def _strategy_centroid(mask, col):
    """Return centroid (mean row) of mask pixels in column, or NaN."""
    col_data = mask[:, col]
    rows = np.where(col_data)[0]
    if len(rows) == 0:
        return float('nan')
    return float(np.mean(rows))


def _strategy_weighted_centroid(arr_gray, mask, col):
    """Return intensity-weighted centroid of mask pixels in column, or NaN.
    Weights: higher weight for darker pixels (lower intensity = more ink)."""
    col_data = mask[:, col]
    rows = np.where(col_data)[0]
    if len(rows) == 0:
        return float('nan')
    intensities = arr_gray[rows, col]
    # Invert: darker pixels get higher weight
    weights = 255.0 - intensities
    total_w = weights.sum()
    if total_w == 0:
        return float(np.mean(rows))
    return float(np.sum(rows * weights) / total_w)


class DiagnosticEngine:
    """Deterministic extraction quality diagnostic engine V2."""

    N_STRIPS = 20
    STRIP_HALF_H = 20
    STRIP_UPSCALE = 6
    HIT_TOLERANCE = 3
    ASYM_HALF = 10
    N_PROFILES = 10
    PROFILE_W = 60
    PROFILE_H = 80
    COVERAGE_H = 40
    STRATEGY_AGREE_TOL = 2  # px tolerance for strategy agreement

    def __init__(self, plot_name):
        self.plot_name = plot_name
        self.out_dir = BENCH_DIR / plot_name / 'diagnostic'

        ext_path = BENCH_DIR / plot_name / 'extraction.json'
        if not ext_path.exists():
            raise FileNotFoundError(f'No extraction.json for {plot_name}')
        with open(ext_path) as f:
            self.extraction = json.load(f)

        img_path = SYNTH_DIR / f'{plot_name}.png'
        if not img_path.exists():
            raise FileNotFoundError(f'No source image for {plot_name}')
        self.orig_img = Image.open(img_path).convert('RGB')

        # Apply same 2x upscale as extraction pipeline
        self.img = self.orig_img.resize(
            (self.orig_img.width * 2, self.orig_img.height * 2),
            Image.LANCZOS,
        )
        self.arr = np.array(self.img)
        self.arr_gray = np.mean(self.arr[:, :, :3].astype(float), axis=2)

        self.bbox = tuple(self.extraction['bbox'])
        self.axis = self.extraction['axis']

    def run(self):
        """Run full diagnostic pipeline. Returns diagnostic dict."""
        # TODO: skip multi-panel plots (bbox=[0,0,0,0]) for now
        if _is_multi_panel(self.extraction):
            print(f'  SKIP (multi-panel): {self.plot_name}')
            return None

        self.out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Full annotated image
        self._generate_full_annotated()

        all_arm_stats = []

        for arm_idx, arm in enumerate(self.extraction['arms']):
            color_name = arm.get('color', 'blue')
            coords = arm.get('coordinates', [])
            if not coords:
                continue

            mask = _make_arm_mask(self.arr, color_name)

            # 1b. Arm mask image
            self._generate_mask_image(arm_idx, mask)

            # 2. Perpendicular profile charts
            self._generate_profiles(arm_idx, arm, mask)

            # 3. Multi-strategy trace comparison
            strategy_agreement = self._generate_strategies(arm_idx, arm, mask)

            # 4. Coverage map
            mean_coverage = self._generate_coverage(arm_idx, arm, mask)

            # 5. Zoomed strips (V1, enhanced with centroid+topmost markers)
            strip_stats = self._generate_strips(arm_idx, arm, mask)

            # 5b. Residual heatmap (V1)
            self._generate_heatmap(arm_idx, arm, mask)

            # 6. Compute arm stats
            arm_stat = self._compute_arm_stats(
                arm_idx, arm, strip_stats, strategy_agreement, mean_coverage
            )
            all_arm_stats.append(arm_stat)

        # Build diagnostic JSON
        diagnostic = {
            'plot_name': self.plot_name,
            'arms': all_arm_stats,
        }

        # Save
        diag_path = self.out_dir / 'diagnostic.json'
        with open(diag_path, 'w') as f:
            json.dump(diagnostic, f, indent=2)
        print(f'  Saved: {diag_path}')

        return diagnostic

    # ─── 1. Full annotated image ───

    def _generate_full_annotated(self):
        """Draw bbox rectangle and extracted tick positions on original image."""
        img_copy = self.img.copy()
        draw = ImageDraw.Draw(img_copy)
        left, top, right, bottom = self.bbox

        # Draw bbox rectangle in yellow
        draw.rectangle([left, top, right, bottom], outline='yellow', width=2)

        # Mark tick positions from extraction coordinates
        for arm in self.extraction['arms']:
            coords = arm.get('coordinates', [])
            for c in coords:
                col, row = data_to_pixel(c['t'], c['s'], self.bbox, self.axis)
                col, row = int(round(col)), int(round(row))
                # Small cross mark
                draw.line([(col - 3, row), (col + 3, row)], fill='red', width=1)
                draw.line([(col, row - 3), (col, row + 3)], fill='red', width=1)

        out_path = self.out_dir / 'full_annotated.png'
        img_copy.save(out_path)

    # ─── 1b. Arm mask image ───

    def _generate_mask_image(self, arm_idx, mask):
        """Save binary mask as white-on-black image."""
        mask_uint8 = (mask.astype(np.uint8) * 255)
        mask_img = Image.fromarray(mask_uint8, mode='L')
        out_path = self.out_dir / f'arm{arm_idx}_mask.png'
        mask_img.save(out_path)

    # ─── 2. Perpendicular profile charts ───

    def _generate_profiles(self, arm_idx, arm, mask):
        """Generate N_PROFILES evenly-spaced perpendicular profile bar charts."""
        coords = arm['coordinates']
        left, top, right, bottom = self.bbox
        h, w = mask.shape

        t_min = self.axis.get('x_min', 0)
        t_max = self.axis['x_max']
        n = self.N_PROFILES

        pw, ph = self.PROFILE_W, self.PROFILE_H
        composite = Image.new('RGB', (pw * n, ph), (30, 30, 30))

        for i in range(n):
            t = t_min + (i + 0.5) / n * (t_max - t_min)
            s_ext = self._interp_s_at_t(coords, t)
            col, row_ext = data_to_pixel(t, s_ext, self.bbox, self.axis)
            col = int(round(col))
            row_ext = int(round(row_ext))

            if col < 0 or col >= w:
                continue

            # Vertical window: +-20px around extracted row
            window_half = 20
            r_lo = max(row_ext - window_half, 0)
            r_hi = min(row_ext + window_half + 1, h)
            col_mask = mask[r_lo:r_hi, col].astype(float)

            # Count mask pixels at each row position
            n_rows = r_hi - r_lo
            if n_rows == 0:
                continue

            # Build bar chart
            chart = Image.new('RGB', (pw, ph), (30, 30, 30))
            chart_draw = ImageDraw.Draw(chart)

            # Draw bars: each row in window -> one bar
            bar_w = max(pw // n_rows, 1)
            for j in range(n_rows):
                val = col_mask[j]
                bar_h = int(val * (ph - 10))  # 0 or full height
                x0 = j * bar_w
                x1 = min(x0 + bar_w - 1, pw - 1)
                if val > 0:
                    chart_draw.rectangle(
                        [x0, ph - bar_h - 1, x1, ph - 1],
                        fill=(200, 200, 200),
                    )

            # Mark extracted position (red line)
            ext_idx = row_ext - r_lo
            ext_x = int(ext_idx * bar_w + bar_w // 2)
            if 0 <= ext_x < pw:
                chart_draw.line([(ext_x, 0), (ext_x, ph - 1)], fill='red', width=1)

            # Mark centroid (green line)
            mask_rows_local = np.where(col_mask > 0)[0]
            if len(mask_rows_local) > 0:
                centroid_idx = float(np.mean(mask_rows_local))
                cent_x = int(centroid_idx * bar_w + bar_w // 2)
                if 0 <= cent_x < pw:
                    chart_draw.line(
                        [(cent_x, 0), (cent_x, ph - 1)], fill='lime', width=1
                    )

            composite.paste(chart, (i * pw, 0))

        out_path = self.out_dir / f'arm{arm_idx}_profiles.png'
        composite.save(out_path)

    # ─── 3. Multi-strategy trace comparison ───

    def _generate_strategies(self, arm_idx, arm, mask):
        """Run three extraction strategies, draw on image crop, return agreement fraction."""
        coords = arm['coordinates']
        left, top, right, bottom = self.bbox
        h, w = mask.shape

        # Crop to bbox region
        crop_left = max(left, 0)
        crop_right = min(right, w - 1)
        crop_top = max(top, 0)
        crop_bottom = min(bottom, h - 1)

        crop = self.arr[crop_top:crop_bottom + 1, crop_left:crop_right + 1].copy()
        crop_img = Image.fromarray(crop)
        draw = ImageDraw.Draw(crop_img)

        crop_w = crop_right - crop_left + 1

        topmost_trace = []
        centroid_trace = []
        weighted_trace = []
        agree_count = 0
        total_count = 0

        for c in range(crop_w):
            abs_col = crop_left + c
            if abs_col >= w:
                continue

            top_r = _strategy_topmost(mask, abs_col)
            cent_r = _strategy_centroid(mask, abs_col)
            wc_r = _strategy_weighted_centroid(self.arr_gray, mask, abs_col)

            topmost_trace.append((c, top_r))
            centroid_trace.append((c, cent_r))
            weighted_trace.append((c, wc_r))

            # Check agreement
            if not (np.isnan(top_r) or np.isnan(cent_r)):
                total_count += 1
                if abs(top_r - cent_r) <= self.STRATEGY_AGREE_TOL:
                    agree_count += 1

        # Draw traces
        def _draw_trace(trace, color):
            pts = [(x, r - crop_top) for x, r in trace if not np.isnan(r)]
            if len(pts) < 2:
                return
            for i in range(len(pts) - 1):
                x0, y0 = pts[i]
                x1, y1 = pts[i + 1]
                # Skip big jumps (different step segments)
                if abs(y1 - y0) > 30:
                    continue
                draw.line([(x0, y0), (x1, y1)], fill=color, width=1)

        _draw_trace(topmost_trace, 'red')
        _draw_trace(centroid_trace, 'lime')
        _draw_trace(weighted_trace, (80, 80, 255))  # blue

        out_path = self.out_dir / f'arm{arm_idx}_strategies.png'
        crop_img.save(out_path)

        return agree_count / total_count if total_count > 0 else 0.0

    # ─── 4. Coverage map ───

    def _generate_coverage(self, arm_idx, arm, mask):
        """Bar chart of mask pixel count per column. Returns mean coverage."""
        left, top, right, bottom = self.bbox
        h, w = mask.shape
        plot_width = min(right, w - 1) - max(left, 0) + 1
        col_start = max(left, 0)

        coverages = []
        ch = self.COVERAGE_H
        bar_img = Image.new('RGB', (plot_width, ch), (0, 0, 0))
        draw = ImageDraw.Draw(bar_img)

        for c in range(plot_width):
            abs_col = col_start + c
            if abs_col >= w:
                count = 0
            else:
                # Count mask pixels only within bbox rows
                r_lo = max(top, 0)
                r_hi = min(bottom, h - 1) + 1
                count = int(mask[r_lo:r_hi, abs_col].sum())
            coverages.append(count)

            # Color coding
            if count == 0:
                color = (255, 0, 0)      # red
            elif count <= 2:
                color = (255, 255, 0)     # yellow
            elif count <= 5:
                color = (0, 200, 0)       # green
            else:
                color = (80, 80, 255)     # blue

            # Bar height proportional to count, max = ch
            bar_h = min(count * 3, ch)
            if bar_h > 0:
                draw.line([(c, ch - bar_h), (c, ch - 1)], fill=color, width=1)
            elif count == 0:
                # Mark zero columns with red dot at bottom
                draw.point((c, ch - 1), fill=(255, 0, 0))

        # Upscale for visibility
        bar_img = bar_img.resize(
            (bar_img.width, bar_img.height * 2), Image.NEAREST
        )
        out_path = self.out_dir / f'arm{arm_idx}_coverage.png'
        bar_img.save(out_path)

        return float(np.mean(coverages)) if coverages else 0.0

    # ─── 5. Zoomed strips (enhanced V1) ───

    def _generate_strips(self, arm_idx, arm, mask):
        """Generate N_STRIPS zoomed strips with centroid + topmost markers."""
        coords = arm['coordinates']
        left, top, right, bottom = self.bbox
        h, w = self.arr.shape[:2]

        t_min = self.axis.get('x_min', 0)
        t_max = self.axis['x_max']
        segment_width = (t_max - t_min) / self.N_STRIPS

        strip_stats = []

        for j in range(self.N_STRIPS):
            t_lo = t_min + j * segment_width
            t_hi = t_lo + segment_width
            t_mid = (t_lo + t_hi) / 2

            s_mid = self._interp_s_at_t(coords, t_mid)
            col_mid, row_mid = data_to_pixel(t_mid, s_mid, self.bbox, self.axis)
            col_lo, _ = data_to_pixel(t_lo, s_mid, self.bbox, self.axis)
            col_hi, _ = data_to_pixel(t_hi, s_mid, self.bbox, self.axis)

            row_mid = int(round(row_mid))
            col_lo = int(round(max(col_lo, 0)))
            col_hi = int(round(min(col_hi, w - 1)))
            col_mid_int = int(round(col_mid))

            crop_top = max(row_mid - self.STRIP_HALF_H, 0)
            crop_bot = min(row_mid + self.STRIP_HALF_H, h - 1)
            crop_left = max(col_lo, 0)
            crop_right = min(col_hi, w - 1)

            if crop_right <= crop_left or crop_bot <= crop_top:
                strip_stats.append(self._empty_strip_stat(j, t_lo, t_hi, s_mid))
                continue

            crop_img = self.arr[crop_top:crop_bot + 1, crop_left:crop_right + 1].copy()
            crop_mask = mask[crop_top:crop_bot + 1, crop_left:crop_right + 1]

            ext_row_in_crop = row_mid - crop_top

            # Centroid
            row_counts = crop_mask.sum(axis=1)
            total = row_counts.sum()
            if total > 0:
                centroid_row = float(
                    np.sum(np.arange(crop_mask.shape[0]) * row_counts) / total
                )
            else:
                centroid_row = float(ext_row_in_crop)

            # Topmost mask row in crop
            mask_rows = np.where(crop_mask.any(axis=1))[0]
            topmost_row = float(mask_rows[0]) if len(mask_rows) > 0 else float(ext_row_in_crop)

            # Bias: extracted - centroid
            bias_px = float(ext_row_in_crop - centroid_row)

            # Asymmetry
            asym_top = max(ext_row_in_crop - self.ASYM_HALF, 0)
            asym_bot = min(ext_row_in_crop + self.ASYM_HALF, crop_mask.shape[0])
            above_count = int(crop_mask[asym_top:ext_row_in_crop, :].sum())
            below_count = int(crop_mask[ext_row_in_crop:asym_bot, :].sum())
            denom = above_count + below_count
            asymmetry = float((below_count - above_count) / denom) if denom > 0 else 0.0

            # Hit rate
            hit_rate = self._compute_hit_rate(coords, t_lo, t_hi, mask)

            # Upscale and annotate
            strip_pil = Image.fromarray(crop_img)
            strip_pil = strip_pil.resize(
                (strip_pil.width * self.STRIP_UPSCALE, strip_pil.height * self.STRIP_UPSCALE),
                Image.NEAREST,
            )
            draw = ImageDraw.Draw(strip_pil)

            # Red circle at extracted position
            ext_x = (col_mid_int - crop_left) * self.STRIP_UPSCALE + self.STRIP_UPSCALE // 2
            ext_y = ext_row_in_crop * self.STRIP_UPSCALE + self.STRIP_UPSCALE // 2
            r = 4
            draw.ellipse([ext_x - r, ext_y - r, ext_x + r, ext_y + r], outline='red', width=2)

            # Green horizontal line at centroid
            centroid_y = int(round(centroid_row)) * self.STRIP_UPSCALE + self.STRIP_UPSCALE // 2
            draw.line(
                [(0, centroid_y), (strip_pil.width - 1, centroid_y)],
                fill='lime', width=1,
            )

            # Cyan horizontal line at topmost
            topmost_y = int(round(topmost_row)) * self.STRIP_UPSCALE + self.STRIP_UPSCALE // 2
            draw.line(
                [(0, topmost_y), (strip_pil.width - 1, topmost_y)],
                fill='cyan', width=1,
            )

            strip_path = self.out_dir / f'arm{arm_idx}_strip_{j:02d}.png'
            strip_pil.save(strip_path)

            stat = {
                'strip': j,
                'bias_px': round(bias_px, 2),
                'asymmetry': round(asymmetry, 4),
                'pixel_hit_rate': round(hit_rate, 4),
                't_range': [round(t_lo, 4), round(t_hi, 4)],
                's_range': [round(s_mid, 6)],
                'col_range': [col_lo, col_hi],
            }
            strip_stats.append(stat)

        return strip_stats

    # ─── 5b. Residual heatmap ───

    def _generate_heatmap(self, arm_idx, arm, mask):
        """Generate residual heatmap for one arm."""
        coords = arm['coordinates']
        left, top, right, bottom = self.bbox
        h, w = mask.shape

        plot_width = min(right, w - 1) - max(left, 0) + 1
        col_start = max(left, 0)
        heatmap_h = 20
        heatmap = np.zeros((heatmap_h, plot_width, 3), dtype=np.uint8)

        for col_offset in range(plot_width):
            col = col_start + col_offset

            t, _ = pixel_to_data(col, top, self.bbox, self.axis)
            s_ext = self._interp_s_at_t(coords, t)
            _, ext_row = data_to_pixel(t, s_ext, self.bbox, self.axis)
            ext_row = int(round(ext_row))

            col_mask = mask[:, col] if 0 <= col < w else np.zeros(h, dtype=bool)
            mask_rows = np.where(col_mask)[0]

            if len(mask_rows) > 0:
                nearest_dist = float(np.abs(mask_rows - ext_row).min())
            else:
                nearest_dist = 6.0

            rgb = _residual_to_rgb(nearest_dist)
            heatmap[:, col_offset] = rgb

        heatmap_img = Image.fromarray(heatmap)
        heatmap_img = heatmap_img.resize(
            (heatmap_img.width * 2, heatmap_img.height * 4), Image.NEAREST
        )
        out_path = self.out_dir / f'arm{arm_idx}_heatmap.png'
        heatmap_img.save(out_path)

    # ─── Stats computation ───

    def _compute_arm_stats(self, arm_idx, arm, strip_stats, strategy_agreement, mean_coverage):
        """Compute global stats for one arm."""
        valid_strips = [s for s in strip_stats if s.get('bias_px') is not None]

        if valid_strips:
            biases = [s['bias_px'] for s in valid_strips]
            asymmetries = [s['asymmetry'] for s in valid_strips]
            hit_rates = [s['pixel_hit_rate'] for s in valid_strips]
            mean_bias = float(np.mean(biases))
            mean_asym = float(np.mean(asymmetries))
            overall_hr = float(np.mean(hit_rates))
        else:
            mean_bias = 0.0
            mean_asym = 0.0
            overall_hr = 0.0

        return {
            'label': arm.get('label', f'Arm {arm_idx}'),
            'mean_bias_px': round(mean_bias, 3),
            'bias_direction': 'above' if mean_bias > 0 else 'below',
            'mean_asymmetry': round(mean_asym, 4),
            'overall_hit_rate': round(overall_hr, 4),
            'mean_coverage': round(mean_coverage, 2),
            'strategy_agreement': round(strategy_agreement, 4),
            'strips': strip_stats,
            'images': {
                'mask': f'arm{arm_idx}_mask.png',
                'profiles': f'arm{arm_idx}_profiles.png',
                'strategies': f'arm{arm_idx}_strategies.png',
                'coverage': f'arm{arm_idx}_coverage.png',
                'heatmap': f'arm{arm_idx}_heatmap.png',
            },
        }

    # ─── Helpers ───

    def _interp_s_at_t(self, coords, t):
        """Step-function interpolation: find S at time t."""
        if not coords:
            return 1.0
        s = coords[0]['s']
        for c in coords:
            if c['t'] <= t:
                s = c['s']
            else:
                break
        return s

    def _compute_hit_rate(self, coords, t_lo, t_hi, mask):
        """Fraction of extracted coords in [t_lo, t_hi] within HIT_TOLERANCE of mask."""
        hits = 0
        total = 0
        h, w = mask.shape

        for c in coords:
            if c['t'] < t_lo or c['t'] > t_hi:
                continue
            col, row = data_to_pixel(c['t'], c['s'], self.bbox, self.axis)
            col = int(round(col))
            row = int(round(row))

            if col < 0 or col >= w or row < 0 or row >= h:
                total += 1
                continue

            r_lo = max(row - self.HIT_TOLERANCE, 0)
            r_hi = min(row + self.HIT_TOLERANCE + 1, h)
            c_lo = max(col - self.HIT_TOLERANCE, 0)
            c_hi = min(col + self.HIT_TOLERANCE + 1, w)

            if mask[r_lo:r_hi, c_lo:c_hi].any():
                hits += 1
            total += 1

        return hits / total if total > 0 else 0.0

    def _empty_strip_stat(self, j, t_lo, t_hi, s_mid):
        """Placeholder stat for a degenerate strip."""
        return {
            'strip': j,
            'bias_px': None,
            'asymmetry': None,
            'pixel_hit_rate': 0.0,
            't_range': [round(t_lo, 4), round(t_hi, 4)],
            's_range': [round(s_mid, 6)],
            'col_range': [0, 0],
        }


def run_diagnostic(plot_name):
    """Run diagnostic for a single plot."""
    print(f'Diagnostic: {plot_name}')
    engine = DiagnosticEngine(plot_name)
    result = engine.run()
    if result and result.get('arms'):
        for a in result['arms']:
            print(f'  {a["label"]}: bias={a["mean_bias_px"]}px ({a["bias_direction"]})  '
                  f'asym={a["mean_asymmetry"]}  hit={a["overall_hit_rate"]}  '
                  f'cov={a["mean_coverage"]}  agree={a["strategy_agreement"]}')
    return result


if __name__ == '__main__':
    plots = sys.argv[1:] if len(sys.argv) > 1 else list(PLOT_CONFIGS.keys())
    for p in plots:
        try:
            run_diagnostic(p)
        except Exception as e:
            print(f'  ERROR: {p}: {e}')
