"""
KM Curve Extraction Diagnostic Engine
Deterministic CV diagnostics for extraction quality assessment — no ground truth needed.

Generates per-arm zoomed strips, residual heatmaps, and bias/asymmetry statistics.
"""

import json
import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jalen.benchmark_extract import (
    make_color_mask_vectorized,
    detect_bbox_safe,
    detect_bbox_refined,
    pixel_to_data,
    PLOT_CONFIGS,
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
    """Map extraction arm color string to a color_spec dict for make_color_mask_vectorized."""
    if color_name in COLOR_SPECS:
        return COLOR_SPECS[color_name]
    return None  # BW or unknown


def _make_arm_mask(arr, color_name):
    """Build binary mask for one arm given the image array and color name."""
    if color_name in BW_COLORS or color_name.startswith('black'):
        gray = np.mean(arr[:, :, :3].astype(float), axis=2)
        return gray < 100
    spec = _get_color_spec(color_name)
    if spec is None:
        # Fallback: try loose HSL-based blue
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
    """Map pixel distance to color: 0→green, 3→yellow, 6+→red."""
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


class DiagnosticEngine:
    """Deterministic extraction quality diagnostic engine."""

    N_STRIPS = 20
    STRIP_HALF_H = 20    # ±20px above/below extracted row
    STRIP_UPSCALE = 6    # nearest-neighbor upscale factor
    HIT_TOLERANCE = 3    # px tolerance for pixel_hit_rate
    ASYM_HALF = 10       # ±10px for asymmetry calculation

    def __init__(self, plot_name):
        self.plot_name = plot_name
        self.out_dir = BENCH_DIR / plot_name / 'diagnostic'

        # Load extraction
        ext_path = BENCH_DIR / plot_name / 'extraction.json'
        if not ext_path.exists():
            raise FileNotFoundError(f'No extraction.json for {plot_name}')
        with open(ext_path) as f:
            self.extraction = json.load(f)

        # Load original image
        img_path = SYNTH_DIR / f'{plot_name}.png'
        if not img_path.exists():
            raise FileNotFoundError(f'No source image for {plot_name}')
        self.orig_img = Image.open(img_path).convert('RGB')

        # Apply same 2x upscale as extraction
        self.img = self.orig_img.resize(
            (self.orig_img.width * 2, self.orig_img.height * 2),
            Image.LANCZOS,
        )
        self.arr = np.array(self.img)

        self.bbox = tuple(self.extraction['bbox'])
        self.axis = self.extraction['axis']

    def run(self):
        """Run full diagnostic pipeline. Returns diagnostic dict."""
        # TODO: skip multi-panel plots for now
        if _is_multi_panel(self.extraction):
            print(f'  SKIP (multi-panel): {self.plot_name}')
            return None

        self.out_dir.mkdir(parents=True, exist_ok=True)

        left, top, right, bottom = self.bbox
        all_arm_stats = []

        for arm_idx, arm in enumerate(self.extraction['arms']):
            color_name = arm.get('color', 'blue')
            coords = arm.get('coordinates', [])
            if not coords:
                continue

            # Build mask
            mask = _make_arm_mask(self.arr, color_name)

            # Generate strips + per-strip measurements
            strip_stats = self._generate_strips(arm_idx, arm, mask)

            # Generate residual heatmap
            self._generate_heatmap(arm_idx, arm, mask)

            # Compute global stats for this arm
            arm_stat = self._compute_arm_stats(arm_idx, arm, strip_stats)
            all_arm_stats.append(arm_stat)

        # Global summary
        diagnostic = {
            'plot_name': self.plot_name,
            'bbox': list(self.bbox),
            'axis': self.axis,
            'arms': all_arm_stats,
        }

        if all_arm_stats:
            biases = [a['mean_bias_px'] for a in all_arm_stats]
            asyms = [a['mean_asymmetry'] for a in all_arm_stats]
            hits = [a['overall_hit_rate'] for a in all_arm_stats]
            diagnostic['global'] = {
                'mean_bias_px': float(np.mean(biases)),
                'bias_direction': 'above' if np.mean(biases) > 0 else 'below',
                'mean_asymmetry': float(np.mean(asyms)),
                'overall_hit_rate': float(np.mean(hits)),
            }

        # Save
        diag_path = self.out_dir / 'diagnostic.json'
        with open(diag_path, 'w') as f:
            json.dump(diagnostic, f, indent=2)
        print(f'  Saved: {diag_path}')

        return diagnostic

    def _generate_strips(self, arm_idx, arm, mask):
        """Generate N_STRIPS zoomed curve-following strips for one arm."""
        coords = arm['coordinates']
        left, top, right, bottom = self.bbox
        h, w = self.arr.shape[:2]

        # Time range from axis
        t_min = self.axis.get('x_min', 0)
        t_max = self.axis['x_max']
        segment_width = (t_max - t_min) / self.N_STRIPS

        strip_stats = []

        for j in range(self.N_STRIPS):
            t_lo = t_min + j * segment_width
            t_hi = t_lo + segment_width
            t_mid = (t_lo + t_hi) / 2

            # Find extracted S at midpoint: interpolate from coordinates
            s_mid = self._interp_s_at_t(coords, t_mid)

            # Convert to pixel coordinates
            col_mid, row_mid = data_to_pixel(t_mid, s_mid, self.bbox, self.axis)
            col_lo, _ = data_to_pixel(t_lo, s_mid, self.bbox, self.axis)
            col_hi, _ = data_to_pixel(t_hi, s_mid, self.bbox, self.axis)

            row_mid = int(round(row_mid))
            col_lo = int(round(max(col_lo, 0)))
            col_hi = int(round(min(col_hi, w - 1)))
            col_mid_int = int(round(col_mid))

            # Crop region: full segment width × ±STRIP_HALF_H
            crop_top = max(row_mid - self.STRIP_HALF_H, 0)
            crop_bot = min(row_mid + self.STRIP_HALF_H, h - 1)
            crop_left = max(col_lo, 0)
            crop_right = min(col_hi, w - 1)

            if crop_right <= crop_left or crop_bot <= crop_top:
                strip_stats.append(self._empty_strip_stat(j, t_lo, t_hi, s_mid))
                continue

            # Extract crop from original image
            crop_img = self.arr[crop_top:crop_bot + 1, crop_left:crop_right + 1].copy()
            crop_mask = mask[crop_top:crop_bot + 1, crop_left:crop_right + 1]

            # Extracted row position within the crop
            ext_row_in_crop = row_mid - crop_top

            # Compute centroid of mask pixels in this crop
            mask_rows = np.where(crop_mask.any(axis=1))[0]
            if len(mask_rows) > 0:
                # Weighted centroid by number of mask pixels per row
                row_counts = crop_mask.sum(axis=1)
                total = row_counts.sum()
                if total > 0:
                    centroid_row = float(np.sum(np.arange(crop_mask.shape[0]) * row_counts) / total)
                else:
                    centroid_row = ext_row_in_crop
            else:
                centroid_row = ext_row_in_crop

            # Bias: extracted - centroid (positive = extracted is below centroid = biased above in data space)
            bias_px = float(ext_row_in_crop - centroid_row)

            # Asymmetry: within ±ASYM_HALF of extracted row
            asym_top = max(ext_row_in_crop - self.ASYM_HALF, 0)
            asym_bot = min(ext_row_in_crop + self.ASYM_HALF, crop_mask.shape[0])
            above_count = int(crop_mask[asym_top:ext_row_in_crop, :].sum())
            below_count = int(crop_mask[ext_row_in_crop:asym_bot, :].sum())
            denom = above_count + below_count
            asymmetry = float((below_count - above_count) / denom) if denom > 0 else 0.0

            # Pixel hit rate: fraction of extracted coords in this strip within HIT_TOLERANCE of mask
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

            # Save strip
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

    def _generate_heatmap(self, arm_idx, arm, mask):
        """Generate residual heatmap for one arm."""
        coords = arm['coordinates']
        left, top, right, bottom = self.bbox

        plot_width = right - left + 1
        heatmap_h = 20
        heatmap = np.zeros((heatmap_h, plot_width, 3), dtype=np.uint8)

        for col_offset in range(plot_width):
            col = left + col_offset

            # Find extracted row at this column via interpolation
            t, _ = pixel_to_data(col, top, self.bbox, self.axis)
            s_ext = self._interp_s_at_t(coords, t)
            _, ext_row = data_to_pixel(t, s_ext, self.bbox, self.axis)
            ext_row = int(round(ext_row))

            # Find nearest mask pixel row in this column
            col_mask = mask[:, col] if 0 <= col < mask.shape[1] else np.zeros(mask.shape[0], dtype=bool)
            mask_rows = np.where(col_mask)[0]

            if len(mask_rows) > 0:
                dists = np.abs(mask_rows - ext_row)
                nearest_dist = float(dists.min())
            else:
                nearest_dist = 6.0  # max out

            rgb = _residual_to_rgb(nearest_dist)
            heatmap[:, col_offset] = rgb

        heatmap_img = Image.fromarray(heatmap)
        # Upscale for visibility
        heatmap_img = heatmap_img.resize(
            (heatmap_img.width * 2, heatmap_img.height * 4),
            Image.NEAREST,
        )
        heatmap_path = self.out_dir / f'arm{arm_idx}_heatmap.png'
        heatmap_img.save(heatmap_path)

    def _compute_arm_stats(self, arm_idx, arm, strip_stats):
        """Compute global stats for one arm from strip measurements."""
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
            'arm_index': arm_idx,
            'label': arm.get('label', f'Arm {arm_idx}'),
            'color': arm.get('color', 'unknown'),
            'mean_bias_px': round(mean_bias, 3),
            'bias_direction': 'above' if mean_bias > 0 else 'below',
            'mean_asymmetry': round(mean_asym, 4),
            'overall_hit_rate': round(overall_hr, 4),
            'strips': strip_stats,
        }

    def _interp_s_at_t(self, coords, t):
        """Step-function interpolation: find S at time t from extracted coordinates."""
        if not coords:
            return 1.0
        # Find the last coordinate with t_coord <= t
        s = coords[0]['s']
        for c in coords:
            if c['t'] <= t:
                s = c['s']
            else:
                break
        return s

    def _compute_hit_rate(self, coords, t_lo, t_hi, mask):
        """Fraction of extracted coords in [t_lo, t_hi] that are within HIT_TOLERANCE of a mask pixel."""
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

            # Check if any mask pixel within HIT_TOLERANCE
            r_lo = max(row - self.HIT_TOLERANCE, 0)
            r_hi = min(row + self.HIT_TOLERANCE + 1, h)
            c_lo = max(col - self.HIT_TOLERANCE, 0)
            c_hi = min(col + self.HIT_TOLERANCE + 1, w)

            if mask[r_lo:r_hi, c_lo:c_hi].any():
                hits += 1
            total += 1

        return hits / total if total > 0 else 0.0

    def _empty_strip_stat(self, j, t_lo, t_hi, s_mid):
        """Return a placeholder stat for a degenerate strip."""
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
    if result:
        g = result.get('global', {})
        print(f'  bias={g.get("mean_bias_px", "?")}px ({g.get("bias_direction", "?")})  '
              f'asym={g.get("mean_asymmetry", "?")}  hit_rate={g.get("overall_hit_rate", "?")}')
    return result


if __name__ == '__main__':
    plots = sys.argv[1:] if len(sys.argv) > 1 else list(PLOT_CONFIGS.keys())
    for p in plots:
        try:
            run_diagnostic(p)
        except Exception as e:
            print(f'  ERROR: {p}: {e}')
