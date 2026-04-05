#!/usr/bin/env python3
"""
Self-correction orchestrator for KM curve extraction.

Uses a CV diagnostic engine to identify extraction errors, then prepares
correction inputs for an LLM subagent (Claude) that writes improved
extraction code — all WITHOUT access to ground truth.

Architecture:
  Phase 1: Run diagnostics on attempt 1 extractions
  Phase 2: Compose correction prompts + save correction inputs to disk
  (The parent agent dispatches a subagent with each correction_input.json)
  Phase 3 (post-subagent): Execute corrected code, save attempt 2, score it

Usage:
  # Phases 1+2: prepare correction inputs
  python self_correct.py synthetic_001

  # Phase 3: after subagent writes corrected code
  python self_correct.py --execute synthetic_001
"""

import json
import sys
import os
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from jalen.benchmark_extract import PLOT_CONFIGS, SYNTH_DIR, BENCH_DIR, annotate_image

try:
    from jalen.diagnostic import run_diagnostic
except ImportError:
    def run_diagnostic(plot_name):
        print(f"  WARNING: diagnostic.py not available, skipping {plot_name}")


# ─── Prompt composition ───

def _sample_coords(coords, max_points=100):
    """Downsample coordinate list to at most max_points, evenly spaced."""
    if len(coords) <= max_points:
        return coords
    step = len(coords) / max_points
    return [coords[int(i * step)] for i in range(max_points)]


def compose_correction_prompt(plot_name, extraction, diagnostic):
    """
    Build the V3 reasoning-first correction prompt for the blind correction agent.

    The agent sees: diagnostic images, attempt 1 coordinates, diagnostic stats.
    It NEVER sees ground truth, IAE, or truth JSON.

    V3 changes over V2:
    - Forces 4 structured sections (image analysis, technique selection,
      diagnostic interpretation, corrections+reasoning) BEFORE code
    - Only shows global stats + worst strips (not all 20)
    - Mentions line styles explicitly for black_solid / black_dashed arms
    - Includes strategy agreement percentage and centroid technique guidance
    """
    parts = []

    # ── Arm color/style info for the prompt ──
    arm_infos = []
    for arm in extraction.get('arms', []):
        color = arm.get('color', 'unknown')
        label = arm.get('label', 'unknown')
        style_note = ''
        if 'dashed' in color:
            style_note = ' (DASHED line — periodic gaps that tracer must bridge, not treat as separate segments)'
        elif 'solid' in color:
            style_note = ' (SOLID line)'
        arm_infos.append({'label': label, 'color': color, 'style_note': style_note})

    # ── System context with V3 4-section reasoning format ──
    parts.append("""You are a KM curve extraction specialist performing blind quality improvement.

You will analyze this Kaplan-Meier survival plot and its extraction diagnostic data.
Output your analysis in 4 clearly-labeled sections, then the corrected code.

You do NOT have access to ground truth data. Use ONLY the image, annotation,
and diagnostic measurements.

\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
SECTION 1 \u2014 IMAGE ANALYSIS
\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
Describe what you observe about this plot:
- How many curves? What colors?
- Line styles: solid, dashed, dotted? (This matters \u2014 dashed lines have periodic
  gaps that the tracer must bridge, not treat as separate segments)
- Image quality: blur, JPEG artifacts, darkness, resolution?
- Annotations, legends, gridlines inside the plot area?
- Dense zones (smooth diagonal) vs steppy zones (clear horizontal segments)?
- Do curves overlap? Where?

\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
SECTION 2 \u2014 TECHNIQUE SELECTION
\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
For each technique, state whether you'll use it and WHY or WHY NOT:

a) Pixel selection strategy:
   - Topmost pixel: good for sharp step-function segments (value IS at top of step)
   - Centroid: good for diagonal/dense zones and blurry images (unbiased under symmetric blur)
   - Gaussian fit: sub-pixel precision by fitting a Gaussian to the perpendicular intensity profile
   - Skeleton (morphological thinning): finds the true centerline, robust to line thickness
   \u2192 Which will you use for which regions, and why?

b) Dashed line handling:
   - Are any curves dashed? If so, how will you bridge gaps?
   - Gap detection: how do you distinguish a dash gap from the curve ending?

c) Color mask strategy:
   - HSL vs RGB thresholds? Loose vs strict?
   - Will legend/annotation text contaminate the mask?

d) What you are specifically NOT doing and why.

Consider using Gaussian fit, skeleton centerline, or edge-midpoint instead of
simple topmost pixel or mean centroid for better precision.

\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
SECTION 3 \u2014 DIAGNOSTIC INTERPRETATION
\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
Based on the diagnostic measurements:
- What is the overall bias PATTERN? (not per-strip numbers, the trend)
- Where do topmost/centroid/skeleton strategies DISAGREE? What does that mean?
- Are there catastrophic regions (arm jumps, annotation contamination)?
- What is the coverage telling you about mask quality?

\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
SECTION 4 \u2014 CORRECTIONS AND REASONING
\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
For each correction you'll make:
- What diagnostic evidence motivated this correction?
- What technique are you applying?
- What is the expected impact?

Then provide the corrected code:

def corrected_extract(image_path, bbox, axis, attempt1_coords):
    # attempt1_coords is a list of arm dicts:
    #   [{"label": "...", "coordinates": [{"t": ..., "s": ...}, ...]}, ...]
    # Apply targeted fixes based on your diagnosis
    # Return same format: {"arms": [{"label": "...", "coordinates": [...]}, ...]}
    ...

Available imports: numpy, PIL (Image, ImageDraw), pathlib, json
bbox = [left, top, right, bottom] in upscaled pixel coordinates.
axis = {"x_min": 0, "x_max": N, "y_min": 0.0, "y_max": 1.0}

Guidelines:
- Steps should start at (0, 1.0) and be monotonically non-increasing
- Use step-function logic: horizontal segments connected by vertical drops
- Each coordinate should have "t" (time) and "s" (survival) keys
- Return ALL arms present in the plot
- Focus corrections on the strips flagged as WARN or ERROR in the diagnostics
- ADJUST the attempt 1 coordinates — do not rewrite extraction from scratch
""")

    # ── Diagnostic image references ──
    parts.append(f"""
=== DIAGNOSTIC IMAGES (view these for visual evidence) ===
  Original plot:     benchmark/{plot_name}/original.png
  Attempt 1 overlay: benchmark/{plot_name}/annotation.png
  Full annotated:    benchmark/{plot_name}/diagnostic/full_annotated.png""")

    num_arms = len(extraction.get('arms', []))
    for i in range(num_arms):
        parts.append(f"""  Arm {i} color mask:  benchmark/{plot_name}/diagnostic/arm{i}_mask.png
  Arm {i} profiles:    benchmark/{plot_name}/diagnostic/arm{i}_profiles.png
  Arm {i} strategies:  benchmark/{plot_name}/diagnostic/arm{i}_strategies.png
  Arm {i} coverage:    benchmark/{plot_name}/diagnostic/arm{i}_coverage.png""")

    # ── Diagnostic summary per arm (global stats + worst strips only) ──
    if 'arms' in diagnostic:
        for arm_diag in diagnostic['arms']:
            # V2 schema: stats at arm level; V1: nested in global_stats
            stats = arm_diag.get('global_stats') or arm_diag
            mb = stats.get('mean_bias_px') or 0.0
            ma = stats.get('mean_asymmetry') or 0.0
            hr = stats.get('overall_hit_rate') or 0.0
            bd = stats.get('bias_direction') or 'unknown'
            mc = stats.get('mean_coverage') or 0.0
            sa = stats.get('strategy_agreement') or 0.0

            # Find matching extraction arm for color/style info
            arm_label = arm_diag['label']
            arm_color = 'unknown'
            style_desc = ''
            for ai in arm_infos:
                if ai['label'] == arm_label:
                    arm_color = ai['color']
                    style_desc = ai['style_note']
                    break

            parts.append(f"""
=== Arm: {arm_label} (color: {arm_color}){style_desc} ===
Global stats:
  Mean bias: {mb:+.1f}px ({bd})
  Mean asymmetry: {ma:.2f} (0=centered, 1=fully one-sided)
  Pixel hit rate: {hr:.2f}
  Mean coverage: {mc:.1f}px
  Strategy agreement: {sa:.1%} (topmost vs centroid within 2px)""")

            # Show only the worst 4 strips (by absolute bias), not all 20
            strips = arm_diag.get('strips', [])
            if strips:
                ranked = sorted(strips, key=lambda s: abs(s.get('bias_px') or 0), reverse=True)
                worst = ranked[:4]
                parts.append(f"\nWorst {len(worst)} strips (of {len(strips)} total):")
                for strip in worst:
                    bias = strip.get('bias_px') or 0.0
                    asym = strip.get('asymmetry') or 0.0
                    verdict = "OK" if abs(bias) < 1.5 else ("WARN" if abs(bias) < 3 else "ERROR")
                    t_range = strip.get('t_range') or [0, 0]
                    c_range = strip.get('col_range') or [0, 0]
                    t_lo, t_hi = t_range[0], t_range[-1]
                    c_lo, c_hi = c_range[0], c_range[-1]
                    parts.append(
                        f"  Strip {strip.get('strip', strip.get('strip_idx', 0)):2d} | "
                        f"t={t_lo:5.1f}-{t_hi:5.1f} | "
                        f"cols {c_lo:4d}-{c_hi:4d} | "
                        f"bias={bias:+5.1f}px | "
                        f"asym={asym:.2f} | {verdict}"
                    )

    # ── Extraction config (no ground truth!) ──
    arm_colors = [a.get('color', 'unknown') for a in extraction.get('arms', [])]
    parts.append(f"""
=== EXTRACTION CONFIG ===
  bbox: {extraction.get('bbox', [])}
  axis: {json.dumps(extraction.get('axis', {}))}
  arms: {len(extraction.get('arms', []))} arms
  arm colors: {arm_colors}
""")

    # ── Attempt 1 coordinates (sampled to keep prompt size reasonable) ──
    parts.append("=== ATTEMPT 1 COORDINATES (adjust these, do not rewrite from scratch) ===")
    for i, arm in enumerate(extraction.get('arms', [])):
        coords = arm.get('coordinates', [])
        n = len(coords)
        sampled = _sample_coords(coords)
        parts.append(
            f"\n  Arm {i} ({arm.get('label', '?')}): "
            f"{n} total points (showing {len(sampled)})"
        )
        for c in sampled:
            parts.append(f"    t={c['t']:.4f}, s={c['s']:.4f}")

    return "\n".join(parts)


# ─── Correction input preparation ───

def prepare_correction_input(plot_name):
    """
    Prepare everything the correction subagent needs, save to disk.

    Produces: benchmark/{plot}/diagnostic/correction_input.json
    containing the prompt, image paths, and diagnostic dir — but
    NEVER ground truth, IAE, or truth file paths.
    """
    config = PLOT_CONFIGS[plot_name]
    bench_dir = BENCH_DIR / plot_name

    # Load attempt 1
    extraction_path = bench_dir / 'extraction.json'
    if not extraction_path.exists():
        raise FileNotFoundError(f"No extraction.json for {plot_name}")
    extraction = json.load(open(extraction_path))

    # Load diagnostic
    diag_path = bench_dir / 'diagnostic' / 'diagnostic.json'
    if not diag_path.exists():
        raise FileNotFoundError(
            f"No diagnostic.json for {plot_name}. Run diagnostics first."
        )
    diagnostic = json.load(open(diag_path))

    # Compose prompt
    prompt = compose_correction_prompt(plot_name, extraction, diagnostic)

    # Image paths (subagent will read these)
    original_img = SYNTH_DIR / f'{plot_name}.png'
    annotation_img = bench_dir / 'annotation.png'
    dashboard_img = bench_dir / 'diagnostic' / 'dashboard.png'

    # Build attempt1 coordinates (sampled for prompt size)
    attempt1_coords = []
    for arm in extraction.get('arms', []):
        attempt1_coords.append({
            'label': arm.get('label', 'unknown'),
            'coordinates': _sample_coords(arm.get('coordinates', [])),
        })

    # Build diagnostic image list
    diag_dir = bench_dir / 'diagnostic'
    num_arms = len(extraction.get('arms', []))
    diagnostic_images = {
        'original': str(original_img),
        'annotation': str(annotation_img),
        'full_annotated': str(diag_dir / 'full_annotated.png'),
    }
    for i in range(num_arms):
        diagnostic_images[f'arm{i}_mask'] = str(diag_dir / f'arm{i}_mask.png')
        diagnostic_images[f'arm{i}_profiles'] = str(diag_dir / f'arm{i}_profiles.png')
        diagnostic_images[f'arm{i}_strategies'] = str(diag_dir / f'arm{i}_strategies.png')
        diagnostic_images[f'arm{i}_coverage'] = str(diag_dir / f'arm{i}_coverage.png')

    # Assemble correction input — NO ground truth references
    correction_input = {
        'plot_name': plot_name,
        'prompt': prompt,
        'prompt_version': 'v3',
        'prompt_sections': [
            'SECTION 1 — IMAGE ANALYSIS',
            'SECTION 2 — TECHNIQUE SELECTION',
            'SECTION 3 — DIAGNOSTIC INTERPRETATION',
            'SECTION 4 — CORRECTIONS AND REASONING',
        ],
        'original_image': str(original_img),
        'annotation_image': str(annotation_img),
        'dashboard_image': str(dashboard_img) if dashboard_img.exists() else None,
        'diagnostic_dir': str(diag_dir),
        'diagnostic_images': diagnostic_images,
        'bbox': extraction.get('bbox', []),
        'axis': extraction.get('axis', {}),
        'arm_colors': [a.get('color', 'unknown') for a in extraction.get('arms', [])],
        'arm_labels': [a.get('label', f'Arm {i}') for i, a in enumerate(extraction.get('arms', []))],
        'attempt1_coords': attempt1_coords,
        'timestamp': datetime.now().isoformat(),
    }

    out_path = bench_dir / 'diagnostic' / 'correction_input.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(correction_input, f, indent=2)

    # Also save the prompt as plain text for easy reading
    prompt_path = bench_dir / 'diagnostic' / 'correction_prompt.txt'
    with open(prompt_path, 'w') as f:
        f.write(prompt)

    return correction_input


# ─── Execution of corrected code ───

def execute_correction(code_str, plot_name, original_extraction):
    """
    Execute the corrected extraction code in a subprocess and return
    the new extraction dict.

    The code_str must define:
      def corrected_extract(image_path, bbox, axis, attempt1_coords) -> dict
    """
    img_path = str(SYNTH_DIR / f'{plot_name}.png')
    bbox = original_extraction.get('bbox', [])
    axis = original_extraction.get('axis', {})

    # Build attempt1_coords from the original extraction
    attempt1_coords = []
    for arm in original_extraction.get('arms', []):
        attempt1_coords.append({
            'label': arm.get('label', 'unknown'),
            'coordinates': arm.get('coordinates', []),
        })

    # Write code to temp file and run isolated
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.py', delete=False, dir='/tmp'
    ) as f:
        f.write(f"""
import sys, json
sys.path.insert(0, '{_REPO}')
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

{code_str}

attempt1_coords = json.loads('''{json.dumps(attempt1_coords)}''')

result = corrected_extract('{img_path}', {bbox}, {json.dumps(axis)}, attempt1_coords)

# Handle numpy types for JSON serialization
import json as _json
class _NumpyEncoder(_json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'item'):
            return obj.item()
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        return super().default(obj)

print(_json.dumps(result, cls=_NumpyEncoder))
""")
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Correction code failed (exit {result.returncode}):\n"
                f"STDERR: {result.stderr[:2000]}"
            )
        output = result.stdout.strip()
        if not output:
            raise RuntimeError("Correction code produced no output")
        return json.loads(output)
    finally:
        os.unlink(tmp_path)


def apply_correction(plot_name, code_str):
    """
    Full attempt 2 pipeline: execute corrected code, save extraction,
    generate annotation, compute metrics (for our eyes only).

    Returns metrics dict or None if no ground truth available.
    """
    config = PLOT_CONFIGS[plot_name]
    bench_dir = BENCH_DIR / plot_name
    attempt2_dir = bench_dir / 'attempt2'
    attempt2_dir.mkdir(parents=True, exist_ok=True)

    # Load attempt 1 for bbox/axis reference
    extraction = json.load(open(bench_dir / 'extraction.json'))

    # Execute corrected code
    new_extraction = execute_correction(code_str, plot_name, extraction)

    # Preserve bbox/axis from attempt 1 if not in new extraction
    if 'bbox' not in new_extraction:
        new_extraction['bbox'] = extraction.get('bbox', [])
    if 'axis' not in new_extraction:
        new_extraction['axis'] = extraction.get('axis', {})
    new_extraction['image'] = extraction.get('image', '')
    new_extraction['attempt'] = 2
    new_extraction['corrected_at'] = datetime.now().isoformat()

    # Save attempt 2 extraction
    with open(attempt2_dir / 'extraction.json', 'w') as f:
        json.dump(new_extraction, f, indent=2)

    # Generate annotation overlay
    img_path = SYNTH_DIR / f'{plot_name}.png'
    annotate_image(img_path, new_extraction, attempt2_dir / 'annotation.png')

    # Compute IAE — WE see this, the subagent NEVER does
    from shared.metrics import compute_score
    truth_path = SYNTH_DIR / f'{plot_name}_truth.json'
    if truth_path.exists():
        truth = json.load(open(truth_path))
        axis_cfg = config.get('axis', {})
        x_max = axis_cfg.get('x_max', axis_cfg.get('panel_a', {}).get('x_max', 24))
        metrics = compute_score(new_extraction, truth, x_max)
        with open(attempt2_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        # Compare with attempt 1
        attempt1_metrics_path = bench_dir / 'metrics.json'
        if attempt1_metrics_path.exists():
            a1 = json.load(open(attempt1_metrics_path))
            metrics['improvement'] = {
                'iae_before': a1.get('iae', None),
                'iae_after': metrics['iae'],
                'iae_delta': (a1.get('iae', 0) - metrics['iae']),
                'improved': bool(metrics['iae'] < a1.get('iae', float('inf'))),
            }
            with open(attempt2_dir / 'metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)

        return metrics

    return None


# ─── CLI entry point ───

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Self-correction orchestrator for KM curve extraction'
    )
    parser.add_argument(
        'plots', nargs='*', default=None,
        help='Plot names to process (default: all in PLOT_CONFIGS)'
    )
    parser.add_argument(
        '--execute', action='store_true',
        help='Phase 3: execute corrected code from correction_code.py files'
    )
    args = parser.parse_args()

    plots = args.plots if args.plots else list(PLOT_CONFIGS.keys())

    if args.execute:
        # Phase 3: apply corrections
        print("Phase 3: Applying corrections...")
        for p in plots:
            code_path = BENCH_DIR / p / 'diagnostic' / 'correction_code.py'
            if not code_path.exists():
                print(f"  {p}: SKIP — no correction_code.py")
                continue
            try:
                code_str = code_path.read_text()
                metrics = apply_correction(p, code_str)
                if metrics:
                    imp = metrics.get('improvement', {})
                    before = imp.get('iae_before', '?')
                    after = imp.get('iae_after', metrics['iae'])
                    improved = imp.get('improved', '?')
                    print(
                        f"  {p}: IAE {before} -> {after:.4f} "
                        f"({'IMPROVED' if improved else 'REGRESSED'})"
                    )
                else:
                    print(f"  {p}: attempt 2 saved (no ground truth for scoring)")
            except Exception as e:
                print(f"  {p}: ERROR — {e}")
        return

    # Phase 1: Run diagnostics
    print("Phase 1: Running diagnostics...")
    for p in plots:
        try:
            bench_dir = BENCH_DIR / p
            if not (bench_dir / 'extraction.json').exists():
                print(f"  {p}: SKIP — no extraction.json")
                continue
            diag_path = bench_dir / 'diagnostic' / 'diagnostic.json'
            if not diag_path.exists():
                run_diagnostic(p)
                if diag_path.exists():
                    print(f"  {p}: diagnostic generated")
                else:
                    print(f"  {p}: diagnostic not generated (diagnostic.py stub?)")
            else:
                print(f"  {p}: diagnostic exists")
        except Exception as e:
            print(f"  {p}: ERROR — {e}")

    # Phase 2: Prepare correction inputs
    print("\nPhase 2: Preparing correction inputs...")
    for p in plots:
        try:
            bench_dir = BENCH_DIR / p
            diag_path = bench_dir / 'diagnostic' / 'diagnostic.json'
            if not diag_path.exists():
                print(f"  {p}: SKIP — no diagnostic data")
                continue
            prepare_correction_input(p)
            print(f"  {p}: correction input ready")
        except Exception as e:
            print(f"  {p}: ERROR — {e}")

    print("\nDone. Correction inputs saved to benchmark/{plot}/diagnostic/correction_input.json")
    print("Next steps:")
    print("  1. Dispatch a subagent per plot with the correction_input.json + images")
    print("  2. Subagent writes corrected code to benchmark/{plot}/diagnostic/correction_code.py")
    print("  3. Run: python self_correct.py --execute [plot_names...]")


if __name__ == '__main__':
    main()
