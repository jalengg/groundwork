"""
KMGen Benchmark Report Generator

Generates an HTML report with:
- Side-by-side comparison: original plot vs annotated extraction
- IAE scores and area-between-curves visualization
- Synthetic data parameters
- Per-arm detailed metrics
- Human annotation checkboxes for visual verification

Usage:
    python report.py [--benchmark-dir /path/to/benchmark] [--output report.html]
"""

import argparse
import base64
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def img_to_base64(path: Path) -> str:
    """Encode an image file as base64 for embedding in HTML."""
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    suffix = path.suffix.lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(suffix.strip("."), "image/png")
    return f"data:{mime};base64,{data}"


def make_area_chart_svg(extraction_json: dict, truth_json: dict, width=600, height=200) -> str:
    """
    Generate an inline SVG showing the area between extracted and truth curves.
    The shaded area IS the IAE — makes the error visually intuitive.
    """
    x_max = truth_json.get("axis", {}).get("x_max", 21)
    y_max = truth_json.get("axis", {}).get("y_max", 1.0)

    n_arms = min(len(extraction_json.get("arms", [])), len(truth_json.get("arms", [])))
    if n_arms == 0:
        return "<p>No arms to compare</p>"

    margin = {"top": 20, "right": 20, "bottom": 30, "left": 40}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]

    def to_svg_x(t):
        return margin["left"] + (t / x_max) * plot_w

    def to_svg_y(s):
        return margin["top"] + (1 - s / y_max) * plot_h

    svgs = []
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    for arm_idx in range(n_arms):
        ext_arm = extraction_json["arms"][arm_idx]
        truth_arm = truth_json["arms"][arm_idx]
        color = colors[arm_idx % len(colors)]
        label = truth_arm.get("label", f"Arm {arm_idx + 1}")

        # Get coordinates
        ext_coords = ext_arm.get("coordinates", ext_arm.get("steps", []))
        truth_coords = truth_arm.get("coordinates", truth_arm.get("steps", []))

        xs_e = np.array([c["t"] for c in ext_coords])
        ys_e = np.array([c["s"] for c in ext_coords])
        xs_t = np.array([c["t"] for c in truth_coords])
        ys_t = np.array([c["s"] for c in truth_coords])

        # Sample both curves at regular intervals for SVG path
        n_samples = 200
        sample_xs = np.linspace(0, x_max, n_samples)

        def interp_step(xs, ys, query):
            indices = np.searchsorted(xs, query, side="right") - 1
            indices = np.clip(indices, 0, len(ys) - 1)
            return ys[indices]

        ext_ys = interp_step(xs_e, ys_e, sample_xs) if len(xs_e) > 0 else np.ones(n_samples)
        truth_ys = interp_step(xs_t, ys_t, sample_xs) if len(xs_t) > 0 else np.ones(n_samples)

        # Build SVG paths
        ext_path = " ".join(f"{'M' if i == 0 else 'L'}{to_svg_x(sample_xs[i]):.1f},{to_svg_y(ext_ys[i]):.1f}" for i in range(n_samples))
        truth_path = " ".join(f"{'M' if i == 0 else 'L'}{to_svg_x(sample_xs[i]):.1f},{to_svg_y(truth_ys[i]):.1f}" for i in range(n_samples))

        # Area between curves (filled polygon)
        area_points = []
        for i in range(n_samples):
            area_points.append(f"{to_svg_x(sample_xs[i]):.1f},{to_svg_y(ext_ys[i]):.1f}")
        for i in range(n_samples - 1, -1, -1):
            area_points.append(f"{to_svg_x(sample_xs[i]):.1f},{to_svg_y(truth_ys[i]):.1f}")

        svg = f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background:#fafafa;border:1px solid #ddd;border-radius:4px;margin:8px 0;">
  <!-- Grid -->
  <line x1="{margin['left']}" y1="{margin['top']}" x2="{margin['left']}" y2="{margin['top']+plot_h}" stroke="#ccc" />
  <line x1="{margin['left']}" y1="{margin['top']+plot_h}" x2="{margin['left']+plot_w}" y2="{margin['top']+plot_h}" stroke="#ccc" />
  <!-- Area between curves (= IAE) -->
  <polygon points="{' '.join(area_points)}" fill="{color}" opacity="0.2" />
  <!-- Truth curve -->
  <path d="{truth_path}" fill="none" stroke="#888" stroke-width="1.5" stroke-dasharray="4,3" />
  <!-- Extracted curve -->
  <path d="{ext_path}" fill="none" stroke="{color}" stroke-width="2" />
  <!-- Labels -->
  <text x="{margin['left']+5}" y="{margin['top']+15}" font-size="11" fill="#333">{label}</text>
  <text x="{width-margin['right']-80}" y="{margin['top']+15}" font-size="10" fill="#888">— extracted</text>
  <text x="{width-margin['right']-80}" y="{margin['top']+28}" font-size="10" fill="#888">--- truth</text>
  <text x="{margin['left']+5}" y="{height-5}" font-size="10" fill="#aaa">Shaded area = IAE</text>
</svg>"""
        svgs.append(svg)

    return "\n".join(svgs)


def generate_report(benchmark_dir: Path, synthetic_dir: Path, output_path: Path):
    """Generate the full HTML benchmark report."""

    # Collect all benchmark results
    results = []
    for subdir in sorted(benchmark_dir.iterdir()):
        if not subdir.is_dir():
            continue

        name = subdir.name
        metrics_path = subdir / "metrics.json"
        extraction_path = subdir / "extraction.json"
        annotation_path = subdir / "annotation.png"

        # Find the original image and truth
        original_path = synthetic_dir / f"{name}.png"
        truth_path = synthetic_dir / f"{name}_truth.json"

        if not original_path.exists():
            continue

        entry = {
            "name": name,
            "original_img": img_to_base64(original_path),
            "annotation_img": img_to_base64(annotation_path) if annotation_path.exists() else "",
            "metrics": {},
            "extraction": {},
            "truth": {},
            "is_edge_case": name.startswith("edge_"),
        }

        if metrics_path.exists():
            with open(metrics_path) as f:
                entry["metrics"] = json.load(f)

        if extraction_path.exists():
            with open(extraction_path) as f:
                entry["extraction"] = json.load(f)

        if truth_path.exists():
            with open(truth_path) as f:
                entry["truth"] = json.load(f)

        # Generate area chart SVG
        if entry["extraction"] and entry["truth"]:
            entry["area_svg"] = make_area_chart_svg(entry["extraction"], entry["truth"])
        else:
            entry["area_svg"] = ""

        # Load diagnostic data if available
        diag_path = subdir / "diagnostic" / "diagnostic.json"
        if diag_path.exists():
            with open(diag_path) as f:
                entry["diagnostic"] = json.load(f)
            # Collect strip and heatmap images
            diag_dir = subdir / "diagnostic"
            entry["diag_strips"] = {}
            entry["diag_heatmaps"] = {}
            for arm_idx in range(len(entry.get("extraction", {}).get("arms", []))):
                strips = []
                for s in range(20):
                    sp = diag_dir / f"arm{arm_idx}_strip_{s:02d}.png"
                    if sp.exists():
                        strips.append(img_to_base64(sp))
                entry["diag_strips"][arm_idx] = strips
                hp = diag_dir / f"arm{arm_idx}_heatmap.png"
                if hp.exists():
                    entry["diag_heatmaps"][arm_idx] = img_to_base64(hp)
        else:
            entry["diagnostic"] = None

        # Load attempt 2 if available
        a2_dir = subdir / "attempt2"
        a2_metrics = a2_dir / "metrics.json"
        a2_annotation = a2_dir / "annotation.png"
        if a2_metrics.exists():
            with open(a2_metrics) as f:
                entry["attempt2_metrics"] = json.load(f)
            entry["attempt2_annotation"] = img_to_base64(a2_annotation) if a2_annotation.exists() else ""
            if (a2_dir / "extraction.json").exists():
                with open(a2_dir / "extraction.json") as f:
                    a2_ext = json.load(f)
                if entry["truth"]:
                    entry["attempt2_area_svg"] = make_area_chart_svg(a2_ext, entry["truth"])
                else:
                    entry["attempt2_area_svg"] = ""
            else:
                entry["attempt2_area_svg"] = ""
        else:
            entry["attempt2_metrics"] = None

        results.append(entry)

    # Summary stats
    all_iaes = [r["metrics"].get("iae", None) for r in results if r["metrics"]]
    all_iaes = [x for x in all_iaes if x is not None]
    standard_iaes = [r["metrics"]["iae"] for r in results if r["metrics"] and not r["is_edge_case"]]
    edge_iaes = [r["metrics"]["iae"] for r in results if r["metrics"] and r["is_edge_case"]]

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>KMGen Benchmark Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f5f5; color: #333; padding: 20px; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  h1 {{ font-size: 24px; margin-bottom: 8px; }}
  h2 {{ font-size: 18px; margin: 24px 0 12px; border-bottom: 2px solid #2196F3; padding-bottom: 4px; }}
  h3 {{ font-size: 15px; margin: 12px 0 8px; }}
  .timestamp {{ color: #888; font-size: 13px; margin-bottom: 20px; }}
  .summary {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; }}
  .stat {{ text-align: center; }}
  .stat-value {{ font-size: 28px; font-weight: bold; color: #2196F3; }}
  .stat-label {{ font-size: 12px; color: #888; text-transform: uppercase; }}
  .benchmark {{ color: #888; font-size: 13px; margin-top: 4px; }}
  .plot-card {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .plot-card.edge {{ border-left: 4px solid #FF9800; }}
  .plot-card.standard {{ border-left: 4px solid #2196F3; }}
  .plot-images {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 12px 0; }}
  .plot-images img {{ width: 100%; border: 1px solid #eee; border-radius: 4px; }}
  .plot-images .label {{ font-size: 11px; color: #888; text-transform: uppercase; margin-bottom: 4px; }}
  .metrics-row {{ display: flex; gap: 20px; flex-wrap: wrap; margin: 8px 0; }}
  .metric {{ background: #f8f8f8; padding: 8px 14px; border-radius: 4px; font-size: 13px; }}
  .metric .val {{ font-weight: bold; font-size: 16px; }}
  .iae-good {{ color: #4CAF50; }}
  .iae-ok {{ color: #FF9800; }}
  .iae-bad {{ color: #f44336; }}
  .arm-detail {{ font-size: 12px; color: #666; margin: 4px 0; }}
  .human-check {{ margin: 12px 0; padding: 12px; background: #FFFDE7; border-radius: 4px; border: 1px solid #FFF9C4; }}
  .human-check label {{ font-size: 13px; cursor: pointer; }}
  .human-check textarea {{ width: 100%; margin-top: 8px; padding: 6px; border: 1px solid #ddd; border-radius: 4px; font-size: 12px; resize: vertical; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 13px; }}
  th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid #eee; }}
  th {{ background: #f8f8f8; font-weight: 600; }}
  .tag {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 500; }}
  .tag-standard {{ background: #E3F2FD; color: #1565C0; }}
  .tag-edge {{ background: #FFF3E0; color: #E65100; }}
  .diagnostic-section {{ margin-top: 16px; padding: 14px; background: #F3E5F5; border-radius: 6px; border: 1px solid #E1BEE7; }}
  .diagnostic-section h4 {{ font-size: 14px; color: #6A1B9A; margin-bottom: 8px; }}
  .diag-stats {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 8px 0; font-size: 12px; }}
  .diag-stat {{ background: white; padding: 6px 12px; border-radius: 4px; }}
  .diag-stat .val {{ font-weight: bold; }}
  .strip-gallery {{ display: flex; overflow-x: auto; gap: 4px; padding: 8px 0; }}
  .strip-gallery img {{ height: 80px; border: 1px solid #ccc; border-radius: 2px; flex-shrink: 0; }}
  .heatmap-img {{ width: 100%; margin: 8px 0; border: 1px solid #ccc; border-radius: 4px; }}
  .strip-table {{ font-size: 11px; max-height: 200px; overflow-y: auto; }}
  .strip-table td, .strip-table th {{ padding: 3px 6px; }}
  .attempt-comparison {{ margin-top: 16px; padding: 14px; background: #E8F5E9; border-radius: 6px; border: 1px solid #C8E6C9; }}
  .attempt-comparison h4 {{ font-size: 14px; color: #2E7D32; margin-bottom: 8px; }}
  .attempt-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  .attempt-grid img {{ width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
  .attempt-grid .label {{ font-size: 11px; color: #666; text-transform: uppercase; margin-bottom: 4px; }}
  .delta-improved {{ color: #2E7D32; font-weight: bold; }}
  .delta-worse {{ color: #C62828; font-weight: bold; }}
  .delta-same {{ color: #888; }}
</style>
</head>
<body>
<div class="container">
<h1>KMGen Benchmark Report</h1>
<p class="timestamp">Generated: {timestamp}</p>

<div class="summary">
<h2>Summary</h2>
<div class="summary-grid">
  <div class="stat">
    <div class="stat-value">{len(results)}</div>
    <div class="stat-label">Plots Tested</div>
  </div>
  <div class="stat">
    <div class="stat-value">{f'{np.mean(all_iaes):.4f}' if all_iaes else 'N/A'}</div>
    <div class="stat-label">Mean IAE (all)</div>
    <div class="benchmark">KM-GPT: 0.018 | Ethan Opus: 0.0418</div>
  </div>
  <div class="stat">
    <div class="stat-value">{f'{np.mean(standard_iaes):.4f}' if standard_iaes else 'N/A'}</div>
    <div class="stat-label">Mean IAE (standard)</div>
  </div>
  <div class="stat">
    <div class="stat-value">{f'{np.mean(edge_iaes):.4f}' if edge_iaes else 'N/A'}</div>
    <div class="stat-label">Mean IAE (edge cases)</div>
  </div>
</div>

<table style="margin-top:16px;">
<tr><th>Plot</th><th>Type</th><th>IAE</th><th>Median AE</th><th>Median OS Err</th><th>Arms</th></tr>
"""

    for r in results:
        m = r["metrics"]
        iae = m.get("iae", None)
        iae_str = f"{iae:.4f}" if iae is not None else "—"
        iae_class = "iae-good" if iae and iae < 0.03 else ("iae-ok" if iae and iae < 0.06 else "iae-bad")
        tag_class = "tag-edge" if r["is_edge_case"] else "tag-standard"
        tag_label = "edge" if r["is_edge_case"] else "standard"
        ae_med = f"{m.get('ae_median', 0):.4f}" if m else "—"
        os_err = f"{m.get('median_os_error', 0):.2f}" if m else "—"
        n_arms = m.get("n_arms", "—")

        html += f"""<tr>
  <td><strong>{r['name']}</strong></td>
  <td><span class="tag {tag_class}">{tag_label}</span></td>
  <td class="{iae_class}"><strong>{iae_str}</strong></td>
  <td>{ae_med}</td>
  <td>{os_err}</td>
  <td>{n_arms}</td>
</tr>\n"""

    html += """</table>
</div>
"""

    # Individual plot cards
    for r in results:
        card_class = "edge" if r["is_edge_case"] else "standard"
        m = r["metrics"]
        iae = m.get("iae", None)
        iae_str = f"{iae:.4f}" if iae is not None else "N/A"

        html += f"""
<div class="plot-card {card_class}">
<h3>{r['name']} <span class="tag {'tag-edge' if r['is_edge_case'] else 'tag-standard'}">{'edge' if r['is_edge_case'] else 'standard'}</span></h3>

<div class="metrics-row">
  <div class="metric">IAE: <span class="val">{iae_str}</span></div>
  <div class="metric">Median AE: <span class="val">{m.get('ae_median', 'N/A')}</span></div>
  <div class="metric">Median OS Error: <span class="val">{m.get('median_os_error', 'N/A')}</span></div>
</div>

<div class="plot-images">
  <div>
    <div class="label">Original</div>
    <img src="{r['original_img']}" alt="Original plot" />
  </div>
  <div>
    <div class="label">Extracted (annotated)</div>
    <img src="{r['annotation_img']}" alt="Annotated extraction" />
  </div>
</div>

{r.get('area_svg', '')}
"""

        # Per-arm details
        if m.get("arms"):
            html += "<h3>Per-Arm Details</h3>\n"
            for arm in m["arms"]:
                html += f"""<div class="arm-detail">
  <strong>{arm.get('label_truth', 'Arm')}</strong>:
  IAE={arm.get('iae', 0):.4f},
  AE median={arm.get('ae_median', 0):.4f},
  AE max={arm.get('ae_max', 0):.4f},
  Steps: {arm.get('n_steps_extracted', '?')} extracted / {arm.get('n_steps_truth', '?')} truth,
  Median OS err={arm.get('median_os_error', 0):.2f}
</div>\n"""

        # Synthetic data info
        truth = r.get("truth", {})
        if truth:
            axis = truth.get("axis", {})
            html += f"""<div class="arm-detail" style="margin-top:8px;">
  <strong>Synthetic data:</strong>
  x_max={axis.get('x_max', '?')},
  y_range=[{axis.get('y_min', 0)}, {axis.get('y_max', 1.0)}],
  {len(truth.get('arms', []))} arms,
  total truth steps: {sum(len(a.get('coordinates', a.get('steps', []))) for a in truth.get('arms', []))}
</div>\n"""

        # Diagnostic dashboard section
        if r.get("diagnostic"):
            diag = r["diagnostic"]
            html += '<div class="diagnostic-section">\n<h4>Diagnostic Dashboard (Self-Correction Input)</h4>\n'

            for arm_data in diag.get("arms", []):
                arm_idx = arm_data.get("arm_index", 0)
                label = arm_data.get("label", f"Arm {arm_idx}")
                bias = arm_data.get("mean_bias_px", 0)
                asym = arm_data.get("mean_asymmetry", 0)
                hit = arm_data.get("overall_hit_rate", 0)
                direction = arm_data.get("bias_direction", "unknown")

                html += f'<h4 style="margin-top:10px;">{label} — bias: {bias:+.1f}px ({direction})</h4>\n'
                html += '<div class="diag-stats">\n'
                html += f'  <div class="diag-stat">Bias: <span class="val">{bias:+.1f}px</span></div>\n'
                html += f'  <div class="diag-stat">Asymmetry: <span class="val">{asym:.2f}</span></div>\n'
                html += f'  <div class="diag-stat">Hit rate: <span class="val">{hit:.2f}</span></div>\n'
                html += '</div>\n'

                # Heatmap
                hm = r.get("diag_heatmaps", {}).get(arm_idx, "")
                if hm:
                    html += f'<div class="label">Residual heatmap</div>\n'
                    html += f'<img class="heatmap-img" src="{hm}" alt="Residual heatmap" />\n'

                # Strip gallery
                strips = r.get("diag_strips", {}).get(arm_idx, [])
                if strips:
                    html += f'<div class="label">Zoomed strips (6x) — red=extracted, green=centroid</div>\n'
                    html += '<div class="strip-gallery">\n'
                    for s_img in strips:
                        html += f'  <img src="{s_img}" />\n'
                    html += '</div>\n'

                # Strip stats table
                strip_data = arm_data.get("strips", [])
                if strip_data:
                    html += '<details><summary style="font-size:12px;cursor:pointer;margin:6px 0;">Strip details table</summary>\n'
                    html += '<table class="strip-table"><tr><th>#</th><th>t range</th><th>bias (px)</th><th>asym</th><th>hit rate</th><th>verdict</th></tr>\n'
                    for sd in strip_data:
                        b = sd.get("bias_px", 0)
                        verdict = "OK" if abs(b) < 1.5 else ("WARN" if abs(b) < 3 else "ERROR")
                        v_color = "#4CAF50" if verdict == "OK" else ("#FF9800" if verdict == "WARN" else "#f44336")
                        t_range = sd.get("t_range", [0, 0])
                        html += f'<tr><td>{sd.get("strip", "?")}</td><td>{t_range[0]:.1f}–{t_range[1]:.1f}</td>'
                        html += f'<td>{b:+.1f}</td><td>{sd.get("asymmetry", 0):.2f}</td>'
                        html += f'<td>{sd.get("pixel_hit_rate", 0):.2f}</td>'
                        html += f'<td style="color:{v_color};font-weight:bold;">{verdict}</td></tr>\n'
                    html += '</table></details>\n'

            html += '</div>\n'

        # Attempt 2 comparison section
        if r.get("attempt2_metrics"):
            a1_iae = m.get("iae", None)
            a2_iae = r["attempt2_metrics"].get("iae", None)

            html += '<div class="attempt-comparison">\n<h4>Self-Correction Result (Attempt 2 — blind, no ground truth)</h4>\n'

            if a1_iae is not None and a2_iae is not None:
                delta = a2_iae - a1_iae
                pct = (delta / a1_iae * 100) if a1_iae > 0 else 0
                if delta < -0.001:
                    delta_class = "delta-improved"
                    delta_label = f"Improved by {abs(delta):.4f} ({abs(pct):.0f}%)"
                elif delta > 0.001:
                    delta_class = "delta-worse"
                    delta_label = f"Worse by {delta:.4f} (+{pct:.0f}%)"
                else:
                    delta_class = "delta-same"
                    delta_label = "No significant change"

                html += '<div class="metrics-row">\n'
                html += f'  <div class="metric">Attempt 1 IAE: <span class="val">{a1_iae:.4f}</span></div>\n'
                html += f'  <div class="metric">Attempt 2 IAE: <span class="val">{a2_iae:.4f}</span></div>\n'
                html += f'  <div class="metric"><span class="{delta_class}">{delta_label}</span></div>\n'
                html += '</div>\n'

            html += '<div class="attempt-grid">\n'
            html += f'  <div><div class="label">Attempt 1</div><img src="{r["annotation_img"]}" /></div>\n'
            a2_img = r.get("attempt2_annotation", "")
            html += f'  <div><div class="label">Attempt 2 (self-corrected)</div><img src="{a2_img}" /></div>\n'
            html += '</div>\n'

            a2_svg = r.get("attempt2_area_svg", "")
            if a2_svg:
                html += '<h4 style="margin-top:10px;">Attempt 2 — Area Between Curves</h4>\n'
                html += a2_svg

            html += '</div>\n'

        # Human annotation section
        html += f"""
<div class="human-check">
  <label><input type="checkbox" id="check_{r['name']}_accurate" /> Extraction looks visually accurate</label><br/>
  <label><input type="checkbox" id="check_{r['name']}_steps" /> Step-downs correctly identified</label><br/>
  <label><input type="checkbox" id="check_{r['name']}_colors" /> Curve separation correct</label><br/>
  <label><input type="checkbox" id="check_{r['name']}_bbox" /> Bounding box aligned</label><br/>
  <textarea id="notes_{r['name']}" rows="2" placeholder="Notes (missed steps, false positives, issues...)"></textarea>
</div>
</div>
"""

    html += """
</div>
</body>
</html>"""

    output_path.write_text(html)
    print(f"Report generated: {output_path}")
    print(f"  {len(results)} plots documented")
    if all_iaes:
        print(f"  Mean IAE: {np.mean(all_iaes):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate KMGen benchmark report")
    _repo = Path(__file__).resolve().parent.parent
    parser.add_argument("--benchmark-dir", type=Path, default=_repo / "benchmark")
    parser.add_argument("--synthetic-dir", type=Path, default=_repo / "shared" / "synthetic")
    parser.add_argument("--output", "-o", type=Path, default=_repo / "benchmark" / "report.html")
    args = parser.parse_args()

    generate_report(args.benchmark_dir, args.synthetic_dir, args.output)
