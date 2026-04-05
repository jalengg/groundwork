# Self-Correction Diagnostic Engine — Design Document

**Date**: 2026-04-04
**Authors**: Jalen + Claude
**Status**: Implementing

## Problem Statement

After extracting KM curves from images, we have no ground truth to know if the extraction is correct. The extraction code runs once ("one chance to get it"). We need a self-correction mechanism that detects and fixes errors WITHOUT access to ground truth or IAE scores.

## Rejected Approaches

| Approach | Why rejected |
|---|---|
| Pixel hit rate | Circular — same logic that placed circles checks if they're on the curve |
| Patients-at-risk table | Not always available |
| Cross-method agreement | Jalen + Ethan now use same method |
| Reported statistics | Not generalizable, just a small trick |
| LLM visual QA (raw) | LLMs can't see fine-grain pixel details (2px bias invisible) |
| Hardcoded rules | User preference against rigid constraints |
| Inverse rendering / degradation modeling | Unknown degradations are hard to replicate |

## Key Insight: CV as Microscope, LLM as Doctor

The LLM can't see a 2px offset on a 600px image. But if CV **amplifies and visualizes** the error first, the LLM becomes a great judge.

- **CV**: precise measurement, counting pixels, computing distances, detecting asymmetry
- **LLM**: pattern interpretation, spatial reasoning, deciding what to fix

## Diagnostic Dashboard Design

### 1. Curve-Following Zoomed Strips (20 per arm)
- Divide time axis into 20 segments
- For each segment, crop ±20px above/below the extracted position
- Upscale 6x so a 2px bias becomes 12px (clearly visible)
- Overlay: red dot (extracted), green line (centroid), yellow bracket (line thickness)
- Label: t_range, s_range, col_range

### 2. Per-Strip Measurements
- `bias_px`: extracted position minus centroid of curve pixels
- `asymmetry`: (pixels_below - pixels_above) / total at extracted position
- `pixel_hit_rate`: fraction of points within 3px of curve pixel
- Spatial mapping: t_range, s_range, col_range back to data coordinates

### 3. Residual Heatmap
- Thin color strip showing error magnitude across full time range
- Green (0px) → Yellow (2-3px) → Red (5+px)

### 4. Global Stats
- Mean bias, direction, mean asymmetry, overall hit rate, max bias

## Self-Correction Loop

```
1. Attempt 1: Standard extraction pipeline → coordinates + annotation
2. CV diagnostic engine (deterministic): coordinates + image → dashboard
3. Self-correction subagent (BLIND):
   - Receives: original image + annotation + diagnostic dashboard
   - Does NOT receive: ground truth, IAE, truth JSON
   - Writes critique + corrected extraction code
4. Attempt 2: Run corrected code → new coordinates
5. Report: Shows both attempts with IAE (computed by us, never seen by subagent)
```

## Mathematical Signal Chain

Each diagnostic strip carries:
1. **Visual**: zoomed image showing gap between extraction and curve center
2. **Coordinates**: WHERE in data space (t-range, s-range) AND pixel space (col-range)
3. **Measurements**: HOW MUCH bias in pixels, which direction, confidence

The LLM uses all three to write targeted corrections:
```python
# Diagnostic: strips 3-14 show +2.8px upward bias
# → switch to centroid in cols 650-1600
if 650 <= col <= 1600:
    row = int(np.mean(col_pixels))  # centroid
else:
    row = col_pixels[0]  # topmost
```

## Perpendicular Symmetry Test

At each extracted point, check the vertical pixel intensity profile:
- If symmetric → on center → correct
- If asymmetric (more pixels below) → biased high → shift toward centroid
- Works because blur is symmetric around true curve center

Validated on stress_combo_stretched_dense:
- Found 0.4px topmost-centroid offset (line is 1-2px thick)
- Actual IAE was 0.037 (mostly from bbox/coordinate mapping, not pixel selection)
- Demonstrates that different error types need different detection mechanisms

## Error Regime Coverage

| Error size | Mechanism | Example |
|---|---|---|
| Gross (wrong arm) | LLM critique on overlay | Circles on blue instead of red |
| Medium (10-20px) | Multi-strategy disagreement + diagnostic strips | Topmost ≠ centroid |
| Subtle (2-5px) | Perpendicular symmetry + zoomed strips | Anti-aliased diagonal |
| Sub-pixel | Report as uncertainty (±spread) | Below blur radius |

## Implementation Files

1. `jalen/diagnostic.py` — Pure CV diagnostic engine (deterministic)
2. `jalen/self_correct.py` — Self-correction orchestrator (LLM subagent)
3. `jalen/report.py` — Updated HTML report with attempt comparison
