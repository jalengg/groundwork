# Self-Correction V2 — Rich Diagnostic Dashboard Design

**Date**: 2026-04-04
**Status**: Implementing

## V1 Postmortem

V1 failed because:
1. Agents only saw 20 tiny zoomed strips with `bias_px` numbers
2. Agents applied pixel-shift corrections (glorified lookup table)
3. No root cause diagnosis — agents didn't know WHY the extraction was wrong
4. Agents rewrote the full extraction pipeline instead of making targeted fixes

## V2 Design: Rich Diagnostic + Root Cause Diagnosis

### Diagnostic Dashboard Components

#### 1. Full-Context Views
- **Original image + annotation overlay** (full size, side by side)
- **Color mask per arm** — show which pixels are being detected. Reveals: missing edge pixels, legend contamination, mask too strict/loose
- **Bbox overlay on original** — draw the detected bbox rectangle on the image so agent can see if tick marks align with bbox boundaries

#### 2. Root Cause Signals
- **Perpendicular profile CHARTS** — at 10-20 points along the curve, plot a bar chart of pixel intensity in a vertical slice through the extraction point. The SHAPE tells the cause:
  - Symmetric but shifted → bbox calibration issue
  - Asymmetric (more pixels below) → topmost-pixel bias, use centroid
  - Bimodal → wrong cluster selected, tracing annotation/legend
  - Very thin (1-2px) → can't improve much, near resolution limit
- **Multi-strategy trace comparison** — run topmost, centroid, and skeleton extraction on same image. Overlay all three as different colored lines. Where they diverge = uncertain regions. Which is closest to curve center = the right strategy for that region
- **Coverage map** — per-column count of detected curve pixels. Low coverage = mask too strict. High with wide spread = mask too loose or anti-aliasing captured
- **Zoomed strips** (kept from V1 but enhanced) — now include the centroid line AND the perpendicular profile bar chart inset

#### 3. Structured Numbers
- Per-strip: bias_px, asymmetry, hit_rate, coverage_count, strategy_agreement (do topmost/centroid/skeleton agree?)
- Global: mean_bias, bias_direction, overall_hit_rate, bbox_quality_score, mask_coverage_ratio

### Correction Agent Prompt (Structured Diagnosis)

```
You are a KM curve extraction specialist performing blind quality improvement.

You have:
1. The original plot image
2. The attempt 1 annotation overlay (circles on curves)
3. A diagnostic dashboard with:
   - Color mask visualization (which pixels were detected)
   - Bbox overlay (is the bounding box aligned with axis ticks?)
   - Perpendicular profile charts at 20 points along each curve
   - Multi-strategy comparison (topmost vs centroid vs skeleton traces)
   - Per-strip measurements (bias, asymmetry, coverage)

You do NOT have ground truth. Use ONLY the diagnostic evidence.

STEP 1 — DIAGNOSE: What is the root cause of any extraction errors?
Choose one or more:
  a) Bbox miscalibration — tick marks don't align with bbox boundaries
  b) Color mask too strict — missing anti-aliased edge pixels
  c) Color mask too loose — including legend text, annotations, or gridlines
  d) Topmost-pixel bias — should use centroid in diagonal/dense zones
  e) Wrong cluster selection — tracing jumped to annotation/legend blob
  f) Arm confusion — curves assigned to wrong labels
  g) No significant issues detected

STEP 2 — PLAN: For each diagnosed issue, describe the specific fix.

STEP 3 — CODE: Write a corrected extraction function that addresses
the diagnosed issues. The function receives the original coordinates
from attempt 1 — apply targeted adjustments, do NOT rewrite from scratch.

def corrected_extract(image_path, bbox, axis, attempt1_coords):
    # attempt1_coords = [{"label": "Arm A", "coordinates": [{"t": ..., "s": ...}]}, ...]
    # Return same format with corrected coordinates
```

### Key Differences from V1

| Aspect | V1 | V2 |
|--------|----|----|
| Agent sees | 20 tiny strips + bias numbers | Full image, masks, bbox overlay, profiles, multi-strategy |
| Agent reasons about | "How many pixels off" | "WHY is it off" (root cause) |
| Agent outputs | Pixel shift per strip | Diagnosis + targeted fix code |
| Input to code | Raw image only | Image + attempt 1 coordinates |
| Fix approach | Rewrite extraction from scratch | Adjust existing coordinates |
