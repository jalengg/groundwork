# Self-Correction V3 — Reasoning-First Design

**Date**: 2026-04-04
**Status**: Implementing

## V2 Postmortem

V2 improved the hardest cases (grid_annotation_bw: 73% improvement) but:
1. Agents are too mechanical — pattern-match "negative bias" → apply pixel shift
2. No awareness of line styles (dashed vs solid, gap bridging)
3. Zoomed strips are too granular ("missing the forest for the trees")
4. Diagnostics don't influence technique selection, just offset correction
5. No exploration of better centroid/extraction techniques
6. Report doesn't show agent reasoning progression

## V3 Design: Reasoning-First Agent

### Agent Output Structure (4 sections, all included in report)

#### Section 1: Image Analysis
"What challenges/obstacles/qualities do I observe about this plot?"
- How many curves? Colors? Solid vs dashed vs dotted?
- Are there annotations, legends, gridlines in the plot area?
- Is the image blurry, JPEG-compressed, dark, tiny?
- Are curves overlapping? Where?
- Is this a dense zone (smooth diagonal) or steppy (clear horizontal segments)?

#### Section 2: Technique Selection
"What tools/heuristics/CV techniques will I use, and why/why not for each?"
- Topmost pixel vs centroid vs Gaussian fit vs skeleton — which and why?
- Dashed line handling: gap bridging? Gap detection? 
- Color mask: HSL vs RGB? Threshold choice?
- Bbox detection: tick marks vs frame?
- What I'm specifically NOT doing and why

#### Section 3: Diagnostic Interpretation
"What does the diagnostic data tell me?"
- Overall bias trend (not per-strip, but the pattern)
- Where strategy agreement is low (topmost ≠ centroid ≠ skeleton)
- Coverage issues (too thin, too thick, zero coverage regions)
- Any catastrophic regions (arm jumps, annotation contamination)

#### Section 4: Correction Plan + Code
"What specific corrections am I making and why?"
- For each correction: what diagnostic evidence motivated it
- The corrected_extract function
- Expected impact

### Diagnostic Dashboard Changes

**Remove**: 20 evenly-spaced zoomed strips (too granular)

**Replace with**:
- Full-width strategy comparison overlay (one image showing all 3 traces)
- Full-width mask visualization
- Bias trend chart (bias_px vs time, as a line graph — shows the pattern, not 20 numbers)
- 3-4 zoomed crops of the WORST regions only (highest |bias| or lowest hit rate)
- Coverage chart (full width)

### Better Centroid Techniques to Explore

1. **Gaussian fit**: For each column, fit a 1D Gaussian to the vertical intensity profile. The mean is the sub-pixel center. Works well for anti-aliased lines.

2. **Skeleton centerline**: Morphological thinning (scipy.ndimage.binary_thin or manual Zhang-Suen) on the curve mask → 1px centerline. This is the gold standard for finding curve centers.

3. **Edge detection + midpoint**: Find upper and lower edges of the curve using gradient magnitude, take the average. Robust to asymmetric anti-aliasing.

4. **Intensity-weighted centroid with Gaussian kernel**: Weight each pixel by exp(-d²/2σ²) where d is distance from expected position. Suppresses outlier pixels.

5. **Local polynomial fit**: Fit a low-degree polynomial to the (col, row) trace over a window of ±20 columns. The polynomial value at the center column is the smoothed position. Reduces jitter without the topmost-pixel bias.

### Report Layout for Top 15 Worst Plots

Each plot card shows:
```
┌─────────────────────────────────────────────────────┐
│ Plot: stress_combo_grid_annotation_bw               │
│ IAE: 0.0744 → 0.0196 (IMPROVED 73%)               │
├─────────────────────────────────────────────────────┤
│ [Original]           [Attempt 1 Annotation]         │
├─────────────────────────────────────────────────────┤
│ § Image Analysis                                    │
│   "Two black curves, solid and dashed. Heavy        │
│    gridlines. Annotation boxes: p=0.03, HR=0.72,   │
│    arrow to median OS. Legend in lower-left."       │
├─────────────────────────────────────────────────────┤
│ § Technique Selection                               │
│   "Using centroid (not topmost) because diagonal    │
│    zones. Dark mask threshold 100 to exclude gray   │
│    gridlines. Legend exclusion in lower-left 35%.   │
│    NOT using skeleton — too noisy for BW overlap."  │
├─────────────────────────────────────────────────────┤
│ § Diagnostic Evidence                               │
│   [Strategy overlay]  [Mask]  [Bias trend chart]    │
│   [3 zoomed worst-regions]                          │
├─────────────────────────────────────────────────────┤
│ § Correction Applied                                │
│   "Annotation text at t=14 caused +15px jump.       │
│    Interpolated through annotation region.           │
│    Applied centroid shift in diagonal zones."        │
├─────────────────────────────────────────────────────┤
│ [Attempt 2 Annotation]  [Area-between-curves]       │
│ IAE: 0.0744 → 0.0196                               │
└─────────────────────────────────────────────────────┘
```

### Cutting to 15 Worst

Sort all 32 plots by attempt 1 IAE descending. Show full cards for top 15. For the remaining 17, show a collapsed row with just the name + IAE (no images, no diagnostics).
