# Prompt 014 — Bias Correction Attempt
**Date:** 2026-03-31

## User Request (verbatim)
> First, did we address the over/under bias?
> Sure [to implementing centroid fix + daily grain + re-running benchmark]

## What We Tried

### Centroid tracing (reverted)
Changed `trace_curve()` to use the vertical centroid (mean row) of all colored pixels per column instead of the topmost pixel.

**Result: Made things WORSE.** Mean IAE 0.0235 → 0.0262.

**Why:** For KM step functions, the survival value IS at the TOP of the horizontal segment — it stays at the higher level until the next event. The topmost pixel is actually the correct position. Using the centroid puts us at the center of the drawn line, which is BELOW the mathematical curve value. This made the downward bias worse.

**Learning:** The bias isn't from "we're reading the wrong part of the line." It's from something else — probably anti-aliasing on vertical drop segments, or subtle bbox calibration errors.

### Daily grain sampling (kept)
Changed sampling interval from `x_max/500` to `1/30` (≈1 day in months).

**Result: Slight improvement.** Mean IAE 0.0235 → 0.0233.

**Why:** Daily granularity aligns with the natural resolution of clinical trial data. Events are recorded by date, so sampling at 1-day intervals matches the data's actual precision.

## Final Run 009 Results
Mean IAE: **0.0233** (down from 0.0235)

| Plot | Run 008 IAE | Run 009 IAE | Change |
|------|-------------|-------------|--------|
| synthetic_001 | 0.0277 | 0.0271 | -0.0006 |
| synthetic_002 | 0.0294 | 0.0297 | +0.0003 |
| synthetic_003 | 0.0169 | 0.0171 | +0.0002 |
| synthetic_004 | 0.0258 | 0.0258 | 0.0000 |
| synthetic_005 | 0.0231 | 0.0231 | 0.0000 |
| edge_high_survival | 0.0171 | 0.0160 | -0.0011 |
| edge_cumulative_incidence | 0.0160 | 0.0149 | -0.0011 |
| edge_four_arms | 0.0239 | 0.0239 | 0.0000 |
| edge_ci_shading | 0.0272 | 0.0275 | +0.0003 |
| edge_near_flat | 0.0333 | 0.0332 | -0.0001 |
| edge_small_dense | 0.0175 | 0.0177 | +0.0002 |
| edge_multi_panel | 0.0240 | 0.0241 | +0.0001 |

## Remaining Bias
The systematic downward bias is still present. The root cause is NOT the line-position reading. Needs further investigation — could be:
1. Vertical segment anti-aliasing creating earlier-than-real step detection
2. Subtle bbox calibration drift across different plots
3. Color mask threshold effects (missing faint anti-aliased pixels at the top edge of the line)
