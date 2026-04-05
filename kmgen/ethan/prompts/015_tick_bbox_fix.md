# Prompt 015 — Root Cause Found: Tick-Mark Bbox Detection
**Date:** 2026-03-31

## Root Cause Analysis (deep dive on synthetic_001)

### The Investigation
- Compared extracted vs truth at 25 evenly-spaced points
- Found **constant ~0.045 offset** across the entire curve (extracted always UNDER truth)
- Traced to pixels: curve at S=1.0 should be at bbox row=45, but first blue pixels are at row=103
- That's 58 pixels of empty white space between the frame and the curve
- 58 pixels × (1.0/1277 S-units per pixel) = 0.045 in S-units — matches the bias exactly

### Root Cause
**Matplotlib axis padding.** Matplotlib adds ~5% padding beyond the data range. The axis frame goes from -0.05 to 1.05, but our data goes from 0.0 to 1.0. `detect_bbox_refined()` was finding the frame/spine edges, not the tick mark positions. This made every single pixel-to-data conversion off by the padding amount.

### The Fix
Rewrote `detect_bbox_refined()` to find **tick marks** (short horizontal dark segments extending from the axis line) instead of the frame edges. Tick marks sit at actual data positions (0.0, 0.2, ..., 1.0), so they give the correct bbox.

This is the same approach we used successfully for the real KM plot (kmgpt_plot.png) in prompt 004 — we just hadn't applied it to the benchmark extraction pipeline.

### Results

| Run | Method | Mean IAE | vs KM-GPT |
|-----|--------|----------|-----------|
| 008 | Frame-based bbox | 0.0235 | 1.3x worse |
| 009 | + daily grain | 0.0233 | 1.3x worse |
| 009 | + centroid (reverted) | 0.0262 | 1.5x worse |
| **010** | **Tick-mark bbox** | **0.0064** | **2.8x better** |
| — | KM-GPT benchmark | 0.018 | — |
| — | Ethan Opus 4.6 | 0.0418 | — |

Per-plot results (run 010):
| Plot | IAE |
|------|-----|
| edge_high_survival | **0.0012** |
| edge_near_flat | 0.0026 |
| synthetic_002 | 0.0030 |
| synthetic_003 | 0.0035 |
| edge_ci_shading | 0.0035 |
| synthetic_001 | 0.0046 |
| synthetic_004 | 0.0055 |
| synthetic_005 | 0.0063 |
| edge_four_arms | 0.0070 |
| edge_small_dense | 0.0096 |
| edge_multi_panel | 0.0237 |
| edge_cumulative_incidence | ERROR |

### Key Insight
> The "systematic bias" was never a pixel-reading problem. It was a calibration problem hiding in plain sight. The same fix we discovered for the real image in prompt 004 (use tick marks, not frame edges) was the answer all along — we just hadn't applied it to the automated pipeline.

### Remaining Issues
1. `edge_cumulative_incidence` errored — tick detection doesn't handle non-standard y-axis (0.0-0.5)
2. `edge_multi_panel` still high at 0.0237 — small panels have fewer/harder-to-detect tick marks
