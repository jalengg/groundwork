# Prompt 004 — Programmatic Bbox Detection
**Date:** 2026-03-16

## User Feedback (verbatim)
> Yes, can you look at it yourself? The blue rectangle doesn't match up with the x and y axes, unless the blue rectangle drawing process itself is flawed.

## Approach
- Stopped guessing bbox by eye — used pixel analysis instead
- `detect_bbox.py`: scanned image for dark pixels to find axis features
- Key findings:
  - Y-axis labels detected at consistent ~102px per 0.2 unit spacing
  - Y-axis label centers: 1.0→94.5, 0.8→196.5, 0.6→299.0, 0.4→400.0, 0.2→502.5, 0.0→604.0
  - X-axis tick marks at cols 163 (x=0) and 340 (x=6) → spacing = 29.5 px/unit
  - x=21 → col 783
- Old bbox: `(126, 78, 714, 536)` — wrong in every dimension
- New bbox: `(163, 95, 783, 604)` — derived from pixel data

## Result
- **Calibration**: yellow grid dots now land precisely on gridlines
- **Annotation**: circles actually sit on the curve lines now
- Remaining work: fine-tune the individual step-down coordinate reads

## Key Insight
Never eyeball the bbox — detect it programmatically from axis features. The error was ~40px in some dimensions, which completely broke the coordinate mapping.
