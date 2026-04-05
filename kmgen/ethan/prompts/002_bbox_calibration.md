# Prompt 002 — Bbox Calibration Fix
**Date:** 2026-03-16

## User Feedback
> To me, it looks like every single circle is off, when you redraw the coordinates, might need to map the chart-coordinates to the image-coordinates first before drawing the red circles, let's start with ensuring we know how to draw the red circles in the correct "guess locations" first.

## Approach
- Created `calibrate.py` to draw green crosses at known axis positions (tick marks)
- Iteratively adjusted bbox from (128, 68, 718, 553) → (130, 84, 714, 533)
- Confirmed calibration: crosses land on axis ticks

## Result (iteration 003)
- **Bbox mapping is now correct** — pixel mapping verified via calibration
- **Circles still off the curves** — the data coordinates I estimated are inaccurate
- This isolates the problem: it's the multimodal coordinate reading, not the pixel math
- Dense sections (months 0-6) are hardest — steps blur together at this resolution

## Key Insight
The two problems are now separated:
1. ~~Data→pixel mapping~~ ✅ SOLVED via calibration
2. **Accurate step-down coordinate extraction** — the remaining challenge
