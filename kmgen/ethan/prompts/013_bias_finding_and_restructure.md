# Prompt 013 — Systematic Bias Finding + Repo Restructure
**Date:** 2026-03-31

## User Finding (verbatim)
> Wow, this looks closer, If you read the HTML area-between-curves, you'll see there's a consistent bias where extracted is always less than truth, except for that one test case where the curve was going up, and it seems like the misestimation is strongest at the leftmost/rightmost/ Can you document that I made this finding, for research purposes, also document what changes you made to the prompt/skill overall. Also fix the benchmark vs skill-test they need to be in an iterative step timesteppable and timeseriesable to show our dev journey, want a more incremental dev history kind of file system.

## Research Finding: Systematic Downward Bias in Extraction
**Discovered by:** Jalen Jiang
**Date:** 2026-03-31
**Source:** Visual inspection of area-between-curves SVGs in benchmark report

### Observation
Across all standard KM plots (curves going DOWN from 1.0), the extracted curve consistently sits **below** the ground truth curve. The extraction systematically **underestimates** survival probability.

The ONE exception: `edge_cumulative_incidence` where curves go UP from 0.0. There, the extracted curve sits **above** the truth — which is the same directional bias (extraction reads the curve as more extreme than it is).

### Pattern: Strongest at Extremes
The bias is largest at the **leftmost** (near t=0, where S≈1.0) and **rightmost** (tail of the curve) portions. The middle section is closer to truth.

### Hypothesized Causes
1. **Topmost-pixel tracing**: The tracer picks the topmost colored pixel per column. With anti-aliasing, the topmost pixel is a faint blend of curve color + white background — its center is slightly ABOVE the actual mathematical curve position. But we use it as the curve position, which maps to a slightly LOWER survival value (since higher pixel row = lower S). This creates systematic downward bias.
2. **Anti-aliasing asymmetry**: The curve line's anti-aliased fringe extends slightly more downward than upward (the step function has a vertical drop that bleeds downward), pulling the detected position lower.
3. **Edge effects at extremes**: Near t=0 (S≈1.0), the curve is very close to the axis/top of plot — fewer pixels above the curve to average, so the fringe effect is strongest. At the tail (S near 0), similar edge compression.
4. **Cumulative incidence confirmation**: When curves go UP, the same pixel-position bias manifests as the extracted curve being ABOVE truth — consistent with the "topmost pixel reads slightly too low in S" hypothesis, which in cumulative incidence (where low row = low incidence) manifests as reading slightly too high.

### Implication
A systematic bias correction (adding ~1-2px upward offset to traced positions) might uniformly improve IAE across all plots. This should be tested.

## Repo Structure Change
User requested: benchmark results and skill_test should be in the same iterative, timestamped system — showing the development journey as a time series, not separate folders.

## Skill/Prompt Evolution Summary
See prompt 013 appendix below.

---

## Appendix: Skill/Prompt Evolution

### Iteration 1 (prompts 001-003): Manual Visual Extraction
- Claude Code visually inspects image, manually lists (time, survival) coordinates
- Draws red circles from hardcoded coordinate arrays
- **Result**: Every circle was off — bbox was wrong, coordinates were guessed

### Iteration 2 (prompt 004): Programmatic Bbox Detection
- `detect_bbox.py` — scans pixel brightness to find axis labels and tick marks
- Bbox shifted from (126,78,714,536) to (163,95,783,604)
- **Result**: Calibration verified via grid overlay; circles now on curves

### Iteration 3 (prompts 005-007): Hybrid LLM+CV Pipeline
- Key insight: LLM should be analyst/strategist, code should be pixel extractor
- `extract_cv.py` — color filter (RGB thresholds) → continuity-tracked tracing → step detection
- Added max step filter (80px) to reject artifacts
- **Result**: 31 blue + 29 orange steps; circles on curves; dense zone gap near dashed median line

### Iteration 4 (prompts 008-010): Adaptive Agent Skill
- Incorporated Ethan's findings: HSL color space, 2x upscaling, IAE metric
- Wrote `skill_kmgen.md` — adaptive agent prompt with technique toolbox
- Agent reasons about each image and picks techniques instead of following fixed recipe
- **Result**: Skill test produced 44 blue + 42 orange steps using HSL + upscaling

### Iteration 5 (prompts 011-013): Benchmark + Metrics
- `metrics.py` — IAE implementation compatible with Ethan's
- `report.py` — HTML report with area charts, human annotation
- 12 synthetic plots (5 standard + 7 edge cases) with ground truth
- **Result**: Mean IAE 0.0235 (beating Ethan's 0.0418, approaching KM-GPT's 0.018)
- **Finding**: Systematic downward bias in extraction (Jalen, 2026-03-31)
