# Prompt 011 — IAE Implementation
**Date:** 2026-03-31

## Changes
- Implemented `metrics.py` with IAE (Integrated Absolute Error), point-wise AE, median survival error
- Matches Ethan's implementation: step-function interpolation, normalized time [0,1], trapezoid integration
- CLI usage: `python metrics.py extraction.json ground_truth.json`
- Library usage: `from metrics import compute_score`

## Self-Consistency Test
Compared our two extractions against each other (no ground truth available):
- extract_cv.py (32 blue, 30 orange steps) vs skill_test (44 blue, 42 orange steps)
- **Cross-IAE: 0.0098** — the two methods agree very closely
- Trilaciclib arm: IAE 0.0080, median OS error 0.04 months
- Placebo arm: IAE 0.0117, AE max 0.2348 (the dense zone gap in extract_cv)

## Interpretation
- IAE of 0.0098 between our two methods means they extract nearly identical curves
- The skill_test finds more steps (44 vs 32) thanks to upscaling, but the overall curve shape is the same
- The Placebo AE max of 0.23 is from the dense zone around month 5-6 where extract_cv had a gap
- **We can't know true accuracy without ground truth** — need to test on synthetic data or NCT03041311 trial data

## Benchmarks
- KM-GPT: IAE 0.018
- Ethan Opus 4.6: IAE 0.0418
- Our cross-consistency: IAE 0.0098 (but this is NOT against ground truth)
