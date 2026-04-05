# Gate Status

## Current Cycle: BW Fix Run (2026-04-02)
- **Verdict**: IMPROVED
- **Focus**: extract_bw_2arm rewrite (coverage-based arm assignment, continuity tracking, jump rejection, run-length filter)
- **BW plots after fix**:
  - `stress_bw`: IAE 0.0088 (was 0.0084, delta: +0.0004 — within tolerance)
  - `stress_combo_bw_overlap`: IAE 0.0075 (was 0.0822, delta: **-0.0747** — major improvement)
  - `stress_combo_grid_annotation_bw`: IAE 0.0744 (was 0.0880, delta: -0.0136 — improved but still Hard tier)
- **Notes**: bw_overlap moved from Hard to Easy tier. grid_annotation_bw improved but remains the hardest BW case. stress_bw held steady (no regression).

## Previous Cycle: Quality Gate Run (2026-04-01)
- **Verdict**: PASS
- **Mean IAE**: 0.0086 (was 0.0086, delta: +0.0000)
- **Median IAE**: 0.0068 (was 0.0068, delta: +0.0000)
- **Plots**: 21
- **Regressions (>0.005)**: none
- **Improvements (>0.002)**: none
- **Errors**: 1 (edge_cumulative_incidence — known issue, y-axis 0.0-0.5 tick detection)
- **Notes**: All 20 successful plots matched baseline exactly. No code changes detected since baseline run 012.

## Initial Cycle: Baseline benchmark (run 012, 2026-03-31)
- **Result**: Baseline established
- **Mean IAE**: 0.0086 (21 plots) / 0.0150 (31 plots incl. combos)
- **Errors**: 1 (edge_cumulative_incidence)
- **Date**: 2026-03-31
