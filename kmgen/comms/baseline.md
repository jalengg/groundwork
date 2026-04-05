# IAE Baseline

Last updated: 2026-03-31 (run 012)

**Mean IAE: 0.0086** | Median: 0.0068 | Plots: 21

| Plot | IAE | Score | Arms |
|------|-----|-------|------|
| edge_high_survival | 0.0012 | 0.9988 | 2 |
| edge_near_flat | 0.0029 | 0.9971 | 2 |
| synthetic_002 | 0.0033 | 0.9967 | 2 |
| edge_ci_shading | 0.0035 | 0.9965 | 2 |
| synthetic_003 | 0.0040 | 0.9960 | 2 |
| stress_gridlines | 0.0051 | 0.9949 | 2 |
| synthetic_001 | 0.0051 | 0.9949 | 2 |
| stress_annotation_heavy | 0.0052 | 0.9948 | 2 |
| stress_three_similar | 0.0060 | 0.9940 | 3 |
| synthetic_004 | 0.0062 | 0.9938 | 2 |
| synthetic_005 | 0.0068 | 0.9932 | 2 |
| edge_four_arms | 0.0070 | 0.9930 | 4 |
| stress_jpeg_artifact | 0.0074 | 0.9926 | 2 |
| stress_bw | 0.0084 | 0.9916 | 2 |
| stress_legend_overlap | 0.0085 | 0.9915 | 2 |
| edge_small_dense | 0.0096 | 0.9904 | 2 |
| stress_tiny | 0.0149 | 0.9851 | 2 |
| stress_stretched_tall | 0.0149 | 0.9851 | 2 |
| stress_blurry | 0.0151 | 0.9849 | 2 |
| stress_stretched_wide | 0.0215 | 0.9785 | 2 |
| edge_multi_panel | 0.0237 | 0.9763 | 4 |
| edge_cumulative_incidence | ERROR | ERROR | ? |

## Benchmarks
| Source | IAE |
|--------|-----|
| **Our pipeline** | **0.0086** |
| KM-GPT | 0.018 |
| Ethan Opus 4.6 | 0.0418 |

## Known Issues
1. edge_cumulative_incidence errors (tick detection doesn't handle y-axis 0.0-0.5)
2. edge_multi_panel worst performer (0.0237) — small panels challenge bbox detection
3. Combo stress tests (not in this baseline) range 0.001-0.088, BW+overlap worst
