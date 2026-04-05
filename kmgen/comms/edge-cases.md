# Known Edge Cases & Failure Modes

## Open Issues

| Category | Worst Plot | IAE | Root Cause | Fix Type |
|----------|-----------|-----|------------|----------|
| Cumulative incidence | edge_cumulative_incidence | ERROR | Tick detection assumes y-axis 0-1 | code |
| Multi-panel | edge_multi_panel | 0.0237 | Small panels, bbox detection fails | code |
| BW + overlap | stress_combo_bw_overlap | ~~0.0822~~ **0.0075** | Fixed: coverage-based arm assignment + continuity tracking | ✅ resolved |
| BW + clutter | stress_combo_grid_annotation_bw | ~~0.0880~~ **0.0744** | Improved: jump rejection + run-length filter, still Hard tier | code+prompt |
| DIEP-like | stress_diep_like | 0.0540 | 8 CI arms, dual tiny panels | code |
| Triple degradation | stress_combo_jpeg_blurry_dark | 0.0412 | JPEG + blur + dark bg | code |

## Resolved Issues
| Issue | Resolution | Run |
|-------|-----------|-----|
| Systematic downward bias | Tick-mark bbox detection (was using frame) | 010 |
| Stray circle in dense zone | Max step filter (80px) | 007 |
| Phantom jump at curve gap | Continuity tracking + max jump rejection | 005 |

## Difficulty Tiers
- **Easy (IAE < 0.01)**: Standard plots, single degradation, color separation works
- **Medium (0.01-0.05)**: Tiny images, stretched, multi-arm, moderate degradation
- **Hard (0.05+)**: BW + overlap, BW + clutter, DIEP-like multi-panel CI
