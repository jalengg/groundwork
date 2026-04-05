# Prompt 017 — Stress Test Results
**Date:** 2026-03-31

## Results
All 10 adversarial stress tests pass. Mean IAE across all 21 scorable plots: **0.0086**.

| Category | Mean IAE | Plots |
|----------|----------|-------|
| Standard | 0.0046 | 5 |
| Edge cases | 0.0079 | 6 (excl. cumulative_incidence error) |
| **Stress tests** | **0.0097** | **10** |
| **All** | **0.0086** | **21** |

### Benchmarks
| Source | IAE |
|--------|-----|
| **Our pipeline** | **0.0086** |
| KM-GPT | 0.018 |
| Ethan Opus 4.6 | 0.0418 |

### New Techniques Added
- Spatial cluster separation (BW, same-color curves)
- 4x upscale for tiny images
- Safe bbox fallback for degenerate geometries
- Widened color thresholds for degraded images

### Key Finding
Spatial separation (top curve = arm 0, bottom = arm 1) is actually more robust than color separation for well-separated curves. Color separation is only needed when curves overlap AND have different colors.
