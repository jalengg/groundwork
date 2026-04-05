# Prompt 018 — Combined Stress Test Results
**Date:** 2026-03-31

## User Request (verbatim)
> We need more tiny stress tests, blurry/stretched/tiny/blackwhite/significantly overlap, lots of flatness, lots of color collisions. I like the last Breast reconstruction complications after postmastectomy proton radiation edge case in the HTML file too, please look at that image and recreate it. The stress tests should start combining and mix/matching anti-patterns and noisiness

## Results (10 combined stress tests)
Mean IAE: 0.0375 | Median: 0.0324 | All 10/10 extracted

### Difficulty tiers emerged:

**Easy (IAE < 0.02):** Color separation still works despite degradation
- flat_overlap (0.0012), multi_panel_clean (0.0064), 4arm_tiny (0.0175), tiny_bw (0.0198)

**Medium (0.02-0.05):** Degradation hurts but results still useful
- tiny_blurry (0.0271), stretched_dense (0.0378), jpeg_blurry_dark (0.0412)

**Hard (0.05+):** Fundamental separation challenges
- diep_like (0.0540) — 8 CI arms across tiny dual panels
- bw_overlap (0.0822) — no color + curves cross each other
- grid_annotation_bw (0.0880) — maximum visual noise + no color

### Key Finding
**BW is the killer anti-pattern.** All three worst results have BW. When curves are the same color, the pipeline relies on spatial separation (top cluster vs bottom cluster). This breaks when:
- Curves overlap (no spatial gap)
- Gridlines/annotations create false clusters
- Resolution is too low to distinguish line styles (solid vs dashed)

Color-based extraction is robust even under triple degradation (JPEG+blur+dark = 0.041).

### New Techniques Added
- `extract_tiny_blurry_2arm` — relative-channel color detection for washed-out images
- `extract_dark_2arm` — contrast stretch for dark backgrounds
- `extract_diep_multi_panel` — dual-panel CI with lenient bbox detection
- `detect_bbox_tiny_panel` — fallback for panels too small for standard detection

## Overall Benchmark (31 scorable plots)
| Category | Count | Mean IAE |
|----------|-------|----------|
| Standard | 5 | 0.0046 |
| Edge cases | 6 | 0.0079 |
| Single stress | 10 | 0.0097 |
| **Combo stress** | **10** | **0.0375** |
| **All** | **31** | **0.0150** |
