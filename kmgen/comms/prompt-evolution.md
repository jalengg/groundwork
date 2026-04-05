# Prompt/Skill Evolution Log

| Date | Change | IAE Before | IAE After | Delta | Verdict |
|------|--------|-----------|----------|-------|---------|
| 2026-03-16 | Manual visual extraction (baseline) | N/A | N/A | — | Failed |
| 2026-03-16 | Programmatic bbox detection | N/A | N/A | — | Improved |
| 2026-03-16 | Hybrid LLM+CV pipeline (color filter + tracing) | N/A | N/A | — | Major improvement |
| 2026-03-16 | Max step filter (80px) | N/A | N/A | — | Fixed stray circles |
| 2026-03-31 | Adaptive agent skill (skill_kmgen.md) | N/A | 0.0235 | — | First benchmark |
| 2026-03-31 | Daily grain sampling | 0.0235 | 0.0233 | -0.0002 | Marginal |
| 2026-03-31 | Centroid tracing (REVERTED) | 0.0233 | 0.0262 | +0.0029 | Worse |
| 2026-03-31 | Tick-mark bbox detection | 0.0233 | 0.0064 | -0.0169 | **Breakthrough** |
| 2026-03-31 | Stress test extraction functions | 0.0064 | 0.0086 | +0.0022 | More plots added |
| 2026-03-31 | Combo stress test functions | 0.0086 | 0.0150 | +0.0064 | Harder plots added |
| 2026-04-02 | BW Curve Separation techniques added to skill | 0.0822 (BW overlap), 0.0880 (grid+BW) | TBD | — | New toolbox section |

### 2026-04-02: Black-and-White Curve Separation Techniques

**Motivation**: Two stress tests exposed BW as the hardest failure mode:
- `stress_combo_bw_overlap` IAE = 0.0822 — spatial separation fails when curves cross
- `stress_combo_grid_annotation_bw` IAE = 0.0880 — gridlines create false clusters in BW

**What was added to skill_kmgen.md** (new "Black-and-White Curve Separation" section with 4 techniques):
1. **Coverage-based arm assignment** — Solid lines have ~100% column coverage; dashed have ~60-70%. Assign arm identity by coverage ratio, not vertical position. Swap arm traces if arm 0 has lower coverage than arm 1.
2. **Continuity tracking over spatial sorting** — Use nearest-to-previous-position for arm assignment in all columns. Spatial sorting (top=arm0, bottom=arm1) fails at crossings. Only use spatial sorting for initial columns before prev_positions are established.
3. **Jump rejection for annotation blobs** — Annotations create dark clusters that hijack the tracer. Set MAX_JUMP_PX (~40px after 2x upscale). Carry forward if nearest cluster exceeds threshold.
4. **Horizontal run-length filter for text removal** — Remove horizontal dark pixel runs shorter than ~8px per row before tracing. Letter strokes are short (3-8px); KM segments are long. Strips annotation text cleanly.

Also added a key principle: "In BW plots, vertical position does not reliably indicate arm identity."
