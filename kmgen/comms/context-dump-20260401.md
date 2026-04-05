# Context Dump — 2026-04-01

This captures the full state of the KMGen project for session continuity.

## What We Built (This Session)

### Pipeline
- **Hybrid LLM+CV extraction**: LLM analyzes image challenges → writes bespoke Python extraction code → code does pixel-level extraction
- **Key file**: `skill_kmgen.md` — adaptive agent skill prompt with technique toolbox
- **Key file**: `benchmark_extract.py` — automated extraction pipeline with per-plot-type configs
- **Key file**: `metrics.py` — IAE metric (compatible with Ethan's implementation)
- **Key file**: `report.py` — HTML benchmark report with area-between-curves SVGs + human annotation

### Results
- **Mean IAE: 0.0086** across 21 standard/edge plots (2x better than KM-GPT's 0.018)
- **Mean IAE: 0.015** across all 31 plots including adversarial combo stress tests
- Best plot: edge_high_survival at 0.0012
- Worst passing: edge_multi_panel at 0.0237
- 1 error: edge_cumulative_incidence (tick detection doesn't handle y-axis 0-0.5)

### Key Breakthrough
**Tick-mark bbox detection** (run 010): matplotlib adds ~5% axis padding. Our bbox detector was finding the frame edges (includes padding) instead of tick marks (actual data positions). Fixing this gave 3.6x IAE improvement (0.0233 → 0.0064).

### Research Finding
**Systematic downward bias** (Jalen, 2026-03-31): extraction consistently underestimates survival. Root cause was bbox calibration, not pixel reading. Centroid tracing made it worse (KM values are at TOP of step, not center).

## Test Suite
- 5 standard synthetic (Weibull, seed=42)
- 7 edge cases (CI shading, cumulative incidence, 4-arm, multi-panel, near-flat, tiny, high survival)
- 10 single stress tests (blur, JPEG, BW, stretched, tiny, legend overlap, gridlines, 3-similar, annotations)
- 10 combo stress tests (tiny+blur, BW+overlap, triple degradation, DIEP-like, etc.)
- 35 degraded variants of real paper images (visual check only)
- Ground truth for all synthetic plots in `synthetic/*_truth.json`

## Team & Collaboration
- **Jalen**: CV pipeline, this work
- **Ethan Rasmussen**: iterative prompt optimization (`kmgen_auto/` package, Streamlit app)
  - His repo: github.com/ethanrasmussen/sunlab-kmgen (`another_idea` branch)
  - His best: IAE 0.0418 (Opus 4.6 on 20 synthetic plots)
- **Andy Gao**: downstream pipeline, paper writing
- **Erick S**: Guyot algorithm implementation
- **jimeng sun**: PI/advisor
- Pushed to: `jalengg/sunlab-kmgen` branch `jalen/cv-pipeline`

## Architectural Decisions (from comms/decisions.md)
1. LLM as analyst, code as extractor (not LLM reading pixels directly)
2. Tick-mark bbox over frame bbox
3. Topmost pixel for KM step functions (not centroid)
4. Daily grain sampling (1/30 month intervals)
5. Adaptive technique toolbox over fixed pipeline

## Agent Team (attempted, OOM'd)
- Tried 7-agent team → laptop OOM, tmux crashed
- Tried 3-agent lean team → quality-gate ran successfully (PASS, baseline confirmed)
- visual-qa was running when session interrupted
- Decision: move to 200k model with better documentation discipline instead

## File Structure
```
/tmp/sunlab-kmgen/
├── skill_kmgen.md           # adaptive extraction skill prompt
├── benchmark_extract.py     # main extraction pipeline
├── metrics.py               # IAE scoring
├── report.py                # HTML report generator
├── generate_synthetic.py    # standard synthetic data
├── generate_edge_cases.py   # edge case data
├── generate_stress_tests.py # single stress tests
├── generate_combo_stress.py # combined stress tests
├── comms/                   # shared knowledge store
│   ├── baseline.md          # current IAE numbers
│   ├── edge-cases.md        # known failures
│   ├── decisions.md         # ADRs
│   ├── prompt-evolution.md  # what helped/hurt
│   ├── gate-status.md       # last quality gate result
│   └── session-log.md       # session summaries
├── synthetic/               # all test plots + ground truth
├── benchmark/               # extraction results + metrics
├── runs/                    # dev history (001-012)
├── prompts/                 # prompt evolution (001-018)
├── edge_cases/              # real paper images
├── kmgen_auto/              # Ethan's optimizer (don't touch)
└── app.py                   # Ethan's Streamlit UI
```

## What's Next
1. Fix edge_cumulative_incidence (handle non-standard y-axis)
2. Improve BW curve separation (line-style detection for solid vs dashed)
3. Build unified harness comparing: one-shot multimodal, one-shot coder, iterative coder, adaptive coder
4. Integrate with Ethan's Streamlit app
5. PR to ethanrasmussen/sunlab-kmgen with unified repo structure
6. Agent team design exists in comms/ but needs lighter execution model
