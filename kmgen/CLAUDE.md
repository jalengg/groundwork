# KMGen — KM Plot Reverse Engineering Research

## Project Goal
Reverse-engineer patient-level survival data from published Kaplan-Meier plot images using multimodal LLM reasoning.

## Team
- **Jalen + Ethan Rasmussen**: curve extraction
- **Andy Gao, Erick S**: downstream data generation pipeline
- **jimeng sun**: PI/advisor
- **Ethan's repo**: github.com/ethanrasmussen/sunlab-kmgen (`another_idea` branch)

## Architecture Decision
The LLM's role is **analyst/strategist** (identify image challenges, write tailored code), NOT pixel reader.
Code handles precision extraction. This plays to each tool's strengths.

## Research Workflow
This is an iterative research project. The conversation, prompt evolution, and dev history ARE the research artifacts.

### Every iteration MUST produce:
1. **Numbered run folder** in `runs/NNN_<slug>/`
   - `annotation.png` — detected step markers overlaid on original image
   - `extraction.json` — extracted coordinates
   - `metrics.json` — IAE scores (when ground truth available)
   - Copy of the extraction script used
2. **Prompt log entry** in `prompts/NNN_<slug>.md` with:
   - Date
   - The user's exact prompt/feedback (verbatim)
   - The approach taken
   - The result and what was learned
3. **Updated `prompts/README.md`** — index of all prompt entries with one-line summaries

### Before claiming an iteration is done:
- Show the annotation to the user
- Log the prompt entry
- Note what changed from the previous run

## Current State (as of run 008)
- **Pipeline**: Adaptive agent skill (`skill_kmgen.md`) — LLM analyzes image, picks from technique toolbox, writes tailored extraction code
- **Metrics**: `metrics.py` — IAE, point-wise AE, median survival error (compatible with Ethan's)
- **Reporting**: `report.py` — HTML report with side-by-side comparison, area-between-curves SVGs, human annotation checkboxes
- **Synthetic data**: 5 standard + 7 edge case plots with ground truth in `synthetic/`
- **Benchmark result**: Mean IAE = 0.0235 across 12 plots (KM-GPT: 0.018, Ethan Opus: 0.0418)
- **Validation**: Extraction validated against NCT03041311 published anchors (within 0.04-0.46 months)

## Research Findings
1. **Systematic downward bias** (Jalen, 2026-03-31): Extracted survival consistently underestimates truth across all standard KM plots. Reversed for cumulative incidence (curves going UP). Strongest at leftmost/rightmost extremes. Likely caused by topmost-pixel tracing + anti-aliasing asymmetry. See prompt 013 for full analysis.

## File Structure
```
sunlab-kmgen/
├── CLAUDE.md                # this file
├── pyproject.toml
├── uv.lock
├── shared/
│   ├── metrics.py           # IAE and related accuracy metrics (shared)
│   └── synthetic/           # synthetic plots + ground truth JSONs
├── jalen/
│   ├── benchmark_extract.py # benchmark extraction runner
│   ├── report.py            # HTML benchmark report generator
│   ├── skill_kmgen.md       # adaptive agent skill prompt (technique toolbox)
│   ├── generate_synthetic.py
│   ├── generate_edge_cases.py
│   ├── generate_stress_tests.py
│   ├── generate_combo_stress.py
│   └── edge_cases/          # real KM plots from papers
├── ethan/
│   ├── kmgen_auto/          # auto-optimization pipeline
│   ├── app.py               # Streamlit app
│   ├── run_iteration.py
│   ├── extract.py           # Ethan's extractor
│   ├── extract_cv.py        # CV-based extraction (legacy)
│   ├── compare_extractions.py
│   ├── prompts/             # prompt evolution log
│   └── iterations/          # iteration history
├── benchmark/               # shared benchmark results
├── comms/                   # shared knowledge store
└── runs/                    # numbered dev history
```

## Benchmarks
| Source | IAE | Notes |
|--------|-----|-------|
| KM-GPT | 0.018 | Published benchmark, larger dataset |
| **Our pipeline** | **0.0235** | 12 plots (5 standard + 7 edge cases) |
| Ethan Opus 4.6 | 0.0418 | 20 synthetic plots, iterative prompt optimization |
| Ethan Sonnet 4.6 | 0.0577 | With upscaling |
