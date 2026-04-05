# Prompt 008 — Ethan's Progress + Team Context
**Date:** 2026-03-31

## Team Structure
- **Jalen + Ethan**: curve extraction
- **Andy, Erick, others**: downstream data generation pipeline
- **jimeng sun**: PI/advisor

## Ethan's Approach (different from ours)
- Streamlit app for iterative prompt optimization
- LLM writes extraction code in one-shot, then iteratively improves it
- Accuracy metric: **IAE (Integrated Absolute Error)** against synthetic KM plots with known ground truth
- 20 synthetic KM plots, seed=42, 15 max iterations, patience k=5
- Repo: https://github.com/ethanrasmussen/sunlab-kmgen/tree/another_idea

## Ethan's Results
| Model | Line Type | IAE | Notes |
|-------|-----------|-----|-------|
| Opus 4.6 | solid | **0.0418** | best result |
| Opus 4.6 | dashed | 0.0485 | surprisingly good |
| Sonnet 4.6 | solid (w/ upscaling) | 0.0577 | new tonight |
| Sonnet 4.6 | solid (no upscaling) | 0.0928 | last week |
| **KM-GPT** | — | **0.018** | benchmark (larger dataset) |

## Key Findings
- Upscaling helps: Sonnet IAE 0.0928 → 0.0577
- Dashed lines work surprisingly well
- **Smooth curves still a huge challenge** — Jalen suggested different processing mode for smooth vs steppy curves
- Erick mentioned Guyot algorithm + ImmPort for open source validation data

## Clinical Trial Data for Validation
- **NCT03041311**: has SAP, Protocol, results. Publication: pubmed 33348420. Figure 3 has survival curves.
- **NCT02499770**: has SAP, Protocol, results. Figure 4 has survival curve.

## Our Image (kmgpt_plot.png)
The plot we've been working with appears to be from the NCT03041311 trial (Trilaciclib vs Placebo PFS) — same figure.
