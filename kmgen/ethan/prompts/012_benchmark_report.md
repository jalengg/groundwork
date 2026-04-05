# Prompt 012 — Benchmark Report Infrastructure
**Date:** 2026-03-31

## User Request (verbatim)
> Yes, also expand the documentation aspect of the repo to have the ability to annotate 12 plots for visual human checks including area calculation, synthetic data used, etc. etc

## Changes
- `report.py` — generates HTML report with:
  - Summary table: all plots with IAE, AE median, median OS error
  - Per-plot cards with side-by-side original vs annotated images
  - Inline SVG area charts showing the IAE (shaded area between extracted and truth curves)
  - Synthetic data info (x_max, n_arms, total truth steps)
  - Human annotation section per plot: checkboxes for accuracy, step detection, color separation, bbox alignment + free-text notes
  - Benchmark references (KM-GPT 0.018, Ethan Opus 0.0418)
  - Color-coded IAE: green (<0.03), orange (<0.06), red (>=0.06)
- Updated CLAUDE.md with current file structure
- Dispatched extraction agent on all 12 synthetic plots

## Status
Extraction agent running on 12 plots. Report will be generated after metrics are computed.
