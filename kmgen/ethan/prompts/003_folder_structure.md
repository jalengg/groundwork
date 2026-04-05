# Prompt 003 — Folder Structure & Research Tracking
**Date:** 2026-03-16

## User Feedback (verbatim)
> Reexamine the bounding box, it looks off?

> I don't see the new bounding box. Can we decide on a more organized file system? under iterations, there should be folders that are dated and timestamped, and within each folder should be the calibration photo, and the red circle photo.

> You didn't update the prompt log, can you add hooks to this project so we don't forget the folder structure, the prompt evolution, etc? This conversation itself, and the claude code learning, is part of the research project.

## Changes Made
- Restructured iterations into `iterations/YYYYMMDD_HHMMSS/` folders containing `calibration.png` + `annotation.png`
- Consolidated `calibrate.py` + `annotate_manual.py` into single `run_iteration.py`
- Added `CLAUDE.md` to kmgen project with workflow conventions
- Created prompt index (`prompts/README.md`)

## Key Insight
The research process (prompt iteration, Claude's learning curve, what works/doesn't) is itself a research artifact. Must be tracked systematically.

## Iteration Output
- `iterations/20260316_105630/calibration.png` — bbox (126, 78, 714, 536)
- `iterations/20260316_105630/annotation.png` — 33 blue + 32 orange steps
- Bbox still potentially off per user feedback — needs further calibration
