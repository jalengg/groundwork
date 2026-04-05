# Prompt 006 — CV-Based Extraction Pipeline
**Date:** 2026-03-16

## User Feedback (verbatim)
> Yes. I suspect that the multimodal processing is having an attention problem, it is unable to "follow the line", detect a stepdown one-by-one, determine with geometric accuracy where the stepdown is, and keep track of stepdowns it has already processed. Maybe just feeding the entire image to the LLM isn't enough. I am a little suspicious of the idea that super-resolutioning will help, like as a human, this is definitely a high enough resolution, why can't a robot handle it? What are some various approaches to make the stepdown detection extraction a bit more methodical? I noticed that Claude might not do well at getting coordinates from the multimodal processing itself, but maybe Claude is good at reading the image as a whole, detecting very high-level challenges (the lines overlap, the colors are low-contrast, the axes are very far away from the edge., etc etc) and writing bespoke code to programmatically detect the bounding box in anticipation of these challenges. Would that be a better approach? And then we can include the python scripts as part of the iteration files, and then start iterating on a skill file or a prompt that is able to do this.

## Approach: Hybrid LLM+CV Pipeline
**LLM role**: Analyst — looks at image, identifies challenges, writes code
**Code role**: Extractor — color filtering, curve tracing, step detection

### Pipeline steps:
1. `analyze_colors.py` — sample actual pixel colors at known curve locations
2. `extract_cv.py` — color filter → trace → detect steps → annotate

### Key technical decisions:
- **Color thresholds**: Blue (B-R>30, R<170), Orange (R>210, R-B>60)
- **Tracking tracer**: follows curve continuously instead of "topmost pixel per column" — prevents jumps to stray text pixels
- **Legend/text exclusion**: mask out top-right 30% of plot area
- **Dashed line exclusion**: ±3 cols around median line at month 5.9
- **Continuity constraint**: reject jumps > 100px

### Results:
- Blue: 31 steps, Orange: 30 steps
- Circles sit ON the curve lines (verified visually)
- Still have overlap confusion in months 0-5 (both masks capture similar shapes)
- Orange step at t=5.42 shows S: 0.54→0.30 drop — might be real but suspiciously large

### Remaining issues:
1. Color separation in overlap regions still imperfect
2. Hard to verify blue vs orange circles are on the correct curve
3. No ground truth to validate accuracy

## Key Insight
> The LLM should be the strategist, not the pixel reader. Code handles precision, LLM handles reasoning about what code to write.
