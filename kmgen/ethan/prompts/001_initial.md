# Prompt 001 — Initial Request
**Date:** 2026-03-16

## User Prompt
Hello, do you know what KM plots are? So the problem is that medical journals will publish studies with KM survival plots, but the underlying patient data is confidential. This is where our research comes in. We can take the KM image and "reverse engineer" the patient data by taking the image and using a multi-modal LLM to detect the exact coordinates where the graph "ticks down" or "ticks over", using that math to define the bounds of our time series patient data. Here's what I want to do: 1) claude takes the image, 2) creates a list of coordinates where the line ticks down, and then draws red circles on those coordinates so the user can see/diagnose where the multimodal reasoning is missing ticks. The rest of the pipeline we will worry about later.

## Approach
- Claude Code visually inspected the KM plot (no API call)
- Estimated plot bounding box in pixels by eye
- Manually listed step-down coordinates in data space (time, survival)
- Mapped data coords → pixel coords and drew red circles

## Result
- 33 blue (Trilaciclib) + 32 orange (Placebo) steps detected
- **Every circle was off** — the data→pixel mapping (plot bbox) was miscalibrated
- Circles followed the general trajectory but didn't sit on the actual curve lines
