# Prompt 010 — Adaptive Agent Design
**Date:** 2026-03-31

## User Feedback (verbatim)
> 1) this is assuming we always have orange/blue etc colors, 2) not a hard requirement, the agent can decide whether to incorporate these techniques, 3) Yes, 4) Sure, High-level: I dont want the prompt to rigidly say "you must have a script that does all this stuff", but have a coding agent consider all of these "features" and "techniques" and determine if it is a good fit for the curve at hand. curve-tracing vs step-detection vs guyot will also be another "toolbox" decision. Do you understand what I'm trying to say?

> Yes. [to drafting the skill/prompt]

## Design Philosophy
The agent encodes **knowledge** (techniques and when they work), not **procedure** (fixed steps).
It reasons about each image's characteristics and picks the right combination from a toolbox.

## Toolbox Techniques
- Color separation: HSL hue matching, RGB thresholds, or both — depends on actual curve colors
- Bbox detection: programmatic axis label/tick detection, or OCR-assisted
- Upscaling: 2x LANCZOS for thin lines, skip if already high-res
- Step detection: for steppy regions with resolvable individual drops
- Curve sampling: for smooth/dense regions, sample at fixed intervals
- Guyot reconstruction: when patients-at-risk table is available
- Dashed/dotted line handling
- Legend/text exclusion
- Continuity tracking for curve tracing
- Monotonicity enforcement

## Status
Drafting the adaptive agent skill/prompt.
