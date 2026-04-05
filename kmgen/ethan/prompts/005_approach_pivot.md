# Prompt 005 — Approach Pivot: LLM as Analyst, Code as Extractor
**Date:** 2026-03-16

## User Feedback (verbatim)
> Yes. I suspect that the multimodal processing is having an attention problem, it is unable to "follow the line", detect a stepdown one-by-one, determine with geometric accuracy where the stepdown is, and keep track of stepdowns it has already processed. Maybe just feeding the entire image to the LLM isn't enough. I am a little suspicious of the idea that super-resolutioning will help, like as a human, this is definitely a high enough resolution, why can't a robot handle it? What are some various approaches to make the stepdown detection extraction a bit more methodical? I noticed that Claude might not do well at getting coordinates from the multimodal processing itself, but maybe Claude is good at reading the image as a whole, detecting very high-level challenges (the lines overlap, the colors are low-contrast, the axes are very far away from the edge., etc etc) and writing bespoke code to programmatically detect the bounding box in anticipation of these challenges. Would that be a better approach? And then we can include the python scripts as part of the iteration files, and then start iterating on a skill file or a prompt that is able to do this.

## Key Insight from User
The LLM's role should be **analyst/strategist**, not **pixel reader**:
- LLM looks at image → identifies high-level challenges (overlapping lines, colors, axis layout)
- LLM writes bespoke Python code tailored to those challenges
- Python code does the actual pixel-level extraction programmatically
- This plays to the LLM's strengths (reasoning, code generation) and avoids its weaknesses (precise coordinate reading)

## Approach Change
- **Old**: LLM visually reads coordinates → draws circles
- **New**: LLM analyzes image → writes tailored extraction code → code detects steps → draws circles
- Include generated Python scripts in iteration folders
- Build toward a reusable skill/prompt for this workflow

## Status
Planning phase — need to design the hybrid LLM+CV pipeline
