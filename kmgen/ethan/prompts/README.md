# Prompt Evolution Log

| # | Date | Slug | Summary |
|---|------|------|---------|
| 001 | 2026-03-16 | initial | First attempt: manual visual extraction, all circles off due to bad bbox |
| 002 | 2026-03-16 | bbox_calibration | Added calibration crosses, iterated bbox from (128,68,718,553) → (130,84,714,533) → (126,78,714,536) |
| 003 | 2026-03-16 | folder_structure | Restructured to timestamped iteration folders, added CLAUDE.md, prompt tracking |
| 004 | 2026-03-16 | bbox_pixel_detection | Programmatic bbox detection via pixel analysis; bbox (126,78,714,536)→(163,95,783,604); circles now on curves |
| 005 | 2026-03-16 | approach_pivot | LLM as analyst/strategist, code as pixel extractor; design hybrid pipeline |
| 006 | 2026-03-16 | cv_extraction | Built hybrid pipeline: color filter→trace→detect; circles land on curves; overlap separation still imperfect |
| 007 | 2026-03-16 | max_step_filter | Added max step size filter (80px); removed stray circle; gap in orange near dashed line |
| 008 | 2026-03-31 | ethan_update | Ethan's iterative prompt approach: IAE 0.0418 (Opus), KM-GPT benchmark 0.018; smooth curves still hard |
| 009 | 2026-03-31 | dual_mode_design | Step detection (steppy) + curve sampling (smooth) + Guyot comparison; IAE convergence test |
| 010 | 2026-03-31 | adaptive_agent | Adaptive agent that reasons about image and picks from technique toolbox; drafting skill |
| 011 | 2026-03-31 | iae_implementation | IAE metric implemented; cross-consistency between our two methods: IAE 0.0098 |
| 012 | 2026-03-31 | benchmark_report | HTML report generator with area charts, human annotation, side-by-side comparison |
| 013 | 2026-03-31 | bias_finding | **Jalen's finding**: systematic downward bias in extraction; restructured to runs/ dev history |
| 014 | 2026-03-31 | bias_correction | Centroid tracing made bias WORSE (reverted); daily grain kept; IAE 0.0235→0.0233 |
| 015 | 2026-03-31 | tick_bbox_fix | **ROOT CAUSE: matplotlib axis padding.** Tick-mark bbox detection: IAE 0.0233→0.0064 (3.6x improvement, 2.8x better than KM-GPT) |
| 016 | 2026-03-31 | stress_tests | Adversarial synthetic data + real paper test (breast reconstruction KM plots) |
| 017 | 2026-03-31 | stress_results | All 10 stress tests pass; mean IAE 0.0086 across 21 plots; 2x better than KM-GPT |
| 018 | 2026-03-31 | combo_stress | 10 combined anti-pattern tests; BW is the killer; DIEP recreation at 0.054; overall 31 plots IAE 0.015 |
