# Architectural Decisions

## ADR-001: LLM as analyst, code as extractor (2026-03-16)
**Decision**: The LLM analyzes image challenges and writes bespoke extraction code. It does NOT read pixel coordinates directly.
**Rationale**: LLMs can't precisely read coordinates from images (proved in runs 001-002). They CAN reason about image challenges and write code.
**Status**: Active, validated by IAE results.

## ADR-002: Tick-mark bbox detection over frame detection (2026-03-31)
**Decision**: Detect axis tick marks, not frame/spine edges, for bounding box calibration.
**Rationale**: Matplotlib adds ~5% padding. Frame detection includes padding → systematic bias. Tick marks are at actual data positions.
**Status**: Active. Caused 3.6x IAE improvement.

## ADR-003: Topmost pixel for KM step functions (2026-03-31)
**Decision**: Use topmost colored pixel per column, not centroid.
**Rationale**: KM survival value IS at the top of the step (stays at higher level until next event). Centroid puts us below the mathematical value.
**Status**: Active. Centroid was tested and reverted (made bias worse).

## ADR-004: Daily grain sampling (2026-03-31)
**Decision**: Sample curve at 1-day intervals (1/30 month) as the natural resolution floor.
**Rationale**: Clinical trial events are recorded by date. Daily = pixel resolution at typical image sizes.
**Status**: Active.

## ADR-005: Adaptive technique toolbox over fixed pipeline (2026-03-31)
**Decision**: Agent picks techniques per-image from a toolbox rather than following a fixed 10-step recipe.
**Rationale**: Different images need different approaches (HSL vs RGB, upscaling vs not, step detection vs curve sampling).
**Status**: Active. Encoded in skill_kmgen.md.

## ADR-006: Unified jalen/ethan/shared repo structure (2026-04-02)
**Decision**: Reorganize the repository into `jalen/`, `ethan/`, and `shared/` subdirectories for semi-independent collaboration.
**Rationale**: Both Jalen's and Ethan's pipelines can coexist in the same repo without stepping on each other. The `shared/` directory holds the benchmark harness (`compare.py`, test plots, ground truth) so both pipelines are evaluated against the same standard. This avoids fork divergence while maintaining development independence.
**Status**: Active. PR ethanrasmussen/sunlab-kmgen#1 opened with this structure.
