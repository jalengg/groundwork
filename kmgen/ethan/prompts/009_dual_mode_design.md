# Prompt 009 — Dual Mode Design: Step Detection + Curve Sampling
**Date:** 2026-03-31

## User Feedback (verbatim)
> So we will have Guyot and take advantage of it whenever a table is provided. However I would also like to have a CV-based way of tracing that smooth curve and getting numbers out of it. It might be difficult to put red circles and get so many close-together coordinates of "step-downs", instead we can define a curve tracing algorithm and create coordinates out of that, maybe with a minimum granularity like once per day or something, in coordination with the axis scale * num_days

> Yep, and then we will do this and Guyot at the same time and see if they converge, we will compare each approach's IAE.

## Design
Two extraction modes, applied per-region of each curve:

### Mode 1: Step Detection (steppy regions)
- Detect discrete step-downs from pixel drops
- Output exact event coordinates
- Red circles on annotation
- High confidence

### Mode 2: Curve Sampling (smooth/dense regions)
- Sample the curve trace at fixed intervals (e.g., daily granularity)
- Output (time, survival) pairs — no circles, just the curve
- Lower confidence but still useful

### Zone Classification
- Compute local slope of the traced curve
- Steep diagonal (slope > threshold) = smooth zone → Mode 2
- Flat + vertical drops = steppy zone → Mode 1

### Comparison Plan
- Run both: CV-based extraction (step detection + curve sampling) AND Guyot (when at-risk table available)
- Compare IAE of each approach
- See if they converge

## Output Format
Each coordinate gets a `method` field:
- `"step"` — detected step-down (Mode 1)
- `"sample"` — curve trace sample (Mode 2)
- `"guyot"` — reconstructed from at-risk table (future)
