# Prompt 007 — Max Step Filter + Lone Circle Fix
**Date:** 2026-03-16

## User Feedback (verbatim)
> It's actually getting quite close to accurate, good job. Curve separation will always remain difficult, and the level of difficulty will depend on the judgment of the initial multimodal script. You're right about the large orange drop, check the annotation you'll see there's a lone circle that's on top of empty space, 3) the ground truth will have to be human for now.

## Changes
- Added `max_drop_px=80` to `detect_stepdowns()` — rejects steps > ~0.16 in S-units
- Rationale: for a KM curve with N patients at risk, max single-event drop ≈ 1/N. Even with clustering, >0.15 is implausible.
- The lone circle at (t≈5.5, S≈0.30) was caused by the orange trace jumping when it lost pixels near the dashed median line exclusion zone

## Result
- Blue: 31 steps, Orange: 29 steps
- Lone stray circle eliminated
- New issue: orange curve has a gap (t=5.32→6.20) where real steps are missed because the dashed line exclusion removed real curve pixels in that region
- Overall: circles sit on curves, no stray circles

## Key Insights from User
1. Curve separation difficulty depends on initial multimodal judgment — this is inherent
2. Ground truth will be human visual verification for now
3. The approach is "getting quite close to accurate"
