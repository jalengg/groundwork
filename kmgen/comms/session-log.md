# Session Log (last 10 sessions)

## Session 001 — 2026-03-31
**Duration**: ~4 hours
**Summary**: Built complete KMGen CV extraction pipeline from scratch. Went from manual visual extraction (every circle off) to automated pipeline with IAE 0.0086 across 21 plots. Key breakthrough: tick-mark bbox detection fixed systematic bias (3.6x improvement). Created 31 synthetic test plots including 10 adversarial stress tests. Validated against NCT03041311 trial data (within 0.04-0.46 months of published anchors). Pushed to jalengg/sunlab-kmgen fork. Designed agent team architecture.
**Key findings**: (1) Jalen's systematic downward bias discovery led to root cause (matplotlib padding), (2) BW is the hardest anti-pattern, (3) Spatial separation > color separation when curves are well-separated.

## Session 002 — 2026-04-02
**Summary**: Major repo reorganization and BW extraction improvement cycle. Restructured repo into `jalen/`, `ethan/`, `shared/` subdirectories for semi-independent collaboration. Built `compare.py` unified comparison harness. Opened PR ethanrasmussen/sunlab-kmgen#1 with the unified structure. Added 4 new BW curve separation techniques to `skill_kmgen.md` (coverage-based arm assignment, continuity tracking over spatial sorting, jump rejection for annotation blobs, horizontal run-length filter for text removal). Engineer working on fixing `extract_bw_2arm` in `benchmark_extract.py`. `stress_bw` regression caught at IAE 0.2159 during post-reorg benchmark.
**Key findings**: (1) BW overlap (0.0822) and grid+BW (0.0880) remain the hardest failures, (2) Vertical position cannot reliably assign arm identity in BW — coverage-based assignment is the fix, (3) Repo reorg enables both pipelines to coexist with shared benchmark harness.
