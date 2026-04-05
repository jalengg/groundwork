"""
Compare two extractions against each other using IAE.
Since we don't have ground truth for the real image, we compare
the skill test output vs the extract_cv.py output to measure consistency.
We can also use either as a pseudo-truth to score the other.
"""

import json
import numpy as np
from metrics import compute_score, print_report

# Load skill test extraction (new format with 'arms')
with open("/mnt/c/Users/jalen/kmgen/iterations/skill_test/extraction.json") as f:
    skill_ext = json.load(f)

# Load extract_cv.py output (old format with 'blue_steps' / 'orange_steps')
with open("/mnt/c/Users/jalen/kmgen/iterations/20260316_144259/results.json") as f:
    cv_raw = json.load(f)

# Convert old format to new format
cv_ext = {
    "axis": {"x_min": 0, "x_max": 21},
    "arms": [
        {
            "label": "Trilaciclib (extract_cv)",
            "coordinates": [{"t": 0.0, "s": 1.0}] + [
                {"t": s["t"], "s": s["s_after"]} for s in cv_raw["blue_steps"]
            ]
        },
        {
            "label": "Placebo (extract_cv)",
            "coordinates": [{"t": 0.0, "s": 1.0}] + [
                {"t": s["t"], "s": s["s_after"]} for s in cv_raw["orange_steps"]
            ]
        }
    ]
}

# Compare: treat skill extraction as "truth", score extract_cv against it
print("=" * 60)
print("  Comparing extract_cv.py vs skill_test extraction")
print("  (skill_test treated as reference)")
print("=" * 60)
result = compute_score(cv_ext, skill_ext, x_max=21)
print_report(result)

# Also compare in reverse
print("\n" + "=" * 60)
print("  Comparing skill_test vs extract_cv.py")
print("  (extract_cv treated as reference)")
print("=" * 60)
result2 = compute_score(skill_ext, cv_ext, x_max=21)
print_report(result2)
