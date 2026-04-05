"""
KMGen Metrics — IAE (Integrated Absolute Error) and related accuracy metrics.

IAE measures the area between the extracted step function and the ground truth
step function, normalized over [0, 1] time range. IAE=0 is perfect, IAE=1 is worst.

Compatible with Ethan's metric implementation in sunlab-kmgen for comparison.

Usage:
    # Compare an extraction JSON against a ground truth JSON:
    python metrics.py extraction.json ground_truth.json

    # Or use as a library:
    from metrics import compute_iae, compute_score
"""

import json
import sys
from pathlib import Path

import numpy as np


def _interp_step(xs, ys, query_xs):
    """
    Left-step interpolation: S(t) = value at the most recent step at or before t.

    For a KM curve, the survival at time t is the value set by the most recent
    event before t. Between events, survival is constant (flat step).
    """
    indices = np.searchsorted(xs, query_xs, side="right") - 1
    indices = np.clip(indices, 0, len(ys) - 1)
    return ys[indices]


def compute_iae(xs_extracted, ys_extracted, xs_truth, ys_truth, x_max):
    """
    Integrated Absolute Error between two step functions.

    Both functions are interpolated onto a merged x-grid, then the absolute
    difference is integrated via trapezoid rule. Time is normalized to [0, 1].

    Args:
        xs_extracted: time points of extracted curve
        ys_extracted: survival values of extracted curve
        xs_truth: time points of ground truth curve
        ys_truth: survival values of ground truth curve
        x_max: maximum time value (for normalization)

    Returns:
        IAE value in [0, 1] where 0 = perfect match
    """
    xs_e = np.asarray(xs_extracted, dtype=float)
    ys_e = np.asarray(ys_extracted, dtype=float)
    xs_t = np.asarray(xs_truth, dtype=float)
    ys_t = np.asarray(ys_truth, dtype=float)

    # Normalize time to [0, 1]
    xs_e_norm = xs_e / x_max
    xs_t_norm = xs_t / x_max

    # Merge x grids
    xs_merged = np.unique(np.concatenate([xs_e_norm, xs_t_norm]))
    xs_merged = xs_merged[(xs_merged >= 0) & (xs_merged <= 1)]

    # Interpolate both curves onto merged grid
    ya = _interp_step(xs_e_norm, ys_e, xs_merged)
    yt = _interp_step(xs_t_norm, ys_t, xs_merged)

    # Integrate absolute difference
    return float(np.trapezoid(np.abs(ya - yt), xs_merged))


def compute_ae_at_points(xs_extracted, ys_extracted, xs_truth, ys_truth, x_max, n_points=100):
    """
    Absolute error at evenly-spaced points across the curve.

    Returns:
        Array of absolute errors at n_points evenly-spaced time points.
    """
    xs_e = np.asarray(xs_extracted, dtype=float)
    ys_e = np.asarray(ys_extracted, dtype=float)
    xs_t = np.asarray(xs_truth, dtype=float)
    ys_t = np.asarray(ys_truth, dtype=float)

    eval_xs = np.linspace(0, x_max, n_points)
    ya = _interp_step(xs_e, ys_e, eval_xs)
    yt = _interp_step(xs_t, ys_t, eval_xs)

    return np.abs(ya - yt)


def compute_median_survival_error(xs_extracted, ys_extracted, xs_truth, ys_truth):
    """
    Absolute error in median survival time (first time S(t) <= 0.5).
    """
    def _find_median(xs, ys):
        for i in range(len(ys)):
            if ys[i] <= 0.5:
                return xs[i]
        return xs[-1]  # never crosses 0.5

    med_e = _find_median(xs_extracted, ys_extracted)
    med_t = _find_median(xs_truth, ys_truth)
    return abs(med_e - med_t)


def compute_score(extraction, ground_truth, x_max=None):
    """
    Compute all metrics comparing extraction against ground truth.

    Args:
        extraction: dict with 'arms' list, each arm has 'coordinates' with 't' and 's'
        ground_truth: dict with same structure
        x_max: max time for normalization (auto-detected if None)

    Returns:
        dict with: score, iae, ae_median, median_os_error, per_arm details
    """
    if x_max is None:
        x_max = extraction.get("axis", {}).get("x_max", 21)

    arm_results = []

    # Match arms by position (first extracted arm vs first truth arm, etc.)
    n_arms = min(len(extraction["arms"]), len(ground_truth["arms"]))

    for i in range(n_arms):
        ext_arm = extraction["arms"][i]
        truth_arm = ground_truth["arms"][i]

        # Extract (t, s) arrays
        xs_e = np.array([c["t"] for c in ext_arm["coordinates"]])
        ys_e = np.array([c["s"] for c in ext_arm["coordinates"]])
        xs_t = np.array([c["t"] for c in truth_arm["coordinates"]])
        ys_t = np.array([c["s"] for c in truth_arm["coordinates"]])

        iae = compute_iae(xs_e, ys_e, xs_t, ys_t, x_max)
        ae_points = compute_ae_at_points(xs_e, ys_e, xs_t, ys_t, x_max)
        median_err = compute_median_survival_error(xs_e, ys_e, xs_t, ys_t)

        arm_results.append({
            "label_extracted": ext_arm.get("label", f"Arm {i}"),
            "label_truth": truth_arm.get("label", f"Arm {i}"),
            "iae": iae,
            "ae_median": float(np.median(ae_points)),
            "ae_mean": float(np.mean(ae_points)),
            "ae_max": float(np.max(ae_points)),
            "median_os_error": median_err,
            "n_steps_extracted": len(xs_e),
            "n_steps_truth": len(xs_t),
        })

    mean_iae = np.mean([r["iae"] for r in arm_results])

    return {
        "score": max(0, 1 - mean_iae),
        "iae": mean_iae,
        "ae_median": float(np.median([r["ae_median"] for r in arm_results])),
        "median_os_error": float(np.mean([r["median_os_error"] for r in arm_results])),
        "n_arms": n_arms,
        "arms": arm_results,
    }


def print_report(result):
    """Print a human-readable metrics report."""
    print(f"\n{'='*50}")
    print(f"  KMGen Extraction Accuracy Report")
    print(f"{'='*50}")
    print(f"  Overall Score:  {result['score']:.4f}  (1.0 = perfect)")
    print(f"  Mean IAE:       {result['iae']:.4f}  (0.0 = perfect)")
    print(f"  Median AE:      {result['ae_median']:.4f}")
    print(f"  Median OS Err:  {result['median_os_error']:.2f} time units")
    print(f"  Arms compared:  {result['n_arms']}")

    for arm in result["arms"]:
        print(f"\n  ── {arm['label_extracted']} ──")
        print(f"     IAE:          {arm['iae']:.4f}")
        print(f"     AE median:    {arm['ae_median']:.4f}")
        print(f"     AE mean:      {arm['ae_mean']:.4f}")
        print(f"     AE max:       {arm['ae_max']:.4f}")
        print(f"     Median OS:    {arm['median_os_error']:.2f} time units")
        print(f"     Steps:        {arm['n_steps_extracted']} extracted vs {arm['n_steps_truth']} truth")

    print(f"\n  Benchmark: KM-GPT IAE = 0.018")
    print(f"  Benchmark: Ethan Opus 4.6 IAE = 0.0418")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python metrics.py extraction.json ground_truth.json")
        print("\nBoth files should have the format:")
        print('  {"arms": [{"coordinates": [{"t": 0.0, "s": 1.0}, ...]}]}')
        sys.exit(1)

    ext_path = Path(sys.argv[1])
    truth_path = Path(sys.argv[2])

    with open(ext_path) as f:
        extraction = json.load(f)
    with open(truth_path) as f:
        ground_truth = json.load(f)

    x_max = extraction.get("axis", {}).get("x_max", None)
    result = compute_score(extraction, ground_truth, x_max)
    print_report(result)

    # Optionally save results
    out_path = ext_path.with_stem(ext_path.stem + "_metrics")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Metrics saved to: {out_path}")
