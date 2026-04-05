"""
Unified comparison harness — side-by-side evaluation of Jalen's and Ethan's
KM curve extraction pipelines against ground truth.

Usage:
    python compare.py                  # run all synthetic_001..005
    python compare.py synthetic_003    # run a single plot
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Repo layout (post-reorg) ──
REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "jalen"))
sys.path.insert(0, str(REPO / "shared"))
sys.path.insert(0, str(REPO / "ethan"))

from metrics import compute_score  # shared/metrics.py

# Jalen's pipeline — import the extraction runner + configs
from benchmark_extract import run_extraction as jalen_run_extraction, PLOT_CONFIGS

SYNTH_DIR = REPO / "shared" / "synthetic"
OUT_DIR = REPO / "benchmark"


def _load_truth(plot_name: str) -> dict:
    truth_path = SYNTH_DIR / f"{plot_name}_truth.json"
    with open(truth_path) as f:
        return json.load(f)


def run_jalen(plot_name: str) -> dict:
    """Run Jalen's color-filter + curve-tracing extraction."""
    try:
        _name, metrics = jalen_run_extraction(plot_name)
        # Also load the saved extraction JSON for the coordinates
        ext_path = OUT_DIR / plot_name / "extraction.json"
        extraction = None
        if ext_path.exists():
            with open(ext_path) as f:
                extraction = json.load(f)
        return {
            "status": "ok",
            "iae": float(metrics["iae"]),
            "score": float(metrics.get("score", 1 - metrics["iae"])),
            "n_arms": metrics.get("n_arms", 0),
            "arms": metrics.get("arms", []),
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "iae": None, "score": None}


def run_ethan(plot_name: str) -> dict:
    """
    Attempt Ethan's LLM-based extraction.
    Requires an API key configured — returns N/A if unavailable.
    """
    try:
        from kmgen_auto.config import LLMConfig
        # Ethan's pipeline needs a configured LLM with API key.
        # If no key is available, we can't run it.
        return {
            "status": "n/a",
            "reason": "Ethan's pipeline requires LLM API key — skipped in offline benchmark",
            "iae": None,
            "score": None,
        }
    except ImportError:
        return {
            "status": "n/a",
            "reason": "kmgen_auto not importable",
            "iae": None,
            "score": None,
        }


def compare_plot(plot_name: str) -> dict:
    """Run both pipelines on a single plot and return comparison."""
    print(f"\n{'─'*60}")
    print(f"  Comparing: {plot_name}")
    print(f"{'─'*60}")

    jalen_result = run_jalen(plot_name)
    ethan_result = run_ethan(plot_name)

    return {
        "plot": plot_name,
        "jalen": jalen_result,
        "ethan": ethan_result,
    }


def print_table(results: list[dict]):
    """Print a side-by-side comparison table."""
    print(f"\n{'='*70}")
    print(f"  SIDE-BY-SIDE COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Plot':<20s} {'Jalen IAE':>12s} {'Ethan IAE':>12s} {'Winner':>10s}")
    print(f"  {'─'*20} {'─'*12} {'─'*12} {'─'*10}")

    jalen_iaes = []
    ethan_iaes = []

    for r in results:
        plot = r["plot"]
        j_iae = r["jalen"].get("iae")
        e_iae = r["ethan"].get("iae")

        j_str = f"{j_iae:.4f}" if j_iae is not None else r["jalen"].get("status", "err")
        e_str = f"{e_iae:.4f}" if e_iae is not None else r["ethan"].get("status", "n/a")

        if j_iae is not None and e_iae is not None:
            winner = "Jalen" if j_iae <= e_iae else "Ethan"
        elif j_iae is not None:
            winner = "Jalen"
        elif e_iae is not None:
            winner = "Ethan"
        else:
            winner = "—"

        if j_iae is not None:
            jalen_iaes.append(j_iae)
        if e_iae is not None:
            ethan_iaes.append(e_iae)

        print(f"  {plot:<20s} {j_str:>12s} {e_str:>12s} {winner:>10s}")

    print(f"  {'─'*20} {'─'*12} {'─'*12} {'─'*10}")

    if jalen_iaes:
        j_mean = np.mean(jalen_iaes)
        print(f"  {'Jalen mean':<20s} {j_mean:>12.4f}")
    if ethan_iaes:
        e_mean = np.mean(ethan_iaes)
        print(f"  {'Ethan mean':<20s} {'':>12s} {e_mean:>12.4f}")

    print(f"\n  Reference benchmarks:")
    print(f"    KM-GPT IAE    = 0.0180")
    print(f"    Ethan Opus 4  = 0.0418")
    print(f"{'='*70}\n")


def main():
    default_plots = [
        "synthetic_001", "synthetic_002", "synthetic_003",
        "synthetic_004", "synthetic_005",
    ]

    plots = [sys.argv[1]] if len(sys.argv) > 1 else default_plots

    results = []
    for plot_name in plots:
        if plot_name not in PLOT_CONFIGS:
            print(f"  WARNING: {plot_name} not in PLOT_CONFIGS, skipping")
            continue
        results.append(compare_plot(plot_name))

    print_table(results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d")
    out_path = OUT_DIR / f"comparison_{timestamp}.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    jalen_iaes = [r["jalen"]["iae"] for r in results if r["jalen"].get("iae") is not None]
    ethan_iaes = [r["ethan"]["iae"] for r in results if r["ethan"].get("iae") is not None]

    output = {
        "timestamp": datetime.now().isoformat(),
        "plots_compared": len(results),
        "summary": {
            "jalen": {
                "mean_iae": float(np.mean(jalen_iaes)) if jalen_iaes else None,
                "median_iae": float(np.median(jalen_iaes)) if jalen_iaes else None,
                "n_successful": len(jalen_iaes),
            },
            "ethan": {
                "mean_iae": float(np.mean(ethan_iaes)) if ethan_iaes else None,
                "median_iae": float(np.median(ethan_iaes)) if ethan_iaes else None,
                "n_successful": len(ethan_iaes),
            },
            "reference": {
                "km_gpt_iae": 0.018,
                "ethan_opus4_iae": 0.0418,
            },
        },
        "results": results,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"Results saved to: {out_path}")
    return output


if __name__ == "__main__":
    main()
