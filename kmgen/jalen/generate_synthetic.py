"""
Synthetic Kaplan-Meier plot generator.

Generates paired (PNG image, JSON ground truth) for evaluating KM extraction
pipelines. Uses Weibull distributions for survival times, matching Ethan's
approach in sunlab-kmgen.

Output JSON format (compatible with metrics.py):
    {"arms": [{"label": "...", "coordinates": [{"t": 0.0, "s": 1.0}, ...]}],
     "axis": {"x_min": 0, "x_max": N}}

Usage:
    python generate_synthetic.py                     # 5 plots, seed=42
    python generate_synthetic.py --n 20 --seed 99    # 20 plots, seed=99
    python generate_synthetic.py --out /path/to/dir  # custom output dir
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Weibull KM simulation ───────────────────────────────────────────────

def simulate_km(n_patients, x_max, weibull_shape, weibull_scale, censor_rate, rng):
    """
    Simulate Kaplan-Meier step function from Weibull event times.

    Returns arrays of (times, survival) representing the step-down coordinates,
    starting at (0, 1.0) and ending at or before x_max.
    """
    # Event times from Weibull
    event_times = weibull_scale * rng.weibull(weibull_shape, size=n_patients)

    # Random censoring: uniform over [0, x_max]
    censor_times = rng.uniform(0, x_max * 1.2, size=n_patients)
    is_censored = rng.random(n_patients) < censor_rate

    # Observed time = min(event, censor) if censored, else event
    observed = np.where(is_censored, np.minimum(event_times, censor_times), event_times)
    event_occurred = np.where(is_censored, event_times <= censor_times, True)

    # Cap at x_max
    observed = np.clip(observed, 0, x_max)

    # Sort by observed time
    order = np.argsort(observed)
    observed = observed[order]
    event_occurred = event_occurred[order]

    # Build KM curve
    times = [0.0]
    survival = [1.0]
    n_at_risk = n_patients
    current_s = 1.0

    for t, is_event in zip(observed, event_occurred):
        if n_at_risk <= 0:
            break
        if is_event and t <= x_max:
            current_s *= (1 - 1 / n_at_risk)
            times.append(float(round(t, 4)))
            survival.append(float(round(current_s, 6)))
        n_at_risk -= 1

    return np.array(times), np.array(survival)


# ── Plot rendering ───────────────────────────────────────────────────────

ARM_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
ARM_STYLES = ["-", "--", "-.", ":"]
ARM_LABELS_POOL = [
    ("Treatment", "Control"),
    ("Drug A", "Placebo"),
    ("Experimental", "Standard"),
    ("Arm A", "Arm B"),
    ("Combination", "Monotherapy"),
]


def render_km_plot(arms_data, x_max, labels, out_path):
    """Render a clinical-style KM plot to PNG."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for i, (times, surv) in enumerate(arms_data):
        color = ARM_COLORS[i % len(ARM_COLORS)]
        style = ARM_STYLES[i % len(ARM_STYLES)]
        ax.step(times, surv, where="post", color=color, linestyle=style,
                linewidth=1.8, label=labels[i])

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time (months)", fontsize=11)
    ax.set_ylabel("Survival probability", fontsize=11)
    ax.legend(loc="lower left", fontsize=10, frameon=True)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Nice tick spacing
    tick_step = {12: 2, 18: 3, 24: 4, 36: 6, 48: 8}.get(x_max, max(1, x_max // 6))
    ax.set_xticks(np.arange(0, x_max + 1, tick_step))
    ax.set_yticks(np.arange(0, 1.1, 0.2))

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


# ── Ground truth JSON ────────────────────────────────────────────────────

def build_truth_json(arms_data, labels, x_max):
    """Build ground truth dict compatible with metrics.py."""
    arms = []
    for i, (times, surv) in enumerate(arms_data):
        coords = [{"t": float(t), "s": float(s)} for t, s in zip(times, surv)]
        arms.append({"label": labels[i], "coordinates": coords})
    return {
        "arms": arms,
        "axis": {"x_min": 0, "x_max": x_max},
    }


# ── Main generator ──────────────────────────────────────────────────────

def generate_dataset(n_plots=5, seed=42, out_dir="synthetic"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    x_max_choices = [12, 18, 24, 36, 48]

    for idx in range(1, n_plots + 1):
        tag = f"synthetic_{idx:03d}"
        x_max = rng.choice(x_max_choices)
        n_patients = int(rng.integers(50, 201))
        labels = list(ARM_LABELS_POOL[rng.integers(0, len(ARM_LABELS_POOL))])

        arms_data = []
        for arm_i in range(2):
            shape = rng.uniform(0.7, 2.0)
            scale = rng.uniform(x_max * 0.3, x_max * 1.2)
            censor_rate = rng.uniform(0.05, 0.35)
            n_arm = int(rng.integers(max(30, n_patients - 40), n_patients + 40))
            times, surv = simulate_km(n_arm, x_max, shape, scale, censor_rate, rng)
            arms_data.append((times, surv))

        # Render plot
        render_km_plot(arms_data, int(x_max), labels, out / f"{tag}.png")

        # Write ground truth
        truth = build_truth_json(arms_data, labels, int(x_max))
        (out / f"{tag}_truth.json").write_text(json.dumps(truth, indent=2))

        n_steps = [len(t) for t, _ in arms_data]
        print(f"  {tag}: x_max={x_max}, arms={labels}, steps={n_steps}")

    print(f"\nGenerated {n_plots} plots in {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic KM plots with ground truth")
    parser.add_argument("--n", type=int, default=5, help="Number of plots to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    out_dir = args.out or str(Path(__file__).parent / "synthetic")
    generate_dataset(n_plots=args.n, seed=args.seed, out_dir=out_dir)
