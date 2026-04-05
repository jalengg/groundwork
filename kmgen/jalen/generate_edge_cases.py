"""
Generate 7 edge-case synthetic KM plots for stress-testing extraction pipelines.

Each produces a .png + _truth.json in the same format as generate_synthetic.py.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUT_DIR = Path(__file__).parent / "synthetic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(2026)


def save_truth(tag, arms_list, axis_dict):
    """Write ground truth JSON in the standard format."""
    truth = {"arms": arms_list, "axis": axis_dict}
    (OUT_DIR / f"{tag}_truth.json").write_text(json.dumps(truth, indent=2))
    print(f"  wrote {tag}_truth.json")


def simulate_km(n_patients, x_max, weibull_shape, weibull_scale, censor_rate, rng_):
    """Simulate KM step function from Weibull event times."""
    event_times = weibull_scale * rng_.weibull(weibull_shape, size=n_patients)
    censor_times = rng_.uniform(0, x_max * 1.2, size=n_patients)
    is_censored = rng_.random(n_patients) < censor_rate
    observed = np.where(is_censored, np.minimum(event_times, censor_times), event_times)
    event_occurred = np.where(is_censored, event_times <= censor_times, True)
    observed = np.clip(observed, 0, x_max)
    order = np.argsort(observed)
    observed = observed[order]
    event_occurred = event_occurred[order]

    times = [0.0]
    survival = [1.0]
    n_at_risk = n_patients
    current_s = 1.0
    # Track censoring times for tick marks
    censor_ticks = []

    for i_obs in range(len(observed)):
        t = observed[i_obs]
        is_event = event_occurred[i_obs]
        if n_at_risk <= 0:
            break
        if is_event and t <= x_max:
            current_s *= (1 - 1 / n_at_risk)
            times.append(float(round(t, 4)))
            survival.append(float(round(current_s, 6)))
        else:
            # censored observation — record for tick marks
            censor_ticks.append((float(t), float(current_s)))
        n_at_risk -= 1

    return np.array(times), np.array(survival), censor_ticks


def arm_to_dict(label, times, surv, color=None):
    coords = [{"t": float(t), "s": float(s)} for t, s in zip(times, surv)]
    d = {"label": label, "coordinates": coords}
    if color:
        d["color"] = color
    return d


# ── Clinical publication styling helpers ─────────────────────────────────

def style_ax(ax, x_max, y_max=1.05, xlabel="Time (months)",
             ylabel="Survival probability", y_min=0):
    ax.set_xlim(0, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def nice_xticks(ax, x_max):
    tick_map = {12: 2, 14: 2, 18: 3, 24: 4, 36: 6, 48: 8, 60: 12, 72: 12,
                84: 12, 96: 12, 120: 12, 144: 24}
    step = tick_map.get(x_max, max(1, x_max // 6))
    ax.set_xticks(np.arange(0, x_max + 1, step))


# ─────────────────────────────────────────────────────────────────────────
# 1. edge_high_survival — near-flat, high survival, censoring ticks
# ─────────────────────────────────────────────────────────────────────────

def gen_high_survival():
    tag = "edge_high_survival"
    x_max = 144
    n = 200
    colors = ["#1f77b4", "#d62728"]
    labels = ["Treatment", "Control"]

    fig, ax = plt.subplots(figsize=(7, 5))
    arms_list = []

    for i, (label, col) in enumerate(zip(labels, colors)):
        # Very high scale → very few events
        shape = 1.5
        scale = x_max * (4.0 + i * 0.5)  # huge scale keeps survival high
        t, s, cticks = simulate_km(n, x_max, shape, scale, 0.3, rng)
        arms_list.append(arm_to_dict(label, t, s, col))

        ax.step(t, s, where="post", color=col, linewidth=1.8, label=label)

        # Draw censoring tick marks
        for ct, cs in cticks:
            if ct <= x_max:
                ax.plot(ct, cs, "|", color=col, markersize=6, markeredgewidth=1.0)

    style_ax(ax, x_max, y_max=1.02, y_min=0.75)
    ax.set_yticks(np.arange(0.75, 1.01, 0.05))
    nice_xticks(ax, x_max)
    ax.legend(loc="lower left", fontsize=10, frameon=True)
    fig.tight_layout()
    fig.savefig(str(OUT_DIR / f"{tag}.png"), dpi=150)
    plt.close(fig)
    save_truth(tag, arms_list, {"x_min": 0, "x_max": x_max, "y_min": 0.75, "y_max": 1.0})
    print(f"  generated {tag}.png")


# ─────────────────────────────────────────────────────────────────────────
# 2. edge_cumulative_incidence — curves go UP from 0, 3 arms, risk table
# ─────────────────────────────────────────────────────────────────────────

def gen_cumulative_incidence():
    tag = "edge_cumulative_incidence"
    x_max = 60
    n = 150
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    labels = ["Arm A", "Arm B", "Arm C"]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    arms_list = []
    risk_table_data = []

    for i, (label, col) in enumerate(zip(labels, colors)):
        shape = 1.2 + i * 0.3
        scale = x_max * (1.8 - i * 0.25)
        t, s, _ = simulate_km(n, x_max, shape, scale, 0.2, rng)
        # Convert survival → cumulative incidence (1 - S)
        ci = 1.0 - s
        arms_list.append({
            "label": label,
            "color": col,
            "coordinates": [{"t": float(tt), "s": float(cc)} for tt, cc in zip(t, ci)]
        })
        ax.step(t, ci, where="post", color=col, linewidth=1.8, label=label)

        # Build risk table row
        risk_counts = []
        tick_step = 12
        for tick_t in range(0, x_max + 1, tick_step):
            at_risk = n - np.searchsorted(t, tick_t, side="right") + 1
            risk_counts.append(max(0, at_risk))
        risk_table_data.append((label, col, risk_counts))

    style_ax(ax, x_max, y_max=0.55, y_min=-0.02,
             ylabel="Cumulative incidence")
    ax.set_yticks(np.arange(0, 0.51, 0.1))
    nice_xticks(ax, x_max)
    ax.legend(loc="upper left", fontsize=10, frameon=True)

    # Patients-at-risk table below plot
    tick_positions = list(range(0, x_max + 1, 12))
    table_y_start = -0.22
    for row_i, (label, col, counts) in enumerate(risk_table_data):
        y = table_y_start - row_i * 0.06
        ax.text(-0.02, y, label, transform=ax.transAxes, fontsize=8,
                color=col, ha="right", va="center", fontweight="bold")
        for j, tick_t in enumerate(tick_positions):
            x_frac = tick_t / x_max
            ax.text(x_frac, y, str(counts[j]), transform=ax.transAxes,
                    fontsize=8, ha="center", va="center")
    ax.text(-0.02, table_y_start + 0.06, "No. at risk", transform=ax.transAxes,
            fontsize=8, ha="right", va="center", fontweight="bold")

    fig.subplots_adjust(bottom=0.28)
    fig.savefig(str(OUT_DIR / f"{tag}.png"), dpi=150)
    plt.close(fig)
    save_truth(tag, arms_list, {"x_min": 0, "x_max": x_max, "y_min": 0.0, "y_max": 0.5})
    print(f"  generated {tag}.png")


# ─────────────────────────────────────────────────────────────────────────
# 3. edge_four_arms — 4 overlapping, crossing arms, distinct line styles
# ─────────────────────────────────────────────────────────────────────────

def gen_four_arms():
    tag = "edge_four_arms"
    x_max = 24
    n = 100
    colors = ["blue", "red", "green", "purple"]
    styles = ["-", "--", ":", "-."]
    labels = ["Arm A", "Arm B", "Arm C", "Arm D"]

    fig, ax = plt.subplots(figsize=(7, 5))
    arms_list = []

    # Tuned so curves cross each other
    params = [
        (1.0, x_max * 0.6),   # concave early drop
        (2.0, x_max * 0.8),   # steeper late
        (0.8, x_max * 0.5),   # early drop then flattens
        (1.5, x_max * 1.0),   # moderate
    ]

    for i in range(4):
        shape, scale = params[i]
        t, s, _ = simulate_km(n, x_max, shape, scale, 0.15, rng)
        arms_list.append(arm_to_dict(labels[i], t, s, colors[i]))
        ax.step(t, s, where="post", color=colors[i], linestyle=styles[i],
                linewidth=1.8, label=labels[i])

    style_ax(ax, x_max)
    nice_xticks(ax, x_max)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.legend(loc="lower left", fontsize=10, frameon=True)
    fig.tight_layout()
    fig.savefig(str(OUT_DIR / f"{tag}.png"), dpi=150)
    plt.close(fig)
    save_truth(tag, arms_list, {"x_min": 0, "x_max": x_max, "y_min": 0, "y_max": 1.0})
    print(f"  generated {tag}.png")


# ─────────────────────────────────────────────────────────────────────────
# 4. edge_ci_shading — two arms with overlapping CI ribbons
# ─────────────────────────────────────────────────────────────────────────

def gen_ci_shading():
    tag = "edge_ci_shading"
    x_max = 36
    n = 120
    colors = ["#1f77b4", "#d62728"]
    labels = ["Treatment", "Control"]

    fig, ax = plt.subplots(figsize=(7, 5))
    arms_list = []

    for i, (label, col) in enumerate(zip(labels, colors)):
        shape = 1.3 + i * 0.3
        scale = x_max * (0.8 + i * 0.2)
        t, s, _ = simulate_km(n, x_max, shape, scale, 0.2, rng)
        arms_list.append(arm_to_dict(label, t, s, col))

        ax.step(t, s, where="post", color=col, linewidth=1.8, label=label)

        # Greenwood CI: SE = S * sqrt(sum(d_i / (n_i*(n_i-d_i))))
        # Approximate: wider CI when fewer at risk
        n_at_risk_approx = np.maximum(1, n - np.arange(len(s)))
        se = s * np.sqrt(np.cumsum(1.0 / n_at_risk_approx))
        se = np.clip(se, 0, 0.15)
        upper = np.clip(s + 1.96 * se, 0, 1)
        lower = np.clip(s - 1.96 * se, 0, 1)

        # Step-style CI ribbon
        t_step = np.repeat(t, 2)[1:]
        upper_step = np.repeat(upper, 2)[:-1]
        lower_step = np.repeat(lower, 2)[:-1]
        ax.fill_between(t_step, lower_step, upper_step, alpha=0.18,
                         color=col, step="post", linewidth=0)

    style_ax(ax, x_max)
    nice_xticks(ax, x_max)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.legend(loc="lower left", fontsize=10, frameon=True)
    fig.tight_layout()
    fig.savefig(str(OUT_DIR / f"{tag}.png"), dpi=150)
    plt.close(fig)
    save_truth(tag, arms_list, {"x_min": 0, "x_max": x_max, "y_min": 0, "y_max": 1.0})
    print(f"  generated {tag}.png")


# ─────────────────────────────────────────────────────────────────────────
# 5. edge_near_flat — one arm near-flat ~95%, other normal decline, x in years
# ─────────────────────────────────────────────────────────────────────────

def gen_near_flat():
    tag = "edge_near_flat"
    x_max = 14  # years
    n = 180
    colors = ["#1f77b4", "#d62728"]
    labels = ["Low risk", "High risk"]

    fig, ax = plt.subplots(figsize=(7, 5))
    arms_list = []

    # Arm 0: near-flat (very high Weibull scale)
    t0, s0, ct0 = simulate_km(n, x_max, 1.5, x_max * 8.0, 0.3, rng)
    arms_list.append(arm_to_dict(labels[0], t0, s0, colors[0]))
    ax.step(t0, s0, where="post", color=colors[0], linewidth=1.8, label=labels[0])
    for ct, cs in ct0:
        if ct <= x_max:
            ax.plot(ct, cs, "|", color=colors[0], markersize=5, markeredgewidth=0.8)

    # Arm 1: normal decline
    t1, s1, ct1 = simulate_km(n, x_max, 1.2, x_max * 0.7, 0.2, rng)
    arms_list.append(arm_to_dict(labels[1], t1, s1, colors[1]))
    ax.step(t1, s1, where="post", color=colors[1], linewidth=1.8, label=labels[1])
    for ct, cs in ct1:
        if ct <= x_max:
            ax.plot(ct, cs, "|", color=colors[1], markersize=5, markeredgewidth=0.8)

    style_ax(ax, x_max, xlabel="Time (years)")
    nice_xticks(ax, x_max)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.legend(loc="lower left", fontsize=10, frameon=True)
    fig.tight_layout()
    fig.savefig(str(OUT_DIR / f"{tag}.png"), dpi=150)
    plt.close(fig)
    save_truth(tag, arms_list, {"x_min": 0, "x_max": x_max, "y_min": 0, "y_max": 1.0})
    print(f"  generated {tag}.png")


# ─────────────────────────────────────────────────────────────────────────
# 6. edge_small_dense — low resolution (400x300), many events
# ─────────────────────────────────────────────────────────────────────────

def gen_small_dense():
    tag = "edge_small_dense"
    x_max = 36
    n = 180
    colors = ["#1f77b4", "#d62728"]
    labels = ["Treatment", "Control"]

    # 400x300 pixels at 100 dpi = 4x3 inches
    fig, ax = plt.subplots(figsize=(4, 3))
    arms_list = []

    for i, (label, col) in enumerate(zip(labels, colors)):
        shape = 1.0 + i * 0.4
        scale = x_max * (0.5 + i * 0.15)
        t, s, _ = simulate_km(n, x_max, shape, scale, 0.08, rng)  # low censor = many events
        arms_list.append(arm_to_dict(label, t, s, col))
        ax.step(t, s, where="post", color=col, linewidth=1.2, label=label)

    style_ax(ax, x_max)
    nice_xticks(ax, x_max)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.legend(loc="lower left", fontsize=7, frameon=True)
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    fig.tight_layout()
    fig.savefig(str(OUT_DIR / f"{tag}.png"), dpi=100)  # 400x300 px
    plt.close(fig)
    save_truth(tag, arms_list, {"x_min": 0, "x_max": x_max, "y_min": 0, "y_max": 1.0})
    print(f"  generated {tag}.png")


# ─────────────────────────────────────────────────────────────────────────
# 7. edge_multi_panel — two side-by-side subplots (Panel A, Panel B)
# ─────────────────────────────────────────────────────────────────────────

def gen_multi_panel():
    tag = "edge_multi_panel"
    x_max_a = 36
    x_max_b = 24
    n = 120
    colors = ["#1f77b4", "#d62728"]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))
    arms_list = []

    # Panel A — OS
    labels_a = ["Treatment (OS)", "Control (OS)"]
    ax_a.set_title("A", fontsize=13, fontweight="bold", loc="left")
    for i, (label, col) in enumerate(zip(labels_a, colors)):
        shape = 1.3 + i * 0.2
        scale = x_max_a * (0.8 + i * 0.15)
        t, s, _ = simulate_km(n, x_max_a, shape, scale, 0.2, rng)
        arms_list.append(arm_to_dict(label, t, s, col))
        ax_a.step(t, s, where="post", color=col, linewidth=1.8, label=label)

    style_ax(ax_a, x_max_a, ylabel="Overall survival")
    nice_xticks(ax_a, x_max_a)
    ax_a.set_yticks(np.arange(0, 1.1, 0.2))
    ax_a.legend(loc="lower left", fontsize=9, frameon=True)

    # Panel B — PFS
    labels_b = ["Treatment (PFS)", "Control (PFS)"]
    ax_b.set_title("B", fontsize=13, fontweight="bold", loc="left")
    for i, (label, col) in enumerate(zip(labels_b, colors)):
        shape = 1.5 + i * 0.3
        scale = x_max_b * (0.5 + i * 0.2)
        t, s, _ = simulate_km(n, x_max_b, shape, scale, 0.15, rng)
        arms_list.append(arm_to_dict(label, t, s, col))
        ax_b.step(t, s, where="post", color=col, linewidth=1.8, label=label)

    style_ax(ax_b, x_max_b, ylabel="Progression-free survival")
    nice_xticks(ax_b, x_max_b)
    ax_b.set_yticks(np.arange(0, 1.1, 0.2))
    ax_b.legend(loc="lower left", fontsize=9, frameon=True)

    fig.tight_layout()
    fig.savefig(str(OUT_DIR / f"{tag}.png"), dpi=150)
    plt.close(fig)

    # Axis info captures both panels
    save_truth(tag, arms_list, {
        "panel_a": {"x_min": 0, "x_max": x_max_a, "y_min": 0, "y_max": 1.0},
        "panel_b": {"x_min": 0, "x_max": x_max_b, "y_min": 0, "y_max": 1.0},
    })
    print(f"  generated {tag}.png")


# ── Run all ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating edge-case synthetic KM plots...\n")
    gen_high_survival()
    gen_cumulative_incidence()
    gen_four_arms()
    gen_ci_shading()
    gen_near_flat()
    gen_small_dense()
    gen_multi_panel()
    print("\nDone. All edge cases written to", OUT_DIR.resolve())
