"""
Generate 10 adversarial/stress-test synthetic KM plots for extraction pipeline testing.

Each produces a .png + _truth.json in synthetic/ matching the standard format.
Post-processing degrades the images; ground truth reflects the actual curve data.
"""

import json
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image, ImageFilter

OUT_DIR = Path(__file__).parent / "synthetic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(123)

X_MAX = 24
N_PATIENTS = 100


# ── Shared helpers (same as generate_edge_cases.py) ─────────────────────

def save_truth(tag, arms_list, axis_dict):
    truth = {"arms": arms_list, "axis": axis_dict}
    (OUT_DIR / f"{tag}_truth.json").write_text(json.dumps(truth, indent=2))
    print(f"  wrote {tag}_truth.json")


def simulate_km(n_patients, x_max, weibull_shape, weibull_scale, censor_rate, rng_):
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

    for i_obs in range(len(observed)):
        t = observed[i_obs]
        is_event = event_occurred[i_obs]
        if n_at_risk <= 0:
            break
        if is_event and t <= x_max:
            current_s *= (1 - 1 / n_at_risk)
            times.append(float(round(t, 4)))
            survival.append(float(round(current_s, 6)))
        n_at_risk -= 1

    return np.array(times), np.array(survival)


def arm_to_dict(label, times, surv, color=None):
    coords = [{"t": float(t), "s": float(s)} for t, s in zip(times, surv)]
    d = {"label": label, "coordinates": coords}
    if color:
        d["color"] = color
    return d


def style_ax(ax, x_max=X_MAX, y_max=1.05, y_min=0):
    ax.set_xlim(0, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Time (months)", fontsize=11)
    ax.set_ylabel("Survival probability", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_xticks(np.arange(0, x_max + 1, 4))
    ax.set_yticks(np.arange(0, 1.1, 0.2))


def make_standard_two_arm(rng_seed_offset=0):
    """Generate a standard 2-arm KM plot data. Returns (arms_list, fig, ax)."""
    local_rng = np.random.default_rng(123 + rng_seed_offset)
    colors = ["#1f77b4", "#d62728"]
    labels = ["Treatment", "Control"]
    params = [
        (1.3, X_MAX * 0.9),   # Treatment: better survival
        (1.5, X_MAX * 0.6),   # Control: worse survival
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    arms_list = []

    for i, (label, col) in enumerate(zip(labels, colors)):
        shape, scale = params[i]
        t, s = simulate_km(N_PATIENTS, X_MAX, shape, scale, 0.15, local_rng)
        arms_list.append(arm_to_dict(label, t, s, col))
        ax.step(t, s, where="post", color=col, linewidth=1.8, label=label)

    style_ax(ax)
    ax.legend(loc="lower left", fontsize=10, frameon=True)
    fig.tight_layout()
    return arms_list, fig, ax


def fig_to_pil(fig, dpi=150):
    """Convert matplotlib figure to PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    return Image.open(buf).copy()


AXIS_DICT = {"x_min": 0, "x_max": X_MAX, "y_min": 0, "y_max": 1.0}


# ── 1. stress_blurry — Gaussian blur radius=3 ──────────────────────────

def gen_blurry():
    tag = "stress_blurry"
    arms_list, fig, ax = make_standard_two_arm(0)
    img = fig_to_pil(fig)
    plt.close(fig)

    img = img.filter(ImageFilter.GaussianBlur(radius=3))
    img.save(str(OUT_DIR / f"{tag}.png"))

    save_truth(tag, arms_list, AXIS_DICT)
    print(f"  generated {tag}.png")


# ── 2. stress_jpeg_artifact — JPEG quality=15 then re-save as PNG ───────

def gen_jpeg_artifact():
    tag = "stress_jpeg_artifact"
    arms_list, fig, ax = make_standard_two_arm(0)
    img = fig_to_pil(fig)
    plt.close(fig)

    # Save as JPEG at very low quality, then reload
    jpeg_buf = io.BytesIO()
    img.convert("RGB").save(jpeg_buf, format="JPEG", quality=15)
    jpeg_buf.seek(0)
    img_degraded = Image.open(jpeg_buf).copy()
    img_degraded.save(str(OUT_DIR / f"{tag}.png"), format="PNG")

    save_truth(tag, arms_list, AXIS_DICT)
    print(f"  generated {tag}.png")


# ── 3. stress_bw — Black and white, solid vs dashed ────────────────────

def gen_bw():
    tag = "stress_bw"
    local_rng = np.random.default_rng(123)
    labels = ["Treatment", "Control"]
    styles = ["-", "--"]
    params = [
        (1.3, X_MAX * 0.9),
        (1.5, X_MAX * 0.6),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    arms_list = []

    for i, (label, ls) in enumerate(zip(labels, styles)):
        shape, scale = params[i]
        t, s = simulate_km(N_PATIENTS, X_MAX, shape, scale, 0.15, local_rng)
        arms_list.append(arm_to_dict(label, t, s, "black"))
        ax.step(t, s, where="post", color="black", linestyle=ls,
                linewidth=1.8, label=label)

    style_ax(ax)
    ax.legend(loc="lower left", fontsize=10, frameon=True)
    fig.tight_layout()
    fig.savefig(str(OUT_DIR / f"{tag}.png"), dpi=150)
    plt.close(fig)

    save_truth(tag, arms_list, AXIS_DICT)
    print(f"  generated {tag}.png")


# ── 4. stress_stretched_wide — 1600x400 (4:1) ──────────────────────────

def gen_stretched_wide():
    tag = "stress_stretched_wide"
    arms_list, fig, ax = make_standard_two_arm(0)
    img = fig_to_pil(fig)
    plt.close(fig)

    img = img.resize((1600, 400), Image.LANCZOS)
    img.save(str(OUT_DIR / f"{tag}.png"))

    save_truth(tag, arms_list, AXIS_DICT)
    print(f"  generated {tag}.png")


# ── 5. stress_stretched_tall — 400x1200 (1:3) ──────────────────────────

def gen_stretched_tall():
    tag = "stress_stretched_tall"
    arms_list, fig, ax = make_standard_two_arm(0)
    img = fig_to_pil(fig)
    plt.close(fig)

    img = img.resize((400, 1200), Image.LANCZOS)
    img.save(str(OUT_DIR / f"{tag}.png"))

    save_truth(tag, arms_list, AXIS_DICT)
    print(f"  generated {tag}.png")


# ── 6. stress_tiny — 200x150 pixels ────────────────────────────────────

def gen_tiny():
    tag = "stress_tiny"
    arms_list, fig, ax = make_standard_two_arm(0)
    img = fig_to_pil(fig)
    plt.close(fig)

    img = img.resize((200, 150), Image.LANCZOS)
    img.save(str(OUT_DIR / f"{tag}.png"))

    save_truth(tag, arms_list, AXIS_DICT)
    print(f"  generated {tag}.png")


# ── 7. stress_legend_overlap — Legend centered ON the curves ────────────

def gen_legend_overlap():
    tag = "stress_legend_overlap"
    local_rng = np.random.default_rng(123)
    colors = ["#1f77b4", "#d62728"]
    labels = ["Treatment", "Control"]
    params = [
        (1.3, X_MAX * 0.9),
        (1.5, X_MAX * 0.6),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    arms_list = []

    for i, (label, col) in enumerate(zip(labels, colors)):
        shape, scale = params[i]
        t, s = simulate_km(N_PATIENTS, X_MAX, shape, scale, 0.15, local_rng)
        arms_list.append(arm_to_dict(label, t, s, col))
        ax.step(t, s, where="post", color=col, linewidth=1.8, label=label)

    style_ax(ax)
    # Place legend dead center of the plot area, with a solid white background
    ax.legend(loc="center", fontsize=12, frameon=True, fancybox=True,
              shadow=True, facecolor="white", edgecolor="black",
              framealpha=1.0)
    fig.tight_layout()
    fig.savefig(str(OUT_DIR / f"{tag}.png"), dpi=150)
    plt.close(fig)

    save_truth(tag, arms_list, AXIS_DICT)
    print(f"  generated {tag}.png")


# ── 8. stress_gridlines — Heavy major + minor gridlines ─────────────────

def gen_gridlines():
    tag = "stress_gridlines"
    local_rng = np.random.default_rng(123)
    colors = ["#1f77b4", "#d62728"]
    labels = ["Treatment", "Control"]
    params = [
        (1.3, X_MAX * 0.9),
        (1.5, X_MAX * 0.6),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    arms_list = []

    for i, (label, col) in enumerate(zip(labels, colors)):
        shape, scale = params[i]
        t, s = simulate_km(N_PATIENTS, X_MAX, shape, scale, 0.15, local_rng)
        arms_list.append(arm_to_dict(label, t, s, col))
        ax.step(t, s, where="post", color=col, linewidth=1.8, label=label)

    style_ax(ax)
    # Heavy gridlines — major and minor
    ax.grid(True, which="major", color="gray", linewidth=0.8, alpha=0.7)
    ax.minorticks_on()
    ax.grid(True, which="minor", color="gray", linewidth=0.4, alpha=0.5)
    ax.legend(loc="lower left", fontsize=10, frameon=True)
    fig.tight_layout()
    fig.savefig(str(OUT_DIR / f"{tag}.png"), dpi=150)
    plt.close(fig)

    save_truth(tag, arms_list, AXIS_DICT)
    print(f"  generated {tag}.png")


# ── 9. stress_three_similar — Three nearly indistinguishable blue curves ─

def gen_three_similar():
    tag = "stress_three_similar"
    local_rng = np.random.default_rng(123)
    # Very similar blues in normalized RGB
    colors_rgb = [
        (0.2, 0.4, 0.7),
        (0.25, 0.45, 0.75),
        (0.15, 0.35, 0.65),
    ]
    labels = ["Arm A", "Arm B", "Arm C"]
    params = [
        (1.3, X_MAX * 0.9),
        (1.5, X_MAX * 0.7),
        (1.1, X_MAX * 0.5),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    arms_list = []

    for i, (label, col_rgb) in enumerate(zip(labels, colors_rgb)):
        shape, scale = params[i]
        t, s = simulate_km(N_PATIENTS, X_MAX, shape, scale, 0.15, local_rng)
        # Store color as hex for truth JSON
        col_hex = "#{:02x}{:02x}{:02x}".format(
            int(col_rgb[0] * 255), int(col_rgb[1] * 255), int(col_rgb[2] * 255))
        arms_list.append(arm_to_dict(label, t, s, col_hex))
        ax.step(t, s, where="post", color=col_rgb, linewidth=1.8, label=label)

    style_ax(ax)
    ax.legend(loc="lower left", fontsize=10, frameon=True)
    fig.tight_layout()
    fig.savefig(str(OUT_DIR / f"{tag}.png"), dpi=150)
    plt.close(fig)

    save_truth(tag, arms_list, AXIS_DICT)
    print(f"  generated {tag}.png")


# ── 10. stress_annotation_heavy — Text annotations, arrows, shading ─────

def gen_annotation_heavy():
    tag = "stress_annotation_heavy"
    local_rng = np.random.default_rng(123)
    colors = ["#1f77b4", "#d62728"]
    labels = ["Treatment", "Control"]
    params = [
        (1.3, X_MAX * 0.9),
        (1.5, X_MAX * 0.6),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    arms_list = []
    all_times = []
    all_surv = []

    for i, (label, col) in enumerate(zip(labels, colors)):
        shape, scale = params[i]
        t, s = simulate_km(N_PATIENTS, X_MAX, shape, scale, 0.15, local_rng)
        arms_list.append(arm_to_dict(label, t, s, col))
        ax.step(t, s, where="post", color=col, linewidth=1.8, label=label)
        all_times.append(t)
        all_surv.append(s)

    style_ax(ax)
    ax.legend(loc="lower left", fontsize=10, frameon=True)

    # Scattered text annotations
    ax.text(14, 0.85, "p = 0.03", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    ax.text(6, 0.35, "HR = 0.72\n(95% CI: 0.55-0.94)", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.6))
    ax.text(18, 0.15, "Log-rank\np < 0.001", fontsize=9, fontstyle="italic",
            ha="center")

    # Arrows pointing to specific curve points
    # Find survival at t~12 for treatment arm
    idx_t12 = np.searchsorted(all_times[0], 12)
    s_at_12 = all_surv[0][min(idx_t12, len(all_surv[0]) - 1)]
    ax.annotate("Median OS", xy=(12, s_at_12), xytext=(15, s_at_12 + 0.15),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray"))

    # Arrow to control arm
    idx_t8 = np.searchsorted(all_times[1], 8)
    s_ctrl_8 = all_surv[1][min(idx_t8, len(all_surv[1]) - 1)]
    ax.annotate("Early separation", xy=(8, s_ctrl_8), xytext=(3, 0.25),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray"))

    # Shaded region highlighting a time interval
    ax.axvspan(6, 14, alpha=0.08, color="green", label="_nolegend_")
    ax.axvspan(16, 22, alpha=0.06, color="orange", label="_nolegend_")

    fig.tight_layout()
    fig.savefig(str(OUT_DIR / f"{tag}.png"), dpi=150)
    plt.close(fig)

    save_truth(tag, arms_list, AXIS_DICT)
    print(f"  generated {tag}.png")


# ── Run all ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating stress-test synthetic KM plots...\n")
    gen_blurry()
    gen_jpeg_artifact()
    gen_bw()
    gen_stretched_wide()
    gen_stretched_tall()
    gen_tiny()
    gen_legend_overlap()
    gen_gridlines()
    gen_three_similar()
    gen_annotation_heavy()
    print(f"\nDone. All stress tests written to {OUT_DIR.resolve()}")
