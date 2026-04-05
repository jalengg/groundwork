"""
Generate 10 combined/mixed stress-test KM plots that layer MULTIPLE anti-patterns.

Each produces a .png + _truth.json in synthetic/ matching the standard format.
These are much harder than single-degradation stress tests.
"""

import json
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

OUT_DIR = Path(__file__).parent / "synthetic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(456)


# ── Shared helpers ────────────────────────────────────────────────────────

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


def style_ax(ax, x_max, y_max=1.05, y_min=0, xlabel="Time (months)",
             ylabel="Survival probability"):
    ax.set_xlim(0, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def nice_xticks(ax, x_max):
    tick_map = {12: 2, 14: 2, 18: 3, 24: 4, 36: 6, 48: 8, 50: 10, 60: 12}
    step = tick_map.get(x_max, max(1, x_max // 6))
    ax.set_xticks(np.arange(0, x_max + 1, step))


def fig_to_pil(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    return Image.open(buf).copy()


# ── 1. stress_combo_tiny_blurry — 200x150 + blur radius=2 ────────────────

def gen_combo_tiny_blurry():
    tag = "stress_combo_tiny_blurry"
    x_max = 24
    local_rng = np.random.default_rng(456)
    colors = ["#1f77b4", "#d62728"]
    labels = ["Treatment", "Control"]
    params = [(1.3, x_max * 0.9), (1.5, x_max * 0.6)]

    fig, ax = plt.subplots(figsize=(7, 5))
    arms_list = []
    for i, (label, col) in enumerate(zip(labels, colors)):
        shape, scale = params[i]
        t, s = simulate_km(100, x_max, shape, scale, 0.15, local_rng)
        arms_list.append(arm_to_dict(label, t, s, col))
        ax.step(t, s, where="post", color=col, linewidth=1.8, label=label)

    style_ax(ax, x_max)
    ax.set_xticks(np.arange(0, x_max + 1, 4))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.legend(loc="lower left", fontsize=10, frameon=True)
    fig.tight_layout()

    img = fig_to_pil(fig)
    plt.close(fig)

    # Resize to tiny THEN blur
    img = img.resize((200, 150), Image.LANCZOS)
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    img.save(str(OUT_DIR / f"{tag}.png"))

    save_truth(tag, arms_list, {"x_min": 0, "x_max": x_max, "y_min": 0, "y_max": 1.0})
    print(f"  generated {tag}.png")


# ── 2. stress_combo_bw_overlap — B&W + heavily overlapping curves ─────────

def gen_combo_bw_overlap():
    tag = "stress_combo_bw_overlap"
    x_max = 36
    local_rng = np.random.default_rng(456)
    labels = ["Arm A", "Arm B"]
    styles = ["-", "--"]
    # Very similar parameters so curves overlap heavily (60%+ of time range)
    params = [(1.3, x_max * 0.75), (1.35, x_max * 0.78)]

    fig, ax = plt.subplots(figsize=(7, 5))
    arms_list = []
    for i, (label, ls) in enumerate(zip(labels, styles)):
        shape, scale = params[i]
        t, s = simulate_km(100, x_max, shape, scale, 0.15, local_rng)
        arms_list.append(arm_to_dict(label, t, s, "black"))
        ax.step(t, s, where="post", color="black", linestyle=ls,
                linewidth=1.8, label=label)

    style_ax(ax, x_max)
    nice_xticks(ax, x_max)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.legend(loc="lower left", fontsize=10, frameon=True)
    fig.tight_layout()
    fig.savefig(str(OUT_DIR / f"{tag}.png"), dpi=150)
    plt.close(fig)

    save_truth(tag, arms_list, {"x_min": 0, "x_max": x_max, "y_min": 0, "y_max": 1.0})
    print(f"  generated {tag}.png")


# ── 3. stress_combo_stretched_dense — 1400x300 + 200 patients/arm ─────────

def gen_combo_stretched_dense():
    tag = "stress_combo_stretched_dense"
    x_max = 48
    local_rng = np.random.default_rng(456)
    colors = ["#1f77b4", "#d62728"]
    labels = ["Treatment", "Control"]
    params = [(1.3, x_max * 0.7), (1.5, x_max * 0.5)]

    fig, ax = plt.subplots(figsize=(7, 5))
    arms_list = []
    for i, (label, col) in enumerate(zip(labels, colors)):
        shape, scale = params[i]
        # 200 patients per arm = very dense steps
        t, s = simulate_km(200, x_max, shape, scale, 0.08, local_rng)
        arms_list.append(arm_to_dict(label, t, s, col))
        ax.step(t, s, where="post", color=col, linewidth=1.8, label=label)

    style_ax(ax, x_max)
    nice_xticks(ax, x_max)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.legend(loc="lower left", fontsize=10, frameon=True)
    fig.tight_layout()

    img = fig_to_pil(fig)
    plt.close(fig)

    # Stretch to 1400x300 (very wide, very short)
    img = img.resize((1400, 300), Image.LANCZOS)
    img.save(str(OUT_DIR / f"{tag}.png"))

    save_truth(tag, arms_list, {"x_min": 0, "x_max": x_max, "y_min": 0, "y_max": 1.0})
    print(f"  generated {tag}.png")


# ── 4. stress_combo_tiny_bw — 200x150 + black/white ──────────────────────

def gen_combo_tiny_bw():
    tag = "stress_combo_tiny_bw"
    x_max = 24
    local_rng = np.random.default_rng(456)
    labels = ["Treatment", "Control"]
    styles = ["-", "--"]
    params = [(1.3, x_max * 0.9), (1.5, x_max * 0.6)]

    fig, ax = plt.subplots(figsize=(7, 5))
    arms_list = []
    for i, (label, ls) in enumerate(zip(labels, styles)):
        shape, scale = params[i]
        t, s = simulate_km(100, x_max, shape, scale, 0.15, local_rng)
        arms_list.append(arm_to_dict(label, t, s, "black"))
        ax.step(t, s, where="post", color="black", linestyle=ls,
                linewidth=1.8, label=label)

    style_ax(ax, x_max)
    ax.set_xticks(np.arange(0, x_max + 1, 4))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.legend(loc="lower left", fontsize=10, frameon=True)
    fig.tight_layout()

    img = fig_to_pil(fig)
    plt.close(fig)

    # Resize to tiny
    img = img.resize((200, 150), Image.LANCZOS)
    img.save(str(OUT_DIR / f"{tag}.png"))

    save_truth(tag, arms_list, {"x_min": 0, "x_max": x_max, "y_min": 0, "y_max": 1.0})
    print(f"  generated {tag}.png")


# ── 5. stress_combo_jpeg_blurry_dark — JPEG q=15 + blur r=2 + 50% brightness

def gen_combo_jpeg_blurry_dark():
    tag = "stress_combo_jpeg_blurry_dark"
    x_max = 24
    local_rng = np.random.default_rng(456)
    colors = ["#1f77b4", "#d62728"]
    labels = ["Treatment", "Control"]
    params = [(1.3, x_max * 0.9), (1.5, x_max * 0.6)]

    fig, ax = plt.subplots(figsize=(7, 5))
    arms_list = []
    for i, (label, col) in enumerate(zip(labels, colors)):
        shape, scale = params[i]
        t, s = simulate_km(100, x_max, shape, scale, 0.15, local_rng)
        arms_list.append(arm_to_dict(label, t, s, col))
        ax.step(t, s, where="post", color=col, linewidth=1.8, label=label)

    style_ax(ax, x_max)
    ax.set_xticks(np.arange(0, x_max + 1, 4))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.legend(loc="lower left", fontsize=10, frameon=True)
    fig.tight_layout()

    img = fig_to_pil(fig)
    plt.close(fig)

    # Triple degradation: blur + JPEG + darken
    # 1) Gaussian blur
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    # 2) JPEG compression at quality=15
    jpeg_buf = io.BytesIO()
    img.convert("RGB").save(jpeg_buf, format="JPEG", quality=15)
    jpeg_buf.seek(0)
    img = Image.open(jpeg_buf).copy()
    # 3) Reduce brightness to 50%
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(0.5)

    img.save(str(OUT_DIR / f"{tag}.png"), format="PNG")

    save_truth(tag, arms_list, {"x_min": 0, "x_max": x_max, "y_min": 0, "y_max": 1.0})
    print(f"  generated {tag}.png")


# ── 6. stress_combo_flat_overlap — two near-flat curves, heavily overlapping

def gen_combo_flat_overlap():
    tag = "stress_combo_flat_overlap"
    x_max = 60
    local_rng = np.random.default_rng(456)
    colors = ["#1f77b4", "#d62728"]
    labels = ["Arm A", "Arm B"]

    fig, ax = plt.subplots(figsize=(7, 5))
    arms_list = []

    # Both arms have very high Weibull scale = very few events, survival stays >0.85
    params = [
        (1.5, x_max * 6.0),   # near-flat, ~95%+ survival
        (1.5, x_max * 5.5),   # near-flat, ~90%+ survival
    ]

    for i, (label, col) in enumerate(zip(labels, colors)):
        shape, scale = params[i]
        t, s = simulate_km(150, x_max, shape, scale, 0.25, local_rng)
        arms_list.append(arm_to_dict(label, t, s, col))
        ax.step(t, s, where="post", color=col, linewidth=1.8, label=label)

    style_ax(ax, x_max, y_min=0.75, y_max=1.02)
    nice_xticks(ax, x_max)
    ax.set_yticks(np.arange(0.75, 1.01, 0.05))
    ax.legend(loc="lower left", fontsize=10, frameon=True)
    fig.tight_layout()
    fig.savefig(str(OUT_DIR / f"{tag}.png"), dpi=150)
    plt.close(fig)

    save_truth(tag, arms_list, {"x_min": 0, "x_max": x_max, "y_min": 0.75, "y_max": 1.0})
    print(f"  generated {tag}.png")


# ── 7. stress_combo_4arm_tiny — 4 arms at 250x200 pixels ─────────────────

def gen_combo_4arm_tiny():
    tag = "stress_combo_4arm_tiny"
    x_max = 36
    local_rng = np.random.default_rng(456)
    colors = ["blue", "red", "green", "purple"]
    labels = ["Arm A", "Arm B", "Arm C", "Arm D"]
    params = [
        (1.0, x_max * 0.6),
        (1.5, x_max * 0.8),
        (0.8, x_max * 0.5),
        (1.3, x_max * 1.0),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    arms_list = []
    for i in range(4):
        shape, scale = params[i]
        t, s = simulate_km(100, x_max, shape, scale, 0.15, local_rng)
        arms_list.append(arm_to_dict(labels[i], t, s, colors[i]))
        ax.step(t, s, where="post", color=colors[i], linewidth=1.8, label=labels[i])

    style_ax(ax, x_max)
    nice_xticks(ax, x_max)
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.legend(loc="lower left", fontsize=9, frameon=True)
    fig.tight_layout()

    img = fig_to_pil(fig)
    plt.close(fig)

    # Shrink to 250x200
    img = img.resize((250, 200), Image.LANCZOS)
    img.save(str(OUT_DIR / f"{tag}.png"))

    save_truth(tag, arms_list, {"x_min": 0, "x_max": x_max, "y_min": 0, "y_max": 1.0})
    print(f"  generated {tag}.png")


# ── 8. stress_combo_grid_annotation_bw — gridlines + annotations + B&W ───

def gen_combo_grid_annotation_bw():
    tag = "stress_combo_grid_annotation_bw"
    x_max = 24
    local_rng = np.random.default_rng(456)
    labels = ["Treatment", "Control"]
    styles = ["-", "--"]
    params = [(1.3, x_max * 0.9), (1.5, x_max * 0.6)]

    fig, ax = plt.subplots(figsize=(7, 5))
    arms_list = []
    all_times = []
    all_surv = []

    for i, (label, ls) in enumerate(zip(labels, styles)):
        shape, scale = params[i]
        t, s = simulate_km(100, x_max, shape, scale, 0.15, local_rng)
        arms_list.append(arm_to_dict(label, t, s, "black"))
        ax.step(t, s, where="post", color="black", linestyle=ls,
                linewidth=1.8, label=label)
        all_times.append(t)
        all_surv.append(s)

    style_ax(ax, x_max)
    ax.set_xticks(np.arange(0, x_max + 1, 4))
    ax.set_yticks(np.arange(0, 1.1, 0.2))

    # Heavy gridlines (major + minor)
    ax.grid(True, which="major", color="gray", linewidth=0.8, alpha=0.7)
    ax.minorticks_on()
    ax.grid(True, which="minor", color="gray", linewidth=0.4, alpha=0.5)

    # Annotation text scattered across plot
    ax.text(14, 0.85, "p = 0.03", fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    ax.text(6, 0.30, "HR = 0.72\n(95% CI: 0.55-0.94)", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.6))
    ax.text(18, 0.15, "Log-rank\np < 0.001", fontsize=9, fontstyle="italic",
            ha="center")

    # Arrow annotations
    idx_t12 = np.searchsorted(all_times[0], 12)
    s_at_12 = all_surv[0][min(idx_t12, len(all_surv[0]) - 1)]
    ax.annotate("Median OS", xy=(12, s_at_12), xytext=(15, s_at_12 + 0.15),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray"))

    ax.legend(loc="lower left", fontsize=10, frameon=True)
    fig.tight_layout()
    fig.savefig(str(OUT_DIR / f"{tag}.png"), dpi=150)
    plt.close(fig)

    save_truth(tag, arms_list, {"x_min": 0, "x_max": x_max, "y_min": 0, "y_max": 1.0})
    print(f"  generated {tag}.png")


# ── 9. stress_diep_like — Realistic dual-panel cumulative incidence ───────
# Mimics the breast reconstruction paper (image7): two side-by-side panels,
# each with 4 cumulative incidence curves going UP, small size, p-values,
# legend overlapping curves.

def gen_diep_like():
    tag = "stress_diep_like"
    x_max = 50  # months
    local_rng = np.random.default_rng(456)
    n = 80
    colors = ["green", "blue", "red", "purple"]
    labels_a = ["DIEP", "Implant", "Latissimus", "TRAM"]
    labels_b = ["Unilateral", "Bilateral", "Immediate", "Delayed"]

    # Target size: ~600x250
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8, 3.3))
    arms_list = []

    # Panel A: different event rates per arm
    ax_a.set_title("A", fontsize=10, fontweight="bold", loc="left")
    event_rates_a = [
        (1.2, x_max * 3.5),   # low incidence (~0.15 at 50mo)
        (1.0, x_max * 2.5),   # moderate (~0.25)
        (1.3, x_max * 2.0),   # higher (~0.35)
        (0.9, x_max * 1.8),   # highest (~0.40)
    ]

    for i, (label, col) in enumerate(zip(labels_a, colors)):
        shape, scale = event_rates_a[i]
        t, s = simulate_km(n, x_max, shape, scale, 0.20, local_rng)
        # Convert to cumulative incidence (1 - S)
        ci = 1.0 - s
        arms_list.append({
            "label": label,
            "color": col,
            "coordinates": [{"t": float(tt), "s": float(cc)} for tt, cc in zip(t, ci)],
            "panel": "A"
        })
        ax_a.step(t, ci, where="post", color=col, linewidth=1.5, label=label)

    ax_a.set_xlim(0, x_max)
    ax_a.set_ylim(-0.02, 0.50)
    ax_a.set_xlabel("Time (months)", fontsize=8)
    ax_a.set_ylabel("Cumulative incidence", fontsize=8)
    ax_a.set_yticks(np.arange(0, 0.51, 0.1))
    nice_xticks(ax_a, x_max)
    ax_a.tick_params(labelsize=7)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    # p-value text in panel
    ax_a.text(0.95, 0.95, "p = 0.042", transform=ax_a.transAxes,
              fontsize=7, ha="right", va="top", fontstyle="italic")
    # Legend that overlaps curves
    ax_a.legend(loc="upper left", fontsize=6, frameon=True, framealpha=0.8,
                handlelength=1.5)

    # Panel B: different grouping
    ax_b.set_title("B", fontsize=10, fontweight="bold", loc="left")
    event_rates_b = [
        (1.1, x_max * 3.0),
        (1.4, x_max * 2.8),
        (1.0, x_max * 2.2),
        (1.2, x_max * 1.5),
    ]

    for i, (label, col) in enumerate(zip(labels_b, colors)):
        shape, scale = event_rates_b[i]
        t, s = simulate_km(n, x_max, shape, scale, 0.20, local_rng)
        ci = 1.0 - s
        arms_list.append({
            "label": label,
            "color": col,
            "coordinates": [{"t": float(tt), "s": float(cc)} for tt, cc in zip(t, ci)],
            "panel": "B"
        })
        ax_b.step(t, ci, where="post", color=col, linewidth=1.5, label=label)

    ax_b.set_xlim(0, x_max)
    ax_b.set_ylim(-0.02, 0.50)
    ax_b.set_xlabel("Time (months)", fontsize=8)
    ax_b.set_ylabel("Cumulative incidence", fontsize=8)
    ax_b.set_yticks(np.arange(0, 0.51, 0.1))
    nice_xticks(ax_b, x_max)
    ax_b.tick_params(labelsize=7)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.text(0.95, 0.95, "p = 0.018", transform=ax_b.transAxes,
              fontsize=7, ha="right", va="top", fontstyle="italic")
    # No legend in panel B (like the real paper — legend only in panel A)

    fig.tight_layout()

    # Save at small size (~600x250)
    img = fig_to_pil(fig, dpi=100)
    plt.close(fig)
    img = img.resize((600, 250), Image.LANCZOS)
    img.save(str(OUT_DIR / f"{tag}.png"))

    save_truth(tag, arms_list, {
        "panel_a": {"x_min": 0, "x_max": x_max, "y_min": 0.0, "y_max": 0.5},
        "panel_b": {"x_min": 0, "x_max": x_max, "y_min": 0.0, "y_max": 0.5},
    })
    print(f"  generated {tag}.png")


# ── 10. stress_multi_panel_clean — clean dual-panel, good resolution ──────
# Isolates multi-panel logic from degradation issues.

def gen_multi_panel_clean():
    tag = "stress_multi_panel_clean"
    x_max_a = 36
    x_max_b = 24
    n = 120
    local_rng = np.random.default_rng(456)
    colors = ["#1f77b4", "#d62728"]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12, 5))
    arms_list = []

    # Panel A — OS
    labels_a = ["Treatment (OS)", "Control (OS)"]
    ax_a.set_title("A", fontsize=13, fontweight="bold", loc="left")
    for i, (label, col) in enumerate(zip(labels_a, colors)):
        shape = 1.3 + i * 0.2
        scale = x_max_a * (0.8 + i * 0.15)
        t, s = simulate_km(n, x_max_a, shape, scale, 0.2, local_rng)
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
        t, s = simulate_km(n, x_max_b, shape, scale, 0.15, local_rng)
        arms_list.append(arm_to_dict(label, t, s, col))
        ax_b.step(t, s, where="post", color=col, linewidth=1.8, label=label)

    style_ax(ax_b, x_max_b, ylabel="Progression-free survival")
    nice_xticks(ax_b, x_max_b)
    ax_b.set_yticks(np.arange(0, 1.1, 0.2))
    ax_b.legend(loc="lower left", fontsize=9, frameon=True)

    fig.tight_layout()

    # Save at good resolution (target ~1000x500)
    img = fig_to_pil(fig, dpi=100)
    plt.close(fig)
    img = img.resize((1000, 500), Image.LANCZOS)
    img.save(str(OUT_DIR / f"{tag}.png"))

    save_truth(tag, arms_list, {
        "panel_a": {"x_min": 0, "x_max": x_max_a, "y_min": 0, "y_max": 1.0},
        "panel_b": {"x_min": 0, "x_max": x_max_b, "y_min": 0, "y_max": 1.0},
    })
    print(f"  generated {tag}.png")


# ── Run all ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating combined/mixed stress-test KM plots...\n")
    gen_combo_tiny_blurry()
    gen_combo_bw_overlap()
    gen_combo_stretched_dense()
    gen_combo_tiny_bw()
    gen_combo_jpeg_blurry_dark()
    gen_combo_flat_overlap()
    gen_combo_4arm_tiny()
    gen_combo_grid_annotation_bw()
    gen_diep_like()
    gen_multi_panel_clean()
    print(f"\nDone. All combined stress tests written to {OUT_DIR.resolve()}")
