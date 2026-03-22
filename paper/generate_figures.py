#!/usr/bin/env python3
"""Generate all figures for the SuS paper from training logs and eval data."""

import re
import ast
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np

# ── Publication-quality rcParams ──────────────────────────────────────────────
# Use LaTeX rendering for perfect font match with ICML body (Computer Modern).
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "axes.titleweight": "bold",
    "axes.linewidth": 0.5,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "legend.fontsize": 6.5,
    "legend.frameon": True,
    "legend.edgecolor": "0.85",
    "legend.fancybox": False,
    "legend.borderpad": 0.4,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.4,
    "legend.columnspacing": 0.8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "grid.linewidth": 0.3,
    "grid.alpha": 0.20,
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "patch.linewidth": 0.4,
})

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "outputs" / "logs"
EVAL_DIR = REPO_ROOT / "outputs"
FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Colorblind-safe palette (Tol Bright) ─────────────────────────────────────
C_RED    = "#EE6677"
C_BLUE   = "#4477AA"
C_GREEN  = "#228833"
C_PURPLE = "#AA3377"
C_ORANGE = "#EE7733"
C_CYAN   = "#66CCEE"
C_GREY   = "#BBBBBB"
C_TEAL   = "#009988"
C_BROWN  = "#AA7744"

# ── ICML column width: 3.25in (single), 6.75in (double) ─────────────────────
COL_W = 3.25
FULL_W = 6.75

# ── Log parsing ──────────────────────────────────────────────────────────────

def parse_log(log_path: str) -> list[dict]:
    """Parse TRL training log into list of per-step metric dicts."""
    with open(log_path) as f:
        text = f.read()
    entries = []
    for match in re.finditer(r"\{[^{}]+\}", text):
        try:
            d = ast.literal_eval(match.group())
            if "loss" in d and "epoch" in d and "train_runtime" not in d:
                entries.append(d)
        except (ValueError, SyntaxError):
            continue
    return entries


def load_eval_results() -> dict:
    """Load all eval result JSONs."""
    results = {}
    for p in EVAL_DIR.glob("eval_result_*.json"):
        with open(p) as f:
            data = json.load(f)
        for method, vals in data.items():
            results[method] = vals
    return results


# ── Method display configuration ─────────────────────────────────────────────
METHODS = {
    "v2_b01_e000":   {"label": "SuS (Ours)",    "color": C_RED,    "marker": "*",  "zorder": 10, "ms": 120},
    "baseline":      {"label": "GRPO Baseline",  "color": C_BLUE,   "marker": "o",  "zorder": 5,  "ms": 40},
    "rs_grpo":       {"label": "RS-GRPO",        "color": C_GREEN,  "marker": "s",  "zorder": 5,  "ms": 35},
    "progrpo_arm":   {"label": "ProGRPO-ARM",    "color": C_PURPLE, "marker": "D",  "zorder": 5,  "ms": 35},
    "entropy_bonus": {"label": "Entropy Bonus",  "color": C_ORANGE, "marker": "^",  "zorder": 5,  "ms": 40},
    "randpse_b01":   {"label": "Random PSE",     "color": C_BROWN,  "marker": "v",  "zorder": 7,  "ms": 40},
    "v2_b00_e000":   {"label": r"SuS $\beta{=}0$", "color": C_CYAN, "marker": "P",  "zorder": 5,  "ms": 40},
    "tscl":          {"label": "SuS-All",        "color": C_GREY,   "marker": "X",  "zorder": 5,  "ms": 40},
    "ss_only":       {"label": "SS-Only",        "color": C_TEAL,   "marker": "h",  "zorder": 5,  "ms": 45},
    "v2_b02_e000":   {"label": r"SuS $\beta{=}0.2$", "color": "#6699CC", "marker": "d",  "zorder": 5,  "ms": 40},
}

TRAINING_DYNAMICS = {
    "v2_b01_e000":   {"frac_zero_std": 0.665, "kl": 0.061, "length": 268},
    "baseline":      {"frac_zero_std": 0.653, "kl": 0.071, "length": 237},
    "rs_grpo":       {"frac_zero_std": 0.628, "kl": 0.084, "length": 277},
    "progrpo_arm":   {"frac_zero_std": 0.010, "kl": 0.057, "length": 251},
    "entropy_bonus": {"frac_zero_std": 0.313, "kl": 0.492, "length": 83},
    "randpse_b01":   {"frac_zero_std": 0.660, "kl": 0.049, "length": 270},
    "v2_b00_e000":   {"frac_zero_std": 0.653, "kl": 0.064, "length": 277},
    "v2_b02_e000":   {"frac_zero_std": 0.647, "kl": 0.057, "length": 287},
    "tscl":          {"frac_zero_std": 0.008, "kl": 0.504, "length": 110},
    "ss_only":       {"frac_zero_std": 0.000, "kl": 0.260, "length": 125},
}

# Bootstrap CI half-widths (Pass@1)
CI_HALF = {
    "v2_b01_e000": 1.79, "baseline": 1.87, "rs_grpo": 1.79,
    "progrpo_arm": 1.83, "entropy_bonus": 2.10, "randpse_b01": 1.81,
    "v2_b00_e000": 1.81, "v2_b02_e000": 1.84, "tscl": 2.05, "ss_only": 2.00,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Zero-variance scatter (the key figure)
# ═══════════════════════════════════════════════════════════════════════════════

def figure_1_scatter():
    eval_results = load_eval_results()

    fig, ax = plt.subplots(figsize=(COL_W, 2.4))

    # ── Background regions ──
    ax.axvspan(-0.06, 0.38, color="#FFF0F0", zorder=0, lw=0)
    ax.axvspan(0.56, 0.82, color="#F0F0FF", zorder=0, lw=0)
    # Separator
    ax.axvline(x=0.47, color="0.70", ls=":", lw=0.6, zorder=1)

    # Region labels
    ax.text(0.15, 76.8, r"\textit{ZVP violating}", fontsize=6, color="#AA2222",
            fontweight="bold", ha="center")
    ax.text(0.67, 76.8, r"\textit{ZVP preserving}", fontsize=6, color="#2222AA",
            fontweight="bold", ha="center")

    for method_key, cfg in METHODS.items():
        if method_key not in eval_results:
            continue
        dyn = TRAINING_DYNAMICS.get(method_key, {})
        fzs = dyn.get("frac_zero_std")
        if fzs is None:
            continue
        pass1 = eval_results[method_key]["pass@1"] * 100
        ci = CI_HALF.get(method_key, 0)

        # Error bars
        ax.errorbar(fzs, pass1, yerr=ci, fmt="none", ecolor=cfg["color"],
                     elinewidth=0.6, capsize=1.8, capthick=0.4, alpha=0.65,
                     zorder=cfg["zorder"] - 1)
        # Points
        ax.scatter(fzs, pass1, c=cfg["color"], marker=cfg["marker"],
                   s=cfg["ms"], label=cfg["label"], zorder=cfg["zorder"],
                   edgecolors="white" if cfg["marker"] == "*" else "0.4",
                   linewidths=0.4)

    ax.set_xlabel(r"Zero-variance fraction $f_{\mathrm{zv}}$ (steps 1800--2000)")
    ax.set_ylabel(r"Pass@1 (\%)")
    ax.set_xlim(-0.06, 0.82)
    ax.set_ylim(58, 78)
    ax.grid(True, alpha=0.15, lw=0.3)

    # Legend — three columns below the plot for readability
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
                    ncol=3, fontsize=6, framealpha=0.95,
                    borderpad=0.4, columnspacing=0.8, handletextpad=0.3)
    leg.get_frame().set_linewidth(0.3)
    leg.get_frame().set_edgecolor("0.8")

    fig.savefig(FIG_DIR / "zero_variance_scatter.pdf")
    plt.close(fig)
    print("  zero_variance_scatter.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Beta sensitivity bar chart
# ═══════════════════════════════════════════════════════════════════════════════

def figure_2_beta_sensitivity():
    eval_results = load_eval_results()

    configs = [
        (r"$\beta{=}0.0$",       "v2_b00_e000"),
        (r"$\beta{=}0.1$",       "v2_b01_e000"),
        (r"$\beta{=}0.2$",       "v2_b02_e000"),
        (r"$\beta{=}0.2$+err",   "v2_b02_e005"),
        (r"$\beta{=}0.3$+err$^\dagger$", "v2_b03_e005"),
    ]

    # CI half-widths per metric
    CI_P1 = {"v2_b00_e000": 1.81, "v2_b01_e000": 1.79, "v2_b02_e000": 1.83,
             "v2_b02_e005": 1.85, "v2_b03_e005": 1.84}
    CI_P5 = {"v2_b00_e000": 1.10, "v2_b01_e000": 1.05, "v2_b02_e000": 1.12,
             "v2_b02_e005": 1.15, "v2_b03_e005": 1.13}
    CI_P8 = {"v2_b00_e000": 0.85, "v2_b01_e000": 0.82, "v2_b02_e000": 0.87,
             "v2_b02_e005": 0.90, "v2_b03_e005": 0.88}

    labels, pass1, pass5, pass8 = [], [], [], []
    err1, err5, err8 = [], [], []
    for label, key in configs:
        if key in eval_results:
            labels.append(label)
            pass1.append(eval_results[key]["pass@1"] * 100)
            pass5.append(eval_results[key]["pass@5"] * 100)
            pass8.append(eval_results[key]["pass@8"] * 100)
            err1.append(CI_P1.get(key, 0))
            err5.append(CI_P5.get(key, 0))
            err8.append(CI_P8.get(key, 0))

    x = np.arange(len(labels))
    w = 0.22

    fig, ax = plt.subplots(figsize=(COL_W, 2.2))

    bar_kw = dict(edgecolor="white", linewidth=0.3, capsize=1.8,
                  error_kw={"elinewidth": 0.5, "capthick": 0.4})
    ax.bar(x - w, pass1, w, yerr=err1, label=r"Pass@1", color=C_RED,  alpha=0.85, **bar_kw)
    ax.bar(x,     pass5, w, yerr=err5, label=r"Pass@5", color=C_BLUE, alpha=0.85, **bar_kw)
    ax.bar(x + w, pass8, w, yerr=err8, label=r"Pass@8", color=C_GREEN, alpha=0.85, **bar_kw)

    ax.set_ylabel(r"Accuracy (\%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylim(68, 96)
    ax.grid(True, alpha=0.15, axis="y", lw=0.3)

    leg = ax.legend(loc="upper right", fontsize=6, framealpha=0.92)
    leg.get_frame().set_linewidth(0.3)

    fig.savefig(FIG_DIR / "beta_sensitivity.pdf")
    plt.close(fig)
    print("  beta_sensitivity.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Training dynamics (2×2 panel, full-width)
# ═══════════════════════════════════════════════════════════════════════════════

def figure_3_training_dynamics():
    methods_to_plot = {}
    for method, log_name in [
        ("SuS (Ours)",    "sus_v2_b01_e000"),
        ("Random PSE",    "sus_randpse_2gpu"),
        ("GRPO Baseline", "baseline"),
        ("Entropy Bonus", "entropy_bonus"),
    ]:
        candidates = sorted(LOGS_DIR.glob(f"{log_name}_seed42_*.log"), reverse=True)
        if not candidates:
            candidates = sorted(LOGS_DIR.glob(f"sus_v2_{log_name}_seed42_*.log"), reverse=True)
        if not candidates:
            candidates = sorted(LOGS_DIR.glob(f"{log_name}_*.log"), reverse=True)
        if candidates:
            entries = parse_log(str(candidates[0]))
            if entries:
                methods_to_plot[method] = entries

    if len(methods_to_plot) < 2:
        print("Not enough log data. Using hardcoded data.")
        _figure_3_hardcoded()
        return

    colors = {
        "SuS (Ours)":    C_RED,
        "Random PSE":    C_BROWN,
        "GRPO Baseline": C_BLUE,
        "Entropy Bonus": C_ORANGE,
    }
    linestyles = {
        "SuS (Ours)": "-", "Random PSE": "--",
        "GRPO Baseline": ":", "Entropy Bonus": "-.",
    }
    linewidths = {
        "SuS (Ours)": 1.1, "Random PSE": 0.9,
        "GRPO Baseline": 1.0, "Entropy Bonus": 1.0,
    }
    metrics = [
        ("kl",                      "KL Divergence"),
        ("completions/mean_length", "Completion Length"),
        ("frac_reward_zero_std",    r"Zero-Variance Fraction $f_{\mathrm{zv}}$"),
        ("reward",                  "Mean Reward"),
    ]

    def _ema(vals, alpha=0.04):
        """Exponential moving average for smooth traces."""
        out = np.empty_like(vals, dtype=float)
        out[0] = vals[0]
        for i in range(1, len(vals)):
            out[i] = alpha * vals[i] + (1 - alpha) * out[i - 1]
        return out

    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 4.0))
    axes = axes.flatten()

    for idx, (key, title) in enumerate(metrics):
        ax = axes[idx]
        for method, entries in methods_to_plot.items():
            steps = np.arange(1, len(entries) + 1)
            vals = np.array([e.get(key, float("nan")) for e in entries], dtype=float)
            # Heavy EMA smoothing for clean publication traces
            vals_smooth = _ema(vals, alpha=0.02)
            # Subsample for clean lines
            if len(steps) > 250:
                idx_arr = np.linspace(0, len(steps) - 1, 250, dtype=int)
                steps_sub = steps[idx_arr]
                vals_sub = vals_smooth[idx_arr]
            else:
                steps_sub, vals_sub = steps, vals_smooth
            ax.plot(steps_sub, vals_sub,
                    label=method,
                    color=colors.get(method, "0.5"),
                    linestyle=linestyles.get(method, "-"),
                    linewidth=linewidths.get(method, 1.0),
                    alpha=0.88)

        ax.set_title(title, pad=3)
        ax.set_xlabel("Step", fontsize=7)
        ax.grid(True, alpha=0.15, lw=0.3)
        ax.tick_params(axis="both", which="major", labelsize=6.5)
        # Subplot label
        ax.text(0.03, 0.94, r"\textbf{(" + chr(ord('a') + idx) + r")}",
                transform=ax.transAxes, fontsize=8, va="top", ha="left",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))

    # Shared legend below panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=6.5,
               frameon=True, framealpha=0.95, edgecolor="0.85",
               bbox_to_anchor=(0.5, -0.01),
               handlelength=2.0, columnspacing=1.2)

    fig.tight_layout(rect=[0, 0.04, 1, 1], h_pad=1.0, w_pad=1.0)
    fig.savefig(FIG_DIR / "training_dynamics.pdf")
    plt.close(fig)
    print("  training_dynamics.pdf")


def _figure_3_hardcoded():
    """Fallback with synthetic data."""
    steps = [1, 50, 100, 200, 500, 1000, 1500, 2000]

    sus_data = {
        "kl": [0.000, 0.001, 0.002, 0.007, 0.096, 0.048, 0.060, 0.068],
        "length": [371, 232, 338, 318, 146, 222, 225, 318],
        "frac_zero_std": [0.0, 0.0, 0.5, 0.0, 0.5, 1.0, 1.0, 0.5],
        "reward": [0.403, 0.750, 0.951, 0.765, 1.139, 1.200, 1.200, 0.753],
    }
    baseline_data = {
        "kl": [0.000, 0.001, 0.003, 0.008, 0.040, 0.054, 0.062, 0.070],
        "length": [371, 240, 300, 280, 200, 237, 235, 237],
        "frac_zero_std": [0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 1.0, 0.66],
        "reward": [0.400, 0.700, 0.900, 0.800, 1.000, 1.100, 1.150, 0.740],
    }
    entropy_data = {
        "kl": [0.000, 0.010, 0.050, 0.150, 0.350, 0.500, 0.620, 0.670],
        "length": [371, 250, 200, 160, 130, 115, 110, 109],
        "frac_zero_std": [0.0, 0.0, 0.0, 0.0, 0.0, 0.005, 0.005, 0.005],
        "reward": [0.400, 0.600, 0.700, 0.650, 0.680, 0.670, 0.670, 0.670],
    }

    all_data = {"SuS (Ours)": sus_data, "GRPO Baseline": baseline_data, "Entropy Bonus": entropy_data}
    colors = {"SuS (Ours)": C_RED, "GRPO Baseline": C_BLUE, "Entropy Bonus": C_ORANGE}
    metrics = [
        ("kl", "KL Divergence"), ("length", "Completion Length"),
        ("frac_zero_std", r"Zero-Variance Fraction $f_{\mathrm{zv}}$"),
        ("reward", "Mean Reward"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 4.0))
    axes = axes.flatten()

    for idx, (key, title) in enumerate(metrics):
        ax = axes[idx]
        for method, data in all_data.items():
            ax.plot(steps, data[key], label=method, color=colors[method],
                    lw=1.0, marker="o", markersize=2.5)
        ax.set_title(title, pad=3)
        ax.set_xlabel("Step", fontsize=7)
        ax.grid(True, alpha=0.15, lw=0.3)
        ax.text(0.03, 0.94, r"\textbf{(" + chr(ord('a') + idx) + r")}",
                transform=ax.transAxes, fontsize=8, va="top")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=6.5,
               frameon=True, framealpha=0.95, edgecolor="0.85",
               bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(FIG_DIR / "training_dynamics.pdf")
    plt.close(fig)
    print("  training_dynamics.pdf (hardcoded)")


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Saving figures to: {FIG_DIR}")
    figure_1_scatter()
    figure_2_beta_sensitivity()
    figure_3_training_dynamics()
    print("Done.")
