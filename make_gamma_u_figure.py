"""
Analyse gamma_U binary CI study results and produce thesis figure.
Reads from gamma_u_ci_results/*.json (assembled by run_gamma_u_aws.py collect).
Saves figure to ~/Desktop/thesis/Tomer-Weiss-Thesis-v2/gfx/gamma_u_binary_ci.pdf
and .png for reference.
"""

import json
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Config ──────────────────────────────────────────────────────────────────

TRUE_BETA_X = 2.0
B_BOOT      = 200
M_ENS       = 10
CI_LEVEL    = (2.5, 97.5)

GAMMA_U_VALUES = [0, 1, 3]
RESULT_DIR     = "gamma_u_ci_results"

GFX_DIR = os.path.expanduser(
    "~/Desktop/thesis/Tomer-Weiss-Thesis-v2/gfx")

# ── Load results ─────────────────────────────────────────────────────────────

def load_complete():
    """Return dict (gu -> list of per-rep dicts) for complete reps only."""
    data = {gu: [] for gu in GAMMA_U_VALUES}
    for path in sorted(glob.glob(os.path.join(RESULT_DIR, "*.json"))):
        with open(path) as f:
            d = json.load(f)
        if d.get("status") != "complete":
            continue
        gu = d["gamma_u"]
        if gu not in data:
            continue
        # Validate completeness
        if (d["logit_point"] is None
                or None in d["logit_bootstrap"]
                or None in d["nn_ensemble"]
                or None in d["nn_bootstrap"]):
            continue
        data[gu].append(d)
    return data


def compute_stats(data):
    """
    For each gamma_u, compute per-rep statistics then aggregate.
    Returns dict: gu -> {"logit": {...}, "nn": {...}}
    """
    stats = {}
    for gu, reps in data.items():
        logit_biases, logit_widths, logit_covers = [], [], []
        nn_biases,    nn_widths,    nn_covers    = [], [], []

        for d in reps:
            lp = d["logit_point"]
            lb = np.array(d["logit_bootstrap"])
            ne = np.array(d["nn_ensemble"])
            nb = np.array(d["nn_bootstrap"])

            # ── Logit: naive percentile bootstrap ──
            logit_biases.append(lp - TRUE_BETA_X)
            lo, hi = np.percentile(lb, CI_LEVEL)
            logit_widths.append(hi - lo)
            logit_covers.append(float(lo <= TRUE_BETA_X <= hi))

            # ── NN: ensemble-corrected bootstrap ──
            ens_mean = float(np.mean(ne))
            nn_biases.append(ens_mean - TRUE_BETA_X)
            b_centered = nb - np.mean(nb)
            lo_nn = ens_mean + np.percentile(b_centered, CI_LEVEL[0])
            hi_nn = ens_mean + np.percentile(b_centered, CI_LEVEL[1])
            nn_widths.append(hi_nn - lo_nn)
            nn_covers.append(float(lo_nn <= TRUE_BETA_X <= hi_nn))

        n = len(reps)
        se = lambda v: np.std(v) / np.sqrt(len(v)) if len(v) > 0 else 0.0

        stats[gu] = {
            "n": n,
            "logit": {
                "bias":       np.mean(logit_biases),
                "bias_se":    se(logit_biases),
                "width":      np.mean(logit_widths),
                "width_se":   se(logit_widths),
                "coverage":   np.mean(logit_covers),
                "coverage_se": se(logit_covers),
            },
            "nn": {
                "bias":       np.mean(nn_biases),
                "bias_se":    se(nn_biases),
                "width":      np.mean(nn_widths),
                "width_se":   se(nn_widths),
                "coverage":   np.mean(nn_covers),
                "coverage_se": se(nn_covers),
            },
        }
    return stats


# ── Plotting ─────────────────────────────────────────────────────────────────

COLORS = {
    "logit": "#2166ac",   # blue
    "nn":    "#d6604d",   # red-orange
}
LABELS = {
    "logit": "Logistic 2SRI",
    "nn":    "NN-2SRI",
}

def make_figure(stats, out_prefix):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    fig.subplots_adjust(wspace=0.38, left=0.07, right=0.97, top=0.88, bottom=0.15)

    gu_vals  = GAMMA_U_VALUES
    x        = np.arange(len(gu_vals))
    width    = 0.32
    x_labels = [f"$\\gamma_U = {g}$" for g in gu_vals]

    methods = ["logit", "nn"]
    offsets = [-width / 2, +width / 2]

    # ── Panel 1: Absolute bias ─────────────────────────────────────────────
    ax = axes[0]
    for m, dx in zip(methods, offsets):
        biases = [abs(stats[gu][m]["bias"]) for gu in gu_vals]
        errs   = [stats[gu][m]["bias_se"]   for gu in gu_vals]
        ax.bar(x + dx, biases, width, color=COLORS[m], alpha=0.85,
               label=LABELS[m], yerr=errs, capsize=3, ecolor="grey")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylabel("|Bias|", fontsize=11)
    ax.set_title("(a) Absolute Bias", fontsize=11)
    ax.tick_params(axis="y", labelsize=9)

    # ── Panel 2: Coverage ──────────────────────────────────────────────────
    ax = axes[1]
    for m, dx in zip(methods, offsets):
        covers = [stats[gu][m]["coverage"] for gu in gu_vals]
        errs   = [stats[gu][m]["coverage_se"] for gu in gu_vals]
        ax.bar(x + dx, covers, width, color=COLORS[m], alpha=0.85,
               label=LABELS[m], yerr=errs, capsize=3, ecolor="grey")
    ax.axhline(0.95, color="#e31a1c", linewidth=1.3, linestyle="--",
               label="Nominal 95%")
    ax.set_ylim(0, 1.12)
    ax.set_xticks(x); ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylabel("Coverage probability", fontsize=11)
    ax.set_title("(b) CI Coverage (95%)", fontsize=11)
    ax.tick_params(axis="y", labelsize=9)

    # ── Panel 3: CI width ──────────────────────────────────────────────────
    ax = axes[2]
    for m, dx in zip(methods, offsets):
        widths = [stats[gu][m]["width"] for gu in gu_vals]
        errs   = [stats[gu][m]["width_se"] for gu in gu_vals]
        ax.bar(x + dx, widths, width, color=COLORS[m], alpha=0.85,
               label=LABELS[m], yerr=errs, capsize=3, ecolor="grey")
    ax.set_xticks(x); ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylabel("Average CI width (log-odds)", fontsize=11)
    ax.set_title("(c) CI Width", fontsize=11)
    ax.tick_params(axis="y", labelsize=9)

    # ── Shared legend ──────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(color=COLORS["logit"], alpha=0.85, label="Logistic 2SRI"),
        mpatches.Patch(color=COLORS["nn"],    alpha=0.85, label="NN-2SRI"),
        plt.Line2D([0], [0], color="#e31a1c", linewidth=1.3,
                   linestyle="--", label="Nominal 95%"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, 1.01), frameon=False)

    os.makedirs(GFX_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(GFX_DIR, f"{out_prefix}.{ext}")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


# ── Summary table ────────────────────────────────────────────────────────────

def print_table(stats):
    print("\n── Results (200 reps) ─────────────────────────────────────────────")
    print(f"{'γ_U':>5}  {'Method':>18}  {'n_reps':>6}  "
          f"{'Bias':>8}  {'Coverage':>9}  {'CI Width':>9}")
    print("─" * 70)
    for gu in GAMMA_U_VALUES:
        n = stats[gu]["n"]
        for m in ["logit", "nn"]:
            s = stats[gu][m]
            lab = LABELS[m]
            print(f"  {gu:>3}  {lab:>18}  {n:>6}  "
                  f"{s['bias']:>+8.3f}  {s['coverage']:>9.2f}  {s['width']:>9.3f}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading results...", flush=True)
    data = load_complete()
    for gu, reps in data.items():
        print(f"  gamma_u={gu}: {len(reps)} complete reps")

    stats = compute_stats(data)
    print_table(stats)

    print("Making figure...", flush=True)
    make_figure(stats, "gamma_u_binary_ci")
    print("Done.")
