"""
Regenerate the cross-population thesis figures from existing caches with the
Oracle estimator removed from every plot.

The Oracle ("Linear 2SRI (oracle)") targets the structural beta_1, not the AME
that every other estimator targets, so displaying it alongside the others is
misleading. This script re-plots from the cached aggregated DataFrames only --
no simulation is re-run -- and writes the figures straight into the thesis gfx
directory.

Usage:
    python examples/replot_cross_pop_no_oracle.py \
        --thesis-gfx-dir ~/Desktop/thesis/Tomer-Weiss-Thesis-v2/gfx
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_cross_pop_study import (  # noqa: E402
    BETA_1,
    TARGET_POP,
    SETTINGS,
    plot_ame_bias_figure,
    plot_bias_convergence,
    plot_rmse_figure,
)
from run_population_ame_demo import (  # noqa: E402
    compute_true_ame,
    compute_true_ame_binary,
)

N_KEEP = [10_000, 20_000, 40_000]

# Continuous and binary caches identified as the source of the current thesis
# figures (partial-overlap Linear 2SRI bias = +0.13 continuous, matches text).
CONT_PKL = "cross_pop_cache/cross_pop_study_all.pkl"
BIN_PKL = "cross_pop_cache_rho0.5_binary/cross_pop_study_all.pkl"


def _cont_targets() -> dict:
    val = compute_true_ame(**TARGET_POP, beta_1=BETA_1, n=1_500_000)
    print(f"  continuous AME target: {val:+.4f}")
    return {name: val for name in SETTINGS}


def _bin_targets() -> dict:
    val = compute_true_ame_binary(**TARGET_POP, beta_1=BETA_1)
    print(f"  binary log-odds AME target: {val:+.4f}")
    return {name: val for name in SETTINGS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thesis-gfx-dir", required=True,
                    help="Destination gfx directory in the thesis repo.")
    args = ap.parse_args()
    out = os.path.expanduser(args.thesis_gfx_dir)
    os.makedirs(out, exist_ok=True)

    # --- Continuous (oracle targets structural beta_1 -> dropped from plots) ---
    print("Continuous figures...")
    df = pd.read_pickle(CONT_PKL)
    tgt = _cont_targets()
    plot_ame_bias_figure(df, tgt, savepath=os.path.join(out, "cross_pop_ame.png"),
                         binary=False, n_filter=N_KEEP, drop_oracle=True)
    plot_bias_convergence(df, tgt, savepath=os.path.join(out, "cross_pop_convergence.png"),
                          binary=False, n_filter=N_KEEP)
    plot_rmse_figure(df, savepath=os.path.join(out, "cross_pop_rmse.png"),
                     n_filter=N_KEEP, drop_oracle=True)

    # --- Binary: main figures drop the oracle to match the continuous panels ---
    print("Binary figures...")
    dfb = pd.read_pickle(BIN_PKL)
    tgtb = _bin_targets()
    # Drop the beta_1 reference line: no estimator targets the structural beta_1
    # in the binary case (they all target the log-odds AME theta*), so showing it
    # is confusing.
    plot_ame_bias_figure(dfb, tgtb, savepath=os.path.join(out, "cross_pop_ame_binary.png"),
                         binary=True, n_filter=N_KEEP, drop_oracle=True, show_beta1=False)
    plot_rmse_figure(dfb, savepath=os.path.join(out, "cross_pop_rmse_binary.png"),
                     n_filter=N_KEEP, drop_oracle=True)

    # --- Binary oracle-support figure: separate plot backing the claim that the
    #     partial-overlap bias is structural (persists even for an oracle that
    #     knows the true nuisance), while DeepPLIV mitigates it. ---
    print("Binary oracle-support figure...")
    # Keep the beta_1 = -2 reference here: the oracle-support figure contrasts
    # the structural beta_1 with the log-odds AME theta* to make the point that
    # the partial-overlap bias is structural.
    plot_ame_bias_figure(
        dfb, tgtb,
        savepath=os.path.join(out, "cross_pop_ame_binary_oracle.png"),
        binary=True, n_filter=N_KEEP,
        methods_override=["Linear 2SRI (oracle)", "DeepPLIV-2SRI"],
        show_beta1=True,
    )

    print("Done.")


if __name__ == "__main__":
    main()
