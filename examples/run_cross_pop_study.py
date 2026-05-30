"""
Cross-population AME study for DeepPLIV.

Compares Linear 2SRI and DeepPLIV-2SRI under three first-stage training
configurations (same-pop, partial overlap, no overlap) at three sample sizes
(10k, 20k, 40k), with 50 replications each.

Each replication is cached to disk so the study can be resumed if interrupted.
Two replications run in parallel by default (--max-workers).

Usage
-----
    python examples/run_cross_pop_study.py --max-workers 2

Resume after crash (cached sims are skipped automatically):
    python examples/run_cross_pop_study.py --max-workers 2
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Ensure the examples directory is importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_deep_iv_simulation import DeepIVData, psi_t  # noqa: E402
from run_population_ame_demo import (  # noqa: E402
    sample_population,
    compute_true_ame,
    compute_true_ame_binary,
    _fit_linear_first_stage,
    _fit_nn_first_stage,
    _predict_linear_first_stage,
    fit_eval_naive_ols,
    fit_eval_linear_2sri_no_int,
    fit_eval_linear_2sri_oracle,
    fit_eval_deeppliv,
)


# ---------------------------------------------------------------------------
# Study configuration
# ---------------------------------------------------------------------------

TARGET_POP = dict(t_range=(4.0, 10.0), s_range=(5.0, 7.0))

SETTINGS = {
    "same-pop": dict(t_range=(4.0, 10.0), s_range=(5.0, 7.0)),
    "partial-overlap": dict(t_range=(0.0, 6.0), s_range=(5.0, 7.0)),
    "no-overlap": dict(t_range=(0.0, 3.0), s_range=(5.0, 7.0)),
}

N_VALUES = [10_000, 20_000, 40_000, 80_000]
NUM_SIMS = 50
BETA_1 = -2.0
DEFAULT_RHO = 0.5
LR = 0.01

METHODS_TO_RUN = [
    "Naive OLS",
    "Linear 2SRI",
    "Linear 2SRI (oracle)",
    "DeepPLIV-2SRI",
]


# ---------------------------------------------------------------------------
# Single-sim worker
# ---------------------------------------------------------------------------


def run_single_sim(
    n: int,
    sim_idx: int,
    cache_dir: str,
    rho: float = DEFAULT_RHO,
    binary: bool = False,
) -> pd.DataFrame:
    """Run all 3 settings for one (n, sim_idx) pair. Cached per-sim."""
    cache_path = os.path.join(cache_dir, f"n{n}_sim{sim_idx:03d}.pkl")
    if os.path.exists(cache_path):
        return pd.read_pickle(cache_path)

    import warnings
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)

    epochs = int(1.5e7 / max(n // 2, 1))
    dropout = 100.0 / (1000.0 + n)

    # Deterministic RNG per (n, sim_idx).
    master_rng = np.random.default_rng(n * 10_000 + sim_idx)

    # Target-population data (shared across all 3 settings).
    train_target = sample_population(
        n=n, rho=rho, beta_1=BETA_1,
        rng=np.random.default_rng(master_rng.integers(0, 2**31 - 1)),
        binary=binary,
        **TARGET_POP,
    )
    test_target = sample_population(
        n=n, rho=rho, beta_1=BETA_1,
        rng=np.random.default_rng(master_rng.integers(0, 2**31 - 1)),
        binary=binary,
        **TARGET_POP,
    )

    records: list[dict] = []

    for setting_name, fs_pop in SETTINGS.items():
        # First-stage training data (same as target for same-pop, different otherwise).
        if setting_name == "same-pop":
            train_fs = train_target
        else:
            train_fs = sample_population(
                n=n, rho=rho, beta_1=BETA_1,
                rng=np.random.default_rng(master_rng.integers(0, 2**31 - 1)),
                binary=binary,
                **fs_pop,
            )

        # Fit first stages on train_fs.
        lin_fs_naive = _fit_linear_first_stage(train_fs, oracle=False)
        lin_fs_oracle = _fit_linear_first_stage(train_fs, oracle=True)
        nn_fs = _fit_nn_first_stage(train_fs, epochs, LR, dropout)

        # Run estimators (second stage always on train_target, eval on test_target).
        res_ols = fit_eval_naive_ols(train_target, test_target, binary=binary)
        res_lin = fit_eval_linear_2sri_no_int(train_target, test_target, first=lin_fs_naive, binary=binary)
        res_oracle = fit_eval_linear_2sri_oracle(train_target, test_target, first=lin_fs_oracle, binary=binary)
        res_nn = fit_eval_deeppliv(
            train_target, test_target, "2sri", epochs, LR, dropout, nn_fs,
        )

        for method, res in [
            ("Naive OLS", res_ols),
            ("Linear 2SRI", res_lin),
            ("Linear 2SRI (oracle)", res_oracle),
            ("DeepPLIV-2SRI", res_nn),
        ]:
            records.append(dict(
                n=n, sim=sim_idx, setting=setting_name, method=method,
                ame=res["ame"], rmse=res["rmse"],
            ))

    df = pd.DataFrame.from_records(records)
    os.makedirs(cache_dir, exist_ok=True)
    df.to_pickle(cache_path)
    return df


# ---------------------------------------------------------------------------
# Aggregation and plotting
# ---------------------------------------------------------------------------


def load_all_cached(cache_dir: str) -> pd.DataFrame:
    """Load and concatenate all cached per-sim DataFrames."""
    frames = []
    for fname in sorted(os.listdir(cache_dir)):
        if fname.endswith(".pkl"):
            frames.append(pd.read_pickle(os.path.join(cache_dir, fname)))
    if not frames:
        raise FileNotFoundError(f"No cached results in {cache_dir}")
    return pd.concat(frames, ignore_index=True)


def print_summary(df: pd.DataFrame, ame_targets: dict) -> None:
    for n in sorted(df.n.unique()):
        print(f"\n{'='*70}")
        print(f"n = {n}")
        print(f"{'='*70}")
        for setting in ["same-pop", "partial-overlap"]:
            sub = df[(df.n == n) & (df.setting == setting)]
            if sub.empty:
                continue
            target = ame_targets.get(setting, float("nan"))
            print(f"\n  {setting} (AME target = {target:+.4f}):")
            for method in METHODS_TO_RUN:
                ms = sub[sub.method == method]
                if ms.empty:
                    continue
                m_ame = ms.ame.mean()
                s_ame = ms.ame.std()
                m_rmse = ms.rmse.mean()
                s_rmse = ms.rmse.std()
                bias = m_ame - target
                print(f"    {method:25s}  AME={m_ame:+.4f}±{s_ame:.4f}  "
                      f"bias={bias:+.4f}  RMSE={m_rmse:.4f}±{s_rmse:.4f}")


def plot_ame_bias_figure(
    df: pd.DataFrame,
    ame_targets: dict,
    savepath: str | None = None,
    binary: bool = False,
    n_filter: list | None = None,
) -> None:
    """2-column (one per setting) × n-row (one per n) grid of AME boxplots."""
    import seaborn as sns
    sns.set_style("whitegrid")

    n_vals = sorted(n for n in df.n.unique() if n_filter is None or n in n_filter)
    settings = ["same-pop", "partial-overlap"]
    setting_labels = {
        "same-pop": "Same population",
        "partial-overlap": "Partial overlap (train t~U(0,6))",
    }
    methods_plot = ["Naive OLS", "Linear 2SRI", "Linear 2SRI (oracle)", "DeepPLIV-2SRI"]

    fig, axes = plt.subplots(
        len(n_vals), len(settings),
        figsize=(7.5, 4.0 * len(n_vals)),
        sharey=True,
    )

    for row, n in enumerate(n_vals):
        for col, setting in enumerate(settings):
            ax = axes[row, col]
            target = ame_targets[setting]
            sub = df[(df.n == n) & (df.setting == setting) & df.method.isin(methods_plot)]
            if sub.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes)
                continue
            sns.boxplot(
                data=sub, x="method", y="ame", order=methods_plot,
                ax=ax, palette="Set2",
            )
            ax.axhline(
                BETA_1, color="grey", linestyle=":", linewidth=1.5,
                label=rf"$\beta_1 = {BETA_1}$",
            )
            ax.axhline(
                target, color="red", linestyle="--", linewidth=1.5,
                label=rf"AME $\theta^\star = {target:+.3f}$",
            )
            if row == 0:
                ax.set_title(setting_labels[setting], fontsize=13)
            if col == 0:
                coef_label = r"$\hat{\beta}_1$ (log-odds)" if binary else r"$\hat{\beta}_1$"
                ax.set_ylabel(f"n = {n:,}\n" + coef_label, fontsize=11)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=20)
            for lbl in ax.get_xticklabels():
                lbl.set_horizontalalignment("right")
            ax.legend(loc="best", fontsize=8, framealpha=0.9)

    outcome_note = " (binary outcome, log-odds scale)" if binary else ""
    fig.suptitle(
        "AME estimates under same-population and partial-overlap "
        f"first-stage training{outcome_note}\n"
        f"Target population: t ~ U(4,10), s ~ U(5,7), "
        rf"$\beta_1 = {BETA_1}$",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
        print(f"  Saved AME figure to {savepath}")
    plt.close(fig)


def plot_bias_convergence(
    df: pd.DataFrame,
    ame_targets: dict,
    savepath: str | None = None,
    binary: bool = False,
    n_filter: list | None = None,
) -> None:
    """Two-panel line plot of |AME bias| vs n, one panel per setting."""
    settings = ["same-pop", "partial-overlap"]
    setting_labels = {
        "same-pop": "Same population",
        "partial-overlap": "Partial overlap (train t~U(0,6))",
    }
    methods_plot = ["Linear 2SRI", "DeepPLIV-2SRI"]
    markers = {"Linear 2SRI": "s", "DeepPLIV-2SRI": "o"}
    colors = {"Linear 2SRI": "#66c2a5", "DeepPLIV-2SRI": "#fc8d62"}

    n_vals = sorted(n for n in df.n.unique() if n_filter is None or n in n_filter)
    outcome_note = " (binary outcome)" if binary else ""
    bias_label = r"|AME bias| (log-odds)" if binary else "|AME bias|"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, setting in zip(axes, settings):
        target = ame_targets[setting]
        for method in methods_plot:
            biases = []
            for n in n_vals:
                sub = df[(df.n == n) & (df.setting == setting) & (df.method == method)]
                biases.append(np.nan if sub.empty else abs(sub.ame.mean() - target))
            ax.plot(
                n_vals, biases,
                marker=markers[method],
                color=colors[method],
                linestyle="-",
                linewidth=2,
                markersize=8,
                label=method,
            )
        ax.set_title(setting_labels[setting], fontsize=13)
        ax.set_xlabel("Sample size (n)", fontsize=12)
        ax.set_xticks(n_vals)
        ax.set_xticklabels([f"{n:,}" for n in n_vals])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.4)

    axes[0].set_ylabel(bias_label, fontsize=12)
    fig.suptitle(
        f"AME bias convergence: Linear 2SRI vs DeepPLIV-2SRI{outcome_note}",
        fontsize=13,
    )
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
        print(f"  Saved convergence figure to {savepath}")
    plt.close(fig)


def plot_rmse_figure(
    df: pd.DataFrame,
    savepath: str | None = None,
    n_filter: list | None = None,
) -> None:
    """1-row × 2-col RMSE boxplots at the largest n in n_filter."""
    import seaborn as sns
    sns.set_style("whitegrid")

    n_vals = sorted(n for n in df.n.unique() if n_filter is None or n in n_filter)
    n_show = max(n_vals)
    settings = ["same-pop", "partial-overlap"]
    setting_labels = {
        "same-pop": "Same population",
        "partial-overlap": "Partial overlap",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for ax, setting in zip(axes, settings):
        ss = df[(df.n == n_show) & (df.setting == setting) & df.method.isin(METHODS_TO_RUN)]
        if ss.empty:
            continue
        sns.boxplot(
            data=ss, x="method", y="rmse", order=METHODS_TO_RUN,
            ax=ax, palette="Set2",
        )
        ax.set_title(setting_labels[setting], fontsize=13)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
        for lbl in ax.get_xticklabels():
            lbl.set_horizontalalignment("right")
    axes[0].set_ylabel("Test RMSE", fontsize=13)
    fig.suptitle(f"Test-set RMSE at n = {n_show:,}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
        print(f"  Saved RMSE figure to {savepath}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-workers", type=int, default=2,
                        help="Number of parallel workers (default 2).")
    parser.add_argument("--cache-dir", default=None,
                        help="Directory for per-sim pickle caches (default: cross_pop_cache_rho{rho}).")
    parser.add_argument("--rho", type=float, default=DEFAULT_RHO,
                        help=f"Endogeneity strength (default {DEFAULT_RHO}).")
    parser.add_argument("--thesis-gfx-dir", default=None,
                        help="Copy final figures to this directory.")
    parser.add_argument("--num-sims", type=int, default=NUM_SIMS)
    parser.add_argument("--binary", action="store_true", default=False,
                        help="Use binary outcome (logistic DGP). Default: continuous.")
    args = parser.parse_args()

    if args.cache_dir is None:
        outcome_tag = "binary" if args.binary else "continuous"
        args.cache_dir = f"cross_pop_cache_rho{args.rho}_{outcome_tag}"
    os.makedirs(args.cache_dir, exist_ok=True)

    # AME targets: binary outcomes have no AME vs structural tension — the target
    # is simply beta_1 on the log-odds scale. Continuous outcomes require MC.
    if args.binary:
        print("Computing MC log-odds AME target (binary, cell fixed-effects logistic)...")
        ame_target_val = compute_true_ame_binary(**TARGET_POP, beta_1=BETA_1)
        ame_targets = {name: ame_target_val for name in SETTINGS}
        print(f"  Log-odds AME target (all settings): {ame_target_val:+.4f}")
    else:
        print("Computing MC AME targets...")
        ame_targets = {}
        for setting_name in SETTINGS:
            ame_targets[setting_name] = compute_true_ame(
                **TARGET_POP, beta_1=BETA_1, n=1_500_000,
            )
        print(f"  AME target (all settings share same target pop): {ame_targets['same-pop']:+.4f}")

    # Build task list: (n, sim_idx) pairs.
    tasks = [
        (n, sim_idx)
        for n in N_VALUES
        for sim_idx in range(args.num_sims)
    ]

    # Count already-cached tasks.
    cached = sum(
        1 for n, sim_idx in tasks
        if os.path.exists(os.path.join(args.cache_dir, f"n{n}_sim{sim_idx:03d}.pkl"))
    )
    remaining = len(tasks) - cached
    print(f"\nTotal tasks: {len(tasks)}, cached: {cached}, remaining: {remaining}")

    if remaining > 0:
        print(f"Running {remaining} tasks with {args.max_workers} workers...\n")
        t0 = time.time()

        with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {
                pool.submit(run_single_sim, n, sim_idx, args.cache_dir, args.rho, args.binary): (n, sim_idx)
                for n, sim_idx in tasks
            }
            done_count = cached
            for future in as_completed(futures):
                n, sim_idx = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"  FAILED n={n} sim={sim_idx}: {exc}")
                else:
                    done_count += 1
                    if done_count % 10 == 0 or done_count == len(tasks):
                        elapsed = time.time() - t0
                        print(f"  [{done_count}/{len(tasks)}] "
                              f"elapsed={elapsed:.0f}s")

        print(f"\nAll tasks done in {time.time() - t0:.0f}s.")
    else:
        print("All tasks already cached.")

    # Aggregate.
    print("\nAggregating results...")
    df = load_all_cached(args.cache_dir)
    print(f"  Total rows: {len(df)}")

    print_summary(df, ame_targets)

    # Figures.
    print("\nGenerating figures...")
    fig_dir = args.cache_dir
    plot_ame_bias_figure(df, ame_targets,
                         savepath=os.path.join(fig_dir, "cross_pop_ame.png"),
                         binary=args.binary)
    plot_bias_convergence(df, ame_targets,
                          savepath=os.path.join(fig_dir, "cross_pop_convergence.png"),
                          binary=args.binary)
    plot_rmse_figure(df,
                     savepath=os.path.join(fig_dir, "cross_pop_rmse.png"))

    if args.thesis_gfx_dir:
        import shutil
        for fname in ["cross_pop_ame.png", "cross_pop_convergence.png", "cross_pop_rmse.png"]:
            src = os.path.join(fig_dir, fname)
            if os.path.exists(src):
                dst = os.path.join(args.thesis_gfx_dir, fname)
                shutil.copy(src, dst)
                print(f"  Copied {fname} to {dst}")

    # Save aggregated DataFrame.
    agg_path = os.path.join(fig_dir, "cross_pop_study_all.pkl")
    df.to_pickle(agg_path)
    print(f"\nSaved aggregated DataFrame to {agg_path}")
    print("Done.")
