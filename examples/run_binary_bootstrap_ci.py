"""
Bootstrap CI coverage study — binary outcome (Deep IV DGP).

Replicates the CI evaluation of Section 5.3 (MR interactions) but for the
binary-outcome Deep IV setting of Section 5.2. Compares:

  DeepPLIV-2SRI   — ensemble-based bootstrap CI (Algorithm 1, Section 3.2)
  Linear 2SRI     — standard percentile bootstrap CI

under two first-stage training configurations:
  same-pop        — first and second stages trained on the same population
  partial-overlap — first stage on t~U(0,6), second stage on target t~U(4,10)

The estimand is the log-odds AME θ* (computed by Monte Carlo; ≈ -2.255 for the
target population).

Usage
-----
    # Quick single-simulation test (M=3, B=5, n=5000):
    python examples/run_binary_bootstrap_ci.py --test

    # Full study (M=50, B=200, multiple n and 200 reps, parallel):
    python examples/run_binary_bootstrap_ci.py --max-workers 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_deep_iv_simulation import DeepIVData, psi_t          # noqa: E402
from run_population_ame_demo import (                          # noqa: E402
    compute_true_ame_binary,
    sample_population,
    _naive_iv_features,
    _fit_linear_first_stage,
    _predict_linear_first_stage,
    NNFirstStage,
)
from deeppliv import DeepPLIV                                  # noqa: E402

# ---------------------------------------------------------------------------
# Study configuration
# ---------------------------------------------------------------------------

TARGET_POP    = dict(t_range=(4.0, 10.0), s_range=(5.0, 7.0))
PARTIAL_FS_POP = dict(t_range=(0.0,  6.0), s_range=(5.0, 7.0))

BETA_1   = -2.0
RHO      = 0.5
CI_LEVEL = 97.5          # percentile for 95% two-sided CI


# ---------------------------------------------------------------------------
# Core NN helpers
# ---------------------------------------------------------------------------


def _fit_nn_first_stage_binary(
    train: DeepIVData,
    epochs: int,
    lr: float,
    dropout: float,
) -> NNFirstStage:
    """Fit NN first stage. (Binary flag is irrelevant for the first stage.)"""
    model  = DeepPLIV()
    scaler = StandardScaler()
    g_iv_1 = scaler.fit_transform(_naive_iv_features(train.z_1, train.t_1))
    g_iv_2 = scaler.transform(_naive_iv_features(train.z_2, train.t_2))
    model.fit_first_stage(
        g_iv_1,
        train.x_1.reshape(-1, 1),
        epochs_first_stage=epochs,
        learning_rate_first_stage=lr,
        dropout=dropout,
        validation_data=(g_iv_2, train.x_2.reshape(-1, 1)),
    )
    return NNFirstStage(model=model, scaler=scaler)


def _fit_deeppliv_2sri(
    train: DeepIVData,
    nn_first: NNFirstStage,
    epochs: int,
    lr: float,
    dropout: float,
) -> float:
    """Fit DeepPLIV-2SRI on train, return β̂ from final linear layer.

    The second stage auto-detects binary outcome from y values and uses
    BCEWithLogitsLoss automatically (see deeppliv/models/second_stage.py).
    """
    x_hat  = nn_first.predict(train)           # shape (n2,)
    resid  = train.x_2 - x_hat

    exog   = np.column_stack([train.s_2, train.t_2, resid])
    scaler = StandardScaler()
    exog_s = scaler.fit_transform(exog)

    second = nn_first.model.fit_second_stage(
        train.x_2.reshape(-1, 1),
        exog_s,
        train.y_2.reshape(-1, 1),
        epochs_second_stage=epochs,
        learning_rate_second_stage=lr,
        dropout=dropout,
    )
    return float(second.final_layer.weight.detach().cpu().numpy()[0, 0])


def _fit_linear_2sri(train: DeepIVData, first=None) -> float:
    """Fit logistic 2SRI (control-function approach) on train. Returns β̂."""
    if first is None:
        first = _fit_linear_first_stage(train, oracle=False)
    x_hat  = _predict_linear_first_stage(first, train, oracle=False)
    resid  = train.x_2 - x_hat
    X      = np.column_stack([train.x_2, train.s_2, train.t_2, resid])
    model  = LogisticRegression(penalty=None, solver="lbfgs", max_iter=500,
                                fit_intercept=True)
    model.fit(X, train.y_2)
    return float(model.coef_.ravel()[0])


# ---------------------------------------------------------------------------
# Resample helpers (keep DeepIVData structure)
# ---------------------------------------------------------------------------


def _resample_data(data: DeepIVData, rng: np.random.Generator) -> DeepIVData:
    """Bootstrap resample both halves of a DeepIVData independently."""
    n1 = len(data.z_1)
    n2 = len(data.z_2)
    idx1 = rng.integers(0, n1, n1)
    idx2 = rng.integers(0, n2, n2)
    return DeepIVData(
        z_1=data.z_1[idx1], t_1=data.t_1[idx1], x_1=data.x_1[idx1],
        z_2=data.z_2[idx2], t_2=data.t_2[idx2], s_2=data.s_2[idx2],
        x_2=data.x_2[idx2], y_2=data.y_2[idx2],
    )


# ---------------------------------------------------------------------------
# Single-simulation bootstrap CI
# ---------------------------------------------------------------------------


def _run_ensemble_bootstrap_deeppliv(
    train_ss: DeepIVData,          # second-stage (target) training data
    train_fs: DeepIVData,          # first-stage training data (may equal train_ss)
    M: int,
    B: int,
    epochs: int,
    lr: float,
    dropout: float,
    seed: int,
) -> tuple[float, float, float]:
    """
    Ensemble-based bootstrap CI for DeepPLIV-2SRI (Algorithm 1).

    Returns (ensemble_mean, ci_lo, ci_hi).
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # ---- Stage 1: ensemble on original data ----
    betas_ens = []
    for m in range(M):
        torch.manual_seed(seed * 10_000 + m)
        nn_fs  = _fit_nn_first_stage_binary(train_fs, epochs, lr, dropout)
        b      = _fit_deeppliv_2sri(train_ss, nn_fs, epochs, lr, dropout)
        betas_ens.append(b)
    mean_ens = float(np.mean(betas_ens))

    # ---- Stage 2: bootstrap deviations ----
    devs = []
    for b_idx in range(B):
        torch.manual_seed(seed * 10_000 + M + b_idx)
        boot_fs = _resample_data(train_fs, rng)
        boot_ss = _resample_data(train_ss, rng)
        nn_fs_b = _fit_nn_first_stage_binary(boot_fs, epochs, lr, dropout)
        b_hat   = _fit_deeppliv_2sri(boot_ss, nn_fs_b, epochs, lr, dropout)
        devs.append(abs(b_hat - mean_ens))

    q = float(np.percentile(devs, CI_LEVEL))
    # Point estimate = first ensemble member (carries full v1+v2 noise)
    pt = betas_ens[0]
    return mean_ens, pt - q, pt + q


def _run_bootstrap_linear_2sri(
    train_ss: DeepIVData,
    train_fs: DeepIVData,
    B: int,
    seed: int,
) -> tuple[float, float, float]:
    """
    Standard percentile bootstrap CI for logistic 2SRI.

    Returns (point_est, ci_lo, ci_hi).
    """
    rng   = np.random.default_rng(seed)
    first = _fit_linear_first_stage(train_fs, oracle=False)
    point = _fit_linear_2sri(train_ss, first)

    boots = []
    for _ in range(B):
        boot_fs = _resample_data(train_fs, rng)
        boot_ss = _resample_data(train_ss, rng)
        first_b = _fit_linear_first_stage(boot_fs, oracle=False)
        boots.append(_fit_linear_2sri(boot_ss, first_b))

    ci_lo = float(np.percentile(boots, 100 - CI_LEVEL))
    ci_hi = float(np.percentile(boots, CI_LEVEL))
    return point, ci_lo, ci_hi


def run_one_sim(
    n: int,
    setting: str,         # "same-pop" | "partial-overlap"
    M: int,
    B: int,
    epochs: int,
    lr: float,
    dropout: float,
    true_ame: float,
    rho: float,
    beta_1: float,
    sim_idx: int,
) -> dict:
    """Run one simulation replicate; return a dict of results."""
    import warnings
    warnings.filterwarnings("ignore")
    torch.set_num_threads(1)

    seed = sim_idx * 1_000 + n
    rng  = np.random.default_rng(seed)

    # Sample data
    train_target = sample_population(
        n=n, rho=rho, t_range=TARGET_POP["t_range"],
        s_range=TARGET_POP["s_range"], beta_1=beta_1,
        rng=np.random.default_rng(rng.integers(0, 2**31)),
        binary=True,
    )

    if setting == "same-pop":
        train_fs = train_target
    else:  # partial-overlap
        train_fs = sample_population(
            n=n, rho=rho, t_range=PARTIAL_FS_POP["t_range"],
            s_range=PARTIAL_FS_POP["s_range"], beta_1=beta_1,
            rng=np.random.default_rng(rng.integers(0, 2**31)),
            binary=True,
        )

    # DeepPLIV ensemble bootstrap
    dp_mean, dp_lo, dp_hi = _run_ensemble_bootstrap_deeppliv(
        train_target, train_fs, M, B, epochs, lr, dropout, seed=seed + 1,
    )

    # Linear 2SRI standard bootstrap
    lin_pt, lin_lo, lin_hi = _run_bootstrap_linear_2sri(
        train_target, train_fs, B, seed=seed + 2,
    )

    return dict(
        sim=sim_idx, n=n, setting=setting,
        # DeepPLIV
        dp_mean=dp_mean,  dp_lo=dp_lo,  dp_hi=dp_hi,
        dp_covered=int(dp_lo <= true_ame <= dp_hi),
        dp_width=dp_hi - dp_lo,
        # Linear 2SRI
        lin_pt=lin_pt,    lin_lo=lin_lo, lin_hi=lin_hi,
        lin_covered=int(lin_lo <= true_ame <= lin_hi),
        lin_width=lin_hi - lin_lo,
        true_ame=true_ame,
    )


# ---------------------------------------------------------------------------
# Aggregation and printing
# ---------------------------------------------------------------------------


def print_results(df: pd.DataFrame) -> None:
    for n in sorted(df.n.unique()):
        print(f"\n{'='*65}")
        print(f"n = {n:,}")
        print(f"{'='*65}")
        for setting in ["same-pop", "partial-overlap"]:
            sub = df[(df.n == n) & (df.setting == setting)]
            if sub.empty:
                continue
            true_ame = sub.true_ame.iloc[0]
            nsims    = len(sub)
            print(f"\n  {setting}  (θ* = {true_ame:+.4f},  {nsims} sims)")
            print(f"  {'Method':<22} {'coverage':>9} {'mean width':>11} "
                  f"{'mean β̂':>9} {'bias':>9}")
            print(f"  {'-'*62}")
            for label, cov_col, width_col, est_col in [
                ("DeepPLIV-2SRI",  "dp_covered",  "dp_width",  "dp_mean"),
                ("Linear 2SRI",    "lin_covered", "lin_width", "lin_pt"),
            ]:
                cov   = sub[cov_col].mean()
                width = sub[width_col].mean()
                est   = sub[est_col].mean()
                bias  = est - true_ame
                print(f"  {label:<22} {cov:>9.3f} {width:>11.4f} "
                      f"{est:>+9.4f} {bias:>+9.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", action="store_true",
                        help="Quick single-sim test (M=3, B=5, n=5000).")
    parser.add_argument("--n", type=int, nargs="+", default=[10_000, 40_000])
    parser.add_argument("--M", type=int, default=50,
                        help="Ensemble size (default 50).")
    parser.add_argument("--B", type=int, default=200,
                        help="Bootstrap resamples (default 200).")
    parser.add_argument("--num-sims", type=int, default=200,
                        help="Simulation replications (default 200).")
    parser.add_argument("--max-workers", type=int, default=1,
                        help="Parallel workers (default 1 = sequential).")
    parser.add_argument("--rho", type=float, default=RHO)
    parser.add_argument("--out-dir", default="binary_bootstrap_results",
                        help="Output directory for results and figures.")
    parser.add_argument("--settings", nargs="+",
                        default=["same-pop", "partial-overlap"],
                        choices=["same-pop", "partial-overlap"])
    args = parser.parse_args()

    # Override for quick test
    if args.test:
        args.n        = [5_000]
        args.M        = 3
        args.B        = 5
        args.num_sims = 1
        args.max_workers = 1
        print("=== TEST MODE: M=3, B=5, n=5000, 1 simulation ===\n")

    os.makedirs(args.out_dir, exist_ok=True)

    # Compute MC AME target once
    print("Computing MC log-odds AME target (binary, K=20 cell logistic)...")
    true_ame = compute_true_ame_binary(**TARGET_POP, beta_1=BETA_1)
    print(f"  θ* ≈ {true_ame:+.5f}\n")

    # Epoch / dropout schedule
    epoch_map = {n: int(1.5e7 / max(n // 2, 1)) for n in args.n}
    drop_map  = {n: 100.0 / (1000.0 + n)        for n in args.n}
    LR = 0.01

    # Build task list
    tasks = [
        (n, setting, sim_idx)
        for n in args.n
        for setting in args.settings
        for sim_idx in range(args.num_sims)
    ]
    print(f"Total tasks: {len(tasks)}  (M={args.M}, B={args.B} per task)")
    print(f"Total NN pipelines: {len(tasks) * (args.M + args.B):,}\n")

    t0 = time.time()
    all_records: list[dict] = []

    if args.max_workers == 1:
        for i, (n, setting, sim_idx) in enumerate(tasks):
            print(f"[{i+1}/{len(tasks)}] n={n:,}  setting={setting}  sim={sim_idx} ...",
                  end=" ", flush=True)
            rec = run_one_sim(
                n=n, setting=setting, M=args.M, B=args.B,
                epochs=epoch_map[n], lr=LR, dropout=drop_map[n],
                true_ame=true_ame, rho=args.rho, beta_1=BETA_1,
                sim_idx=sim_idx,
            )
            all_records.append(rec)
            print(f"DP: [{rec['dp_lo']:+.4f}, {rec['dp_hi']:+.4f}]  "
                  f"covered={rec['dp_covered']}  |  "
                  f"Lin: [{rec['lin_lo']:+.4f}, {rec['lin_hi']:+.4f}]  "
                  f"covered={rec['lin_covered']}")
    else:
        with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {
                pool.submit(
                    run_one_sim,
                    n=n, setting=setting, M=args.M, B=args.B,
                    epochs=epoch_map[n], lr=LR, dropout=drop_map[n],
                    true_ame=true_ame, rho=args.rho, beta_1=BETA_1,
                    sim_idx=sim_idx,
                ): (i, n, setting, sim_idx)
                for i, (n, setting, sim_idx) in enumerate(tasks)
            }
            done = 0
            for fut in as_completed(futures):
                i, n, setting, sim_idx = futures[fut]
                try:
                    rec = fut.result()
                    all_records.append(rec)
                except Exception as exc:
                    print(f"  FAILED n={n} {setting} sim={sim_idx}: {exc}")
                done += 1
                if done % 10 == 0 or done == len(tasks):
                    elapsed = time.time() - t0
                    print(f"  [{done}/{len(tasks)}] elapsed={elapsed:.0f}s")

    df = pd.DataFrame(all_records)

    # Save
    out_path = os.path.join(args.out_dir, "binary_bootstrap_ci_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # Print summary
    print_results(df)
    print(f"\nTotal time: {time.time() - t0:.0f}s")
