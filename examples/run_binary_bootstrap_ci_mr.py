#!/usr/bin/env python3
"""
Bootstrap CI coverage simulation — MR Interactions DGP, binary outcome.

Replicates the continuous-outcome CI study from example_genetic_iv.py / aws_genetic_iv_parallel.py
but replaces Y with a binary outcome via logistic link:

    logit_p = beta_x * X + beta_u * U + eps_y   (eps_y ~ N(0,1))
    Y ~ Bernoulli(sigma(logit_p))

True structural coefficient: beta_x = 2.0 (log-odds of X on P(Y=1)).

Estimators
----------
- NN-2SRI  : DeepPLIV with ensemble-based bootstrap CI (Algorithm 1, Arie & Gorfine 2024)
- Logit-2SRI: Linear first-stage + logistic second-stage with standard percentile bootstrap CI

Usage
-----
    # Quick test (M=3, B=5, n=500, 1 sim)
    python examples/run_binary_bootstrap_ci_mr.py --test

    # Single full sim
    python examples/run_binary_bootstrap_ci_mr.py --n 10000 --beta-u 2 --M 50 --B 200

    # Coverage study: 200 sims, confounded, same-pop
    python examples/run_binary_bootstrap_ci_mr.py --n 10000 --beta-u 2 --M 50 --B 200 --n-sims 200

    # Cross-population (p01y=0.2)
    python examples/run_binary_bootstrap_ci_mr.py --n 10000 --p01y 0.2 --beta-u 2 --M 50 --B 200
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deeppliv import DeepPLIV

# ================================================================
# Constants
# ================================================================
K1 = K2 = 7
beta_x = 2.0        # structural log-odds coefficient (the estimand)
TRUE_BETA = beta_x

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SNP_COLS = [f"S1_{j+1}" for j in range(K1)] + [f"S2_{j+1}" for j in range(K2)]


# ================================================================
# Fixed effects
# ================================================================
def load_fixed_effects(n=40000):
    """Load gamma coefficients from the nearest available pickle file."""
    candidates = [
        os.path.join(SCRIPT_DIR, f"fixed_effects_{n}_new.pkl"),
        os.path.join(SCRIPT_DIR, "fixed_effects_40000_new.pkl"),
        os.path.join(SCRIPT_DIR, "fixed_effects.pkl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "rb") as f:
                d = pickle.load(f)
            return d["gamma_j1"], d["gamma_j2"], d["gamma_jm12"], d["gamma_j1121"]
    raise FileNotFoundError(f"No fixed_effects pickle found in {SCRIPT_DIR}")


# ================================================================
# DGP functions (MR Interactions, same structure as example_genetic_iv.py)
# ================================================================
def simulate_snp_block(n, K, p0, rho):
    Z = np.random.normal(size=n)
    snps = np.zeros((n, K))
    for j in range(K):
        eps = np.random.normal(size=n)
        p = norm.cdf(norm.ppf(p0) + np.sqrt(rho) * Z + np.sqrt(1 - rho) * eps)
        snps[:, j] = np.random.binomial(2, p)
    return snps


def simulate_snp_dataset(n, p01, p02, rho):
    S1 = simulate_snp_block(n, K1, p01, rho)
    S2 = simulate_snp_block(n, K2, p02, rho)
    return S1, S2


def x_eq(S1, S2, n, gamma_j1, gamma_j2, gamma_jm12, gamma_j1121, gamma_u):
    interaction_term = np.sum(
        S1[:, :, None] * S2[:, None, :] * gamma_jm12[None, :, :], axis=(1, 2)
    )
    non_linear_part = (
        S1 @ gamma_j1
        + S2 @ gamma_j2
        + interaction_term
        + (S1[:, 0] * S1[:, 1]) * (S2 @ gamma_j1121)
    )
    epsilon_x = np.random.normal(0, 1.0, size=n)
    u = np.random.normal(0, 1.0, size=n)
    return non_linear_part + epsilon_x + gamma_u * u, epsilon_x, u


def gen_data_binary(n, p01, p02, rho, gamma_u, beta_u,
                    gamma_j1, gamma_j2, gamma_jm12, gamma_j1121,
                    dgp="threshold"):
    """Generate MR-interactions dataset with binary Y.

    dgp="threshold"  (default)
        Latent variable model: Y* = beta_x*X + beta_u*U + eps_y, Y = I(Y* > 0).
        eps_y is additive noise outside any link function — structural coefficient
        beta_x = 2 is preserved as the probit coefficient.

    dgp="logistic"
        Pure logistic model: Y ~ Bernoulli(sigma(beta_x*X + beta_u*U)).
        No additive eps_y at all — randomness comes only from the Bernoulli draw,
        exactly as in standard logistic regression theory.
    """
    S1, S2 = simulate_snp_dataset(n, p01, p02, rho)
    X, eps_x, u = x_eq(S1, S2, n, gamma_j1, gamma_j2, gamma_jm12, gamma_j1121, gamma_u)

    if dgp == "logistic":
        prob = expit(beta_x * X + beta_u * u)
        Y = np.random.binomial(1, prob).astype(float)
    else:  # threshold
        epsilon_y = np.random.normal(0, 1, size=n)
        Y_star = beta_x * X + beta_u * u + epsilon_y
        Y = (Y_star > 0).astype(float)

    df = pd.DataFrame({
        **{f"S1_{j+1}": S1[:, j] for j in range(K1)},
        **{f"S2_{j+1}": S2[:, j] for j in range(K2)},
        "X": X,
        "Y": Y,
    })
    return df


# ================================================================
# Estimators
# ================================================================
def fit_nn(df_x, df_y, method="2sri", epochs=1000, lr=0.01, dropout=0.01):
    """
    Fit DeepPLIV for binary Y — either 2SRI or 2SPS variant.

    2SRI: second stage receives raw X (linear head) + residual ê (deep network).
    2SPS: second stage receives predicted X̂ (linear head) + constant (deep network).

    Binary Y auto-detected via BCEWithLogitsLoss.
    Returns beta_1 = final_layer.weight[0, 0].
    """
    model = DeepPLIV()

    Z_X = df_x[SNP_COLS].values
    Z_Y = df_y[SNP_COLS].values
    X = df_x["X"].values
    X_Y = df_y["X"].values
    Y = df_y["Y"].values.astype(float)

    model.fit_first_stage(
        Z_X, X,
        epochs_first_stage=epochs,
        learning_rate_first_stage=lr,
        dropout=dropout,
        validation_data=(Z_Y, X_Y),
    )

    x_pred = model.first_stage_model.predict(Z_Y).reshape(-1, 1)

    if method == "2sri":
        x_err = X_Y.reshape(-1, 1) - x_pred
        m = model.fit_second_stage(
            X_Y.reshape(-1, 1), x_err, Y.reshape(-1, 1),
            epochs_second_stage=epochs,
            learning_rate_second_stage=lr,
            dropout=dropout,
            method="2sri",
        )
    else:  # 2sps
        m = model.fit_second_stage(
            x_pred, Z_Y, Y.reshape(-1, 1),
            epochs_second_stage=epochs,
            learning_rate_second_stage=lr,
            dropout=dropout,
            method="2sps",
        )
    return float(m.final_layer.weight.detach().numpy()[0, 0])


def fit_nn_2sri(df_x, df_y, epochs=1000, lr=0.01, dropout=0.01):
    return fit_nn(df_x, df_y, method="2sri", epochs=epochs, lr=lr, dropout=dropout)


def fit_nn_2sps(df_x, df_y, epochs=1000, lr=0.01, dropout=0.01):
    return fit_nn(df_x, df_y, method="2sps", epochs=epochs, lr=lr, dropout=dropout)


def fit_naive_logistic(df_y):
    """Naive logistic regression of Y on X — no IV correction, biased under confounding."""
    X_stk = sm.add_constant(df_y["X"].values)
    result = sm.Logit(df_y["Y"].values, X_stk).fit(disp=False, maxiter=200)
    return float(result.params[1])


def fit_logit_2sri(df_x, df_y):
    """
    Linear first-stage OLS + logistic second-stage 2SRI for binary Y.

    Persistent bias expected under same-population confounding due to
    non-collapsibility of the logistic link.
    """
    Z = df_x[SNP_COLS]
    Z_Y = df_y[SNP_COLS]
    X_hat = LinearRegression().fit(Z, df_x["X"]).predict(Z_Y)
    x_err = df_y["X"].values - X_hat
    X_stk = sm.add_constant(np.column_stack([df_y["X"].values, x_err]))
    result = sm.Logit(df_y["Y"].values, X_stk).fit(disp=False, maxiter=200)
    # params layout: [const, X, residual]
    return float(result.params[1])


# ================================================================
# Ensemble-based bootstrap (Algorithm 1, Arie & Gorfine 2024)
# ================================================================
def ensemble_bootstrap_ci_nn(df_x, df_y, M, B, method="2sri", alpha=0.05, nn_kwargs=None, verbose=True):
    """
    Stage 1: train M models on original data → ensemble mean beta_hat_M.
    Stage 2: resample B bootstrap datasets → |beta_b - beta_hat_M|.
    CI: beta_k ± q_{1-alpha}(deviations), where beta_k is a single ensemble member.

    method: "2sri" or "2sps"
    Returns (beta_k, (ci_lo, ci_hi), beta_hat_M, ensemble_betas).
    """
    if nn_kwargs is None:
        nn_kwargs = {}

    _fit = lambda dx, dy: fit_nn(dx, dy, method=method, **nn_kwargs)

    # --- Stage 1: ensemble ---
    ensemble_betas = []
    for m in range(M):
        b = _fit(df_x, df_y)
        ensemble_betas.append(b)
        if verbose:
            print(f"  Ensemble {m+1}/{M}: beta={b:.4f}")

    beta_hat_M = float(np.mean(ensemble_betas))
    beta_k = ensemble_betas[0]
    if verbose:
        print(f"  Ensemble mean: {beta_hat_M:.4f}  (point estimate: {beta_k:.4f})")

    # --- Stage 2: bootstrap deviations ---
    n_x, n_y = len(df_x), len(df_y)
    deviations = []
    for b in range(B):
        idx_x = np.random.choice(n_x, n_x, replace=True)
        idx_y = np.random.choice(n_y, n_y, replace=True)
        dfx_b = df_x.iloc[idx_x].reset_index(drop=True)
        dfy_b = df_y.iloc[idx_y].reset_index(drop=True)
        beta_b = _fit(dfx_b, dfy_b)
        deviations.append(abs(beta_b - beta_hat_M))
        if verbose:
            print(f"  Bootstrap {b+1}/{B}: beta={beta_b:.4f}  |dev|={deviations[-1]:.4f}")

    q = float(np.quantile(deviations, 1 - alpha))
    ci = (beta_k - q, beta_k + q)
    return beta_k, ci, beta_hat_M, ensemble_betas


def percentile_bootstrap_ci_logit(df_x, df_y, B, alpha=0.05, verbose=True):
    """
    Standard percentile bootstrap CI for logistic 2SRI.

    Logistic regression has negligible optimization noise (convex, unique solution),
    so the standard bootstrap is appropriate (no ensemble step needed).
    """
    point = fit_logit_2sri(df_x, df_y)
    n_x, n_y = len(df_x), len(df_y)
    estimates = []
    for b in range(B):
        idx_x = np.random.choice(n_x, n_x, replace=True)
        idx_y = np.random.choice(n_y, n_y, replace=True)
        est = fit_logit_2sri(
            df_x.iloc[idx_x].reset_index(drop=True),
            df_y.iloc[idx_y].reset_index(drop=True),
        )
        estimates.append(est)
        if verbose and (b + 1) % 20 == 0:
            print(f"  Logit bootstrap {b+1}/{B}")

    lo = float(np.percentile(estimates, 100 * alpha / 2))
    hi = float(np.percentile(estimates, 100 * (1 - alpha / 2)))
    return point, (lo, hi)


# ================================================================
# Single simulation
# ================================================================
def run_one_sim(n, p01, p01_y, rho, gamma_u, beta_u,
                M, B, nn_kwargs, gamma_j1, gamma_j2, gamma_jm12, gamma_j1121,
                dgp="threshold", verbose=True):
    t0 = time.time()

    df_x = gen_data_binary(n, p01, p01, rho, gamma_u, beta_u,
                           gamma_j1, gamma_j2, gamma_jm12, gamma_j1121, dgp=dgp)
    df_y = gen_data_binary(n, p01_y, p01_y, rho, gamma_u, beta_u,
                           gamma_j1, gamma_j2, gamma_jm12, gamma_j1121, dgp=dgp)

    y_mean = df_y["Y"].mean()
    if verbose:
        print(f"  df_x shape: {df_x.shape}, df_y shape: {df_y.shape}, Y mean: {y_mean:.3f}")

    if verbose:
        print(f"\n  --- NN-2SRI (M={M}, B={B}) ---")
    nn_sri_beta, nn_sri_ci, nn_sri_mean, _ = ensemble_bootstrap_ci_nn(
        df_x, df_y, M=M, B=B, method="2sri", nn_kwargs=nn_kwargs, verbose=verbose
    )

    naive_beta = fit_naive_logistic(df_y)

    if verbose:
        print(f"\n  --- Logit-2SRI (B={B}) ---")
    log_beta, log_ci = percentile_bootstrap_ci_logit(
        df_x, df_y, B=B, verbose=verbose
    )

    return {
        "true_beta": TRUE_BETA,
        "naive_beta": naive_beta,
        "nn_sri_beta": nn_sri_beta,
        "nn_sri_ensemble_mean": nn_sri_mean,
        "nn_sri_ci_lo": nn_sri_ci[0],
        "nn_sri_ci_hi": nn_sri_ci[1],
        "nn_sri_ci_width": nn_sri_ci[1] - nn_sri_ci[0],
        "nn_sri_covers": int(nn_sri_ci[0] <= TRUE_BETA <= nn_sri_ci[1]),
        "logit_beta": log_beta,
        "logit_ci_lo": log_ci[0],
        "logit_ci_hi": log_ci[1],
        "logit_ci_width": log_ci[1] - log_ci[0],
        "logit_covers": int(log_ci[0] <= TRUE_BETA <= log_ci[1]),
        "y_mean": y_mean,
        "elapsed_s": time.time() - t0,
    }


# ================================================================
# Entry point
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Binary-outcome bootstrap CI coverage: MR Interactions DGP"
    )
    parser.add_argument("--n", type=int, default=10000, help="Sample size per cohort")
    parser.add_argument("--rho", type=float, default=0.5, help="SNP LD correlation")
    parser.add_argument("--p01", type=float, default=0.4, help="Minor allele freq (exposure cohort)")
    parser.add_argument("--p01y", type=float, default=0.4,
                        help="Minor allele freq (outcome cohort); 0.4=same-pop, 0.2=cross-pop")
    parser.add_argument("--gamma-u", type=float, default=1.0,
                        help="Confounding strength in exposure equation")
    parser.add_argument("--beta-u", type=float, default=2.0,
                        help="Confounding in outcome: 0=unconfounded, 2=confounded")
    parser.add_argument("--M", type=int, default=50, help="Ensemble size")
    parser.add_argument("--B", type=int, default=200, help="Bootstrap resamples")
    parser.add_argument("--n-sims", type=int, default=1, help="Number of Monte Carlo replications")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", action="store_true",
                        help="Quick sanity check: M=3, B=5, n=500, epochs=300, 1 sim")
    parser.add_argument("--dgp", type=str, default="threshold",
                        choices=["threshold", "logistic"],
                        help="Binary DGP: 'threshold' = I(Y*>0), 'logistic' = Bernoulli(sigma(2X+2U))")
    parser.add_argument("--out", type=str, default=None,
                        help="CSV path for results (default: binary_ci_results_<params>.csv)")
    args = parser.parse_args()

    if args.test:
        args.M = 3
        args.B = 5
        args.n = 500
        args.epochs = 300
        args.n_sims = 1
        print("TEST MODE: M=3, B=5, n=500, epochs=300, 1 sim\n")

    np.random.seed(args.seed)

    gamma_j1, gamma_j2, gamma_jm12, gamma_j1121 = load_fixed_effects(args.n)
    nn_kwargs = {"epochs": args.epochs, "lr": args.lr, "dropout": args.dropout}

    tag = f"{args.dgp}_n{args.n}_p01y{args.p01y}_bu{args.beta_u}"
    out_path = args.out or os.path.join(
        SCRIPT_DIR, f"binary_ci_results_{tag}.csv"
    )

    print("=" * 60)
    print("MR Interactions DGP — Binary Outcome Bootstrap CI")
    print("=" * 60)
    print(f"DGP             : {args.dgp}")
    print(f"True beta_x     : {TRUE_BETA}")
    print(f"n per cohort    : {args.n}")
    print(f"p01 / p01_y     : {args.p01} / {args.p01y}  ({'same-pop' if args.p01y == args.p01 else 'cross-pop'})")
    print(f"rho             : {args.rho}")
    print(f"gamma_u / beta_u: {args.gamma_u} / {args.beta_u}  ({'confounded' if args.beta_u != 0 else 'unconfounded'})")
    print(f"M={args.M}, B={args.B}, n_sims={args.n_sims}")
    print(f"NN epochs={args.epochs}, lr={args.lr}, dropout={args.dropout}")
    print("=" * 60)

    all_results = []
    for i in range(args.n_sims):
        np.random.seed(args.seed + i)
        print(f"\n=== Simulation {i+1}/{args.n_sims} (seed={args.seed + i}) ===")
        r = run_one_sim(
            n=args.n, p01=args.p01, p01_y=args.p01y,
            rho=args.rho, gamma_u=args.gamma_u, beta_u=args.beta_u,
            M=args.M, B=args.B, nn_kwargs=nn_kwargs,
            gamma_j1=gamma_j1, gamma_j2=gamma_j2,
            gamma_jm12=gamma_jm12, gamma_j1121=gamma_j1121,
            dgp=args.dgp, verbose=True,
        )
        all_results.append(r)

        print(f"\n  Naive Logit: beta={r['naive_beta']:.4f}  bias={r['naive_beta']-TRUE_BETA:.4f}")
        print(f"  NN-2SRI   : beta={r['nn_sri_beta']:.4f}  "
              f"CI=[{r['nn_sri_ci_lo']:.4f}, {r['nn_sri_ci_hi']:.4f}]  "
              f"covers={bool(r['nn_sri_covers'])}  "
              f"(ensemble mean={r['nn_sri_ensemble_mean']:.4f})")
        print(f"  Logit-2SRI: beta={r['logit_beta']:.4f}  "
              f"CI=[{r['logit_ci_lo']:.4f}, {r['logit_ci_hi']:.4f}]  "
              f"covers={bool(r['logit_covers'])}")
        print(f"  Elapsed: {r['elapsed_s']:.1f}s")

    df_res = pd.DataFrame(all_results)
    df_res.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 60)
    print(f"SUMMARY ({args.n_sims} simulations, true beta={TRUE_BETA})")
    print("=" * 60)
    print(f"Naive Logit beta mean: {df_res['naive_beta'].mean():.4f}  "
          f"bias={df_res['naive_beta'].mean()-TRUE_BETA:.4f}")
    print()
    print(f"NN-2SRI   coverage : {df_res['nn_sri_covers'].mean():.3f}  (target 0.95)")
    print(f"NN-2SRI   beta mean: {df_res['nn_sri_beta'].mean():.4f}  "
          f"bias={df_res['nn_sri_beta'].mean()-TRUE_BETA:.4f}")
    print(f"NN-2SRI   CI width : {df_res['nn_sri_ci_width'].mean():.4f}")
    print()
    print(f"Logit-2SRI coverage : {df_res['logit_covers'].mean():.3f}  "
          f"(target 0.95)")
    print(f"Logit-2SRI beta mean: {df_res['logit_beta'].mean():.4f}  "
          f"bias={df_res['logit_beta'].mean()-TRUE_BETA:.4f}")
    print(f"Logit-2SRI CI width : {df_res['logit_ci_width'].mean():.4f}")
