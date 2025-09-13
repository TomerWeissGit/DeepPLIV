import os
import pickle

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

from core.trainer import DeepPLIV


# ---------------------
# Simulation Components
# ---------------------
def simulate_snp_block(n, K, p0, rho):
    Z = np.random.normal(size=n)
    snps = np.zeros((n, K))
    for j in range(K):
        eps = np.random.normal(size=n)
        p = norm.cdf(norm.ppf(p0) + np.sqrt(rho) * Z + np.sqrt(1 - rho) * eps)
        snps[:, j] = np.random.binomial(2, p)
    return snps


def simulate_snp_dataset(n, p01, p02):
    S1 = simulate_snp_block(n, K1, p01, rho)
    S2 = simulate_snp_block(n, K2, p02, rho)
    return S1, S2


def x_eq(S1, S2, n):
    interaction_term = np.sum(S1[:, :, None] * S2[:, None, :] * gamma_jm12[None, :, :], axis=(1, 2))
    non_linear_part = (S1 @ gamma_j1 + S2 @ gamma_j2 + interaction_term +
                       ((S1[:, 0] * S1[:, 1]) * (S2 @ gamma_j1121)))
    epsilon_x = np.random.normal(0, 1.0, size=n)
    u = np.random.normal(0, 1.0, size=n)
    return non_linear_part + epsilon_x + gamma_u * u, epsilon_x, u


def gen_data(n, p01, p02):
    S1, S2 = simulate_snp_dataset(n, p01, p02)
    X, eps_x, u = x_eq(S1, S2, n)
    epsilon_y = np.random.normal(0, 1, size=n)
    Y = beta_x * X + beta_u * u + epsilon_y
    df = pd.DataFrame({
        **{f"S1_{j + 1}": S1[:, j] for j in range(K1)},
        **{f"S2_{j + 1}": S2[:, j] for j in range(K2)},
        "X": X,
        "Y": Y
    })
    return df


# -----------------
# Estimation Models
# -----------------
def run_naive_ols(df_y):
    X = sm.add_constant(df_y[["X"]])
    y = df_y["Y"]
    return sm.OLS(y, X).fit().params.iloc[1]


def run_2sps(df_x, df_y):
    Z = df_x[[f"S1_{j + 1}" for j in range(K1)] + [f"S2_{j + 1}" for j in range(K2)]]
    Z_Y = df_y[[f"S1_{j + 1}" for j in range(K1)] + [f"S2_{j + 1}" for j in range(K2)]]
    X_hat = LinearRegression().fit(Z, df_x["X"]).predict(Z_Y)
    return sm.OLS(df_y["Y"], sm.add_constant(X_hat)).fit().params.iloc[1]


def run_2sri(df_x, df_y):
    Z = df_x[[f"S1_{j + 1}" for j in range(K1)] + [f"S2_{j + 1}" for j in range(K2)]]
    Z_Y = df_y[[f"S1_{j + 1}" for j in range(K1)] + [f"S2_{j + 1}" for j in range(K2)]]
    X_hat = LinearRegression().fit(Z, df_x["X"]).predict(Z_Y)
    X_err = df_y["X"] - X_hat
    X_stk = sm.add_constant(np.column_stack((df_y["X"], X_err)))
    return sm.OLS(df_y["Y"], X_stk).fit().params.iloc[1]


def estimating_sri_sps_with_nn(df_x, df_y, epochs, lr, dropout):
    model = DeepPLIV()

    Z_X = df_x[[f"S1_{j + 1}" for j in range(K1)] + [f"S2_{j + 1}" for j in range(K2)]].values
    Z_Y = df_y[[f"S1_{j + 1}" for j in range(K1)] + [f"S2_{j + 1}" for j in range(K2)]].values

    X = df_x["X"].values
    X_Y = df_y["X"].values
    Y = df_y["Y"].values
    dummy1 = np.ones((X_Y.shape[0], 1))

    model.fit_first_stage(Z_X, X, epochs_first_stage=epochs,
                          learning_rate_first_stage=lr,
                          dropout=dropout,
                          validation_data=(Z_Y, X_Y))

    x_pred = model.first_stage_model.predict(Z_Y).reshape(-1, 1)
    x_err = X_Y.reshape(-1, 1) - x_pred

    m_sri = model.fit_second_stage(X_Y.reshape(-1, 1), x_err, Y.reshape(-1, 1),
                                   epochs_second_stage=epochs, learning_rate_second_stage=lr,
                                   dropout=dropout)

    m_sps = model.fit_second_stage(x_pred, dummy1, Y.reshape(-1, 1),
                                   epochs_second_stage=epochs, learning_rate_second_stage=lr,
                                   dropout=dropout)

    m_naive_feed_forward = model.fit_second_stage(X_Y.reshape(-1, 1), dummy1, Y.reshape(-1, 1),
                                                  epochs_second_stage=epochs, learning_rate_second_stage=lr,
                                                  dropout=dropout)

    return (m_sps.final_layer.weight.detach().numpy()[0, 0],
            m_sri.final_layer.weight.detach().numpy()[0, 0],
            m_naive_feed_forward.final_layer.weight.detach().numpy()[0, 0])


# --------------------------
# Helpers for NEW features
# --------------------------
def ensemble_nn(df_x, df_y, M, **nn_kwargs):
    sps_list, sri_list, naive_feed_forward_list = [], [], []
    for _ in range(M):
        sps, sri, naive_feed_forward = estimating_sri_sps_with_nn(df_x, df_y, **nn_kwargs)
        sps_list.append(sps)
        sri_list.append(sri)
        naive_feed_forward_list.append(naive_feed_forward)
    return np.mean(sps_list), np.mean(sri_list), np.mean(naive_feed_forward_list)


def bootstrap_ci(base_func, df_x, df_y, B):
    if B == 1:
        est = base_func(df_x, df_y)
        return est, (est, est)

    estimates = []
    n_x, n_y = len(df_x), len(df_y)
    for _ in range(B):
        idx_x = np.random.choice(n_x, n_x, replace=True)
        idx_y = np.random.choice(n_y, n_y, replace=True)
        estimates.append(base_func(df_x.iloc[idx_x], df_y.iloc[idx_y]))
    ci_low, ci_high = np.percentile(estimates, CI_LEVEL)
    return np.mean(estimates), (ci_low, ci_high)


# -----------------
# Run One Simulation
# -----------------
def run_single_sim():
    df_x = gen_data(n_X, p01, p02)
    df_y = gen_data(n_Y, p01_y, p02_y)

    # ----- NN (ensemble) -----
    sps, sri_nn, naive_feed_forward = ensemble_nn(df_x, df_y,
                                                  M=ENSEMBLE_SIZE,
                                                  epochs=1000, lr=LR, dropout=DROPOUT)

    # ----- Linear models -----
    ols = run_naive_ols(df_y)

    sls_point, (sls_low, sls_high) = bootstrap_ci(run_2sps, df_x, df_y, BOOTSTRAPS)
    sri_point, (sri_low, sri_high) = bootstrap_ci(run_2sri, df_x, df_y, BOOTSTRAPS)

    return {
        "naive_ols": ols,
        "iv_2sps": sls_point,
        # "iv_2sps_ci_lo":  sls_low,
        # "iv_2sps_ci_hi":  sls_high,
        "iv_2sri": sri_point,
        # "iv_2sri_ci_lo":  sri_low,
        # "iv_2sri_ci_hi":  sri_high,
        "nn-2sps": sps,
        "nn-2sri": sri_nn,
        "naive_feed_forward": naive_feed_forward
    }


# -------------
# Run All Sims
# -------------
if __name__ == "__main__":
    os.makedirs("simulation_interaction", exist_ok=True)
    # -------------------------
    # Global knobs (NEW)
    # -------------------------
    ENSEMBLE_SIZE = 10  # M – NN repetitions on the SAME data
    BOOTSTRAPS = 1  # B – bootstrap resamples per simulation
    CI_LEVEL = (2.5, 97.5)

    # -------------------
    # Simulation Settings
    # -------------------
    n_snps_total = 200_000
    n_top_snps = 500
    K1 = K2 = 7
    p01 = p02 = 0.4

    NUM_SIMULATIONS = 100
    N_JOBS = 5
    beta_x = 2
    from itertools import product

    # All parameter combinations
    N_VALUES = [10000, 40000]
    RHO_VALUES = [0.5]
    P01Y_VALUES = [0.2, 0.4]
    GAMMA_U_VALUES = [1]
    BETA_U_VALUES = [0, 2]
    DROPOUT = 0.01
    LR = 0.01
    all_configs = list(product(N_VALUES, RHO_VALUES, P01Y_VALUES, GAMMA_U_VALUES, BETA_U_VALUES))

    print(f"Total configurations to run: {len(all_configs)}")

    for i, (n_val, rho_val, p01y_val, gamma_u_val, beta_u_val) in enumerate(all_configs):
        print(f"\nRunning configuration {i + 1}/{len(all_configs)}")
        print(
            f"n_X = n_Y = {n_val}, rho = {rho_val}, p01_y = p02_y = {p01y_val}, gamma_u = {gamma_u_val}, beta_u = {beta_u_val}")

        # Set parameters
        n_X = n_Y = n_val
        rho = rho_val
        p01_y = p02_y = p01y_val
        gamma_u = gamma_u_val
        beta_u = beta_u_val
        # Adjust effects path (so you reuse where possible, or regenerate when needed)
        effects_path = f"fixed_effects.pkl"

        # # Re-run effects loading
        with open(effects_path, "rb") as f:
            effect_dict = pickle.load(f)
        gamma_j1 = effect_dict["gamma_j1"]
        gamma_j2 = effect_dict["gamma_j2"]
        gamma_jm12 = effect_dict["gamma_jm12"]
        gamma_j1121 = effect_dict["gamma_j1121"]

        # Directory and filename
        folder = "simulation_interaction_grid"
        os.makedirs(folder, exist_ok=True)
        filename = (
            f"{folder}/num_sim{NUM_SIMULATIONS}_n{n_X}_rho{rho}_p01{p01}_p01y{p01_y}_gamma_u{gamma_u}_beta_u{beta_u}_ensemble_size{ENSEMBLE_SIZE}.csv"
        )

        results = Parallel(n_jobs=N_JOBS)(
            delayed(run_single_sim)() for _ in range(NUM_SIMULATIONS)
        )
        df_results = pd.DataFrame(results)
        df_results.to_csv(filename, index=False)
        print(f"Saved results to {filename}")
