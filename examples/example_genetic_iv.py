import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

from core.trainer import DeepPLIV

# -------------------
# Simulation Settings
# -------------------
n_snps_total = 200_000
n_top_snps = 500
K1 = K2 = 7
rho = 0.9
p01 = p02 = 0.2
p01_y = p02_y = 0.3

n_X = n_Y = 5000
NUM_SIMULATIONS = 100
rho_y = 0.9
N_JOBS = 5
beta_x = 2
effects_path = f"fixed_effects_{n_X}_new.pkl"

# ---------------------x----------
# Step 1: Load or Generate Gammas
# -------------------------------
if os.path.exists(effects_path):
    with open(effects_path, "rb") as f:
        effect_dict = pickle.load(f)
    gamma_j1 = effect_dict["gamma_j1"]
    gamma_j2 = effect_dict["gamma_j2"]
    gamma_jm12 = effect_dict["gamma_jm12"]
    gamma_j1121 = effect_dict["gamma_j1121"]
else:
    alpha = np.random.normal(0, np.sqrt(0.005), n_snps_total)
    top_indices = np.argsort(np.abs(alpha))[-n_top_snps:]
    gamma_pool = alpha[top_indices]
    selected_effects = np.random.choice(gamma_pool, size=70, replace=False)

    gamma_j1 = selected_effects[0:K1]
    gamma_j2 = selected_effects[K1:K1 + K2]
    gamma_jm12 = selected_effects[K1 + K2:K1 + K2 + K1 * K2].reshape(K1, K2)
    gamma_j1121 = selected_effects[-K2:]

    with open(effects_path, "wb") as f:
        pickle.dump({
            "gamma_j1": gamma_j1,
            "gamma_j2": gamma_j2,
            "gamma_jm12": gamma_jm12,
            "gamma_j1121": gamma_j1121
        }, f)


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
    return non_linear_part + epsilon_x, epsilon_x


def gen_data(n, p01, p02):
    S1, S2 = simulate_snp_dataset(n, p01, p02)
    X, epsilon_x = x_eq(S1, S2, n)
    epsilon_y = np.array([np.random.normal(eps * rho_y, 1 - (rho_y ** 2)) for eps in epsilon_x])
    Y = beta_x * X + epsilon_y
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


def run_2sls(df_x, df_y):
    Z = df_x[[f"S1_{j + 1}" for j in range(K1)] + [f"S2_{j + 1}" for j in range(K2)]]
    Z_Y = df_y[[f"S1_{j + 1}" for j in range(K1)] + [f"S2_{j + 1}" for j in range(K2)]]
    X_hat = LinearRegression().fit(Z, df_x["X"]).predict(Z_Y)
    X_stack = sm.add_constant(X_hat)
    return sm.OLS(df_y["Y"], X_stack).fit().params.iloc[1]


def run_2sri(df_x, df_y):
    Z = df_x[[f"S1_{j + 1}" for j in range(K1)] + [f"S2_{j + 1}" for j in range(K2)]]
    Z_Y = df_y[[f"S1_{j + 1}" for j in range(K1)] + [f"S2_{j + 1}" for j in range(K2)]]
    X_hat = LinearRegression().fit(Z, df_x["X"]).predict(Z_Y)
    X_err = df_y["X"] - X_hat
    X_stack = sm.add_constant(np.column_stack((df_y["X"], X_err)))
    return sm.OLS(df_y["Y"], X_stack).fit().params.iloc[1]


def estimating_sri_sps_with_nn(df_x, df_y, epochs, learning_rate, dropout):
    model = DeepPLIV()
    Z_X = df_x[[f"S1_{j + 1}" for j in range(K1)] + [f"S2_{j + 1}" for j in range(K2)]].values
    Z_Y = df_y[[f"S1_{j + 1}" for j in range(K1)] + [f"S2_{j + 1}" for j in range(K2)]].values

    X = df_x["X"].values
    X_Y = df_y["X"].values
    Y = df_y["Y"].values
    dummy_v = np.ones((X_Y.shape[0], 1))

    model.fit_first_stage(Z_X, X, epochs_first_stage=epochs,
                          learning_rate_first_stage=learning_rate,
                          dropout=dropout,
                          validation_data=(Z_Y, X_Y))

    x_pred = model.first_stage_model.predict(Z_Y).reshape(-1, 1)
    x_err = X_Y.reshape(-1, 1) - x_pred

    model_sri = model.fit_second_stage(X_Y.reshape(-1, 1),
                                       x_err,
                                       Y.reshape(-1, 1),
                                       epochs_second_stage=epochs,
                                       learning_rate_second_stage=learning_rate,
                                       dropout=dropout)

    model_sps = model.fit_second_stage(x_pred, dummy_v, Y.reshape(-1, 1),
                                       epochs_second_stage=epochs,
                                       learning_rate_second_stage=learning_rate,
                                       dropout=dropout)

    model_nff = model.fit_second_stage(X_Y.reshape(-1, 1), dummy_v, Y.reshape(-1, 1),
                                       epochs_second_stage=epochs,
                                       learning_rate_second_stage=learning_rate,
                                       dropout=dropout)

    return (
        model_sps.final_layer.weight.detach().numpy()[0, 0],
        model_sri.final_layer.weight.detach().numpy()[0, 0],
        model_nff.final_layer.weight.detach().numpy()[0, 0]
    )


# -----------------
# Run One Simulation
# -----------------
def run_single_sim():
    df_x = gen_data(n_X, p01, p02)
    df_y = gen_data(n_Y, p01_y, p02_y)
    sps, sri, nff = estimating_sri_sps_with_nn(df_x, df_y, epochs=1000, learning_rate=0.01, dropout=0.1)
    return {
        "naive_ols": run_naive_ols(df_y),
        "iv_2sls": run_2sls(df_x, df_y),
        "iv_2sri": run_2sri(df_x, df_y),
        "nn-sps": sps,
        "nn-sri": sri,
        "nff": nff
    }


# -------------
# Run All Sims
# -------------
if __name__ == "__main__":
    print("Running simulations in parallel...")
    results = Parallel(n_jobs=N_JOBS)(
        delayed(run_single_sim)()
        for i in range(NUM_SIMULATIONS)
    )

    df_results = pd.DataFrame(results)
    df_results.to_csv("simulation_interaction/simulation_results_optional_G.csv", index=False)
    print("Results saved to simulation_results_optional_G.csv")

    print("\n--- Summary Statistics ---")
    for key in df_results.columns:
        values = df_results[key]
        mean = values.mean()
        std = values.std()
        ci = np.percentile(values, [2.5, 97.5])
        print(f"{key:<8} - Mean: {mean:.4f}, Std: {std:.4f}, CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

    df_melted = df_results.melt(var_name="Model", value_name="Coefficient")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_melted, x="Model", y="Coefficient")
    plt.title("Coefficient Distribution by Model")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("boxplot_optional_G.png")
    plt.show()
