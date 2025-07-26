from scipy.stats import norm
import pickle
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
# Default simulation run if file is executed directly
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
from core.trainer import DeepPLIV

# Settings
n_snps_total = 200_000
n_top_snps = 500
K1 = K2 = 7
rho = 0.9  # Can also try 0.1 or 0.9
p01 = p02 = 0.3
n_X = n_Y = 5000  # Can vary over [1000, 5000, 10000, 20000]
n_samples = n_X + n_Y  # Total number of samples for SNPs
# Separate sample sizes for exposure and outcome models

effects_path = f"fixed_effects_{n_X}_new.pkl"

# Step 1: Load or generate true SNP effects and estimated effects
# if os.path.exists(effects_path):
#     with open(effects_path, "rb") as f:
#         effect_dict = pickle.load(f)
#     gamma_j1 = effect_dict["gamma_j1"]
#     gamma_j2 = effect_dict["gamma_j2"]
#     gamma_jm12 = effect_dict["gamma_jm12"]
#     gamma_j1121 = effect_dict["gamma_j1121"]
# else:
np.random.seed(42)
alpha = np.random.normal(0, np.sqrt(0.1), n_snps_total)
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


# Step 2: Simulate SNPs for each gene with LD using latent probit model
def simulate_snp_block(n, K, p0, rho):
    Z = np.random.normal(size=n)  # shared latent variable per subject
    snps = np.zeros((n, K))
    for j in range(K):
        eps = np.random.normal(size=n)
        p = norm.cdf(norm.ppf(p0) + np.sqrt(rho) * Z + np.sqrt(1 - rho) * eps)
        snps[:, j] = np.random.binomial(2, p)
    return snps


# Simulate dataset for X or Y
def simulate_snp_dataset(n, noise_sd=1.0):
    U = np.random.normal(0, 1, size=n)
    S1 = simulate_snp_block(n, K1, p01, rho)
    S2 = simulate_snp_block(n, K2, p02, rho)
    return U, S1, S2


# IV Estimation methods
def run_naive_ols(df_y):
    X = sm.add_constant(df_y[["X"]])
    y = df_y["Y"]
    return sm.OLS(y, X).fit().params.iloc[1]


def run_2sls(df_x, df_y):
    Z = df_x[[f"S1_{j + 1}" for j in range(K1)] + [f"S2_{j + 1}" for j in range(K2)]]
    Z_Y = df_y[[f"S1_{j + 1}" for j in range(K1)] + [f"S2_{j + 1}" for j in range(K2)]]
    X_hat = LinearRegression().fit(Z, df_x["X"]).predict(Z_Y)
    X_stack = sm.add_constant(X_hat)
    y = df_y["Y"]
    return sm.OLS(y, X_stack).fit().params.iloc[1]


def run_2sri(df_x, df_y):
    # Instruments
    Z = df_x[[f"S1_{j + 1}" for j in range(K1)] + [f"S2_{j + 1}" for j in range(K2)]]
    Z_Y = df_y[[f"S1_{j + 1}" for j in range(K1)] + [f"S2_{j + 1}" for j in range(K2)]]

    # First-stage: Predict X from Z
    first_stage = LinearRegression().fit(Z, df_x["X"])
    X_hat = first_stage.predict(Z_Y)

    # Residuals (instruments for the endogeneity)
    X_err = df_y["X"] - X_hat

    # Second-stage: Y ~ X + residuals
    X_stack = sm.add_constant(np.column_stack((df_y["X"], X_err)))
    y = df_y["Y"]

    # Fit OLS
    model = sm.OLS(y, X_stack).fit()
    coef = model.params.iloc[1]  # coefficient on X

    return coef


def gen_data():
    U_X, S1_X, S2_X = simulate_snp_dataset(n_X)
    U_Y, S1_Y, S2_Y = simulate_snp_dataset(n_Y)

    X = (
            S1_X @ gamma_j1 +
            S2_X @ gamma_j2 +
            np.sum(S1_X[:, :, None] * S2_X[:, None, :] * gamma_jm12[None, :, :], axis=(1, 2)) +
            (S1_X[:, 0] * S1_X[:, 1]) * (S2_X @ gamma_j1121) +
            U_X +
            np.random.normal(0, 1.0, size=n_X)
    )
    X_Y = (
            S1_Y @ gamma_j1 +
            S2_Y @ gamma_j2 +
            np.sum(S1_X[:, :, None] * S2_Y[:, None, :] * gamma_jm12[None, :, :], axis=(1, 2)) +
            (S1_Y[:, 0] * S1_Y[:, 1]) * (S2_Y @ gamma_j1121) +
            U_Y +
            np.random.normal(0, 1.0, size=n_Y)
    )

    Y = X_Y - U_Y + np.random.normal(0, 1.0, size=n_Y)
    df_x = pd.DataFrame({
        **{f"S1_{j + 1}": S1_X[:, j] for j in range(K1)},
        **{f"S2_{j + 1}": S2_X[:, j] for j in range(K2)},
        "X": X,
        "U": U_X
    })
    df_y = pd.DataFrame({
        **{f"S1_{j + 1}": S1_Y[:, j] for j in range(K1)},
        **{f"S2_{j + 1}": S2_Y[:, j] for j in range(K2)},
        "X": X_Y,
        "Y": Y,
        "U": U_Y
    })
    return df_x, df_y


def estimating_sri_sps_with_nn(df_x, df_y, epochs=5000, learning_rate=0.01, dropout=0.01):
    model = DeepPLIV()
    Z_X = df_x[[f"S1_{j + 1}" for j in range(K1)] + [f"S2_{j + 1}" for j in range(K2)]].values
    Z_Y = df_y[[f"S1_{j + 1}" for j in range(K1)] + [f"S2_{j + 1}" for j in range(K2)]].values
    X = df_x["X"].values
    X_Y = df_y["X"].values
    Y = df_y["Y"].values

    n = len(df_x)
    dummy_v = np.ones((X_Y.shape[0], 1))

    scaler = StandardScaler()
    zx = scaler.fit_transform(Z_X)
    zy = scaler.transform(Z_Y)
    model.fit_first_stage(zx,
                          X,
                          epochs_first_stage=epochs,
                          learning_rate_first_stage=learning_rate,
                          dropout=dropout,
                          validation_data=(Z_Y, X_Y))
    x_pred = model.first_stage_model.predict(zy).reshape(-1, 1)
    x_err = X_Y.reshape(-1, 1) - x_pred
    x_exog_sri = StandardScaler().fit_transform(x_err)
    model_sri = model.fit_second_stage(X_Y.reshape(-1, 1), x_exog_sri, Y.reshape(-1, 1),
                                       epochs_second_stage=epochs,
                                       learning_rate_second_stage=learning_rate,
                                       dropout=dropout)
    sri_coef = model_sri.final_layer.weight.detach().numpy()[0, 0]
    model_sps = model.fit_second_stage(x_pred, dummy_v, Y.reshape(-1, 1),
                                       epochs_second_stage=epochs,
                                       learning_rate_second_stage=learning_rate,
                                       dropout=dropout)
    # Coefficient for the SPS model
    print(f'sps model coefficients:{model_sps.final_layer.weight}')
    sps_coef = model_sps.final_layer.weight.detach().numpy()[0, 0]

    model_nff = model.fit_second_stage(X_Y.reshape(-1, 1), dummy_v, Y.reshape(-1, 1),
                                       epochs_second_stage=epochs,
                                       learning_rate_second_stage=learning_rate,
                                       dropout=dropout)
    nff_coef = model_nff.final_layer.weight.detach().numpy()[0, 0]
    return sps_coef, sri_coef, nff_coef


def run_single_sim(seed=None):
    if seed is not None:
        np.random.seed(seed)
    df_x, df_y = gen_data()
    coefs = estimating_sri_sps_with_nn(df_x, df_y, epochs=5000, learning_rate=0.0001, dropout=0.01)
    return {
        "naive_ols": run_naive_ols(df_y),
        "iv_2sls": run_2sls(df_x, df_y),
        "iv_2sri": run_2sri(df_x, df_y),
        "nn-sps": coefs[0],
        "nn-sri": coefs[1],
        "nff": coefs[2]
    }


if __name__ == "__main__":
    NUM_SIMULATIONS = 5
    N_JOBS = 5  # Use all cores

    print("Running simulations in parallel...")
    results = Parallel(n_jobs=N_JOBS)(delayed(run_single_sim)(i) for i in range(NUM_SIMULATIONS))

    # Convert to DataFrame and save
    df_results = pd.DataFrame(results)
    df_results.to_csv("simulation_interaction/simulation_results_low_dropout_50.csv", index=False)
    print("Results saved to simulation_results.csv")

    # Summary
    print("\n--- Summary Statistics ---")
    for key in df_results.columns:
        values = df_results[key]
        mean = values.mean()
        std = values.std()
        ci = np.percentile(values, [2.5, 97.5])
        print(f"{key:<8} - Mean: {mean:.4f}, Std: {std:.4f}, CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

    # Boxplot
    df_melted = df_results.melt(var_name="Model", value_name="Coefficient")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_melted, x="Model", y="Coefficient")
    plt.title("Coefficient Distribution by Model")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("boxplot_hdo.png")
    plt.show()
