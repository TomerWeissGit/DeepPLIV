import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import concurrent.futures

from core.trainer import DeepPLIV

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("GeneticIVSim")

# Global interaction weights (to be initialized once)
interaction_weights = None


def generate_marker_snp(n, maf=0.3):
    return np.random.binomial(1, maf, size=n)


def generate_causal_snp_with_target_r2(marker_snp, target_r2):
    n = len(marker_snp)
    rho = np.sqrt(target_r2)
    z1 = norm.ppf((np.argsort(np.argsort(marker_snp)) + 1) / (n + 1))
    z2 = rho * z1 + np.sqrt(1 - rho ** 2) * np.random.normal(size=n)
    return (z2 > np.median(z2)).astype(int)


def generate_interactions(snp_matrix):
    n, m = snp_matrix.shape
    interaction_indices = [(i, j) for i in range(m) for j in range(i+1, m)]
    interactions = np.column_stack([snp_matrix[:, i] * snp_matrix[:, j] for i, j in interaction_indices])
    return interactions, len(interaction_indices)


def simulate_disease(snp_matrix, u, weights, noise_sd=0.1):
    interactions, _ = generate_interactions(snp_matrix)
    nonlinear = np.sin(interactions @ weights)  # or np.tanh, np.exp, etc.
    return nonlinear + u + np.random.normal(0, noise_sd, size=snp_matrix.shape[0])


def simulate_life_expectancy(disease, u, w1, w2, baseline_life=80, treatment_effect=-5,
                             confounder_effect=3, gamma1=2.5, gamma2=-4, noise_sd=3.0):
    return (baseline_life + treatment_effect * disease + confounder_effect * u +
            gamma1 * np.sin(3 * np.pi * w1) + gamma2 * w2 + np.random.normal(0, noise_sd, size=len(disease)))


def simulate_dataset(n=10000, m=10, maf=0.3, r2=0.8, noise_sd=1.0):
    global interaction_weights

    u = np.random.normal(0, 1, size=n)
    marker_snps = np.column_stack([generate_marker_snp(n, maf) for _ in range(m)])
    causal_snps = np.column_stack([generate_causal_snp_with_target_r2(marker_snps[:, i], r2) for i in range(m)])

    interactions, total_interactions = generate_interactions(causal_snps)

    if interaction_weights is None:
        interaction_weights = np.random.normal(0, 1, total_interactions)

    disease = simulate_disease(causal_snps, u, interaction_weights, noise_sd)

    w1 = np.random.uniform(0, 1, size=n)
    w2 = np.random.binomial(1, 0.3, size=n)

    y = simulate_life_expectancy(disease, u, w1, w2, noise_sd=noise_sd)

    return pd.DataFrame({
        **{f"s{i+1}_marker": marker_snps[:, i] for i in range(m)},
        **{f"s{i+1}_star": causal_snps[:, i] for i in range(m)},
        "u": u,
        "w1": w1,
        "w2": w2,
        "disease": disease,
        "life_expectancy": y
    })


def run_naive_ols(df):
    X = sm.add_constant(np.column_stack((df["disease"], np.sin(3 * np.pi * df["w1"]), df["w2"])))
    y = df["life_expectancy"]
    return sm.OLS(y, X).fit().params[1]


def run_2sls(df, m=10):
    Z = df[[f"s{i+1}_star" for i in range(m)]]
    stage1 = LinearRegression().fit(Z, df["disease"])
    df["disease_hat"] = stage1.predict(Z)
    X = sm.add_constant(np.column_stack((df["disease_hat"], df["w1"], df["w2"])))
    y = df["life_expectancy"]
    return sm.OLS(y, X).fit().params[1]


def run_2sls_with_proxies(df, m=10):
    Z = df[[f"s{i+1}_marker" for i in range(m)]]
    stage1 = LinearRegression().fit(Z, df["disease"])
    df["disease_hat"] = stage1.predict(Z)
    X = sm.add_constant(np.column_stack((df["disease_hat"], df["w1"], df["w2"])))
    y = df["life_expectancy"]
    return sm.OLS(y, X).fit().params[1]


def run_nn_estimates(df, m=10, epochs=5000, learning_rate=0.01, dropout=0.05):
    model = DeepPLIV()
    z = df[[f"s{i+1}_marker" for i in range(m)]].values
    z_split = int(len(z) // 2)
    z1, z2 = z[:z_split], z[z_split:]
    x1, x2 = df["disease"].values[:z_split], df["disease"].values[z_split:]
    y = df["life_expectancy"].values[z_split:]
    w1 = df["w1"].values[z_split:]
    w2 = df["w2"].values[z_split:]
    w_obs = np.column_stack((w1, w2))
    scaler = StandardScaler()
    z1 = scaler.fit_transform(z1)
    z2 = scaler.transform(z2)

    model.fit_first_stage(x1, z1, epochs_first_stage=epochs,
                          learning_rate_first_stage=learning_rate,
                          dropout=dropout, validation_data=(z2, x2))
    x_pred = model.first_stage_model.predict(z2).reshape(-1, 1)
    x_err = x2.reshape(-1, 1) - x_pred

    sri_model = model.fit_second_stage(x2.reshape(-1, 1),
                                       np.concatenate((x_err, w_obs), axis=1),
                                       y.reshape(-1, 1),
                                       epochs_second_stage=epochs,
                                       learning_rate_second_stage=learning_rate,
                                       dropout=dropout)

    sps_model = model.fit_second_stage(x_pred,
                                       w_obs,
                                       y.reshape(-1, 1),
                                       epochs_second_stage=epochs,
                                       learning_rate_second_stage=learning_rate,
                                       dropout=dropout)

    sri_coef = sri_model.final_layer.weight.detach().numpy()[0, 0]
    sps_coef = sps_model.final_layer.weight.detach().numpy()[0, 0]
    return sri_coef, sps_coef


def single_simulation_run(n=5000):
    df = simulate_dataset(n=n)
    return (
        run_naive_ols(df),
        run_2sls_with_proxies(df),
        run_2sls(df),
        *run_nn_estimates(df)
    )


def run_multiple_simulations_with_proxies_parallel(n_sim=5, n=20000, max_workers=5):
    ols_estimates = []
    proxy_iv_estimates = []
    true_iv_estimates = []
    sri_nn_estimates = []
    sps_nn_estimates = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(single_simulation_run, n=n) for _ in range(n_sim)]
        for fut in tqdm(concurrent.futures.as_completed(futures), total=n_sim):
            try:
                ols, proxy_iv, true_iv, sri, sps = fut.result()
                ols_estimates.append(ols)
                proxy_iv_estimates.append(proxy_iv)
                true_iv_estimates.append(true_iv)
                sri_nn_estimates.append(sri)
                sps_nn_estimates.append(sps)
            except Exception as e:
                logger.warning(f"Simulation failed: {e}")

    return ols_estimates, proxy_iv_estimates, true_iv_estimates, sri_nn_estimates, sps_nn_estimates


if __name__ == '__main__':
    ols, proxy_iv, true_iv, sri_nn, sps_nn = run_multiple_simulations_with_proxies_parallel()
    print("Naive OLS:", np.mean(ols))
    print("2SLS with proxies:", np.mean(proxy_iv))
    print("2SLS with true SNPs:", np.mean(true_iv))
    print("SRI with NN:", np.mean(sri_nn))
    print("SPS with NN:", np.mean(sps_nn))
    print("OLS std:", np.std(ols))
    print("2SLS with proxies std:", np.std(proxy_iv))
    print("2SLS with true SNPs std:", np.std(true_iv))
    print("SRI with NN std:", np.std(sri_nn))
    print("SPS with NN std:", np.std(sps_nn))