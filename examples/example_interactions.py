import numpy as np
from scipy.stats import norm
import pandas as pd
import os
import pickle

# Settings
n_samples = 10000
n_snps_total = 200_000
n_top_snps = 500
K1 = K2 = 7
rho = 0.5  # Can also try 0.1 or 0.9
p01 = p02 = 0.3
n_X = 5000  # Can vary over [1000, 5000, 10000, 20000]

effects_path = "fixed_effects.pkl"

# Step 1: Load or generate true SNP effects and estimated effects
if os.path.exists(effects_path):
    with open(effects_path, "rb") as f:
        effect_dict = pickle.load(f)
    gamma_j1 = effect_dict["gamma_j1"]
    gamma_j2 = effect_dict["gamma_j2"]
    gamma_jm12 = effect_dict["gamma_jm12"]
    gamma_j1121 = effect_dict["gamma_j1121"]
else:
    np.random.seed(42)
    alpha = np.random.normal(0, np.sqrt(0.005), n_snps_total)
    est_alpha = np.random.normal(alpha, 1/np.sqrt(n_X))
    top_indices = np.argsort(np.abs(est_alpha))[-n_top_snps:]
    gamma_pool = est_alpha[top_indices]
    selected_effects = np.random.choice(gamma_pool, size=70, replace=False)

    gamma_j1 = selected_effects[0:K1]
    gamma_j2 = selected_effects[K1:K1+K2]
    gamma_jm12 = selected_effects[K1+K2:K1+K2+K1*K2].reshape(K1, K2)
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

# Simulate U, SNPs, X, Y
U = np.random.normal(size=n_samples)
S1 = simulate_snp_block(n_samples, K1, p01, rho)
S2 = simulate_snp_block(n_samples, K2, p02, rho)

# Compute X
X = (
    S1 @ gamma_j1 +
    S2 @ gamma_j2 +
    np.sum(S1[:, :, None] * S2[:, None, :] * gamma_jm12[None, :, :], axis=(1, 2)) +
    (S1[:, 0] * S1[:, 1]) * (S2 @ gamma_j1121) +
    U +
    np.random.normal(size=n_samples)
)

# Compute Y
Y = X + U + np.random.normal(size=n_samples)

# Create DataFrame
columns = [f"S1_{j+1}" for j in range(K1)] + [f"S2_{j+1}" for j in range(K2)]
data = pd.DataFrame(np.hstack([S1, S2]), columns=columns)
data["X"] = X
data["Y"] = Y
data["U"] = U

data.head()
if __name__ == "__main__":
    print("Generated dataset with shape:", data.shape)
    print("Columns:", data.columns.tolist())
    print(data.head())
    print(data.shape)
    # Save to CSV if needed
    # data.to_csv("simulated_data.csv", index=False)
