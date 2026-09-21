import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Set filters (None = no filter)
# ---------------------------
n_filter = 40000               # or None
rho_filter = 0.5               # or None
ensemble_size_filter = 10
# n_sim_filter = 20 # or None

# ---------------------------
# Load and Parse Simulation Data
# ---------------------------
folder = "simulation_interaction_grid/results"
files = [f for f in os.listdir(folder) if f.endswith(".csv")]

import re

def parse_filename(filename):
    match = re.match(
        r"num_sim(\d+)_n(\d+)_rho([\d.]+)_p01([\d.]+)_p01y([\d.]+)_gamma_u([\d.]+)_beta_u([\d.]+)_ensemble_size(\d+)\.csv",
        filename
    )
    if match:
        return {
            "filename": filename,
            "num_sim": int(match.group(1)),
            "n": int(match.group(2)),
            "rho": float(match.group(3)),
            "p01": float(match.group(4)),
            "p01y": float(match.group(5)),
            "gamma_u": float(match.group(6)),
            "beta_u": float(match.group(7)),
            "ensemble_size": int(match.group(8)),
        }
    return None

# Filter files by parameters
filtered_files = []
for f in files:
    info = parse_filename(f)
    if info is None:
        continue
    if (n_filter is None or info["n"] == n_filter) and \
       (rho_filter is None or info["rho"] == rho_filter) and \
       (ensemble_size_filter is None or info["ensemble_size"] == ensemble_size_filter):
        filtered_files.append(info)

if not filtered_files:
    raise ValueError("No files matched the specified filters.")

# Load data
df_all = []
for info in filtered_files:
    df = pd.read_csv(os.path.join(folder, info["filename"]))
    for key, val in info.items():
        if key != "filename":
            df[key] = val
    df_all.append(df)

df_all = pd.concat(df_all, ignore_index=True)

# ---------------------------
# Annotate Conditions and Melt
# ---------------------------
df_all["has_confounding"] = df_all["beta_u"] > 0
df_all["diff_snp_distribution"] = df_all["p01y"] != 0.4  # p01 fixed at 0.2

model_order = ["naive_ols", "iv_2sps", "iv_2sri", "nn-2sps", "nn-2sri", "naive_feed_forward"]

df_melted = df_all.melt(
    id_vars=["has_confounding", "diff_snp_distribution"],
    value_vars=model_order,
    var_name="model",
    value_name="estimate"
)
df_melted["model"] = pd.Categorical(df_melted["model"], categories=model_order, ordered=True)

# ---------------------------
# Helper for Mean and 95% CI
# ---------------------------
def compute_summary_stats(data):
    grouped = data.groupby("model")["estimate"]
    summary = grouped.agg(
        mean="mean",
        ci_low=lambda x: x.quantile(0.025),
        ci_high=lambda x: x.quantile(0.975)
    ).reset_index()
    return summary

# ---------------------------
# Plot Grid of Results
# ---------------------------
sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
actual_effect = 2

# ---------------------------
# Custom Model Order and Label Colors
# ---------------------------
df_melted["model"] = df_melted["model"].replace("naive_feed_forward", "naive_nn")

# Update model order and neural network label list
model_order = [
    "naive_ols",
    "naive_nn",
    "iv_2sps",
    "iv_2sri",
    "nn-2sps",
    "nn-2sri"
]

nn_models = {"nn-2sps", "nn-2sri"}

df_melted["model"] = pd.Categorical(df_melted["model"], categories=model_order, ordered=True)

# ---------------------------
# Plot Grid of Results
# ---------------------------
sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
actual_effect = 2

for i, diff_snp in enumerate([False, True]):
    for j, confounding in enumerate([False, True]):
        ax = axes[j, i]
        subset = df_melted[
            (df_melted["diff_snp_distribution"] == diff_snp) &
            (df_melted["has_confounding"] == confounding)
        ]

        sns.boxplot(
            data=subset,
            x="model",
            y="estimate",
            ax=ax,
            order=model_order,
            color="skyblue"
        )

        # Overlay mean ± 95% CI
        summary = compute_summary_stats(subset)
        summary = summary.set_index("model").loc[model_order].reset_index()
        x_positions = np.arange(len(model_order))

        ax.errorbar(
            x=x_positions,
            y=summary["mean"],
            yerr=[summary["mean"] - summary["ci_low"], summary["ci_high"] - summary["mean"]],
            fmt='o',
            color='black',
            capsize=5,
            elinewidth=2,
            markeredgewidth=2,
            label="Mean ± 95% CI"
        )

        # Red dashed line = true causal effect
        ax.axhline(actual_effect, ls="--", color="red", label="True Effect")

        # Titles and labels
        ax.set_title(f"{'With' if confounding else 'No'} Confounding, "
                     f"{'Different' if diff_snp else 'Same'} SNP Distribution")
        ax.set_ylabel("Estimated Effect")
        ax.set_xlabel("")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(model_order, rotation=45)

        # Color specific x-axis labels (nn models in green)
        for tick_label in ax.get_xticklabels():
            if tick_label.get_text() in nn_models:
                tick_label.set_color("green")

        ax.legend()

plt.tight_layout()
plt.suptitle("Causal Effect Estimates by Model Across Simulation Settings", fontsize=18, y=1.03)

# Save as PNG
plt.savefig("causal_estimates_grid.png", dpi=300, bbox_inches="tight")
plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# import pickle
#
# # Parameters
# n_snps_total = 200_000
# n_top_snps = 500
# K1 = K2 = 7
# save_path = "fixed_effects.pkl"
#
# # Step 1: Generate effects
# alpha = np.random.normal(0, np.sqrt(0.005), n_snps_total)
# top_indices = np.argsort(np.abs(alpha))[-n_top_snps:]
# gamma_pool = alpha[top_indices]
# selected = np.random.choice(gamma_pool, size=70, replace=False)
#
# gamma_j1 = selected[0:K1]
# gamma_j2 = selected[K1:K1 + K2]
# gamma_jm12 = selected[K1 + K2:K1 + K2 + K1 * K2].reshape(K1, K2)
# gamma_j1121 = selected[-K2:]
#
# # Save gamma coefficients to file
# with open(save_path, "wb") as f:
#     pickle.dump({
#         "gamma_j1": gamma_j1,
#         "gamma_j2": gamma_j2,
#         "gamma_jm12": gamma_jm12,
#         "gamma_j1121": gamma_j1121
#     }, f)
#
# # Combine all values into one array
# all_gammas = np.concatenate([
#     gamma_j1,
#     gamma_j2,
#     gamma_jm12.flatten(),
#     gamma_j1121
# ])
#
# # Step 2: Plot histograms
# fig, axes = plt.subplots(1, 2, figsize=(12, 4))
#
# # Histogram of raw values
# axes[0].hist(all_gammas, bins=30, color='steelblue', edgecolor='black')
# axes[0].set_title("Distribution of γ values")
# axes[0].set_xlabel("γ")
# axes[0].set_ylabel("Frequency")
#
# # Histogram of absolute values
# axes[1].hist(np.abs(all_gammas), bins=30, color='darkorange', edgecolor='black')
# axes[1].set_title("Distribution of |γ| values")
# axes[1].set_xlabel("|γ|")
# axes[1].set_ylabel("Frequency")
#
# plt.tight_layout()
# plt.savefig("gamma_distribution.png", dpi=300)
# plt.show()
