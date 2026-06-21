"""
Same-population binary outcome — Deep IV-style correlated-error DGP.
Endogeneity enters only through eps_x (correlated errors), NOT a shared U.
  X  = f(S) + eps_x
  Y  ~ Bernoulli(sigma(beta_x*X + rho_end*eps_x))
  ê  = X - X̂ ≈ eps_x  →  perfect proxy for the endogeneity source

This matches the Deep IV binary formulation. Both logit-2SRI and NN
should recover beta_x = 2.0 in same-pop (confirmed by oracle).

Run: python -B test_samepop_binary.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, LogisticRegression
from deeppliv.core.trainer import DeepPLIV

# ── fixed DGP constants ──────────────────────────────────────────────────────
np.random.seed(0)

K1 = K2 = 7
p01 = p02 = 0.4      # same for both cohorts
rho = 0.5
beta_x  = 2.0
rho_end = 0.5   # endogeneity strength (correlated-error, like Deep IV binary)
EPOCHS = 500
LR = 0.01
DROPOUT = 0.01
M = 5

gamma_j1    = np.random.normal(0, 0.3, size=K1)
gamma_j2    = np.random.normal(0, 0.3, size=K2)
gamma_jm12  = np.random.normal(0, 0.1, size=(K1, K2))
gamma_j1121 = np.random.normal(0, 0.1, size=K2)

# ── DGP ─────────────────────────────────────────────────────────────────────

def simulate_snp_block(n, K, p0):
    Z = np.random.normal(size=n)
    snps = np.zeros((n, K))
    for j in range(K):
        eps = np.random.normal(size=n)
        p = norm.cdf(norm.ppf(p0) + np.sqrt(rho) * Z + np.sqrt(1 - rho) * eps)
        snps[:, j] = np.random.binomial(2, p)
    return snps

def gen_data(n, seed, p01_val=None):
    """Deep IV-style binary DGP: endogeneity via correlated errors only.
    X = f(S) + eps_x
    Y ~ Bernoulli(sigma(beta_x*X + rho_end*eps_x))
    ê ≈ eps_x → perfect proxy for the endogeneity source.
    """
    if p01_val is None:
        p01_val = p01
    np.random.seed(seed)
    S1 = simulate_snp_block(n, K1, p01_val)
    S2 = simulate_snp_block(n, K2, p02)
    interaction = np.sum(S1[:,:,None]*S2[:,None,:]*gamma_jm12[None,:,:], axis=(1,2))
    f_S   = S1@gamma_j1 + S2@gamma_j2 + interaction + (S1[:,0]*S1[:,1])*(S2@gamma_j1121)
    eps_x = np.random.normal(0, 1, n)
    X     = f_S + eps_x
    prob  = 1.0 / (1.0 + np.exp(-(beta_x * X + rho_end * eps_x)))
    Y     = np.random.binomial(1, prob).astype(float)
    cols  = {f"S1_{j+1}": S1[:,j] for j in range(K1)}
    cols.update({f"S2_{j+1}": S2[:,j] for j in range(K2)})
    import pandas as pd
    df = pd.DataFrame({**cols, "X": X, "Y": Y})
    df["eps_x"] = eps_x   # stored for oracle check
    return df

def snp_cols():
    return [f"S1_{j+1}" for j in range(K1)] + [f"S2_{j+1}" for j in range(K2)]

def run_naive(df_y):
    m = LogisticRegression(penalty=None).fit(df_y[["X"]].values, df_y["Y"].values)
    return float(m.coef_.ravel()[0])

def run_oracle(df_y):
    """True eps_x as control — ceiling: logit-2SRI with perfect first stage."""
    m = LogisticRegression(penalty=None).fit(
        np.column_stack([df_y["X"].values, df_y["eps_x"].values]), df_y["Y"].values)
    return float(m.coef_.ravel()[0])

def run_logit_2sri(df_x, df_y):
    cols = snp_cols()
    lr   = LinearRegression().fit(df_x[cols].values, df_x["X"].values)
    e    = df_y["X"].values - lr.predict(df_y[cols].values)
    des  = np.column_stack([df_y["X"].values, e])
    m    = LogisticRegression(penalty=None).fit(des, df_y["Y"].values)
    return float(m.coef_.ravel()[0])

def run_nn_2sri(df_x, df_y):
    cols = snp_cols()
    Z_X, Z_Y = df_x[cols].values, df_y[cols].values
    X, X_Y, Y = df_x["X"].values, df_y["X"].values, df_y["Y"].values
    model = DeepPLIV()
    model.fit_first_stage(Z_X, X,
        epochs_first_stage=EPOCHS, learning_rate_first_stage=LR, dropout=DROPOUT,
        validation_data=(Z_Y, X_Y))
    e   = X_Y.reshape(-1,1) - model.predict_first_stage(Z_Y).reshape(-1,1)
    ss  = model.fit_second_stage(
        v_hat=X_Y.reshape(-1,1), x=e, y=Y.reshape(-1,1),
        epochs_second_stage=EPOCHS, learning_rate_second_stage=LR, dropout=DROPOUT,
        method="2sri")
    return float(ss.final_layer.weight[0,0].item())

# ── Sweep ────────────────────────────────────────────────────────────────────

N = 20000

p01y = 0.05  # Y-cohort allele frequency (cross-pop, very different from p01=0.4)

N_VALUES = [5000, 10000, 20000, 40000]

print(f"\n{'='*85}")
print(f"Cross-pop binary  |  Correlated-error DGP (Deep IV style)")
print(f"X = f(S)+eps_x,  Y~Bern(sigma(beta_x*X + rho_end*eps_x)),  rho_end={rho_end}")
print(f"FS cohort p01={p01},  SS cohort p01y={p01y},  M={M} NN fits,  true beta_x={beta_x}")
print(f"{'='*85}")
print(f"{'n':>8}  {'naive':>8}  {'logit2sri':>10}  {'nn2sri':>10}  {'oracle':>8}  {'lin_bias':>9}  {'nn_bias':>9}")
print("-" * 85)

for N in N_VALUES:
    df_x = gen_data(N, seed=10, p01_val=p01)
    df_y = gen_data(N, seed=20, p01_val=p01y)

    naive  = run_naive(df_y)
    oracle = run_oracle(df_y)
    l2sri  = run_logit_2sri(df_x, df_y)

    nn_betas = []
    for m in range(M):
        try:
            nn_betas.append(run_nn_2sri(df_x, df_y))
        except Exception as ex:
            print(f"  [!] nn fit {m} failed: {ex}")
    nn_mean = float(np.mean(nn_betas)) if nn_betas else float('nan')

    print(f"{N:>8}  {naive:>8.4f}  {l2sri:>10.4f}  {nn_mean:>10.4f}  {oracle:>8.4f}"
          f"  {l2sri-beta_x:>+9.4f}  {nn_mean-beta_x:>+9.4f}", flush=True)

print("=" * 85)
print()
print("Expected (same-pop, binary, non-collapsibility theory):")
print("  naive:      large positive bias (confounding)")
print("  logit-2SRI: partially corrects; residual non-collapsibility bias persists")
print("  nn-2SRI:    similar residual bias to logit-2SRI")
print("  oracle:     ~2.0 (ceiling — true U used directly)")
print("  All biases should be FLAT across n (structural, not statistical)")
