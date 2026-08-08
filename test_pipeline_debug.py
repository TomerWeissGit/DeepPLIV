"""
Diagnostic: trace exactly what enters each stage of 2SRI for binary outcome.
Single run at n=20000, beta_u=2, cross-pop (p01y=0.2 vs p01=0.4).
Prints first-stage quality, ê statistics, and what the second stage receives.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm, pearsonr
from sklearn.linear_model import LinearRegression
from deeppliv.core.trainer import DeepPLIV

np.random.seed(42)

K1 = K2 = 7
p01 = p02 = 0.4
p01y = 0.2
rho = 0.5
beta_x = 2.0
beta_u = 2.0
gamma_u = 1.0
EPOCHS = 500

np.random.seed(0)
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
        p = norm.cdf(norm.ppf(p0) + np.sqrt(rho)*Z + np.sqrt(1-rho)*eps)
        snps[:, j] = np.random.binomial(2, p)
    return snps

def gen_data(n, p01_val, seed):
    np.random.seed(seed)
    S1 = simulate_snp_block(n, K1, p01_val)
    S2 = simulate_snp_block(n, K2, p02)
    interaction = np.sum(S1[:,:,None]*S2[:,None,:]*gamma_jm12[None,:,:], axis=(1,2))
    f_S = S1@gamma_j1 + S2@gamma_j2 + interaction + (S1[:,0]*S1[:,1])*(S2@gamma_j1121)
    u   = np.random.normal(0, 1, n)
    eps_x = np.random.normal(0, 1, n)
    X = f_S + eps_x + gamma_u * u
    p = 1.0 / (1.0 + np.exp(-(beta_x * X + beta_u * u)))
    Y = np.random.binomial(1, p).astype(float)
    snp_cols = {f"S1_{j+1}": S1[:,j] for j in range(K1)}
    snp_cols.update({f"S2_{j+1}": S2[:,j] for j in range(K2)})
    df = pd.DataFrame({**snp_cols, "X": X, "Y": Y})
    df["U"] = u      # keep U for diagnostics
    df["f_S"] = f_S  # keep true instrument effect
    return df

n = 20000
df_x = gen_data(n, p01,  seed=1)
df_y = gen_data(n, p01y, seed=2)

cols = [f"S1_{j+1}" for j in range(K1)] + [f"S2_{j+1}" for j in range(K2)]
Z_X  = df_x[cols].values
Z_Y  = df_y[cols].values
X    = df_x["X"].values
X_Y  = df_y["X"].values
Y    = df_y["Y"].values
U_Y  = df_y["U"].values    # true U in Y-cohort (for diagnostics only)
fS_Y = df_y["f_S"].values  # true f(S) in Y-cohort

print("=" * 70)
print(f"PIPELINE DIAGNOSTIC  |  n={n}, beta_u={beta_u}, cross-pop (p01y={p01y})")
print("=" * 70)

# ── 1. Data summary ──────────────────────────────────────────────────────────
print("\n--- 1. DATA SUMMARY ---")
print(f"X-cohort  X: mean={X.mean():.3f}  std={X.std():.3f}")
print(f"Y-cohort  X: mean={X_Y.mean():.3f}  std={X_Y.std():.3f}")
print(f"Y-cohort  Y: mean={Y.mean():.3f}  (Y is binary)")
print(f"Y-cohort  U: mean={U_Y.mean():.3f}  std={U_Y.std():.3f}")
print(f"Corr(X_Y, U_Y)  = {pearsonr(X_Y, U_Y)[0]:.4f}  (endogeneity)")
print(f"True f(S_Y) std = {fS_Y.std():.3f}  (IV strength in Y-cohort)")

# ── 2. Linear first stage ────────────────────────────────────────────────────
print("\n--- 2. LINEAR FIRST STAGE ---")
print("Input:  Z_X → X  (trained on X-cohort)")
print("Output: X̂_Y = lr.predict(Z_Y)  (applied to Y-cohort)")
lr = LinearRegression().fit(Z_X, X)
xhat_lin = lr.predict(Z_Y)
e_lin = X_Y - xhat_lin
r2_lin = pearsonr(xhat_lin, X_Y)[0]**2
print(f"  R²(X̂_lin, X_Y)    = {r2_lin:.4f}")
print(f"  MSE on Y-cohort    = {np.mean((X_Y - xhat_lin)**2):.4f}")
print(f"  Corr(X̂_lin, U_Y)  = {pearsonr(xhat_lin, U_Y)[0]:.4f}  (should be ~0, IV validity)")
print(f"  Corr(ê_lin, U_Y)   = {pearsonr(e_lin, U_Y)[0]:.4f}  (should be >0, control power)")
print(f"  ê_lin: mean={e_lin.mean():.3f}  std={e_lin.std():.3f}")

# ── 3. NN first stage ────────────────────────────────────────────────────────
print("\n--- 3. NN FIRST STAGE ---")
print("Input:  Z_X → X  (trained on X-cohort, validated on Z_Y → X_Y)")
model = DeepPLIV()
model.fit_first_stage(Z_X, X,
    epochs_first_stage=EPOCHS, learning_rate_first_stage=0.01, dropout=0.01,
    validation_data=(Z_Y, X_Y))
xhat_nn = model.predict_first_stage(Z_Y).reshape(-1)
e_nn = X_Y - xhat_nn
r2_nn = pearsonr(xhat_nn, X_Y)[0]**2
print(f"  R²(X̂_nn, X_Y)     = {r2_nn:.4f}")
print(f"  MSE on Y-cohort    = {np.mean((X_Y - xhat_nn)**2):.4f}")
print(f"  Corr(X̂_nn, U_Y)   = {pearsonr(xhat_nn, U_Y)[0]:.4f}  (should be ~0, IV validity)")
print(f"  Corr(ê_nn, U_Y)    = {pearsonr(e_nn, U_Y)[0]:.4f}  (should be >0, control power)")
print(f"  ê_nn:  mean={e_nn.mean():.3f}  std={e_nn.std():.3f}")

# ── 4. Second stage inputs ───────────────────────────────────────────────────
print("\n--- 4. SECOND STAGE INPUTS ---")
print("Logit-2SRI design matrix: [const, X_Y, ê_lin]")
print("  X_Y  : observed treatment  (causal variable)")
print("  ê_lin: X_Y - X̂_lin        (control for endogeneity)")
print()
print("NN-2SRI:")
print("  v_hat = X_Y.reshape(-1,1)  → enters linear (final) layer")
print("  x     = ê_nn.reshape(-1,1) → enters deep network")
print("  y     = Y.reshape(-1,1)")
print(f"  Shapes: v_hat={X_Y.reshape(-1,1).shape}, x={e_nn.reshape(-1,1).shape}, y={Y.reshape(-1,1).shape}")

# ── 5. Logit-2SRI result ─────────────────────────────────────────────────────
print("\n--- 5. LOGIT-2SRI RESULT ---")
design = sm.add_constant(np.column_stack([X_Y, e_lin]))
res = sm.Logit(Y, design).fit(disp=0)
print(f"  coef on X_Y  = {res.params[1]:.4f}  (should be ~2.0)")
print(f"  coef on ê    = {res.params[2]:.4f}  (selection correction)")

# ── 6. NN-2SRI result ────────────────────────────────────────────────────────
print("\n--- 6. NN-2SRI RESULT ---")
ss = model.fit_second_stage(
    v_hat=X_Y.reshape(-1,1),
    x=e_nn.reshape(-1,1),
    y=Y.reshape(-1,1),
    epochs_second_stage=EPOCHS,
    learning_rate_second_stage=0.01,
    dropout=0.01,
    method="2sri"
)
beta_nn = float(ss.final_layer.weight[0, 0].item())
print(f"  final_layer.weight[0,0] = {beta_nn:.4f}  (should be ~2.0)")

# ── 7. Oracle: what if we had perfect ê (true U) ────────────────────────────
print("\n--- 7. ORACLE CHECK (using true U as control) ---")
design_oracle = sm.add_constant(np.column_stack([X_Y, U_Y]))
res_oracle = sm.Logit(Y, design_oracle).fit(disp=0)
print(f"  Oracle logit coef on X_Y (control=true U): {res_oracle.params[1]:.4f}")
print(f"  (This is the ceiling for 2SRI — best possible with perfect control)")

print("\n" + "="*70)
