"""
gamma_U Binary Outcome CI Study — 3 gamma_U values × 200 reps × 210 NN fits = 126,000 fits.

DGP (shared_u family, sigma_x=1 fixed, gamma_U varied):
  X = f(S) + gamma_u * U + eps_x,   eps_x ~ N(0,1),  U ~ N(0,1)
  Y ~ Bern(sigmoid(beta_x*X + beta_u*U))

gamma_U values: [0, 1, 3]
  gamma_U=0: ê = pure noise — both methods biased (~-0.62)
  gamma_U=1: ê has SNR=1 — moderate proxy, both biased
  gamma_U=3: ê dominated by U (SNR=9) — NN near-unbiased, logit retains bias

Per simulation:
  - logit_point      : 1 logit-2SRI estimate on full data
  - logit_bootstrap  : B=200 logit-2SRI estimates on bootstrap resamples
  - nn_ensemble      : M=10 NN-2SRI estimates on full data (different random inits)
  - nn_bootstrap     : B=200 NN-2SRI estimates on bootstrap resamples (1 fit each)

Output: gamma_u_ci_results/gamma{GU}_rep{NNN}.json per simulation.

Usage:
  python run_gamma_u_study.py                  # auto workers, all 200 reps
  python run_gamma_u_study.py --workers 32     # explicit worker count
  python run_gamma_u_study.py --max-reps 5     # test run (5 reps)
  python run_gamma_u_study.py --progress       # print progress and exit

Restart: re-run the same command — completed simulations are skipped,
         partial simulations resume from the last completed fit.
"""

import sys
import os
import json
import time
import argparse
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, LogisticRegression

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────

RESULT_DIR   = "gamma_u_ci_results"
TRUE_BETA_X  = 2.0
TRUE_BETA_U  = 2.0
SIGMA_X      = 1.0   # fixed noise on X
K1 = K2      = 10
P01 = P02    = 0.3
RHO_SNP      = 0.3
N            = 20000
M_ENS        = 10    # ensemble size
B_BOOT       = 200   # bootstrap resamples
EPOCHS       = 500
LR           = 0.01
DROPOUT      = 0.1
ES_MIN_DELTA = 1e-4
N_REPS       = 200

GAMMA_U_VALUES = [0, 1, 3]
GAMMA_U_IDX    = {g: i for i, g in enumerate(GAMMA_U_VALUES)}
SCOLS          = [f"S1_{j+1}" for j in range(K1)] + [f"S2_{j+1}" for j in range(K2)]


# ── DGP parameters (fixed global seed) ───────────────────────────────────────

def _make_gammas(seed=0):
    rng = np.random.RandomState(seed)
    return (rng.normal(0, 0.3, K1),
            rng.normal(0, 0.3, K2),
            rng.normal(0, 0.1, (K1, K2)),
            rng.normal(0, 0.1, K2))

GAMMAS = _make_gammas(seed=0)   # (gj1, gj2, gjm12, gj1121)


# ── Data generation ───────────────────────────────────────────────────────────

def _snp_block(n, K, p0, seed):
    rng = np.random.RandomState(seed)
    Z = rng.normal(size=n)
    s = np.zeros((n, K))
    for j in range(K):
        e = rng.normal(size=n)
        p = norm.cdf(norm.ppf(p0) + np.sqrt(RHO_SNP)*Z + np.sqrt(1-RHO_SNP)*e)
        s[:, j] = rng.binomial(2, p)
    return s


def gen_data(gamma_u, n, seed, gammas):
    """Generate DataFrame with SNP cols, X, Y for given gamma_u."""
    gj1, gj2, gjm12, gj1121 = gammas
    rng = np.random.RandomState(seed)
    S1  = _snp_block(n, K1, P01, seed * 1000 + 1)
    S2  = _snp_block(n, K2, P02, seed * 1000 + 2)
    fS  = (S1@gj1 + S2@gj2
           + np.sum(S1[:,:,None]*S2[:,None,:]*gjm12[None,:,:], axis=(1,2))
           + (S1[:,0]*S1[:,1])*(S2@gj1121))

    U     = rng.normal(0, 1, n)
    eps_x = rng.normal(0, SIGMA_X, n)
    X     = fS + gamma_u * U + eps_x
    prob  = 1 / (1 + np.exp(-(TRUE_BETA_X * X + TRUE_BETA_U * U)))
    Y     = rng.binomial(1, np.clip(prob, 1e-9, 1-1e-9)).astype(float)

    cols = {f"S1_{j+1}": S1[:,j] for j in range(K1)}
    cols.update({f"S2_{j+1}": S2[:,j] for j in range(K2)})
    return pd.DataFrame({**cols, "X": X, "Y": Y})


# ── Estimators ────────────────────────────────────────────────────────────────

def logit_2sri(df_train, df_eval):
    """Linear OLS first stage + sklearn LogisticRegression second stage (2SRI)."""
    lr  = LinearRegression().fit(df_train[SCOLS].values, df_train["X"].values)
    e   = df_eval["X"].values - lr.predict(df_eval[SCOLS].values)
    clf = LogisticRegression(penalty=None, max_iter=1000, solver="lbfgs")
    clf.fit(np.column_stack([df_eval["X"].values, e]), df_eval["Y"].values)
    return float(clf.coef_.ravel()[0])


def nn_2sri(df_train, df_eval):
    """NN first stage + DeepPLIV second stage (2SRI). Returns β̂_x."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from deeppliv.core.trainer import DeepPLIV

    model = DeepPLIV()
    model.fit_first_stage(
        df_train[SCOLS].values, df_train["X"].values,
        epochs_first_stage=EPOCHS,
        learning_rate_first_stage=LR,
        dropout=DROPOUT,
        validation_data=(df_eval[SCOLS].values, df_eval["X"].values),
        early_stopping_min_delta=ES_MIN_DELTA)

    e  = df_eval["X"].values - model.predict_first_stage(df_eval[SCOLS].values).reshape(-1)
    ss = model.fit_second_stage(
        v_hat=df_eval["X"].values.reshape(-1, 1),
        x=e.reshape(-1, 1),
        y=df_eval["Y"].values.reshape(-1, 1),
        epochs_second_stage=EPOCHS,
        learning_rate_second_stage=LR,
        dropout=DROPOUT,
        method="2sri",
        early_stopping_min_delta=ES_MIN_DELTA)
    return float(ss.final_layer.weight[0, 0].item())


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def _sim_key(gamma_u):
    return f"gamma{gamma_u}"

def _result_path(result_dir, gamma_u, rep):
    return os.path.join(result_dir, f"{_sim_key(gamma_u)}_rep{rep:03d}.json")

def _save(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def _load_or_init(path, gamma_u, rep):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        "gamma_u": gamma_u,
        "rep":     rep,
        "config": {
            "true_beta_x": TRUE_BETA_X,
            "true_beta_u": TRUE_BETA_U,
            "sigma_x":     SIGMA_X,
            "n": N, "m": M_ENS, "b": B_BOOT, "epochs": EPOCHS,
        },
        "logit_point":     None,
        "logit_bootstrap": [],
        "nn_ensemble":     [],
        "nn_bootstrap":    [],
        "status":          "partial",
    }


def _is_complete(data):
    return (data.get("status") == "complete"
            and data.get("logit_point") is not None
            and len(data.get("logit_bootstrap", [])) >= B_BOOT
            and len(data.get("nn_ensemble",     [])) >= M_ENS
            and len(data.get("nn_bootstrap",    [])) >= B_BOOT)


# ── Worker ────────────────────────────────────────────────────────────────────

def run_one_sim(args):
    """Run one full simulation. Saves after every fit. Resumes if partial."""
    import os, sys
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    sys.stdout = open(os.devnull, "w")
    import warnings; warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(1)

    gamma_u, rep, result_dir, gammas = args
    filepath = _result_path(result_dir, gamma_u, rep)

    data = _load_or_init(filepath, gamma_u, rep)
    if _is_complete(data):
        return f"SKIP  gamma_u={gamma_u} rep {rep:03d}"

    sim_seed = rep * len(GAMMA_U_VALUES) + GAMMA_U_IDX[gamma_u]
    df       = gen_data(gamma_u, N, sim_seed, gammas)

    # 1. Logit point estimate
    if data["logit_point"] is None:
        data["logit_point"] = logit_2sri(df, df)
        _save(filepath, data)

    # 2. Logit bootstrap
    while len(data["logit_bootstrap"]) < B_BOOT:
        b_idx = len(data["logit_bootstrap"])
        rng   = np.random.RandomState(sim_seed * 10_000 + b_idx)
        idx   = rng.choice(N, N, replace=True)
        df_b  = df.iloc[idx].reset_index(drop=True)
        data["logit_bootstrap"].append(logit_2sri(df_b, df_b))
        _save(filepath, data)

    # 3. NN ensemble
    while len(data["nn_ensemble"]) < M_ENS:
        data["nn_ensemble"].append(nn_2sri(df, df))
        _save(filepath, data)

    # 4. NN bootstrap
    while len(data["nn_bootstrap"]) < B_BOOT:
        b_idx = len(data["nn_bootstrap"])
        rng   = np.random.RandomState(sim_seed * 10_000 + 50_000 + b_idx)
        idx   = rng.choice(N, N, replace=True)
        df_b  = df.iloc[idx].reset_index(drop=True)
        data["nn_bootstrap"].append(nn_2sri(df_b, df_b))
        _save(filepath, data)

    data["status"] = "complete"
    _save(filepath, data)
    return f"DONE  gamma_u={gamma_u} rep {rep:03d}"


# ── Progress summary ──────────────────────────────────────────────────────────

def print_progress(result_dir, n_reps):
    print(f"\nProgress ({n_reps} reps per gamma_u):")
    for gu in GAMMA_U_VALUES:
        done, partial, pending = 0, 0, 0
        for rep in range(n_reps):
            path = _result_path(result_dir, gu, rep)
            if not os.path.exists(path):
                pending += 1
            else:
                with open(path) as f:
                    d = json.load(f)
                if _is_complete(d):
                    done += 1
                else:
                    partial += 1
        pct = 100 * done / n_reps
        print(f"  gamma_u={gu}: {done:3d} done, {partial:2d} partial, "
              f"{pending:3d} pending  ({pct:.0f}%)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="gamma_U binary outcome CI study.")
    parser.add_argument("--workers",  type=int, default=0,
                        help="Number of parallel workers (default: auto = cpu_count-2).")
    parser.add_argument("--max-reps", type=int, default=N_REPS,
                        help="Max reps per gamma_u (default 200; use 5 for a test run).")
    parser.add_argument("--progress", action="store_true",
                        help="Print progress summary and exit.")
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)

    if args.progress:
        print_progress(RESULT_DIR, args.max_reps or N_REPS)
        return

    n_reps   = args.max_reps
    n_cpus   = os.cpu_count() or 8
    n_workers = args.workers if args.workers > 0 else max(1, n_cpus - 2)

    # Build job list — interleaved so partial runs have balanced coverage
    jobs = []
    for rep in range(n_reps):
        for gu in GAMMA_U_VALUES:
            path = _result_path(RESULT_DIR, gu, rep)
            if os.path.exists(path):
                with open(path) as f:
                    d = json.load(f)
                if _is_complete(d):
                    continue
            jobs.append((gu, rep, RESULT_DIR, GAMMAS))

    n_total   = n_reps * len(GAMMA_U_VALUES)
    n_pending = len(jobs)
    n_done    = n_total - n_pending

    print(f"\ngamma_U CI study — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  gamma_U  : {GAMMA_U_VALUES}")
    print(f"  Reps     : {n_reps}  ×  {len(GAMMA_U_VALUES)} gamma_U = {n_total} simulations")
    print(f"  Per sim  : M={M_ENS} ensemble + B={B_BOOT} bootstrap = {M_ENS+B_BOOT} NN fits")
    print(f"  Workers  : {n_workers}  (cpu_count={n_cpus})")
    print(f"  Already done: {n_done} / {n_total}  —  {n_pending} pending")
    print(f"  Results  : {os.path.abspath(RESULT_DIR)}")

    if not jobs:
        print("\nAll simulations complete.")
        print_progress(RESULT_DIR, n_reps)
        return

    ctx = mp.get_context("spawn")
    t0  = time.time()
    n_completed = 0
    n_errors    = 0

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        futures = {executor.submit(run_one_sim, job): job for job in jobs}
        for fut in as_completed(futures):
            try:
                result = fut.result()
            except Exception as e:
                job = futures[fut]
                print(f"  ERROR  gamma_u={job[0]} rep {job[1]:03d}: {e}", flush=True)
                n_errors += 1
                continue
            n_completed += 1
            elapsed   = time.time() - t0
            rate      = n_completed / elapsed if elapsed > 0 else 0
            remaining = (n_pending - n_completed - n_errors) / rate if rate > 0 else float("inf")
            print(f"  [{n_completed:4d}/{n_pending}]  {result}"
                  f"   ETA {remaining/3600:.1f}h", flush=True)

    print(f"\nFinished {n_completed} simulations ({n_errors} errors) in {(time.time()-t0)/3600:.2f}h")
    print_progress(RESULT_DIR, n_reps)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
