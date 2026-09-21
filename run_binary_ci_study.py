"""
Local Binary Outcome CI Study — 3 DGPs × 200 reps × 210 NN fits = 126,000 fits total.

DGPs (same-population, n=20k):
  corr_err      : X = f(S) + eps_x,          Y ~ Bern(σ(β_x·X + ρ·eps_x))
  shared_u_high : X = f(S) + U + 1.0·eps_x,  Y ~ Bern(σ(β_x·X + β_u·U))
  shared_u_low  : X = f(S) + U + 0.1·eps_x,  Y ~ Bern(σ(β_x·X + β_u·U))

Per simulation:
  - logit_point      : 1 logit-2SRI estimate on full data
  - logit_bootstrap  : B=200 logit-2SRI estimates on bootstrap resamples
  - nn_ensemble      : M=10 NN-2SRI estimates on full data (different random inits)
  - nn_bootstrap     : B=200 NN-2SRI estimates on bootstrap resamples (1 fit each)

Output: binary_ci_results/{dgp}_rep{NNN}.json per simulation.

Usage:
  python run_binary_ci_study.py                  # 6 workers, all 200 reps
  python run_binary_ci_study.py --workers 4      # fewer workers
  python run_binary_ci_study.py --max-reps 10    # test run (10 reps)

Restart: re-run the same command — completed simulations are skipped,
         partial simulations resume from the last completed fit.

See CLAUDE.md §"Binary Outcome CI Study" for full restart instructions.
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

RESULT_DIR   = "binary_ci_results"
TRUE_BETA_X  = 2.0
TRUE_BETA_U  = 2.0
RHO_ERR      = 0.5    # correlation in corr_err DGP
K1 = K2      = 7
P01 = P02    = 0.4
RHO_SNP      = 0.5    # SNP block correlation
N            = 20000
M_ENS        = 10     # ensemble size
B_BOOT       = 100    # bootstrap resamples
EPOCHS       = 500
LR           = 0.01
DROPOUT      = 0.01
ES_MIN_DELTA = 1e-4   # early stopping: minimum improvement to reset patience counter
N_REPS       = 200

DGP_TYPES = ["corr_err", "shared_u_high", "shared_u_low"]
DGP_IDX   = {d: i for i, d in enumerate(DGP_TYPES)}
SCOLS     = [f"S1_{j+1}" for j in range(K1)] + [f"S2_{j+1}" for j in range(K2)]


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


def _fS(S1, S2, gj1, gj2, gjm12, gj1121):
    return (S1@gj1 + S2@gj2
            + np.sum(S1[:,:,None]*S2[:,None,:]*gjm12[None,:,:], axis=(1,2))
            + (S1[:,0]*S1[:,1])*(S2@gj1121))


def gen_data(dgp_type, n, seed, gammas):
    """Generate a DataFrame with SNP cols, X, Y. Seed is (rep*3 + dgp_idx)."""
    gj1, gj2, gjm12, gj1121 = gammas
    rng = np.random.RandomState(seed)
    S1  = _snp_block(n, K1, P01, seed * 1000 + 1)
    S2  = _snp_block(n, K2, P02, seed * 1000 + 2)
    fS  = _fS(S1, S2, gj1, gj2, gjm12, gj1121)

    if dgp_type == "corr_err":
        eps_x = rng.normal(0, 1, n)
        X     = fS + eps_x
        prob  = 1 / (1 + np.exp(-(TRUE_BETA_X * X + RHO_ERR * eps_x)))
    else:
        sigma_x = 1.0 if dgp_type == "shared_u_high" else 0.1
        U       = rng.normal(0, 1, n)
        eps_x   = rng.normal(0, sigma_x, n)
        X       = fS + eps_x + U
        prob    = 1 / (1 + np.exp(-(TRUE_BETA_X * X + TRUE_BETA_U * U)))

    Y    = rng.binomial(1, np.clip(prob, 1e-9, 1-1e-9)).astype(float)
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
    # Import here so workers can find it after spawn
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

def _save(path, data):
    """Atomic JSON save (write to .tmp then rename)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def _load_or_init(path, dgp_type, rep):
    """Load existing checkpoint or create fresh one."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        "dgp":    dgp_type,
        "rep":    rep,
        "config": {
            "true_beta_x": TRUE_BETA_X,
            "true_beta_u": TRUE_BETA_U if dgp_type != "corr_err" else None,
            "rho_err":     RHO_ERR     if dgp_type == "corr_err"  else None,
            "sigma_x":     (1.0 if dgp_type == "shared_u_high" else 0.1)
                           if dgp_type != "corr_err" else None,
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
    """
    Run one full simulation: logit point + B logit bootstraps +
    M NN ensemble fits + B NN bootstrap fits. Saves after every fit.
    Skips if already complete; resumes if partial.
    """
    import os, sys
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    # Suppress worker stdout (epoch prints) to keep the main log clean
    sys.stdout = open(os.devnull, "w")
    import warnings; warnings.filterwarnings("ignore")
    import torch
    torch.set_num_threads(1)

    dgp_type, rep, result_dir, gammas = args
    filepath = os.path.join(result_dir, f"{dgp_type}_rep{rep:03d}.json")

    data = _load_or_init(filepath, dgp_type, rep)
    if _is_complete(data):
        return f"SKIP  {dgp_type} rep {rep:03d}"

    # Deterministic data seed: each (dgp, rep) gets a unique seed
    sim_seed = rep * len(DGP_TYPES) + DGP_IDX[dgp_type]
    df       = gen_data(dgp_type, N, sim_seed, gammas)

    # 1. Logit point estimate ─────────────────────────────────────────────────
    if data["logit_point"] is None:
        data["logit_point"] = logit_2sri(df, df)
        _save(filepath, data)

    # 2. Logit bootstrap ──────────────────────────────────────────────────────
    while len(data["logit_bootstrap"]) < B_BOOT:
        b_idx = len(data["logit_bootstrap"])
        rng   = np.random.RandomState(sim_seed * 10_000 + b_idx)
        idx   = rng.choice(N, N, replace=True)
        df_b  = df.iloc[idx].reset_index(drop=True)
        data["logit_bootstrap"].append(logit_2sri(df_b, df_b))
        _save(filepath, data)

    # 3. NN ensemble ──────────────────────────────────────────────────────────
    while len(data["nn_ensemble"]) < M_ENS:
        data["nn_ensemble"].append(nn_2sri(df, df))
        _save(filepath, data)

    # 4. NN bootstrap ─────────────────────────────────────────────────────────
    while len(data["nn_bootstrap"]) < B_BOOT:
        b_idx = len(data["nn_bootstrap"])
        rng   = np.random.RandomState(sim_seed * 10_000 + 50_000 + b_idx)
        idx   = rng.choice(N, N, replace=True)
        df_b  = df.iloc[idx].reset_index(drop=True)
        data["nn_bootstrap"].append(nn_2sri(df_b, df_b))
        _save(filepath, data)

    data["status"] = "complete"
    _save(filepath, data)
    return f"DONE  {dgp_type} rep {rep:03d}"


# ── Progress summary ──────────────────────────────────────────────────────────

def print_progress(result_dir, n_reps):
    counts = {d: {"complete": 0, "partial": 0, "pending": 0} for d in DGP_TYPES}
    for dgp in DGP_TYPES:
        for rep in range(n_reps):
            path = os.path.join(result_dir, f"{dgp}_rep{rep:03d}.json")
            if not os.path.exists(path):
                counts[dgp]["pending"] += 1
            else:
                with open(path) as f:
                    d = json.load(f)
                if _is_complete(d):
                    counts[dgp]["complete"] += 1
                else:
                    counts[dgp]["partial"] += 1
    print(f"\nProgress ({n_reps} reps per DGP):")
    for dgp, c in counts.items():
        pct = 100 * c["complete"] / n_reps
        print(f"  {dgp:<20}: {c['complete']:3d} done, {c['partial']:2d} partial, "
              f"{c['pending']:3d} pending  ({pct:.0f}%)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Binary outcome CI study (local).")
    parser.add_argument("--workers",  type=int, default=6,
                        help="Number of parallel workers (default 6; leave 2 free).")
    parser.add_argument("--max-reps", type=int, default=N_REPS,
                        help="Max reps per DGP (default 200; use 5 for a test run).")
    parser.add_argument("--progress", action="store_true",
                        help="Print progress summary and exit.")
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)

    if args.progress:
        print_progress(RESULT_DIR, args.max_reps)
        return

    n_reps = args.max_reps

    # Build job list — interleaved (rep0 of all DGPs, then rep1, …)
    # so partial runs have balanced coverage across DGP types.
    jobs = []
    for rep in range(n_reps):
        for dgp in DGP_TYPES:
            path = os.path.join(RESULT_DIR, f"{dgp}_rep{rep:03d}.json")
            if os.path.exists(path):
                with open(path) as f:
                    d = json.load(f)
                if _is_complete(d):
                    continue   # skip at submission time
            jobs.append((dgp, rep, RESULT_DIR, GAMMAS))

    n_total   = n_reps * len(DGP_TYPES)
    n_pending = len(jobs)
    n_done    = n_total - n_pending

    print(f"\nBinary CI study — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  DGPs     : {DGP_TYPES}")
    print(f"  Reps     : {n_reps}  ×  {len(DGP_TYPES)} DGPs = {n_total} simulations")
    print(f"  Per sim  : M={M_ENS} ensemble + B={B_BOOT} bootstrap = {M_ENS+B_BOOT} NN fits")
    print(f"  Workers  : {args.workers}")
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

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as executor:
        futures = {executor.submit(run_one_sim, job): job for job in jobs}
        for fut in as_completed(futures):
            try:
                result = fut.result()
            except Exception as e:
                job = futures[fut]
                print(f"  ERROR  {job[0]} rep {job[1]:03d}: {e}", flush=True)
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
