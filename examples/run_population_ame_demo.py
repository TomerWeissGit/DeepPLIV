"""
Population AME + prediction RMSE + cross-population robustness demo for DeepPLIV.

Three-part analysis of the Hartford Deep IV DGP
------------------------------------------------
Part 1 --- Same-population AME.
    Confirms that the structural parameter beta_1 and the population Average
    Marginal Effect (AME) are two different estimands, and that linear IV
    methods can target either one by choosing whether to include the oracle
    interaction feature s * psi_t(t) * x_hat. DeepPLIV, which enforces only a
    scalar linear treatment effect, targets the AME; so do linear 2SPS/2SRI
    without the interaction feature, because the partial-linear (Robinson)
    projection is first-order robust to nuisance misspecification.

Part 2 --- Same-population test-set prediction RMSE.
    Even when linear 2SRI correctly recovers the AME scalar, its overall
    model of E[y | x, s, t] is misspecified and yields a worse fit to
    individual observations. We quantify this by computing the test-set RMSE
    of each estimator on a held-out sample drawn from the same population.
    Under wide (s, t) ranges where the psi_t nonlinearity is genuinely
    expressed, DeepPLIV delivers a meaningfully lower prediction error even
    though its AME estimate is statistically indistinguishable from linear
    2SRI's.

Part 3 --- Cross-population robustness.
    Two-sample instrumental variable analyses (classically: two-sample
    Mendelian randomization) fit the first stage on one population and the
    second stage on another. We simulate this by training the first stage on
    a wide training population and fitting the second stage on a narrow
    target population whose covariate marginals differ. A linear first stage
    averages psi_t across the wide range and is systematically wrong on the
    narrow target region; DeepPLIV's neural first stage learns psi_t and
    transfers. The test-set RMSE gap grows sharply in this setting, which is
    the regime that motivates flexible nuisance estimation in the MR
    simulations later in the thesis.

Usage
-----
    python examples/run_population_ame_demo.py                 # full run
    python examples/run_population_ame_demo.py --num-simulations 5   # quick
    python examples/run_population_ame_demo.py \
        --thesis-gfx-dir /path/to/thesis/gfx                   # also copy figures
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_deep_iv_simulation import DeepIVData, psi_t  # noqa: E402

from deeppliv import DeepPLIV  # noqa: E402


# ---------------------------------------------------------------------------
# DGP
# ---------------------------------------------------------------------------


def sample_population(
    n: int,
    rho: float,
    t_range: Sequence[float],
    s_range: Sequence[float],
    beta_1: float,
    rng: np.random.Generator,
) -> DeepIVData:
    """Draw a sample from the Hartford Deep IV DGP on continuous (t, s) ranges.

    Uses the thesis calibration (both structural blocks are standardized to
    unit variance before adding the linear ``beta_1 * x`` term and the noise).
    """
    n1 = n // 2
    n2 = n - n1

    z_1 = rng.standard_normal(n1)
    z_2 = rng.standard_normal(n2)
    t_1 = rng.uniform(*t_range, size=n1)
    t_2 = rng.uniform(*t_range, size=n2)
    s_2 = rng.uniform(*s_range, size=n2)

    eps_x_1 = rng.standard_normal(n1)
    eps_x_2 = rng.standard_normal(n2)
    x_1_raw = 25.0 + psi_t(t_1) * (z_1 + 3.0)
    x_2_raw = 25.0 + psi_t(t_2) * (z_2 + 3.0)
    scaler = StandardScaler()
    x_1 = scaler.fit_transform(x_1_raw.reshape(-1, 1))[:, 0] + eps_x_1
    x_2 = scaler.transform(x_2_raw.reshape(-1, 1))[:, 0] + eps_x_2

    eps_y_2 = rng.normal(
        loc=rho * eps_x_2,
        scale=np.sqrt(1.0 - rho ** 2),
        size=n2,
    )
    y_struct = 100.0 + (10.0 + x_2) * s_2 * psi_t(t_2)
    y_struct = StandardScaler().fit_transform(y_struct.reshape(-1, 1))[:, 0]
    y_2 = y_struct + beta_1 * x_2 + eps_y_2

    return DeepIVData(
        z_1=z_1, t_1=t_1, x_1=x_1,
        z_2=z_2, t_2=t_2, s_2=s_2, x_2=x_2, y_2=y_2,
    )


def compute_true_ame(
    t_range: Sequence[float],
    s_range: Sequence[float],
    beta_1: float,
    n: int = 1_500_000,
    K: int = 30,
    rng: np.random.Generator | None = None,
) -> float:
    """MC ground truth for the partial-linear projection (AME) on a K x K cell grid."""
    rng = rng or np.random.default_rng(0)
    data = sample_population(
        n=n, rho=0.0, t_range=t_range, s_range=s_range,
        beta_1=beta_1, rng=rng,
    )

    def bin_idx(vals: np.ndarray, lo: float, hi: float) -> np.ndarray:
        return np.clip(
            ((vals - lo) / (hi - lo + 1e-12) * K).astype(int),
            0, K - 1,
        )

    si = bin_idx(data.s_2, *s_range)
    ti = bin_idx(data.t_2, *t_range)
    key = si * K + ti
    cells = K * K
    ybar = np.zeros(cells)
    xbar = np.zeros(cells)
    cnt = np.zeros(cells)
    np.add.at(ybar, key, data.y_2)
    np.add.at(xbar, key, data.x_2)
    np.add.at(cnt, key, 1)
    mask = cnt > 0
    ybar[mask] /= cnt[mask]
    xbar[mask] /= cnt[mask]
    y_res = data.y_2 - ybar[key]
    x_res = data.x_2 - xbar[key]
    return float((x_res * y_res).sum() / (x_res * x_res).sum())


# ---------------------------------------------------------------------------
# First-stage helpers
# ---------------------------------------------------------------------------


def _naive_iv_features(z: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.column_stack([z, t])


def _oracle_iv_features(z: np.ndarray, t: np.ndarray) -> np.ndarray:
    """The true (standardized) first stage is
        x = const + a * psi_t(t) + b * psi_t(t) * z + eps_x,
    so the oracle analyst uses [z, psi_t(t), z * psi_t(t)] as first-stage features.
    """
    psi = psi_t(t)
    return np.column_stack([z, psi, z * psi])


def _fit_linear_first_stage(train: DeepIVData, *, oracle: bool) -> LinearRegression:
    feat = _oracle_iv_features if oracle else _naive_iv_features
    return LinearRegression().fit(feat(train.z_1, train.t_1), train.x_1)


def _predict_linear_first_stage(
    first: LinearRegression, data: DeepIVData, *, oracle: bool
) -> np.ndarray:
    feat = _oracle_iv_features if oracle else _naive_iv_features
    return first.predict(feat(data.z_2, data.t_2))


@dataclass
class NNFirstStage:
    """Package a fitted DeepPLIV first-stage network with its feature scaler."""

    model: DeepPLIV
    scaler: StandardScaler

    def predict(self, data: DeepIVData) -> np.ndarray:
        g_iv = self.scaler.transform(_naive_iv_features(data.z_2, data.t_2))
        return self.model.first_stage_model.predict(g_iv).reshape(-1)


def _fit_nn_first_stage(
    train: DeepIVData,
    epochs: int,
    lr: float,
    dropout: float,
) -> NNFirstStage:
    """Fit the DeepPLIV NN first stage on the training sample."""
    model = DeepPLIV()
    scaler = StandardScaler()
    g_iv_1 = scaler.fit_transform(_naive_iv_features(train.z_1, train.t_1))
    g_iv_2 = scaler.transform(_naive_iv_features(train.z_2, train.t_2))
    model.fit_first_stage(
        g_iv_1,
        train.x_1.reshape(-1, 1),
        epochs_first_stage=epochs,
        learning_rate_first_stage=lr,
        dropout=dropout,
        validation_data=(g_iv_2, train.x_2.reshape(-1, 1)),
    )
    return NNFirstStage(model=model, scaler=scaler)


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


def _rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def fit_eval_naive_ols(train: DeepIVData, test: DeepIVData) -> dict:
    X_tr = np.column_stack([train.x_2, train.s_2, train.t_2])
    model = LinearRegression().fit(X_tr, train.y_2)
    X_te = np.column_stack([test.x_2, test.s_2, test.t_2])
    y_pred = model.predict(X_te)
    return dict(y_pred=y_pred, ame=float(model.coef_[0]), rmse=_rmse(y_pred, test.y_2))


def fit_eval_linear_2sps_no_int(
    train: DeepIVData,
    test: DeepIVData,
    first: LinearRegression | None = None,
) -> dict:
    """Linear 2SPS with no interaction feature and a naive linear first stage."""
    if first is None:
        first = _fit_linear_first_stage(train, oracle=False)
    x_hat_tr = _predict_linear_first_stage(first, train, oracle=False)
    X_tr = np.column_stack([x_hat_tr, train.s_2, train.t_2])
    second = LinearRegression().fit(X_tr, train.y_2)

    x_hat_te = _predict_linear_first_stage(first, test, oracle=False)
    X_te = np.column_stack([x_hat_te, test.s_2, test.t_2])
    y_pred = second.predict(X_te)
    return dict(y_pred=y_pred, ame=float(second.coef_[0]), rmse=_rmse(y_pred, test.y_2))


def fit_eval_linear_2sri_no_int(
    train: DeepIVData,
    test: DeepIVData,
    first: LinearRegression | None = None,
) -> dict:
    """Linear 2SRI with no interaction feature and a naive linear first stage."""
    if first is None:
        first = _fit_linear_first_stage(train, oracle=False)
    x_hat_tr = _predict_linear_first_stage(first, train, oracle=False)
    residual_tr = train.x_2 - x_hat_tr
    X_tr = np.column_stack([train.x_2, train.s_2, train.t_2, residual_tr])
    second = LinearRegression().fit(X_tr, train.y_2)

    x_hat_te = _predict_linear_first_stage(first, test, oracle=False)
    residual_te = test.x_2 - x_hat_te
    X_te = np.column_stack([test.x_2, test.s_2, test.t_2, residual_te])
    y_pred = second.predict(X_te)
    return dict(y_pred=y_pred, ame=float(second.coef_[0]), rmse=_rmse(y_pred, test.y_2))


def fit_eval_linear_2sps_oracle(
    train: DeepIVData,
    test: DeepIVData,
    first: LinearRegression | None = None,
) -> dict:
    """Linear 2SPS with oracle psi_t in both stages."""
    if first is None:
        first = _fit_linear_first_stage(train, oracle=True)
    x_hat_tr = _predict_linear_first_stage(first, train, oracle=True)
    J_tr = train.s_2 * psi_t(train.t_2)
    X_tr = np.column_stack([x_hat_tr, J_tr, J_tr * x_hat_tr])
    second = LinearRegression().fit(X_tr, train.y_2)

    x_hat_te = _predict_linear_first_stage(first, test, oracle=True)
    J_te = test.s_2 * psi_t(test.t_2)
    X_te = np.column_stack([x_hat_te, J_te, J_te * x_hat_te])
    y_pred = second.predict(X_te)
    return dict(y_pred=y_pred, ame=float(second.coef_[0]), rmse=_rmse(y_pred, test.y_2))


def fit_eval_linear_2sri_oracle(
    train: DeepIVData,
    test: DeepIVData,
    first: LinearRegression | None = None,
) -> dict:
    """Linear 2SRI with oracle psi_t in both stages."""
    if first is None:
        first = _fit_linear_first_stage(train, oracle=True)
    x_hat_tr = _predict_linear_first_stage(first, train, oracle=True)
    residual_tr = train.x_2 - x_hat_tr
    J_tr = train.s_2 * psi_t(train.t_2)
    X_tr = np.column_stack([train.x_2, J_tr, J_tr * train.x_2, residual_tr])
    second = LinearRegression().fit(X_tr, train.y_2)

    x_hat_te = _predict_linear_first_stage(first, test, oracle=True)
    residual_te = test.x_2 - x_hat_te
    J_te = test.s_2 * psi_t(test.t_2)
    X_te = np.column_stack([test.x_2, J_te, J_te * test.x_2, residual_te])
    y_pred = second.predict(X_te)
    return dict(y_pred=y_pred, ame=float(second.coef_[0]), rmse=_rmse(y_pred, test.y_2))


def fit_eval_deeppliv(
    train: DeepIVData,
    test: DeepIVData,
    variant: str,
    epochs: int,
    lr: float,
    dropout: float,
    nn_first: NNFirstStage | None = None,
) -> dict:
    """Fit and evaluate a DeepPLIV estimator. `variant` in {'2sps', '2sri', 'naive'}.

    If `nn_first` is provided it is reused as the first stage (this is how the
    cross-population setup passes in a first stage trained on a different
    sample). Otherwise a fresh first stage is trained on `train`.
    """
    if nn_first is None:
        nn_first = _fit_nn_first_stage(train, epochs, lr, dropout)

    x_hat_tr = nn_first.predict(train)
    residual_tr = train.x_2 - x_hat_tr
    if variant == "2sps":
        linear_tr = x_hat_tr
        exog_tr = np.column_stack([train.s_2, train.t_2])
    elif variant == "2sri":
        linear_tr = train.x_2
        exog_tr = np.column_stack([train.s_2, train.t_2, residual_tr])
    elif variant == "naive":
        linear_tr = train.x_2
        exog_tr = np.column_stack([train.s_2, train.t_2])
    else:
        raise ValueError(f"Unknown DeepPLIV variant: {variant!r}")

    exog_scaler = StandardScaler()
    exog_tr_s = exog_scaler.fit_transform(exog_tr)

    second_nn = nn_first.model.fit_second_stage(
        linear_tr.reshape(-1, 1),
        exog_tr_s,
        train.y_2.reshape(-1, 1),
        epochs_second_stage=epochs,
        learning_rate_second_stage=lr,
        dropout=dropout,
    )

    x_hat_te = nn_first.predict(test)
    residual_te = test.x_2 - x_hat_te
    if variant == "2sps":
        linear_te = x_hat_te
        exog_te = np.column_stack([test.s_2, test.t_2])
    elif variant == "2sri":
        linear_te = test.x_2
        exog_te = np.column_stack([test.s_2, test.t_2, residual_te])
    else:  # naive
        linear_te = test.x_2
        exog_te = np.column_stack([test.s_2, test.t_2])
    exog_te_s = exog_scaler.transform(exog_te)

    second_nn.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(linear_te.reshape(-1, 1), dtype=torch.float32)
        e_tensor = torch.tensor(exog_te_s, dtype=torch.float32)
        y_pred = second_nn(e_tensor, x_tensor).numpy().reshape(-1)

    ame = float(second_nn.final_layer.weight.detach().cpu().numpy()[0, 0])
    return dict(y_pred=y_pred, ame=ame, rmse=_rmse(y_pred, test.y_2))


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


METHOD_ORDER = [
    "Naive OLS",
    "Naive NN",
    "Linear 2SPS",
    "Linear 2SRI",
    "Linear 2SPS (oracle)",
    "Linear 2SRI (oracle)",
    "DeepPLIV-2SPS",
    "DeepPLIV-2SRI",
]


def run_all_estimators(
    train: DeepIVData,
    test: DeepIVData,
    epochs: int,
    lr: float,
    dropout: float,
    *,
    linear_first_naive: LinearRegression | None = None,
    linear_first_oracle: LinearRegression | None = None,
    nn_first: NNFirstStage | None = None,
) -> dict:
    results: dict[str, dict] = {}
    results["Naive OLS"] = fit_eval_naive_ols(train, test)
    results["Linear 2SPS"] = fit_eval_linear_2sps_no_int(
        train, test, linear_first_naive
    )
    results["Linear 2SRI"] = fit_eval_linear_2sri_no_int(
        train, test, linear_first_naive
    )
    results["Linear 2SPS (oracle)"] = fit_eval_linear_2sps_oracle(
        train, test, linear_first_oracle
    )
    results["Linear 2SRI (oracle)"] = fit_eval_linear_2sri_oracle(
        train, test, linear_first_oracle
    )

    if nn_first is None:
        nn_first = _fit_nn_first_stage(train, epochs, lr, dropout)
    results["Naive NN"] = fit_eval_deeppliv(
        train, test, "naive", epochs, lr, dropout, nn_first
    )
    results["DeepPLIV-2SPS"] = fit_eval_deeppliv(
        train, test, "2sps", epochs, lr, dropout, nn_first
    )
    results["DeepPLIV-2SRI"] = fit_eval_deeppliv(
        train, test, "2sri", epochs, lr, dropout, nn_first
    )
    return results


def run_same_pop_sweep(
    pop_spec: dict,
    num_sims: int,
    n: int,
    rho: float,
    beta_1: float,
    epochs: int,
    lr: float,
    dropout: float,
    seed: int,
    label: str,
) -> pd.DataFrame:
    master_rng = np.random.default_rng(seed)
    records: list[dict] = []
    for sim in range(num_sims):
        train_rng = np.random.default_rng(master_rng.integers(0, 2**31 - 1))
        test_rng = np.random.default_rng(master_rng.integers(0, 2**31 - 1))
        train = sample_population(
            n=n, rho=rho, beta_1=beta_1, rng=train_rng, **pop_spec,
        )
        test = sample_population(
            n=n, rho=rho, beta_1=beta_1, rng=test_rng, **pop_spec,
        )
        results = run_all_estimators(train, test, epochs, lr, dropout)
        for method, res in results.items():
            records.append(
                dict(
                    sim=sim,
                    method=method,
                    ame=res["ame"],
                    rmse=res["rmse"],
                    population=label,
                    setting="same",
                )
            )
        print(
            f"  [{label}] sim {sim + 1}/{num_sims}  "
            + "  ".join(
                f"{m}:ame={r['ame']:+.3f},rmse={r['rmse']:.3f}"
                for m, r in results.items()
                if m in ("Linear 2SRI", "DeepPLIV-2SRI")
            )
        )
    return pd.DataFrame.from_records(records)


def run_cross_pop_sweep(
    pop_train_spec: dict,
    pop_target_spec: dict,
    num_sims: int,
    n: int,
    rho: float,
    beta_1: float,
    epochs: int,
    lr: float,
    dropout: float,
    seed: int,
    label: str,
) -> pd.DataFrame:
    """Train first stage on `pop_train_spec`, fit second stage on `pop_target_spec`,
    evaluate test RMSE on an independent sample from `pop_target_spec`.
    """
    master_rng = np.random.default_rng(seed)
    records: list[dict] = []
    for sim in range(num_sims):
        train_A_rng = np.random.default_rng(master_rng.integers(0, 2**31 - 1))
        train_B_rng = np.random.default_rng(master_rng.integers(0, 2**31 - 1))
        test_B_rng = np.random.default_rng(master_rng.integers(0, 2**31 - 1))
        train_A = sample_population(
            n=n, rho=rho, beta_1=beta_1, rng=train_A_rng, **pop_train_spec,
        )
        train_B = sample_population(
            n=n, rho=rho, beta_1=beta_1, rng=train_B_rng, **pop_target_spec,
        )
        test_B = sample_population(
            n=n, rho=rho, beta_1=beta_1, rng=test_B_rng, **pop_target_spec,
        )

        # Precompute first stages on Pop A's training data.
        linear_first_naive = _fit_linear_first_stage(train_A, oracle=False)
        linear_first_oracle = _fit_linear_first_stage(train_A, oracle=True)
        nn_first = _fit_nn_first_stage(train_A, epochs, lr, dropout)

        # All estimators now fit their second stage on train_B, but with
        # first stages precomputed on train_A.
        results = run_all_estimators(
            train_B, test_B, epochs, lr, dropout,
            linear_first_naive=linear_first_naive,
            linear_first_oracle=linear_first_oracle,
            nn_first=nn_first,
        )
        for method, res in results.items():
            records.append(
                dict(
                    sim=sim,
                    method=method,
                    ame=res["ame"],
                    rmse=res["rmse"],
                    population=label,
                    setting="cross",
                )
            )
        print(
            f"  [{label}] sim {sim + 1}/{num_sims}  "
            + "  ".join(
                f"{m}:ame={r['ame']:+.3f},rmse={r['rmse']:.3f}"
                for m, r in results.items()
                if m in ("Linear 2SRI", "DeepPLIV-2SRI")
            )
        )
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_ame_figure(
    df: pd.DataFrame,
    targets: dict[str, float],
    beta_1: float = -2.0,
    savepath: str | None = None,
) -> None:
    import seaborn as sns

    sns.set_style("whitegrid")
    pops = list(targets.keys())
    fig, axes = plt.subplots(
        1, len(pops), figsize=(9.0 * len(pops), 5.5), sharey=True,
    )
    if len(pops) == 1:
        axes = [axes]

    for ax, pop in zip(axes, pops):
        sub = df[df.population == pop]
        sns.boxplot(
            data=sub, x="method", y="ame", order=METHOD_ORDER,
            ax=ax, palette="Set2",
        )
        theta = targets[pop]
        ax.axhline(
            beta_1, color="grey", linestyle=":", linewidth=1.8,
            label=rf"Structural $\beta_1 = {beta_1:+.2f}$",
        )
        ax.axhline(
            theta, color="red", linestyle="--", linewidth=1.8,
            label=rf"Population AME $\theta^\star = {theta:+.3f}$",
        )
        ax.set_title(pop, fontsize=12)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)
        for lbl in ax.get_xticklabels():
            lbl.set_horizontalalignment("right")
        ax.legend(loc="best", fontsize=9, framealpha=0.9)
    axes[0].set_ylabel(r"$\hat{\beta}_1$", fontsize=13)
    fig.suptitle(
        "Same-population AME estimates: "
        r"Linear IV with oracle $\psi_t$ targets $\beta_1$; "
        r"linear IV without the interaction and DeepPLIV target $\theta^\star$",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
        print(f"  Saved AME figure to {savepath}")
    plt.close(fig)


def plot_rmse_figure(
    df_same_target: pd.DataFrame,
    df_cross: pd.DataFrame,
    savepath: str | None = None,
) -> None:
    """Two-panel RMSE comparison.

    Left panel: test RMSE when both stages are trained and evaluated on the
    target population (the baseline "same-population" case).
    Right panel: test RMSE when the first stage is trained on a different
    (wider) training population but the second stage is still fit and
    evaluated on the target population (two-sample cross-population setup).

    Only 2SRI variants and naive baselines are shown, because 2SPS plugs
    x_hat (noiseless predicted x) into the second stage and therefore has
    inflated predictive RMSE that is not comparable across stages.
    """
    import seaborn as sns

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True)

    same_filtered = df_same_target[df_same_target.method.isin(METHOD_ORDER_RMSE)]
    cross_filtered = df_cross[df_cross.method.isin(METHOD_ORDER_RMSE)]

    sns.boxplot(
        data=same_filtered, x="method", y="rmse", order=METHOD_ORDER_RMSE,
        ax=axes[0], palette="Set2",
    )
    axes[0].set_title(
        "Same-population: first and second stages trained on target population",
        fontsize=11,
    )
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Test RMSE", fontsize=13)
    axes[0].tick_params(axis="x", rotation=20)
    for lbl in axes[0].get_xticklabels():
        lbl.set_horizontalalignment("right")

    sns.boxplot(
        data=cross_filtered, x="method", y="rmse", order=METHOD_ORDER_RMSE,
        ax=axes[1], palette="Set2",
    )
    axes[1].set_title(
        "Cross-population: first stage trained on wider training population, "
        "second stage on target population",
        fontsize=11,
    )
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="x", rotation=20)
    for lbl in axes[1].get_xticklabels():
        lbl.set_horizontalalignment("right")

    fig.suptitle(
        "Test-set prediction RMSE on the target population: "
        "DeepPLIV-2SRI transfers across populations, linear 2SRI does not",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
        print(f"  Saved RMSE figure to {savepath}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


# Default populations. The AME figure uses two narrow populations (Pop A and
# Pop B) where the population AME differs enough to be visible; these are the
# pedagogical "structural vs AME" populations. The RMSE figure uses Pop Target
# (a moderately wide population where psi_t is genuinely nonlinear) as the
# evaluation population for both the same-population and the cross-population
# comparisons. For the cross-population comparison the first stage is trained
# on Pop Train (the full Hartford t range, so the NN first stage sees every
# psi_t value) and then applied to data drawn from Pop Target.
POP_A_AME = dict(t_range=(1.0, 3.0), s_range=(5.0, 7.0))
POP_B_AME = dict(t_range=(9.0, 10.0), s_range=(5.0, 7.0))
POP_TRAIN = dict(t_range=(0.0, 10.0), s_range=(1.0, 7.0))
POP_TARGET = dict(t_range=(4.0, 6.0), s_range=(5.0, 7.0))
LABEL_A_AME = "A  (t ~ U(1,3),  s ~ U(5,7))"
LABEL_B_AME = "B  (t ~ U(9,10), s ~ U(5,7))"
LABEL_TARGET = "Target (t ~ U(4,6), s ~ U(5,7))"
LABEL_CROSS = "Cross (FS on t~U(0,10), SS on target)"


# Methods shown in the RMSE figure. 2SPS variants are dropped because 2SPS is
# a causal-coefficient estimator that plugs x_hat (noiseless predicted x) into
# the second stage, so its predictive RMSE is dominated by the
# beta_1 * (x - x_hat) term and is not comparable to 2SRI or naive baselines
# that use the raw observed x in the outcome model. Reporting 2SPS in the AME
# figure is still fair because we care about the scalar coefficient there.
METHOD_ORDER_RMSE = [
    "Naive OLS",
    "Naive NN",
    "Linear 2SRI",
    "Linear 2SRI (oracle)",
    "DeepPLIV-2SRI",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--num-simulations", type=int, default=20)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--beta-1", type=float, default=-2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--output-dir",
        default="deep_iv_population_results",
        help="Where to save the DataFrame pickles and PNG figures.",
    )
    parser.add_argument(
        "--thesis-gfx-dir",
        default=None,
        help="Optional path to also copy the final figures (e.g. thesis gfx dir).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    epochs = int(1.5e7 / max(args.n // 2, 1))
    dropout = 100.0 / (1000.0 + args.n)

    # Compute ground-truth AMEs once.
    print("Computing Monte Carlo AME targets...")
    ame_A = compute_true_ame(**POP_A_AME, beta_1=args.beta_1)
    ame_B = compute_true_ame(**POP_B_AME, beta_1=args.beta_1)
    ame_target = compute_true_ame(**POP_TARGET, beta_1=args.beta_1)
    ame_train = compute_true_ame(**POP_TRAIN, beta_1=args.beta_1)
    print(f"  {LABEL_A_AME:40s}  theta* = {ame_A:+.4f}")
    print(f"  {LABEL_B_AME:40s}  theta* = {ame_B:+.4f}")
    print(f"  {LABEL_TARGET:40s}  theta* = {ame_target:+.4f}")
    print(f"  Train (t ~ U(0,10), s ~ U(1,7))            theta* = {ame_train:+.4f}")

    # ----- Sweep 1+2: same-population AME on Pop A and Pop B -----
    print(f"\n=== Same-population sweep: {LABEL_A_AME} (for AME figure) ===")
    df_same_A = run_same_pop_sweep(
        POP_A_AME, args.num_simulations, args.n, args.rho, args.beta_1,
        epochs, args.lr, dropout, args.seed, LABEL_A_AME,
    )
    print(f"\n=== Same-population sweep: {LABEL_B_AME} (for AME figure) ===")
    df_same_B = run_same_pop_sweep(
        POP_B_AME, args.num_simulations, args.n, args.rho, args.beta_1,
        epochs, args.lr, dropout, args.seed + 1, LABEL_B_AME,
    )
    df_ame = pd.concat([df_same_A, df_same_B], ignore_index=True)

    # ----- Sweep 3: same-population on Pop Target (baseline for RMSE figure) -----
    print(f"\n=== Same-population sweep: {LABEL_TARGET} (RMSE figure left panel) ===")
    df_same_target = run_same_pop_sweep(
        POP_TARGET, args.num_simulations, args.n, args.rho, args.beta_1,
        epochs, args.lr, dropout, args.seed + 2, LABEL_TARGET,
    )

    # ----- Sweep 4: cross-population (train on Pop Train, fit/eval on Pop Target) -----
    print(f"\n=== Cross-population sweep: {LABEL_CROSS} (RMSE figure right panel) ===")
    df_cross = run_cross_pop_sweep(
        POP_TRAIN, POP_TARGET,
        args.num_simulations, args.n, args.rho, args.beta_1,
        epochs, args.lr, dropout, args.seed + 3, LABEL_CROSS,
    )

    # Persist the DataFrames.
    tag = f"n{args.n}_k{args.num_simulations}"
    df_ame.to_pickle(os.path.join(args.output_dir, f"ame_sweep_{tag}.pkl"))
    df_same_target.to_pickle(os.path.join(args.output_dir, f"same_target_{tag}.pkl"))
    df_cross.to_pickle(os.path.join(args.output_dir, f"cross_pop_{tag}.pkl"))

    def _fmt(df: pd.DataFrame, group) -> pd.DataFrame:
        grp = df.groupby(group)[["ame", "rmse"]].agg(["mean", "std"])
        if isinstance(group, list):
            return grp.reindex(METHOD_ORDER, level="method")
        return grp.reindex(METHOD_ORDER)

    print("\n=== Part 1: Same-population AME estimates (Pop A vs Pop B) ===")
    print(_fmt(df_ame, ["population", "method"]).to_string())

    print(f"\n=== Part 2: Same-population test RMSE on {LABEL_TARGET} ===")
    print(_fmt(df_same_target, "method").to_string())

    print(f"\n=== Part 3: Cross-population test RMSE ({LABEL_CROSS}) ===")
    print(_fmt(df_cross, "method").to_string())

    # Figures.
    print("\nSaving figures...")
    targets = {LABEL_A_AME: ame_A, LABEL_B_AME: ame_B}
    ame_fig = os.path.join(args.output_dir, "deepiv_population_ame.png")
    rmse_fig = os.path.join(args.output_dir, "deepiv_population_rmse.png")
    plot_ame_figure(df_ame, targets, beta_1=args.beta_1, savepath=ame_fig)
    plot_rmse_figure(df_same_target, df_cross, savepath=rmse_fig)

    if args.thesis_gfx_dir is not None:
        import shutil
        for src in (ame_fig, rmse_fig):
            dst = os.path.join(args.thesis_gfx_dir, os.path.basename(src))
            shutil.copy(src, dst)
            print(f"  Copied {os.path.basename(src)} to {dst}")

    print("\nDone.")
