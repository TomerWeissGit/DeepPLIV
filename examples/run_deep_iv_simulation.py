"""
Deep IV simulation runner using the DeepPLIV package.

Reproduces the Hartford et al. (2017) "Deep IV" data-generating process and
compares six estimators for the linear-in-X coefficient beta_1 in

    x = 25 + z * psi_t + 3 * psi_t + eps_x
    y = 100 + (10 + x) * s * psi_t + beta_1 * x + eps_y

with

    psi_t    = (t - 5)^4 / 300 + 2 * exp(-4 * (t - 5)^2) + t / 5 - 4
    z        ~ N(0, 1)
    t        ~ Uniform on {1, ..., 10}           (integer; Hartford 2017)
    s        ~ Uniform on {1, ..., 7}             (integer; Hartford 2017)
    eps_x    ~ N(0, 1)
    eps_y    ~ N(rho * eps_x, 1 - rho^2)           (endogeneity controlled by rho)

The marginal/direct linear effect of x on y is beta_1 (default -2). The
interaction term (10 + x) * s * psi_t makes the conditional effect depend on
(s, t); following the thesis, we give the linear 2SPS/2SRI baselines the
advantage of including s * psi_t * x_hat as an explicit covariate in the
second stage.

Estimators
----------
1. Naive Regression        : OLS of y on [x, s, t, s * psi_t * x].
2. 2SPS (linear)           : OLS first stage (x ~ z, t), then OLS of y on
                             [x_hat, s, t, s * psi_t * x_hat].
3. 2SRI (linear)           : OLS first stage as above, residual e_hat = x - x_hat,
                             then OLS of y on [x, s, t, s * psi_t * x, e_hat].
4. Naive Feed Forward      : DeepPLIV second stage on raw x (no IV correction).
5. DeepPLIV-2SPS           : DeepPLIV first stage predicts x_hat from (z, t);
                             DeepPLIV partial-linear second stage uses x_hat
                             as the linear treatment and (s, t) as the
                             nonlinear covariates.
6. DeepPLIV-2SRI           : Same first stage, but the second stage keeps the
                             raw x as the linear treatment and adds the
                             first-stage residual to the nonlinear inputs.

Usage
-----
Smoke test (single cell):

    python examples/run_deep_iv_simulation.py

Full grid: edit the `_N_VALUES` / `_RHO_VALUES` lists at the bottom, or import
`run_deep_iv_cell` / `run_grid` from this module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from deeppliv import DeepPLIV


# ---------------------------------------------------------------------------
# Data generating process
# ---------------------------------------------------------------------------


def psi_t(t: np.ndarray) -> np.ndarray:
    """Hartford 2017 instrument-driven nonlinearity."""
    return (t - 5.0) ** 4 / 300.0 + 2.0 * np.exp(-4.0 * (t - 5.0) ** 2) + t / 5.0 - 4.0


@dataclass
class DeepIVData:
    # First-stage sample (used to fit x ~ instruments).
    z_1: np.ndarray  # (n1,)  instrument
    t_1: np.ndarray  # (n1,)  instrument (categorical 1..10)
    x_1: np.ndarray  # (n1,)  endogenous treatment

    # Second-stage sample (used to fit the outcome model).
    z_2: np.ndarray  # (n2,)
    t_2: np.ndarray  # (n2,)
    s_2: np.ndarray  # (n2,)  observed covariate 1..7
    x_2: np.ndarray  # (n2,)
    y_2: np.ndarray  # (n2,)


class DeepIVDGP:
    """Hartford 2017 Deep IV simulation (as calibrated for the thesis).

    The outcome equation as written in Hartford 2017 has a nonlinear structural
    block `(10 + x) * s * psi_t` whose standard deviation dwarfs the
    linear-in-x term `beta_1 * x` by roughly two orders of magnitude. An
    MSE-trained network has no incentive to recover beta_1 from a signal that
    is <1% of the outcome variance, so the linear head absorbs whatever helps
    fit the dominant nonlinear term. The thesis (and the original
    `SimDataCreatorDeepIV`) standardizes the nonlinear block to unit variance
    before adding `beta_1 * x + eps_y`, so the actual DGP behind the thesis
    figures is

        x = standardize(25 + z * psi_t + 3 * psi_t) + eps_x
        y = standardize(100 + (10 + x) * s * psi_t) + beta_1 * x + eps_y.

    This is the default. Set `standardize_nuisance=False` only to see the raw
    Hartford formulas (expect all neural estimators to return nonsense on the
    linear coefficient).

    Parameters
    ----------
    n : total sample size (split evenly into two halves).
    beta_1 : true direct linear effect of x on y. Default -2.
    rho : endogeneity strength in [0, 1). 0 means no confounding.
    rng : numpy Generator.
    standardize_nuisance : standardize the structural x and y blocks to unit
        variance before adding noise. Default True (matches thesis).
    """

    def __init__(
        self,
        n: int,
        beta_1: float = -2.0,
        rho: float = 0.5,
        rng: np.random.Generator | None = None,
        standardize_nuisance: bool = True,
        binary: bool = False,
    ):
        if not 0.0 <= rho < 1.0:
            raise ValueError(f"rho must lie in [0, 1); got {rho}.")
        self.n1 = n // 2
        self.n2 = n - self.n1
        self.beta_1 = float(beta_1)
        self.rho = float(rho)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.standardize_nuisance = bool(standardize_nuisance)
        self.binary = bool(binary)

    def sample(self) -> DeepIVData:
        rng = self.rng

        # Instruments and covariates.
        z_1 = rng.standard_normal(self.n1)
        z_2 = rng.standard_normal(self.n2)
        t_1 = rng.integers(low=1, high=11, size=self.n1).astype(float)  # {1,...,10}
        t_2 = rng.integers(low=1, high=11, size=self.n2).astype(float)
        s_2 = rng.integers(low=1, high=8, size=self.n2).astype(float)   # {1,...,7}

        # First-stage structural block: 25 + z * psi_t + 3 * psi_t.
        eps_x_1 = rng.standard_normal(self.n1)
        eps_x_2 = rng.standard_normal(self.n2)
        x_1_raw = 25.0 + psi_t(t_1) * (z_1 + 3.0)
        x_2_raw = 25.0 + psi_t(t_2) * (z_2 + 3.0)

        if self.standardize_nuisance:
            scaler = StandardScaler()
            x_1 = scaler.fit_transform(x_1_raw.reshape(-1, 1))[:, 0] + eps_x_1
            x_2 = scaler.transform(x_2_raw.reshape(-1, 1))[:, 0] + eps_x_2
        else:
            x_1 = x_1_raw + eps_x_1
            x_2 = x_2_raw + eps_x_2

        # Second-stage structural block: 100 + (10 + x) * s * psi_t.
        y_struct = 100.0 + (10.0 + x_2) * s_2 * psi_t(t_2)
        if self.standardize_nuisance:
            y_struct = StandardScaler().fit_transform(y_struct.reshape(-1, 1))[:, 0]

        if self.binary:
            latent = y_struct + self.beta_1 * x_2 + self.rho * eps_x_2
            prob = 1.0 / (1.0 + np.exp(-latent))
            y_2 = rng.binomial(1, prob).astype(float)
        else:
            eps_y_2 = rng.normal(
                loc=self.rho * eps_x_2,
                scale=np.sqrt(1.0 - self.rho ** 2),
                size=self.n2,
            )
            y_2 = y_struct + self.beta_1 * x_2 + eps_y_2

        return DeepIVData(
            z_1=z_1, t_1=t_1, x_1=x_1,
            z_2=z_2, t_2=t_2, s_2=s_2, x_2=x_2, y_2=y_2,
        )


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


def _iv_inputs(z: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Column-stack (z, t) as the instrument / first-stage input."""
    return np.column_stack([z, t])


def _linear_second_stage_features(
    x: np.ndarray, s: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Feature matrix for linear baselines.

    Columns: [x, s, t, s * psi_t(t) * x]. The interaction is given to the
    linear baselines as an explicit covariate (matches the thesis description).
    """
    interaction = s * psi_t(t) * x
    return np.column_stack([x, s, t, interaction])


def _regression_model(binary: bool):
    """Linear or logistic (no regularization) second-stage model."""
    return LogisticRegression(penalty=None) if binary else LinearRegression()


def estimate_naive_ols(data: DeepIVData, binary: bool = False) -> float:
    features = _linear_second_stage_features(data.x_2, data.s_2, data.t_2)
    model = _regression_model(binary).fit(features, data.y_2)
    return float(model.coef_.ravel()[0])


def estimate_2sps_linear(data: DeepIVData, binary: bool = False) -> float:
    first = LinearRegression().fit(_iv_inputs(data.z_1, data.t_1), data.x_1)
    x_hat = first.predict(_iv_inputs(data.z_2, data.t_2))
    features = _linear_second_stage_features(x_hat, data.s_2, data.t_2)
    model = _regression_model(binary).fit(features, data.y_2)
    return float(model.coef_.ravel()[0])


def estimate_2sri_linear(data: DeepIVData, binary: bool = False) -> float:
    first = LinearRegression().fit(_iv_inputs(data.z_1, data.t_1), data.x_1)
    x_hat = first.predict(_iv_inputs(data.z_2, data.t_2))
    residual = data.x_2 - x_hat
    features = np.column_stack(
        [_linear_second_stage_features(data.x_2, data.s_2, data.t_2), residual]
    )
    model = _regression_model(binary).fit(features, data.y_2)
    return float(model.coef_.ravel()[0])


def _deep_first_stage(
    data: DeepIVData,
    epochs: int,
    learning_rate: float,
    dropout: float,
) -> tuple[DeepPLIV, np.ndarray, np.ndarray]:
    """Fit a DeepPLIV first stage and return the model + first-stage features.

    Returns (model, x_hat_on_second_stage, residual_on_second_stage).
    The instrument features (z, t) are standardized the same way for both
    halves before being fed to the network.
    """
    model = DeepPLIV()
    scaler = StandardScaler()
    g_iv_1 = scaler.fit_transform(_iv_inputs(data.z_1, data.t_1))
    g_iv_2 = scaler.transform(_iv_inputs(data.z_2, data.t_2))

    first_stage = model.fit_first_stage(
        g_iv_1,                              # z_1: instruments (input)
        data.x_1.reshape(-1, 1),             # v_1: endogenous target
        epochs_first_stage=epochs,
        learning_rate_first_stage=learning_rate,
        dropout=dropout,
        validation_data=(g_iv_2, data.x_2.reshape(-1, 1)),
    )
    x_hat = first_stage.predict(g_iv_2).reshape(-1)
    residual = data.x_2 - x_hat
    return model, x_hat, residual


def _fit_deep_second_stage(
    model: DeepPLIV,
    treatment: np.ndarray,      # (n,)   linear-entering treatment
    exog: np.ndarray,           # (n, d) nonlinear covariates
    y: np.ndarray,              # (n,)
    epochs: int,
    learning_rate: float,
    dropout: float,
) -> float:
    """Fit a DeepPLIV partial-linear second stage and return the linear coeff."""
    scaler = StandardScaler()
    exog_scaled = scaler.fit_transform(exog)
    trained = model.fit_second_stage(
        treatment.reshape(-1, 1),
        exog_scaled,
        y.reshape(-1, 1),
        epochs_second_stage=epochs,
        learning_rate_second_stage=learning_rate,
        dropout=dropout,
    )
    # The linear-in-treatment weight is the first column of the final layer.
    return float(trained.final_layer.weight.detach().cpu().numpy()[0, 0])


def estimate_deeppliv(
    data: DeepIVData,
    epochs: int,
    learning_rate: float,
    dropout: float,
    k_averages: int = 1,
) -> dict[str, float]:
    """Fit DeepPLIV-2SPS, DeepPLIV-2SRI, and the naive feed-forward baseline.

    Each network is refit `k_averages` times and the mean coefficient is
    returned, which reduces per-replication variance from NN initialization.
    """
    sps_runs: list[float] = []
    sri_runs: list[float] = []
    nff_runs: list[float] = []

    for _ in range(k_averages):
        model, x_hat, residual = _deep_first_stage(
            data, epochs=epochs, learning_rate=learning_rate, dropout=dropout,
        )

        # DeepPLIV-2SPS: linear term = x_hat, nonlinear covariates = (s, t).
        exog_sps = np.column_stack([data.s_2, data.t_2])
        sps_runs.append(
            _fit_deep_second_stage(
                model, x_hat, exog_sps, data.y_2,
                epochs=epochs, learning_rate=learning_rate, dropout=dropout,
            )
        )

        # DeepPLIV-2SRI: linear term = raw x, nonlinear covariates = (s, t, residual).
        exog_sri = np.column_stack([data.s_2, data.t_2, residual])
        sri_runs.append(
            _fit_deep_second_stage(
                model, data.x_2, exog_sri, data.y_2,
                epochs=epochs, learning_rate=learning_rate, dropout=dropout,
            )
        )

        # Naive FFN: no IV correction, treats x like any other covariate.
        nff_runs.append(
            _fit_deep_second_stage(
                model, data.x_2, exog_sps, data.y_2,
                epochs=epochs, learning_rate=learning_rate, dropout=dropout,
            )
        )

    return {
        "DeepPLIV-2SPS": float(np.mean(sps_runs)),
        "DeepPLIV-2SRI": float(np.mean(sri_runs)),
        "Naive Feed Forward": float(np.mean(nff_runs)),
    }


# ---------------------------------------------------------------------------
# Simulation driver
# ---------------------------------------------------------------------------


def run_deep_iv_cell(
    n: int,
    rho: float,
    num_simulations: int = 100,
    beta_1: float = -2.0,
    learning_rate: float = 0.01,
    epochs: int | None = None,
    dropout: float | None = None,
    k_averages: int = 1,
    seed: int | None = 0,
    cache_dir: str | None = None,
    standardize_nuisance: bool = True,
    binary: bool = False,
) -> pd.DataFrame:
    """Run the full estimator comparison for one (n, rho) configuration.

    Defaults for `epochs` and `dropout` follow the thesis description:
        dropout = 100 / (1000 + n)
        epochs  = floor(1.5e7 / (n // 2))     (maximum; early stopping may cut short)
    """
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        outcome_tag = "_binary" if binary else ""
        cache_path = os.path.join(
            cache_dir,
            f"deep_iv_n{n}_rho{rho}_k{num_simulations}_beta{beta_1}{outcome_tag}.pkl",
        )
        if os.path.exists(cache_path):
            return pd.read_pickle(cache_path)

    if dropout is None:
        dropout = 100.0 / (1000.0 + n)
    if epochs is None:
        epochs = int(1.5e7 / max(n // 2, 1))

    records: list[dict] = []
    master_rng = np.random.default_rng(seed)

    for sim_idx in range(num_simulations):
        dgp = DeepIVDGP(
            n=n,
            beta_1=beta_1,
            rho=rho,
            rng=np.random.default_rng(master_rng.integers(0, 2**31 - 1)),
            standardize_nuisance=standardize_nuisance,
            binary=binary,
        )
        data = dgp.sample()

        estimates: dict[str, float] = {
            "Naive Regression": estimate_naive_ols(data, binary=binary),
            "2SPS": estimate_2sps_linear(data, binary=binary),
            "2SRI": estimate_2sri_linear(data, binary=binary),
        }
        estimates.update(
            estimate_deeppliv(
                data,
                epochs=epochs,
                learning_rate=learning_rate,
                dropout=dropout,
                k_averages=k_averages,
            )
        )

        for method, value in estimates.items():
            records.append(
                dict(
                    sim=sim_idx,
                    method=method,
                    value=value,
                    n=n,
                    rho=rho,
                    beta_1=beta_1,
                )
            )

        print(
            f"[n={n} rho={rho}] sim {sim_idx + 1}/{num_simulations} "
            + " ".join(f"{m}={v:+.3f}" for m, v in estimates.items())
        )

    df = pd.DataFrame.from_records(records)
    if cache_dir is not None:
        df.to_pickle(cache_path)
    return df


def run_grid(
    n_values: Iterable[int],
    rho_values: Iterable[float],
    **kwargs,
) -> pd.DataFrame:
    """Iterate `run_deep_iv_cell` over a grid and return one stacked DataFrame."""
    frames = []
    for n in n_values:
        for rho in rho_values:
            frames.append(run_deep_iv_cell(n=n, rho=rho, **kwargs))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Smoke test: run both continuous and binary outcomes side-by-side to
    # confirm the binary path produces sensible log-odds estimates near beta_1.
    for outcome, binary_flag in [("continuous", False), ("binary", True)]:
        print(f"\n=== {outcome.capitalize()} outcome (true beta_1 = -2.0) ===")
        df = run_deep_iv_cell(
            n=10000,
            rho=0.5,
            num_simulations=5,
            seed=0,
            binary=binary_flag,
            cache_dir="deep_iv_sim_results",
        )
        print(df.groupby("method")["value"].agg(["mean", "std", "count"]))

    # Full grid (uncomment to run):
    # _N_VALUES = [2000, 10000, 20000, 40000]
    # _RHO_VALUES = [0.1, 0.25, 0.5, 0.75, 0.9]
    # grid = run_grid(
    #     n_values=_N_VALUES,
    #     rho_values=_RHO_VALUES,
    #     num_simulations=100,
    #     seed=0,
    #     cache_dir="deep_iv_sim_results",
    # )
    # grid.to_pickle("deep_iv_sim_results/full_grid.pkl")