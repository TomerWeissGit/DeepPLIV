"""
AWS S3 Parallel Execution System for Binary-Outcome IV Simulation
Adapted from aws_genetic_iv_parallel.py for binary Y ~ Bernoulli DGP.

Uses Deep IV-style correlated-error DGP:
  X  = f(S) + eps_x
  Y  ~ Bernoulli(sigma(beta_x * X + rho_end * eps_x))
  ê  ≈ eps_x  →  perfect proxy for endogeneity; P(Y=1|X,ê) is exactly logistic

Binary-specific changes:
  - gen_data_binary(): correlated-error DGP, no shared U
  - No 2SPS estimator (inconsistent for binary outcomes)
  - run_naive_logistic(): sklearn LogisticRegression(penalty=None), no IV
  - run_logit_2sri(): linear OLS first stage + LogisticRegression second stage
  - run_nn_2sri() / run_naive_nn(): regular DeepPLIV (BCEWithLogitsLoss auto-detected)
  - ensemble_nn_binary() / bootstrap_ci_ensemble_binary()
"""

import json
import gzip
import asyncio
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
import pickle
import logging

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, LogisticRegression

# AWS dependencies
import boto3
from botocore.exceptions import NoCredentialsError

# Local imports — use the regular (packaged) DeepPLIV which auto-detects binary Y
from deeppliv.core.trainer import DeepPLIV
import torch

# CRITICAL: Set multiprocessing start method to 'spawn' for CUDA compatibility
if __name__ != "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

# Optimize PyTorch threading for parallel workers (guard: can only be set once)
try:
    torch.set_num_threads(1)
except RuntimeError:
    pass
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def get_gpu_count() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # AWS Settings
    S3_URI: str = os.getenv("S3_URI")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET: str = None   # parsed from S3_URI
    BASE_S3_PATH: str = None  # parsed from S3_URI

    def __post_init_s3(self):
        if self.S3_URI:
            if self.S3_URI.startswith("s3://"):
                parts = self.S3_URI[5:].split("/", 1)
                self.S3_BUCKET = parts[0]
                self.BASE_S3_PATH = parts[1].rstrip("/") if len(parts) > 1 else ""
                # Use BINARY_SIM_CORR_ERR subfolder (correlated-error DGP, Deep IV style)
                if self.BASE_S3_PATH:
                    self.BASE_S3_PATH = f"{self.BASE_S3_PATH}/BINARY_SIM_CORR_ERR"
                else:
                    self.BASE_S3_PATH = "BINARY_SIM_CORR_ERR"
            else:
                raise ValueError(
                    f"Invalid S3_URI format: {self.S3_URI}. Expected format: s3://bucket-name/path/"
                )
        else:
            self.S3_BUCKET = os.getenv("S3_BUCKET")
            self.BASE_S3_PATH = "BINARY_SIM_CORR_ERR"

        if self.AWS_REGION:
            self.AWS_REGION = self.AWS_REGION.replace("_", "-")

    # Execution mode
    LOCAL_MODE: bool = os.getenv("LOCAL_MODE", "false").lower() == "true"

    # Simulation parameters (env vars with defaults)
    ENSEMBLE_SIZE: int = int(os.getenv("ENSEMBLE_SIZE", "50"))
    BOOTSTRAPS: int = int(os.getenv("BOOTSTRAPS", "200"))
    NUM_SIMULATIONS: int = int(os.getenv("NUM_SIMULATIONS", "200"))
    CI_LEVEL: Tuple[float, float] = (2.5, 97.5)

    # Model parameters
    EPOCHS: int = 1000
    LEARNING_RATE: float = 0.01
    DROPOUT: float = 0.01

    # Parameter grid
    N_VALUES: List[int] = None
    RHO_VALUES: List[float] = None
    P01Y_VALUES: List[float] = None
    RHO_END_VALUES: List[float] = None

    @property
    def MAX_WORKERS(self) -> int:
        return mp.cpu_count()

    @property
    def MAX_OUTER_WORKERS(self) -> int:
        env_val = os.getenv("MAX_OUTER_WORKERS")
        if env_val:
            return int(env_val)
        gc = get_gpu_count()
        return gc if gc > 0 else 2

    @property
    def MAX_INNER_WORKERS(self) -> int:
        env_val = os.getenv("MAX_INNER_WORKERS")
        if env_val:
            return int(env_val)
        return 10 if get_gpu_count() > 0 else 7

    def __post_init__(self):
        self.__post_init_s3()

        # Threading environment variables
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'

        if self.N_VALUES is None:
            self.N_VALUES = [10000, 20000, 40000]
        if self.RHO_VALUES is None:
            self.RHO_VALUES = [0.5]
        if self.P01Y_VALUES is None:
            self.P01Y_VALUES = [0.05, 0.4]  # cross-pop (diff allele freq) and same-pop
        if self.RHO_END_VALUES is None:
            self.RHO_END_VALUES = [0, 0.5]  # unconfounded and confounded (correlated-error)

        mode = "LOCAL TESTING" if self.LOCAL_MODE else "FULL AWS"
        logger.info(f"Configuration loaded for {mode} mode")
        logger.info(
            f"S3 Configuration: Bucket={self.S3_BUCKET}, Path={self.BASE_S3_PATH}, Region={self.AWS_REGION}"
        )
        logger.info(
            f"CPU cores: {mp.cpu_count()}, "
            f"Outer workers: {self.MAX_OUTER_WORKERS}, "
            f"Inner workers: {self.MAX_INNER_WORKERS}"
        )
        logger.info(
            f"Parameters: ENSEMBLE_SIZE={self.ENSEMBLE_SIZE}, "
            f"BOOTSTRAPS={self.BOOTSTRAPS}, "
            f"NUM_SIMULATIONS={self.NUM_SIMULATIONS}"
        )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global fixed effects (loaded from S3 at startup)
# ---------------------------------------------------------------------------

gamma_j1 = None
gamma_j2 = None
gamma_jm12 = None
gamma_j1121 = None

# Simulation constants
K1 = K2 = 7
p01 = p02 = 0.4
beta_x = 2.0  # structural treatment coefficient


# ---------------------------------------------------------------------------
# S3 Manager
# ---------------------------------------------------------------------------

class S3Manager:
    """S3 operations manager using standard boto3 with async wrapper"""

    def __init__(self, bucket_name: str, region: str, base_path: str):
        self.bucket_name = bucket_name
        self.region = region
        self.base_path = base_path
        self.s3_client = None

    async def __aenter__(self):
        self.s3_client = boto3.client('s3', region_name=self.region)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    def _get_s3_key(self, relative_path: str) -> str:
        return f"{self.base_path}/{relative_path}"

    async def upload_object(self, local_path: str, s3_key: str, compress: bool = True) -> bool:
        loop = asyncio.get_event_loop()

        def _sync_upload():
            try:
                full_key = self._get_s3_key(s3_key)
                if compress and not s3_key.endswith('.gz'):
                    with open(local_path, 'rb') as f:
                        data = f.read()
                    compressed_data = gzip.compress(data)
                    full_key += '.gz'
                    self.s3_client.put_object(
                        Bucket=self.bucket_name,
                        Key=full_key,
                        Body=compressed_data
                    )
                else:
                    self.s3_client.upload_file(local_path, self.bucket_name, full_key)
                return True
            except Exception as e:
                logger.error(f"Failed to upload {s3_key}: {e}")
                return False

        return await loop.run_in_executor(None, _sync_upload)

    async def download_object(self, s3_key: str, local_path: str, decompress: bool = True) -> bool:
        loop = asyncio.get_event_loop()

        def _sync_download():
            try:
                full_key = self._get_s3_key(s3_key)
                if decompress and not full_key.endswith('.gz'):
                    gz_key = full_key + '.gz'
                    try:
                        response = self.s3_client.get_object(Bucket=self.bucket_name, Key=gz_key)
                        compressed_data = response['Body'].read()
                        data = gzip.decompress(compressed_data)
                        with open(local_path, 'wb') as f:
                            f.write(data)
                        return True
                    except Exception:
                        pass

                if decompress and full_key.endswith('.gz'):
                    response = self.s3_client.get_object(Bucket=self.bucket_name, Key=full_key)
                    compressed_data = response['Body'].read()
                    data = gzip.decompress(compressed_data)
                    with open(local_path, 'wb') as f:
                        f.write(data)
                else:
                    self.s3_client.download_file(self.bucket_name, full_key, local_path)
                return True
            except Exception as e:
                logger.error(f"Failed to download {s3_key}: {e}")
                return False

        return await loop.run_in_executor(None, _sync_download)

    async def list_objects(self, prefix: str) -> List[str]:
        loop = asyncio.get_event_loop()

        def _sync_list():
            try:
                full_prefix = self._get_s3_key(prefix)
                objects = []
                paginator = self.s3_client.get_paginator('list_objects_v2')
                for page in paginator.paginate(Bucket=self.bucket_name, Prefix=full_prefix):
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            key = obj['Key']
                            if key.startswith(f"{self.base_path}/"):
                                relative_key = key[len(f"{self.base_path}/"):]
                                objects.append(relative_key)
                return objects
            except Exception as e:
                logger.error(f"Failed to list objects with prefix {prefix}: {e}")
                return []

        return await loop.run_in_executor(None, _sync_list)

    async def put_json(self, data: Dict, s3_key: str) -> bool:
        loop = asyncio.get_event_loop()

        def _sync_put_json():
            try:
                full_key = self._get_s3_key(s3_key)
                json_data = json.dumps(data, indent=2)
                self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=full_key,
                    Body=json_data.encode('utf-8'),
                    ContentType='application/json'
                )
                return True
            except Exception as e:
                logger.error(f"Failed to upload JSON {s3_key}: {e}")
                return False

        return await loop.run_in_executor(None, _sync_put_json)

    async def get_json(self, s3_key: str) -> Optional[Dict]:
        loop = asyncio.get_event_loop()

        def _sync_get_json():
            try:
                full_key = self._get_s3_key(s3_key)
                response = self.s3_client.get_object(Bucket=self.bucket_name, Key=full_key)
                data = response['Body'].read()
                return json.loads(data.decode('utf-8'))
            except Exception as e:
                logger.error(f"Failed to download JSON {s3_key}: {e}")
                return None

        return await loop.run_in_executor(None, _sync_get_json)


# ---------------------------------------------------------------------------
# DGP
# ---------------------------------------------------------------------------

def simulate_snp_block(n, K, p0, rho):
    Z = np.random.normal(size=n)
    snps = np.zeros((n, K))
    for j in range(K):
        eps = np.random.normal(size=n)
        p = norm.cdf(norm.ppf(p0) + np.sqrt(rho) * Z + np.sqrt(1 - rho) * eps)
        snps[:, j] = np.random.binomial(2, p)
    return snps


def simulate_snp_dataset(n, p01_val, p02_val, rho):
    S1 = simulate_snp_block(n, K1, p01_val, rho)
    S2 = simulate_snp_block(n, K2, p02_val, rho)
    return S1, S2


def x_eq(S1, S2, n):
    """Compute X from SNPs. No shared U — endogeneity enters only via eps_x."""
    global gamma_j1, gamma_j2, gamma_jm12, gamma_j1121

    interaction_term = np.sum(
        S1[:, :, None] * S2[:, None, :] * gamma_jm12[None, :, :], axis=(1, 2)
    )
    non_linear_part = (
        S1 @ gamma_j1
        + S2 @ gamma_j2
        + interaction_term
        + (S1[:, 0] * S1[:, 1]) * (S2 @ gamma_j1121)
    )
    epsilon_x = np.random.normal(0, 1.0, size=n)
    return non_linear_part + epsilon_x, epsilon_x


def gen_data_binary(n, p01_val, p02_val, rho, rho_end):
    """
    Generate a dataset with binary outcome via correlated-error DGP (Deep IV style).

    X  = f(S) + eps_x
    Y  ~ Bernoulli(sigma(beta_x * X + rho_end * eps_x))

    ê = X - X̂ ≈ eps_x → P(Y=1|X,ê) is exactly logistic in (X,ê),
    so 2SRI with logistic second stage is correctly specified.

    rho_end=0: unconfounded (naive logit recovers beta_x).
    rho_end>0: confounded; 2SRI corrects via the residual.

    Returns DataFrame with S1_1..S1_7, S2_1..S2_7, X, Y columns.
    Y is binary (0/1).
    """
    S1, S2 = simulate_snp_dataset(n, p01_val, p02_val, rho)
    X, eps_x = x_eq(S1, S2, n)
    prob = 1.0 / (1.0 + np.exp(-(beta_x * X + rho_end * eps_x)))
    Y = np.random.binomial(1, prob).astype(np.float64)
    df = pd.DataFrame({
        **{f"S1_{j + 1}": S1[:, j] for j in range(K1)},
        **{f"S2_{j + 1}": S2[:, j] for j in range(K2)},
        "X": X,
        "Y": Y
    })
    return df


# ---------------------------------------------------------------------------
# Linear / Logistic estimators
# ---------------------------------------------------------------------------

def _snp_cols():
    return [f"S1_{j + 1}" for j in range(K1)] + [f"S2_{j + 1}" for j in range(K2)]


def run_naive_logistic(df_y: pd.DataFrame) -> float:
    """
    Simple logistic regression of Y on X (no IV correction).
    Returns the coefficient on X.
    """
    m = LogisticRegression(penalty=None).fit(df_y[["X"]].values, df_y["Y"].values)
    return float(m.coef_.ravel()[0])


def run_logit_2sri(df_x: pd.DataFrame, df_y: pd.DataFrame) -> float:
    """
    Two-Stage Residual Inclusion (2SRI) with logistic second stage.

    First stage : linear OLS of X on SNPs (fit on df_x, predict on df_y SNPs).
    Second stage: LogisticRegression(penalty=None) of Y on [X, residual].
    Returns the coefficient on X.
    """
    snp_cols = _snp_cols()
    Z_X = df_x[snp_cols].values
    Z_Y = df_y[snp_cols].values

    lr = LinearRegression().fit(Z_X, df_x["X"].values)
    X_hat_Y = lr.predict(Z_Y)
    X_err_Y = df_y["X"].values - X_hat_Y

    des = np.column_stack([df_y["X"].values, X_err_Y])
    m = LogisticRegression(penalty=None).fit(des, df_y["Y"].values)
    return float(m.coef_.ravel()[0])


# ---------------------------------------------------------------------------
# MultiGPUDeepPLIV — kept for reference, NOT used in binary estimators
# ---------------------------------------------------------------------------

class MultiGPUDeepPLIV:
    """
    Multi-GPU optimized version of DeepPLIV with explicit device assignment.
    Kept for reference / future use. Binary estimators use the regular packaged
    DeepPLIV which auto-detects binary Y and switches to BCEWithLogitsLoss.
    """

    def __init__(self, device_id=None):
        if device_id is not None and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device_id}")
            self.device_id = device_id
        else:
            self.device = torch.device("cpu")
            self.device_id = None

        self.first_stage_model = None
        self.second_stage_model = None

    def fit_first_stage(self, z_train, v_train, z_val, v_val, epochs, lr, dropout):
        from deeppliv.models.first_stage import NeuralNetworkFirstStage

        z_train_tensor = torch.tensor(z_train, dtype=torch.float32, device=self.device)
        v_train_tensor = torch.tensor(v_train, dtype=torch.float32, device=self.device).view(-1, 1)
        z_val_tensor = torch.tensor(z_val, dtype=torch.float32, device=self.device)
        v_val_tensor = torch.tensor(v_val, dtype=torch.float32, device=self.device).view(-1, 1)

        self.first_stage_model = NeuralNetworkFirstStage(
            input_dim=z_train.shape[1],
            output_dim=1,
            dropout=dropout
        ).to(self.device)

        optimizer = torch.optim.Adam(self.first_stage_model.parameters(), lr=lr, weight_decay=0.001)
        criterion = torch.nn.MSELoss()

        self.first_stage_model.train()
        best_val_loss = float('inf')
        patience_counter = 0
        patience = int(np.sqrt(epochs))

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.first_stage_model(z_train_tensor)
            loss = criterion(outputs, v_train_tensor)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                self.first_stage_model.eval()
                with torch.no_grad():
                    val_outputs = self.first_stage_model(z_val_tensor)
                    val_loss = criterion(val_outputs, v_val_tensor)

                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break

                self.first_stage_model.train()

        return self.first_stage_model

    def predict_first_stage(self, z_new):
        self.first_stage_model.eval()
        with torch.no_grad():
            z_tensor = torch.tensor(z_new, dtype=torch.float32, device=self.device)
            predictions = self.first_stage_model(z_tensor)
            return predictions.cpu().numpy()

    def fit_second_stage(self, v_input, x_input, y_target, epochs, lr, dropout):
        from deeppliv.models.second_stage import NeuralNetworkSecondStage

        v_tensor = torch.tensor(v_input, dtype=torch.float32, device=self.device)
        x_tensor = torch.tensor(x_input, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y_target, dtype=torch.float32, device=self.device).view(-1, 1)

        model = NeuralNetworkSecondStage(
            x=x_input.shape[1],
            v=v_input.shape[1],
            dropout=dropout
        ).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        model.train()
        best_loss = float('inf')
        patience_counter = 0
        patience = int(np.sqrt(epochs))

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(x_tensor, v_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0 and loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            elif epoch % 10 == 0:
                patience_counter += 1

            if patience_counter >= patience:
                break

        return model


# ---------------------------------------------------------------------------
# NN estimators (binary-aware, use regular DeepPLIV)
# ---------------------------------------------------------------------------

def run_nn_2sri(
    df_x: pd.DataFrame,
    df_y: pd.DataFrame,
    epochs: int,
    lr: float,
    dropout: float,
    device_id=None
) -> Optional[float]:
    """
    NN-based 2SRI for binary outcome.

    Uses the regular DeepPLIV from deeppliv.core.trainer which auto-detects
    binary Y (all 0/1) and switches to BCEWithLogitsLoss in the second stage.

    First stage : fit_first_stage(Z_X, X, ..., validation_data=(Z_Y, X_Y))
    Second stage: fit_second_stage(v_hat=X_Y, x=x_err, y=Y, method="2sri")

    Returns the final_layer.weight[0, 0] coefficient (log-odds scale for binary Y).
    """
    snp_cols = _snp_cols()
    Z_X = df_x[snp_cols].values
    Z_Y = df_y[snp_cols].values
    X = df_x["X"].values
    X_Y = df_y["X"].values
    Y = df_y["Y"].values

    try:
        model = DeepPLIV()
        model.fit_first_stage(
            Z_X, X,
            epochs_first_stage=epochs,
            learning_rate_first_stage=lr,
            dropout=dropout,
            validation_data=(Z_Y, X_Y)
        )
        x_pred = model.predict_first_stage(Z_Y).reshape(-1, 1)
        x_err = X_Y.reshape(-1, 1) - x_pred

        second_stage = model.fit_second_stage(
            v_hat=X_Y.reshape(-1, 1),
            x=x_err,
            y=Y.reshape(-1, 1),
            epochs_second_stage=epochs,
            learning_rate_second_stage=lr,
            dropout=dropout,
            method="2sri"
        )
        return float(second_stage.final_layer.weight[0, 0].item())
    except Exception as e:
        logger.error(f"run_nn_2sri failed: {e}")
        return None


def run_naive_nn(
    df_x: pd.DataFrame,
    df_y: pd.DataFrame,
    epochs: int,
    lr: float,
    dropout: float
) -> Optional[float]:
    """
    Naive NN second stage: observed X as v_hat, dummy ones as x (no IV correction).
    Binary Y is auto-detected by DeepPLIV -> BCEWithLogitsLoss.

    Returns the final_layer.weight[0, 0] coefficient.
    """
    snp_cols = _snp_cols()
    Z_X = df_x[snp_cols].values
    Z_Y = df_y[snp_cols].values
    X = df_x["X"].values
    X_Y = df_y["X"].values
    Y = df_y["Y"].values
    ones = np.ones((X_Y.shape[0], 1))

    try:
        model = DeepPLIV()
        model.fit_first_stage(
            Z_X, X,
            epochs_first_stage=epochs,
            learning_rate_first_stage=lr,
            dropout=dropout,
            validation_data=(Z_Y, X_Y)
        )

        second_stage = model.fit_second_stage(
            v_hat=X_Y.reshape(-1, 1),
            x=ones,
            y=Y.reshape(-1, 1),
            epochs_second_stage=epochs,
            learning_rate_second_stage=lr,
            dropout=dropout,
            method=None   # no binary guard for None
        )
        return float(second_stage.final_layer.weight[0, 0].item())
    except Exception as e:
        logger.error(f"run_naive_nn failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Parallel ensemble / bootstrap (ProcessPoolExecutor-compatible)
# ---------------------------------------------------------------------------

def run_one_nn_fit(args) -> Tuple[Optional[float], Optional[float]]:
    """
    Worker function for ProcessPoolExecutor.

    Fits a fresh DeepPLIV:
      1. fit_first_stage once on df_x (SNPs->X), validate on df_y SNPs.
      2. fit_second_stage for 2SRI (method="2sri").
      3. fit_second_stage for naive (method=None, v_hat=X_Y, x=ones).

    Returns (nn_2sri_beta, naive_nn_beta).
    """
    df_x, df_y, nn_kwargs = args
    epochs = nn_kwargs.get('epochs', 1000)
    lr = nn_kwargs.get('lr', 0.01)
    dropout = nn_kwargs.get('dropout', 0.01)

    snp_cols = _snp_cols()
    Z_X = df_x[snp_cols].values
    Z_Y = df_y[snp_cols].values
    X = df_x["X"].values
    X_Y = df_y["X"].values
    Y = df_y["Y"].values
    ones = np.ones((X_Y.shape[0], 1))

    try:
        model = DeepPLIV()
        model.fit_first_stage(
            Z_X, X,
            epochs_first_stage=epochs,
            learning_rate_first_stage=lr,
            dropout=dropout,
            validation_data=(Z_Y, X_Y)
        )
        x_pred = model.predict_first_stage(Z_Y).reshape(-1, 1)
        x_err = X_Y.reshape(-1, 1) - x_pred

        # 2SRI second stage
        ss_2sri = model.fit_second_stage(
            v_hat=X_Y.reshape(-1, 1),
            x=x_err,
            y=Y.reshape(-1, 1),
            epochs_second_stage=epochs,
            learning_rate_second_stage=lr,
            dropout=dropout,
            method="2sri"
        )
        beta_2sri = float(ss_2sri.final_layer.weight[0, 0].item())

        # Naive second stage (same first stage, new second stage instance)
        ss_naive = model.fit_second_stage(
            v_hat=X_Y.reshape(-1, 1),
            x=ones,
            y=Y.reshape(-1, 1),
            epochs_second_stage=epochs,
            learning_rate_second_stage=lr,
            dropout=dropout,
            method=None
        )
        beta_naive = float(ss_naive.final_layer.weight[0, 0].item())

        return beta_2sri, beta_naive

    except Exception as e:
        logger.error(f"run_one_nn_fit failed: {e}")
        return None, None


def ensemble_nn_binary(
    df_x: pd.DataFrame,
    df_y: pd.DataFrame,
    M: int,
    max_workers: int,
    **nn_kwargs
) -> Tuple[Optional[float], Optional[float], List[float], List[float]]:
    """
    Run M independent NN fits in parallel via ProcessPoolExecutor.

    Returns (mean_2sri, mean_naive, list_2sri, list_naive).
    """
    args_list = [(df_x, df_y, nn_kwargs)] * M
    sri_list = []
    naive_list = []

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_one_nn_fit, a) for a in args_list]
        for f in as_completed(futures):
            beta_2sri, beta_naive = f.result()
            if beta_2sri is not None:
                sri_list.append(beta_2sri)
            if beta_naive is not None:
                naive_list.append(beta_naive)

    mean_2sri = float(np.mean(sri_list)) if sri_list else None
    mean_naive = float(np.mean(naive_list)) if naive_list else None
    return mean_2sri, mean_naive, sri_list, naive_list


def _bootstrap_nn_binary_worker(args):
    """
    Worker for a single bootstrap resample.

    Resamples both df_x and df_y independently (two-sample MR design),
    runs run_one_nn_fit, and returns (beta_2sri_b, beta_naive_b).
    """
    df_x, df_y, n_x, n_y, nn_kwargs, seed = args
    rng = np.random.default_rng(seed)
    idx_x = rng.integers(0, n_x, n_x)
    idx_y = rng.integers(0, n_y, n_y)
    dfx_b = df_x.iloc[idx_x].reset_index(drop=True)
    dfy_b = df_y.iloc[idx_y].reset_index(drop=True)
    return run_one_nn_fit((dfx_b, dfy_b, nn_kwargs))


def bootstrap_ci_ensemble_binary(
    ensemble_ests: Tuple[Optional[float], Optional[float]],
    df_x: pd.DataFrame,
    df_y: pd.DataFrame,
    B: int,
    max_workers: int,
    **nn_kwargs
) -> Dict:
    """
    Ensemble-based bootstrap CI for binary NN estimators.

    For each bootstrap resample b:
      dev_2sri_b  = |beta_2sri_b  - ensemble_mean_2sri|
      dev_naive_b = |beta_naive_b - ensemble_mean_naive|

    CI = ensemble_est ± q95(deviations).

    Returns dict with:
      nn_2sri_ci_lo, nn_2sri_ci_hi,
      naive_nn_ci_lo, naive_nn_ci_hi,
      raw_sri, raw_naive
    """
    ensemble_2sri, ensemble_naive = ensemble_ests
    n_x, n_y = len(df_x), len(df_y)
    seeds = np.random.SeedSequence().spawn(B)

    args_list = [
        (df_x, df_y, n_x, n_y, nn_kwargs, int(s.generate_state(1)[0]))
        for s in seeds
    ]

    sri_boot = []
    naive_boot = []
    sri_dev = []
    naive_dev = []

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_bootstrap_nn_binary_worker, a) for a in args_list]
        for f in as_completed(futures):
            b2sri, bnaive = f.result()
            if b2sri is not None:
                sri_boot.append(b2sri)
                if ensemble_2sri is not None:
                    sri_dev.append(abs(b2sri - ensemble_2sri))
            if bnaive is not None:
                naive_boot.append(bnaive)
                if ensemble_naive is not None:
                    naive_dev.append(abs(bnaive - ensemble_naive))

    def _ci(est, devs):
        if est is None or not devs:
            return None, None
        q95 = float(np.percentile(devs, 95))
        return float(est - q95), float(est + q95)

    sri_lo, sri_hi = _ci(ensemble_2sri, sri_dev)
    naive_lo, naive_hi = _ci(ensemble_naive, naive_dev)

    return {
        'nn_2sri_ci_lo': sri_lo,
        'nn_2sri_ci_hi': sri_hi,
        'naive_nn_ci_lo': naive_lo,
        'naive_nn_ci_hi': naive_hi,
        'raw_sri': sri_boot,
        'raw_naive': naive_boot,
    }


# ---------------------------------------------------------------------------
# Fixed effects loader
# ---------------------------------------------------------------------------

def load_fixed_effects():
    """Load fixed effects from pickle file in S3 (same path as original MR_SIM study)."""
    global gamma_j1, gamma_j2, gamma_jm12, gamma_j1121

    s3_uri = os.getenv("S3_URI") + "MR_SIM/"
    file_uri = s3_uri + "fixed_effects.pkl"

    _, _, bucket, *key_parts = file_uri.split("/")
    key = "/".join(key_parts)

    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))
    obj = s3.get_object(Bucket=bucket, Key=key)
    effect_dict = pickle.loads(obj["Body"].read())

    gamma_j1 = effect_dict["gamma_j1"]
    gamma_j2 = effect_dict["gamma_j2"]
    gamma_jm12 = effect_dict["gamma_jm12"]
    gamma_j1121 = effect_dict["gamma_j1121"]

    logger.info("Fixed effects loaded successfully from %s", file_uri)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def convert_to_json_serializable(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(v) for v in obj]
    return obj


def create_estimates_csv(estimates_dict: Dict, config_id: int, run_id: int,
                         worker_id: int, estimation_type: str) -> str:
    """Create CSV content from individual estimates."""
    import csv
    from io import StringIO

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow([
        'estimate_id', 'model_type', 'estimate_value', 'config_id', 'run_id',
        'worker_id', 'estimation_type', 'timestamp'
    ])

    timestamp = datetime.now().isoformat()
    for model_type, estimates in estimates_dict.items():
        for i, estimate in enumerate(estimates):
            if estimate is not None:
                writer.writerow([
                    i + 1, model_type, float(estimate), config_id, run_id,
                    worker_id, estimation_type, timestamp
                ])

    return output.getvalue()


async def save_estimates_csv(s3_manager: S3Manager, csv_content: str, s3_key: str) -> bool:
    """Save CSV content to S3."""
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        success = await s3_manager.upload_object(temp_path, s3_key, compress=True)
        os.remove(temp_path)
        return success
    except Exception as e:
        logger.error(f"Failed to save CSV to S3 {s3_key}: {e}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        return False


# ---------------------------------------------------------------------------
# ProgressTracker
# ---------------------------------------------------------------------------

class ProgressTracker:
    """Track and checkpoint progress."""

    def __init__(self, s3_manager: S3Manager):
        self.s3_manager = s3_manager
        self.start_time = datetime.now()

    async def rebuild_from_results(self) -> None:
        """Rebuild checkpoint from existing result files in S3."""
        logger.info("Rebuilding checkpoint from S3 result files...")
        result_files = await self.s3_manager.list_objects('results/')
        result_files = [f for f in result_files if f.endswith('_results.json')]
        logger.info(f"Found {len(result_files)} result files in S3")

        completed_tasks = []
        for file_path in result_files:
            parts = file_path.replace('results/', '').split('/')
            if len(parts) == 2:
                config_part = parts[0]
                run_part = parts[1].replace('_results.json', '')
                task_key = f"{config_part}_{run_part}"
                completed_tasks.append(task_key)

        completed_tasks.sort()

        if not completed_tasks:
            logger.info("No existing results found to rebuild from")
            return

        existing = await self.s3_manager.get_json('checkpoints/binary_progress.json')

        checkpoint = {
            'completed_tasks': completed_tasks,
            'failed_tasks': existing.get('failed_tasks', []) if existing else [],
            'start_time': (
                existing.get('start_time', self.start_time.isoformat())
                if existing else self.start_time.isoformat()
            ),
            'last_update': datetime.now().isoformat(),
            'total_completed': len(completed_tasks),
            'total_failed': len(existing.get('failed_tasks', [])) if existing else 0,
            'rebuilt_at': datetime.now().isoformat(),
            'rebuilt_from': 'S3 results scan'
        }

        await self.s3_manager.put_json(checkpoint, 'checkpoints/binary_progress.json')
        logger.info(f"Checkpoint rebuilt: {len(completed_tasks)} tasks recovered from results")

    async def load_checkpoint(self) -> Dict:
        """Load existing checkpoint from S3."""
        checkpoint = await self.s3_manager.get_json('checkpoints/binary_progress.json')
        if checkpoint is None:
            checkpoint = {
                'completed_tasks': [],
                'failed_tasks': [],
                'start_time': self.start_time.isoformat(),
                'last_update': self.start_time.isoformat()
            }
        return checkpoint

    async def save_checkpoint(self, completed_tasks: List[str], failed_tasks: List[str]) -> None:
        """Save current progress to S3."""
        checkpoint = {
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks,
            'start_time': self.start_time.isoformat(),
            'last_update': datetime.now().isoformat(),
            'total_completed': len(completed_tasks),
            'total_failed': len(failed_tasks)
        }
        await self.s3_manager.put_json(checkpoint, 'checkpoints/binary_progress.json')

    async def log_progress(self, total_tasks: int, completed: int, failed: int) -> None:
        """Log current progress."""
        elapsed = datetime.now() - self.start_time
        rate = completed / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
        remaining = total_tasks - completed - failed
        eta_seconds = remaining / rate if rate > 0 else 0
        eta = str(pd.Timedelta(seconds=eta_seconds))
        logger.info(
            f"Progress: {completed}/{total_tasks} completed, {failed} failed, "
            f"{rate:.2f} tasks/sec, ETA: {eta}"
        )


# ---------------------------------------------------------------------------
# Module-level worker (must be picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _x_eq_inline(S1, S2, n, gj1, gj2, gjm12, gj1121):
    """Self-contained version of x_eq that takes gamma arrays explicitly (no globals)."""
    interaction_term = np.sum(S1[:, :, None] * S2[:, None, :] * gjm12[None, :, :], axis=(1, 2))
    non_linear_part = (
        S1 @ gj1
        + S2 @ gj2
        + interaction_term
        + (S1[:, 0] * S1[:, 1]) * (S2 @ gj1121)
    )
    epsilon_x = np.random.normal(0, 1.0, size=n)
    return non_linear_part + epsilon_x, epsilon_x


def _gen_data_inline(n, p01_val, p02_val, rho, rho_end_val, gj1, gj2, gjm12, gj1121):
    """Self-contained dataset generator (no module globals needed). Correlated-error DGP."""
    S1, S2 = simulate_snp_dataset(n, p01_val, p02_val, rho)
    X, eps_x = _x_eq_inline(S1, S2, n, gj1, gj2, gjm12, gj1121)
    prob = 1.0 / (1.0 + np.exp(-(beta_x * X + rho_end_val * eps_x)))
    Y = np.random.binomial(1, prob).astype(np.float64)
    df = pd.DataFrame({
        **{f"S1_{j + 1}": S1[:, j] for j in range(K1)},
        **{f"S2_{j + 1}": S2[:, j] for j in range(K2)},
        "X": X,
        "Y": Y
    })
    return df


def _generate_dataset_worker(args) -> Optional[str]:
    """Generate one dataset and save to a local gzip-pickle file.

    Args tuple: (config_id, config_params, run_id, temp_dir, gj1, gj2, gjm12, gj1121)
    Gamma arrays are passed explicitly — no module globals, no re-import, no torch calls.
    """
    config_id, config_params, run_id, temp_dir, gj1, gj2, gjm12, gj1121 = args
    try:
        n_val, rho_val, p01y_val, rho_end_val = (
            config_params['n'], config_params['rho'], config_params['p01y'],
            config_params['rho_end']
        )
        df_x = _gen_data_inline(n_val, p01, p02, rho_val, rho_end_val, gj1, gj2, gjm12, gj1121)
        df_y = _gen_data_inline(n_val, p01y_val, p02, rho_val, rho_end_val, gj1, gj2, gjm12, gj1121)
        dataset = {
            'df_x': df_x,
            'df_y': df_y,
            'params': config_params,
            'run_id': run_id,
            'timestamp': datetime.now().isoformat()
        }
        local_path = f"{temp_dir}/dataset_{run_id}.pkl.gz"
        with gzip.open(local_path, 'wb', compresslevel=9) as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
        file_size = os.path.getsize(local_path)
        logger.info(f"Generated config_{config_id}_run_{run_id}: {file_size / 1024:.1f}KB (compressed)")
        return local_path
    except Exception as e:
        logger.error(f"Failed to generate dataset config_{config_id}_run_{run_id}: {e}")
        return None


# ---------------------------------------------------------------------------
# DatasetGenerator
# ---------------------------------------------------------------------------

class DatasetGenerator:
    """Generate binary-outcome datasets and upload to S3."""

    def __init__(self, config: Config, s3_manager: S3Manager):
        self.config = config
        self.s3_manager = s3_manager

    def generate_single_dataset(self, config_params: Dict, run_id: int) -> Dict:
        n_val, rho_val, p01y_val, rho_end_val = (
            config_params['n'], config_params['rho'], config_params['p01y'],
            config_params['rho_end']
        )
        # df_x uses same-population SNP frequencies (p01, p02)
        df_x = gen_data_binary(n_val, p01, p02, rho_val, rho_end_val)
        # df_y uses p01y (cross-pop or same-pop as configured)
        df_y = gen_data_binary(n_val, p01y_val, p02, rho_val, rho_end_val)

        return {
            'df_x': df_x,
            'df_y': df_y,
            'params': config_params,
            'run_id': run_id,
            'timestamp': datetime.now().isoformat()
        }

    def generate_single_dataset_to_file(
        self, config_id: int, config_params: Dict, run_id: int, temp_dir: str
    ) -> Optional[str]:
        return _generate_dataset_worker(
            (config_id, config_params, run_id, temp_dir,
             gamma_j1, gamma_j2, gamma_jm12, gamma_j1121)
        )

    async def generate_all_datasets(self) -> Dict:
        """Generate all datasets using multiprocessing and upload to S3."""
        from itertools import product

        all_configs = list(product(
            self.config.N_VALUES,
            self.config.RHO_VALUES,
            self.config.P01Y_VALUES,
            self.config.RHO_END_VALUES
        ))

        logger.info(
            f"Generating {len(all_configs)} configurations × "
            f"{self.config.NUM_SIMULATIONS} runs = "
            f"{len(all_configs) * self.config.NUM_SIMULATIONS} total datasets"
        )

        generation_checkpoint = await self.s3_manager.get_json(
            'checkpoints/generation_progress.json'
        )
        completed_datasets = set()
        if generation_checkpoint:
            completed_datasets = set(generation_checkpoint.get('completed_datasets', []))
            logger.info(
                f"Found existing generation checkpoint with {len(completed_datasets)} completed datasets"
            )

        task_manifest = {
            'total_configs': len(all_configs),
            'total_datasets': len(all_configs) * self.config.NUM_SIMULATIONS,
            'configs': {},
            'generation_timestamp': datetime.now().isoformat(),
            'generation_checkpoint_loaded': len(completed_datasets) > 0
        }

        # Build all generation tasks
        all_generation_tasks = []
        for config_id, (n_val, rho_val, p01y_val, rho_end_val) in enumerate(all_configs):
            config_params = {
                'n': n_val,
                'rho': rho_val,
                'p01y': p01y_val,
                'rho_end': rho_end_val
            }
            temp_dir = f"/tmp/binary_datasets_config_{config_id}"
            os.makedirs(temp_dir, exist_ok=True)
            for run_id in range(self.config.NUM_SIMULATIONS):
                all_generation_tasks.append((config_id, config_params, run_id, temp_dir))

        logger.info(f"Submitting {len(all_generation_tasks)} dataset generation tasks...")

        with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            future_to_task = {}
            for config_id, config_params, run_id, temp_dir in all_generation_tasks:
                future = executor.submit(
                    _generate_dataset_worker,
                    (config_id, config_params, run_id, temp_dir,
                     gamma_j1, gamma_j2, gamma_jm12, gamma_j1121)
                )
                future_to_task[future] = (config_id, run_id)

            generated_files: Dict[int, List[str]] = {}
            completed_count = 0

            for future in as_completed(future_to_task):
                config_id, run_id = future_to_task[future]
                try:
                    local_path = future.result()
                    if local_path:
                        if config_id not in generated_files:
                            generated_files[config_id] = []
                        generated_files[config_id].append(local_path)
                        completed_count += 1
                        if completed_count % 4 == 0:
                            logger.info(
                                f"Generated {completed_count}/{len(all_generation_tasks)} datasets"
                            )
                except Exception as e:
                    logger.error(f"Failed to generate dataset config_{config_id}_run_{run_id}: {e}")

        logger.info("Generation completed. Starting parallel upload to S3...")

        # Build upload tasks
        upload_tasks = []
        for config_id in range(len(all_configs)):
            if config_id in generated_files:
                for local_path in generated_files[config_id]:
                    filename = os.path.basename(local_path)
                    if filename.endswith('.pkl.gz'):
                        run_id = int(filename.replace('dataset_', '').replace('.pkl.gz', ''))
                        s3_key = f"datasets/config_{config_id}/dataset_{run_id}.pkl.gz"
                    else:
                        run_id = int(filename.replace('dataset_', '').replace('.pkl', ''))
                        s3_key = f"datasets/config_{config_id}/dataset_{run_id}.pkl"
                    upload_tasks.append((local_path, s3_key, config_id, run_id))

        logger.info(f"Starting parallel upload of {len(upload_tasks)} files to S3...")

        total_size = sum(
            os.path.getsize(lp) for lp, _, _, _ in upload_tasks if os.path.exists(lp)
        )
        logger.info(f"Total data to upload: {total_size / 1024 / 1024:.2f} MB")

        uploaded_files: Dict[int, int] = {}
        failed_uploads = []
        max_concurrent_uploads = 20
        upload_semaphore = asyncio.Semaphore(max_concurrent_uploads)

        async def upload_single_file(local_path, s3_key, config_id, run_id):
            async with upload_semaphore:
                try:
                    file_size = os.path.getsize(local_path) if os.path.exists(local_path) else 0
                    compress = not s3_key.endswith('.gz')
                    success = await self.s3_manager.upload_object(local_path, s3_key, compress=compress)
                    if success:
                        os.remove(local_path)
                        logger.info(
                            f"Uploaded config_{config_id}_run_{run_id} ({file_size / 1024:.1f}KB)"
                        )
                        return config_id, True
                    else:
                        failed_uploads.append((config_id, run_id, s3_key))
                        return config_id, False
                except Exception as e:
                    logger.error(f"Exception uploading {local_path}: {e}")
                    failed_uploads.append((config_id, run_id, s3_key))
                    return config_id, False

        upload_coroutines = [
            upload_single_file(lp, sk, cid, rid)
            for lp, sk, cid, rid in upload_tasks
        ]
        upload_results = await asyncio.gather(*upload_coroutines, return_exceptions=True)

        for result in upload_results:
            if isinstance(result, Exception):
                logger.error(f"Upload failed with exception: {result}")
            elif isinstance(result, tuple) and len(result) == 2:
                config_id, success = result
                if config_id not in uploaded_files:
                    uploaded_files[config_id] = 0
                if success:
                    uploaded_files[config_id] += 1

        for config_id, (n_val, rho_val, p01y_val, rho_end_val) in enumerate(all_configs):
            config_params = {
                'n': n_val, 'rho': rho_val, 'p01y': p01y_val,
                'rho_end': rho_end_val
            }
            uploaded_count = uploaded_files.get(config_id, 0)
            expected_count = len(generated_files.get(config_id, []))

            if uploaded_count > 0:
                logger.info(f"Config {config_id}: Uploaded {uploaded_count}/{expected_count} datasets")
            else:
                logger.warning(f"Config {config_id}: No datasets uploaded")

            task_manifest['configs'][config_id] = {
                'params': config_params,
                'datasets_generated': uploaded_count,
                's3_prefix': f'datasets/config_{config_id}/'
            }

            temp_dir = f"/tmp/binary_datasets_config_{config_id}"
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)

        if failed_uploads:
            logger.warning(f"Failed to upload {len(failed_uploads)} files: {failed_uploads}")

        total_uploaded = sum(uploaded_files.values())
        logger.info(
            f"Upload completed: {total_uploaded}/{len(upload_tasks)} files uploaded successfully"
        )

        await self.s3_manager.put_json(task_manifest, 'metadata/task_manifest.json')
        logger.info("Dataset generation completed. Task manifest saved to S3.")
        return task_manifest


# ---------------------------------------------------------------------------
# AsyncWorker — processes one dataset (download -> estimate -> upload results)
# ---------------------------------------------------------------------------

class AsyncWorker:
    """Individual worker for processing datasets."""

    def __init__(self, worker_id: int, config: Config, s3_manager: S3Manager, device_id: int = None):
        self.worker_id = worker_id
        self.config = config
        self.s3_manager = s3_manager
        self.device_id = device_id
        self.processed_count = 0
        if device_id is not None and get_gpu_count() > 0:
            logger.info(f"Worker {worker_id}: using physical GPU {device_id}")
        else:
            logger.info(f"Worker {worker_id}: Initialized (CPU mode)")

    async def process_single_dataset(self, task: Dict) -> Optional[Dict]:
        """
        Download dataset, run all estimators, upload results.

        Result dict keys:
          config_id, run_id, worker_id, timestamp,
          naive_logit, logit_2sri,
          nn_2sri, nn_2sri_ci_lo, nn_2sri_ci_hi,
          naive_nn, naive_nn_ci_lo, naive_nn_ci_hi,
          params
        """
        config_id = task['config_id']
        run_id = task['run_id']
        s3_key = task['s3_key']
        local_dataset_path = (
            f"/tmp/binary_worker_{self.worker_id}_dataset_{config_id}_{run_id}.pkl"
        )

        logger.info(f"Worker {self.worker_id}: Starting to process {s3_key}")

        try:
            # Download dataset
            success = await self.s3_manager.download_object(s3_key, local_dataset_path, decompress=True)
            if not success:
                raise Exception(f"Failed to download dataset {s3_key}")

            with open(local_dataset_path, 'rb') as f:
                dataset = pickle.load(f)

            df_x = dataset['df_x']
            df_y = dataset['df_y']
            logger.info(
                f"Worker {self.worker_id}: Dataset loaded, "
                f"df_x shape: {df_x.shape}, df_y shape: {df_y.shape}, "
                f"Y mean: {df_y['Y'].mean():.3f}"
            )

            nn_kwargs = dict(
                epochs=self.config.EPOCHS,
                lr=self.config.LEARNING_RATE,
                dropout=self.config.DROPOUT
            )

            # --- NN Ensemble ---
            logger.info(
                f"Worker {self.worker_id}: Starting NN ensemble "
                f"({self.config.ENSEMBLE_SIZE} models, "
                f"{self.config.MAX_INNER_WORKERS} inner workers) for {s3_key}"
            )
            loop = asyncio.get_running_loop()

            mean_2sri, mean_naive, sri_raw, naive_raw = await loop.run_in_executor(
                None,
                lambda: ensemble_nn_binary(
                    df_x, df_y,
                    M=self.config.ENSEMBLE_SIZE,
                    max_workers=self.config.MAX_INNER_WORKERS,
                    **nn_kwargs
                )
            )
            logger.info(
                f"Worker {self.worker_id}: NN ensemble completed: "
                f"nn_2sri={mean_2sri}, naive_nn={mean_naive}"
            )

            # Save raw ensemble estimates
            ensemble_estimates = {'nn_2sri': sri_raw, 'naive_nn': naive_raw}
            ensemble_csv = create_estimates_csv(
                ensemble_estimates, config_id, run_id, self.worker_id, 'ensemble'
            )
            ensemble_s3_key = f"raw_estimates/config_{config_id}/run_{run_id}_ensemble.csv"
            await save_estimates_csv(self.s3_manager, ensemble_csv, ensemble_s3_key)
            logger.info(
                f"Worker {self.worker_id}: Saved {len(sri_raw)} ensemble estimates to {ensemble_s3_key}"
            )

            # --- NN Bootstrap CI ---
            logger.info(
                f"Worker {self.worker_id}: Starting NN bootstrap CI "
                f"({self.config.BOOTSTRAPS} samples) for {s3_key}"
            )

            ci_results = await loop.run_in_executor(
                None,
                lambda: bootstrap_ci_ensemble_binary(
                    (mean_2sri, mean_naive),
                    df_x, df_y,
                    B=self.config.BOOTSTRAPS,
                    max_workers=self.config.MAX_INNER_WORKERS,
                    **nn_kwargs
                )
            )
            logger.info(f"Worker {self.worker_id}: NN bootstrap CI completed")

            # Save raw bootstrap estimates
            bootstrap_estimates = {
                'nn_2sri': ci_results['raw_sri'],
                'naive_nn': ci_results['raw_naive']
            }
            bootstrap_csv = create_estimates_csv(
                bootstrap_estimates, config_id, run_id, self.worker_id, 'bootstrap_ensemble'
            )
            bootstrap_s3_key = (
                f"raw_estimates/config_{config_id}/run_{run_id}_bootstrap_ensemble.csv"
            )
            await save_estimates_csv(self.s3_manager, bootstrap_csv, bootstrap_s3_key)

            # --- Linear / Logistic estimators (no CI — convex, cheap) ---
            logger.info(
                f"Worker {self.worker_id}: Starting logistic model analysis for {s3_key}"
            )
            naive_logit = run_naive_logistic(df_y)
            logger.info(f"Worker {self.worker_id}: Naive logit completed: {naive_logit}")

            logit_2sri = run_logit_2sri(df_x, df_y)
            logger.info(f"Worker {self.worker_id}: Logit 2SRI completed: {logit_2sri}")

            # Compile results
            results = {
                'config_id': config_id,
                'run_id': run_id,
                'worker_id': self.worker_id,
                'timestamp': datetime.now().isoformat(),
                'naive_logit': naive_logit,
                'logit_2sri': logit_2sri,
                'nn_2sri': mean_2sri,
                'nn_2sri_ci_lo': ci_results['nn_2sri_ci_lo'],
                'nn_2sri_ci_hi': ci_results['nn_2sri_ci_hi'],
                'naive_nn': mean_naive,
                'naive_nn_ci_lo': ci_results['naive_nn_ci_lo'],
                'naive_nn_ci_hi': ci_results['naive_nn_ci_hi'],
                'params': dataset['params']
            }

            results = convert_to_json_serializable(results)
            result_s3_key = f"results/config_{config_id}/run_{run_id}_results.json"
            await self.s3_manager.put_json(results, result_s3_key)

            os.remove(local_dataset_path)
            self.processed_count += 1
            logger.info(
                f"Worker {self.worker_id}: Completed {s3_key} ({self.processed_count} total)"
            )
            return results

        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Failed to process {s3_key}: {e}")
            if os.path.exists(local_dataset_path):
                os.remove(local_dataset_path)
            return None


# ---------------------------------------------------------------------------
# ParallelExecutor
# ---------------------------------------------------------------------------

class ParallelExecutor:
    """Main orchestrator for parallel execution."""

    def __init__(self, config: Config):
        self.config = config

    async def execute_all_tasks(self) -> None:
        """Execute all simulation tasks in parallel."""
        load_fixed_effects()
        logger.info("Fixed effects loaded for parallel execution")

        async with S3Manager(
            self.config.S3_BUCKET, self.config.AWS_REGION, self.config.BASE_S3_PATH
        ) as s3_manager:
            tracker = ProgressTracker(s3_manager)

            # Rebuild checkpoint from existing results
            await tracker.rebuild_from_results()
            checkpoint = await tracker.load_checkpoint()

            manifest = await s3_manager.get_json('metadata/task_manifest.json')
            if manifest is None:
                logger.error("Task manifest not found. Please run dataset generation first.")
                return

            # Optional: restrict to a subset of config IDs (comma-separated env var)
            # e.g. CONFIG_IDS=0,1,2,3 — lets two machines split the work cleanly
            _config_ids_env = os.getenv("CONFIG_IDS")
            allowed_config_ids = (
                {int(x) for x in _config_ids_env.split(",")}
                if _config_ids_env else None
            )
            if allowed_config_ids:
                logger.info(f"CONFIG_IDS filter active: processing configs {sorted(allowed_config_ids)}")

            all_tasks = []
            for config_id, config_info in manifest['configs'].items():
                if allowed_config_ids and int(config_id) not in allowed_config_ids:
                    continue
                for run_id in range(config_info['datasets_generated']):
                    task_key = f"config_{config_id}_run_{run_id}"
                    if task_key not in checkpoint['completed_tasks']:
                        all_tasks.append({
                            'config_id': int(config_id),
                            'run_id': run_id,
                            's3_key': f"datasets/config_{config_id}/dataset_{run_id}.pkl",
                            'task_key': task_key
                        })

            logger.info(f"Starting parallel execution: {len(all_tasks)} tasks remaining")

            task_queue = asyncio.Queue()
            for task in all_tasks:
                await task_queue.put(task)

            workers = []
            GPU_COUNT = get_gpu_count()
            for worker_id in range(self.config.MAX_OUTER_WORKERS):
                device_id = worker_id if GPU_COUNT > 0 else None
                worker = AsyncWorker(worker_id, self.config, s3_manager, device_id=device_id)
                workers.append(worker)

            async def worker_loop(worker: AsyncWorker):
                tasks_processed = 0
                while True:
                    try:
                        task = await asyncio.wait_for(task_queue.get(), timeout=10.0)
                        result = await worker.process_single_dataset(task)
                        if result is not None:
                            tasks_processed += 1
                        task_queue.task_done()
                    except asyncio.TimeoutError:
                        break
                    except Exception as e:
                        logger.error(f"Worker {worker.worker_id} error: {e}")
                        break
                logger.info(
                    f"Worker {worker.worker_id} finished: {tasks_processed} tasks processed"
                )
                return tasks_processed

            async def progress_monitor():
                while not task_queue.empty():
                    await tracker.log_progress(
                        len(all_tasks),
                        len(all_tasks) - task_queue.qsize(),
                        0
                    )
                    await asyncio.sleep(30)

            worker_tasks = [asyncio.create_task(worker_loop(w)) for w in workers]
            monitor_task = asyncio.create_task(progress_monitor())

            await task_queue.join()
            monitor_task.cancel()

            for wt in worker_tasks:
                wt.cancel()

            worker_results = await asyncio.gather(*worker_tasks, return_exceptions=True)
            total_processed = sum(r for r in worker_results if isinstance(r, int))
            logger.info(f"Execution completed: {total_processed} tasks processed in this run")

    async def run_full_pipeline(self) -> None:
        """Run the complete pipeline: generate datasets + execute tasks."""
        async with S3Manager(
            self.config.S3_BUCKET, self.config.AWS_REGION, self.config.BASE_S3_PATH
        ) as s3_manager:
            load_fixed_effects()

            logger.info("Starting binary dataset generation...")
            generator = DatasetGenerator(self.config, s3_manager)
            await generator.generate_all_datasets()

            logger.info("Starting parallel task execution...")
            await self.execute_all_tasks()

            logger.info("Full pipeline completed!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # CRITICAL: spawn for CUDA compatibility (must be first)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    config = Config()

    # Verify AWS credentials
    try:
        boto3.Session().get_credentials()
        logger.info("AWS credentials found")
    except NoCredentialsError:
        logger.error("AWS credentials not found. Please configure them.")
        sys.exit(1)

    # Log configuration
    logger.info("=" * 60)
    GPU_COUNT = get_gpu_count()
    if GPU_COUNT > 0:
        logger.info("MULTI-GPU CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"GPUs Detected: {GPU_COUNT}")
        for i in range(GPU_COUNT):
            logger.info(
                f"  GPU {i}: {torch.cuda.get_device_name(i)} "
                f"({torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.1f} GB)"
            )
        logger.info(f"Max Outer Workers (GPU Workers): {config.MAX_OUTER_WORKERS}")
        logger.info(f"Max Inner Workers (Models per GPU): {config.MAX_INNER_WORKERS}")
        logger.info(
            f"Total GPU Concurrency: "
            f"{config.MAX_OUTER_WORKERS} GPUs × {config.MAX_INNER_WORKERS} models = "
            f"{config.MAX_OUTER_WORKERS * config.MAX_INNER_WORKERS} concurrent NN trainings"
        )
        logger.info(f"PyTorch Version: {torch.__version__}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
    else:
        logger.info("CPU-ONLY CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"CPU Count: {mp.cpu_count()}")
        logger.info(f"Max Outer Workers: {config.MAX_OUTER_WORKERS}")
        logger.info(f"Max Inner Workers: {config.MAX_INNER_WORKERS}")
        logger.info(f"PyTorch Threads per Task: 2")
        logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info("=" * 60)

    if len(sys.argv) > 1:
        if sys.argv[1] == "generate":
            async def run_generation():
                async with S3Manager(
                    config.S3_BUCKET, config.AWS_REGION, config.BASE_S3_PATH
                ) as s3_manager:
                    load_fixed_effects()
                    generator = DatasetGenerator(config, s3_manager)
                    await generator.generate_all_datasets()

            asyncio.run(run_generation())

        elif sys.argv[1] == "execute":
            executor = ParallelExecutor(config)
            asyncio.run(executor.execute_all_tasks())

        elif sys.argv[1] == "full":
            executor = ParallelExecutor(config)
            asyncio.run(executor.run_full_pipeline())

        else:
            print("Usage: python aws_binary_iv_parallel.py [generate|execute|full]")
    else:
        print("Usage: python aws_binary_iv_parallel.py [generate|execute|full]")
        print("  generate: Generate and upload datasets only")
        print("  execute:  Execute simulation tasks only (requires existing datasets)")
        print("  full:     Run complete pipeline (generate + execute)")
