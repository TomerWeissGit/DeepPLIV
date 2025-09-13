"""
AWS S3 Parallel Execution System for Genetic IV Simulation
Optimized for large-scale parallel processing on EC2 with S3 storage
"""

import os
import json
import pickle
import gzip
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

# AWS and async dependencies
import boto3
import aioboto3
from botocore.exceptions import NoCredentialsError

# Local imports
from core.trainer import DeepPLIV


# Configuration
@dataclass
class Config:
    # AWS Settings
    S3_URI: str = os.getenv("S3_URI")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET: str = None  # Will be parsed from S3_URI
    BASE_S3_PATH: str = None  # Will be parsed from S3_URI

    # Parse S3 URI to extract bucket and base path
    def __post_init_s3(self):
        if self.S3_URI:
            # Parse s3://bucket-name/path/to/folder/ format
            if self.S3_URI.startswith("s3://"):
                parts = self.S3_URI[5:].split("/", 1)
                self.S3_BUCKET = parts[0]
                self.BASE_S3_PATH = parts[1].rstrip("/") if len(parts) > 1 else ""
                # Add MR_SIM to the path
                if self.BASE_S3_PATH:
                    self.BASE_S3_PATH = f"{self.BASE_S3_PATH}/MR_SIM"
                else:
                    self.BASE_S3_PATH = "MR_SIM"
            else:
                raise ValueError(f"Invalid S3_URI format: {self.S3_URI}. Expected format: s3://bucket-name/path/")
        else:
            # Fallback to old format
            self.S3_BUCKET = os.getenv("S3_BUCKET")
            self.BASE_S3_PATH = "MR_SIM"

        # Fix region format (replace underscores with hyphens)
        if self.AWS_REGION:
            self.AWS_REGION = self.AWS_REGION.replace("_", "-")

    # Execution Mode
    LOCAL_MODE: bool = os.getenv("LOCAL_MODE", "false").lower() == "true"

    # Simulation Parameters (from env vars with defaults)
    ENSEMBLE_SIZE: int = int(os.getenv("ENSEMBLE_SIZE", "50"))
    BOOTSTRAPS: int = int(os.getenv("BOOTSTRAPS", "200"))
    NUM_SIMULATIONS: int = int(os.getenv("NUM_SIMULATIONS", "200"))
    CI_LEVEL: Tuple[float, float] = (2.5, 97.5)

    # Model Parameters (reduced for local testing)
    EPOCHS: int = 1000
    LEARNING_RATE: float = 0.01
    DROPOUT: float = 0.01

    # Processing Optimization
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "32"))
    BATCH_SIZE: int = 8  # Datasets per worker batch

    # Parameter Grid
    N_VALUES: List[int] = None
    RHO_VALUES: List[float] = None
    P01Y_VALUES: List[float] = None
    GAMMA_U_VALUES: List[float] = None
    BETA_U_VALUES: List[float] = None

    def __post_init__(self):
        # Parse S3 configuration first
        self.__post_init_s3()

        if self.N_VALUES is None:
            # Smaller values for local testing
            self.N_VALUES = [10000, 40000]
        if self.RHO_VALUES is None:
            self.RHO_VALUES = [0.5]
        if self.P01Y_VALUES is None:
            self.P01Y_VALUES = [0.2, 0.4]
        if self.GAMMA_U_VALUES is None:
            self.GAMMA_U_VALUES = [1]
        if self.BETA_U_VALUES is None:
            self.BETA_U_VALUES = [0, 2]

        # Log configuration mode
        mode = "LOCAL TESTING" if self.LOCAL_MODE else "FULL AWS"
        logger.info(f"Configuration loaded for {mode} mode")
        logger.info(f"S3 Configuration: Bucket={self.S3_BUCKET}, Path={self.BASE_S3_PATH}, Region={self.AWS_REGION}")
        logger.info(
            f"Parameters: ENSEMBLE_SIZE={self.ENSEMBLE_SIZE}, BOOTSTRAPS={self.BOOTSTRAPS}, NUM_SIMULATIONS={self.NUM_SIMULATIONS}, MAX_WORKERS={self.MAX_WORKERS}")


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global parameters (loaded from pickle)
gamma_j1 = None
gamma_j2 = None
gamma_jm12 = None
gamma_j1121 = None

# Simulation constants
K1 = K2 = 7
p01 = p02 = 0.4
beta_x = 2


class S3Manager:
    """Async S3 operations manager"""

    def __init__(self, bucket_name: str, region: str, base_path: str):
        self.bucket_name = bucket_name
        self.region = region
        self.base_path = base_path
        self.session = None

    async def __aenter__(self):
        self.session = aioboto3.Session()
        self.s3_client = await self.session.client('s3', region_name=self.region).__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.s3_client.__aexit__(exc_type, exc_val, exc_tb)

    def _get_s3_key(self, relative_path: str) -> str:
        """Convert relative path to full S3 key"""
        return f"{self.base_path}/{relative_path}"

    async def upload_object(self, local_path: str, s3_key: str, compress: bool = True) -> bool:
        """Upload object to S3 with optional compression"""
        try:
            full_key = self._get_s3_key(s3_key)

            if compress and not s3_key.endswith('.gz'):
                # Compress data in memory
                with open(local_path, 'rb') as f:
                    data = f.read()
                compressed_data = gzip.compress(data)
                full_key += '.gz'

                await self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=full_key,
                    Body=compressed_data
                )
            else:
                await self.s3_client.upload_file(local_path, self.bucket_name, full_key)

            return True
        except Exception as e:
            logger.error(f"Failed to upload {s3_key}: {e}")
            return False

    async def download_object(self, s3_key: str, local_path: str, decompress: bool = True) -> bool:
        """Download object from S3 with optional decompression"""
        try:
            full_key = self._get_s3_key(s3_key)

            # If decompressing and file doesn't end with .gz, try with .gz extension
            if decompress and not full_key.endswith('.gz'):
                gz_key = full_key + '.gz'
                try:
                    # Try to get the compressed version first
                    response = await self.s3_client.get_object(Bucket=self.bucket_name, Key=gz_key)
                    compressed_data = await response['Body'].read()
                    data = gzip.decompress(compressed_data)

                    with open(local_path, 'wb') as f:
                        f.write(data)
                    return True
                except:
                    # If compressed version doesn't exist, fall back to uncompressed
                    pass

            if decompress and full_key.endswith('.gz'):
                response = await self.s3_client.get_object(Bucket=self.bucket_name, Key=full_key)
                compressed_data = await response['Body'].read()
                data = gzip.decompress(compressed_data)

                with open(local_path, 'wb') as f:
                    f.write(data)
            else:
                await self.s3_client.download_file(self.bucket_name, full_key, local_path)

            return True
        except Exception as e:
            logger.error(f"Failed to download {s3_key}: {e}")
            return False

    async def list_objects(self, prefix: str) -> List[str]:
        """List objects with given prefix"""
        try:
            full_prefix = self._get_s3_key(prefix)
            objects = []

            paginator = self.s3_client.get_paginator('list_objects_v2')
            async for page in paginator.paginate(Bucket=self.bucket_name, Prefix=full_prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        # Remove base path to get relative path
                        key = obj['Key']
                        if key.startswith(f"{self.base_path}/"):
                            relative_key = key[len(f"{self.base_path}/"):]
                            objects.append(relative_key)

            return objects
        except Exception as e:
            logger.error(f"Failed to list objects with prefix {prefix}: {e}")
            return []

    async def put_json(self, data: Dict, s3_key: str) -> bool:
        """Upload JSON data to S3"""
        try:
            full_key = self._get_s3_key(s3_key)
            json_data = json.dumps(data, indent=2)

            await self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=full_key,
                Body=json_data.encode('utf-8'),
                ContentType='application/json'
            )
            return True
        except Exception as e:
            logger.error(f"Failed to upload JSON {s3_key}: {e}")
            return False

    async def get_json(self, s3_key: str) -> Optional[Dict]:
        """Download JSON data from S3"""
        try:
            full_key = self._get_s3_key(s3_key)
            response = await self.s3_client.get_object(Bucket=self.bucket_name, Key=full_key)
            data = await response['Body'].read()
            return json.loads(data.decode('utf-8'))
        except Exception as e:
            logger.error(f"Failed to download JSON {s3_key}: {e}")
            return None


# Simulation functions (copied and adapted from example_genetic_iv.py)
def simulate_snp_block(n, K, p0, rho):
    Z = np.random.normal(size=n)
    snps = np.zeros((n, K))
    for j in range(K):
        eps = np.random.normal(size=n)
        p = norm.cdf(norm.ppf(p0) + np.sqrt(rho) * Z + np.sqrt(1 - rho) * eps)
        snps[:, j] = np.random.binomial(2, p)
    return snps


def simulate_snp_dataset(n, p01, p02, rho):
    S1 = simulate_snp_block(n, K1, p01, rho)
    S2 = simulate_snp_block(n, K2, p02, rho)
    return S1, S2


def x_eq(S1, S2, n, gamma_u):
    global gamma_j1, gamma_j2, gamma_jm12, gamma_j1121

    interaction_term = np.sum(S1[:, :, None] * S2[:, None, :] * gamma_jm12[None, :, :], axis=(1, 2))
    non_linear_part = (S1 @ gamma_j1 + S2 @ gamma_j2 + interaction_term +
                       ((S1[:, 0] * S1[:, 1]) * (S2 @ gamma_j1121)))
    epsilon_x = np.random.normal(0, 1.0, size=n)
    u = np.random.normal(0, 1.0, size=n)
    return non_linear_part + epsilon_x + gamma_u * u, epsilon_x, u


def gen_data(n, p01, p02, rho, gamma_u, beta_u):
    S1, S2 = simulate_snp_dataset(n, p01, p02, rho)
    X, eps_x, u = x_eq(S1, S2, n, gamma_u)
    epsilon_y = np.random.normal(0, 1, size=n)
    Y = beta_x * X + beta_u * u + epsilon_y
    df = pd.DataFrame({
        **{f"S1_{j + 1}": S1[:, j] for j in range(K1)},
        **{f"S2_{j + 1}": S2[:, j] for j in range(K2)},
        "X": X,
        "Y": Y
    })
    return df


# Model execution functions
def run_naive_ols(df_y):
    X = sm.add_constant(df_y[["X"]])
    y = df_y["Y"]
    return sm.OLS(y, X).fit().params.iloc[1]


def run_2sps(df_x, df_y):
    Z = df_x[[f"S1_{j + 1}" for j in range(K1)] + [f"S2_{j + 1}" for j in range(K2)]]
    Z_Y = df_y[[f"S1_{j + 1}" for j in range(K1)] + [f"S2_{j + 1}" for j in range(K2)]]
    X_hat = LinearRegression().fit(Z, df_x["X"]).predict(Z_Y)
    return sm.OLS(df_y["Y"], sm.add_constant(X_hat)).fit().params.iloc[1]


def run_2sri(df_x, df_y):
    Z = df_x[[f"S1_{j + 1}" for j in range(K1)] + [f"S2_{j + 1}" for j in range(K2)]]
    Z_Y = df_y[[f"S1_{j + 1}" for j in range(K1)] + [f"S2_{j + 1}" for j in range(K2)]]
    X_hat = LinearRegression().fit(Z, df_x["X"]).predict(Z_Y)
    X_err = df_y["X"] - X_hat
    X_stk = sm.add_constant(np.column_stack((df_y["X"], X_err)))
    return sm.OLS(df_y["Y"], X_stk).fit().params.iloc[1]


def estimating_sri_sps_with_nn(df_x, df_y, epochs, lr, dropout):
    """Single NN estimation - memory optimized version"""
    model = DeepPLIV()

    Z_X = df_x[[f"S1_{j + 1}" for j in range(K1)] + [f"S2_{j + 1}" for j in range(K2)]].values
    Z_Y = df_y[[f"S1_{j + 1}" for j in range(K1)] + [f"S2_{j + 1}" for j in range(K2)]].values

    X = df_x["X"].values
    X_Y = df_y["X"].values
    Y = df_y["Y"].values
    dummy1 = np.ones((X_Y.shape[0], 1))

    try:
        # First stage
        model.fit_first_stage(Z_X, X, epochs_first_stage=epochs,
                              learning_rate_first_stage=lr,
                              dropout=dropout,
                              validation_data=(Z_Y, X_Y))

        x_pred = model.first_stage_model.predict(Z_Y).reshape(-1, 1)
        x_err = X_Y.reshape(-1, 1) - x_pred

        # Second stage models
        m_sri = model.fit_second_stage(X_Y.reshape(-1, 1), x_err, Y.reshape(-1, 1),
                                       epochs_second_stage=epochs, learning_rate_second_stage=lr,
                                       dropout=dropout)

        m_sps = model.fit_second_stage(x_pred, dummy1, Y.reshape(-1, 1),
                                       epochs_second_stage=epochs, learning_rate_second_stage=lr,
                                       dropout=dropout)

        m_naive_feed_forward = model.fit_second_stage(X_Y.reshape(-1, 1), dummy1, Y.reshape(-1, 1),
                                                      epochs_second_stage=epochs, learning_rate_second_stage=lr,
                                                      dropout=dropout)

        results = (
            m_sps.final_layer.weight.detach().numpy()[0, 0],
            m_sri.final_layer.weight.detach().numpy()[0, 0],
            m_naive_feed_forward.final_layer.weight.detach().numpy()[0, 0]
        )

        # Explicit cleanup
        del model, m_sri, m_sps, m_naive_feed_forward
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    except Exception as e:
        logger.error(f"NN estimation failed: {e}")
        return None, None, None


async def ensemble_nn(df_x, df_y, M, **nn_kwargs):
    """Async ensemble NN estimation"""
    loop = asyncio.get_event_loop()

    def run_single_nn():
        return estimating_sri_sps_with_nn(df_x, df_y, **nn_kwargs)

    tasks = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for _ in range(M):
            task = loop.run_in_executor(executor, run_single_nn)
            tasks.append(task)

        results = await asyncio.gather(*tasks)

    # Filter out failed results
    valid_results = [r for r in results if r[0] is not None]
    if not valid_results:
        return None, None, None

    sps_list = [r[0] for r in valid_results]
    sri_list = [r[1] for r in valid_results]
    naive_list = [r[2] for r in valid_results]

    return np.mean(sps_list), np.mean(sri_list), np.mean(naive_list)


async def bootstrap_ci(base_func, df_x, df_y, B, ci_level):
    """Async bootstrap confidence interval for linear models"""
    if B == 1:
        est = base_func(df_x, df_y)
        return est, (est, est)

    loop = asyncio.get_event_loop()
    estimates = []
    n_x, n_y = len(df_x), len(df_y)

    def bootstrap_sample():
        idx_x = np.random.choice(n_x, n_x, replace=True)
        idx_y = np.random.choice(n_y, n_y, replace=True)
        return base_func(df_x.iloc[idx_x], df_y.iloc[idx_y])

    with ThreadPoolExecutor(max_workers=8) as executor:
        tasks = [loop.run_in_executor(executor, bootstrap_sample) for _ in range(B)]
        estimates = await asyncio.gather(*tasks)

    ci_low, ci_high = np.percentile(estimates, ci_level)
    return np.mean(estimates), (ci_low, ci_high)


async def bootstrap_ci_ensemble(ensemble_estimates, df_x, df_y, B, **nn_kwargs):
    """
    Bootstrap confidence interval for ensemble NN estimates
    Uses 95% quantile of absolute errors from ensemble estimate

    Args:
        ensemble_estimates: Tuple of (sps_ensemble, sri_ensemble, naive_ensemble)
        df_x, df_y: Original datasets
        B: Number of bootstrap samples (each runs single model, not ensemble)
        **nn_kwargs: Neural network parameters

    Returns:
        Tuple of (sps_ci, sri_ci, naive_ci) where each is (low, high)
    """
    sps_ensemble, sri_ensemble, naive_ensemble = ensemble_estimates

    if B == 1:
        return (sps_ensemble, sps_ensemble), (sri_ensemble, sri_ensemble), (naive_ensemble, naive_ensemble)

    sps_errors = []
    sri_errors = []
    naive_errors = []
    n_x, n_y = len(df_x), len(df_y)

    async def bootstrap_single_sample():
        # Create bootstrap sample
        idx_x = np.random.choice(n_x, n_x, replace=True)
        idx_y = np.random.choice(n_y, n_y, replace=True)
        df_x_boot = df_x.iloc[idx_x].reset_index(drop=True)
        df_y_boot = df_y.iloc[idx_y].reset_index(drop=True)

        # Run single NN model on bootstrap sample (not ensemble)
        sps_boot, sri_boot, naive_boot = estimating_sri_sps_with_nn(
            df_x_boot, df_y_boot, **nn_kwargs
        )

        # Calculate absolute errors from original ensemble estimates
        sps_error = abs(sps_boot - sps_ensemble) if sps_boot is not None else 0
        sri_error = abs(sri_boot - sri_ensemble) if sri_boot is not None else 0
        naive_error = abs(naive_boot - naive_ensemble) if naive_boot is not None else 0

        return sps_error, sri_error, naive_error

    # Run bootstrap samples
    bootstrap_results = []
    for _ in range(B):
        result = await bootstrap_single_sample()
        bootstrap_results.append(result)

    # Extract errors
    for sps_err, sri_err, naive_err in bootstrap_results:
        sps_errors.append(sps_err)
        sri_errors.append(sri_err)
        naive_errors.append(naive_err)

    # Calculate 95% quantile errors
    error_95_sps = np.percentile(sps_errors, 95)
    error_95_sri = np.percentile(sri_errors, 95)
    error_95_naive = np.percentile(naive_errors, 95)

    # Construct confidence intervals: ensemble_est ± error_95%
    sps_ci = (sps_ensemble - error_95_sps, sps_ensemble + error_95_sps)
    sri_ci = (sri_ensemble - error_95_sri, sri_ensemble + error_95_sri)
    naive_ci = (naive_ensemble - error_95_naive, naive_ensemble + error_95_naive)

    return sps_ci, sri_ci, naive_ci


import os
import pickle
import boto3
import logging

logger = logging.getLogger(__name__)

def load_fixed_effects():
    """Load fixed effects from pickle file in S3"""
    global gamma_j1, gamma_j2, gamma_jm12, gamma_j1121

    # Get S3 URI from environment variable
    s3_uri = os.getenv("S3_URI") + "MR_SIM/"


    # Full file path
    file_uri = s3_uri + "fixed_effects.pkl"

    # Parse bucket and key
    _, _, bucket, *key_parts = file_uri.split("/")
    key = "/".join(key_parts)

    # Download object from S3
    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))
    obj = s3.get_object(Bucket=bucket, Key=key)
    effect_dict = pickle.loads(obj["Body"].read())

    # Assign globals
    gamma_j1 = effect_dict["gamma_j1"]
    gamma_j2 = effect_dict["gamma_j2"]
    gamma_jm12 = effect_dict["gamma_jm12"]
    gamma_j1121 = effect_dict["gamma_j1121"]

    logger.info("Fixed effects loaded successfully from %s", file_uri)


class DatasetGenerator:
    """Generate and upload datasets to S3"""

    def __init__(self, config: Config, s3_manager: S3Manager):
        self.config = config
        self.s3_manager = s3_manager

    def generate_single_dataset(self, config_params: Dict, run_id: int) -> Dict:
        """Generate a single dataset for given parameters"""
        n_val, rho_val, p01y_val, gamma_u_val, beta_u_val = (
            config_params['n'], config_params['rho'], config_params['p01y'],
            config_params['gamma_u'], config_params['beta_u']
        )

        # Generate training and prediction datasets
        df_x = gen_data(n_val, p01, p02, rho_val, gamma_u_val, beta_u_val)
        df_y = gen_data(n_val, p01y_val, p02, rho_val, gamma_u_val, beta_u_val)

        dataset = {
            'df_x': df_x,
            'df_y': df_y,
            'params': config_params,
            'run_id': run_id,
            'timestamp': datetime.now().isoformat()
        }

        return dataset

    def generate_single_dataset_to_file(self, config_id: int, config_params: Dict, run_id: int, temp_dir: str) -> str:
        """Generate a single dataset and save it to file with aggressive compression"""
        try:
            dataset = self.generate_single_dataset(config_params, run_id)

            # Save to compressed file
            local_path = f"{temp_dir}/dataset_{run_id}.pkl.gz"
            with gzip.open(local_path, 'wb', compresslevel=9) as f:
                # Use highest pickle protocol with maximum gzip compression
                pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Log file size for debugging
            file_size = os.path.getsize(local_path)
            logger.info(f"Generated config_{config_id}_run_{run_id}: {file_size/1024:.1f}KB (compressed)")

            return local_path
        except Exception as e:
            logger.error(f"Failed to generate dataset config_{config_id}_run_{run_id}: {e}")
            return None

    def generate_datasets_batch(self, config_id: int, config_params: Dict, start_run: int, end_run: int) -> List[str]:
        """Generate a batch of datasets and return their local paths"""
        local_paths = []
        temp_dir = f"/tmp/datasets_config_{config_id}"
        os.makedirs(temp_dir, exist_ok=True)

        for run_id in range(start_run, end_run):
            try:
                dataset = self.generate_single_dataset(config_params, run_id)

                # Save locally first
                local_path = f"{temp_dir}/dataset_{run_id}.pkl"
                with open(local_path, 'wb') as f:
                    pickle.dump(dataset, f)

                local_paths.append(local_path)

            except Exception as e:
                logger.error(f"Failed to generate dataset config_{config_id}_run_{run_id}: {e}")

        return local_paths

    async def upload_datasets_batch(self, config_id: int, local_paths: List[str]) -> int:
        """Upload batch of datasets to S3"""
        uploaded_count = 0

        for local_path in local_paths:
            try:
                run_id = int(local_path.split('_')[-1].split('.')[0])  # Extract run_id
                s3_key = f"datasets/config_{config_id}/dataset_{run_id}.pkl"

                success = await self.s3_manager.upload_object(local_path, s3_key, compress=True)
                if success:
                    uploaded_count += 1
                    # Clean up local file
                    os.remove(local_path)

            except Exception as e:
                logger.error(f"Failed to upload {local_path}: {e}")

        return uploaded_count

    async def generate_all_datasets(self) -> Dict:
        """Generate all datasets using multiprocessing and upload to S3"""
        from itertools import product

        all_configs = list(product(
            self.config.N_VALUES,
            self.config.RHO_VALUES,
            self.config.P01Y_VALUES,
            self.config.GAMMA_U_VALUES,
            self.config.BETA_U_VALUES
        ))

        logger.info(
            f"Generating {len(all_configs)} configurations × {self.config.NUM_SIMULATIONS} runs = {len(all_configs) * self.config.NUM_SIMULATIONS} total datasets")

        task_manifest = {
            'total_configs': len(all_configs),
            'total_datasets': len(all_configs) * self.config.NUM_SIMULATIONS,
            'configs': {},
            'generation_timestamp': datetime.now().isoformat()
        }

        # Process ALL configurations and datasets in parallel
        logger.info("Starting fully parallel dataset generation...")

        # Create all individual dataset generation tasks
        all_generation_tasks = []
        for config_id, (n_val, rho_val, p01y_val, gamma_u_val, beta_u_val) in enumerate(all_configs):
            config_params = {
                'n': n_val,
                'rho': rho_val,
                'p01y': p01y_val,
                'gamma_u': gamma_u_val,
                'beta_u': beta_u_val
            }

            # Create temp directory for this config
            temp_dir = f"/tmp/datasets_config_{config_id}"
            os.makedirs(temp_dir, exist_ok=True)

            # Add each dataset as a separate task
            for run_id in range(self.config.NUM_SIMULATIONS):
                all_generation_tasks.append((config_id, config_params, run_id, temp_dir))

        logger.info(f"Submitting {len(all_generation_tasks)} dataset generation tasks to thread pool...")

        # Generate ALL datasets in parallel using all CPU cores
        with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
            # Submit all tasks
            future_to_task = {}
            for config_id, config_params, run_id, temp_dir in all_generation_tasks:
                future = executor.submit(self.generate_single_dataset_to_file, config_id, config_params, run_id, temp_dir)
                future_to_task[future] = (config_id, run_id)

            # Collect results as they complete
            generated_files = {}  # config_id -> list of local paths
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
                        if completed_count % 4 == 0:  # Log progress every 4 completions
                            logger.info(f"Generated {completed_count}/{len(all_generation_tasks)} datasets")
                except Exception as e:
                    logger.error(f"Failed to generate dataset config_{config_id}_run_{run_id}: {e}")

        logger.info(f"Generation completed. Starting parallel upload to S3...")

        # Create all upload tasks for parallel execution
        upload_tasks = []
        for config_id, config_info in enumerate([(n_val, rho_val, p01y_val, gamma_u_val, beta_u_val)
                                                for n_val, rho_val, p01y_val, gamma_u_val, beta_u_val in all_configs]):
            if config_id in generated_files:
                for local_path in generated_files[config_id]:
                    # Extract run_id from filename (handle .pkl.gz extension)
                    filename = os.path.basename(local_path)
                    if filename.endswith('.pkl.gz'):
                        run_id = int(filename.replace('dataset_', '').replace('.pkl.gz', ''))
                        s3_key = f"datasets/config_{config_id}/dataset_{run_id}.pkl.gz"
                    else:
                        run_id = int(filename.replace('dataset_', '').replace('.pkl', ''))
                        s3_key = f"datasets/config_{config_id}/dataset_{run_id}.pkl"
                    upload_tasks.append((local_path, s3_key, config_id, run_id))

        logger.info(f"Starting parallel upload of {len(upload_tasks)} files to S3...")

        # Check file sizes for debugging
        total_size = 0
        for local_path, _, _, _ in upload_tasks:
            if os.path.exists(local_path):
                size = os.path.getsize(local_path)
                total_size += size
        logger.info(f"Total data to upload: {total_size / 1024 / 1024:.2f} MB")

        # Upload all files in parallel
        uploaded_files = {}  # config_id -> count
        failed_uploads = []
        upload_start_time = datetime.now()

        async def upload_single_file(local_path, s3_key, config_id, run_id):
            file_start = datetime.now()
            try:
                # Get file size for logging
                file_size = os.path.getsize(local_path) if os.path.exists(local_path) else 0

                # Don't double-compress if file is already compressed
                compress = not s3_key.endswith('.gz')
                success = await self.s3_manager.upload_object(local_path, s3_key, compress=compress)

                upload_time = (datetime.now() - file_start).total_seconds()

                if success:
                    # Clean up local file after successful upload
                    os.remove(local_path)
                    logger.info(f"Uploaded config_{config_id}_run_{run_id} ({file_size/1024:.1f}KB) in {upload_time:.2f}s")
                    return config_id, True
                else:
                    logger.error(f"Failed to upload config_{config_id}_run_{run_id} after {upload_time:.2f}s")
                    failed_uploads.append((config_id, run_id, s3_key))
                    return config_id, False
            except Exception as e:
                upload_time = (datetime.now() - file_start).total_seconds()
                logger.error(f"Exception uploading {local_path} to {s3_key} after {upload_time:.2f}s: {e}")
                failed_uploads.append((config_id, run_id, s3_key))
                return config_id, False

        # Execute all uploads concurrently
        upload_results = await asyncio.gather(*[
            upload_single_file(local_path, s3_key, config_id, run_id)
            for local_path, s3_key, config_id, run_id in upload_tasks
        ])

        # Count successful uploads per config
        for config_id, success in upload_results:
            if config_id not in uploaded_files:
                uploaded_files[config_id] = 0
            if success:
                uploaded_files[config_id] += 1

        # Update manifest and log results
        for config_id, config_info in enumerate([(n_val, rho_val, p01y_val, gamma_u_val, beta_u_val)
                                                for n_val, rho_val, p01y_val, gamma_u_val, beta_u_val in all_configs]):
            n_val, rho_val, p01y_val, gamma_u_val, beta_u_val = config_info
            config_params = {
                'n': n_val,
                'rho': rho_val,
                'p01y': p01y_val,
                'gamma_u': gamma_u_val,
                'beta_u': beta_u_val
            }

            uploaded_count = uploaded_files.get(config_id, 0)
            expected_count = len(generated_files.get(config_id, []))

            if uploaded_count > 0:
                logger.info(f"Config {config_id}: Uploaded {uploaded_count}/{expected_count} datasets")
            else:
                logger.warning(f"Config {config_id}: No datasets uploaded")

            # Update manifest
            task_manifest['configs'][config_id] = {
                'params': config_params,
                'datasets_generated': uploaded_count,
                's3_prefix': f'datasets/config_{config_id}/'
            }

            # Clean up temp directory
            temp_dir = f"/tmp/datasets_config_{config_id}"
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)

        if failed_uploads:
            logger.warning(f"Failed to upload {len(failed_uploads)} files: {failed_uploads}")

        total_uploaded = sum(uploaded_files.values())
        logger.info(f"Parallel upload completed: {total_uploaded}/{len(upload_tasks)} files uploaded successfully")

        # Save task manifest to S3
        await self.s3_manager.put_json(task_manifest, 'metadata/task_manifest.json')
        logger.info("Dataset generation completed. Task manifest saved to S3.")

        return task_manifest


def convert_to_json_serializable(obj):
    """Convert numpy types to JSON-serializable Python types"""
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


class AsyncWorker:
    """Individual worker for processing datasets"""

    def __init__(self, worker_id: int, config: Config, s3_manager: S3Manager):
        self.worker_id = worker_id
        self.config = config
        self.s3_manager = s3_manager
        self.processed_count = 0

    async def process_single_dataset(self, task: Dict) -> Dict:
        """Process a single dataset task"""
        config_id = task['config_id']
        run_id = task['run_id']
        s3_key = task['s3_key']
        local_dataset_path = f"/tmp/worker_{self.worker_id}_dataset_{config_id}_{run_id}.pkl"

        logger.info(f"Worker {self.worker_id}: Starting to process {s3_key}")

        try:
            # Download dataset
            logger.info(f"Worker {self.worker_id}: Downloading {s3_key}")
            success = await self.s3_manager.download_object(s3_key, local_dataset_path, decompress=True)

            if not success:
                raise Exception(f"Failed to download dataset {s3_key}")

            # Load dataset
            logger.info(f"Worker {self.worker_id}: Loading dataset from {local_dataset_path}")
            with open(local_dataset_path, 'rb') as f:
                dataset = pickle.load(f)

            df_x = dataset['df_x']
            df_y = dataset['df_y']
            logger.info(f"Worker {self.worker_id}: Dataset loaded, df_x shape: {df_x.shape}, df_y shape: {df_y.shape}")

            # Run NN ensemble (2 models for testing)
            logger.info(
                f"Worker {self.worker_id}: Starting NN ensemble ({self.config.ENSEMBLE_SIZE} models) for {s3_key}")
            try:
                sps_nn, sri_nn, naive_nn = await ensemble_nn(
                    df_x, df_y, self.config.ENSEMBLE_SIZE,
                    epochs=self.config.EPOCHS,
                    lr=self.config.LEARNING_RATE,
                    dropout=self.config.DROPOUT
                )
                logger.info(
                    f"Worker {self.worker_id}: NN ensemble completed: sps={sps_nn}, sri={sri_nn}, naive={naive_nn}")
            except Exception as e:
                logger.error(f"Worker {self.worker_id}: NN ensemble failed: {e}")
                raise

            # Run bootstrap CI for NN ensemble estimates (5 bootstrap samples for testing)
            logger.info(
                f"Worker {self.worker_id}: Starting NN bootstrap CI ({self.config.BOOTSTRAPS} samples) for {s3_key}")
            try:
                sps_ci, sri_ci, naive_ci = await bootstrap_ci_ensemble(
                    (sps_nn, sri_nn, naive_nn), df_x, df_y, self.config.BOOTSTRAPS,
                    epochs=self.config.EPOCHS,
                    lr=self.config.LEARNING_RATE,
                    dropout=self.config.DROPOUT
                )
                logger.info(f"Worker {self.worker_id}: NN bootstrap CI completed")
            except Exception as e:
                logger.error(f"Worker {self.worker_id}: NN bootstrap CI failed: {e}")
                raise

            # Run bootstrap for linear models (5 bootstrap samples for testing)
            logger.info(f"Worker {self.worker_id}: Starting linear model analysis for {s3_key}")
            try:
                ols_point = run_naive_ols(df_y)
                logger.info(f"Worker {self.worker_id}: OLS completed: {ols_point}")

                sls_point, (sls_low, sls_high) = await bootstrap_ci(
                    run_2sps, df_x, df_y, self.config.BOOTSTRAPS, self.config.CI_LEVEL
                )
                logger.info(f"Worker {self.worker_id}: 2SPS bootstrap completed: {sls_point}")

                sri_point, (sri_low, sri_high) = await bootstrap_ci(
                    run_2sri, df_x, df_y, self.config.BOOTSTRAPS, self.config.CI_LEVEL
                )
                logger.info(f"Worker {self.worker_id}: 2SRI bootstrap completed: {sri_point}")
            except Exception as e:
                logger.error(f"Worker {self.worker_id}: Linear model analysis failed: {e}")
                raise

            # Compile results
            results = {
                'config_id': config_id,
                'run_id': run_id,
                'worker_id': self.worker_id,
                'timestamp': datetime.now().isoformat(),
                'naive_ols': ols_point,
                'iv_2sps': sls_point,
                'iv_2sps_ci_lo': sls_low,
                'iv_2sps_ci_hi': sls_high,
                'iv_2sri': sri_point,
                'iv_2sri_ci_lo': sri_low,
                'iv_2sri_ci_hi': sri_high,
                'nn_2sps': sps_nn,
                'nn_2sps_ci_lo': sps_ci[0],
                'nn_2sps_ci_hi': sps_ci[1],
                'nn_2sri': sri_nn,
                'nn_2sri_ci_lo': sri_ci[0],
                'nn_2sri_ci_hi': sri_ci[1],
                'naive_nn': naive_nn,
                'naive_nn_ci_lo': naive_ci[0],
                'naive_nn_ci_hi': naive_ci[1],
                'params': dataset['params']
            }

            # Convert results to JSON-serializable format and upload
            results = convert_to_json_serializable(results)
            result_s3_key = f"results/config_{config_id}/run_{run_id}_results.json"
            await self.s3_manager.put_json(results, result_s3_key)

            # Cleanup
            os.remove(local_dataset_path)
            self.processed_count += 1

            logger.info(f"Worker {self.worker_id}: Completed {s3_key} ({self.processed_count} total)")
            return results

        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Failed to process {s3_key}: {e}")
            # Cleanup on error
            if os.path.exists(local_dataset_path):
                os.remove(local_dataset_path)
            return None


class ProgressTracker:
    """Track and checkpoint progress"""

    def __init__(self, s3_manager: S3Manager):
        self.s3_manager = s3_manager
        self.start_time = datetime.now()

    async def load_checkpoint(self) -> Dict:
        """Load existing checkpoint from S3"""
        checkpoint = await self.s3_manager.get_json('checkpoints/progress.json')
        if checkpoint is None:
            checkpoint = {
                'completed_tasks': [],
                'failed_tasks': [],
                'start_time': self.start_time.isoformat(),
                'last_update': self.start_time.isoformat()
            }
        return checkpoint

    async def save_checkpoint(self, completed_tasks: List[str], failed_tasks: List[str]) -> None:
        """Save current progress to S3"""
        checkpoint = {
            'completed_tasks': completed_tasks,
            'failed_tasks': failed_tasks,
            'start_time': self.start_time.isoformat(),
            'last_update': datetime.now().isoformat(),
            'total_completed': len(completed_tasks),
            'total_failed': len(failed_tasks)
        }

        await self.s3_manager.put_json(checkpoint, 'checkpoints/progress.json')

    async def log_progress(self, total_tasks: int, completed: int, failed: int) -> None:
        """Log current progress"""
        elapsed = datetime.now() - self.start_time
        rate = completed / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
        remaining = total_tasks - completed - failed

        eta_seconds = remaining / rate if rate > 0 else 0
        eta = str(pd.Timedelta(seconds=eta_seconds))

        logger.info(f"Progress: {completed}/{total_tasks} completed, {failed} failed, {rate:.2f} tasks/sec, ETA: {eta}")


class ParallelExecutor:
    """Main orchestrator for parallel execution"""

    def __init__(self, config: Config):
        self.config = config

    async def execute_all_tasks(self) -> None:
        """Execute all simulation tasks in parallel"""
        # Load fixed effects before starting workers
        load_fixed_effects()
        logger.info("Fixed effects loaded for parallel execution")

        async with S3Manager(self.config.S3_BUCKET, self.config.AWS_REGION, self.config.BASE_S3_PATH) as s3_manager:
            # Initialize progress tracker
            tracker = ProgressTracker(s3_manager)
            checkpoint = await tracker.load_checkpoint()

            # Load task manifest
            manifest = await s3_manager.get_json('metadata/task_manifest.json')
            if manifest is None:
                logger.error("Task manifest not found. Please run dataset generation first.")
                return

            # Build task queue
            all_tasks = []
            for config_id, config_info in manifest['configs'].items():
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

            # Create task queue
            task_queue = asyncio.Queue()
            for task in all_tasks:
                await task_queue.put(task)

            # Create workers
            workers = []
            for worker_id in range(self.config.MAX_WORKERS):
                worker = AsyncWorker(worker_id, self.config, s3_manager)
                workers.append(worker)

            # Worker coroutines
            async def worker_loop(worker: AsyncWorker):
                completed_tasks = []
                failed_tasks = []

                while True:
                    try:
                        task = await asyncio.wait_for(task_queue.get(), timeout=10.0)
                        result = await worker.process_single_dataset(task)

                        if result is not None:
                            completed_tasks.append(task['task_key'])
                        else:
                            failed_tasks.append(task['task_key'])

                        task_queue.task_done()

                        # Periodic checkpoint
                        if (len(completed_tasks) + len(failed_tasks)) % 10 == 0:
                            await tracker.save_checkpoint(
                                checkpoint['completed_tasks'] + completed_tasks,
                                checkpoint['failed_tasks'] + failed_tasks
                            )

                    except asyncio.TimeoutError:
                        break
                    except Exception as e:
                        logger.error(f"Worker {worker.worker_id} error: {e}")
                        break

                return completed_tasks, failed_tasks

            # Start all workers as tasks
            worker_tasks = [asyncio.create_task(worker_loop(worker)) for worker in workers]

            # Progress monitoring
            async def progress_monitor():
                while not task_queue.empty():
                    await tracker.log_progress(
                        len(all_tasks),
                        len(all_tasks) - task_queue.qsize(),
                        0  # We'll track failures separately
                    )
                    await asyncio.sleep(30)  # Update every 30 seconds

            monitor_task = asyncio.create_task(progress_monitor())

            # Wait for all tasks to complete
            await task_queue.join()
            monitor_task.cancel()

            # Cancel any remaining worker tasks
            for task in worker_tasks:
                task.cancel()

            # Collect results from workers
            all_completed = checkpoint['completed_tasks'].copy()
            all_failed = checkpoint['failed_tasks'].copy()

            worker_results = await asyncio.gather(*worker_tasks, return_exceptions=True)
            for result in worker_results:
                if isinstance(result, tuple):
                    completed, failed = result
                    all_completed.extend(completed)
                    all_failed.extend(failed)

            # Final checkpoint
            await tracker.save_checkpoint(all_completed, all_failed)

            logger.info(f"Execution completed: {len(all_completed)} successful, {len(all_failed)} failed")

    async def run_full_pipeline(self) -> None:
        """Run the complete pipeline: generate datasets + execute tasks"""
        async with S3Manager(self.config.S3_BUCKET, self.config.AWS_REGION, self.config.BASE_S3_PATH) as s3_manager:
            # Load fixed effects
            load_fixed_effects()

            # Step 1: Generate and upload datasets
            logger.info("Starting dataset generation...")
            generator = DatasetGenerator(self.config, s3_manager)
            await generator.generate_all_datasets()

            # Step 2: Execute all tasks
            logger.info("Starting parallel task execution...")
            await self.execute_all_tasks()

            logger.info("Full pipeline completed!")


if __name__ == "__main__":
    import sys

    config = Config()

    # Verify AWS credentials
    try:
        boto3.Session().get_credentials()
        logger.info("AWS credentials found")
    except NoCredentialsError:
        logger.error("AWS credentials not found. Please configure them.")
        sys.exit(1)

    if len(sys.argv) > 1:
        if sys.argv[1] == "generate":
            # Generate datasets only
            async def run_generation():
                async with S3Manager(config.S3_BUCKET, config.AWS_REGION, config.BASE_S3_PATH) as s3_manager:
                    load_fixed_effects()
                    generator = DatasetGenerator(config, s3_manager)
                    await generator.generate_all_datasets()


            asyncio.run(run_generation())

        elif sys.argv[1] == "execute":
            # Execute tasks only (assumes datasets already generated)
            executor = ParallelExecutor(config)
            asyncio.run(executor.execute_all_tasks())

        elif sys.argv[1] == "full":
            # Run full pipeline
            executor = ParallelExecutor(config)
            asyncio.run(executor.run_full_pipeline())

        else:
            print("Usage: python aws_genetic_iv_parallel.py [generate|execute|full]")
    else:
        print("Usage: python aws_genetic_iv_parallel.py [generate|execute|full]")
        print("  generate: Generate and upload datasets only")
        print("  execute:  Execute simulation tasks only (requires existing datasets)")
        print("  full:     Run complete pipeline (generate + execute)")
