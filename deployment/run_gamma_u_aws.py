"""
run_gamma_u_aws.py — Distributed gamma_U study via SQS + S3.

Tasks are individual fits (logit_point, logit_boot, nn_ens, nn_boot).
Workers pull from SQS, run the fit, write result to S3, delete message.
Crash-safe: SQS re-queues any message held by a dead worker (600s timeout).
Idempotent: workers skip tasks whose S3 result already exists.

Total tasks: 200 reps × 3 gamma_U × (1 + 200 + 10 + 200) = 246,600

Subcommands:
  populate   Create SQS queue + S3 bucket and fill with all pending tasks
  worker     Run N workers pulling from SQS (run on each EC2 instance)
  progress   Count completed results in S3
  collect    Download S3 results → local gamma_u_ci_results/*.json

Typical flow:
  # Once (locally or on one instance):
  python run_gamma_u_aws.py populate

  # On each of 4 instances:
  PYTHONPATH=/home/ec2-user python3.11 run_gamma_u_aws.py worker --workers 190

  # When done:
  python run_gamma_u_aws.py collect
"""

import os
import sys
import json
import time
import argparse
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, LogisticRegression

warnings.filterwarnings("ignore")

# ── AWS config ────────────────────────────────────────────────────────────────

AWS_REGION = "eu-central-1"
S3_BUCKET  = "tomer-gamma-u-study"
SQS_NAME   = "gamma-u-tasks"

# ── DGP constants ─────────────────────────────────────────────────────────────

TRUE_BETA_X    = 2.0
TRUE_BETA_U    = 2.0
SIGMA_X        = 1.0
K1 = K2        = 10
P01 = P02      = 0.3
RHO_SNP        = 0.3
N              = 20000
M_ENS          = 10
B_BOOT         = 200
EPOCHS         = 500
LR             = 0.01
DROPOUT        = 0.1
ES_MIN_DELTA   = 1e-4
GAMMA_U_VALUES = [0, 1, 3]
N_REPS         = 200
SCOLS          = [f"S1_{j+1}" for j in range(K1)] + [f"S2_{j+1}" for j in range(K2)]


def _make_gammas(seed=0):
    rng = np.random.RandomState(seed)
    return (rng.normal(0, 0.3, K1), rng.normal(0, 0.3, K2),
            rng.normal(0, 0.1, (K1, K2)), rng.normal(0, 0.1, K2))

_GAMMAS = _make_gammas(seed=0)


# ── Data generation ───────────────────────────────────────────────────────────

def _snp_block(n, K, p0, seed):
    rng = np.random.RandomState(seed)
    Z   = rng.normal(size=n)
    s   = np.zeros((n, K))
    for j in range(K):
        e = rng.normal(size=n)
        p = norm.cdf(norm.ppf(p0) + np.sqrt(RHO_SNP)*Z + np.sqrt(1-RHO_SNP)*e)
        s[:, j] = rng.binomial(2, p)
    return s


def gen_data(gamma_u, seed):
    gj1, gj2, gjm12, gj1121 = _GAMMAS
    rng = np.random.RandomState(seed)
    S1  = _snp_block(N, K1, P01, seed * 1000 + 1)
    S2  = _snp_block(N, K2, P02, seed * 1000 + 2)
    fS  = (S1@gj1 + S2@gj2
           + np.sum(S1[:,:,None]*S2[:,None,:]*gjm12[None,:,:], axis=(1,2))
           + (S1[:,0]*S1[:,1])*(S2@gj1121))
    U     = rng.normal(0, 1, N)
    eps_x = rng.normal(0, SIGMA_X, N)
    X     = fS + gamma_u * U + eps_x
    prob  = 1 / (1 + np.exp(-(TRUE_BETA_X * X + TRUE_BETA_U * U)))
    Y     = rng.binomial(1, np.clip(prob, 1e-9, 1-1e-9)).astype(float)
    cols  = {f"S1_{j+1}": S1[:,j] for j in range(K1)}
    cols.update({f"S2_{j+1}": S2[:,j] for j in range(K2)})
    return pd.DataFrame({**cols, "X": X, "Y": Y})


def bootstrap_resample(df, boot_seed):
    rng = np.random.RandomState(boot_seed)
    idx = rng.choice(N, N, replace=True)
    return df.iloc[idx].reset_index(drop=True)


# ── Estimators ────────────────────────────────────────────────────────────────

def logit_2sri(df_train, df_eval):
    lr  = LinearRegression().fit(df_train[SCOLS].values, df_train["X"].values)
    e   = df_eval["X"].values - lr.predict(df_eval[SCOLS].values)
    clf = LogisticRegression(penalty=None, max_iter=1000, solver="lbfgs")
    clf.fit(np.column_stack([df_eval["X"].values, e]), df_eval["Y"].values)
    return float(clf.coef_.ravel()[0])


def nn_2sri(df_train, df_eval):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from deeppliv.core.trainer import DeepPLIV
    model = DeepPLIV()
    model.fit_first_stage(
        df_train[SCOLS].values, df_train["X"].values,
        epochs_first_stage=EPOCHS, learning_rate_first_stage=LR, dropout=DROPOUT,
        validation_data=(df_eval[SCOLS].values, df_eval["X"].values),
        early_stopping_min_delta=ES_MIN_DELTA)
    e  = df_eval["X"].values - model.predict_first_stage(df_eval[SCOLS].values).reshape(-1)
    ss = model.fit_second_stage(
        v_hat=df_eval["X"].values.reshape(-1, 1),
        x=e.reshape(-1, 1),
        y=df_eval["Y"].values.reshape(-1, 1),
        epochs_second_stage=EPOCHS, learning_rate_second_stage=LR,
        dropout=DROPOUT, method="2sri", early_stopping_min_delta=ES_MIN_DELTA)
    return float(ss.final_layer.weight[0, 0].item())


# ── Task definitions ──────────────────────────────────────────────────────────

def _sim_seed(gamma_u, rep):
    return rep * len(GAMMA_U_VALUES) + GAMMA_U_VALUES.index(gamma_u)


def make_task_id(gamma_u, rep, task_type, idx):
    return f"gu{gamma_u}_rep{rep:03d}_{task_type}_{idx:03d}"


def s3_key(task_id):
    return f"gamma_u_results/{task_id}.json"


def all_tasks():
    """Enumerate all 246,600 tasks as dicts."""
    tasks = []
    for gu in GAMMA_U_VALUES:
        for rep in range(N_REPS):
            ss = _sim_seed(gu, rep)
            tasks.append({
                "task_id": make_task_id(gu, rep, "logit_point", 0),
                "gamma_u": gu, "rep": rep, "task_type": "logit_point",
                "idx": 0, "sim_seed": ss,
            })
            for b in range(B_BOOT):
                tasks.append({
                    "task_id": make_task_id(gu, rep, "logit_boot", b),
                    "gamma_u": gu, "rep": rep, "task_type": "logit_boot",
                    "idx": b, "sim_seed": ss,
                    "boot_seed": ss * 10_000 + b,
                })
                tasks.append({
                    "task_id": make_task_id(gu, rep, "nn_boot", b),
                    "gamma_u": gu, "rep": rep, "task_type": "nn_boot",
                    "idx": b, "sim_seed": ss,
                    "boot_seed": ss * 10_000 + 50_000 + b,
                })
            for m in range(M_ENS):
                tasks.append({
                    "task_id": make_task_id(gu, rep, "nn_ens", m),
                    "gamma_u": gu, "rep": rep, "task_type": "nn_ens",
                    "idx": m, "sim_seed": ss,
                })
    return tasks


# ── Run one task ──────────────────────────────────────────────────────────────

def run_task(task):
    """Execute one fit. Returns {"task_id": ..., "value": float}."""
    gu    = task["gamma_u"]
    ttype = task["task_type"]
    ss    = task["sim_seed"]

    df = gen_data(gu, ss)

    if ttype == "logit_point":
        value = logit_2sri(df, df)

    elif ttype == "logit_boot":
        df_b  = bootstrap_resample(df, task["boot_seed"])
        value = logit_2sri(df_b, df_b)

    elif ttype == "nn_ens":
        value = nn_2sri(df, df)

    elif ttype == "nn_boot":
        df_b  = bootstrap_resample(df, task["boot_seed"])
        value = nn_2sri(df_b, df_b)

    else:
        raise ValueError(f"Unknown task_type: {ttype}")

    return {"task_id": task["task_id"], "value": value}


# ── Worker process ─────────────────────────────────────────────────────────────

def worker_loop(queue_url, bucket):
    """Single worker: pull → check → run → save → delete, until queue empty."""
    import os, sys
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    sys.stdout = open(os.devnull, "w")
    import warnings; warnings.filterwarnings("ignore")
    import torch; torch.set_num_threads(1)
    import boto3
    from botocore.exceptions import ClientError

    sqs = boto3.client("sqs", region_name=AWS_REGION)
    s3  = boto3.client("s3",  region_name=AWS_REGION)

    while True:
        resp = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=1,
            VisibilityTimeout=600,   # 10 min — enough for any single fit
            WaitTimeSeconds=20)      # long-poll to avoid busy-wait

        msgs = resp.get("Messages", [])
        if not msgs:
            break   # queue drained

        msg    = msgs[0]
        handle = msg["ReceiptHandle"]
        task   = json.loads(msg["Body"])
        key    = s3_key(task["task_id"])

        # Skip if another worker already completed this task
        try:
            s3.head_object(Bucket=bucket, Key=key)
            sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=handle)
            continue
        except ClientError:
            pass

        try:
            result = run_task(task)
            s3.put_object(Bucket=bucket, Key=key,
                          Body=json.dumps(result).encode())
            sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=handle)
        except Exception:
            # Leave message in queue — visibility timeout will re-queue it
            pass


# ── populate ──────────────────────────────────────────────────────────────────

def _send_batch(args):
    """Send one batch of ≤10 SQS messages (thread worker)."""
    queue_url, batch = args
    import boto3
    sqs = boto3.client("sqs", region_name=AWS_REGION)
    for attempt in range(3):
        try:
            sqs.send_message_batch(QueueUrl=queue_url, Entries=batch)
            return len(batch)
        except Exception:
            time.sleep(2 ** attempt)
    return 0


def cmd_populate(args):
    import boto3
    from botocore.exceptions import ClientError

    s3  = boto3.client("s3",  region_name=AWS_REGION)
    sqs = boto3.client("sqs", region_name=AWS_REGION)

    # Create S3 bucket
    try:
        s3.head_bucket(Bucket=S3_BUCKET)
        print(f"S3 bucket '{S3_BUCKET}' already exists")
    except ClientError:
        s3.create_bucket(Bucket=S3_BUCKET,
            CreateBucketConfiguration={"LocationConstraint": AWS_REGION})
        print(f"Created S3 bucket '{S3_BUCKET}'")

    # Create SQS queue
    resp = sqs.create_queue(
        QueueName=SQS_NAME,
        Attributes={"VisibilityTimeout": "600",
                    "MessageRetentionPeriod": "1209600"})
    queue_url = resp["QueueUrl"]
    print(f"SQS queue: {queue_url}")

    # Find already-completed tasks in S3
    print("Scanning S3 for completed tasks...")
    done = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix="gamma_u_results/"):
        for obj in page.get("Contents", []):
            fname = obj["Key"].split("/")[-1].replace(".json", "")
            done.add(fname)
    print(f"  {len(done)} tasks already done in S3")

    tasks   = all_tasks()
    pending = [t for t in tasks if t["task_id"] not in done]
    print(f"Submitting {len(pending):,} / {len(tasks):,} tasks to SQS...")

    # Build batches of 10
    batches = []
    batch   = []
    for t in pending:
        batch.append({"Id": str(len(batch)), "MessageBody": json.dumps(t)})
        if len(batch) == 10:
            batches.append((queue_url, batch))
            batch = []
    if batch:
        batches.append((queue_url, batch))

    # Send with 50 threads
    n_sent = 0
    with ThreadPoolExecutor(max_workers=50) as pool:
        for n in pool.map(_send_batch, batches):
            n_sent += n
            if n_sent % 20_000 == 0:
                print(f"  {n_sent:,} / {len(pending):,} submitted", flush=True)

    print(f"Done. {n_sent:,} tasks in queue.")
    print(f"\nRun on each EC2 instance:")
    print(f"  PYTHONPATH=/home/ec2-user python3.11 run_gamma_u_aws.py worker --workers 190")


# ── worker ────────────────────────────────────────────────────────────────────

def cmd_worker(args):
    import boto3
    sqs = boto3.client("sqs", region_name=AWS_REGION)
    queue_url = sqs.get_queue_url(QueueName=SQS_NAME)["QueueUrl"]

    n_workers = args.workers
    n_cpus    = os.cpu_count() or 8
    if n_workers == 0:
        n_workers = max(1, n_cpus - 2)

    print(f"Starting {n_workers} workers  (cpu_count={n_cpus})", flush=True)
    print(f"Queue: {queue_url}", flush=True)

    ctx = mp.get_context("spawn")
    t0  = time.time()

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        futures = [ex.submit(worker_loop, queue_url, S3_BUCKET)
                   for _ in range(n_workers)]
        n_done = 0
        for _ in as_completed(futures):
            n_done += 1
            elapsed = time.time() - t0
            print(f"  Worker {n_done}/{n_workers} finished  ({elapsed/3600:.2f}h elapsed)",
                  flush=True)

    print(f"All workers done in {(time.time()-t0)/3600:.2f}h", flush=True)


# ── progress ──────────────────────────────────────────────────────────────────

def cmd_progress(args):
    import boto3
    s3  = boto3.client("s3",  region_name=AWS_REGION)
    sqs = boto3.client("sqs", region_name=AWS_REGION)

    # Queue depth (approximate)
    try:
        queue_url = sqs.get_queue_url(QueueName=SQS_NAME)["QueueUrl"]
        attrs = sqs.get_queue_attributes(
            QueueUrl=queue_url,
            AttributeNames=["ApproximateNumberOfMessages",
                            "ApproximateNumberOfMessagesNotVisible"])["Attributes"]
        visible    = int(attrs["ApproximateNumberOfMessages"])
        in_flight  = int(attrs["ApproximateNumberOfMessagesNotVisible"])
        print(f"SQS: {visible:,} pending, {in_flight:,} in-flight")
    except Exception as e:
        print(f"SQS check failed: {e}")

    # S3 counts by task type
    counts = {"logit_point": 0, "logit_boot": 0, "nn_ens": 0, "nn_boot": 0}
    total  = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix="gamma_u_results/"):
        for obj in page.get("Contents", []):
            key = obj["Key"].split("/")[-1].replace(".json", "")
            for ttype in counts:
                if ttype in key:
                    counts[ttype] += 1
                    break
            total += 1

    grand = N_REPS * len(GAMMA_U_VALUES)
    print(f"\nS3 completed: {total:,} / 246,600 ({100*total/246600:.1f}%)")
    print(f"  logit_point : {counts['logit_point']:5d} / {grand}")
    print(f"  logit_boot  : {counts['logit_boot']:6d} / {grand * B_BOOT}")
    print(f"  nn_ens      : {counts['nn_ens']:6d} / {grand * M_ENS}")
    print(f"  nn_boot     : {counts['nn_boot']:6d} / {grand * B_BOOT}")


# ── collect ───────────────────────────────────────────────────────────────────

def _download_page(args):
    """Download one S3 page of results (thread worker)."""
    bucket, keys = args
    import boto3
    s3 = boto3.client("s3", region_name=AWS_REGION)
    rows = []
    for key in keys:
        try:
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            rows.append(json.loads(body))
        except Exception:
            pass
    return rows


def cmd_collect(args):
    import boto3
    s3 = boto3.client("s3", region_name=AWS_REGION)
    os.makedirs("gamma_u_ci_results", exist_ok=True)

    print("Listing S3 objects...")
    all_keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix="gamma_u_results/"):
        for obj in page.get("Contents", []):
            all_keys.append(obj["Key"])
    print(f"  {len(all_keys):,} result files found")

    # Download in parallel batches of 100 keys
    chunk_size = 100
    chunks = [(S3_BUCKET, all_keys[i:i+chunk_size])
              for i in range(0, len(all_keys), chunk_size)]

    # Accumulators
    store = {(gu, rep): {
        "logit_point": None,
        "logit_bootstrap": [None] * B_BOOT,
        "nn_ensemble":     [None] * M_ENS,
        "nn_bootstrap":    [None] * B_BOOT,
    } for gu in GAMMA_U_VALUES for rep in range(N_REPS)}

    print("Downloading and assembling...")
    n_done = 0
    with ThreadPoolExecutor(max_workers=50) as pool:
        for rows in pool.map(_download_page, chunks):
            for r in rows:
                tid   = r["task_id"]
                val   = r["value"]
                parts = tid.split("_")
                gu    = int(parts[0][2:])
                rep   = int(parts[1][3:])
                ttype = "_".join(parts[2:-1])
                idx   = int(parts[-1])
                s     = store[(gu, rep)]
                if ttype == "logit_point":
                    s["logit_point"] = val
                elif ttype == "logit_boot":
                    s["logit_bootstrap"][idx] = val
                elif ttype == "nn_ens":
                    s["nn_ensemble"][idx] = val
                elif ttype == "nn_boot":
                    s["nn_bootstrap"][idx] = val
                n_done += 1
            if n_done % 50_000 == 0:
                print(f"  {n_done:,} assembled...", flush=True)

    print(f"Writing {N_REPS * len(GAMMA_U_VALUES)} JSON files...")
    n_complete = 0
    for (gu, rep), s in store.items():
        complete = (
            s["logit_point"] is not None
            and None not in s["logit_bootstrap"]
            and None not in s["nn_ensemble"]
            and None not in s["nn_bootstrap"]
        )
        out = {
            "gamma_u": gu, "rep": rep,
            "config": {
                "true_beta_x": TRUE_BETA_X, "true_beta_u": TRUE_BETA_U,
                "sigma_x": SIGMA_X, "n": N, "m": M_ENS, "b": B_BOOT,
            },
            "logit_point":     s["logit_point"],
            "logit_bootstrap": s["logit_bootstrap"],
            "nn_ensemble":     s["nn_ensemble"],
            "nn_bootstrap":    s["nn_bootstrap"],
            "status": "complete" if complete else "partial",
        }
        path = f"gamma_u_ci_results/gamma{gu}_rep{rep:03d}.json"
        with open(path, "w") as f:
            json.dump(out, f)
        if complete:
            n_complete += 1

    print(f"Done. {n_complete} / {N_REPS * len(GAMMA_U_VALUES)} simulations complete.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Distributed gamma_U study (SQS + S3)")
    sub    = parser.add_subparsers(dest="cmd")

    sub.add_parser("populate", help="Create queue/bucket and fill with tasks")

    wp = sub.add_parser("worker", help="Run workers pulling from SQS")
    wp.add_argument("--workers", type=int, default=0,
                    help="Number of parallel workers (default: cpu_count-2)")

    sub.add_parser("progress", help="Show completion counts from S3 + SQS")
    sub.add_parser("collect",  help="Download S3 results → local JSON files")

    args = parser.parse_args()

    if   args.cmd == "populate":  cmd_populate(args)
    elif args.cmd == "worker":    cmd_worker(args)
    elif args.cmd == "progress":  cmd_progress(args)
    elif args.cmd == "collect":   cmd_collect(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
