#!/usr/bin/env python3
"""
Hyperparameter sweep runner for BCNN_ECG.ipynb.

This leaves the original notebook unchanged and recreates the notebook's ECG data
loading, BayesianSmallCNN model, SVI training loop, and test-accuracy evaluation
inside a resumable script. Results are appended to CSV after every configuration.

Default training/evaluation settings match the notebook's main training cell:
  - num_epochs=50
  - batch_size=32 train / 128 test
  - num_prediction_samples=10
  - test_max_batches=5

The original notebook hard-codes tanh activations, so this runner adds activation
as an experiment parameter while keeping the rest of the architecture the same.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

try:
    import pyro
    import pyro.distributions as dist
    from pyro.infer import SVI, Trace_ELBO, Predictive
    from pyro.infer.autoguide import AutoDiagonalNormal
    from pyro.nn import PyroModule, PyroSample
    from pyro.optim import Adam
except Exception as exc:  # pragma: no cover - gives a clearer message for local runs
    raise RuntimeError(
        "Pyro is required for this sweep. Install it in the same environment as "
        "PyTorch, for example: pip install pyro-ppl"
    ) from exc


KERNEL_SIZES = [5, 7, 9, 11]
PRIOR_SIZES = [ 0.5, 0.75, 1.0]
ACTIVATIONS = ["tanh", "sigmoid", "relu"]
NUM_FILTERS = [8, 12, 16, 32, 64, 128]
LEARNING_RATES = [1e-3, 5e-3, 10e-3]

FIELDNAMES = [
    "run_index",
    "kernel_size",
    "prior_size",
    "activation_function",
    "num_filters",
    "lr",
    "num_epochs",
    "num_prediction_samples",
    "test_accuracy",
    "final_avg_negative_elbo_per_example",
    "elapsed_seconds",
    "status",
    "error",
]


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def load_ecg_dataset(data_dir: Path, label_source: str = "auto") -> tuple[np.ndarray, np.ndarray]:
    """Load the ECG waveforms and labels using notebook-compatible file names.

    Label source order for label_source='auto':
      1. EchoNext_EchoData_5K.csv column shd_moderate_or_greater_flag
      2. First column of ECG_metrics_1K.npy / ECG_metrics_1K(2).npy
    """
    waveform_path = _first_existing(
        [
            data_dir / "ECG_waveforms_1K.npy",
            data_dir / "ECG_waveforms_1K(2).npy",
        ]
    )
    if waveform_path is None:
        raise FileNotFoundError(
            "Could not find ECG_waveforms_1K.npy or ECG_waveforms_1K(2).npy"
        )

    ECG_data_1K = np.load(waveform_path)
    X = ECG_data_1K[:1000]
    X_ts = np.swapaxes(X, 1, 2).astype(np.float32)

    labels = None
    if label_source in {"auto", "echo_csv"}:
        echo_path = data_dir / "EchoNext_EchoData_5K.csv"
        if echo_path.exists():
            echo = pd.read_csv(echo_path)
            if "shd_moderate_or_greater_flag" not in echo.columns:
                raise KeyError(
                    "EchoNext_EchoData_5K.csv is missing "
                    "shd_moderate_or_greater_flag"
                )
            labels = echo["shd_moderate_or_greater_flag"].to_numpy()[:1000]
        elif label_source == "echo_csv":
            raise FileNotFoundError("EchoNext_EchoData_5K.csv was not found")

    if labels is None and label_source in {"auto", "metrics_first_col"}:
        metrics_path = _first_existing(
            [
                data_dir / "ECG_metrics_1K.npy",
                data_dir / "ECG_metrics_1K(2).npy",
            ]
        )
        if metrics_path is None:
            raise FileNotFoundError(
                "No labels found: missing EchoNext_EchoData_5K.csv and "
                "ECG_metrics_1K.npy / ECG_metrics_1K(2).npy"
            )
        metrics = np.load(metrics_path)
        labels = metrics[:1000, 0]

    y_ts = np.asarray(labels[:1000], dtype=np.int64)
    unique = set(np.unique(y_ts).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"Expected binary labels 0/1, got labels: {sorted(unique)}")

    return X_ts, y_ts


def make_loaders(
    X_ts: np.ndarray,
    y_ts: np.ndarray,
    batch_size: int,
    test_batch_size: int,
    split_seed: int,
) -> tuple[DataLoader, DataLoader, int, int]:
    X_ts_tensor = torch.tensor(X_ts, dtype=torch.float32)
    y_ts_tensor = torch.tensor(y_ts, dtype=torch.long)
    ts_dataset = TensorDataset(X_ts_tensor, y_ts_tensor)

    train_size = int(0.8 * len(ts_dataset))
    test_size = len(ts_dataset) - train_size

    ts_train_dataset, ts_test_dataset = random_split(
        ts_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(split_seed),
    )

    ts_train_loader = DataLoader(ts_train_dataset, batch_size=batch_size, shuffle=True)
    ts_test_loader = DataLoader(ts_test_dataset, batch_size=test_batch_size, shuffle=False)
    return ts_train_loader, ts_test_loader, train_size, test_size


def get_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    cleaned = name.lower().replace("()", "")
    if cleaned == "tanh":
        return torch.tanh
    if cleaned == "sigmoid":
        return torch.sigmoid
    if cleaned == "relu":
        return F.relu
    raise ValueError(f"Unknown activation function: {name}")


class BayesianSmallCNN(PyroModule):
    def __init__(
        self,
        prior_scale: float = 0.10,
        num_filters: int = 8,
        kernel_size: int = 5,
        activation_function: str = "tanh",
        input_channels: int = 12,
        input_length: int = 2500,
        hidden_dim: int = 32,
        num_classes: int = 2,
    ):
        super().__init__()

        if prior_scale <= 0:
            raise ValueError(
                "prior_size/prior_scale must be > 0 for dist.Normal scale. "
                "The value 0 is recorded as an error and skipped."
            )

        padding = kernel_size // 2

        self.input_channels = input_channels
        self.input_length = input_length
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.prior_scale = prior_scale
        self.activation_function = activation_function
        self.activation = get_activation(activation_function)

        self.conv1 = PyroModule[nn.Conv1d](
            input_channels,
            num_filters,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv1.weight = PyroSample(
            dist.Normal(0.0, prior_scale)
            .expand([num_filters, input_channels, kernel_size])
            .to_event(3)
        )
        self.conv1.bias = PyroSample(
            dist.Normal(0.0, prior_scale).expand([num_filters]).to_event(1)
        )

        self.conv2 = PyroModule[nn.Conv1d](
            num_filters,
            input_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv2.weight = PyroSample(
            dist.Normal(0.0, prior_scale)
            .expand([input_channels, num_filters, kernel_size])
            .to_event(3)
        )
        self.conv2.bias = PyroSample(
            dist.Normal(0.0, prior_scale).expand([input_channels]).to_event(1)
        )

        pooled_length = input_length // 4
        flattened_dim = input_channels * pooled_length

        self.fc1 = PyroModule[nn.Linear](flattened_dim, hidden_dim)
        self.fc1.weight = PyroSample(
            dist.Normal(0.0, prior_scale).expand([hidden_dim, flattened_dim]).to_event(2)
        )
        self.fc1.bias = PyroSample(
            dist.Normal(0.0, prior_scale).expand([hidden_dim]).to_event(1)
        )

        self.fc2 = PyroModule[nn.Linear](hidden_dim, num_classes)
        self.fc2.weight = PyroSample(
            dist.Normal(0.0, prior_scale).expand([num_classes, hidden_dim]).to_event(2)
        )
        self.fc2.bias = PyroSample(
            dist.Normal(0.0, prior_scale).expand([num_classes]).to_event(1)
        )

    def forward(self, x, y=None, dataset_size=None):
        batch_size = x.shape[0]

        x = self.activation(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=2)

        x = self.activation(self.conv2(x))
        x = F.max_pool1d(x, kernel_size=2)

        x = torch.flatten(x, start_dim=1)
        x = self.activation(self.fc1(x))
        logits = self.fc2(x)

        if y is not None:
            if dataset_size is not None:
                scale = dataset_size / batch_size
            else:
                scale = 1.0

            with pyro.plate("data", batch_size):
                with pyro.poutine.scale(scale=scale):
                    pyro.sample("obs", dist.Categorical(logits=logits), obs=y)

        return logits


def train_one_epoch_svi(svi, data_loader, device, dataset_size, max_batches=None):
    total_loss = 0.0
    total_examples = 0

    for batch_idx, (x, y) in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        loss = svi.step(x, y, dataset_size)
        total_loss += loss
        total_examples += x.shape[0]

    return total_loss / max(1, total_examples)


@torch.no_grad()
def predict_proba_bayesian(model, guide, x, num_samples=250):
    predictive = Predictive(
        model,
        guide=guide,
        num_samples=num_samples,
        return_sites=("_RETURN",),
    )
    out = predictive(x, None)
    logits_samples = out["_RETURN"]
    probs_samples = torch.softmax(logits_samples, dim=-1)
    mean_probs = probs_samples.mean(dim=0)
    return mean_probs


@torch.no_grad()
def evaluate_bayesian_classifier(
    model, guide, data_loader, device, num_samples=2500, max_batches=None
):
    correct = 0
    total = 0

    for batch_idx, (x, y) in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)
        mean_probs = predict_proba_bayesian(model, guide, x, num_samples=num_samples)
        preds = mean_probs.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.numel()

    return correct / max(1, total)


def all_configs():
    return list(
        itertools.product(
            KERNEL_SIZES, PRIOR_SIZES, ACTIVATIONS, NUM_FILTERS, LEARNING_RATES
        )
    )


def load_completed_keys(output_csv: Path) -> set[tuple[int, float, str, int, float]]:
    if not output_csv.exists():
        return set()
    completed = set()
    with output_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") in {"ok", "error"}:
                completed.add(
                    (
                        int(row["kernel_size"]),
                        float(row["prior_size"]),
                        row["activation_function"],
                        int(row["num_filters"]),
                        float(row["lr"]),
                    )
                )
    return completed


def append_row(output_csv: Path, row: dict) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_csv.exists() or output_csv.stat().st_size == 0
    with output_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def run_one_config(
    *,
    run_index: int,
    kernel_size: int,
    prior_size: float,
    activation_function: str,
    num_filters: int,
    lr: float,
    args,
    ts_train_loader,
    ts_test_loader,
    train_size: int,
    device: torch.device,
) -> dict:
    start = time.time()
    base_row = {
        "run_index": run_index,
        "kernel_size": kernel_size,
        "prior_size": prior_size,
        "activation_function": f"{activation_function}()",
        "num_filters": num_filters,
        "lr": lr,
        "num_epochs": args.num_epochs,
        "num_prediction_samples": args.num_prediction_samples,
        "test_accuracy": "",
        "final_avg_negative_elbo_per_example": "",
        "elapsed_seconds": "",
        "status": "error",
        "error": "",
    }

    try:
        pyro.clear_param_store()
        set_seed(args.seed)

        bcnn = BayesianSmallCNN(
            prior_scale=prior_size,
            num_filters=num_filters,
            kernel_size=kernel_size,
            activation_function=activation_function,
        ).to(device)

        bcnn_guide = AutoDiagonalNormal(bcnn)
        bcnn_optimizer = Adam({"lr": lr, "betas": (0.9, 0.999)})
        bcnn_svi = SVI(
            model=bcnn,
            guide=bcnn_guide,
            optim=bcnn_optimizer,
            loss=Trace_ELBO(),
        )

        avg_loss = math.nan
        epoch_test_accs = []

        for _epoch in range(args.num_epochs):
            avg_loss = train_one_epoch_svi(
                bcnn_svi,
                ts_train_loader,
                device,
                dataset_size=train_size,
                max_batches=args.max_batches_per_epoch,
            )

            epoch_test_acc = evaluate_bayesian_classifier(
                bcnn,
                bcnn_guide,
                ts_test_loader,
                device,
                num_samples=args.num_prediction_samples,
                max_batches=args.test_max_batches,
            )
            epoch_test_accs.append(epoch_test_acc)

        test_acc = float(np.mean(epoch_test_accs)) if epoch_test_accs else float("nan")

        base_row.update(
            {
                "test_accuracy": test_acc,
                "final_avg_negative_elbo_per_example": avg_loss,
                "elapsed_seconds": round(time.time() - start, 3),
                "status": "ok",
                "error": "",
            }
        )
        return base_row
    except Exception as exc:
        base_row.update(
            {
                "elapsed_seconds": round(time.time() - start, 3),
                "status": "error",
                "error": " ".join(str(exc).split()),
            }
        )
        if args.verbose_errors:
            traceback.print_exc()
        return base_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--output-csv", type=Path, default=Path("BCNN_ECG_grid_results.csv")
    )
    parser.add_argument(
        "--label-source",
        choices=["auto", "echo_csv", "metrics_first_col"],
        default="auto",
        help="Use EchoNext CSV labels if present; otherwise first ECG_metrics column.",
    )
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-batch-size", type=int, default=128)
    parser.add_argument("--num-prediction-samples", type=int, default=10)
    parser.add_argument("--max-batches-per-epoch", type=int, default=None)
    parser.add_argument("--test-max-batches", type=int, default=5)
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--verbose-errors", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    set_seed(args.seed)
    pyro.set_rng_seed(args.seed)

    X_ts, y_ts = load_ecg_dataset(args.data_dir, label_source=args.label_source)
    ts_train_loader, ts_test_loader, train_size, test_size = make_loaders(
        X_ts,
        y_ts,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        split_seed=args.split_seed,
    )

    configs = all_configs()
    configs = configs[args.start_index :]
    if args.max_configs is not None:
        configs = configs[: args.max_configs]

    completed = set() if args.no_resume else load_completed_keys(args.output_csv)

    print(f"Using device: {device}")
    print(f"Training samples: {train_size}; test samples: {test_size}")
    print(f"Total selected configurations: {len(configs)}")
    print(f"Writing results to: {args.output_csv}")

    for offset, (kernel_size, prior_size, activation_function, num_filters, lr) in enumerate(
        configs
    ):
        run_index = args.start_index + offset
        key = (kernel_size, float(prior_size), f"{activation_function}()", num_filters, float(lr))
        if key in completed:
            print(f"Skipping completed run_index={run_index}: {key}")
            continue

        print(
            "Running "
            f"{run_index + 1}/{len(all_configs())}: "
            f"kernel={kernel_size}, prior={prior_size}, activation={activation_function}, "
            f"filters={num_filters}, lr={lr}"
        )
        row = run_one_config(
            run_index=run_index,
            kernel_size=kernel_size,
            prior_size=float(prior_size),
            activation_function=activation_function,
            num_filters=int(num_filters),
            lr=float(lr),
            args=args,
            ts_train_loader=ts_train_loader,
            ts_test_loader=ts_test_loader,
            train_size=train_size,
            device=device,
        )
        append_row(args.output_csv, row)
        if (
            row["status"] == "ok"
            and float(row["test_accuracy"]) > 0.65
        ):
            print(
                f"*** GOOD MODEL *** "
                f"test_accuracy={row['test_accuracy']:.4f} "
                f"kernel={row['kernel_size']} "
                f"prior={row['prior_size']} "
                f"filters={row['num_filters']} "
                f"lr={row['lr']}"
            )

    print("Done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

