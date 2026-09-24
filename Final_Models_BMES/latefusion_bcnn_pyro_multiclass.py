"""Pyro late-fusion Bayesian CNN for EchoNext waveform + demographics data.

This is the Pyro/SVI equivalent of ``LateFusion_BCNN_FINAL``.  It preserves
that model's late-fusion structure: an ECG CNN and a demographic MLP learn
independent embeddings, which are concatenated only before classification.

Install dependencies, if needed:
    pip install pyro-ppl torch numpy pandas scipy scikit-learn
"""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score,
    f1_score, log_loss, average_precision_score
)
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import constraints
from torch.utils.data import DataLoader, Dataset

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.nn import PyroModule, PyroSample
from pyro.optim import ReduceLROnPlateau

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 0. CONFIGURATION
# ---------------------------------------------------------------------
RANDOM_SEED = 42
PRIOR_SIGMA_1 = 2.5
PRIOR_SIGMA_2 = 0.5
PRIOR_PI = 0.5
POSTERIOR_RHO_INIT = -6.0

ECG_SAMPLING_RATE_HZ = 250
CATEGORICAL_COLS = ["race_ethnicity", "sex"]
CONTINUOUS_COLS = [
    "age_at_ecg", "ventricular_rate", "atrial_rate", "pr_interval",
    "qrs_duration", "qt_corrected",
]
ALL_DEMO_COLS = CATEGORICAL_COLS + CONTINUOUS_COLS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# 1. PREPROCESSING
# ---------------------------------------------------------------------
def bandpass_filter_ecg(signal_array, fs=ECG_SAMPLING_RATE_HZ, low=0.5, high=40.0, order=4):
    """Filter a waveform of shape (time, leads) along its time axis."""
    nyquist = 0.5 * fs
    b, a = butter(order, [low / nyquist, high / nyquist], btype="band")
    return filtfilt(b, a, np.asarray(signal_array), axis=0).astype(np.float32)


def preprocess_demographics(df, scaler=None, imputer=None, categories=None, knn_k=5, fit=True):
    """Fit preprocessing only on training demographics, then reuse it for other splits."""
    df = df[ALL_DEMO_COLS].copy()
    if categories is None:
        if not fit:
            raise ValueError("categories must be supplied when fit=False")
        categories = {c: df[c].astype("category").cat.categories for c in CATEGORICAL_COLS}

    coded = pd.DataFrame(index=df.index)
    for col in CATEGORICAL_COLS:
        coded[col] = pd.Categorical(df[col], categories=categories[col]).codes
    coded = coded.replace(-1, np.nan)
    combined = pd.concat([coded, df[CONTINUOUS_COLS].astype(float)], axis=1)

    if imputer is None:
        if not fit:
            raise ValueError("imputer must be supplied when fit=False")
        imputer = KNNImputer(n_neighbors=knn_k)
    values = imputer.fit_transform(combined) if fit else imputer.transform(combined)
    values = pd.DataFrame(values, columns=ALL_DEMO_COLS, index=df.index)

    for col in CATEGORICAL_COLS:
        n_categories = len(categories[col])
        codes = values[col].round().astype(int).clip(0, n_categories - 1)
        values[col] = pd.Categorical.from_codes(codes, categories=categories[col])

    encoded = pd.get_dummies(values, columns=CATEGORICAL_COLS, dtype=float)
    feature_names = list(encoded.columns)
    array = encoded.to_numpy(dtype=np.float32)
    continuous_indices = [feature_names.index(c) for c in CONTINUOUS_COLS]
    if scaler is None:
        if not fit:
            raise ValueError("scaler must be supplied when fit=False")
        scaler = StandardScaler()
    array[:, continuous_indices] = (
        scaler.fit_transform(array[:, continuous_indices]) if fit
        else scaler.transform(array[:, continuous_indices])
    )
    return array, scaler, imputer, categories, feature_names


class FusedDataset(Dataset):
    def __init__(self, x_ts, x_demo, y):
        if not (len(x_ts) == len(x_demo) == len(y)):
            raise ValueError("Waveform, demographic, and label lengths must match")
        self.X_ts = torch.as_tensor(x_ts, dtype=torch.float32)
        self.X_demo = torch.as_tensor(x_demo, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X_ts[index], self.X_demo[index], self.y[index]


# ---------------------------------------------------------------------
# 2. PYRO BAYESIAN LAYERS
# ---------------------------------------------------------------------
class RealSupportMixtureSameFamily(dist.MixtureSameFamily):
    """Gaussian mixture whose support is explicitly real for AutoNormal."""
    support = constraints.real


class _ScaleMixturePriorMixin:
    def _register_prior_buffers(self):
        self.register_buffer("_prior_probs", torch.tensor([PRIOR_PI, 1.0 - PRIOR_PI]))
        self.register_buffer("_prior_locs", torch.zeros(2))
        self.register_buffer("_prior_scales", torch.tensor([PRIOR_SIGMA_1, PRIOR_SIGMA_2]))

    def _prior(self, shape):
        mixture = RealSupportMixtureSameFamily(
            dist.Categorical(probs=self._prior_probs),
            dist.Normal(self._prior_locs, self._prior_scales),
        )
        return mixture.expand(shape).to_event(len(shape))


class PyroBayesianLinear(_ScaleMixturePriorMixin, PyroModule):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self._register_prior_buffers()
        self.weight = PyroSample(lambda m: m._prior((out_features, in_features)))
        self.bias = PyroSample(lambda m: m._prior((out_features,))) if bias else None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class PyroBayesianConv1d(_ScaleMixturePriorMixin, PyroModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.stride, self.padding = stride, padding
        self._register_prior_buffers()
        self.weight = PyroSample(lambda m: m._prior((out_channels, in_channels, kernel_size)))
        self.bias = PyroSample(lambda m: m._prior((out_channels,))) if bias else None

    def forward(self, x):
        return F.conv1d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)


# ---------------------------------------------------------------------
# 3. LATE-FUSION NETWORK (same mathematical architecture as BLITZ code)
# ---------------------------------------------------------------------
class BayesianDemographicMLP(PyroModule):
    """Independent demographic branch: demog_dim -> 32 -> 16."""
    def __init__(self, input_dim, output_dim=16):
        super().__init__()
        self.fc1 = PyroBayesianLinear(input_dim, 32)
        self.fc2 = PyroBayesianLinear(32, output_dim)
        self.activ = nn.GELU()

    def forward(self, x):
        return self.activ(self.fc2(self.activ(self.fc1(x))))


class BayesianLateFusionHeartDiseaseNet(PyroModule):
    """ECG CNN -> 32, demographics MLP -> 16, concatenate -> 32 -> classes."""
    def __init__(self, in_channels=12, demog_dim=8, num_classes=2):
        super().__init__()
        # ECG branch: exactly 12->16->32->32 with 15/9/5 kernels and 10/10 pooling.
        self.conv1 = PyroBayesianConv1d(in_channels, 16, kernel_size=15, padding=7)
        self.pool1 = nn.AvgPool1d(kernel_size=10)
        self.conv2 = PyroBayesianConv1d(16, 32, kernel_size=9, padding=4)
        self.pool2 = nn.AvgPool1d(kernel_size=10)
        self.conv3 = PyroBayesianConv1d(32, 32, kernel_size=5, padding=2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.activ = nn.GELU()

        self.demog_features = BayesianDemographicMLP(demog_dim, output_dim=16)
        self.fc_out1 = PyroBayesianLinear(32 + 16, 32)
        self.fc_out2 = PyroBayesianLinear(32, num_classes)

    def forward(self, ts_input, demog_input):
        x = self.pool1(self.activ(self.conv1(ts_input)))
        x = self.pool2(self.activ(self.conv2(x)))
        ts_embedding = torch.flatten(self.gap(self.activ(self.conv3(x))), start_dim=1)
        demographic_embedding = self.demog_features(demog_input)
        fused = torch.cat((ts_embedding, demographic_embedding), dim=1)
        return self.fc_out2(self.activ(self.fc_out1(fused)))


class PyroLateFusionClassifier(PyroModule):
    """Network plus minibatch-scaled categorical observation likelihood."""
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x_ts, x_demo, y=None, dataset_size=None):
        logits = self.network(x_ts, x_demo)
        if y is not None:
            scale = float(dataset_size) / float(x_ts.shape[0]) if dataset_size else 1.0
            with poutine.scale(scale=scale), pyro.plate("data", x_ts.shape[0]):
                pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        return logits


def build_mean_field_guide(model):
    return AutoNormal(model, init_scale=float(F.softplus(torch.tensor(POSTERIOR_RHO_INIT))))


# ---------------------------------------------------------------------
# 4. SVI TRAINING AND POSTERIOR PREDICTION
# ---------------------------------------------------------------------
def sample_posterior_logits(model, guide, x_ts, x_demo, mc_samples=1):
    samples = []
    for _ in range(mc_samples):
        trace = poutine.trace(guide).get_trace(x_ts, x_demo, None, None)
        samples.append(poutine.replay(model, trace=trace)(x_ts, x_demo, None, None))
    return torch.stack(samples)


def train_bayesian_classifier(model, guide, train_loader, val_loader, epochs=60, learning_rate=1e-3, sample_nbr=3):
    pyro.clear_param_store()
    model, guide = model.to(DEVICE), guide.to(DEVICE)
    optimizer = ReduceLROnPlateau({
        "optimizer": torch.optim.Adam,
        "optim_args": {"lr": learning_rate, "weight_decay": 1e-4},
        "mode": "min", "factor": 0.5, "patience": 5,
    })
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO(num_particles=sample_nbr))
    n_train = len(train_loader.dataset)

    for epoch in range(epochs):
        model.train(); guide.train()
        train_loss = 0.0
        for x_ts, x_demo, y in train_loader:
            train_loss += svi.step(x_ts.to(DEVICE), x_demo.to(DEVICE), y.to(DEVICE), n_train)

        model.eval(); guide.eval()
        val_nll, correct, n_val = 0.0, 0, 0
        with torch.no_grad():
            for x_ts, x_demo, y in val_loader:
                x_ts, x_demo, y = x_ts.to(DEVICE), x_demo.to(DEVICE), y.to(DEVICE)
                logits = sample_posterior_logits(model, guide, x_ts, x_demo, mc_samples=10)
                log_mean_probs = torch.logsumexp(torch.log_softmax(logits, -1), 0) - np.log(10)
                val_nll += F.nll_loss(log_mean_probs, y, reduction="sum").item()
                correct += (log_mean_probs.argmax(-1) == y).sum().item()
                n_val += len(y)
        val_nll /= max(n_val, 1)
        optimizer.step(val_nll)
        print(f"Epoch {epoch + 1:03d} | negative ELBO/patient: {train_loss / n_train:.5f} | "
              f"validation NLL: {val_nll:.5f} | validation accuracy: {correct / max(n_val, 1):.4f}", flush=True)
    return svi, optimizer


# 6. PERFORMANCE METRICS GENERATION & CALIBRATION ESTIMATION
# ──────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────
# 6. POSTERIOR PREDICTION, ERROR METRICS, CALIBRATION, AND UNCERTAINTY
# ──────────────────────────────────────────────────────────────────────
def calculate_confidence_ece(probs, labels, bins=10):
    """Multiclass ECE using the probability of the predicted class."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    preds = probs.argmax(axis=1)
    confidence = probs[np.arange(len(labels)), preds]
    correctness = (preds == labels).astype(np.float64)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0

    for i in range(bins):
        if i == bins - 1:
            in_bin = (confidence >= boundaries[i]) & (confidence <= boundaries[i + 1])
        else:
            in_bin = (confidence >= boundaries[i]) & (confidence < boundaries[i + 1])

        if np.any(in_bin):
            bin_confidence = confidence[in_bin].mean()
            bin_accuracy = correctness[in_bin].mean()
            ece += in_bin.mean() * abs(bin_accuracy - bin_confidence)

    return float(ece)


def collect_global_posterior_predictive(
    model,
    guide,
    dataset,
    mc_samples=200,
    evaluation_device=None,
):
    """
    Collect globally consistent posterior predictive draws.

    Each row of posterior_probs uses one guide draw (one sampled Bayesian
    network) evaluated over the entire dataset, so dataset-level posterior
    metrics remain statistically coherent.

    Important: the full dataset must fit in memory on evaluation_device.
    Use CPU when the full dataset does not fit on the GPU.
    """
    if evaluation_device is None:
        evaluation_device = device

    model_parameter = next(model.parameters(), None)
    guide_parameter = next(guide.parameters(), None)
    original_model_device = (
        model_parameter.device if model_parameter is not None else evaluation_device
    )
    original_guide_device = (
        guide_parameter.device if guide_parameter is not None else original_model_device
    )

    model = model.to(evaluation_device)
    guide = guide.to(evaluation_device)
    model.eval()
    guide.eval()

    X_ts = dataset.X_ts.to(evaluation_device)
    X_demo = dataset.X_demo.to(evaluation_device)
    labels = dataset.y.cpu().numpy().astype(np.int64)

    posterior_draws = []

    with torch.no_grad():
        for draw_index in range(mc_samples):
            logits = sample_posterior_logits(
                model,
                guide,
                X_ts,
                X_demo,
                mc_samples=1,
            )[0]
            class_probs = torch.softmax(logits, dim=-1)
            posterior_draws.append(class_probs.cpu().numpy())

            print(
                f"Posterior draw {draw_index + 1}/{mc_samples}",
                end="\r",
                flush=True,
            )

    print(flush=True)

    posterior_probs = np.stack(posterior_draws, axis=0).astype(np.float32)

    if posterior_probs.ndim != 3 or posterior_probs.shape[:2] != (mc_samples, len(dataset)):
        raise RuntimeError(
            "Unexpected posterior probability shape: "
            f"{posterior_probs.shape}; expected {(mc_samples, len(dataset))}."
        )

    if mc_samples > 1:
        mean_draw_difference = float(
            np.mean(np.abs(posterior_probs[0] - posterior_probs[1]))
        )
        print(
            "Mean absolute difference between first two posterior draws: "
            f"{mean_draw_difference:.8f}",
            flush=True,
        )

    model.to(original_model_device)
    guide.to(original_guide_device)
    return posterior_probs, labels


def calculate_multiclass_metrics(probs, labels, calibration_bins=10):
    """Calculate predictive metrics for ``(n_samples, n_classes)`` probabilities."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2 or probs.shape[0] != len(labels):
        raise ValueError("probs must have shape (n_samples, n_classes).")
    num_classes = probs.shape[1]
    classes = np.arange(num_classes)
    if np.any((labels < 0) | (labels >= num_classes)):
        raise ValueError("labels must be integer class indices in [0, n_classes).")

    probs = np.clip(probs, 1e-8, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)
    one_hot = np.eye(num_classes)[labels]

    auroc = np.nan
    if len(np.unique(labels)) > 1:
        try:
            auroc = roc_auc_score(labels, probs, labels=classes, multi_class="ovr", average="macro")
        except ValueError:
            pass  # A class absent from the split makes its one-vs-rest AUROC undefined.

    per_class_ap = []
    for class_id in classes:
        target = one_hot[:, class_id]
        if target.min() != target.max():
            per_class_ap.append(average_precision_score(target, probs[:, class_id]))

    return {
        "accuracy": float(np.mean(preds == labels)),
        "balanced_accuracy": float(recall_score(labels, preds, labels=classes, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(labels, preds, labels=classes, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, labels=classes, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, labels=classes, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, labels=classes, average="weighted", zero_division=0)),
        "auroc_ovr_macro": float(auroc),
        "auprc_macro": float(np.mean(per_class_ap)) if per_class_ap else np.nan,
        "multiclass_brier_score": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "negative_log_likelihood": float(log_loss(labels, probs, labels=classes)),
        "confidence_ece": calculate_confidence_ece(probs, labels, bins=calibration_bins),
    }


def posterior_metric_distribution(posterior_probs, labels, calibration_bins=10):
    """Calculate every error metric separately for each posterior probability draw."""
    records = []
    for draw_index, probs in enumerate(posterior_probs):
        row = calculate_multiclass_metrics(
            probs,
            labels,
            calibration_bins=calibration_bins,
        )
        row["draw"] = draw_index
        records.append(row)
    return pd.DataFrame(records)


def summarize_posterior_metrics(metric_draws_df):
    """Summarize posterior metric distributions with central 95% intervals."""
    metric_columns = [column for column in metric_draws_df.columns if column != "draw"]
    rows = []
    for metric in metric_columns:
        values = metric_draws_df[metric].dropna().to_numpy(dtype=np.float64)
        rows.append({
            "metric": metric,
            "posterior_mean": float(np.mean(values)),
            "posterior_median": float(np.median(values)),
            "posterior_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
        })

    return pd.DataFrame(rows)


def summarize_patient_posteriors(posterior_probs, labels, metadata_ids=None):
    """Create patient-level posterior predictive and uncertainty summaries."""
    mean_probs = posterior_probs.mean(axis=0)
    median_probs = np.median(posterior_probs, axis=0)
    std_probs = posterior_probs.std(axis=0, ddof=1)
    lower = np.quantile(posterior_probs, 0.025, axis=0)
    upper = np.quantile(posterior_probs, 0.975, axis=0)
    predictive_entropy = -np.sum(mean_probs * np.log(np.clip(mean_probs, 1e-8, 1.0)), axis=1)
    expected_entropy = -np.sum(posterior_probs * np.log(np.clip(posterior_probs, 1e-8, 1.0)), axis=2).mean(axis=0)
    mutual_information = np.maximum(predictive_entropy - expected_entropy, 0.0)
    predictions = mean_probs.argmax(axis=1)
    confidence = mean_probs[np.arange(len(labels)), predictions]

    result = pd.DataFrame({
        "sample_index": np.arange(len(labels)),
        "true_label": labels,
        "predicted_label": predictions,
        "prediction_confidence": confidence,
        "predictive_entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": mutual_information,
    })

    for class_id in range(mean_probs.shape[1]):
        result[f"posterior_mean_class{class_id}"] = mean_probs[:, class_id]
        result[f"posterior_median_class{class_id}"] = median_probs[:, class_id]
        result[f"posterior_sd_class{class_id}"] = std_probs[:, class_id]
        result[f"posterior_lower_95_class{class_id}"] = lower[:, class_id]
        result[f"posterior_upper_95_class{class_id}"] = upper[:, class_id]

    if metadata_ids is not None:
        if len(metadata_ids) != len(result):
            raise ValueError("metadata_ids length does not match posterior predictions")
        result.insert(1, "metadata_index", np.asarray(metadata_ids))

    return result


def evaluate_posterior_model(
    model,
    guide,
    dataset,
    split_name,
    results_dir,
    mc_samples=200,
    metadata_ids=None,
    calibration_bins=10,
    evaluation_device=None,
):
    """Run globally consistent posterior evaluation and save all outputs."""
    print(
        f"Collecting {mc_samples} global posterior predictive draws for {split_name}...",
        flush=True,
    )
    posterior_probs, labels = collect_global_posterior_predictive(
        model=model,
        guide=guide,
        dataset=dataset,
        mc_samples=mc_samples,
        evaluation_device=evaluation_device,
    )
    mean_probs = posterior_probs.mean(axis=0)

    posterior_mean_metrics = calculate_multiclass_metrics(
        mean_probs,
        labels,
        calibration_bins=calibration_bins,
    )
    posterior_mean_metrics_df = pd.DataFrame([
        {"metric": metric, "value": value}
        for metric, value in posterior_mean_metrics.items()
    ])

    metric_draws_df = posterior_metric_distribution(
        posterior_probs,
        labels,
        calibration_bins=calibration_bins,
    )
    metric_summary_df = summarize_posterior_metrics(metric_draws_df)
    patient_summary_df = summarize_patient_posteriors(
        posterior_probs,
        labels,
        metadata_ids=metadata_ids,
    )

    np.savez_compressed(
        os.path.join(results_dir, f"{split_name}_posterior_predictive.npz"),
        posterior_class_probs=posterior_probs,
        posterior_mean_class_probs=mean_probs,
        labels=labels,
    )
    posterior_mean_metrics_df.to_csv(
        os.path.join(results_dir, f"{split_name}_posterior_mean_metrics.csv"), index=False
    )
    metric_draws_df.to_csv(
        os.path.join(results_dir, f"{split_name}_posterior_metric_draws.csv"), index=False
    )
    metric_summary_df.to_csv(
        os.path.join(results_dir, f"{split_name}_posterior_metric_summary.csv"), index=False
    )
    patient_summary_df.to_csv(
        os.path.join(results_dir, f"{split_name}_patient_posterior_summary.csv"), index=False
    )
    probability_columns = [
        column for column in patient_summary_df.columns
        if column in {"sample_index", "metadata_index", "true_label", "predicted_label", "prediction_confidence"}
        or column.startswith("posterior_mean_class")
    ]
    patient_summary_df[probability_columns].to_csv(
        os.path.join(results_dir, f"{split_name}_probability_predictions.csv"),
        index=False,
    )

    print(f"\n{split_name.capitalize()} posterior-mean predictive metrics:", flush=True)
    for metric, value in posterior_mean_metrics.items():
        print(f"{metric:<30} {value:.6f}", flush=True)

    return {
        "posterior_probs": posterior_probs,
        "labels": labels,
        "mean_probs": mean_probs,
        "posterior_mean_metrics": posterior_mean_metrics_df,
        "metric_draws": metric_draws_df,
        "metric_summary": metric_summary_df,
        "patient_summary": patient_summary_df,
    }


def save_evaluation_plots(train_results, test_results, results_dir):
    """Save ROC, precision-recall, and test confusion-matrix panels as one PNG.

    Solid lines are training curves and dashed lines are test curves. Curves
    are produced one-vs-rest for every class present in the corresponding
    split; a class absent from a split is skipped because its curve is
    undefined.
    """
    os.makedirs(results_dir, exist_ok=True)
    train_probs = train_results["mean_probs"]
    train_labels = train_results["labels"]
    test_probs = test_results["mean_probs"]
    test_labels = test_results["labels"]
    test_preds = test_probs.argmax(axis=1)
    classes = np.arange(train_probs.shape[1])

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for class_id in classes:
        train_target = (train_labels == class_id).astype(int)
        test_target = (test_labels == class_id).astype(int)
        if train_target.min() != train_target.max():
            fpr, tpr, _ = roc_curve(train_target, train_probs[:, class_id])
            ax[0].plot(fpr, tpr, lw=1.5, label=f"Train class {class_id}")
        if test_target.min() != test_target.max():
            fpr, tpr, _ = roc_curve(test_target, test_probs[:, class_id])
            ax[0].plot(fpr, tpr, lw=1.5, linestyle="--", label=f"Test class {class_id}")
    ax[0].plot([0, 1], [0, 1], linestyle="--")
    ax[0].set_xlabel("False Positive Rate")
    ax[0].set_ylabel("True Positive Rate")
    ax[0].set_title("Posterior-Mean One-vs-Rest ROC")
    ax[0].legend(loc="lower right")

    for class_id in classes:
        train_target = (train_labels == class_id).astype(int)
        test_target = (test_labels == class_id).astype(int)
        if train_target.min() != train_target.max():
            precision, recall, _ = precision_recall_curve(train_target, train_probs[:, class_id])
            ax[1].plot(recall, precision, lw=1.5, label=f"Train class {class_id}")
        if test_target.min() != test_target.max():
            precision, recall, _ = precision_recall_curve(test_target, test_probs[:, class_id])
            ax[1].plot(recall, precision, lw=1.5, linestyle="--", label=f"Test class {class_id}")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].set_title("Posterior-Mean One-vs-Rest Precision-Recall")
    ax[1].legend(loc="lower left")

    cm_test = confusion_matrix(test_labels, test_preds, labels=classes)
    ConfusionMatrixDisplay(cm_test, display_labels=classes).plot(ax=ax[2], values_format="d")
    ax[2].set_title("Posterior-Mean Test Confusion Matrix")

    fig.tight_layout()
    output_path = os.path.join(results_dir, "late_fusion_posterior_metrics.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------
# 5. EXECUTION PIPELINE
# ---------------------------------------------------------------------
if __name__ == "__main__":
    set_seed()
    print(f"Using device: {DEVICE}", flush=True)
    print("Beginning execution pipeline...", flush = True)

    ECG_train_raw = np.load("../EchoNextData/EchoNext_train_waveforms.npy")
    ECG_val_raw = np.load("../EchoNextData/EchoNext_val_waveforms.npy")
    ECG_test_raw = np.load("../EchoNextData/EchoNext_test_waveforms.npy")
    Echo_data = pd.read_csv("../EchoNextData/echonext_metadata_100k.csv")

    X_train_lead1 = ECG_train_raw[:, 0, :, :]
    X_val_lead1   = ECG_val_raw[:, 0, :, :]
    X_test_lead1  = ECG_test_raw[:, 0, :, :]

    X_filt_train = np.array([bandpass_filter_ecg(x, fs=ECG_SAMPLING_RATE_HZ) for x in X_train_lead1])
    X_filt_val   = np.array([bandpass_filter_ecg(x, fs=ECG_SAMPLING_RATE_HZ) for x in X_val_lead1])
    X_filt_test  = np.array([bandpass_filter_ecg(x, fs=ECG_SAMPLING_RATE_HZ) for x in X_test_lead1])

    X_ts_train = np.swapaxes(X_filt_train, 1, 2)
    X_ts_val   = np.swapaxes(X_filt_val, 1, 2)
    X_ts_test  = np.swapaxes(X_filt_test, 1, 2)

    # ── STEP C: EXTRACTION FROM EXPLICIT SPLITS ───────────────────────
    train_ids = Echo_data.index[Echo_data['split'] == 'train'].tolist()
    val_ids   = Echo_data.index[Echo_data['split'] == 'val'].tolist()
    test_ids  = Echo_data.index[Echo_data['split'] == 'test'].tolist()


    ###########################################################################
    # ── Only use a small subset of data for testing if needed
    # ECG_train_raw = np.load("../EchoNextData/ECG_waveforms_2K.npy")
    # ECG_val_raw = np.load("../EchoNextData/ECG_waveforms_2K.npy")
    # ECG_test_raw = np.load("../EchoNextData/ECG_waveforms_2K.npy")
    # Echo_data = pd.read_csv("../EchoNextData/EchoNext_EchoData_2K.csv")

    # # ── STEP B: WAVEFORM PRE-FILTERING ──────────────────────────────


    # X_train_lead1 = ECG_train_raw
    # X_val_lead1   = ECG_val_raw
    # X_test_lead1  = ECG_test_raw
    

    # X_filt_train = np.array([bandpass_filter_ecg(x, fs=ECG_SAMPLING_RATE_HZ) for x in X_train_lead1])
    # X_filt_val   = np.array([bandpass_filter_ecg(x, fs=ECG_SAMPLING_RATE_HZ) for x in X_val_lead1])
    # X_filt_test  = np.array([bandpass_filter_ecg(x, fs=ECG_SAMPLING_RATE_HZ) for x in X_test_lead1])

    # X_ts_train = np.swapaxes(X_filt_train, 1, 2)
    # X_ts_val   = np.swapaxes(X_filt_val, 1, 2)
    # X_ts_test  = np.swapaxes(X_filt_test, 1, 2)
    # print(np.shape(X_filt_train),np.shape(X_ts_train))

    # # ── STEP C: EXTRACTION FROM EXPLICIT SPLITS ───────────────────────
    # train_ids = Echo_data.index[Echo_data['split'] == 'train'].tolist()
    # val_ids   = Echo_data.index[Echo_data['split'] == 'train'].tolist()
    # test_ids  = Echo_data.index[Echo_data['split'] == 'train'].tolist()
    ###########################################################################

    left = (
            Echo_data["lvef_lte_45_flag"].eq(1)
        | Echo_data["lvwt_gte_13_flag"].eq(1)
        )
    
    right = (
        Echo_data["rv_systolic_dysfunction_moderate_or_greater_flag"].eq(1)
        | Echo_data["pasp_gte_45_flag"].eq(1)
        | Echo_data["tr_max_gte_32_flag"].eq(1)
    )

    valve = (
        Echo_data["aortic_stenosis_moderate_or_greater_flag"].eq(1)
        | Echo_data["aortic_regurgitation_moderate_or_greater_flag"].eq(1)
        | Echo_data["mitral_regurgitation_moderate_or_greater_flag"].eq(1)
        | Echo_data["tricuspid_regurgitation_moderate_or_greater_flag"].eq(1)
        | Echo_data["pulmonary_regurgitation_moderate_or_greater_flag"].eq(1)
    )

    effusion = Echo_data["pericardial_effusion_moderate_large_flag"].eq(1)

    Echo_data["disease_class"] = np.select(
        [
            ~left & ~right & ~valve & ~effusion,
            left & ~right & ~valve & ~effusion,
            ~left & right & ~valve & ~effusion,
            ~left & ~right & valve & ~effusion,
            ~left & ~right & ~valve & effusion,
            left & ~right & valve & ~effusion,
            ~left & right & valve & ~effusion,
        ],
        [
            0,  # healthy
            1,  # left-sided, non-valvular
            2,  # right-sided, non-valvular
            3,  # valvular only
            4,  # effusion only
            5,  # left + valve
            6,  # right + valve
        ],
        default=7,  # other mixed
    )

    y_train = np.array(Echo_data['disease_class'][train_ids])
    y_val   = np.array(Echo_data['disease_class'][val_ids])
    y_test  = np.array(Echo_data['disease_class'][test_ids])

    X_demo_train, scaler, imputer, categories, feature_names = preprocess_demographics(Echo_data.loc[train_ids], fit=True)
    X_demo_val, *_ = preprocess_demographics(Echo_data.loc[val_ids], scaler, imputer, categories, fit=False)
    X_demo_test, *_ = preprocess_demographics(Echo_data.loc[test_ids], scaler, imputer, categories, fit=False)
    train_dataset = FusedDataset(X_ts_train, X_demo_train, y_train)
    val_dataset = FusedDataset(X_ts_val, X_demo_val, y_val)
    test_dataset = FusedDataset(X_ts_test, X_demo_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    network = BayesianLateFusionHeartDiseaseNet(X_ts_train.shape[1], X_demo_train.shape[1], num_classes=2)
    model = PyroLateFusionClassifier(network)
    guide = build_mean_field_guide(model)
    print("Training late-fusion model with Pyro SVI...", flush=True)
    svi, optimizer = train_bayesian_classifier(model, guide, train_loader, val_loader, epochs=10, learning_rate=1e-3, sample_nbr=3)

    results_dir = "latefusion_pyro_results_GELU_100K_prior2pt5_0pt5_postprior6"
    model.eval(); guide.eval()
    example_ts, example_demo = train_dataset.X_ts[:1].to(DEVICE), train_dataset.X_demo[:1].to(DEVICE)
    posterior_median_weights = {k: v.detach().cpu() for k, v in guide.median(example_ts, example_demo, None, len(train_dataset)).items()}
    checkpoint = {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "guide_state_dict": {k: v.detach().cpu() for k, v in guide.state_dict().items()},
        "pyro_param_store_state": pyro.get_param_store().get_state(),
        "optimizer_state": optimizer.get_state(), "posterior_median_weights": posterior_median_weights,
        "in_channels": X_ts_train.shape[1], "sequence_length": X_ts_train.shape[2],
        "demo_dim": X_demo_train.shape[1], "num_classes": 2, "feature_names": feature_names,
        "demographic_scaler": scaler, "demographic_imputer": imputer, "demographic_categories": categories,
        "prior_sigma_1": PRIOR_SIGMA_1, "prior_sigma_2": PRIOR_SIGMA_2,
        "prior_pi": PRIOR_PI, "guide_type": "AutoNormal",
    }
    os.makedirs(results_dir, exist_ok=True)
    torch.save(checkpoint, Path(results_dir) / "latefusion_pyro_checkpoint.pt")
    torch.save(posterior_median_weights, Path(results_dir) / "latefusion_pyro_posterior_median_weights.pt")
    

    # for name, dataset, ids in (("train", train_dataset, train_ids), ("test", test_dataset, test_ids)):
    #     metrics = save_posterior_results(model, guide, dataset, name, results_dir, ids, mc_samples=200)
    #     print(f"{name} posterior-mean metrics: {metrics}", flush=True)
    # print(f"All model weights and posterior predictions were saved to {results_dir}", flush=True)


    # ── STEP F: FULL POSTERIOR PREDICTIVE EVALUATION ──────────────────
    POSTERIOR_MC_SAMPLES = 200

    # Full-dataset posterior draws must use one forward pass per draw.
    # CPU is safer for memory; switch to device only if the entire split fits.
    posterior_evaluation_device = torch.device("cpu")

    train_results = evaluate_posterior_model(
        model=model,
        guide=guide,
        dataset=train_dataset,
        split_name="train",
        results_dir=results_dir,
        mc_samples=POSTERIOR_MC_SAMPLES,
        metadata_ids=train_ids,
        evaluation_device=posterior_evaluation_device,
    )

    test_results = evaluate_posterior_model(
        model=model,
        guide=guide,
        dataset=test_dataset,
        split_name="test",
        results_dir=results_dir,
        mc_samples=POSTERIOR_MC_SAMPLES,
        metadata_ids=test_ids,
        evaluation_device=posterior_evaluation_device,
    )

    plot_path = save_evaluation_plots(train_results, test_results, results_dir)
    print(f"Posterior evaluation plot saved to: {plot_path}", flush=True)
    print(f"\nAll posterior outputs saved to: {results_dir}", flush=True)
