import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import constraints
from torch.utils.data import Dataset, DataLoader

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.nn import PyroModule, PyroSample
from pyro.optim import ReduceLROnPlateau
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, roc_curve,
    brier_score_loss, confusion_matrix, ConfusionMatrixDisplay
)

# Install with: pip install pyro-ppl

# ──────────────────────────────────────────────────────────────────────
# 0. CONFIGURATION & HYPERPARAMETERS
# ──────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
pyro.set_rng_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}" , flush = True)

PRIOR_SIGMA_1 = 3.0#1.5
PRIOR_SIGMA_2 = 0.1#0.01#0.0025
PRIOR_PI = 1.0
POSTERIOR_RHO_INIT = -6.0#-5.0

CATEGORICAL_COLS = ["race_ethnicity", "sex"]
CONTINUOUS_COLS = [
    "age_at_ecg", "ventricular_rate", "atrial_rate",
    "pr_interval", "qrs_duration", "qt_corrected"
]
ALL_DEMO_COLS = CATEGORICAL_COLS + CONTINUOUS_COLS

# ──────────────────────────────────────────────────────────────────────
# 1. SIGNAL PROCESSING & SIGNAL FILTERING
# ──────────────────────────────────────────────────────────────────────
ECG_SAMPLING_RATE_HZ = 250
FILTER_LOW_HZ = 0.5
FILTER_HIGH_HZ = 40.0
FILTER_ORDER = 4

def bandpass_filter_ecg(signal_array, fs=ECG_SAMPLING_RATE_HZ,
                         low=FILTER_LOW_HZ, high=FILTER_HIGH_HZ, order=FILTER_ORDER):
    nyquist = 0.5 * fs
    b, a = butter(order, [low / nyquist, high / nyquist], btype="band")
    # signal_array shape: (time, leads) — filter along time axis (axis=0), leads handled independently
    filtered = filtfilt(b, a, signal_array, axis=0)
    return filtered.astype(np.float32)

# ──────────────────────────────────────────────────────────────────────
# 2. DEMOGRAPHIC PREPROCESSING
# ──────────────────────────────────────────────────────────────────────
def preprocess_demographics(df, scaler=None, imputer=None, categories=None, knn_k=5, fit=True):
    df = df[ALL_DEMO_COLS].copy()

    if categories is None:
        if not fit:
            raise ValueError("`categories` dictionary map must be provided when fit=False.")
        categories = {col: df[col].astype("category").cat.categories for col in CATEGORICAL_COLS}

    cat_codes = pd.DataFrame(index=df.index)
    for col in CATEGORICAL_COLS:
        cat_codes[col] = pd.Categorical(df[col], categories=categories[col]).codes
    cat_codes = cat_codes.replace(-1, np.nan)

    cont_part = df[CONTINUOUS_COLS].astype(float)
    imputer_input = pd.concat([cat_codes, cont_part], axis=1)

    if imputer is None:
        if not fit:
            raise ValueError("`imputer` object instance must be explicitly provided when fit=False.")
        imputer = KNNImputer(n_neighbors=knn_k)

    if fit:
        imputed = imputer.fit_transform(imputer_input)
    else:
        imputed = imputer.transform(imputer_input)

    imputed_df = pd.DataFrame(imputed, columns=ALL_DEMO_COLS, index=df.index)

    for col in CATEGORICAL_COLS:
        cats = categories[col]
        codes_rounded = imputed_df[col].round().astype(int).clip(0, len(cats) - 1)
        imputed_df[col] = pd.Categorical.from_codes(codes_rounded, categories=cats)

    ohe_df = pd.get_dummies(imputed_df, columns=CATEGORICAL_COLS, dtype=float)
    feature_names = list(ohe_df.columns)

    if scaler is None:
        if not fit:
            raise ValueError("`scaler` must be provided when fit=False.")
        scaler = StandardScaler()

    cont_indices = [feature_names.index(c) for c in CONTINUOUS_COLS]
    arr = ohe_df.values.astype(np.float32)

    if fit:
        arr[:, cont_indices] = scaler.fit_transform(arr[:, cont_indices])
    else:
        arr[:, cont_indices] = scaler.transform(arr[:, cont_indices])

    return arr, scaler, imputer, categories, feature_names

# ──────────────────────────────────────────────────────────────────────
# 3. PYTORCH DATA STRUCTURING
# ──────────────────────────────────────────────────────────────────────
class FusedDataset(Dataset):
    def __init__(self, X_ts, X_demo, y):
        print(np.shape(X_ts),np.shape(X_demo),np.shape(y))
        assert len(X_ts) == len(X_demo) == len(y), "Length mismatch across tensor targets."
        self.X_ts = torch.tensor(X_ts, dtype=torch.float32)
        self.X_demo = torch.tensor(X_demo, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_ts[idx], self.X_demo[idx], self.y[idx]

# ──────────────────────────────────────────────────────────────────────
# 4. BAYESIAN NETWORK ARCHITECTURE
# ──────────────────────────────────────────────────────────────────────
class RealSupportMixtureSameFamily(dist.MixtureSameFamily):
    """MixtureSameFamily with the mathematically correct real-valued support.

    PyTorch currently reports a MixtureSameFamilyConstraint for this
    distribution. AutoNormal then tries to construct a bijective transform for
    that constraint, but no such transform is registered. A Gaussian mixture is
    supported on the whole real line, so declaring constraints.real preserves
    the exact sampling and log_prob behavior while allowing AutoNormal to build
    its unconstrained Normal guide.
    """

    support = constraints.real


class _ScaleMixturePriorMixin:
    """BLITZ-compatible zero-mean two-component Gaussian scale-mixture prior."""

    def _register_prior_buffers(self):
        self.register_buffer(
            "_prior_probs",
            torch.tensor([PRIOR_PI, 1.0 - PRIOR_PI], dtype=torch.float32),
        )
        self.register_buffer(
            "_prior_locs",
            torch.zeros(2, dtype=torch.float32),
        )
        self.register_buffer(
            "_prior_scales",
            torch.tensor([PRIOR_SIGMA_1, PRIOR_SIGMA_2], dtype=torch.float32),
        )

    def _scale_mixture_prior(self, shape):
        mixture = RealSupportMixtureSameFamily(
            dist.Categorical(probs=self._prior_probs),
            dist.Normal(self._prior_locs, self._prior_scales),
        )
        return mixture.expand(shape).to_event(len(shape))


class PyroBayesianLinear(_ScaleMixturePriorMixin, PyroModule):
    """Bayesian linear layer with PyroSample weight and bias priors."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._register_prior_buffers()

        weight_shape = (out_features, in_features)
        self.weight = PyroSample(
            lambda module: module._scale_mixture_prior(weight_shape)
        )

        if bias:
            bias_shape = (out_features,)
            self.bias = PyroSample(
                lambda module: module._scale_mixture_prior(bias_shape)
            )
        else:
            self.bias = None

    def forward(self, inputs):
        return F.linear(inputs, self.weight, self.bias)


class PyroBayesianConv1d(_ScaleMixturePriorMixin, PyroModule):
    """Bayesian Conv1d layer with the same tensor layout as nn.Conv1d."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self._register_prior_buffers()

        weight_shape = (out_channels, in_channels // groups, kernel_size)
        self.weight = PyroSample(
            lambda module: module._scale_mixture_prior(weight_shape)
        )

        if bias:
            bias_shape = (out_channels,)
            self.bias = PyroSample(
                lambda module: module._scale_mixture_prior(bias_shape)
            )
        else:
            self.bias = None

    def forward(self, inputs):
        return F.conv1d(
            inputs,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class BayesianFiLMLayer(PyroModule):
    def __init__(self, demo_embed_dim, num_channels):
        super().__init__()
        self.film_proj = PyroBayesianLinear(
            demo_embed_dim,
            2 * num_channels,
        )

    def forward(self, h, demo_embed):
        params = self.film_proj(demo_embed)
        gamma, beta = params.chunk(2, dim=-1)
        return gamma.unsqueeze(-1) * h + beta.unsqueeze(-1)


class BayesianFiLMConvBlock(PyroModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        pool_size,
        demo_embed_dim,
        pool_type="avg",
    ):
        super().__init__()
        self.conv = PyroBayesianConv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.film = BayesianFiLMLayer(demo_embed_dim, out_channels)
        self.act = nn.GELU()
        self.pool = (
            nn.AdaptiveAvgPool1d(1)
            if pool_type == "gap"
            else nn.AvgPool1d(kernel_size=pool_size)
        )

    def forward(self, x, demo_embed):
        x = self.conv(x)
        x = self.bn(x)
        x = self.film(x, demo_embed)
        x = self.act(x)
        return self.pool(x)


class BayesianHeartDiseaseNet(PyroModule):
    """Same early-fusion FiLM architecture as the original BLITZ network."""

    def __init__(
        self,
        sequence_length,
        in_channels,
        demo_dim,
        num_classes=2,
        demo_embed_dim=16,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.in_channels = in_channels
        self.demo_dim = demo_dim
        self.num_classes = num_classes
        self.demo_embed_dim = demo_embed_dim

        self.demo_linear = PyroBayesianLinear(demo_dim, demo_embed_dim)
        self.demo_activation = nn.GELU()

        self.block1 = BayesianFiLMConvBlock(
            in_channels,
            16,
            kernel_size=15,
            padding=7,
            pool_size=10,
            demo_embed_dim=demo_embed_dim,
            pool_type="avg",
        )
        self.block2 = BayesianFiLMConvBlock(
            16,
            32,
            kernel_size=9,
            padding=4,
            pool_size=10,
            demo_embed_dim=demo_embed_dim,
            pool_type="avg",
        )
        self.block3 = BayesianFiLMConvBlock(
            32,
            32,
            kernel_size=5,
            padding=2,
            pool_size=None,
            demo_embed_dim=demo_embed_dim,
            pool_type="gap",
        )

        # block3 ends in AdaptiveAvgPool1d(1), so flattening produces 32 features.
        self.head_linear1 = PyroBayesianLinear(32, 32)
        self.head_activation = nn.GELU()
        self.head_linear2 = PyroBayesianLinear(32, num_classes)

    def forward(self, x_ts, x_demo):
        demo_embed = self.demo_activation(self.demo_linear(x_demo))
        x = self.block1(x_ts, demo_embed)
        x = self.block2(x, demo_embed)
        x = self.block3(x, demo_embed)
        x = torch.flatten(x, start_dim=1)
        x = self.head_activation(self.head_linear1(x))
        return self.head_linear2(x)


class PyroHeartDiseaseClassifier(PyroModule):
    """Probabilistic wrapper that adds the categorical observation model."""

    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x_ts, x_demo, y=None, dataset_size=None):
        logits = self.network(x_ts, x_demo)

        if y is not None:
            batch_size = x_ts.shape[0]
            likelihood_scale = (
                float(dataset_size) / float(batch_size)
                if dataset_size is not None
                else 1.0
            )
            with poutine.scale(scale=likelihood_scale):
                with pyro.plate("data", batch_size):
                    pyro.sample("obs", dist.Categorical(logits=logits), obs=y)

        return logits


def build_mean_field_guide(model):
    """Diagonal Normal variational posterior, initialized from BLITZ rho."""
    posterior_init_scale = float(F.softplus(torch.tensor(POSTERIOR_RHO_INIT)))
    return AutoNormal(model, init_scale=posterior_init_scale)


# 5. TRAINING ROUTINE
# ──────────────────────────────────────────────────────────────────────
def sample_posterior_logits(model, guide, x_ts, x_demo, mc_samples=1):
    """Draw posterior weights from the guide and replay them through the model."""
    sampled_logits = []
    for _ in range(mc_samples):
        guide_trace = poutine.trace(guide).get_trace(
            x_ts,
            x_demo,
            None,
            None,
        )
        replayed_model = poutine.replay(model, trace=guide_trace)
        sampled_logits.append(
            replayed_model(x_ts, x_demo, None, None)
        )
    return torch.stack(sampled_logits, dim=0)


def train_bayesian_classifier(
    model,
    guide,
    train_loader,
    val_loader,
    epochs=20,
    learning_rate=1e-4,
    sample_nbr=3,
):
    pyro.clear_param_store()
    model = model.to(device)
    guide = guide.to(device)

    scheduler = ReduceLROnPlateau({
        "optimizer": torch.optim.Adam,
        "optim_args": {
            "lr": learning_rate,
            "weight_decay": 1e-4,
        },
        "mode": "min",
        "factor": 0.5,
        "patience": 5,
    })
    svi = SVI(
        model,
        guide,
        scheduler,
        loss=Trace_ELBO(num_particles=sample_nbr),
    )
    n_train = len(train_loader.dataset)

    for epoch in range(epochs):
        model.train()
        guide.train()
        epoch_elbo = 0.0

        for X_ts_b, X_demo_b, y_b in train_loader:
            X_ts_b = X_ts_b.to(device)
            X_demo_b = X_demo_b.to(device)
            y_b = y_b.to(device)

            # The likelihood is scaled to the full training-set size inside model().
            epoch_elbo += svi.step(
                X_ts_b,
                X_demo_b,
                y_b,
                n_train,
            )

        model.eval()
        guide.eval()
        val_loss = 0.0
        val_total = 0
        epoch_acc = 0.0

        with torch.no_grad():
            val_correct = 0  # Track total correct predictions

            for step, (X_ts_v, X_demo_v, y_v) in enumerate(val_loader):
                X_ts_v = X_ts_v.to(device)
                X_demo_v = X_demo_v.to(device)
                y_v = y_v.to(device)

                pass_logits = sample_posterior_logits(
                    model,
                    guide,
                    X_ts_v,
                    X_demo_v,
                    mc_samples=10,
                )
                pass_log_probs = torch.log_softmax(pass_logits, dim=-1)
                log_mean_probs = torch.logsumexp(pass_log_probs, dim=0) - np.log(10)

                # 1. Compute loss
                batch_loss = F.nll_loss(
                    log_mean_probs,
                    y_v,
                    reduction="sum",
                )
                val_loss += batch_loss.item()
                
                # 2. Get class predictions
                # For multi-class classification: take argmax along class dimension
                # For binary classification with single logit: use (log_mean_probs > np.log(0.5))
                preds = log_mean_probs.argmax(dim=-1)
                
                # 3. Calculate batch accuracy
                batch_correct = (preds == y_v).sum().item()
                batch_acc = batch_correct / y_v.size(0)
                
                # 4. Accumulate totals
                val_correct += batch_correct
                val_total += y_v.size(0)

                # Print / log accuracy at each step
                # print(f"Step {step+1}/{len(val_loader)} - Batch Acc: {batch_acc:.4f} ({batch_correct}/{y_v.size(0)})")

        # Final metric calculations after the loop
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        epoch_elbo_per_patient = epoch_elbo / max(n_train, 1)
        
        print(
            f"Epoch {epoch + 1:02d} | "
            f"Train negative ELBO/patient: {epoch_elbo_per_patient:.6f} | "
            f"Dedicated Validation Loss: {epoch_val_loss:.6f}",
            f"Accuracy: {epoch_val_acc:.6f}",
            flush=True,
        )
        scheduler.step(epoch_val_loss)

    return svi, scheduler


# 6. PERFORMANCE METRICS GENERATION & CALIBRATION ESTIMATION
# ──────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────
# 6. POSTERIOR PREDICTION, ERROR METRICS, CALIBRATION, AND UNCERTAINTY
# ──────────────────────────────────────────────────────────────────────
def _binary_entropy(probs, eps=1e-8):
    probs = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0 - eps)
    return -(probs * np.log(probs) + (1.0 - probs) * np.log(1.0 - probs))


def calculate_binary_risk_ece(probs, labels, bins=10):
    """ECE for positive-class risk: predicted event probability vs observed event rate."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0

    for i in range(bins):
        if i == bins - 1:
            in_bin = (probs >= boundaries[i]) & (probs <= boundaries[i + 1])
        else:
            in_bin = (probs >= boundaries[i]) & (probs < boundaries[i + 1])

        if np.any(in_bin):
            predicted_rate = probs[in_bin].mean()
            observed_rate = labels[in_bin].mean()
            ece += in_bin.mean() * abs(predicted_rate - observed_rate)

    return float(ece)


def calculate_confidence_ece(probs, labels, bins=10):
    """Classification ECE using confidence in the predicted class."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    preds = (probs >= 0.5).astype(np.int64)
    confidence = np.where(preds == 1, probs, 1.0 - probs)
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
            class1_probs = torch.softmax(logits, dim=-1)[:, 1]
            posterior_draws.append(class1_probs.cpu().numpy())

            print(
                f"Posterior draw {draw_index + 1}/{mc_samples}",
                end="\r",
                flush=True,
            )

    print(flush=True)

    posterior_probs = np.stack(posterior_draws, axis=0).astype(np.float32)

    if posterior_probs.shape != (mc_samples, len(dataset)):
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


def calculate_binary_metrics(probs, labels, threshold=0.5, calibration_bins=10):
    """Calculate metrics for one vector of positive-class probabilities."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    preds = (probs >= threshold).astype(np.int64)
    eps = 1e-8

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    # npv = tn / (tn + fn) if (tn + fn) else np.nan
    f1 = 2.0 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else np.nan
    balanced_accuracy = np.nanmean([sensitivity, specificity])

    precision_curve, recall_curve, _ = precision_recall_curve(labels, probs)
    true_class_probs = np.where(labels == 1, probs, 1.0 - probs)

    return {
        "accuracy": float(np.mean(preds == labels)),
        "balanced_accuracy": float(balanced_accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision_ppv": float(precision),
        # "negative_predictive_value": float(npv),
        "f1_score": float(f1),
        "auroc": float(roc_auc_score(labels, probs)),
        "auprc": float(auc(recall_curve, precision_curve)),
        "brier_score": float(brier_score_loss(labels, probs)),
        "negative_log_likelihood": float(-np.mean(np.log(np.clip(true_class_probs, eps, 1.0)))),
        "binary_risk_ece": calculate_binary_risk_ece(probs, labels, bins=calibration_bins),
        "confidence_ece": calculate_confidence_ece(probs, labels, bins=calibration_bins),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def posterior_metric_distribution(posterior_probs, labels, threshold=0.5, calibration_bins=10):
    """Calculate every error metric separately for each posterior probability draw."""
    records = []
    for draw_index, probs in enumerate(posterior_probs):
        row = calculate_binary_metrics(
            probs,
            labels,
            threshold=threshold,
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


def summarize_patient_posteriors(posterior_probs, labels, metadata_ids=None, threshold=0.5):
    """Create patient-level posterior predictive and uncertainty summaries."""
    mean_probs = posterior_probs.mean(axis=0)
    median_probs = np.median(posterior_probs, axis=0)
    std_probs = posterior_probs.std(axis=0, ddof=1)
    lower = np.quantile(posterior_probs, 0.025, axis=0)
    upper = np.quantile(posterior_probs, 0.975, axis=0)
    predictive_entropy = _binary_entropy(mean_probs)
    expected_entropy = _binary_entropy(posterior_probs).mean(axis=0)
    mutual_information = np.maximum(predictive_entropy - expected_entropy, 0.0)
    predictions = (mean_probs >= threshold).astype(np.int64)
    confidence = np.where(predictions == 1, mean_probs, 1.0 - mean_probs)

    result = pd.DataFrame({
        "sample_index": np.arange(len(labels)),
        "true_label": labels,
        "predicted_label": predictions,
        "posterior_mean_class0": 1.0 - mean_probs,
        "posterior_mean_class1": mean_probs,
        "posterior_median_class1": median_probs,
        "posterior_sd_class1": std_probs,
        "posterior_lower_95_class1": lower,
        "posterior_upper_95_class1": upper,
        "prediction_confidence": confidence,
        "predictive_entropy": predictive_entropy,
        "expected_entropy": expected_entropy,
        "mutual_information": mutual_information,
    })

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
    threshold=0.5,
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

    posterior_mean_metrics = calculate_binary_metrics(
        mean_probs,
        labels,
        threshold=threshold,
        calibration_bins=calibration_bins,
    )
    posterior_mean_metrics_df = pd.DataFrame([
        {"metric": metric, "value": value}
        for metric, value in posterior_mean_metrics.items()
    ])

    metric_draws_df = posterior_metric_distribution(
        posterior_probs,
        labels,
        threshold=threshold,
        calibration_bins=calibration_bins,
    )
    metric_summary_df = summarize_posterior_metrics(metric_draws_df)
    patient_summary_df = summarize_patient_posteriors(
        posterior_probs,
        labels,
        metadata_ids=metadata_ids,
        threshold=threshold,
    )

    np.savez_compressed(
        os.path.join(results_dir, f"{split_name}_posterior_predictive.npz"),
        posterior_class0_probs=1.0 - posterior_probs,
        posterior_class1_probs=posterior_probs,
        posterior_mean_class0=1.0 - mean_probs,
        posterior_mean_class1=mean_probs,
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
        column for column in [
            "sample_index",
            "metadata_index",
            "true_label",
            "predicted_label",
            "posterior_mean_class0",
            "posterior_mean_class1",
            "prediction_confidence",
        ]
        if column in patient_summary_df.columns
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
    train_probs = train_results["mean_probs"]
    train_labels = train_results["labels"]
    test_probs = test_results["mean_probs"]
    test_labels = test_results["labels"]
    test_preds = (test_probs >= 0.5).astype(np.int64)

    train_fpr, train_tpr, _ = roc_curve(train_labels, train_probs)
    test_fpr, test_tpr, _ = roc_curve(test_labels, test_probs)
    train_precision, train_recall, _ = precision_recall_curve(train_labels, train_probs)
    test_precision, test_recall, _ = precision_recall_curve(test_labels, test_probs)

    train_auroc = roc_auc_score(train_labels, train_probs)
    test_auroc = roc_auc_score(test_labels, test_probs)
    train_auprc = auc(train_recall, train_precision)
    test_auprc = auc(test_recall, test_precision)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(train_fpr, train_tpr, lw=2, label=f"Train ROC (AUC={train_auroc:.3f})")
    ax[0].plot(test_fpr, test_tpr, lw=2, label=f"Test ROC (AUC={test_auroc:.3f})")
    ax[0].plot([0, 1], [0, 1], linestyle="--")
    ax[0].set_xlabel("False Positive Rate")
    ax[0].set_ylabel("True Positive Rate")
    ax[0].set_title("Posterior-Mean ROC")
    ax[0].legend(loc="lower right")

    ax[1].plot(train_recall, train_precision, lw=2, label=f"Train PR (AUC={train_auprc:.3f})")
    ax[1].plot(test_recall, test_precision, lw=2, label=f"Test PR (AUC={test_auprc:.3f})")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].set_title("Posterior-Mean Precision-Recall")
    ax[1].legend(loc="lower left")

    cm_test = confusion_matrix(test_labels, test_preds, labels=[0, 1])
    ConfusionMatrixDisplay(cm_test, display_labels=["No SHD", "SHD"]).plot(ax=ax[2], values_format="d")
    ax[2].set_title("Posterior-Mean Test Confusion Matrix")

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "early_fusion_posterior_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    summary = test_results["metric_summary"].copy()
    selected = summary[summary["metric"].isin([
        "accuracy", "balanced_accuracy", "auroc", "auprc",
        "brier_score", "negative_log_likelihood",
        "binary_risk_ece", "confidence_ece"
    ])].reset_index(drop=True)

    # fig, ax = plt.subplots(figsize=(11, 5.5))
    # x = np.arange(len(selected))
    # means = selected["posterior_mean"].to_numpy()
    # lower_error = means - selected["lower_95"].to_numpy()
    # upper_error = selected["upper_95"].to_numpy() - means
    # ax.errorbar(x, means, yerr=np.vstack([lower_error, upper_error]), fmt="o", capsize=4)
    # ax.set_xticks(x)
    # ax.set_xticklabels(selected["metric"], rotation=35, ha="right")
    # ax.set_ylabel("Metric value")
    # ax.set_title("Test Posterior Metric Means and 95% Intervals")
    # ax.grid(axis="y", alpha=0.3)
    # fig.tight_layout()
    # fig.savefig(os.path.join(results_dir, "test_posterior_metric_intervals.png"), dpi=150, bbox_inches="tight")
    # plt.close(fig)

# ──────────────────────────────────────────────────────────────────────
# 7. EXECUTION PIPELINE
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Beginning execution pipeline...", flush = True)
###################################################################################

    ###### ── STEP A: DATA INGESTION ────
    ####### ── FULL DATASET
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

###################################################################################
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

###################################################################################

    y_train = np.array(Echo_data['shd_moderate_or_greater_flag'][train_ids])
    y_val   = np.array(Echo_data['shd_moderate_or_greater_flag'][val_ids])
    y_test  = np.array(Echo_data['shd_moderate_or_greater_flag'][test_ids])

    # ── STEP D: TRANSFORM DEMOGRAPHICS ──
    demo_df_train = Echo_data[ALL_DEMO_COLS].iloc[train_ids].copy()
    demo_df_val   = Echo_data[ALL_DEMO_COLS].iloc[val_ids].copy()
    demo_df_test  = Echo_data[ALL_DEMO_COLS].iloc[test_ids].copy()

    X_demo_train, scaler, imputer, categories, feature_names = preprocess_demographics(
        demo_df_train, fit=True
    )
    X_demo_val, _, _, _, _ = preprocess_demographics(
        demo_df_val, scaler=scaler, imputer=imputer, categories=categories, fit=False
    )
    X_demo_test, _, _, _, _ = preprocess_demographics(
        demo_df_test, scaler=scaler, imputer=imputer, categories=categories, fit=False
    )

    train_dataset = FusedDataset(X_ts_train, X_demo_train, y_train)
    val_dataset = FusedDataset(X_ts_val, X_demo_val, y_val)
    test_dataset = FusedDataset(X_ts_test, X_demo_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # ── STEP E: INITIALIZE & TRAIN THE MODEL ──────────────────────────
    network = BayesianHeartDiseaseNet(
        sequence_length=X_ts_train.shape[2],
        in_channels=X_ts_train.shape[1],
        demo_dim=X_demo_train.shape[1],
        num_classes=2,
        demo_embed_dim=16,
    )
    model = PyroHeartDiseaseClassifier(network)
    guide = build_mean_field_guide(model)

    print("Training model with Pyro SVI...", flush=True)
    svi, scheduler = train_bayesian_classifier(
        model,
        guide,
        train_loader,
        val_loader,
        epochs=60,
        learning_rate=1e-3,
        sample_nbr=3,
    )

    RESULTS_DIR = "earlyfusion_pyro_results_GELU_8_24_26_prior3_postprior6_FULL"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model.eval()
    guide.eval()
    example_ts = train_dataset.X_ts[:1].to(device)
    example_demo = train_dataset.X_demo[:1].to(device)
    with torch.no_grad():
        posterior_median_weights = guide.median(
            example_ts,
            example_demo,
            None,
            len(train_dataset),
        )

    posterior_median_weights_cpu = {
        name: value.detach().cpu()
        for name, value in posterior_median_weights.items()
    }
    deterministic_state_cpu = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }

    checkpoint = {
        "model_state_dict": deterministic_state_cpu,
        "guide_state_dict": {
            name: value.detach().cpu()
            for name, value in guide.state_dict().items()
        },
        "pyro_param_store_state": pyro.get_param_store().get_state(),
        "optimizer_state": scheduler.get_state(),
        "posterior_median_weights": posterior_median_weights_cpu,
        "sequence_length": X_ts_train.shape[2],
        "in_channels": X_ts_train.shape[1],
        "demo_dim": X_demo_train.shape[1],
        "demo_embed_dim": 16,
        "num_classes": 2,
        "random_seed": RANDOM_SEED,
        "feature_names": feature_names,
        "demographic_scaler": scaler,
        "demographic_imputer": imputer,
        "demographic_categories": categories,
        "prior_sigma_1": PRIOR_SIGMA_1,
        "prior_sigma_2": PRIOR_SIGMA_2,
        "prior_pi": PRIOR_PI,
        "posterior_rho_init": POSTERIOR_RHO_INIT,
        "guide_type": "AutoNormal",
    }

    torch.save(
        checkpoint,
        os.path.join(RESULTS_DIR, "earlyfusion_pyro_checkpoint.pt"),
    )
    torch.save(
        {
            "posterior_median_weights": posterior_median_weights_cpu,
            "deterministic_state_dict": deterministic_state_cpu,
        },
        os.path.join(RESULTS_DIR, "earlyfusion_pyro_posterior_median_weights.pt"),
    )

    torch.manual_seed(RANDOM_SEED)
    pyro.set_rng_seed(RANDOM_SEED)

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
        results_dir=RESULTS_DIR,
        mc_samples=POSTERIOR_MC_SAMPLES,
        metadata_ids=train_ids,
        evaluation_device=posterior_evaluation_device,
    )

    test_results = evaluate_posterior_model(
        model=model,
        guide=guide,
        dataset=test_dataset,
        split_name="test",
        results_dir=RESULTS_DIR,
        mc_samples=POSTERIOR_MC_SAMPLES,
        metadata_ids=test_ids,
        evaluation_device=posterior_evaluation_device,
    )

    save_evaluation_plots(train_results, test_results, RESULTS_DIR)
    print(f"\nAll posterior outputs saved to: {RESULTS_DIR}", flush=True)
    
