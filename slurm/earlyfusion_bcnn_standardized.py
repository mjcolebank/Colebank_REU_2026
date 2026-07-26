import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, roc_curve,
    brier_score_loss, confusion_matrix, ConfusionMatrixDisplay
)

!pip install blitz-bayesian-pytorch
from blitz.modules import BayesianLinear, BayesianConv1d
from blitz.utils import variational_estimator

# ──────────────────────────────────────────────────────────────────────
# 0. CONFIGURATION & HYPERPARAMETERS
# ──────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}" , flush = True)

PRIOR_SIGMA_1 = 1.0
PRIOR_SIGMA_2 = 0.0025
PRIOR_PI = 0.5
POSTERIOR_RHO_INIT = -5.0

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
@variational_estimator
class BayesianFiLMLayer(nn.Module):
    def __init__(self, demo_embed_dim, num_channels):
        super().__init__()
        self.film_proj = BayesianLinear(
            demo_embed_dim, 2 * num_channels,
            prior_sigma_1=PRIOR_SIGMA_1, prior_sigma_2=PRIOR_SIGMA_2, prior_pi=PRIOR_PI,
            posterior_rho_init=POSTERIOR_RHO_INIT
        )

    def forward(self, h, demo_embed):
        params = self.film_proj(demo_embed)
        gamma, beta = params.chunk(2, dim=-1)
        return gamma.unsqueeze(-1) * h + beta.unsqueeze(-1)

@variational_estimator
class BayesianFiLMConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, pool_size, demo_embed_dim, pool_type="avg"):
        super().__init__()
        self.conv = BayesianConv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding,
            prior_sigma_1=PRIOR_SIGMA_1, prior_sigma_2=PRIOR_SIGMA_2, prior_pi=PRIOR_PI,
            posterior_rho_init=POSTERIOR_RHO_INIT
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.film = BayesianFiLMLayer(demo_embed_dim, out_channels)
        self.act = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(1) if pool_type == "gap" else nn.AvgPool1d(kernel_size=pool_size)

    def forward(self, x, demo_embed):
        return self.pool(self.act(self.film(self.bn(self.conv(x)), demo_embed)))

@variational_estimator
class BayesianHeartDiseaseNet(nn.Module):
    def __init__(self, sequence_length, in_channels, demo_dim, num_classes=2, demo_embed_dim=16):
        super().__init__()
        def blinear(in_f, out_f):
            return BayesianLinear(
                in_f, out_f, prior_sigma_1=PRIOR_SIGMA_1, prior_sigma_2=PRIOR_SIGMA_2,
                prior_pi=PRIOR_PI, posterior_rho_init=POSTERIOR_RHO_INIT
            )

        self.demo_encoder = nn.Sequential(blinear(demo_dim, demo_embed_dim), nn.GELU())
        self.block1 = BayesianFiLMConvBlock(in_channels, 16, kernel_size=15, padding=7, pool_size=10, demo_embed_dim=demo_embed_dim, pool_type="avg")
        self.block2 = BayesianFiLMConvBlock(16, 32, kernel_size=9, padding=4, pool_size=10, demo_embed_dim=demo_embed_dim, pool_type="avg")
        self.block3 = BayesianFiLMConvBlock(32, 32, kernel_size=5, padding=2, pool_size=None, demo_embed_dim=demo_embed_dim, pool_type="gap")

        with torch.no_grad():
            dummy_ts = torch.zeros(1, in_channels, sequence_length)
            dummy_demo = torch.zeros(1, demo_embed_dim)
            x = self.block3(self.block2(self.block1(dummy_ts, dummy_demo), dummy_demo), dummy_demo)
            flat_sz = x.shape[1] * x.shape[2]

        self.head = nn.Sequential(nn.Flatten(), blinear(flat_sz, 32), nn.GELU(), blinear(32, num_classes))

    def forward(self, x_ts, x_demo):
        demo_embed = self.demo_encoder(x_demo)
        return self.head(self.block3(self.block2(self.block1(x_ts, demo_embed), demo_embed), demo_embed))

# ──────────────────────────────────────────────────────────────────────
# 5. TRAINING ROUTINE
# ──────────────────────────────────────────────────────────────────────
def train_bayesian_classifier(model, train_loader, val_loader, epochs=60, learning_rate=1e-4, sample_nbr=3):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    n_train = len(train_loader.dataset)

    for epoch in range(epochs):
        model.train()
        for X_ts_b, X_demo_b, y_b in train_loader:
            X_ts_b, X_demo_b, y_b = X_ts_b.to(device), X_demo_b.to(device), y_b.to(device)
            X_ts_b += 0.05 * torch.randn_like(X_ts_b)

            optimizer.zero_grad()
            nll = 0.0
            for _ in range(sample_nbr):
                logits = model(X_ts_b, X_demo_b)
                nll += criterion(logits, y_b)

            loss = (nll / sample_nbr) + (model.nn_kl_divergence() / n_train)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss, val_total = 0.0, 0
        with torch.no_grad():
            for X_ts_v, X_demo_v, y_v in val_loader:
                X_ts_v, X_demo_v, y_v = X_ts_v.to(device), X_demo_v.to(device), y_v.to(device)

                # Correct MC predictive: average probabilities, not logits
                pass_logits = torch.stack([model(X_ts_v, X_demo_v) for _ in range(10)], dim=0)
                pass_log_probs = torch.log_softmax(pass_logits, dim=-1)
                log_mean_probs = torch.logsumexp(pass_log_probs, dim=0) - np.log(10)

                batch_loss = nn.functional.nll_loss(log_mean_probs, y_v, reduction='sum')
                val_loss += batch_loss.item()
                val_total += X_ts_v.shape[0]

        epoch_val_loss = val_loss / val_total
        print(f"Epoch {epoch+1:02d} | Dedicated Validation Loss: {epoch_val_loss:.6f}", flush = True)
        scheduler.step(epoch_val_loss)

# ──────────────────────────────────────────────────────────────────────
# 6. PERFORMANCE METRICS GENERATION & CALIBRATION ESTIMATION
# ──────────────────────────────────────────────────────────────────────
def calculate_calibration_error(probs, labels, bins=10):
    bin_boundaries = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin] == (probs[in_bin] >= 0.5))
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += prop_in_bin * np.abs(avg_confidence_in_bin - accuracy_in_bin)
    return ece

def get_predictions_and_labels(model, loader, sample_nbr=50):
    model.eval()
    all_probs, all_labels = [], []
    all_nll = 0.0

    with torch.no_grad():
        for X_ts_b, X_demo_b, y_b in loader:
            X_ts_b, X_demo_b, y_b = X_ts_b.to(device), X_demo_b.to(device), y_b.to(device)
            pass_logits = torch.stack([model(X_ts_b, X_demo_b) for _ in range(sample_nbr)], dim=0)  # (S, B, C)

            # Correct MC predictive: average PROBABILITIES across samples, not logits.
            # Use logsumexp for numerical stability instead of mean(softmax(.)).log()
            pass_log_probs = torch.log_softmax(pass_logits, dim=-1)                       # (S, B, C)
            log_mean_probs = torch.logsumexp(pass_log_probs, dim=0) - np.log(sample_nbr)  # (B, C)
            mean_probs = log_mean_probs.exp()

            nll_batch = nn.functional.nll_loss(log_mean_probs, y_b, reduction='sum')
            all_nll += nll_batch.item()

            all_probs.extend(mean_probs[:, 1].cpu().numpy())
            all_labels.extend(y_b.cpu().numpy())

    return np.array(all_probs), np.array(all_labels), all_nll / len(loader.dataset)

def generate_and_display_comparative_metrics(model, train_loader, test_loader, sample_nbr=50):
    train_probs, train_labels, train_nll = get_predictions_and_labels(model, train_loader, sample_nbr)
    test_probs, test_labels, test_nll = get_predictions_and_labels(model, test_loader, sample_nbr)

    train_preds = (train_probs >= 0.5).astype(int)
    test_preds = (test_probs >= 0.5).astype(int)

    metrics_map = {
        "accuracy": (np.mean(train_preds == train_labels), np.mean(test_preds == test_labels)),
        "auroc": (roc_auc_score(train_labels, train_probs), roc_auc_score(test_labels, test_probs)),
        "auprc": (None, None),
        "brier_score": (brier_score_loss(train_labels, train_probs), brier_score_loss(test_labels, test_probs)),
        "negative_log_likelihood": (train_nll, test_nll),
        "expected_calibration_error": (calculate_calibration_error(train_probs, train_labels), calculate_calibration_error(test_probs, test_labels))
    }

    train_precision, train_recall, _ = precision_recall_curve(train_labels, train_probs)
    train_fpr, train_tpr, _ = roc_curve(train_labels, train_probs)
    metrics_map["auprc"] = (auc(train_recall, train_precision), metrics_map["auprc"][1])

    test_precision, test_recall, _ = precision_recall_curve(test_labels, test_probs)
    test_fpr, test_tpr, _ = roc_curve(test_labels, test_probs)
    metrics_map["auprc"] = (metrics_map["auprc"][0], auc(test_recall, test_precision))

    tn_tr, fp_tr, fn_tr, tp_tr = confusion_matrix(train_labels, train_preds).ravel()
    tn_te, fp_te, fn_te, tp_te = confusion_matrix(test_labels, test_preds).ravel()

    metrics_map["tn"] = (float(tn_tr), float(tn_te))
    metrics_map["fp"] = (float(fp_tr), float(fp_te))
    metrics_map["fn"] = (float(fn_tr), float(fn_te))
    metrics_map["tp"] = (float(tp_tr), float(tp_te))

    print("\nClassification metrics (train vs. test):", flush = True)
    print(f"{'':<28} {'train':<10} {'test':<10}", flush = True)
    for k, v in metrics_map.items():
        print(f"{k:<28} {v[0]:.6f}   {v[1]:.6f}", flush = True)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    ax[0].plot(train_fpr, train_tpr, color='teal', lw=2, label=f'Train ROC (AUC = {metrics_map["auroc"][0]:.3f})')
    ax[0].plot(test_fpr, test_tpr, color='orange', lw=2, label=f'Test ROC (AUC = {metrics_map["auroc"][1]:.3f})')
    ax[0].plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate')
    ax[0].set_title('Receiver Operating Characteristic (ROC) Curve')
    ax[0].legend(loc='lower right')

    ax[1].plot(train_recall, train_precision, color='teal', lw=2, label=f'Train PR (AUC = {metrics_map["auprc"][0]:.3f})')
    ax[1].plot(test_recall, test_precision, color='orange', lw=2, label=f'Test PR (AUC = {metrics_map["auprc"][1]:.3f})')
    ax[1].set_xlabel('Recall')
    ax[1].set_ylabel('Precision')
    ax[1].set_title('Precision-Recall (PR) Curve')
    ax[1].legend(loc='lower left')

    cm_test = confusion_matrix(test_labels, test_preds)
    ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=["No SHD", "SHD"]).plot(ax=ax[2], cmap="Blues", values_format="d")
    ax[2].set_title("Confusion Matrix (Holdout Test Split)")

    plt.tight_layout()
    plt.savefig("earlyfusion_bcnn_standardized.png", dpi=150)

# ──────────────────────────────────────────────────────────────────────
# 7. EXECUTION PIPELINE
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Beginning execution pipeline...", flush = True)

    # ── STEP A: DATA INGESTION ────
    ECG_train_raw = np.load('/work/sm222/data/physionet.org/files/echonext/1.1.1/EchoNext_train_waveforms.npy')
    ECG_val_raw   = np.load('/work/sm222/data/physionet.org/files/echonext/1.1.1/EchoNext_val_waveforms.npy')
    ECG_test_raw  = np.load('/work/sm222/data/physionet.org/files/echonext/1.1.1/EchoNext_test_waveforms.npy')
    Echo_data     = pd.read_csv('/work/sm222/data/physionet.org/files/echonext/1.1.1/echonext_metadata_100k.csv')

        # ── STEP B: WAVEFORM PRE-FILTERING ──────────────────────────────
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

    train_loader = DataLoader(FusedDataset(X_ts_train, X_demo_train, y_train), batch_size=32, shuffle=True)
    val_loader   = DataLoader(FusedDataset(X_ts_val,   X_demo_val,   y_val),   batch_size=128, shuffle=False)
    test_loader  = DataLoader(FusedDataset(X_ts_test,  X_demo_test,  y_test),  batch_size=128, shuffle=False)

    # ── STEP E: INITIALIZE & TRAIN THE MODEL ──────────────────────────
    model = BayesianHeartDiseaseNet(
        sequence_length=X_ts_train.shape[2], in_channels=X_ts_train.shape[1],
        demo_dim=X_demo_train.shape[1], num_classes=2, demo_embed_dim=32
    )

    print("Training model...", flush = True)
    train_bayesian_classifier(model, train_loader, val_loader, epochs=60, learning_rate=1e-4)

    torch.manual_seed(RANDOM_SEED)

    # ── STEP F: PRINT SAMPLE INDIVIDUAL INFERENCE RECORDS ─────────────
    print("\n===== Individual Test Sample Inference Profiles ====", flush = True)
    model.eval()
    sample_counter = 0

    with torch.no_grad():
        for X_ts_b, X_demo_b, y_b in test_loader:
            X_ts_b, X_demo_b = X_ts_b.to(device), X_demo_b.to(device)
            pass_logits = torch.stack([model(X_ts_b, X_demo_b) for _ in range(50)], dim=0)
            pass_probs = torch.softmax(pass_logits, dim=-1).cpu().numpy()

            mean_probs = pass_probs.mean(axis=0)
            std_probs = pass_probs.std(axis=0)

            for idx in range(X_ts_b.size(0)):
                true_val = y_b[idx].item()
                pred_val = np.argmax(mean_probs[idx])
                confidence = mean_probs[idx][pred_val]
                uncertainty = std_probs[idx][pred_val]
                global_df_idx = test_ids[sample_counter]

                print(f"Sample {sample_counter:03d} (DF Index: {global_df_idx:03d}) | "
                      f"True: {true_val} | Pred: {pred_val} | "
                      f"Confidence: {confidence:.4f} | Uncertainty (Std): {uncertainty:.4f}", flush = True)
                sample_counter += 1

    # ── STEP G: SYSTEM METRICS ANALYSIS VISUALIZATIONS ────────────────
    generate_and_display_comparative_metrics(model, train_loader, test_loader, sample_nbr=50)
