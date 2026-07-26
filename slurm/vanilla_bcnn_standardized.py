import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, roc_curve,
    brier_score_loss, confusion_matrix, ConfusionMatrixDisplay
)

# Install blitz environment if not already present
try:
    from blitz.modules import BayesianLinear, BayesianConv1d
    from blitz.utils import variational_estimator
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "blitz-bayesian-pytorch"])
    from blitz.modules import BayesianLinear, BayesianConv1d
    from blitz.utils import variational_estimator

# ──────────────────────────────────────────────────────────────────────
# 0. CONFIGURATION & HYPERPARAMETERS
# ──────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush = True)

# Bayesian prior/posterior settings
PRIOR_SIGMA_1 = 1.0
PRIOR_SIGMA_2 = 0.0025
PRIOR_PI = 0.5
POSTERIOR_RHO_INIT = -5.0

# ECG filtering settings
ECG_SAMPLING_RATE_HZ = 250   # Matches EchoNext waveform acquisition rate
FILTER_LOW_HZ = 0.5
FILTER_HIGH_HZ = 40.0
FILTER_ORDER = 4

# ──────────────────────────────────────────────────────────────────────
# 1. SIGNAL PROCESSING & SIGNAL FILTERING
# ──────────────────────────────────────────────────────────────────────
def bandpass_filter_ecg(signal_array, fs=ECG_SAMPLING_RATE_HZ,
                         low=FILTER_LOW_HZ, high=FILTER_HIGH_HZ, order=FILTER_ORDER):
    nyquist = 0.5 * fs
    b, a = butter(order, [low / nyquist, high / nyquist], btype="band")
    # signal_array shape: (time, leads) — filter along time axis (axis=0), leads handled independently
    filtered = filtfilt(b, a, signal_array, axis=0)
    return filtered.astype(np.float32)

# ──────────────────────────────────────────────────────────────────────
# 2. PYTORCH DATA STRUCTURING (ECG waveform only)
# ──────────────────────────────────────────────────────────────────────
class ECGDataset(Dataset):
    def __init__(self, X_ts, y):
        assert len(X_ts) == len(y), "Length mismatch across tensor targets."
        self.X_ts = torch.tensor(X_ts, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_ts[idx], self.y[idx]

# ──────────────────────────────────────────────────────────────────────
# 3. BAYESIAN NETWORK ARCHITECTURE
# ──────────────────────────────────────────────────────────────────────
@variational_estimator
class BayesianVanillaHeartDiseaseNet(nn.Module):
    def __init__(self, sequence_length, in_channels, num_classes=2):
        super().__init__()

        def bconv(in_c, out_c, k, p):
            return BayesianConv1d(
                in_c, out_c, kernel_size=k, padding=p,
                prior_sigma_1=PRIOR_SIGMA_1, prior_sigma_2=PRIOR_SIGMA_2, prior_pi=PRIOR_PI,
                posterior_rho_init=POSTERIOR_RHO_INIT
            )

        def blinear(in_f, out_f):
            return BayesianLinear(
                in_f, out_f, prior_sigma_1=PRIOR_SIGMA_1, prior_sigma_2=PRIOR_SIGMA_2,
                prior_pi=PRIOR_PI, posterior_rho_init=POSTERIOR_RHO_INIT
            )

        self.conv1 = bconv(in_channels, 16, 15, 7)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.AvgPool1d(kernel_size=10)

        self.conv2 = bconv(16, 32, 9, 4)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.AvgPool1d(kernel_size=10)

        self.conv3 = bconv(32, 32, 5, 2)
        self.bn3 = nn.BatchNorm1d(32)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.relu = nn.ReLU()

        self.fc_out1 = blinear(32, 32)
        self.bn_out = nn.BatchNorm1d(32)
        self.fc_out2 = blinear(32, num_classes)

    def forward(self, x_ts):
        x = self.relu(self.bn1(self.conv1(x_ts)))
        x = self.pool1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)

        out = self.relu(self.bn_out(self.fc_out1(x)))
        out = self.fc_out2(out)
        return out

# ──────────────────────────────────────────────────────────────────────
# 4. TRAINING ROUTINE
# ──────────────────────────────────────────────────────────────────────
def train_bayesian_classifier(model, train_loader, val_loader, epochs=60, learning_rate=1e-4, sample_nbr=3):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    n_train = len(train_loader.dataset)

    for epoch in range(epochs):
        model.train()
        for X_ts_b, y_b in train_loader:
            X_ts_b, y_b = X_ts_b.to(device), y_b.to(device)
            X_ts_b += 0.05 * torch.randn_like(X_ts_b)  # light input noise

            optimizer.zero_grad()
            nll = 0.0
            for _ in range(sample_nbr):
                logits = model(X_ts_b)
                nll += criterion(logits, y_b)

            loss = (nll / sample_nbr) + (model.nn_kl_divergence() / n_train)
            loss.backward()
            optimizer.step()

        # Validation loop (already utilizing corrected logit accumulation)
        model.eval()
        val_loss, val_total = 0.0, 0
        with torch.no_grad():
            for X_ts_v, y_v in val_loader:
                X_ts_v, y_v = X_ts_v.to(device), y_v.to(device)

                # 1. Accumulate raw logits across 10 MC forward samples
                logits_accum = sum(model(X_ts_v) for _ in range(10)) / 10

                # 2. Pass mean logits straight to CrossEntropyLoss
                val_loss += criterion(logits_accum, y_v).item() * X_ts_v.shape[0]
                val_total += X_ts_v.shape[0]

        epoch_val_loss = val_loss / val_total
        print(f"Epoch {epoch+1:02d} | Dedicated Validation Loss: {epoch_val_loss:.6f}", flush = True)
        scheduler.step(epoch_val_loss)

# ──────────────────────────────────────────────────────────────────────
# 5. PERFORMANCE METRICS GENERATION & CALIBRATION ESTIMATION
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
    criterion = nn.CrossEntropyLoss(reduction='sum')

    with torch.no_grad():
        for X_ts_b, y_b in loader:
            X_ts_b = X_ts_b.to(device)
            # Gather raw logit outputs across multiple forward iterations
            pass_logits = torch.stack([model(X_ts_b) for _ in range(sample_nbr)], dim=0)

            # FIX: Average the raw logits first across the MC dimension
            mean_logits = pass_logits.mean(dim=0)

            # Pass unnormalized mean logits directly to CrossEntropyLoss
            all_nll += criterion(mean_logits, y_b.to(device)).item()

            # Calculate final probabilities from the averaged logits for structural evaluations
            mean_probs = torch.softmax(mean_logits, dim=-1)
            all_probs.extend(mean_probs[:, 1].cpu().numpy())
            all_labels.extend(y_b.numpy())

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

    print("\n==================================================", flush = True)
    print("Classification metrics (train vs. test):", flush = True)
    print(f"{'':<28} {'train':<10} {'test':<10}", flush = True)
    print("--------------------------------------------------", flush = True)
    for k, v in metrics_map.items():
        print(f"{k:<28} {v[0]:.6f}   {v[1]:.6f}", flush = True)
    print("==================================================", flush = True)

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
    plt.savefig("vanilla_bcnn_standardized_metrics.png", dpi = 150)

# ──────────────────────────────────────────────────────────────────────
# 6. EXECUTION PIPELINE
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Beginning execution pipeline...", flush = True)

    # ── STEP A: DATA INGESTION (Using EchoNext 100K explicitly split arrays) ────
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

    # ── STEP C: TARGET EXTRACTION FROM EXPLICIT SPLITS ────────────────────────
    train_ids = Echo_data.index[Echo_data['split'] == 'train'].tolist()
    val_ids   = Echo_data.index[Echo_data['split'] == 'val'].tolist()
    test_ids  = Echo_data.index[Echo_data['split'] == 'test'].tolist()

    y_train = np.array(Echo_data['shd_moderate_or_greater_flag'][train_ids])
    y_val   = np.array(Echo_data['shd_moderate_or_greater_flag'][val_ids])
    y_test  = np.array(Echo_data['shd_moderate_or_greater_flag'][test_ids])

    # Create DataLoaders (Plain single-branch targets)
    train_loader = DataLoader(ECGDataset(X_ts_train, y_train), batch_size=32, shuffle=True)
    val_loader   = DataLoader(ECGDataset(X_ts_val,   y_val),   batch_size=128, shuffle=False)
    test_loader  = DataLoader(ECGDataset(X_ts_test,  y_test),  batch_size=128, shuffle=False)

    # ── STEP D: INITIALIZE & TRAIN THE MODEL ──────────────────────────
    model = BayesianVanillaHeartDiseaseNet(
        sequence_length=X_ts_train.shape[2], in_channels=X_ts_train.shape[1], num_classes=2
    )

    print("Training model...", flush = True)
    train_bayesian_classifier(model, train_loader, val_loader, epochs=60, learning_rate=1e-4)

    torch.manual_seed(RANDOM_SEED)

    # ── STEP E: PRINT SAMPLE INDIVIDUAL INFERENCE RECORDS ─────────────
    print("\n===== Individual Test Sample Inference Profiles ====", flush = True)
    model.eval()
    sample_counter = 0

    with torch.no_grad():
        for X_ts_b, y_b in test_loader:
            X_ts_b = X_ts_b.to(device)
            pass_logits = torch.stack([model(X_ts_b) for _ in range(50)], dim=0)
            pass_probs = torch.softmax(pass_logits, dim=-1).cpu().numpy()

            mean_probs = pass_probs.mean(axis=0)
            std_probs = pass_probs.std(axis=0)

            # Cap display out for large inference test splits
            for idx in range(X_ts_b.size(0)):
                if sample_counter >= 100:
                    break
                true_val = y_b[idx].item()
                pred_val = np.argmax(mean_probs[idx])
                confidence = mean_probs[idx][pred_val]
                uncertainty = std_probs[idx][pred_val]
                global_df_idx = test_ids[sample_counter]

                print(f"Sample {sample_counter:03d} (DF Index: {global_df_idx:03d}) | "
                      f"True: {true_val} | Pred: {pred_val} | "
                      f"Confidence: {confidence:.4f} | Uncertainty (Std): {uncertainty:.4f}", flush = True)
                sample_counter += 1

    # ── STEP F: SYSTEM METRICS ANALYSIS VISUALIZATIONS ────────────────
    generate_and_display_comparative_metrics(model, train_loader, test_loader, sample_nbr=50)
