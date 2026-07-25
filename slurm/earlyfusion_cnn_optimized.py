import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# ──────────────────────────────────────────────────────────────────────
# 0. CONFIGURATION & HYPERPARAMETERS
# ──────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush = True)

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
    # signal_array shape: (leads, time) — filter along time axis, leads handled independently
    filtered = filtfilt(b, a, signal_array, axis=-1)
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


# ─────────────────────────────────────────────
# 4. Model
# ─────────────────────────────────────────────
class FiLMLayer(nn.Module):
    def __init__(self, demo_embed_dim: int, num_channels: int):
        super().__init__()
        self.film_proj = nn.Linear(demo_embed_dim, 2 * num_channels)
        self.num_channels = num_channels

        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)
        with torch.no_grad():
            self.film_proj.bias[:num_channels] = 1.0

    def forward(self, h: torch.Tensor, demo_embed: torch.Tensor) -> torch.Tensor:
        params = self.film_proj(demo_embed)
        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)
        beta  = beta.unsqueeze(-1)
        return gamma * h + beta


class FiLMConvBlock(nn.Module):
    def __init__(
        self,
        in_channels:    int,
        out_channels:   int,
        kernel_size:    int,
        padding:        int,
        pool_size:      int,
        dropout:        float,
        demo_embed_dim: int,
        use_pool:       bool = True # Added flag to bypass pooling safely if spatial sizes drop
    ):
        super().__init__()
        self.conv    = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.film    = FiLMLayer(demo_embed_dim, out_channels)
        self.act     = nn.ReLU()
        self.drop    = nn.Dropout(dropout)
        self.use_pool = use_pool
        if use_pool:
            self.pool    = nn.AvgPool1d(kernel_size=pool_size)

    def forward(self, x: torch.Tensor, demo_embed: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.film(x, demo_embed)
        x = self.act(x)
        x = self.drop(x)
        if self.use_pool:
            x = self.pool(x)
        return x

class HeartDiseaseNet(nn.Module):
    def __init__(
        self,
        sequence_length: int = 2500,
        in_channels:     int = 12,
        demo_dim:        int = 20,
        num_classes:     int = 2,
        demo_embed_dim:  int = 32,
    ):
        super().__init__()

        self.demo_encoder = nn.Sequential(
            nn.Linear(demo_dim, demo_embed_dim),
            nn.ReLU(),
        )

        # FIX: The 4th layer uses use_pool=False to avoid crashing on short sequence lengths
        self.block1 = FiLMConvBlock(in_channels, 4,  kernel_size=15, padding=7,  pool_size=5, dropout=0.20, demo_embed_dim=demo_embed_dim)
        self.block2 = FiLMConvBlock(4,           8,  kernel_size=9,  padding=4,  pool_size=5, dropout=0.40, demo_embed_dim=demo_embed_dim)
        self.block3 = FiLMConvBlock(8,           12, kernel_size=5,  padding=2,  pool_size=5, dropout=0.40, demo_embed_dim=demo_embed_dim)
        self.block4 = FiLMConvBlock(12,          16, kernel_size=3,  padding=1,  pool_size=1, dropout=0.30, demo_embed_dim=demo_embed_dim, use_pool=False)

        with torch.no_grad():
            dummy_ts   = torch.zeros(1, in_channels, sequence_length)
            dummy_demo = torch.zeros(1, demo_embed_dim)
            x = self.block1(dummy_ts,   dummy_demo)
            x = self.block2(x,          dummy_demo)
            x = self.block3(x,          dummy_demo)
            x = self.block4(x,          dummy_demo)
            flat_sz = x.shape[1] * x.shape[2]
        print(f"Flattened size after FiLM-CNN blocks: {flat_sz}", flush = True)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_sz, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, num_classes),
        )

    def forward(self, x_ts: torch.Tensor, x_demo: torch.Tensor) -> torch.Tensor:
        demo_embed = self.demo_encoder(x_demo)

        x = self.block1(x_ts, demo_embed)
        x = self.block2(x,    demo_embed)
        x = self.block3(x,    demo_embed)
        x = self.block4(x,    demo_embed)

        return self.head(x)

# ──────────────────────────────────────────────────────────────────────
# 5. TRAINING ROUTINE
# ──────────────────────────────────────────────────────────────────────
def train_classifier(model, train_loader, val_loader, epochs=60, learning_rate=1e-4):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    for epoch in range(epochs):
        model.train()
        for X_ts_b, X_demo_b, y_b in train_loader:
            X_ts_b, X_demo_b, y_b = X_ts_b.to(device), X_demo_b.to(device), y_b.to(device)
            X_ts_b += 0.05 * torch.randn_like(X_ts_b)

            optimizer.zero_grad()
            logits = model(X_ts_b, X_demo_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss, val_total = 0.0, 0
        with torch.no_grad():
            for X_ts_v, X_demo_v, y_v in val_loader:
                X_ts_v, X_demo_v, y_v = X_ts_v.to(device), X_demo_v.to(device), y_v.to(device)
                logits = model(X_ts_v, X_demo_v)
                val_loss += criterion(logits, y_v).item() * X_ts_v.shape[0]
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

def get_predictions_and_labels(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    all_nll = 0.0
    criterion = nn.CrossEntropyLoss(reduction='sum')

    with torch.no_grad():
        for X_ts_b, X_demo_b, y_b in loader:
            X_ts_b, X_demo_b = X_ts_b.to(device), X_demo_b.to(device)
            logits = model(X_ts_b, X_demo_b)
            probs = torch.softmax(logits, dim=-1)

            all_nll += criterion(logits, y_b.to(device)).item()
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(y_b.numpy())

    return np.array(all_probs), np.array(all_labels), all_nll / len(loader.dataset)

def generate_and_display_comparative_metrics(model, train_loader, test_loader):
    train_probs, train_labels, train_nll = get_predictions_and_labels(model, train_loader)
    test_probs, test_labels, test_nll = get_predictions_and_labels(model, test_loader)

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
    plt.savefig("earlyfusion_cnn_optimized_metrics.png", dpi = 150)

# ──────────────────────────────────────────────────────────────────────
# 7. EXECUTION PIPELINE
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Beginning execution pipeline...", flush = True)

    # ── STEP A: DATA INGESTION ────
    ECG_train_raw = np.load('../EchoNextData/EchoNext_train_waveforms.npy')
    ECG_val_raw   = np.load('../EchoNextData/EchoNext_val_waveforms.npy')
    ECG_test_raw  = np.load('../EchoNextData/EchoNext_test_waveforms.npy')
    Echo_data     = pd.read_csv('../EchoNextData/echonext_metadata_100k.csv')

    # ── STEP B: WAVEFORM PRE-FILTERING (all 12 leads) ────────────────
    # ECG_*_raw shape: (N, 12, T) — already channel-first, no swapaxes needed
    X_ts_train = np.array([bandpass_filter_ecg(x, fs=ECG_SAMPLING_RATE_HZ) for x in ECG_train_raw])
    X_ts_val   = np.array([bandpass_filter_ecg(x, fs=ECG_SAMPLING_RATE_HZ) for x in ECG_val_raw])
    X_ts_test  = np.array([bandpass_filter_ecg(x, fs=ECG_SAMPLING_RATE_HZ) for x in ECG_test_raw])

    X_ts_train = np.swapaxes(X_ts_train, 1, 2)
    X_ts_val   = np.swapaxes(X_ts_val, 1, 2)
    X_ts_test  = np.swapaxes(X_ts_test, 1, 2)

    # ── STEP C: EXTRACTION FROM EXPLICIT SPLITS ───────────────────────
    train_ids = Echo_data.index[Echo_data['split'] == 'train'].tolist()
    val_ids   = Echo_data.index[Echo_data['split'] == 'val'].tolist()
    test_ids  = Echo_data.index[Echo_data['split'] == 'test'].tolist()

    y_train = np.array(Echo_data['shd_moderate_or_greater_flag'][train_ids])
    y_val   = np.array(Echo_data['shd_moderate_or_greater_flag'][val_ids])
    y_test  = np.array(Echo_data['shd_moderate_or_greater_flag'][test_ids])

    # ── STEP D: TRANSFORM DEMOGRAPHICS ──
    # FIX: Changed `.iloc` to `.loc` to reference target rows by label index safely
    demo_df_train = Echo_data[ALL_DEMO_COLS].loc[train_ids].copy()
    demo_df_val   = Echo_data[ALL_DEMO_COLS].loc[val_ids].copy()
    demo_df_test  = Echo_data[ALL_DEMO_COLS].loc[test_ids].copy()

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
    model = HeartDiseaseNet(
        sequence_length=X_ts_train.shape[2], in_channels=X_ts_train.shape[1],
        demo_dim=X_demo_train.shape[1], num_classes=2, demo_embed_dim=32
    )

    print("Training model...", flush = True)
    train_classifier(model, train_loader, val_loader, epochs=60, learning_rate=1e-4)

    torch.manual_seed(RANDOM_SEED)

    # ── STEP F: PRINT SAMPLE INDIVIDUAL INFERENCE RECORDS ─────────────
    print("\n===== Individual Test Sample Inference Profiles ====", flush = True)
    model.eval()
    sample_counter = 0

    with torch.no_grad():
        for X_ts_b, X_demo_b, y_b in test_loader:
            X_ts_b, X_demo_b = X_ts_b.to(device), X_demo_b.to(device)
            logits = model(X_ts_b, X_demo_b)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for idx in range(X_ts_b.size(0)):
                true_val = y_b[idx].item()
                pred_val = np.argmax(probs[idx])
                confidence = probs[idx][pred_val]
                global_df_idx = test_ids[sample_counter]

                print(f"Sample {sample_counter:03d} (DF Index: {global_df_idx:03d}) | "
                      f"True: {true_val} | Pred: {pred_val} | "
                      f"Confidence: {confidence:.4f}", flush = True)
                sample_counter += 1

    # ── STEP G: SYSTEM METRICS ANALYSIS VISUALIZATIONS ────────────────
    generate_and_display_comparative_metrics(model, train_loader, test_loader)
