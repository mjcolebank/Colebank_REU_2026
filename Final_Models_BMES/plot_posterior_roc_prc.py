"""Plot ROC and precision-recall curves from saved Bayesian posterior predictions.

This script is designed for the ``*_posterior_predictive.npz`` files written by
``latefusion_bcnn_pyro.py`` (and ``earlyfusion_bcnn_pyro.py``).  It treats each
row of ``posterior_class1_probs`` as one posterior draw of the complete model,
so each draw produces one ROC curve, PR curve, AUROC, and AUPRC.

Example
-------
python plot_posterior_roc_prc.py \
    --input-npz latefusion_pyro_results/test_posterior_predictive.npz \
    --output-dir latefusion_pyro_results/figures \
    --model-name "Late-fusion Pyro BCNN"

Outputs
-------
* ``posterior_roc_prc.png``: a two-panel figure with 95% posterior curve bands.
* ``posterior_curve_metrics.csv``: posterior and bootstrap summary metrics.
* ``posterior_curve_data.npz``: grids and uncertainty bands used in the figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True,
                        help="NPZ file containing labels and posterior_class1_probs.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for plot and summary files (default: NPZ parent/figures).")
    parser.add_argument("--model-name", default="Bayesian classifier",
                        help="Name used in the figure title.")
    parser.add_argument("--n-grid", type=int, default=201,
                        help="Number of equally spaced ROC/recall grid points.")
    parser.add_argument("--n-bootstrap", type=int, default=2000,
                        help="Number of patient-level bootstrap resamples for posterior-mean metrics.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrapping.")
    return parser.parse_args()


def validate_inputs(labels: np.ndarray, posterior_probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels).astype(int).ravel()
    posterior_probs = np.asarray(posterior_probs, dtype=float)
    if posterior_probs.ndim != 2:
        raise ValueError("posterior_class1_probs must have shape (n_posterior_samples, n_patients).")
    if posterior_probs.shape[1] != labels.size:
        raise ValueError("The posterior probability array and labels have incompatible patient counts.")
    if not np.isin(labels, [0, 1]).all() or np.unique(labels).size != 2:
        raise ValueError("labels must contain both binary classes coded as 0 and 1.")
    if not np.isfinite(posterior_probs).all() or ((posterior_probs < 0) | (posterior_probs > 1)).any():
        raise ValueError("posterior probabilities must be finite values in [0, 1].")
    return labels, posterior_probs


def interpolate_roc(labels: np.ndarray, scores: np.ndarray, fpr_grid: np.ndarray) -> np.ndarray:
    fpr, tpr, _ = roc_curve(labels, scores)
    return np.interp(fpr_grid, fpr, tpr)


def interpolate_prc(labels: np.ndarray, scores: np.ndarray, recall_grid: np.ndarray) -> np.ndarray:
    precision, recall, _ = precision_recall_curve(labels, scores)
    # sklearn returns recall from 1 to 0.  Sorting and keeping the maximum
    # precision at duplicate recall values yields a stable interpolation.
    recall = recall[::-1]
    precision = precision[::-1]
    unique_recall, inverse = np.unique(recall, return_inverse=True)
    max_precision = np.full(unique_recall.size, -np.inf)
    np.maximum.at(max_precision, inverse, precision)
    return np.interp(recall_grid, unique_recall, max_precision)


def bootstrap_metric_intervals(labels: np.ndarray, mean_probs: np.ndarray, n_bootstrap: int, seed: int):
    rng = np.random.default_rng(seed)
    auc_values, ap_values = [], []
    n = labels.size
    while len(auc_values) < n_bootstrap:
        idx = rng.integers(0, n, n)
        if np.unique(labels[idx]).size < 2:
            continue
        auc_values.append(roc_auc_score(labels[idx], mean_probs[idx]))
        ap_values.append(average_precision_score(labels[idx], mean_probs[idx]))
    return np.asarray(auc_values), np.asarray(ap_values)


def interval(values: np.ndarray) -> tuple[float, float]:
    return tuple(np.percentile(values, [2.5, 97.5]))


def main() -> None:
    args = parse_args()
    if args.n_grid < 2 or args.n_bootstrap < 1:
        raise ValueError("--n-grid must be >= 2 and --n-bootstrap must be >= 1.")
    output_dir = args.output_dir or args.input_npz.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.input_npz) as saved:
        required = {"labels", "posterior_class1_probs"}
        missing = required.difference(saved.files)
        if missing:
            raise KeyError(f"{args.input_npz} is missing required keys: {sorted(missing)}")
        labels, posterior_probs = validate_inputs(saved["labels"], saved["posterior_class1_probs"])

    mean_probs = posterior_probs.mean(axis=0)
    fpr_grid = np.linspace(0.0, 1.0, args.n_grid)
    recall_grid = np.linspace(0.0, 1.0, args.n_grid)
    posterior_roc = np.asarray([interpolate_roc(labels, sample, fpr_grid) for sample in posterior_probs])
    posterior_prc = np.asarray([interpolate_prc(labels, sample, recall_grid) for sample in posterior_probs])
    posterior_auc = np.asarray([roc_auc_score(labels, sample) for sample in posterior_probs])
    posterior_ap = np.asarray([average_precision_score(labels, sample) for sample in posterior_probs])

    mean_roc = interpolate_roc(labels, mean_probs, fpr_grid)
    mean_prc = interpolate_prc(labels, mean_probs, recall_grid)
    mean_auc = roc_auc_score(labels, mean_probs)
    mean_ap = average_precision_score(labels, mean_probs)
    boot_auc, boot_ap = bootstrap_metric_intervals(labels, mean_probs, args.n_bootstrap, args.seed)

    roc_low, roc_high = np.percentile(posterior_roc, [2.5, 97.5], axis=0)
    prc_low, prc_high = np.percentile(posterior_prc, [2.5, 97.5], axis=0)
    posterior_auc_ci, posterior_ap_ci = interval(posterior_auc), interval(posterior_ap)
    bootstrap_auc_ci, bootstrap_ap_ci = interval(boot_auc), interval(boot_ap)
    prevalence = labels.mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    ax = axes[0]
    ax.plot(fpr_grid, mean_roc, color="#0072B2", lw=2.5,
            label=f"Posterior mean AUROC = {mean_auc:.3f}")
    ax.fill_between(fpr_grid, roc_low, roc_high, color="#0072B2", alpha=0.22,
                    label="95% posterior credible band")
    ax.plot([0, 1], [0, 1], "--", color="0.45", lw=1.2, label="No-discrimination")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False-positive rate", ylabel="True-positive rate",
           title="Receiver operating characteristic")
    ax.legend(loc="lower right", frameon=False)

    ax = axes[1]
    ax.plot(recall_grid, mean_prc, color="#D55E00", lw=2.5,
            label=f"Posterior mean AUPRC = {mean_ap:.3f}")
    ax.fill_between(recall_grid, prc_low, prc_high, color="#D55E00", alpha=0.22,
                    label="95% posterior credible band")
    ax.axhline(prevalence, ls="--", color="0.45", lw=1.2,
               label=f"Prevalence = {prevalence:.3f}")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision",
           title="Precision-recall curve")
    ax.legend(loc="lower left", frameon=False)

    fig.suptitle(args.model_name, fontsize=14, fontweight="bold")
    figure_path = output_dir / "posterior_roc_prc.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    metrics = pd.DataFrame([
        {"metric": "AUROC", "estimate_type": "posterior_mean", "estimate": mean_auc,
         "lower_95": posterior_auc_ci[0], "upper_95": posterior_auc_ci[1],
         "interval_source": "posterior draws"},
        {"metric": "AUROC", "estimate_type": "posterior_mean", "estimate": mean_auc,
         "lower_95": bootstrap_auc_ci[0], "upper_95": bootstrap_auc_ci[1],
         "interval_source": f"{args.n_bootstrap} patient bootstrap resamples"},
        {"metric": "AUPRC", "estimate_type": "posterior_mean", "estimate": mean_ap,
         "lower_95": posterior_ap_ci[0], "upper_95": posterior_ap_ci[1],
         "interval_source": "posterior draws"},
        {"metric": "AUPRC", "estimate_type": "posterior_mean", "estimate": mean_ap,
         "lower_95": bootstrap_ap_ci[0], "upper_95": bootstrap_ap_ci[1],
         "interval_source": f"{args.n_bootstrap} patient bootstrap resamples"},
    ])
    metrics.to_csv(output_dir / "posterior_curve_metrics.csv", index=False)
    np.savez_compressed(
        output_dir / "posterior_curve_data.npz",
        fpr_grid=fpr_grid, mean_tpr=mean_roc, lower_tpr_95=roc_low, upper_tpr_95=roc_high,
        recall_grid=recall_grid, mean_precision=mean_prc, lower_precision_95=prc_low,
        upper_precision_95=prc_high, posterior_auroc=posterior_auc, posterior_auprc=posterior_ap,
        bootstrap_auroc=boot_auc, bootstrap_auprc=boot_ap,
    )
    print(f"Saved ROC/PRC figure: {figure_path}")
    print(f"AUROC: {mean_auc:.3f} | posterior 95% CI [{posterior_auc_ci[0]:.3f}, {posterior_auc_ci[1]:.3f}] "
          f"| bootstrap 95% CI [{bootstrap_auc_ci[0]:.3f}, {bootstrap_auc_ci[1]:.3f}]")
    print(f"AUPRC: {mean_ap:.3f} | posterior 95% CI [{posterior_ap_ci[0]:.3f}, {posterior_ap_ci[1]:.3f}] "
          f"| bootstrap 95% CI [{bootstrap_ap_ci[0]:.3f}, {bootstrap_ap_ci[1]:.3f}]")


if __name__ == "__main__":
    main()
