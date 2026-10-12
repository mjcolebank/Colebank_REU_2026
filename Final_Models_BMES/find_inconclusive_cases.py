"""Find 1K-set patients whose predictions are inconclusive under BOTH the
early/intermediate-fusion (FiLM) BCNN and the late-fusion BCNN.

Per patient we draw S global posterior predictive samples of P(class 1) from
each trained model, then flag a patient as "inconclusive" for a model when the
draws straddle the decision threshold, i.e. the full predictive posterior has
mass on both sides (class overlap). Patients flagged by both models are saved
as candidate test cases for posterior-predictive plotting.

Usage:
  python find_inconclusive_patients.py \
      --early-script  early_fusion_bcnn.py \
      --late-script   late_fusion_bcnn.py \
      --early-ckpt    earlyfusion_.../earlyfusion_pyro_checkpoint.pt \
      --late-ckpt     latefusion_.../latefusion_pyro_checkpoint.pt \
      --waveforms ECG_waveforms_1K.npy --metrics ECG_metrics_1K.npy \
      --out-dir inconclusive_1K

Notes
- Each model is loaded in its own pass because Pyro's param store is global
  and both models use identical parameter names.
- The 1K metrics file has 7 columns, so the checkpoints must have been
  trained with demo_dim == 7 (the script checks and tells you if not).
- Waveforms are (N, 2500, 12); we apply the same 0.5-40 Hz bandpass as the
  training pipeline (disable with --no-filter if already filtered).
"""
import argparse
import importlib.util
import os
import sys
import types

import numpy as np
import pandas as pd
import torch
import pyro
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The early-fusion script imports sympy but never uses it; shim if missing.
try:
    import sympy  # noqa: F401
except ImportError:
    sys.modules["sympy"] = types.ModuleType("sympy")


def import_from_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class ArrayDataset:
    """Minimal stand-in for FusedDataset (what collect_global_posterior_predictive needs)."""
    def __init__(self, x_ts, x_demo):
        self.X_ts = torch.as_tensor(x_ts, dtype=torch.float32)
        self.X_demo = torch.as_tensor(x_demo, dtype=torch.float32)
        self.y = torch.zeros(len(x_ts), dtype=torch.long)  # no labels in the 1K files

    def __len__(self):
        return len(self.y)


def posterior_draws(kind, script_path, ckpt_path, ds, mc_samples, seed):
    """Load one model + guide from checkpoint and return (S, N) draws of P(class 1)."""
    mod = import_from_path(script_path, f"{kind}_mod")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if ckpt["demo_dim"] != ds.X_demo.shape[1]:
        raise ValueError(
            f"[{kind}] checkpoint demo_dim={ckpt['demo_dim']} but metrics file has "
            f"{ds.X_demo.shape[1]} columns. The model must be trained on the same "
            f"demographic encoding as ECG_metrics_1K.npy."
        )
    if ckpt.get("sequence_length") not in (None, ds.X_ts.shape[2]):
        print(f"[{kind}] WARNING: checkpoint sequence_length={ckpt['sequence_length']} "
              f"vs data {ds.X_ts.shape[2]}")

    # Restore the priors used at training time (they live in module globals).
    for k, attr in [("prior_sigma_1", "PRIOR_SIGMA_1"), ("prior_sigma_2", "PRIOR_SIGMA_2"),
                    ("prior_pi", "PRIOR_PI")]:
        if k in ckpt:
            setattr(mod, attr, ckpt[k])

    pyro.clear_param_store()
    in_ch, demo_dim = ds.X_ts.shape[1], ds.X_demo.shape[1]
    if kind == "early":
        net = mod.BayesianHeartDiseaseNet(
            sequence_length=ds.X_ts.shape[2], in_channels=in_ch, demo_dim=demo_dim,
            num_classes=2, demo_embed_dim=ckpt.get("demo_embed_dim", 16))
        model = mod.PyroHeartDiseaseClassifier(net)
    else:
        net = mod.BayesianLateFusionHeartDiseaseNet(in_ch, demo_dim, num_classes=2)
        model = mod.PyroLateFusionClassifier(net)
    guide = mod.build_mean_field_guide(model)

    # Run the guide once so AutoNormal creates its parameters, then restore them.
    model.eval(); guide.eval()
    with torch.no_grad():
        guide(ds.X_ts[:2], ds.X_demo[:2], None, None)
    pyro.get_param_store().set_state(ckpt["pyro_param_store_state"])
    model.load_state_dict(ckpt["model_state_dict"])  # includes BatchNorm running stats (early)
    try:
        guide.load_state_dict(ckpt["guide_state_dict"])
    except RuntimeError as e:
        print(f"[{kind}] guide.load_state_dict strict load failed ({e}); "
              f"relying on restored Pyro param store.")
        guide.load_state_dict(ckpt["guide_state_dict"], strict=False)

    torch.manual_seed(seed); pyro.set_rng_seed(seed)
    probs, _ = mod.collect_global_posterior_predictive(
        model, guide, ds, mc_samples=mc_samples, evaluation_device=torch.device("cpu"))
    return probs, mod


def summarize(probs, threshold, minority_frac_min):
    """Per-patient summary of an (S, N) matrix of P(class 1) draws."""
    mean = probs.mean(0)
    lo, hi = np.quantile(probs, [0.025, 0.975], axis=0)
    frac_pos = (probs >= threshold).mean(0)
    minority = np.minimum(frac_pos, 1.0 - frac_pos)  # share of draws on the "other" side
    straddle = (lo < threshold) & (hi > threshold)
    return pd.DataFrame({
        "mean": mean, "sd": probs.std(0, ddof=1), "lower_95": lo, "upper_95": hi,
        "frac_draws_class1": frac_pos, "minority_frac": minority,
        "pred": (mean >= threshold).astype(int),
        "ci_straddles": straddle,
        "inconclusive": straddle & (minority >= minority_frac_min),
    })


def plot_cases(probs_e, probs_l, sum_e, sum_l, idx, title, path, threshold):
    if len(idx) == 0:
        return
    fig, axes = plt.subplots(len(idx), 2, figsize=(10, 2.1 * len(idx)), squeeze=False)
    bins = np.linspace(0, 1, 41)
    for r, i in enumerate(idx):
        for c, (name, pr, sm) in enumerate([("Early fusion", probs_e, sum_e),
                                            ("Late fusion", probs_l, sum_l)]):
            ax = axes[r, c]
            ax.hist(pr[:, i], bins=bins, color="C0" if c == 0 else "C1", alpha=0.8)
            ax.axvline(threshold, color="k", ls="--", lw=1)
            ax.set_xlim(0, 1)
            ax.set_title(f"#{i} {name}: mean={sm.at[i, 'mean']:.2f}, "
                         f"minority={sm.at[i, 'minority_frac']:.2f}", fontsize=9)
            if r == len(idx) - 1:
                ax.set_xlabel("P(class 1) across posterior draws")
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--early-script", required=True)
    p.add_argument("--late-script", required=True)
    p.add_argument("--early-ckpt", required=True)
    p.add_argument("--late-ckpt", required=True)
    p.add_argument("--waveforms", default="ECG_waveforms_1K.npy")
    p.add_argument("--metrics", default="ECG_metrics_1K.npy")
    p.add_argument("--out-dir", default="inconclusive_1K")
    p.add_argument("--mc-samples", type=int, default=200)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--minority-frac", type=float, default=0.20,
                   help="min share of posterior draws on the opposite side of the threshold")
    p.add_argument("--n-plot", type=int, default=8)
    p.add_argument("--no-filter", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    wave = np.load(a.waveforms)       # (N, 2500, 12)
    metrics = np.load(a.metrics)      # (N, 7), already encoded/scaled
    assert wave.ndim == 3 and wave.shape[2] == 12 and len(wave) == len(metrics), \
        f"unexpected shapes {wave.shape}, {metrics.shape}"

    # Same preprocessing as the training pipeline: bandpass over time, then (N, leads, time).
    if a.no_filter:
        ts = wave.astype(np.float32)
    else:
        ref = import_from_path(a.early_script, "ref_mod")
        ts = np.array([ref.bandpass_filter_ecg(x, fs=ref.ECG_SAMPLING_RATE_HZ) for x in wave])
    ts = np.swapaxes(ts, 1, 2)
    ds = ArrayDataset(ts, metrics)

    probs_e, _ = posterior_draws("early", a.early_script, a.early_ckpt, ds, a.mc_samples, a.seed)
    probs_l, _ = posterior_draws("late", a.late_script, a.late_ckpt, ds, a.mc_samples, a.seed)
    np.savez_compressed(os.path.join(a.out_dir, "posterior_draws_1K.npz"),
                        early=probs_e, late=probs_l)

    se = summarize(probs_e, a.threshold, a.minority_frac)
    sl = summarize(probs_l, a.threshold, a.minority_frac)
    both = se.add_prefix("early_").join(sl.add_prefix("late_"))
    both.insert(0, "patient_index", np.arange(len(both)))
    both["both_inconclusive"] = both["early_inconclusive"] & both["late_inconclusive"]
    both["models_disagree_on_label"] = both["early_pred"] != both["late_pred"]
    both["overlap_score"] = np.minimum(both["early_minority_frac"], both["late_minority_frac"])
    both.to_csv(os.path.join(a.out_dir, "all_patients_summary.csv"), index=False)

    hits = both[both["both_inconclusive"]].sort_values("overlap_score", ascending=False)
    hits.to_csv(os.path.join(a.out_dir, "inconclusive_both_models.csv"), index=False)

    # Contrast cases: confidently decided by both models, and models in label disagreement.
    confident = both[(both["early_minority_frac"] == 0) & (both["late_minority_frac"] == 0)]
    confident = confident.assign(margin=(confident["early_mean"] - a.threshold).abs()
                                 + (confident["late_mean"] - a.threshold).abs())
    confident = confident.sort_values("margin", ascending=False)
    disagree = both[both["models_disagree_on_label"]].sort_values("overlap_score", ascending=False)

    print(f"\nN patients: {len(both)}  (threshold={a.threshold}, "
          f"minority_frac>={a.minority_frac}, S={a.mc_samples})")
    print(f"Early-fusion inconclusive : {int(se['inconclusive'].sum())}")
    print(f"Late-fusion inconclusive  : {int(sl['inconclusive'].sum())}")
    print(f"INCONCLUSIVE IN BOTH      : {len(hits)}")
    print(f"Models disagree on label  : {len(disagree)}")
    if len(hits):
        cols = ["patient_index", "early_mean", "early_minority_frac",
                "late_mean", "late_minority_frac", "overlap_score"]
        print("\nTop both-inconclusive patients:")
        print(hits[cols].head(15).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    else:
        print("No patients met both criteria; try a lower --minority-frac (e.g. 0.1).")

    plot_cases(probs_e, probs_l, se, sl, hits["patient_index"].head(a.n_plot).tolist(),
               "Inconclusive under BOTH models", os.path.join(a.out_dir, "cases_both_inconclusive.png"),
               a.threshold)
    plot_cases(probs_e, probs_l, se, sl, disagree["patient_index"].head(4).tolist(),
               "Models disagree on label", os.path.join(a.out_dir, "cases_model_disagreement.png"),
               a.threshold)
    plot_cases(probs_e, probs_l, se, sl, confident["patient_index"].head(4).tolist(),
               "Confident under both models (controls)", os.path.join(a.out_dir, "cases_confident_controls.png"),
               a.threshold)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(both["early_minority_frac"], both["late_minority_frac"], s=8, alpha=0.5)
    ax.scatter(hits["early_minority_frac"], hits["late_minority_frac"], s=14, color="red",
               label="inconclusive in both")
    ax.axvline(a.minority_frac, color="k", ls=":"); ax.axhline(a.minority_frac, color="k", ls=":")
    ax.set_xlabel("Early fusion: share of draws on minority side")
    ax.set_ylabel("Late fusion: share of draws on minority side")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "overlap_scatter.png"), dpi=150)
    plt.close(fig)
    print(f"\nOutputs written to {a.out_dir}/")


if __name__ == "__main__":
    main()
