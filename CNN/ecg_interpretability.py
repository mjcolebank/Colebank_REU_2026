"""
ecg_interpretability.py

Utilities for interpretable machine learning with 12-lead ECG neural networks.

The functions in this module are designed for PyTorch models that accept ECG tensors
with shape:

    x.shape == (batch_size, n_leads, n_timepoints)

For standard 12-lead ECGs, n_leads is usually 12.

Included methods
----------------
1. Vanilla saliency
2. Integrated gradients
3. Occlusion sensitivity over lead/time windows
4. Lead-wise occlusion importance
5. Bayesian/Pyro-compatible predictive wrapper
6. Monte Carlo attribution summaries
7. Region/window aggregation
8. ECG attribution plotting

Typical usage
-------------
from ecg_interpretability import integrated_gradients, plot_ecg_with_attribution

ig = integrated_gradients(model, ecg_batch, steps=100)
plot_ecg_with_attribution(ecg_batch, ig, sample_idx=0, sampling_rate=500)
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


Tensor = torch.Tensor


DEFAULT_12_LEAD_NAMES: List[str] = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]


def get_target_scores(
    logits: Tensor,
    target_class: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Select one scalar class score per sample.

    Parameters
    ----------
    logits:
        Tensor with shape ``(batch, n_classes)``.
    target_class:
        Optional integer tensor with shape ``(batch,)`` containing the class
        index to explain for each sample. If ``None``, the predicted class is
        used.

    Returns
    -------
    scores:
        Tensor with shape ``(batch,)`` containing the selected class score.
    target_class:
        Tensor with shape ``(batch,)`` containing the selected class index for
        each sample.
    """
    if logits.ndim != 2:
        raise ValueError(
            f"Expected logits with shape (batch, n_classes), got {tuple(logits.shape)}"
        )

    if target_class is None:
        target_class = logits.argmax(dim=1)

    if target_class.ndim != 1:
        raise ValueError(
            f"Expected target_class with shape (batch,), got {tuple(target_class.shape)}"
        )

    target_class = target_class.to(device=logits.device, dtype=torch.long)
    scores = logits.gather(1, target_class.view(-1, 1)).squeeze(1)

    return scores, target_class


def normalize_attribution(attr: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Normalize attribution values per sample to the interval [0, 1].

    Parameters
    ----------
    attr:
        Attribution tensor with shape ``(batch, leads, time)``.
    eps:
        Numerical stability constant.

    Returns
    -------
    norm_attr:
        Tensor with the same shape as ``attr``.
    """
    if attr.ndim != 3:
        raise ValueError(
            f"Expected attribution with shape (batch, leads, time), got {tuple(attr.shape)}"
        )

    batch = attr.shape[0]
    flat = attr.reshape(batch, -1)

    min_vals = flat.min(dim=1, keepdim=True).values
    max_vals = flat.max(dim=1, keepdim=True).values

    norm = (flat - min_vals) / (max_vals - min_vals + eps)
    return norm.reshape_as(attr)


def vanilla_saliency(
    model: torch.nn.Module,
    x: Tensor,
    target_class: Optional[Tensor] = None,
    absolute: bool = True,
    normalize: bool = True,
) -> Tensor:
    """
    Compute vanilla gradient saliency for ECG input.

    Parameters
    ----------
    model:
        Trained PyTorch model mapping ``x`` to logits.
    x:
        ECG tensor with shape ``(batch, leads, time)``.
    target_class:
        Optional integer tensor with shape ``(batch,)``. If ``None``, the
        predicted class is explained.
    absolute:
        If ``True``, return absolute gradients.
    normalize:
        If ``True``, normalize attribution per sample to [0, 1].

    Returns
    -------
    saliency:
        Tensor with shape ``(batch, leads, time)``.
    """
    _validate_ecg_tensor(x)
    model.eval()

    x_grad = x.detach().clone().requires_grad_(True)

    logits = model(x_grad)
    scores, target_class = get_target_scores(logits, target_class)

    model.zero_grad(set_to_none=True)
    scores.sum().backward()

    if x_grad.grad is None:
        raise RuntimeError("Gradient was not computed. Check that model output depends on x.")

    saliency = x_grad.grad.detach()

    if absolute:
        saliency = saliency.abs()

    if normalize:
        saliency = normalize_attribution(saliency)

    return saliency


def integrated_gradients(
    model: torch.nn.Module,
    x: Tensor,
    target_class: Optional[Tensor] = None,
    baseline: Optional[Tensor] = None,
    steps: int = 64,
    absolute: bool = True,
    normalize: bool = True,
) -> Tensor:
    """
    Compute integrated gradients for ECG input.

    Integrated gradients estimate the importance of input features by integrating
    gradients along a straight-line path from a baseline ECG to the observed ECG.

    Parameters
    ----------
    model:
        Trained PyTorch model mapping ``x`` to logits.
    x:
        ECG tensor with shape ``(batch, leads, time)``.
    target_class:
        Optional integer tensor with shape ``(batch,)``. If ``None``, the
        predicted class is explained.
    baseline:
        Baseline ECG tensor. Must be broadcast-compatible with ``x``. If
        ``None``, uses zeros.
    steps:
        Number of interpolation steps between baseline and input.
    absolute:
        If ``True``, return absolute attributions.
    normalize:
        If ``True``, normalize attribution per sample to [0, 1].

    Returns
    -------
    attr:
        Tensor with shape ``(batch, leads, time)``.
    """
    _validate_ecg_tensor(x)
    if steps < 1:
        raise ValueError("steps must be >= 1")

    model.eval()
    x_detached = x.detach()

    if baseline is None:
        baseline = torch.zeros_like(x_detached)
    else:
        baseline = baseline.detach().to(device=x.device, dtype=x.dtype)
        baseline = torch.broadcast_to(baseline, x_detached.shape)

    if target_class is None:
        with torch.no_grad():
            logits = model(x_detached)
            target_class = logits.argmax(dim=1)
    else:
        target_class = target_class.to(device=x.device, dtype=torch.long)

    total_gradients = torch.zeros_like(x_detached)

    for alpha in torch.linspace(0.0, 1.0, steps, device=x.device, dtype=x.dtype):
        x_interp = baseline + alpha * (x_detached - baseline)
        x_interp = x_interp.detach().clone().requires_grad_(True)

        logits = model(x_interp)
        scores, _ = get_target_scores(logits, target_class)

        model.zero_grad(set_to_none=True)
        scores.sum().backward()

        if x_interp.grad is None:
            raise RuntimeError("Gradient was not computed during integrated gradients.")

        total_gradients += x_interp.grad.detach()

    avg_gradients = total_gradients / float(steps)
    attr = (x_detached - baseline) * avg_gradients

    if absolute:
        attr = attr.abs()

    if normalize:
        attr = normalize_attribution(attr)

    return attr


@torch.no_grad()
def occlusion_sensitivity(
    model: torch.nn.Module,
    x: Tensor,
    target_class: Optional[Tensor] = None,
    window_size: int = 50,
    stride: int = 10,
    baseline_value: float = 0.0,
    use_probabilities: bool = True,
    normalize: bool = True,
) -> Tensor:
    """
    Compute occlusion sensitivity by masking lead-specific time windows.

    For each lead and sliding time window, this function replaces the segment
    with ``baseline_value`` and measures how much the target score decreases.

    Parameters
    ----------
    model:
        Trained PyTorch model mapping ``x`` to logits.
    x:
        ECG tensor with shape ``(batch, leads, time)``.
    target_class:
        Optional integer tensor with shape ``(batch,)``. If ``None``, the
        predicted class is explained.
    window_size:
        Number of timepoints to occlude.
    stride:
        Sliding-window stride.
    baseline_value:
        Value used to replace occluded segments.
    use_probabilities:
        If ``True``, use softmax probabilities. If ``False``, use logits.
    normalize:
        If ``True``, normalize attribution per sample to [0, 1].

    Returns
    -------
    importance:
        Tensor with shape ``(batch, leads, time)``. Larger values indicate
        stronger decrease in target score when occluded.
    """
    _validate_ecg_tensor(x)
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    model.eval()

    batch, n_leads, n_time = x.shape

    logits = model(x)
    outputs = F.softmax(logits, dim=1) if use_probabilities else logits

    if target_class is None:
        target_class = outputs.argmax(dim=1)
    else:
        target_class = target_class.to(device=x.device, dtype=torch.long)

    original_scores = outputs.gather(1, target_class.view(-1, 1)).squeeze(1)

    importance = torch.zeros_like(x)
    counts = torch.zeros_like(x)

    for lead in range(n_leads):
        for start in range(0, max(n_time - window_size + 1, 1), stride):
            end = min(start + window_size, n_time)

            x_occ = x.clone()
            x_occ[:, lead, start:end] = baseline_value

            occ_logits = model(x_occ)
            occ_outputs = F.softmax(occ_logits, dim=1) if use_probabilities else occ_logits
            occ_scores = occ_outputs.gather(1, target_class.view(-1, 1)).squeeze(1)

            score_drop = original_scores - occ_scores

            importance[:, lead, start:end] += score_drop.view(batch, 1)
            counts[:, lead, start:end] += 1

            if end == n_time:
                break

    importance = importance / counts.clamp_min(1.0)
    importance = importance.clamp_min(0.0)

    if normalize:
        importance = normalize_attribution(importance)

    return importance


@torch.no_grad()
def lead_occlusion_importance(
    model: torch.nn.Module,
    x: Tensor,
    target_class: Optional[Tensor] = None,
    baseline_value: float = 0.0,
    use_probabilities: bool = True,
    normalize: bool = False,
) -> Tensor:
    """
    Estimate importance of each ECG lead by occluding one full lead at a time.

    Parameters
    ----------
    model:
        Trained PyTorch model mapping ``x`` to logits.
    x:
        ECG tensor with shape ``(batch, leads, time)``.
    target_class:
        Optional integer tensor with shape ``(batch,)``. If ``None``, the
        predicted class is explained.
    baseline_value:
        Replacement value for occluded lead.
    use_probabilities:
        If ``True``, use softmax probabilities. If ``False``, use logits.
    normalize:
        If ``True``, normalize per sample across leads.

    Returns
    -------
    lead_importance:
        Tensor with shape ``(batch, leads)``.
    """
    _validate_ecg_tensor(x)
    model.eval()

    batch, n_leads, _ = x.shape

    logits = model(x)
    outputs = F.softmax(logits, dim=1) if use_probabilities else logits

    if target_class is None:
        target_class = outputs.argmax(dim=1)
    else:
        target_class = target_class.to(device=x.device, dtype=torch.long)

    original_scores = outputs.gather(1, target_class.view(-1, 1)).squeeze(1)

    lead_importance = torch.zeros(batch, n_leads, device=x.device, dtype=x.dtype)

    for lead in range(n_leads):
        x_occ = x.clone()
        x_occ[:, lead, :] = baseline_value

        occ_logits = model(x_occ)
        occ_outputs = F.softmax(occ_logits, dim=1) if use_probabilities else occ_logits
        occ_scores = occ_outputs.gather(1, target_class.view(-1, 1)).squeeze(1)

        lead_importance[:, lead] = original_scores - occ_scores

    lead_importance = lead_importance.clamp_min(0.0)

    if normalize:
        denom = lead_importance.sum(dim=1, keepdim=True).clamp_min(1e-8)
        lead_importance = lead_importance / denom

    return lead_importance


class BayesianPredictiveWrapper(torch.nn.Module):
    """
    Wrap a Pyro ``Predictive`` object so it behaves like a PyTorch model.

    The wrapped object should return a dictionary containing posterior samples
    of logits under ``logits_key``. The expected shape is usually:

        samples[logits_key].shape == (num_samples, batch, n_classes)

    The wrapper returns posterior mean logits with shape:

        (batch, n_classes)

    Notes
    -----
    This class does not import Pyro. It only assumes the object passed to
    ``predictive`` is callable and returns a dictionary.
    """

    def __init__(
        self,
        predictive: Callable[..., Dict[str, Tensor]],
        logits_key: str = "logits",
    ) -> None:
        super().__init__()
        self.predictive = predictive
        self.logits_key = logits_key

    def forward(self, x: Tensor) -> Tensor:
        samples = self.predictive(x)

        if self.logits_key not in samples:
            raise KeyError(
                f"Predictive output does not contain logits_key={self.logits_key!r}. "
                f"Available keys: {list(samples.keys())}"
            )

        logits_samples = samples[self.logits_key]

        if logits_samples.ndim == 3:
            return logits_samples.mean(dim=0)
        if logits_samples.ndim == 2:
            return logits_samples

        raise ValueError(
            "Expected logits samples with shape (mc, batch, classes) or "
            f"(batch, classes), got {tuple(logits_samples.shape)}"
        )


def monte_carlo_attribution(
    model_sampler: Callable[[Tensor], torch.nn.Module],
    x: Tensor,
    attribution_fn: Callable[..., Tensor],
    n_samples: int = 30,
    target_class: Optional[Tensor] = None,
    normalize_outputs: bool = True,
    **attr_kwargs,
) -> Tuple[Tensor, Tensor]:
    """
    Compute attribution mean and standard deviation over posterior model samples.

    This is useful for Bayesian neural networks when you can sample a concrete
    PyTorch model from the posterior.

    Parameters
    ----------
    model_sampler:
        Callable that returns a sampled PyTorch model from the posterior. It is
        called as ``model_sampler(x)``.
    x:
        ECG tensor with shape ``(batch, leads, time)``.
    attribution_fn:
        Attribution function such as ``vanilla_saliency`` or
        ``integrated_gradients``.
    n_samples:
        Number of posterior model samples.
    target_class:
        Optional integer tensor with shape ``(batch,)``.
    normalize_outputs:
        If ``True``, normalize mean and standard deviation maps separately.
    **attr_kwargs:
        Extra arguments passed to ``attribution_fn``.

    Returns
    -------
    attr_mean:
        Mean attribution tensor with shape ``(batch, leads, time)``.
    attr_std:
        Attribution standard deviation tensor with shape ``(batch, leads, time)``.
    """
    _validate_ecg_tensor(x)
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    attrs: List[Tensor] = []

    for _ in range(n_samples):
        sampled_model = model_sampler(x)

        attr = attribution_fn(
            sampled_model,
            x,
            target_class=target_class,
            normalize=False,
            **attr_kwargs,
        )

        attrs.append(attr.detach())

    stacked = torch.stack(attrs, dim=0)

    attr_mean = stacked.mean(dim=0)
    attr_std = stacked.std(dim=0, unbiased=False)

    if normalize_outputs:
        attr_mean = normalize_attribution(attr_mean)
        attr_std = normalize_attribution(attr_std)

    return attr_mean, attr_std


def pyro_integrated_gradients_mc(
    predictive: Callable[..., Dict[str, Tensor]],
    x: Tensor,
    logits_key: str = "logits",
    target_class: Optional[Tensor] = None,
    baseline: Optional[Tensor] = None,
    steps: int = 32,
    n_mc: int = 50,
    absolute: bool = True,
    normalize: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Compute Monte Carlo integrated gradients for a Pyro Bayesian neural network.

    Parameters
    ----------
    predictive:
        Pyro ``Predictive`` object or compatible callable. It should return a
        dictionary containing logits under ``logits_key``.
    x:
        ECG tensor with shape ``(batch, leads, time)``.
    logits_key:
        Dictionary key containing logits in the predictive output.
    target_class:
        Optional integer tensor with shape ``(batch,)``. If ``None``, the class
        with largest posterior mean logit is explained.
    baseline:
        Baseline ECG tensor. If ``None``, uses zeros.
    steps:
        Number of integrated-gradient interpolation steps.
    n_mc:
        Number of Monte Carlo attribution repetitions.
    absolute:
        If ``True``, return absolute attributions.
    normalize:
        If ``True``, normalize mean and standard deviation maps separately.

    Returns
    -------
    attr_mean:
        Mean attribution over MC samples, shape ``(batch, leads, time)``.
    attr_std:
        Attribution standard deviation over MC samples, shape
        ``(batch, leads, time)``.
    """
    _validate_ecg_tensor(x)
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if n_mc < 1:
        raise ValueError("n_mc must be >= 1")

    x_detached = x.detach()

    if baseline is None:
        baseline = torch.zeros_like(x_detached)
    else:
        baseline = baseline.detach().to(device=x.device, dtype=x.dtype)
        baseline = torch.broadcast_to(baseline, x_detached.shape)

    if target_class is None:
        with torch.no_grad():
            samples = predictive(x_detached)
            logits_samples = _extract_logits_samples(samples, logits_key)
            mean_logits = _mean_logits(logits_samples)
            target_class = mean_logits.argmax(dim=1)
    else:
        target_class = target_class.to(device=x.device, dtype=torch.long)

    all_attrs: List[Tensor] = []

    for _ in range(n_mc):
        total_gradients = torch.zeros_like(x_detached)

        for alpha in torch.linspace(0.0, 1.0, steps, device=x.device, dtype=x.dtype):
            x_interp = baseline + alpha * (x_detached - baseline)
            x_interp = x_interp.detach().clone().requires_grad_(True)

            samples = predictive(x_interp)
            logits_samples = _extract_logits_samples(samples, logits_key)
            logits = _mean_logits(logits_samples)

            scores, _ = get_target_scores(logits, target_class)

            scores.sum().backward()

            if x_interp.grad is None:
                raise RuntimeError(
                    "Gradient was not computed during Pyro integrated gradients."
                )

            total_gradients += x_interp.grad.detach()

        avg_gradients = total_gradients / float(steps)
        attr = (x_detached - baseline) * avg_gradients

        if absolute:
            attr = attr.abs()

        all_attrs.append(attr)

    stacked = torch.stack(all_attrs, dim=0)

    attr_mean = stacked.mean(dim=0)
    attr_std = stacked.std(dim=0, unbiased=False)

    if normalize:
        attr_mean = normalize_attribution(attr_mean)
        attr_std = normalize_attribution(attr_std)

    return attr_mean, attr_std


def confidence_weighted_attribution(
    attr_mean: Tensor,
    attr_std: Tensor,
    eps: float = 1e-6,
    normalize: bool = True,
) -> Tensor:
    """
    Compute a confidence-weighted attribution map.

    High values indicate high mean attribution and low posterior attribution
    uncertainty.

    Parameters
    ----------
    attr_mean:
        Mean attribution tensor with shape ``(batch, leads, time)``.
    attr_std:
        Attribution standard deviation tensor with the same shape.
    eps:
        Numerical stability constant.
    normalize:
        If ``True``, normalize per sample to [0, 1].

    Returns
    -------
    stable_attr:
        Confidence-weighted attribution tensor with shape ``(batch, leads, time)``.
    """
    if attr_mean.shape != attr_std.shape:
        raise ValueError(
            f"attr_mean and attr_std must have the same shape, got "
            f"{tuple(attr_mean.shape)} and {tuple(attr_std.shape)}"
        )

    stable_attr = attr_mean / (attr_std + eps)

    if normalize:
        stable_attr = normalize_attribution(stable_attr)

    return stable_attr


def aggregate_attribution_windows(
    attribution: Tensor,
    window_size: int,
    stride: int,
    reduce: str = "mean",
) -> Tensor:
    """
    Aggregate attribution into sliding lead-specific time windows.

    Parameters
    ----------
    attribution:
        Tensor with shape ``(batch, leads, time)``.
    window_size:
        Window length in time samples.
    stride:
        Window stride in time samples.
    reduce:
        Aggregation method. Options: ``"mean"``, ``"sum"``, ``"max"``.

    Returns
    -------
    window_scores:
        Tensor with shape ``(batch, leads, n_windows)``.
    """
    _validate_ecg_tensor(attribution)
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if reduce not in {"mean", "sum", "max"}:
        raise ValueError("reduce must be one of {'mean', 'sum', 'max'}")

    _, _, n_time = attribution.shape

    scores: List[Tensor] = []

    for start in range(0, max(n_time - window_size + 1, 1), stride):
        end = min(start + window_size, n_time)
        window = attribution[:, :, start:end]

        if reduce == "mean":
            score = window.mean(dim=2)
        elif reduce == "sum":
            score = window.sum(dim=2)
        else:
            score = window.max(dim=2).values

        scores.append(score)

        if end == n_time:
            break

    return torch.stack(scores, dim=2)


def top_ecg_regions(
    window_scores: Tensor,
    top_k: int = 10,
    lead_names: Optional[Sequence[str]] = None,
    window_size: int = 100,
    stride: int = 25,
    sampling_rate: Optional[float] = None,
    sample_idx: int = 0,
) -> List[Dict[str, Union[str, int, float]]]:
    """
    Return top lead-window regions by attribution score.

    Parameters
    ----------
    window_scores:
        Tensor with shape ``(batch, leads, n_windows)``.
    top_k:
        Number of top regions to return.
    lead_names:
        Names of ECG leads. If ``None``, uses standard 12-lead names when the
        number of leads is 12, otherwise uses ``Lead_0``, ``Lead_1``, etc.
    window_size:
        Window length in time samples.
    stride:
        Window stride in time samples.
    sampling_rate:
        Sampling rate in Hz. If provided, start/end are returned in seconds.
        Otherwise, start/end are returned as sample indices.
    sample_idx:
        Batch index to inspect.

    Returns
    -------
    regions:
        List of dictionaries describing the highest-scoring regions.
    """
    if window_scores.ndim != 3:
        raise ValueError(
            "Expected window_scores with shape (batch, leads, n_windows), got "
            f"{tuple(window_scores.shape)}"
        )

    batch, n_leads, n_windows = window_scores.shape

    if sample_idx < 0 or sample_idx >= batch:
        raise IndexError(f"sample_idx={sample_idx} is outside batch size {batch}")

    if lead_names is None:
        lead_names = _default_lead_names(n_leads)

    if len(lead_names) != n_leads:
        raise ValueError(
            f"lead_names length must match number of leads: {len(lead_names)} != {n_leads}"
        )

    scores = window_scores[sample_idx]
    flat = scores.flatten()
    k = min(top_k, flat.numel())

    values, indices = torch.topk(flat, k=k)

    regions: List[Dict[str, Union[str, int, float]]] = []

    for value, idx in zip(values.detach().cpu(), indices.detach().cpu()):
        idx_int = int(idx.item())
        lead_idx = idx_int // n_windows
        window_idx = idx_int % n_windows

        start = int(window_idx * stride)
        end = int(start + window_size)

        if sampling_rate is not None:
            region: Dict[str, Union[str, int, float]] = {
                "lead": lead_names[lead_idx],
                "start_sec": start / sampling_rate,
                "end_sec": end / sampling_rate,
                "score": float(value.item()),
            }
        else:
            region = {
                "lead": lead_names[lead_idx],
                "start_idx": start,
                "end_idx": end,
                "score": float(value.item()),
            }

        regions.append(region)

    return regions


def summarize_lead_importance(
    lead_importance: Tensor,
    lead_names: Optional[Sequence[str]] = None,
    sample_idx: int = 0,
    normalize: bool = True,
) -> List[Dict[str, Union[str, float]]]:
    """
    Return lead-wise importance as a sorted list of dictionaries.

    Parameters
    ----------
    lead_importance:
        Tensor with shape ``(batch, leads)``.
    lead_names:
        Lead names. If ``None``, uses standard 12-lead names when possible.
    sample_idx:
        Batch index to inspect.
    normalize:
        If ``True``, normalize the selected sample's lead importance to sum to 1.

    Returns
    -------
    summary:
        Sorted list of dictionaries with keys ``lead`` and ``importance``.
    """
    if lead_importance.ndim != 2:
        raise ValueError(
            f"Expected lead_importance with shape (batch, leads), got {tuple(lead_importance.shape)}"
        )

    batch, n_leads = lead_importance.shape

    if sample_idx < 0 or sample_idx >= batch:
        raise IndexError(f"sample_idx={sample_idx} is outside batch size {batch}")

    if lead_names is None:
        lead_names = _default_lead_names(n_leads)

    values = lead_importance[sample_idx].detach().cpu().clone()

    if normalize:
        denom = values.sum().clamp_min(1e-8)
        values = values / denom

    order = torch.argsort(values, descending=True)

    return [
        {"lead": lead_names[int(i)], "importance": float(values[int(i)].item())}
        for i in order
    ]


def plot_ecg_with_attribution(
    x: Tensor,
    attribution: Tensor,
    sample_idx: int = 0,
    lead_names: Optional[Sequence[str]] = None,
    sampling_rate: Optional[float] = None,
    figsize: Tuple[float, float] = (14.0, 2.5),
    alpha: float = 0.3,
    show: bool = True,
):
    """
    Plot each ECG lead with attribution intensity underlaid.

    This function imports matplotlib only when called.

    Parameters
    ----------
    x:
        ECG tensor with shape ``(batch, leads, time)``.
    attribution:
        Attribution tensor with shape ``(batch, leads, time)``.
    sample_idx:
        Batch index to plot.
    lead_names:
        Optional lead names.
    sampling_rate:
        Sampling rate in Hz. If provided, x-axis is in seconds.
    figsize:
        Figure size for each lead plot.
    alpha:
        Transparency of the attribution overlay.
    show:
        If ``True``, call ``plt.show()`` for each figure.

    Returns
    -------
    figures:
        List of matplotlib Figure objects.
    """
    import matplotlib.pyplot as plt

    _validate_ecg_tensor(x)
    _validate_ecg_tensor(attribution)

    if x.shape != attribution.shape:
        raise ValueError(
            f"x and attribution must have the same shape, got {tuple(x.shape)} and "
            f"{tuple(attribution.shape)}"
        )

    batch, n_leads, n_time = x.shape

    if sample_idx < 0 or sample_idx >= batch:
        raise IndexError(f"sample_idx={sample_idx} is outside batch size {batch}")

    if lead_names is None:
        lead_names = _default_lead_names(n_leads)

    ecg = x[sample_idx].detach().cpu()
    attr = attribution[sample_idx].detach().cpu()

    if sampling_rate is not None:
        t = torch.arange(n_time, dtype=torch.float32) / float(sampling_rate)
        xlabel = "Time (s)"
    else:
        t = torch.arange(n_time)
        xlabel = "Time index"

    figures = []

    for lead in range(n_leads):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        ax.plot(t, ecg[lead])

        attr_scaled = attr[lead]
        if attr_scaled.max() > attr_scaled.min():
            attr_scaled = (attr_scaled - attr_scaled.min()) / (
                attr_scaled.max() - attr_scaled.min() + 1e-8
            )

        y_min = ecg[lead].min()
        y_max = ecg[lead].max()
        y_range = (y_max - y_min).clamp_min(1e-8)

        ax.fill_between(
            t,
            y_min,
            y_min + attr_scaled * y_range,
            alpha=alpha,
        )

        ax.set_title(f"Lead {lead_names[lead]}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Amplitude")
        fig.tight_layout()

        figures.append(fig)

        if show:
            plt.show()

    return figures


def plot_lead_importance(
    lead_importance: Tensor,
    sample_idx: int = 0,
    lead_names: Optional[Sequence[str]] = None,
    normalize: bool = True,
    show: bool = True,
):
    """
    Plot lead-wise importance for one ECG sample.

    This function imports matplotlib only when called.

    Parameters
    ----------
    lead_importance:
        Tensor with shape ``(batch, leads)``.
    sample_idx:
        Batch index to plot.
    lead_names:
        Optional lead names.
    normalize:
        If ``True``, normalize values to sum to 1.
    show:
        If ``True``, call ``plt.show()``.

    Returns
    -------
    fig:
        Matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    summary = summarize_lead_importance(
        lead_importance=lead_importance,
        lead_names=lead_names,
        sample_idx=sample_idx,
        normalize=normalize,
    )

    leads = [item["lead"] for item in summary]
    values = [item["importance"] for item in summary]

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.bar(leads, values)
    ax.set_xlabel("Lead")
    ax.set_ylabel("Importance")
    ax.set_title("Lead-wise ECG importance")
    fig.tight_layout()

    if show:
        plt.show()

    return fig


def _validate_ecg_tensor(x: Tensor) -> None:
    """Validate that input is a 3D ECG-like tensor."""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")
    if x.ndim != 3:
        raise ValueError(
            f"Expected ECG tensor with shape (batch, leads, time), got {tuple(x.shape)}"
        )


def _default_lead_names(n_leads: int) -> List[str]:
    """Return standard 12-lead names when possible, otherwise generic names."""
    if n_leads == 12:
        return DEFAULT_12_LEAD_NAMES.copy()
    return [f"Lead_{i}" for i in range(n_leads)]


def _extract_logits_samples(samples: Dict[str, Tensor], logits_key: str) -> Tensor:
    """Extract logits from a Pyro Predictive-style output dictionary."""
    if logits_key not in samples:
        raise KeyError(
            f"Predictive output does not contain logits_key={logits_key!r}. "
            f"Available keys: {list(samples.keys())}"
        )
    return samples[logits_key]


def _mean_logits(logits_samples: Tensor) -> Tensor:
    """Convert logits samples to mean logits."""
    if logits_samples.ndim == 3:
        return logits_samples.mean(dim=0)
    if logits_samples.ndim == 2:
        return logits_samples
    raise ValueError(
        "Expected logits samples with shape (mc, batch, classes) or "
        f"(batch, classes), got {tuple(logits_samples.shape)}"
    )


__all__ = [
    "DEFAULT_12_LEAD_NAMES",
    "BayesianPredictiveWrapper",
    "aggregate_attribution_windows",
    "confidence_weighted_attribution",
    "get_target_scores",
    "integrated_gradients",
    "lead_occlusion_importance",
    "monte_carlo_attribution",
    "normalize_attribution",
    "occlusion_sensitivity",
    "plot_ecg_with_attribution",
    "plot_lead_importance",
    "pyro_integrated_gradients_mc",
    "summarize_lead_importance",
    "top_ecg_regions",
    "vanilla_saliency",
]
