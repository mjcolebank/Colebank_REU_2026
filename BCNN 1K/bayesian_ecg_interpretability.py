"""
bayesian_ecg_interpretability.py

Interpretability utilities for a Pyro BayesianSmallCNN-style model.

This module is designed for a PyroModule Bayesian CNN like:

    class BayesianSmallCNN(PyroModule):
        ...
        def forward(self, x, y=None, dataset_size=None):
            ...
            return logits

Important design choice
-----------------------
Pyro's Predictive is convenient for posterior prediction, but it is not ideal
inside gradient-based attribution because Predictive may run the model under
torch.no_grad(). For input-gradient interpretability, this module uses the
following workflow:

1. Draw posterior samples from the trained guide.
2. Condition the Bayesian model on one posterior sample at a time using
   pyro.poutine.condition.
3. Run the conditioned model with gradients enabled with respect to the ECG input.
4. Aggregate attributions across posterior samples.

Expected ECG shape
------------------
    x.shape == (batch_size, 12, n_timepoints)

Typical usage
-------------
from pyro.infer import Predictive
from bayesian_ecg_interpretability import (
    draw_posterior_samples_from_guide,
    bayesian_integrated_gradients,
    confidence_weighted_attribution,
    plot_ecg_with_attribution,
)

model.eval()

x_batch = x_batch.to(model._device_anchor.device)

posterior_samples = draw_posterior_samples_from_guide(
    guide=guide,
    x=x_batch,
    num_samples=50,
)

attr_mean, attr_std = bayesian_integrated_gradients(
    model=model,
    posterior_samples=posterior_samples,
    x=x_batch,
    steps=32,
)

stable_attr = confidence_weighted_attribution(attr_mean, attr_std)

plot_ecg_with_attribution(
    x=x_batch,
    attribution=stable_attr,
    sample_idx=0,
    sampling_rate=500,
)
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

import pyro
import pyro.poutine as poutine
from pyro.infer import Predictive


Tensor = torch.Tensor


DEFAULT_12_LEAD_NAMES: List[str] = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------


def get_model_device(model: torch.nn.Module) -> torch.device:
    """
    Return the model device.

    Your BayesianSmallCNN registers a buffer called ``_device_anchor``.
    This function uses it when available. Otherwise, it falls back to
    parameters or buffers.
    """
    if hasattr(model, "_device_anchor"):
        return model._device_anchor.device

    try:
        return next(model.parameters()).device
    except StopIteration:
        pass

    try:
        return next(model.buffers()).device
    except StopIteration:
        pass

    return torch.device("cpu")


def validate_ecg_tensor(x: Tensor) -> None:
    """
    Validate an ECG-like tensor with shape ``(batch, leads, time)``.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")

    if x.ndim != 3:
        raise ValueError(
            f"Expected ECG tensor with shape (batch, leads, time), got {tuple(x.shape)}"
        )


def normalize_attribution(attr: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Normalize attribution values per sample to [0, 1].

    Parameters
    ----------
    attr:
        Tensor with shape ``(batch, leads, time)``.
    eps:
        Numerical stability constant.

    Returns
    -------
    Tensor with same shape as ``attr``.
    """
    validate_ecg_tensor(attr)

    batch = attr.shape[0]
    flat = attr.reshape(batch, -1)

    min_vals = flat.min(dim=1, keepdim=True).values
    max_vals = flat.max(dim=1, keepdim=True).values

    norm = (flat - min_vals) / (max_vals - min_vals + eps)
    return norm.reshape_as(attr)


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
        Optional tensor with shape ``(batch,)``. If ``None``, the predicted
        class is used.

    Returns
    -------
    scores:
        Tensor with shape ``(batch,)``.
    target_class:
        Tensor with shape ``(batch,)``.
    """
    if logits.ndim != 2:
        raise ValueError(
            f"Expected logits with shape (batch, n_classes), got {tuple(logits.shape)}"
        )

    if target_class is None:
        target_class = logits.argmax(dim=1)
    else:
        target_class = target_class.to(device=logits.device, dtype=torch.long)

    scores = logits.gather(1, target_class.view(-1, 1)).squeeze(1)
    return scores, target_class


def _default_lead_names(n_leads: int) -> List[str]:
    if n_leads == 12:
        return DEFAULT_12_LEAD_NAMES.copy()
    return [f"Lead_{idx}" for idx in range(n_leads)]


# ---------------------------------------------------------------------
# Posterior sample helpers for Pyro BayesianSmallCNN
# ---------------------------------------------------------------------


@torch.no_grad()
def draw_posterior_samples_from_guide(
    guide: Callable,
    x: Tensor,
    num_samples: int = 50,
    y: Optional[Tensor] = None,
    dataset_size: Optional[int] = None,
    return_sites: Optional[Sequence[str]] = None,
) -> Dict[str, Tensor]:
    """
    Draw posterior latent samples from a trained Pyro guide.

    This is usually the first step before Bayesian attribution.

    Parameters
    ----------
    guide:
        Trained Pyro guide, for example an AutoDiagonalNormal guide.
    x:
        ECG tensor with shape ``(batch, leads, time)``.
    num_samples:
        Number of posterior samples to draw.
    y:
        Optional labels. Usually leave as ``None`` for attribution.
    dataset_size:
        Optional dataset size argument if your guide expects it.
    return_sites:
        Optional list of latent site names to return. Usually leave as ``None``.

    Returns
    -------
    posterior_samples:
        Dictionary mapping latent site names to tensors. Each tensor should
        have leading dimension ``num_samples``.
    """
    validate_ecg_tensor(x)

    predictive = Predictive(
        guide,
        num_samples=num_samples,
        return_sites=return_sites,
    )

    posterior_samples = predictive(
        x,
        y=y,
        dataset_size=dataset_size,
    )

    posterior_samples = {
        name: value for name, value in posterior_samples.items()
        if torch.is_tensor(value)
    }

    if len(posterior_samples) == 0:
        raise RuntimeError(
            "No tensor posterior samples were returned from the guide. "
            "Check that the guide is trained and that return_sites is not excluding "
            "all latent parameters."
        )

    return posterior_samples


def get_num_posterior_samples(posterior_samples: Dict[str, Tensor]) -> int:
    """
    Infer number of posterior samples from a posterior sample dictionary.
    """
    for value in posterior_samples.values():
        if torch.is_tensor(value):
            return int(value.shape[0])

    raise ValueError("posterior_samples contains no tensors.")


def select_single_posterior_sample(
    posterior_samples: Dict[str, Tensor],
    sample_idx: int,
) -> Dict[str, Tensor]:
    """
    Select one posterior sample from a dictionary of posterior samples.

    Parameters
    ----------
    posterior_samples:
        Dictionary where each tensor has leading dimension ``num_samples``.
    sample_idx:
        Posterior sample index.

    Returns
    -------
    single_sample:
        Dictionary suitable for ``pyro.poutine.condition(model, data=single_sample)``.
    """
    single_sample: Dict[str, Tensor] = {}

    for name, value in posterior_samples.items():
        if not torch.is_tensor(value):
            continue

        if value.ndim == 0:
            single_sample[name] = value
        else:
            single_sample[name] = value[sample_idx]

    return single_sample


def conditioned_logits(
    model: torch.nn.Module,
    posterior_samples: Dict[str, Tensor],
    sample_idx: int,
    x: Tensor,
) -> Tensor:
    """
    Run a Pyro Bayesian model conditioned on one posterior sample.

    This function preserves gradients with respect to ``x``.

    Parameters
    ----------
    model:
        Your BayesianSmallCNN PyroModule.
    posterior_samples:
        Dictionary of posterior samples from ``draw_posterior_samples_from_guide``.
    sample_idx:
        Which posterior sample to use.
    x:
        ECG tensor with shape ``(batch, 12, time)``.

    Returns
    -------
    logits:
        Tensor with shape ``(batch, n_classes)``.
    """
    validate_ecg_tensor(x)

    single_sample = select_single_posterior_sample(posterior_samples, sample_idx)

    conditioned_model = poutine.condition(model, data=single_sample)

    # For attribution, do not pass y. The model's obs site is unobserved and
    # irrelevant because we use only the returned logits.
    logits = conditioned_model(
        x,
        y=None,
        dataset_size=None,
    )

    if logits.ndim != 2:
        raise ValueError(
            f"Expected conditioned model to return logits with shape "
            f"(batch, classes), got {tuple(logits.shape)}"
        )

    return logits


@torch.no_grad()
def posterior_mean_logits(
    model: torch.nn.Module,
    posterior_samples: Dict[str, Tensor],
    x: Tensor,
    max_samples: Optional[int] = None,
) -> Tensor:
    """
    Compute posterior mean logits by averaging conditioned model outputs.

    Parameters
    ----------
    model:
        Bayesian PyroModule.
    posterior_samples:
        Posterior sample dictionary.
    x:
        ECG tensor.
    max_samples:
        Optional cap on posterior samples used.

    Returns
    -------
    mean_logits:
        Tensor with shape ``(batch, n_classes)``.
    """
    validate_ecg_tensor(x)
    model.eval()

    n_samples = get_num_posterior_samples(posterior_samples)
    if max_samples is not None:
        n_samples = min(n_samples, int(max_samples))

    logits_list = []

    for sample_idx in range(n_samples):
        logits_i = conditioned_logits(
            model=model,
            posterior_samples=posterior_samples,
            sample_idx=sample_idx,
            x=x,
        )
        logits_list.append(logits_i)

    return torch.stack(logits_list, dim=0).mean(dim=0)


@torch.no_grad()
def posterior_mean_probs(
    model: torch.nn.Module,
    posterior_samples: Dict[str, Tensor],
    x: Tensor,
    max_samples: Optional[int] = None,
) -> Tensor:
    """
    Compute posterior mean class probabilities.

    This averages probabilities, not logits.
    """
    validate_ecg_tensor(x)
    model.eval()

    n_samples = get_num_posterior_samples(posterior_samples)
    if max_samples is not None:
        n_samples = min(n_samples, int(max_samples))

    probs_list = []

    for sample_idx in range(n_samples):
        logits_i = conditioned_logits(
            model=model,
            posterior_samples=posterior_samples,
            sample_idx=sample_idx,
            x=x,
        )
        probs_list.append(F.softmax(logits_i, dim=1))

    return torch.stack(probs_list, dim=0).mean(dim=0)


@torch.no_grad()
def predict_target_class(
    model: torch.nn.Module,
    posterior_samples: Dict[str, Tensor],
    x: Tensor,
    use_probabilities: bool = True,
    max_samples: Optional[int] = None,
) -> Tensor:
    """
    Predict a target class using posterior mean probabilities or logits.

    Parameters
    ----------
    use_probabilities:
        If ``True``, use posterior mean probabilities. If ``False``, use
        posterior mean logits.
    """
    if use_probabilities:
        mean_output = posterior_mean_probs(
            model=model,
            posterior_samples=posterior_samples,
            x=x,
            max_samples=max_samples,
        )
    else:
        mean_output = posterior_mean_logits(
            model=model,
            posterior_samples=posterior_samples,
            x=x,
            max_samples=max_samples,
        )

    return mean_output.argmax(dim=1)


# ---------------------------------------------------------------------
# Bayesian gradient attribution
# ---------------------------------------------------------------------


def bayesian_vanilla_saliency(
    model: torch.nn.Module,
    posterior_samples: Dict[str, Tensor],
    x: Tensor,
    target_class: Optional[Tensor] = None,
    n_mc: Optional[int] = None,
    absolute: bool = True,
    normalize: bool = True,
    use_probabilities_for_target: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Compute Bayesian vanilla saliency by averaging gradients across posterior samples.

    Parameters
    ----------
    model:
        BayesianSmallCNN PyroModule.
    posterior_samples:
        Posterior samples from the trained guide.
    x:
        ECG tensor with shape ``(batch, 12, time)``.
    target_class:
        Optional target class tensor. If ``None``, class is chosen using
        posterior mean prediction.
    n_mc:
        Number of posterior samples to use. If ``None``, use all samples.
    absolute:
        If ``True``, use absolute gradients.
    normalize:
        If ``True``, normalize returned mean/std maps per sample.
    use_probabilities_for_target:
        Whether to choose default target class using posterior mean probabilities.

    Returns
    -------
    attr_mean:
        Mean saliency over posterior samples, shape ``(batch, leads, time)``.
    attr_std:
        Standard deviation of saliency over posterior samples.
    """
    validate_ecg_tensor(x)
    model.eval()

    device = get_model_device(model)
    x = x.to(device)

    total_samples = get_num_posterior_samples(posterior_samples)
    if n_mc is None:
        n_mc = total_samples
    else:
        n_mc = min(int(n_mc), total_samples)

    if target_class is None:
        target_class = predict_target_class(
            model=model,
            posterior_samples=posterior_samples,
            x=x,
            use_probabilities=use_probabilities_for_target,
            max_samples=n_mc,
        )
    else:
        target_class = target_class.to(device=x.device, dtype=torch.long)

    attrs = []

    for sample_idx in range(n_mc):
        x_grad = x.detach().clone().requires_grad_(True)

        logits = conditioned_logits(
            model=model,
            posterior_samples=posterior_samples,
            sample_idx=sample_idx,
            x=x_grad,
        )

        scores, _ = get_target_scores(logits, target_class)

        model.zero_grad(set_to_none=True)

        if x_grad.grad is not None:
            x_grad.grad.zero_()

        scores.sum().backward()

        if x_grad.grad is None:
            raise RuntimeError(
                "x_grad.grad is None. Check that gradients are enabled and that "
                "the returned logits depend on x."
            )

        attr = x_grad.grad.detach()

        if absolute:
            attr = attr.abs()

        attrs.append(attr)

    stacked = torch.stack(attrs, dim=0)

    attr_mean = stacked.mean(dim=0)
    attr_std = stacked.std(dim=0, unbiased=False)

    if normalize:
        attr_mean = normalize_attribution(attr_mean)
        attr_std = normalize_attribution(attr_std)

    return attr_mean, attr_std


def bayesian_integrated_gradients(
    model: torch.nn.Module,
    posterior_samples: Dict[str, Tensor],
    x: Tensor,
    target_class: Optional[Tensor] = None,
    baseline: Optional[Tensor] = None,
    steps: int = 32,
    n_mc: Optional[int] = None,
    absolute: bool = True,
    normalize: bool = True,
    use_probabilities_for_target: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Compute Bayesian integrated gradients for BayesianSmallCNN.

    This uses posterior samples from the guide and runs the model under
    ``pyro.poutine.condition`` so that gradients with respect to input ECG
    remain enabled.

    Parameters
    ----------
    model:
        BayesianSmallCNN PyroModule.
    posterior_samples:
        Posterior sample dictionary from ``draw_posterior_samples_from_guide``.
    x:
        ECG tensor with shape ``(batch, 12, time)``.
    target_class:
        Optional target class tensor. If ``None``, class is chosen using
        posterior mean prediction.
    baseline:
        Baseline ECG. If ``None``, uses zeros.
    steps:
        Number of integrated-gradient interpolation points.
    n_mc:
        Number of posterior samples to use. If ``None``, use all samples.
    absolute:
        If ``True``, use absolute attribution values.
    normalize:
        If ``True``, normalize returned mean/std maps per sample.
    use_probabilities_for_target:
        Whether to choose default target class using posterior mean probabilities.

    Returns
    -------
    attr_mean:
        Mean integrated gradients over posterior samples.
    attr_std:
        Standard deviation of integrated gradients over posterior samples.
    """
    validate_ecg_tensor(x)

    if steps < 1:
        raise ValueError("steps must be >= 1")

    model.eval()

    device = get_model_device(model)
    x = x.detach().to(device)

    if baseline is None:
        baseline = torch.zeros_like(x)
    else:
        baseline = baseline.detach().to(device=device, dtype=x.dtype)
        baseline = torch.broadcast_to(baseline, x.shape)

    total_samples = get_num_posterior_samples(posterior_samples)
    if n_mc is None:
        n_mc = total_samples
    else:
        n_mc = min(int(n_mc), total_samples)

    if target_class is None:
        target_class = predict_target_class(
            model=model,
            posterior_samples=posterior_samples,
            x=x,
            use_probabilities=use_probabilities_for_target,
            max_samples=n_mc,
        )
    else:
        target_class = target_class.to(device=x.device, dtype=torch.long)

    all_attrs = []

    alphas = torch.linspace(
        0.0,
        1.0,
        steps,
        device=x.device,
        dtype=x.dtype,
    )

    for sample_idx in range(n_mc):
        total_gradients = torch.zeros_like(x)

        for alpha in alphas:
            x_interp = baseline + alpha * (x - baseline)
            x_interp = x_interp.detach().clone().requires_grad_(True)

            logits = conditioned_logits(
                model=model,
                posterior_samples=posterior_samples,
                sample_idx=sample_idx,
                x=x_interp,
            )

            scores, _ = get_target_scores(logits, target_class)

            model.zero_grad(set_to_none=True)

            if x_interp.grad is not None:
                x_interp.grad.zero_()

            scores.sum().backward()

            if x_interp.grad is None:
                raise RuntimeError(
                    "x_interp.grad is None. Check that gradients are enabled and "
                    "that the conditioned model output depends on x."
                )

            total_gradients += x_interp.grad.detach()

        avg_gradients = total_gradients / float(steps)
        attr = (x - baseline) * avg_gradients

        if absolute:
            attr = attr.abs()

        all_attrs.append(attr.detach())

    stacked = torch.stack(all_attrs, dim=0)

    attr_mean = stacked.mean(dim=0)
    attr_std = stacked.std(dim=0, unbiased=False)

    if normalize:
        attr_mean = normalize_attribution(attr_mean)
        attr_std = normalize_attribution(attr_std)

    return attr_mean, attr_std


# ---------------------------------------------------------------------
# Bayesian occlusion attribution
# ---------------------------------------------------------------------


@torch.no_grad()
def bayesian_occlusion_sensitivity(
    model: torch.nn.Module,
    posterior_samples: Dict[str, Tensor],
    x: Tensor,
    target_class: Optional[Tensor] = None,
    window_size: int = 50,
    stride: int = 10,
    baseline_value: float = 0.0,
    n_mc: Optional[int] = None,
    use_probabilities: bool = True,
    normalize: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Compute Bayesian occlusion sensitivity over lead-specific time windows.

    Parameters
    ----------
    model:
        BayesianSmallCNN PyroModule.
    posterior_samples:
        Posterior sample dictionary.
    x:
        ECG tensor with shape ``(batch, 12, time)``.
    target_class:
        Optional target class tensor.
    window_size:
        Time-window size to occlude.
    stride:
        Sliding-window stride.
    baseline_value:
        Value used to replace occluded segments.
    n_mc:
        Number of posterior samples to use. If ``None``, use all samples.
    use_probabilities:
        If ``True``, use class probabilities. If ``False``, use logits.
    normalize:
        If ``True``, normalize returned mean/std maps.

    Returns
    -------
    attr_mean:
        Mean occlusion importance across posterior samples.
    attr_std:
        Standard deviation across posterior samples.
    """
    validate_ecg_tensor(x)

    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    model.eval()

    device = get_model_device(model)
    x = x.to(device)

    batch, n_leads, n_time = x.shape

    total_samples = get_num_posterior_samples(posterior_samples)
    if n_mc is None:
        n_mc = total_samples
    else:
        n_mc = min(int(n_mc), total_samples)

    if target_class is None:
        target_class = predict_target_class(
            model=model,
            posterior_samples=posterior_samples,
            x=x,
            use_probabilities=use_probabilities,
            max_samples=n_mc,
        )
    else:
        target_class = target_class.to(device=x.device, dtype=torch.long)

    sample_attrs = []

    for sample_idx in range(n_mc):
        logits = conditioned_logits(
            model=model,
            posterior_samples=posterior_samples,
            sample_idx=sample_idx,
            x=x,
        )

        output = F.softmax(logits, dim=1) if use_probabilities else logits
        original_scores = output.gather(1, target_class.view(-1, 1)).squeeze(1)

        importance = torch.zeros_like(x)
        counts = torch.zeros_like(x)

        for lead in range(n_leads):
            for start in range(0, max(n_time - window_size + 1, 1), stride):
                end = min(start + window_size, n_time)

                x_occ = x.clone()
                x_occ[:, lead, start:end] = baseline_value

                occ_logits = conditioned_logits(
                    model=model,
                    posterior_samples=posterior_samples,
                    sample_idx=sample_idx,
                    x=x_occ,
                )

                occ_output = F.softmax(occ_logits, dim=1) if use_probabilities else occ_logits
                occ_scores = occ_output.gather(1, target_class.view(-1, 1)).squeeze(1)

                score_drop = original_scores - occ_scores

                importance[:, lead, start:end] += score_drop.view(batch, 1)
                counts[:, lead, start:end] += 1

                if end == n_time:
                    break

        importance = importance / counts.clamp_min(1.0)
        importance = importance.clamp_min(0.0)

        sample_attrs.append(importance)

    stacked = torch.stack(sample_attrs, dim=0)

    attr_mean = stacked.mean(dim=0)
    attr_std = stacked.std(dim=0, unbiased=False)

    if normalize:
        attr_mean = normalize_attribution(attr_mean)
        attr_std = normalize_attribution(attr_std)

    return attr_mean, attr_std


@torch.no_grad()
def bayesian_lead_occlusion_importance(
    model: torch.nn.Module,
    posterior_samples: Dict[str, Tensor],
    x: Tensor,
    target_class: Optional[Tensor] = None,
    baseline_value: float = 0.0,
    n_mc: Optional[int] = None,
    use_probabilities: bool = True,
    normalize: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Compute Bayesian lead-wise importance by occluding one lead at a time.

    Parameters
    ----------
    model:
        BayesianSmallCNN PyroModule.
    posterior_samples:
        Posterior sample dictionary.
    x:
        ECG tensor with shape ``(batch, 12, time)``.
    target_class:
        Optional target class tensor.
    baseline_value:
        Replacement value for occluded lead.
    n_mc:
        Number of posterior samples to use.
    use_probabilities:
        If ``True``, use class probabilities. If ``False``, use logits.
    normalize:
        If ``True``, normalize lead importances per sample to sum to 1.

    Returns
    -------
    lead_mean:
        Tensor with shape ``(batch, leads)``.
    lead_std:
        Tensor with shape ``(batch, leads)``.
    """
    validate_ecg_tensor(x)
    model.eval()

    device = get_model_device(model)
    x = x.to(device)

    batch, n_leads, _ = x.shape

    total_samples = get_num_posterior_samples(posterior_samples)
    if n_mc is None:
        n_mc = total_samples
    else:
        n_mc = min(int(n_mc), total_samples)

    if target_class is None:
        target_class = predict_target_class(
            model=model,
            posterior_samples=posterior_samples,
            x=x,
            use_probabilities=use_probabilities,
            max_samples=n_mc,
        )
    else:
        target_class = target_class.to(device=x.device, dtype=torch.long)

    all_lead_importance = []

    for sample_idx in range(n_mc):
        logits = conditioned_logits(
            model=model,
            posterior_samples=posterior_samples,
            sample_idx=sample_idx,
            x=x,
        )

        output = F.softmax(logits, dim=1) if use_probabilities else logits
        original_scores = output.gather(1, target_class.view(-1, 1)).squeeze(1)

        lead_importance = torch.zeros(batch, n_leads, device=x.device, dtype=x.dtype)

        for lead in range(n_leads):
            x_occ = x.clone()
            x_occ[:, lead, :] = baseline_value

            occ_logits = conditioned_logits(
                model=model,
                posterior_samples=posterior_samples,
                sample_idx=sample_idx,
                x=x_occ,
            )

            occ_output = F.softmax(occ_logits, dim=1) if use_probabilities else occ_logits
            occ_scores = occ_output.gather(1, target_class.view(-1, 1)).squeeze(1)

            lead_importance[:, lead] = original_scores - occ_scores

        lead_importance = lead_importance.clamp_min(0.0)
        all_lead_importance.append(lead_importance)

    stacked = torch.stack(all_lead_importance, dim=0)

    lead_mean = stacked.mean(dim=0)
    lead_std = stacked.std(dim=0, unbiased=False)

    if normalize:
        denom = lead_mean.sum(dim=1, keepdim=True).clamp_min(1e-8)
        lead_mean = lead_mean / denom
        lead_std = lead_std / denom

    return lead_mean, lead_std


# ---------------------------------------------------------------------
# Attribution post-processing
# ---------------------------------------------------------------------


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
        Mean attribution tensor.
    attr_std:
        Standard deviation attribution tensor.
    eps:
        Numerical stability constant.
    normalize:
        If ``True``, normalize per sample to [0, 1].

    Returns
    -------
    stable_attr:
        Tensor with same shape as ``attr_mean``.
    """
    if attr_mean.shape != attr_std.shape:
        raise ValueError(
            f"attr_mean and attr_std must have the same shape, got "
            f"{tuple(attr_mean.shape)} and {tuple(attr_std.shape)}"
        )

    stable_attr = attr_mean / (attr_std + eps)

    if normalize:
        if stable_attr.ndim == 3:
            stable_attr = normalize_attribution(stable_attr)
        elif stable_attr.ndim == 2:
            denom = stable_attr.sum(dim=1, keepdim=True).clamp_min(eps)
            stable_attr = stable_attr / denom
        else:
            raise ValueError(
                f"Expected attribution with 2 or 3 dimensions, got {stable_attr.ndim}"
            )

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
        Window length in samples.
    stride:
        Window stride in samples.
    reduce:
        One of ``'mean'``, ``'sum'``, or ``'max'``.

    Returns
    -------
    window_scores:
        Tensor with shape ``(batch, leads, n_windows)``.
    """
    validate_ecg_tensor(attribution)

    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    if reduce not in {"mean", "sum", "max"}:
        raise ValueError("reduce must be one of {'mean', 'sum', 'max'}")

    _, _, n_time = attribution.shape
    scores = []

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
    """
    if window_scores.ndim != 3:
        raise ValueError(
            f"Expected window_scores with shape (batch, leads, windows), got "
            f"{tuple(window_scores.shape)}"
        )

    batch, n_leads, n_windows = window_scores.shape

    if sample_idx < 0 or sample_idx >= batch:
        raise IndexError(f"sample_idx={sample_idx} outside batch size {batch}")

    if lead_names is None:
        lead_names = _default_lead_names(n_leads)

    if len(lead_names) != n_leads:
        raise ValueError(
            f"lead_names length must equal number of leads: {len(lead_names)} != {n_leads}"
        )

    scores = window_scores[sample_idx]
    flat = scores.flatten()

    k = min(int(top_k), flat.numel())
    values, indices = torch.topk(flat, k=k)

    regions: List[Dict[str, Union[str, int, float]]] = []

    for value, index in zip(values.detach().cpu(), indices.detach().cpu()):
        idx = int(index.item())
        lead_idx = idx // n_windows
        window_idx = idx % n_windows

        start = int(window_idx * stride)
        end = int(start + window_size)

        if sampling_rate is not None:
            region = {
                "lead": lead_names[lead_idx],
                "start_sec": start / float(sampling_rate),
                "end_sec": end / float(sampling_rate),
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
    lead_std: Optional[Tensor] = None,
    lead_names: Optional[Sequence[str]] = None,
    sample_idx: int = 0,
    normalize: bool = True,
) -> List[Dict[str, Union[str, float]]]:
    """
    Return sorted lead-wise importance summary.

    Parameters
    ----------
    lead_importance:
        Tensor with shape ``(batch, leads)``.
    lead_std:
        Optional tensor with shape ``(batch, leads)``.
    lead_names:
        Optional lead names.
    sample_idx:
        Batch index.
    normalize:
        If ``True``, normalize importance values to sum to 1.

    Returns
    -------
    summary:
        List of dictionaries sorted by descending importance.
    """
    if lead_importance.ndim != 2:
        raise ValueError(
            f"Expected lead_importance with shape (batch, leads), got "
            f"{tuple(lead_importance.shape)}"
        )

    batch, n_leads = lead_importance.shape

    if sample_idx < 0 or sample_idx >= batch:
        raise IndexError(f"sample_idx={sample_idx} outside batch size {batch}")

    if lead_names is None:
        lead_names = _default_lead_names(n_leads)

    values = lead_importance[sample_idx].detach().cpu().clone()

    if normalize:
        denom = values.sum().clamp_min(1e-8)
        values = values / denom

    std_values = None
    if lead_std is not None:
        std_values = lead_std[sample_idx].detach().cpu().clone()
        if normalize:
            std_values = std_values / denom

    order = torch.argsort(values, descending=True)

    summary = []

    for idx in order:
        idx_int = int(idx.item())
        item: Dict[str, Union[str, float]] = {
            "lead": lead_names[idx_int],
            "importance": float(values[idx_int].item()),
        }

        if std_values is not None:
            item["std"] = float(std_values[idx_int].item())

        summary.append(item)

    return summary


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------


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

    Parameters
    ----------
    x:
        ECG tensor with shape ``(batch, leads, time)``.
    attribution:
        Attribution tensor with same shape as ``x``.
    sample_idx:
        Batch index.
    lead_names:
        Optional ECG lead names.
    sampling_rate:
        Optional sampling rate in Hz. If supplied, x-axis uses seconds.
    figsize:
        Figure size for each lead.
    alpha:
        Attribution overlay transparency.
    show:
        If ``True``, call ``plt.show()``.

    Returns
    -------
    figures:
        List of matplotlib Figure objects.
    """
    import matplotlib.pyplot as plt

    validate_ecg_tensor(x)
    validate_ecg_tensor(attribution)

    if x.shape != attribution.shape:
        raise ValueError(
            f"x and attribution must have same shape, got {tuple(x.shape)} and "
            f"{tuple(attribution.shape)}"
        )

    batch, n_leads, n_time = x.shape

    if sample_idx < 0 or sample_idx >= batch:
        raise IndexError(f"sample_idx={sample_idx} outside batch size {batch}")

    if lead_names is None:
        lead_names = _default_lead_names(n_leads)

    ecg = x[sample_idx].detach().cpu()
    attr = attribution[sample_idx].detach().cpu()

    if sampling_rate is None:
        t = torch.arange(n_time)
        xlabel = "Time index"
    else:
        t = torch.arange(n_time, dtype=torch.float32) / float(sampling_rate)
        xlabel = "Time (s)"

    figures = []

    for lead in range(n_leads):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        signal = ecg[lead]
        lead_attr = attr[lead]

        if lead_attr.max() > lead_attr.min():
            lead_attr = (lead_attr - lead_attr.min()) / (
                lead_attr.max() - lead_attr.min() + 1e-8
            )

        y_min = signal.min()
        y_max = signal.max()
        y_range = (y_max - y_min).clamp_min(1e-8)

        ax.plot(t, signal)
        ax.fill_between(
            t,
            y_min,
            y_min + lead_attr * y_range,
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
    lead_std: Optional[Tensor] = None,
    sample_idx: int = 0,
    lead_names: Optional[Sequence[str]] = None,
    normalize: bool = True,
    show: bool = True,
):
    """
    Plot lead-wise importance for one ECG.
    """
    import matplotlib.pyplot as plt

    summary = summarize_lead_importance(
        lead_importance=lead_importance,
        lead_std=lead_std,
        lead_names=lead_names,
        sample_idx=sample_idx,
        normalize=normalize,
    )

    leads = [item["lead"] for item in summary]
    values = [float(item["importance"]) for item in summary]

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)

    if lead_std is not None:
        errors = [float(item["std"]) for item in summary]
        ax.bar(leads, values, yerr=errors)
    else:
        ax.bar(leads, values)

    ax.set_xlabel("Lead")
    ax.set_ylabel("Importance")
    ax.set_title("Bayesian lead-wise ECG importance")
    fig.tight_layout()

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------
# End-to-end convenience function
# ---------------------------------------------------------------------


def explain_batch_with_bayesian_ig(
    model: torch.nn.Module,
    guide: Callable,
    x: Tensor,
    num_posterior_samples: int = 50,
    n_mc: Optional[int] = None,
    steps: int = 32,
    target_class: Optional[Tensor] = None,
    baseline: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """
    Convenience function: draw guide samples and compute Bayesian IG.

    Parameters
    ----------
    model:
        BayesianSmallCNN.
    guide:
        Trained Pyro guide.
    x:
        ECG batch.
    num_posterior_samples:
        Number of posterior samples to draw from the guide.
    n_mc:
        Number of samples to use for attribution. If ``None``, use all drawn.
    steps:
        Integrated-gradient steps.
    target_class:
        Optional target class.
    baseline:
        Optional baseline ECG.

    Returns
    -------
    result:
        Dictionary containing ``attr_mean``, ``attr_std``, ``stable_attr``,
        ``target_class``, and ``posterior_samples``.
    """
    model.eval()

    device = get_model_device(model)
    x = x.to(device)

    posterior_samples = draw_posterior_samples_from_guide(
        guide=guide,
        x=x,
        num_samples=num_posterior_samples,
    )

    if target_class is None:
        target_class = predict_target_class(
            model=model,
            posterior_samples=posterior_samples,
            x=x,
            use_probabilities=True,
            max_samples=n_mc,
        )

    attr_mean, attr_std = bayesian_integrated_gradients(
        model=model,
        posterior_samples=posterior_samples,
        x=x,
        target_class=target_class,
        baseline=baseline,
        steps=steps,
        n_mc=n_mc,
    )

    stable_attr = confidence_weighted_attribution(attr_mean, attr_std)

    return {
        "attr_mean": attr_mean,
        "attr_std": attr_std,
        "stable_attr": stable_attr,
        "target_class": target_class,
        "posterior_samples": posterior_samples,
    }


__all__ = [
    "DEFAULT_12_LEAD_NAMES",
    "aggregate_attribution_windows",
    "bayesian_integrated_gradients",
    "bayesian_lead_occlusion_importance",
    "bayesian_occlusion_sensitivity",
    "bayesian_vanilla_saliency",
    "conditioned_logits",
    "confidence_weighted_attribution",
    "draw_posterior_samples_from_guide",
    "explain_batch_with_bayesian_ig",
    "get_model_device",
    "get_num_posterior_samples",
    "get_target_scores",
    "normalize_attribution",
    "plot_ecg_with_attribution",
    "plot_lead_importance",
    "posterior_mean_logits",
    "posterior_mean_probs",
    "predict_target_class",
    "select_single_posterior_sample",
    "summarize_lead_importance",
    "top_ecg_regions",
    "validate_ecg_tensor",
]
