"""Reusable, bounded per-series weighting for probabilistic training objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

SERIES_LOSS_WEIGHTING_MODES = frozenset({"uniform", "revenue", "output_gradient"})
_GRADIENT_WEIGHT_MIN = 0.25
_GRADIENT_WEIGHT_MAX = 4.0


@dataclass(frozen=True)
class SeriesLossWeights:
    """Detached batch weights plus compact diagnostics for training logs."""

    values: Tensor
    mode: str

    def diagnostics(self) -> dict[str, float | str]:
        positive = self.values[self.values > 0]
        return {
            "series_loss_weighting": self.mode,
            "series_weight_min": float(positive.min()) if positive.numel() else 0.0,
            "series_weight_mean": float(positive.mean()) if positive.numel() else 0.0,
            "series_weight_max": float(positive.max()) if positive.numel() else 0.0,
        }


def series_loss_weights(
    mode: str,
    per_series_loss: Tensor,
    *,
    revenue_weight: Tensor | None = None,
    distribution_outputs: tuple[Tensor, ...] = (),
) -> SeriesLossWeights:
    """Return one detached weight per batch row without populating parameter gradients.

    ``output_gradient`` uses the loss gradient with respect to distribution outputs. It is a
    scalable influence proxy, not the full per-series model-parameter gradient norm.
    """
    if mode not in SERIES_LOSS_WEIGHTING_MODES:
        raise ValueError(f"unsupported series loss weighting mode {mode!r}")
    if per_series_loss.ndim != 1 or not per_series_loss.is_floating_point():
        raise ValueError("per-series loss must be one floating value per batch row")
    if not bool(torch.isfinite(per_series_loss).all()) or bool((per_series_loss < 0).any()):
        raise ValueError("per-series loss must be finite and nonnegative")
    informative = per_series_loss.detach() > 0
    if mode == "uniform":
        values = torch.ones_like(per_series_loss)
    elif mode == "revenue":
        if revenue_weight is None or revenue_weight.shape != per_series_loss.shape:
            raise ValueError("revenue weighting requires one pre-origin weight per batch row")
        if (
            not revenue_weight.is_floating_point()
            or not bool(torch.isfinite(revenue_weight).all())
            or bool((revenue_weight < 0).any())
        ):
            raise ValueError("revenue weights must be finite nonnegative floating values")
        values = revenue_weight.detach().to(per_series_loss)
    else:
        if not bool(informative.any()):
            raise ValueError("output gradients are undefined without an informative row")
        outputs = tuple(value for value in distribution_outputs if value.requires_grad)
        if not outputs or any(value.shape[0] != per_series_loss.shape[0] for value in outputs):
            raise ValueError("output-gradient weighting requires aligned differentiable outputs")
        gradients = torch.autograd.grad(
            per_series_loss.sum(),
            outputs,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        squared = torch.zeros_like(per_series_loss)
        for gradient in gradients:
            if gradient is None:
                continue
            squared = squared + gradient.detach().square().reshape(len(per_series_loss), -1).sum(1)
        values = squared.sqrt()
        positive = values[informative]
        if (
            not positive.numel()
            or not bool(torch.isfinite(positive).all())
            or not bool((positive > 0).any())
        ):
            raise ValueError("output gradients are undefined or zero for every informative row")
        values = values / positive.mean()
        clamped = values.clamp(_GRADIENT_WEIGHT_MIN, _GRADIENT_WEIGHT_MAX)
        clamped_mean = clamped[informative].mean()
        if bool(clamped_mean < 1):
            values = clamped + (1 - clamped_mean) * (
                _GRADIENT_WEIGHT_MAX - clamped
            ) / (_GRADIENT_WEIGHT_MAX - clamped_mean)
        elif bool(clamped_mean > 1):
            values = _GRADIENT_WEIGHT_MIN + (1 - _GRADIENT_WEIGHT_MIN) * (
                clamped - _GRADIENT_WEIGHT_MIN
            ) / (clamped_mean - _GRADIENT_WEIGHT_MIN)
        else:
            values = clamped
        values = torch.where(
            informative,
            values,
            torch.zeros_like(values),
        )
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("series loss weights must be finite and nonnegative")
    if informative.any() and not bool((values[informative] > 0).any()):
        raise ValueError("series loss weights remove every informative row")
    return SeriesLossWeights(values.detach(), mode)


def weighted_series_mean(
    per_series_loss: Tensor,
    denominator: float | Tensor,
    weights: SeriesLossWeights,
) -> Tensor:
    """Reduce per-series contributions using the caller's governed denominator."""
    if weights.values.shape != per_series_loss.shape:
        raise ValueError("series loss and weights must have the same batch shape")
    return (per_series_loss * weights.values).sum() / denominator
