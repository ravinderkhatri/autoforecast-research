"""HMFNet's active count objective on the shared TFT-Q/TFT-HNB TFT decoder output."""

from __future__ import annotations

import pandas as pd
import torch
from forecast.models.mft.sft import (
    SFT_LOG_K_MAX,
    SFT_LOG_K_MIN,
    ShiftedHurdlePrediction,
)
from pytorch_forecasting.data.encoders import TorchNormalizer
from pytorch_forecasting.metrics.base_metrics import MultiHorizonMetric


class RawCountNormalizer(TorchNormalizer):
    """Identity scale that also accepts PyTorch-Forecasting's one-column lag frame."""

    def __init__(self) -> None:
        super().__init__(method="identity")

    def transform(self, y, *args, **kwargs):
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError("raw count normalizer expects one lagged target column")
            y = y.iloc[:, 0]
        return super().transform(y, *args, **kwargs)


class HMFNetHurdleLoss(MultiHorizonMetric):
    """HMFNet's shifted-hurdle NB NLL plus its active 0.2 mean-error term.

    The three TFT channels are occurrence logit, log shifted-NB excess, and log dispersion.
    HMFNet's configured expert and event terms have no active contribution in its M5 default.
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _decode(output: torch.Tensor) -> ShiftedHurdlePrediction:
        if output.ndim != 3 or output.shape[-1] != 3:
            raise ValueError("HMFNet hurdle TFT output must have three channels per forecast cell")
        # HMFNet evaluates this law in float64. MPS has no float64 tensors, so the
        # decoder channels and target/scale cross to CPU; autograd returns gradients.
        output = output.to(device="cpu").double()
        return ShiftedHurdlePrediction(
            logit=output[..., 0],
            excess=output[..., 1].double().exp(),
            dispersion=output[..., 2].double().clamp(SFT_LOG_K_MIN, SFT_LOG_K_MAX).exp(),
        )

    def loss(
        self, y_pred: torch.Tensor, target: torch.Tensor, mean_scale: torch.Tensor
    ) -> torch.Tensor:
        prediction = self._decode(y_pred)
        if mean_scale.shape != target.shape:
            raise ValueError("HMFNet mean scale must match the target cells")
        actual = target.to(device="cpu").double()
        scale = mean_scale.to(device="cpu").double()
        nll = prediction.nll(actual, torch.ones_like(actual, dtype=torch.bool))
        # HMFNet validates the series-reduced likelihood, not each rounded cell.
        series_nll = nll.mean(dim=1)
        if (not torch.isfinite(nll).all()
                or not torch.isfinite(series_nll).all()
                or (series_nll < 0).any()):
            raise FloatingPointError("HMFNet hurdle NLL is negative or nonfinite")
        mean_error = (prediction.mean() - actual).square()
        return nll + 0.2 * scale * mean_error

    def forward(self, y_pred: torch.Tensor, target) -> torch.Tensor:
        """Reduce on CPU in float64, bypassing PF's nonfinite-to-1e9 fallback."""
        if not isinstance(target, tuple | list) or len(target) != 2:
            raise ValueError("HMFNet hurdle loss requires target and origin-safe mean scale")
        actual, mean_scale = target
        losses = self.loss(y_pred, actual, mean_scale)
        if not torch.isfinite(losses).all():
            raise FloatingPointError("HMFNet hurdle loss has a nonfinite forecast cell")
        # Dividing before summing prevents an otherwise finite mean from
        # overflowing float64 for a large batch. Keep the ordinary reduction
        # for tiny subnormal losses, where early division could underflow.
        threshold = torch.finfo(losses.dtype).max / losses.numel()
        result = (
            (losses / losses.numel()).sum()
            if losses.max() > threshold
            else losses.mean()
        )
        if not torch.isfinite(result):
            raise FloatingPointError("HMFNet hurdle loss reduction is nonfinite")
        return result

    def update(self, y_pred, target) -> None:
        raise RuntimeError("HMFNet hurdle loss is reduced directly by forward()")

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        return self._decode(y_pred).mean()
