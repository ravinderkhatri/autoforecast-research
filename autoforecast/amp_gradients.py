"""Gradient-norm handling shared by the trainable PyTorch adapters."""

from __future__ import annotations

import math
from collections.abc import Iterable


def clip_gradients(
    parameters: Iterable, max_norm: float, *, allow_amp_overflow: bool
):
    """Clip with a scale-first norm; leave genuine component overflow for GradScaler."""
    import torch

    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("gradient clip max_norm must be finite and positive")
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return torch.tensor(0.0, dtype=torch.float64), False
    nonempty = [gradient for gradient in gradients if gradient.numel()]
    if not nonempty:
        return torch.tensor(0.0, dtype=torch.float64), False
    # Squaring raw FP16/FP32 components before reduction can overflow while every component
    # is finite. Normalize by the largest component first, accumulate in at least FP32,
    # then form the clipping coefficient in Python float64 without materializing a large norm
    # on MPS (which has no float64 tensor dtype).
    work_dtype = (
        torch.float64 if any(gradient.dtype == torch.float64 for gradient in nonempty)
        else torch.float32
    )
    maximum_tensor = torch.stack([
        gradient.detach().abs().amax().to(work_dtype) for gradient in nonempty
    ]).amax()
    maximum = float(maximum_tensor)
    if not math.isfinite(maximum):
        if allow_amp_overflow:
            return torch.tensor(math.inf, dtype=torch.float64), True
        raise FloatingPointError("gradient norm is nonfinite due to nonfinite components")
    if maximum == 0:
        return torch.tensor(0.0, dtype=torch.float64), False
    squared = torch.zeros((), dtype=work_dtype, device=nonempty[0].device)
    for gradient in nonempty:
        scaled = gradient.detach().to(work_dtype) / maximum_tensor
        squared = squared + scaled.square().sum()
    norm = maximum * math.sqrt(float(squared))
    if not math.isfinite(norm):
        raise FloatingPointError("gradient norm is nonfinite with finite components")
    coefficient = min(1.0, max_norm / norm)
    if coefficient < 1.0:
        for gradient in nonempty:
            gradient.mul_(coefficient)
    return torch.tensor(norm, dtype=torch.float64), False
