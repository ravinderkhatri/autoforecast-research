"""Lightning TFT AMP hooks that preserve one successful Adam update per batch."""

from __future__ import annotations

import math


def fp32_count_output_head(model):
    """Keep the exact HMFNet count loss fed by an FP32 TFT decoder projection."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    class FP32Linear(nn.Linear):
        def forward(self, inputs):
            with torch.autocast(inputs.device.type, enabled=False):
                return F.linear(inputs.to(self.weight.dtype), self.weight, self.bias)

    old = model.output_layer
    if not isinstance(old, nn.Linear):
        raise TypeError("HMFNet count TFT output head must be linear")
    head = FP32Linear(
        old.in_features, old.out_features, bias=old.bias is not None,
        device=old.weight.device, dtype=old.weight.dtype,
    )
    head.load_state_dict(old.state_dict())
    model.output_layer = head
    return model


def mixed_precision_tft_class(base_class, counter: dict[str, int], device: str):
    """Retry the same stochastic batch after GradScaler backs off an overflowing scale."""
    import torch

    class MixedPrecisionTFT(base_class):
        def log(self, *args, **kwargs):
            if getattr(self, "_defer_training_logs", False):
                self._pending_training_logs.append(("log", args, kwargs))
                return None
            return super().log(*args, **kwargs)

        def log_dict(self, *args, **kwargs):
            if getattr(self, "_defer_training_logs", False):
                self._pending_training_logs.append(("log_dict", args, kwargs))
                return None
            return super().log_dict(*args, **kwargs)

        def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
            for _ in range(24):
                cpu_rng = torch.get_rng_state()
                accelerator_rng = (
                    torch.mps.get_rng_state() if device == "mps"
                    else torch.cuda.get_rng_state() if device == "cuda" else None
                )
                self._pending_training_logs = []
                self._defer_training_logs = True
                raw_optimizer = getattr(optimizer, "optimizer", optimizer)
                plugin = getattr(getattr(self, "trainer", None), "precision_plugin", None)
                attempt_scaler = getattr(plugin, "scaler", None)
                if hasattr(self, "trainer") and attempt_scaler is None:
                    raise RuntimeError("TFT 16-mixed requires an active GradScaler")
                before_scale = attempt_scaler.get_scale() if attempt_scaler is not None else None
                if before_scale is not None and (
                    not math.isfinite(before_scale) or before_scale <= 0
                ):
                    raise FloatingPointError("TFT AMP has an invalid loss scale")
                calls = {"count": 0}
                handle = raw_optimizer.register_step_post_hook(
                    lambda _optimizer, _args, _kwargs, calls=calls: calls.__setitem__(
                        "count", calls["count"] + 1
                    )
                )
                try:
                    result = super().optimizer_step(
                        epoch, batch_idx, optimizer, optimizer_closure
                    )
                finally:
                    handle.remove()
                    self._defer_training_logs = False
                if calls["count"] not in (0, 1):
                    raise RuntimeError("TFT completed multiple optimizer updates for one batch")
                if calls["count"]:
                    counter["count"] += 1
                    for method, args, kwargs in self._pending_training_logs:
                        getattr(super(), method)(*args, **kwargs)
                    self._pending_training_logs = []
                    return result
                if attempt_scaler is not None:
                    after_scale = attempt_scaler.get_scale()
                    if after_scale >= before_scale:
                        # A skipped step is authoritative even when visible gradients are
                        # finite: MPS/GradScaler may retain a nonfinite found_inf flag from
                        # unscale. Back off and replay this exact batch; after 24 skips fail.
                        attempt_scaler.update(new_scale=before_scale * 0.5)
                counter["retries"] += 1
                self.training_step_outputs.pop()
                self._pending_training_logs = []
                torch.set_rng_state(cpu_rng)
                if device == "mps":
                    torch.mps.set_rng_state(accelerator_rng)
                elif device == "cuda":
                    torch.cuda.set_rng_state(accelerator_rng)
            raise FloatingPointError("TFT AMP could not complete an optimizer update in 24 tries")

    return MixedPrecisionTFT


def finite_gradient_clip_callback(*, precision: str, max_norm: float = 1.0):
    """Let GradScaler recover component overflow, but reject a bad finite-component norm."""
    from lightning.pytorch import Callback

    from autoforecast.amp_gradients import clip_gradients

    class FiniteGradientClip(Callback):
        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            clip_gradients(
                pl_module.parameters(), max_norm, allow_amp_overflow=precision == "16-mixed"
            )

    return FiniteGradientClip()
