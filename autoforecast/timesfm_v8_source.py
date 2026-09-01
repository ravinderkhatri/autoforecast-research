"""PEFT LoRA for the pinned TimesFM3 network; original parameters remain frozen."""

from __future__ import annotations

import hashlib
import importlib.metadata

import numpy as np
import torch
from torch.overrides import TorchFunctionMode

from autoforecast.timesfm_hnb_source import SourceRequest, TimesFM3Source

LORA_RECIPE = {
    "rank": 8,
    "alpha": 16,
    "dropout": 0.05,
    "target_modules": ["query_proj", "key_proj", "value_proj", "out_proj"],
    "learning_rate": 0.0001,
    "peft_version": "0.19.1",
    "autograd_policy": "sqrt_zero_derivative_v1",
}


class _ZeroSafeSqrt(TorchFunctionMode):
    """Keep source sqrt values; choose a zero derivative at zero variance.

    Inference-only CPM normalization evaluates constant and unused channels too.
    Its sqrt(0) backward produces 0 * inf, contaminating LoRA gradients even when
    that channel's output is masked. This scoped dispatch avoids the singular
    operation without changing positive-variance derivatives or provider globals.
    """

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.sqrt and args[0].requires_grad:
            value = args[0]
            zero = value == 0
            return torch.where(
                zero,
                torch.zeros_like(value),
                torch.sqrt(torch.where(zero, torch.ones_like(value), value)),
            )
        return func(*args, **kwargs)


def state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class LoRASource(TimesFM3Source):
    """Train only maintained PEFT adapters, saving them separately from base weights."""

    def __init__(self, device: str = "cpu") -> None:
        super().__init__(device)
        self.enable_lora()

    def enable_lora(self) -> None:
        from peft import LoraConfig, inject_adapter_in_model

        if importlib.metadata.version("peft") != LORA_RECIPE["peft_version"]:
            raise ValueError("V8 requires pinned PEFT 0.19.1")
        targets = [
            name
            for name, layer in self.model.named_modules()
            if isinstance(layer, torch.nn.Linear)
            and name.rsplit(".", 1)[-1] in LORA_RECIPE["target_modules"]
        ]
        if not targets:
            raise ValueError("V8 found no supported TimesFM3 attention projections")
        self.model = inject_adapter_in_model(
            LoraConfig(
                r=LORA_RECIPE["rank"],
                lora_alpha=LORA_RECIPE["alpha"],
                lora_dropout=LORA_RECIPE["dropout"],
                target_modules=targets,
                bias="none",
            ),
            self.model,
        )
        trainable = [name for name, value in self.model.named_parameters() if value.requires_grad]
        if not trainable or any(".lora_" not in name for name in trainable):
            raise ValueError("V8 must train only LoRA parameters inside TimesFM3")
        self.identity = {
            **self.identity,
            "adaptation": dict(LORA_RECIPE),
            "adapted_modules": targets,
        }
        self.sealed = False

    def adapter_state(self) -> dict[str, torch.Tensor]:
        from peft import get_peft_model_state_dict

        return {
            name: value.detach().cpu().clone()
            for name, value in get_peft_model_state_dict(
                self.model, save_embedding_layers=False
            ).items()
        }

    def load_adapter(self, state: dict[str, torch.Tensor]) -> None:
        from peft import set_peft_model_state_dict

        expected = self.adapter_state()
        if state.keys() != expected.keys() or any(
            value.shape != expected[name].shape
            or value.dtype != expected[name].dtype
            or not torch.isfinite(value).all()
            for name, value in state.items()
        ):
            raise ValueError("V8 LoRA state differs from the recorded architecture")
        result = set_peft_model_state_dict(self.model, state)
        if result.unexpected_keys or any(".lora_" in name for name in result.missing_keys):
            raise ValueError("V8 did not load the complete LoRA adapter")
        self.seal()

    def seal(self) -> None:
        self.model.eval()
        self.identity = {**self.identity, "adapter_sha256": state_digest(self.adapter_state())}
        self.sealed = True

    def tensor_quantiles(self, request: SourceRequest) -> torch.Tensor:
        # Reuse the pinned decode BODY, including padding, masks, normalization and stitching.
        # Remove its no_grad decorator and define sqrt's derivative at zero variance.
        # Tests cover exact forward parity and finite gradients on constant/masked feeds.
        kwargs = {"horizon": request.horizon}
        names = {
            "target": "target",
            "target_mask": "target_mask",
            "past_only": "past_only_covariates",
            "past_only_mask": "past_only_mask",
            "past_future": "past_future_covariates",
            "past_future_mask": "past_future_mask",
        }
        for name, argument in names.items():
            value = getattr(request, name)
            if value is not None:
                kwargs[argument] = (
                    torch.from_numpy(value.copy()).unsqueeze(0).to(self.provider.device)
                )
        decode = type(self.model).decode.__wrapped__
        with _ZeroSafeSqrt():
            raw = decode(self.model, **kwargs)[0, : len(request.target)]
        return raw.sort(dim=-1).values

    def predict(self, request: SourceRequest) -> np.ndarray:
        if not self.sealed or self.model.training:
            raise ValueError("V8 inference requires a sealed, evaluated LoRA state")
        with torch.inference_mode():
            return self.tensor_quantiles(request).cpu().numpy()
