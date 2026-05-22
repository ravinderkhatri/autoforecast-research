"""The future-conditioned decoder: encoder representation plus what is KNOWN about the horizon.

FUTURE-CONDITIONED, and the distinction is the one that decides whether a forecast is honest.
Some things about the horizon are known at forecast time -- the calendar, a planned price, a
scheduled promotion -- and some are not, above all the demand itself. A decoder that attends over
the encoder AND over known-future covariates uses the first without touching the second.

WHAT MAY ENTER, stated because getting it wrong is invisible in training and fatal in production:

  · KNOWN: day of week, holiday distance, planned price, planned promotion, planned display.
  · NOT KNOWN: demand, a competitor's price, a promotion nobody has announced yet.

The second list is why `event_known_at` exists in the spec. Without it a model trained on
historical promotions learns to FORESEE unannounced ones, scores beautifully in backtest, and
cannot do it live.

CROSS-ATTENTION ONLY TO THE ENCODER, never from the encoder back. The horizon must not inform the
representation of the past.
"""

from __future__ import annotations

import torch
from forecast.models.mft.trunk import RMSNorm
from torch import Tensor, nn

__all__ = ["Decoder", "DecoderError"]


class DecoderError(ValueError):
    """The decoder was given a horizon or a covariate set it cannot use."""


class _DecoderLayer(nn.Module):
    """Self-attention over the horizon, then cross-attention into the encoder."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4) -> None:
        super().__init__()
        self.norm_self = RMSNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_cross = RMSNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_ff = RMSNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model), nn.GELU(), nn.Linear(ff_mult * d_model, d_model)
        )

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        h = self.norm_self(x)
        attended, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + attended
        h = self.norm_cross(x)
        crossed, _ = self.cross_attn(
            h,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        x = x + crossed
        return x + self.ff(self.norm_ff(x))


class Decoder(nn.Module):
    """Two layers, conditioned on known-future covariates.

    No causal mask over the horizon, deliberately. This is a DIRECT multi-horizon decoder: every
    step is produced from the encoder and the known future, not from the step before it. Masking
    would impose an autoregressive order the model does not use, and an autoregressive decoder
    would consume its own predictions -- which is how a leak-free-looking pipeline starts training
    on its own output.
    """

    def __init__(
        self,
        d_model: int,
        n_known: int,
        *,
        n_layers: int = 2,
        n_heads: int = 4,
        max_horizon: int = 60,
    ) -> None:
        super().__init__()
        self.max_horizon = max_horizon
        self.known = nn.Linear(n_known, d_model)
        self.step_id = nn.Parameter(torch.randn(1, max_horizon, d_model) * 0.02)
        self.layers = nn.ModuleList(_DecoderLayer(d_model, n_heads) for _ in range(n_layers))
        self.out = RMSNorm(d_model)

    def forward(
        self,
        memory: Tensor,
        known_future: Tensor,
        memory_mask: Tensor | None = None,
    ) -> Tensor:
        """Args:
            memory: `[batch, n_positions, d_model]` -- the trunk's output, flattened over
                branches and encoder time. The decoder attends into it; nothing flows back.
            known_future: `[batch, horizon, n_known]` covariates KNOWN at forecast time. Passing
                anything here that is not known live is the leak this module's docstring exists
                to prevent, and no runtime check can detect it -- the tensor looks identical.
            memory_mask: Optional `[batch, n_positions]`, True only for real encoder positions.

        Returns:
            `[batch, horizon, d_model]`, one representation per horizon step, ready for the head.

        Raises:
            DecoderError: the horizon exceeds `max_horizon`.
        """
        b, h, _ = known_future.shape
        if h > self.max_horizon:
            raise DecoderError(
                f"horizon {h} exceeds max_horizon={self.max_horizon}; the step embeddings past "
                "that point were never trained, so those days would be produced from noise"
            )
        key_padding_mask = None
        if memory_mask is not None:
            if (
                not isinstance(memory_mask, Tensor)
                or memory_mask.shape != memory.shape[:2]
                or memory_mask.device != memory.device
            ):
                raise DecoderError("memory_mask must match [batch, n_positions]")
            if memory_mask.dtype != torch.bool:
                if memory_mask.is_complex() or not bool(torch.isfinite(memory_mask).all()) or bool(
                    ((memory_mask != 0) & (memory_mask != 1)).any()
                ):
                    raise DecoderError("memory_mask must contain exact binary values")
                memory_mask = memory_mask.bool()
            if bool((~memory_mask).all(dim=1).any()):
                raise DecoderError("each row needs at least one available memory position")
            key_padding_mask = ~memory_mask
        x = self.known(known_future) + self.step_id[:, :h, :]
        for layer in self.layers:
            x = layer(x, memory, key_padding_mask)
        return self.out(x)
