"""PACT decoder-only forecasting model and coherent segment allocation law."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from forecast.models.mft.heads import (
    LOG_K_MAX,
    LOG_K_MIN,
    hurdle_mean,
    log1mexp,
    nb_log_pmf,
    ztnb_log_pmf,
)
from forecast.models.mft.sample import sample_paths, sample_paths_negbin
from torch import Tensor, nn

PACT_LAWS = frozenset({"negbin", "zero_truncated_hurdle_negbin"})


def _require_bool(name: str, value: Tensor, shape: tuple[int, ...]) -> None:
    if not isinstance(value, Tensor) or value.dtype != torch.bool or value.shape != shape:
        raise ValueError(f"{name} must be bool tensor {shape}, got {getattr(value, 'shape', None)}")


def _require_int64(name: str, value: Tensor, shape: tuple[int, ...]) -> None:
    if not isinstance(value, Tensor) or value.dtype != torch.int64 or value.shape != shape:
        raise ValueError(
            f"{name} must be int64 tensor {shape}, got {getattr(value, 'shape', None)}"
        )


def prefix_attention_mask(
    token_exists: Tensor, history_length: int
) -> tuple[Tensor, Tensor]:
    """Return the PACT allowed relation and invalid-query rows made safe for SDPA."""
    if (
        not isinstance(token_exists, Tensor)
        or token_exists.dtype != torch.bool
        or token_exists.ndim != 2
        or type(history_length) is not int
        or not 1 <= history_length < token_exists.shape[1]
    ):
        raise ValueError("prefix mask needs bool [batch,tokens] existence and both token regions")
    batch, tokens = token_exists.shape
    index = torch.arange(tokens, device=token_exists.device)
    history = index < history_length
    future = ~history
    relation = (history[:, None] & history[None, :] & (index[None, :] <= index[:, None])) | (
        future[:, None] & (history[None, :] | future[None, :])
    )
    allowed = (
        token_exists[:, :, None]
        & token_exists[:, None, :]
        & relation.unsqueeze(0).expand(batch, -1, -1)
    )
    invalid_query = ~token_exists
    safe = allowed.clone()
    rows = torch.nonzero(invalid_query, as_tuple=False)
    if rows.numel():
        safe[rows[:, 0], rows[:, 1], 0] = True
    return allowed, safe


def full_attention_mask(token_exists: Tensor) -> tuple[Tensor, Tensor]:
    """Full valid-token relation with one harmless key for every invalid query row."""
    if (
        not isinstance(token_exists, Tensor)
        or token_exists.dtype != torch.bool
        or token_exists.ndim != 2
    ):
        raise ValueError("full attention mask needs a bool [batch,tokens] tensor")
    allowed = token_exists[:, :, None] & token_exists[:, None, :]
    safe = allowed.clone()
    rows = torch.nonzero(~token_exists, as_tuple=False)
    if rows.numel():
        safe[rows[:, 0], rows[:, 1], 0] = True
    return allowed, safe


def _validate_day_mapping(active: Tensor, index: Tensor, horizon: int) -> None:
    """Require each forecast day once and one or more active days in every segment."""
    if type(horizon) is not int or horizon < 1:
        raise ValueError("PACT horizon must be a positive built-in integer")
    if bool((~active.any(dim=-1)).any()) or bool((index.masked_select(~active) != -1).any()):
        raise ValueError("PACT day mapping needs active segments and canonical inactive slots")
    days = index.masked_select(active)
    if days.numel() != horizon or not torch.equal(
        torch.sort(days).values, torch.arange(horizon, device=days.device)
    ):
        raise ValueError("PACT active day indices must be an exact horizon permutation")


@dataclass(frozen=True)
class PACTModelInputs:
    """Typed, target-free tensors accepted by the PACT decoder."""

    history_features: Tensor
    history_exists: Tensor
    history_positions: Tensor
    history_types: Tensor
    future_features: Tensor
    future_exists: Tensor
    future_positions: Tensor
    future_types: Tensor
    future_active: Tensor
    future_day_index: Tensor
    statistics: Tensor
    attributes: Tensor
    identity_indices: Tensor
    horizon: int

    def __post_init__(self) -> None:
        if not isinstance(self.history_features, Tensor) or self.history_features.ndim != 4:
            raise ValueError("typed PACT history must be [batch,streams,tokens,features]")
        batch, streams, history, _width = self.history_features.shape
        _require_bool(
            "typed PACT history existence", self.history_exists, (batch, streams, history)
        )
        _require_int64("typed PACT history positions", self.history_positions, (history,))
        _require_int64("typed PACT history types", self.history_types, (history,))
        if not isinstance(self.future_features, Tensor) or (
            self.future_features.ndim != 3 or self.future_features.shape[0] != batch
        ):
            raise ValueError("typed PACT future must be [batch,segments,features]")
        segments = self.future_features.shape[1]
        _require_bool("typed PACT future existence", self.future_exists, (batch, segments))
        _require_int64("typed PACT future positions", self.future_positions, (segments,))
        _require_int64("typed PACT future types", self.future_types, (segments,))
        _require_bool("typed PACT active-day mask", self.future_active, (segments, 7))
        _require_int64("typed PACT day index", self.future_day_index, (segments, 7))
        if not isinstance(self.statistics, Tensor) or (
            self.statistics.ndim != 2 or self.statistics.shape[0] != batch
        ):
            raise ValueError("typed PACT statistics must be [batch,features]")
        if not isinstance(self.attributes, Tensor) or (
            self.attributes.ndim != 2 or self.attributes.shape[0] != batch
        ):
            raise ValueError("typed PACT attributes must be [batch,features]")
        if not isinstance(self.identity_indices, Tensor) or (
            self.identity_indices.ndim != 2 or self.identity_indices.shape[0] != batch
        ):
            raise ValueError("typed PACT identities must be [batch,tables]")
        if self.identity_indices.dtype != torch.int64:
            raise ValueError("typed PACT identity indices must be int64")
        if type(self.horizon) is not int or self.horizon < 1:
            raise ValueError("typed PACT horizon must be a positive built-in integer")
        _validate_day_mapping(self.future_active, self.future_day_index, self.horizon)
        numeric = (
            self.history_features,
            self.future_features,
            self.statistics,
            self.attributes,
        )
        if any(
            not value.is_floating_point() or not bool(torch.isfinite(value).all())
            for value in numeric
        ):
            raise ValueError("typed PACT numeric inputs must be finite floating tensors")
        tensors = tuple(value for value in self.as_kwargs().values() if isinstance(value, Tensor))
        if len({value.device for value in tensors}) != 1:
            raise ValueError("typed PACT model inputs must be on one device")

    def as_kwargs(self) -> dict[str, Tensor | int]:
        return {
            "history_features": self.history_features,
            "history_exists": self.history_exists,
            "history_positions": self.history_positions,
            "history_types": self.history_types,
            "future_features": self.future_features,
            "future_exists": self.future_exists,
            "future_positions": self.future_positions,
            "future_types": self.future_types,
            "future_active": self.future_active,
            "future_day_index": self.future_day_index,
            "statistics": self.statistics,
            "attributes": self.attributes,
            "identity_indices": self.identity_indices,
            "horizon": self.horizon,
        }

    def to_device(self, device: str | torch.device) -> PACTModelInputs:
        moved = {
            name: value.to(device) if isinstance(value, Tensor) else value
            for name, value in self.as_kwargs().items()
        }
        return replace(self, **moved)


@dataclass(frozen=True)
class PACTLabels:
    """PACT supervision kept structurally separate from target-free model inputs."""

    daily_target: Tensor
    objective_day_mask: Tensor
    target_counts: Tensor
    segment_eligible: Tensor
    series_loss_weight: Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.daily_target, Tensor) or self.daily_target.ndim != 2:
            raise ValueError("PACT daily labels must be [batch,horizon]")
        batch, horizon = self.daily_target.shape
        _require_bool("PACT objective-day mask", self.objective_day_mask, (batch, horizon))
        if not isinstance(self.target_counts, Tensor) or (
            self.target_counts.ndim != 3 or self.target_counts.shape[0] != batch
        ):
            raise ValueError("PACT segment labels must be [batch,segments,7]")
        if self.target_counts.shape[-1] != 7:
            raise ValueError("PACT segment labels must contain seven weekday slots")
        _require_bool(
            "PACT segment eligibility", self.segment_eligible, self.target_counts.shape[:2]
        )
        if self.series_loss_weight is not None and (
            self.series_loss_weight.shape != (batch,)
            or not self.series_loss_weight.is_floating_point()
            or not bool(torch.isfinite(self.series_loss_weight).all())
            or bool((self.series_loss_weight < 0).any())
        ):
            raise ValueError("PACT series loss weights must be finite nonnegative [batch]")
        for value in (self.daily_target, self.target_counts):
            if (
                not value.is_floating_point()
                or not bool(torch.isfinite(value).all())
                or bool((value < 0).any())
                or bool((value != torch.floor(value)).any())
            ):
                raise ValueError("PACT labels must be finite nonnegative integer-valued tensors")
        if len(
            {
                value.device
                for value in (
                    self.daily_target,
                    self.objective_day_mask,
                    self.target_counts,
                    self.segment_eligible,
                    *(() if self.series_loss_weight is None else (self.series_loss_weight,)),
                )
            }
        ) != 1:
            raise ValueError("PACT labels must be on one device")

    def to_device(self, device: str | torch.device) -> PACTLabels:
        return replace(
            self,
            daily_target=self.daily_target.to(device),
            objective_day_mask=self.objective_day_mask.to(device),
            target_counts=self.target_counts.to(device),
            segment_eligible=self.segment_eligible.to(device),
            series_loss_weight=(
                self.series_loss_weight.to(device)
                if self.series_loss_weight is not None
                else None
            ),
        )


@dataclass(frozen=True)
class PACTBatch:
    """One typed PACT handoff from deterministic assembly to model/loss code."""

    inputs: PACTModelInputs
    known_future: Tensor
    labels: PACTLabels | None = None

    def __post_init__(self) -> None:
        expected = (self.inputs.history_features.shape[0], self.inputs.horizon)
        if not isinstance(self.known_future, Tensor) or (
            self.known_future.ndim != 3 or self.known_future.shape[:2] != expected
        ):
            raise ValueError("PACT known-future tensor must be [batch,horizon,features]")
        if not self.known_future.is_floating_point() or not bool(
            torch.isfinite(self.known_future).all()
        ):
            raise ValueError("PACT known-future values must be finite floating tensors")
        if self.labels is not None and self.labels.daily_target.shape != expected:
            raise ValueError("PACT batch labels disagree with its batch/horizon")
        if self.labels is not None and self.labels.target_counts.shape != (
            expected[0],
            self.inputs.future_features.shape[1],
            7,
        ):
            raise ValueError("PACT batch labels disagree with its future segments")
        devices = {self.inputs.history_features.device, self.known_future.device}
        if self.labels is not None:
            devices.add(self.labels.daily_target.device)
        if len(devices) != 1:
            raise ValueError("PACT batch inputs, known future and labels must share one device")

    def to_device(self, device: str | torch.device) -> PACTBatch:
        return replace(
            self,
            inputs=self.inputs.to_device(device),
            known_future=self.known_future.to(device),
            labels=self.labels.to_device(device) if self.labels is not None else None,
        )


def apply_signed_rope(value: Tensor, positions: Tensor, theta: float) -> Tensor:
    """Apply adjacent-pair RoPE directly at signed real-day coordinates."""
    if value.ndim != 4:
        raise ValueError("RoPE value must be [batch,heads,tokens,head_dim]")
    if value.shape[-1] % 2:
        raise ValueError("RoPE head dimension must be even")
    if positions.shape != (value.shape[0], value.shape[2]):
        raise ValueError("RoPE positions must be [batch,tokens]")
    if not positions.is_floating_point() and positions.dtype not in (
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise ValueError("RoPE positions must be real numeric coordinates")
    if not math.isfinite(theta) or theta <= 0:
        raise ValueError("RoPE theta must be finite and positive")
    half = value.shape[-1] // 2
    frequency = theta ** (
        -torch.arange(half, device=value.device, dtype=value.dtype) * 2 / value.shape[-1]
    )
    angle = positions.to(value.dtype)[:, None, :, None] * frequency[None, None, None, :]
    cosine, sine = torch.cos(angle), torch.sin(angle)
    even, odd = value[..., 0::2], value[..., 1::2]
    rotated = torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1)
    return rotated.flatten(-2)


class MaskedSelfAttention(nn.Module):
    """Self-attention with explicit allowed edges and optional signed-coordinate RoPE."""

    def __init__(self, d_model: int, n_heads: int, *, rope_theta: float | None) -> None:
        super().__init__()
        if d_model % n_heads or (d_model // n_heads) % 2:
            raise ValueError("PACT attention needs d_model divisible into even head dimensions")
        self.d_model, self.n_heads = d_model, n_heads
        self.head_dim = d_model // n_heads
        self.rope_theta = rope_theta
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(
        self,
        value: Tensor,
        exists: Tensor,
        *,
        history_length: int | None = None,
        positions: Tensor | None = None,
    ) -> Tensor:
        batch, tokens, width = value.shape
        _require_bool("attention existence", exists, (batch, tokens))
        if width != self.d_model:
            raise ValueError("attention hidden width differs from its fingerprinted model width")
        if history_length is None:
            _allowed, safe = full_attention_mask(exists)
        else:
            _allowed, safe = prefix_attention_mask(exists, history_length)
        qkv = self.qkv(value).reshape(batch, tokens, 3, self.n_heads, self.head_dim)
        query, key, payload = qkv.unbind(2)
        query, key, payload = (item.transpose(1, 2) for item in (query, key, payload))
        query = query * torch.rsqrt(query.square().mean(-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt(key.square().mean(-1, keepdim=True) + 1e-6)
        if self.rope_theta is not None:
            if positions is None:
                raise ValueError("temporal attention requires fingerprinted token coordinates")
            query = apply_signed_rope(query, positions, self.rope_theta)
            key = apply_signed_rope(key, positions, self.rope_theta)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            payload,
            attn_mask=safe[:, None, :, :],
            dropout_p=0.0,
            scale=1.0 / math.sqrt(self.head_dim),
        )
        result = self.output(attended.transpose(1, 2).reshape(batch, tokens, width))
        return result * exists[..., None]


class GatedFeedForward(nn.Module):
    def __init__(self, d_model: int, multiplier: int) -> None:
        super().__init__()
        hidden = d_model * multiplier
        self.input = nn.Linear(d_model, 2 * hidden)
        self.output = nn.Linear(hidden, d_model)

    def forward(self, value: Tensor) -> Tensor:
        content, gate = self.input(value).chunk(2, dim=-1)
        return self.output(content * F.silu(gate))


class PACTDecoderBlock(nn.Module):
    """Temporal attention, then historical stream attention, then gated FFN."""

    def __init__(
        self, d_model: int, n_heads: int, ff_multiplier: int, *, rope_theta: float
    ) -> None:
        super().__init__()
        self.temporal_norm = nn.LayerNorm(d_model)
        self.temporal = MaskedSelfAttention(d_model, n_heads, rope_theta=rope_theta)
        self.stream_norm = nn.LayerNorm(d_model)
        self.stream = MaskedSelfAttention(d_model, n_heads, rope_theta=None)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = GatedFeedForward(d_model, ff_multiplier)

    def forward(
        self,
        value: Tensor,
        exists: Tensor,
        positions: Tensor,
        *,
        history_length: int,
    ) -> Tensor:
        batch, streams, tokens, width = value.shape
        temporal_in = self.temporal_norm(value).reshape(batch * streams, tokens, width)
        temporal_exists = exists.reshape(batch * streams, tokens)
        temporal_positions = positions.reshape(batch * streams, tokens)
        temporal_out = self.temporal(
            temporal_in,
            temporal_exists,
            history_length=history_length,
            positions=temporal_positions,
        ).reshape(batch, streams, tokens, width)
        value = (value + temporal_out) * exists[..., None]
        if streams > 1:
            stream_in = self.stream_norm(value).permute(0, 2, 1, 3).reshape(
                batch * tokens, streams, width
            )
            stream_exists = exists.permute(0, 2, 1).reshape(batch * tokens, streams)
            stream_out = self.stream(stream_in, stream_exists).reshape(
                batch, tokens, streams, width
            ).permute(0, 2, 1, 3)
            historical = torch.arange(tokens, device=value.device) < history_length
            stream_out = stream_out * historical[None, None, :, None]
            value = (value + stream_out) * exists[..., None]
        value = (value + self.ff(self.ff_norm(value))) * exists[..., None]
        return value


@dataclass(frozen=True)
class PACTPrediction:
    """One coherent segment-total and daily-allocation distribution."""

    law: str
    p_positive: Tensor | None
    mu: Tensor
    k: Tensor
    allocation: Tensor
    allocation_log: Tensor
    active: Tensor
    day_index: Tensor
    horizon: int

    def __post_init__(self) -> None:
        if self.law not in PACT_LAWS:
            raise ValueError(f"unknown PACT law {self.law!r}")
        if self.mu.shape != self.k.shape or self.mu.ndim != 2:
            raise ValueError("PACT total parameters must be aligned [batch,segments]")
        if (
            self.allocation.shape != (*self.mu.shape, 7)
            or self.allocation_log.shape != self.allocation.shape
        ):
            raise ValueError("PACT allocation must have seven slots per segment")
        if self.active.shape != (self.mu.shape[1], 7) or self.active.dtype != torch.bool:
            raise ValueError("PACT active-day mask must be bool [segments,7]")
        if self.day_index.shape != self.active.shape or self.day_index.dtype != torch.int64:
            raise ValueError("PACT day index must be int64 [segments,7]")
        _validate_day_mapping(self.active, self.day_index, self.horizon)
        if self.law == "zero_truncated_hurdle_negbin":
            if self.p_positive is None or self.p_positive.shape != self.mu.shape:
                raise ValueError("PACT hurdle law requires aligned occurrence probabilities")
        elif self.p_positive is not None:
            raise ValueError("plain PACT NB must not carry a hurdle probability")
        tensors = (self.mu, self.k, self.allocation, self.allocation_log)
        if self.p_positive is not None:
            tensors += (self.p_positive,)
        if any(not value.is_floating_point() for value in tensors):
            raise ValueError("PACT prediction parameters must be floating tensors")
        if len({value.dtype for value in tensors}) != 1:
            raise ValueError("PACT prediction parameters must share one floating dtype")
        if len({value.device for value in (*tensors, self.active, self.day_index)}) != 1:
            raise ValueError("PACT prediction tensors must be on one device")
        if any(not torch.isfinite(value).all() for value in tensors):
            raise ValueError("PACT prediction parameters must be finite")
        if bool((self.mu <= 0).any() or (self.k <= 0).any()):
            raise ValueError("PACT size parameters must be strictly positive")
        if self.p_positive is not None and bool(
            (self.p_positive <= 0).any() or (self.p_positive >= 1).any()
        ):
            raise ValueError("PACT hurdle probabilities must be strictly inside (0,1)")
        if not torch.allclose(
            self.allocation.sum(-1), torch.ones_like(self.mu), atol=2e-6, rtol=2e-6
        ):
            raise ValueError("PACT active allocation probabilities must sum to one")
        if bool((self.allocation < 0).any() or (self.allocation > 1).any()):
            raise ValueError("PACT allocation probabilities must lie in [0,1]")
        active = self.active[None, :, :].expand_as(self.allocation)
        if bool((self.allocation.masked_select(~active) != 0).any()):
            raise ValueError("inactive PACT allocation slots must be exactly zero")
        if bool((self.allocation_log.masked_select(~active) != 0).any()):
            raise ValueError("inactive PACT allocation log probabilities must be canonical zero")
        tolerance = max(2e-6, 10 * torch.finfo(self.allocation.dtype).eps)
        if not torch.allclose(
            torch.exp(self.allocation_log.masked_fill(~active, float("-inf"))),
            self.allocation, atol=tolerance, rtol=tolerance,
        ) or not torch.allclose(
            torch.logsumexp(self.allocation_log.masked_fill(~active, float("-inf")), -1),
            torch.zeros_like(self.mu), atol=tolerance, rtol=tolerance,
        ):
            raise ValueError("PACT allocation log probabilities disagree with the simplex")

    def segment_mean(self) -> Tensor:
        if self.p_positive is None:
            return self.mu
        return hurdle_mean(self.p_positive, self.mu, self.k)

    def mean(self) -> Tensor:
        daily = self.segment_mean()[..., None] * self.allocation
        result = torch.zeros(
            self.mu.shape[0], self.horizon, dtype=daily.dtype, device=daily.device
        )
        for segment in range(self.active.shape[0]):
            for slot in torch.nonzero(self.active[segment], as_tuple=False).squeeze(-1).tolist():
                result[:, self.day_index[segment, slot]] = daily[:, segment, slot]
        return result

    def joint_nll(self, counts: Tensor, eligible: Tensor) -> Tensor:
        """Per-segment joint NLL; ineligible complete-case segments contribute zero."""
        if counts.shape != self.allocation.shape:
            raise ValueError("PACT labels must be [batch,segments,7]")
        _require_bool("segment eligibility", eligible, self.mu.shape)
        if not torch.isfinite(counts).all() or bool((counts < 0).any()):
            raise ValueError("PACT counts must be finite and nonnegative")
        if counts.is_floating_point() and bool((counts != torch.floor(counts)).any()):
            raise ValueError("PACT counts must be integer-valued")
        inactive = ~self.active[None, :, :]
        if bool((counts.masked_select(inactive) != 0).any()):
            raise ValueError("inactive PACT label slots must be zero")
        total = counts.sum(-1)
        if self.p_positive is None:
            total_log = nb_log_pmf(total, self.mu, self.k)
        else:
            total_log = torch.log1p(-self.p_positive)
            positive = total > 0
            if bool(positive.any()):
                positive_log = torch.log(self.p_positive[positive]) + ztnb_log_pmf(
                    total[positive], self.mu[positive], self.k[positive]
                )
                total_log = total_log.masked_scatter(positive, positive_log)
        allocation_log = (
            torch.lgamma(total + 1)
            - torch.lgamma(counts + 1).sum(-1)
            + (counts * self.allocation_log).sum(-1)
        )
        joint = total_log + allocation_log
        return torch.where(eligible, -joint, torch.zeros_like(joint))

    def nll(self, daily_counts: Tensor, observed: Tensor) -> Tensor:
        """Daily-shaped validation contributions whose sum equals the eligible joint NLL."""
        if daily_counts.shape != (self.mu.shape[0], self.horizon):
            raise ValueError("daily validation labels disagree with the PACT horizon")
        _require_bool("daily observation mask", observed, daily_counts.shape)
        padded = torch.zeros_like(self.allocation)
        exact = torch.zeros_like(self.allocation, dtype=torch.bool)
        for segment in range(self.active.shape[0]):
            for slot in torch.nonzero(self.active[segment], as_tuple=False).squeeze(-1).tolist():
                index = self.day_index[segment, slot]
                padded[:, segment, slot] = torch.where(
                    observed[:, index], daily_counts[:, index], 0
                )
                exact[:, segment, slot] = observed[:, index]
        eligible = torch.all(~self.active[None, :, :] | exact, dim=-1)
        segment_nll = self.joint_nll(padded, eligible)
        result = torch.zeros_like(daily_counts, dtype=segment_nll.dtype)
        for segment in range(self.active.shape[0]):
            slots = torch.nonzero(self.active[segment], as_tuple=False).squeeze(-1)
            contribution = segment_nll[:, segment] / len(slots)
            for slot in slots.tolist():
                result[:, self.day_index[segment, slot]] = contribution
        return result

    def daily_log_prob(self, daily_counts: Tensor) -> Tensor:
        """Exact daily marginal log probability under the coherent segment construction."""
        if daily_counts.shape != (self.mu.shape[0], self.horizon):
            raise ValueError("daily marginal labels disagree with the PACT horizon")
        if not torch.isfinite(daily_counts).all() or bool((daily_counts < 0).any()):
            raise ValueError("daily marginal counts must be finite and nonnegative")
        if daily_counts.is_floating_point() and bool(
            (daily_counts != torch.floor(daily_counts)).any()
        ):
            raise ValueError("daily marginal counts must be integer-valued")
        result = torch.empty_like(daily_counts, dtype=self.mu.dtype)
        for segment in range(self.active.shape[0]):
            for slot in torch.nonzero(self.active[segment], as_tuple=False).squeeze(-1).tolist():
                index = self.day_index[segment, slot]
                share = self.allocation[:, segment, slot]
                daily_mu = self.mu[:, segment] * share
                label = daily_counts[:, index]
                if self.p_positive is None:
                    positive_share = share > 0
                    result[~positive_share, index] = torch.where(
                        label[~positive_share] == 0,
                        torch.zeros_like(label[~positive_share], dtype=result.dtype),
                        torch.full_like(label[~positive_share], -torch.inf, dtype=result.dtype),
                    )
                    if bool(positive_share.any()):
                        result[positive_share, index] = nb_log_pmf(
                            label[positive_share], daily_mu[positive_share],
                            self.k[:, segment][positive_share],
                        )
                    continue
                zero_share = share == 0
                if bool(zero_share.any()):
                    result[zero_share, index] = torch.where(
                        label[zero_share] == 0,
                        torch.zeros_like(label[zero_share], dtype=result.dtype),
                        torch.full_like(label[zero_share], -torch.inf, dtype=result.dtype),
                    )
                positive_share = ~zero_share
                if not bool(positive_share.any()):
                    continue
                log_q0 = nb_log_pmf(
                    torch.zeros_like(self.mu[:, segment]),
                    self.mu[:, segment],
                    self.k[:, segment],
                )
                log_qd0 = nb_log_pmf(
                    torch.zeros_like(daily_mu), daily_mu, self.k[:, segment]
                )
                log_daily_positive = (
                    torch.log(self.p_positive[:, segment])
                    + log1mexp(log_qd0)
                    - log1mexp(log_q0)
                )
                daily_positive = torch.exp(log_daily_positive).clamp(max=1 - 1e-7)
                cell = torch.log1p(-daily_positive)
                positive = (label > 0) & positive_share
                if bool(positive.any()):
                    selected = torch.log(daily_positive[positive]) + ztnb_log_pmf(
                        label[positive], daily_mu[positive], self.k[:, segment][positive]
                    )
                    cell = cell.masked_scatter(positive, selected)
                result[positive_share, index] = cell[positive_share]
        return result

    def sample(self, n_draws: int, *, generator: torch.Generator | None = None) -> Tensor:
        """Ancestral integer daily paths whose days sum exactly to their segment total."""
        totals = (
            sample_paths_negbin(self.mu, self.k, n_draws, generator=generator)
            if self.p_positive is None
            else sample_paths(
                self.p_positive, self.mu, self.k, n_draws, generator=generator
            )
        )
        result = torch.zeros(
            n_draws,
            self.mu.shape[0],
            self.horizon,
            dtype=torch.int64,
            device=self.mu.device,
        )
        for draw in range(n_draws):
            for row in range(self.mu.shape[0]):
                for segment in range(self.mu.shape[1]):
                    total = int(totals[draw, row, segment])
                    if total == 0:
                        continue
                    slots = torch.multinomial(
                        self.allocation[row, segment],
                        total,
                        replacement=True,
                        generator=generator,
                    )
                    allocated = torch.bincount(slots, minlength=7)
                    active_slots = torch.nonzero(
                        self.active[segment], as_tuple=False
                    ).squeeze(-1).tolist()
                    for slot in active_slots:
                        result[draw, row, self.day_index[segment, slot]] = allocated[slot]
        return result


class PACTModel(nn.Module):
    """One decoder-only stack over historical streams and masked future segment queries."""

    def __init__(
        self,
        *,
        history_width: int,
        future_width: int,
        statistic_width: int,
        attribute_width: int,
        identity_cardinalities: tuple[int, ...],
        d_model: int,
        n_heads: int,
        n_layers: int,
        ff_multiplier: int,
        rope_theta: float,
        position_mode: str,
        max_streams: int,
        head_law: str,
    ) -> None:
        super().__init__()
        if (
            any(type(value) is not int or value < 1 for value in (
                history_width,
                future_width,
                statistic_width,
                d_model,
                n_heads,
                n_layers,
                ff_multiplier,
                max_streams,
            ))
            or type(attribute_width) is not int
            or attribute_width < 0
            or head_law not in PACT_LAWS
            or position_mode not in {"elapsed_day", "token_index"}
        ):
            raise ValueError("invalid PACT model dimensions or law")
        if d_model % n_heads or (d_model // n_heads) % 2:
            raise ValueError("PACT d_model must form even per-head rotary dimensions")
        self.d_model = d_model
        self.max_streams = max_streams
        self.head_law = head_law
        self.position_mode = position_mode
        self.history_projection = nn.Linear(history_width, d_model)
        self.future_projection = nn.Linear(future_width, d_model)
        self.statistic_projection = nn.Linear(statistic_width, d_model)
        self.attribute_projection = (
            nn.Linear(attribute_width, d_model, bias=False) if attribute_width else None
        )
        self.identity_embeddings = nn.ModuleList(
            nn.Embedding(cardinality, d_model) for cardinality in identity_cardinalities
        )
        self.token_type = nn.Embedding(7, d_model)
        self.stream_type = nn.Embedding(max_streams, d_model)
        self.masked_target = nn.Parameter(torch.zeros(d_model))
        self.embedding_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.layers = nn.ModuleList(
            PACTDecoderBlock(
                d_model, n_heads, ff_multiplier, rope_theta=float(rope_theta)
            )
            for _ in range(n_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.occurrence = (
            nn.Linear(d_model, 1) if head_law == "zero_truncated_hurdle_negbin" else None
        )
        self.size = nn.Linear(d_model, 1)
        self.dispersion = nn.Linear(d_model, 1)
        self.allocation = nn.Linear(d_model, 7)

    def forward(
        self,
        *,
        history_features: Tensor,
        history_exists: Tensor,
        history_positions: Tensor,
        history_types: Tensor,
        future_features: Tensor,
        future_exists: Tensor,
        future_positions: Tensor,
        future_types: Tensor,
        future_active: Tensor,
        future_day_index: Tensor,
        statistics: Tensor,
        attributes: Tensor,
        identity_indices: Tensor,
        horizon: int,
    ) -> PACTPrediction:
        if history_features.ndim == 3:
            history_features = history_features[:, None, :, :]
            history_exists = history_exists[:, None, :]
        if history_features.ndim != 4:
            raise ValueError("PACT history must be [batch,streams,tokens,features]")
        batch, streams, history_length, _ = history_features.shape
        if not 1 <= streams <= self.max_streams:
            raise ValueError("PACT stream count exceeds its registered cap")
        _require_bool("history existence", history_exists, (batch, streams, history_length))
        if future_features.ndim != 3 or future_features.shape[0] != batch:
            raise ValueError("PACT future features must be [batch,segments,features]")
        segments = future_features.shape[1]
        _require_bool("future existence", future_exists, (batch, segments))
        if statistics.shape[0] != batch or attributes.shape[0] != batch:
            raise ValueError("PACT static/statistic rows differ from token rows")
        if identity_indices.shape != (batch, len(self.identity_embeddings)):
            raise ValueError("PACT identity index layout differs from registered tables")
        if future_positions.shape != (segments,) or future_types.shape != (segments,):
            raise ValueError("PACT future descriptors must be shared segment vectors")
        if history_positions.shape != (history_length,) or history_types.shape != (history_length,):
            raise ValueError("PACT history descriptors must be shared chronological vectors")
        _require_bool("future active-day mask", future_active, (segments, 7))
        if future_day_index.shape != (segments, 7) or future_day_index.dtype != torch.int64:
            raise ValueError("PACT future day index must be int64 [segments,7]")
        if type(horizon) is not int or horizon < 1:
            raise ValueError("PACT horizon must be a positive built-in integer")
        _validate_day_mapping(future_active, future_day_index, horizon)

        history = self.history_projection(history_features)
        history = history + self.token_type(history_types)[None, None, :, :]
        stream_ids = torch.arange(streams, device=history.device)
        history = history + self.stream_type(stream_ids)[None, :, None, :]
        static = self.statistic_projection(statistics)
        if self.attribute_projection is not None:
            static = static + self.attribute_projection(attributes)
        for column, embedding in enumerate(self.identity_embeddings):
            static = static + embedding(identity_indices[:, column])
        history[:, 0] = history[:, 0] + static[:, None, :]

        future = self.future_projection(future_features)
        future = (
            future
            + self.token_type(future_types)[None, :, :]
            + self.stream_type.weight[0][None, None, :]
            + static[:, None, :]
            + self.masked_target[None, None, :]
        )
        future_grid = torch.zeros(
            batch,
            streams,
            segments,
            self.d_model,
            dtype=future.dtype,
            device=future.device,
        )
        future_grid[:, 0] = future
        aggregate_future_exists = torch.zeros(
            batch,
            max(streams - 1, 0),
            segments,
            dtype=torch.bool,
            device=future.device,
        )
        exists = torch.cat(
            (
                history_exists,
                torch.cat((future_exists[:, None, :], aggregate_future_exists), dim=1),
            ),
            dim=2,
        )
        value = torch.cat((history, future_grid), dim=2)
        value = (value + self.embedding_mlp(value)) * exists[..., None]
        coordinates = (
            torch.arange(history_length + segments, device=value.device)
            if self.position_mode == "token_index"
            else torch.cat((history_positions, future_positions)).to(value.device)
        )
        positions = coordinates[None, None, :].expand(batch, streams, -1)
        for layer in self.layers:
            value = layer(
                value, exists, positions, history_length=history_length
            )
        hidden = self.final_norm(value[:, 0, history_length:])
        # The count parameters and seven-day simplex need FP32 range and summation accuracy.
        # Keep the learned trunk under autocast, but not this probability head.
        with torch.autocast(hidden.device.type, enabled=False):
            head_dtype = self.size.weight.dtype
            if head_dtype not in (torch.float32, torch.float64):
                raise ValueError("PACT count head requires FP32 or FP64 parameters")
            head_hidden = hidden.to(head_dtype)
            mu = F.softplus(self.size(head_hidden).squeeze(-1)) + 1e-6
            log_k = self.dispersion(head_hidden).squeeze(-1).clamp(LOG_K_MIN, LOG_K_MAX)
            k = torch.exp(log_k)
            p_positive = None
            if self.occurrence is not None:
                eps = torch.finfo(head_hidden.dtype).eps
                p_positive = torch.sigmoid(self.occurrence(head_hidden).squeeze(-1)).clamp(
                    eps, 1 - eps
                )
            logits = self.allocation(head_hidden)
            masked_logits = logits.masked_fill(~future_active[None, :, :], -torch.inf)
            allocation_log = torch.log_softmax(masked_logits, dim=-1)
            allocation = torch.exp(allocation_log)
            allocation = torch.where(
                future_active[None, :, :], allocation, torch.zeros_like(allocation)
            )
            allocation_log = torch.where(
                future_active[None, :, :], allocation_log, torch.zeros_like(allocation_log)
            )
        return PACTPrediction(
            law=self.head_law,
            p_positive=p_positive,
            mu=mu,
            k=k,
            allocation=allocation,
            allocation_log=allocation_log,
            active=future_active,
            day_index=future_day_index,
            horizon=horizon,
        )


__all__ = [
    "PACTBatch",
    "PACTDecoderBlock",
    "PACTLabels",
    "PACTModel",
    "PACTModelInputs",
    "PACTPrediction",
    "apply_signed_rope",
    "full_attention_mask",
    "prefix_attention_mask",
]
