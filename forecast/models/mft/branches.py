"""Shallow encoders for the supported input families, each producing a branch token. Per-channel
masks retain available features when other columns are missing. The initial model uses fixed
calendar windows and excludes adaptive event kernels, interactions and the peer branch.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = [
    "BranchError",
    "CalendarEventBranch",
    "CommercialBranch",
    "DemandHistoryBranch",
    "RevIN",
    "StaticIdentityBranch",
]


class BranchError(ValueError):
    """A branch was given a shape or a mask it cannot interpret."""


def _binary_mask(
    mask: Tensor,
    reference: Tensor,
    name: str,
    *,
    error_type: type[ValueError] = BranchError,
) -> Tensor:
    """Require a Boolean availability tensor with the exact current shape and device."""
    if (
        not isinstance(mask, Tensor)
        or mask.shape != reference.shape
        or mask.device != reference.device
    ):
        shape = None if not isinstance(mask, Tensor) else tuple(mask.shape)
        device = None if not isinstance(mask, Tensor) else mask.device
        raise error_type(
            f"{name} must be a Boolean binary 0/1 mask matching shape {tuple(reference.shape)} "
            f"and device {reference.device}; got shape {shape} and device {device}"
        )
    if mask.dtype != torch.bool:
        raise error_type(f"{name} must be a Boolean binary 0/1 mask; got dtype {mask.dtype}")
    return mask



class RevIN(nn.Module):
    """Reversible instance normalisation over a series' own history.

    Attention is a dot product, so without per-series
    normalisation the fast movers dominate the softmax and the gradient, and the slow movers --
    which are most of the catalogue -- contribute almost nothing. Normalising the OUTPUT would not
    fix it: by then the attention has already been computed on unnormalised magnitudes.

    Statistics come from the ENCODER WINDOW ONLY. Using the full series would leak the horizon's
    scale into the encoder, which is a leak that looks like unusually good performance on exactly
    the series whose level shifts.

    THE EPSILON IS RELATIVE, `var + eps * mean^2`, NOT `var + eps`. This looks like a detail and
    is not. An absolute floor is not scale-free: rescaling a series by `f` multiplies its variance
    by `f^2` while the floor stays put, so the SAME series at two scales normalises differently.
    Measured on a quiet series (variance 2.1e-3): an absolute `eps = 1e-5` distorts the normalised
    values by 0.24%, and the distortion grows as the series gets quieter -- which is exactly the
    slow movers this module exists to protect. With `eps * mean^2` both terms scale as `f^2` and
    the invariance is exact.
    """

    def __init__(self, n_channels: int, eps: float = 1e-5) -> None:
        """Args:
        n_channels: Channels to normalise independently.
        eps: RELATIVE variance floor, applied as `eps * mean^2`. See the class docstring.
            A last-resort absolute floor still applies to a series that is identically zero,
            where there is no scale to be relative to -- that one normalises to zeros, which
            is the correct "this series does not move" signal rather than an inf.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_channels))
        self.bias = nn.Parameter(torch.zeros(n_channels))

    def forward(self, x: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """Normalise `[batch, time, channels]`. Returns `(normalised, mean, std)`.

        The mean and std are returned rather than stored, because the same module instance serves
        every series in a batch and stashing them on `self` would make two concurrent calls
        interfere -- a bug that only appears under a dataloader with workers.

        Args:
            x: `[batch, time, channels]`.
            mask: Optional `[batch, time]`, True where the step is observed. Masked steps are
                excluded from the statistics, so a run of never-ranged days does not drag the mean
                toward zero and make the series look smaller than it is.
        """
        if x.dim() != 3:
            raise BranchError(f"RevIN expects [batch, time, channels], got {tuple(x.shape)}")
        if mask is None:
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, unbiased=False, keepdim=True)
        else:
            mask = _binary_mask(mask, x[..., 0], "RevIN mask")
            m = mask.unsqueeze(-1).to(x.dtype)
            n = m.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean = (x * m).sum(dim=1, keepdim=True) / n
            var = (((x - mean) ** 2) * m).sum(dim=1, keepdim=True) / n
        # Scale-free floor: both `var` and `mean^2` scale as f^2 under a rescaling, so the
        # normalised output does not move. The absolute clamp only catches an identically-zero
        # series, where there is no scale to be relative to.
        variance = (var + self.eps * mean.pow(2)).clamp_min(torch.finfo(x.dtype).tiny)
        std = torch.sqrt(variance)
        return (x - mean) / std * self.weight + self.bias, mean, std


class DemandHistoryBranch(nn.Module):
    """Attends over TIME, on daily and weekly patches, after RevIN.

    Two patch sizes rather than one: a daily view carries day-of-week shape, and a weekly view
    carries level and trend without making the attention span 180 positions to see them.

    Returns BOTH a pooled token and the weekly patch SEQUENCE. The token is the branch's summary
    for the fusion gate; the sequence is the trunk's TIME axis -- the thing its time stages
    attend over. Before the sequence existed the trunk received the fused token copied across
    the horizon, and attention over identical copies is a fixed average: the "time" half of
    every trunk stage was arithmetic that could not affect the output. The weekly grain is the
    v3 table's own resolution ("52 timesteps"), and "how does this week relate to the last one"
    is the trunk docstring's own description of the question.

    MASKED DAYS CARRY ZERO, NOT (0 - mean)/std. RevIN excludes masked days from its statistics
    but still normalises their stored zeros, so a fast mover's pre-launch days became large
    negative values -- and two paths read them: any partially observed patch mixed them into its
    weekly token, and the output's weekly summary averaged every patch regardless of the mask.
    Measured before the fix: perturbing ONLY masked days moved the output token by four orders
    of magnitude. Zero after RevIN is the "this series does not move" neutral, the same value an
    identically-zero series normalises to, and the summary now averages only patches with at
    least one observed day.
    """

    def __init__(
        self,
        d_model: int,
        encoder_days: int,
        *,
        weekly_patch: int = 7,
        n_heads: int = 4,
        dropout: float = 0.15,
        attention_layers: int = 1,
        n_covariates: int = 0,
        position_availability: bool = False,
    ) -> None:
        super().__init__()
        if type(n_covariates) is not int or n_covariates < 0:
            raise BranchError("n_covariates must be a nonnegative integer")
        if type(position_availability) is not bool:
            raise BranchError("position_availability must be boolean")
        self.n_covariates = n_covariates
        self.position_availability = position_availability
        if type(attention_layers) is not int or attention_layers < 1:
            raise BranchError(
                f"attention_layers must be a positive integer, got {attention_layers!r}"
            )
        if encoder_days % weekly_patch:
            raise BranchError(
                f"encoder_days={encoder_days} is not a multiple of weekly_patch={weekly_patch}; a "
                "ragged final patch would summarise a different number of days from the others"
            )
        self.encoder_days = encoder_days
        self.weekly_patch = weekly_patch
        self.n_patches = encoder_days // weekly_patch
        self.revin = RevIN(1)
        # The optional fourth channel distinguishes a real calendar position with an unavailable
        # target from global left-padding before the dataset begins. Observation and OOS remain
        # separate: a genuine zero, an unobserved real date and an absent date are not synonyms.
        input_width = 3 + int(position_availability) + 2 * n_covariates
        self.daily = nn.Linear(input_width, d_model)
        # Attention over the daily tokens is permutation-invariant too. Weekly positions alone
        # cannot distinguish Monday OOS from Sunday OOS inside one patch, so daily position is a
        # first-class part of the representation rather than something the weekly projection is
        # expected to reconstruct after pooling.
        self.daily_pos = nn.Parameter(torch.randn(1, encoder_days, d_model) * 0.02)
        # Flatten VALUE / OBSERVED / OOS for every day in the patch. Using only the seven demand
        # magnitudes here dropped both state channels from the sequence consumed by the trunk,
        # even though the pooled branch token happened to see one of them.
        self.weekly = nn.Linear(weekly_patch * input_width, d_model)
        # The weekly tokens are a SEQUENCE now, and attention is permutation-invariant, so
        # position must be in the token: without this, "week 51 then week 52" and the reverse
        # are the same input, and a trend has no representation to live in.
        self.weekly_pos = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attention_layers = attention_layers
        self.attn_refinements = nn.ModuleList(
            nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            for _ in range(attention_layers - 1)
        )
        for refinement in self.attn_refinements:
            nn.init.zeros_(refinement.out_proj.weight)
            nn.init.zeros_(refinement.out_proj.bias)
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        # `ForecastModelSpec.demand_branch_dropout`. Applied to the PATCH TOKENS rather than to
        # the raw history: dropping raw days would delete demand, and a deleted day is
        # indistinguishable from a day that sold nothing -- which is the one confusion the whole
        # observation contract exists to prevent. Dropping token dimensions regularises the
        # representation without lying about the series.
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(2 * d_model, d_model)

    def forward(
        self, history: Tensor, mask: Tensor | None = None, censored: Tensor | None = None,
        covariates: Tensor | None = None, covariate_mask: Tensor | None = None,
        position_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Args:
            history: `[batch, encoder_days]` demand.
            mask: Optional `[batch, encoder_days]`, True where the day is observed.
            censored: Optional `[batch, encoder_days]`, True where the recorded value is a
                lower bound rather than demand, including a zero-valued all-day stockout. The
                value mask may be False there while this state remains visible.
            position_mask: Optional `[batch, encoder_days]`, True for real calendar positions and
                False only for global padding before the dataset begins.

        Returns:
            `(token, sequence)` -- `[batch, d_model]` for the fusion stack, and
            `[batch, n_patches, d_model]` weekly tokens for the trunk's time axis.
        """
        observed = (
            torch.ones_like(history, dtype=torch.bool)
            if mask is None
            else _binary_mask(mask, history, "demand observation mask")
        )
        if self.position_availability and position_mask is None:
            raise BranchError("calendar-position-aware history requires position_mask")
        available = (
            torch.ones_like(history, dtype=torch.bool)
            if position_mask is None
            else _binary_mask(position_mask, history, "calendar position mask")
        )
        oos = (
            torch.zeros_like(history, dtype=torch.bool)
            if censored is None
            else _binary_mask(censored, history, "demand censoring mask")
        )
        if bool(((observed | oos) & ~available).any()):
            raise BranchError("globally padded positions cannot be observed or censored")
        x = history.unsqueeze(-1)
        normed, _, _ = self.revin(x, observed)
        b, t, _ = normed.shape
        if mask is not None:
            # Zero AFTER normalisation: the statistics already excluded these days, but their
            # stored zeros were still standardised to (0 - mean)/std, which for any active
            # series is a large negative number pretending to be demand.
            normed = normed * observed.unsqueeze(-1).to(normed.dtype)
        observed_f = observed.unsqueeze(-1).to(normed.dtype)
        oos_f = oos.unsqueeze(-1).to(normed.dtype)
        day_state = torch.cat([normed, observed_f, oos_f], dim=-1)
        if self.position_availability:
            day_state = torch.cat(
                [day_state, available.unsqueeze(-1).to(normed.dtype)], dim=-1
            )
        covariate_visible = torch.zeros_like(observed)
        if self.n_covariates:
            expected = (*history.shape, self.n_covariates)
            if (covariates is None or covariate_mask is None or covariates.shape != expected
                    or covariate_mask.shape != expected):
                raise BranchError("historical covariates and masks must match the declared layout")
            covariate_mask = _binary_mask(
                covariate_mask, covariates, "historical covariate mask"
            )
            if bool((covariate_mask & ~available.unsqueeze(-1)).any()):
                raise BranchError("globally padded positions cannot contain covariates")
            safe = torch.where(covariate_mask, covariates, 0.0)
            if not torch.isfinite(safe).all():
                raise BranchError("available historical covariates must be finite")
            day_state = torch.cat([day_state, safe, covariate_mask.to(safe.dtype)], -1)
            covariate_visible = covariate_mask.any(-1)
        elif covariates is not None or covariate_mask is not None:
            raise BranchError("historical covariates supplied to a covariate-free branch")
        daily = self.dropout(self.daily(day_state) + self.daily_pos)  # [b, t, d]
        weekly = self.dropout(
            self.weekly(day_state.reshape(b, self.n_patches, -1))
            + self.weekly_pos
        )
        tokens = torch.cat([daily, weekly], dim=1)
        key_mask = None
        weekly_mask = None
        if mask is not None or censored is not None or position_mask is not None:
            # An OOS day is visible even when its recorded lower bound is zero and therefore has
            # no usable magnitude. `observed` alone would mask out exactly that state signal.
            visible = (
                available
                if position_mask is not None
                else observed | oos | covariate_visible
            )
            weekly_mask = visible.reshape(b, self.n_patches, self.weekly_patch).any(-1)
            key_mask = ~torch.cat([visible, weekly_mask], dim=1)
            # A ROW WITH NO OBSERVED DAY AT ALL. Attention over an empty key set is undefined and
            # returns NaN -- the same failure `CommercialBranch` names below, arriving here from a
            # different direction. It appears as soon as a training window reaches back far enough
            # to precede a product's launch: measured at a 2014 origin on M5, 3,101 of 30,490
            # series had every encoder day unobserved, and one NaN row poisons the whole batch
            # through the bound's logsumexp.
            #
            # Such a row is unmasked rather than dropped. The masked values are zeroed above, so
            # attending over them produces the branch's "nothing is known here" representation --
            # the honest answer for a product that did not exist yet, and what the fusion gate
            # should receive.
            empty = key_mask.all(dim=-1)
            if bool(empty.any()):
                key_mask = key_mask.clone()
                key_mask[empty] = False
            if position_mask is not None:
                weekly = weekly * weekly_mask.unsqueeze(-1).to(weekly.dtype)
        pooled, _ = self.attn(
            self.query.expand(b, -1, -1),
            tokens,
            tokens,
            key_padding_mask=key_mask,
            need_weights=False,
        )
        # Each refinement asks the same time-token sequence a follow-up question using the
        # previous pooled state.  Zero-initialised residual projections make a newly deepened
        # branch start as the proven one-layer branch, then learn the extra depth from data.
        for refinement in self.attn_refinements:
            update, _ = refinement(
                pooled, tokens, tokens, key_padding_mask=key_mask, need_weights=False
            )
            pooled = pooled + update
        if weekly_mask is None:
            summary = weekly.mean(dim=1)
        else:
            m = weekly_mask.unsqueeze(-1).to(weekly.dtype)
            summary = (weekly * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return self.out(torch.cat([pooled.squeeze(1), summary], dim=-1)), weekly


class CommercialBranch(nn.Module):
    """Attends over CHANNELS -- price, discount depth, promotion -- not over time.

    Time is the demand branch's job. What this branch has to resolve is that a discount and a
    promotion flag say overlapping things, and which one carries the signal differs by client. A
    channel-wise attention lets the model weigh them; a concatenation would fix the weighting at
    training time for whichever client dominated the batch.

    A MISSING CHANNEL IS MASKED, NOT ZERO-FILLED. Zero is a legitimate discount depth, so filling
    an absent price channel with zeros teaches "this product is free" rather than "we do not know".
    """

    def __init__(self, n_channels: int, d_model: int, *, n_heads: int = 2,
                 attention_layers: int = 1) -> None:
        super().__init__()
        if type(attention_layers) is not int or attention_layers < 1:
            raise BranchError(
                f"attention_layers must be a positive integer, got {attention_layers!r}"
            )
        self.n_channels = n_channels
        self.embed = nn.Linear(1, d_model)
        self.channel_id = nn.Parameter(torch.randn(1, n_channels, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attention_layers = attention_layers
        self.attn_refinements = nn.ModuleList(
            nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            for _ in range(attention_layers - 1)
        )
        for refinement in self.attn_refinements:
            nn.init.zeros_(refinement.out_proj.weight)
            nn.init.zeros_(refinement.out_proj.bias)
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, channels: Tensor, channel_mask: Tensor | None = None) -> Tensor:
        """Args:
            channels: `[batch, n_channels]` values at the forecast step.
            channel_mask: Optional `[batch, n_channels]`, True where the channel exists for this
                client. Per CHANNEL, deliberately -- coverage differs by column, not by row.

        Returns:
            `[batch, d_model]`.
        """
        if channels.shape[-1] != self.n_channels:
            raise BranchError(
                f"expected {self.n_channels} channels, got {channels.shape[-1]}; a silently "
                "reordered channel set would train each weight on the wrong column"
            )
        tokens = self.embed(channels.unsqueeze(-1)) + self.channel_id
        channel_mask = (
            None
            if channel_mask is None
            else _binary_mask(channel_mask, channels, "commercial channel mask")
        )
        key_mask = None if channel_mask is None else ~channel_mask
        if key_mask is not None and bool(key_mask.all(dim=-1).any()):
            raise BranchError(
                "a row masks out every commercial channel; attention over an empty key set is "
                "undefined and returns NaN. Drop the branch for that client instead"
            )
        pooled, _ = self.attn(
            self.query.expand(channels.shape[0], -1, -1),
            tokens,
            tokens,
            key_padding_mask=key_mask,
            need_weights=False,
        )
        for refinement in self.attn_refinements:
            update, _ = refinement(
                pooled, tokens, tokens, key_padding_mask=key_mask, need_weights=False
            )
            pooled = pooled + update
        return pooled.squeeze(1)


class _KernelBasisRefinement(nn.Module):
    """One residual refinement of an event kernel's compact spline weights.

    This operates on ``[event, basis]`` weights, never on the full
    ``[series, forecast_day, d_model]`` activation.  Four refinements therefore make the
    *kernel shape* deeper without multiplying the memory required by a large forecasting panel.
    The final projection starts at zero, so adding depth preserves the one-kernel initial
    behaviour rather than changing an existing model merely because a knob was introduced.
    """

    def __init__(self, n_basis: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(n_basis)
        self.expand = nn.Linear(n_basis, 2 * n_basis, bias=False)
        self.activate = nn.GELU()
        self.reduce = nn.Linear(2 * n_basis, n_basis, bias=False)
        nn.init.zeros_(self.reduce.weight)

    def forward(self, weights: Tensor) -> Tensor:
        return weights + self.reduce(self.activate(self.expand(self.norm(weights))))


class CalendarEventBranch(nn.Module):
    """A DISTRIBUTED-LAG kernel over signed distance to each event. Not attention.

    Attention was the wrong instrument here and the reason is worth keeping: its weights are
    RELATIVE and never reach exactly zero, so an event always contributes something at every lag.
    What is needed is a kernel that can be zero far from the event and non-zero near it.

    Nor are 61 free daily coefficients tenable: a product has seen about TWO Easters. So the
    kernel is a small spline basis over signed distance -- about 8 blocks rather than 61 points --
    with a shared shape and a shrunk per-event deviation, which is the partial pooling that makes
    an N=2 event estimable at all.

    `first_build=True` REFUSES the adaptive version. The ordered gates that LEARN how far an
    effect reaches are Phase 5; here the window is fixed at `max_support` and the kernel decides
    only the shape within it.
    """

    def __init__(
        self,
        n_events: int,
        d_model: int,
        *,
        max_support: int = 30,
        n_basis: int = 8,
        shrink: float = 0.1,
        n_series: int = 0,
        series_shrink: float = 0.1,
        event_windows: tuple[tuple[int, int], ...] | None = None,
        kernel_layers: int = 1,
    ) -> None:
        """Args:
        n_events: Distinct event NAMES (a recurring event's occurrences share one).
        d_model: Output width.
        max_support: Fixed window either side, matching the feature builder's.
        n_basis: Spline blocks over signed distance.
        shrink: Scale on the per-EVENT shape deviation from the shared shape.
        n_series: Rows in the per-(series, event) AMPLITUDE deviation, 0 to disable.
            `event_pooling='partial'` is a per-event response shared across the catalogue
            PLUS a weight-decayed per-(SKU, event) deviation -- the deviation was never
            built, so pooling only ever ran across event TYPES. A scalar amplitude per
            (series, event) rather than a free per-series basis: a series has seen an
            annual event about twice, and six free coefficients against two observations
            is the fit the pooling exists to prevent. Zero-initialised, so an untrained
            model is exactly the pooled one.
        series_shrink: Scale on that deviation, same construction as `SeasonalAmplitude`.
        event_windows: Per-event `(lower, upper)` ceilings, one per event in column order,
            overriding the shared symmetric `max_support`. The kernel's hard zero applies
            per event: a pay-day effect gated at (0, +5) contributes nothing before its
            day however the shared basis is shaped.
        kernel_layers: Residual refinements of the compact spline-basis weights.  This is
            depth in the event *kernel*, not attention over dates or event names.  It is applied
            before the panel-sized kernel evaluation, so it has negligible panel-memory cost.
        """
        super().__init__()
        if type(kernel_layers) is not int or kernel_layers < 1:
            raise BranchError(f"kernel_layers must be a positive integer, got {kernel_layers!r}")
        self.n_events = n_events
        self.max_support = max_support
        self.n_basis = n_basis
        self.kernel_layers_count = kernel_layers
        self.shrink = shrink
        self.series_shrink = series_shrink
        if event_windows is not None:
            if len(event_windows) != n_events:
                raise BranchError(
                    f"{len(event_windows)} event windows for {n_events} events; a positional "
                    "mismatch gates one event with another's window"
                )
            if any(low > 0 or high < 0 for low, high in event_windows):
                raise BranchError(f"every event window must bracket zero, got {event_windows}")
            lows = torch.tensor([float(low) for low, _ in event_windows])
            highs = torch.tensor([float(high) for _, high in event_windows])
        else:
            lows = torch.full((n_events,), float(-max_support))
            highs = torch.full((n_events,), float(max_support))
        self.register_buffer("support_low", lows)
        self.register_buffer("support_high", highs)
        envelope = max(float(highs.max()), float(-lows.min()), 1.0) if n_events else max_support
        centres = torch.linspace(-envelope, envelope, n_basis)
        self.register_buffer("centres", centres)
        self.log_width = nn.Parameter(
            torch.log(torch.tensor(float(2 * max_support) / max(n_basis - 1, 1)))
        )
        self.shared = nn.Parameter(torch.zeros(n_basis))
        self.deviation = nn.Parameter(torch.zeros(n_events, n_basis))
        self.kernel_layers = nn.ModuleList(
            _KernelBasisRefinement(n_basis) for _ in range(kernel_layers)
        )
        self.series_deviation = nn.Embedding(n_series, n_events) if n_series else None
        if self.series_deviation is not None:
            nn.init.zeros_(self.series_deviation.weight)
        self.project = nn.Linear(n_events, d_model)

    def kernel_weights(self) -> Tensor:
        """Per-event basis weights: a shared shape plus a SHRUNK per-event deviation.

        The shrinkage is what lets an event seen twice borrow the shape of every other event
        rather than fit two points. Without it the per-event deviation is unconstrained and the
        kernel memorises the two instances it has.
        """
        weights = self.shared.unsqueeze(0) + self.shrink * self.deviation
        for layer in self.kernel_layers:
            weights = layer(weights)
        return weights

    def forward(
        self,
        signed_distance: Tensor,
        present: Tensor | None = None,
        series_index: Tensor | None = None,
    ) -> Tensor:
        """Args:
            signed_distance: `[batch, n_events]` or `[batch, horizon, n_events]` days to each
                event, negative before. The latter evaluates the learned kernel independently at
                every forecast step; applying it only at day one and feeding raw distance to a
                generic decoder bypasses the architecture's distributed-lag constraint.
            present: Optional tensor of the same shape, True where the event is in range at all.
            series_index: `[batch]` rows into the per-(series, event) amplitude deviation.
                Required when the branch was built with `n_series`, refused when it was not --
                a silently ignored index and a silently skipped deviation are the same bug.

        Returns:
            `[batch, d_model]` or `[batch, horizon, d_model]`, matching the leading dimensions.

        Raises:
            BranchError: `series_index` and the deviation table do not travel together.
        """
        if signed_distance.dim() not in (2, 3):
            raise BranchError(
                "event distance must be [batch, events] or [batch, horizon, events], got "
                f"{tuple(signed_distance.shape)}"
            )
        if signed_distance.shape[-1] != self.n_events:
            raise BranchError(
                f"expected {self.n_events} event channels, got {signed_distance.shape[-1]}; "
                "broadcasting one distance across several learned event kernels would assign "
                "the same event observation to different event identities"
            )
        if present is not None and present.shape != signed_distance.shape:
            raise BranchError(
                f"event presence {tuple(present.shape)} does not match distance "
                f"{tuple(signed_distance.shape)}"
            )
        if present is not None:
            present = _binary_mask(present, signed_distance, "event presence mask")
        d = signed_distance.unsqueeze(-1)  # [..., e, 1]
        weights = self.kernel_weights().reshape(
            *((1,) * (signed_distance.dim() - 1)), *self.kernel_weights().shape
        )
        basis = torch.exp(-0.5 * ((d - self.centres) / torch.exp(self.log_width)) ** 2)
        # Hard zero outside the support: a kernel that leaks past its window is the failure that
        # made attention the wrong instrument in the first place.
        inside = (
            (signed_distance >= self.support_low) & (signed_distance <= self.support_high)
        ).to(basis.dtype)
        effect = (basis * weights).sum(-1) * inside
        if self.series_deviation is not None:
            if series_index is None:
                raise BranchError(
                    "this calendar branch carries a per-(series, event) deviation but no "
                    "series index was given; skipping it would make the table dead weight"
                )
            amplitude = 1.0 + self.series_shrink * self.series_deviation(series_index)
            if signed_distance.dim() == 3:
                amplitude = amplitude.unsqueeze(1)
            effect = effect * amplitude
        elif series_index is not None:
            raise BranchError(
                "a series index was given but this branch has no per-(series, event) "
                "deviation; the caller believes a mechanism exists that was never built"
            )
        if present is not None:
            effect = effect * present.to(effect.dtype)
        return self.project(effect)


class StaticIdentityBranch(nn.Module):
    """Product and store identity. NO attention of its own, but it IS a group token.

    Nothing to attend over -- identity is one vector per cell, not a sequence. It still becomes a
    branch token because the trunk's group attention is where identity should interact with the
    other families, and folding it in earlier would let identity dominate a branch that is
    supposed to be about price or demand.
    """

    def __init__(
        self,
        n_attributes: int,
        d_model: int,
        *,
        text_dim: int = 0,
        residual_dim: int = 16,
        n_series: int = 0,
        n_products: int = 0,
        n_stores: int = 0,
        identity_cardinalities: tuple[int, ...] = (),
    ) -> None:
        """Args:
            n_attributes: Width of the structured attribute vector.
            d_model: Output width.
            text_dim: Width of a FROZEN text embedding, 0 if absent. Frozen because fine-tuning it
                on this objective would let a description memorise its series' demand level.
            residual_dim: Width of the learned residual. Small and weight-decayed: it exists to
                carry what the attributes miss, not to become a lookup table.
            n_series: Rows of a PER-SERIES residual -- the fallback for a panel whose series
                identity has no declared product/store split, where the series IS the product.
            n_products: Rows of the factorised PRODUCT table. When given (with `n_stores`), the
                residual is `product_row + store_row` -- "a table for shops and a table for
                products", as the spec's factorisation rule states -- and `n_series` is refused,
                because a per-(product, store)-pair table is exactly the joint lookup
                `allow_joint_embedding=False` forbids: it has NO ROW for a pair that has never
                occurred, and an assortment change is precisely a new pair.
            n_stores: Rows of the factorised STORE table.

        Raises:
            BranchError: both a per-series and a factorised residual were requested, or a
                factorised one was requested with only one of its two tables.
        """
        super().__init__()
        self.n_attributes = n_attributes
        if (any(type(n) is not int or n < 1 for n in identity_cardinalities)
                or identity_cardinalities and (n_series or n_products or n_stores)):
            raise BranchError("generic identity tables require positive sizes and no legacy tables")
        self.residual_identities = nn.ModuleList(
            nn.Embedding(n, residual_dim) for n in identity_cardinalities)
        if (n_products > 0) != (n_stores > 0):
            raise BranchError(
                f"a factorised residual needs BOTH tables, got n_products={n_products}, "
                f"n_stores={n_stores}; composing with a missing half is a per-one-id lookup "
                "wearing the factorised name"
            )
        if n_products and n_series:
            raise BranchError(
                "both a factorised (product + store) residual and a per-series one were "
                "requested; the per-series table on a product x store panel is the joint "
                "lookup the factorisation exists to replace"
            )
        self.factorised = bool(n_products)
        self.text_dim = text_dim
        self.residual_dim = (
            residual_dim if (n_series or n_products or identity_cardinalities) else 0)
        self.residual = nn.Embedding(n_series, residual_dim) if n_series else None
        self.residual_product = nn.Embedding(n_products, residual_dim) if n_products else None
        self.residual_store = nn.Embedding(n_stores, residual_dim) if n_products else None
        for table in (self.residual, self.residual_product, self.residual_store,
                      *self.residual_identities):
            if table is not None:
                assert isinstance(table, nn.Embedding)
                nn.init.normal_(table.weight, std=0.01)
        width = n_attributes + text_dim + self.residual_dim
        self.project = nn.Linear(width, d_model)

    def forward(
        self,
        attributes: Tensor,
        text: Tensor | None = None,
        series_index: Tensor | None = None,
        product_index: Tensor | None = None,
        store_index: Tensor | None = None,
        identity_indices: Tensor | None = None,
    ) -> Tensor:
        """Returns `[batch, d_model]`.

        Raises:
            BranchError: the index the configured residual needs was not given. Defaulting
                would give every series the same residual, silently.
        """
        parts = [attributes]
        if self.residual_identities:
            if (identity_indices is None
                    or identity_indices.shape != (
                        attributes.shape[0], len(self.residual_identities))
                    or identity_indices.dtype not in (torch.int32, torch.int64)
                    or identity_indices.device != attributes.device):
                raise BranchError("identity_indices must have one integral row per series and role")
            residuals = []
            for column, table in enumerate(self.residual_identities):
                assert isinstance(table, nn.Embedding)
                index = identity_indices[:, column]
                if (index < 0).any() or (index >= table.num_embeddings).any():
                    raise BranchError("identity index is outside its declared vocabulary")
                residuals.append(table(index))
            parts.append(torch.stack(residuals).sum(0))
        elif identity_indices is not None:
            raise BranchError("generic identity indices supplied without generic identity tables")
        if text is not None:
            parts.append(text)
        if self.factorised:
            product_residual = self.residual_product
            store_residual = self.residual_store
            if (
                product_index is None
                or store_index is None
                or product_residual is None
                or store_residual is None
            ):
                raise BranchError(
                    "a factorised residual was configured but a product or store index is "
                    "missing; composing with a default row would hand every new pair the same "
                    "identity"
                )
            parts.append(product_residual(product_index) + store_residual(store_index))
        elif self.residual is not None:
            if series_index is None:
                raise BranchError(
                    "a per-series residual was configured but no series index was given; "
                    "defaulting to row 0 would give every series the same residual"
                )
            parts.append(self.residual(series_index))
        return self.project(torch.cat(parts, dim=-1))
