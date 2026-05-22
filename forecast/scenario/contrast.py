"""Prospective scenario contrasts under a fitted model. Compare the planned inputs with a coherent
reference bundle from a pre-treatment origin. Every reported contrast requires campaign-level
resampling and an overlap check. Paired draws reduce simulation noise but do not identify a
causal effect. Driver allocation is disabled because co-occurring inputs are not separately
identified.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Literal

import torch

__all__ = ["Contrast", "ContrastError", "InterventionBundle", "campaign_bootstrap", "contrast"]

_WINDOWS = frozenset({"horizon", "event_support", "custom"})
_CAMPAIGN_DTYPES = (torch.int32, torch.int64)


def _is_finite_real(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _finite_tensor_sum(values: torch.Tensor, failure: str) -> float:
    """Sum finite effects without overflowing before cancellation can occur."""
    work = values.detach().to(torch.float64)
    scale = float(work.abs().max())
    if scale == 0.0:
        return 0.0
    total = float((work / scale).sum()) * scale
    if not math.isfinite(total):
        raise ContrastError(failure)
    return total


def _finite_float_sum(values: list[float], failure: str) -> float:
    """The Python-float counterpart used after whole-campaign aggregation."""
    scale = max((abs(value) for value in values), default=0.0)
    if scale == 0.0:
        return 0.0
    total = math.fsum(value / scale for value in values) * scale
    if not math.isfinite(total):
        raise ContrastError(failure)
    return total


def _canonical_date(value: object, label: str) -> dt.date:
    if type(value) is not str:
        raise ContrastError(f"{label} must be a canonical ISO YYYY-MM-DD date, got {value!r}")
    try:
        parsed = dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ContrastError(
            f"{label} must be a canonical ISO YYYY-MM-DD date, got {value!r}") from exc
    if parsed.isoformat() != value:
        raise ContrastError(f"{label} must be a canonical ISO YYYY-MM-DD date, got {value!r}")
    return parsed


def _validate_bootstrap_controls(n_bootstrap: object, seed: object, alpha: object) -> None:
    if type(n_bootstrap) is not int or n_bootstrap < 1:
        raise ContrastError(
            f"n_bootstrap must be a positive built-in integer, got {n_bootstrap!r}")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ContrastError(f"seed must be a built-in integer in [0, 2**63), got {seed!r}")
    if not _is_finite_real(alpha) or not 0.0 < alpha < 1.0:
        raise ContrastError(f"alpha must be finite and lie in (0,1), got {alpha!r}")


def _validate_driver_range(value: object, label: str) -> None:
    if (type(value) is not tuple or len(value) != 2
            or any(not _is_finite_real(bound) for bound in value)
            or value[0] > value[1]):
        raise ContrastError(
            f"{label} must be an ordered pair of finite built-in numbers, got {value!r}")


def _validate_window_request(
    window: object,
    event_support: object,
    custom_window: object,
) -> None:
    """Validate the selector and reject bounds that the selected integration would ignore."""
    if window is not None and (type(window) is not str or window not in _WINDOWS):
        raise ContrastError(
            f"window must be None or exactly one of {sorted(_WINDOWS)}, got {window!r}")
    if event_support is not None and window != "event_support":
        raise ContrastError(
            "event_support is only valid with window='event_support'; refusing ignored bounds")
    if custom_window is not None and window != "custom":
        raise ContrastError(
            "custom_window is only valid with window='custom'; refusing ignored bounds")
    if window == "event_support":
        if (type(event_support) is not tuple or len(event_support) != 2
                or any(type(value) is not int for value in event_support)
                or event_support[0] > 0 or event_support[1] < 0):
            raise ContrastError(
                "window='event_support' needs event_support=(lower<=0, upper>=0) signed "
                "built-in integer days around the campaign start")
    if window == "custom":
        if (type(custom_window) is not tuple or len(custom_window) != 2
                or any(type(value) is not int for value in custom_window)):
            raise ContrastError(
                "window='custom' needs custom_window=(first_day, last_day) built-in integer "
                "day indices")
        first, last = custom_window
        if not 1 <= first <= last:
            raise ContrastError(
                f"custom_window {custom_window} must satisfy 1 <= first <= last")


def _validate_campaign_index(
    campaign_index: object,
    *,
    n_cells: int,
    device: torch.device,
) -> torch.Tensor:
    """Return unique campaigns after validating the episode-label boundary."""
    if not isinstance(campaign_index, torch.Tensor):
        raise ContrastError(
            f"campaign_index must be a torch.Tensor, got {type(campaign_index).__name__}")
    if campaign_index.dim() != 1 or campaign_index.shape[0] != n_cells:
        raise ContrastError(
            f"campaign_index must be rank 1 with one of its campaign labels per cell; got "
            f"{campaign_index.shape} for {n_cells} cells")
    if campaign_index.device != device:
        raise ContrastError(
            f"campaign_index must be on {device} with its cell effects, got "
            f"{campaign_index.device}")
    if campaign_index.dtype not in _CAMPAIGN_DTYPES:
        raise ContrastError(
            "campaign_index must contain exact int32 or int64 episode labels; floating or "
            "unsigned aliases can merge distinct identifiers")
    campaigns = torch.unique(campaign_index)
    if campaigns.numel() < 2:
        raise ContrastError(
            f"{campaigns.numel()} campaign(s) in the data. A bootstrap over one campaign "
            "returns its own estimate with zero width, which reads as certainty and is the "
            "opposite of what this interval is for")
    return campaigns


class ContrastError(ValueError):
    """A contrast was requested that would not mean what it appears to."""


@dataclass(frozen=True)
class InterventionBundle:
    """A set of planned inputs that move TOGETHER, and the reference they are set to.

    Attributes:
        name: What the bundle is, e.g. `"price_promotion"`.
        columns: Indices into the known-future tensor that this bundle controls. **Plural on
            purpose** -- a single-column bundle is nearly always a mistake, because a discount
            without its display and media is a world that never existed.
        reference: Scalar value each column takes in the counterfactual. Usually the
            non-promoted level, and only correct when every column's reference IS one number --
            promo off, depth zero.
        reference_values: Per-cell references, `[..., len(columns)]` aligned with the
            known-future tensor's leading shape. **Required whenever a scalar cannot express
            the reference**: "no promotion" for a raw price column means each product's own
            LIST price, and a scalar 0 there sets every price to free -- an off-manifold world
            no client asked about, returned with a straight face. Takes precedence over
            `reference` when given.

    Raises:
        ContrastError: no columns, a non-finite reference, or reference_values whose last axis
            does not match the columns.
    """

    name: str
    columns: tuple[int, ...]
    reference: float = 0.0
    reference_values: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ContrastError("an intervention bundle needs a non-blank name")
        if not self.columns:
            raise ContrastError(f"bundle {self.name!r} controls no columns")
        if (any(type(column) is not int or column < 0
                for column in self.columns) or len(set(self.columns)) != len(self.columns)):
            raise ContrastError(
                f"bundle {self.name!r} columns must be unique non-negative integer indices")
        if not _is_finite_real(self.reference):
            raise ContrastError(f"bundle {self.name!r} has a non-finite reference")
        if self.reference_values is not None:
            if (not isinstance(self.reference_values, torch.Tensor)
                    or self.reference_values.dim() < 1
                    or not self.reference_values.is_floating_point()):
                raise ContrastError(
                    f"bundle {self.name!r} reference_values must be a floating tensor")
            if self.reference_values.shape[-1] != len(self.columns):
                raise ContrastError(
                    f"bundle {self.name!r} has reference_values with last axis "
                    f"{self.reference_values.shape[-1]} against {len(self.columns)} columns; a "
                    "broadcast here would set a column to another column's reference")
            try:
                finite_reference = bool(torch.isfinite(self.reference_values).all())
            except (RuntimeError, TypeError) as exc:
                raise ContrastError(
                    f"bundle {self.name!r} reference_values must be a materialized finite tensor"
                ) from exc
            if not finite_reference:
                raise ContrastError(f"bundle {self.name!r} has non-finite reference values")


@dataclass(frozen=True)
class Contrast:
    """A scenario contrast and everything that has to be read alongside it.

    Attributes:
        bundle: What was set to reference.
        effect: `E[Y | planned] - E[Y | reference]`, in units.
        low: Lower end of the campaign-bootstrap spread.
        high: Upper end.
        n_campaigns: Campaigns resampled. **The effective sample size is EPISODES, not rows** --
            a promotion spans days, and treating each day as an observation overstates the
            evidence by the length of the campaign.
        overlap: Share of the planned driver's range covered by campaigns actually observed.
        origin_is_pre_treatment: Whether the origin precedes the campaign.
        origin: Forecast origin used for the contrast.
        campaign_start: First campaign date.
        n_bootstrap: Number of whole-campaign resamples used for the interval.
        seed: Bootstrap random seed.
        alpha: Two-sided bootstrap tail probability.
        observed_driver_range: Driver range represented by observed campaigns.
        planned_driver_range: Driver range evaluated by the scenario.
        horizon: Positive number of forecast days against which window bounds are interpreted.
        window_bounds: Exact 1-based inclusive horizon-day bounds integrated into the effect.
        requested_window: The caller's selector. `None` records use of the compatibility default,
            distinct from an explicit `"horizon"` selection even though both integrate all days.
        requested_window_bounds: Exact bounds supplied for an event-support or custom request,
            before horizon clipping. `None` for implicit or explicit full-horizon integration.
        episode_validated: Whether a held-out whole-episode diagnostic established that this
            contrast procedure transfers across campaigns. Query rows labelled with campaign ids
            are not such a diagnostic and cannot make a model-derived association quotable.
        notes: Anything a reader must see before quoting the number.
    """

    bundle: str
    effect: float
    low: float
    high: float
    n_campaigns: int
    overlap: float
    origin_is_pre_treatment: bool
    origin: str
    campaign_start: str
    n_bootstrap: int
    seed: int
    alpha: float
    observed_driver_range: tuple[float, float]
    planned_driver_range: tuple[float, float]
    horizon: int
    window_bounds: tuple[int, int]
    requested_window_bounds: tuple[int, int] | None = None
    episode_validated: bool = False
    window: Literal["horizon", "event_support", "custom"] = "horizon"
    requested_window: Literal["horizon", "event_support", "custom"] | None = None
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Raises: ContrastError: the interval does not contain the estimate, or the origin was
        not pre-treatment -- the second is refused rather than warned about, because a
        post-treatment contrast understates the campaign silently."""
        if not isinstance(self.bundle, str) or not self.bundle.strip():
            raise ContrastError("contrast bundle must be a non-blank name")
        values = {"effect": self.effect, "low": self.low, "high": self.high,
                  "overlap": self.overlap}
        invalid = {name: value for name, value in values.items()
                   if not _is_finite_real(value)}
        if invalid:
            raise ContrastError(f"contrast evidence must be finite: {invalid}")
        if type(self.n_campaigns) is not int or self.n_campaigns < 1:
            raise ContrastError(
                f"n_campaigns must be a positive built-in integer, got {self.n_campaigns!r}")
        if not 0.0 <= self.overlap <= 1.0:
            raise ContrastError(f"overlap must lie in [0,1], got {self.overlap!r}")
        if not isinstance(self.origin_is_pre_treatment, bool) \
                or not isinstance(self.episode_validated, bool):
            raise ContrastError("pre-treatment and episode-validation flags must be booleans")
        origin_day = _canonical_date(self.origin, "origin")
        campaign_day = _canonical_date(self.campaign_start, "campaign_start")
        _validate_bootstrap_controls(self.n_bootstrap, self.seed, self.alpha)
        _validate_driver_range(self.observed_driver_range, "observed_driver_range")
        _validate_driver_range(self.planned_driver_range, "planned_driver_range")
        if type(self.horizon) is not int or self.horizon < 1:
            raise ContrastError(
                f"horizon must be a positive built-in integer, got {self.horizon!r}")
        if self.origin_is_pre_treatment is not (origin_day < campaign_day):
            raise ContrastError(
                "origin_is_pre_treatment contradicts the recorded origin and campaign_start")
        if type(self.window) is not str or self.window not in _WINDOWS:
            raise ContrastError(
                f"window must be exactly one of {sorted(_WINDOWS)}, got {self.window!r}")
        if (type(self.window_bounds) is not tuple or len(self.window_bounds) != 2
                or any(type(bound) is not int for bound in self.window_bounds)
                or not 1 <= self.window_bounds[0] <= self.window_bounds[1] <= self.horizon):
            raise ContrastError(
                "window_bounds must be exact 1-based inclusive built-in integer day bounds, "
                f"within horizon={self.horizon}, got {self.window_bounds!r}")
        if self.requested_window is None:
            if self.window != "horizon":
                raise ContrastError(
                    "requested_window=None is the compatibility horizon default and cannot "
                    f"resolve to window={self.window!r}")
        elif (type(self.requested_window) is not str
              or self.requested_window not in _WINDOWS
              or self.requested_window != self.window):
            raise ContrastError(
                "requested_window must be None or exactly match the resolved window; got "
                f"{self.requested_window!r} and {self.window!r}")
        if self.requested_window in (None, "horizon"):
            if self.requested_window_bounds is not None:
                raise ContrastError(
                    "requested_window_bounds must be None for implicit or explicit horizon")
            expected_bounds = (1, self.horizon)
        elif self.requested_window == "custom":
            _validate_window_request("custom", None, self.requested_window_bounds)
            assert self.requested_window_bounds is not None
            if self.requested_window_bounds[1] > self.horizon:
                raise ContrastError(
                    f"custom requested_window_bounds {self.requested_window_bounds} exceed "
                    f"horizon={self.horizon}")
            expected_bounds = self.requested_window_bounds
        else:
            _validate_window_request("event_support", self.requested_window_bounds, None)
            assert self.requested_window_bounds is not None
            offset = (campaign_day - origin_day).days
            first = offset + self.requested_window_bounds[0]
            last = offset + self.requested_window_bounds[1]
            expected_bounds = (max(first, 1), min(last, self.horizon))
            if expected_bounds[0] > expected_bounds[1]:
                raise ContrastError(
                    "requested event support lies entirely outside the recorded horizon")
        if self.window_bounds != expected_bounds:
            raise ContrastError(
                f"window_bounds {self.window_bounds!r} do not match requested window "
                f"{self.requested_window!r} bounds {self.requested_window_bounds!r}; expected "
                f"{expected_bounds!r}")
        if (not isinstance(self.notes, list)
                or any(not isinstance(note, str) or not note.strip() for note in self.notes)):
            raise ContrastError("contrast notes must be non-blank strings")
        # Tolerance relative to the estimate: the point estimate sums every cell once while the
        # bootstrap sums resampled campaigns, so the two accumulate float error differently. A
        # strict comparison fails on a difference of 1e-6 that means nothing.
        slack = max(abs(self.effect), 1.0) * 1e-6
        if not (self.low - slack) <= self.effect <= (self.high + slack):
            raise ContrastError(
                f"effect {self.effect} lies outside its own interval [{self.low}, {self.high}]")
        if not self.origin_is_pre_treatment:
            raise ContrastError(
                "the origin is not before the campaign. Re-running from inside it puts the sales "
                "the campaign caused into the encoder, so the counterfactual has already seen the "
                "effect it is meant to remove -- and the contrast understates it silently")

    @property
    def is_quotable(self) -> bool:
        """Whether this number may be shown to a client.

        Not a formatting question. Insufficient overlap or too few campaigns means the model is
        extrapolating, and a number produced by extrapolation looks exactly like one produced by
        evidence.
        """
        return (self.n_campaigns >= 5 and self.overlap >= 0.5 and self.episode_validated
                and not self.notes)

    def __str__(self) -> str:
        mark = "" if self.is_quotable else "  [NOT QUOTABLE]"
        return (f"{self.bundle}: {self.effect:+.1f} units "
                f"[{self.low:+.1f}, {self.high:+.1f}] from {self.n_campaigns} campaigns, "
                f"overlap {self.overlap:.0%}{mark}")

    def as_record(self) -> dict[str, object]:
        return {"bundle": self.bundle, "effect": self.effect, "low": self.low, "high": self.high,
                "n_campaigns": self.n_campaigns, "overlap": self.overlap,
                "origin": self.origin, "campaign_start": self.campaign_start,
                "origin_is_pre_treatment": self.origin_is_pre_treatment,
                "n_bootstrap": self.n_bootstrap, "seed": self.seed, "alpha": self.alpha,
                "observed_driver_range": list(self.observed_driver_range),
                "planned_driver_range": list(self.planned_driver_range),
                "horizon": self.horizon,
                "episode_validated": self.episode_validated,
                "window": self.window,
                "requested_window": self.requested_window,
                "window_bounds": list(self.window_bounds),
                "requested_window_bounds": (None if self.requested_window_bounds is None else
                                            list(self.requested_window_bounds)),
                "quotable": self.is_quotable, "notes": self.notes,
                "estimand": "prospective scenario contrast, NOT a causal effect"}


def _apply_window(per_cell: torch.Tensor, *,
                  window: Literal["horizon", "event_support", "custom"] | None, origin,
                  campaign_start, event_support: tuple[int, int] | None,
                  custom_window: tuple[int, int] | None,
                  horizon: int,
                  ) -> tuple[torch.Tensor, str, tuple[int, int], list[str]]:
    """Return integrated effects, resolved selector, 1-based inclusive bounds, and notes.

    `attribution_window` was a spec field nothing read: contrast() summed every horizon cell
    regardless, which is silently the 'horizon' choice -- the one the spec warns counts a
    promotion's spike and none of its payback beyond the edge.
    """
    if per_cell.dim() == 1:
        if window in (None, "horizon"):
            # Per-cell totals CAN only integrate the whole horizon; the recorded window field
            # says so. The truncation caveat rides the notes only when the caller explicitly
            # CHOSE horizon -- notes gate quotability, and an undeclared window on output that
            # permits no other is a recorded fact, not a disqualification.
            notes = [] if window is None else [
                "attribution_window='horizon': truncated at the forecast edge BY CHOICE, "
                "so a payback beyond it is never debited"]
            return per_cell, "horizon", (1, horizon), notes
        raise ContrastError(
            f"window={window!r} needs per-day model output [cells, horizon]; the model_fn "
            "returned per-cell totals, which cannot be integrated over a sub-window")
    if window in (None, "horizon"):
        note = ("no attribution window was declared; integrating every horizon day"
                ) if window is None else (
                "attribution_window='horizon': truncated at the forecast edge BY CHOICE, so a "
                "payback beyond it is never debited")
        return per_cell.sum(dim=1), "horizon", (1, horizon), [note]
    if window == "custom":
        assert custom_window is not None  # validated before model evaluation
        first, last = custom_window
        if last > horizon:
            raise ContrastError(
                f"custom_window {custom_window} must satisfy 1 <= first <= last <= {horizon}")
        return per_cell[:, first - 1:last].sum(dim=1), "custom", (first, last), []
    if window == "event_support":
        assert event_support is not None  # validated before model evaluation
        offset = (campaign_start - origin).days           # campaign start as a 1-based day index
        first = offset + event_support[0]
        last = offset + event_support[1]
        clipped_first, clipped_last = max(first, 1), min(last, horizon)
        if clipped_first > clipped_last:
            raise ContrastError(
                f"the event support around day {offset} lies entirely outside the "
                f"{horizon}-day horizon; there is nothing to integrate")
        notes = []
        if last > horizon:
            notes.append(
                f"the event support reaches day {last} but the horizon ends at {horizon}: part "
                "of any payback lies beyond the forecast, so this contrast is GROSS, not net, "
                "and must not be presented as the promotion's net effect")
        if first < 1:
            notes.append(
                f"the support's first {1 - first} pre-campaign day(s) precede the forecast "
                "origin and are excluded; anticipation before the origin is not in this window")
        return (per_cell[:, clipped_first - 1:clipped_last].sum(dim=1), "event_support",
                (clipped_first, clipped_last), notes)
    raise ContrastError(
        f"unknown window {window!r}; the spec defines event_support, horizon and custom")


def _mean_under(model_fn, known_future: torch.Tensor, draws: torch.Tensor) -> torch.Tensor:
    """Expected demand under a given known-future tensor, on FIXED random draws.

    The draws are passed in and reused across both scenarios. That is what makes the difference a
    contrast rather than two independent estimates whose gap is mostly sampling noise -- and it is
    also why the paired spread must NOT be quoted as a confidence interval: it is narrow precisely
    because the noise cancels.
    """
    return model_fn(known_future, draws)


def contrast(
    model_fn,
    known_future: torch.Tensor,
    bundle: InterventionBundle,
    draws: torch.Tensor,
    *,
    origin: str,
    campaign_start: str,
    campaign_index: torch.Tensor,
    observed_driver_range: tuple[float, float],
    planned_driver_range: tuple[float, float],
    n_bootstrap: int = 200,
    seed: int = 0,
    alpha: float = 0.1,
    episode_validated: bool = False,
    window: Literal["horizon", "event_support", "custom"] | None = None,
    event_support: tuple[int, int] | None = None,
    custom_window: tuple[int, int] | None = None,
) -> Contrast:
    """Compute a scenario contrast with its campaign-bootstrap spread and its overlap check.

    Args:
        model_fn: `(known_future, draws) -> expected demand per cell`. Takes the SAME draws under
            both scenarios.
        known_future: `[cells, horizon, n_known]` as planned.
        bundle: What to set to reference.
        draws: Fixed random draws, reused across scenarios.
        origin: Forecast origin, ISO.
        campaign_start: When the campaign begins, ISO. Must be after the origin.
        campaign_index: `[cells]` which campaign each cell belongs to. The bootstrap resamples
            THESE, because the effective sample size is episodes rather than rows. Exact int32
            or int64 labels are required so floating/unsigned aliases cannot merge identifiers.
        observed_driver_range: `(min, max)` of the driver across campaigns actually seen.
        planned_driver_range: `(min, max)` the client is asking about.
        n_bootstrap: Campaign resamples.
        seed: Reproducibility.
        alpha: Two-sided bootstrap tail probability.
        episode_validated: True only when a separate held-out whole-campaign diagnostic has
            validated this contrast procedure. Campaign labels on the query rows do not qualify.
        window: OVER WHICH DAYS the contrast integrates -- `ForecastModelSpec.attribution_window`,
            and the choice can invert the answer: a promotion produces a spike then a payback
            dip, and integrating the promotion week counts the spike and none of the payback.
            'event_support' integrates the fitted kernel's support around `campaign_start`
            (requires `event_support` and per-day model output); 'horizon' integrates every
            forecast day, truncated at the forecast edge BY CHOICE and noted as such; 'custom'
            takes explicit day bounds. None -- the compatibility default -- behaves as 'horizon'
            and says so in the notes, because a window nobody declared is not a window.
        event_support: `(lower, upper)` signed-day support around the campaign start, e.g.
            `(-30, 30)`. Days beyond the horizon are clipped and the contrast is labelled GROSS:
            the payback outside the forecast is never debited.
        custom_window: `(first_day, last_day)`, 1-based horizon day indices, inclusive.

    Returns:
        A `Contrast`.

    Raises:
        ContrastError: the origin is not pre-treatment, or fewer than two campaigns exist -- a
            bootstrap over one campaign returns that campaign's estimate with zero width, which
            reads as certainty.
    """
    origin_day = _canonical_date(origin, "origin")
    campaign_day = _canonical_date(campaign_start, "campaign_start")
    _validate_bootstrap_controls(n_bootstrap, seed, alpha)
    _validate_driver_range(observed_driver_range, "observed_driver_range")
    _validate_driver_range(planned_driver_range, "planned_driver_range")
    _validate_window_request(window, event_support, custom_window)
    if (not isinstance(known_future, torch.Tensor) or known_future.dim() != 3
            or not known_future.is_floating_point()):
        raise ContrastError(
            "known_future must be a floating rank-3 tensor [cells, horizon, n_known]")
    if any(size < 1 for size in known_future.shape):
        raise ContrastError(
            "known_future [cells, horizon, n_known] axes must all be non-empty, got "
            f"{tuple(known_future.shape)}")
    if not isinstance(draws, torch.Tensor):
        raise ContrastError(f"draws must be a torch.Tensor, got {type(draws).__name__}")
    if not bool(torch.isfinite(known_future).all()) or not bool(torch.isfinite(draws).all()):
        raise ContrastError("known_future and shared draws must be finite")
    if not isinstance(bundle, InterventionBundle):
        raise ContrastError(f"bundle must be an InterventionBundle, got {type(bundle).__name__}")
    if any(column >= known_future.shape[-1] for column in bundle.columns):
        raise ContrastError(
            f"bundle {bundle.name!r} indexes columns {bundle.columns} outside known_future "
            f"width {known_future.shape[-1]}")
    campaigns = _validate_campaign_index(
        campaign_index, n_cells=known_future.shape[0], device=known_future.device)
    if not isinstance(episode_validated, bool):
        raise ContrastError("episode_validated must be a boolean from an external validation")
    pre_treatment = origin_day < campaign_day

    counterfactual = known_future.clone()
    if bundle.reference_values is not None:
        if bundle.reference_values.shape[:-1] != known_future.shape[:-1]:
            raise ContrastError(
                f"bundle reference_values leading shape "
                f"{tuple(bundle.reference_values.shape[:-1])} does not align with known_future "
                f"{tuple(known_future.shape[:-1])}")
        if (bundle.reference_values.dtype != known_future.dtype
                or bundle.reference_values.device != known_future.device):
            raise ContrastError(
                "bundle reference_values must have the same floating dtype and device as "
                "known_future")
    for j, col in enumerate(bundle.columns):
        if bundle.reference_values is not None:
            counterfactual[..., col] = bundle.reference_values[..., j]
        else:
            counterfactual[..., col] = bundle.reference
    factual = _mean_under(model_fn, known_future, draws)
    reference = _mean_under(model_fn, counterfactual, draws)
    expected_shapes = {
        (known_future.shape[0],),
        (known_future.shape[0], known_future.shape[1]),
    }
    if (not isinstance(factual, torch.Tensor) or not isinstance(reference, torch.Tensor)
            or tuple(factual.shape) not in expected_shapes
            or tuple(reference.shape) not in expected_shapes
            or factual.shape != reference.shape):
        raise ContrastError(
            "factual and reference model outputs must be aligned tensors shaped exactly "
            f"[cells] or [cells, horizon] ({known_future.shape[0]} cells, "
            f"{known_future.shape[1]} horizon days); got "
            f"{getattr(factual, 'shape', None)} and {getattr(reference, 'shape', None)}")
    if not factual.is_floating_point() or not reference.is_floating_point():
        raise ContrastError(
            "factual and reference model outputs must be floating expected-demand tensors")
    if factual.dtype != reference.dtype or factual.device != reference.device:
        raise ContrastError(
            "factual and reference model outputs must have the same floating dtype and device")
    if factual.device != known_future.device:
        raise ContrastError(
            "factual and reference model outputs must be on the same device as known_future")
    if not bool(torch.isfinite(factual).all()) or not bool(torch.isfinite(reference).all()):
        raise ContrastError("factual and reference model outputs must be finite")
    per_cell = factual - reference
    if not bool(torch.isfinite(per_cell).all()):
        raise ContrastError(
            "model scenario difference must have one finite leading value per query cell")
    per_cell, window_label, window_bounds, window_notes = _apply_window(
        per_cell, window=window, origin=origin_day, campaign_start=campaign_day,
        event_support=event_support, custom_window=custom_window,
        horizon=known_future.shape[1])
    effect = _finite_tensor_sum(
        per_cell, "finite per-cell effects overflowed while forming the overall contrast")

    low, high = campaign_bootstrap(
        per_cell, campaign_index, n_bootstrap=n_bootstrap, seed=seed, alpha=alpha)
    overlap = _overlap(observed_driver_range, planned_driver_range)

    notes = list(window_notes)
    # A degenerate interval is a FINDING, not a rounding artefact. It means every campaign
    # produced the same effect, so resampling them tells you nothing -- which is the "count is
    # not identification" problem showing up as a width rather than as a count. Reported here
    # because an interval of [16.0000041, 16.0000050] reads as extraordinary precision.
    width = high - low
    if abs(effect) > 0 and width / max(abs(effect), 1e-12) < 1e-4:
        notes.append(
            f"the campaign interval has width {width:.2e} on an effect of {effect:.2f} -- the "
            "campaigns are indistinguishable, so resampling them measures nothing. This reads as "
            "precision and is its absence")
    if overlap < 0.5:
        notes.append(
            f"the planned range is only {overlap:.0%} covered by campaigns actually observed; "
            "outside that the model is extrapolating and the number looks the same either way")
    if campaigns.numel() < 5:
        notes.append(f"only {campaigns.numel()} campaigns; episode COUNT is not identification, "
                     "and this few cannot separate the driver from what co-moves with it")
    if not episode_validated:
        notes.append(
            "no held-out whole-episode validation was supplied. Query campaign labels and a "
            "bootstrap describe heterogeneity in this modelled contrast; they do not establish "
            "that it transfers across historical campaigns")
    return Contrast(bundle=bundle.name, effect=effect, low=low, high=high,
                    n_campaigns=int(campaigns.numel()), overlap=overlap,
                    origin_is_pre_treatment=pre_treatment,
                    origin=origin, campaign_start=campaign_start,
                    n_bootstrap=n_bootstrap, seed=seed, alpha=float(alpha),
                    observed_driver_range=tuple(float(v) for v in observed_driver_range),
                    planned_driver_range=tuple(float(v) for v in planned_driver_range),
                    horizon=known_future.shape[1], window_bounds=window_bounds,
                    requested_window_bounds=(event_support if window == "event_support" else
                                             custom_window if window == "custom" else None),
                    episode_validated=episode_validated, notes=notes, window=window_label,
                    requested_window=window)


def campaign_bootstrap(
    per_cell: torch.Tensor, campaign_index: torch.Tensor, *, n_bootstrap: int = 200,
    seed: int = 0, alpha: float = 0.1,
) -> tuple[float, float]:
    """Resample WHOLE CAMPAIGNS to get the spread. Returns `(low, high)`.

    **Campaigns, not cells.** A promotion spans days and stores, so the cells within one campaign
    are anything but independent -- resampling them would produce an interval that shrinks with
    the length of the campaign rather than with the amount of evidence, and it would be
    spectacularly narrow.

    This is also not the paired-draw spread. That one is narrow BY CONSTRUCTION because the same
    random draws appear under both scenarios and the noise cancels; it was quoted as a confidence
    interval here once, and it is not one.

    Raises:
        ContrastError: invalid controls, effects or exact int32/int64 campaign labels; fewer than
            two campaigns; or a campaign/bootstrap aggregate outside finite float range.
    """
    _validate_bootstrap_controls(n_bootstrap, seed, alpha)
    if not isinstance(per_cell, torch.Tensor):
        raise ContrastError(
            f"per_cell must be a torch.Tensor, got {type(per_cell).__name__}")
    if per_cell.dim() != 1:
        raise ContrastError(
            f"per_cell must be a rank-1 tensor with one effect per cell, got {per_cell.shape}")
    if not per_cell.is_floating_point():
        raise ContrastError(
            f"per_cell must contain floating scenario effects, got {per_cell.dtype}")
    campaigns = _validate_campaign_index(
        campaign_index, n_cells=per_cell.shape[0], device=per_cell.device)
    if not bool(torch.isfinite(per_cell).all()):
        raise ContrastError("campaign effects must be finite before bootstrapping")
    campaign_totals = [
        _finite_tensor_sum(
            per_cell[campaign_index == campaign],
            "finite per-cell effects overflowed while forming whole-campaign totals",
        )
        for campaign in campaigns
    ]
    g = torch.Generator().manual_seed(seed)
    totals = []
    for _ in range(n_bootstrap):
        picks = torch.randint(len(campaigns), (len(campaigns),), generator=g)
        total = _finite_float_sum(
            [campaign_totals[int(index)] for index in picks],
            "finite campaign effects overflowed while forming bootstrap totals",
        )
        totals.append(total)
    ordered = sorted(totals)
    lo = ordered[int(alpha / 2 * len(ordered))]
    hi = ordered[min(len(ordered) - 1, int((1 - alpha / 2) * len(ordered)))]
    return lo, hi


def _overlap(observed: tuple[float, float], planned: tuple[float, float]) -> float:
    """Share of the planned driver range covered by what was actually observed.

    The check that separates interpolation from extrapolation. A client asking about a 30%
    discount when nothing above 15% has ever run is asking the model to invent, and the answer
    looks identical to one built on evidence.
    """
    lo = max(observed[0], planned[0])
    hi = min(observed[1], planned[1])
    width = planned[1] - planned[0]
    if width <= 0:
        return 1.0 if observed[0] <= planned[0] <= observed[1] else 0.0
    if hi <= lo:
        return 0.0
    covered = hi - lo
    if not math.isfinite(width) or not math.isfinite(covered):
        # Finite endpoints can still overflow when opposite-signed extremes are subtracted.
        # Scaling every clipped endpoint by the same positive value preserves the interval ratio.
        scale = max(abs(planned[0]), abs(planned[1]), 1.0)
        width = planned[1] / scale - planned[0] / scale
        covered = hi / scale - lo / scale
    return min(1.0, max(0.0, covered / width))
