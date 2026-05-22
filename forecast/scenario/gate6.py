"""Gate 6: the evidence that lets a scenario contrast be QUOTED, produced nowhere else.

`contrast()` accepts `episode_validated` and its docstring has always said what qualifies:
"a separate held-out whole-campaign diagnostic". Until this module existed nothing could
produce that diagnostic, so every contrast was permanently unquotable -- correct, and useless.
This is the machinery, and it is deliberately the ONLY sanctioned producer of
`episode_validated=True`:

  1. **Held-out campaign evaluation.** For each historical campaign, the caller forecasts from
     a PRE-campaign origin under the factual plan and records the realised outcome over the
     campaign window. The per-campaign error -- realised minus predicted, relative to realised
     scale -- is the direct measurement of the thing a contrast asks the model to do: predict a
     campaign it did not train through.
  2. **Campaign-level error bootstrap.** Resampling whole campaigns' errors gives the honest
     band for future contrasts -- campaigns, never cells, because the days of one campaign are
     not replications of it.
  3. **Pretrend diagnostic.** Treated and untreated series must have been MOVING TOGETHER
     before the campaign; a standardised gap in their pre-window trends means the "effect" was
     already underway before the lever moved.
  4. **Assignment balance.** Retailers promote what is already selling: a standardised mean
     difference in baseline demand between treated and untreated says how far the assignment is
     from comparable, which is the confound the spec's own docstrings name.

Thresholds are REGISTERED inputs recorded with the evidence, not derived from the results --
a bar chosen after seeing the numbers is a description, not a gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

__all__ = ["Gate6Error", "Gate6Evidence", "Gate6Thresholds", "assignment_imbalance",
           "campaign_error_bootstrap", "evaluate", "load_evidence", "pretrend_gap"]

_MIN_CAMPAIGNS = "min_campaigns"
_ABSOLUTE_ERROR = "median_absolute_relative_error"
_PRETREND = "pretrend"
_IMBALANCE = "assignment_imbalance"


def _is_finite_number(value: object) -> bool:
    """Strict JSON-number boundary: bool is not a measurement and huge ints fail closed."""
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _is_finite_float(value: object) -> bool:
    return type(value) is float and math.isfinite(value)


class Gate6Error(ValueError):
    """The evidence requested cannot mean what it would appear to mean."""


@dataclass(frozen=True)
class Gate6Thresholds:
    """Registered thresholds for held-out campaign replication, relative prediction error,
    pre-treatment trends and baseline imbalance. Set them before running the evaluation.
    """

    min_campaigns: int = 5
    max_absolute_relative_error: float = 0.30
    max_pretrend: float = 0.25
    max_imbalance: float = 0.25

    def __post_init__(self) -> None:
        if type(self.min_campaigns) is not int or self.min_campaigns < 2:
            raise Gate6Error("min_campaigns must be a built-in integer >= 2")
        for name in (
            "max_absolute_relative_error", "max_pretrend", "max_imbalance",
        ):
            value = getattr(self, name)
            if not _is_finite_float(value) or value < 0.0:
                raise Gate6Error(
                    f"{name} must be a finite non-negative built-in float, got {value!r}")


@dataclass(frozen=True)
class Gate6Evidence:
    """The record a quotable contrast must cite. `episode_validated` is derived, never set.

    Attributes:
        campaign_ids: The held-out campaigns evaluated.
        relative_errors: Per-campaign (realised - predicted) / max(|realised|, 1).
        error_low, error_high: Campaign-level bootstrap band over the relative errors.
        pretrend: Standardised pre-window trend gap.
        imbalance: Standardised baseline difference at assignment.
        thresholds: The registered bars this evidence was judged against.
        notes: Human-readable renderings of the derived stable failure codes.
    """

    campaign_ids: tuple[str, ...]
    relative_errors: tuple[float, ...]
    error_low: float
    error_high: float
    pretrend: float
    imbalance: float
    thresholds: Gate6Thresholds
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if (type(self.campaign_ids) is not tuple or len(self.campaign_ids) < 2
                or any(type(value) is not str or value != value.strip() or not value
                       for value in self.campaign_ids)
                or len(set(self.campaign_ids)) != len(self.campaign_ids)):
            raise Gate6Error("campaign_ids must be at least two unique canonical strings")
        if (type(self.relative_errors) is not tuple
                or len(self.relative_errors) != len(self.campaign_ids)
                or any(not _is_finite_float(value) for value in self.relative_errors)):
            raise Gate6Error("relative_errors must contain one finite built-in float per campaign")
        if (not _is_finite_float(self.error_low) or not _is_finite_float(self.error_high)
                or self.error_low > self.error_high):
            raise Gate6Error("bootstrap bounds must be finite ordered built-in floats")
        for name in ("pretrend", "imbalance"):
            value = getattr(self, name)
            if not _is_finite_float(value) or value < 0.0:
                raise Gate6Error(
                    f"{name} must be a finite non-negative built-in float, got {value!r}")
        if type(self.thresholds) is not Gate6Thresholds:
            raise Gate6Error("thresholds must be a Gate6Thresholds contract")
        _, expected_notes = _failures(
            self.campaign_ids, self.relative_errors, self.pretrend, self.imbalance,
            self.thresholds)
        if type(self.notes) is not tuple or self.notes != expected_notes:
            raise Gate6Error(
                "failure notes must be the derived renderings of the recorded measurements")

    @property
    def failure_codes(self) -> tuple[str, ...]:
        """Stable machine-readable failures derived from measurements and registered bars."""
        return _failures(
            self.campaign_ids, self.relative_errors, self.pretrend, self.imbalance,
            self.thresholds)[0]

    @property
    def episode_validated(self) -> bool:
        return not self.failure_codes

    def as_record(self) -> dict[str, object]:
        return {"campaign_ids": list(self.campaign_ids),
                "relative_errors": list(self.relative_errors),
                "error_low": self.error_low, "error_high": self.error_high,
                "pretrend": self.pretrend, "imbalance": self.imbalance,
                "thresholds": {"min_campaigns": self.thresholds.min_campaigns,
                               "max_absolute_relative_error":
                                   self.thresholds.max_absolute_relative_error,
                               "max_pretrend": self.thresholds.max_pretrend,
                               "max_imbalance": self.thresholds.max_imbalance},
                "episode_validated": self.episode_validated,
                "failure_codes": list(self.failure_codes),
                "notes": list(self.notes)}


def _failures(
    campaign_ids: tuple[str, ...], errors: tuple[float, ...], trend: float, balance: float,
    thresholds: Gate6Thresholds,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Derive stable decision codes and their human renderings from measurements and bars."""
    codes, notes = [], []
    if len(campaign_ids) < thresholds.min_campaigns:
        codes.append(_MIN_CAMPAIGNS)
        notes.append(
            f"{len(campaign_ids)} held-out campaigns against a registered minimum of "
            f"{thresholds.min_campaigns}; episode count is not identification, and this few "
            "cannot separate the driver from what co-moves with it")
    # This preserves the existing implementation's even-sample convention. The conventional
    # median ambiguity is reported for owner resolution rather than changed during hardening.
    median_abs = sorted(abs(error) for error in errors)[len(errors) // 2]
    if median_abs > thresholds.max_absolute_relative_error:
        codes.append(_ABSOLUTE_ERROR)
        notes.append(
            f"median held-out campaign error {median_abs:.0%} exceeds the registered "
            f"{thresholds.max_absolute_relative_error:.0%}; a model that cannot predict a "
            "held-out campaign cannot contrast one")
    if trend > thresholds.max_pretrend:
        codes.append(_PRETREND)
        notes.append(
            f"pre-window trend gap {trend:.2f} standard deviations exceeds "
            f"{thresholds.max_pretrend:.2f}; the 'effect' was underway before the lever moved")
    if balance > thresholds.max_imbalance:
        codes.append(_IMBALANCE)
        notes.append(
            f"assignment imbalance {balance:.2f} standard deviations exceeds "
            f"{thresholds.max_imbalance:.2f}; treated series were already different, which is "
            "the promote-what-sells confound")
    return tuple(codes), tuple(notes)


def pretrend_gap(treated_pre: torch.Tensor, untreated_pre: torch.Tensor) -> float:
    """Standardised difference of mean pre-window TRENDS. Zero means moving together.

    Trend per series is the OLS slope over the pre-window days; the gap is the difference of
    group mean slopes, standardised by the pooled slope spread -- so "0.25" means a quarter of
    a standard deviation, whatever the units.

    Raises:
        Gate6Error: fewer than two days, or an empty group -- a trend of one point and a gap
            against nobody both look like zero and are its absence.
    """
    for name, group in (("treated", treated_pre), ("untreated", untreated_pre)):
        if not isinstance(group, torch.Tensor) or group.dim() != 2 or group.shape[0] < 1 \
                or group.shape[1] < 2:
            raise Gate6Error(
                f"{name} pre-window must be [series, days>=2]; got "
                f"{getattr(group, 'shape', None)}")
        if group.dtype == torch.bool or group.is_complex() or not bool(torch.isfinite(group).all()):
            raise Gate6Error(f"{name} pre-window must contain finite real demand")
    def slopes(group: torch.Tensor) -> torch.Tensor:
        days = torch.arange(group.shape[1], dtype=torch.float64)
        days = days - days.mean()
        centred = group.double() - group.double().mean(dim=1, keepdim=True)
        return (centred * days).sum(dim=1) / (days * days).sum()
    treated, untreated = slopes(treated_pre), slopes(untreated_pre)
    difference = float((treated.mean() - untreated.mean()).abs())
    pooled = float(torch.cat([treated, untreated]).std(unbiased=False))
    if pooled < 1e-6:
        # Every series trends identically: standardising by a zero spread is 0/0 dressed as a
        # gap. Identical group means are exactly parallel (zero); different means with no
        # within-spread are deterministic divergence, which no threshold should pass.
        return 0.0 if difference < 1e-6 else float("inf")
    return difference / pooled


def assignment_imbalance(treated_baseline: torch.Tensor,
                         untreated_baseline: torch.Tensor) -> float:
    """Standardised mean difference of baseline demand at assignment -- the promote-what-sells
    confound, measured. Zero means comparable groups.

    Raises:
        Gate6Error: an empty group.
    """
    for name, group in (("treated", treated_baseline), ("untreated", untreated_baseline)):
        if not isinstance(group, torch.Tensor) or group.numel() < 1:
            raise Gate6Error(f"{name} baseline group is empty; balance against nobody is zero "
                             "and means nothing")
        if group.dtype == torch.bool or group.is_complex() or not bool(torch.isfinite(group).all()):
            raise Gate6Error(f"{name} baseline must contain finite real demand")
    pooled = float(torch.cat([treated_baseline.double().flatten(),
                              untreated_baseline.double().flatten()]).std(unbiased=False))
    difference = float((treated_baseline.double().mean()
                        - untreated_baseline.double().mean()).abs())
    if pooled < 1e-6:
        # Zero within-group spread: identical groups are perfectly balanced; identical-within
        # but different-between is deterministic separation no threshold should pass.
        return 0.0 if difference < 1e-6 else float("inf")
    return difference / pooled


def campaign_error_bootstrap(errors: list[float], *, n_bootstrap: int = 2000, seed: int = 0,
                             alpha: float = 0.1) -> tuple[float, float]:
    """Bootstrap band over CAMPAIGN-level errors. Campaigns are the replication unit.

    Raises:
        Gate6Error: fewer than two campaigns -- a band over one is its own point.
    """
    if type(errors) is not list or len(errors) < 2:
        raise Gate6Error("a bootstrap over fewer than two campaigns has zero width by "
                         "construction, which reads as certainty")
    if any(not _is_finite_float(error) for error in errors):
        raise Gate6Error(f"campaign errors must be finite built-in floats, got {errors}")
    if type(n_bootstrap) is not int or n_bootstrap < 1:
        raise Gate6Error("n_bootstrap must be a positive built-in integer")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise Gate6Error("seed must be a built-in integer in [0, 2**63)")
    if type(alpha) is not float or not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise Gate6Error("alpha must be a finite built-in float in (0, 1)")
    generator = torch.Generator().manual_seed(seed)
    values = torch.tensor(errors, dtype=torch.float64)
    picks = torch.randint(len(errors), (n_bootstrap, len(errors)), generator=generator)
    means = values[picks].mean(dim=1)
    ordered = means.sort().values
    lo = float(ordered[int(alpha / 2 * n_bootstrap)])
    hi = float(ordered[min(n_bootstrap - 1, int((1 - alpha / 2) * n_bootstrap))])
    return lo, hi


def evaluate(
    *,
    campaigns: list[tuple[str, float, float]],
    treated_pre: torch.Tensor,
    untreated_pre: torch.Tensor,
    treated_baseline: torch.Tensor,
    untreated_baseline: torch.Tensor,
    thresholds: Gate6Thresholds | None = None,
    seed: int = 0,
) -> Gate6Evidence:
    """Judge the held-out evidence against the registered bars. Returns the record.

    Args:
        campaigns: `(campaign_id, realised_total, predicted_total)` per HELD-OUT campaign --
            each predicted from a pre-campaign origin under the factual plan, by the caller,
            which is the expensive part and the part that cannot be faked cheaply here.
        treated_pre: `[series, days]` pre-window demand of treated series.
        untreated_pre: Same for untreated.
        treated_baseline: `[series]` baseline demand of treated series at assignment.
        untreated_baseline: Same for untreated.
        thresholds: The bars to record with this non-consumable diagnostic record. File
            consumption remains disabled until an external registration exists.
        seed: Bootstrap reproducibility.

    Raises:
        Gate6Error: malformed campaign tuples or non-finite outcomes.
    """
    if thresholds is None:
        thresholds = Gate6Thresholds()
    elif type(thresholds) is not Gate6Thresholds:
        raise Gate6Error("thresholds must be a Gate6Thresholds contract")
    if type(campaigns) is not list or not campaigns:
        raise Gate6Error("no held-out campaigns; evidence of nothing validates nothing")
    ids, errors = [], []
    for entry in campaigns:
        if type(entry) not in (tuple, list) or len(entry) != 3:
            raise Gate6Error(f"each campaign is (id, realised, predicted); got {entry!r}")
        campaign_id, realised, predicted = entry
        if (type(campaign_id) is not str or campaign_id != campaign_id.strip()
                or not campaign_id or campaign_id in ids):
            raise Gate6Error(
                f"campaign ids must be unique canonical strings, got {campaign_id!r}")
        if not all(_is_finite_number(value) for value in (realised, predicted)):
            raise Gate6Error(
                f"campaign {campaign_id!r} outcomes must be finite built-in numbers")
        realised_float, predicted_float = float(realised), float(predicted)
        error = ((realised_float - predicted_float)
                 / max(abs(realised_float), 1.0))
        if not math.isfinite(error):
            raise Gate6Error(f"campaign {campaign_id!r} produced a non-finite relative error")
        ids.append(campaign_id)
        errors.append(error)
    low, high = campaign_error_bootstrap(errors, seed=seed)
    trend = pretrend_gap(treated_pre, untreated_pre)
    balance = assignment_imbalance(treated_baseline, untreated_baseline)

    ids_tuple, errors_tuple = tuple(ids), tuple(errors)
    _, notes = _failures(ids_tuple, errors_tuple, trend, balance, thresholds)
    return Gate6Evidence(campaign_ids=ids_tuple, relative_errors=errors_tuple,
                         error_low=low, error_high=high, pretrend=trend, imbalance=balance,
                         thresholds=thresholds, notes=notes)


def load_evidence(path) -> dict[str, object]:
    """Refuse file evidence until Gate 6 has a real external authority boundary.

    A JSON file can repeat measurements, thresholds, verdicts and hashes consistently without
    proving that any campaign was held out or any threshold was registered beforehand. No Gate-6
    registration or governed backtest-source artifact exists in this repository, so parsing one
    would make a self-authored record look authoritative.

    Raises:
        Gate6Error: always, until the upgrade trigger below is satisfied.
    """
    # File consumption requires independently registered thresholds and verified source evidence.
    raise Gate6Error(
        "Gate 6 evidence consumption is disabled: add a committed Gate-6 registration and "
        "governed backtest-source binding, pin their immutable identities, then add "
        "a loader that verifies those external bytes. Self-authored JSON is not authority; "
        f"refused {path}")
