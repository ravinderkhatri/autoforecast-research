"""Calibrated cumulative lead-time coverage: is the interval an ordering decision rests on honest?

**COVERAGE USES A q-SPECIFIC RANDOMISED BOUNDARY.** Demand is integer, so the forecast quantile is
an atom and `P(Y <= Q(q))` overshoots its nominal level by up to that mass. Giving every boundary
observation a fixed half weight (mid-P) works only when q lies halfway through the atom; it calls a
calibrated deterministic-zero forecast 50% covered at every level. Instead the boundary weight is
`(q - F(Q^-)) / P(Y=Q)`, estimated from the paths. Under the forecast distribution this makes
coverage exactly q even for a point mass. Plain operational coverage is reported too.

**GATE 4 REQUIRES THIS, AND IT CANNOT BE COMPUTED FROM A POINT FORECAST.** That is the whole
reason the model emits sample paths rather than quantiles, so this is where that choice is
cashed in or found wanting.

WHAT AN ORDERING DECISION ACTUALLY ASKS. Not "how much will sell on Thursday" but "how much will
sell between now and the next delivery" -- demand over the LEAD TIME plus the REVIEW PERIOD. The
replenishment system takes a quantile of that distribution to hit a service level, so the quantile
is the number that decides how much stock is bought.

**WHY PER-DAY QUANTILES CANNOT BE SUMMED INTO IT.** Adding each day's 95th percentile gives a
number far above the 95th percentile of the sum, by an amount that depends on the dependence
between days. Independent days would need the sum's variance to grow as the sum of variances;
correlated ones grow faster. Neither generally equals the naive sum of marginal quantiles.
Quantiles are not additive or universally subadditive: depending on dependence and tail mass,
summing daily quantiles can over-order OR under-order. That ambiguity is why coherent paths are
required, not a presumed direction of error.

A path has the dependence in it, so the sum is simply taken. That is what `factor.py`'s AR(1) is
FOR, and this module is the test of whether it worked: if the factor's persistence is wrong, the
lead-time intervals will be too narrow or too wide and the coverage will say so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from autoforecast._numeric import is_finite_real

__all__ = ["Calibration", "CalibrationError", "coverage", "reliability_table"]


class CalibrationError(ValueError):
    """Calibration was asked for on something that cannot support the claim."""


@dataclass(frozen=True)
class Calibration:
    """Coverage of the lead-time interval at each nominal level.

    Attributes:
        cover_days: Lead time plus review period, in days.
        nominal: The service levels asked for, e.g. `(0.5, 0.8, 0.95)`.
        empirical: Randomised discrete coverage at each level. Boundary observations receive the
            q-specific fraction needed to place q within the quantile atom. See `coverage`.
        empirical_plain: `P(Y <= q)`, which is what an ordering policy actually experiences.
            Reported alongside, because the business gets this one even though the DISTRIBUTIONAL
            question is answered by the randomised figure.
        n_cells: Cells the coverage was measured over.
        naive_sum_ratio: Sum of per-day quantiles divided by the quantile of the sum, at the
            highest nominal level. Above 1 means the marginal shortcut over-orders; below 1
            means it under-orders. Neither direction is ruled out mathematically.
        notes: What the numbers mean for the interval.
    """

    cover_days: int
    nominal: tuple[float, ...]
    empirical: tuple[float, ...]
    empirical_plain: tuple[float, ...]
    n_cells: int
    naive_sum_ratio: float
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Raises: CalibrationError: the two level tuples disagree, or nothing was measured."""
        if (not isinstance(self.cover_days, int) or isinstance(self.cover_days, bool)
                or self.cover_days < 1):
            raise CalibrationError(
                f"cover_days must be a positive integer, got {self.cover_days!r}")
        if not len(self.nominal) == len(self.empirical) == len(self.empirical_plain):
            raise CalibrationError(
                f"{len(self.nominal)} nominal levels against {len(self.empirical)} randomised and "
                f"{len(self.empirical_plain)} plain empirical values")
        if not self.nominal:
            raise CalibrationError("no nominal levels; a calibration curve over nothing is not one")
        if (not isinstance(self.n_cells, int) or isinstance(self.n_cells, bool)
                or self.n_cells < 1):
            raise CalibrationError("no cells; a coverage figure over nothing is not one")
        invalid_nominal = [q for q in self.nominal
                           if not is_finite_real(q) or not 0.0 < q < 1.0]
        if invalid_nominal:
            raise CalibrationError(
                f"nominal levels must be finite and lie in (0,1): {invalid_nominal}")
        for label, values in (("empirical", self.empirical),
                              ("empirical_plain", self.empirical_plain)):
            invalid = [value for value in values
                       if not is_finite_real(value) or not 0.0 <= value <= 1.0]
            if invalid:
                raise CalibrationError(
                    f"{label} coverage must be finite and lie in [0,1]: {invalid}")
        if not is_finite_real(self.naive_sum_ratio) or self.naive_sum_ratio < 0:
            raise CalibrationError(
                f"naive_sum_ratio must be finite and non-negative, got {self.naive_sum_ratio!r}")
        if (not isinstance(self.notes, list)
                or any(not isinstance(note, str) or not note.strip() for note in self.notes)):
            raise CalibrationError("calibration notes must be non-blank strings")

    @property
    def worst_gap(self) -> float:
        """Largest absolute distance between a nominal level and what it actually covered."""
        return max(abs(n - e) for n, e in zip(self.nominal, self.empirical, strict=True))

    def is_calibrated(self, tolerance: float = 0.05) -> bool:
        """Whether every level lands within `tolerance` of its nominal.

        Both directions matter and they fail differently. **Under-covering** means stockouts at a
        service level the business believes it is buying. **Over-covering** means capital and
        shelf space spent on a level nobody asked for -- quieter, and it never generates a
        complaint, which is why it survives.
        """
        if not is_finite_real(tolerance) or tolerance < 0.0:
            raise CalibrationError(
                f"tolerance must be a finite non-negative built-in number, got {tolerance!r}")
        return self.worst_gap <= tolerance

    def __str__(self) -> str:
        pairs = "  ".join(f"{n:.0%}→{e:.0%}" for n, e in
                          zip(self.nominal, self.empirical, strict=True))
        mark = "" if self.is_calibrated() else "  [NOT CALIBRATED]"
        return (f"{self.cover_days}-day cover, {self.n_cells:,} cells: {pairs}{mark}\n"
                f"  marginal-quantile sum / path-total quantile = "
                f"{self.naive_sum_ratio:.2f}x")

    def as_record(self) -> dict[str, object]:
        return {"cover_days": self.cover_days, "nominal": list(self.nominal),
                "empirical_randomized": list(self.empirical),
                "empirical_plain": list(self.empirical_plain), "n_cells": self.n_cells,
                "worst_gap": self.worst_gap, "calibrated": self.is_calibrated(),
                "naive_sum_ratio": self.naive_sum_ratio, "notes": self.notes}


def coverage(
    paths: torch.Tensor,
    actual: torch.Tensor,
    *,
    lead_time: int,
    review_period: int,
    nominal: tuple[float, ...] = (0.5, 0.8, 0.95, 0.99),
) -> Calibration:
    """Measure how often actual cumulative demand fell inside the forecast interval.

    Args:
        paths: `[n_paths, n_cells, horizon]` integer sample paths.
        actual: `[n_cells, horizon]` realised demand.
        lead_time: Days until an order arrives.
        review_period: Days until the next ordering opportunity.
        nominal: Service levels to check.

    Returns:
        A `Calibration`.

    Raises:
        CalibrationError: the cover period exceeds the horizon drawn -- extrapolating past it
            would invent dependence beyond the last day the factor was drawn for -- or the shapes
            disagree, or a nominal level lies outside (0,1).
    """
    if not isinstance(paths, torch.Tensor):
        raise CalibrationError(f"paths must be a torch.Tensor, got {type(paths).__name__}")
    if not isinstance(actual, torch.Tensor):
        raise CalibrationError(f"actual must be a torch.Tensor, got {type(actual).__name__}")
    if paths.dim() != 3:
        raise CalibrationError(
            f"paths must be [n_paths, n_cells, horizon], got {tuple(paths.shape)}")
    if any(size < 1 for size in paths.shape):
        raise CalibrationError(
            f"paths must contain at least one path, cell, and horizon day, got "
            f"{tuple(paths.shape)}"
        )
    if (not isinstance(lead_time, int) or isinstance(lead_time, bool) or lead_time < 0
            or not isinstance(review_period, int) or isinstance(review_period, bool)
            or review_period < 0):
        raise CalibrationError(
            f"lead_time and review_period must be non-negative integers, got "
            f"{lead_time!r} and {review_period!r}")
    cover = lead_time + review_period
    if cover < 1:
        raise CalibrationError(f"cover period must be >= 1, got {lead_time} + {review_period}")
    if cover > paths.shape[-1]:
        raise CalibrationError(
            f"cover period {cover} exceeds the {paths.shape[-1]}-day horizon drawn. Extending it "
            "would invent dependence past the last day the factor was drawn for")
    if actual.shape != paths.shape[1:]:
        raise CalibrationError(
            f"actual is {tuple(actual.shape)} against paths {tuple(paths.shape[1:])}")
    if not nominal or any(not 0.0 < q < 1.0 for q in nominal):
        raise CalibrationError(f"nominal levels must lie in (0,1), got {nominal}")

    _validate_counts("paths", paths)
    _validate_counts("actual", actual)

    path_totals = paths[..., :cover].double().sum(dim=-1)          # [n_paths, n_cells]
    actual_total = actual[:, :cover].double().sum(dim=-1)          # [n_cells]
    if (not bool(torch.isfinite(path_totals).all())
            or not bool(torch.isfinite(actual_total).all())
            or bool((path_totals < 0).any()) or bool((actual_total < 0).any())):
        raise CalibrationError(
            "finite daily counts overflowed while forming cumulative demand; coverage is not "
            "representable at this scale")

    # RANDOMISED DISCRETE COVERAGE. For threshold Q(q), choose alpha so
    # F(Q^-) + alpha P(Y=Q) = q. A fixed alpha=.5 is not q-specific: an all-zero distribution
    # would score .5 at 50%, 80%, 95%, and 99%, falsely failing three perfect levels.
    empirical, empirical_plain = [], []
    for q in nominal:
        threshold = _order_statistic(path_totals, q, dim=0)       # [n_cells]
        forecast_below = (path_totals < threshold).double().mean(dim=0)
        forecast_at = (path_totals == threshold).double().mean(dim=0)
        if bool((forecast_at <= 0).any()):
            raise CalibrationError(
                "empirical quantile has no boundary mass; path order statistic is inconsistent")
        boundary_weight = ((q - forecast_below) / forecast_at).clamp(0.0, 1.0)
        below = (actual_total < threshold).double()
        at = (actual_total == threshold).double()
        empirical.append(float((below + boundary_weight * at).mean()))
        # What an ordering policy actually experiences: order the quantile, cover what you cover.
        empirical_plain.append(float((actual_total <= threshold).double().mean()))

    # The inequality this module exists to demonstrate, measured rather than asserted.
    top = max(nominal)
    per_day_q = _order_statistic(paths[..., :cover].double(), top, dim=0).sum(dim=-1)
    total_q = _order_statistic(path_totals, top, dim=0)
    numerator, denominator = float(per_day_q.sum()), float(total_q.sum())
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        raise CalibrationError("quantile totals overflowed while computing naive_sum_ratio")
    if denominator == 0.0 and numerator != 0.0:
        raise CalibrationError(
            "path-total quantile is zero while summed marginal quantiles are positive; the "
            "naive_sum_ratio is undefined rather than JSON-safe evidence")
    ratio = 1.0 if numerator == 0.0 and denominator == 0.0 else numerator / denominator

    notes = []
    for q, e in zip(nominal, empirical, strict=True):
        if e < q - 0.05:
            notes.append(
                f"{q:.0%} covers only {e:.0%}: the interval is TOO NARROW, so stockouts happen at "
                "a service level the business believes it is buying")
        elif e > q + 0.05:
            notes.append(
                f"{q:.0%} covers {e:.0%}: the interval is TOO WIDE, so capital and shelf space go "
                "on a level nobody asked for. This never generates a complaint, which is why it "
                "survives")
    if ratio > 1.05:
        notes.append(
            f"summing per-day quantiles would order {ratio:.2f}x what the quantile of the sum "
            "requires -- the over-ordering a point-forecast pipeline cannot see")
    elif ratio < 0.95:
        notes.append(
            f"summing per-day quantiles would order only {ratio:.2f}x the quantile of the sum -- "
            "the under-ordering a marginal-quantile pipeline cannot see")
    return Calibration(cover_days=cover, nominal=tuple(nominal), empirical=tuple(empirical),
                       empirical_plain=tuple(empirical_plain),
                       n_cells=int(actual.shape[0]), naive_sum_ratio=ratio, notes=notes)


def _validate_counts(name: str, values: torch.Tensor) -> None:
    """Demand paths and actuals must be finite, non-negative integer counts.

    Torch's count samplers return floating tensors, so this validates VALUES rather than insisting
    on an integer storage dtype.  That accepts ``tensor([0., 1.])`` but rejects interpolated
    pseudo-demand such as ``0.5``, which would make discrete coverage meaningless.
    """
    if values.dtype == torch.bool or values.is_complex():
        raise CalibrationError(f"{name} must contain integer demand counts, got {values.dtype}")
    try:
        finite = torch.isfinite(values)
    except (RuntimeError, TypeError) as exc:
        raise CalibrationError(f"{name} must contain numeric demand counts") from exc
    if not bool(finite.all()):
        raise CalibrationError(f"{name} must contain only finite demand counts")
    if bool((values < 0).any()):
        raise CalibrationError(f"{name} must contain only non-negative demand counts")
    if values.is_floating_point() and not bool((values == values.round()).all()):
        raise CalibrationError(f"{name} must contain integer demand counts")


def _order_statistic(values: torch.Tensor, q: float, *, dim: int) -> torch.Tensor:
    """The empirical inverse CDF: the smallest observed value whose CDF reaches ``q``.

    ``torch.quantile`` defaults to linear interpolation.  On paths ``{0, 1}`` its median is 0.5,
    a demand value no path contains; comparing an integer actual against that invented boundary
    changes the boundary entirely. The ``ceil(q*n)`` order statistic is the discrete
    quantile the ordering decision can actually place.
    """
    n = values.shape[dim]
    k = max(1, math.ceil(q * n))
    return torch.kthvalue(values, k, dim=dim).values


def reliability_table(
    paths: torch.Tensor, actual: torch.Tensor, *, lead_time: int, review_period: int,
    bins: int = 10,
) -> list[tuple[float, float, int]]:
    """`(nominal, empirical, n)` across evenly spaced levels -- the reliability diagram's data.

    A single service level can be right by accident. A model that is well calibrated at 95% and
    badly at 50% has a distribution of the wrong SHAPE, and only the ordering policy at 95% would
    ever notice -- until the policy changes.

    Raises:
        CalibrationError: `bins` below 2, which cannot describe a curve.
    """
    if bins < 2:
        raise CalibrationError(f"a reliability table needs at least 2 bins, got {bins}")
    levels = tuple((i + 1) / (bins + 1) for i in range(bins))
    cal = coverage(paths, actual, lead_time=lead_time, review_period=review_period,
                   nominal=levels)
    return [(n, e, cal.n_cells) for n, e in zip(cal.nominal, cal.empirical, strict=True)]
