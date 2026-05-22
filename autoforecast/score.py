"""The honest metric: an evaluation harness that can see the errors the deployed one cannot.

MOVED from `migration/honest_metric.py` on 2026-08-22, unchanged, because it JUDGES. Anything that
judges belongs to the harness and must be reachable by every competitor on equal terms; a metric
living inside the package it scores is the oldest way to win a benchmark by accident.
The Spark path over Parquet is `score_spark.py`, deliberately separate so THIS
module keeps its stdlib-only guarantee -- the arithmetic below takes plain tuples, which is what
lets it be checked against hand-computed cases and reduced across partitions without a JVM.

The deployed metric has four defects, and they compound in the same direction: they all make the
forecast look better than it is.

  1. SCORED AT A COARSER GRAIN than the forecast is published at. The forecast is per store x
     product x day; the metric sums across stores first. Store-level errors of opposite sign
     cancel before anything is measured, so a forecast that is wrong everywhere can score well.

  2. INNER-JOINED TO ACTUALS. A cell with a forecast and no actual row is dropped. But "no actual
     row" is exactly what a zero-sales day looks like in a sparse fact table -- so every
     OVER-FORECAST ON A DAY THAT SOLD NOTHING silently leaves the evaluation set. The metric is
     structurally incapable of seeing the error it most needs to catch.

  3. UNDEFINED AT ZERO. Percentage errors divide by the actual. With 36%-77% zero rates the
     denominator is zero for most cells, and whatever convention handles that decides the score.

  4. NO FROZEN ORIGIN. Features are recomputed with data that did not exist at forecast time.

This module scores on the FULL forecast x calendar cross-product restricted to ELIGIBLE cells,
at the published grain, with a metric defined at zero, and reports bias separately from error --
because a chronic over-forecast and a chronic under-forecast are different business failures and
an absolute-error summary hides which one you have.

Pure Python, no dependencies, no I/O. Bring your own rows.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from autoforecast._numeric import is_finite_real


@dataclass(frozen=True)
class Cell:
    """One published forecast cell, including the evaluation stratum and source provenance.

    `actual=None` is deliberately ambiguous until `actual_state` says whether it is a known sparse
    zero or missing source data. Treating both as zero is the exact four-zero violation this metric
    exists to prevent. Client and origin are required because rolling-origin results may overlap;
    aggregating them together lets errors cancel across separate forecast decisions.
    """
    client: str
    origin: str
    store: str
    product: str
    day: str
    forecast: float
    actual: float | None
    eligible: bool                 # from the exposure contract: could this cell have sold?
    actual_state: Literal["observed", "sparse_zero", "no_data"] = "observed"


@dataclass(frozen=True)
class Score:
    """A score, plus the RAW SUMS needed to combine partial scores.

    The raw sums are not decoration. `honest_metric` must run under Spark at estate scale, and a
    metric that stores only ratios CANNOT BE REDUCED ACROSS PARTITIONS -- you cannot average
    per-partition MAEs to get the global MAE unless the partitions are equal-sized. Keeping
    `sum_abs_error`, `sum_error`, `sum_actual` and `n_scored` lets `Score.combine()` recover the
    global formula without averaging partition ratios. As with every floating reduction, a
    different summation order may change the last bit.
    """
    n_scored: int
    n_dropped_ineligible: int
    n_dropped_no_data: int
    n_zero_filled: int
    mae: float
    rmse: float
    """Root mean squared error -- THE metric a mean forecast must be selected by.

    MAE is minimised by the conditional MEDIAN; squared error is minimised by the conditional
    MEAN. This model emits a mean (`heads.hurdle_mean`), demand is heavily right-skewed and
    zero-inflated (M5 measures 55.1% zeros on eligible cells), so median and mean are far apart
    and ranking mean forecasts by MAE prefers whichever model sits closer to the median. That is
    selection pressure toward under-forecasting.

    It also disagreed with the publication metric: WRMSSE is squared-error based, so a rung could
    win the ladder on MAE and then be scored by something that wanted a different forecast.
    Until 2026-08-25 this object carried no squared-error metric at all.

    MAE stays -- it is a fine descriptive error -- but it is a diagnostic, not the selector.
    """
    bias: float                    # signed: positive means OVER-forecasting
    bias_pct_of_volume: float | None   # None when volume is 0 -- inf poisons any aggregate
    mase: float | None
    # Raw sufficient statistics are REQUIRED, including for an empty partition. A default can
    # fabricate an internally plausible ratio: before this guard, a caller could provide one
    # unit of absolute error, omit squared error, and publish RMSE=0 because the dataclass supplied
    # a zero SSE. These fields are the evidence behind the displayed ratios, not optional metadata.
    sum_abs_error: float
    sum_squared_error: float          # RMSE reduces only through the SUM, never averaged ratios
    sum_error: float
    sum_actual: float
    sum_naive_abs_error: float        # the MASE denominator's numerator, so MASE reduces too
    grain: str = "cell"            # "cell" or "aggregated" -- the two are NOT comparable

    def __post_init__(self) -> None:
        """Validate the ratios and the raw sufficient statistics behind them."""
        counts = {
            "n_scored": self.n_scored,
            "n_dropped_ineligible": self.n_dropped_ineligible,
            "n_dropped_no_data": self.n_dropped_no_data,
            "n_zero_filled": self.n_zero_filled,
        }
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
               for value in counts.values()):
            raise ValueError(f"score counts must be non-negative integers: {counts}")
        if self.n_zero_filled > self.n_scored:
            raise ValueError(
                f"n_zero_filled={self.n_zero_filled} exceeds n_scored={self.n_scored}")
        numeric = {
            "mae": self.mae,
            "rmse": self.rmse,
            "bias": self.bias,
            "sum_abs_error": self.sum_abs_error,
            "sum_squared_error": self.sum_squared_error,
            "sum_error": self.sum_error,
            "sum_actual": self.sum_actual,
            "sum_naive_abs_error": self.sum_naive_abs_error,
        }
        optional = {"bias_pct_of_volume": self.bias_pct_of_volume, "mase": self.mase}
        invalid = {name: value for name, value in numeric.items()
                   if not is_finite_real(value)}
        invalid.update({name: value for name, value in optional.items()
                        if value is not None and not is_finite_real(value)})
        if invalid:
            raise ValueError(
                f"score values must be finite numbers (or None where allowed): {invalid}")
        if (self.mae < 0 or self.rmse < 0 or self.sum_abs_error < 0
                or self.sum_squared_error < 0 or self.sum_actual < 0
                or self.sum_naive_abs_error < 0 or (self.mase is not None and self.mase < 0)):
            raise ValueError(
                "MAE, RMSE, absolute-error, squared-error, actual, and naive sums cannot be "
                "negative")
        if self.grain not in ("cell", "aggregated"):
            raise ValueError(f"unsupported score grain {self.grain!r}")
        if self.n_scored:
            expected_mae = self.sum_abs_error / self.n_scored
            expected_rmse = (self.sum_squared_error / self.n_scored) ** 0.5
            expected_bias = self.sum_error / self.n_scored
            if not math.isclose(self.mae, expected_mae, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(
                    f"mae {self.mae} disagrees with sum_abs_error/n_scored {expected_mae}")
            if not math.isclose(self.rmse, expected_rmse, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(
                    f"rmse {self.rmse} disagrees with sqrt(sum_squared_error/n_scored) "
                    f"{expected_rmse}")
            if not math.isclose(self.bias, expected_bias, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(
                    f"bias {self.bias} disagrees with sum_error/n_scored {expected_bias}")
            # These inequalities hold for every real residual vector, independently of partition
            # boundaries: |mean(e)| <= mean(|e|) <= RMS(e), and RMS(e) cannot exceed
            # sqrt(n)*mean(|e|). Ratio-vs-sum checks alone do not enforce them; without this guard
            # a stored record can provide mutually impossible sufficient statistics while making
            # every displayed ratio agree with its own fabricated sum.
            tolerance = 1e-12
            if (abs(self.bias) > self.mae
                    and not math.isclose(abs(self.bias), self.mae,
                                         rel_tol=tolerance, abs_tol=tolerance)):
                raise ValueError("absolute bias cannot exceed MAE for any residual vector")
            if (self.mae > self.rmse
                    and not math.isclose(self.mae, self.rmse,
                                         rel_tol=tolerance, abs_tol=tolerance)):
                raise ValueError("MAE cannot exceed RMSE for any residual vector")
            maximum_rmse = math.sqrt(self.n_scored) * self.mae
            if (self.rmse > maximum_rmse
                    and not math.isclose(self.rmse, maximum_rmse,
                                         rel_tol=tolerance, abs_tol=tolerance)):
                raise ValueError(
                    "RMSE cannot exceed sqrt(n_scored) * MAE for any residual vector")
            # Forecasts and actuals are non-negative at both scorer entry points. Their aggregate
            # forecast is recoverable from these sufficient statistics:
            #
            #     sum(forecast) = sum(actual) + sum(forecast - actual).
            #
            # The other norm inequalities do not imply this constraint: one cell with actual=1,
            # error=-2 satisfies all of them while requiring forecast=-1. Allow only the tiny
            # cancellation noise possible when two large independently reduced sums should meet at
            # zero; a materially negative implied forecast proves the record was fabricated or
            # corrupted. A relative tolerance is wrong here: at an estate-scale volume of 1e15,
            # ``rel_tol=1e-12`` accepted an impossible -100 aggregate forecast. Bound only the
            # final cancellation noise in ULPs of the two stored operands. This stays a small
            # number of representable values at every scale instead of admitting a fixed fraction
            # of estate volume.
            implied_forecast = self.sum_actual + self.sum_error
            cancellation_slack = 4.0 * max(
                math.ulp(self.sum_actual), math.ulp(self.sum_error))
            if implied_forecast < -cancellation_slack:
                raise ValueError(
                    "sum_actual + sum_error implies a negative total forecast, impossible for "
                    "non-negative forecast cells")
        else:
            if any(value != 0 for value in (
                    self.sum_abs_error, self.sum_squared_error, self.sum_error, self.sum_actual,
                    self.sum_naive_abs_error)):
                raise ValueError("a zero-cell score cannot carry non-zero raw sums")
            if any(value != 0 for value in (self.mae, self.rmse, self.bias)):
                raise ValueError(
                    "a zero-cell score cannot claim non-zero MAE, RMSE, or bias without evidence")
        expected_pct = self.sum_error / self.sum_actual if self.sum_actual else None
        if ((expected_pct is None) != (self.bias_pct_of_volume is None)
                or (expected_pct is not None and not math.isclose(
                    self.bias_pct_of_volume, expected_pct, rel_tol=1e-12, abs_tol=1e-12))):
            raise ValueError(
                "bias_pct_of_volume disagrees with sum_error/sum_actual or is defined at zero "
                "volume")
        expected_mase = (self.sum_abs_error / self.sum_naive_abs_error
                         if self.sum_naive_abs_error else None)
        if ((expected_mase is None) != (self.mase is None)
                or (expected_mase is not None and not math.isclose(
                    self.mase, expected_mase, rel_tol=1e-12, abs_tol=1e-12))):
            raise ValueError(
                "mase disagrees with sum_abs_error/sum_naive_abs_error or is defined with no "
                "naive denominator")

    @property
    def absolute_error_pct_of_volume(self) -> float | None:
        """Absolute error as a share of volume (WAPE), comparable across the two grains.

        Mean absolute errors are not comparable because one mean is over cells and the other over
        store-aggregated groups. Their raw absolute-error sums and common actual volume are in the
        same units, so this ratio exposes aggregation cancellation without a store-count artefact.
        """
        return self.sum_abs_error / self.sum_actual if self.sum_actual else None

    @staticmethod
    def combine(parts: list[Score]) -> Score:
        """Sufficient-statistic reduction; ratios are recomputed from sums, never averaged."""
        if not parts:
            raise ValueError("nothing to combine")
        scored = [part for part in parts if part.n_scored]
        if not scored:
            raise ValueError("nothing to combine")
        grains = {p.grain for p in parts}
        if len(grains) > 1:
            raise ValueError(f"refusing to combine different grains: {grains}")
        has_mase = [p.mase is not None for p in scored]
        if any(has_mase) and not all(has_mase):
            raise ValueError(
                "cannot combine partial MASE coverage: every partition needs a valid denominator")
        n = sum(p.n_scored for p in scored)
        sae = sum(p.sum_abs_error for p in scored)
        sse = sum(p.sum_squared_error for p in scored)
        se = sum(p.sum_error for p in scored)
        vol = sum(p.sum_actual for p in scored)
        naive = sum(p.sum_naive_abs_error for p in scored)
        return Score(n_scored=n,
                     n_dropped_ineligible=sum(p.n_dropped_ineligible for p in parts),
                     n_dropped_no_data=sum(p.n_dropped_no_data for p in parts),
                     n_zero_filled=sum(p.n_zero_filled for p in parts),
                     mae=sae / n, rmse=(sse / n) ** 0.5, bias=se / n,
                     bias_pct_of_volume=(se / vol) if vol else None,
                     # MASE now survives reduction: recomputed from summed numerator and
                     # denominator, never averaged from per-partition MASEs, which would be wrong
                     # unless the partitions were equal-sized.
                     mase=(sae / naive) if all(has_mase) else None,
                     sum_abs_error=sae, sum_squared_error=sse, sum_error=se, sum_actual=vol,
                     sum_naive_abs_error=naive, grain=parts[0].grain)

    def __str__(self) -> str:
        pct = f"{self.bias_pct_of_volume:+.1%} of volume" if self.bias_pct_of_volume is not None \
            else "volume=0"
        return (f"[{self.grain}] n={self.n_scored} (zero-filled {self.n_zero_filled}, "
                f"ineligible dropped {self.n_dropped_ineligible}, "
                f"no-data dropped {self.n_dropped_no_data})  "
                f"RMSE={self.rmse:.3f}  MAE={self.mae:.3f}  bias={self.bias:+.3f} ({pct})"
                + (f"  MASE={self.mase:.3f}" if self.mase is not None else ""))


def _validate_cells(cells: list[Cell]) -> None:
    """Validate the parts of a cell that would otherwise change a score silently."""
    seen = set()
    for c in cells:
        key = (c.client, c.origin, c.store, c.product, c.day)
        if any(not isinstance(v, str) or not v.strip() for v in key):
            raise ValueError(f"cell identity and evaluation stratum cannot be blank: {key}")
        if key in seen:
            raise ValueError(f"duplicate published forecast cell: {key}")
        seen.add(key)
        if not isinstance(c.eligible, bool):
            raise ValueError(f"cell {key} needs explicit boolean exposure eligibility")
        if not is_finite_real(c.forecast) or c.forecast < 0:
            raise ValueError(f"cell {key} has invalid non-negative forecast: {c.forecast!r}")
        if c.actual is not None and (not is_finite_real(c.actual) or c.actual < 0):
            raise ValueError(f"cell {key} has invalid non-negative actual: {c.actual!r}")
        if c.actual_state not in ("observed", "sparse_zero", "no_data"):
            raise ValueError(f"cell {key} has unsupported actual_state: {c.actual_state!r}")
        if (c.actual_state == "observed") != (c.actual is not None):
            raise ValueError(
                f"cell {key} must use actual_state='observed' exactly when an actual is present")


def score(cells: list[Cell], naive_mae: float | None = None) -> Score:
    """Score at the published grain, on the full cross-product, defined at zero.

    `actual=None, actual_state='sparse_zero'` on an eligible cell is a true zero. A no-data row is
    excluded and counted separately. Requiring the caller to distinguish them prevents a source
    outage from becoming either a perfect zero forecast or a fabricated over-forecast.

    Ineligible cells -- never ranged, shop shut, outside lifecycle -- are excluded, because a
    forecast for a product the shop does not stock is not a forecasting error.
    """
    if naive_mae is not None and (not is_finite_real(naive_mae) or naive_mae <= 0):
        raise ValueError("naive_mae must be finite and strictly positive when supplied")
    _validate_cells(cells)
    scored, dropped, dropped_no_data, zero_filled = [], 0, 0, 0
    for c in cells:
        if not c.eligible:
            dropped += 1
            continue
        if c.actual_state == "no_data":
            dropped_no_data += 1
            continue
        actual = c.actual
        if c.actual_state == "sparse_zero":
            actual = 0.0
            zero_filled += 1
        scored.append((c.forecast, actual))

    if not scored:
        raise ValueError("no eligible cells to score")

    n = len(scored)
    errors = [f - a for f, a in scored]
    volume = sum(a for _, a in scored)
    mae = sum(abs(e) for e in errors) / n
    bias = sum(errors) / n
    return Score(
        n_scored=n,
        n_dropped_ineligible=dropped,
        n_dropped_no_data=dropped_no_data,
        n_zero_filled=zero_filled,
        mae=mae,
        rmse=(sum(e * e for e in errors) / n) ** 0.5,
        bias=bias,
        bias_pct_of_volume=(sum(errors) / volume) if volume else None,
        mase=(mae / naive_mae) if naive_mae is not None else None,
        sum_abs_error=sum(abs(e) for e in errors),
        sum_squared_error=sum(e * e for e in errors),
        sum_error=sum(errors),
        sum_actual=volume,
        sum_naive_abs_error=(naive_mae * n) if naive_mae is not None else 0.0,
        grain="cell",
    )


def score_the_old_way(cells: list[Cell]) -> Score:
    """Demonstrate aggregation and inner-join scoring bias on the same input cells.

    This structural example is not any production metric. Aggregating residuals
    across stores hides offsetting errors, and inner joins discard missing actuals.
    Its aggregated MAE has different units from cell-level MAE; compare normalized
    totals only. The returned grain makes incompatible comparisons detectable.
    """
    _validate_cells(cells)
    dropped = sum(1 for c in cells if not c.eligible)
    dropped_no_data = sum(1 for c in cells if c.eligible and c.actual_state == "no_data")
    by_key: dict[tuple[str, str, str, str], list[Cell]] = {}
    for c in cells:
        if c.eligible:
            by_key.setdefault((c.client, c.origin, c.product, c.day), []).append(c)

    scored = []
    for (_client, _origin, _product, _day), group in by_key.items():
        present = [c for c in group if c.actual is not None]      # the inner join
        if not present:
            continue
        scored.append((sum(c.forecast for c in present),          # aggregate, THEN residual
                       sum(c.actual for c in present)))

    n = len(scored)
    if not n:
        raise ValueError("nothing survived the inner join")
    errors = [f - a for f, a in scored]
    volume = sum(a for _, a in scored)
    return Score(
        n_scored=n,
        n_dropped_ineligible=dropped,
        n_dropped_no_data=dropped_no_data,
        n_zero_filled=0,
        mae=sum(abs(e) for e in errors) / n,
        rmse=(sum(e * e for e in errors) / n) ** 0.5,
        bias=sum(errors) / n,
        bias_pct_of_volume=(sum(errors) / volume) if volume else None,
        mase=None,
        sum_abs_error=sum(abs(e) for e in errors),
        sum_squared_error=sum(e * e for e in errors),
        sum_error=sum(errors),
        sum_actual=volume,
        sum_naive_abs_error=0.0,
        grain="aggregated",
    )


def naive_seasonal_mae(history: list[float], season: int = 7) -> float:
    """Denominator for MASE: the in-sample MAE of a seasonal-naive forecast.

    Scale-free, defined at zero, and it does not reward predicting zero everywhere -- which is
    what percentage errors do on intermittent series.
    """
    if isinstance(season, bool) or not isinstance(season, int) or season < 1:
        raise ValueError("season must be a positive integer")
    if any(not is_finite_real(v) or v < 0 for v in history):
        raise ValueError("seasonal history must contain finite, non-negative demand")
    if len(history) <= season:
        raise ValueError(f"need more than {season} points to form a seasonal-naive baseline")
    diffs = [abs(history[i] - history[i - season]) for i in range(season, len(history))]
    denom = math.fsum(diff / len(diffs) for diff in diffs)
    if denom == 0:
        raise ValueError(
            "seasonal-naive MAE is 0 -- the series is exactly periodic over this window, so MASE "
            "would divide by zero. Intermittent catalogues contain many such windows; the caller "
            "must exclude the series or widen the window rather than receive an infinity")
    return denom


def compare(deployed: Score, honest: Score) -> dict[str, float | None]:
    """The Gate 1 comparison, on quantities that survive the grain difference.

    Gate 1 requires reporting the deployed metric and the honest metric for the same run. Their
    MAEs are in different units (see `score_the_old_way`), so this returns only the scale-free
    quantities plus the counts that explain the gap.
    """
    if deployed.grain != "aggregated" or honest.grain != "cell":
        raise ValueError(
            f"expected deployed='aggregated' and honest='cell'; got "
            f"{deployed.grain!r} and {honest.grain!r}")
    if not math.isclose(deployed.sum_actual, honest.sum_actual, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            "deployed and honest scores cover different actual volume; the cohorts are not equal")
    return {
        "deployed_bias_pct_of_volume": deployed.bias_pct_of_volume,
        "honest_bias_pct_of_volume": honest.bias_pct_of_volume,
        "deployed_absolute_error_pct_of_volume": deployed.absolute_error_pct_of_volume,
        "honest_absolute_error_pct_of_volume": honest.absolute_error_pct_of_volume,
        "deployed_sum_abs_error": deployed.sum_abs_error,
        "honest_sum_abs_error": honest.sum_abs_error,
        "cells_invisible_to_deployed": honest.n_zero_filled,
        "groups_scored_deployed": deployed.n_scored,
        "cells_scored_honest": honest.n_scored,
        # MAE deliberately absent: the two are different units and the ratio is meaningless
    }
