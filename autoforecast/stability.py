"""Forecast stability: how much a cell's number moves between consecutive origins.

**GATE 4 REQUIRES THIS AND NO EARLIER GATE MEASURED IT**, which is the whole reason the module
exists. Every gate up to here measures accuracy, and accuracy-only gates miss the failure that
actually gets forecasting systems switched off:

    A model 5% more accurate that swings 40% week to week generates order churn, and operations
    will reject it whatever its MASE.

The churn is not a modelling curiosity. A forecast that moves generates a replenishment order that
moves, and a buyer who has already committed to a supplier sees the system change its mind. After
a few weeks of that they stop looking at it, and the accuracy improvement is worth nothing because
nobody is acting on the number.

**WHAT IS COMPARED**: the same (series, date) cell, forecast from two origins a week apart. The
second origin has seen a week more data, so SOME movement is correct -- a model that never
revised would be ignoring new information. What is measured is whether the movement is
proportionate to the news, and the reference is what the comparators do on the same cells.

**WHY IT IS RELATIVE, NOT ABSOLUTE.** A cell moving from 2 to 3 units matters more than one moving
from 200 to 201, and an absolute mean would be dominated by the fast movers -- which are exactly
the cells where a proportional swing is least disruptive.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType

from autoforecast import grid
from autoforecast._numeric import is_finite_real
from autoforecast.frequency import parse_frequency, parse_origin, period_points
from autoforecast.panel import Forecast, ForecastError

__all__ = ["Stability", "StabilityError", "compare", "measure"]


class StabilityError(ValueError):
    """Stability was requested between forecasts that cannot be compared."""


@dataclass(frozen=True)
class Stability:
    """How much a forecast moved between two origins, on the cells they share.

    Attributes:
        adapter: Which model this describes.
        n_shared_cells: Cells forecast from BOTH origins. Only these can be compared -- a cell the
            later origin dropped has no revision, it has a disappearance.
        mean_absolute_revision: Mean `|later - earlier|` in units. Reported because operations
            think in units, but not the number to gate on.
        mean_relative_revision: Mean `|later - earlier| / max(earlier, 1)`. **The gating number.**
            A cell moving 2 to 3 matters more than one moving 200 to 201, and an absolute mean is
            dominated by fast movers -- exactly the cells where a swing is least disruptive.
        p95_relative_revision: The tail. A good mean with a bad tail is a model that is stable
            except when it is not, and the exceptions are what a buyer remembers.
        share_moving_over_20pct: Share of cells whose forecast moved by more than a fifth. The
            figure a planner recognises.
    """

    adapter: str
    n_shared_cells: int
    mean_absolute_revision: float
    mean_relative_revision: float
    p95_relative_revision: float
    share_moving_over_20pct: float

    def __post_init__(self) -> None:
        """Refuse an empty or non-finite measurement."""
        if not isinstance(self.adapter, str) or not self.adapter.strip():
            raise StabilityError("stability evidence needs a non-blank adapter identity")
        if (not isinstance(self.n_shared_cells, int) or isinstance(self.n_shared_cells, bool)
                or self.n_shared_cells < 1):
            raise StabilityError(
                f"{self.adapter}: the two origins share no cells, so no revision was measured. "
                "Comparing forecasts that do not overlap produces a number about nothing")
        values = {
            "mean_absolute_revision": self.mean_absolute_revision,
            "mean_relative_revision": self.mean_relative_revision,
            "p95_relative_revision": self.p95_relative_revision,
            "share_moving_over_20pct": self.share_moving_over_20pct,
        }
        invalid = {
            name: value for name, value in values.items()
            if not is_finite_real(value) or value < 0.0
        }
        if invalid:
            raise StabilityError(
                f"{self.adapter}: stability statistics must be finite and non-negative: "
                f"{invalid}")
        if self.share_moving_over_20pct > 1.0:
            raise StabilityError(
                f"{self.adapter}: share_moving_over_20pct must lie in [0,1], got "
                f"{self.share_moving_over_20pct}")

    def __str__(self) -> str:
        return (f"{self.adapter}: revision {self.mean_relative_revision:.1%} mean, "
                f"{self.p95_relative_revision:.1%} p95, "
                f"{self.share_moving_over_20pct:.1%} of cells moved >20% "
                f"({self.n_shared_cells:,} cells)")

    def as_record(self) -> dict[str, object]:
        return {"adapter": self.adapter, "n_shared_cells": self.n_shared_cells,
                "mean_absolute_revision": self.mean_absolute_revision,
                "mean_relative_revision": self.mean_relative_revision,
                "p95_relative_revision": self.p95_relative_revision,
                "share_moving_over_20pct": self.share_moving_over_20pct}


def measure(
    earlier: Forecast,
    later: Forecast,
    spark: SparkSession,
    *,
    expected_overlap: str | pathlib.Path,
    expected_overlap_hash: str,
) -> Stability:
    """Measure revision on exactly the pre-registered overlap key set.

    Args:
        earlier: The forecast made first.
        later: The forecast made from a subsequent origin. It has seen more data, so SOME
            revision is correct -- a model that never revised would be ignoring new information.
        spark: Active session.
        expected_overlap: Parquet containing the canonical ``series_id · date`` keys which the
            two registered origin grids share.
        expected_overlap_hash: SHA-256 committed before the forecasts ran. The Parquet path alone
            is mutable; checking its content hash is what makes this registered evidence.

    Returns:
        A `Stability`.

    Raises:
        StabilityError: either forecast is malformed; the expected overlap is empty, changed,
            outside the mathematical shared date window, or differs by even one key from the
            forecasts' actual overlap; the adapters/origin order are incompatible.
    """
    if earlier.adapter != later.adapter:
        raise StabilityError(
            f"comparing {earlier.adapter!r} against {later.adapter!r}. Stability is a property of "
            "ONE model across origins; comparing two models measures their disagreement instead")
    if earlier.target_estimand != later.target_estimand:
        raise StabilityError(
            f"stability forecasts different quantities: {earlier.target_estimand!r} versus "
            f"{later.target_estimand!r}")
    if earlier.frequency != later.frequency:
        raise StabilityError(
            "stability forecasts have different frequency declarations: "
            f"{earlier.frequency!r} versus {later.frequency!r}")
    try:
        cadence = parse_frequency(earlier.frequency)
        early_origin = parse_origin(earlier.origin, cadence)
        late_origin = parse_origin(later.origin, cadence)
    except (TypeError, ValueError) as exc:
        raise StabilityError(
            f"forecast origins must be canonical for {earlier.frequency!r}, got "
            f"{earlier.origin!r} and "
            f"{later.origin!r}") from exc
    if late_origin <= early_origin:
        raise StabilityError(
            f"the 'later' forecast has origin {later.origin}, not after {earlier.origin}. "
            "Measuring revision backwards produces a number that looks exactly the same")

    if not isinstance(expected_overlap_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_overlap_hash):
        raise StabilityError("expected_overlap_hash must be a lowercase SHA-256 digest")
    try:
        actual_overlap_hash = grid.content_hash(
            expected_overlap, spark, frequency=earlier.frequency
        )
    except Exception as exc:
        raise StabilityError(f"invalid expected-overlap artifact {expected_overlap}: {exc}") \
            from exc
    if actual_overlap_hash != expected_overlap_hash:
        raise StabilityError(
            "expected-overlap artifact changed after registration: "
            f"registered {expected_overlap_hash}, materialised {actual_overlap_hash}")

    a = _validated_forecast_frame(earlier, spark, "earlier", "_before")
    b = _validated_forecast_frame(later, spark, "later", "_after")
    raw_expected = grid.read_time_parquet(spark, expected_overlap, "date")
    if cadence.key_kind == "timestamp" and (
        raw_expected.schema["date"].dataType.typeName() != "timestamp_ntz"
    ):
        raise StabilityError("hourly expected overlap date must have timestamp_ntz physical type")
    expected_key, valid_expected_key = grid.canonical_key("date", cadence)
    expected = raw_expected.select(
        F.col("series_id").cast("string").alias("series_id"),
        expected_key.alias("date"),
        valid_expected_key.alias("_valid_date"),
    )
    if expected.filter(
        F.col("series_id").isNull()
        | (F.trim("series_id") == F.lit(""))
        | ~F.col("_valid_date")
    ).limit(1).count():
        raise StabilityError("expected overlap contains an invalid series or time key")
    expected = expected.drop("_valid_date")

    early_dates = set(period_points(early_origin, cadence, earlier.horizon))
    late_dates = set(period_points(late_origin, cadence, later.horizon))
    overlap_dates = early_dates & late_dates
    if not overlap_dates:
        raise StabilityError(
            f"{earlier.adapter}: origins {earlier.origin} and {later.origin} have no mathematical "
            "horizon overlap")
    if expected.filter(
        ~grid.period_membership("date", sorted(overlap_dates), cadence)
    ).limit(1).count():
        raise StabilityError(
            "expected overlap contains a date outside the exact shared period grid "
            f"{min(overlap_dates)}..{max(overlap_dates)}")
    n_expected = expected.count()
    if n_expected == 0:
        raise StabilityError(
            f"{earlier.adapter}: registered expected overlap is empty; a stability number over "
            "nothing cannot pass")

    actual_keys = (a.select("series_id", "date")
                   .join(b.select("series_id", "date"), ["series_id", "date"], "inner"))
    missing = expected.join(actual_keys, ["series_id", "date"], "left_anti").count()
    extra = actual_keys.join(expected, ["series_id", "date"], "left_anti").count()
    if missing or extra:
        raise StabilityError(
            f"{earlier.adapter}: forecast overlap is not the registered exact overlap: "
            f"{missing} expected key(s) missing, {extra} unregistered key(s) present")

    both = (expected.join(a, ["series_id", "date"], "inner")
            .join(b, ["series_id", "date"], "inner"))

    # Denominator floored at 1 unit: a cell moving from 0.01 to 0.02 is a 100% revision and moves
    # nothing anybody orders. Flooring keeps the metric about decisions rather than about ratios.
    rel = F.abs(F.col("_after") - F.col("_before")) / F.greatest(F.col("_before"), F.lit(1.0))
    stats = both.select(
        F.abs(F.col("_after") - F.col("_before")).alias("abs"),
        rel.alias("rel"),
    ).agg(
        F.avg("abs").alias("mean_abs"),
        F.avg("rel").alias("mean_rel"),
        F.expr("percentile_approx(rel, 0.95)").alias("p95"),
        F.avg((F.col("rel") > 0.2).cast("double")).alias("over20"),
    ).collect()[0]

    return Stability(adapter=earlier.adapter, n_shared_cells=n_expected,
                     mean_absolute_revision=float(stats["mean_abs"]),
                     mean_relative_revision=float(stats["mean_rel"]),
                     p95_relative_revision=float(stats["p95"]),
                     share_moving_over_20pct=float(stats["over20"]))


def _validated_forecast_frame(
    forecast: Forecast, spark: SparkSession, label: str, value_column: str,
):
    """Return canonical Spark columns after validating the complete forecast artifact.

    This validation is deliberately local to stability as well as present at adapter emission.
    Reuse means an artifact may be read long after it was produced, and evidence must validate the
    bytes it consumes rather than trust that some earlier caller happened to do so.
    """
    try:
        forecast.validate()
    except ForecastError as exc:
        raise StabilityError(f"{label} forecast is invalid: {exc}") from exc
    try:
        raw = grid.read_time_parquet(spark, forecast.path, "date")
    except Exception as exc:
        raise StabilityError(f"cannot read {label} forecast {forecast.path}: {exc}") from exc
    missing = sorted({"series_id", "date", "yhat"} - set(raw.columns))
    if missing:
        raise StabilityError(f"{label} forecast lacks required columns {missing}")
    if not isinstance(raw.schema["yhat"].dataType, NumericType):
        raise StabilityError(
            f"{label} forecast yhat must have a numeric schema, got "
            f"{raw.schema['yhat'].dataType.simpleString()}")

    cadence = parse_frequency(forecast.frequency)
    if cadence.key_kind == "timestamp" and (
        raw.schema["date"].dataType.typeName() != "timestamp_ntz"
    ):
        raise StabilityError(
            f"{label} hourly forecast date must have timestamp_ntz physical type"
        )
    time_key, valid_time_key = grid.canonical_key("date", cadence)
    frame = raw.select(
        F.col("series_id").cast("string").alias("series_id"),
        time_key.alias("date"),
        valid_time_key.alias("_valid_date"),
        F.col("yhat").cast("double").alias(value_column),
    )
    bad_key = frame.filter(
        F.col("series_id").isNull()
        | (F.trim("series_id") == F.lit(""))
        | ~F.col("_valid_date")).limit(1).count()
    if bad_key:
        raise StabilityError(
            f"{label} forecast contains a null/blank series id or invalid time key"
        )
    frame = frame.drop("_valid_date")
    value = F.col(value_column)
    if frame.filter(
            value.isNull() | F.isnan(value)
            | value.isin(float("inf"), float("-inf"))
            | (value < 0.0)).limit(1).count():
        raise StabilityError(f"{label} forecast contains a null, non-finite, or negative yhat")

    try:
        allowed_dates = period_points(
            forecast.origin, cadence, forecast.horizon
        )
    except (TypeError, ValueError) as exc:
        raise StabilityError(
            f"{label} forecast origin must be canonical for {forecast.frequency!r}, got "
            f"{forecast.origin!r}"
        ) from exc
    lo, hi = allowed_dates[0], allowed_dates[-1]
    if frame.filter(
        ~grid.period_membership("date", allowed_dates, cadence)
    ).limit(1).count():
        raise StabilityError(
            f"{label} forecast contains a date outside requested period grid {lo}..{hi}")
    if (frame.groupBy("series_id", "date").count()
            .filter(F.col("count") != 1).limit(1).count()):
        raise StabilityError(f"{label} forecast contains duplicate (series_id, date) keys")
    return frame


def compare(results: dict[str, Stability], *, bound: float) -> dict[str, object]:
    """Rank adapters by stability and say which clear a registered bound.

    Args:
        results: Adapter name -> `Stability`.
        bound: Registered maximum mean relative revision. **Registered BEFORE the run**, like every
            other threshold here -- a stability bound chosen after seeing the numbers is a
            description of what happened rather than a decision about what is acceptable.

    Returns:
        A record naming the most stable adapter, those that clear the bound, and those that do not.

    Raises:
        StabilityError: no results, or a non-positive bound.
    """
    if not results:
        raise StabilityError("nothing to compare")
    if not is_finite_real(bound) or bound <= 0:
        raise StabilityError(f"a stability bound must be positive, got {bound}")
    ordered = sorted(results, key=lambda a: results[a].mean_relative_revision)
    passing = [a for a in ordered if results[a].mean_relative_revision <= bound]
    return {
        "bound": bound,
        "most_stable": ordered[0],
        "order": ordered,
        "within_bound": passing,
        "outside_bound": [a for a in ordered if a not in passing],
        "detail": {a: results[a].as_record() for a in ordered},
        "note": ("a model more accurate but less stable than the incumbent generates order churn, "
                 "and operations will reject it whatever its MASE -- which is why this is a "
                 "separate PASS condition in Gate 4 rather than a tiebreak"),
    }
