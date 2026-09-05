"""WRMSSE — the M5 competition's published metric, implemented from its published definition.

We run NONE of the winning solution. Its SCORE is published for M5's private-evaluation window;
repository measurements use the preceding public-validation window. This module implements the
same metric definition, so the values share a metric scale, but their different labels prevent a
like-for-like performance ranking.

    RMSSE_i = sqrt(  mean_{t in horizon} (Y_it - Yhat_it)^2
                   / mean_{t=2..n}       (Y_it - Y_i,t-1)^2  )

    WRMSSE  = mean over the 12 aggregation levels of  sum_i w_i * RMSSE_i

The denominator is a naive-one-step benchmark computed on the TRAINING period only, from each
series' first non-zero observation -- M5 excludes leading zeros there for the same reason the
exposure contract does, and including them would inflate the denominator and flatter every
forecast. `w_i` is the series' share of dollar sales over the last 28 training days, so the
weights sum to one within each level.

GENERIC BY CONSTRUCTION. The levels are an ARGUMENT, not a constant: the formula is a definition
and the twelve levels are a fact about M5, which is why they live in `benchmarks/m5.py` as
`M5_LEVELS` and arrive from there.

PROTOCOL STATUS (2026-08-27): the repository's `2016-04-24` public window was observed before the
Phase 2 manifest reserved it and was reused afterward. It is retired from acceptance, so Phase 3
scores there are descriptive only and cannot support acceptance, holdout, Gate 4, independent-
confirmation, or stable-winner claims. WRMSSE also weights by recent sales value and is dominated
by fast movers, and M5 has no stock data, so exposure and censoring cannot show a benefit here.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from autoforecast._numeric import is_finite_real
from autoforecast.panel import ELIGIBLE_STATES, Forecast, Panel

WEIGHT_DAYS = 28
"""Training days the dollar weights are computed over. 28 is M5's published choice, not a knob."""


class WrmsseError(ValueError):
    """WRMSSE was asked for on data that cannot support its definition."""


@dataclass(frozen=True)
class Wrmsse:
    """A WRMSSE score and the per-level scores it averages.

    Attributes:
        score: The mean of `per_level`. It uses the published M5 metric scale; comparability to a
            published result additionally requires the same evaluation labels and evidence role.
        per_level: Level name -> weighted RMSSE at that level.
        n_series_scored: Level name -> series included in the published formula.
        n_series_dropped: Per-level evidence that the full scoring population was retained.
            Every value must be zero. Selection checks this evidence because dropping an
            undefined scale changes the population and is not published WRMSSE.
    """

    score: float
    per_level: dict[str, float]
    n_series_scored: dict[str, int] = field(default_factory=dict)
    n_series_dropped: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Refuse a score that is not the mean of the levels it reports.

        Raises:
            WrmsseError: no levels were scored, or `score` disagrees with `per_level`.
        """
        if not self.per_level:
            raise WrmsseError("no aggregation level produced a score")
        invalid_levels = {
            name: value for name, value in self.per_level.items()
            if not is_finite_real(value) or value < 0
        }
        if invalid_levels:
            raise WrmsseError(
                f"per-level WRMSSE values must be finite and non-negative, got {invalid_levels}")
        if not is_finite_real(self.score) or self.score < 0:
            raise WrmsseError(f"WRMSSE score must be finite and non-negative, got {self.score!r}")
        for label, counts in (("n_series_scored", self.n_series_scored),
                              ("n_series_dropped", self.n_series_dropped)):
            if counts and set(counts) != set(self.per_level):
                raise WrmsseError(
                    f"{label} must describe exactly the scored levels; got {sorted(counts)}, "
                    f"expected {sorted(self.per_level)}")
            if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
                   for value in counts.values()):
                raise WrmsseError(f"{label} values must be non-negative integers")
        if any(self.n_series_dropped.values()):
            raise WrmsseError(
                "a WRMSSE record may not drop undefined-scale series; that changes the metric's "
                "population rather than computing the published score")
        expected = math.fsum(float(v) / len(self.per_level)
                             for v in self.per_level.values())
        if abs(self.score - expected) > 1e-9:
            raise WrmsseError(
                f"score {self.score} is not the mean of the {len(self.per_level)} level scores "
                f"({expected}). WRMSSE is defined as that mean; anything else is another metric")

    def as_record(self) -> dict[str, object]:
        """A JSON-safe record for the gate result."""
        return {"wrmsse": self.score, "per_level": self.per_level,
                "n_series_scored": self.n_series_scored,
                "n_series_dropped": self.n_series_dropped}


def score(
    panel: Panel,
    forecast: Forecast,
    spark: SparkSession,
    *,
    levels: tuple[tuple[str, ...], ...],
    weight_col: str,
    weight_days: int = WEIGHT_DAYS,
) -> Wrmsse:
    """Compute WRMSSE for `forecast` against `panel`.

    Args:
        panel: Actuals and hierarchy. Must carry every column named in `levels` and `weight_col`.
        forecast: What to score. Its origin separates training from horizon.
        spark: Active session.
        levels: Aggregation levels, each a tuple of grouping columns. `()` is the grand total.
            Twelve of them for M5 — see `benchmarks.m5.M5_LEVELS`.
        weight_col: Column holding per-cell sales VALUE, not units. WRMSSE weights by dollars.
        weight_days: Training days the weights are computed over.

    Returns:
        A `Wrmsse`.

    Raises:
        WrmsseError: required data is missing or malformed, the forecast is not exactly the full
            series-by-horizon grid, or no level scored.
    """
    if not levels:
        raise WrmsseError("WRMSSE needs at least one aggregation level")
    if panel.frequency != "D" or forecast.frequency != "D":
        raise WrmsseError(
            "published M5 WRMSSE requires daily panel and forecast frequency; "
            f"got panel={panel.frequency!r}, forecast={forecast.frequency!r}")
    if panel.target_estimand != "sales_given_availability" \
            or forecast.target_estimand != "sales_given_availability":
        raise WrmsseError(
            "published M5 WRMSSE is defined on observed unit sales; it cannot score or compare "
            f"panel={panel.target_estimand!r}, forecast={forecast.target_estimand!r}")
    if len(set(levels)) != len(levels):
        raise WrmsseError("WRMSSE aggregation levels must be unique")
    level_names = ["total" if not lv else "+".join(lv) for lv in levels]
    if len(set(level_names)) != len(level_names):
        raise WrmsseError(
            f"WRMSSE aggregation levels have ambiguous display names: {level_names}")
    if type(weight_days) is not int or weight_days < 1:
        raise WrmsseError(
            f"weight_days must be a built-in integer >= 1, got {weight_days!r}")

    try:
        origin_date = dt.date.fromisoformat(forecast.origin)
    except (TypeError, ValueError) as exc:
        raise WrmsseError(
            f"forecast origin must be ISO YYYY-MM-DD, got {forecast.origin!r}") from exc
    if origin_date.isoformat() != forecast.origin:
        raise WrmsseError(
            f"forecast origin must be canonical ISO YYYY-MM-DD, got {forecast.origin!r}")
    if (
        type(forecast.horizon) is not int
        or forecast.horizon < 1
        or type(forecast.n_series) is not int
        or forecast.n_series < 1
    ):
        raise WrmsseError(
            "forecast horizon and n_series must both be positive integers, got "
            f"horizon={forecast.horizon!r}, n_series={forecast.n_series!r}"
        )
    try:
        weight_start = origin_date - dt.timedelta(days=weight_days - 1)
        end_date = origin_date + dt.timedelta(days=forecast.horizon)
    except OverflowError as exc:
        raise WrmsseError(
            "forecast or weight window is outside the representable calendar for "
            f"origin={forecast.origin!r}, horizon={forecast.horizon!r}, "
            f"weight_days={weight_days!r}") from exc

    raw = spark.read.parquet(panel.path)
    needed = ({panel.series_col, panel.time_col, panel.target_col, weight_col}
              | {c for lv in levels for c in lv})
    missing = sorted(needed - set(raw.columns))
    if missing:
        raise WrmsseError(
            f"panel is missing columns WRMSSE needs: {missing}. The levels and the weight column "
            "describe the benchmark's hierarchy, and a metric cannot invent one")

    requested = forecast.n_series * forecast.horizon
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0
           for value in forecast.disposition.values()):
        raise WrmsseError("forecast disposition counts must be non-negative integers")
    if sum(forecast.disposition.values()) != requested:
        raise WrmsseError(
            f"forecast disposition accounts for {sum(forecast.disposition.values())} cells, "
            f"but {requested} were requested")

    origin = F.to_date(F.lit(forecast.origin))
    end = end_date.isoformat()
    hierarchy = sorted({c for lv in levels for c in lv})
    raw_weight = F.expr(f"try_cast(`{weight_col}` as double)")
    base = raw.select(
        F.col(panel.series_col).cast("string").alias("_sid"),
        F.expr(f"try_cast(`{panel.time_col}` as date)").alias("_d"),
        F.expr(f"try_cast(`{panel.target_col}` as double)").alias("_y"),
        raw_weight.alias("_raw_w"),
        *[F.col(c) for c in hierarchy],
    )
    invalid_base = (
        F.col("_sid").isNull()
        | (F.trim("_sid") == F.lit(""))
        | F.col("_d").isNull()
        | F.col("_y").isNull()
        | F.isnan("_y")
        | (F.abs("_y") == float("inf"))
        | (F.col("_y") < 0)
    )
    for column in hierarchy:
        invalid_base = invalid_base | F.col(column).isNull()
    if base.filter(invalid_base).limit(1).count():
        raise WrmsseError(
            "panel contains a null/blank series id, invalid date, invalid/non-finite/negative "
            "target, "
            "or null hierarchy key")
    if (base.groupBy("_sid", "_d").count()
            .filter(F.col("count") != 1).limit(1).count()):
        raise WrmsseError("panel contains duplicate (series, date) rows")

    # Hierarchy membership is an identity, not a time-varying feature. If it changes over the
    # score history, an aggregate's denominator, weight, and numerator refer to different series.
    if hierarchy and (base.groupBy("_sid")
                       .agg(F.countDistinct(F.struct(*hierarchy)).alias("_memberships"))
                       .filter(F.col("_memberships") != 1).limit(1).count()):
        raise WrmsseError("a series changes WRMSSE hierarchy membership over time")

    bad_weight = base.filter(
        (F.col("_d") >= F.lit(weight_start.isoformat()))
        & (F.col("_d") <= origin)
        & (((F.col("_raw_w").isNull()) & (F.col("_y") != 0))
           | F.isnan("_raw_w")
           | (F.abs("_raw_w") == float("inf"))
           | (F.col("_raw_w") < 0))
    )
    if bad_weight.limit(1).count():
        raise WrmsseError(
            "a non-zero sale in the weight window has missing, non-finite, or negative sales "
            "value")
    # A missing price on a zero-sale cell contributes zero dollars, as it does in the published
    # aggregation. Missing value on a non-zero sale was rejected above.
    base = base.withColumn(
        "_w", F.when(F.col("_raw_w").isNull(), F.lit(0.0)).otherwise(F.col("_raw_w"))
    ).drop("_raw_w")

    # RMSSE's scale is a one-day naive benchmark. Once a series has started selling, adjacent
    # rows therefore have to be adjacent calendar days through the origin. A lag over a sparse
    # table would bridge a missing day and silently compute a different benchmark.
    history = base.filter(F.col("_d") <= origin)
    coverage = history.groupBy("_sid").agg(
        F.min(F.when(F.col("_y") > 0, F.col("_d"))).alias("_first_nz"),
        F.max("_d").alias("_last_d"),
    )
    missing_history = base.select("_sid").distinct().join(
        coverage.select("_sid"), "_sid", "left_anti").limit(1).count()
    if missing_history or coverage.filter(
            F.col("_first_nz").isNull() | (F.col("_last_d") != origin)).limit(1).count():
        raise WrmsseError(
            "every series must have a positive training observation and daily history through "
            "the forecast origin; otherwise its published RMSSE scale is undefined")
    active = history.join(coverage.select("_sid", "_first_nz"), "_sid", "inner").filter(
        F.col("_d") >= F.col("_first_nz"))
    day_window = Window.partitionBy("_sid").orderBy("_d")
    if (active.withColumn("_prev_d", F.lag("_d").over(day_window))
            .filter(F.col("_prev_d").isNotNull()
                    & (F.datediff("_d", "_prev_d") != 1)).limit(1).count()):
        raise WrmsseError(
            "training dates must be consecutive daily observations from each series' first "
            "non-zero sale through the forecast origin; a lag may not bridge missing dates")

    series: DataFrame | None = base.select("_sid").distinct().cache()
    expected: DataFrame | None = None
    preds: DataFrame | None = None
    try:
        n_series = series.count()
        if forecast.n_series != n_series:
            raise WrmsseError(
                f"forecast declares {forecast.n_series} series, but the panel contains {n_series}")
        dates = spark.createDataFrame(
            [(origin_date + dt.timedelta(days=step),)
             for step in range(1, forecast.horizon + 1)],
            "_d date",
        )
        expected = series.crossJoin(F.broadcast(dates)).cache()

        actual_keys = base.filter(
            (F.col("_d") > origin) & (F.col("_d") <= F.lit(end))
        ).select("_sid", "_d")
        missing_actuals = expected.join(actual_keys, ["_sid", "_d"], "left_anti").count()
        if missing_actuals:
            raise WrmsseError(
                f"panel is missing {missing_actuals} actual cell(s) from the full forecast grid")

        try:
            raw_preds = spark.read.parquet(forecast.path)
        except Exception as exc:
            raise WrmsseError(f"cannot read forecast artifact {forecast.path!r}: {exc}") from exc
        missing_pred_columns = sorted({"series_id", "date", "yhat"} - set(raw_preds.columns))
        if missing_pred_columns:
            raise WrmsseError(
                f"forecast is missing required columns {missing_pred_columns}")
        preds = raw_preds.select(
            F.col("series_id").cast("string").alias("_sid"),
            F.expr("try_cast(date as date)").alias("_d"),
            F.expr("try_cast(yhat as double)").alias("_yhat"),
        ).cache()
        if preds.filter(
                F.col("_sid").isNull()
                | (F.trim("_sid") == F.lit(""))
                | F.col("_d").isNull()
                | F.col("_yhat").isNull()
                | F.isnan("_yhat")
                | (F.abs("_yhat") == float("inf"))
                ).limit(1).count():
            raise WrmsseError(
                "forecast contains an invalid key/date or a non-finite prediction")
        if (preds.groupBy("_sid", "_d").count()
                .filter(F.col("count") != 1).limit(1).count()):
            raise WrmsseError("forecast contains duplicate (series_id, date) keys")
        n_pred = preds.count()
        if n_pred != forecast.n_emitted:
            raise WrmsseError(
                f"forecast disposition claims {forecast.n_emitted} emitted cells, but its "
                f"artifact contains {n_pred}")
        missing_predictions = expected.join(preds, ["_sid", "_d"], "left_anti").count()
        extra_predictions = preds.join(expected, ["_sid", "_d"], "left_anti").count()
        if missing_predictions or extra_predictions:
            raise WrmsseError(
                "forecast does not equal the published full series-by-horizon grid: "
                f"missing={missing_predictions}, extra={extra_predictions}")

        expected.unpersist()
        expected = None
        series.unpersist()
        series = None
        per_level: dict[str, float] = {}
        scored: dict[str, int] = {}
        dropped: dict[str, int] = {}
        for lv in levels:
            name = "total" if not lv else "+".join(lv)
            value, n_ok, n_drop = _level_score(base, preds, lv, origin, end, weight_days)
            per_level[name] = value
            scored[name] = n_ok
            dropped[name] = n_drop
        return Wrmsse(score=sum(per_level.values()) / len(per_level), per_level=per_level,
                      n_series_scored=scored, n_series_dropped=dropped)
    finally:
        if preds is not None:
            preds.unpersist()
        if expected is not None:
            expected.unpersist()
        if series is not None:
            series.unpersist()


def _level_score(
    base: DataFrame, preds: DataFrame, level: tuple[str, ...],
    origin, end: str, weight_days: int,
) -> tuple[float, int, int]:
    """Weighted RMSSE at one aggregation level. Returns (score, n_scored, zero).

    At an aggregated level the series IS the sum of its constituents, and the naive benchmark is
    computed on that aggregate -- not averaged from the benchmarks of its parts. Those are
    different numbers, and using the second is the commonest way a hand-rolled WRMSSE disagrees
    with the leaderboard.
    """
    keys = list(level) if level else []
    grp = keys or [F.lit(1).alias("_all")]

    agg = (base.groupBy(*grp, "_d").agg(F.sum("_y").alias("_y"), F.sum("_w").alias("_w"))
           if keys else
           base.groupBy(F.lit(1).alias("_all"), "_d").agg(F.sum("_y").alias("_y"),
                                                          F.sum("_w").alias("_w")))
    gcols = keys or ["_all"]

    if agg.filter(
            F.col("_y").isNull() | F.isnan("_y") | (F.abs("_y") == float("inf"))
            | F.col("_w").isNull() | F.isnan("_w") | (F.abs("_w") == float("inf"))
            | (F.col("_y") < 0) | (F.col("_w") < 0)).limit(1).count():
        raise WrmsseError(
            f"level {level or '(total)'} overflows or produces a non-finite aggregate")

    train = agg.filter(F.col("_d") <= origin)
    w = Window.partitionBy(*gcols).orderBy("_d")

    # Denominator: mean squared one-step change over the training period, from the first non-zero.
    first_nz = (train.filter(F.col("_y") > 0).groupBy(*gcols)
                .agg(F.min("_d").alias("_first_nz")))
    diffs = (train.join(first_nz, gcols, "inner")
             .filter(F.col("_d") >= F.col("_first_nz"))
             .withColumn("_prev", F.lag("_y").over(w))
             .filter(F.col("_prev").isNotNull())
             .withColumn("_sq", F.pow(F.col("_y") - F.col("_prev"), F.lit(2))))
    denom = diffs.groupBy(*gcols).agg(F.avg("_sq").alias("_denom"))

    # Weight: share of value over the last `weight_days` of training. Sums to 1 within the level.
    wlo = F.date_sub(origin, weight_days - 1)
    weights = (train.filter(F.col("_d") >= wlo).groupBy(*gcols)
               .agg(F.sum("_w").alias("_wsum")))
    total_raw = weights.agg(F.sum("_wsum")).collect()[0][0]
    total = float(total_raw) if total_raw is not None else 0.0
    if not math.isfinite(total) or total <= 0:
        raise WrmsseError(
            f"level {level or '(total)'} has zero or non-finite sales value over the last "
            f"{weight_days} training days, so its weights are undefined")

    # Numerator: mean squared error over the horizon, on the SAME aggregate.
    pred_agg = (base.join(preds, ["_sid", "_d"], "inner")
                .filter((F.col("_d") > origin) & (F.col("_d") <= F.lit(end))))
    pred_agg = (pred_agg.groupBy(*(keys or [F.lit(1).alias("_all")]), "_d")
                .agg(F.sum("_y").alias("_y"), F.sum("_yhat").alias("_yhat")))
    if pred_agg.filter(
            F.col("_y").isNull() | F.isnan("_y") | (F.abs("_y") == float("inf"))
            | F.col("_yhat").isNull() | F.isnan("_yhat")
            | (F.abs("_yhat") == float("inf"))).limit(1).count():
        raise WrmsseError(
            f"level {level or '(total)'} has a non-finite actual or prediction aggregate")
    num = (pred_agg.withColumn("_sq", F.pow(F.col("_y") - F.col("_yhat"), F.lit(2)))
           .groupBy(*gcols).agg(F.avg("_sq").alias("_num")))

    # Start from weights so an absent denominator cannot disappear through an inner join. Every
    # group belongs to the published population; silently dropping one makes the score smaller
    # and is a different metric.
    joined = weights.join(denom, gcols, "left").join(num, gcols, "left")
    invalid = joined.filter(
        F.col("_denom").isNull() | F.isnan("_denom")
        | (F.abs("_denom") == float("inf")) | (F.col("_denom") <= 0)
        | F.col("_num").isNull() | F.isnan("_num")
        | (F.abs("_num") == float("inf")) | (F.col("_num") < 0)
        | F.col("_wsum").isNull() | F.isnan("_wsum")
        | (F.abs("_wsum") == float("inf")) | (F.col("_wsum") < 0)
    ).limit(1).count()
    if invalid:
        raise WrmsseError(
            f"level {level or '(total)'} has a missing, zero, or non-finite RMSSE scale, "
            "numerator, or weight; published WRMSSE cannot drop that group")
    stats = joined.agg(
        F.sum((F.col("_wsum") / F.lit(total))
              * F.sqrt(F.col("_num") / F.col("_denom"))).alias("_score"),
        F.count(F.lit(1)).alias("_ok"),
    ).collect()[0]
    value = float(stats["_score"]) if stats["_score"] is not None else float("nan")
    if not math.isfinite(value) or value < 0:
        raise WrmsseError(
            f"level {level or '(total)'} produced non-finite weighted RMSSE {value!r}")
    return value, int(stats["_ok"]), 0


def eligible_only(panel: Panel) -> str:
    """The eligibility filter WRMSSE does NOT apply, named so its absence is deliberate.

    M5's published score is computed over the full grid, leading zeros included, because that is
    what the leaderboard did. Filtering to `ELIGIBLE_STATES` here would produce a number that is
    better than the published one for a reason that has nothing to do with forecasting, and it
    would not be comparable to anything.

    Returns a description, so a caller quoting a WRMSSE can state which convention it used.
    """
    return (f"WRMSSE computed on the FULL grid, matching the published definition. Eligibility "
            f"({sorted(ELIGIBLE_STATES)}) is deliberately NOT applied: the leaderboard number "
            f"does not apply it, and a filtered score is not comparable to it.")


__all__ = ["WEIGHT_DAYS", "Wrmsse", "WrmsseError", "eligible_only", "score"]
