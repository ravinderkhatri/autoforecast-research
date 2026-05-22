"""The honest metric over Parquet, at estate scale.

`score.py` holds the arithmetic and imports nothing but the stdlib. This module holds the join and
the aggregation, and produces the SAME `Score`. Two implementations of one definition is a real
risk, and the answer taken here is to make disagreement fail the build: `test_score_spark.py`
scores a real M5 slice both ways -- constructing `Cell`s independently from the raw columns -- and
requires the two to be identical.

The aggregation is native Catalyst. Mapping partitions through `score()` would keep exactly one
implementation, which is tempting, but it pays a Python round-trip per cell across 59M cells on
M5 and far more on an estate. The parity test buys the same guarantee for the cost of one test.
"""

from __future__ import annotations

import math
from typing import Literal

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F
from spec.forecast_spec import STATE_CODES

from autoforecast._numeric import is_finite_real
from autoforecast.frequency import Cadence, FrequencyError, parse_frequency, period_points
from autoforecast.grid import GridError, canonical_key, read_time_parquet
from autoforecast.panel import ELIGIBLE_STATES, Forecast, Panel
from autoforecast.score import Score

MissingActual = Literal["sparse_zero", "no_data"]


class ScoreError(ValueError):
    """A forecast and the panel it is scored against do not agree, or nothing is left to score."""


def _read_time_storage(
    spark: SparkSession, path: str, time_col: str, *, context: str,
) -> DataFrame:
    """Keep storage validation failures inside the scorer's public exception contract."""
    try:
        return read_time_parquet(spark, path, time_col)
    except GridError as exc:
        raise ScoreError(f"{context} has invalid time storage: {exc}") from exc


def score_forecast(
    panel: Panel,
    forecast: Forecast,
    spark: SparkSession,
    *,
    missing_actual: MissingActual,
    naive_mae: float | None = None,
    restrict_to: str | None = None,
    eligible_grid: str | None = None,
    original_missing_column: str | None = None,
    series_scores_path: str | None = None,
) -> Score:
    """Score a forecast against the panel's actuals, on eligible cells only.

    Args:
        panel: The dataset the actuals are read from. Must be the same panel the forecast was
            produced from; the join is on `series_col` and `time_col`.
        forecast: What an adapter returned. Only its EMITTED cells are scored -- a declined cell
            was never forecast, so scoring it as a zero would invent an error the adapter did not
            make. What the adapter declined is in `forecast.disposition`, and any comparison
            between adapters has to carry it, because declining the hard cells is otherwise a
            way to score well.
        spark: Active session.
        missing_actual: What a forecast cell with NO actual row means. Required, with no default,
            because it is not derivable from the join and it silently decides the score:

            · `"sparse_zero"` -- the panel is a sparse fact table where a day that sold nothing
              has no row. The cell scores as a true zero. This is the case the deployed metric
              gets wrong by inner-joining, so every over-forecast on a zero day vanishes.
            · `"no_data"` -- the panel is dense and a missing row is a source failure. The cell
              is excluded and counted, so an outage cannot become a perfect zero forecast.

            M5 is DENSE, so `"no_data"` is correct there and fires almost never; a client fact
            table is typically sparse. A sparse zero is only identifiable when `eligible_grid`
            proves the absent series-date was a scorable cell; series history alone is not proof.
        naive_mae: Denominator for MASE. Omitted, `Score.mase` is None.
        restrict_to: Optional Parquet of `series_id · date` to inner-join the forecast against
            before scoring. This is how two adapters that declined DIFFERENT cells are compared
            fairly: both are scored on the cells both produced. Without it, an adapter that
            declines the hard cells is rewarded for declining them, since its remaining cells are
            the easy ones. `select.common_cells` builds the intersection.
        eligible_grid: Optional canonical Parquet of eligible `series_id · date` cells,
            normally materialised by `autoforecast.grid.build`. Required to classify an absent
            panel row as `sparse_zero`. When supplied it is also the eligibility source: emitted
            keys outside the grid are ineligible, never zero-filled.

    Returns:
        A `Score` at `grain="cell"`, carrying the raw sums so it reduces across partitions.

    Raises:
        ScoreError: the forecast Parquet does not hold the number of rows its manifest claims,
            or no eligible cell survives.
        ValueError: `missing_actual` is not one of the two permitted values.
    """
    if naive_mae is not None and (not is_finite_real(naive_mae) or naive_mae <= 0):
        raise ValueError("naive_mae must be finite and strictly positive when supplied")
    if missing_actual not in ("sparse_zero", "no_data"):
        raise ValueError(
            f"missing_actual must be 'sparse_zero' or 'no_data', got {missing_actual!r}. It is "
            "not derivable from the join and it decides the score, so it has no default"
        )
    if forecast.target_estimand != panel.target_estimand:
        raise ScoreError(
            f"{forecast.adapter}: forecast target_estimand {forecast.target_estimand!r} does not "
            f"match panel actuals {panel.target_estimand!r}. These are different quantities and "
            "cannot share an error metric")
    try:
        forecast_cadence = parse_frequency(forecast.frequency)
        panel_cadence = parse_frequency(panel.frequency)
    except FrequencyError as exc:
        raise ScoreError(f"{forecast.adapter}: invalid forecast/panel frequency: {exc}") from exc
    if forecast_cadence.code != panel_cadence.code:
        raise ScoreError(
            f"{forecast.adapter}: forecast frequency {forecast.frequency!r} does not match "
            f"panel frequency {panel.frequency!r}")

    if (
        type(forecast.horizon) is not int
        or forecast.horizon < 1
        or type(forecast.n_series) is not int
        or forecast.n_series < 1
    ):
        raise ScoreError(
            f"{forecast.adapter}: horizon and n_series must be positive integers, got "
            f"{forecast.horizon!r} and {forecast.n_series!r}"
        )
    try:
        allowed_dates = period_points(
            forecast.origin, forecast_cadence, forecast.horizon,
        )
    except (TypeError, ValueError, OverflowError, FrequencyError) as exc:
        raise ScoreError(
            f"{forecast.adapter}: invalid canonical origin/frequency forecast grid: {exc}") from exc
    key_type = "timestamp_ntz" if forecast_cadence.key_kind == "timestamp" else "date"
    allowed_grid = spark.createDataFrame(
        [(point,) for point in allowed_dates], f"date {key_type}",
    )

    fc = _read_time_storage(
        spark, forecast.path, "date", context=f"{forecast.adapter}: forecast"
    )
    required = {"series_id", "date", "yhat"}
    missing_columns = sorted(required - set(fc.columns))
    if missing_columns:
        raise ScoreError(
            f"{forecast.adapter}: forecast Parquet lacks {missing_columns}; required columns are "
            f"{sorted(required)}")
    if fc.schema["yhat"].dataType.typeName() == "boolean":
        raise ScoreError(
            f"{forecast.adapter}: yhat must be numeric, not boolean. Casting True/False to 1/0 "
            "would create a forecast value the local scorer explicitly refuses")
    bad_identity = (fc.filter(
        F.col("series_id").isNull()
        | (F.trim(F.col("series_id").cast("string")) == F.lit(""))
    ).limit(1).count())
    if bad_identity:
        raise ScoreError(
            f"{forecast.adapter}: forecast series_id must be non-null and non-blank")

    # Normalise the relational key before checking duplicates, window membership or joining.
    # Hourly keys remain timestamps; date cadences refuse non-midnight timestamps instead of
    # silently truncating them.
    fc = _normalise_key(
        fc,
        "date",
        forecast_cadence,
        f"{forecast.adapter}: forecast date",
    )
    duplicates = (fc.groupBy("series_id", "date").count().filter(F.col("count") != 1).limit(1)
                  .count())
    if duplicates:
        raise ScoreError(
            f"{forecast.adapter}: forecast contains duplicate (series_id, date) keys. A duplicate "
            "easy cell can otherwise replace a missing hard cell while preserving the row count")
    # `cast("double")` returns NULL for malformed strings when ANSI mode is disabled.  Testing
    # only the source NULL therefore accepted values such as "not-a-number": all of the later
    # predicates also became NULL and Spark's filter discarded the row.  `try_cast` gives one
    # deterministic validation path under either ANSI setting, and its result is the exact value
    # used by the metric after validation.
    yhat = F.expr("try_cast(yhat as double)")
    bad_value = (fc.filter(yhat.isNull() | F.isnan(yhat)
                           | (F.abs(yhat) == float("inf"))
                           | (yhat < 0)).limit(1).count())
    if bad_value:
        raise ScoreError(f"{forecast.adapter}: yhat must be finite and non-negative")
    fc = fc.withColumn("yhat", yhat)

    start, end = allowed_dates[0], allowed_dates[-1]
    bad_date = fc.select("date").join(
        F.broadcast(allowed_grid), ["date"], "left_anti",
    ).limit(1).count()
    if bad_date:
        raise ScoreError(
            f"{forecast.adapter}: forecast keys fall outside [{start}, {end}] or contain null "
            "dates")
    n_rows = fc.count()
    if n_rows != forecast.n_emitted:
        raise ScoreError(
            f"{forecast.adapter}: manifest claims {forecast.n_emitted} emitted cells but the "
            f"Parquet holds {n_rows}. The account and the data disagree, and the disposition "
            "table is what makes a silent shortfall detectable -- it cannot be trusted here"
        )

    grid = _read_eligible_grid(
        eligible_grid, forecast, forecast_cadence, allowed_dates, allowed_grid, spark,
    ) \
        if eligible_grid is not None else None

    # A missing DATE for a known series can mean a true sparse zero.  A series identifier that
    # never occurs anywhere in the panel cannot: zero-filling it would let a malformed forecast
    # manufacture an arbitrary collection of perfect zero cells.  Membership is checked against
    # the complete panel rather than the scoring window because a legitimate sparse series may
    # have no rows at all during that window.
    forecast_series = fc.select(F.col("series_id").alias("_forecast_sid")).distinct()
    panel_series = (spark.read.parquet(panel.path)
                    .select(F.col(panel.series_col).alias("_panel_sid"))
                    .filter(F.col("_panel_sid").isNotNull()).distinct())
    unknown_series = (forecast_series.join(
        panel_series,
        forecast_series["_forecast_sid"] == panel_series["_panel_sid"],
        "left_anti",
    ).limit(1).count())
    if unknown_series:
        raise ScoreError(
            f"{forecast.adapter}: forecast contains a series_id absent from the panel. "
            "Only missing dates for known series can use the sparse-zero convention")

    if restrict_to is not None:
        keep_raw = _read_time_storage(
            spark, restrict_to, "date", context="restrict_to"
        )
        missing_keep = sorted({"series_id", "date"} - set(keep_raw.columns))
        if missing_keep:
            raise ScoreError(f"restrict_to lacks required columns {missing_keep}")
        keep = _normalise_key(
            keep_raw.select("series_id", "date"),
            "date",
            forecast_cadence,
            "restrict_to date",
        ).distinct()
        fc = fc.join(keep, ["series_id", "date"], "inner")
    if (original_missing_column is not None
            and (eligible_grid is None or series_scores_path is None)):
        raise ScoreError("original missingness requires a fixed grid and per-series score output")
    joined = _join_actuals(fc, panel, panel_cadence, spark, original_missing_column)
    actual = F.col("_actual")
    bad_actual = (joined.filter(
        F.col("_actual_present")
        & (actual.isNull() | F.isnan(actual)
           | (F.abs(actual) == float("inf")) | (actual < 0))
    ).limit(1).count())
    if bad_actual:
        raise ScoreError(
            f"{forecast.adapter}: a present actual row has a target that is null, non-numeric, "
            "non-finite, or negative. Actuals must be finite and non-negative; missing_actual "
            "applies only when the panel row itself is absent")
    if panel.has_exposure:
        invalid_state = (joined.filter(
            F.col("_actual_present")
            & (F.col("_state").isNull()
               | ~F.col("_state").isin(*sorted(STATE_CODES)))).limit(1).count())
        if invalid_state:
            raise ScoreError(
                f"{forecast.adapter}: an actual row has a null or unknown observation state. "
                "Only exact/lower_bound/unobserved are defined; null cannot default to eligible")
    if grid is not None:
        joined = joined.join(grid, ["series_id", "date"], "left")
    elif missing_actual == "sparse_zero" and joined.filter(
            F.col("_actual_present").isNull()).limit(1).count():
        raise ScoreError(
            f"{forecast.adapter}: sparse_zero cannot classify an absent panel row without an "
            "eligible_grid proving that exact series-date is a canonical scorable cell")
    agg = joined.agg(*_aggregations(panel, missing_actual, grid is not None)).collect()[0]

    n = int(agg["n_scored"])
    if n == 0 and original_missing_column is None:
        raise ScoreError(
            "no eligible cell survived: every forecast cell was ineligible or had no actual. "
            f"ineligible={agg['n_ineligible']}, no_data={agg['n_no_data']}"
        )
    sae = float(agg["sum_abs_error"] or 0.0)
    sse = float(agg["sum_squared_error"] or 0.0)
    se = float(agg["sum_error"] or 0.0)
    vol = float(agg["sum_actual"] or 0.0)
    if any(not math.isfinite(value) for value in (sae, sse, se, vol)):
        raise ScoreError(
            f"{forecast.adapter}: finite cells overflowed the aggregate error/volume; the "
            "requested score is not representable as float64")
    if series_scores_path is not None:
        joined.groupBy("series_id").agg(
            F.count("*").alias("expected"),
            F.count(F.when(F.col("_actual_present").isNull(), 1)).alias("originally_missing"),
            *_aggregations(panel, missing_actual, grid is not None),
        ).write.parquet(series_scores_path)
    return Score(
        n_scored=n,
        n_dropped_ineligible=int(agg["n_ineligible"]),
        n_dropped_no_data=int(agg["n_no_data"]),
        n_zero_filled=int(agg["n_zero_filled"]),
        mae=sae / n if n else 0.0,
        rmse=(sse / n) ** 0.5 if n else 0.0,
        bias=se / n if n else 0.0,
        bias_pct_of_volume=(se / vol) if vol else None,
        mase=((sae / n) / naive_mae) if naive_mae is not None and n else None,
        sum_abs_error=sae,
        sum_squared_error=sse,
        sum_error=se,
        sum_actual=vol,
        sum_naive_abs_error=(naive_mae * n) if naive_mae is not None else 0.0,
        grain="cell",
    )


def _read_eligible_grid(
    path: str,
    forecast: Forecast,
    cadence: Cadence,
    allowed_dates: tuple[object, ...],
    allowed_grid: DataFrame,
    spark: SparkSession,
) -> DataFrame:
    """Load and validate one origin's canonical eligible series-date set."""
    raw = _read_time_storage(spark, path, "date", context="eligible_grid")
    required = {"series_id", "date"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ScoreError(
            f"eligible_grid lacks {missing}; required columns are {sorted(required)}")
    grid = _normalise_key(
        raw.select("series_id", "date"), "date", cadence, "eligible_grid date",
    )
    bad_identity = (grid.filter(
        F.col("series_id").isNull()
        | (F.trim(F.col("series_id").cast("string")) == F.lit(""))
    ).limit(1).count())
    if bad_identity:
        raise ScoreError("eligible_grid contains a null/blank series_id")
    if grid.groupBy("series_id", "date").count().filter(F.col("count") != 1).limit(1).count():
        raise ScoreError(
            "eligible_grid contains duplicate (series_id, date) keys; joining it would multiply "
            "scored cells")

    start, end = allowed_dates[0], allowed_dates[-1]
    if grid.select("date").join(
        F.broadcast(allowed_grid), ["date"], "left_anti",
    ).limit(1).count():
        raise ScoreError(
            f"eligible_grid contains dates outside [{start}, {end}] for this forecast")
    return grid.withColumn("_grid_eligible", F.lit(True))


def _join_actuals(
    fc: DataFrame,
    panel: Panel,
    cadence: Cadence,
    spark: SparkSession,
    original_missing_column: str | None = None,
) -> DataFrame:
    """Left-join emitted forecast cells to the panel's actuals.

    LEFT, not inner. An inner join drops every forecast cell with no actual row, which is exactly
    the over-forecast-on-a-zero-day the deployed metric cannot see (`score.py`, defect 2). The
    join has to preserve those and let `missing_actual` decide what they mean.
    """
    raw = _read_time_storage(
        spark, panel.path, panel.time_col, context=f"panel time column {panel.time_col!r}"
    )
    if raw.schema[panel.target_col].dataType.typeName() == "boolean":
        raise ScoreError(
            f"panel target {panel.target_col!r} must be numeric, not boolean. Casting True/False "
            "to 1/0 would create an actual value the local scorer explicitly refuses")
    cols = [F.col(panel.series_col).alias("_sid"),
            F.col(panel.time_col).alias("_raw_date"),
            F.col(panel.target_col).alias("_raw_actual")]
    if panel.has_exposure:
        assert panel.exposure_col is not None
        cols.append(F.col(panel.exposure_col).alias("_state"))
    if original_missing_column is not None:
        if (original_missing_column not in raw.columns
                or raw.schema[original_missing_column].dataType.typeName() != "boolean"):
            raise ScoreError("prepared actuals lack the original-missing boolean indicator")
        flag = F.col(original_missing_column)
        cols.append(flag.alias("_original_missing"))
        cols.append(F.when(~flag, F.lit(True)).alias("_actual_present"))
    else:
        cols.append(F.lit(True).alias("_actual_present"))
    actuals = raw.select(*cols)
    actuals = _normalise_key(
        actuals, "_raw_date", cadence, f"panel time column {panel.time_col!r}", output="_date",
    ).withColumn("_actual", F.expr("try_cast(_raw_actual as double)")) \
        .drop("_raw_actual", "_raw_date")
    relevant = actuals.join(fc.select(F.col("series_id").alias("_f_sid"),
                                      F.col("date").alias("_f_date")),
                            (F.col("_sid") == F.col("_f_sid"))
                            & (F.col("_date") == F.col("_f_date")), "left_semi")
    if original_missing_column is not None:
        flag, actual = F.col("_original_missing"), F.col("_actual")
        if (relevant.filter(flag.isNull() | (flag & (actual.isNull() | (actual != 0))))
                .limit(1).count()):
            raise ScoreError("original-missing indicator requires a zero-filled target")
    if relevant.groupBy("_sid", "_date").count().filter(F.col("count") != 1).limit(1).count():
        raise ScoreError(
            "panel contains duplicate actual (series, date) keys in the forecast window; the join "
            "would multiply both errors and volume")
    return fc.join(actuals,
                   (fc["series_id"] == actuals["_sid"]) & (fc["date"] == actuals["_date"]),
                   "left")


def _normalise_key(
    frame: DataFrame,
    column: str,
    cadence: Cadence,
    context: str,
    *,
    output: str | None = None,
) -> DataFrame:
    """Validate and losslessly normalise one Spark time-key column."""
    if cadence.key_kind == "timestamp" and (
        frame.schema[column].dataType.typeName() != "timestamp_ntz"
    ):
        raise ScoreError(
            f"{context} must have timestamp_ntz physical type for hourly cadence, got "
            f"{frame.schema[column].dataType.simpleString()}"
        )
    key, valid = canonical_key(column, cadence)
    if frame.filter(~valid).limit(1).count():
        if cadence.key_kind == "timestamp":
            detail = "a valid timezone-naive exact-hour timestamp"
        else:
            detail = (
                "a valid calendar date; an invalid calendar date, lossy non-midnight timestamp, "
                "or off-anchor key is refused"
            )
        raise ScoreError(f"{context} must be {detail}")
    return frame.withColumn(output or column, key)


def _eligible(panel: Panel, has_grid: bool) -> Column:
    """Whether a joined row may be scored at all.

    A panel with no exposure column carries no eligibility information, so every cell is eligible.
    That is a property of the data -- most public benchmarks are like this -- and it is why the
    M5 run cannot demonstrate anything about the exposure contract at a recent origin.
    """
    if has_grid:
        return F.col("_grid_eligible").isNotNull()
    if not panel.has_exposure:
        return F.lit(True)
    # A row absent from the panel is distinguished by the presence marker. A PRESENT row with a
    # null state is corrupt and is refused before this expression; treating both nulls alike used
    # to make corrupt actuals silently eligible.
    return F.col("_actual_present").isNull() | F.col("_state").isin(*sorted(ELIGIBLE_STATES))


def _aggregations(
    panel: Panel,
    missing_actual: MissingActual,
    has_grid: bool = False,
) -> list[Column]:
    """The raw sums, computed natively so nothing crosses into Python per cell."""
    eligible = _eligible(panel, has_grid)
    # Absence is relational provenance, not a property of the target value.  A present row with
    # a null/non-numeric target is corrupt and is refused before this aggregation.
    absent = F.col("_actual_present").isNull()
    zero_filled = eligible & absent & F.lit(missing_actual == "sparse_zero")
    no_data = eligible & absent & F.lit(missing_actual == "no_data")
    scored = eligible & (~absent | F.lit(missing_actual == "sparse_zero"))
    actual = F.coalesce(F.col("_actual"), F.lit(0.0))
    err = F.col("yhat").cast("double") - actual
    return [
        F.count(F.when(scored, F.lit(1))).alias("n_scored"),
        F.count(F.when(~eligible, F.lit(1))).alias("n_ineligible"),
        F.count(F.when(no_data, F.lit(1))).alias("n_no_data"),
        F.count(F.when(zero_filled, F.lit(1))).alias("n_zero_filled"),
        F.sum(F.when(scored, F.abs(err))).alias("sum_abs_error"),
        # Squared error reduces through its sufficient-statistic SUM like the others: RMSE is the
        # selector. Floating summation order may still change the last bit.
        # for a MEAN forecast (MAE is minimised by the median), so it must survive partitioning.
        F.sum(F.when(scored, err * err)).alias("sum_squared_error"),
        F.sum(F.when(scored, err)).alias("sum_error"),
        F.sum(F.when(scored, actual)).alias("sum_actual"),
    ]


__all__ = ["ScoreError", "score_forecast"]
