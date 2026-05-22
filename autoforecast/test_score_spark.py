"""Parity between the two scoring paths, on REAL M5 data.

The whole reason two implementations of one metric are allowed to exist: `score.py` is stdlib and
takes plain tuples, `score_spark.py` aggregates natively over Parquet. Either alone is defensible;
both without this test is how a metric quietly starts disagreeing with itself.

The independent path here is genuinely independent -- it reads the raw columns with pandas and
builds `Cell`s from them, rather than borrowing any classification from `score_spark`. Comparing a
helper to itself would prove nothing.
"""

from __future__ import annotations

import datetime as dt
import os
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyspark.sql import SparkSession

from autoforecast.adapters.naive import seasonal_naive
from autoforecast.panel import ELIGIBLE_STATES, Forecast, Panel
from autoforecast.score import Cell, score
from autoforecast.score_spark import ScoreError, score_forecast

ROOT = pathlib.Path(__file__).resolve().parent.parent
STORE = ROOT / "data" / "m5_parquet" / "store_id=CA_1"
ORIGIN = "2016-04-24"
HORIZON = 7

_SPARK: SparkSession | None = None


def spark() -> SparkSession:
    """One local session for the file; starting a JVM per test dominates the runtime."""
    global _SPARK
    if _SPARK is None:
        if not os.environ.get("JAVA_HOME"):
            raise RuntimeError(
                "JAVA_HOME is not set, so Spark cannot start and this parity test would prove "
                "nothing. Set JAVA_HOME to a Java 17 installation.")
        _SPARK = (SparkSession.builder.master("local[*]").appName("test_score_spark")
                  .config("spark.sql.shuffle.partitions", "8")
                  .config("spark.sql.session.timeZone", "Asia/Kolkata")
                  .config("spark.ui.enabled", "false")
                  .config("spark.ui.showConsoleProgress", "false")
                  .getOrCreate())
        _SPARK.sparkContext.setLogLevel("ERROR")
    return _SPARK


def panel(**over: object) -> Panel:
    if "path" not in over and not STORE.is_dir():
        pytest.skip("public M5 Parquet is not installed")
    kw: dict[str, object] = dict(path=STORE, series_col="id", time_col="date", target_col="qty",
                                 exposure_col="observation_state", season_length=7)
    kw.update(over)
    return Panel(**kw).validate()  # type: ignore[arg-type]


def _forecast(p: Panel, origin: str = ORIGIN, horizon: int = HORIZON):
    d = tempfile.mkdtemp()
    return seasonal_naive(p, origin, horizon, spark=spark(), out_dir=d)


def _cells_the_independent_way(p: Panel, fc, missing_actual: str) -> list[Cell]:
    """Build `Cell`s from the raw columns with pandas, borrowing nothing from `score_spark`."""
    pred = pd.read_parquet(fc.path)
    act = pd.read_parquet(str(p.path), columns=["id", "date", "qty", "observation_state"]) \
        if p.has_exposure else pd.read_parquet(str(p.path), columns=["id", "date", "qty"])
    merged = pred.merge(act, how="left", left_on=["series_id", "date"], right_on=["id", "date"])
    cells = []
    for r in merged.itertuples(index=False):
        present = pd.notna(r.qty)
        state = getattr(r, "observation_state", None) if p.has_exposure else None
        eligible = True if state is None or pd.isna(state) else state in ELIGIBLE_STATES
        if present:
            actual, actual_state = float(r.qty), "observed"
        else:
            actual, actual_state = None, missing_actual
        cells.append(Cell(client="m5", origin=ORIGIN, store="CA_1", product=str(r.series_id),
                          day=str(r.date), forecast=float(r.yhat), actual=actual,
                          eligible=bool(eligible), actual_state=actual_state))
    return cells


def test_the_two_scoring_paths_agree_on_real_m5() -> None:
    p = panel()
    fc = _forecast(p)
    spark_score = score_forecast(p, fc, spark(), missing_actual="no_data")
    plain_score = score(_cells_the_independent_way(p, fc, "no_data"))
    for field in ("n_scored", "n_dropped_ineligible", "n_dropped_no_data", "n_zero_filled"):
        assert getattr(spark_score, field) == getattr(plain_score, field), \
            f"{field}: spark={getattr(spark_score, field)} plain={getattr(plain_score, field)}"
    for field in ("sum_abs_error", "sum_squared_error", "sum_error", "sum_actual"):
        a, b = getattr(spark_score, field), getattr(plain_score, field)
        assert abs(a - b) < 1e-6, f"{field}: spark={a} plain={b}"
    assert abs(spark_score.mae - plain_score.mae) < 1e-9
    assert abs(spark_score.rmse - plain_score.rmse) < 1e-9, \
        f"rmse: spark={spark_score.rmse} plain={plain_score.rmse}"
    assert abs(spark_score.bias - plain_score.bias) < 1e-9


def test_the_two_paths_agree_when_eligibility_actually_bites() -> None:
    # At a recent origin M5 has nothing ineligible (measured: 0.0% of the source window), so the
    # test above never exercises the eligibility branch. This one does: 71.2% of the early origin's
    # source window is unobserved, and the paths must agree on which cells that removes.
    p = panel()
    fc = _forecast(p, origin="2011-02-01")
    spark_score = score_forecast(p, fc, spark(), missing_actual="no_data")
    plain_score = score(_cells_the_independent_way(p, fc, "no_data"))
    assert spark_score.n_dropped_ineligible == plain_score.n_dropped_ineligible
    assert abs(spark_score.sum_abs_error - plain_score.sum_abs_error) < 1e-6
    assert abs(spark_score.sum_squared_error - plain_score.sum_squared_error) < 1e-6


def test_a_sparse_panel_and_a_dense_one_score_the_same_forecast_differently() -> None:
    # M5 is dense, so neither convention fires and the two must agree exactly. That is the point:
    # it proves the parameter is not silently changing a score it has no business changing here.
    p = panel()
    fc = _forecast(p)
    dense = score_forecast(p, fc, spark(), missing_actual="no_data")
    sparse = score_forecast(p, fc, spark(), missing_actual="sparse_zero")
    assert dense.n_scored == sparse.n_scored
    assert dense.n_zero_filled == 0 and sparse.n_zero_filled == 0
    assert dense.n_dropped_no_data == 0


def test_missing_actual_has_no_default_and_refuses_anything_else() -> None:
    p = Panel("/not-read", "id", "date", "qty")
    fc = Forecast("not-read", "/not-read", ORIGIN, 1, 1, {"emitted": 1})
    try:
        score_forecast(p, fc, None, missing_actual="whatever")  # type: ignore[arg-type]
    except ValueError as e:
        assert "no default" in str(e); return
    raise AssertionError("accepted an unrecognised missing_actual convention")


def test_a_forecast_whose_manifest_disagrees_with_its_parquet_is_refused() -> None:
    # SYNTHETIC (structure): a real adapter's manifest matches. The check exists because the
    # disposition table is what makes a silent shortfall detectable, and a manifest that does not
    # match the data it describes cannot support that.
    import dataclasses
    p = _synthetic_panel([("known", "2020-01-02", 3.0)])
    fc = _synthetic_forecast([("known", "2020-01-02", 1.0)])
    lying = dataclasses.replace(
        fc,
        disposition={"emitted": 0, "declined_no_source": 1},
    )
    try:
        score_forecast(p, lying, spark(), missing_actual="no_data")
    except ScoreError as e:
        assert "the account and the data disagree" in str(e).lower(); return
    raise AssertionError("scored a forecast whose manifest does not match its Parquet")


def test_duplicate_forecast_keys_cannot_replace_a_hard_cell_at_the_same_row_count() -> None:
    """The relational counterexample row-count validation cannot catch."""
    p = _synthetic_panel([
        ("known", "2020-01-02", 0.0),
        ("known", "2020-01-03", 100.0),
    ])
    fc = _synthetic_forecast(
        [("known", "2020-01-02", 0.0), ("known", "2020-01-02", 0.0)],
        horizon=2,
    )
    try:
        score_forecast(p, fc, spark(), missing_actual="no_data")
    except ScoreError as exc:
        assert "duplicate" in str(exc).lower(); return
    raise AssertionError("two copies of one easy cell replaced a missing hard key")


def _synthetic_panel(
    rows: list[tuple[object, str, object]],
    *,
    frequency: str = "D",
) -> Panel:
    out = pathlib.Path(tempfile.mkdtemp()) / "panel"
    key_cast = "cast(date as timestamp_ntz)" if frequency == "H" else "to_date(date)"
    spark().createDataFrame(rows, "id string, date string, qty double") \
        .selectExpr("id", f"{key_cast} as date", "qty") \
        .write.mode("overwrite").parquet(str(out))
    return Panel(str(out), "id", "date", "qty", frequency=frequency).validate()


def _synthetic_forecast(
    rows: list[tuple[object, str, object]],
    *,
    schema: str = "series_id string, date string, yhat double",
    horizon: int = 1,
    n_series: int = 1,
    frequency: str = "D",
    origin: str | None = None,
) -> Forecast:
    out = pathlib.Path(tempfile.mkdtemp()) / "forecast"
    key_cast = "cast(date as timestamp_ntz)" if frequency == "H" else "to_date(date)"
    spark().createDataFrame(rows, schema) \
        .selectExpr("series_id", f"{key_cast} as date", "yhat") \
        .write.mode("overwrite").parquet(str(out))
    if origin is None:
        origin = "2020-01-01T00:00:00" if frequency == "H" else "2020-01-01"
    return Forecast("synthetic", str(out), origin, horizon, n_series,
                    {"emitted": len(rows)}, frequency=frequency)


def test_scorer_uses_the_declared_period_grid_and_refuses_frequency_mismatch() -> None:
    p = _synthetic_panel([
        ("known", "2020-01-08", 3.0),
        ("known", "2020-01-15", 5.0),
    ])
    p = Panel(p.path, p.series_col, p.time_col, p.target_col, frequency="W")
    fc = _synthetic_forecast([
        ("known", "2020-01-08", 2.0),
        ("known", "2020-01-15", 7.0),
    ], horizon=2, frequency="W")
    result = score_forecast(p, fc, spark(), missing_actual="no_data")
    assert result.n_scored == 2 and result.sum_abs_error == 3.0

    with pytest.raises(ScoreError, match="frequency"):
        score_forecast(
            Panel(p.path, p.series_col, p.time_col, p.target_col, frequency="7D"),
            fc,
            spark(),
            missing_actual="no_data",
        )


def test_hourly_scorer_preserves_two_hours_on_one_day() -> None:
    assert spark().conf.get("spark.sql.session.timeZone") == "Asia/Kolkata"
    p = _synthetic_panel([
        ("known", "2020-01-01 23:00:00", 3.0),
        ("known", "2020-01-02 00:00:00", 8.0),
    ], frequency="H")
    fc = _synthetic_forecast([
        ("known", "2020-01-01 23:00:00", 2.0),
        ("known", "2020-01-02 00:00:00", 5.0),
    ], horizon=2, frequency="H", origin="2020-01-01T22:00:00")
    result = score_forecast(p, fc, spark(), missing_actual="no_data")
    assert result.n_scored == 2
    assert result.sum_abs_error == 4.0


@pytest.mark.parametrize(
    "frequency,origin,actuals,forecasts",
    [
        ("M", "2024-01-31", [("known", "2024-02-29", 3.0),
                              ("known", "2024-03-31", 5.0)],
         [("known", "2024-02-29", 2.0), ("known", "2024-03-31", 7.0)]),
        ("MS", "2024-01-01", [("known", "2024-02-01", 3.0),
                                ("known", "2024-03-01", 5.0)],
         [("known", "2024-02-01", 2.0), ("known", "2024-03-01", 7.0)]),
        ("Y", "2023-12-31", [("known", "2024-12-31", 3.0),
                               ("known", "2025-12-31", 5.0)],
         [("known", "2024-12-31", 2.0), ("known", "2025-12-31", 7.0)]),
        ("YS", "2024-01-01", [("known", "2025-01-01", 3.0),
                                ("known", "2026-01-01", 5.0)],
         [("known", "2025-01-01", 2.0), ("known", "2026-01-01", 7.0)]),
    ],
)
def test_calendar_scorer_uses_exact_anchored_period_keys(
    frequency, origin, actuals, forecasts,
) -> None:
    p = _synthetic_panel(actuals, frequency=frequency)
    fc = _synthetic_forecast(
        forecasts, horizon=2, frequency=frequency, origin=origin,
    )
    result = score_forecast(p, fc, spark(), missing_actual="no_data")
    assert result.n_scored == 2 and result.sum_abs_error == 3.0


def test_month_end_aliases_compare_by_canonical_semantics() -> None:
    p = _synthetic_panel([("known", "2024-02-29", 3.0)], frequency="M")
    fc = _synthetic_forecast(
        [("known", "2024-02-29", 2.0)], frequency="ME", origin="2024-01-31",
    )
    result = score_forecast(p, fc, spark(), missing_actual="no_data")
    assert result.n_scored == 1 and result.sum_abs_error == 1.0


def test_scorer_rejects_hourly_date_type_and_nonmidnight_daily_key() -> None:
    hourly_panel = _synthetic_panel([
        ("known", "2020-01-01 23:00:00", 3.0),
    ], frequency="H")
    date_out = pathlib.Path(tempfile.mkdtemp()) / "date-forecast"
    spark().createDataFrame(
        [("known", dt.date(2020, 1, 1), 2.0)],
        "series_id string, date date, yhat double",
    ).write.mode("overwrite").parquet(str(date_out))
    date_fc = Forecast(
        "synthetic", str(date_out), "2020-01-01T22:00:00", 1, 1, {"emitted": 1},
        frequency="H",
    )
    with pytest.raises(ScoreError, match="timestamp_ntz physical type"):
        score_forecast(hourly_panel, date_fc, spark(), missing_actual="no_data")

    daily_panel = _synthetic_panel([("known", "2020-01-02", 3.0)])
    timestamp_out = pathlib.Path(tempfile.mkdtemp()) / "timestamp-forecast"
    spark().createDataFrame(
        [("known", dt.datetime(2020, 1, 2, 1), 2.0)],
        "series_id string, date timestamp_ntz, yhat double",
    ).write.mode("overwrite").parquet(str(timestamp_out))
    timestamp_fc = Forecast(
        "synthetic", str(timestamp_out), "2020-01-01", 1, 1, {"emitted": 1}, frequency="D",
    )
    with pytest.raises(ScoreError, match="lossy non-midnight"):
        score_forecast(daily_panel, timestamp_fc, spark(), missing_actual="no_data")


def test_scorer_reports_invalid_parquet_time_storage_as_score_error() -> None:
    panel = _synthetic_panel([
        ("known", "2020-01-01 01:00:00", 3.0),
    ], frequency="H")
    output = pathlib.Path(tempfile.mkdtemp()) / "aware-int96-forecast"
    output.mkdir()
    pq.write_table(
        pa.table({
            "series_id": ["known"],
            "date": pa.array(
                [dt.datetime(2020, 1, 1, 1)], type=pa.timestamp("us", tz="Asia/Kathmandu")
            ),
            "yhat": [2.0],
        }),
        output / "part.parquet",
        use_deprecated_int96_timestamps=True,
    )
    forecast = Forecast(
        "synthetic", output, "2020-01-01T00:00:00", 1, 1, {"emitted": 1}, frequency="H"
    )

    with pytest.raises(ScoreError, match="invalid time storage.*timezone-bearing"):
        score_forecast(panel, forecast, spark(), missing_actual="no_data")


def test_scorer_rejects_calendar_key_off_the_declared_grid() -> None:
    p = _synthetic_panel([("known", "2024-02-29", 3.0)], frequency="M")
    fc = _synthetic_forecast(
        [("known", "2024-02-28", 2.0)], frequency="M", origin="2024-01-31",
    )
    with pytest.raises(ScoreError, match="off-anchor|outside"):
        score_forecast(p, fc, spark(), missing_actual="no_data")


@pytest.mark.parametrize("origin,horizon", [("20200101", 1), ("2020-01-01", True)])
def test_scorer_refuses_noncanonical_span_before_spark_io(origin, horizon) -> None:
    p = Panel("/not-read", "id", "date", "qty")
    fc = Forecast("candidate", "/not-read", origin, horizon, 1, {"emitted": 1})
    with pytest.raises(ScoreError, match="canonical|horizon"):
        score_forecast(p, fc, object(), missing_actual="no_data")


def _synthetic_grid(
    rows: list[tuple[object, str]],
    *,
    frequency: str = "D",
) -> str:
    out = pathlib.Path(tempfile.mkdtemp()) / "grid"
    key_cast = "cast(date as timestamp_ntz)" if frequency == "H" else "to_date(date)"
    spark().createDataFrame(rows, "series_id string, date string") \
        .selectExpr("series_id", f"{key_cast} as date") \
        .write.mode("overwrite").parquet(str(out))
    return str(out)


def test_non_castable_yhat_is_refused_with_ansi_disabled() -> None:
    """Regression: a failed cast used to become NULL and disappear from every predicate."""
    p = _synthetic_panel([("known", "2020-01-02", 3.0)])
    fc = _synthetic_forecast(
        [("known", "2020-01-02", "not-a-number")],
        schema="series_id string, date string, yhat string",
    )
    old_ansi = spark().conf.get("spark.sql.ansi.enabled")
    spark().conf.set("spark.sql.ansi.enabled", "false")
    try:
        try:
            score_forecast(p, fc, spark(), missing_actual="no_data")
        except ScoreError as exc:
            assert "yhat must be finite" in str(exc).lower(); return
        raise AssertionError("a malformed yhat was scored after its failed cast became null")
    finally:
        spark().conf.set("spark.sql.ansi.enabled", old_ansi)


def test_boolean_forecast_and_actual_values_are_not_silently_cast_to_numbers() -> None:
    p = _synthetic_panel([("known", "2020-01-02", 3.0)])
    boolean_fc = _synthetic_forecast(
        [("known", "2020-01-02", True)],
        schema="series_id string, date string, yhat boolean",
    )
    try:
        score_forecast(p, boolean_fc, spark(), missing_actual="no_data")
    except ScoreError as exc:
        assert "not boolean" in str(exc).lower()
    else:
        raise AssertionError("Spark cast a boolean forecast to 1/0 although local score refuses it")

    panel_out = pathlib.Path(tempfile.mkdtemp()) / "boolean-panel"
    spark().createDataFrame(
        [("known", "2020-01-02", True)], "id string, date string, qty boolean",
    ).selectExpr("id", "to_date(date) as date", "qty") \
        .write.mode("overwrite").parquet(str(panel_out))
    boolean_panel = Panel(str(panel_out), "id", "date", "qty")
    fc = _synthetic_forecast([("known", "2020-01-02", 1.0)])
    try:
        score_forecast(boolean_panel, fc, spark(), missing_actual="no_data")
    except ScoreError as exc:
        assert "not boolean" in str(exc).lower()
    else:
        raise AssertionError("Spark cast a boolean actual to 1/0 although local score refuses it")


def test_spark_naive_mae_obeys_the_local_score_numeric_domain() -> None:
    p = _synthetic_panel([("known", "2020-01-02", 3.0)])
    fc = _synthetic_forecast([("known", "2020-01-02", 1.0)])
    for invalid in (True, False, 0.0, -1.0, 10**400, float("nan"), float("inf"), "1"):
        try:
            score_forecast(p, fc, spark(), missing_actual="no_data", naive_mae=invalid)
        except ValueError as exc:
            assert "finite and strictly positive" in str(exc)
        else:
            raise AssertionError(f"accepted invalid naive_mae={invalid!r}")


def test_malformed_forecast_date_cannot_hide_a_large_error_as_no_data() -> None:
    p = _synthetic_panel([
        ("known", "2020-01-02", 1.0),
        ("known", "2020-01-03", 1.0),
    ])
    out = pathlib.Path(tempfile.mkdtemp()) / "string-date-forecast"
    spark().createDataFrame([
        ("known", "2020-01-02", 1.0),
        ("known", "2020-01-02x", 999.0),
    ], "series_id string, date string, yhat double") \
        .write.mode("overwrite").parquet(str(out))
    fc = Forecast("synthetic", str(out), "2020-01-01", 2, 1, {"emitted": 2})
    try:
        score_forecast(p, fc, spark(), missing_actual="no_data")
    except ScoreError as exc:
        assert "valid calendar date" in str(exc).lower()
        return
    raise AssertionError("a malformed key dropped yhat=999 from the score as no-data")


@pytest.mark.parametrize(
    "value",
    [
        "2020-01-02 00:00:00 America/New_York",
        "2020-01-02 00:00:00+00:00:30",
    ],
)
def test_timezone_bearing_forecast_text_cannot_reach_scoring(value: str) -> None:
    panel = _synthetic_panel([("known", "2020-01-02", 5.0)])
    out = pathlib.Path(tempfile.mkdtemp()) / "timezone-forecast"
    spark().createDataFrame(
        [("known", value, 5.0)], "series_id string, date string, yhat double",
    ).write.mode("overwrite").parquet(str(out))
    forecast = Forecast("synthetic", str(out), "2020-01-01", 1, 1, {"emitted": 1})

    with pytest.raises(ScoreError, match="valid calendar date"):
        score_forecast(panel, forecast, spark(), missing_actual="no_data")


def test_malformed_panel_date_cannot_hide_a_large_error_as_no_data() -> None:
    panel_out = pathlib.Path(tempfile.mkdtemp()) / "string-date-panel"
    spark().createDataFrame([
        ("known", "2020-01-02", 1.0),
        ("known", "2020-01-03x", 1.0),
    ], "id string, date string, qty double") \
        .write.mode("overwrite").parquet(str(panel_out))
    p = Panel(str(panel_out), "id", "date", "qty")
    fc = _synthetic_forecast([
        ("known", "2020-01-02", 1.0),
        ("known", "2020-01-03", 999.0),
    ], horizon=2)
    old_ansi = spark().conf.get("spark.sql.ansi.enabled")
    spark().conf.set("spark.sql.ansi.enabled", "false")
    try:
        try:
            score_forecast(p, fc, spark(), missing_actual="no_data")
        except ScoreError as exc:
            assert "invalid calendar date" in str(exc).lower()
            return
        raise AssertionError("a malformed actual key dropped yhat=999 from the score as no-data")
    finally:
        spark().conf.set("spark.sql.ansi.enabled", old_ansi)


def test_null_forecast_series_id_cannot_score_as_a_sparse_zero() -> None:
    p = _synthetic_panel([("known", "2020-01-02", 3.0)])
    fc = _synthetic_forecast([(None, "2020-01-02", 0.0)])
    try:
        score_forecast(p, fc, spark(), missing_actual="sparse_zero")
    except ScoreError as exc:
        assert "series_id must be non-null" in str(exc).lower(); return
    raise AssertionError("a null forecast series_id was converted into a perfect sparse zero")


def test_unknown_forecast_series_id_cannot_score_as_a_sparse_zero() -> None:
    p = _synthetic_panel([("known", "2020-01-02", 3.0)])
    fc = _synthetic_forecast([("never-seen", "2020-01-02", 0.0)])
    try:
        score_forecast(p, fc, spark(), missing_actual="sparse_zero")
    except ScoreError as exc:
        assert "absent from the panel" in str(exc).lower(); return
    raise AssertionError("an unknown forecast series_id was converted into a perfect sparse zero")


def test_known_series_history_is_not_enough_to_infer_a_sparse_zero() -> None:
    """Historical existence does not prove that this series was eligible on an absent date."""
    p = _synthetic_panel([("known", "2020-01-01", 3.0)])
    fc = _synthetic_forecast([("known", "2020-01-02", 2.0)])
    try:
        score_forecast(p, fc, spark(), missing_actual="sparse_zero")
    except ScoreError as exc:
        assert "eligible_grid" in str(exc); return
    raise AssertionError("series history alone was accepted as absent-cell eligibility proof")


def test_canonical_grid_can_prove_a_known_series_sparse_zero() -> None:
    p = _synthetic_panel([("known", "2020-01-01", 3.0)])
    fc = _synthetic_forecast([("known", "2020-01-02", 2.0)])
    grid = _synthetic_grid([("known", "2020-01-02")])
    result = score_forecast(
        p, fc, spark(), missing_actual="sparse_zero", eligible_grid=grid)
    assert result.n_scored == 1
    assert result.n_zero_filled == 1
    assert result.mae == 2.0


def test_hourly_canonical_grid_proves_the_exact_missing_hour() -> None:
    p = _synthetic_panel([
        ("known", "2020-01-01 21:00:00", 3.0),
    ], frequency="H")
    fc = _synthetic_forecast(
        [("known", "2020-01-01 23:00:00", 2.0)],
        frequency="H",
        origin="2020-01-01T22:00:00",
    )
    eligible = _synthetic_grid(
        [("known", "2020-01-01 23:00:00")], frequency="H",
    )
    result = score_forecast(
        p, fc, spark(), missing_actual="sparse_zero", eligible_grid=eligible,
    )
    assert result.n_scored == 1
    assert result.n_zero_filled == 1
    assert result.mae == 2.0


def test_absent_forecast_key_outside_canonical_grid_is_ineligible_not_zero() -> None:
    p = _synthetic_panel([
        ("eligible", "2020-01-01", 1.0),
        ("not-eligible", "2020-01-01", 1.0),
    ])
    fc = _synthetic_forecast([
        ("eligible", "2020-01-02", 2.0),
        ("not-eligible", "2020-01-02", 1000.0),
    ], n_series=2)
    grid = _synthetic_grid([("eligible", "2020-01-02")])
    result = score_forecast(
        p, fc, spark(), missing_actual="sparse_zero", eligible_grid=grid)
    assert result.n_scored == 1
    assert result.n_zero_filled == 1
    assert result.n_dropped_ineligible == 1
    assert result.mae == 2.0


def test_present_row_with_null_target_is_not_a_sparse_zero() -> None:
    """`missing_actual` classifies a missing row, never a present row's missing value."""
    p = _synthetic_panel([("known", "2020-01-02", None)])
    fc = _synthetic_forecast([("known", "2020-01-02", 0.0)])
    try:
        score_forecast(p, fc, spark(), missing_actual="sparse_zero")
    except ScoreError as exc:
        assert "present actual row" in str(exc).lower(); return
    raise AssertionError("a present null target was fabricated into a sparse zero")


def test_present_non_finite_or_negative_actual_is_refused() -> None:
    """Spark aggregation must enforce the same actual-value domain as pure `score()`."""
    for value in (float("nan"), float("inf"), float("-inf"), -1.0):
        p = _synthetic_panel([("known", "2020-01-02", value)])
        fc = _synthetic_forecast([("known", "2020-01-02", 0.0)])
        try:
            score_forecast(p, fc, spark(), missing_actual="no_data")
        except ScoreError as exc:
            assert "finite and non-negative" in str(exc).lower()
            continue
        raise AssertionError(f"invalid present actual was scored: {value!r}")


def test_scorer_refuses_cross_estimand_actuals_before_reading_artifacts() -> None:
    p = Panel("/tmp/not-read", "id", "date", "qty")
    fc = Forecast("latent", "/tmp/not-read", "2020-01-01", 1, 1, {"emitted": 1},
                  target_estimand="unconstrained_demand")
    try:
        score_forecast(p, fc, None, missing_actual="no_data")
    except ScoreError as exc:
        assert "different quantities" in str(exc)
        return
    raise AssertionError("scored latent demand against observed-sales actuals")


def test_spark_scorer_refuses_aggregate_overflow_from_finite_cells() -> None:
    p = _synthetic_panel([
        ("a", "2020-01-02", 0.0),
        ("b", "2020-01-02", 0.0),
    ])
    fc = _synthetic_forecast([
        ("a", "2020-01-02", 1e308),
        ("b", "2020-01-02", 1e308),
    ], n_series=2)
    try:
        score_forecast(p, fc, spark(), missing_actual="no_data")
    except ScoreError as exc:
        assert "overflowed" in str(exc)
        return
    raise AssertionError("Spark published an infinite metric from finite forecast cells")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
