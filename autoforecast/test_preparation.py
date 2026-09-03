"""Preparation examples use exactly the explicitly listed synthetic series."""

import datetime as dt

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import types as T

from autoforecast import preparation as prep
from autoforecast import study_references as ref
from autoforecast.preparation import prepare_time_series
from autoforecast.study_references import prepare_wrmsse_references


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[2]")
        .appName("preparation-tests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def test_preparation_distinguishes_inserted_zero_explicit_null_and_duplicates(spark):
    # One meter; days 1, 3, 4 observed. Day 2 absent, day 3 explicitly unknown.
    frame = spark.createDataFrame(
        [
            (7, dt.date(2024, 1, 1), 2.0),
            (7, dt.date(2024, 1, 3), None),
            (7, dt.date(2024, 1, 4), 5.0),
            (7, dt.date(2024, 1, 4), 1.0),
        ],
        "meter int, day date, energy double",
    )
    options = dict(
        series_id=["meter"],
        date_col="day",
        value_cols=["energy"],
        source_frequency="D",
        frequency="D",
        fill_missing="zero",
    )
    with pytest.raises(ValueError, match="duplicate"):
        prepare_time_series(frame, **options, duplicate_policy="error")
    prepared = prepare_time_series(
        frame, **options, duplicate_policy="sum", observed_col="observed"
    )
    assert [(r.energy, r.observed) for r in prepared.orderBy("day").collect()] == [
        (2.0, True),
        (0.0, False),
        (None, True),
        (6.0, True),
    ]
    assert prepared.schema["meter"].dataType == frame.schema["meter"].dataType
    with pytest.raises(ValueError, match="without finer"):
        prepare_time_series(frame, **{**options, "frequency": "H"}, duplicate_policy="sum")


def test_hourly_energy_aggregation_respects_cutoff_and_nulls(spark):
    frame = spark.createDataFrame(
        [
            ("meter-a", dt.datetime(2024, 1, 1, 0), 2.0),
            ("meter-a", dt.datetime(2024, 1, 1, 1), 3.0),
            ("meter-a", dt.datetime(2024, 1, 2, 0), 999.0),
        ],
        "meter string, at timestamp, energy double",
    )
    result = prepare_time_series(
        frame,
        series_id=["meter"],
        date_col="at",
        value_cols=["energy"],
        source_frequency="H",
        frequency="D",
        fill_missing="na",
        duplicate_policy="error",
        series_end_date=dt.datetime(2024, 1, 1, 1, tzinfo=dt.UTC),
    ).collect()
    assert len(result) == 1 and result[0].energy == 5.0
    assert result[0].at == dt.date(2024, 1, 1)


def test_references_hand_calculation_and_future_independence(spark):
    # Exactly A/B, three daily targets each. The fourth day's large values must not enter.
    rows = [
        (key, dt.date(2024, 1, day), target, revenue)
        for key, revenue in (("A", 300.0), ("B", 100.0 / 3))
        for day, target in ((1, 2.0), (2, 0.0), (3, 4.0), (4, 10000.0))
    ]
    data = spark.createDataFrame(rows, "product string, day date, units double, amount double")
    kwargs = dict(
        series_id=["product"],
        date_col="day",
        target_col="units",
        weight_col="amount",
        origins=[("tuning", dt.date(2024, 1, 3))],
        missing_history="zero",
        duplicate_policy="error",
        scale_start="history_start",
    )
    refs = prepare_wrmsse_references(data, **kwargs).orderBy("keys.product").collect()
    assert [r.weight for r in refs] == pytest.approx([900.0, 100.0])
    assert [r.scale for r in refs] == [10.0, 10.0]
    earlier = prepare_wrmsse_references(data.filter("day < '2024-01-04'"), **kwargs)
    assert earlier.orderBy("keys.product").collect() == refs
    unknown = data.replace(0.0, None, subset=["units"])
    with pytest.raises(ValueError, match="invalid weight or scale"):
        prepare_wrmsse_references(unknown, **kwargs)


def _daily_frame(spark, rows):
    return spark.createDataFrame(rows, "id string, day date, value double")


def _prepare_rows(data, **kw):
    options = dict(
        series_id=["id"],
        date_col="day",
        value_cols=["value"],
        source_frequency="D",
        frequency="D",
        fill_missing="na",
        duplicate_policy="sum",
    )
    return prepare_time_series(data, **(options | kw)).orderBy("day").collect()


def test_missing_period_na_is_not_zero(spark):
    rows = _prepare_rows(
        _daily_frame(spark, [("A", dt.date(2024, 1, 1), 2.0), ("A", dt.date(2024, 1, 3), 4.0)])
    )
    assert [r.value for r in rows] == [2.0, None, 4.0]


def test_null_in_mixed_duplicate_period_is_not_ignored(spark):
    rows = _prepare_rows(
        _daily_frame(spark, [("A", dt.date(2024, 1, 1), 2.0), ("A", dt.date(2024, 1, 1), None)])
    )
    assert rows[0].value is None


def test_null_in_mixed_coarse_period_is_not_ignored(spark):
    data = spark.createDataFrame(
        [("A", dt.datetime(2024, 1, 1, 0), 2.0), ("A", dt.datetime(2024, 1, 1, 1), None)],
        "id string, day timestamp, value double",
    )
    assert _prepare_rows(data, source_frequency="H")[0].value is None


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -float("inf")])
def test_rejects_invalid_ids_or_values(spark, bad):
    data = _daily_frame(
        spark, [(None if bad is None else "A", dt.date(2024, 1, 1), 1.0 if bad is None else bad)]
    )
    with pytest.raises(ValueError, match="null IDs|non-finite"):
        _prepare_rows(data)


@pytest.mark.parametrize(
    "frequency,dates,expected",
    [
        (
            "W",
            [dt.date(2023, 12, 31), dt.date(2024, 1, 1)],
            [dt.date(2023, 12, 25), dt.date(2024, 1, 1)],
        ),
        (
            "MS",
            [dt.date(2024, 1, 31), dt.date(2024, 3, 1)],
            [dt.date(2024, 1, 1), dt.date(2024, 2, 1), dt.date(2024, 3, 1)],
        ),
        (
            "YS",
            [dt.date(2023, 12, 31), dt.date(2024, 1, 1)],
            [dt.date(2023, 1, 1), dt.date(2024, 1, 1)],
        ),
    ],
)
def test_coarse_calendar_labels_and_completion(spark, frequency, dates, expected):
    rows = _prepare_rows(_daily_frame(spark, [("A", d, 1.0) for d in dates]), frequency=frequency)
    assert [r.day for r in rows] == expected


def _reference_rows(spark, targets, revenues=None, **kw):
    revenues = revenues or [10.0] * len(targets)
    data = spark.createDataFrame(
        [
            ("A", dt.date(2024, 1, i + 1), float(v) if v is not None else None, float(revenues[i]))
            for i, v in enumerate(targets)
        ],
        "id string, day date, target double, money double",
    )
    options = dict(
        series_id=["id"],
        date_col="day",
        target_col="target",
        weight_col="money",
        origins=[("tuning", dt.date(2024, 1, len(targets)))],
        missing_history="zero",
        duplicate_policy="sum",
    )
    return prepare_wrmsse_references(data, **(options | kw)).orderBy("origin").collect()


def test_reference_inclusive_weight_lookback(spark):
    row = _reference_rows(spark, [2, 3, 5], [1000, 20, 30], weight_window_days=2)[0]
    assert row.weight == 50.0


def test_reference_first_positive_start(spark):
    row = _reference_rows(spark, [0, 0, 2, 4], scale_start="first_positive")[0]
    assert row.scale == 4.0


def test_reference_nondefault_lag(spark):
    row = _reference_rows(spark, [1, 2, 4, 8], scale_lag=2, scale_start="history_start")[0]
    assert row.scale == 22.5


def test_reference_keeps_multiple_origins_same_stage(spark):
    rows = _reference_rows(
        spark, [1, 2, 4], origins=[("tuning", dt.date(2024, 1, 2)), ("tuning", dt.date(2024, 1, 3))]
    )
    assert [(r.stage, r.origin, r.weight, r.scale) for r in rows] == [
        ("tuning", "2024-01-02", 20.0, 1.0),
        ("tuning", "2024-01-03", 30.0, 2.5),
    ]


def test_reference_report_mode_keeps_invalid_scale(spark):
    row = _reference_rows(spark, [1, 1, 1], validate=False)[0]
    assert row.scale == 0.0


def _paired_reference_rows(spark, revenue, targets=(1.0, 2.0, 4.0), **kw):
    # Exactly A/B, each has the same 3-day target history. No eligibility exclusions.
    rows = [
        (key, dt.date(2024, 1, i + 1), target, revenue[j][i])
        for j, key in enumerate(("A", "B"))
        for i, target in enumerate(targets)
    ]
    frame = spark.createDataFrame(rows, "id string, day date, target double, revenue double")
    return (
        prepare_wrmsse_references(
            frame,
            series_id=["id"],
            date_col="day",
            target_col="target",
            weight_col="revenue",
            origins={"validation": dt.date(2024, 1, 3)},
            missing_history="na",
            duplicate_policy="error",
            **kw,
        )
        .orderBy("keys.id")
        .collect()
    )


@pytest.mark.parametrize(
    "revenue",
    [
        ((-1.0, -1.0, -1.0), (10.0, 10.0, 10.0)),
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ((1.0, None, 1.0), (1.0, 1.0, 1.0)),
        ((1e308, 1e308, 1e308), (1.0, 1.0, 1.0)),
    ],
)
def test_reference_rejects_invalid_weights_and_total(spark, revenue):
    with pytest.raises(ValueError, match="weight|total"):
        _paired_reference_rows(spark, revenue)


def test_zero_weight_series_is_preserved(spark):
    rows = _paired_reference_rows(spark, ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    assert len(rows) == 2 and [r.weight for r in rows] == [0.0, 3.0]


@pytest.mark.parametrize("targets", [(1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (1e308, -1e308, 1e308)])
def test_reference_rejects_invalid_scales(spark, targets):
    with pytest.raises(ValueError, match="scale"):
        _paired_reference_rows(spark, ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)), targets=targets)


def test_reference_report_mode_does_not_drop_invalid_weight(spark):
    rows = _paired_reference_rows(spark, ((-1.0, -1.0, -1.0), (10.0, 10.0, 10.0)), validate=False)
    assert len(rows) == 2 and [r.weight for r in rows] == [-3.0, 30.0]


def _calendar_rows(frame, **kw):
    options = dict(
        series_id=["id"],
        date_col="at",
        value_cols=["v"],
        source_frequency="D",
        frequency="D",
        fill_missing="na",
        duplicate_policy="error",
    )
    return prep.prepare_time_series(frame, **options | kw).orderBy("id", "at").collect()


def test_common_cutoff_extends_each_retained_series(spark):
    data = spark.createDataFrame(
        [
            ("A", dt.date(2024, 1, 1), 1.0),
            ("B", dt.date(2024, 1, 2), 2.0),
            ("C", dt.date(2024, 1, 6), 3.0),
        ],
        "id string, at date, v double",
    )
    rows = _calendar_rows(data, series_end_date="2024-01-04")
    assert [(r.id, r.at.day, r.v) for r in rows] == [
        ("A", 1, 1.0),
        ("A", 2, None),
        ("A", 3, None),
        ("A", 4, None),
        ("B", 2, 2.0),
        ("B", 3, None),
        ("B", 4, None),
    ]


def test_automatic_bounds_are_per_series(spark):
    data = spark.createDataFrame(
        [
            ("A", dt.date(2024, 1, 1), 1.0),
            ("A", dt.date(2024, 1, 3), 3.0),
            ("B", dt.date(2024, 1, 2), 2.0),
            ("B", dt.date(2024, 1, 4), 4.0),
        ],
        "id string, at date, v double",
    )
    rows = _calendar_rows(data)
    assert [(r.id, r.at.day) for r in rows] == [
        ("A", 1),
        ("A", 2),
        ("A", 3),
        ("B", 2),
        ("B", 3),
        ("B", 4),
    ]


@pytest.mark.parametrize(
    "cadence,value",
    [
        ("hourly", dt.datetime(2024, 1, 1, 0, 30)),
        ("daily", dt.datetime(2024, 1, 1, 1)),
        ("weekly", dt.date(2024, 1, 2)),
        ("monthly", dt.date(2024, 1, 2)),
        ("yearly", dt.date(2024, 2, 1)),
    ],
)
def test_manual_cutoff_rejects_every_misaligned_family(cadence, value):
    with pytest.raises(ValueError, match="start of a source period"):
        prep._cutoff(value, cadence)


@pytest.mark.parametrize(
    "cadence,value",
    [
        ("H", dt.datetime(2024, 1, 1, 0, 30)),
        ("D", dt.datetime(2024, 1, 1, 1)),
        ("W", dt.datetime(2024, 1, 2)),
        ("MS", dt.datetime(2024, 1, 2)),
        ("YS", dt.datetime(2024, 2, 1)),
    ],
)
def test_source_labels_reject_every_misaligned_family(spark, cadence, value):
    data = spark.createDataFrame([("A", value, 1.0)], "id string, at timestamp, v double")
    with pytest.raises(ValueError, match="aligned to source periods"):
        _calendar_rows(data, source_frequency=cadence, frequency=cadence)


def test_timestamp_requires_utc_without_changing_session(spark):
    before = spark.conf.get("spark.sql.session.timeZone")
    spark.conf.set("spark.sql.session.timeZone", "America/New_York")
    try:
        data = spark.createDataFrame(
            [("A", dt.datetime(2024, 1, 1), 1.0)], "id string, at timestamp, v double"
        )
        with pytest.raises(ValueError, match="timeZone=UTC"):
            _calendar_rows(data, source_frequency="H", frequency="H")
        assert spark.conf.get("spark.sql.session.timeZone") == "America/New_York"
    finally:
        spark.conf.set("spark.sql.session.timeZone", before)


@pytest.mark.parametrize("target", ["MS", "YS"])
def test_weekly_input_cannot_be_split_across_coarse_periods(spark, target):
    data = spark.createDataFrame([("A", dt.date(2024, 1, 1), 1.0)], "id string, at date, v double")
    with pytest.raises(ValueError, match="without finer source data"):
        _calendar_rows(data, source_frequency="W", frequency=target)


def test_cutoff_before_all_data_rejects(spark):
    data = spark.createDataFrame([("A", dt.date(2024, 1, 1), 1.0)], "id string, at date, v double")
    with pytest.raises(ValueError, match="no input rows remain"):
        _calendar_rows(data, series_end_date="2023-12-31")


def test_iso_and_aware_cutoffs_use_utc():
    expected = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
    assert prep._cutoff("2024-01-01", "daily") == expected
    assert prep._cutoff("2024-01-01T01:00:00+01:00", "hourly") == expected
    assert prep._cutoff(dt.datetime(2024, 1, 1), "hourly") == expected


def test_original_literal_names_composite_keys_values_and_numeric_widening(spark):
    schema = T.StructType(
        [
            T.StructField(n, t)
            for n, t in [
                ("item.id", T.StringType()),
                ("region`id", T.IntegerType()),
                ("day at", T.DateType()),
                ("unit.val", T.IntegerType()),
                ("cash`value", T.DoubleType()),
            ]
        ]
    )
    data = spark.createDataFrame(
        [
            ("A", 1, dt.date(2024, 1, 1), 2, 1.0),
            ("A", 1, dt.date(2024, 1, 3), 4, 3.0),
            ("A", 2, dt.date(2024, 1, 1), 7, 8.0),
        ],
        schema,
    )
    out = prep.prepare_time_series(
        data,
        series_id=["item.id", "region`id"],
        date_col="day at",
        value_cols=["unit.val", "cash`value"],
        source_frequency="D",
        fill_missing="na",
        duplicate_policy="sum",
    )
    assert out.columns == data.columns
    assert (
        out.schema["region`id"].dataType == T.IntegerType()
        and out.schema["unit.val"].dataType == T.LongType()
    )
    rows = sorted([tuple(r) for r in out.collect()])
    assert rows == [
        ("A", 1, dt.date(2024, 1, 1), 2, 1.0),
        ("A", 1, dt.date(2024, 1, 2), None, None),
        ("A", 1, dt.date(2024, 1, 3), 4, 3.0),
        ("A", 2, dt.date(2024, 1, 1), 7, 8.0),
    ]


def _history_reference_rows(spark, rows, **kw):
    data = spark.createDataFrame(rows, "id string, at date, units double, revenue double")
    options = dict(
        series_id=["id"],
        date_col="at",
        target_col="units",
        weight_col="revenue",
        origins={"validation": dt.date(2024, 1, 3)},
        missing_history="na",
        duplicate_policy="error",
    )
    return ref.prepare_wrmsse_references(data, **options | kw).orderBy("keys.id").collect()


@pytest.mark.parametrize("kind", ["future_only", "all_zero", "all_null"])
@pytest.mark.parametrize("validate", [True, False])
def test_reference_invalid_member_is_never_silently_removed(spark, kind, validate):
    rows = [("A", dt.date(2024, 1, i + 1), x, 1.0) for i, x in enumerate((1.0, 2.0, 4.0))]
    rows += (
        [("B", dt.date(2024, 1, 4), 10.0, 1.0)]
        if kind == "future_only"
        else [
            ("B", dt.date(2024, 1, i), 0.0 if kind == "all_zero" else None, 1.0)
            for i in range(1, 4)
        ]
    )
    if validate:
        with pytest.raises(ValueError, match="invalid weight or scale"):
            _history_reference_rows(spark, rows)
    else:
        result = _history_reference_rows(spark, rows, validate=False)
        assert [r["keys"]["id"] for r in result] == ["A", "B"]
        assert result[1].scale is None
        if kind == "future_only":
            assert result[1].weight is None


def test_reference_missing_history_policy_reaches_weight_and_scale(spark):
    rows = [("A", dt.date(2024, 1, 1), 1.0, 1.0), ("A", dt.date(2024, 1, 3), 4.0, 1.0)]
    result = _history_reference_rows(spark, rows, missing_history="zero")
    assert len(result) == 1 and result[0].weight == 2 and (result[0].scale == 8.5)
    with pytest.raises(ValueError, match="invalid weight or scale"):
        _history_reference_rows(spark, rows, missing_history="na")


def test_generated_references_register_and_score_the_same_population(spark, tmp_path):
    import math

    import yaml

    from autoforecast.benchmark import freeze_benchmark
    from autoforecast.test_study_tuning import config, implementation, open_study, predict

    # Exactly A/B, nine days each; no exclusions. Targets grow by one, so scales are one.
    history = spark.createDataFrame([
        (key, dt.date(2024, 1, day), float(day), revenue)
        for key, revenue in (("A", 9.0), ("B", 1.0)) for day in range(1, 10)
    ], "sku string, at date, units double, revenue double")
    references = prepare_wrmsse_references(
        history, series_id=["sku"], date_col="at", target_col="units", weight_col="revenue",
        origins=[(stage, dt.date(2024, 1, day)) for stage, day in (
            ("tuning", 3), ("validation", 5), ("test", 7)
        )], missing_history="na", duplicate_policy="error", weight_window_days=3,
    )
    rows = references.collect()
    assert len(rows) == 6 and {row["keys"]["sku"] for row in rows} == {"A", "B"}
    assert all(row["scale"] == 1.0 for row in rows)
    ref_path, source = tmp_path / "references", tmp_path / "history"
    references.write.parquet(str(ref_path))
    history.select("sku", "at", "units").write.parquet(str(source))
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text(yaml.safe_dump({"schema_version": 1, "dataset": {
        "id": "generated-references", "panel": {
            "series_col": "sku", "time_col": "at", "target_col": "units", "frequency": "D",
            "season_length": 1, "target_estimand": "observed_nonnegative",
        }
    }}))
    recipe = {
        "schema_version": 2, "source_keys": ["sku"], "training_start": "2024-01-01",
        "frequency": "D", "validation_origins": ["2024-01-05"], "validation_horizon": 2,
        "final_train_cutoff": "2024-01-07", "final_test_horizon": 2,
        "history_policy": "fixed", "new_series_policy": "error", "target_unit": "units",
        "feature_availability": {}, "metric": "wrmsse", "actual_missing": "error",
        "tuning_windows": [{"origin": "2024-01-03", "horizon": 2}],
    }
    manifest = freeze_benchmark(
        contract_path=dataset, sources=dict.fromkeys(
            ("selection", "final_training", "validation_score", "final_score"), source
        ), recipe=recipe, scoring_references=ref_path, output_root=tmp_path / "registry", spark=spark,
    )
    with open_study(manifest, spark) as study:
        run = study.run_forecast_workflow(
            fit_predict=predict, implementation=implementation(), config=config(tmp_path),
            fixed_parameters={}, output_dir=tmp_path / "runs",
        )
    for stage, expected in ((run.validation, math.sqrt(42.5)), (run.testing, math.sqrt(72.5))):
        assert stage["score"]["wrmsse"] == pytest.approx(expected)
        assert stage["score"]["population"]["scored_series"] == 2
        assert stage["score"]["population"]["expected_cells"] == 4
