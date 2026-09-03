"""Strict numeric reports use only the explicit synthetic keys and boundaries below."""

import datetime as dt
import math

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from autoforecast import study_references as ref

ORIGIN = dt.date(2024, 1, 3)
SCHEMA = "keys struct<`item.id`:int,region:string>, stage string, origin string, weight double, scale double"


@pytest.fixture(scope="module")
def spark():
    session = (SparkSession.builder.master("local[2]").appName("reference-issues-tests")
               .config("spark.sql.shuffle.partitions", "2")
               .config("spark.ui.enabled", "false").getOrCreate())
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def references(spark, values):
    return spark.createDataFrame([
        ((i, "north"), "validation", ORIGIN.isoformat(), weight, scale)
        for i, (weight, scale) in enumerate(values)
    ], SCHEMA)


def check(frame, **kwargs):
    return ref.check_wrmsse_references(frame, origins=kwargs.get("origins", {"validation": ORIGIN}))


def test_valid_report_has_stable_schema_and_preserves_zero_weight(spark):
    valid = references(spark, [(0.0, 2.0), (3.0, 4.0)])
    report = check(valid)
    assert report.collect() == []
    invalid = check(references(spark, [(None, None)]))
    assert report.schema == invalid.schema
    assert report.columns == ["keys", "stage", "origin", "field", "issue_code", "observed_value", "message"]
    assert report.schema["keys"].dataType == valid.schema["keys"].dataType
    assert report.schema["observed_value"].dataType == T.DoubleType()
    assert report.schema["keys"].nullable
    assert report.schema["observed_value"].nullable


def test_every_numeric_issue_is_reported_without_modifying_input(spark):
    bad = [None, float("nan"), float("inf"), float("-inf"), -2.0]
    frame = references(spark, [(value, value) for value in bad] + [(1.0, 0.0), (0.0, 4.0), (9.0, 4.0)])
    before = frame.toJSON().collect()
    rows = check(frame).collect()
    assert len(rows) == 11
    found = {(r["keys"]["item.id"], r.field): r for r in rows}
    assert set(found) == {(i, field) for i in range(5) for field in ("weight", "scale")} | {(5, "scale")}
    for (i, field), row in found.items():
        assert row["keys"]["region"] == "north"
        assert row.stage == "validation" and row.origin == ORIGIN.isoformat()
        assert row.issue_code == f"INVALID_{field.upper()}"
        expected = bad[i] if i < 5 else 0.0
        if expected is not None and math.isnan(expected):
            assert math.isnan(row.observed_value)
        else:
            assert row.observed_value == expected
        assert row.message and "revenue" not in row.message and "constant" not in row.message
    assert frame.toJSON().collect() == before


def test_totals_use_only_usable_rows_and_keep_origins_separate(spark):
    # Jan 1 is clean. Jan 2 has zero usable weight; Jan 3 overflows; Jan 4 has no usable
    # rows; Jan 5 has no rows at all. These are five separate tuning boundaries.
    origins = [("tuning", dt.date(2024, 1, day)) for day in range(1, 6)]
    rows = [("2024-01-01", 2.0, 1.0), ("2024-01-02", 0.0, 1.0),
            ("2024-01-02", 99.0, 0.0), ("2024-01-03", 1e308, 1.0),
            ("2024-01-03", 1e308, 1.0), ("2024-01-04", -1.0, 1.0)]
    frame = spark.createDataFrame([
        ((i, "north"), "tuning", origin, weight, scale)
        for i, (origin, weight, scale) in enumerate(rows)
    ], SCHEMA)
    issues = check(frame, origins=origins).collect()
    totals = {r.origin: r for r in issues if r.field == "total_weight"}
    assert set(totals) == {f"2024-01-0{i}" for i in range(2, 6)}
    assert totals["2024-01-02"].observed_value == 0.0
    assert totals["2024-01-03"].observed_value == float("inf")
    assert totals["2024-01-04"].observed_value is None
    assert totals["2024-01-05"].observed_value is None
    assert all(r["keys"] is None and r.stage == "tuning" and r.issue_code == "INVALID_TOTAL_WEIGHT" for r in totals.values())
    assert {(r.origin, r.field) for r in issues if r["keys"] is not None} == {
        ("2024-01-02", "scale"), ("2024-01-04", "weight"),
    }
    emptied = check(frame.filter(F.lit(False)), origins=origins).collect()
    assert len(emptied) == 5
    assert all(r.issue_code == "INVALID_TOTAL_WEIGHT" and r.observed_value is None for r in emptied)


def test_stages_sharing_a_cutoff_have_independent_totals(spark):
    frame = spark.createDataFrame([
        ((1, "north"), stage, ORIGIN.isoformat(), weight, 1.0)
        for stage, weight in (("tuning", 9.0), ("validation", 0.0), ("test", -1.0))
    ], SCHEMA)
    rows = check(frame, origins=dict.fromkeys(("tuning", "validation", "test"), ORIGIN)).collect()
    assert {(r.stage, r.issue_code) for r in rows} == {
        ("validation", "INVALID_TOTAL_WEIGHT"), ("test", "INVALID_WEIGHT"),
        ("test", "INVALID_TOTAL_WEIGHT"),
    }
    assert len(rows) == 3 and all(r.origin == ORIGIN.isoformat() for r in rows)
    totals = {r.stage: r for r in rows if r.field == "total_weight"}
    assert totals["validation"].observed_value == 0.0
    assert totals["test"].observed_value is None
    assert all(r["keys"] is None for r in totals.values())


def test_exclusions_can_be_rechecked_without_recalculating(spark):
    frame = references(spark, [(100.0, 4.0), (-10.0, 2.0), (30.0, 0.0)])
    issues = check(frame).collect()
    assert {(r["keys"]["item.id"], r.issue_code) for r in issues} == {(1, "INVALID_WEIGHT"), (2, "INVALID_SCALE")}
    retained = frame.filter(F.col("keys")["item.id"] == 0)
    assert check(retained).collect() == []
    assert retained.first().weight == 100.0
    assert check(retained.filter(F.lit(False))).first().issue_code == "INVALID_TOTAL_WEIGHT"


@pytest.mark.parametrize("origins", [None, [], [None], "validation", [("validation",)], [("validation", ORIGIN)] * 2,
                                     [("unknown", ORIGIN)], [("validation", "2024-01-03")],
                                     [("validation", dt.datetime(2024, 1, 3))], [([], ORIGIN)]])
def test_malformed_origins_raise_input_error(spark, origins):
    with pytest.raises(ValueError, match="origins"):
        check(references(spark, [(1.0, 1.0)]), origins=origins)
    history = spark.createDataFrame(
        [("A", ORIGIN, 1.0, 1.0)], "id string, at date, target double, weight double",
    )
    for validate in (True, False):
        with pytest.raises(ValueError, match="origins"):
            ref.prepare_wrmsse_references(
                history, series_id=["id"], date_col="at", target_col="target", weight_col="weight",
                origins=origins, missing_history="na", duplicate_policy="error", validate=validate,
            )


@pytest.mark.parametrize("case", ["not_frame", "missing", "duplicate", "key_type", "empty_keys", "duplicate_keys", "weight_type", "scale_type", "stage_type", "origin_type"])
def test_malformed_schema_never_returns_a_clean_report(spark, case):
    frame = references(spark, [(1.0, 1.0)])
    if case == "not_frame":
        frame = None
    elif case == "missing":
        frame = frame.drop("scale")
    elif case == "duplicate":
        frame = frame.select("*", F.col("weight").alias("WEIGHT"))
    elif case == "key_type":
        frame = frame.withColumn("keys", F.lit("series"))
    elif case == "empty_keys":
        frame = frame.withColumn("keys", F.struct())
    elif case == "duplicate_keys":
        frame = frame.withColumn("keys", F.struct(F.lit(1).alias("id"), F.lit(2).alias("ID")))
    else:
        name = case.removesuffix("_type")
        frame = frame.withColumn(name, F.col(name).cast("string" if name in {"weight", "scale"} else "date"))
    with pytest.raises(ValueError, match="reference"):
        check(frame)


@pytest.mark.parametrize("case", ["null_keys", "null_key_member", "null_stage", "null_origin", "unknown_stage", "unknown_origin", "malformed_origin"])
def test_malformed_identities_raise_input_error(spark, case):
    frame = references(spark, [(1.0, 1.0)])
    if case == "null_keys":
        frame = frame.withColumn("keys", F.lit(None).cast(frame.schema["keys"].dataType))
    elif case == "null_key_member":
        frame = frame.withColumn("keys", F.struct(F.lit(None).cast("int").alias("item.id"), F.lit("north").alias("region")))
    else:
        name = "stage" if case.endswith("stage") else "origin"
        value = None if case.startswith("null") else ("test" if name == "stage" else "2024-01-04")
        if case == "malformed_origin":
            value = "2024-1-3"
        frame = frame.withColumn(name, F.lit(value).cast("string"))
    with pytest.raises(ValueError, match="keys|boundary"):
        check(frame)


def test_numeric_input_types_are_normalized_only_in_report(spark):
    frame = references(spark, [(2.0, 0.0)]).select(
        "keys", "stage", "origin", F.col("weight").cast("long").alias("weight"),
        F.col("scale").cast("decimal(12,2)").alias("scale"),
    )
    issues = check(frame).collect()
    assert {r.issue_code for r in issues} == {"INVALID_SCALE", "INVALID_TOTAL_WEIGHT"}
    assert next(r for r in issues if r.field == "scale").observed_value == 0.0
    assert frame.schema["weight"].dataType == T.LongType()
    assert frame.schema["scale"].dataType == T.DecimalType(12, 2)


def test_prepare_uses_public_checker_and_false_preserves_values(spark, monkeypatch):
    data = spark.createDataFrame([
        ("A", dt.date(2024, 1, i), float(i), 2.0) for i in range(1, 4)
    ], "id string, at date, target double, weight double")
    options = dict(series_id=["id"], date_col="at", target_col="target", weight_col="weight",
                   origins={"validation": ORIGIN}, missing_history="na", duplicate_policy="error")
    calls = []
    checker = ref.check_wrmsse_references

    def report(frame, *, origins):
        calls.append(origins)
        return checker(frame.withColumn("scale", F.lit(0.0)), origins=origins)

    monkeypatch.setattr(ref, "check_wrmsse_references", report)
    with pytest.raises(ValueError, match="invalid weight or scale at validation/2024-01-03"):
        ref.prepare_wrmsse_references(data, **options)
    assert calls == [[("validation", ORIGIN)]]
    raw = ref.prepare_wrmsse_references(data, **options, validate=False).first()
    assert len(calls) == 1
    assert raw.weight == 6.0 and raw.scale == 1.0
