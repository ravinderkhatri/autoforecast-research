"""Distributed identity agrees with the offline format and preserves trust boundaries."""

import copy
import datetime as dt
import hashlib
import json
from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyspark.sql import SparkSession

from autoforecast import _logical_hash as logical
from provenance.lineage import directory_digest, parquet_files


@pytest.fixture(autouse=True)
def distributed_parity_cases(monkeypatch):
    # These fixtures deliberately use tiny data to exercise the distributed
    # implementation. The local-selection cases below set their own limits.
    monkeypatch.setattr(logical, "_LOCAL_MAX_ROWS", -1)


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[2]")
        .appName("logical-identity-tests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "2")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def test_hand_computed_format_and_bounded_summaries(tmp_path, spark, monkeypatch):
    path = tmp_path / "rows.parquet"
    pq.write_table(pa.table({"x": pa.array([1, None, 1, 2], type=pa.int64())}), path)
    buckets = [[] for _ in range(256)]
    for payload in (b"v1:1", b"n", b"v1:1", b"v1:2"):
        sha = hashlib.sha256(payload).digest()
        buckets[sha[0]].append(sha)
    summaries = [
        [
            i,
            len(rows),
            hashlib.sha256(
                b"autoforecast.bucket-sha256.v1/bucket\x00" + b"".join(sorted(rows))
            ).hexdigest(),
        ]
        for i, rows in enumerate(buckets)
    ]
    record = {
        "domain": "dataset",
        "algorithm": "autoforecast.bucket-sha256.v1",
        "encoding": "typed-ascii.v1",
        "bucket_count": 256,
        "schema": [["x", "i64"]],
        "rows": 4,
        "buckets": summaries,
    }
    expected = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    from pyspark.sql.classic.dataframe import DataFrame

    collect = DataFrame.collect
    observed = []

    def bounded(frame):
        assert frame.columns == ["bucket", "rows", "sha256"]
        rows = collect(frame)
        assert len(rows) <= 257
        observed.append(len(rows))
        return rows

    monkeypatch.setattr(DataFrame, "collect", bounded)
    assert logical.fingerprint(path, spark=spark)["sha256"] == expected
    assert observed
    assert logical.fingerprint(path)["sha256"] == expected


def test_supported_types_layout_timezone_and_nulls(tmp_path, spark):
    schema = pa.schema(
        [
            ("unicode", pa.string()),
            ("bytes", pa.binary()),
            ("decimal", pa.decimal128(25, 4)),
            ("date", pa.date32()),
            ("instant", pa.timestamp("us", tz="UTC")),
            ("wall", pa.timestamp("us")),
            ("float", pa.float32()),
            ("double", pa.float64()),
            ("nested", pa.struct([("z", pa.list_(pa.int16())), ("a", pa.bool_())])),
            ("map", pa.map_(pa.string(), pa.list_(pa.int64()))),
        ]
    )
    rows = [
        {
            "unicode": "a\x00é🙂",
            "bytes": b"\x00\xff",
            "decimal": Decimal("-12.3400"),
            "date": dt.date(1969, 12, 31),
            "instant": dt.datetime(1969, 12, 31, 23, 59, 59, 1, tzinfo=dt.UTC),
            "wall": dt.datetime(2024, 10, 27, 1, 30, 0, 123456),
            "float": value,
            "double": value,
            "nested": {"a": True, "z": [1, None, -2]},
            "map": [("b", []), ("a", [None, 3])],
        }
        for value in [0.0, -0.0, float("nan"), float("inf"), -float("inf"), 1.25, 5e-324]
    ]
    rows += [{name: None for name in schema.names}]
    path = tmp_path / "original.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    expected = logical.fingerprint(path)
    logical.validate_metadata(expected)
    assert logical.fingerprint(path, spark=spark) == expected
    try:
        spark.conf.set("spark.sql.session.timeZone", "America/New_York")
        assert logical.fingerprint(path, spark=spark) == expected
    finally:
        spark.conf.set("spark.sql.session.timeZone", "UTC")
    for row in rows[:-1]:
        row["map"].reverse()
    rearranged = tmp_path / "rearranged"
    rearranged.mkdir()
    for i, row in enumerate(reversed(rows)):
        pq.write_table(
            pa.Table.from_pylist([row], schema=schema),
            rearranged / f"{i}.parquet",
            compression="gzip",
        )
    spark.conf.set("spark.sql.shuffle.partitions", "3")
    try:
        assert logical.fingerprint(rearranged, spark=spark) == expected
        assert logical.fingerprint(rearranged) == expected
    finally:
        spark.conf.set("spark.sql.shuffle.partitions", "2")


def test_values_multiplicity_schema_empty_and_skew(tmp_path, spark):
    def check(values, dtype=None):
        path = tmp_path / "values.parquet"
        pq.write_table(pa.table({"x": pa.array(values, type=dtype or pa.float64())}), path)
        local = logical.fingerprint(path)
        assert logical.fingerprint(path, spark=spark) == local
        return local["sha256"]

    hashes = [check(values) for values in [[], [None], [0.0], [-0.0], [1.0], [1.0, 1.0]]]
    assert len(set(hashes)) == len(hashes)
    assert check([1.0]) != check([1], pa.int64())
    assert check([2.0] * 101) != check([2.0] * 102)


@pytest.mark.parametrize(
    "summaries,rows",
    [
        ([{"bucket": 0, "rows": 1, "sha256": "a" * 64}] * 2, 2),
        ([{"bucket": 256, "rows": 1, "sha256": "a" * 64}], 1),
        ([{"bucket": 0, "rows": 0, "sha256": "a" * 64}], 0),
        ([{"bucket": 0, "rows": 1, "sha256": "z" * 64}], 1),
        ([], 1),
    ],
)
def test_rejects_malformed_bucket_summaries(summaries, rows):
    with pytest.raises(ValueError):
        logical._finish([["x", "i64"]], summaries, rows)


def test_trusted_bytes_reuse_and_untrusted_full_verification(tmp_path, monkeypatch):
    path = tmp_path / "data"
    path.mkdir()
    pq.write_table(pa.table({"x": [1, 2]}), path / "part.parquet", compression="NONE")
    metadata = logical.fingerprint(path)
    physical = directory_digest(path, suffixes=(".parquet",))
    calls = []
    original = logical.fingerprint

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(logical, "fingerprint", counted)
    assert (
        logical.verify_files(path, physical_sha256=physical, logical=metadata, trusted=True)
        == metadata["sha256"]
    )
    assert not calls
    assert (
        logical.verify_files(path, physical_sha256=physical, logical=metadata, trusted=False)
        == metadata["sha256"]
    )
    assert calls
    before = (path / "part.parquet").stat().st_size
    pq.write_table(pa.table({"x": [3, 4]}), path / "part.parquet", compression="NONE")
    assert (path / "part.parquet").stat().st_size == before
    with pytest.raises(ValueError, match="bytes or inventory changed"):
        logical.verify_files(path, physical_sha256=physical, logical=metadata, trusted=True)
    changed = directory_digest(path, suffixes=(".parquet",))
    with pytest.raises(ValueError, match="logical identity changed"):
        logical.verify_files(path, physical_sha256=changed, logical=metadata, trusted=False)
    corrupt = copy.deepcopy(metadata)
    corrupt["algorithm"] = "unknown"
    with pytest.raises(ValueError, match="unsupported"):
        logical.verify_files(path, physical_sha256=changed, logical=corrupt, trusted=True)


def test_unsupported_and_conflicting_schemas(tmp_path):
    path = tmp_path / "rows"
    path.mkdir()
    pq.write_table(pa.table({"x": pa.array([1], type=pa.uint64())}), path / "a.parquet")
    with pytest.raises(ValueError, match="unsupported"):
        logical.fingerprint(path)
    pq.write_table(pa.table({"x": [1]}), path / "a.parquet")
    pq.write_table(pa.table({"x": ["1"]}), path / "b.parquet")
    with pytest.raises(ValueError, match="conflicting"):
        logical.fingerprint(path)


def test_int96_and_microsecond_timestamps_match_offline_and_spark(tmp_path, spark):
    from pyspark.sql.types import StructField, StructType, TimestampType

    metadata = {
        b"org.apache.spark.sql.parquet.row.metadata": StructType(
            [StructField("at", TimestampType())]
        )
        .json()
        .encode()
    }
    schema = pa.schema([("at", pa.timestamp("us", tz="UTC"))], metadata=metadata)
    table = pa.Table.from_pylist(
        [{"at": dt.datetime(year, 1, 1, tzinfo=dt.UTC)} for year in (1500, 2300)], schema=schema
    )
    old = tmp_path / "int96.parquet"
    new = tmp_path / "micros.parquet"
    pq.write_table(table, old, use_deprecated_int96_timestamps=True)
    pq.write_table(table, new)
    expected = logical.fingerprint(new)
    assert logical.fingerprint(old) == expected
    assert logical.fingerprint(old, spark=spark) == expected
    # Raw nanoseconds must never be silently rounded into the supported format.
    pq.write_table(pa.table({"at": pa.array([1], type=pa.timestamp("ns"))}), new)
    with pytest.raises(ValueError, match="unsupported"):
        logical.fingerprint(new)


def _versioned_example(tmp_path):
    from autoforecast import benchmark
    from autoforecast.test_benchmark import _frozen_example

    path, record, _ = _frozen_example(tmp_path)
    record["schema"] = benchmark.DISTRIBUTED_SCHEMA
    record["logical_hashes"] = {}
    for name in record["artifacts"]:
        result = logical.fingerprint(path.parent / record["role_paths"][name])
        record["logical_hashes"][name] = result
        record["logical_snapshots"][name] = result["sha256"]
    for group in ("grids", "coverage"):
        for key, item in record[group].items():
            result = logical.fingerprint(path.parent / item["path"])
            record["logical_hashes"][group + "/" + key] = result
            identities = "grid_identities" if group == "grids" else "coverage_identities"
            record[identities][key]["content_sha256"] = result["sha256"]
    record["study_id"] = record["benchmark_id"] = benchmark.digest(benchmark._identity(record))
    path.write_text(json.dumps(record))
    return path, record


def test_manifest_trust_reuse_and_tampered_physical_binding(tmp_path, monkeypatch):
    from autoforecast import benchmark

    path, record = _versioned_example(tmp_path)
    original = path.read_bytes()
    expected = hashlib.sha256(original).hexdigest()
    calls = []
    fingerprint = logical.fingerprint

    def counted(*args, **kwargs):
        calls.append(True)
        return fingerprint(*args, **kwargs)

    monkeypatch.setattr(logical, "fingerprint", counted)
    benchmark.verify_benchmark(path, expected, _reuse_verified=True)
    assert not calls
    benchmark.verify_benchmark(path)
    assert calls
    target = path.parent / record["role_paths"]["source-keys"]
    file = next(target.glob("*.parquet"))
    saved = file.read_bytes()
    file.rename(target / "extra.parquet")
    with pytest.raises(ValueError, match="artifact changed"):
        benchmark.verify_benchmark(path, expected, _reuse_verified=True)
    (target / "extra.parquet").rename(file)
    table = pq.read_table(file)
    pq.write_table(table.slice(1), file)
    record["artifacts"]["source-keys"] = directory_digest(target, suffixes=(".parquet",))
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="manifest changed"):
        benchmark.verify_benchmark(path, expected, _reuse_verified=True)
    with pytest.raises(ValueError, match="logical identity changed"):
        benchmark.verify_benchmark(path)
    with pytest.raises(ValueError, match="logical identity changed"):
        benchmark.verify_benchmark(path, hashlib.sha256(path.read_bytes()).hexdigest())
    path.write_bytes(original)
    file.write_bytes(saved)
    file.unlink()
    with pytest.raises(ValueError):
        benchmark.verify_benchmark(path, expected, _reuse_verified=True)


def test_verification_keeps_original_expected_manifest_bytes(tmp_path, monkeypatch):
    from autoforecast import benchmark

    path, record = _versioned_example(tmp_path)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    read = benchmark.read_benchmark_manifest
    calls = []

    def replace_after_read(*args, **kwargs):
        result = read(*args, **kwargs)
        if not calls:
            calls.append(True)
            path.write_text(json.dumps(record, indent=2))
        return result

    monkeypatch.setattr(benchmark, "read_benchmark_manifest", replace_after_read)
    with pytest.raises(ValueError, match="manifest changed"):
        benchmark.verify_benchmark(path, expected, _reuse_verified=True)


def test_distinct_digests_share_bucket_across_arrow_batches(tmp_path, spark):
    buckets = {}
    for value in range(1000):
        payload = f"v{len(str(value))}:{value}".encode()
        bucket = hashlib.sha256(payload).digest()[0]
        buckets.setdefault(bucket, []).append(value)
    values = next(values for values in buckets.values() if len(values) >= 4)
    path = tmp_path / "colliding-prefixes.parquet"
    pq.write_table(pa.table({"x": list(reversed(values)) + values}), path)
    assert logical.fingerprint(path, spark=spark) == logical.fingerprint(path)


@pytest.mark.parametrize(
    "version,expected",
    [
        (4, "c339e39b1e394e2b2c41759ac83066606f6dd9251e8f622f3936ea3fe3813697"),
        (5, "5fc30d328256ba2f314c0f6a22f304fd9e4726263516e4418ca7517733ef9768"),
    ],
)
def test_legacy_study_ids_remain_exact(tmp_path, version, expected):
    from autoforecast import benchmark
    from autoforecast.test_benchmark import _frozen_example

    path, record, _ = _frozen_example(tmp_path)
    if version == 5:
        record["schema"] = benchmark.OPTIONAL_STAGE_SCHEMA
        record["recipe"]["schema_version"] = 3
    record["study_id"] = record["benchmark_id"] = benchmark.digest(benchmark._identity(record))
    path.write_text(json.dumps(record))
    assert benchmark.verify_benchmark(path)[1]["study_id"] == expected
    assert "logical_hashes" not in record


def test_typed_ascii_reference_vectors(spark):
    from pyspark.sql import functions as F
    from pyspark.sql.types import StructField, StructType

    # These literal encodings specify the format independently of either encoder.
    vectors = [
        (None, "string", "n"),
        ("é🙂", "string", "v12:c3a9f09f9982"),
        (b"\x00\xff", "binary", "v4:00ff"),
        (True, "bool", "v1:1"),
        (-12, "i64", "v3:-12"),
        (Decimal("-12.3400"), {"decimal": [25, 4]}, "v8:-12.3400"),
        (Decimal("-0.0000"), {"decimal": [25, 4]}, "v6:0.0000"),
        (dt.date(1969, 12, 31), "date", "v2:-1"),
        (dt.datetime(1969, 12, 31, 23, 59, 59, 1, tzinfo=dt.UTC),
         "timestamp_utc", "v7:-999999"),
        (dt.datetime(1970, 1, 1, 0, 0, 0, 1), "timestamp_ntz", "v1:1"),
        (0.0, "f64", "v7:0x0.0p0"),
        (-0.0, "f64", "v8:-0x0.0p0"),
        (1.25, "f32", "v7:0x1.4p0"),
        (float("nan"), "f64", "v3:NaN"),
        (float("inf"), "f64", "v8:Infinity"),
        (-float("inf"), "f64", "v9:-Infinity"),
        ([1, None, -2], {"array": "i16"}, "v10:v1:1nv2:-2"),
        ({"z": [1, None, -2], "a": True},
         {"struct": [["a", "bool"], ["z", {"array": "i16"}]]},
         "v18:v1:1v10:v1:1nv2:-2"),
        ({"b": 2, "a": None}, {"map": ["string", "i64"]}, "v15:v2:61nv2:62v1:2"),
    ]
    schema = StructType([
        StructField(f"v{i}", logical._spark_type(spec))
        for i, (_, spec, _) in enumerate(vectors)
    ])
    frame = spark.createDataFrame([tuple(value for value, _, _ in vectors)], schema)
    encoded = frame.select(*[
        logical._expression(F.col(f"v{i}"), spec).alias(f"v{i}")
        for i, (_, spec, _) in enumerate(vectors)
    ]).first()
    for i, (value, spec, expected) in enumerate(vectors):
        assert logical._value(value, spec) == expected
        assert encoded[i] == expected


def test_canonical_schema_identity_contract(tmp_path):
    baseline = pa.schema([
        pa.field("z", pa.int64(), nullable=True),
        pa.field("a", pa.decimal128(5, 2), nullable=True),
    ])
    values = [{"z": 1, "a": Decimal("1.20")}]

    def hashed(schema, rows=values):
        target = tmp_path / "schema.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), target)
        return logical.fingerprint(target)["sha256"]

    expected = hashed(baseline)
    incidental = pa.schema([
        pa.field("a", pa.decimal128(5, 2), nullable=False, metadata={b"note": b"display"}),
        pa.field("z", pa.int64(), nullable=False),
    ], metadata={b"owner": b"example"})
    assert hashed(incidental) == expected
    for dtype in (pa.decimal128(6, 2), pa.decimal128(5, 3)):
        assert hashed(pa.schema([("z", pa.int64()), ("a", dtype)])) != expected
    assert hashed(pa.schema([("z", pa.int32()), ("a", pa.decimal128(5, 2))])) != expected
    renamed = pa.schema([("renamed", pa.int64()), ("a", pa.decimal128(5, 2))])
    assert hashed(renamed, [{"renamed": 1, "a": Decimal("1.20")}]) != expected
    assert logical._schema(pa.schema([("nested", pa.struct([
        pa.field("z", pa.int64(), nullable=True), pa.field("a", pa.string()),
    ]))])) == logical._schema(pa.schema([("nested", pa.struct([
        pa.field("a", pa.string(), nullable=False), pa.field("z", pa.int64(), nullable=False),
    ]))]))


def test_v1_offline_spill_matches_reference_and_cleans_failure(tmp_path, monkeypatch):
    from autoforecast import benchmark

    # One extra row forces two sorted runs; the duplicate crosses their boundary.
    values = list(range(65536)) + [0]
    target = tmp_path / "spill.parquet"
    pq.write_table(pa.table({"x": values}), target)
    buckets = [[] for _ in range(256)]
    for value in values:
        payload = f"v{len(str(value))}:{value}".encode("ascii")
        sha = hashlib.sha256(payload).digest()
        buckets[sha[0]].append(sha)
    record = {
        "domain": "dataset", "algorithm": "autoforecast.bucket-sha256.v1",
        "encoding": "typed-ascii.v1", "bucket_count": 256,
        "schema": [["x", "i64"]], "rows": len(values),
        "buckets": [
            [i, len(rows), hashlib.sha256(
                b"autoforecast.bucket-sha256.v1/bucket\x00" + b"".join(sorted(rows))
            ).hexdigest()] for i, rows in enumerate(buckets)
        ],
    }
    expected = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    merge = benchmark._merge_hash_runs
    observed = []

    def counted(paths, destination):
        observed.append((tuple(paths), destination))
        return merge(paths, destination)

    monkeypatch.setattr(benchmark, "_merge_hash_runs", counted)
    assert logical.fingerprint(target)["sha256"] == expected
    assert len(observed) == 1 and len(observed[0][0]) == 2
    assert not observed[0][1].parent.exists()

    def fail(paths, destination):
        observed.append((tuple(paths), destination))
        destination.write_bytes(b"partial output")
        raise OSError("merge interrupted")

    monkeypatch.setattr(benchmark, "_merge_hash_runs", fail)
    with pytest.raises(OSError, match="merge interrupted"):
        logical.fingerprint(target)
    assert not observed[-1][1].parent.exists()
    assert pq.read_metadata(target).num_rows == len(values)


def test_v6_manifest_rejects_malformed_logical_metadata(tmp_path, monkeypatch):
    from autoforecast import benchmark

    path, original = _versioned_example(tmp_path)

    def no_data_read(*args, **kwargs):
        pytest.fail("malformed metadata must fail before reading artifacts")

    monkeypatch.setattr(logical, "verify_files", no_data_read)
    cases = [
        ("algorithm", "unknown", "unsupported or malformed"),
        ("encoding", "unknown", "unsupported or malformed"),
        ("bucket_count", 257, "unsupported or malformed"),
        ("bucket_count", True, "unsupported or malformed"),
        ("rows", -1, "unsupported or malformed"),
        ("rows", True, "unsupported or malformed"),
        ("schema", [["x", "uint64"]], "invalid logical hash schema"),
        ("schema", [["x", "i64"], ["x", "i64"]], "invalid logical hash schema"),
        ("schema", [["z", "i64"], ["a", "i64"]], "invalid logical hash schema"),
        ("schema", [["x", {"decimal": [39, 0]}]], "invalid logical hash schema"),
        ("sha256", "z" * 64, "unsupported or malformed"),
        ("sha256", "0" * 64, "differs from study identity"),
    ]
    for field, value, message in cases:
        record = copy.deepcopy(original)
        record["logical_hashes"]["source-keys"][field] = value
        path.write_text(json.dumps(record))
        with pytest.raises(benchmark.BenchmarkError, match=message):
            benchmark.verify_benchmark(path)
    for field in original["logical_hashes"]["source-keys"]:
        record = copy.deepcopy(original)
        record["logical_hashes"]["source-keys"].pop(field)
        path.write_text(json.dumps(record))
        with pytest.raises(benchmark.BenchmarkError, match="unsupported or malformed"):
            benchmark.verify_benchmark(path)
    record = copy.deepcopy(original)
    record["logical_hashes"].pop("source-keys")
    path.write_text(json.dumps(record))
    with pytest.raises(benchmark.BenchmarkError, match="logical_hashes fields changed or missing"):
        benchmark.verify_benchmark(path)


def test_decimal_hash_is_ansi_independent(tmp_path, spark):
    from pyspark.sql import functions as F

    schema = pa.schema([
        (f"d{scale}", pa.decimal128(38, scale)) for scale in (0, 4, 7, 38)
    ] + [("nested", pa.struct([
        ("values", pa.list_(pa.decimal128(38, 38))),
    ]))])
    rows = [
        {**{f"d{scale}": Decimal(sign + "0." + "0" * scale) for scale in (0, 4, 7, 38)},
         "nested": {"values": [Decimal(sign + "0E-38"), None]}}
        for sign in ("", "-")
    ]
    for sign in ("", "-"):
        rows.append({
            **{f"d{scale}": Decimal(sign + ("1" if scale == 0 else "0." + "0" * (scale - 1) + "1"))
               for scale in (0, 4, 7, 38)},
            "nested": {"values": [Decimal(sign + "1E-38"), None]},
        })
    rows.append({
        "d0": Decimal("99999999999999999999999999999999999999"),
        "d4": Decimal("1234567890123456789012345678901234.5678"),
        "d7": Decimal("1234567890123456789012345678901.2345678"),
        "d38": Decimal("0.12345678901234567890123456789012345678"),
        "nested": {"values": [Decimal("0.12345678901234567890123456789012345678")]},
    })
    target = tmp_path / "decimals.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), target)
    expected = logical.fingerprint(target)
    original = spark.conf.get("spark.sql.ansi.enabled")
    try:
        for ansi in ("false", "true"):
            spark.conf.set("spark.sql.ansi.enabled", ansi)
            assert logical.fingerprint(target, spark=spark) == expected
            # Assert exact plain decimal spelling, including the zero's recorded scale.
            encoded = spark.read.parquet(str(target)).select(
                logical._expression(F.col("d7"), {"decimal": [38, 7]}).alias("value")
            ).collect()
            assert [row.value for row in encoded] == [
                "v9:0.0000000", "v9:0.0000000", "v9:0.0000001", "v10:-0.0000001",
                "v39:1234567890123456789012345678901.2345678",
            ]
    finally:
        spark.conf.set("spark.sql.ansi.enabled", original)


def test_parent_reuse_requires_authenticated_child(tmp_path, monkeypatch):
    import shutil

    from autoforecast import benchmark

    parent_path, parent = _versioned_example(tmp_path / "parent")
    child_root = tmp_path / "child"
    shutil.copytree(parent_path.parent, child_root)
    path = child_root / "benchmark.json"
    record = copy.deepcopy(parent)
    ids = sorted(row["__af_series_id"] for row in pq.read_table(
        parent_path.parent / "source-keys"
    ).to_pylist())
    record["execution_policy"] = {
        "series_policy": "use_available", "minimum_history": 1,
        "original_ids": ids, "retained_ids": ids, "excluded_ids": [],
        "cohort_sha256": benchmark.digest(ids), "interpretation": "retrospective_availability",
        "windows": {key: {
            "eligible_ids": ids, "mask_sha256": "0" * 64, "common_dates": [],
            "observed_cells": item["n_cells"],
        } for key, item in record["grids"].items()},
    }
    record["parent"] = {
        "path": str(parent_path), "sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        "study_id": parent["study_id"],
    }
    record["inherited_roles"] = ["source-keys"]

    def save_child():
        record["study_id"] = record["benchmark_id"] = benchmark.digest(benchmark._identity(record))
        path.write_text(json.dumps(record))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    expected = save_child()
    fingerprint = logical.fingerprint
    calls = []

    def counted(target, **kwargs):
        calls.append(target)
        return fingerprint(target, **kwargs)

    monkeypatch.setattr(logical, "fingerprint", counted)
    benchmark.verify_benchmark(path, expected, _reuse_verified=True)
    assert not calls
    benchmark.verify_benchmark(path)
    assert parent_path.parent / "source-keys" in calls
    assert child_root / "final_training" in calls

    # An editable child may not turn an editable parent's claims into authority.
    target = parent_path.parent / "source-keys"
    file = next(target.glob("*.parquet"))
    pq.write_table(pq.read_table(file).slice(1), file)
    parent["artifacts"]["source-keys"] = directory_digest(target, suffixes=(".parquet",))
    parent_path.write_text(json.dumps(parent))
    record["parent"]["sha256"] = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    record["artifacts"]["source-keys"] = parent["artifacts"]["source-keys"]
    self_pinned = save_child()
    with pytest.raises(benchmark.BenchmarkError, match="manifest changed"):
        benchmark.verify_benchmark(path, expected, _reuse_verified=True)
    for supplied in (None, self_pinned):
        with pytest.raises(benchmark.BenchmarkError, match="logical identity changed"):
            benchmark.verify_benchmark(path, supplied)


def test_logical_hash_events_describe_inventory_and_final_outcome(tmp_path, spark, monkeypatch):
    from autoforecast import _execution as events

    source = tmp_path / "captured.parquet"
    pq.write_table(pa.table({"x": [1, 2]}), source)
    audit = tmp_path / "events.jsonl"
    expected_bytes = source.stat().st_size
    with events.Execution(audit, ["hash"]):
        with events._stage(audit, "hash"):
            metadata = logical.fingerprint(source)
            assert logical.fingerprint(source, spark=spark) == metadata
            logical.verify_files(
                source, physical_sha256=directory_digest(source, suffixes=(".parquet",)),
                logical=metadata, trusted=True,
            )
    records = [json.loads(line) for line in audit.read_text().splitlines()]
    for label in ("Calculate distributed logical SHA", "Calculate logical SHA offline"):
        outcome = next(row for row in records if row["stage"] == label and row["status"] == "PASS")
        assert {key: outcome[key] for key in ("dataset", "files", "bytes", "rows")} == {
            "dataset": str(source), "files": 1, "bytes": expected_bytes, "rows": 2,
        }
    reused = next(row for row in records if row["stage"] == "Reuse verified logical identity")
    assert {key: reused[key] for key in ("dataset", "files", "bytes", "rows")} == {
        "dataset": str(source), "files": 1, "bytes": expected_bytes, "rows": 2,
    }
    offline = logical._offline

    def changed_after_hash(*args, **kwargs):
        result = offline(*args, **kwargs)
        pq.write_table(pa.table({"x": [3, 4]}), source)
        return result

    monkeypatch.setattr(logical, "_offline", changed_after_hash)
    failure_audit = tmp_path / "failure.jsonl"
    with pytest.raises(ValueError, match="changed during logical hashing"):
        with events.Execution(failure_audit, ["hash"]):
            with events._stage(failure_audit, "hash"):
                logical.fingerprint(source)
    records = [json.loads(line) for line in failure_audit.read_text().splitlines()]
    outcomes = [row["status"] for row in records if row["stage"] == "Calculate logical SHA offline"]
    assert "FAIL" in outcomes and "PASS" not in outcomes


@pytest.mark.parametrize(
    "artifact", ["selection", "final_training", "validation_score", "final_score", "source-keys",
                 "grid", "coverage"],
)
def test_public_scoring_contract_rejects_self_pinned_artifact_changes(tmp_path, artifact):
    from autoforecast import benchmark
    from autoforecast.study_scoring import verify_study_evaluation

    path, record = _versioned_example(tmp_path)
    recipe = record["recipe"]
    origin, horizon = recipe["final_train_cutoff"], recipe["final_test_horizon"]
    key = benchmark.boundary("test", origin, horizon)
    if artifact in {"grid", "coverage"}:
        item = record["grids" if artifact == "grid" else "coverage"][key]
        target = path.parent / item["path"]
    else:
        target = path.parent / record["role_paths"][artifact]
    file = next(target.glob("*.parquet"))
    # Keep the old logical identity while replacing the rows and updating editable byte claims.
    pq.write_table(pq.read_table(file).slice(1), file)
    replacement_sha = directory_digest(target, suffixes=(".parquet",))
    if artifact in {"grid", "coverage"}:
        item["sha256"] = replacement_sha
    else:
        record["artifacts"][artifact] = replacement_sha
    path.write_text(json.dumps(record))
    contract = benchmark._evaluation_contract_from_record(
        path, record, stage="test", origin=origin, horizon=horizon, metric=recipe["metric"],
    )
    assert contract["study"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(benchmark.BenchmarkError, match="logical identity changed"):
        benchmark.verify_benchmark(path)
    with pytest.raises(benchmark.BenchmarkError, match="logical identity changed"):
        verify_study_evaluation(contract)


def test_scoring_contract_reuses_only_separately_authenticated_study_sha(tmp_path, monkeypatch):
    from autoforecast import benchmark
    from autoforecast.study_scoring import verify_study_evaluation

    path, record = _versioned_example(tmp_path)
    recipe = record["recipe"]
    contract = benchmark._evaluation_contract_from_record(
        path, record, stage="test", origin=recipe["final_train_cutoff"],
        horizon=recipe["final_test_horizon"], metric=recipe["metric"],
    )
    protected_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert verify_study_evaluation(contract)[1] == record

    def no_logical_scan(*args, **kwargs):
        pytest.fail("a separately authenticated study SHA should reuse verified logical identities")

    monkeypatch.setattr(logical, "fingerprint", no_logical_scan)
    assert verify_study_evaluation(
        contract, _expected_study_sha256=protected_sha,
    )[1] == record
    with pytest.raises(ValueError):
        verify_study_evaluation(contract, _expected_study_sha256="f" * 64)
    altered = copy.deepcopy(contract)
    altered["study"]["sha256"] = "f" * 64
    with pytest.raises(ValueError):
        verify_study_evaluation(altered, _expected_study_sha256=protected_sha)
    # Reuse still authenticates physical bytes before any scoring can use them.
    target = path.parent / record["role_paths"]["final_score"]
    file = next(target.glob("*.parquet"))
    pq.write_table(pq.read_table(file).slice(1), file)
    with pytest.raises(benchmark.BenchmarkError, match="artifact changed"):
        verify_study_evaluation(contract, _expected_study_sha256=protected_sha)


@pytest.mark.parametrize("cache_mode", ["cache", "persist", "catalog", "sql", "sql_lazy"])
@pytest.mark.parametrize("owned_copy", [False, True])
def test_untrusted_verification_does_not_read_stale_spark_cache(
    tmp_path, spark, monkeypatch, cache_mode, owned_copy,
):
    target = tmp_path / "source"
    target.mkdir()
    part = target / "part.parquet"
    pq.write_table(pa.table({"x": [1, 2]}), part)
    original = logical.fingerprint(target)
    frame = spark.read.parquet(str(target))
    name = "logical_verification_cached_input"
    if cache_mode in {"cache", "persist"}:
        frame = getattr(frame, cache_mode)()
    else:
        frame.createOrReplaceTempView(name)
        if cache_mode == "catalog":
            spark.catalog.cacheTable(name)
        else:
            spark.sql(f"CACHE {'LAZY ' if cache_mode == 'sql_lazy' else ''}TABLE {name}")
        frame = spark.table(name)
    try:
        assert sorted(row.x for row in frame.collect()) == [1, 2]
        pq.write_table(pa.table({"x": [3, 4]}), part)
        replacement_sha = directory_digest(target, suffixes=(".parquet",))
        # The stale cache is real, including an independently resolved DataFrame.
        assert sorted(row.x for row in spark.read.parquet(str(target)).collect()) == [1, 2]
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        observed = []
        fingerprint = logical.fingerprint

        def captured(path, *, spark=None):
            observed.append((path, spark))
            return fingerprint(path, spark=spark)

        monkeypatch.setattr(logical, "fingerprint", captured)
        options = dict(
            physical_sha256=replacement_sha, trusted=False, spark=spark,
            _scratch_root=scratch if owned_copy else None,
        )
        with pytest.raises(ValueError, match="logical identity changed"):
            logical.verify_files(target, logical=original, **options)
        assert len(observed) == 1
        verified_path, engine = observed[0]
        if owned_copy:
            assert engine is spark and verified_path != target
            assert verified_path.is_relative_to(scratch) and not verified_path.exists()
        else:
            assert engine is None and verified_path == target
        assert not list(scratch.iterdir())
        # Valid replacement metadata succeeds through the same isolated route.
        replacement = fingerprint(target)
        assert logical.verify_files(target, logical=replacement, **options) == replacement["sha256"]
        assert not list(scratch.iterdir())
        assert sorted(row.x for row in frame.collect()) == [1, 2]
        if cache_mode in {"cache", "persist"}:
            assert frame.is_cached
        else:
            assert spark.catalog.isCached(name)
    finally:
        if cache_mode in {"cache", "persist"}:
            frame.unpersist(blocking=True)
        else:
            spark.catalog.uncacheTable(name)
            spark.catalog.dropTempView(name)


@pytest.mark.parametrize("change", ["source", "copy", "cancel"])
def test_owned_logical_verification_rechecks_bytes_and_always_cleans(
    tmp_path, spark, monkeypatch, change,
):
    target = tmp_path / "source"
    target.mkdir()
    part = target / "part.parquet"
    table = pa.table({"x": [1, 2]})
    pq.write_table(table, part, compression="NONE")
    metadata = logical.fingerprint(target)
    physical = directory_digest(target, suffixes=(".parquet",))
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    fingerprint = logical.fingerprint
    captured = []

    def change_after_hash(path, *, spark=None):
        captured.append(path)
        assert path != target and path.is_relative_to(scratch)
        if change == "cancel":
            raise KeyboardInterrupt("cancel verification")
        result = fingerprint(path, spark=spark)
        changed = part if change == "source" else path / part.name
        # Equal rows still need the exact authenticated bytes; recompression must fail.
        pq.write_table(table, changed, compression="gzip")
        return result

    monkeypatch.setattr(logical, "fingerprint", change_after_hash)
    error = KeyboardInterrupt if change == "cancel" else ValueError
    with pytest.raises(error, match="cancel verification|bytes changed"):
        logical.verify_files(
            target, physical_sha256=physical, logical=metadata, trusted=False,
            spark=spark, _scratch_root=scratch,
        )
    assert len(captured) == 1 and not captured[0].exists()
    assert not list(scratch.iterdir())
    assert target.exists()


@pytest.mark.parametrize("storage", ["DATE", "INT96", "TIMESTAMP_MICROS", "TIMESTAMP_MILLIS"])
@pytest.mark.parametrize("year", [1500, 2024])
def test_legacy_calendar_captures_are_rejected_before_hashing(tmp_path, spark, storage, year):
    # A separate session represents a caller using legacy writes without changing the fixture.
    session = spark.newSession()
    session.conf.set("spark.sql.parquet.datetimeRebaseModeInWrite", "LEGACY")
    session.conf.set("spark.sql.parquet.int96RebaseModeInWrite", "LEGACY")
    session.conf.set("spark.sql.parquet.outputTimestampType", "INT96" if storage == "DATE" else storage)
    dtype = "DATE" if storage == "DATE" else "TIMESTAMP"
    frame = session.sql(f"SELECT CAST('{year}-01-01' AS {dtype}) AS value")
    target = tmp_path / "legacy"
    frame.write.parquet(str(target))
    metadata = pq.read_metadata(next(target.glob("*.parquet"))).metadata
    assert {b"org.apache.spark.legacyDateTime", b"org.apache.spark.legacyINT96"}.intersection(metadata)
    for engine in (None, session):
        with pytest.raises(ValueError, match="legacy-rebased Parquet.*corrected-calendar"):
            logical.fingerprint(target, spark=engine)


@pytest.mark.parametrize("storage", ["INT96", "TIMESTAMP_MICROS", "TIMESTAMP_MILLIS"])
def test_temporal_epoch_encoding_has_no_python_year_limit(tmp_path, spark, storage):
    session = spark.newSession()
    session.conf.set("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")
    session.conf.set("spark.sql.parquet.int96RebaseModeInWrite", "CORRECTED")
    session.conf.set("spark.sql.parquet.outputTimestampType", storage)
    frame = session.sql("""
        SELECT CAST(value AS DATE) AS d, CAST(value AS TIMESTAMP) AS t,
               CAST(value AS TIMESTAMP_NTZ) AS n
        FROM VALUES ('0000-01-01'), ('+10000-01-01') AS input(value)
    """).selectExpr(
        "*", "array(d, CAST(NULL AS DATE)) AS dates",
        "struct(t AS instant, n AS wall, array(d) AS dates) AS nested",
        "map(d, struct(t AS instant, n AS wall)) AS keyed",
        "array(map(t, n)) AS nested_map",
    )
    target = tmp_path / "extended-calendar"
    frame.write.parquet(str(target))
    offline = logical.fingerprint(target)
    assert offline["rows"] == 2
    assert logical.fingerprint(target, spark=session) == offline
    assert logical._value(-719528, "date") == "v7:-719528"
    assert logical._value(2932897, "date") == "v7:2932897"
    assert logical._value(-62167219200000000, "timestamp_utc") == "v18:-62167219200000000"
    assert logical._value(253402300800000000, "timestamp_ntz") == "v18:253402300800000000"


@pytest.mark.parametrize("int96", [False, True])
def test_new_hash_uses_corrected_calendar_independently_of_read_session(tmp_path, spark, int96):
    from pyspark.sql.types import DateType, StructField, StructType, TimestampType

    # Arrow writes proleptic Gregorian values without Spark's version/rebase markers.
    declared = StructType([StructField("day", DateType()), StructField("at", TimestampType())])
    schema = pa.schema(
        [("day", pa.date32()), ("at", pa.timestamp("us", tz="UTC"))],
        metadata={b"org.apache.spark.sql.parquet.row.metadata": declared.json().encode()},
    )
    table = pa.Table.from_pylist([
        {"day": dt.date(1500, 1, 1), "at": dt.datetime(1500, 1, 1, tzinfo=dt.UTC)},
    ], schema=schema)
    target = tmp_path / "corrected.parquet"
    pq.write_table(table, target, use_deprecated_int96_timestamps=int96)
    expected = logical.fingerprint(target)
    session = spark.newSession()
    session.conf.set("spark.sql.parquet.datetimeRebaseModeInRead", "LEGACY")
    session.conf.set("spark.sql.parquet.int96RebaseModeInRead", "LEGACY")
    assert logical.fingerprint(target, spark=session) == expected
    assert session.conf.get("spark.sql.parquet.datetimeRebaseModeInRead") == "LEGACY"
    assert session.conf.get("spark.sql.parquet.int96RebaseModeInRead") == "LEGACY"


@pytest.mark.parametrize("version,int96", [("2.4.8", False), ("2.4.8", True), ("3.0.3", True)])
def test_old_spark_calendar_metadata_requires_corrected_recapture(tmp_path, version, int96):
    schema = pa.schema([("at", pa.timestamp("us", tz="UTC"))], metadata={
        b"org.apache.spark.version": version.encode(),
    })
    target = tmp_path / "old.parquet"
    pq.write_table(pa.Table.from_pylist([
        {"at": dt.datetime(1500, 1, 1, tzinfo=dt.UTC)},
    ], schema=schema), target, use_deprecated_int96_timestamps=int96)
    with pytest.raises(ValueError, match="legacy-rebased Parquet"):
        logical.fingerprint(target)


def test_concurrent_winner_rejects_cached_rebound_rows(tmp_path, spark):
    from autoforecast import benchmark

    path, record = _versioned_example(tmp_path)
    target = benchmark.artifact_path(path, record, "source-keys")
    schema = logical._spark_type({"struct": record["logical_hashes"]["source-keys"]["schema"]})
    cached = spark.read.schema(schema).parquet(str(target)).cache()
    original = cached.collect()
    try:
        part = next(target.glob("*.parquet"))
        table = pq.read_table(part)
        values = table.to_pylist()
        values[0]["sku"] += 1000
        pq.write_table(pa.Table.from_pylist(values, schema=table.schema), part)
        record["artifacts"]["source-keys"] = directory_digest(target, suffixes=(".parquet",))
        path.write_text(json.dumps(record))
        losing_capture = tmp_path / "losing-capture"
        losing_capture.mkdir()
        # A publisher now carries its original manifest bytes as the handoff.
        (losing_capture / "benchmark.json").write_text(json.dumps(record))
        with pytest.raises(benchmark.BenchmarkError, match="logical identity changed"):
            benchmark._publish_frozen_benchmark(
                losing_capture, path.parent, record["study_id"], spark=spark,
            )
        assert not losing_capture.exists()
        assert cached.is_cached and cached.collect() == original
        assert not list(tmp_path.glob("af-verify-hash-*"))
    finally:
        cached.unpersist(blocking=True)


@pytest.mark.parametrize("values", [[1000], [1001], [], [None]], ids=["aligned", "fractional", "empty", "null"])
@pytest.mark.parametrize("position", ["scalar", "struct", "array", "map_key", "map_value", "nested", "omitted"])
@pytest.mark.parametrize("instant", [False, True], ids=["wall", "instant"])
def test_declared_spark_schema_cannot_hide_physical_nanoseconds(
    tmp_path, spark, values, position, instant,
):
    dtype = pa.timestamp("ns", tz="UTC" if instant else None)
    temporal = "timestamp_utc" if instant else "timestamp_ntz"
    containers = {
        "scalar": (dtype, temporal, lambda v: v),
        "struct": (pa.struct([("at", dtype)]), {"struct": [["at", temporal]]}, lambda v: {"at": v}),
        "array": (pa.list_(dtype), {"array": temporal}, lambda v: [v]),
        "map_key": (pa.map_(dtype, pa.string()), {"map": [temporal, "string"]},
                    lambda v: [(v, "value")] if v is not None else None),
        "map_value": (pa.map_(pa.string(), dtype), {"map": ["string", temporal]}, lambda v: [("key", v)]),
        "nested": (pa.list_(pa.map_(pa.string(), pa.struct([("at", dtype)]))),
                   {"array": {"map": ["string", {"struct": [["at", temporal]]}]}},
                   lambda v: [[("key", {"at": v})]]),
    }
    if position == "omitted":
        schema = pa.schema([("x", pa.int64()), ("hidden", dtype)])
        declared = [["x", "i64"]]
        rows = [{"x": 1, "hidden": value} for value in values]
    else:
        column_type, column_spec, convert = containers[position]
        schema = pa.schema([("x", column_type)])
        declared = [["x", column_spec]]
        rows = [{"x": convert(value)} for value in values]
    schema = schema.with_metadata({
        b"org.apache.spark.sql.parquet.row.metadata":
            logical._spark_type({"struct": declared}).json().encode(),
    })
    target = tmp_path / "nanos.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), target)
    for engine in (None, spark):
        with pytest.raises(ValueError, match="unsupported.*physical timestamp"):
            logical.fingerprint(target, spark=engine)


@pytest.mark.parametrize("position", ["top", "struct", "array", "map_key", "map_value", "nested"])
def test_spark_metadata_cannot_project_away_physical_fields(tmp_path, spark, position):
    fields = pa.struct([("visible", pa.int64()), ("hidden", pa.int64())])
    projected = {"struct": [["visible", "i64"]]}
    value = {"visible": 1, "hidden": 2}
    containers = {
        "struct": (fields, projected, value),
        "array": (pa.list_(fields), {"array": projected}, [value]),
        "map_key": (pa.map_(fields, pa.string()), {"map": [projected, "string"]}, [(value, "value")]),
        "map_value": (pa.map_(pa.string(), fields), {"map": ["string", projected]}, [("key", value)]),
        "nested": (pa.list_(pa.map_(pa.string(), fields)),
                   {"array": {"map": ["string", projected]}}, [[("key", value)]]),
    }
    if position == "top":
        schema, rows, declared = pa.schema(list(fields)), [value], projected["struct"]
    else:
        dtype, spec, values = containers[position]
        schema, rows, declared = pa.schema([("x", dtype)]), [{"x": values}], [["x", spec]]
    schema = schema.with_metadata({
        b"org.apache.spark.sql.parquet.row.metadata":
            logical._spark_type({"struct": declared}).json().encode(),
    })
    target = tmp_path / "projected.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), target)
    for engine in (None, spark):
        with pytest.raises(ValueError, match="Spark-declared schema differs from physical"):
            logical.fingerprint(target, spark=engine)


@pytest.mark.parametrize("schema,rows,declared", [
    (pa.schema([("x", pa.int32())]), [{"x": 1}], [["x", "i64"]]),
    (pa.schema([("x", pa.int64())]), [{"x": 1}], [["x", "i32"]]),
    (pa.schema([("x", pa.int64())]), [{"x": 1}, {"x": 2}], [["x", "bool"]]),
    (pa.schema([("x", pa.decimal128(8, 2))]), [{"x": Decimal("1.20")}],
     [["x", {"decimal": [9, 2]}]]),
    (pa.schema([("x", pa.decimal128(8, 3))]), [{"x": Decimal("1.231")}, {"x": Decimal("1.232")}],
     [["x", {"decimal": [7, 2]}]]),
    (pa.schema([("x", pa.int64())]), [{"x": 1}], [["renamed", "i64"]]),
    (pa.schema([("x", pa.struct([("actual", pa.int64())]))]), [{"x": {"actual": 1}}],
     [["x", {"struct": [["renamed", "i64"]]}]]),
    (pa.schema([("x", pa.list_(pa.int32()))]), [{"x": [1]}], [["x", {"array": "i64"}]]),
    (pa.schema([("x", pa.map_(pa.string(), pa.int32()))]), [{"x": [("key", 1)]}],
     [["x", {"map": ["string", "i64"]}]]),
], ids=["widen", "narrow", "boolean", "decimal-precision", "decimal-scale", "name", "struct", "array", "map"])
def test_declared_spark_schema_must_match_physical_fields(tmp_path, spark, schema, rows, declared):
    schema = schema.with_metadata({
        b"org.apache.spark.sql.parquet.row.metadata":
            logical._spark_type({"struct": declared}).json().encode(),
    })
    target = tmp_path / "mismatched.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), target)
    for engine in (None, spark):
        with pytest.raises(ValueError, match="Spark-declared schema differs from physical"):
            logical.fingerprint(target, spark=engine)


@pytest.mark.parametrize("position", ["scalar", "struct", "array", "map_key", "map_value", "nested"])
@pytest.mark.parametrize("scala_class", [False, True])
def test_spark_footer_types_are_inert_at_every_depth(tmp_path, monkeypatch, position, scala_class):
    from pyspark.sql import types as T

    def forbidden(*args, **kwargs):
        pytest.fail("untrusted footer must not invoke Spark's custom-type parser")

    monkeypatch.setattr(T.UserDefinedType, "fromJson", forbidden)
    containers = {
        "scalar": (pa.int64(), 1),
        "struct": (pa.struct([("n", pa.int64())]), {"n": 1}),
        "array": (pa.list_(pa.int64()), [1]),
        "map_key": (pa.map_(pa.int64(), pa.string()), [(1, "a")]),
        "map_value": (pa.map_(pa.string(), pa.int64()), [("a", 1)]),
        "nested": (pa.list_(pa.map_(pa.string(), pa.struct([("n", pa.int64())]))),
                   [[("a", {"n": 1})]]),
    }
    dtype, value = containers[position]
    schema = pa.schema([("x", dtype)])
    declared = logical._spark_type({"struct": logical._schema(schema)}).jsonValue()

    def replace(value):
        if value == "long":
            return {"type": "udt", "pyClass": "example.CustomType", "sqlType": "long",
                    **({"class": "example.CustomType"} if scala_class else {})}
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        return value

    schema = schema.with_metadata({
        b"org.apache.spark.sql.parquet.row.metadata": json.dumps(replace(declared)).encode(),
    })
    target = tmp_path / "custom.parquet"
    pq.write_table(pa.Table.from_pylist([{"x": value}], schema=schema), target)
    with pytest.raises(ValueError, match="Spark-declared schema differs from physical"):
        logical.fingerprint(target)


@pytest.mark.parametrize("declared", [
    b"null", b"[]", b"{}", b"invalid", b'{"type":"struct","fields":null}',
    b'{"type":"struct","fields":[{}]}',
    b'{"type":"struct","fields":[{"name":"x","type":null}]}',
    b'{"type":"struct"}', b'{"type":"struct","fields":[null]}',
    b'{"type":"struct","fields":[{"name":"x"}]}',
    b'{"type":"struct","fields":[{"name":"x","type":{"type":"udt",'
    b'"pyClass":"example.CustomType","sqlType":"long"},"nullable":true,"metadata":{}}]}',
])
def test_public_verification_reports_malformed_spark_footer(tmp_path, monkeypatch, declared):
    from pyspark.sql import types as T

    from autoforecast import benchmark

    monkeypatch.setattr(T.UserDefinedType, "fromJson", lambda *args, **kwargs: pytest.fail(
        "public verification must not invoke custom-type parsing",
    ))
    path, record = _versioned_example(tmp_path)
    target = path.parent / record["role_paths"]["source-keys"]
    part = next(target.glob("*.parquet"))
    table = pq.read_table(part).replace_schema_metadata({
        b"org.apache.spark.sql.parquet.row.metadata": declared,
    })
    pq.write_table(table, part)
    record["artifacts"]["source-keys"] = directory_digest(target, suffixes=(".parquet",))
    path.write_text(json.dumps(record))
    with pytest.raises(benchmark.BenchmarkError, match="Spark-declared"):
        benchmark.verify_benchmark(path)


def test_one_spark_declaration_cannot_hide_conflicting_part_schemas(tmp_path, spark):
    metadata = {b"org.apache.spark.sql.parquet.row.metadata":
                logical._spark_type({"struct": [["x", "i64"]]}).json().encode()}
    target = tmp_path / "parts"
    target.mkdir()
    for field in ("first", "second"):
        table = pa.table({"x": [1], field: [2]}).replace_schema_metadata(metadata)
        pq.write_table(table, target / f"{field}.parquet")
    for engine in (None, spark):
        with pytest.raises(ValueError, match="Spark-declared schema differs from physical"):
            logical.fingerprint(target, spark=engine)


@pytest.mark.parametrize("native", [False, True])
def test_public_full_verification_rejects_rebound_hidden_fields(tmp_path, spark, native):
    from autoforecast import benchmark

    path, record = _versioned_example(tmp_path)
    target = path.parent / record["role_paths"]["selection"]
    part = next(target.glob("*.parquet"))
    table = pq.read_table(part)
    declaration = logical._spark_type({"struct": record["logical_hashes"]["selection"]["schema"]})
    changed = table.append_column("hidden", pa.array([1] * table.num_rows)).replace_schema_metadata({
        b"org.apache.spark.sql.parquet.row.metadata": declaration.json().encode(),
    })
    pq.write_table(changed, part)
    record["artifacts"]["selection"] = directory_digest(target, suffixes=(".parquet",))
    path.write_text(json.dumps(record))
    self_pinned = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(benchmark.BenchmarkError, match="Spark-declared schema differs from physical"):
        benchmark.verify_benchmark(
            path, self_pinned, spark=spark if native else None,
            _scratch_root=tmp_path if native else None,
        )
    assert not list(tmp_path.glob("af-verify-hash-*"))


@pytest.mark.parametrize("writer", ["arrow", "spark"])
@pytest.mark.parametrize("conversion", ["false", "true"])
def test_int96_conversion_uses_safe_backend_without_changing_session(
    tmp_path, spark, writer, conversion,
):
    from autoforecast import _execution as events

    session = spark.newSession()
    settings = {
        "spark.sql.session.timeZone": "America/New_York",
        "spark.sql.parquet.int96TimestampConversion": conversion,
        "spark.sql.parquet.datetimeRebaseModeInWrite": "CORRECTED",
        "spark.sql.parquet.int96RebaseModeInWrite": "CORRECTED",
        "spark.sql.parquet.outputTimestampType": "INT96",
    }
    for name, value in settings.items():
        session.conf.set(name, value)
    if writer == "arrow":
        target = tmp_path / "foreign.parquet"
        table = pa.table({"at": pa.array([
            dt.datetime(2024, 1, 1, tzinfo=dt.UTC),
        ], type=pa.timestamp("us", tz="UTC"))})
        pq.write_table(table, target, use_deprecated_int96_timestamps=True)
        assert not pq.read_metadata(target).created_by.startswith("parquet-mr")
    else:
        target = tmp_path / "spark-capture"
        session.sql("SELECT TIMESTAMP '2024-01-01 00:00:00' AS at").write.parquet(str(target))
        assert pq.read_metadata(next(target.glob("*.parquet"))).created_by.startswith("parquet-mr")
    expected = logical.fingerprint(target)
    audit = tmp_path / "events.jsonl"
    with events.Execution(audit, ["hash"]):
        with events._stage(audit, "hash"):
            assert logical.fingerprint(target, spark=session) == expected
    records = [json.loads(line) for line in audit.read_text().splitlines()]
    fallback = writer == "arrow" and conversion == "true"
    label = "Calculate logical SHA offline" if fallback else "Calculate distributed logical SHA"
    outcome = next(row for row in records if row["stage"] == label and row["status"] == "PASS")
    assert outcome["engine"] == ("driver" if fallback else "spark")
    if fallback:
        assert "session timezone conversion" in outcome["reason"]
        assert not any(row["stage"] == "Calculate distributed logical SHA" for row in records)
    assert logical.verify_files(
        target, physical_sha256=directory_digest(target, suffixes=(".parquet",)),
        logical=expected, trusted=False, spark=session, _scratch_root=tmp_path,
    ) == expected["sha256"]
    assert not list(tmp_path.glob("af-verify-hash-*"))
    assert {name: session.conf.get(name) for name in settings} == settings


@pytest.mark.parametrize('rows,byte_limit,local', [(4, 100000, True), (5, 100000, False),
                                                 (4, 1, False)])
def test_small_capture_engine_selection_preserves_identity(
    tmp_path, spark, monkeypatch, rows, byte_limit, local,
):
    target = tmp_path / 'input.parquet'
    pq.write_table(pa.table({'x': [None, -1, 0, 1, 2][:rows]}), target)
    expected = logical.fingerprint(target)
    monkeypatch.setattr(logical, '_LOCAL_MAX_ROWS', 4)
    monkeypatch.setattr(logical, '_LOCAL_MAX_BYTES', byte_limit)
    offline = logical._offline
    calls = []

    def recorded(*args):
        calls.append(True)
        return offline(*args)

    monkeypatch.setattr(logical, '_offline', recorded)
    assert logical.fingerprint(target, spark=spark) == expected
    assert bool(calls) is local


def test_small_capture_does_not_launch_spark_and_checks_every_schema(tmp_path, monkeypatch):
    class NoJobs:
        @property
        def read(self):
            raise AssertionError('small evidence must not launch Spark work')

    monkeypatch.setattr(logical, '_LOCAL_MAX_ROWS', 250_000)
    pq.write_table(pa.table({'x': pa.array([], type=pa.int64())}), tmp_path / 'empty.parquet')
    pq.write_table(pa.table({'x': [1, -1, None]}), tmp_path / 'rows.parquet')
    expected = logical.fingerprint(tmp_path)
    assert logical.fingerprint(tmp_path, spark=NoJobs()) == expected
    # Key and finding tables of tens of thousands of rows also stay on the driver.
    keys = tmp_path.parent / 'keys'
    keys.mkdir()
    pq.write_table(pa.table({'series': [f's{i}' for i in range(20000)],
                             'date': list(range(20000))}), keys / 'part.parquet')
    assert logical.fingerprint(keys, spark=NoJobs()) == logical.fingerprint(keys)
    pq.write_table(pa.table({'x': ['different']}), tmp_path / 'conflict.parquet')
    with pytest.raises(ValueError, match='conflicting logical schemas'):
        logical.fingerprint(tmp_path, spark=NoJobs())


@pytest.mark.parametrize('empty', [False, True])
def test_native_frame_identity_matches_file_and_offline(tmp_path, spark, empty):
    path = tmp_path / 'native'
    frame = spark.createDataFrame(
        [(1, -0.0, Decimal('1.20')), (1, -0.0, Decimal('1.20')), (None, None, None)],
        'item long, value double, price decimal(10,2)',
    )
    if empty:
        frame = frame.limit(0)
    frame.write.parquet(str(path))
    assert logical.fingerprint_frame(frame) == logical.fingerprint(path)
    assert logical.fingerprint_frame(frame) == logical.fingerprint(path, spark=spark)


@pytest.mark.parametrize('value_size,local', [(2048, True), (20000, False)])
def test_offline_batches_bound_dictionary_expansion(tmp_path, spark, monkeypatch,
                                                    value_size, local):
    path = tmp_path / 'dictionary.parquet'
    pq.write_table(pa.table({'x': ['x' * value_size] * 64}), path, use_dictionary=True)
    expected = logical.fingerprint(path, spark=spark)
    monkeypatch.setattr(logical, '_LOCAL_MAX_ROWS', 4096)
    monkeypatch.setattr(logical, '_LOCAL_MAX_BYTES', 10000)
    offline = logical._offline
    batches = []

    def recorded(path, schema, rows, batch_rows):
        batches.append(batch_rows)
        return offline(path, schema, rows, batch_rows)

    monkeypatch.setattr(logical, '_offline', recorded)
    assert logical.fingerprint(path, spark=spark) == expected
    # The file holds one dictionary entry, but every decoded row repeats it in full.
    # Offline batches must stay within the byte budget; a value above it goes to Spark.
    assert bool(batches) is local
    assert all(1 <= rows * value_size <= 10000 for rows in batches)


@pytest.mark.parametrize("kind", ["physical", "repeated"])
def test_local_limits_are_independent(tmp_path, spark, monkeypatch, kind):
    path = tmp_path / "input.parquet"
    if kind == "physical":
        table = pa.table({"x": [1]}).replace_schema_metadata({b"padding": b"a" * 10000})
    else:
        table = pa.table({"x": [["a" * 2048] * 128]})
    pq.write_table(table, path, use_dictionary=True)
    with pq.ParquetFile(path) as footer:
        decoded = sum(footer.metadata.row_group(i).total_byte_size for i in range(footer.metadata.num_row_groups))
    limit = 10000 if kind == "repeated" else 4096
    assert decoded < limit
    assert path.stat().st_size > limit if kind == "physical" else path.stat().st_size < limit
    monkeypatch.setattr(logical, "_LOCAL_MAX_ROWS", 4096)
    monkeypatch.setattr(logical, "_LOCAL_MAX_BYTES", limit)
    offline = logical._offline
    expected = logical.fingerprint(path)
    calls = []
    def local(*args):
        calls.append(True)
        return offline(*args)
    monkeypatch.setattr(logical, "_offline", local)
    assert logical.fingerprint(path, spark=spark) == expected
    assert not calls


def test_offline_batches_bound_deep_nesting_overhead(tmp_path, monkeypatch):
    # Each row holds one value under 64 list levels: tiny on disk, about 1 KiB per row
    # once Arrow materializes the offsets and validity of every level.
    dtype, value = pa.int64(), 1
    for _ in range(64):
        dtype, value = pa.list_(dtype), [value]
    column = pa.array([value] * 8192, type=dtype)
    path = tmp_path / "nested.parquet"
    pq.write_table(pa.table({name: column for name in "abcd"}), path)
    monkeypatch.setattr(logical, "_LOCAL_MAX_ROWS", 250_000)
    offline = logical._offline
    sizes = []

    def measured(path, schema, rows, batch_rows):
        for file in parquet_files(path):
            with pq.ParquetFile(file) as reader:
                sizes.extend(batch.nbytes for batch in reader.iter_batches(batch_size=batch_rows))
        return offline(path, schema, rows, batch_rows)

    class NoJobs:
        @property
        def read(self):
            raise AssertionError("a capture within the offline bound must not launch Spark")

    monkeypatch.setattr(logical, "_offline", measured)
    assert logical.fingerprint(path, spark=NoJobs()) == logical.fingerprint(path)
    assert sizes and max(sizes) <= logical._LOCAL_MAX_BYTES
