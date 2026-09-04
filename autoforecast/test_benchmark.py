"""Bounded hashing, manifest admission, compilation, and publication contracts."""

import copy
import datetime as dt
import errno
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
import yaml

from autoforecast import benchmark
from autoforecast.cli import create_example


def _reference_hash(path):
    dataset = ds.dataset(path, format="parquet", exclude_invalid_files=True)
    columns = sorted(dataset.schema.names)
    schema = [(name, str(dataset.schema.field(name).type)) for name in columns]
    hashed = hashlib.sha256(json.dumps(schema).encode())
    rows = [
        hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":"), default=str)
                       .encode()).digest()
        for row in dataset.to_table(columns=columns).to_pylist()
    ]
    for row in sorted(rows):
        hashed.update(row)
    return hashed.hexdigest()


@pytest.mark.parametrize("size", [0, 1, 3, 4, 29])
def test_external_sort_matches_reference_and_bounds_each_sort_and_merge(tmp_path, monkeypatch, size):
    path = tmp_path / "data.parquet"
    table = pa.table({
        "key": pa.array([f"key-{index % 5}" for index in range(size)], type=pa.string()),
        "value": pa.array([None if index % 3 == 0 else index / 2 for index in range(size)],
                          type=pa.float64()),
        "day": pa.array([dt.date(2024, 1, 1)] * size, type=pa.date32()),
    })
    pq.write_table(table, path)
    monkeypatch.setattr(benchmark, "_ROW_HASH_CHUNK_SIZE", 3)
    monkeypatch.setattr(benchmark, "_ROW_HASH_BATCH_SIZE", 2)
    monkeypatch.setattr(benchmark, "_ROW_HASH_MERGE_FAN_IN", 2)
    writes, merges, scans = [], [], []
    write, merge, dataset = benchmark._write_hash_run, benchmark._merge_hash_runs, ds.dataset

    def checked_write(output, rows):
        writes.append(len(rows))
        assert len(rows) <= 3 and all(len(row) == 32 for row in rows)
        return write(output, rows)

    def checked_merge(paths, output):
        merges.append(len(paths))
        assert len(paths) <= 2
        return merge(paths, output)

    class CheckedDataset:
        def __init__(self, value):
            self.value, self.schema = value, value.schema

        def scanner(self, **kwargs):
            scans.append(kwargs)
            assert kwargs["batch_size"] == 2 and kwargs["batch_readahead"] == 1
            assert kwargs["fragment_readahead"] == 1 and kwargs["use_threads"] is False
            return self.value.scanner(**kwargs)

    expected = _reference_hash(path)
    monkeypatch.setattr(benchmark, "_write_hash_run", checked_write)
    monkeypatch.setattr(benchmark, "_merge_hash_runs", checked_merge)
    monkeypatch.setattr(ds, "dataset", lambda *args, **kwargs: CheckedDataset(dataset(*args, **kwargs)))
    assert benchmark.parquet_content_hash(path) == expected
    assert sum(writes) == size and scans
    assert bool(merges) == (size > 3)


def test_external_sort_retains_duplicates_and_is_layout_independent(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "_ROW_HASH_CHUNK_SIZE", 2)
    monkeypatch.setattr(benchmark, "_ROW_HASH_MERGE_FAN_IN", 2)
    original, rearranged = tmp_path / "original", tmp_path / "rearranged"
    original.mkdir()
    rearranged.mkdir()
    rows = [{"value": value, "id": "series"} for value in (3, 3, 1, 2, 3)]
    pq.write_table(pa.Table.from_pylist(rows), original / "one.parquet")
    pq.write_table(pa.Table.from_pylist(rows[1:][::-1]), rearranged / "one.parquet")
    pq.write_table(pa.Table.from_pylist(rows[:1]), rearranged / "two.parquet")
    assert benchmark.parquet_content_hash(original) == benchmark.parquet_content_hash(rearranged)
    pq.write_table(pa.Table.from_pylist(rows[1:]), original / "one.parquet")
    assert benchmark.parquet_content_hash(original) != benchmark.parquet_content_hash(rearranged)


def test_external_sort_cleans_scratch_after_merge_failure(tmp_path, monkeypatch):
    path, scratch = tmp_path / "data.parquet", tmp_path / "scratch"
    scratch.mkdir()
    pq.write_table(pa.table({"value": range(7)}), path)
    temporary_directory = benchmark.tempfile.TemporaryDirectory
    monkeypatch.setattr(benchmark.tempfile, "TemporaryDirectory",
                        lambda **kwargs: temporary_directory(dir=scratch, **kwargs))
    monkeypatch.setattr(benchmark, "_ROW_HASH_CHUNK_SIZE", 2)
    monkeypatch.setattr(benchmark, "_ROW_HASH_MERGE_FAN_IN", 2)

    def fail_merge(*_args):
        raise OSError("injected sort write failure")

    monkeypatch.setattr(benchmark, "_merge_hash_runs", fail_merge)
    with pytest.raises(OSError, match="injected sort write failure"):
        benchmark.parquet_content_hash(path)
    assert list(scratch.iterdir()) == []


def _frozen_example(tmp_path):
    """Two synthetic series, 42 dates each; no exclusions; 14 cells per evaluation boundary."""
    project = tmp_path / "project"
    create_example(project)
    root = project / "frozen"
    root.mkdir()
    recipe = benchmark._normalise_recipe(json.loads((project / "recipe.json").read_text()))
    contract = yaml.safe_load((project / "dataset.yaml").read_text())
    contract["schema_version"] = 2
    contract["dataset"]["required_series_count"] = 2
    contract["dataset"]["panel"]["series_col"] = "__af_series_id"
    (root / "dataset.yaml").write_text(yaml.safe_dump(contract))
    rows = pq.read_table(project / "data").to_pylist()
    for row in rows:
        row[benchmark.MISSING_COLUMN] = False
        row["__af_series_id"] = json.dumps(
            {key: row[key] for key in recipe["source_keys"]}, separators=(",", ":"),
        )
    keys = {row["__af_series_id"]: {key: row[key] for key in ["__af_series_id", "shop", "sku"]}
            for row in rows}
    for key in keys.values():
        key[benchmark.START_COLUMN] = dt.date.fromisoformat(recipe["training_start"])
    cutoff = dt.date.fromisoformat(recipe["final_train_cutoff"])
    validation_end = dt.date.fromisoformat(recipe["validation_origins"][-1])
    tables = {
        "selection": [row for row in rows if row["day"] <= validation_end],
        "final_training": [row for row in rows if row["day"] <= cutoff],
        "validation_score": [row for row in rows if row["day"] <= cutoff],
        "final_score": [row for row in rows if row["day"] > cutoff],
        "source-keys": list(keys.values()),
    }
    for name, values in tables.items():
        (root / name).mkdir()
        pq.write_table(pa.Table.from_pylist(values), root / name / "part.parquet")
    grids, identities, coverage, coverage_identities = {}, {}, {}, {}
    for stage, origin in [("validation", day) for day in recipe["validation_origins"]] + [
        ("test", recipe["final_train_cutoff"]),
    ]:
        key = benchmark.boundary(stage, origin, 7)
        destination = root / "grids" / key
        destination.mkdir(parents=True)
        grid_rows = [{"series_id": series, "date": dt.date.fromisoformat(origin) + dt.timedelta(days=i)}
                     for series in keys for i in range(1, 8)]
        pq.write_table(pa.Table.from_pylist(grid_rows), destination / "part.parquet")
        logical = benchmark.parquet_content_hash(destination)
        grids[key] = {"path": f"grids/{key}", "sha256": benchmark.directory_digest(destination),
                      "logical_sha256": logical, "n_cells": 14}
        identities[key] = {"content_sha256": logical, "logical_sha256": logical, "n_cells": 14}
        coverage_path = root / "coverage" / key
        coverage_path.mkdir(parents=True)
        counts = {"expected": 7, "originally_missing": 0, "zero_filled": 0,
                  "excluded": 0, "scored": 7}
        pq.write_table(pa.Table.from_pylist([
            {"series_id": series, **counts} for series in keys
        ]), coverage_path / "part.parquet")
        coverage[key] = {"path": f"coverage/{key}",
                         "sha256": benchmark.directory_digest(coverage_path)}
        coverage_identities[key] = {
            "content_sha256": benchmark.parquet_content_hash(coverage_path),
            "counts": {name: 2 * count for name, count in counts.items()},
        }
    record = {
        "schema": benchmark.STUDY_SCHEMA, "recipe": recipe, "key_types": {"shop": "string", "sku": "bigint"},
        "contract_sha256": hashlib.sha256((project / "dataset.yaml").read_bytes()).hexdigest(),
        "logical_snapshots": {name: benchmark.parquet_content_hash(root / name) for name in tables},
        "source_series_count": 2, "excluded_new_series": 0, "cohort_series_count": 2,
        "dataset_sha256": hashlib.sha256((root / "dataset.yaml").read_bytes()).hexdigest(),
        "grid_identities": identities, "grids": grids,
        "counts": {name: {"required_observations": len(tables[name]),
                          "written_rows": len(tables[name]), "originally_missing": 0, "zero_filled": 0}
                   for name in ("selection", "final_training", "validation_score", "final_score")},
        "policies": {"history_missing": "preserve", "actual_missing": "error",
                     "missing_definition": "absent_row_or_null_target", "confirmed_zero": "observed",
                     "missing_indicator": benchmark.MISSING_COLUMN},
        "metric_definition": benchmark._metric_definition(recipe),
        "role_paths": {name: name for name in tables},
        "coverage": coverage, "coverage_identities": coverage_identities,
        "sources": {name: {"path": str(project / "data"), "sha256": "0" * 64}
                    for name in ("selection", "final_training", "validation_score", "final_score")},
        "artifacts": {name: benchmark.directory_digest(root / name) for name in tables},
    }
    record["benchmark_id"] = benchmark.digest(benchmark._identity(record))
    record["study_id"] = record["benchmark_id"]
    path = root / "benchmark.json"
    path.write_text(json.dumps(record))
    return path, record, project


def _malformed_records(record):
    for key in record:
        missing = copy.deepcopy(record)
        missing.pop(key)
        yield missing
    for key, value in (
        ("recipe", []), ("key_types", []), ("artifacts", []), ("sources", None), ("counts", []),
        ("logical_snapshots", []), ("grids", []), ("grid_identities", []),
        ("source_series_count", True), ("excluded_new_series", -1), ("cohort_series_count", 0),
        ("benchmark_id", "not-a-digest"), ("policies", []),
    ):
        malformed = copy.deepcopy(record)
        malformed[key] = value
        yield malformed
    key = next(iter(record["grids"]))
    for field in ("path", "sha256", "logical_sha256", "n_cells"):
        malformed = copy.deepcopy(record)
        del malformed["grids"][key][field]
        yield malformed
    for field, value in (("path", "../../elsewhere"), ("n_cells", True), ("sha256", [])):
        malformed = copy.deepcopy(record)
        malformed["grids"][key][field] = value
        yield malformed
    for field in ("validation_origins", "source_keys", "feature_availability"):
        malformed = copy.deepcopy(record)
        malformed["recipe"][field] = None
        yield malformed
    malformed = copy.deepcopy(record)
    malformed["sources"]["selection"] = []
    yield malformed
    malformed = copy.deepcopy(record)
    malformed["counts"]["selection"]["written_rows"] = True
    yield malformed
    malformed = copy.deepcopy(record)
    malformed["grid_identities"][key] = []
    yield malformed


def test_manifest_structure_is_validated_before_identity_or_artifact_reads(tmp_path, monkeypatch):
    path, record, _ = _frozen_example(tmp_path)
    assert benchmark.verify_benchmark(path)[1] == record
    monkeypatch.setattr(benchmark, "_identity", lambda *_: pytest.fail("identity saw malformed record"))
    monkeypatch.setattr(benchmark, "directory_digest", lambda *_a, **_k: pytest.fail("read artifacts"))
    malformed = list(_malformed_records(record)) + [[], None, {"schema": benchmark.STUDY_SCHEMA}]
    assert len(malformed) > 40
    for value in malformed:
        path.write_text(json.dumps(value))
        with pytest.raises(benchmark.BenchmarkError):
            benchmark.verify_benchmark(path)


def test_run_rejects_malformed_benchmark_without_traceback(tmp_path):
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps({"schema": benchmark.STUDY_SCHEMA}))
    project = tmp_path / "project"
    create_example(project)
    result = subprocess.run(
        [sys.executable, "-m", "autoforecast.cli", "run", "--project-root", str(project),
         "--request", str(project / "campaign.yaml"), "--benchmark", str(path),
         "--output-root", str(tmp_path / "output")], text=True, capture_output=True,
    )
    assert result.returncode == 2
    assert "benchmark" in result.stderr and "Traceback" not in result.stderr


def test_compile_verifies_the_benchmark_once_for_all_boundaries(tmp_path, monkeypatch):
    from autoforecast import forecast_campaign
    from autoforecast.project import experiment_project

    path, record, project = _frozen_example(tmp_path)
    request = yaml.safe_load((project / "campaign.yaml").read_text())
    request["forecast_campaign"]["dataset"]["contract"] = "frozen/dataset.yaml"
    (project / "campaign.yaml").write_text(yaml.safe_dump(request))
    calls = []
    verify = benchmark.verify_benchmark

    def counted_verify(*args, **kwargs):
        calls.append(args)
        return verify(*args, **kwargs)

    monkeypatch.setattr(benchmark, "verify_benchmark", counted_verify)
    # Wheel integrity is exercised by installed-release tests; this test owns compilation scans.
    monkeypatch.setattr(forecast_campaign, "_git_source_identity", lambda *_args: {})
    with experiment_project(project):
        result = forecast_campaign.compile_forecast_campaign(
            project / "campaign.yaml", selection_data_path=path.parent / "selection",
            final_training_data_path=path.parent / "final_training",
            final_score_data_path=path.parent / "final_score", benchmark_path=path,
            compiled_root=project / "compiled",
        )
    resolved = json.loads(result.resolved_path.read_text())
    assert len(calls) == 1
    assert set(resolved["benchmark"]["evaluations"]) == set(record["grids"])
    assert resolved["benchmark"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("error_number", [errno.EEXIST, errno.ENOTEMPTY])
def test_concurrent_identical_publication_authenticates_winner_and_cleans_loser(
    tmp_path, monkeypatch, error_number,
):
    path, record, _ = _frozen_example(tmp_path)
    scratch = tmp_path / "scratch"
    shutil.copytree(path.parent, scratch)
    destination = tmp_path / "published"
    original_rename = pathlib.Path.rename
    verify = benchmark.verify_benchmark
    checked = []

    def race(source, target):
        if source == scratch:
            shutil.copytree(source, target)
            raise OSError(error_number, "another publisher won")
        return original_rename(source, target)

    def checked_verify(*args, **kwargs):
        checked.append(args[0])
        return verify(*args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "rename", race)
    monkeypatch.setattr(benchmark, "verify_benchmark", checked_verify)
    benchmark._publish_frozen_benchmark(scratch, destination, record["benchmark_id"])
    assert checked == [destination / "benchmark.json"]
    assert not scratch.exists()
    assert verify(destination / "benchmark.json")[1]["benchmark_id"] == record["benchmark_id"]


@pytest.mark.parametrize("defect", ["different_identity", "changed_artifact", "write_error"])
def test_publication_refuses_a_different_or_corrupted_winner(tmp_path, monkeypatch, defect):
    path, record, _ = _frozen_example(tmp_path)
    scratch, destination = tmp_path / "scratch", tmp_path / "published"
    shutil.copytree(path.parent, scratch)
    shutil.copytree(path.parent, destination)
    if defect == "different_identity":
        record["recipe"]["target_unit"] = "other units"
        record["benchmark_id"] = benchmark.digest(benchmark._identity(record))
        record["study_id"] = record["benchmark_id"]
        (destination / "benchmark.json").write_text(json.dumps(record))
        expected_error, match = benchmark.BenchmarkError, "different benchmark identity"
    elif defect == "changed_artifact":
        (destination / "selection/part.parquet").write_bytes(b"changed")
        expected_error, match = benchmark.BenchmarkError, "artifact changed"
    else:
        def denied(*_args):
            raise OSError(errno.EACCES, "denied publication")
        monkeypatch.setattr(pathlib.Path, "rename", denied)
        expected_error, match = OSError, "denied publication"
    original = json.loads(path.read_text())["benchmark_id"]
    with pytest.raises(expected_error, match=match):
        benchmark._publish_frozen_benchmark(scratch, destination, original)
    assert not scratch.exists()


@pytest.mark.parametrize("name", ["store.id", "tick`name", "méter"])
@pytest.mark.parametrize("role", ["source_key", "feature", "known_at"])
def test_recipe_rejects_names_with_expression_meaning_before_runtime(tmp_path, name, role):
    project = tmp_path / "project"
    create_example(project)
    recipe = json.loads((project / "recipe.json").read_text())
    if role == "source_key":
        recipe["source_keys"] = [name]
    elif role == "feature":
        recipe["feature_availability"] = {name: "known_at"}
    else:
        recipe["feature_availability"] = {"promotion": name}
    with pytest.raises(benchmark.BenchmarkError, match="ASCII identifiers"):
        benchmark.validate_recipe(recipe)


@pytest.mark.parametrize(
    "frequency,start,origin,cutoff",
    [
        ("H", "2024-03-01T22:00:00", "2024-03-02T00:00:00", "2024-03-02T02:00:00"),
        ("W", "2024-01-03", "2024-01-10", "2024-01-24"),
        ("M", "2024-01-31", "2024-02-29", "2024-04-30"),
        ("Y", "2023-12-31", "2024-12-31", "2026-12-31"),
    ],
)
def test_recipe_windows_count_cadence_periods(frequency, start, origin, cutoff):
    recipe = {
        "schema_version": 2,
        "metric": "rmse",
        "source_keys": ["series"],
        "training_start": start,
        "validation_origins": [origin],
        "validation_horizon": 1,
        "final_train_cutoff": cutoff,
        "final_test_horizon": 1,
        "history_policy": "fixed",
        "new_series_policy": "error",
        "target_unit": "units",
        "feature_availability": {},
        "frequency": frequency,
    }
    benchmark.validate_recipe(recipe)
    assert benchmark.boundary("validation", origin, 1, frequency).endswith(f"/1/{frequency}")


@pytest.mark.parametrize(
    "frequency,start,origin,cutoff",
    [
        ("H", "2024-03-01T22:00:00", "2024-03-01T22:30:00", "2024-03-02T00:00:00"),
        ("W", "2024-01-03", "2024-01-11", "2024-01-24"),
        ("M", "2024-01-31", "2024-02-29", "2024-03-30"),
        ("Y", "2023-12-31", "2024-02-29", "2026-12-31"),
    ],
)
def test_recipe_refuses_off_grid_hour_week_month_or_year(frequency, start, origin, cutoff):
    recipe = {
        "schema_version": 2,
        "metric": "rmse",
        "source_keys": ["series"],
        "training_start": start,
        "validation_origins": [origin],
        "validation_horizon": 1,
        "final_train_cutoff": cutoff,
        "final_test_horizon": 1,
        "history_policy": "fixed",
        "new_series_policy": "error",
        "target_unit": "units",
        "feature_availability": {},
        "frequency": frequency,
    }
    with pytest.raises(benchmark.BenchmarkError):
        benchmark.validate_recipe(recipe)


def test_large_hourly_recipe_validation_does_not_enumerate_history(monkeypatch):
    recipe = {
        "schema_version": 2,
        "metric": "rmse",
        "source_keys": ["series"],
        "training_start": "0001-01-01T00:00:00",
        "validation_origins": ["9998-01-01T00:00:00"],
        "validation_horizon": 1,
        "final_train_cutoff": "9998-01-01T01:00:00",
        "final_test_horizon": 1,
        "history_policy": "fixed",
        "new_series_policy": "error",
        "target_unit": "units",
        "feature_availability": {},
        "frequency": "H",
    }
    calls = 0
    real_advance = benchmark.advance

    def bounded_advance(*args):
        nonlocal calls
        calls += 1
        assert calls <= 8, "recipe validation enumerated the historical grid"
        return real_advance(*args)

    monkeypatch.setattr(benchmark, "advance", bounded_advance)
    benchmark.validate_recipe(recipe)
    assert calls > 0


@pytest.mark.parametrize(
    "frequency,start,end,count",
    [
        ("H", "2024-02-29T23:00:00", "2024-03-01T01:00:00", 3),
        ("D", "2024-02-28", "2024-03-01", 3),
        ("W", "2024-02-23", "2024-03-08", 3),
        ("3D", "2024-02-26", "2024-03-03", 3),
        ("M", "2024-02-29", "2024-04-30", 3),
        ("MS", "2023-12-01", "2024-02-01", 3),
        ("Y", "2099-12-31", "2101-12-31", 3),
        ("YS", "2099-01-01", "2101-01-01", 3),
        ("M", "2024-02-29", "2024-02-29", 1),
    ],
)
def test_inclusive_period_count_preserves_calendar_and_stride(frequency, start, end, count):
    cadence = benchmark.parse_frequency(frequency)
    assert benchmark._inclusive_period_count(
        benchmark.parse_origin(start, cadence), benchmark.parse_origin(end, cadence),
        cadence, label="test window",
    ) == count


@pytest.mark.parametrize("frequency,start,end", [
    ("W", "2024-01-03", "2024-01-11"),
    ("3D", "2024-01-01", "2024-01-08"),
    ("D", "2024-01-03", "2024-01-01"),
])
def test_inclusive_period_count_refuses_reversed_or_shifted_endpoints(frequency, start, end):
    cadence = benchmark.parse_frequency(frequency)
    with pytest.raises(benchmark.BenchmarkError):
        benchmark._inclusive_period_count(
            benchmark.parse_origin(start, cadence), benchmark.parse_origin(end, cadence),
            cadence, label="test window",
        )


def test_typed_row_hash_canonicalizes_nested_maps_without_reordering_arrays(tmp_path):
    from decimal import Decimal

    from provenance.lineage import directory_digest

    schema = pa.schema([
        ("id", pa.string()), ("amount", pa.decimal128(12, 2)),
        ("at", pa.timestamp("us")),
        ("nested", pa.struct([("map", pa.map_(pa.string(), pa.list_(pa.int32())))])),
    ])
    first = {"id": "雪,\n", "amount": Decimal("12.30"),
             "at": dt.datetime(2024, 1, 1, 12, 3, 4, 123456),
             "nested": {"map": [("z", [2, 1]), ("a", [None, 3])]}}
    null = {"id": "B", "amount": None, "at": None, "nested": None}
    left, right = tmp_path / "one.parquet", tmp_path / "two.parquet"
    pq.write_table(pa.Table.from_pylist([first, null], schema=schema), left)
    reverse = copy.deepcopy(first)
    reverse["nested"]["map"].reverse()
    pq.write_table(pa.Table.from_pylist([null, reverse], schema=schema), right, compression="gzip")
    expected = benchmark.parquet_content_hash(left)
    assert benchmark.parquet_content_hash(right) == expected
    assert directory_digest(left) != directory_digest(right)
    reverse["nested"]["map"][1][1].reverse()
    pq.write_table(pa.Table.from_pylist([null, reverse], schema=schema), right)
    assert benchmark.parquet_content_hash(right) != expected
    pq.write_table(pa.Table.from_pylist([first, null, null], schema=schema), right)
    assert benchmark.parquet_content_hash(right) != expected
    pq.write_table(pa.Table.from_pylist([], schema=schema), right)
    empty = benchmark.parquet_content_hash(right)
    pq.write_table(pa.Table.from_pylist([], schema=schema.remove(1)), right)
    assert benchmark.parquet_content_hash(right) != empty


def test_logical_hash_rejects_duplicate_map_keys_and_corrupt_members(tmp_path):
    path = tmp_path / "rows"
    path.mkdir()
    schema = pa.schema([("m", pa.map_(pa.string(), pa.int32()))])
    pq.write_table(pa.Table.from_pylist([{"m": [("a", 1), ("a", 2)]}], schema=schema),
                   path / "part.parquet")
    with pytest.raises(benchmark.BenchmarkError, match="unique map keys"):
        benchmark.parquet_content_hash(path)
    pq.write_table(pa.Table.from_pylist([{"m": []}], schema=schema), path / "part.parquet")
    (path / "corrupt.parquet").write_bytes(b"not parquet")
    with pytest.raises(pa.ArrowInvalid):
        benchmark.parquet_content_hash(path)
