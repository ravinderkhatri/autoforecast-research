"""Result boundaries preserve missing data and identity instead of inventing forecast rows."""

import datetime as dt
import hashlib
import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pyarrow as pa
import pytest
import yaml

from autoforecast.campaign import CampaignStore, CampaignSupervisor
from autoforecast.results import (
    ROW_SCHEMA,
    CampaignResults,
    comparison_plot,
    export_results,
    read_results,
)


def _reader_boundary(tmp_path, monkeypatch, children, stages, benchmark=None):
    """Isolate journal lookup while retaining reader byte and forecast authentication."""
    import autoforecast.results as reader

    resolved = tmp_path / "resolved.json"
    resolved.write_text(
        json.dumps(
            {
                "schema": "autoforecast.forecast-campaign-resolved.v4",
                "source": {},
                "selection": {"metric": "rmse"},
                "children": children,
                "benchmark": benchmark,
            }
        )
    )
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    store = SimpleNamespace(
        root=tmp_path,
        plan=SimpleNamespace(
            stages=[
                SimpleNamespace(
                    command=[
                        "--resolved",
                        str(resolved),
                        "--resolved-sha256",
                        digest,
                    ]
                )
            ]
        ),
        state=lambda: {"stages": stages},
        verify_completed_outputs=lambda: None,
    )
    monkeypatch.setattr(reader.CampaignStore, "from_root", lambda _: store)
    monkeypatch.setattr(reader, "_completed_output", lambda root, stage, name: root / stage / name)
    return digest


def test_initialization_parent_is_not_validation_evidence(tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    from autoforecast.panel import Forecast
    from provenance.lineage import directory_digest

    child = {
        "id": "parent",
        "kind": "initialization_parent",
        "model_id": "example",
        "variant_id": "default",
        "seed": 0,
        "initialization": "fresh_parent",
        "origin": "2024-01-01",
        "horizon": 1,
    }
    _reader_boundary(
        tmp_path,
        monkeypatch,
        [child],
        [
            {"id": "parent", "status": "completed", "attempts": 1},
        ],
    )
    parent = tmp_path / "parent"
    forecast_dir = parent / "forecast"
    forecast_dir.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"series_id": "a", "date": dt.date(2024, 1, 2), "yhat": 2.0},
            ]
        ),
        forecast_dir / "part.parquet",
    )
    forecast = Forecast("seasonal_naive", str(forecast_dir), child["origin"], 1, 1, {"emitted": 1})
    record = parent / "forecast.json"
    record.write_text(json.dumps(forecast.as_record()))
    (parent / "child-result.json").write_text(
        json.dumps(
            {
                **child,
                "adapter": "seasonal_naive",
                "run_path": str(parent / "run"),
                "forecast_record_path": str(record),
                "forecast_record_sha256": hashlib.sha256(record.read_bytes()).hexdigest(),
                "forecast_artifact_sha256": directory_digest(forecast_dir, suffixes=(".parquet",)),
                "score": None,
            }
        )
    )
    results = read_results(tmp_path)
    assert results.rows["stage"].to_pylist() == ["initialization_parent"]
    assert results.runs[0]["stage"] == "initialization_parent"


def test_failed_final_fit_retains_locked_identity(tmp_path, monkeypatch):
    digest = _reader_boundary(
        tmp_path,
        monkeypatch,
        [],
        [
            {"id": "aggregate_selection", "status": "completed", "attempts": 1},
            {"id": "final_fit", "status": "failed", "attempts": 2},
        ],
    )
    chosen = {
        "model_id": "example",
        "variant_id": "tuned",
        "final_seed": 9,
        "initialization": "fresh",
        "parameter_resolution": {"effective_parameters": {}},
        "parameter_resolution_sha256": "a" * 64,
    }
    path = tmp_path / "aggregate_selection/selection-lock.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"resolved_sha256": digest, "chosen": chosen}))
    result = read_results(tmp_path)
    assert result.rows.num_rows == 0
    (run,) = result.runs
    assert run["stage"] == "test" and run["attempt"] == 2 and run["status"] == "failed"
    assert run["seed"] == 9
    assert all(run[key] == value for key, value in chosen.items() if key != "final_seed")
    assert run["selection_lock_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    original_bytes = path.read_bytes()
    original_read = type(path).read_bytes
    reads = 0

    def changing_lock(file):
        nonlocal reads
        if file == path:
            reads += 1
            if reads > 1:
                return json.dumps(
                    {"resolved_sha256": digest, "chosen": {**chosen, "model_id": "replacement"}}
                ).encode()
            return original_bytes
        return original_read(file)

    with monkeypatch.context() as patch:
        patch.setattr(type(path), "read_bytes", changing_lock)
        (retained,) = read_results(tmp_path).runs
    assert retained["model_id"] == chosen["model_id"]
    assert retained["selection_lock_sha256"] == hashlib.sha256(original_bytes).hexdigest()
    path.write_text(json.dumps({"resolved_sha256": "b" * 64, "chosen": chosen}))
    with pytest.raises(ValueError, match="selection lock differs"):
        read_results(tmp_path)


def test_reader_parses_the_authenticated_resolved_bytes(tmp_path, monkeypatch):
    _reader_boundary(tmp_path, monkeypatch, [], [])
    path = tmp_path / "resolved.json"
    original_bytes = path.read_bytes()
    replacement = json.loads(original_bytes)
    replacement["source"] = {"unauthenticated": True}
    original_read = type(path).read_bytes
    reads = 0

    def changing_resolved(file):
        nonlocal reads
        if file == path:
            reads += 1
            return original_bytes if reads == 1 else json.dumps(replacement).encode()
        return original_read(file)

    monkeypatch.setattr(type(path), "read_bytes", changing_resolved)
    assert read_results(tmp_path).metadata["source"] == {}


def test_reader_retains_selection_actual_snapshot_for_revised_history(tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    import autoforecast.results as reader

    benchmark_path = tmp_path / "benchmark.json"
    benchmark = {
        "id": "benchmark",
        "path": str(benchmark_path),
        "sha256": "a" * 64,
        "evaluations": {},
    }
    _reader_boundary(tmp_path, monkeypatch, [], [], benchmark)
    frozen = {
        "benchmark_id": "benchmark",
        "recipe": {"target_unit": "units", "history_policy": "revised_snapshot"},
        "key_types": {},
        "cohort_series_count": 1,
        "artifacts": {"selection": "1" * 64, "final_training": "2" * 64, "final_score": "3" * 64},
    }
    (tmp_path / "source-keys").mkdir()
    pq.write_table(
        pa.Table.from_pylist([{"series_id": "a"}]), tmp_path / "source-keys/part.parquet"
    )
    monkeypatch.setattr(reader, "verify_benchmark", lambda *args: (benchmark_path, frozen))
    (tmp_path / "dataset.yaml").write_text(
        "schema_version: 1\ndataset:\n  id: meters\n  panel:\n"
        "    series_col: meter\n    time_col: date\n    target_col: kwh\n"
        "    target_estimand: observed_nonnegative\n"
    )
    result = read_results(tmp_path)
    assert result.metadata["target_estimand"] == "observed_nonnegative"
    assert result.metadata["actual_snapshots"] == {
        role: {"path": str(tmp_path / role), "sha256": digest}
        for role, digest in frozen["artifacts"].items()
    }


def test_failed_run_without_fake_forecast(tmp_path):
    resolved = tmp_path / "resolved.json"
    resolved.write_text(
        json.dumps(
            {
                "schema": "autoforecast.forecast-campaign-resolved.v4",
                "source": {},
                "selection": {"metric": "rmse"},
                "children": [
                    {
                        "id": "v0",
                        "model_id": "example",
                        "variant_id": "default",
                        "seed": 0,
                        "initialization": "fresh",
                    }
                ],
            }
        )
    )
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "campaign": {
                    "name": "failed-reader",
                    "stages": [
                        {
                            "id": "v0",
                            "command": [
                                sys.executable,
                                "-c",
                                "raise SystemExit(1)",
                                "--resolved",
                                str(resolved),
                                "--resolved-sha256",
                                hashlib.sha256(resolved.read_bytes()).hexdigest(),
                            ],
                            "needs": [],
                            "timeout_seconds": 10,
                            "max_attempts": 1,
                            "retry_mode": "restart",
                            "required_outputs": ["child-result.json"],
                            "resources": {
                                "device": "cpu",
                                "memory_gib": 0.5,
                                "min_free_disk_gib": 0.01,
                            },
                        }
                    ],
                },
            }
        )
    )
    store = CampaignStore.open_or_create(
        plan, output_root=tmp_path / "runs", working_directory=tmp_path
    )
    CampaignSupervisor(store, poll_seconds=0.02).run()
    result = read_results(store.root)
    assert result.rows.num_rows == 0
    assert len(result.runs) == 1 and result.runs[0]["status"] == "failed"


def _results():
    rows = [
        {
            "benchmark_id": "benchmark",
            "model_id": "same-model",
            "variant_id": variant,
            "run_id": run,
            "stage": stage,
            "origin": "2024-01-01",
            "target_date": dt.date(2024, 1, 2),
            "series_id": "a",
            "prediction": value,
            "status": "emitted",
        }
        for variant, run, stage, value in (
            ("short", "one", "validation", 2.0),
            ("long", "two", "test", 9.0),
        )
    ]
    return CampaignResults(
        pa.Table.from_pylist(rows, schema=ROW_SCHEMA),
        ({"run_id": "one", "horizon": 3}, {"run_id": "two", "horizon": 3}),
        {"comparison_scope": "fixed_benchmark", "target_unit": "units"},
        None,
    )


def test_result_export_identity_roundtrip_and_publication_retry_idempotent(tmp_path):
    import pyarrow.parquet as pq

    result = _results()
    export_results(result, tmp_path / "export")
    export_results(result, tmp_path / "export")
    assert pq.read_table(tmp_path / "export/predictions.parquet").equals(result.rows)
    with pytest.raises(OSError, match="immutable artifact differs"):
        export_results(replace(result, metadata={"changed": True}), tmp_path / "export")


def test_plot_matches_exported_values_and_preserves_missing_predictions(tmp_path, monkeypatch):
    import math

    import matplotlib.axes

    observed = []
    original = matplotlib.axes.Axes.plot

    def capture(self, dates, values, **kwargs):
        observed.append((list(dates), list(values), kwargs["label"]))
        return original(self, dates, values, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", capture)
    comparison_plot(_results(), series_id="a", output=tmp_path / "comparison.png")
    assert sorted(values[0] for _, values, _ in observed) == [2.0, 9.0]
    assert all(
        len(values) == 3 and all(math.isnan(v) for v in values[1:]) for _, values, _ in observed
    )
    assert len({label for _, _, label in observed}) == 2
    assert (tmp_path / "comparison.png").stat().st_size > 0
