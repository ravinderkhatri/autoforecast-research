"""Regression coverage for study identity and stable input consumption."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import pathlib
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from autoforecast import benchmark, grid
from autoforecast.data_admission import load_dataset_contract
from autoforecast.panel import Forecast
from autoforecast.test_study import freeze
from autoforecast.test_study import spark as _spark_fixture
from provenance.lineage import directory_digest

spark = _spark_fixture


def _validation_forecast(
    study_path: pathlib.Path, output: pathlib.Path, *, yhat: float = 2.0, metric: str = "rmse"
) -> tuple[dict, Forecast]:
    fixed = benchmark.evaluation_contract(
        study_path,
        stage="validation",
        origin="2024-01-05",
        horizon=2,
        metric=metric,
    )
    rows = pq.read_table(fixed["grid"]["path"]).to_pylist()
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([{**row, "yhat": yhat} for row in rows]), output)
    forecast = Forecast(
        "test",
        str(output),
        "2024-01-05",
        2,
        2,
        {"emitted": len(rows)},
        target_estimand="observed_nonnegative",
    )
    return fixed, forecast


def _part(path: pathlib.Path) -> pathlib.Path:
    return path if path.is_file() else next(path.glob("*.parquet"))


def test_resolve_study_rejects_obsolete_schema_before_artifact_reads(tmp_path, monkeypatch):
    """An obsolete format fails at admission, before identity or artifact reads."""
    registry = tmp_path / "registry"
    identity = "a" * 64
    target = registry / identity
    target.mkdir(parents=True)
    (target / "benchmark.json").write_text(json.dumps({
        "schema": "autoforecast.frozen-benchmark.v1",
    }))
    monkeypatch.setattr(benchmark, "_identity", lambda *_: pytest.fail("read obsolete identity"))
    monkeypatch.setattr(benchmark, "directory_digest", lambda *_a, **_k: pytest.fail("read artifacts"))
    with pytest.raises(benchmark.BenchmarkError, match="unsupported benchmark schema"):
        benchmark.resolve_study(registry, identity)


@pytest.mark.parametrize("metric,policy", [("wrmsse", "zero"), ("rmse", "zero"), ("rmse", "mask")])
def test_study_compiles_for_shared_scorer_and_enforces_declared_adapter_support(
    tmp_path, spark, monkeypatch, metric, policy,
):
    """A recipe-v2 single-level WRMSSE study is not an M5 hierarchy campaign."""
    from autoforecast import forecast_campaign as campaign
    from autoforecast.cli import create_example
    from autoforecast.project import experiment_project

    project = tmp_path / "project"
    create_example(project)
    recipe = json.loads((project / "recipe.json").read_text())
    recipe.update(metric=metric, actual_missing=policy)
    references = project / "scoring-references.parquet"
    rows = [
        {
            "keys": {"shop": shop, "sku": sku},
            "stage": stage,
            "origin": origin,
            "weight": weight,
            "scale": scale,
        }
        for stage, origin in [
            *(('validation', item) for item in recipe["validation_origins"]),
            ("test", recipe["final_train_cutoff"]),
        ]
        for shop, sku, weight, scale in (("a|b", 1, 14.0, 2.5), ("a", 2, 9.0, 4.0))
    ]
    pq.write_table(pa.Table.from_pylist(rows), references)
    sources = dict.fromkeys(
        ("selection", "final_training", "validation_score", "final_score"),
        project / "data",
    )
    study_path = benchmark.freeze_benchmark(
        contract_path=project / "dataset.yaml",
        sources=sources,
        recipe=recipe,
        output_root=project / "frozen",
        spark=spark,
        scoring_references=references if metric == "wrmsse" else None,
    )
    _, record = benchmark.verify_benchmark(study_path)
    request_path = project / "campaign.yaml"
    request = yaml.safe_load(request_path.read_text())
    request["forecast_campaign"]["dataset"]["contract"] = str(
        (study_path.parent / "dataset.yaml").relative_to(project)
    )
    request["forecast_campaign"]["selection"].update(
        metric=metric, missing_actual="sparse_zero" if policy == "zero" else "no_data",
    )
    request_path.write_text(yaml.safe_dump(request))
    monkeypatch.setattr(campaign, "_git_source_identity", lambda *_args: {})

    with experiment_project(project):
        compiled = campaign.compile_forecast_campaign(
            request_path,
            benchmark_path=study_path,
            compiled_root=project / "compiled",
            **{
                f"{role}_data_path": benchmark.artifact_path(study_path, record, role)
                for role in ("selection", "final_training", "final_score")
            },
        )
    resolved = json.loads(compiled.resolved_path.read_text())
    assert resolved["study_id"] == record["study_id"]
    assert resolved["selection"]["metric"] == metric

    original_discover = campaign.discover_candidates
    original_descriptor = campaign.descriptor
    with experiment_project(project):
        for adapter in ("sft", "hmfnet", "pact", "tft_q", "tft_hnb", "timesfm_hnb", "timesfm_v8"):
            assert original_descriptor(adapter).supports_shared_studies
            monkeypatch.setattr(
                campaign, "descriptor", lambda name: dataclasses.replace(
                    original_descriptor(name), supports_shared_studies=False,
                ),
            )
            monkeypatch.setattr(
                campaign, "discover_candidates",
                lambda root, adapter=adapter: [
                    dataclasses.replace(candidate, adapter=adapter)
                    for candidate in original_discover(root)
                ],
            )
            with pytest.raises(campaign.ForecastCampaignError, match="does not support shared studies"):
                campaign.compile_forecast_campaign(
                    request_path, benchmark_path=study_path,
                    compiled_root=project / ("refused-" + adapter),
                    **{f"{role}_data_path": benchmark.artifact_path(study_path, record, role)
                       for role in ("selection", "final_training", "final_score")},
                )



def test_freeze_uses_the_source_bytes_whose_hash_it_records(tmp_path, spark, monkeypatch):
    """Prepared targets must come from the exact source bytes named in provenance."""
    from autoforecast.test_study import inputs

    contract, sources, recipe = inputs(tmp_path)
    source = pathlib.Path(sources["selection"])
    original_bytes = source.read_bytes()
    original_table = pq.read_table(source)
    rows = original_table.to_pylist()
    for row in rows:
        if row["y"] is not None:
            row["y"] = float(row["y"]) + 100.0
    changed = pa.Table.from_pylist(rows, schema=original_table.schema)
    real_digest = benchmark.directory_digest
    source_calls = 0

    def racing_digest(path, *args, **kwargs):
        nonlocal source_calls
        current = pathlib.Path(path)
        if current.resolve() != source.resolve():
            return real_digest(path, *args, **kwargs)
        source_calls += 1
        if 5 <= source_calls <= 8:
            source.write_bytes(original_bytes)
            result = real_digest(path, *args, **kwargs)
            pq.write_table(changed, source)
            return result
        if source_calls == 9:
            source.write_bytes(original_bytes)
            return real_digest(path, *args, **kwargs)
        return real_digest(path, *args, **kwargs)

    real_read = grid.read_time_parquet
    injected = 0

    def racing_read(*args, **kwargs):
        nonlocal injected
        if not injected:
            pq.write_table(changed, source)
            injected += 1
        return real_read(*args, **kwargs)

    monkeypatch.setattr(grid, "read_time_parquet", racing_read)
    monkeypatch.setattr(benchmark, "directory_digest", racing_digest)
    try:
        study_path = benchmark.freeze_benchmark(
            contract_path=contract,
            sources=sources,
            recipe=recipe,
            output_root=tmp_path / "frozen",
            spark=spark,
        )
    except benchmark.BenchmarkError as error:
        assert "changed" in str(error)
        assert source_calls >= 1
        assert not list((tmp_path / "frozen").rglob("benchmark.json"))
        return
    finally:
        source.write_bytes(original_bytes)
    assert injected == 1
    assert source_calls >= 1
    _, record = benchmark.verify_benchmark(study_path)
    selection = pq.read_table(benchmark.artifact_path(study_path, record, "selection")).to_pylist()
    observed = next(row["y"] for row in selection if row["id"] == "A" and row["date"].day == 1)
    assert record["sources"]["selection"]["sha256"] == real_digest(
        source, suffixes=(".parquet",)
    )
    assert observed == 1.0, (
        "frozen target came from transient source bytes while provenance names restored bytes"
    )


def test_freeze_uses_the_contract_bytes_whose_hash_it_records(tmp_path, spark, monkeypatch):
    """Prepared role semantics must come from the exact contract bytes named in provenance."""
    from autoforecast.test_study import inputs

    contract, sources, recipe = inputs(tmp_path)
    source = pathlib.Path(sources["selection"])
    table = pq.read_table(source)
    rows = table.to_pylist()
    for row in rows:
        row["y2"] = None if row["y"] is None else float(row["y"]) + 100.0
    schema = pa.schema([*table.schema, pa.field("y2", pa.float64())])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), source)
    original_contract = pathlib.Path(contract).read_bytes()
    changed_contract = yaml.safe_load(original_contract)
    changed_contract["dataset"]["panel"]["target_col"] = "y2"
    changed_bytes = yaml.safe_dump(changed_contract).encode()
    real_read = pathlib.Path.read_bytes
    reads = 0

    def racing_read(path):
        nonlocal reads
        if path.resolve() != pathlib.Path(contract).resolve():
            return real_read(path)
        reads += 1
        payload = real_read(path)
        if reads == 1:
            path.write_bytes(changed_bytes)
        elif reads == 2:
            path.write_bytes(original_contract)
        return payload

    monkeypatch.setattr(pathlib.Path, "read_bytes", racing_read)
    try:
        study_path = benchmark.freeze_benchmark(
            contract_path=contract,
            sources=sources,
            recipe=recipe,
            output_root=tmp_path / "frozen",
            spark=spark,
        )
    except benchmark.BenchmarkError as error:
        assert "changed" in str(error)
        assert reads >= 1
        assert not list((tmp_path / "frozen").rglob("benchmark.json"))
        return
    finally:
        pathlib.Path(contract).write_bytes(original_contract)
    assert reads >= 1
    _, record = benchmark.verify_benchmark(study_path)
    prepared = load_dataset_contract(study_path.parent / "dataset.yaml")
    assert record["contract_sha256"] == hashlib.sha256(original_contract).hexdigest()
    assert prepared.panel_roles["target_col"] == "y", (
        "prepared target role came from transient contract bytes while provenance names the "
        "restored contract"
    )


def test_training_materialization_uses_pinned_source_bytes(tmp_path, monkeypatch):
    """The training scan consumes its authenticated copy during a live-source rewrite."""
    from autoforecast import forecast_campaign as campaign
    from autoforecast.cli import create_example

    project = tmp_path / "project"
    create_example(project)
    source = project / "data"
    part = next(source.glob("*.parquet"))
    original = part.read_bytes()
    table = pq.read_table(part)
    rows = table.to_pylist()
    for row in rows:
        row["units"] = float(row["units"]) + 100.
    changed = pa.Table.from_pylist(rows, schema=table.schema)
    expected = directory_digest(source, suffixes=(".parquet",))
    writer = campaign._write_training_panel
    real_digest = campaign.directory_digest
    scanning = False
    calls = 0

    def racing_digest(path, *args, **kwargs):
        if scanning and pathlib.Path(path).resolve() == source.resolve():
            # The producer restores the original for each check, then resumes its rewrite.
            part.write_bytes(original)
            value = real_digest(path, *args, **kwargs)
            pq.write_table(changed, part)
            return value
        return real_digest(path, *args, **kwargs)

    def concurrent_rewrite(*args, **kwargs):
        nonlocal calls, scanning
        calls += 1
        scanning = True
        pq.write_table(changed, part)
        try:
            return writer(*args, **kwargs)
        finally:
            scanning = False
            part.write_bytes(original)

    monkeypatch.setattr(campaign, "_write_training_panel", concurrent_rewrite)
    monkeypatch.setattr(campaign, "directory_digest", racing_digest)
    destination = tmp_path / "training"
    campaign._materialize_training_panel(
        source, destination, contract_path=project / "dataset.yaml",
        origin="2024-01-21", horizon=7, training_start="2024-01-01",
        expected_source_sha256=expected,
    )
    assert calls == 1
    output = pq.read_table(destination).to_pylist()
    observed = next(row["units"] for row in output
                    if row["shop"] == "a|b" and row["sku"] == 1 and row["day"].day == 1)
    assert observed == 3.
    assert part.read_bytes() == original


def test_score_study_uses_pinned_actual_bytes(tmp_path, spark, monkeypatch):
    """A transient actuals rewrite must not produce a score under the original digest."""
    from autoforecast import study_scoring
    from autoforecast.run_evaluation import compare_run_forecasts

    study_path = freeze(tmp_path / "study", spark)
    fixed, forecast = _validation_forecast(study_path, tmp_path / "predictions.parquet")
    panel = load_dataset_contract(study_path.parent / "dataset.yaml").panel(
        fixed["actuals"]["path"]
    )
    actual_part = _part(pathlib.Path(panel.path))
    original_bytes = actual_part.read_bytes()
    checksum = actual_part.with_name(f".{actual_part.name}.crc")
    checksum_bytes = checksum.read_bytes() if checksum.exists() else None
    original_table = pq.read_table(actual_part)
    rows = original_table.to_pylist()
    for row in rows:
        if not row[benchmark.MISSING_COLUMN]:
            row["y"] = float(row["y"]) + 100.0
    changed = pa.Table.from_pylist(rows, schema=original_table.schema)
    real_score = study_scoring.score_forecast

    changed_during_call = False

    def transient_rewrite(*args, **kwargs):
        nonlocal changed_during_call
        changed_during_call = True
        pq.write_table(changed, actual_part)
        # Hadoop's local checksum sidecar is mutable state too.  A legitimate
        # concurrent Spark/Hadoop rewrite updates it; removing it models the
        # same readable-file state without invoking a second Spark writer.
        if checksum.exists():
            checksum.unlink()
        try:
            return real_score(*args, **kwargs)
        finally:
            actual_part.write_bytes(original_bytes)
            if checksum_bytes is not None:
                checksum.write_bytes(checksum_bytes)

    monkeypatch.setattr(study_scoring, "score_forecast", transient_rewrite)
    comparison = compare_run_forecasts(
        panel,
        {"test": forecast},
        spark,
        missing_actual="no_data",
        metric="rmse",
        output_dir=tmp_path / "comparison",
        fixed_contract=fixed,
    )
    candidate = json.loads(comparison.read_text())["candidates"][0]
    assert changed_during_call
    assert candidate["status"] == "scored"
    assert candidate["score"]["rmse"] == pytest.approx(3.5), (
        "score was computed from transient bytes even though final provenance re-authenticated "
        "the original digest"
    )


def test_fixed_study_rejects_extra_declined_series_in_forecast_manifest(tmp_path, spark):
    """A full two-series grid cannot be accepted as a three-series forecast with declines."""
    from autoforecast.run_evaluation import compare_run_forecasts

    study_path = freeze(tmp_path / "study", spark)
    fixed, ordinary = _validation_forecast(study_path, tmp_path / "predictions.parquet")
    malformed = Forecast(
        ordinary.adapter,
        ordinary.path,
        ordinary.origin,
        ordinary.horizon,
        3,
        {"emitted": 4, "declined_no_source": 2},
        target_estimand=ordinary.target_estimand,
    ).validate()
    panel = load_dataset_contract(study_path.parent / "dataset.yaml").panel(
        fixed["actuals"]["path"]
    )
    comparison = compare_run_forecasts(
        panel,
        {"test": malformed},
        spark,
        missing_actual="no_data",
        metric="rmse",
        output_dir=tmp_path / "comparison",
        fixed_contract=fixed,
    )
    candidate = json.loads(comparison.read_text())["candidates"][0]
    assert candidate["status"] != "scored", (
        "fixed study accepted a complete two-series grid labeled as three requested series "
        "with two declines"
    )


@pytest.mark.parametrize("changed_input", ["forecast", "comparison"])
def test_rescore_uses_pinned_forecast_bytes(tmp_path, spark, monkeypatch, changed_input):
    """A transient forecast rewrite must not be labeled with the original forecast digest."""
    from autoforecast import rescore
    from autoforecast.run_evaluation import compare_run_forecasts as real_compare

    study_path = freeze(tmp_path / "study", spark)
    fixed, forecast = _validation_forecast(study_path, tmp_path / "predictions.parquet")
    forecast_record = tmp_path / "forecast.json"
    forecast_record.write_text(json.dumps(forecast.as_record()))
    original = SimpleNamespace(
        metadata={"resolved_sha256": "a" * 64, "study_id": fixed["study_id"]},
        runs=(
            {
                "status": "completed",
                "stage": "validation",
                "adapter": "test",
                "model_id": "test",
                "variant_id": "default",
                "run_id": "retained-run",
                "origin": forecast.origin,
                "horizon": forecast.horizon,
                "forecast_record_path": str(forecast_record),
                "forecast_record_sha256": hashlib.sha256(
                    forecast_record.read_bytes()
                ).hexdigest(),
                "forecast_artifact_sha256": directory_digest(
                    forecast.path, suffixes=(".parquet",)
                ),
            },
        ),
    )
    monkeypatch.setattr(rescore, "read_results", lambda _: original)
    forecast_part = _part(pathlib.Path(forecast.path))
    original_bytes = forecast_part.read_bytes()
    original_table = pq.read_table(forecast_part)
    rows = original_table.to_pylist()
    for row in rows:
        row["yhat"] = 102.0
    changed = pa.Table.from_pylist(rows, schema=original_table.schema)

    changed_during_call = False
    comparison_reads = []
    real_read = pathlib.Path.read_bytes

    def transient_rewrite(*args, **kwargs):
        nonlocal changed_during_call
        changed_during_call = True
        if changed_input == "forecast":
            pq.write_table(changed, forecast_part)
        try:
            result = real_compare(*args, **kwargs)
            if changed_input == "comparison":
                captured = real_read(result)
                different = json.loads(captured)
                different["candidates"][0]["score"]["rmse"] = 987654.0

                def transient_comparison_read(path):
                    if path == result:
                        comparison_reads.append(path)
                        return (captured if len(comparison_reads) == 1
                                else json.dumps(different).encode())
                    return real_read(path)

                monkeypatch.setattr(pathlib.Path, "read_bytes", transient_comparison_read)
            return result
        finally:
            forecast_part.write_bytes(original_bytes)

    monkeypatch.setattr(rescore, "compare_run_forecasts", transient_rewrite)
    result = rescore.rescore_campaign(
        tmp_path / "campaign", study_path, tmp_path / "rescore", spark
    )
    record = json.loads(result.read_text())
    comparison_path = pathlib.Path(record["scores"][0]["comparison"])
    comparison = json.loads(real_read(comparison_path))
    candidate = comparison["candidates"][0]
    assert changed_during_call
    assert candidate["status"] == "scored"
    assert candidate["score"]["rmse"] == pytest.approx(3.5), (
        "rescore accepted a score from transient forecast bytes while recording the original hash"
    )
    assert record["scores"][0]["comparison_sha256"] == hashlib.sha256(
        real_read(comparison_path)
    ).hexdigest()
    if changed_input == "comparison":
        assert len(comparison_reads) == 1



@pytest.mark.parametrize("role", ["actuals", "forecast", "grid", "coverage", "references"])
def test_study_scores_private_copies_of_all_inputs(tmp_path, spark, monkeypatch, role):
    """A live input rewrite cannot change any value consumed by a scoring operation."""
    from autoforecast import study_scoring
    from autoforecast.run_evaluation import compare_run_forecasts
    from autoforecast.test_study import inputs, scoring_references

    contract, sources, recipe = inputs(tmp_path)
    recipe["metric"] = "wrmsse"
    study_path = benchmark.freeze_benchmark(
        contract_path=contract, sources=sources, recipe=recipe,
        output_root=tmp_path / "frozen", spark=spark,
        scoring_references=scoring_references(tmp_path),
    )
    study = benchmark.verify_benchmark(study_path)[1]
    fixed, forecast = _validation_forecast(
        study_path, tmp_path / "predictions.parquet", metric="wrmsse",
    )
    paths = {name: pathlib.Path(fixed[name]["path"]) for name in ("actuals", "grid", "coverage")}
    paths["forecast"] = pathlib.Path(forecast.path)
    paths["references"] = benchmark.artifact_path(study_path, study, "scoring_references")
    part = _part(paths[role])
    original = part.read_bytes()
    checksum = part.with_name(f".{part.name}.crc")
    checksum_bytes = checksum.read_bytes() if checksum.exists() else None
    table = pq.read_table(part)
    rows = table.to_pylist()
    for row in rows:
        if role == "actuals" and not row[benchmark.MISSING_COLUMN]:
            row["y"] += 100.0
        elif role == "forecast":
            row["yhat"] += 100.0
        elif role == "grid":
            row["series_id"] = "different-series"
        elif role == "coverage":
            row["scored"] += 1
        elif role == "references":
            row["scale"] *= 100.0
    changed = pa.Table.from_pylist(rows, schema=table.schema)
    score = study_scoring._score_study_snapshot
    calls = 0

    def concurrent_rewrite(*args, **kwargs):
        nonlocal calls
        calls += 1
        pq.write_table(changed, part)
        if checksum.exists():
            checksum.unlink()
        try:
            return score(*args, **kwargs)
        finally:
            part.write_bytes(original)
            if checksum_bytes is not None:
                checksum.write_bytes(checksum_bytes)

    monkeypatch.setattr(study_scoring, "_score_study_snapshot", concurrent_rewrite)
    panel = load_dataset_contract(study_path.parent / "dataset.yaml").panel(paths["actuals"])
    comparison = compare_run_forecasts(
        panel, {"test": forecast}, spark, metric="wrmsse", missing_actual="no_data",
        fixed_contract=fixed, output_dir=tmp_path / "comparison",
    )
    candidate = json.loads(comparison.read_bytes())["candidates"][0]
    assert calls == 1 and candidate["status"] == "scored"
    assert candidate["score"]["rmse"] == pytest.approx(3.5)
    # A's errors are 4 and 25; B's are 16 and 4. Supplied scales are 2.5 and 4.
    expected = (14 * math.sqrt(14.5 / 2.5) + 9 * math.sqrt(10 / 4)) / 23
    assert candidate["score"]["wrmsse"] == pytest.approx(expected)
    assert part.read_bytes() == original


def test_known_future_features_do_not_extend_validation_actuals_into_test(tmp_path, spark):
    import datetime as dt

    contract = tmp_path / "dataset.yaml"
    contract.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "dataset": {
                    "id": "known-future-study",
                    "panel": {
                        "series_col": "id",
                        "time_col": "date",
                        "target_col": "y",
                        "exposure_col": "state",
                        "known_future": ["promotion"],
                        "frequency": "D",
                        "season_length": 2,
                        "target_estimand": "observed_nonnegative",
                    },
                },
            }
        )
    )
    source = tmp_path / "source"
    source.mkdir()
    rows = [
        {
            "id": "A",
            "date": dt.date(2024, 1, day),
            "y": float(day),
            "state": "exact",
            "promotion": float(day % 2),
            "promotion_known_at": dt.datetime(2024, 1, 1),
        }
        for day in range(1, 10)
    ]
    pq.write_table(pa.Table.from_pylist(rows), source / "part.parquet")
    recipe = {
        "schema_version": 2,
        "source_keys": ["id"],
        "training_start": "2024-01-01",
        "validation_origins": ["2024-01-05"],
        "validation_horizon": 2,
        "final_train_cutoff": "2024-01-07",
        "final_test_horizon": 2,
        "history_policy": "fixed",
        "new_series_policy": "exclude_after_first_origin",
        "target_unit": "units",
        "feature_availability": {"promotion": "promotion_known_at"},
        "metric": "rmse",
        "actual_missing": "zero",
    }
    sources = dict.fromkeys(
        ("selection", "final_training", "validation_score", "final_score"), source
    )
    manifest = benchmark.freeze_benchmark(
        contract_path=contract,
        sources=sources,
        recipe=recipe,
        output_root=tmp_path / "frozen",
        spark=spark,
    )
    _, record = benchmark.verify_benchmark(manifest)
    validation = pq.read_table(
        benchmark.artifact_path(manifest, record, "validation_score")
    ).to_pylist()
    leaked = [
        (row["date"].isoformat(), row["y"])
        for row in validation
        if row["date"] > dt.date(2024, 1, 7)
    ]

    assert leaked == []
    assert max(row["date"] for row in validation) == dt.date(2024, 1, 7)
