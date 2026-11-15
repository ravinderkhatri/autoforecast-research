"""Registered model lists replay protected native inputs in separate processes."""

import json
import os

import pytest

from autoforecast.integrations.test_snapshots import postgres_store as _postgres_store
from autoforecast.test_study_tuning import spark as _spark


@pytest.fixture(scope="module")
def spark():
    yield from _spark.__wrapped__()


@pytest.fixture
def postgres_store():
    yield from _postgres_store.__wrapped__()


@pytest.mark.parametrize("execution,parallel", [("sequential", 1), ("local", 2)])
def test_postgres_public_model_list_preserves_protected_table_inputs(
    postgres_store, spark, tmp_path, monkeypatch, execution, parallel,
):
    from psycopg import errors, sql
    from psycopg.conninfo import make_conninfo

    from autoforecast import _logical_hash, load_study_campaign, register_study, run_study
    from autoforecast.benchmark import read_benchmark_manifest
    from autoforecast.study_registration import StudyRegistry
    from autoforecast.test_explicit_study import stages

    # Population: all 17 observations of two synthetic series; no eligibility exclusions.
    monkeypatch.setattr(_logical_hash, "fingerprint", lambda *a, **k: pytest.fail(
        "table-only registration and campaign must not hash input rows"))
    worker_config = tmp_path / "worker.json"
    worker_config.write_text(json.dumps({"connections": {"archive": {
        "kind": "postgresql", "dsn_env": "AF_LIST_READER_DSN",
        "writer_dsn_env": "AF_LIST_WRITER_DSN",
    }}}))
    monkeypatch.setenv("AUTOFORECAST_CONFIG", str(worker_config))
    for variable, role in (("AF_LIST_READER_DSN", "reader"), ("AF_LIST_WRITER_DSN", "writer")):
        monkeypatch.setenv(variable, make_conninfo(
            os.environ["AUTOFORECAST_TEST_INPUT_POSTGRES_DSN"],
            user=postgres_store.test_roles[role], password="local-test-reader-only",
        ))
    with StudyRegistry(tmp_path / "registry", spark=spark,
                       connections={"archive": postgres_store}, snapshots={
                           "connection": "archive", "namespace": postgres_store.schema,
                       }) as registry:
        saved_id = register_study(registry=registry, snapshot_mode="table",
                                  feature_roles={"season_length": 1}, **stages(spark))
        definition, _ = registry.definition(saved_id)
        _, manifest = read_benchmark_manifest(
            registry.root / "benchmarks" / saved_id / "benchmark.json",
            definition["library_sha256"],
        )
        records = list(manifest["owned_snapshots"].values())
        assert records
        assert all("logical_hash" not in record for record in records)
        # Source, registration and model-training accounts cannot rewrite retained rows.
        for record in records:
            assert postgres_store.reader.execute(
                sql.SQL("SELECT COUNT(*) FROM {}").format(postgres_store._name(record)),
            ).fetchone()[0] == record["rows"]
            for role in ("source", "writer", "reader"):
                with pytest.raises(errors.InsufficientPrivilege):
                    postgres_store.test_clients[role].execute(
                        sql.SQL("DELETE FROM {}").format(postgres_store._name(record)),
                    )
        arguments = dict(
            study_id=saved_id, registry=registry, models=["seasonal_naive", "window_average"],
            execution=execution, max_parallel_models=parallel, timeout_seconds=600,
            model_options={name: {"memory_gib": 2.0} for name in
                           ("seasonal_naive", "window_average")},
            output_dir=tmp_path / "campaign",
        )
        result = run_study(**arguments)
        assert result.status == "completed", result.summary()
        assert result.study_id == saved_id
        for row in result.candidates:
            assert row["validation_metrics"]["mae"] == 1.5
            assert row["testing_metrics"]["mae"] == 1.5
        assert load_study_campaign(result.path).summary() == result.summary()
        assert run_study(**arguments).candidates == result.candidates
