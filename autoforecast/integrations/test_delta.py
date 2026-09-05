"""Actual local Delta transactions and typed shared-snapshot verification."""

import datetime as dt
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from autoforecast.integrations.delta import DeltaStudyStore, latest_delta_version


@pytest.fixture(scope="module")
def spark(tmp_path_factory):
    if not os.environ.get("JAVA_HOME"):
        pytest.skip("Delta integration requires Java 17")
    root = tmp_path_factory.mktemp("delta-runtime")
    session = (SparkSession.builder.master("local[2]").appName("study-delta-tests")
               .config("spark.jars.packages", "io.delta:delta-spark_4.2_2.13:4.4.1")
               .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
               .config("spark.sql.catalog.spark_catalog",
                       "org.apache.spark.sql.delta.catalog.DeltaCatalog")
               .config("spark.sql.warehouse.dir", str(root / "warehouse"))
               .config("spark.sql.session.timeZone", "UTC")
               .config("spark.sql.shuffle.partitions", "2")
               .config("spark.databricks.delta.snapshotPartitions", "2")
               .config("spark.ui.enabled", "false")
               .config("spark.ui.showConsoleProgress", "false").getOrCreate())
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture
def store(spark, tmp_path):
    namespace = "study_" + tmp_path.name.replace("-", "_")
    result = DeltaStudyStore(spark, namespace=namespace)
    result.prepare(lock_path=tmp_path / "provision.lock")
    yield result
    spark.sql(f"DROP SCHEMA `{namespace}` CASCADE")


def _source(spark, root, *, integer=False, reverse=False):
    if integer:
        frame = spark.createDataFrame([(1, 2)], "item int,amount int")
    else:
        values = [("z", [2, 1]), ("a", [None, 3])]
        if reverse:
            values.reverse()
        frame = spark.createDataFrame([
            ("雪,\n", Decimal("12.30"), dt.datetime(2024, 1, 1, 12, 3, 4, 123456),
             dt.datetime(2024, 1, 1, 12, 3, 4, 123456), b"\x00\xff", {"map": dict(values)}),
            ("B", None, None, None, None, None),
            ("B", None, None, None, None, None),
        ], "item string,amount decimal(12,2),at timestamp_ntz,utc timestamp,bytes binary,"
           "nested struct<map:map<string,array<int>>>")
    path = root / "source"
    frame.write.parquet(str(path))
    return path


def test_shared_snapshots_roundtrip_typed_rows_and_retry_without_duplicates(store, tmp_path):
    first = store.freeze_observations(_source(store.spark, tmp_path / "one"),
                                      keys=["item"], scratch_root=tmp_path)
    assert first["rows"] == 3 and first["series"] == 2
    version = latest_delta_version(store.spark, store.tables["frozen_observations"])
    same = store.freeze_observations(_source(store.spark, tmp_path / "same", reverse=True),
                                     keys=["item"], scratch_root=tmp_path)
    assert same == first
    assert latest_delta_version(store.spark, store.tables["frozen_observations"]) == version
    other = store.freeze_observations(_source(store.spark, tmp_path / "two", integer=True),
                                      keys=["item"], scratch_root=tmp_path)
    assert other["snapshot_id"] != first["snapshot_id"]
    store.verify_observations(first, scratch_root=tmp_path)
    store.verify_observations(other, scratch_root=tmp_path)
    assert store.spark.table(store.tables["frozen_observations"]).count() == 4


def test_snapshot_verifier_rejects_changed_transport_and_preserves_other_scopes(store, tmp_path):
    source = _source(store.spark, tmp_path / "one", integer=True)
    record = store.freeze_observations(source, keys=["item"], scratch_root=tmp_path)
    table = store.tables["frozen_observations"]
    store.spark.sql(f"UPDATE {table} SET row_json = '{{\"item\":1,\"amount\":3}}'")
    with pytest.raises(ValueError, match="protected snapshot"):
        store.verify_observations(record, scratch_root=tmp_path)
    with pytest.raises(ValueError, match="conflicting immutable"):
        store.freeze_observations(source, keys=["item"], scratch_root=tmp_path)
    assert store.spark.table(store.tables["studies"]).count() == 0


@pytest.mark.parametrize("conflict", [False, True])
def test_simultaneous_writers_verify_the_same_logical_transaction(store, monkeypatch, conflict):
    barrier = threading.Barrier(2)
    local = threading.local()
    read = DeltaStudyStore.read

    def synchronized(self, kind, scope):
        result = read(self, kind, scope)
        if kind == "studies" and not getattr(local, "started", False):
            local.started = True
            # Force both readers to finish evaluating the initially empty scope.
            assert result[0].count() == 0
            barrier.wait(timeout=60)
        return result

    monkeypatch.setattr(DeltaStudyStore, "read", synchronized)

    def write(index):
        own = DeltaStudyStore(store.spark.newSession(), namespace=store.namespace)
        payload = "different" if conflict and index else "same"
        frame = own.spark.createDataFrame([("id", "sha", payload)],
                                         "study_id string,study_sha string,definition_json string")
        try:
            own.append_once("studies", frame, scope={"study_id": "id"})
            return "success"
        except Exception as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(write, range(2)))
    monkeypatch.setattr(DeltaStudyStore, "read", read)
    rows, _ = store.read("studies", {"study_id": "id"})
    assert rows.count() == 1
    assert sum(result == "success" for result in outcomes) == (1 if conflict else 2), outcomes
    # A later retry uses the exact content even if transaction retention changes.
    store.append_once("studies", rows, scope={"study_id": "id"})
    with pytest.raises(ValueError, match="conflicting immutable"):
        store.append_once("studies", rows.withColumn("definition_json", F.lit("wrong")),
                          scope={"study_id": "id"})


def test_unknown_study_read_does_not_provision_storage(spark):
    store = DeltaStudyStore(spark, namespace="absent_study_namespace")
    with pytest.raises(Exception, match="TABLE_OR_VIEW_NOT_FOUND"):
        store.read("studies", {"study_id": "missing"})
    assert not spark.catalog.databaseExists("absent_study_namespace")


@pytest.mark.parametrize("table", ["", ".bad", "bad..name", None])
def test_invalid_managed_table_identifiers_fail_before_runtime_access(table):
    from autoforecast.integrations.delta import read_sources

    with pytest.raises(ValueError, match="table names require"):
        read_sources(None, queries={"primary": "SELECT * FROM valid"},
                     tables={"valid": "neutral.sensors", "invalid": table})


def test_source_query_pins_versions_and_rejects_commands_and_undeclared_tables(store):
    from autoforecast.integrations.delta import read_source

    table = store.namespace + ".input"
    store.spark.createDataFrame([(1,)], "item int").write.format("delta").saveAsTable(table)
    frame, record = read_source(
        store.spark, sql="SELECT item FROM source", tables={"source": table},
    )
    (store.spark.createDataFrame([(2,)], "item int").write.format("delta")
     .mode("append").saveAsTable(table))
    assert [r.item for r in frame.collect()] == [1]
    assert record["tables"]["source"]["version"] == 0
    for query in ("DROP TABLE source", "SET spark.sql.shuffle.partitions=1",
                  "SELECT * FROM another", "INSERT INTO source VALUES (3)",
                  "SELECT * FROM range(3)",
                  "SELECT item FROM source WHERE item IN (SELECT item FROM another)"):
        with pytest.raises(ValueError):
            read_source(store.spark, sql=query, tables={"source": table})
    assert store.spark.table(table).count() == 2


@pytest.mark.parametrize("spelling", ["same", "qualified", "unqualified", "shared_location"])
@pytest.mark.parametrize("change", ["append", "update"])
def test_managed_queries_pin_shared_physical_table_once_before_any_execution(
    store, monkeypatch, spelling, change,
):
    from autoforecast.integrations import delta

    table = store.namespace + ".managed_input"
    store.spark.createDataFrame([(1,)], "item int").write.format("delta").saveAsTable(table)
    session = store.spark.newSession()
    session.catalog.setCurrentDatabase(store.namespace)
    audit_table = table
    if spelling == "qualified":
        audit_table = "spark_catalog." + table
    elif spelling == "unqualified":
        audit_table = "managed_input"
    elif spelling == "shared_location":
        audit_table = store.namespace + ".alternate_input"
        location = session.sql(f"DESCRIBE DETAIL {delta.quote_table(table)}").first()["location"]
        quoted_location = location.replace("'", "''")
        session.sql(f"CREATE TABLE {delta.quote_table(audit_table)} "
                    f"USING DELTA LOCATION '{quoted_location}'")
    original = delta.latest_delta_version
    lookups = []

    def changing(spark, physical):
        version = original(spark, physical)
        lookups.append(physical)
        if change == "append":
            (spark.createDataFrame([(2,)], "item int").write.format("delta")
             .mode("append").saveAsTable(physical))
        else:
            spark.sql(f"UPDATE {delta.quote_table(physical)} SET item = 2")
        return version

    monkeypatch.setattr(delta, "latest_delta_version", changing)
    tables = {"observations": table, "audit_source": audit_table}
    queries = {"primary": "SELECT item FROM observations",
               "audit": "SELECT item FROM audit_source"}
    managed = delta.read_sources(session, queries=queries, tables=tables)
    assert lookups == ["spark_catalog." + table]
    for frame, provenance in managed.values():
        assert [row.item for row in frame.collect()] == [1]
        assert {item["version"] for item in provenance["tables"].values()} == {0}
        assert provenance["sql"] in queries.values()
    for name in queries:
        for bad in ("DROP TABLE observations", "SELECT * FROM undeclared",
                    "SELECT * FROM observations WHERE item IN (SELECT item FROM undeclared)"):
            lookups.clear()
            with pytest.raises(ValueError):
                delta.read_sources(session, queries={**queries, name: bad}, tables=tables)
            assert not lookups


def test_prepared_registration_owns_audit_query_evidence_and_links_before_publication(
    store, tmp_path, monkeypatch,
):
    import dataclasses
    import json

    import yaml

    from autoforecast import register_study
    from autoforecast.study_registration import EXCLUSION_REASON_COLUMN, StudyRegistry
    from autoforecast.test_study_preparation import SensorPreparation
    from autoforecast.test_study_registration import configuration

    path = configuration(tmp_path)
    table = store.namespace + ".sensors"
    (store.spark.read.parquet(str(tmp_path / "source.parquet")).write.format("delta")
     .saveAsTable(table))
    (tmp_path / "primary.sql").write_text("SELECT item, day, value FROM measurements")
    (tmp_path / "population.sql").write_text("SELECT DISTINCT item FROM inventory")
    value = yaml.safe_load(path.read_text())
    value["source"] = {"kind": "delta", "sql": "primary.sql",
                       "tables": {"measurements": table, "inventory": table},
                       "audit_queries": {"population": "population.sql"},
                       "population": "population"}
    value["records"] = {"format": "yaml"}
    path.write_text(yaml.safe_dump(value))
    registry = StudyRegistry(tmp_path / "registry", spark=store.spark, namespace=store.namespace)

    class SensorAudit(SensorPreparation):
        def prepare(self, context):
            assert set(context.audits) == {"population"}
            assert context.audits["population"].count() == 2
            return dataclasses.replace(
                super().prepare(context), findings=context.findings.withColumn(
                    "issue_code", F.lit("SENSOR_NOTE")))

    append = registry.store.append_once

    def unavailable(role, frame, *, scope):
        if role == "study_data_issues" and scope.get("check_id", "").endswith("-application"):
            raise OSError("injected findings transport failure")
        return append(role, frame, scope=scope)

    with monkeypatch.context() as patch:
        patch.setattr(registry.store, "append_once", unavailable)
        with pytest.raises(OSError, match="findings transport failure"):
            register_study(path, registry=registry, prepare=SensorAudit())
    assert not list((registry.root / "registrations").glob("*.*"))
    assert store.spark.table(store.tables["studies"]).count() == 0
    assert list((registry.root / "checks").glob("*/application/*.parquet"))
    link = registry.store.link_issues

    def interrupted(check_id, study_id):
        link(check_id, study_id)
        if check_id.endswith("-application"):
            raise OSError("injected application linkage failure")

    with monkeypatch.context() as patch:
        patch.setattr(registry.store, "link_issues", interrupted)
        with pytest.raises(OSError, match="application linkage failure"):
            register_study(path, registry=registry, prepare=SensorAudit())
    assert not list((registry.root / "registrations").glob("*.*"))
    assert store.spark.table(store.tables["studies"]).count() == 0
    study_id = register_study(path, registry=registry, prepare=SensorAudit())
    definition, _ = registry.definition(study_id)
    record = yaml.safe_load((registry.root / definition["admission_evidence"]["path"]).read_text())
    assert record["identity"]["population_query"] == "population"
    assert record["identity"]["sources"]["population"]["rows"] == 2
    for source in record["identity"]["sources"].values():
        assert {item["version"] for item in source["tables"].values()} == {0}
    issues = store.spark.table(store.tables["study_data_issues"])
    successful = [p.parent.name for p in (registry.root / "checks").glob("*/registration.yaml")]
    assert len(successful) == 1
    completed = issues.filter(F.col("check_id").startswith(successful[0]))
    assert completed.count() > 0
    assert completed.filter(F.col("study_id").isNull()).count() == 0
    assert completed.filter(F.col("issue_json").contains("SENSOR_NOTE")).count() > 0
    # The interrupted attempt remains visible as failed evidence; its unfinished
    # report links must not be attributed to the later successful attempt.
    assert len(list((registry.root / "checks").glob("*/attempt.yaml"))) == 2
    stored = store.spark.table(store.tables["studies"]).first()
    assert json.loads(stored.definition_json) == definition
    # Population: A and B from inventory. The primary query loses B before
    # eligibility. Strict matching rejects this; explicit exclusion must account
    # for B when the caller permits a primary/population mismatch.
    (tmp_path / "primary.sql").write_text(
        "SELECT item, day, value FROM measurements WHERE item = 'A'")
    with pytest.raises(ValueError, match="membership"):
        register_study(path, registry=registry, prepare=SensorAudit())
    value["source"]["require_matching_population"] = False
    path.write_text(yaml.safe_dump(value))

    class MissingSensor(SensorAudit):
        def prepare(self, context):
            result = super().prepare(context)
            retained = context.observations.select("item").distinct()
            excluded = context.starting_keys.join(retained, ["item"], "left_anti")
            return dataclasses.replace(
                result, retained_keys=retained,
                exclusions=excluded.withColumn(EXCLUSION_REASON_COLUMN, F.lit("NO_OBSERVATIONS")),
            )

    filtered_id = register_study(path, registry=registry, prepare=MissingSensor())
    filtered, _ = registry.definition(filtered_id)
    evidence = registry.root / filtered["admission_evidence"]["path"]
    missing = store.spark.read.parquet(str(evidence.parent / "input_missing_keys")).collect()
    excluded = store.spark.read.parquet(str(evidence.parent / "exclusions")).collect()
    assert {row.item for row in missing} == {"B"}
    assert {(row.item, row[EXCLUSION_REASON_COLUMN]) for row in excluded} == {
        ("B", "NO_OBSERVATIONS"),
    }
    record = yaml.safe_load(evidence.read_text())
    assert record["outcome"]["starting"] == 2 and record["outcome"]["final_retained"] == 1
    assert filtered_id != study_id
    store.spark.sql(f"UPDATE {table} SET value = 200.0")
    path.unlink()
    (tmp_path / "primary.sql").unlink()
    (tmp_path / "population.sql").unlink()
    with registry.load_study(study_id) as study:
        assert study.training_data.filter("value = 200.0").count() == 0


@pytest.mark.parametrize("tracking,publish", [("none", False), ("none", True),
                                             ("mlflow", False), ("mlflow", True)])
def test_tracking_publication_matrix_and_retry(store, tmp_path, monkeypatch, tracking, publish):
    import yaml

    from autoforecast import register_study, run_mlflow, run_study
    from autoforecast.study_registration import StudyRegistry
    from autoforecast.test_study_registration import configuration
    from autoforecast.test_study_tuning import config, implementation, predict

    path = configuration(tmp_path)
    registry = StudyRegistry(tmp_path / "registry", spark=store.spark, namespace=store.namespace)
    study_id = register_study(path, registry=registry)
    profile = config(tmp_path)
    options = yaml.safe_load(profile.read_text())
    options.pop("hpo")
    if tracking == "none" and not publish:
        options.update(series_policy="exclude_unavailable", minimum_history=3)
    profile.write_text(yaml.safe_dump(options))
    settings = {}
    if tracking == "none":
        monkeypatch.setattr(run_mlflow, "_mlflow", lambda: pytest.fail("tracking disabled"))
    else:
        import mlflow

        uri = "sqlite:///" + str(tmp_path / "tracking.db")
        client = mlflow.tracking.MlflowClient(tracking_uri=uri)
        experiment = client.create_experiment(
            "registered", artifact_location=str(tmp_path / "mlruns"),
        )
        settings = {"tracking_uri": uri, "mlflow_experiment": experiment}
    calls = []

    def model(request):
        calls.append(request.purpose)
        return predict(request)

    arguments = dict(registry=registry, model_config=profile, implementation=implementation(),
                     tracking=tracking, publish=publish, **settings)
    if tracking == "none" and publish:
        from autoforecast.study_workflow import load_forecast_workflow

        append = registry.store.append_once
        failed = False

        def interrupt_final_record(kind, frame, *, scope):
            nonlocal failed
            if kind == "study_results" and not failed:
                failed = True
                raise OSError("interrupted after forecasts")
            return append(kind, frame, scope=scope)

        monkeypatch.setattr(registry.store, "append_once", interrupt_final_record)
        with pytest.raises(OSError, match="interrupted") as caught:
            run_study(study_id, model, **arguments)
        assert store.read("study_results", {"study_id": study_id})[0].count() == 0
        assert store.read("forecasts", {"study_id": study_id})[0].count() == 8
        events = [json.loads(line) for line in caught.value.audit_path.read_text().splitlines()]
        assert [(e["stage"], e["status"]) for e in events if e["status"] == "FAIL"] == [
            ("table_publication", "FAIL"), ("run", "FAIL"),
        ]
        assert events[-1]["status"] == "FAIL"
        result = load_forecast_workflow(caught.value.workflow_path)
        registry.publish_results(study_id, result.path)
    else:
        result = run_study(study_id, model, **arguments)
    assert calls == ["validation", "refit"]
    if tracking == "mlflow" and not publish:
        import dataclasses

        from autoforecast import study_scoring, study_tuning

        links = run_mlflow.export_forecast_workflow(
            result.path, tracking_uri=uri, experiment_id=experiment,
        )
        alien = client.create_run(experiment)
        wrong_link = dataclasses.replace(links["validation"], run_id=alien.info.run_id)
        for bad_links in ({"validation": links["validation"]},
                          {"validation": links["test"], "test": links["validation"]},
                          {**links, "validation": wrong_link},
                          {**links, "validation": "not-a-link"}):
            with pytest.raises(
                ValueError, match="both workflow stages|different workflow|verified MLflow"
            ):
                registry.publish_results(study_id, result.path, mlflow_links=bad_links)
            assert store.read("study_results", {"study_id": study_id})[0].count() == 0
            assert store.read("forecasts", {"study_id": study_id})[0].count() == 0
        client.set_terminated(alien.info.run_id)
        # Guard the separate publication-only retry against both scorer import paths.
        monkeypatch.setattr(study_scoring, "score_study", lambda *a, **k: pytest.fail("rescored"))
        monkeypatch.setattr(study_tuning, "score_study", lambda *a, **k: pytest.fail("rescored"))
    if tracking == "none" and not publish:
        assert result.parent_study_id == study_id
        parent, _ = registry.definition(study_id)
        child, _ = registry.definition(result.study_id)
        assert child["snapshot"] == parent["snapshot"]
        assert child["effective_series"] == 1
        assert store.read("frozen_observations", {
            "snapshot_id": child["snapshot"]["snapshot_id"],
        })[0].count() == 17
    results, _ = store.read("study_results", {"study_id": study_id})
    forecasts, _ = store.read("forecasts", {"study_id": study_id})
    assert results.count() == (2 if publish else 0)
    assert forecasts.count() == (8 if publish else 0)
    if publish:
        rows = {r.stage: r for r in results.collect()}
        assert json.loads(rows["validation"].evidence_json)["mae"] == pytest.approx(
            result.validation_metrics["mae"],
        )
        assert (rows["validation"].mlflow_uri is None) == (tracking == "none")
        links = (run_mlflow.export_forecast_workflow(result.path, tracking_uri=uri,
                                                    experiment_id=experiment)
                 if tracking == "mlflow" else None)
        from autoforecast import study_scoring, study_tuning

        monkeypatch.setattr(study_scoring, "score_study", lambda *a, **k: pytest.fail("rescored"))
        monkeypatch.setattr(study_tuning, "score_study", lambda *a, **k: pytest.fail("rescored"))
        assert registry.publish_results(
            study_id, result.path, mlflow_links=links,
        )["status"] == "published"
        assert calls == ["validation", "refit"]
        assert store.read("study_results", {"study_id": study_id})[0].count() == 2
        assert store.spark.sql(store.dashboard_sql()).count() == 2
        if tracking == "none":
            changed = yaml.safe_load(path.read_text())
            changed["recipe"]["actual_missing"] = "zero"
            path.write_text(yaml.safe_dump(changed))
            other_id = register_study(path, registry=registry)
            with pytest.raises(ValueError, match="different registered study"):
                registry.publish_results(other_id, result.path)
            assert store.read("study_results", {"study_id": other_id})[0].count() == 0
            model_file = result.validation["model_artifacts"] / "constant.json"
            model_file.write_text("changed")
            with pytest.raises(ValueError, match="not verified and completed"):
                registry.publish_results(study_id, result.path)
            assert store.read("study_results", {"study_id": study_id})[0].count() == 2


def test_simultaneous_initial_table_provisioning_is_retryable(spark, tmp_path):
    namespace = "provision_" + tmp_path.name.replace("-", "_")
    stores = [DeltaStudyStore(spark, namespace=namespace) for _ in range(2)]
    try:
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(
                lambda store: store.prepare(lock_path=tmp_path / "provision.lock"), stores,
            ))
        for store in stores:
            store.validate_destinations()
    finally:
        spark.sql(f"DROP SCHEMA IF EXISTS `{namespace}` CASCADE")


def test_sql_registration_partial_definition_recovery_and_live_source_changes(store, tmp_path,
                                                                           monkeypatch):
    import yaml

    from autoforecast import register_study
    from autoforecast.study_registration import StudyRegistry
    from autoforecast.test_study_registration import configuration

    path = configuration(tmp_path)
    table = store.namespace + ".input"
    (store.spark.read.parquet(str(tmp_path / "source.parquet")).write.format("delta")
     .saveAsTable(table))
    sql = tmp_path / "source.sql"
    sql.write_text("SELECT item, day, value FROM observations")
    value = yaml.safe_load(path.read_text())
    value["source"] = {"kind": "delta", "sql": "source.sql", "tables": {"observations": table}}
    value["storage"]["namespace"] = store.namespace
    path.write_text(yaml.safe_dump(value))
    registry = StudyRegistry.from_config(path, spark=store.spark)
    append = registry.store.append_once

    def interrupted(kind, frame, *, scope):
        if kind == "studies":
            raise OSError("registration interrupted")
        return append(kind, frame, scope=scope)

    monkeypatch.setattr(registry.store, "append_once", interrupted)
    with pytest.raises(OSError, match="registration interrupted"):
        register_study(path, registry=registry)
    assert store.spark.table(store.tables["studies"]).count() == 0
    assert not list((registry.root / "registrations").glob("*.json"))
    monkeypatch.setattr(registry.store, "append_once", append)
    study_id = register_study(path, registry=registry)
    baseline = registry.root / "registrations" / (study_id + ".json")
    before, before_sha = registry.definition(study_id)
    assert before["source"]["tables"]["observations"]["version"] == 0
    baseline_bytes = baseline.read_bytes()
    sql.write_text(sql.read_text() + "\n")
    with pytest.raises(ValueError, match="conflicting immutable study registration"):
        register_study(path, registry=registry)
    assert baseline.read_bytes() == baseline_bytes
    assert registry.definition(study_id) == (before, before_sha)
    store.spark.sql(f"UPDATE {table} SET value = 123.0 WHERE item = 'A'")
    sql.unlink()
    fresh = StudyRegistry.from_config(path, spark=store.spark)
    assert fresh.definition(study_id) == (before, before_sha)
    with fresh.load_study(study_id) as study:
        assert study.training_data.filter("value = 123.0").count() == 0
    file_only = StudyRegistry(registry.root, spark=store.spark)
    with pytest.raises(ValueError, match="storage does not match"):
        file_only.load_study(study_id)
    store.spark.sql(f"UPDATE {store.tables['studies']} SET definition_json = 'changed'")
    with pytest.raises(ValueError, match="table definition differs"):
        fresh.load_study(study_id)
    assert baseline.read_bytes() == baseline_bytes


def test_direct_registered_derived_id_can_publish(store, tmp_path, monkeypatch):
    import pathlib
    import shutil

    import yaml

    from autoforecast import run_study
    from autoforecast.study_registration import StudyRegistry, register_study
    from autoforecast.test_study_registration import configuration
    from autoforecast.test_study_tuning import config as model_profile
    from autoforecast.test_study_tuning import implementation, predict

    path = configuration(tmp_path)
    registry = StudyRegistry(tmp_path / "registry", spark=store.spark, namespace=store.namespace)
    parent = register_study(path, registry=registry)
    profile = model_profile(tmp_path)
    value = yaml.safe_load(profile.read_text())
    value.pop("hpo")
    value.update(series_policy="exclude_unavailable", minimum_history=3)
    profile.write_text(yaml.safe_dump(value))
    first = run_study(parent, predict, registry=registry, model_config=profile,
                      implementation=implementation())
    assert first.study_id != parent
    # Both documented routes to the derived registration must work: immediate
    # publication of the parent-started run, then direct execution of the child.
    assert registry.publish_results(first.study_id, first.path)["status"] == "published"
    second = run_study(first.study_id, predict, registry=registry, model_config=profile,
                       implementation=implementation(), publish=True)
    assert second.study_id == first.study_id
    assert second.publication["status"] == "published"
    import dataclasses

    from autoforecast import study_results

    load = study_results.load_forecast_workflow
    for field in ("study_id", "sha256"):
        invoked = []
        def wrong_binding(path, field=field, invoked=invoked):
            invoked.append(path)
            workflow = load(path)
            binding = {**workflow.provenance["registration"], field: "wrong"}
            return dataclasses.replace(workflow, provenance={**workflow.provenance,
                                                            "registration": binding})

        with monkeypatch.context() as patch:
            patch.setattr(study_results, "load_forecast_workflow", wrong_binding)
            with pytest.raises(ValueError, match="registration binding differs"):
                registry.publish_results(second.study_id, second.path)
            assert invoked
    manifest = pathlib.Path(first.provenance["study"]["path"])
    external = tmp_path / "external" / manifest.parent.name
    shutil.copytree(manifest.parent, external)
    with pytest.raises(ValueError, match="within the configured registry"):
        registry.retain_derived(parent, external / manifest.name)
    other_path = configuration(tmp_path / "other", actual_missing="zero")
    other_id = register_study(other_path, registry=registry)
    with pytest.raises(ValueError, match="does not belong to this registered parent"):
        registry.retain_derived(other_id, manifest)


def test_typed_snapshot_verification_does_not_trust_transport_hash_alone(store, tmp_path):
    import copy

    source = tmp_path / "typed-source"
    store.spark.createDataFrame([("A", 1), ("B", 1)], "item string,value int").write.parquet(
        str(source),
    )
    record = store.freeze_observations(source, keys=["item"], scratch_root=tmp_path)
    store.verify_observations(record, scratch_root=tmp_path)
    for field, value in (("logical_sha256", "0" * 64), ("rows", 3), ("series", 3),
                         ("columns", ["value", "item"]), ("source_keys", ["value"])):
        changed = {**record, field: value}
        with pytest.raises(ValueError, match="typed snapshot"):
            store.verify_observations(changed, scratch_root=tmp_path)
    changed = copy.deepcopy(record)
    changed["schema"]["fields"][1]["type"] = "string"
    with pytest.raises(ValueError, match="typed snapshot"):
        store.verify_observations(changed, scratch_root=tmp_path)
    for field, value in (("algorithm", "unknown"), ("codec", {})):
        with pytest.raises(ValueError, match="unsupported study snapshot"):
            store.verify_observations({**record, field: value}, scratch_root=tmp_path)
    assert store.spark.table(store.tables["frozen_observations"]).count() == 2


def test_unsupported_transport_and_non_utc_session_never_append(store, tmp_path):
    from pyspark.sql import types as T

    from autoforecast.integrations.delta import _transport_type

    for schema, message in ((T.StructType([]), "empty structs"),
                            (T.MapType(T.IntegerType(), T.StringType()), "string map keys"),
                            (T.ArrayType(T.CalendarIntervalType()), "unsupported study storage")):
        with pytest.raises(ValueError, match=message):
            _transport_type(schema)
    source = tmp_path / "unsupported"
    store.spark.createDataFrame([({1: "a"},)], "mapping map<int,string>").write.parquet(str(source))
    with pytest.raises(ValueError, match="string map keys"):
        store.freeze_observations(source, keys=["mapping"], scratch_root=tmp_path)
    session = store.spark.newSession()
    session.conf.set("spark.sql.session.timeZone", "Asia/Kolkata")
    other = DeltaStudyStore(session, namespace=store.namespace)
    with pytest.raises(ValueError, match="timeZone=UTC"):
        other.freeze_observations(_source(store.spark, tmp_path / "valid", integer=True),
                                  keys=["item"], scratch_root=tmp_path)
    assert store.spark.table(store.tables["frozen_observations"]).count() == 0
    assert store.spark.conf.get("spark.sql.session.timeZone") == "UTC"


def test_immutable_write_rejects_missing_null_and_conflicting_scope(store):
    from autoforecast.integrations.delta import SCHEMAS

    for value in (None, "wrong"):
        frame = store.spark.createDataFrame([(value, "sha", "{}")], SCHEMAS["studies"])
        with pytest.raises(ValueError, match="escape their immutable write scope"):
            store.append_once("studies", frame, scope={"study_id": "expected"})
    frame = store.spark.createDataFrame([("sha", "{}")], "study_sha string,definition_json string")
    with pytest.raises(ValueError, match="escape their immutable write scope"):
        store.append_once("studies", frame, scope={"study_id": "expected"})
    for scope in ({}, {"study_id": ""}, {"study_id": None}):
        with pytest.raises(ValueError, match="nonempty logical scope"):
            store.append_once("studies", frame, scope=scope)
    assert store.spark.table(store.tables["studies"]).count() == 0


@pytest.mark.parametrize("bad_kind", ["schema", "parquet"])
def test_bad_publication_destination_stops_before_fit_or_provision(store, tmp_path,
                                                                  monkeypatch, bad_kind):
    from pyspark.errors import AnalysisException

    from autoforecast import register_study, run_study
    from autoforecast.adapters import naive
    from autoforecast.integrations.delta import SCHEMAS
    from autoforecast.study_registration import StudyRegistry
    from autoforecast.test_study_registration import configuration

    path = configuration(tmp_path)
    registry = StudyRegistry(tmp_path / "registry", spark=store.spark, namespace=store.namespace)
    study_id = register_study(path, registry=registry)
    wrong = store.namespace + ".wrong_results"
    schema = "wrong string" if bad_kind == "schema" else SCHEMAS["study_results"]
    store.spark.createDataFrame([], schema).write.format("parquet").saveAsTable(wrong)
    registry.store = DeltaStudyStore(store.spark, namespace=store.namespace,
                                     tables={"study_results": wrong})
    monkeypatch.setattr(registry.store, "prepare", lambda **_: pytest.fail("implicit provision"))
    monkeypatch.setattr(naive, "fit_study_baseline", lambda _: pytest.fail("fit reached"))
    with pytest.raises(ValueError if bad_kind == "schema" else AnalysisException):
        run_study(study_id, "seasonal_naive", registry=registry, publish=True)
    assert store.spark.table(store.tables["study_results"]).count() == 0
    assert store.spark.table(store.tables["forecasts"]).count() == 0


def test_registration_links_reports_before_discovery_and_keeps_failed_checks(
    store, tmp_path, monkeypatch,
):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml

    from autoforecast import register_study
    from autoforecast.study_registration import StudyRegistry
    from autoforecast.test_study_registration import configuration

    path = configuration(tmp_path)
    value = yaml.safe_load(path.read_text())
    value["recipe"]["metric"] = "wrmsse"
    value["references"] = {"path": "references.parquet"}
    value["invalid_references"] = "exclude"
    path.write_text(yaml.safe_dump(value))
    rows = [{"keys": {"item": item}, "stage": stage, "origin": origin,
             "weight": 1.0, "scale": 1.0 if item == "A" else 0.0}
            for item in ("A", "B") for stage, origin in
            (("tuning", "2024-01-03"), ("validation", "2024-01-05"), ("test", "2024-01-07"))]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "references.parquet")
    monkeypatch.setenv("AUTOFORECAST_REGISTRY_ROOT", str(tmp_path / "environment-registry"))
    monkeypatch.setenv("AUTOFORECAST_SCRATCH_ROOT", str(tmp_path / "environment-scratch"))
    monkeypatch.setenv("AUTOFORECAST_DELTA_NAMESPACE", store.namespace)
    registry = StudyRegistry.from_environment(spark=store.spark)
    original = StudyRegistry._publish_definition

    def checked_publish(self, definition, **options):
        issues = store.spark.table(store.tables["study_data_issues"])
        assert issues.count() > 0
        assert issues.filter(F.col("study_id").isNull()).count() == 0
        assert issues.filter(F.col("study_id") != definition["study_id"]).count() == 0
        return original(self, definition, **options)

    monkeypatch.setattr(StudyRegistry, "_publish_definition", checked_publish)
    study_id = register_study(path, spark=store.spark)
    assert not (tmp_path / "registry").exists()
    assert registry.scratch_root == tmp_path / "environment-scratch"
    assert register_study(path, spark=store.spark) == study_id
    with registry.load_study(study_id) as study:
        assert study.registration["definition"]["effective_series"] == 1
    issues = store.spark.table(store.tables["study_data_issues"])
    checks = [row.check_id for row in issues.select("check_id").distinct().collect()]
    assert len(checks) == 4 and sum(c.endswith("-references") for c in checks) == 2
    count = issues.count()
    for check in checks:
        store.link_issues(check, study_id)
        with pytest.raises(ValueError, match="different study"):
            store.link_issues(check, "unrelated-study")
    assert store.spark.table(store.tables["study_data_issues"]).count() == count
    assert store.read("studies", {"study_id": study_id})[0].count() == 1

    # Failed admission leaves factual findings available, without a successful study link.
    value["invalid_references"] = "error"
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="invalid scoring references"):
        register_study(path, spark=store.spark)
    issues = store.spark.table(store.tables["study_data_issues"])
    assert issues.filter(F.col("study_id").isNull()).count() > 0
    assert issues.filter(F.col("study_id") == study_id).count() == count
    assert store.read("studies", {"study_id": study_id})[0].count() == 1


def test_issue_link_conflict_or_write_failure_never_publishes_registration(
    store, tmp_path, monkeypatch,
):
    from autoforecast import register_study
    from autoforecast.study_registration import StudyRegistry
    from autoforecast.test_study_registration import configuration

    path = configuration(tmp_path)
    registry = StudyRegistry(tmp_path / "registry", spark=store.spark, namespace=store.namespace)
    native = store.link_issues
    registry.store = store

    def fail_link(check_id, study_id):
        native(check_id, study_id)
        raise OSError("injected issue linkage failure")

    monkeypatch.setattr(store, "link_issues", fail_link)
    with pytest.raises(OSError, match="issue linkage failure"):
        register_study(path, registry=registry)
    assert store.spark.table(store.tables["studies"]).count() == 0
    assert not list((registry.root / "registrations").glob("*.json"))
    monkeypatch.setattr(store, "link_issues", native)
    study_id = register_study(path, registry=registry)
    assert registry.definition(study_id)[0]["study_id"] == study_id
    for check_id, bad_id in (("", study_id), ("check", ""), (None, study_id)):
        with pytest.raises(ValueError, match="check and study IDs"):
            store.link_issues(check_id, bad_id)
    check_id, linked_id = "check'\\{}雪", "study'\\{}雪"
    store.append_once("study_data_issues", store.spark.createDataFrame(
        [(check_id, None, "{}")], "check_id string,study_id string,issue_json string",
    ), scope={"check_id": check_id})
    native_sql = store.spark.sql

    def skip_update(query, *args, **kwargs):
        return None if query.startswith("UPDATE ") else native_sql(query, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(store.spark, "sql", skip_update)
        with pytest.raises(ValueError, match="differs from the checked report"):
            store.link_issues(check_id, linked_id)
    store.link_issues(check_id, linked_id)
    assert store.read("study_data_issues", {"check_id": check_id})[0].first().study_id == linked_id


def test_publish_cross_validation_windows_with_final_stages_and_retry(store, tmp_path, monkeypatch):
    """Two trials share three windows, one study and one parent workflow in Delta."""
    import copy
    import dataclasses
    import pathlib
    from collections import Counter

    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml

    from autoforecast import register_study, run_study, study_results, study_scoring, study_tuning
    from autoforecast.study_registration import StudyRegistry
    from autoforecast.test_study_registration import configuration
    from autoforecast.test_study_tuning import config, implementation, predict

    path = configuration(tmp_path)
    settings = yaml.safe_load(path.read_text())
    settings['recipe']['tuning_windows'] = [
        {'origin': f'2024-01-0{day}', 'horizon': 2 if day < 4 else 1} for day in (2, 3, 4)
    ]
    path.write_text(yaml.safe_dump(settings))
    registry = StudyRegistry(tmp_path / 'registry', spark=store.spark, namespace=store.namespace)
    study_id = register_study(path, registry=registry)
    profile = config(tmp_path, budget=2)
    options = yaml.safe_load(profile.read_text())
    options['selection'] = {'stage': 'tuning', 'origins': 'all', 'aggregation': 'mean'}
    profile.write_text(yaml.safe_dump(options))
    calls = []

    def model(request):
        calls.append((request.purpose, request.origin))
        output = predict(request)
        part = pathlib.Path(output.forecasts.path) / 'part.parquet'
        table = pq.read_table(part)
        table = table.set_column(table.schema.get_field_index('yhat'), 'yhat',
                                 pa.array([-(request.model_parameters['level'] + 2.5 + i) if i % 2
                                           else -(request.model_parameters['level'] + 1) * 1e-12
                                           for i in range(table.num_rows)]))
        pq.write_table(table, part)
        return output

    workflow = run_study(study_id, model, registry=registry, model_config=profile,
                         implementation=implementation(), tracking='none', publish=False)
    assert len(calls) == 8  # Two trials * three windows, validation, then refit.
    windows, _, _, _ = study_results._search_windows(workflow.tuning_path)
    assert len(windows) == 6
    first_windows = [pq.read_table(window['forecasts'].path).to_pylist()
                     for window, trial in windows if trial['window'] == 0]
    assert first_windows[0] != first_windows[1]  # A trial-path swap must be observable.
    monkeypatch.setattr(study_scoring, 'score_study', lambda *a, **k: pytest.fail('rescored'))
    monkeypatch.setattr(study_tuning, 'score_study', lambda *a, **k: pytest.fail('rescored'))

    # A valid search for another model must not be attached to this workflow.
    read_search = study_results._search_windows

    def wrong_model(path):
        frames, runs, info, sha = read_search(path)
        info = copy.deepcopy(info)
        info['provenance']['model_id'] = 'another-model'
        return frames, runs, info, sha

    with monkeypatch.context() as patch:
        patch.setattr(study_results, '_search_windows', wrong_model)
        with pytest.raises(ValueError, match='different workflow configuration'):
            registry.publish_results(study_id, workflow.path)

    # Changing verified parameters cannot be hidden by a later read of the winner file.
    load_selected = study_results.load_tuning_result

    def changed_selected(path):
        selected = load_selected(path)
        return dataclasses.replace(selected, parameters={'model.level': 999})

    with monkeypatch.context() as patch:
        patch.setattr(study_results, 'load_tuning_result', changed_selected)
        with pytest.raises(ValueError, match='different workflow configuration'):
            registry.publish_results(study_id, workflow.path)

    for duplicate in (True, False):
        def repeated_or_relocated(path, duplicate=duplicate):
            frames, runs, info, sha = read_search(path)
            if duplicate:
                frames = [*frames, frames[0]]
            else:
                frame, trial = frames[0]
                frames[0] = (frame, {**trial, 'path': str(path / 'copied-attempt')})
            return frames, runs, info, sha

        with monkeypatch.context() as patch:
            patch.setattr(study_results, '_search_windows', repeated_or_relocated)
            with pytest.raises(ValueError, match='duplicate tuning window|mismatched attempt path'):
                registry.publish_results(study_id, workflow.path)

    # A changed search between the first check and publication must also stop the write.
    reads = 0

    def changed_search(path):
        nonlocal reads
        frames, runs, info, sha = read_search(path)
        reads += 1
        return frames, runs, info, sha if reads == 1 else 'changed'

    with monkeypatch.context() as patch:
        patch.setattr(study_results, '_search_windows', changed_search)
        with pytest.raises(ValueError, match='changed during publication'):
            registry.publish_results(study_id, workflow.path)
    assert store.read('forecasts', {'study_id': study_id})[0].count() == 0

    # A partial publication can be retried without repeating any fit or score.
    append = registry.store.append_once

    def interrupt_results(kind, frame, *, scope):
        if kind == 'study_results':
            raise OSError('after forecast append')
        # Mutate the original after collection: lazy Spark reads must use the checked copy.
        part = next(pathlib.Path(windows[0][0]['forecasts'].path).rglob('*.parquet'))
        original = part.read_bytes()
        try:
            part.write_bytes(b'changed after snapshot')
            return append(kind, frame, scope=scope)
        finally:
            part.write_bytes(original)

    with monkeypatch.context() as patch:
        patch.setattr(registry.store, 'append_once', interrupt_results)
        with pytest.raises(OSError, match='after forecast append'):
            registry.publish_results(study_id, workflow.path)
    assert store.read('forecasts', {'study_id': study_id})[0].count() == 28
    assert store.read('study_results', {'study_id': study_id})[0].count() == 0
    for _ in range(2):
        assert registry.publish_results(study_id, workflow.path)['status'] == 'published'
    rows = store.read('study_results', {'study_id': study_id})[0].collect()
    forecasts = store.read('forecasts', {'study_id': study_id})[0]
    assert len(rows) == 8 and forecasts.count() == 28 and len(calls) == 8
    definition, sha = registry.definition(study_id)
    assert all(row.study_sha == sha and row.run_id == workflow.run_id for row in rows)
    assert store.spark.sql(store.dashboard_sql()).count() == 8
    expected = {(trial['trial_id'], trial['attempt_id'], trial['window']): window
                for window, trial in windows}
    chosen_windows = 0
    for row in rows:
        info = json.loads(row.additional_info)
        if row.stage != 'tuning':
            assert row.stage in {'validation', 'test'}
            assert (row.search_id, row.trial_id, row.attempt_id, row.window_index) == (None,) * 4
            assert row.horizon == 2
            evidence = workflow.validation if row.stage == 'validation' else workflow.testing
            assert json.loads(row.evidence_json) == evidence['score']
            actual = forecasts.filter(F.col('stage') == row.stage).collect()
            assert Counter((value.series_id, value.date, value.yhat)
                           for value in actual) == Counter(
                (value['series_id'], str(value['date']), value['yhat'])
                for value in pq.read_table(evidence['forecasts'].path).to_pylist()
            )
            continue
        assert row.search_id == workflow.tuning_path.name
        window = expected[(row.trial_id, row.attempt_id, row.window_index)]
        assert row.origin == window['forecasts'].origin
        assert row.horizon == window['forecasts'].horizon
        assert json.loads(row.evidence_json) == window['score']
        assert row.model_sha == window['model_sha256']
        assert row.model_uri == str(window['model_artifacts'])
        assert info['trial_status'] == 'completed'
        assert info['parameters'] and len(info['search_sha']) == 64
        assert row.mlflow_uri is None  # No invented tracking links for trial fits.
        chosen_windows += info['selected']
        matching = forecasts.filter(
            (F.col('stage') == 'tuning') & (F.col('run_id') == row.run_id)
            & (F.col('search_id') == row.search_id) & (F.col('trial_id') == row.trial_id)
            & (F.col('attempt_id') == row.attempt_id) & (F.col('window_index') == row.window_index)
            & (F.col('study_sha') == row.study_sha)
        )
        expected_values = Counter((value['series_id'], str(value['date']), value['yhat'])
                                  for value in pq.read_table(window['forecasts'].path).to_pylist())
        assert Counter((value.series_id, value.date, value.yhat)
                       for value in matching.collect()) == expected_values
    assert chosen_windows == 3
    analysis = study_results.read_workflow_results(workflow.path)
    assert Counter((row['series_id'], str(row['target_date']), row['prediction'])
                   for row in analysis.rows.to_pylist()) == Counter(
        (row.series_id, row.date, row.yhat) for row in forecasts.collect()
    )

    # Changing retained CV evidence must fail even after an earlier successful publish.
    model_file = next((windows[0][0]['model_artifacts']).glob('*.json'))
    model_file.write_text('changed')
    with pytest.raises(ValueError, match='artifact|manifest|changed'):
        registry.publish_results(study_id, workflow.path)
    assert store.read('study_results', {'study_id': study_id})[0].count() == 8
    assert store.read('forecasts', {'study_id': study_id})[0].count() == 28


def test_explicit_result_destination_overrides_delta_store(store, tmp_path, monkeypatch):
    from autoforecast.study_registration import StudyRegistry, register_study
    from autoforecast.test_study_registration import configuration

    registry = StudyRegistry(tmp_path / "registry", spark=store.spark, namespace=store.namespace)
    study_id = register_study(configuration(tmp_path), registry=registry)
    definition, sha = registry.definition(study_id)
    calls = []

    class Destination:
        def publish_results(self, *args, **kwargs):
            calls.append((args, kwargs))
            return {"status": "destination-selected"}

    monkeypatch.setattr(registry.store, "publish_results",
                        lambda *a, **k: pytest.fail("wrong writer"))
    path = tmp_path / "workflow"
    links = {"validation": object(), "test": object()}
    assert registry.publish_results(
        study_id, path, destination=Destination(), mlflow_links=links,
    ) == {
        "status": "destination-selected",
    }
    assert calls == [((definition, sha, path), {"scratch_root": registry.scratch_root,
                                              "mlflow_links": links})]
    assert registry.definition(study_id) == (definition, sha)
