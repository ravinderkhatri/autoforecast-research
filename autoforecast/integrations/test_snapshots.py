"""Native input tests distinguish local contracts from opt-in service guarantees."""

import datetime as dt
import os
import uuid
from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autoforecast.snapshots import RetainedInputs, SnapshotReference, resolve_mode


def _assert_original_stage_values(study, rows, *, windows=None):
    """Check all values in the shared 17-row fixture, including missing-value policy."""
    windows = windows or (("training", 1, 5), ("validation", 6, 7),
                          ("refit", 1, 7), ("testing", 8, 9))
    for stage, start, end in windows:
        expected = []
        for row in rows:
            if start <= row["day"].day <= end:
                value = row["value"]
                if value is None and stage in {"training", "refit"}:
                    value = 0.0
                expected.append((row["item"], row["day"], value))
        frame = getattr(study, stage + "_data")
        actual = [tuple(row) for row in frame.select("item", "day", "value").collect()]
        assert sorted(actual) == sorted(expected), stage


def _assert_native_manifest_bindings(record, tmp_path):
    """A freshly recomputed envelope cannot repair inconsistent saved bindings."""
    import copy
    import json

    from autoforecast import benchmark

    path = tmp_path / "binding-check.json"
    path.write_text(json.dumps(record))
    assert benchmark.read_benchmark_manifest(path)[1] == record
    boundary = next(iter(record["grids"]))
    for failure in ("missing_snapshot", "wrong_role", "wrong_source_mode", "wrong_mode",
                    "grid_path", "grid_identity", "coverage_path", "coverage_identity"):
        changed = copy.deepcopy(record)
        if failure == "missing_snapshot":
            changed["snapshots"].pop("source-keys")
        elif failure == "wrong_role":
            changed["artifacts"]["selection"] = changed["snapshots"]["final_score"]
        elif failure == "wrong_source_mode":
            changed["sources"]["selection"]["mode"] = "hash"
        elif failure == "wrong_mode":
            changed["snapshot_mode"] = "hash"
        elif failure == "grid_path":
            changed["grids"][boundary]["path"] = "coverage/" + boundary
        elif failure == "grid_identity":
            changed["grid_identities"][boundary]["snapshot"] = changed["snapshots"]["final_score"]
        elif failure == "coverage_path":
            changed["coverage"][boundary]["path"] = "grids/" + boundary
        else:
            changed["coverage_identities"][boundary]["snapshot"] = (
                changed["snapshots"]["final_score"])
        identity = benchmark.digest(benchmark._identity(changed))
        changed["study_id"] = changed["benchmark_id"] = identity
        path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="fields|mode|role|identity"):
            benchmark.read_benchmark_manifest(path)


@pytest.mark.parametrize("explicit,configured,default,expected", [
    (None, None, None, "hash"), ("hash", "table", "table+hash", "hash"),
    (None, "table", "hash", "table"), (None, None, "table+hash", "table+hash"),
])
def test_snapshot_mode_precedence(explicit, configured, default, expected):
    assert resolve_mode(explicit, configured, default) == expected


@pytest.mark.parametrize("value", ["", "auto", True, 1, {}, []])
def test_snapshot_mode_refuses_unknown_settings(value):
    with pytest.raises(ValueError, match="snapshot_mode"):
        resolve_mode(value)


@pytest.mark.parametrize("backend", ["bigquery", "postgresql", "mysql", "sqlserver"])
@pytest.mark.parametrize("mode", ["hash", "table", "table+hash"])
def test_configuration_infers_explicit_connection_and_reports_active_storage(
    tmp_path, backend, mode,
):
    import yaml

    from autoforecast.study_registration import validate_study_config
    from autoforecast.test_study_registration import configuration

    path = configuration(tmp_path)
    value = yaml.safe_load(path.read_text())
    (tmp_path / "source.sql").write_text("SELECT item, day, value FROM observations")
    value["source"] = {"connection": "source", "sql": "source.sql",
                       "tables": {"observations": "public.sales"}}
    value["connections"] = {"source": {"kind": backend}, "archive": {"kind": "postgresql"}}
    value["snapshot_mode"] = mode
    value["storage"]["snapshots"] = {"connection": "archive", "namespace": "retained"}
    path.write_text(yaml.safe_dump(value))
    checked = validate_study_config(path)
    assert checked.configuration["source"]["kind"] == backend
    assert "source SQL syntax" in checked.checked
    assert checked.input_storage == {
        "snapshot_mode": mode, "source_kind": backend,
        "snapshot_destination": "inactive" if mode == "hash" else value["storage"]["snapshots"],
    }
    assert not (tmp_path / "registry").exists()
    value["source"]["kind"] = "delta"
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="conflicts"):
        validate_study_config(path)
    del value["source"]["kind"]
    value["connections"] = {}
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="missing connection alias"):
        validate_study_config(path)


def test_reader_ownership_survives_a_lost_handle_and_blocks_new_reads_after_release(tmp_path):
    import json
    import subprocess
    import sys

    from mlops._safe_files import publish_bytes

    from autoforecast.snapshots import StudyLease

    study_id, sha = "a" * 64, "b" * 64
    manifest = tmp_path / "benchmarks" / study_id / "benchmark.json"
    reader = StudyLease(manifest, sha)
    assert StudyLease.active(tmp_path, study_id) == [reader.identifier]
    # A new process can see ownership without the original in-memory handle.
    output = subprocess.check_output([
        sys.executable, "-c", "import json,sys; from autoforecast.snapshots import StudyLease; "
        "print(json.dumps(StudyLease.active(sys.argv[1],sys.argv[2])))", str(tmp_path), study_id,
    ], text=True)
    assert json.loads(output) == [reader.identifier]
    reader.close()
    reader.close()
    assert StudyLease.active(tmp_path, study_id) == []
    publish_bytes(tmp_path, "released/" + study_id + ".json", b"{}")
    with pytest.raises(ValueError, match="released"):
        StudyLease(manifest, sha)


@pytest.fixture
def typed_input(tmp_path):
    # Four synthetic rows, three distinct IDs, one duplicate and no exclusions.
    schema = pa.schema([
        ("id", pa.int64()), ("value", pa.float64()), ("amount", pa.decimal128(25, 12)),
        ("day", pa.date32()), ("at", pa.timestamp("us")), ("label", pa.string()),
    ])
    rows = [dict(id=index, value=value, amount=Decimal("1234567890123.000000000001"),
                 day=dt.date(2024, 1, 1), at=dt.datetime(2024, 1, 1, 12, 1, 2, 123456),
                 label=label) for index, value, label in (
                     (1, -3.5, "one"), (2, 0.0, None), (3, None, "three"), (1, -3.5, "one"),
                 )]
    table = pa.Table.from_pylist(rows, schema=schema)
    path = tmp_path / "source.parquet"
    pq.write_table(table, path)
    return path, table


@pytest.fixture
def bigquery_store():
    dataset = os.environ.get("AUTOFORECAST_TEST_INPUT_BIGQUERY_DATASET")
    location = os.environ.get("AUTOFORECAST_TEST_INPUT_BIGQUERY_LOCATION")
    if not dataset or not location:
        if os.environ.get("AUTOFORECAST_REQUIRE_INPUT_BIGQUERY") == "1":
            pytest.fail("the required BigQuery input gate needs an authorized dataset and location")
        pytest.skip("BigQuery service unavailable: configure an authorized dataset and location")
    from google.cloud import bigquery

    from autoforecast.integrations.bigquery import BigQuerySnapshots

    with bigquery.Client(project=dataset.split(".")[0]) as client:
        yield BigQuerySnapshots(client, dataset=dataset, location=location)


@pytest.mark.parametrize("mode", ["table", "table+hash"])
def test_bigquery_retains_exact_typed_inputs(bigquery_store, typed_input, tmp_path, mode,
                                           monkeypatch):
    from google.api_core.exceptions import BadRequest, NotFound

    from autoforecast import _logical_hash

    path, expected = typed_input
    manager = RetainedInputs(mode=mode, connection="warehouse", store=bigquery_store,
                             scratch_root=tmp_path)
    if mode == "table":
        monkeypatch.setattr(_logical_hash, "fingerprint", lambda *a, **k: pytest.fail(
            "table mode must not hash input rows"))
    reference = manager.capture(path)
    record = reference.record()
    try:
        assert record["rows"] == 4
        assert ("logical_hash" in record) == (mode == "table+hash")
        # Reloading the inert record does not need the mutable original input.
        path.unlink()
        restored = SnapshotReference.from_record(record)
        with manager.open(restored) as retained:
            actual = pq.read_table(retained)
            assert actual.schema.equals(expected.schema, check_metadata=False)
            assert sorted(actual.to_pylist(), key=repr) == sorted(
                    expected.to_pylist(), key=repr,
                )
        assert not retained.exists()
        with pytest.raises(BadRequest, match="(?i)snapshot"):
            bigquery_store.client.query(
                f"UPDATE `{record['object']}` SET value = 1 WHERE TRUE",
                location=bigquery_store.location,
            ).result()
        table = bigquery_store.client.get_table(record["object"])
        assert table.expires is None
        assert table.table_type == "SNAPSHOT"
    finally:
        bigquery_store.release(record)
    with pytest.raises(NotFound):
        bigquery_store.client.get_table(record["object"])


def test_bigquery_refuses_negative_zero_before_writes(typed_input, tmp_path):
    from autoforecast.integrations.bigquery import BigQuerySnapshots

    path, table = typed_input
    values = pa.array([-0.0] + [1.0] * 3, type=pa.float64())
    pq.write_table(table.set_column(1, "value", values), path)
    with pytest.raises(ValueError, match="negative zero"):
        BigQuerySnapshots._input(path)


def test_bigquery_retained_input_detects_name_reuse(bigquery_store, typed_input, tmp_path):
    from google.cloud import bigquery

    path, _ = typed_input
    record = bigquery_store.capture(path, capture_id=uuid.uuid4().hex)
    try:
        bigquery_store.client.delete_table(record["object"])
        # A same-name mutable table must never satisfy a saved snapshot reference.
        table = bigquery.Table(record["object"], schema=[bigquery.SchemaField("id", "INTEGER")])
        table.expires = dt.datetime.now(dt.UTC) + dt.timedelta(hours=1)
        bigquery_store.client.create_table(table)
        with pytest.raises(ValueError, match="read-only snapshot"):
            bigquery_store.inspect(record)
        with pytest.raises(ValueError, match="read-only snapshot"):
            bigquery_store.release(record)
    finally:
        # The test owns both incarnations; production release must not delete the replacement.
        bigquery_store.client.delete_table(record["object"], not_found_ok=True)


@pytest.fixture
def postgres_store():
    dsn = os.environ.get("AUTOFORECAST_TEST_INPUT_POSTGRES_DSN")
    if not dsn:
        if os.environ.get("AUTOFORECAST_REQUIRE_INPUT_POSTGRES") == "1":
            pytest.fail("the required PostgreSQL test service has no DSN")
        pytest.skip("PostgreSQL service unavailable: configure a test DSN or local container")
    from contextlib import ExitStack

    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    from autoforecast.integrations.postgres import PostgresSnapshots, provision_snapshot_schema

    suffix = uuid.uuid4().hex
    roles = {kind: "af_" + kind + "_" + suffix
             for kind in ("owner", "writer", "reader", "release", "source")}
    namespace = "af_archive_" + suffix
    source_schema = "af_source_" + suffix
    with psycopg.connect(dsn, autocommit=True) as admin:
        try:
            for kind, role in roles.items():
                admin.execute(sql.SQL("CREATE ROLE {} {} NOSUPERUSER NOCREATEDB NOCREATEROLE")
                              .format(sql.Identifier(role), sql.SQL(
                                  "NOLOGIN" if kind == "owner" else
                                  "LOGIN PASSWORD 'local-test-reader-only'")))
            provision_snapshot_schema(admin, schema=namespace, owner=roles["owner"],
                                      registration_role=roles["writer"],
                                      reader_role=roles["reader"], release_role=roles["release"])
            admin.execute(sql.SQL("CREATE SCHEMA {} AUTHORIZATION {}").format(
                sql.Identifier(source_schema), sql.Identifier(roles["writer"]),
            ))
            with ExitStack() as stack:
                clients = {kind: stack.enter_context(psycopg.connect(make_conninfo(
                    dsn, user=role, password="local-test-reader-only",
                ), autocommit=True)) for kind, role in roles.items() if kind != "owner"}
                clients["writer"].execute(sql.SQL("SET search_path TO {}, public").format(
                    sql.Identifier(source_schema),
                ))
                store = PostgresSnapshots(clients["writer"], clients["reader"], schema=namespace,
                                          releaser=clients["release"])
                # Only the test harness has administrator and release credentials.
                store.test_admin, store.test_roles = admin, roles
                store.test_clients = clients
                store.test_source_schema = source_schema
                yield store
        finally:
            for schema in (namespace, source_schema):
                admin.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(schema),
                ))
            for role in roles.values():
                if admin.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (role,)).fetchone():
                    admin.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)))
                    admin.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))
            assert not admin.execute("SELECT 1 FROM pg_namespace WHERE nspname=ANY(%s)",
                                     ([namespace, source_schema],)).fetchall()
            assert not admin.execute("SELECT 1 FROM pg_roles WHERE rolname=ANY(%s)",
                                     (list(roles.values()),)).fetchall()


@pytest.mark.parametrize("mode", ["table", "table+hash"])
def test_postgres_retains_exact_typed_inputs(postgres_store, typed_input, tmp_path, mode,
                                           monkeypatch):
    from psycopg import errors, sql

    from autoforecast import _logical_hash

    path, expected = typed_input
    manager = RetainedInputs(mode=mode, connection="archive", store=postgres_store,
                             scratch_root=tmp_path)
    if mode == "table":
        monkeypatch.setattr(_logical_hash, "fingerprint", lambda *a, **k: pytest.fail(
            "table mode must not hash input rows"))
    reference = manager.capture(path)
    record = reference.record()
    try:
        path.unlink()
        with manager.open(SnapshotReference.from_record(record)) as retained:
            actual = pq.read_table(retained)
            assert actual.schema.equals(expected.schema, check_metadata=False)
            assert sorted(actual.to_pylist(), key=repr) == sorted(
                    expected.to_pylist(), key=repr,
                )
        assert not retained.exists()
        with pytest.raises(errors.InsufficientPrivilege):
            postgres_store.reader.execute(sql.SQL("UPDATE {} SET value=0").format(
                postgres_store._name(record),
            ))
        postgres_store.test_admin.execute(sql.SQL("GRANT UPDATE ON {} TO {}").format(
            postgres_store._name(record), sql.Identifier(postgres_store.reader_role),
        ))
        with pytest.raises(ValueError, match="without mutation"):
            postgres_store.inspect(record)
    finally:
        postgres_store.release(record)
    with pytest.raises(FileNotFoundError):
        postgres_store.inspect(record)


def test_postgres_refuses_same_reader_and_owner(postgres_store):
    from autoforecast.integrations.postgres import PostgresSnapshots

    with pytest.raises(ValueError, match="restricted reader"):
        PostgresSnapshots(postgres_store.writer, postgres_store.writer,
                          schema=postgres_store.schema).preflight()


@pytest.mark.parametrize("mode", ["table", "table+hash"])
def test_postgres_content_check_distinguishes_privileged_retained_row_changes(
    postgres_store, tmp_path, mode,
):
    from psycopg import sql

    # One synthetic input row, with no filtering or eligibility exclusions.
    path = tmp_path / "input.parquet"
    pq.write_table(pa.table({"id": [1], "value": [100]}), path)
    manager = RetainedInputs(mode=mode, connection="archive", store=postgres_store,
                             scratch_root=tmp_path)
    reference = manager.capture(path)
    record = reference.record()
    try:
        with manager.open(reference) as retained:
            assert pq.read_table(retained).to_pylist() == [{"id": 1, "value": 100}]
        # A trusted administrator can bypass application permissions. This changes the retained
        # table itself without changing its name, schema, identity or row count.
        postgres_store.test_admin.execute(sql.SQL("UPDATE {} SET value=120").format(
            postgres_store._name(record),
        ))
        postgres_store.inspect(record)
        if mode == "table+hash":
            with pytest.raises(ValueError, match="native input content changed"):
                with manager.open(reference):
                    pytest.fail("changed retained rows must not reach the experiment")
        else:
            # Table-only mode trusts the privileged owner; it cannot establish
            # content equality from metadata. Keep this limitation explicit.
            with manager.open(reference) as retained:
                assert pq.read_table(retained).to_pylist() == [{"id": 1, "value": 120}]
    finally:
        postgres_store.release(record)


@pytest.mark.parametrize("grant_before_capture", [True, False])
def test_postgres_refuses_reader_ownership_of_snapshot_schema(
    postgres_store, typed_input, grant_before_capture,
):
    from psycopg import sql

    store = postgres_store
    if not grant_before_capture:
        record = store.capture(typed_input[0], capture_id=uuid.uuid4().hex)
    store.test_admin.execute(sql.SQL("ALTER SCHEMA {} OWNER TO {}").format(
        sql.Identifier(store.schema), sql.Identifier(store.test_roles["reader"]),
    ))
    with pytest.raises(ValueError, match="protected NOLOGIN owner"):
        if grant_before_capture:
            store.preflight()
        else:
            store.inspect(record)


def test_postgres_refuses_reader_ownership_of_snapshot_database(postgres_store, typed_input):
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    from autoforecast.integrations.postgres import PostgresSnapshots, provision_snapshot_schema

    admin, roles = postgres_store.test_admin, postgres_store.test_roles
    dsn = os.environ["AUTOFORECAST_TEST_INPUT_POSTGRES_DSN"]
    database = "af_database_" + uuid.uuid4().hex
    admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    try:
        with (psycopg.connect(make_conninfo(dsn, dbname=database), autocommit=True) as setup,
              psycopg.connect(make_conninfo(dsn, dbname=database, user=roles["writer"],
                                           password="local-test-reader-only"),
                              autocommit=True) as writer,
              psycopg.connect(make_conninfo(dsn, dbname=database, user=roles["reader"],
                                           password="local-test-reader-only"),
                              autocommit=True) as reader):
            provision_snapshot_schema(setup, schema="retained", owner=roles["owner"],
                                      registration_role=roles["writer"],
                                      reader_role=roles["reader"], release_role=roles["release"])
            store = PostgresSnapshots(writer, reader, schema="retained")
            record = store.capture(typed_input[0], capture_id=uuid.uuid4().hex)
            admin.execute(sql.SQL("ALTER DATABASE {} OWNER TO {}").format(
                sql.Identifier(database), sql.Identifier(roles["reader"]),
            ))
            with pytest.raises(ValueError, match="must not own"):
                store.preflight()
            with pytest.raises(ValueError, match="must not own"):
                store.inspect(record)
    finally:
        admin.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(database)))


def test_postgres_refuses_name_replacement(postgres_store, typed_input):
    from psycopg import sql

    path, _ = typed_input
    record = postgres_store.capture(path, capture_id=uuid.uuid4().hex)
    table = postgres_store._name(record)
    try:
        postgres_store.test_admin.execute(sql.SQL("DROP TABLE {}").format(table))
        postgres_store.test_admin.execute(sql.SQL("CREATE TABLE {} (id BIGINT)").format(table))
        with pytest.raises(ValueError, match="sealed persistent"):
            postgres_store.inspect(record)
        with pytest.raises(Exception, match="different capture incarnation"):
            postgres_store.release(record)
    finally:
        postgres_store.test_admin.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(table))


@pytest.mark.parametrize("dialect", ["postgres", "bigquery", "mysql", "tsql"])
@pytest.mark.parametrize("query", [
    "DELETE FROM sales", "SELECT * FROM undeclared", "SELECT * INTO created FROM sales",
    "SELECT * FROM sales; DELETE FROM sales", "SELECT * FROM other.sales",
])
def test_native_source_queries_refuse_writes_and_unbound_tables(dialect, query):
    from autoforecast.snapshots import bind_source_queries

    with pytest.raises(ValueError):
        bind_source_queries({"primary": query}, {"sales": "declared"}, dialect=dialect,
                            sources={"sales": "captured"})


def test_postgres_sources_share_one_snapshot_and_block_ddl(postgres_store, tmp_path, monkeypatch):
    import psycopg
    from psycopg import errors, sql

    from autoforecast.integrations import postgres

    # Population: three synthetic IDs in each source, three joined output rows.
    names = {alias: postgres_store.test_source_schema + ".af_source_" + uuid.uuid4().hex
             for alias in ("sales", "prices")}
    writer = postgres_store.writer
    try:
        for index, name in enumerate(names.values()):
            writer.execute(sql.SQL("CREATE TABLE {} AS SELECT id, id+%s AS value "
                                   "FROM generate_series(1,3) AS id").format(
                sql.Identifier(*name.split(".")),
            ), (index * 100 + 10,))
        describe = postgres._source_arrow_schema
        changes = []
        with psycopg.connect(os.environ["AUTOFORECAST_TEST_INPUT_POSTGRES_DSN"],
                             autocommit=True) as source:
            def after_snapshot(connection, columns):
                if not changes:
                    for name in names.values():
                        writer.execute(sql.SQL("UPDATE {} SET value=999").format(
                            sql.Identifier(*name.split(".")),
                        ))
                    writer.execute("SET lock_timeout='50ms'")
                    with pytest.raises(errors.LockNotAvailable):
                        writer.execute(sql.SQL("TRUNCATE {}").format(
                            sql.Identifier(*names["prices"].split(".")),
                        ))
                    writer.execute("SET lock_timeout=0")
                    changes.append(True)
                return describe(connection, columns)
            monkeypatch.setattr(postgres, "_source_arrow_schema", after_snapshot)
            result = postgres.read_sources(source, queries={
                "primary": "SELECT s.id, s.value AS value, p.value AS price "
                           "FROM sales s JOIN prices p USING(id)",
                "audit": "SELECT CAST(SUM(value) AS BIGINT) AS total FROM prices",
            }, tables=names, output_root=tmp_path / "capture")
        assert changes
        assert sorted(pq.read_table(result["primary"][0]).to_pylist(), key=lambda r: r["id"]) == [
            {"id": i, "value": i + 10, "price": i + 110} for i in (1, 2, 3)
        ]
        assert pq.read_table(result["audit"][0]).to_pylist() == [{"total": 336}]
        assert result["primary"][1]["snapshot"] == result["audit"][1]["snapshot"]
    finally:
        for name in names.values():
            writer.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(
                sql.Identifier(*name.split(".")),
            ))


def test_bigquery_sources_share_one_timestamp(bigquery_store, tmp_path, monkeypatch):
    from google.cloud import bigquery

    from autoforecast.integrations.bigquery import read_sources

    # Population: three synthetic IDs per table, no exclusions, three joined rows.
    client = bigquery_store.client
    names = {alias: bigquery_store.dataset + ".af_source_" + uuid.uuid4().hex
             for alias in ("sales", "prices")}
    query = client.query
    try:
        for index, name in enumerate(names.values()):
            table = bigquery.Table(name, schema=[bigquery.SchemaField("id", "INTEGER"),
                                                bigquery.SchemaField("value", "INTEGER")])
            table.expires = dt.datetime.now(dt.UTC) + dt.timedelta(hours=1)
            client.create_table(table)
            client.load_table_from_json([{"id": i, "value": i + index * 100 + 10}
                                         for i in (1, 2, 3)], name,
                                        location=bigquery_store.location).result()
        changed = []

        def after_timestamp(sql, *args, **kwargs):
            job = query(sql, *args, **kwargs)
            if sql == "SELECT CURRENT_TIMESTAMP() AS captured":
                list(job.result())
                for name in names.values():
                    query(f"UPDATE `{name}` SET value=999 WHERE TRUE",
                          location=bigquery_store.location).result()
                changed.append(True)
            return job

        monkeypatch.setattr(client, "query", after_timestamp)
        result = read_sources(client, queries={
            "primary": "SELECT s.id, s.value AS value, p.value AS price "
                       "FROM sales s JOIN prices p USING(id)",
            "audit": "SELECT SUM(value) AS total FROM prices",
        }, tables=names, location=bigquery_store.location, output_root=tmp_path / "capture")
        assert changed
        assert sorted(pq.read_table(result["primary"][0]).to_pylist(), key=lambda r: r["id"]) == [
            {"id": i, "value": i + 10, "price": i + 110} for i in (1, 2, 3)
        ]
        assert pq.read_table(result["audit"][0]).to_pylist() == [{"total": 336}]
        assert result["primary"][1]["capture_time"] == result["audit"][1]["capture_time"]
    finally:
        for name in names.values():
            client.delete_table(name, not_found_ok=True)


def _assert_legacy_runners_refuse_native_inputs(study, root):
    import hashlib
    import json

    from scripts.run_autoforecast import run_request

    from autoforecast.benchmark import _evaluation_contract_from_record, read_benchmark_manifest
    from autoforecast.forecast_campaign import ForecastCampaignError, compile_forecast_campaign
    from autoforecast.hpo import HPOError
    from autoforecast.hpo_bridge import ForecastTrialBridge
    from autoforecast.train_workflow import TrainWorkflowError

    # Use the real registered manifest. No resolver or row export is needed to refuse it.
    manifest, record = read_benchmark_manifest(study._path, study._sha256)
    unused = root / "not-read"
    with pytest.raises(ForecastCampaignError, match="use Study.run_campaign"):
        compile_forecast_campaign(unused, selection_data_path=unused,
                                  final_training_data_path=unused, final_score_data_path=unused,
                                  benchmark_path=manifest)
    contract = _evaluation_contract_from_record(
        manifest, record, stage="test", origin=record["recipe"]["final_train_cutoff"],
        horizon=record["recipe"]["final_test_horizon"], metric=record["recipe"]["metric"],
    )
    path = root / "native-evaluation.json"
    raw = json.dumps(contract).encode()
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    bridge = ForecastTrialBridge(
        request_path=unused, data_path=unused, data_snapshot_id="native-input",
        data_reference="local://native-input", epochs=1, timeout_seconds=10,
        project_root=root, source_data_path=unused, source_data_fingerprint="a" * 64,
        evaluation_contract_path=path, evaluation_contract_sha256=digest,
    )
    with pytest.raises(HPOError, match="use Study.tune"):
        bridge._evaluation_contract()
    with pytest.raises(TrainWorkflowError, match="use run_study"):
        run_request(unused, data_path=unused, data_snapshot_id="native-input",
                    data_reference="local://native-input", output_root=unused,
                    epochs=1, timeout_seconds=10,
                    evaluation_contract_path=path, evaluation_contract_sha256=digest)
    assert not unused.exists()


@pytest.mark.parametrize("credential", [
    "administrator", "release", "inherited_release", "server_program",
])
def test_registration_refuses_privileged_postgres_source_accounts(
    postgres_store, tmp_path, credential,
):
    import yaml
    from psycopg import sql
    from pyspark.sql import SparkSession

    from autoforecast import register_study
    from autoforecast.study_registration import StudyRegistry
    from autoforecast.test_study_registration import configuration

    # No rows should be read: every case is refused before the source query runs.
    path = configuration(tmp_path)
    (tmp_path / "source.sql").write_text("SELECT item, day, value FROM observations")
    value = yaml.safe_load(path.read_text())
    value["source"] = {"kind": "postgresql", "connection": "source",
                       "tables": {"observations": "public.not_read"}, "sql": "source.sql"}
    path.write_text(yaml.safe_dump(value))
    source = postgres_store.test_clients["source"]
    grant = None
    if credential == "administrator":
        source = postgres_store.test_admin
    elif credential == "release":
        source = postgres_store.test_clients["release"]
    else:
        grant = (postgres_store.test_roles["release"] if credential == "inherited_release"
                 else "pg_execute_server_program")
        postgres_store.test_admin.execute(sql.SQL("GRANT {} TO {}").format(
            sql.Identifier(grant), sql.Identifier(postgres_store.test_roles["source"]),
        ))
    session = (SparkSession.builder.master("local[2]").appName("source-account-preflight")
               .config("spark.ui.enabled", "false").getOrCreate())
    session.sparkContext.setLogLevel("ERROR")
    try:
        registry = StudyRegistry(tmp_path / "registry", spark=session, connections={
            "source": source, "archive": postgres_store,
        }, snapshots={"connection": "archive", "namespace": postgres_store.schema})
        with pytest.raises(ValueError, match="ordinary source account"):
            register_study(path, registry=registry, snapshot_mode="table")
        assert not list((tmp_path / "registry" / "studies").glob("*/study.json"))
        assert postgres_store.reader.execute(sql.SQL("SELECT count(*) FROM {} ").format(
            sql.Identifier(postgres_store.schema, "af_captures"),
        )).fetchone()[0] == 0
    finally:
        session.stop()
        if grant:
            postgres_store.test_admin.execute(sql.SQL("REVOKE {} FROM {}").format(
                sql.Identifier(grant), sql.Identifier(postgres_store.test_roles["source"]),
            ))


@pytest.mark.parametrize("mode", ["hash", "table", "table+hash"])
@pytest.mark.parametrize("prepare_case", ["none", "prepared"])
def test_postgres_source_registers_through_the_existing_study_pipeline(
    postgres_store, tmp_path, mode, monkeypatch, prepare_case,
):
    import yaml
    from psycopg import sql
    from pyspark.sql import SparkSession

    from autoforecast import register_study
    from autoforecast.study_registration import StudyRegistry
    from autoforecast.test_study_registration import configuration

    # Population: the existing two-series, 17-row fixture; no client data is read.
    config = configuration(tmp_path)
    rows = pq.read_table(tmp_path / "source.parquet").to_pylist()
    name = "af_source_" + uuid.uuid4().hex
    table = sql.Identifier(postgres_store.test_source_schema, name)
    writer = postgres_store.writer
    session = (SparkSession.builder.master("local[2]").appName("native-source-registration")
               .config("spark.sql.shuffle.partitions", "2")
               .config("spark.ui.enabled", "false").getOrCreate())
    session.sparkContext.setLogLevel("ERROR")
    captures = []
    capture = postgres_store.capture

    def owned(*args, **kwargs):
        reference = capture(*args, **kwargs)
        captures.append(reference)
        return reference

    monkeypatch.setattr(postgres_store, "capture", owned)
    if mode == "table":
        from autoforecast import _logical_hash

        monkeypatch.setattr(_logical_hash, "fingerprint",
                            lambda *_a, **_k: pytest.fail("table registration hashed rows"))
    try:
        writer.execute(sql.SQL("CREATE TABLE {} (item TEXT, day DATE, value DOUBLE PRECISION)")
                       .format(table))
        with writer.cursor() as cursor:
            cursor.executemany(sql.SQL("INSERT INTO {} VALUES (%s,%s,%s)").format(table),
                               [(r["item"], r["day"], r["value"]) for r in rows])
        (tmp_path / "source.sql").write_text("SELECT item, day, value FROM observations")
        value = yaml.safe_load(config.read_text())
        value["source"] = {"kind": "postgresql", "connection": "source",
                           "tables": {
                               "observations": postgres_store.test_source_schema + "." + name,
                           },
                           "sql": "source.sql"}
        config.write_text(yaml.safe_dump(value))
        registry = StudyRegistry(tmp_path / "registry", spark=session,
                                 connections={"source": writer, "archive": postgres_store},
                                 snapshot_mode="hash", snapshots={
                                     "connection": "archive", "namespace": postgres_store.schema,
                                 })
        from autoforecast.test_study_preparation import SensorPreparation

        hook = SensorPreparation() if prepare_case == "prepared" else None
        study_id = register_study(config, registry=registry, prepare=hook, snapshot_mode=mode)
        writer.execute(sql.SQL("UPDATE {} SET value=999").format(table))
        with registry.load_study(study_id) as study:
            assert study.n_series == 2
            assert study.training_data.filter("value = 999").count() == 0
            assert study.testing_data.count() == 4
            _assert_original_stage_values(study, rows)
            if mode != "hash":
                _assert_legacy_runners_refuse_native_inputs(study, tmp_path)
                with pytest.raises(ValueError, match="active readers"):
                    registry.release(study_id)
        definition, _ = registry.definition(study_id)
        assert definition["source"]["kind"] == "postgresql"
        assert definition["source"]["rows"] == 17
        if prepare_case == "prepared" and mode != "hash":
            evidence = registry.root / definition["admission_evidence"]["path"]
            assert not list(evidence.parent.rglob("*.parquet"))
            assert yaml.safe_load(evidence.read_text())["schema"] == "autoforecast.admission.v2"
        if mode != "hash":
            assert not list(registry.root.rglob("*.parquet"))
            assert definition["native_findings"]
            capture_id = definition["native_findings"][0]["snapshot"]["capture_id"]
            with pytest.raises(ValueError, match="published"):
                registry.recover_capture(capture_id)
            marker = registry.root / definition["native_findings"][0]["path"]
            original_marker = marker.read_bytes()
            try:
                marker.write_bytes(b"{}")
                with pytest.raises(ValueError, match="native findings differ"):
                    registry.definition(study_id)
            finally:
                marker.write_bytes(original_marker)
            release, failed = postgres_store.release, []

            def lost_acknowledgement(reference):
                release(reference)
                if not failed:
                    failed.append(reference["capture_id"])
                    raise ConnectionError("synthetic lost deletion acknowledgement")

            monkeypatch.setattr(postgres_store, "release", lost_acknowledgement)
            with pytest.raises(ConnectionError, match="lost deletion"):
                registry.release(study_id)
            with pytest.raises(ValueError, match="released"):
                registry.load_study(study_id)
            assert registry.release(study_id) == study_id
            assert registry.release(study_id) == study_id
            with pytest.raises(ValueError, match="released"):
                registry.load_study(study_id)
            for reference in captures:
                with pytest.raises(FileNotFoundError):
                    postgres_store.inspect(reference)
    finally:
        for reference in captures:
            postgres_store.release(reference)
        writer.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(table))
        session.stop()


def test_bigquery_source_registers_through_the_existing_study_pipeline(bigquery_store, tmp_path):
    import yaml
    from google.cloud import bigquery
    from pyspark.sql import SparkSession

    from autoforecast import register_study
    from autoforecast.study_registration import StudyRegistry
    from autoforecast.test_study_registration import configuration

    # Population: two synthetic series, 17 rows, including the declared missing values.
    config = configuration(tmp_path)
    expected_rows = pq.read_table(tmp_path / "source.parquet").to_pylist()
    name = bigquery_store.dataset + ".af_source_" + uuid.uuid4().hex
    client = bigquery_store.client
    session = (SparkSession.builder.master("local[2]").appName("cloud-source-registration")
               .config("spark.sql.shuffle.partitions", "2")
               .config("spark.ui.enabled", "false").getOrCreate())
    session.sparkContext.setLogLevel("ERROR")
    try:
        table = bigquery.Table(name, schema=[bigquery.SchemaField("item", "STRING"),
                                            bigquery.SchemaField("day", "DATE"),
                                            bigquery.SchemaField("value", "FLOAT")])
        table.expires = dt.datetime.now(dt.UTC) + dt.timedelta(hours=1)
        client.create_table(table)
        with (tmp_path / "source.parquet").open("rb") as stream:
            client.load_table_from_file(
                stream, name, location=bigquery_store.location,
                job_config=bigquery.LoadJobConfig(source_format="PARQUET"),
            ).result()
        (tmp_path / "source.sql").write_text("SELECT item, day, value FROM observations")
        value = yaml.safe_load(config.read_text())
        value["source"] = {"kind": "bigquery", "connection": "warehouse",
                           "tables": {"observations": name}, "sql": "source.sql"}
        config.write_text(yaml.safe_dump(value))
        registry = StudyRegistry(tmp_path / "registry", spark=session,
                                 connections={"warehouse": client}, connection_settings={
                                     "warehouse": {"kind": "bigquery",
                                                   "project": bigquery_store.dataset.split('.')[0],
                                                   "location": bigquery_store.location},
                                 })
        study_id = register_study(config, registry=registry)
        client.query(f"UPDATE `{name}` SET value=999 WHERE TRUE",
                     location=bigquery_store.location).result()
        with registry.load_study(study_id) as study:
            assert study.n_series == 2
            assert study.training_data.filter("value = 999").count() == 0
            assert study.testing_data.count() == 4
            _assert_original_stage_values(study, expected_rows)
        definition, _ = registry.definition(study_id)
        assert definition["source"]["kind"] == "bigquery"
        assert definition["source"]["rows"] == 17
    finally:
        client.delete_table(name, not_found_ok=True)
        session.stop()


@pytest.mark.parametrize("mode", ["table", "table+hash"])
def test_postgres_freezer_retains_native_rows_without_publishing_parquet(
    postgres_store, tmp_path, mode, monkeypatch,
):
    from pyspark.sql import SparkSession

    from autoforecast import _logical_hash, benchmark
    from autoforecast.snapshots import Connections, NativeReadScope
    from autoforecast.test_study import inputs

    # Three input series become two eligible series; the late arrival is excluded.
    contract, sources, recipe = inputs(tmp_path)
    session = (SparkSession.builder.master("local[2]").appName("native-benchmark")
               .config("spark.sql.shuffle.partitions", "2")
               .config("spark.ui.enabled", "false").getOrCreate())
    session.sparkContext.setLogLevel("ERROR")
    manager = RetainedInputs(mode=mode, connection="archive", store=postgres_store,
                             scratch_root=tmp_path)
    scope = NativeReadScope(Connections(supplied={"archive": postgres_store}),
                            scratch_root=tmp_path, spark=session)
    references = []
    capture = postgres_store.capture

    def track(*args, **kwargs):
        reference = capture(*args, **kwargs)
        references.append(reference)
        return reference

    monkeypatch.setattr(postgres_store, "capture", track)
    if mode == "table":
        from autoforecast import study_tuning

        monkeypatch.setattr(_logical_hash, "fingerprint",
                            lambda *_a, **_k: pytest.fail("table mode hashed input rows"))
        monkeypatch.setattr(benchmark, "directory_digest",
                            lambda *_a, **_k: pytest.fail("table mode hashed input files"))
        output_digest = study_tuning.directory_digest

        def outputs_only(path, **kwargs):
            if str(path).split("/")[-1] in {"training", "series-mapping"}:
                pytest.fail("table mode hashed callback inputs")
            return output_digest(path, **kwargs)

        monkeypatch.setattr(study_tuning, "directory_digest", outputs_only)
    try:
        with scope.active():
            frozen = benchmark._freeze_verified(
                contract_path=contract, sources=sources, recipe=recipe,
                output_root=tmp_path / "benchmarks", spark=session, native_inputs=manager,
            )
            path, record = benchmark.verify_benchmark(frozen.path, frozen.sha256)
            assert record["schema"] == benchmark.NATIVE_SCHEMA
            assert "logical_snapshots" not in record and "logical_hashes" not in record
            assert record["cohort_series_count"] == 2
            assert record["excluded_new_series"] == 1
            _assert_native_manifest_bindings(record, tmp_path)
            assert not list(path.parent.rglob("*.parquet"))
            rows = pq.read_table(benchmark.artifact_path(path, record, "final_score")).to_pylist()
            assert len(rows) == 4
            assert sum(row[benchmark.MISSING_COLUMN] for row in rows) == 2
            assert any(row["y"] == 0 and not row[benchmark.MISSING_COLUMN] for row in rows)
            # Check each saved value at its own series and date. Aggregate scores
            # cannot detect values that were accidentally swapped between dates.
            expected_series = {
                "A": [1.0, 2.0, 3.0, None, 5.0, None, 7.0, 0.0, 9.0],
                "B": [None, None, 3.0, 4.0, None, 6.0, None, None, None],
            }
            for name, start, end in (("selection", 1, 5),
                                      ("final_training", 1, 7),
                                      ("validation_score", 1, 7),
                                      ("final_score", 8, 9)):
                saved = pq.read_table(benchmark.artifact_path(path, record, name)).to_pylist()
                actual = sorted((r["id"], r["date"].day, r["y"],
                                 r[benchmark.MISSING_COLUMN]) for r in saved)
                expected = sorted(
                    (series, day, values[day - 1], values[day - 1] is None)
                    for series, values in expected_series.items()
                    for day in range(max(start, 3 if series == "B" else 1), end + 1)
                )
                assert actual == expected, name
            for item in record["grids"].values():
                assert item["n_cells"] == 4
            # Removing the source does not change retained model inputs.
            next(iter(sources.values())).unlink()
            benchmark.verify_benchmark(path, frozen.sha256)
        from autoforecast.study import load_study

        with load_study(record["study_id"], registry_root=tmp_path / "benchmarks",
                        spark=session, scratch_root=tmp_path,
                        connections={"archive": postgres_store}) as study:
            assert study.n_series == 2
            assert study.training_data.count() == 8
            assert study.testing_data.count() == 4
            assert study.series_mapping.count() == 2
            assert study.scoring_references is None
            from autoforecast.panel import Forecast

            predictions = {}
            with study._native_inputs.active():
                for stage, origin in (("validation", "2024-01-05"), ("test", "2024-01-07")):
                    fixed = benchmark._evaluation_contract_from_record(
                        path, record, stage=stage, origin=origin, horizon=2, metric="rmse",
                    )
                    cells = pq.read_table(benchmark.bound_input_path(fixed["grid"])).to_pylist()
                    prediction_path = tmp_path / (stage + "-predictions.parquet")
                    pq.write_table(pa.Table.from_pylist([
                        {**row, "yhat": -1.0} for row in cells
                    ]), prediction_path)
                    predictions[stage] = Forecast(
                        "native_fixture", str(prediction_path), origin, 2, 2,
                        {"emitted": 4}, target_estimand="observed_nonnegative",
                    )
            result = study.evaluate_forecasts(
                validation_forecasts=predictions["validation"],
                testing_forecasts=predictions["test"], model_id="native_fixture",
                output_dir=tmp_path / "evaluations", forecast_accuracy_breakdown="none",
            )
            # Test observed values are 0 and 9. Predictions stay at -1, giving these errors.
            assert result["test"]["score"]["rmse"] == pytest.approx((101 / 2) ** 0.5)
            assert result["test"]["evaluation_contract"]["schema"].endswith("v3")
            assert "path" not in result["test"]["evaluation_contract"]["actuals"]
            from autoforecast.results import read_workflow_results
            from autoforecast.test_study_tuning import config, implementation, predict

            workflow = study.run_forecast_workflow(
                fit_predict=predict, config=config(tmp_path), implementation=implementation(),
                fixed_parameters={}, output_dir=tmp_path / "workflow",
            )
            collected = read_workflow_results(
                workflow.path, connections={"archive": postgres_store}, scratch_root=tmp_path,
            )
            assert len(collected.rows) == 8
            assert set(collected.rows["prediction"].to_pylist()) == {0.0}

            def changed_input(request):
                import pathlib

                result = predict(request)
                file = next(pathlib.Path(request.panel.path).glob("*.parquet"))
                original = pq.read_table(file)
                file.chmod(0o600)
                column = original.schema.get_field_index(request.panel.target_col)
                changed = original.set_column(column, original.schema.field(column),
                                              pa.array([999.0] * len(original)))
                pq.write_table(changed, file)
                return result

            with pytest.raises(ValueError, match="owned native input transport changed"):
                study.run_forecast_workflow(
                    fit_predict=changed_input, config=config(tmp_path),
                    implementation=implementation(), fixed_parameters={},
                    output_dir=tmp_path / "changed-input-workflow",
                )
            assert study.training_data.filter("y = 999").count() == 0
            from autoforecast import study_preflight

            if mode == "table":
                monkeypatch.setattr(study_preflight, "directory_digest",
                                    lambda *_a, **_k: pytest.fail("derived inputs were hashed"))
            with study._native_inputs.active():
                derived = study_preflight.preflight_study(study, {
                    "series_policy": "use_available", "minimum_history": 1,
                    "metrics": ["rmse"], "metric": "rmse", "selection_boundaries": [],
                }, output_dir=tmp_path / "preflight")
                _, derived_record = benchmark.verify_benchmark(
                    derived.study_path, derived.study_sha256,
                )
                assert derived_record["snapshot_mode"] == mode
                assert derived_record["parent"]["study_id"] == study.study_id
                assert not list(derived.study_path.parent.rglob("*.parquet"))
                assert derived_record["artifacts"] == record["artifacts"]
            captures_before_reopen = len(references)
            with load_study(derived.study_id, registry_root=tmp_path / "benchmarks",
                            spark=session, scratch_root=tmp_path,
                            connections={"archive": postgres_store}) as reopened:
                with reopened._native_inputs.active():
                    checked = study_preflight.preflight_study(reopened, {
                        "series_policy": "use_available", "minimum_history": 1,
                        "metrics": ["rmse"], "metric": "rmse", "selection_boundaries": [],
                    }, output_dir=tmp_path / "repeat-preflight")
                    assert checked.study_id == derived.study_id
            assert len(references) == captures_before_reopen
            selected = study._native_inputs.materialize(record["artifacts"]["selection"])
        assert not selected.exists()
        with pytest.raises(ValueError, match="closed"):
            _ = study.training_data
    finally:
        scope.close()
        for reference in references:
            postgres_store.release(reference)
        session.stop()


@pytest.mark.parametrize("mode", ["table", "table+hash"])
@pytest.mark.parametrize("backend", ["postgres", "bigquery"])
def test_native_preserves_nested_admission_keys_and_null_structs(
    request, backend, tmp_path, mode,
):
    store = request.getfixturevalue(backend + "_store")
    # Null keys, keys containing nulls, and actual keys remain distinct.
    schema = pa.schema([
        ("keys", pa.struct([("item", pa.string()), ("amount", pa.decimal128(25, 12))])),
        ("issue", pa.string()),
    ])
    rows = [
        {"keys": None, "issue": "global"},
        {"keys": {"item": None, "amount": None}, "issue": "null key"},
        {"keys": {"item": "A", "amount": Decimal("1234567890123.000000000001")},
         "issue": "example"},
    ]
    original = pa.Table.from_pylist(rows, schema=schema)
    source = tmp_path / "findings.parquet"
    pq.write_table(original, source)
    manager = RetainedInputs(mode=mode, connection="archive", store=store,
                             scratch_root=tmp_path)
    reference = manager.capture(source)
    try:
        with manager.open(reference) as path:
            actual = pq.read_table(path)
            assert actual.schema == original.schema
            assert sorted(actual.to_pylist(), key=repr) == sorted(rows, key=repr)
    finally:
        store.release(reference.record())


@pytest.mark.parametrize("published", [False, True])
def test_native_capture_lifecycle_preserves_retained_inputs_for_recovery(
    postgres_store, typed_input, tmp_path, published,
):
    from autoforecast.snapshots import CaptureAttempt, current_capture, managed_path

    source, expected = typed_input
    second = tmp_path / "temporary.parquet"
    pq.write_table(expected.slice(0, 1), second)
    manager = RetainedInputs(mode="table", connection="archive", store=postgres_store,
                             scratch_root=tmp_path)
    attempt = CaptureAttempt(manager, spark=None)
    kept = temporary = None
    try:
        with attempt.active():
            assert current_capture() is attempt
            kept = attempt.bind(source)["snapshot"]
            temporary = attempt.bind(second)["snapshot"]
            attempt.retain(kept)
            source.unlink()
            assert pq.read_table(managed_path(source)).num_rows == expected.num_rows
            if published:
                attempt.published()
        assert current_capture() is None
        with pytest.raises(FileNotFoundError):
            postgres_store.inspect(temporary)
        # Retention is one-way. A failed later publication leaves a protected
        # orphan for an authorized recovery process, never ordinary cleanup.
        postgres_store.inspect(kept, retained=True)
    finally:
        for reference in (kept, temporary):
            if reference is not None:
                postgres_store.release(reference)


@pytest.mark.parametrize("mode", ["table", "table+hash"])
def test_postgres_experiments_keep_old_values_after_source_corrections(
    postgres_store, tmp_path, mode,
):
    from psycopg import sql

    from autoforecast.integrations.postgres import read_sources

    # Experiment 1 has one product. Experiment 2 has the corrected product plus a new one.
    name = "af_source_" + uuid.uuid4().hex
    identifier = sql.Identifier(postgres_store.test_source_schema, name)
    writer = postgres_store.writer
    manager = RetainedInputs(mode=mode, connection="archive", store=postgres_store,
                             scratch_root=tmp_path)
    captures = []
    first_rows = [{"item": "A", "day": dt.date(2026, 11, 2), "value": 100.0}]
    second_rows = [{"item": "A", "day": dt.date(2026, 11, 2), "value": 120.0},
                   {"item": "B", "day": dt.date(2026, 11, 3), "value": 50.0}]

    def capture(index):
        result = read_sources(
            writer, queries={"primary": "SELECT item, day, value FROM observations"},
            tables={"observations": postgres_store.test_source_schema + "." + name},
            output_root=tmp_path / str(index),
        )
        reference = manager.capture(result["primary"][0])
        captures.append(reference)
        return reference

    def check_saved():
        for reference, expected in zip(captures, (first_rows, second_rows), strict=True):
            with manager.open(reference) as saved:
                assert sorted(pq.read_table(saved).to_pylist(), key=repr) == expected

    try:
        writer.execute(sql.SQL("CREATE TABLE {} (item TEXT, day DATE, value DOUBLE PRECISION)")
                       .format(identifier))
        writer.execute(sql.SQL("INSERT INTO {} VALUES ('A', '2026-11-02', 100)").format(identifier))
        first = capture(1)
        writer.execute(sql.SQL("UPDATE {} SET value=120 WHERE item='A'").format(identifier))
        writer.execute(sql.SQL("INSERT INTO {} VALUES ('B', '2026-11-03', 50)").format(identifier))
        second = capture(2)
        assert first.record()["capture_id"] != second.record()["capture_id"]
        # Filtering a business date sees today's corrected value, not its old version.
        assert writer.execute(sql.SQL("SELECT value FROM {} WHERE day <= '2026-11-02'")
                              .format(identifier)).fetchone()[0] == 120
        check_saved()
        writer.execute(sql.SQL("DELETE FROM {} WHERE item='A'").format(identifier))
        check_saved()
        writer.execute(sql.SQL("TRUNCATE {}").format(identifier))
        check_saved()
        writer.execute(sql.SQL("DROP TABLE {}").format(identifier))
        writer.execute(sql.SQL("CREATE TABLE {} (replacement TEXT)").format(identifier))
        check_saved()
    finally:
        writer.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(identifier))
        for reference in captures:
            postgres_store.release(reference.record())


@pytest.mark.parametrize("actor", ["writer", "reader", "source"])
def test_postgres_ordinary_accounts_cannot_change_retained_inputs(
    postgres_store, typed_input, tmp_path, actor, monkeypatch,
):
    from psycopg import errors, sql

    from autoforecast import _logical_hash
    from autoforecast.integrations.postgres import PostgresSnapshots

    store = postgres_store
    # This public application instance receives no administrator or release credentials.
    application = PostgresSnapshots(store.writer, store.reader, schema=store.schema)
    monkeypatch.setattr(_logical_hash, "fingerprint", lambda *a, **k: pytest.fail(
        "table-only protection must not hash rows"))
    manager = RetainedInputs(mode="table", connection="archive", store=application,
                             scratch_root=tmp_path)
    reference = manager.capture(typed_input[0])
    record = reference.record()
    application.retain(record)
    client = store.test_clients[actor]
    target = store._name(record)
    statements = [
        sql.SQL("UPDATE {} SET value=0").format(target),
        sql.SQL("INSERT INTO {} SELECT * FROM {}").format(target, target),
        sql.SQL("DELETE FROM {}").format(target),
        sql.SQL("TRUNCATE {}").format(target),
        sql.SQL("DROP TABLE {}").format(target),
        sql.SQL("ALTER TABLE {} RENAME TO replaced").format(target),
        sql.SQL("ALTER TABLE {} OWNER TO {}").format(
            target, sql.Identifier(store.test_roles[actor]),
        ),
        sql.SQL("COMMENT ON TABLE {} IS 'changed'").format(target),
        sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(store.schema)),
        sql.SQL("SET ROLE {}").format(sql.Identifier(store.test_roles["owner"])),
        sql.SQL("SET ROLE {}").format(sql.Identifier(store.test_roles["release"])),
        sql.SQL("UPDATE {} SET state='sealed'").format(sql.Identifier(store.schema, "af_captures")),
    ]
    try:
        for statement in statements:
            with pytest.raises(errors.InsufficientPrivilege):
                client.execute(statement)
        # A denied GRANT can be a warning with no effect; check the resulting privilege.
        try:
            client.execute(sql.SQL("GRANT UPDATE ON {} TO {}").format(
                target, sql.Identifier(store.test_roles[actor]),
            ))
        except errors.InsufficientPrivilege:
            pass
        assert not store.test_admin.execute(
            "SELECT has_table_privilege(%s,%s,'UPDATE')",
            (store.test_roles[actor], record["object"]),
        ).fetchone()[0]
        with pytest.raises(errors.InsufficientPrivilege):
            application.drop_unretained(record)
        with pytest.raises(ValueError, match="separate release credentials"):
            application.release(record)
        with manager.open(reference) as retained:
            actual = pq.read_table(retained)
            assert sorted(actual.to_pylist(), key=repr) == sorted(
                typed_input[1].to_pylist(), key=repr,
            )
        application.inspect(record, retained=True)
    finally:
        store.release(record)


def test_postgres_failed_seal_rolls_back_data_and_creation_record(
    postgres_store, typed_input, monkeypatch,
):
    from psycopg import sql

    store = postgres_store
    capture_id = uuid.uuid4().hex
    operation = store._operation

    def stop_before_seal(action, *args, **kwargs):
        if action == "seal":
            raise RuntimeError("injected failure before seal")
        return operation(action, *args, **kwargs)

    monkeypatch.setattr(store, "_operation", stop_before_seal)
    with pytest.raises(RuntimeError, match="injected failure"):
        store.capture(typed_input[0], capture_id=capture_id)
    assert store.reader.execute("SELECT to_regclass(%s)",
                                (store.schema + ".af_input_" + capture_id,)).fetchone() == (None,)
    assert not store.reader.execute(sql.SQL("SELECT 1 FROM {} WHERE capture_id=%s").format(
        sql.Identifier(store.schema, "af_captures"),
    ), (capture_id,)).fetchall()


@pytest.mark.parametrize("drift", ["owner_member", "column_write", "write_all", "release_member",
                                    "public_execute", "function_body"])
def test_postgres_replay_refuses_protection_drift(postgres_store, typed_input, drift):
    from psycopg import sql

    store = postgres_store
    record = store.capture(typed_input[0], capture_id=uuid.uuid4().hex)
    admin, roles = store.test_admin, store.test_roles
    if drift == "owner_member":
        command = sql.SQL("GRANT {} TO {}").format(
            sql.Identifier(roles["owner"]), sql.Identifier(roles["source"]),
        )
    elif drift == "column_write":
        command = sql.SQL("GRANT UPDATE(value) ON {} TO {}").format(
            store._name(record), sql.Identifier(roles["source"]),
        )
    elif drift == "write_all":
        command = sql.SQL("GRANT pg_write_all_data TO {}").format(sql.Identifier(roles["source"]))
    elif drift == "release_member":
        command = sql.SQL("GRANT {} TO {}").format(
            sql.Identifier(roles["release"]), sql.Identifier(roles["writer"]),
        )
    elif drift == "public_execute":
        command = sql.SQL("GRANT EXECUTE ON FUNCTION {}(text,text,jsonb) TO PUBLIC").format(
            sql.Identifier(store.schema, "af_snapshot"),
        )
    else:
        command = sql.SQL("""CREATE OR REPLACE FUNCTION {}(operation text,capture_id text,
            details jsonb DEFAULT '{{}}'::jsonb) RETURNS void LANGUAGE plpgsql SECURITY DEFINER
            SET search_path=pg_catalog,pg_temp AS 'BEGIN RETURN; END'""").format(
                sql.Identifier(store.schema, "af_snapshot"),
            )
    admin.execute(command)
    with pytest.raises(ValueError):
        store.inspect(record)
    # Fixture cleanup uses the trusted administrator, not the corrupted application policy.


def test_postgres_resolver_loads_release_credentials_only_for_release(
    postgres_store, typed_input, monkeypatch,
):
    from psycopg.conninfo import make_conninfo

    from autoforecast.snapshots import Connections

    dsn = os.environ["AUTOFORECAST_TEST_INPUT_POSTGRES_DSN"]
    roles = postgres_store.test_roles
    variables = {kind: "AUTOFORECAST_UNIT_" + kind.upper() + "_DSN"
                 for kind in ("writer", "reader", "release")}
    for kind in ("writer", "reader"):
        monkeypatch.setenv(variables[kind], make_conninfo(
            dsn, user=roles[kind], password="local-test-reader-only",
        ))
    monkeypatch.delenv(variables["release"], raising=False)
    settings = {"archive": {"kind": "postgresql", "dsn_env": variables["reader"],
                            "writer_dsn_env": variables["writer"],
                            "release_dsn_env": variables["release"]}}
    destination = {"connection": "archive", "namespace": postgres_store.schema}
    connections = Connections(settings)
    record = None
    try:
        application = connections.snapshot_store(destination)
        assert application.releaser is None
        record = application.capture(typed_input[0], capture_id=uuid.uuid4().hex)
        application.retain(record)
        with pytest.raises(ValueError, match="release_dsn_env"):
            connections.snapshot_store(destination, release=True)
        application.inspect(record, retained=True)
        monkeypatch.setenv(variables["release"], make_conninfo(
            dsn, user=roles["release"], password="local-test-reader-only",
        ))
        cleanup = connections.snapshot_store(destination, release=True)
        assert cleanup.writer is None
        cleanup.release(record)
        cleanup.release(record)
        with pytest.raises(FileNotFoundError):
            application.inspect(record)
    finally:
        connections.close()
        if record is not None:
            postgres_store.release(record)


def test_postgres_provisioning_preserves_existing_archive_and_callers_role(postgres_store):
    from psycopg import errors, sql

    from autoforecast.integrations.postgres import provision_snapshot_schema

    store, roles = postgres_store, postgres_store.test_roles
    admin = store.test_admin
    kwargs = dict(owner=roles["owner"], registration_role=roles["writer"],
                  reader_role=roles["reader"], release_role=roles["release"])
    with pytest.raises(errors.DuplicateSchema):
        provision_snapshot_schema(admin, schema=store.schema, **kwargs)
    store.preflight()
    namespace = "af_provision_" + uuid.uuid4().hex
    database = admin.execute("SELECT current_database()").fetchone()[0]
    admin.execute(sql.SQL("GRANT CREATE ON DATABASE {} TO {}").format(
        sql.Identifier(database), sql.Identifier(roles["owner"]),
    ))
    try:
        admin.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(roles["owner"])))
        provision_snapshot_schema(admin, schema=namespace, **kwargs)
        assert admin.execute("SELECT current_user").fetchone()[0] == roles["owner"]
    finally:
        admin.execute("RESET ROLE")
        admin.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(namespace)))
        admin.execute(sql.SQL("REVOKE CREATE ON DATABASE {} FROM {}").format(
            sql.Identifier(database), sql.Identifier(roles["owner"]),
        ))


@pytest.mark.parametrize("placement", ["top", "storage"])
@pytest.mark.parametrize("mode", ["hash", "table", "table+hash"])
def test_configured_registry_keeps_the_selected_input_mode(tmp_path, placement, mode):
    import yaml

    from autoforecast.study_registration import StudyRegistry, validate_study_config
    from autoforecast.test_study_registration import configuration

    path = configuration(tmp_path)
    value = yaml.safe_load(path.read_text())
    container = value if placement == "top" else value["storage"]
    container["snapshot_mode"] = mode
    path.write_text(yaml.safe_dump(value))
    assert validate_study_config(path).input_storage["snapshot_mode"] == mode
    # Constructing a registry must not issue Spark or database queries.
    with StudyRegistry.from_config(path, spark=object()) as registry:
        assert registry.snapshot_mode == mode
    container["snapshot_mode"] = "automatic"
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="snapshot_mode"):
        validate_study_config(path)


@pytest.mark.parametrize("mode", ["table", "table+hash"])
def test_postgres_direct_stages_run_builtin_model_and_reopen_in_a_worker(
    postgres_store, tmp_path, mode, monkeypatch,
):
    import json
    import subprocess
    import sys

    import yaml
    from psycopg.conninfo import make_conninfo
    from pyspark.sql import SparkSession

    from autoforecast import _logical_hash, register_study, run_study
    from autoforecast.study_registration import StudyRegistry
    from autoforecast.test_explicit_study import stages

    # Population: two synthetic series, 17 observations; all series are retained.
    session = (SparkSession.builder.master("local[2]").appName("native-direct-stages")
               .config("spark.sql.shuffle.partitions", "2")
               .config("spark.ui.enabled", "false").getOrCreate())
    session.sparkContext.setLogLevel("ERROR")
    registry = StudyRegistry(tmp_path / "registry", spark=session,
                             connections={"archive": postgres_store}, snapshots={
                                 "connection": "archive", "namespace": postgres_store.schema,
                             })
    if mode == "table":
        monkeypatch.setattr(_logical_hash, "fingerprint", lambda *a, **k: pytest.fail(
            "table-only direct inputs must not hash rows"))
    study_id = None
    try:
        study_id = register_study(registry=registry, snapshot_mode=mode,
                                  feature_roles={"season_length": 1}, **stages(session))
        definition, _ = registry.definition(study_id)
        assert all("snapshot" in value and "logical_hash" not in value
                   for value in definition["source"]["stages"].values())
        with registry.load_study(study_id) as saved_study:
            _assert_original_stage_values(saved_study, [
                {"item": key, "day": dt.date(2024, 1, day), "value": float(day)}
                for key in ("A", "B") for day in range(1 if key == "A" else 2, 10)
            ])
        first = run_study(study_id, "seasonal_naive", registry=registry, publish=False)
        assert first.testing_metrics["mae"] == 1.5
        assert first.validation_metrics["mae"] == 1.5
        session.stop()
        settings = tmp_path / "worker.yaml"
        settings.write_text(yaml.safe_dump({
            "storage": {"root": str(registry.root)},
            "connections": {"archive": {"kind": "postgresql", "dsn_env": "AF_REPLAY_DSN",
                                         "writer_dsn_env": "AF_CAPTURE_DSN"}},
        }))
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith("AUTOFORECAST_")}
        environment["AUTOFORECAST_CONFIG"] = str(settings)
        for variable, role in (("AF_REPLAY_DSN", "reader"), ("AF_CAPTURE_DSN", "writer")):
            environment[variable] = make_conninfo(
                os.environ["AUTOFORECAST_TEST_INPUT_POSTGRES_DSN"],
                user=postgres_store.test_roles[role],
                password="local-test-reader-only",
            )
        script = '''
import json, sys
from pyspark.sql import SparkSession
from autoforecast import run_study
from autoforecast import _logical_hash
spark=(SparkSession.builder.master("local[2]").config("spark.ui.enabled","false")
       .config("spark.sql.shuffle.partitions","2").getOrCreate())
if sys.argv[2]=='table':
    def forbidden(*args, **kwargs):
        raise AssertionError("worker hashed native input rows")
    _logical_hash.fingerprint=forbidden
try:
    from autoforecast.study_workflow import load_forecast_workflow
    from autoforecast.snapshots import Connections
    closed=[]
    original_close=Connections.close
    def tracked_close(self):
        owned=list(self._owned)
        original_close(self)
        closed.extend(owned)
    Connections.close=tracked_close
    from autoforecast.study_registration import StudyRegistry
    from autoforecast.integrations.test_snapshots import _assert_original_stage_values
    import datetime as dt
    with StudyRegistry.from_environment(spark=spark) as registry:
        with registry.load_study(sys.argv[1]) as study:
            _assert_original_stage_values(study, [
                {"item": key, "day": dt.date(2024, 1, day), "value": float(day)}
                for key in ("A", "B") for day in range(1 if key == "A" else 2, 10)
            ])
    result=run_study(sys.argv[1],"seasonal_naive",publish=False)
    assert closed and all(client.closed for client in closed)
    replay=load_forecast_workflow(result.path)
    assert replay.testing_metrics["mae"]==1.5
    assert all(client.closed for client in closed)
    print("RESULT="+json.dumps(result.summary()))
finally:
    spark.stop()
'''
        completed = subprocess.run([sys.executable, "-c", script, study_id, mode], env=environment,
                                   capture_output=True, text=True, timeout=180)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        result = json.loads(next(line.removeprefix("RESULT=")
                                 for line in completed.stdout.splitlines()
                                 if line.startswith("RESULT=")))
        assert result["study_id"] == study_id
        assert result["testing"]["mae"] == 1.5
    finally:
        session.stop()
        if study_id is not None:
            registry.release(study_id)


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("supplied", [False, True])
def test_registration_closes_only_its_own_registry(tmp_path, monkeypatch, failure, supplied):
    from autoforecast import study_registration as registration
    from autoforecast.test_study_registration import configuration

    config = configuration(tmp_path / "input")
    registry = registration.StudyRegistry(tmp_path / "registry", spark=object())
    closed = []
    monkeypatch.setattr(registry, "close", lambda: closed.append(True))
    monkeypatch.setattr(registration.StudyRegistry, "_from_storage",
                        lambda *args, **kwargs: registry)

    def perform(*args, **kwargs):
        if failure:
            raise ValueError("registration failed")
        return "registered"

    monkeypatch.setattr(registration, "_register_study", perform)
    if failure:
        with pytest.raises(ValueError, match="registration failed"):
            registration.register_study(config, registry=registry if supplied else None)
    else:
        assert registration.register_study(
            config, registry=registry if supplied else None,
        ) == "registered"
    assert closed == ([] if supplied else [True])


@pytest.mark.parametrize("supplied", [False, True])
@pytest.mark.parametrize("close_failure", [False, True])
def test_run_study_closes_owned_registry_after_invalid_settings(
    tmp_path, monkeypatch, supplied, close_failure,
):
    from autoforecast import run_study
    from autoforecast.study_registration import StudyRegistry

    registry = StudyRegistry(tmp_path / "registry", spark=object())
    closed = []

    def close():
        closed.append(True)
        if close_failure:
            raise OSError("connection close failed")

    monkeypatch.setattr(registry, "close", close)
    monkeypatch.setattr(StudyRegistry, "from_environment", lambda: registry)
    with pytest.raises(ValueError, match="tracking must be"):
        run_study("a" * 64, "seasonal_naive", tracking="invalid",
                  registry=registry if supplied else None)
    assert closed == ([] if supplied else [True])


def test_connections_close_all_owned_clients_after_failure():
    from autoforecast.snapshots import Connections

    closed = []

    class Client:
        def __init__(self, name, failure=False):
            self.name, self.failure = name, failure

        def close(self):
            closed.append(self.name)
            if self.failure:
                raise OSError("close failed")

    supplied = Client("supplied")
    connections = Connections(supplied={"external": supplied})
    connections._owned.extend([Client("first"), Client("last", failure=True)])
    connections._clients["owned"] = connections._owned[0]
    with pytest.raises(OSError, match="close failed"):
        connections.close()
    assert closed == ["last", "first"]
    assert not connections._clients
    connections.close()
    assert closed == ["last", "first"]


@pytest.mark.parametrize("mode", ["table", "table+hash"])
@pytest.mark.parametrize("execution", ["sequential", "local"])
def test_postgres_native_tuning_and_campaign_replay(
    postgres_store, tmp_path, mode, monkeypatch, execution,
):
    import json

    from psycopg.conninfo import make_conninfo
    from pyspark.sql import SparkSession

    from autoforecast import _logical_hash
    from autoforecast.results import read_tuning_results
    from autoforecast.snapshots import Connections, NativeReadScope
    from autoforecast.study import load_study
    from autoforecast.study_campaign import StudyModel
    from autoforecast.test_study_tuning import config, freeze, implementation, predict

    # Two synthetic series, 18 rows, no exclusions, one two-day tuning window.
    session = (SparkSession.builder.master("local[2]").appName("native-search-campaign")
               .config("spark.sql.shuffle.partitions", "2")
               .config("spark.ui.enabled", "false").getOrCreate())
    session.sparkContext.setLogLevel("ERROR")
    captured = []
    original_capture = postgres_store.capture

    def capture(path, *, capture_id, checkpoint=None):
        record = original_capture(path, capture_id=capture_id, checkpoint=checkpoint)
        captured.append(record)
        return record

    monkeypatch.setattr(postgres_store, "capture", capture)
    if mode == "table":
        monkeypatch.setattr(_logical_hash, "fingerprint", lambda *a, **k: pytest.fail(
            "native tuning/campaign hashed input rows"))
    manager = RetainedInputs(mode=mode, connection="archive", store=postgres_store,
                             scratch_root=tmp_path)
    scope = NativeReadScope(Connections(supplied={"archive": postgres_store}),
                            scratch_root=tmp_path, spark=session)
    # Worker processes receive aliases and environment credential names only.
    worker_config = tmp_path / "worker.json"
    worker_config.write_text(json.dumps({"connections": {"archive": {
        "kind": "postgresql", "dsn_env": "AF_NATIVE_SEARCH_READER_DSN",
        "writer_dsn_env": "AF_NATIVE_SEARCH_WRITER_DSN",
    }}}))
    monkeypatch.setenv("AUTOFORECAST_CONFIG", str(worker_config))
    monkeypatch.setenv("AF_NATIVE_SEARCH_READER_DSN", make_conninfo(
        os.environ["AUTOFORECAST_TEST_INPUT_POSTGRES_DSN"],
        user=postgres_store.test_roles["reader"], password="local-test-reader-only",
    ))
    # Fitting seals private training windows under the same protected owner.
    # This role can create a new window, but cannot change retained study rows.
    monkeypatch.setenv("AF_NATIVE_SEARCH_WRITER_DSN", make_conninfo(
        os.environ["AUTOFORECAST_TEST_INPUT_POSTGRES_DSN"],
        user=postgres_store.test_roles["writer"], password="local-test-reader-only",
    ))
    try:
        with scope.active():
            manifest = freeze(tmp_path / "frozen", session, native_inputs=manager)
        with load_study(manifest.parent.name, registry_root=manifest.parent.parent,
                        spark=session, scratch_root=tmp_path,
                        connections={"archive": postgres_store}) as study:
            profile = config(tmp_path, budget=2)
            tuned = study.tune(fit_predict=predict, config=profile, implementation=implementation(),
                               output_dir=tmp_path / "search")
            rows = read_tuning_results(tuned.path, connections={"archive": postgres_store},
                                       scratch_root=tmp_path)
            assert len(rows.rows) == 8
            assert set(rows.rows["prediction"].to_pylist()) == {0.0, 10.0}
            assert set(rows.rows["target_date"].to_pylist()) == {
                dt.date(2024, 1, 4), dt.date(2024, 1, 5),
            }
            assert tuned.parameters == {"model.level": 0.0, "training.offset": 0}
            assert tuned.tuning_score == pytest.approx((20.5) ** 0.5)
            model = StudyModel("autoforecast.test_study_tuning:predict", implementation(), profile,
                               fixed_parameters={"model.level": 0.0}, memory_gib=2.0)
            options = dict(models=[model], execution=execution, output_dir=tmp_path / "campaign",
                           final_evaluation="all_models", timeout_seconds=300)
            result = study.run_campaign(**options)
            assert result["state"]["status"] == "completed", result
            summary = result["results"]
            assert summary["selected_model"] == "custom-constant"
            assert set(summary["results"]) == {"0"}
            assert len(summary["candidates"]) == 1
            candidate = summary["candidates"][0]
            assert candidate["status"] == "completed"
            assert candidate["validation"]["rmse"] == pytest.approx(42.5 ** 0.5)
            assert candidate["testing"]["rmse"] == pytest.approx(72.5 ** 0.5)
            from autoforecast.results import read_workflow_results

            saved = read_workflow_results(
                summary["results"]["0"]["workflow_path"],
                connections={"archive": postgres_store}, scratch_root=tmp_path,
            )
            assert len(saved.rows) == 8
            assert set(saved.rows["prediction"].to_pylist()) == {0.0}
            assert sorted(saved.rows["target_date"].to_pylist()) == [
                dt.date(2024, 1, day) for day in (6, 6, 7, 7, 8, 8, 9, 9)
            ]
            repeated = study.run_campaign(**options)
            assert repeated["results"] == result["results"]
    finally:
        scope.close()
        for record in captured:
            postgres_store.release(record)
        session.stop()


@pytest.fixture
def mysql_source():
    dsn = os.environ.get("AUTOFORECAST_TEST_INPUT_MYSQL_DSN")
    if not dsn:
        if os.environ.get("AUTOFORECAST_REQUIRE_INPUT_MYSQL") == "1":
            pytest.fail("requested MySQL gate requires AUTOFORECAST_TEST_INPUT_MYSQL_DSN")
        pytest.skip("MySQL engine unavailable: configure a test DSN or local container")
    from autoforecast.integrations.mysql import connect

    admin = connect(dsn)
    namespace = "af_source_" + uuid.uuid4().hex
    from urllib.parse import urlsplit

    import pymysql

    address = urlsplit(dsn)
    user = "af_reader_" + uuid.uuid4().hex[:12]
    reader = None
    try:
        with admin.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE `{namespace}`")
            cursor.execute(f"CREATE USER '{user}'@'%%' IDENTIFIED BY %s", ("local-test-only",))
            cursor.execute(f"GRANT SELECT ON `{namespace}`.* TO '{user}'@'%'")
        reader = pymysql.connect(host=address.hostname, port=address.port or 3306, user=user,
                                 password="local-test-only", charset="utf8mb4", autocommit=True)
        yield admin, reader, namespace
    finally:
        if reader is not None:
            reader.close()
        with admin.cursor() as cursor:
            cursor.execute(f"DROP DATABASE IF EXISTS `{namespace}`")
            cursor.execute(f"DROP USER IF EXISTS '{user}'@'%'")
            cursor.execute("SELECT SCHEMA_NAME FROM information_schema.SCHEMATA "
                           "WHERE SCHEMA_NAME=%s", (namespace,))
            assert cursor.fetchone() is None
            cursor.execute("SELECT User FROM mysql.user WHERE User=%s", (user,))
            assert cursor.fetchone() is None
        admin.close()


def test_mysql_sources_use_one_read_view_and_preserve_types(mysql_source, tmp_path, monkeypatch):
    import struct

    import pymysql

    from autoforecast.integrations.mysql import read_sources

    admin, reader, namespace = mysql_source
    # Three synthetic IDs per table, six starting rows and three joined outputs.
    with admin.cursor() as cursor:
        cursor.execute(f"CREATE TABLE `{namespace}`.facts (id BIGINT, value DOUBLE, "
                       "amount DECIMAL(25,12), day DATE, at DATETIME(6), small FLOAT) "
                       "ENGINE=InnoDB")
        cursor.execute(f"CREATE TABLE `{namespace}`.labels (id BIGINT, label TEXT) ENGINE=InnoDB")
        cursor.executemany(f"INSERT INTO `{namespace}`.facts VALUES(%s,%s,%s,%s,%s,%s)", [
            (i, value, Decimal("1234567890123.000000000001"), dt.date(2024, 1, 1),
             dt.datetime(2024, 1, 1, 12, 1, 2, 123456), 1.23456789)
            for i, value in ((1, 100.0), (2, 0.0), (3, None))
        ])
        cursor.executemany(f"INSERT INTO `{namespace}`.labels VALUES(%s,%s)",
                           [(1, "old"), (2, "two"), (3, None)])
    changed = []
    fetch = pymysql.cursors.SSCursor.fetchmany

    def change_source(self, *args, **kwargs):
        rows = fetch(self, *args, **kwargs)
        if rows and not changed:
            changed.append(True)
            with admin.cursor() as cursor:
                cursor.execute(f"UPDATE `{namespace}`.facts SET value=999 WHERE id=1")
                cursor.execute(f"UPDATE `{namespace}`.labels SET label='new' WHERE id=1")
        return rows

    monkeypatch.setattr(pymysql.cursors.SSCursor, "fetchmany", change_source)
    query = ("SELECT f.id,f.value,f.amount,f.day,f.at,f.small,l.label "
             "FROM facts f JOIN labels l ON f.id=l.id")
    before = reader.get_autocommit()
    results = read_sources(reader, queries={"primary": query, "audit": query + " WHERE f.id=1"},
                           tables={"facts": f"{namespace}.facts", "labels": f"{namespace}.labels"},
                           output_root=tmp_path / "source")
    assert changed == [True] and reader.get_autocommit() == before
    primary = pq.read_table(results["primary"][0]).sort_by("id")
    audit = pq.read_table(results["audit"][0])
    assert primary.num_rows == 3 and audit.num_rows == 1
    assert primary["value"].to_pylist() == [100.0, 0.0, None]
    assert audit["value"].to_pylist() == [100.0]
    assert audit["label"].to_pylist() == ["old"]
    assert primary["amount"].type == pa.decimal128(25, 12)
    assert primary["amount"].to_pylist() == [Decimal("1234567890123.000000000001")] * 3
    assert primary["at"].to_pylist() == [dt.datetime(2024, 1, 1, 12, 1, 2, 123456)] * 3
    assert primary["small"].type == pa.float32()
    assert [struct.pack("f", value) for value in primary["small"].to_pylist()] == [
        struct.pack("f", 1.23456789),
    ] * 3
    with admin.cursor() as cursor:
        cursor.execute(f"SELECT value FROM `{namespace}`.facts WHERE id=1")
        assert cursor.fetchone() == (999.0,)


@pytest.mark.parametrize("engine", ["MyISAM", "view"])
def test_mysql_refuses_sources_without_supported_transaction_semantics(
    mysql_source, tmp_path, engine,
):
    from autoforecast.integrations.mysql import read_sources

    admin, reader, namespace = mysql_source
    with admin.cursor() as cursor:
        if engine == "view":
            cursor.execute(f"CREATE VIEW `{namespace}`.source AS SELECT 1 AS id")
        else:
            cursor.execute(f"CREATE TABLE `{namespace}`.source (id INT) ENGINE=MyISAM")
    with pytest.raises(ValueError, match="persistent InnoDB"):
        read_sources(reader, queries={"primary": "SELECT * FROM source"},
                     tables={"source": f"{namespace}.source"}, output_root=tmp_path / "source")
    assert not (tmp_path / "source").exists()


@pytest.mark.parametrize("mode", ["hash", "table", "table+hash"])
def test_mysql_source_registers_and_replays_original_values(
    mysql_source, tmp_path, mode, request, monkeypatch,
):
    import yaml
    from pyspark.sql import SparkSession

    from autoforecast import register_study, run_study
    from autoforecast.study_registration import StudyRegistry
    from autoforecast.test_study_registration import configuration

    admin, reader, namespace = mysql_source
    store = request.getfixturevalue("mysql_store") if mode != "hash" else None
    if mode == "table":
        from autoforecast import _logical_hash

        monkeypatch.setattr(_logical_hash, "fingerprint", lambda *a, **k: pytest.fail(
            "table mode must not hash input rows"))
    # The existing fixture has two series, 17 rows, three null observations, no exclusions.
    config = configuration(tmp_path)
    rows = pq.read_table(tmp_path / "source.parquet").to_pylist()
    with admin.cursor() as cursor:
        cursor.execute(f"CREATE TABLE `{namespace}`.observations "
                       "(item VARCHAR(10), day DATE, value DOUBLE) ENGINE=InnoDB")
        cursor.executemany(f"INSERT INTO `{namespace}`.observations VALUES (%s,%s,%s)",
                           [(row["item"], row["day"], row["value"]) for row in rows])
    (tmp_path / "source.sql").write_text("SELECT item,day,value FROM observations")
    value = yaml.safe_load(config.read_text())
    # The actual connection identifies MySQL; the table name does not select a provider.
    value["source"] = {"connection": "source", "sql": "source.sql",
                       "tables": {"observations": namespace + ".observations"}}
    config.write_text(yaml.safe_dump(value))
    session = (SparkSession.builder.master("local[2]").appName("mysql-study-source")
               .config("spark.sql.shuffle.partitions", "2")
               .config("spark.ui.enabled", "false").getOrCreate())
    session.sparkContext.setLogLevel("ERROR")
    try:
        with StudyRegistry(tmp_path / "registry", spark=session,
                           snapshot_mode=mode,
                           connections={"source": reader, **({"archive": store} if store else {})},
                           snapshots=({"connection": "archive", "namespace": store.database}
                                      if store else None)) as registry:
            study_id = register_study(config, registry=registry)
            with admin.cursor() as cursor:
                cursor.execute(f"UPDATE `{namespace}`.observations SET value=999")
            definition, _ = registry.definition(study_id)
            assert definition["source"]["kind"] == "mysql"
            assert definition["source"]["rows"] == 17
            with registry.load_study(study_id) as study:
                assert study.n_series == 2
                assert study.training_data.filter("value=999").count() == 0
                _assert_original_stage_values(study, rows)
            result = run_study(study_id, "seasonal_naive", registry=registry, publish=False)
            assert result.testing_metrics["mae"] == pytest.approx(5 / 3)
            assert not (registry.scratch_root / "capture-journal").exists()
            if store:
                registry.release(study_id)
        assert reader.open
    finally:
        session.stop()


@pytest.mark.parametrize("failure", ["before_create", "before_commit", "lost_acknowledgement"])
def test_capture_journal_preserves_original_identity_across_interruption(
    postgres_store, typed_input, tmp_path, monkeypatch, failure,
):
    import json

    from mlops import _safe_files
    from psycopg import sql
    from psycopg.errors import InsufficientPrivilege

    journal = tmp_path / "durable-evidence"
    manager = RetainedInputs(mode="table", connection="archive", store=postgres_store,
                             scratch_root=tmp_path, journal_root=journal)
    publish = _safe_files.publish_bytes
    original_capture = postgres_store.capture
    identifiers = []

    def fail_write(root, name, payload):
        if (failure == "before_create" and name == "intent.json"
                or failure == "before_commit" and name == "sealed.json"):
            raise OSError("synthetic evidence write failure")
        return publish(root, name, payload)

    def interrupted_capture(path, *, capture_id, checkpoint=None):
        # The intent must already be durable before any database write starts.
        intent = json.loads((journal / capture_id / "intent.json").read_bytes())
        assert intent["capture_id"] == capture_id
        assert intent["namespace"] == postgres_store.schema
        identifiers.append(capture_id)
        record = original_capture(path, capture_id=capture_id, checkpoint=checkpoint)
        if failure == "lost_acknowledgement":
            raise ConnectionError("synthetic lost commit acknowledgement")
        return record

    monkeypatch.setattr(_safe_files, "publish_bytes", fail_write)
    monkeypatch.setattr(postgres_store, "capture", interrupted_capture)
    with pytest.raises((OSError, ConnectionError), match="synthetic"):
        manager.capture(typed_input[0])
    if failure == "before_create":
        assert identifiers == []
        return
    capture_id, = identifiers
    name = postgres_store.schema + ".af_input_" + capture_id
    if failure == "before_commit":
        # Failure to save the identity aborts the creating transaction too.
        found = postgres_store.reader.execute("SELECT to_regclass(%s)", (name,)).fetchone()[0]
        assert found is None
        assert not (journal / capture_id / "sealed.json").exists()
        return
    record = json.loads((journal / capture_id / "sealed.json").read_bytes())
    assert record["object"] == name
    assert not (journal / capture_id / "ready.json").exists()
    try:
        postgres_store.inspect(record)
        with manager.open(SnapshotReference.from_record(record)) as saved:
            assert pq.read_table(saved).equals(typed_input[1])
        # A lost response never restores the ordinary writer's mutation rights.
        with pytest.raises(InsufficientPrivilege):
            postgres_store.writer.execute(sql.SQL("DELETE FROM {}").format(
                sql.Identifier(postgres_store.schema, "af_input_" + capture_id)))
    finally:
        # Recovery uses the originally saved identity and separate release account.
        postgres_store.release(record)
    assert postgres_store.reader.execute("SELECT to_regclass(%s)", (name,)).fetchone()[0] is None


@pytest.mark.parametrize("failure", ["none", "lost_acknowledgement", "changed_identity"])
def test_unpublished_capture_recovery_requires_quiescence_and_original_identity(
    postgres_store, typed_input, tmp_path, monkeypatch, failure,
):
    import json

    from autoforecast.study_registration import StudyRegistry

    registry = StudyRegistry(tmp_path / "registry", spark=object(),
                             connections={"archive": postgres_store})
    manager = RetainedInputs(mode="table", connection="archive", store=postgres_store,
                             scratch_root=tmp_path,
                             journal_root=registry.root / "capture-journal")
    reference = manager.capture(typed_input[0]).record()
    capture_id = reference["capture_id"]
    try:
        # Retain before a simulated publication failure. Recovery must use the
        # separate release account; ordinary registration cannot delete this table.
        postgres_store.retain(reference)
        with pytest.raises(ValueError, match="active owner"):
            registry.recover_capture(capture_id)
        manager.close()
        manager.close()
        with pytest.raises(ValueError, match="closed"):
            manager.capture(typed_input[0])
        if failure == "changed_identity":
            journal = registry.root / "capture-journal" / capture_id / "sealed.json"
            original = journal.read_bytes()
            changed = json.loads(original)
            changed["incarnation"]["seal"] = str(uuid.uuid4())
            journal.write_text(json.dumps(changed))
            with pytest.raises(Exception, match="identity|sealed"):
                registry.recover_capture(capture_id)
            postgres_store.inspect(reference, retained=True)
            journal.write_bytes(original)
            # A conflicting attempted recovery is evidence, not an overwrite target.
            with pytest.raises(ValueError, match="recovery identity changed"):
                registry.recover_capture(capture_id)
            return
        release = postgres_store.release
        calls = []

        def lose_response(record):
            release(record)
            calls.append(record)
            if len(calls) == 1:
                raise ConnectionError("synthetic lost release response")

        if failure == "lost_acknowledgement":
            monkeypatch.setattr(postgres_store, "release", lose_response)
            with pytest.raises(ConnectionError, match="lost release"):
                registry.recover_capture(capture_id)
        assert registry.recover_capture(capture_id) == capture_id
        assert registry.recover_capture(capture_id) == capture_id
        with pytest.raises(FileNotFoundError):
            postgres_store.inspect(reference)
        if failure == "lost_acknowledgement":
            assert len(calls) == 2
    finally:
        manager.close()
        postgres_store.release(reference)
        registry.close()



def test_reader_recovery_refuses_live_owner_and_closes_crashed_process(tmp_path):
    import json
    import subprocess
    import sys

    from autoforecast.snapshots import StudyLease
    from autoforecast.study_registration import StudyRegistry

    study_id, sha = "a" * 64, "b" * 64
    manifest = tmp_path / "benchmarks" / study_id / "benchmark.json"
    registry = StudyRegistry(tmp_path, spark=object())
    reader = StudyLease(manifest, sha)
    try:
        with pytest.raises(ValueError, match="active owner"):
            registry.recover_reader(study_id, reader.identifier)
        assert StudyLease.active(tmp_path, study_id) == [reader.identifier]
    finally:
        reader.close()
    # Exit without normal cleanup, as a killed notebook process would do.
    output = subprocess.check_output([
        sys.executable, "-c",
        "import os,sys; from autoforecast.snapshots import StudyLease; "
        "reader=StudyLease(sys.argv[1],sys.argv[2]); "
        "print(reader.identifier,flush=True); os._exit(0)", str(manifest), sha,
    ], text=True)
    lease_id = output.strip()
    assert StudyLease.active(tmp_path, study_id) == [lease_id]
    assert registry.recover_reader(study_id, lease_id) == lease_id
    assert registry.recover_reader(study_id, lease_id) == lease_id
    assert StudyLease.active(tmp_path, study_id) == []
    marker = tmp_path / "readers" / study_id / (lease_id + ".closed.json")
    saved = json.loads(marker.read_bytes())
    assert saved["manifest_sha256"] == sha
    saved["manifest_sha256"] = "c" * 64
    marker.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="completion record changed"):
        registry.recover_reader(study_id, lease_id)
    registry.close()



def test_failed_capture_cleanup_preserves_the_operation_error(
    postgres_store, typed_input, tmp_path, monkeypatch,
):
    from autoforecast.snapshots import CaptureAttempt
    from autoforecast.study_registration import StudyRegistry

    registry = StudyRegistry(tmp_path / "registry", spark=object(),
                             connections={"archive": postgres_store})
    manager = RetainedInputs(mode="table", connection="archive", store=postgres_store,
                             scratch_root=tmp_path,
                             journal_root=registry.root / "capture-journal")
    attempt = CaptureAttempt(manager, spark=None)
    reference = None

    def failed_cleanup(_record):
        raise ConnectionError("synthetic cleanup failure")

    monkeypatch.setattr(postgres_store, "drop_unretained", failed_cleanup)
    try:
        with pytest.raises(ValueError, match="original preparation failure") as raised:
            with attempt.active():
                reference = attempt.bind(typed_input[0])["snapshot"]
                raise ValueError("original preparation failure")
        assert any("cleanup needs recovery" in note for note in raised.value.__notes__)
        assert reference is not None
        assert registry.recover_capture(reference["capture_id"]) == reference["capture_id"]
    finally:
        if reference is not None:
            postgres_store.release(reference)
        registry.close()



@pytest.mark.parametrize("mode", ["hash", "table", "table+hash"])
def test_reusable_capture_preserves_raw_rows_and_recalculates_child_eligibility(
    postgres_store, tmp_path, mode, monkeypatch,
):
    import json

    import yaml
    from pyspark.sql import SparkSession

    from autoforecast import _logical_hash
    from autoforecast.benchmark import digest
    from autoforecast.study_registration import StudyRegistry, register_study, validate_study_config
    from autoforecast.test_study_registration import configuration

    # Two synthetic series and 17 observations. B begins one day after A.
    # The first study keeps both; an earlier selection boundary must exclude B.
    config = configuration(tmp_path)
    value = yaml.safe_load(config.read_text())
    session = (SparkSession.builder.master("local[2]").appName("reusable-source-capture")
               .config("spark.sql.shuffle.partitions", "2")
               .config("spark.ui.enabled", "false").getOrCreate())
    session.sparkContext.setLogLevel("ERROR")
    registry = StudyRegistry(tmp_path / "registry", spark=session,
                             connections={"archive": postgres_store}, snapshot_mode=mode,
                             snapshots={"connection": "archive",
                                        "namespace": postgres_store.schema})
    if mode == "table":
        monkeypatch.setattr(_logical_hash, "fingerprint", lambda *a, **k: pytest.fail(
            "table-only reusable capture hashed input rows"))
    try:
        capture_id = registry.capture_sources(config)
        captured, _, _ = registry._source_capture(capture_id)
        assert captured["starting_series"] == 2
        assert captured["provenance"]["primary"]["rows"] == 17
        # Overwrite the former source. Child studies must never read it again.
        rows = pq.read_table(tmp_path / "source.parquet").to_pylist()
        pq.write_table(pa.Table.from_pylist([dict(row, value=999.0) for row in rows]),
                       tmp_path / "source.parquet")
        value["source"] = {"kind": "capture", "capture_id": capture_id}
        config.write_text(yaml.safe_dump(value))
        checked = validate_study_config(config)
        assert "source SQL syntax" not in checked.checked
        # Recomputing an edited envelope checksum must not reuse the original ID.
        manifest = registry.root / "captures" / capture_id / "capture.json"
        original = manifest.read_bytes()
        changed = json.loads(original)
        changed["capture"]["scope"]["end"] = "2024-02-01"
        changed["sha256"] = digest(changed["capture"])
        manifest.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="capture identity changed"):
            register_study(config, registry=registry)
        manifest.write_bytes(original)
        value["recipe"]["final_test_horizon"] = 3
        config.write_text(yaml.safe_dump(value))
        with pytest.raises(ValueError, match="exceed the saved source capture scope"):
            register_study(config, registry=registry)
        value["recipe"]["final_test_horizon"] = 2
        config.write_text(yaml.safe_dump(value))
        other_mode = "table" if mode == "hash" else "hash"
        with pytest.raises(ValueError, match="same snapshot mode"):
            register_study(config, registry=registry, snapshot_mode=other_mode)
        from autoforecast import study_registration

        acquire = study_registration._capture_source_inputs
        guarded = []

        def read_while_capture_is_leased(*args, **kwargs):
            with pytest.raises(ValueError, match="active readers"):
                registry.release(capture_id)
            guarded.append(True)
            return acquire(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(study_registration, "_capture_source_inputs",
                          read_while_capture_is_leased)
            first = register_study(config, registry=registry)
        assert guarded == [True]
        with registry.load_study(first) as study:
            assert study.n_series == 2
            assert study.testing_data.count() == 4
            assert study.training_data.filter("value = 999").count() == 0
            _assert_original_stage_values(study, rows)
        value["recipe"]["validation_origins"] = ["2024-01-01"]
        value["recipe"]["new_series_policy"] = "exclude_after_first_origin"
        value["recipe"].pop("tuning_windows")
        config.write_text(yaml.safe_dump(value))
        second = register_study(config, registry=registry)
        assert second != first
        with registry.load_study(second) as study:
            assert study.n_series == 1
            assert study.testing_data.count() == 2
            assert study.training_data.filter("value = 999").count() == 0
            _assert_original_stage_values(
                study, [row for row in rows if row["item"] == "A"],
                windows=(("training", 1, 1), ("validation", 2, 3),
                         ("refit", 1, 7), ("testing", 8, 9)),
            )
        if mode != "hash":
            with pytest.raises(ValueError, match="dependent study"):
                registry.release(capture_id)
        for identifier in (first, second):
            definition, _ = registry.definition(identifier)
            assert definition["source"]["capture_id"] == capture_id
            if mode != "hash":
                assert definition["source"]["snapshot"] == captured["inputs"]["primary"]["snapshot"]
                registry.release(identifier)
                # Child cleanup must not remove the raw capture shared with siblings.
                postgres_store.inspect(captured["inputs"]["primary"]["snapshot"], retained=True)
        assert registry.release(capture_id) == capture_id
        assert registry.release(capture_id) == capture_id
        with pytest.raises(ValueError, match="released"):
            registry._source_capture(capture_id)
        if mode == "hash":
            assert not list((registry.root / "captures" / capture_id).rglob("*.parquet"))
            # Each child owns its files and still opens after the raw capture is gone.
            for identifier, expected_series in ((first, 2), (second, 1)):
                with registry.load_study(identifier) as study:
                    assert study.n_series == expected_series
                    assert study.training_data.filter("value = 999").count() == 0
            with pytest.raises(ValueError, match="released"):
                register_study(config, registry=registry)
            # A fresh, unused portable capture also has an explicit release path.
            value["source"] = {"kind": "parquet", "path": "source.parquet"}
            config.write_text(yaml.safe_dump(value))
            unused = registry.capture_sources(config)
            registry.release(unused)
            assert not list((registry.root / "captures" / unused).rglob("*.parquet"))
    finally:
        registry.close()
        session.stop()


@pytest.mark.parametrize("mode", ["table", "table+hash"])
@pytest.mark.parametrize("supplied", [False, True])
def test_unqualified_native_destination_refuses_capture_and_replay_before_client_access(
    tmp_path, mode, supplied, monkeypatch,
):
    from autoforecast.snapshots import Connections, NativeReadScope, arrow_schema
    from autoforecast.study_registration import StudyRegistry
    from autoforecast.test_study_registration import configuration

    class UnqualifiedStore:
        backend = "bigquery"
        namespace = "example-project.retained"

        def preflight(self):
            pytest.fail("unqualified destination must be refused before provider access")

    settings = {"archive": {"kind": "bigquery", "project": "example-project", "location": "US"}}
    monkeypatch.setattr(Connections, "client", lambda *a, **kw: pytest.fail(
        "unqualified destination must not acquire credentials"))
    with StudyRegistry(tmp_path / "registry", spark=object(),
                       connections={"archive": UnqualifiedStore()} if supplied else {},
                       connection_settings=settings, snapshot_mode=mode,
                       snapshots={"connection": "archive", "namespace": "example-project.retained"}
                       ) as registry:
        assert registry._native_inputs("hash") is None
        with pytest.raises(ValueError, match="live protection gate"):
            registry.capture_sources(configuration(tmp_path))
        record = {
            "schema": "autoforecast.native-input.v1", "mode": mode, "backend": "bigquery",
            "connection": "archive", "capture_id": "a" * 32,
            "object": "example-project.retained.af_input_" + "a" * 32,
            "incarnation": {"created": "2024-01-01T00:00:00Z"},
            "arrow_schema": arrow_schema(pa.schema([("id", pa.int64())])), "rows": 1,
            "retention": "until_released",
        }
        if mode == "table+hash":
            from autoforecast import _logical_hash

            path = tmp_path / "input.parquet"
            pq.write_table(pa.table({"id": [1]}), path)
            record["logical_hash"] = _logical_hash.fingerprint(path)
        with NativeReadScope(registry.connections, scratch_root=tmp_path).active() as scope:
            with pytest.raises(ValueError, match="live protection gate"):
                scope.inspect(record, retained=True)
        assert not (registry.root / "registrations").exists()


@pytest.fixture
def sqlserver_database(request):
    dsn = os.environ.get("AUTOFORECAST_TEST_INPUT_SQLSERVER_DSN")
    if not dsn:
        if os.environ.get("AUTOFORECAST_REQUIRE_INPUT_SQLSERVER") == "1":
            pytest.fail("SQL Server engine gate requires its test DSN")
        pytest.skip("SQL Server engine unavailable: configure a test DSN or local container")
    from mssql_python import connect as open_connection
    from mssql_python import pooling

    from autoforecast.integrations.sqlserver import connect

    pooling(enabled=False)
    name = "af_test_" + uuid.uuid4().hex
    source_login = "af_source_" + uuid.uuid4().hex[:16]
    admin = connect(dsn)
    opened = []
    try:
        with admin.cursor() as cursor:
            cursor.execute("SELECT CAST(SERVERPROPERTY('ProductVersion') AS varchar(30))")
            request.node.user_properties.append(("sqlserver_version", cursor.fetchone()[0]))
            cursor.execute("CREATE DATABASE [" + name + "]")
            cursor.execute("ALTER DATABASE [" + name + "] SET ALLOW_SNAPSHOT_ISOLATION ON")
            cursor.execute("USE master")
            cursor.execute("CREATE LOGIN [" + source_login + "] "
                           "WITH PASSWORD='Local-source-test-only-2026!'")
            cursor.execute("USE [" + name + "]")
            cursor.execute("CREATE USER [" + source_login + "] FOR LOGIN [" + source_login + "]")
            cursor.execute("GRANT SELECT ON SCHEMA::dbo TO [" + source_login + "]")
            cursor.execute("USE master")
        for index in range(2):
            connection = (open_connection(dsn, autocommit=True, UID=source_login,
                                          PWD="Local-source-test-only-2026!", DATABASE=name)
                          if index == 0 else connect(dsn))
            opened.append(connection)
            with connection.cursor() as cursor:
                cursor.execute("USE [" + name + "]")
        yield opened[0], opened[1], admin, name
    finally:
        for connection in opened:
            # The driver may pool closed sessions. Leave the owned database before
            # returning a session to the pool so cleanup can remove it immediately.
            try:
                with connection.cursor() as cursor:
                    cursor.execute("IF @@TRANCOUNT>0 ROLLBACK TRANSACTION")
                    cursor.execute("USE master")
            finally:
                connection.close()
        try:
            with admin.cursor() as cursor:
                cursor.execute("DROP DATABASE IF EXISTS [" + name + "]")
                cursor.execute("SELECT name FROM sys.databases WHERE name=?", (name,))
                assert cursor.fetchone() is None
                cursor.execute("SELECT name FROM sys.server_principals WHERE name=?",
                               (source_login,))
                if cursor.fetchone() is not None:
                    cursor.execute("DROP LOGIN [" + source_login + "]")
                cursor.execute("SELECT name FROM sys.server_principals WHERE name=?",
                               (source_login,))
                assert cursor.fetchone() is None
        finally:
            admin.close()


def test_sqlserver_queries_share_one_read_view_and_preserve_declared_types(
    sqlserver_database, tmp_path, monkeypatch,
):
    from autoforecast.integrations import sqlserver

    source, writer, _, _ = sqlserver_database
    with writer.cursor() as cursor:
        cursor.execute("CREATE TABLE dbo.sales (id int NOT NULL, value float, "
                       "amount decimal(25,12), at datetime2(6), label nvarchar(40), "
                       "flag bit, raw varbinary(8))")
        cursor.execute("CREATE TABLE dbo.prices (id int NOT NULL, value int NOT NULL)")
        for i in (1, 2, 3):
            cursor.execute("INSERT INTO dbo.sales VALUES (?,?,?,?,?,?,?)", (
                i, float(i * 100), Decimal("1234567890123.000000000001") if i != 2 else None,
                dt.datetime(2024, 1, 1, 12, 0, 0, 123456), "雪" if i == 1 else None,
                i == 1, b"\x00\xff" if i == 3 else None,
            ))
            cursor.execute("INSERT INTO dbo.prices VALUES (?,?)", (i, i * 10))
    original = sqlserver._source_schema
    changed = []

    def describe_after_source_update(connection, query):
        if not changed:
            with writer.cursor() as cursor:
                for table in ("sales", "prices"):
                    cursor.execute("UPDATE dbo." + table + " SET value=999")
            changed.append(True)
        return original(connection, query)

    monkeypatch.setattr(sqlserver, "_source_schema", describe_after_source_update)
    result = sqlserver.read_sources(source, queries={
        "primary": "SELECT s.*, p.value AS price FROM sales s JOIN prices p ON s.id=p.id",
        "audit": "SELECT SUM(value) AS total FROM prices",
        "empty": "SELECT id,CAST(NULL AS decimal(10,2)) AS amount FROM sales WHERE id<0",
    }, tables={"sales": "dbo.sales", "prices": "dbo.prices"}, output_root=tmp_path / "capture")
    saved = pq.read_table(result["primary"][0])
    values = sorted(saved.to_pylist(), key=lambda row: row["id"])
    assert [row["value"] for row in values] == [100, 200, 300]
    assert [row["price"] for row in values] == [10, 20, 30]
    assert values[0]["amount"] == Decimal("1234567890123.000000000001")
    assert values[1]["amount"] is None and values[0]["label"] == "雪"
    assert values[0]["at"].microsecond == 123456
    assert values[0]["flag"] is True and values[1]["flag"] is False
    assert values[2]["raw"] == b"\x00\xff"
    assert saved.schema.field("id").type == pa.int32()
    assert pq.read_table(result["audit"][0]).to_pylist() == [{"total": 60}]
    empty = pq.read_table(result["empty"][0])
    assert empty.num_rows == 0 and empty.schema.field("amount").type == pa.decimal128(10, 2)
    assert result["primary"][1]["capture_boundary"] == result["audit"][1]["capture_boundary"]
    with source.cursor() as cursor:
        cursor.execute("SELECT @@TRANCOUNT")
        assert cursor.fetchone()[0] == 0
        cursor.execute("SELECT transaction_isolation_level FROM sys.dm_exec_sessions "
                       "WHERE session_id=@@SPID")
        assert cursor.fetchone()[0] == 2


@pytest.mark.parametrize("failure", ["isolation", "view", "precision", "transaction"])
def test_sqlserver_refuses_unsupported_capture_without_partial_publication(
    sqlserver_database, tmp_path, failure,
):
    from autoforecast.integrations.sqlserver import read_sources

    source, writer, admin, database = sqlserver_database
    with writer.cursor() as cursor:
        cursor.execute("CREATE TABLE dbo.sales (id int, at datetime2(7))")
        cursor.execute("INSERT INTO dbo.sales VALUES (1,'2024-01-01T00:00:00.1234567')")
        cursor.execute("CREATE VIEW dbo.visible_sales AS SELECT * FROM dbo.sales")
    table, query, message = "dbo.sales", "SELECT id FROM sales", ""
    if failure == "isolation":
        with admin.cursor() as cursor:
            cursor.execute("ALTER DATABASE [" + database + "] SET ALLOW_SNAPSHOT_ISOLATION OFF")
        message = "ALLOW_SNAPSHOT_ISOLATION"
    elif failure == "view":
        table, message = "dbo.visible_sales", "base tables"
    elif failure == "precision":
        query, message = "SELECT at FROM sales", "cast explicitly"
    else:
        with source.cursor() as cursor:
            cursor.execute("BEGIN TRANSACTION")
        message = "idle autocommit"
    try:
        with pytest.raises(ValueError, match=message):
            read_sources(source, queries={"primary": query}, tables={"sales": table},
                         output_root=tmp_path / "capture")
        assert not list(tmp_path.rglob("*.parquet"))
        with source.cursor() as cursor:
            cursor.execute("SELECT @@TRANCOUNT")
            assert cursor.fetchone()[0] == int(failure == "transaction")
    finally:
        with source.cursor() as cursor:
            cursor.execute("IF @@TRANCOUNT>0 ROLLBACK TRANSACTION")


@pytest.mark.parametrize("query", [
    "SELECT * FROM sales WITH (NOLOCK)", "SELECT * INTO dbo.copy FROM sales",
    "DELETE FROM sales", "SELECT * FROM sales FOR SYSTEM_TIME AS OF '2024-01-01'",
])
def test_sqlserver_rejects_sql_read_view_overrides_before_connecting(tmp_path, query):
    from autoforecast.integrations.sqlserver import read_sources

    with pytest.raises(ValueError, match="source SQL"):
        read_sources(object(), queries={"primary": query}, tables={"sales": "dbo.sales"},
                     output_root=tmp_path)


@pytest.mark.parametrize("mode,destination", [
    ("hash", None), ("table", "postgres_store"), ("table+hash", "postgres_store"),
    ("table", "sqlserver_store"), ("table+hash", "sqlserver_store"),
])
def test_sqlserver_source_registers_and_replays_original_values(
    sqlserver_database, tmp_path, request, mode, destination, monkeypatch,
):
    import pyarrow.parquet as pq
    import yaml
    from pyspark.sql import SparkSession

    from autoforecast import register_study, run_study
    from autoforecast.study_registration import StudyRegistry
    from autoforecast.test_study_registration import configuration

    source, writer, _, _ = sqlserver_database
    store = request.getfixturevalue(destination) if destination else None
    if destination == "sqlserver_store":
        from autoforecast.integrations.sqlserver import _execute

        login = _execute(source, "SELECT ORIGINAL_LOGIN()")[0][0]
        _execute(store.test_admin, f"CREATE USER [{login}] FOR LOGIN [{login}]")
        _execute(store.test_admin, f"GRANT CONNECT TO [{login}]")
    if mode == "table":
        from autoforecast import _logical_hash

        monkeypatch.setattr(_logical_hash, "fingerprint", lambda *a, **kw: pytest.fail(
            "table-only source capture must not hash dataset rows"))
    # Two declared series, 17 observations and three null values; no series are excluded.
    config = configuration(tmp_path)
    rows = pq.read_table(tmp_path / "source.parquet").to_pylist()
    with writer.cursor() as cursor:
        cursor.execute("CREATE TABLE dbo.observations (item nvarchar(10), day date, value float)")
        cursor.executemany("INSERT INTO dbo.observations VALUES (?,?,?)",
                           [(row["item"], row["day"], row["value"]) for row in rows])
    (tmp_path / "source.sql").write_text("SELECT item,day,value FROM observations")
    value = yaml.safe_load(config.read_text())
    value["source"] = {"connection": "source", "sql": "source.sql",
                       "tables": {"observations": "dbo.observations"}}
    config.write_text(yaml.safe_dump(value))
    session = (SparkSession.builder.master("local[2]").appName("sqlserver-study-source")
               .config("spark.sql.shuffle.partitions", "2")
               .config("spark.ui.enabled", "false").getOrCreate())
    session.sparkContext.setLogLevel("ERROR")
    try:
        with StudyRegistry(
            tmp_path / "registry", spark=session, snapshot_mode=mode,
            connections={"source": source, **({"archive": store} if store is not None else {})},
            snapshots=({"connection": "archive", "namespace": store.schema}
                       if store is not None else None),
        ) as registry:
            study_id = register_study(config, registry=registry)
            with writer.cursor() as cursor:
                cursor.execute("UPDATE dbo.observations SET value=999")
            definition, _ = registry.definition(study_id)
            assert definition["source"]["kind"] == "sqlserver"
            assert definition["source"]["rows"] == 17
            with registry.load_study(study_id) as study:
                assert study.n_series == 2
                assert study.training_data.filter("value=999").count() == 0
                _assert_original_stage_values(study, rows)
            result = run_study(study_id, "seasonal_naive", registry=registry, publish=False)
            assert result.testing_metrics["mae"] == pytest.approx(5 / 3)
            assert not (registry.scratch_root / "capture-journal").exists()
            if store is not None:
                registry.release(study_id)
        assert not source.closed
    finally:
        session.stop()


def test_sqlserver_source_ddl_change_is_not_relabelled_as_the_original_capture(
    sqlserver_database, tmp_path, monkeypatch,
):
    from mssql_python import DatabaseError

    from autoforecast.integrations import sqlserver

    source, writer, _, _ = sqlserver_database
    with writer.cursor() as cursor:
        cursor.execute("CREATE TABLE dbo.sales (id int, value float)")
        cursor.execute("INSERT INTO dbo.sales VALUES (1,100)")
    describe = sqlserver._source_schema

    def after_ddl(connection, query):
        with writer.cursor() as cursor:
            cursor.execute("ALTER TABLE dbo.sales ADD changed int NULL")
        return describe(connection, query)

    monkeypatch.setattr(sqlserver, "_source_schema", after_ddl)
    with pytest.raises((ValueError, DatabaseError), match="(?i)snapshot|changed during capture"):
        sqlserver.read_sources(source, queries={"primary": "SELECT id,value FROM sales"},
                               tables={"sales": "dbo.sales"}, output_root=tmp_path / "capture")
    with source.cursor() as cursor:
        cursor.execute("SELECT @@TRANCOUNT")
        assert cursor.fetchone()[0] == 0


@pytest.fixture
def mysql_store():
    dsn = os.environ.get("AUTOFORECAST_TEST_INPUT_MYSQL_DSN")
    if not dsn:
        if os.environ.get("AUTOFORECAST_REQUIRE_INPUT_MYSQL") == "1":
            pytest.fail("the required MySQL test service has no DSN")
        pytest.skip("MySQL service unavailable: configure a test DSN or local container")
    from contextlib import ExitStack

    import pymysql

    from autoforecast.integrations.mysql import (
        MySQLSnapshots,
        _execute,
        connect,
        provision_snapshot_database,
    )

    suffix = uuid.uuid4().hex[:16]
    database = "af_archive_" + suffix
    roles = {kind: "af_" + kind + "_" + suffix
             for kind in ("owner", "writer", "reader", "release", "source")}
    with connect(dsn) as admin:
        try:
            for kind, role in roles.items():
                _execute(admin, "CREATE USER %s@%s IDENTIFIED BY %s" +
                         (" ACCOUNT LOCK" if kind == "owner" else ""),
                         (role, "%", "local-test-reader-only"))
            provision_snapshot_database(admin, database=database, owner=roles["owner"],
                                        registration_role=roles["writer"],
                                        reader_role=roles["reader"], release_role=roles["release"])
            _execute(admin, f"GRANT SELECT ON `{database}`.* TO '{roles['source']}'@'%'")
            with ExitStack() as stack:
                clients = {kind: stack.enter_context(pymysql.connect(
                    host=admin.host, port=admin.port, user=role, password="local-test-reader-only",
                    database=database, charset="utf8mb4", autocommit=True,
                )) for kind, role in roles.items() if kind != "owner"}
                store = MySQLSnapshots(clients["writer"], clients["reader"], database=database,
                                       releaser=clients["release"])
                store.test_admin, store.test_roles, store.test_clients = admin, roles, clients
                yield store
        finally:
            _execute(admin, f"DROP DATABASE IF EXISTS `{database}`")
            for role in roles.values():
                _execute(admin, "DROP USER IF EXISTS %s@%s", (role, "%"))
            assert not _execute(admin, "SELECT SCHEMA_NAME FROM information_schema.SCHEMATA "
                                "WHERE SCHEMA_NAME=%s", (database,))
            assert not _execute(admin, "SELECT User FROM mysql.user WHERE User IN (" +
                                ",".join(["%s"] * len(roles)) + ")", tuple(roles.values()))


@pytest.mark.parametrize("commit_extra_row", [False, True])
def test_mysql_seal_waits_for_an_open_writer_transaction(
    mysql_store, typed_input, tmp_path, monkeypatch, commit_extra_row,
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from time import monotonic, sleep

    import pymysql

    from autoforecast.integrations.mysql import _execute

    store, (path, expected) = mysql_store, typed_input
    capture_id = uuid.uuid4().hex
    ready, proceed = Event(), Event()
    call = store._call

    def pause_before_seal(operation, *args, **kwargs):
        if operation == "seal":
            ready.set()
            assert proceed.wait(15), "test did not release the sealing operation"
        return call(operation, *args, **kwargs)

    monkeypatch.setattr(store, "_call", pause_before_seal)
    _execute(store.writer, "SET SESSION lock_wait_timeout=15")
    admin = store.test_admin
    writer = pymysql.connect(
        host=admin.host, port=admin.port, database=store.database,
        user=store.test_roles["writer"], password="local-test-reader-only", autocommit=False,
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(store.capture, path, capture_id=capture_id)
        record = None
        try:
            assert ready.wait(15), "capture did not reach sealing"
            # An ordinary registration account holds a real write transaction open.
            _execute(writer, f"INSERT INTO `{store.database}`.`af_build_{capture_id}` "
                     "(id) VALUES (999)")
            proceed.set()
            deadline = monotonic() + 10
            while monotonic() < deadline:
                pending = _execute(admin, "SELECT 1 FROM performance_schema.metadata_locks "
                                   "WHERE OBJECT_SCHEMA=%s AND OBJECT_NAME=%s "
                                   "AND LOCK_STATUS='PENDING'",
                                   (store.database, "af_build_" + capture_id))
                if pending:
                    break
                if future.done():
                    future.result()
                    pytest.fail("sealing finished while the write transaction was open")
                sleep(0.05)
            assert pending, "sealing did not wait on the table's metadata lock"
            assert not future.done()
            if commit_extra_row:
                writer.commit()
                with pytest.raises(ValueError, match="row count changed during transport"):
                    future.result(timeout=15)
                assert not _execute(admin, "SELECT 1 FROM information_schema.TABLES "
                                    "WHERE TABLE_SCHEMA=%s AND TABLE_NAME IN (%s,%s)",
                                    (store.database, "af_build_" + capture_id,
                                     "af_input_" + capture_id))
            else:
                writer.rollback()
                record = future.result(timeout=15)
                destination = tmp_path / "replayed"
                store.read(record, destination)
                assert pq.read_table(destination).equals(expected)
        finally:
            proceed.set()
            writer.rollback()
            writer.close()
            if record is not None:
                store.drop_unretained(record)


@pytest.mark.parametrize("mode", ["table", "table+hash"])
def test_mysql_retains_typed_inputs_without_ordinary_mutation(
    mysql_store, typed_input, tmp_path, mode, monkeypatch,
):
    import pymysql

    from autoforecast import _logical_hash
    from autoforecast.integrations.mysql import _execute

    path, expected = typed_input
    if mode == "table":
        monkeypatch.setattr(_logical_hash, "fingerprint", lambda *a, **k: pytest.fail(
            "table mode must not hash input rows"))
    with RetainedInputs(mode=mode, connection="archive", store=mysql_store,
                        scratch_root=tmp_path) as manager:
        reference = manager.capture(path)
        record = reference.record()
        try:
            mysql_store.retain(record)
            name = mysql_store._name(record)
            for actor in ("source", "writer", "reader"):
                client = mysql_store.test_clients[actor]
                for command in (
                    "UPDATE " + name + " SET value=999", "DELETE FROM " + name,
                    "INSERT INTO " + name + " (id) VALUES (999)", "TRUNCATE TABLE " + name,
                    "DROP TABLE " + name, "ALTER TABLE " + name + " ADD extra INT",
                    "RENAME TABLE " + name + " TO `" + mysql_store.database + "`.stolen",
                    "GRANT UPDATE ON " + name + " TO '" + mysql_store.test_roles[actor] + "'@'%'",
                ):
                    with pytest.raises(pymysql.Error, match="denied"):
                        _execute(client, command)
            with pytest.raises(pymysql.Error, match="not permitted"):
                mysql_store.drop_unretained(record)
            with manager.open(reference) as retained:
                actual = pq.read_table(retained)
                assert actual.schema.equals(expected.schema)
                assert sorted(actual.to_pylist(), key=repr) == sorted(
                    expected.to_pylist(), key=repr,
                )
            mysql_store.inspect(record, retained=True)
        finally:
            mysql_store.release(record)
            mysql_store.release(record)
        with pytest.raises(pymysql.Error, match="not permitted"):
            mysql_store.inspect(record)


@pytest.mark.parametrize("drift", ["table_update", "column_update", "schema_write", "owner_member",
                                    "release_member", "global_role_admin", "procedure_body"])
def test_mysql_replay_refuses_protection_drift(mysql_store, typed_input, drift):
    import pymysql

    from autoforecast.integrations.mysql import _execute

    store = mysql_store
    record = store.capture(typed_input[0], capture_id=uuid.uuid4().hex)
    target = store._name(record)
    actor = "'" + store.test_roles["source"] + "'@'%'"
    command = {
        "table_update": f"GRANT UPDATE ON {target} TO {actor}",
        "column_update": f"GRANT UPDATE(value) ON {target} TO {actor}",
        "schema_write": f"GRANT INSERT ON `{store.database}`.* TO {actor}",
        "owner_member": f"GRANT '{store.test_roles['owner']}'@'%' TO {actor}",
        "release_member": f"GRANT '{store.test_roles['release']}'@'%' TO {actor}",
        "global_role_admin": f"GRANT ROLE_ADMIN ON *.* TO {actor}",
    }
    if drift == "procedure_body":
        _execute(store.test_admin, f"DROP PROCEDURE `{store.database}`.af_snapshot")
        _execute(store.test_admin, f"CREATE PROCEDURE `{store.database}`.af_snapshot() SELECT 1")
    else:
        _execute(store.test_admin, command[drift])
    with pytest.raises((ValueError, pymysql.Error), match="changed"):
        store.inspect(record)
    # The fixture administrator removes the deliberately broken test archive.


@pytest.mark.parametrize("failure", ["before_ddl", "after_begin", "after_seal", "after_rename"])
def test_mysql_interrupted_capture_retains_original_cleanup_identity(
    mysql_store, typed_input, tmp_path, failure, monkeypatch,
):
    import json

    from autoforecast.integrations.mysql import _execute

    store = mysql_store
    call = store._call
    captured = []

    def interrupted(operation, record=None, **kwargs):
        if record is not None:
            captured.append(record)
        if operation == "begin" and failure == "before_ddl":
            raise ConnectionError("synthetic lost operation")
        if operation == "seal" and failure == "after_rename":
            _execute(store.test_admin, f"RENAME TABLE `{store.database}`.af_build_" +
                     record["capture_id"] + " TO " + store._name(record))
            raise ConnectionError("synthetic lost operation")
        result = call(operation, record, **kwargs)
        if ((operation == "begin" and failure == "after_begin")
                or (operation == "seal" and failure == "after_seal")):
            raise ConnectionError("synthetic lost operation")
        return result

    def disconnected(*_args, **_kwargs):
        raise ConnectionError("synthetic disconnected cleanup")

    monkeypatch.setattr(store, "_call", interrupted)
    monkeypatch.setattr(store, "drop_unretained", disconnected)
    with RetainedInputs(mode="table", connection="archive", store=store,
                        scratch_root=tmp_path) as manager:
        with pytest.raises(ConnectionError, match="lost operation"):
            manager.capture(typed_input[0])
    monkeypatch.setattr(store, "_call", call)
    saved = list((tmp_path / "capture-journal").glob("*/sealed.json"))
    assert len(saved) == 1 and captured
    record = json.loads(saved[0].read_bytes())
    assert record["incarnation"] == captured[0]["incarnation"]
    assert not list((tmp_path / "capture-journal").glob("*/ready.json"))
    store.release(record)
    store.release(record)
    assert not _execute(store.reader, "SELECT TABLE_NAME FROM information_schema.TABLES "
                        "WHERE TABLE_SCHEMA=%s AND TABLE_NAME IN (%s,%s)",
                        (store.database, "af_input_" + record["capture_id"],
                         "af_build_" + record["capture_id"]))


def test_mysql_retained_identity_cannot_be_reused_or_replaced(mysql_store, typed_input):
    import pymysql

    from autoforecast.integrations.mysql import _execute

    store = mysql_store
    record = store.capture(typed_input[0], capture_id=uuid.uuid4().hex)
    store.retain(record)
    template = f"`{store.database}`.`replacement_template`"
    _execute(store.test_admin, "CREATE TABLE " + template + " LIKE " + store._name(record))
    store.release(record)
    with pytest.raises(pymysql.Error, match="Duplicate"):
        store.capture(typed_input[0], capture_id=record["capture_id"])
    # A privileged replacement must not be accepted or deleted as the original object.
    _execute(store.test_admin, "CREATE TABLE " + store._name(record) + " LIKE " + template)
    with pytest.raises(pymysql.Error, match="name was reused"):
        store.inspect(record)
    with pytest.raises(pymysql.Error, match="name was reused"):
        store.release(record)
    assert _execute(store.reader, "SELECT * FROM " + store._name(record)) == ()


@pytest.mark.parametrize("store_fixture", ["mysql_store", "sqlserver_store"])
@pytest.mark.parametrize("empty", [False, True])
def test_relational_retains_scalar_bits_and_nested_nulls(request, store_fixture, tmp_path, empty):
    import math
    import struct

    store = request.getfixturevalue(store_fixture)
    schema = pa.schema([
        ("small", pa.float32()), ("negative_zero", pa.float64()), ("flag", pa.bool_()),
        ("unsigned", pa.uint64()),
        ("text", pa.string()), ("bytes", pa.binary()),
        ("at", pa.timestamp("us", tz="Asia/Kolkata")),
        ("nested", pa.struct([("n", pa.int32()), ("flag", pa.bool_())])),
    ])
    table = pa.Table.from_pylist([] if empty else [
        {"small": 1.23456789, "negative_zero": -0.0, "flag": True,
         "unsigned": 2**64 - 1,
         "text": "café 水", "bytes": b"\x00\xff", "nested": None,
         "at": dt.datetime(2024, 1, 1, 12, 1, 2, 123456, tzinfo=dt.UTC)},
        {"small": None, "negative_zero": 0.0, "flag": False,
         "text": None, "bytes": None, "nested": {"n": None, "flag": None}},
    ], schema=schema)
    path = tmp_path / "bits.parquet"
    pq.write_table(table, path)
    with RetainedInputs(mode="table", connection="archive", store=store,
                        scratch_root=tmp_path) as manager:
        reference = manager.capture(path)
        try:
            with manager.open(reference) as result:
                actual = pq.read_table(result)
                assert actual.equals(table)
                if not empty:
                    assert struct.pack("f", actual["small"][0].as_py()) == struct.pack(
                        "f", 1.23456789,
                    )
                    assert math.copysign(1, actual["negative_zero"][0].as_py()) == -1
        finally:
            store.drop_unretained(reference.record())


def test_mysql_resolver_separates_registration_replay_and_release(
    mysql_store, typed_input, tmp_path, monkeypatch,
):
    import pymysql

    from autoforecast.snapshots import Connections, NativeReadScope

    store = mysql_store
    variables = {role: "AF_MYSQL_" + role.upper() + "_DSN"
                 for role in ("writer", "reader", "release")}

    def dsn(role):
        return ("mysql://" + store.test_roles[role] + ":local-test-reader-only@" +
                store.test_admin.host + ":" + str(store.test_admin.port) + "/" + store.database)

    for role in ("writer", "reader"):
        monkeypatch.setenv(variables[role], dsn(role))
    monkeypatch.delenv(variables["release"], raising=False)
    connections = Connections({"archive": {
        "kind": "mysql", "dsn_env": variables["reader"],
        "writer_dsn_env": variables["writer"], "release_dsn_env": variables["release"],
    }})
    destination = {"connection": "archive", "namespace": store.database}
    record = None
    try:
        application = connections.snapshot_store(destination)
        assert application.releaser is None
        with RetainedInputs(mode="table", connection="archive", store=application,
                            scratch_root=tmp_path) as manager:
            reference = manager.capture(typed_input[0])
            record = reference.record()
            application.retain(record)
        with pytest.raises(ValueError, match="release_dsn_env"):
            connections.snapshot_store(destination, release=True)
        scope = NativeReadScope(connections, scratch_root=tmp_path)
        try:
            restored = pq.read_table(scope.materialize(record))
            assert sorted(restored.to_pylist(), key=repr) == sorted(
                typed_input[1].to_pylist(), key=repr,
            )
        finally:
            scope.close()
        monkeypatch.setenv(variables["release"], dsn("release"))
        cleanup = connections.snapshot_store(destination, release=True)
        assert cleanup.writer is None
        cleanup.release(record)
        cleanup.release(record)
        with pytest.raises(pymysql.Error, match="not permitted"):
            application.inspect(record)
    finally:
        connections.close()
        if record is not None:
            store.release(record)


def test_mysql_rejects_privileged_sources_and_lossy_session_settings(mysql_store):

    from autoforecast.integrations.mysql import _execute

    store = mysql_store
    store.preflight()
    store.check_source(store.test_clients["source"])
    for client in (store.test_admin, store.test_clients["release"]):
        with pytest.raises(ValueError, match="source account has archive administration rights"):
            store.check_source(client)
    before = _execute(store.writer, "SELECT @@session.sql_mode")[0][0]
    try:
        _execute(store.writer, "SET SESSION sql_mode='' ")
        with pytest.raises(ValueError, match="strict SQL mode"):
            store.preflight()
    finally:
        _execute(store.writer, "SET SESSION sql_mode=%s", (before,))


@pytest.mark.parametrize("drift", ["replace", "column_type", "nullable", "collation"])
def test_mysql_retained_object_or_schema_changes_fail_closed(mysql_store, typed_input, drift):
    import pymysql

    from autoforecast.integrations.mysql import _execute

    store = mysql_store
    record = store.capture(typed_input[0], capture_id=uuid.uuid4().hex)
    name = store._name(record)
    if drift == "replace":
        template = f"`{store.database}`.`replacement_template`"
        _execute(store.test_admin, "CREATE TABLE " + template + " LIKE " + name)
        _execute(store.test_admin, "DROP TABLE " + name)
        _execute(store.test_admin, "CREATE TABLE " + name + " LIKE " + template)
        _execute(store.test_admin, "ALTER TABLE " + name + " COMMENT=%s",
                 (record["incarnation"]["seal"],))
    elif drift == "column_type":
        _execute(store.test_admin, "ALTER TABLE " + name + " MODIFY value FLOAT")
    elif drift == "nullable":
        _execute(store.test_admin, "ALTER TABLE " + name + " MODIFY id BIGINT NOT NULL")
    else:
        _execute(store.test_admin, "ALTER TABLE " + name +
                 " MODIFY label LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci")
    with pytest.raises((ValueError, pymysql.Error), match="identity changed|columns changed"):
        store.inspect(record)
    if drift == "replace":
        with pytest.raises(pymysql.Error, match="identity changed"):
            store.release(record)


@pytest.fixture
def sqlserver_store():
    dsn = os.environ.get("AUTOFORECAST_TEST_INPUT_SQLSERVER_DSN")
    if not dsn:
        if os.environ.get("AUTOFORECAST_REQUIRE_INPUT_SQLSERVER") == "1":
            pytest.fail("the required SQL Server test service has no DSN")
        pytest.skip("SQL Server engine unavailable: configure a test DSN or local container")
    from contextlib import ExitStack

    from mssql_python import connect, pooling

    from autoforecast.integrations.sqlserver import (
        SQLServerSnapshots,
        _execute,
        provision_snapshot_schema,
    )

    # Fixture-owned logins must have no pooled session left when they are removed.
    pooling(enabled=False)
    suffix = uuid.uuid4().hex[:16]
    database = "af_archive_" + suffix
    roles = {kind: "af_" + kind + "_" + suffix
             for kind in ("owner", "writer", "reader", "release", "source")}
    clients = {}
    with connect(dsn, autocommit=True) as admin:
        try:
            _execute(admin, "USE master")
            _execute(admin, f"CREATE DATABASE [{database}]")
            for kind, name in roles.items():
                if kind != "owner":
                    _execute(admin, f"CREATE LOGIN [{name}] "
                             "WITH PASSWORD='Local-source-test-only-2026!'")
            _execute(admin, f"USE [{database}]")
            for kind, name in roles.items():
                _execute(admin, f"CREATE USER [{name}] " + (
                    "WITHOUT LOGIN" if kind == "owner" else f"FOR LOGIN [{name}]"))
            provision_snapshot_schema(admin, schema="archive", owner=roles["owner"],
                                      registration_role=roles["writer"],
                                      reader_role=roles["reader"], release_role=roles["release"])
            _execute(admin, f"GRANT SELECT ON SCHEMA::archive TO [{roles['source']}]")
            with ExitStack() as stack:
                clients = {kind: stack.enter_context(connect(
                    dsn, autocommit=True, UID=name, PWD="Local-source-test-only-2026!",
                    DATABASE=database,
                )) for kind, name in roles.items() if kind != "owner"}
                store = SQLServerSnapshots(clients["writer"], clients["reader"], schema="archive",
                                           releaser=clients["release"])
                store.test_admin, store.test_clients, store.test_roles = admin, clients, roles
                yield store
        finally:
            _execute(admin, "IF @@TRANCOUNT>0 ROLLBACK")
            _execute(admin, "USE master")
            _execute(admin, f"DROP DATABASE IF EXISTS [{database}]")
            for kind, name in roles.items():
                if kind != "owner" and _execute(admin,
                    "SELECT name FROM sys.server_principals WHERE name=?", (name,),
                ):
                    _execute(admin, f"DROP LOGIN [{name}]")
            assert not _execute(admin, "SELECT name FROM sys.databases WHERE name=?", (database,))
            assert not _execute(admin, "SELECT name FROM sys.server_principals WHERE name IN (" +
                                ",".join(["?"] * len(roles)) + ")", tuple(roles.values()))


@pytest.mark.parametrize("mode", ["table", "table+hash"])
def test_sqlserver_retains_exact_typed_inputs(
    sqlserver_store, typed_input, tmp_path, mode, monkeypatch,
):
    from autoforecast import _logical_hash
    from autoforecast.integrations.sqlserver import _execute

    store = sqlserver_store
    if mode == "table":
        monkeypatch.setattr(_logical_hash, "fingerprint", lambda *a, **k: pytest.fail(
            "table mode must not hash input rows"))
    with RetainedInputs(mode=mode, connection="archive", store=store,
                        scratch_root=tmp_path) as manager:
        reference = manager.capture(typed_input[0])
        record = reference.record()
        try:
            store.retain(record)
            for role in ("source", "writer", "reader"):
                client = store.test_clients[role]
                name = store._name(record)
                for statement in (
                    "INSERT INTO " + name + " (id) VALUES (999)",
                    "UPDATE " + name + " SET value=999", "DELETE FROM " + name,
                    "TRUNCATE TABLE " + name, "DROP TABLE " + name,
                    "ALTER TABLE " + name + " ADD extra INT",
                ):
                    with pytest.raises(Exception, match="denied|permission|Cannot find"):
                        _execute(client, statement)
                _execute(client, "GRANT UPDATE ON OBJECT::" + name +
                         " TO [" + store.test_roles[role] + "]")
                assert _execute(client, "SELECT HAS_PERMS_BY_NAME(?,'OBJECT','UPDATE')",
                                (record["object"],)) == [(0,)]
            with pytest.raises(Exception, match="not permitted"):
                store.drop_unretained(record)
            with manager.open(reference) as retained:
                actual = pq.read_table(retained)
                assert actual.schema.equals(typed_input[1].schema)
                assert sorted(actual.to_pylist(), key=repr) == sorted(
                    typed_input[1].to_pylist(), key=repr,
                )
            store.inspect(record, retained=True)
        finally:
            store.release(record)
            store.release(record)
        with pytest.raises(FileNotFoundError):
            store.inspect(record)


@pytest.mark.parametrize("drift", [
    "table_update", "column_update", "schema_write", "database_write", "owner_impersonate",
    "release_impersonate", "database_role", "server_role", "metadata_visibility",
    "procedure_body", "trustworthy",
])
def test_sqlserver_replay_refuses_protection_drift(sqlserver_store, typed_input, drift):
    from autoforecast.integrations.sqlserver import _execute

    store = sqlserver_store
    record = store.capture(typed_input[0], capture_id=uuid.uuid4().hex)
    name = store._name(record)
    source = "[" + store.test_roles["source"] + "]"
    writer = "[" + store.test_roles["writer"] + "]"
    reader = "[" + store.test_roles["reader"] + "]"
    commands = {
        "table_update": "GRANT UPDATE ON OBJECT::" + name + " TO " + source,
        "column_update": "GRANT UPDATE ON OBJECT::" + name + " (value) TO " + source,
        "schema_write": "GRANT INSERT ON SCHEMA::archive TO " + source,
        "database_write": "GRANT UPDATE TO " + source,
        "owner_impersonate": "GRANT IMPERSONATE ON USER::[" + store.test_roles["owner"] +
                             "] TO " + source,
        "release_impersonate": "GRANT IMPERSONATE ON USER::[" + store.test_roles["release"] +
                               "] TO " + source,
        "database_role": "ALTER ROLE db_datawriter ADD MEMBER " + writer,
        "metadata_visibility": "REVOKE VIEW SECURITY DEFINITION FROM " + reader,
        "procedure_body": "ALTER PROCEDURE archive.af_snapshot AS SELECT 1",
        "trustworthy": "ALTER DATABASE [" + store._database(store.reader)["name"] +
                       "] SET TRUSTWORTHY ON",
    }
    if drift in {"server_role", "metadata_visibility"}:
        database = store._database(store.reader)["name"]
        try:
            _execute(store.test_admin, "USE master")
            if drift == "server_role":
                _execute(store.test_admin, "ALTER SERVER ROLE securityadmin ADD MEMBER " + writer)
            else:
                _execute(store.test_admin, "REVOKE VIEW ANY SECURITY DEFINITION FROM " + reader)
        finally:
            _execute(store.test_admin, "USE [" + database + "]")
    else:
        _execute(store.test_admin, commands[drift])
    with pytest.raises(ValueError, match="changed|roles|visibility|rights|TRUSTWORTHY"):
        store.inspect(record)
    # The fixture administrator removes this deliberately invalid archive.


@pytest.mark.parametrize("failure", ["before_seal", "checkpoint", "commit_ack"])
def test_sqlserver_capture_recovery_preserves_original_identity(
    sqlserver_store, typed_input, tmp_path, failure, monkeypatch,
):
    import json

    from autoforecast.integrations import sqlserver

    store = sqlserver_store
    operation, execute = store._operation, sqlserver._execute
    journal = []
    capture_id = uuid.uuid4().hex

    def interrupted(connection, action, record, **kwargs):
        if action == "seal" and failure == "before_seal":
            raise ConnectionError("synthetic capture interruption")
        return operation(connection, action, record, **kwargs)

    def checkpoint(record):
        journal.append(json.loads(json.dumps(record)))
        if failure == "checkpoint":
            raise ConnectionError("synthetic capture interruption")

    def lost_ack(connection, statement, values=()):
        result = execute(connection, statement, values)
        if connection is store.writer and statement == "COMMIT" and failure == "commit_ack":
            raise ConnectionError("synthetic capture interruption")
        return result

    monkeypatch.setattr(store, "_operation", interrupted)
    monkeypatch.setattr(sqlserver, "_execute", lost_ack)
    with pytest.raises(ConnectionError, match="synthetic capture interruption"):
        store.capture(typed_input[0], capture_id=capture_id, checkpoint=checkpoint)
    monkeypatch.setattr(store, "_operation", operation)
    monkeypatch.setattr(sqlserver, "_execute", execute)
    assert execute(store.writer, "SELECT @@TRANCOUNT") == [(0,)]
    exists = execute(store.reader, "SELECT object_id FROM sys.tables WHERE name=?",
                     ("af_input_" + capture_id,))
    assert bool(exists) == (failure == "commit_ack")
    if failure == "before_seal":
        assert not journal
    else:
        assert len(journal) == 1
        record = journal[0]
        if failure == "commit_ack":
            store.inspect(record)
            # A different recorded seal must never authorize removal of this capture.
            wrong = json.loads(json.dumps(record))
            wrong["incarnation"]["seal"] = "0" * 32
            with pytest.raises(ValueError, match="incarnation or schema changed"):
                store.release(wrong)
            store.inspect(record)
        store.release(record)
        store.release(record)
    assert not execute(store.reader, "SELECT object_id FROM sys.tables WHERE name=?",
                       ("af_input_" + capture_id,))


@pytest.mark.parametrize("drift", ["replace", "column_type", "nullable", "collation"])
def test_sqlserver_retained_object_or_schema_changes_fail_closed(
    sqlserver_store, typed_input, drift,
):
    from autoforecast.integrations.sqlserver import _execute

    store = sqlserver_store
    record = store.capture(typed_input[0], capture_id=uuid.uuid4().hex)
    name = store._name(record)
    if drift == "replace":
        _execute(store.test_admin, "DROP TABLE " + name)
        _execute(store.test_admin, "CREATE TABLE " + name + " (id BIGINT)")
    else:
        declaration = {
            "column_type": "value REAL NULL", "nullable": "id BIGINT NOT NULL",
            "collation": "label NVARCHAR(MAX) COLLATE Latin1_General_100_BIN2 NULL",
        }[drift]
        _execute(store.test_admin, "ALTER TABLE " + name + " ALTER COLUMN " + declaration)
    with pytest.raises((ValueError, FileNotFoundError), match="changed|original seal"):
        store.inspect(record)
    if drift == "replace":
        with pytest.raises(FileNotFoundError, match="original seal"):
            store.release(record)


def test_sqlserver_retained_identity_cannot_be_reused(sqlserver_store, typed_input):
    from autoforecast.integrations.sqlserver import _execute

    store = sqlserver_store
    record = store.capture(typed_input[0], capture_id=uuid.uuid4().hex)
    store.retain(record)
    store.release(record)
    with pytest.raises(Exception, match="identity already exists"):
        store.capture(typed_input[0], capture_id=record["capture_id"])
    _execute(store.test_admin, "CREATE TABLE " + store._name(record) + " (id BIGINT)")
    with pytest.raises(FileNotFoundError, match="original seal"):
        store.release(record)
    assert _execute(store.reader, "SELECT COUNT_BIG(*) FROM " + store._name(record)) == [(0,)]


def test_sqlserver_refuses_privileged_source_accounts(sqlserver_store):
    store = sqlserver_store
    store.check_source(store.test_clients["source"])
    for client in (store.test_admin, store.test_clients["release"]):
        with pytest.raises(ValueError, match="ordinary source account"):
            store.check_source(client)


def test_sqlserver_resolver_separates_registration_replay_and_release(
    sqlserver_store, typed_input, tmp_path, monkeypatch,
):
    from mssql_python.connection_string_parser import _ConnectionStringParser

    from autoforecast.snapshots import Connections, NativeReadScope

    store = sqlserver_store
    variables = {role: "AF_SQLSERVER_" + role.upper() + "_DSN"
                 for role in ("writer", "reader", "release")}
    database = store._database(store.reader)["name"]
    # These fixture-only identities never come from an application configuration.
    base = _ConnectionStringParser._normalize_params(_ConnectionStringParser()._parse(
        os.environ["AUTOFORECAST_TEST_INPUT_SQLSERVER_DSN"],
    ))

    def dsn(role):
        values = {**base, "Database": database, "UID": store.test_roles[role],
                  "PWD": "Local-source-test-only-2026!"}
        return ";".join(key + "={" + value.replace("}", "}}") + "}"
                        for key, value in values.items())

    for role in ("writer", "reader"):
        monkeypatch.setenv(variables[role], dsn(role))
    monkeypatch.delenv(variables["release"], raising=False)
    connections = Connections({"archive": {
        "kind": "sqlserver", "dsn_env": variables["reader"],
        "writer_dsn_env": variables["writer"], "release_dsn_env": variables["release"],
    }})
    destination = {"connection": "archive", "namespace": store.schema}
    record = None
    try:
        application = connections.snapshot_store(destination)
        assert application.releaser is None
        with RetainedInputs(mode="table", connection="archive", store=application,
                            scratch_root=tmp_path) as manager:
            reference = manager.capture(typed_input[0])
            record = reference.record()
            application.retain(record)
        with pytest.raises(ValueError, match="release_dsn_env"):
            connections.snapshot_store(destination, release=True)
        scope = NativeReadScope(connections, scratch_root=tmp_path)
        try:
            restored = pq.read_table(scope.materialize(record))
            assert sorted(restored.to_pylist(), key=repr) == sorted(
                typed_input[1].to_pylist(), key=repr,
            )
        finally:
            scope.close()
        monkeypatch.setenv(variables["release"], dsn("release"))
        cleanup = connections.snapshot_store(destination, release=True)
        assert cleanup.writer is None
        cleanup.release(record)
        cleanup.release(record)
        with pytest.raises(FileNotFoundError):
            application.inspect(record)
    finally:
        connections.close()
        if record is not None:
            store.release(record)


@pytest.mark.parametrize("fixture_name,backend", [
    ("postgres_store", "POSTGRES"), ("mysql_source", "MYSQL"), ("mysql_store", "MYSQL"),
    ("sqlserver_database", "SQLSERVER"), ("sqlserver_store", "SQLSERVER"),
    ("bigquery_store", "BIGQUERY"),
])
@pytest.mark.parametrize("required", [False, True])
def test_input_service_selection_is_explicit(monkeypatch, fixture_name, backend, required):
    # Credentials for result-publishing tests must not select input-storage tests.
    for suffix in ("DSN", "DATASET", "LOCATION"):
        monkeypatch.setenv("AUTOFORECAST_TEST_" + backend + "_" + suffix, "unrelated-service")
        monkeypatch.delenv("AUTOFORECAST_TEST_INPUT_" + backend + "_" + suffix, raising=False)
    monkeypatch.setenv("AUTOFORECAST_REQUIRE_INPUT_" + backend, "1" if required else "0")
    arguments = (None,) if fixture_name == "sqlserver_database" else ()
    fixture = globals()[fixture_name].__wrapped__(*arguments)
    outcome = pytest.fail.Exception if required else pytest.skip.Exception
    with pytest.raises(outcome):
        next(fixture)


@pytest.mark.parametrize('top', ['hash', 'table', 'table+hash'])
@pytest.mark.parametrize('stored', [None, 'hash', 'table', 'table+hash'])
def test_environment_registry_preserves_mode_precedence(tmp_path, monkeypatch, top, stored):
    import json

    from autoforecast.study_registration import StudyRegistry

    settings = {'storage': {'root': str(tmp_path / 'registry')}, 'snapshot_mode': top}
    if stored is not None:
        settings['storage']['snapshot_mode'] = stored
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(settings))
    monkeypatch.setenv('AUTOFORECAST_CONFIG', str(config))
    with StudyRegistry.from_config(config, spark=object()) as explicit:
        assert explicit.snapshot_mode == top
    with StudyRegistry.from_environment(spark=object()) as implicit:
        assert implicit.snapshot_mode == top


@pytest.mark.parametrize('mode', ['table', 'table+hash'])
@pytest.mark.parametrize('prepared', [False, True])
def test_releasing_registered_derived_study_keeps_parent_evidence(
    postgres_store, tmp_path, mode, prepared,
):
    import copy
    import json

    from pyspark.sql import SparkSession

    from autoforecast.study_preflight import preflight_study
    from autoforecast.study_registration import StudyRegistry, register_study
    from autoforecast.test_study_preparation import SensorPreparation
    from autoforecast.test_study_registration import configuration

    session = (SparkSession.builder.master('local[2]').appName('derived-release-check')
               .config('spark.ui.enabled', 'false')
               .config('spark.sql.shuffle.partitions', '2').getOrCreate())
    session.sparkContext.setLogLevel('ERROR')
    registry = StudyRegistry(tmp_path / 'registry', spark=session, snapshot_mode=mode,
                             connections={'archive': postgres_store},
                             snapshots={'connection': 'archive',
                                        'namespace': postgres_store.schema})
    try:
        parent_id = register_study(configuration(tmp_path), registry=registry,
                                   prepare=SensorPreparation() if prepared else None)
        parent_definition, _ = registry.definition(parent_id)
        assert parent_definition['native_findings']
        with registry.load_study(parent_id) as parent:
            result = preflight_study(parent, {
                'series_policy': 'use_available', 'minimum_history': 1,
                'metrics': ['mae'], 'metric': 'mae', 'selection_boundaries': [],
            }, output_dir=tmp_path / 'preflight')
            assert result.study_id != parent_id
            from autoforecast.benchmark import read_benchmark_manifest

            _, derived = read_benchmark_manifest(result.study_path)
            for reference in derived["snapshots"].values():
                assert (registry.root / "capture-journal" / reference["capture_id"]
                        / "sealed.json").is_file()
            assert not (registry.scratch_root / "capture-journal").exists()
            from autoforecast import benchmark

            # Recomputing a child's checksum cannot change its parent's identity.
            changed = copy.deepcopy(derived)
            changed["parent"]["study_id"] = "f" * 64
            identity = benchmark.digest(benchmark._identity(changed))
            changed["study_id"] = changed["benchmark_id"] = identity
            invalid = tmp_path / "invalid-derived.json"
            invalid.write_text(json.dumps(changed))
            with parent._native_inputs.active():
                with pytest.raises(ValueError, match="invalid derived study parent"):
                    benchmark.verify_benchmark(invalid)
            registry.retain_derived(parent_id, result.study_path)
            with pytest.raises(ValueError, match="dependent study"):
                registry.release(parent_id)
            # Preflight keeps its child open until this scope closes.
            with pytest.raises(ValueError, match="active readers"):
                registry.release(result.study_id)
        registry.definition(parent_id)
        with registry.load_study(parent_id) as parent:
            # A separate parent reader must survive releasing the closed child.
            registry.release(result.study_id)
            assert parent.testing_data.count() == 4
            with pytest.raises(ValueError, match="active readers"):
                registry.release(parent_id)
        registry.release(result.study_id)
        # A released child must leave all parent inputs and evidence usable.
        registry.definition(parent_id)
        with registry.load_study(parent_id) as parent:
            assert parent.testing_data.count() == 4
        registry.release(parent_id)
    finally:
        registry.close()
        session.stop()


def test_native_comparison_uses_input_binding_without_file_hash(
    postgres_store, tmp_path, monkeypatch,
):
    import dataclasses
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq
    from pyspark.sql import SparkSession

    from autoforecast import _logical_hash
    from autoforecast.benchmark import _evaluation_contract_from_record, bound_input_path
    from autoforecast.panel import Forecast
    from autoforecast.run_evaluation import compare_run_forecasts
    from autoforecast.study_registration import StudyRegistry, register_study
    from autoforecast.test_study_registration import configuration

    session = (SparkSession.builder.master('local[2]').appName('native-comparison-check')
               .config('spark.ui.enabled', 'false')
               .config('spark.sql.shuffle.partitions', '2').getOrCreate())
    session.sparkContext.setLogLevel('ERROR')
    registry = StudyRegistry(tmp_path / 'registry', spark=session, snapshot_mode='table',
                             connections={'archive': postgres_store},
                             snapshots={'connection': 'archive',
                                        'namespace': postgres_store.schema})
    try:
        study_id = register_study(configuration(tmp_path), registry=registry)
        with registry.load_study(study_id) as study, study._native_inputs.active():
            record = study._record(data=True)
            origin = record['recipe']['final_train_cutoff']
            horizon = record['recipe']['final_test_horizon']
            fixed = _evaluation_contract_from_record(study._path, record, stage='test',
                                                      origin=origin, horizon=horizon, metric='mae')
            rows = pq.read_table(bound_input_path(fixed['grid'])).to_pylist()
            predictions = tmp_path / 'predictions.parquet'
            pq.write_table(pa.Table.from_pylist([dict(row, yhat=0.0) for row in rows]), predictions)
            forecast = Forecast('fixture', str(predictions), origin, horizon, study.n_series,
                                {'emitted': len(rows)}, target_estimand='observed_nonnegative')
            monkeypatch.setattr(_logical_hash, "fingerprint", lambda *_a, **_k: pytest.fail(
                "table-only comparison must not hash input rows",
            ))
            result = compare_run_forecasts(
                dataclasses.replace(study.panel, path=str(bound_input_path(fixed['actuals']))),
                {'fixture': forecast}, session, missing_actual='no_data', metric='mae',
                output_dir=tmp_path / 'comparison', fixed_contract=fixed,
            )
            assert json.loads(result.read_text())['ranking']['order'] == ['fixture']
        registry.release(study_id)
    finally:
        registry.close()
        session.stop()


def test_run_study_keeps_registry_connections_through_local_tracking(
    postgres_store, tmp_path, monkeypatch,
):
    import mlflow
    from pyspark.sql import SparkSession

    from autoforecast import run_study
    from autoforecast.study_registration import StudyRegistry, register_study
    from autoforecast.test_study_registration import configuration

    monkeypatch.delenv('AUTOFORECAST_CONFIG', raising=False)
    uri = 'sqlite:///' + str(tmp_path / 'tracking.db')
    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    experiment_id = client.create_experiment(
        'native-tracking', artifact_location=(tmp_path / 'mlflow-artifacts').as_uri(),
    )
    session = (SparkSession.builder.master('local[2]').appName('native-tracking-check')
               .config('spark.ui.enabled', 'false')
               .config('spark.sql.shuffle.partitions', '2').getOrCreate())
    session.sparkContext.setLogLevel('ERROR')
    registry = StudyRegistry(tmp_path / 'registry', spark=session, snapshot_mode='table',
                             connections={'archive': postgres_store},
                             snapshots={'connection': 'archive',
                                        'namespace': postgres_store.schema})
    try:
        study_id = register_study(configuration(tmp_path), registry=registry)
        result = run_study(study_id, 'seasonal_naive', registry=registry, publish=False,
                           tracking='mlflow', tracking_uri=uri, mlflow_experiment='native-tracking')
        assert result.testing_metrics['mae'] == pytest.approx(5 / 3)
        assert len(client.search_runs([experiment_id])) == 2
        from psycopg import sql

        from autoforecast.integrations.postgres import PostgresResults
        from autoforecast.results import read_workflow_results

        # A lost publication response must be retryable without fitting again.
        schema = "af_results_" + uuid.uuid4().hex
        admin = postgres_store.test_admin
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            destination = PostgresResults(admin, schema=schema)
            destination.create_tables()
            publish = destination.publish_results
            calls = []

            def lose_first_response(*args, **kwargs):
                saved = publish(*args, **kwargs)
                calls.append(saved)
                if len(calls) == 1:
                    raise ConnectionError("publication response lost after commit")
                return saved

            monkeypatch.setattr(destination, "publish_results", lose_first_response)
            with pytest.raises(ConnectionError, match="after commit"):
                registry.publish_results(study_id, result.path, destination=destination)
            registry.publish_results(study_id, result.path, destination=destination)
            assert len(calls) == 2
            saved = admin.execute(sql.SQL(
                "SELECT stage, series_id, date, yhat FROM {}.forecasts"
            ).format(sql.Identifier(schema))).fetchall()
            expected = read_workflow_results(
                result.path, connections={"archive": postgres_store}, scratch_root=tmp_path,
            ).rows.to_pylist()
            assert sorted(saved) == sorted(
                (row["stage"], row["series_id"], str(row["target_date"]), row["prediction"])
                for row in expected
            )
            assert len(saved) == 8
            assert admin.execute(sql.SQL("SELECT count(*) FROM {}.study_results").format(
                sql.Identifier(schema))).fetchone()[0] == 2
            assert len(client.search_runs([experiment_id])) == 2
        finally:
            admin.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
        registry.release(study_id)
    finally:
        registry.close()
        session.stop()


@pytest.mark.parametrize("store_fixture", ["postgres_store", "mysql_store", "sqlserver_store"])
def test_temporary_capture_recovers_from_the_registry_journal(
    request, store_fixture, typed_input, tmp_path, monkeypatch,
):
    from autoforecast.snapshots import NativeReadScope
    from autoforecast.study_registration import StudyRegistry

    store = request.getfixturevalue(store_fixture)
    path, expected = typed_input
    registry = StudyRegistry(tmp_path / "registry", spark=object(), connections={"archive": store})
    registry.scratch_root.mkdir(parents=True)
    journal = registry.root / "capture-journal"
    manager = RetainedInputs(mode="table", connection="archive", store=store,
                             scratch_root=registry.scratch_root, journal_root=journal)
    scope = NativeReadScope(registry.connections, scratch_root=registry.scratch_root)
    template = manager.capture(path).record()
    reference = None
    drop = store.drop_unretained
    interrupt_cleanup = False

    def interrupted_cleanup(*args, **kwargs):
        if interrupt_cleanup:
            raise OSError("cleanup connection unavailable")
        return drop(*args, **kwargs)

    monkeypatch.setattr(store, "drop_unretained", interrupted_cleanup)
    try:
        reference = scope.capture_temporary(path, template=template, journal_root=journal)
        capture_id = reference["capture_id"]
        assert (journal / capture_id / "sealed.json").is_file()
        assert not (registry.scratch_root / "capture-journal").exists()
        with pytest.raises(ValueError, match="active owner"):
            registry.recover_capture(capture_id)
        with scope.active():
            assert pq.read_table(scope.materialize(reference)).equals(expected)

        # Failed cleanup must leave the original identity in the durable registry.
        interrupt_cleanup = True
        with pytest.raises(OSError, match="cleanup connection unavailable"):
            scope.close()
        store.inspect(reference)
        interrupt_cleanup = False
        assert registry.recover_capture(capture_id) == capture_id
        assert registry.recover_capture(capture_id) == capture_id
        assert (journal / capture_id / "removed.json").is_file()
    finally:
        interrupt_cleanup = False
        scope.close()
        manager.close()
        if reference is not None:
            store.release(reference)
        store.release(template)
        registry.close()


@pytest.mark.parametrize("consumer_fails", [False, True])
def test_native_reader_keeps_the_original_error_during_final_verification(tmp_path, consumer_fails):
    from autoforecast.snapshots import arrow_schema

    table = pa.table({"id": [1]})
    record = {"schema": "autoforecast.native-input.v1", "mode": "table",
              "backend": "postgresql", "connection": "archive", "capture_id": "a" * 32,
              "object": "archive.af_input_" + "a" * 32, "incarnation": {"seal": "one"},
              "arrow_schema": arrow_schema(table.schema), "rows": 1, "retention": "until_released"}

    class Store:
        backend = "postgresql"

        def __init__(self):
            self.calls = 0

        def inspect(self, _record):
            self.calls += 1
            if self.calls == 3:
                raise OSError("secondary verification failure")

        def read(self, _record, destination):
            destination.mkdir()
            pq.write_table(table, destination / "part.parquet")

    failure = ValueError("original consumer failure") if consumer_fails else None
    with RetainedInputs(mode="table", connection="archive", store=Store(),
                        scratch_root=tmp_path) as manager:
        with pytest.raises(ValueError if consumer_fails else OSError) as caught:
            with manager.open(SnapshotReference.from_record(record)):
                if failure is not None:
                    raise failure
        if failure is not None:
            assert caught.value is failure
            assert failure.__notes__ == ["Final native input verification failed: OSError"]
        else:
            assert str(caught.value) == "secondary verification failure"


@pytest.mark.parametrize("operation_fails", [False, True])
def test_worker_scope_keeps_the_original_error_during_cleanup(tmp_path, operation_fails):
    from autoforecast.snapshots import Connections, input_scope

    def interrupted_cleanup():
        raise OSError("secondary cleanup failure")

    failure = ValueError("original worker failure") if operation_fails else None
    with pytest.raises(ValueError if operation_fails else OSError) as caught:
        with input_scope(connections=Connections(), scratch_root=tmp_path) as scope:
            scope._stack.callback(interrupted_cleanup)
            if failure is not None:
                raise failure
    if failure is not None:
        assert caught.value is failure
        assert failure.__notes__ == ["Native input cleanup needs recovery: OSError"]
    else:
        assert str(caught.value) == "secondary cleanup failure"


@pytest.mark.parametrize("mode", ["table", "table+hash"])
def test_native_rescoring_uses_configured_readers_and_preserves_forecasts(
    postgres_store, tmp_path, monkeypatch, mode,
):
    import hashlib
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from psycopg.conninfo import make_conninfo
    from pyspark.sql import SparkSession

    from autoforecast import benchmark, rescore
    from autoforecast.panel import Forecast
    from autoforecast.snapshots import StudyLease
    from autoforecast.study_registration import StudyRegistry, register_study
    from autoforecast.test_study_registration import configuration
    from provenance.lineage import directory_digest

    session = (SparkSession.builder.master("local[2]").appName("native-rescoring")
               .config("spark.ui.enabled", "false")
               .config("spark.sql.shuffle.partitions", "2").getOrCreate())
    session.sparkContext.setLogLevel("ERROR")
    registry = StudyRegistry(
        tmp_path / "registry", spark=session, snapshot_mode=mode,
        connections={"archive": postgres_store},
        snapshots={"connection": "archive", "namespace": postgres_store.schema},
    )
    try:
        study_id = register_study(configuration(tmp_path), registry=registry)
        with registry.load_study(study_id) as study, study._native_inputs.active():
            study_path = study._path
            fixed = benchmark.evaluation_contract(
                study_path, stage="test", origin="2024-01-07", horizon=2, metric="mae",
            )
            rows = pq.read_table(benchmark.bound_input_path(fixed["grid"])).to_pylist()
        predictions = tmp_path / "predictions.parquet"
        pq.write_table(pa.Table.from_pylist([{**row, "yhat": 0.0} for row in rows]), predictions)
        forecast = Forecast("test", str(predictions), "2024-01-07", 2, 2,
                            {"emitted": len(rows)}, target_estimand="observed_nonnegative")
        forecast_record = tmp_path / "forecast.json"
        forecast_record.write_text(json.dumps(forecast.as_record()))
        original_bytes = predictions.read_bytes()
        original_digest = directory_digest(predictions, suffixes=(".parquet",))
        original = SimpleNamespace(
            metadata={"resolved_sha256": "a" * 64, "study_id": study_id},
            runs=({"status": "completed", "stage": "test", "adapter": "test", "model_id": "test",
                   "variant_id": "default", "run_id": "retained-run", "origin": forecast.origin,
                   "horizon": forecast.horizon, "forecast_record_path": str(forecast_record),
                   "forecast_record_sha256": hashlib.sha256(
                       forecast_record.read_bytes(),
                   ).hexdigest(),
                   "forecast_artifact_sha256": original_digest},),
        )
        # Campaign loading is unchanged; actual forecast verification and scoring run below.
        monkeypatch.setattr(rescore, "read_results", lambda _: original)
        worker = tmp_path / "reader.json"
        worker.write_text(json.dumps({"connections": {"archive": {
            "kind": "postgresql", "dsn_env": "AF_TEST_RESCORE_READER",
        }}}))
        monkeypatch.setenv("AUTOFORECAST_CONFIG", str(worker))
        monkeypatch.setenv("AF_TEST_RESCORE_READER", make_conninfo(
            os.environ["AUTOFORECAST_TEST_INPUT_POSTGRES_DSN"],
            user=postgres_store.test_roles["reader"], password="local-test-reader-only",
        ))
        result = rescore.rescore_campaign(tmp_path / "campaign", study_path,
                                         tmp_path / "rescored", session)
        record = json.loads(result.read_text())
        assert record["scoring_study_id"] == study_id and len(record["scores"]) == 1
        score = record["scores"][0]
        comparison = json.loads(Path(score["comparison"]).read_text())
        assert comparison["candidates"][0]["score"]["mae"] == pytest.approx(26 / 3)
        assert score["stage"] == "test" and score["forecast_artifact_sha256"] == original_digest
        assert predictions.read_bytes() == original_bytes
        assert StudyLease.active(registry.root, study_id) == []
        monkeypatch.delenv("AF_TEST_RESCORE_READER")
        with pytest.raises(ValueError, match="connection .archive. needs its DSN"):
            rescore.rescore_campaign(tmp_path / "campaign", study_path,
                                     tmp_path / "missing-reader", session)
        assert not (tmp_path / "missing-reader").exists()
        assert StudyLease.active(registry.root, study_id) == []
        registry.release(study_id)
    finally:
        registry.close()
        session.stop()


def test_manager_context_preserves_error_when_owner_close_fails(tmp_path):
    manager = RetainedInputs(
        mode="table", connection="archive", store=object(), scratch_root=tmp_path
    )

    def failed_close():
        raise OSError("secondary owner cleanup failure")

    manager._locks.callback(failed_close)
    with pytest.raises(ValueError, match="original capture consumer failure"):
        with manager:
            raise ValueError("original capture consumer failure")


def test_postgres_capture_preserves_inspection_error_when_drop_fails(
    postgres_store, typed_input, monkeypatch
):
    import uuid

    def failed_inspection(*a, **k):
        raise ValueError("original capture inspection failure")

    def failed_drop(*a, **k):
        raise OSError("secondary provider cleanup failure")

    monkeypatch.setattr(postgres_store, "inspect", failed_inspection)
    monkeypatch.setattr(postgres_store, "drop_unretained", failed_drop)
    with pytest.raises(ValueError, match="original capture inspection failure"):
        postgres_store.capture(typed_input[0], capture_id=uuid.uuid4().hex)


@pytest.mark.parametrize("phase", ["rollback", "timezone"])
def test_mysql_source_preserves_error_when_rollback_fails(tmp_path, phase):
    from autoforecast.integrations.mysql import read_sources

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, statement, *args):
            self.statement = statement
            if phase == "timezone" and statement == "SET SESSION time_zone=%s":
                raise OSError("secondary source timezone failure")
            if statement.startswith("START TRANSACTION"):
                raise ValueError("original source transaction failure")

        def fetchone(self):
            if self.statement.startswith("SELECT @@session.character_set_client"):
                return ("utf8mb4", "utf8mb4", "utf8mb4")
            return ("+00:00",)

    class Connection:
        server_status = 0
        charset = "utf8mb4"
        encoding = "utf8"

        def get_autocommit(self):
            return True

        def cursor(self):
            return Cursor()

        def rollback(self):
            if phase == "rollback":
                raise OSError("secondary source rollback failure")

    with pytest.raises(ValueError, match="original source transaction failure"):
        read_sources(
            Connection(),
            queries={"primary": "SELECT * FROM sales"},
            tables={"sales": "source.sales"},
            output_root=tmp_path,
        )


@pytest.mark.parametrize("phase", ["metadata", "delete"])
def test_bigquery_primitive_preserves_create_error_when_cleanup_fails(tmp_path, phase):
    from autoforecast.integrations.bigquery import BigQuerySnapshots

    table = pa.table({"id": [1]})
    path = tmp_path / "rows.parquet"
    pq.write_table(table, path)

    class Client:
        def create_table(self, table):
            self.table = table
            raise ValueError("original create failure")

        def get_table(self, *args):
            if phase == "metadata":
                raise OSError("secondary cleanup metadata failure")
            return self.table

        def delete_table(self, *args, **kwargs):
            raise OSError("secondary deletion failure")

    store = BigQuerySnapshots(Client(), dataset="example-project.archive", location="US")
    store.preflight = lambda: None
    with pytest.raises(ValueError, match="original create failure"):
        store.capture(path, capture_id="b" * 32)


@pytest.mark.parametrize("phase", ["rollback", "restore_database"])
def test_sqlserver_provision_preserves_error_when_rollback_fails(monkeypatch, phase):
    from autoforecast.integrations import sqlserver

    names = ("owner", "writer", "reader", "release")

    def execute(connection, statement, values=()):
        if statement == "SELECT @@TRANCOUNT":
            return [(0,)]
        if statement == "SELECT DB_NAME()":
            return [("owned_archive",)]
        if "FROM sys.database_principals" in statement:
            return [
                (names[0], "SQL_USER", "NONE", None),
                *((name, "SQL_USER", "INSTANCE", name) for name in names[1:]),
            ]
        if statement.startswith("CREATE SCHEMA"):
            raise ValueError("original provisioning failure")
        if phase == "rollback" and "ROLLBACK" in statement:
            raise OSError("secondary provisioning rollback failure")
        if phase == "restore_database" and statement.startswith("USE ["):
            raise OSError("secondary database restore failure")
        return []

    class Connection:
        autocommit = True

    monkeypatch.setattr(sqlserver, "_execute", execute)
    with pytest.raises(ValueError, match="original provisioning failure"):
        sqlserver.provision_snapshot_schema(
            Connection(),
            schema="archive",
            owner=names[0],
            registration_role=names[1],
            reader_role=names[2],
            release_role=names[3],
        )


def test_freezer_preserves_error_when_input_cleanup_fails(monkeypatch):
    from autoforecast import benchmark

    def fail(**settings):
        def cleanup():
            raise OSError("secondary freezer cleanup failure")

        settings["inputs"].callback(cleanup)
        raise ValueError("original freeze failure")

    monkeypatch.setattr(benchmark, "_freeze_benchmark", fail)
    with pytest.raises(ValueError, match="original freeze failure"):
        benchmark._freeze_verified()


def test_study_close_preserves_native_close_error_when_scratch_cleanup_fails():
    from autoforecast.study import Study

    class Native:
        def close(self):
            raise ValueError("original native close failure")

    class Scratch:
        def cleanup(self):
            raise OSError("secondary scratch cleanup failure")

    study = object.__new__(Study)
    object.__setattr__(study, "_closed", False)
    object.__setattr__(study, "_native_inputs", Native())
    object.__setattr__(study, "_snapshots", Scratch())
    with pytest.raises(ValueError, match="original native close failure"):
        study.close()


def test_freezer_preserves_error_when_new_artifact_cleanup_fails(
    postgres_store, tmp_path, monkeypatch
):
    from pathlib import Path

    from pyspark.sql import SparkSession

    from autoforecast import benchmark
    from autoforecast.test_study import inputs

    contract, sources, recipe = inputs(tmp_path)
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("freeze-cleanup-failure-probe")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    manager = RetainedInputs(
        mode="table", connection="archive", store=postgres_store, scratch_root=tmp_path
    )
    original_capture, original_drop = manager.capture, postgres_store.drop_unretained
    artifact_id = None

    def capture(path, **kwargs):
        nonlocal artifact_id
        if "coverage" in Path(path).parts:
            if artifact_id is not None:
                raise ValueError("original second artifact failure")
            result = original_capture(path, **kwargs)
            artifact_id = result.record()["capture_id"]
            return result
        return original_capture(path, **kwargs)

    def drop(reference):
        if reference["capture_id"] == artifact_id:
            raise OSError("secondary new artifact cleanup failure")
        return original_drop(reference)

    monkeypatch.setattr(manager, "capture", capture)
    monkeypatch.setattr(postgres_store, "drop_unretained", drop)
    try:
        with pytest.raises(ValueError, match="original second artifact failure"):
            benchmark._freeze_verified(
                contract_path=contract,
                sources=sources,
                recipe=recipe,
                output_root=tmp_path / "frozen",
                spark=spark,
                native_inputs=manager,
            )
        assert artifact_id is not None
    finally:
        manager.close()
        spark.stop()


@pytest.mark.parametrize("drift", ["extra_column", "column_type", "nullable", "collation"])
@pytest.mark.parametrize("backend", ["mysql", "sqlserver"])
@pytest.mark.parametrize("operation", ["release", "drop_unretained"])
def test_release_refuses_changed_sealed_schema(request, typed_input, backend, drift, operation):
    store = request.getfixturevalue(backend + "_store")
    record = store.capture(typed_input[0], capture_id=uuid.uuid4().hex)
    if operation == "release":
        store.retain(record)
    target = store._name(record)
    if backend == "mysql":
        from pymysql import Error as ProviderError

        from autoforecast.integrations.mysql import _execute

        command = {
            "extra_column": "ALTER TABLE " + target + " ADD extra INT NULL, ALGORITHM=INSTANT",
            "column_type": "ALTER TABLE " + target + " MODIFY value FLOAT",
            "nullable": "ALTER TABLE " + target + " MODIFY id BIGINT NOT NULL",
            "collation": "ALTER TABLE "
            + target
            + " MODIFY label LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci",
        }[drift]
    else:
        from mssql_python import DatabaseError as ProviderError

        from autoforecast.integrations.sqlserver import _execute

        command = {
            "extra_column": "ALTER TABLE " + target + " ADD extra INT NULL",
            "column_type": "ALTER TABLE " + target + " ALTER COLUMN value REAL NULL",
            "nullable": "ALTER TABLE " + target + " ALTER COLUMN id BIGINT NOT NULL",
            "collation": "ALTER TABLE "
            + target
            + " ALTER COLUMN label NVARCHAR(MAX) COLLATE Latin1_General_100_BIN2 NULL",
        }[drift]
    _execute(store.test_admin, command)
    with pytest.raises((ValueError, ProviderError), match="identity|columns|schema|changed"):
        store.inspect(record)
    with pytest.raises((ValueError, ProviderError), match="identity|columns|schema|changed"):
        getattr(store, operation)(record)


def test_worker_cleanup_does_not_hide_a_new_error_inside_an_outer_exception_handler(tmp_path):
    from autoforecast.snapshots import Connections, input_scope

    def interrupted_cleanup():
        raise OSError("current cleanup failed")

    try:
        raise LookupError("an older error that the application has already handled")
    except LookupError:
        # The current operation succeeded, so its cleanup failure must still be raised.
        with pytest.raises(OSError, match="current cleanup failed"):
            with input_scope(connections=Connections(), scratch_root=tmp_path) as scope:
                scope._stack.callback(interrupted_cleanup)


def test_native_checkpoint_handoff_checks_targets_and_binding(postgres_store, tmp_path):
    import dataclasses
    import hashlib
    import json
    from pathlib import Path

    import numpy as np
    import pyarrow.parquet as pq
    from pyspark.sql import SparkSession

    from autoforecast.benchmark import _evaluation_contract_from_record, bound_input_path
    from autoforecast.study_registration import StudyRegistry, register_study
    from autoforecast.study_validation import StudyValidation
    from autoforecast.test_study_registration import configuration

    session = (SparkSession.builder.master('local[2]').appName('native-checkpoint')
               .config('spark.ui.enabled', 'false')
               .config('spark.sql.shuffle.partitions', '2').getOrCreate())
    session.sparkContext.setLogLevel('ERROR')
    registry = StudyRegistry(tmp_path / 'registry', spark=session, snapshot_mode='table',
                             connections={'archive': postgres_store},
                             snapshots={'connection': 'archive',
                                        'namespace': postgres_store.schema})
    try:
        study_id = register_study(configuration(tmp_path), registry=registry)
        with registry.load_study(study_id) as study, study._native_inputs.active():
            record = study._record(data=True)
            fixed = _evaluation_contract_from_record(study._path, record, stage='validation',
                                                      origin='2024-01-05', horizon=2, metric='mae')
            grid = pq.read_table(bound_input_path(fixed['grid']))
            identifiers = tuple(sorted(set(grid['series_id'].to_pylist())))
            panel = dataclasses.replace(study.panel, path=str(bound_input_path(fixed['actuals'])))
            path = tmp_path / 'checkpoint.json'
            raw = json.dumps(fixed).encode()
            path.write_bytes(raw)
            sha = hashlib.sha256(raw).hexdigest()
            window = StudyValidation.read(path, sha, panel=panel, origin='2024-01-05',
                                           horizon=2, series_ids=identifiers)
            np.testing.assert_allclose(window.target, [[6.0, 7.0], [np.nan, 7.0]], equal_nan=True)
            assert window.score(np.zeros((2, 2)))['mae'] == pytest.approx(20 / 3)
            from autoforecast.snapshots import current_inputs
            from autoforecast.study_tuning import _prepare_window

            inputs = _prepare_window(study, "validation", "2024-01-05", 2,
                                     tmp_path / "window")
            bindings = inputs.verify()["native_inputs"]
            scope = current_inputs()
            for role in ("training", "mapping"):
                original_path = inputs.panel.path if role == "training" else inputs.mapping_path
                other = scope.capture_temporary(
                    Path(original_path), template=bindings[role]["snapshot"],
                    journal_root=registry.root / "capture-journal",
                )
                replacement = scope.materialize(other)
                swapped = (dataclasses.replace(
                    inputs, panel=dataclasses.replace(inputs.panel, path=str(replacement)),
                ) if role == "training" else dataclasses.replace(inputs, mapping_path=replacement))
                with pytest.raises(ValueError, match="native transport differs"):
                    swapped.verify()
            with pytest.raises(ValueError, match='native transport differs'):
                StudyValidation.read(path, sha, panel=dataclasses.replace(panel, path='other'),
                                      origin='2024-01-05', horizon=2, series_ids=identifiers)
            changed = json.loads(raw)
            changed['actuals']['snapshot']['capture_id'] = 'f' * 32
            raw = json.dumps(changed).encode()
            path.write_bytes(raw)
            with pytest.raises(ValueError):
                StudyValidation.read(path, hashlib.sha256(raw).hexdigest(), panel=panel,
                                      origin='2024-01-05', horizon=2, series_ids=identifiers)
        registry.release(study_id)
    finally:
        registry.close()
        session.stop()


@pytest.fixture
def transport_reference():
    from autoforecast.snapshots import arrow_schema

    table = pa.table({"id": [1], "value": [2.5]})
    record = {"schema": "autoforecast.native-input.v1", "mode": "table",
              "backend": "postgresql", "connection": "archive", "capture_id": "a" * 32,
              "object": "archive.af_input_" + "a" * 32, "incarnation": {"seal": "original"},
              "arrow_schema": arrow_schema(table.schema), "rows": 1,
              "retention": "until_released"}
    return table, record


@pytest.mark.parametrize("change", [
    {"mode": "hash"}, {"mode": "table+hash"}, {"logical_hash": {}}, {"backend": "unknown"},
    {"rows": True}, {"rows": -1}, {"retention": "one_day"}, {"password": "not-allowed"},
    {"schema": "old"}, {"connection": "../archive"}, {"capture_id": "../capture"},
    {"object": ""}, {"incarnation": {}}, {"arrow_schema": "bad-base64"},
])
def test_native_reference_refuses_ambiguous_or_incomplete_identity(transport_reference, change):
    _, original = transport_reference
    with pytest.raises(ValueError):
        SnapshotReference.from_record({**original, **change})
    reference = SnapshotReference.from_record(original)
    original["incarnation"]["seal"] = "changed"
    returned = reference.record()
    returned["incarnation"]["seal"] = "also_changed"
    assert reference.record()["incarnation"]["seal"] == "original"


@pytest.mark.parametrize("failure", [
    "missing_files", "empty_without_files", "schema", "rows", "wrong_resolver", "content",
])
def test_native_transport_refuses_incomplete_or_wrong_data(tmp_path, transport_reference, failure):
    from types import SimpleNamespace

    from autoforecast import _logical_hash

    table, record = transport_reference
    mode = "table"
    if failure == "empty_without_files":
        record["rows"] = 0
    if failure == "content":
        path = tmp_path / "original.parquet"
        pq.write_table(table, path)
        mode = record["mode"] = "table+hash"
        record["logical_hash"] = _logical_hash.fingerprint(path)

    def read(_record, destination):
        destination.mkdir()
        if failure not in {"missing_files", "empty_without_files"}:
            value = (pa.table({"other": [1]}) if failure == "schema"
                     else table.slice(0, 0) if failure == "rows"
                     else pa.table({"id": [1], "value": [9.5]}) if failure == "content" else table)
            pq.write_table(value, destination / "part.parquet")

    store = SimpleNamespace(backend="postgresql", inspect=lambda _: None, read=read)
    connection = "different" if failure == "wrong_resolver" else "archive"
    with RetainedInputs(mode=mode, connection=connection, store=store,
                        scratch_root=tmp_path) as manager:
        error = "typed data|schema changed|row count|resolver|content"
        with pytest.raises(ValueError, match=error):
            with manager.open(SnapshotReference.from_record(record)):
                pytest.fail("invalid input reached the consumer")
    assert not list(tmp_path.glob("native-input-*"))


@pytest.mark.parametrize("failure", [
    "missing_checkpoint", "wrong_capture_id", "changed_return", "changed_journal",
])
def test_native_capture_requires_the_original_durable_checkpoint(
    tmp_path, transport_reference, failure,
):
    from types import SimpleNamespace

    table, record = transport_reference
    dropped = []
    journal = tmp_path / "journal"

    def capture(_path, *, capture_id, checkpoint):
        value = {key: value for key, value in record.items()
                 if key not in {"schema", "mode", "connection", "backend", "retention"}}
        value["capture_id"] = capture_id
        value["object"] = "archive.af_input_" + capture_id
        if failure == "wrong_capture_id":
            value["capture_id"] = "wrong_capture_id"
        if failure != "missing_checkpoint":
            checkpoint(value)
        if failure == "changed_return":
            value["incarnation"] = {"seal": "replacement"}
        if failure == "changed_journal":
            (journal / capture_id / "sealed.json").write_bytes(b"{}")
        return value

    store = SimpleNamespace(backend="postgresql", namespace="archive",
                            preflight=lambda: None, capture=capture,
                            inspect=lambda _: None, drop_unretained=dropped.append)
    path = tmp_path / "input.parquet"
    pq.write_table(table, path)
    with RetainedInputs(mode="table", connection="archive", store=store,
                        scratch_root=tmp_path, journal_root=journal) as manager:
        error = "sealed capture identity|after checkpointing|journal|ID"
        with pytest.raises(ValueError, match=error):
            manager.capture(path)
    assert not list(journal.glob("*/ready.json"))
    if failure in {"missing_checkpoint", "wrong_capture_id"}:
        # Without a sealed identity, cleanup must not guess which object to delete.
        assert dropped == []
    else:
        assert len(dropped) == 1
        assert dropped[0]["incarnation"]["seal"] == "original"


@pytest.mark.parametrize("change", ["replace_file", "file_symlink", "directory_symlink"])
def test_native_transport_rejects_replacement_during_owned_reads(
    tmp_path, transport_reference, change,
):
    from types import SimpleNamespace

    from autoforecast.snapshots import Connections, NativeReadScope

    table, record = transport_reference
    original = tmp_path / "original.parquet"
    pq.write_table(table, original)

    def read(_record, destination):
        destination.mkdir()
        pq.write_table(table, destination / "part.parquet")

    store = SimpleNamespace(backend="postgresql", inspect=lambda _: None, read=read)
    scope = NativeReadScope(Connections(supplied={"archive": store}), scratch_root=tmp_path)
    try:
        path = scope.materialize(record)
        file = path / "part.parquet"
        if change == "replace_file":
            file.chmod(0o600)
            pq.write_table(pa.table({"id": [1], "value": [9.5]}), file)
        elif change == "file_symlink":
            file.unlink()
            file.symlink_to(original)
        else:
            (path / "extra").symlink_to(tmp_path, target_is_directory=True)
        with pytest.raises(ValueError, match="changed|symbolic link"):
            scope.materialize(record)
    finally:
        scope.close()
    assert not path.exists()


@pytest.mark.parametrize("store_fixture", ["postgres_store", "mysql_store", "sqlserver_store"])
@pytest.mark.parametrize("failure", [
    "unsupported_type", "timestamp_precision", "mixed_schema", "missing_files",
])
def test_native_adapters_refuse_invalid_inputs_before_creating_tables(
    request, tmp_path, store_fixture, failure,
):
    store = request.getfixturevalue(store_fixture)
    path = tmp_path / "input"
    path.mkdir()
    if failure == "unsupported_type":
        pq.write_table(pa.table({"values": [[1, 2]]}), path / "part.parquet")
    elif failure == "timestamp_precision":
        pq.write_table(pa.table({"at": pa.array([1], type=pa.timestamp("ns"))}),
                       path / "part.parquet")
    elif failure == "mixed_schema":
        pq.write_table(pa.table({"value": [1]}), path / "one.parquet")
        pq.write_table(pa.table({"value": ["different"]}), path / "two.parquet")
    capture_id = uuid.uuid4().hex
    error = "unsupported|support|preserve|schemas|Parquet|parquet|files"
    with pytest.raises(ValueError, match=error):
        store.capture(path, capture_id=capture_id)
    # Check the real engine, rather than trusting an exception to mean rollback.
    placeholder = "?" if store.backend == "sqlserver" else "%s"
    query = ("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema="
             + placeholder + " AND table_name IN (" + placeholder + "," + placeholder + ")")
    with store.test_admin.cursor() as cursor:
        cursor.execute(query, (store.namespace, "af_input_" + capture_id, "af_build_" + capture_id))
        assert cursor.fetchone()[0] == 0


@pytest.mark.parametrize("actor", ["writer", "reader", "source"])
@pytest.mark.parametrize("indirect", [False, True])
@pytest.mark.parametrize("authority", [
    "SUPERUSER", "CREATEROLE", "CREATEDB", "REPLICATION", "BYPASSRLS",
    "pg_write_all_data", "pg_write_server_files", "pg_execute_server_program", "release",
])
def test_postgres_refuses_inherited_snapshot_administration(
    postgres_store, tmp_path, actor, indirect, authority,
):
    import uuid

    import pyarrow as pa
    import pyarrow.parquet as pq
    import pytest
    from psycopg import sql

    from autoforecast.snapshots import RetainedInputs

    # One complete synthetic row; no filters or eligibility exclusions.
    store = postgres_store
    admin, roles = store.test_admin, store.test_roles
    member = roles[actor]
    created = []
    grants = []
    path = tmp_path / "input.parquet"
    pq.write_table(pa.table({"id": [1], "value": [100]}), path)
    with RetainedInputs(mode="table", connection="archive", store=store,
                        scratch_root=tmp_path) as manager:
        reference = manager.capture(path)
        record = reference.record()
        store.retain(record)
        try:
            if authority.startswith("pg_"):
                parent = authority
            elif authority == "release":
                parent = roles["release"]
            else:
                parent = "af_privileged_" + uuid.uuid4().hex
                admin.execute(sql.SQL("CREATE ROLE {} NOLOGIN {}").format(
                    sql.Identifier(parent), sql.SQL(authority),
                ))
                created.append(parent)
            target = parent
            if indirect:
                target = "af_bridge_" + uuid.uuid4().hex
                admin.execute(sql.SQL("CREATE ROLE {} NOLOGIN").format(sql.Identifier(target)))
                created.append(target)
                admin.execute(sql.SQL("GRANT {} TO {}").format(
                    sql.Identifier(parent), sql.Identifier(target),
                ))
                grants.append((parent, target))
            # NOINHERIT still permits the explicit SET ROLE escalation being checked.
            admin.execute(sql.SQL("ALTER ROLE {} NOINHERIT").format(sql.Identifier(member)))
            admin.execute(sql.SQL("GRANT {} TO {}").format(
                sql.Identifier(target), sql.Identifier(member),
            ))
            grants.append((target, member))
            client = store.test_clients[actor]
            client.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(parent)))
            try:
                assert client.execute("SELECT current_user").fetchone()[0] == parent
                if authority == "SUPERUSER" and actor != "source":
                    client.execute(sql.SQL("UPDATE {} SET value=999").format(store._name(record)))
            finally:
                client.execute("RESET ROLE")
            if actor == "source":
                with pytest.raises(ValueError, match="ordinary source account|write-all-data"):
                    store.check_source(client)
            else:
                journals = set((tmp_path / "capture-journal").iterdir())
                with pytest.raises(ValueError, match="administrator|release|write-all-data"):
                    manager.capture(path)
                assert set((tmp_path / "capture-journal").iterdir()) == journals
                with pytest.raises(ValueError, match="administrator|release|write-all-data"):
                    with manager.open(reference):
                        pytest.fail("unsafe application authority reached retained rows")
                assert not list(tmp_path.glob("native-input-*"))
        finally:
            for parent, member in reversed(grants):
                admin.execute(sql.SQL("REVOKE {} FROM {}").format(
                    sql.Identifier(parent), sql.Identifier(member),
                ))
            for role in reversed(created):
                admin.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))
            store.release(record)


@pytest.mark.parametrize("actor", ["writer", "reader", "source"])
@pytest.mark.parametrize("indirect", [False, True])
def test_postgres_ordinary_memberships_preserve_capture_and_replay(
    postgres_store, tmp_path, actor, indirect,
):
    import uuid

    import pyarrow as pa
    import pyarrow.parquet as pq
    from psycopg import sql

    from autoforecast.snapshots import RetainedInputs

    store = postgres_store
    admin, member = store.test_admin, store.test_roles[actor]
    created = []
    path = tmp_path / "input.parquet"
    pq.write_table(pa.table({"id": [1], "value": [100]}), path)
    try:
        parent = "af_ordinary_" + uuid.uuid4().hex
        admin.execute(sql.SQL("CREATE ROLE {} NOLOGIN").format(sql.Identifier(parent)))
        created.append(parent)
        target = parent
        if indirect:
            target = "af_bridge_" + uuid.uuid4().hex
            admin.execute(sql.SQL("CREATE ROLE {} NOLOGIN").format(sql.Identifier(target)))
            created.append(target)
            admin.execute(sql.SQL("GRANT {} TO {}").format(
                sql.Identifier(parent), sql.Identifier(target),
            ))
        admin.execute(sql.SQL("GRANT {} TO {}").format(
            sql.Identifier(target), sql.Identifier(member),
        ))
        store.check_source(store.test_clients["source"])
        with RetainedInputs(mode="table", connection="archive", store=store,
                            scratch_root=tmp_path) as manager:
            reference = manager.capture(path)
            record = reference.record()
            store.retain(record)
            try:
                with manager.open(reference) as retained:
                    assert pq.read_table(retained).to_pylist() == [{"id": 1, "value": 100}]
            finally:
                store.release(record)
    finally:
        for role in reversed(created):
            admin.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))


@pytest.mark.parametrize("setting", ["off", "false", "no", "0"])
def test_postgres_source_refuses_nonstandard_strings_before_query_or_output(
    postgres_store, tmp_path, monkeypatch, setting,
):
    import pytest
    from psycopg import ServerCursor, sql
    from psycopg.pq import TransactionStatus

    from autoforecast.integrations.postgres import read_sources

    # The requested input is the one sales row. The hidden row is not declared.
    connection = postgres_store.writer
    namespace = postgres_store.test_source_schema
    with connection.transaction():
        for name, value in (("sales", 1), ("hidden", 999)):
            table = sql.Identifier(namespace, name)
            connection.execute(sql.SQL("CREATE TABLE {} (id INT)").format(table))
            connection.execute(sql.SQL("INSERT INTO {} VALUES (%s)").format(table), (value,))
    original = connection.execute("SHOW standard_conforming_strings").fetchone()[0]
    output = tmp_path / "capture"
    try:
        connection.execute("SELECT set_config('standard_conforming_strings',%s,false)", (setting,))
        assert connection.execute("SHOW standard_conforming_strings").fetchone()[0] == "off"

        def user_query_must_not_run(*args, **kwargs):
            pytest.fail("incompatible source grammar reached user SQL")

        monkeypatch.setattr(ServerCursor, "execute", user_query_must_not_run)
        query = ("SELECT id FROM sales WHERE length('x\\' = ' ) > 0 "
                 f"UNION ALL SELECT id FROM {namespace}.hidden -- ') > 0\n")
        with pytest.raises(ValueError, match="standard_conforming_strings=on"):
            read_sources(connection, queries={"primary": query},
                         tables={"sales": namespace + ".sales"}, output_root=output)
        assert not output.exists()
        assert connection.execute("SHOW standard_conforming_strings").fetchone()[0] == "off"
        assert connection.info.transaction_status == TransactionStatus.IDLE
    finally:
        connection.execute("SELECT set_config('standard_conforming_strings',%s,false)", (original,))


@pytest.mark.parametrize("setting", ["on", "true", "yes", "1"])
def test_postgres_source_standard_strings_preserve_rows_and_caller_settings(
    postgres_store, tmp_path, setting,
):
    import pyarrow.parquet as pq
    from psycopg import sql
    from psycopg.pq import TransactionStatus

    from autoforecast.integrations.postgres import read_sources

    # One complete synthetic row is selected; there are no exclusions.
    connection = postgres_store.writer
    name = postgres_store.test_source_schema + ".sales"
    table = sql.Identifier(*name.split("."))
    connection.execute(sql.SQL("CREATE TABLE {} (id INT)").format(table))
    connection.execute(sql.SQL("INSERT INTO {} VALUES (1)").format(table))
    original = connection.execute("SHOW standard_conforming_strings").fetchone()[0]
    timezone = connection.execute("SHOW TIME ZONE").fetchone()[0]
    try:
        connection.execute("SELECT set_config('standard_conforming_strings',%s,false)", (setting,))
        result = read_sources(
            connection, queries={"primary": "SELECT id, 'x\\' AS label FROM sales"},
            tables={"sales": name}, output_root=tmp_path / "capture",
        )
        assert pq.read_table(result["primary"][0]).to_pylist() == [{"id": 1, "label": "x\\"}]
        assert result["primary"][1]["rows"] == 1
        assert connection.execute("SHOW standard_conforming_strings").fetchone()[0] == "on"
        assert connection.execute("SHOW TIME ZONE").fetchone()[0] == timezone
        assert connection.info.transaction_status == TransactionStatus.IDLE
    finally:
        connection.execute("SELECT set_config('standard_conforming_strings',%s,false)", (original,))


@pytest.mark.parametrize("include_later_query", [False, True])
def test_postgres_source_select_cannot_change_validated_string_grammar(
    postgres_store, tmp_path, monkeypatch, include_later_query,
):
    import pytest
    from psycopg import ServerCursor, sql
    from psycopg.pq import TransactionStatus

    from autoforecast.integrations.postgres import read_sources

    # Only the one sales row belongs to the requested input population.
    connection = postgres_store.writer
    namespace = postgres_store.test_source_schema
    for name, value in (("sales", 1), ("hidden", 999)):
        table = sql.Identifier(namespace, name)
        connection.execute(sql.SQL("CREATE TABLE {} (id INT)").format(table))
        connection.execute(sql.SQL("INSERT INTO {} VALUES (%s)").format(table), (value,))
    original = connection.execute("SHOW standard_conforming_strings").fetchone()[0]
    queries = {"primary": "SELECT id,set_config('standard_conforming_strings','off',false) "
                          "AS setting FROM sales"}
    if include_later_query:
        queries["later"] = ("SELECT id FROM sales WHERE length('x\\' = ' ) > 0 "
                            f"UNION ALL SELECT id FROM {namespace}.hidden -- ') > 0\n")
    executed = []
    execute = ServerCursor.execute

    def execute_only_with_standard_strings(cursor, *args, **kwargs):
        assert connection.execute("SHOW standard_conforming_strings").fetchone()[0] == "on"
        executed.append(True)
        return execute(cursor, *args, **kwargs)

    monkeypatch.setattr(ServerCursor, "execute", execute_only_with_standard_strings)
    output = tmp_path / "capture"
    with pytest.raises(ValueError, match="standard_conforming_strings=on"):
        read_sources(connection, queries=queries, tables={"sales": namespace + ".sales"},
                     output_root=output)
    assert executed == [True]
    assert not (output / "query-1").exists()
    assert connection.execute("SHOW standard_conforming_strings").fetchone()[0] == original
    assert connection.info.transaction_status == TransactionStatus.IDLE


@pytest.mark.parametrize("actor", ["writer", "reader", "source"])
@pytest.mark.parametrize("indirect", [False, True])
def test_mysql_sources_refuse_inherited_administration(mysql_store, tmp_path, actor, indirect):
    import uuid

    import pymysql

    from autoforecast.integrations.mysql import _execute

    store = mysql_store
    admin = store.test_admin
    parent = "af_probe_admin_" + uuid.uuid4().hex[:16]
    bridge = "af_probe_bridge_" + uuid.uuid4().hex[:16]
    member = store.test_roles[actor]
    try:
        _execute(admin, f"CREATE ROLE '{parent}'@'%'")
        _execute(admin, f"GRANT ALL PRIVILEGES ON *.* TO '{parent}'@'%'")
        if indirect:
            _execute(admin, f"CREATE ROLE '{bridge}'@'%'")
            _execute(admin, f"GRANT '{parent}'@'%' TO '{bridge}'@'%'")
        _execute(admin, f"GRANT '{bridge if indirect else parent}'@'%' TO '{member}'@'%'")
        # Unactivated membership is enough: the account can activate it later.
        with pytest.raises(ValueError if actor == "source" else pymysql.Error,
                           match="changed|administration"):
            if actor == "source":
                store.check_source(store.test_clients[actor])
            else:
                store.preflight()
        assert not list(tmp_path.rglob("*.parquet"))
    finally:
        if indirect:
            _execute(admin, f"DROP ROLE IF EXISTS '{bridge}'@'%'")
        _execute(admin, f"DROP ROLE IF EXISTS '{parent}'@'%'")


@pytest.mark.parametrize("indirect", [False, True])
def test_mysql_replay_refuses_inherited_source_administration(
    mysql_store, typed_input, tmp_path, indirect,
):
    import uuid

    import pymysql

    from autoforecast.integrations.mysql import _execute
    from autoforecast.snapshots import RetainedInputs

    store = mysql_store
    admin = store.test_admin
    parent = "af_probe_admin_" + uuid.uuid4().hex[:16]
    bridge = "af_probe_bridge_" + uuid.uuid4().hex[:16]
    client = store.test_clients["source"]
    with RetainedInputs(mode="table", connection="archive", store=store,
                        scratch_root=tmp_path) as manager:
        reference = manager.capture(typed_input[0])
        record = reference.record()
        store.retain(record)
        try:
            _execute(admin, f"CREATE ROLE '{parent}'@'%'")
            _execute(admin, f"GRANT ALL PRIVILEGES ON *.* TO '{parent}'@'%'")
            if indirect:
                _execute(admin, f"CREATE ROLE '{bridge}'@'%'")
                _execute(admin, f"GRANT '{parent}'@'%' TO '{bridge}'@'%'")
            _execute(admin, f"GRANT '{bridge if indirect else parent}'@'%' "
                     f"TO '{store.test_roles['source']}'@'%'")
            _execute(client, "SET ROLE ALL")
            with pytest.raises(ValueError, match="changed|administration"):
                store.check_source(client)
            with pytest.raises(pymysql.Error, match="changed"):
                with manager.open(reference):
                    pytest.fail("unsafe source role reached retained rows")
        finally:
            _execute(client, "SET ROLE NONE")
            if indirect:
                _execute(admin, f"DROP ROLE IF EXISTS '{bridge}'@'%'")
            _execute(admin, f"DROP ROLE IF EXISTS '{parent}'@'%'")
            store.release(record)


@pytest.mark.parametrize("mode", ["ANSI_QUOTES", "NO_BACKSLASH_ESCAPES", "ANSI"])
@pytest.mark.parametrize("query_name", ["primary", "audit"])
def test_mysql_sources_refuse_incompatible_sql_grammar(
    mysql_source, tmp_path, monkeypatch, mode, query_name,
):
    import pymysql
    from pymysql.constants import SERVER_STATUS

    from autoforecast.integrations.mysql import _execute, read_sources

    admin, reader, namespace = mysql_source
    with admin.cursor() as cursor:
        cursor.execute(f"CREATE TABLE `{namespace}`.sales (id INT, `a\\` INT) ENGINE=InnoDB")
        cursor.execute(f"INSERT INTO `{namespace}`.sales VALUES (1,1)")
        cursor.execute(f"CREATE TABLE `{namespace}`.hidden (id INT) ENGINE=MyISAM")
        cursor.execute(f"INSERT INTO `{namespace}`.hidden VALUES (999)")
    original_mode = _execute(reader, "SELECT @@session.sql_mode")[0][0]
    original_timezone = _execute(reader, "SELECT @@session.time_zone")[0][0]
    try:
        _execute(reader, "SET SESSION time_zone='+05:30'")
        _execute(reader, "SET SESSION sql_mode=%s", (original_mode + "," + mode,))
        settings = _execute(reader, "SELECT @@session.sql_mode,@@session.time_zone")
        assert set(settings[0][0].split(",")) & {"ANSI_QUOTES", "NO_BACKSLASH_ESCAPES"}
        if mode == "NO_BACKSLASH_ESCAPES":
            query = (f"SELECT id FROM sales WHERE 'a\\' UNION ALL SELECT id FROM "
                     f"`{namespace}`.hidden -- ' = 'a'\n")
        else:
            query = (f'SELECT id FROM sales WHERE "a\\" UNION ALL SELECT id FROM '
                     f'`{namespace}`.hidden -- " = "a"\n')
        queries = {"primary": "SELECT id FROM sales", query_name: query}
        executed = []
        execute = pymysql.cursors.Cursor.execute

        def observe(self, statement, *args, **kwargs):
            executed.append(statement)
            return execute(self, statement, *args, **kwargs)

        monkeypatch.setattr(pymysql.cursors.Cursor, "execute", observe)
        with pytest.raises(ValueError, match="ANSI_QUOTES.*NO_BACKSLASH_ESCAPES"):
            read_sources(reader, queries=queries, tables={"sales": namespace + ".sales"},
                         output_root=tmp_path / "capture")
        assert not (tmp_path / "capture").exists()
        assert not any("FROM" in statement.upper() for statement in executed)
        assert _execute(reader, "SELECT @@session.sql_mode,@@session.time_zone") == settings
        assert reader.get_autocommit()
        assert not reader.server_status & SERVER_STATUS.SERVER_STATUS_IN_TRANS
    finally:
        _execute(reader, "SET SESSION sql_mode=%s", (original_mode,))
        _execute(reader, "SET SESSION time_zone=%s", (original_timezone,))


@pytest.mark.parametrize("query_name", ["primary", "audit"])
@pytest.mark.parametrize("comment", [
    "/*!UNION ALL SELECT id FROM {hidden} */",
    "/*! UNION ALL SELECT id FROM {hidden} */",
    "/*!00000 UNION ALL SELECT id FROM {hidden} */",
    "/*!50700 UNION ALL SELECT id FROM {hidden} */",
    "/*!80000 UNION ALL SELECT id FROM {hidden} */",
    "/*!080000 UNION ALL SELECT id FROM {hidden} */",
    "/*!99999 UNION ALL SELECT id FROM {hidden} */",
    "/*!\nUNION ALL SELECT id FROM {hidden} */",
    "/*! FOR SHARE */",
    "/*! LOCK IN SHARE MODE */",
    "/*! FOR UPDATE */",
    "/*!80000 FOR SHARE */",
])
def test_mysql_sources_refuse_executable_comments(
    mysql_source, tmp_path, monkeypatch, query_name, comment,
):
    import pymysql
    from pymysql.constants import SERVER_STATUS

    from autoforecast.integrations.mysql import _execute, read_sources

    admin, reader, namespace = mysql_source
    with admin.cursor() as cursor:
        cursor.execute(f"CREATE TABLE `{namespace}`.sales (id INT) ENGINE=InnoDB")
        cursor.execute(f"INSERT INTO `{namespace}`.sales VALUES (1)")
        cursor.execute(f"CREATE TABLE `{namespace}`.hidden (id INT) ENGINE=MyISAM")
        cursor.execute(f"INSERT INTO `{namespace}`.hidden VALUES (999)")
    settings = _execute(reader, "SELECT @@session.sql_mode,@@session.time_zone")
    query = "SELECT id FROM sales " + comment.format(hidden=f"`{namespace}`.hidden")
    queries = {"primary": "SELECT id FROM sales", query_name: query}
    executed = []
    execute = pymysql.cursors.Cursor.execute

    def observe(self, statement, *args, **kwargs):
        executed.append(statement)
        return execute(self, statement, *args, **kwargs)

    monkeypatch.setattr(pymysql.cursors.Cursor, "execute", observe)
    with pytest.raises(ValueError, match="executable"):
        read_sources(reader, queries=queries, tables={"sales": namespace + ".sales"},
                     output_root=tmp_path / "capture")
    assert not (tmp_path / "capture").exists()
    assert not any("FROM" in statement.upper() for statement in executed)
    assert _execute(reader, "SELECT @@session.sql_mode,@@session.time_zone") == settings
    assert reader.get_autocommit()
    assert not reader.server_status & SERVER_STATUS.SERVER_STATUS_IN_TRANS


@pytest.mark.parametrize("indirect", [False, True])
def test_mysql_sources_accept_read_only_role_closure(mysql_store, indirect):
    import uuid

    from autoforecast.integrations.mysql import _execute

    store = mysql_store
    admin = store.test_admin
    parent = "af_probe_read_" + uuid.uuid4().hex[:16]
    bridge = "af_probe_bridge_" + uuid.uuid4().hex[:16]
    source = store.test_roles["source"]
    client = store.test_clients["source"]
    try:
        _execute(admin, f"CREATE ROLE '{parent}'@'%'")
        _execute(admin, f"GRANT SELECT ON `{store.database}`.* TO '{parent}'@'%'")
        if indirect:
            _execute(admin, f"CREATE ROLE '{bridge}'@'%'")
            _execute(admin, f"GRANT '{parent}'@'%' TO '{bridge}'@'%'")
        _execute(admin, f"GRANT '{bridge if indirect else parent}'@'%' TO '{source}'@'%'")
        _execute(client, "SET ROLE ALL")
        store.check_source(client)
        store.preflight()
    finally:
        _execute(client, "SET ROLE NONE")
        if indirect:
            _execute(admin, f"DROP ROLE IF EXISTS '{bridge}'@'%'")
        _execute(admin, f"DROP ROLE IF EXISTS '{parent}'@'%'")


@pytest.mark.parametrize("indirect", [False, True])
def test_mysql_sources_refuse_inherited_registration_identity(mysql_store, indirect):
    import uuid

    from autoforecast.integrations.mysql import _execute

    store = mysql_store
    admin = store.test_admin
    parent = store.test_roles["writer"]
    bridge = "af_probe_bridge_" + uuid.uuid4().hex[:16]
    source = store.test_roles["source"]
    assigned = bridge if indirect else parent
    try:
        if indirect:
            _execute(admin, f"CREATE ROLE '{bridge}'@'%'")
            _execute(admin, f"GRANT '{parent}'@'%' TO '{bridge}'@'%'")
        _execute(admin, f"GRANT '{assigned}'@'%' TO '{source}'@'%'")
        with pytest.raises(ValueError, match="source account has archive administration rights"):
            store.check_source(store.test_clients["source"])
    finally:
        _execute(admin, f"REVOKE '{assigned}'@'%' FROM '{source}'@'%'")
        if indirect:
            _execute(admin, f"DROP ROLE IF EXISTS '{bridge}'@'%'")


def test_mysql_archive_refuses_mandatory_roles(mysql_store):
    import uuid

    import pymysql

    from autoforecast.integrations.mysql import _execute

    store = mysql_store
    admin = store.test_admin
    role = "af_probe_required_" + uuid.uuid4().hex[:14]
    original = _execute(admin, "SELECT @@global.mandatory_roles")[0][0]
    try:
        _execute(admin, f"CREATE ROLE '{role}'@'%'")
        _execute(admin, "SET GLOBAL mandatory_roles=%s", (f"'{role}'@'%'",))
        with pytest.raises(pymysql.Error, match="archive account isolation changed"):
            store.preflight()
        with pytest.raises(ValueError, match="archive account isolation changed"):
            store.check_source(store.test_clients["source"])
    finally:
        _execute(admin, "SET GLOBAL mandatory_roles=%s", (original,))
        _execute(admin, f"DROP ROLE IF EXISTS '{role}'@'%'")


def test_mysql_read_only_archive_reader_can_be_a_source(mysql_store):
    mysql_store.preflight()
    mysql_store.check_source(mysql_store.reader)


@pytest.mark.parametrize("details", [(), (2013, "source policy connection interrupted")])
def test_mysql_source_check_preserves_provider_failures(mysql_store, monkeypatch, details):
    import pymysql

    error = pymysql.OperationalError(*details)

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(mysql_store, "_call", fail)
    with pytest.raises(pymysql.OperationalError) as caught:
        mysql_store.check_source(mysql_store.test_clients["source"])
    assert caught.value is error


def _assert_sqlserver_permission_drift_refused(
    store, typed_input, tmp_path, grants, revokes, *, actor=None, identities=(),
):
    import pyarrow.parquet as pq

    from autoforecast.integrations.sqlserver import _execute
    from autoforecast.snapshots import RetainedInputs

    with RetainedInputs(mode="table", connection="archive", store=store,
                        scratch_root=tmp_path) as manager:
        reference = manager.capture(typed_input[0])
        record = reference.record()
        store.retain(record)
        try:
            for statement in grants:
                _execute(store.test_admin, statement)
            if actor is not None:
                client = store.test_clients[actor]
                entered = 0
                try:
                    for identity in identities:
                        _execute(client, "EXECUTE AS USER=N'" + identity + "'")
                        entered += 1
                    assert _execute(client, "SELECT HAS_PERMS_BY_NAME(?,'OBJECT','UPDATE')",
                                    (record["object"],)) == [(1,)]
                finally:
                    for _ in range(entered):
                        _execute(client, "REVERT")
            with pytest.raises(ValueError, match="mutation|roles|rights|ordinary"):
                manager.capture(typed_input[0])
            with pytest.raises(ValueError, match="mutation|roles|rights|ordinary"):
                store.check_source(store.test_clients["source"])
            with pytest.raises(ValueError, match="mutation|roles|rights|ordinary"):
                with manager.open(reference):
                    pytest.fail("replay admitted unsafe principal authority")
            assert _execute(store.reader, "SELECT COUNT(*) FROM archive.af_captures") == [(1,)]
        finally:
            for statement in revokes:
                _execute(store.test_admin, statement)
        with manager.open(reference) as path:
            actual = pq.read_table(path)
            assert actual.schema.equals(typed_input[1].schema)
            assert sorted(actual.to_pylist(), key=repr) == sorted(
                typed_input[1].to_pylist(), key=repr,
            )
        store.release(record)


@pytest.mark.parametrize("actor", ["source", "writer", "reader"])
@pytest.mark.parametrize("permission", ["IMPERSONATE", "CONTROL"])
@pytest.mark.parametrize("target", ["dbo", "db_owner_member"])
def test_sqlserver_refuses_administrator_user_impersonation(
    sqlserver_store, typed_input, tmp_path, actor, permission, target,
):
    import uuid

    from autoforecast.integrations.sqlserver import _execute

    store = sqlserver_store
    target_name = "dbo"
    if target == "db_owner_member":
        target_name = "af_probe_owner_" + uuid.uuid4().hex[:16]
        _execute(store.test_admin, f"CREATE USER [{target_name}] WITHOUT LOGIN")
        _execute(store.test_admin, f"ALTER ROLE db_owner ADD MEMBER [{target_name}]")
    member = store.test_roles[actor]
    _assert_sqlserver_permission_drift_refused(
        store, typed_input, tmp_path,
        [f"GRANT {permission} ON USER::[{target_name}] TO [{member}]"],
        [f"REVOKE {permission} ON USER::[{target_name}] FROM [{member}]"],
        actor=actor, identities=(target_name,),
    )


@pytest.mark.parametrize("actor", ["source", "writer", "reader"])
@pytest.mark.parametrize("permission", ["IMPERSONATE", "CONTROL"])
@pytest.mark.parametrize("route", ["intermediary", "role", "public"])
def test_sqlserver_refuses_indirect_user_impersonation(
    sqlserver_store, typed_input, tmp_path, actor, permission, route,
):
    import uuid

    from autoforecast.integrations.sqlserver import _execute

    store = sqlserver_store
    member = store.test_roles[actor]
    bridge = "af_probe_bridge_" + uuid.uuid4().hex[:16]
    role = "af_probe_role_" + uuid.uuid4().hex[:16]
    if route == "public":
        grants = [f"GRANT {permission} ON USER::dbo TO public"]
        revokes = [f"REVOKE {permission} ON USER::dbo FROM public"]
        identities = ("dbo",)
    else:
        _execute(store.test_admin, f"CREATE USER [{bridge}] WITHOUT LOGIN")
        grants = [f"GRANT IMPERSONATE ON USER::[{bridge}] TO [{member}]"]
        revokes = [f"REVOKE IMPERSONATE ON USER::[{bridge}] FROM [{member}]"]
        recipient = bridge
        if route == "role":
            _execute(store.test_admin, f"CREATE ROLE [{role}]")
            _execute(store.test_admin, f"ALTER ROLE [{role}] ADD MEMBER [{bridge}]")
            recipient = role
        grants.append(f"GRANT {permission} ON USER::dbo TO [{recipient}]")
        revokes.insert(0, f"REVOKE {permission} ON USER::dbo FROM [{recipient}]")
        identities = (bridge, "dbo")
    _assert_sqlserver_permission_drift_refused(
        store, typed_input, tmp_path, grants, revokes, actor=actor, identities=identities,
    )


@pytest.mark.parametrize("target,permission", [
    ("USER", "ALTER"), ("ROLE", "ALTER"), ("ROLE", "TAKE OWNERSHIP"),
])
def test_sqlserver_refuses_mutating_security_principals(
    sqlserver_store, typed_input, tmp_path, permission, target,
):
    import uuid

    from autoforecast.integrations.sqlserver import _execute

    store = sqlserver_store
    name = "af_probe_principal_" + uuid.uuid4().hex[:16]
    _execute(store.test_admin, f"CREATE {target} [{name}]" + (
        " WITHOUT LOGIN" if target == "USER" else ""))
    source = store.test_roles["source"]
    _assert_sqlserver_permission_drift_refused(
        store, typed_input, tmp_path,
        [f"GRANT {permission} ON {target}::[{name}] TO [{source}]"],
        [f"REVOKE {permission} ON {target}::[{name}] FROM [{source}]"],
    )


@pytest.mark.parametrize("authority", [
    "login_impersonate", "login_control", "server_control", "alter_login",
    "impersonate_any_login", "alter_server_role", "alter_database", "server_role",
    "nested_server_role", "public_login",
])
def test_sqlserver_refuses_source_server_authority(
    sqlserver_store, typed_input, tmp_path, authority,
):
    import uuid

    from autoforecast.integrations.sqlserver import _execute

    store = sqlserver_store
    source = store.test_roles["source"]
    release = store.test_roles["release"]
    database = store._database(store.reader)["name"]
    role = "af_probe_server_role_" + uuid.uuid4().hex[:16]
    bridge = "af_probe_server_bridge_" + uuid.uuid4().hex[:16]
    grants, revokes = ["USE master"], ["USE master"]
    if authority in {"server_role", "nested_server_role"}:
        _execute(store.test_admin, "USE master")
        try:
            _execute(store.test_admin, f"CREATE SERVER ROLE [{role}]")
            _execute(store.test_admin,
                     f"GRANT IMPERSONATE ON LOGIN::[{release}] TO [{role}]")
            recipient = role
            if authority == "nested_server_role":
                _execute(store.test_admin, f"CREATE SERVER ROLE [{bridge}]")
                _execute(store.test_admin, f"ALTER SERVER ROLE [{role}] ADD MEMBER [{bridge}]")
                recipient = bridge
            grants.append(f"ALTER SERVER ROLE [{recipient}] ADD MEMBER [{source}]")
            revokes.append(f"ALTER SERVER ROLE [{recipient}] DROP MEMBER [{source}]")
            if authority == "nested_server_role":
                revokes.extend([f"ALTER SERVER ROLE [{role}] DROP MEMBER [{bridge}]",
                                f"DROP SERVER ROLE [{bridge}]"])
            revokes.append(f"DROP SERVER ROLE [{role}]")
        finally:
            _execute(store.test_admin, f"USE [{database}]")
    elif authority in {"login_impersonate", "login_control", "public_login"}:
        permission = "CONTROL" if authority == "login_control" else "IMPERSONATE"
        recipient = "public" if authority == "public_login" else source
        grants.append(f"GRANT {permission} ON LOGIN::[{release}] TO [{recipient}]")
        revokes.append(f"REVOKE {permission} ON LOGIN::[{release}] FROM [{recipient}]")
    else:
        permission = {"server_control": "CONTROL SERVER", "alter_login": "ALTER ANY LOGIN",
                      "impersonate_any_login": "IMPERSONATE ANY LOGIN",
                      "alter_server_role": "ALTER ANY SERVER ROLE",
                      "alter_database": "ALTER ANY DATABASE"}[authority]
        grants.append(f"GRANT {permission} TO [{source}]")
        revokes.append(f"REVOKE {permission} FROM [{source}]")
    grants.append(f"USE [{database}]")
    revokes.append(f"USE [{database}]")
    _assert_sqlserver_permission_drift_refused(store, typed_input, tmp_path, grants, revokes)


@pytest.mark.parametrize("permission", ["IMPERSONATE", "CONTROL", "ALTER"])
@pytest.mark.parametrize("inherited", [False, True])
def test_sqlserver_checks_effective_source_login_authority(
    sqlserver_store, permission, inherited,
):
    import os
    import uuid

    from mssql_python import connect

    from autoforecast.integrations.sqlserver import _execute

    store = sqlserver_store
    database = store._database(store.reader)["name"]
    source = "af_probe_source_" + uuid.uuid4().hex[:16]
    role = "af_probe_server_role_" + uuid.uuid4().hex[:16]
    release = store.test_roles["release"]
    joined = False
    _execute(store.test_admin, "USE master")
    _execute(store.test_admin, f"CREATE LOGIN [{source}] "
             "WITH PASSWORD='Local-source-test-only-2026!'")
    try:
        if inherited:
            _execute(store.test_admin, f"CREATE SERVER ROLE [{role}]")
        _execute(store.test_admin, f"USE [{database}]")
        _execute(store.test_admin, f"CREATE USER [{source}] FOR LOGIN [{source}]")
        _execute(store.test_admin, f"GRANT CONNECT TO [{source}]")
        with connect(os.environ["AUTOFORECAST_TEST_INPUT_SQLSERVER_DSN"], autocommit=True,
                     UID=source, PWD="Local-source-test-only-2026!", DATABASE="master") as client:
            store.check_source(client)
        _execute(store.test_admin, "USE master")
        recipient = role if inherited else source
        _execute(store.test_admin, f"GRANT {permission} ON LOGIN::[{release}] TO [{recipient}]")
        if inherited:
            _execute(store.test_admin, f"ALTER SERVER ROLE [{role}] ADD MEMBER [{source}]")
            joined = True
        with connect(os.environ["AUTOFORECAST_TEST_INPUT_SQLSERVER_DSN"], autocommit=True,
                     UID=source, PWD="Local-source-test-only-2026!", DATABASE="master") as client:
            assert _execute(client, "SELECT HAS_PERMS_BY_NAME(?,'LOGIN',?)",
                            (release, permission)) == [(1,)]
            with pytest.raises(ValueError, match="ordinary|roles|rights"):
                store.check_source(client)
    finally:
        _execute(store.test_admin, "USE master")
        if joined:
            _execute(store.test_admin, f"ALTER SERVER ROLE [{role}] DROP MEMBER [{source}]")
        if inherited:
            _execute(store.test_admin, f"DROP SERVER ROLE [{role}]")
        _execute(store.test_admin, f"USE [{database}]")
        _execute(store.test_admin, f"DROP USER [{source}]")
        _execute(store.test_admin, "USE master")
        _execute(store.test_admin, f"DROP LOGIN [{source}]")
        _execute(store.test_admin, f"USE [{database}]")


def test_sqlserver_allows_read_only_principal_metadata(sqlserver_store):
    from autoforecast.integrations.sqlserver import _execute

    store = sqlserver_store
    source = store.test_roles["source"]
    _execute(store.test_admin, f"GRANT VIEW DEFINITION ON USER::dbo TO [{source}]")
    store.preflight()
    store.check_source(store.test_clients["source"])


def test_sqlserver_refuses_impersonated_source_sessions(sqlserver_store):
    from autoforecast.integrations.sqlserver import _execute

    store = sqlserver_store
    source = store.test_roles["source"]
    _execute(store.test_admin, f"EXECUTE AS LOGIN=N'{source}'")
    try:
        with pytest.raises(ValueError, match="ordinary source account"):
            store.check_source(store.test_admin)
    finally:
        _execute(store.test_admin, "REVERT")


@pytest.mark.parametrize("indirect", [False, True])
def test_sqlserver_refuses_source_database_owner_roles(
    sqlserver_store, typed_input, tmp_path, indirect,
):
    import uuid

    from autoforecast.integrations.sqlserver import _execute

    store = sqlserver_store
    source = store.test_roles["source"]
    bridge = "af_probe_database_role_" + uuid.uuid4().hex[:16]
    grants, revokes = [], []
    recipient = source
    if indirect:
        _execute(store.test_admin, f"CREATE ROLE [{bridge}]")
        grants.append(f"ALTER ROLE [{bridge}] ADD MEMBER [{source}]")
        revokes.append(f"ALTER ROLE [{bridge}] DROP MEMBER [{source}]")
        recipient = bridge
    grants.append(f"ALTER ROLE db_owner ADD MEMBER [{recipient}]")
    revokes.insert(0, f"ALTER ROLE db_owner DROP MEMBER [{recipient}]")
    _assert_sqlserver_permission_drift_refused(
        store, typed_input, tmp_path, grants, revokes, actor="source",
    )


@pytest.mark.parametrize('backend', ['postgresql', 'mysql', 'sqlserver'])
def test_source_capture_preserves_rows_after_the_first_transport_batch(request, tmp_path, backend):
    expected = list(range(8193))
    if backend == 'postgresql':
        from psycopg import sql

        from autoforecast.integrations.postgres import read_sources
        store = request.getfixturevalue('postgres_store')
        connection = store.writer
        name = store.test_source_schema + '.sales'
        relation = sql.Identifier(*name.split('.'))
        with connection.transaction():
            connection.execute(sql.SQL('CREATE TABLE {} (id INT)').format(relation))
            with connection.cursor() as cursor:
                cursor.executemany(sql.SQL('INSERT INTO {} VALUES (%s)').format(relation),
                                   [(i,) for i in expected])
    elif backend == 'mysql':
        from autoforecast.integrations.mysql import read_sources
        admin, connection, namespace = request.getfixturevalue('mysql_source')
        name = namespace + '.sales'
        with admin.cursor() as cursor:
            cursor.execute(f'CREATE TABLE `{namespace}`.sales (id INT) ENGINE=InnoDB')
            cursor.executemany(f'INSERT INTO `{namespace}`.sales VALUES (%s)',
                               [(i,) for i in expected])
    else:
        from autoforecast.integrations.sqlserver import read_sources
        connection, writer, _, _ = request.getfixturevalue('sqlserver_database')
        name = 'dbo.sales'
        with writer.cursor() as cursor:
            cursor.execute('CREATE TABLE dbo.sales (id INT)')
            cursor.execute('BEGIN TRANSACTION')
            cursor.executemany('INSERT INTO dbo.sales VALUES (?)', [(i,) for i in expected])
            cursor.execute('COMMIT')
    result = read_sources(connection, queries={'primary':'SELECT id FROM sales'},
                          tables={'sales':name}, output_root=tmp_path/'capture')
    actual = sorted(pq.read_table(result['primary'][0])['id'].to_pylist())
    print('FULL_INPUT_CHECKED', backend, 'expected', len(expected), 'actual', len(actual))
    assert actual == expected, 'Source capture lost or changed observations across batch boundaries'


@pytest.mark.parametrize('store_fixture', ['postgres_store','mysql_store','sqlserver_store'])
def test_native_roundtrip_preserves_multiple_files_and_batches(request, tmp_path, store_fixture):
    store = request.getfixturevalue(store_fixture)
    expected = pa.table({'id':list(range(8195)) + [17],
                         'value':[float(i)+0.25 for i in range(8195)] + [17.25]})
    source = tmp_path/'input'
    source.mkdir()
    pq.write_table(expected.slice(0,8193), source/'one.parquet')
    pq.write_table(expected.slice(8193), source/'two.parquet')
    with RetainedInputs(mode='table', connection='archive', store=store,
                        scratch_root=tmp_path) as manager:
        reference = manager.capture(source)
        try:
            store.retain(reference.record())
            with manager.open(reference) as retained:
                actual = pq.read_table(retained)
                assert actual.schema == expected.schema
                assert sorted(actual.to_pylist(), key=repr) == sorted(
                    expected.to_pylist(), key=repr)
        finally:
            store.release(reference.record())


@pytest.mark.parametrize("query_name", ["primary", "audit"])
@pytest.mark.parametrize("query,expected", [
    ("SELECT /* ordinary selection */ id FROM sales /* ordinary source */", {"id": 1}),
    ("SELECT id FROM sales -- ordinary line comment\n", {"id": 1}),
    ("SELECT id FROM sales # ordinary line comment\n", {"id": 1}),
    ("SELECT id, '/*! UNION ALL SELECT 999 */' AS note FROM sales",
     {"id": 1, "note": "/*! UNION ALL SELECT 999 */"}),
    ('SELECT id, "/*! FOR UPDATE */" AS note FROM sales',
     {"id": 1, "note": "/*! FOR UPDATE */"}),
    ("SELECT id, 'a\\' /*! FOR SHARE */ z' AS note FROM sales",
     {"id": 1, "note": "a' /*! FOR SHARE */ z"}),
    ("SELECT id, '/*M! UNION ALL SELECT 999 */' AS note FROM sales",
     {"id": 1, "note": "/*M! UNION ALL SELECT 999 */"}),
    ("SELECT id, '/*m! FOR SHARE */' AS note FROM sales",
     {"id": 1, "note": "/*m! FOR SHARE */"}),
])
def test_mysql_sources_preserve_ordinary_comments_and_executable_looking_strings(
    mysql_source, tmp_path, query_name, query, expected,
):
    import pyarrow.parquet as pq
    from pymysql.constants import SERVER_STATUS

    from autoforecast.integrations.mysql import _execute, read_sources

    admin, reader, namespace = mysql_source
    with admin.cursor() as cursor:
        cursor.execute(f"CREATE TABLE `{namespace}`.sales (id INT) ENGINE=InnoDB")
        cursor.execute(f"INSERT INTO `{namespace}`.sales VALUES (1)")
    settings = _execute(reader, "SELECT @@session.sql_mode,@@session.time_zone")
    queries = {"primary": "SELECT id FROM sales", query_name: query}
    result = read_sources(reader, queries=queries, tables={"sales": namespace + ".sales"},
                          output_root=tmp_path / "capture")
    assert set(result) == set(queries)
    for name, (path, provenance) in result.items():
        assert pq.read_table(path).to_pylist() == [expected if name == query_name else {"id": 1}]
        assert provenance["sql"] == queries[name]
    assert _execute(reader, "SELECT @@session.sql_mode,@@session.time_zone") == settings
    assert reader.get_autocommit()
    assert not reader.server_status & SERVER_STATUS.SERVER_STATUS_IN_TRANS



@pytest.fixture(scope="module")
def lifecycle_spark():
    from pyspark.sql import SparkSession

    session = (SparkSession.builder.master("local[2]").appName("native-owned-lifecycle")
               .config("spark.sql.shuffle.partitions", "2")
               .config("spark.ui.enabled", "false").getOrCreate())
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def _lifecycle_registry(store, spark, path, *, mode="table"):
    from autoforecast.study_registration import StudyRegistry

    return StudyRegistry(path, spark=spark, snapshot_mode=mode,
                         connections={"archive": store},
                         snapshots={"connection": "archive", "namespace": store.namespace})


@pytest.mark.parametrize("mode", ["table", "table+hash"])
@pytest.mark.parametrize("prepared", [False, True])
def test_interrupted_native_base_registration_releases_complete_inventory(
    postgres_store, lifecycle_spark, tmp_path, monkeypatch, mode, prepared,
):
    import hashlib
    import json
    import uuid

    import pytest
    from psycopg import sql

    from autoforecast import _logical_hash
    from autoforecast.benchmark import read_benchmark_manifest
    from autoforecast.snapshots import StudyLease
    from autoforecast.study_registration import _publication_lock, register_study
    from autoforecast.test_study_preparation import SensorPreparation
    from autoforecast.test_study_registration import configuration

    # The source has all 17 synthetic observations for two series; no series are excluded.
    registry = _lifecycle_registry(postgres_store, lifecycle_spark, tmp_path / "registry",
                                   mode=mode)
    active_refusals = []

    def stop_before_registration(definition, **kwargs):
        with pytest.raises(ValueError, match="active owner|active readers"):
            registry.release(definition["study_id"])
        assert not (registry.root / "released" / (definition["study_id"] + ".json")).exists()
        active_refusals.append(True)
        raise OSError("injected registration publication failure")

    monkeypatch.setattr(registry, "_publish_definition", stop_before_registration)
    if mode == "table":
        monkeypatch.setattr(_logical_hash, "fingerprint", lambda *a, **k: pytest.fail(
            "table-only recovery must not hash input rows"))
    try:
        with pytest.raises(OSError, match="injected registration"):
            register_study(configuration(tmp_path / "source"), registry=registry,
                           prepare=SensorPreparation() if prepared else None)
        assert active_refusals == [True]
        manifests = list((registry.root / "benchmarks").glob("*/benchmark.json"))
        assert len(manifests) == 1
        manifest, record = read_benchmark_manifest(manifests[0])
        study_id = record["study_id"]
        assert not list((registry.root / "registrations").glob(study_id + ".*"))
        owned = record["owned_snapshots"]
        retained = postgres_store.reader.execute(sql.SQL(
            "SELECT capture_id FROM {} WHERE state='retained'",
        ).format(sql.Identifier(postgres_store.schema, "af_captures"))).fetchall()
        assert set(owned) == {row[0] for row in retained}
        assert len(owned) > len({item["capture_id"] for item in record["snapshots"].values()})
        if prepared:
            admission = json.loads(next((registry.root / "admissions" / study_id).glob(
                "admission.json",
            )).read_bytes())
            assert admission["study_id"] == study_id
            assert {item["snapshot"]["capture_id"] for item in admission["artifacts"].values()
                    } <= set(owned)
        preserved = {path: path.read_bytes() for path in registry.root.rglob("*")
                     if path.is_file() and path.suffix in {".json", ".yaml", ".yml"}}
        reference = next(iter(owned.values()))
        with pytest.raises(ValueError, match="published"):
            registry.recover_capture(reference["capture_id"])
        lease = StudyLease(manifest, hashlib.sha256(manifest.read_bytes()).hexdigest())
        try:
            with pytest.raises(ValueError, match="active readers"):
                registry.release(study_id)
            assert not (registry.root / "released" / (study_id + ".json")).exists()
        finally:
            lease.close()
        journal = registry.root / "capture-journal" / reference["capture_id"] / "sealed.json"
        with _publication_lock(journal.parent, blocking=False):
            with pytest.raises(ValueError, match="active owner"):
                registry.release(study_id)
            assert not (registry.root / "released" / (study_id + ".json")).exists()
        original = journal.read_bytes()
        changed = json.loads(original)
        changed["incarnation"]["seal"] = str(uuid.uuid4())
        journal.write_text(json.dumps(changed))
        try:
            with pytest.raises(ValueError, match="original sealed ownership"):
                registry.release(study_id)
            assert not (registry.root / "released" / (study_id + ".json")).exists()
        finally:
            journal.write_bytes(original)
        assert registry.release(study_id) == study_id
        assert registry.release(study_id) == study_id
        assert postgres_store.reader.execute(sql.SQL("SELECT count(*) FROM {}").format(
            sql.Identifier(postgres_store.schema, "af_captures"),
        )).fetchone()[0] == 0
        for item in owned.values():
            assert postgres_store.reader.execute("SELECT to_regclass(%s)",
                                                (item["object"],)).fetchone()[0] is None
        assert all(path.read_bytes() == payload for path, payload in preserved.items())
        assert (registry.root / "released" / (study_id + ".done.json")).is_file()
    finally:
        registry.close()


def test_release_refuses_foreign_manifest_before_tombstone(
    postgres_store, lifecycle_spark, tmp_path,
):
    import hashlib
    import json

    import pytest

    from autoforecast.benchmark import digest, read_benchmark_manifest
    from autoforecast.study_registration import register_study
    from autoforecast.test_study_registration import configuration

    registry = _lifecycle_registry(postgres_store, lifecycle_spark, tmp_path / "registry")
    config = configuration(tmp_path / "source")
    try:
        first = register_study(config, registry=registry)
        second = register_study(config, registry=registry)
        first_manifest = registry.root / "benchmarks" / first / "benchmark.json"
        second_manifest = registry.root / "benchmarks" / second / "benchmark.json"
        _, first_record = read_benchmark_manifest(first_manifest)
        _, second_record = read_benchmark_manifest(second_manifest)
        registration = registry._registration_path(first)
        original_manifest = first_manifest.read_bytes()
        original_registration = registration.read_bytes()
        with registry.load_study(second):
            try:
                first_manifest.write_bytes(second_manifest.read_bytes())
                envelope = json.loads(original_registration)
                envelope["definition"]["library_sha256"] = hashlib.sha256(
                    second_manifest.read_bytes(),
                ).hexdigest()
                envelope["sha256"] = digest(envelope["definition"])
                registration.write_text(json.dumps(envelope))
                with pytest.raises(ValueError, match="library study identity differs"):
                    registry.release(first)
                assert not (registry.root / "released" / (first + ".json")).exists()
                for owner in (first_record, second_record):
                    for reference in owner["owned_snapshots"].values():
                        postgres_store.inspect(reference, retained=True)
            finally:
                first_manifest.write_bytes(original_manifest)
                registration.write_bytes(original_registration)
        registry.release(first)
        registry.release(second)
    finally:
        registry.close()


@pytest.mark.parametrize("site", [
    "native_findings", "admission_artifact", "admission_subject", "admission_schema",
])
def test_release_refuses_foreign_auxiliary_before_tombstone(
    postgres_store, lifecycle_spark, tmp_path, site,
):
    import hashlib
    import json

    import pytest

    from autoforecast.benchmark import digest, read_benchmark_manifest
    from autoforecast.study_registration import register_study
    from autoforecast.test_study_preparation import SensorPreparation
    from autoforecast.test_study_registration import configuration

    registry = _lifecycle_registry(postgres_store, lifecycle_spark, tmp_path / "registry")
    config = configuration(tmp_path / "source")
    try:
        first = register_study(config, registry=registry, prepare=SensorPreparation())
        second = register_study(config, registry=registry, prepare=SensorPreparation())
        registration = registry._registration_path(first)
        original_registration = registration.read_bytes()
        envelope = json.loads(original_registration)
        _, first_record = read_benchmark_manifest(
            registry.root / "benchmarks" / first / "benchmark.json",
        )
        _, second_record = read_benchmark_manifest(
            registry.root / "benchmarks" / second / "benchmark.json",
        )
        victim = second_record["snapshots"]["selection"]
        definition = envelope["definition"]
        if site == "native_findings":
            evidence = registry.root / definition["native_findings"][0]["path"]
            payload = json.dumps({"schema": "autoforecast.native-findings.v1",
                                  "snapshot": victim}).encode()
            definition["native_findings"][0].update(
                sha256=hashlib.sha256(payload).hexdigest(), snapshot=victim,
            )
        else:
            binding = definition["admission_evidence"]
            evidence = registry.root / binding["path"]
            admission = json.loads(evidence.read_bytes())
            if site == "admission_artifact":
                # Keep the correct schema, subject and admission identity. Only this
                # final artifact changes; envelope checksums alone cannot bind its owner.
                admission["artifacts"]["final_keys"] = {"snapshot": victim}
            elif site == "admission_subject":
                admission["study_id"] = second
            else:
                admission["schema"] = "autoforecast.admission.v1"
            payload = json.dumps(admission).encode()
            binding.update(sha256=digest(admission),
                           physical_sha256=hashlib.sha256(payload).hexdigest())
        original_evidence = evidence.read_bytes()
        envelope["sha256"] = digest(definition)
        with registry.load_study(second):
            try:
                evidence.write_bytes(payload)
                registration.write_text(json.dumps(envelope))
                with pytest.raises(ValueError, match="ownership|belong|artifact directory"):
                    registry.release(first)
                assert not (registry.root / "released" / (first + ".json")).exists()
                for owner in (first_record, second_record):
                    for reference in owner["owned_snapshots"].values():
                        postgres_store.inspect(reference, retained=True)
            finally:
                evidence.write_bytes(original_evidence)
                registration.write_bytes(original_registration)
        registry.release(first)
        registry.release(second)
    finally:
        registry.close()


def test_native_owned_inventory_release_resumes_after_partial_deletion(
    postgres_store, lifecycle_spark, tmp_path, monkeypatch,
):
    import json

    import pytest
    from psycopg import sql

    from autoforecast.benchmark import read_benchmark_manifest
    from autoforecast.study_registration import register_study
    from autoforecast.test_study_preparation import SensorPreparation
    from autoforecast.test_study_registration import configuration

    registry = _lifecycle_registry(postgres_store, lifecycle_spark, tmp_path / "registry")
    try:
        study_id = register_study(configuration(tmp_path / "source"), registry=registry,
                                  prepare=SensorPreparation())
        manifest = registry.root / "benchmarks" / study_id / "benchmark.json"
        _, record = read_benchmark_manifest(manifest)
        release = postgres_store.release
        calls = []

        def lose_first_acknowledgement(reference):
            release(reference)
            calls.append(reference["capture_id"])
            if len(calls) == 1:
                raise ConnectionError("synthetic lost release acknowledgement")

        monkeypatch.setattr(postgres_store, "release", lose_first_acknowledgement)
        with pytest.raises(ConnectionError, match="lost release"):
            registry.release(study_id)
        tombstone = registry.root / "released" / (study_id + ".json")
        original = tombstone.read_bytes()
        removal = json.loads(original)
        assert {item["capture_id"] for item in removal["snapshots"]} == set(
            record["owned_snapshots"],
        )
        assert not (registry.root / "released" / (study_id + ".done.json")).exists()
        changed = {**removal, "snapshots": removal["snapshots"][1:]}
        tombstone.write_text(json.dumps(changed))
        try:
            with pytest.raises(ValueError, match="release record differs"):
                registry.release(study_id)
            assert len(calls) == 1
        finally:
            tombstone.write_bytes(original)
        assert registry.release(study_id) == study_id
        assert registry.release(study_id) == study_id
        assert len(calls) == len(record["owned_snapshots"]) + 1
        assert postgres_store.reader.execute(sql.SQL("SELECT count(*) FROM {}").format(
            sql.Identifier(postgres_store.schema, "af_captures"),
        )).fetchone()[0] == 0
    finally:
        registry.close()


def test_release_refuses_substituted_dependency_manifest(
    postgres_store, lifecycle_spark, tmp_path,
):
    import pytest

    from autoforecast.benchmark import read_benchmark_manifest
    from autoforecast.study_preflight import preflight_study
    from autoforecast.study_registration import register_study
    from autoforecast.test_study_registration import configuration

    registry = _lifecycle_registry(postgres_store, lifecycle_spark, tmp_path / "registry")
    config = configuration(tmp_path / "source")
    try:
        parent_id = register_study(config, registry=registry)
        other_id = register_study(config, registry=registry)
        with registry.load_study(parent_id) as parent:
            result = preflight_study(parent, {
                "series_policy": "use_available", "minimum_history": 1,
                "metrics": ["mae"], "metric": "mae", "selection_boundaries": [],
            }, output_dir=tmp_path / "preflight")
            assert result.study_id != parent_id
        other_path = registry.root / "benchmarks" / other_id / "benchmark.json"
        parent_path = registry.root / "benchmarks" / parent_id / "benchmark.json"
        _, other_record = read_benchmark_manifest(other_path)
        _, parent_record = read_benchmark_manifest(parent_path)
        original = result.study_path.read_bytes()
        try:
            result.study_path.write_bytes(other_path.read_bytes())
            with pytest.raises(ValueError, match="dependent library study identity differs"):
                registry.release(parent_id)
            assert not (registry.root / "released" / (parent_id + ".json")).exists()
            for owner in (parent_record, other_record):
                for reference in owner["owned_snapshots"].values():
                    postgres_store.inspect(reference, retained=True)
        finally:
            result.study_path.write_bytes(original)
        registry.release(result.study_id)
        registry.release(parent_id)
        registry.release(other_id)
    finally:
        registry.close()


@pytest.mark.parametrize("mode", ["table", "table+hash"])
def test_prepared_reusable_native_source_keeps_its_original_owner(
    postgres_store, lifecycle_spark, tmp_path, monkeypatch, mode,
):
    import json

    import pytest
    import yaml

    from autoforecast import _logical_hash
    from autoforecast.benchmark import read_benchmark_manifest
    from autoforecast.study_registration import register_study
    from autoforecast.test_study_preparation import SensorPreparation
    from autoforecast.test_study_registration import configuration

    # All 17 source observations and both series are reused; preparation excludes none.
    registry = _lifecycle_registry(postgres_store, lifecycle_spark, tmp_path / "registry",
                                   mode=mode)
    config = configuration(tmp_path / "source")
    if mode == "table":
        monkeypatch.setattr(_logical_hash, "fingerprint", lambda *a, **k: pytest.fail(
            "table-only prepared reuse must not hash input rows"))
    try:
        capture_id = registry.capture_sources(config)
        captured, _, _ = registry._source_capture(capture_id)
        raw = captured["inputs"]["primary"]["snapshot"]
        value = yaml.safe_load(config.read_text())
        value["source"] = {"kind": "capture", "capture_id": capture_id}
        config.write_text(yaml.safe_dump(value))
        study_id = register_study(config, registry=registry, prepare=SensorPreparation())
        definition, _ = registry.definition(study_id)
        _, record = read_benchmark_manifest(
            registry.root / "benchmarks" / study_id / "benchmark.json",
        )
        assert definition["source"]["snapshot"] == raw
        assert raw["capture_id"] not in record["owned_snapshots"]
        admission_path = registry.root / definition["admission_evidence"]["path"]
        admission = json.loads(admission_path.read_bytes())
        assert all(item["snapshot"]["capture_id"] in record["owned_snapshots"]
                   for item in admission["artifacts"].values())
        with registry.load_study(study_id) as study:
            assert study.n_series == 2
            assert study.testing_data.count() == 4
            with pytest.raises(ValueError, match="dependent study"):
                registry.release(capture_id)
        registry.release(study_id)
        postgres_store.inspect(raw, retained=True)
        assert registry.release(capture_id) == capture_id
        with pytest.raises(FileNotFoundError):
            postgres_store.inspect(raw)
    finally:
        registry.close()


def _sqlserver_source_session_state(connection):
    from autoforecast.integrations.sqlserver import _execute

    return connection.autocommit, _execute(
        connection,
        "SELECT DB_NAME(),@@TRANCOUNT,@@LOCK_TIMEOUT,@@OPTIONS,transaction_isolation_level "
        "FROM sys.dm_exec_sessions WHERE session_id=@@SPID",
    )


def test_sqlserver_requires_source_archive_user_mapping(sqlserver_store):
    from autoforecast.integrations.sqlserver import _execute

    store = sqlserver_store
    source = store.test_clients["source"]
    _execute(source, "USE master")
    _execute(store.test_admin, f"DROP USER [{store.test_roles['source']}]")
    before = [_sqlserver_source_session_state(client) for client in (source, store.reader)]
    with pytest.raises(ValueError, match="source user mapping.*archive"):
        store.check_source(source)
    assert not store._source_users
    assert [_sqlserver_source_session_state(client) for client in (source, store.reader)] == before


def test_sqlserver_separate_source_database_needs_only_archive_connect(
    sqlserver_store, sqlserver_database, tmp_path, monkeypatch,
):
    import json

    import pyarrow.parquet as pq

    from autoforecast import _logical_hash
    from autoforecast.integrations.sqlserver import _execute, read_sources
    from autoforecast.snapshots import RetainedInputs

    store = sqlserver_store
    source, writer, _, _ = sqlserver_database
    login = _execute(source, "SELECT ORIGINAL_LOGIN()")[0][0]
    _execute(store.test_admin, f"CREATE USER [{login}] FOR LOGIN [{login}]")
    _execute(store.test_admin, f"GRANT CONNECT TO [{login}]")
    _execute(writer, "CREATE TABLE dbo.observations (id BIGINT,value BIGINT)")
    _execute(writer, "INSERT INTO dbo.observations VALUES(1,100),(2,200)")
    before = [_sqlserver_source_session_state(client) for client in (source, store.reader)]
    store.check_source(source)
    assert [_sqlserver_source_session_state(client) for client in (source, store.reader)] == before
    assert _execute(source, "SELECT DB_NAME()")[0][0] != store._database(store.reader)["name"]
    # Two synthetic observations, no exclusions, captured from the declared source table.
    captured, _ = read_sources(
        source, queries={"primary": "SELECT id,value FROM observations"},
        tables={"observations": "dbo.observations"}, output_root=tmp_path / "source",
    )["primary"]
    monkeypatch.setattr(_logical_hash, "fingerprint", lambda *a, **k: pytest.fail(
        "table-only source capture must not hash rows"))
    with RetainedInputs(mode="table", connection="archive", store=store,
                        scratch_root=tmp_path) as manager:
        reference = manager.capture(captured)
        record = reference.record()
        try:
            store.retain(record)
            assert len(record["incarnation"]["sources"]) == 1
            assert record["incarnation"]["sources"][0][0::2] == [login, login]
            stored = _execute(
                store.reader,
                "SELECT CONVERT(NVARCHAR(3750),value) FROM sys.extended_properties "
                "WHERE major_id=OBJECT_ID(?) AND name='autoforecast_sources'",
                (record["object"],),
            )
            assert json.loads(stored[0][0]) == record["incarnation"]["sources"]
            database = store._database(store.reader)["name"]
            with pytest.raises(Exception, match="denied|permission|Cannot find"):
                _execute(source, f"UPDATE [{database}]." + store._name(record) + " SET value=999")
            _execute(writer, "UPDATE dbo.observations SET value=999")
            with manager.open(reference) as path:
                assert sorted(pq.read_table(path).to_pylist(), key=lambda row: row["id"]) == [
                    {"id": 1, "value": 100}, {"id": 2, "value": 200},
                ]
        finally:
            store.release(record)


@pytest.mark.parametrize("failure", ["source_error", "unheld_resource", "release_error", "busy"])
def test_sqlserver_source_server_proof_is_nonblocking_and_always_releases(
    sqlserver_store, monkeypatch, failure,
):
    from autoforecast.integrations import sqlserver

    store = sqlserver_store
    source = store.test_clients["source"]
    execute = sqlserver._execute
    resources, releases = [], []
    admin_locked = False
    before = [_sqlserver_source_session_state(client) for client in (source, store.reader)]

    def probe(connection, statement, values=()):
        nonlocal admin_locked
        if connection is store.reader and "sp_getapplock" in statement:
            assert "@LockTimeout=0" in statement
            resources.append(values[0])
            if failure == "busy":
                assert execute(store.test_admin, statement, values) == [(0,)]
                admin_locked = True
        if connection is source and "APPLOCK_TEST" in statement:
            if failure == "source_error":
                raise ConnectionError("synthetic source identity failure")
            if failure == "unheld_resource":
                # Another DB/server cannot see the reader's random database-scoped lock.
                result = execute(connection, statement, (values[0] + "_unheld",))
                assert result == [(1,)]
                return result
        releasing = connection is store.reader and "sp_releaseapplock" in statement
        if releasing:
            releases.append(values[0])
        result = execute(connection, statement, values)
        if releasing and failure == "release_error":
            raise ConnectionError("synthetic identity lock acknowledgement failure")
        return result

    monkeypatch.setattr(sqlserver, "_execute", probe)
    error = ConnectionError if failure in {"source_error", "release_error"} else ValueError
    try:
        with pytest.raises(error, match="identity"):
            store.check_source(source)
        assert len(resources) == 1
        assert releases == resources
        assert execute(store.reader, "SELECT APPLOCK_MODE('public',?,'Session')",
                       (resources[0],)) == [("NoLock",)]
        assert not store._source_users
        after = [_sqlserver_source_session_state(client) for client in (source, store.reader)]
        assert after == before
    finally:
        monkeypatch.setattr(sqlserver, "_execute", execute)
        if admin_locked:
            assert execute(
                store.test_admin,
                "DECLARE @r INT; EXEC @r=sys.sp_releaseapplock @Resource=?,"
                "@LockOwner='Session',@DbPrincipal='public'; SELECT @r",
                (resources[0],),
            ) == [(0,)]
    store.check_source(source)
    assert len(store._source_users) == 1


@pytest.mark.parametrize("drift", ["drop", "replace", "privilege_after_drop"])
def test_sqlserver_retained_source_mapping_cannot_disappear(
    sqlserver_store, typed_input, tmp_path, drift,
):
    import pyarrow.parquet as pq

    from autoforecast.integrations.sqlserver import _execute
    from autoforecast.snapshots import RetainedInputs

    store = sqlserver_store
    source_name = store.test_roles["source"]
    source = store.test_clients["source"]
    database = store._database(store.reader)["name"]
    admin_login = _execute(store.test_admin, "SELECT ORIGINAL_LOGIN()")[0][0]
    store.check_source(source)
    with RetainedInputs(mode="table", connection="archive", store=store,
                        scratch_root=tmp_path) as manager:
        reference = manager.capture(typed_input[0])
        record = reference.record()
        store.retain(record)
        _execute(source, "USE master")
        _execute(store.test_admin, f"DROP USER [{source_name}]")
        if drift == "replace":
            _execute(store.test_admin, f"CREATE USER [{source_name}] WITHOUT LOGIN")
        elif drift == "privilege_after_drop":
            _execute(store.test_admin, "USE master")
            _execute(store.test_admin,
                     f"GRANT IMPERSONATE ON LOGIN::[{admin_login}] TO [{source_name}]")
            _execute(store.test_admin, f"USE [{database}]")
            assert _execute(source, "SELECT HAS_PERMS_BY_NAME(?,'LOGIN','IMPERSONATE')",
                            (admin_login,)) == [(1,)]
        try:
            with pytest.raises(ValueError, match="source.*mapping"):
                store.inspect(record)
            with pytest.raises(ValueError, match="source.*mapping"):
                manager.capture(typed_input[0])
            with pytest.raises(ValueError, match="source.*mapping"):
                with manager.open(reference):
                    pytest.fail("replay forgot its original source user")
            with pytest.raises(ValueError, match="source.*mapping"):
                store.release(record)
            with pytest.raises(ValueError, match="source.*mapping"):
                store.drop_unretained(record)
        finally:
            if drift == "privilege_after_drop":
                _execute(store.test_admin, "USE master")
                _execute(store.test_admin,
                         f"REVOKE IMPERSONATE ON LOGIN::[{admin_login}] FROM [{source_name}]")
                _execute(store.test_admin, f"USE [{database}]")
            if drift == "replace":
                _execute(store.test_admin, f"DROP USER [{source_name}]")
            _execute(store.test_admin, f"CREATE USER [{source_name}] FOR LOGIN [{source_name}]")
            _execute(store.test_admin, f"GRANT CONNECT TO [{source_name}]")
        with manager.open(reference) as path:
            assert sorted(pq.read_table(path).to_pylist(), key=repr) == sorted(
                typed_input[1].to_pylist(), key=repr,
            )
        store.release(record)


def test_sqlserver_source_binding_is_protected_and_required_on_all_paths(
    sqlserver_store, typed_input, tmp_path,
):
    import copy
    import json
    import uuid

    from autoforecast.integrations.sqlserver import _execute

    store = sqlserver_store
    store.check_source(store.test_clients["source"])
    record = store.capture(typed_input[0], capture_id=uuid.uuid4().hex)
    original = record["incarnation"]["sources"]
    assert len(original) == 1
    name = record["object"].split(".")[1]
    try:
        for actor in ("source", "writer", "reader"):
            with pytest.raises(Exception, match="denied|permission|Cannot find|not exist"):
                _execute(
                    store.test_clients[actor],
                    "EXEC sys.sp_updateextendedproperty @name=N'autoforecast_sources',"
                    "@value=N'[]',@level0type=N'SCHEMA',@level0name=N'archive',"
                    "@level1type=N'TABLE',@level1name=?", (name,),
                )
        for operation in ("inspect", "read", "release", "drop_unretained"):
            for mutation in ("omit", "empty"):
                changed = copy.deepcopy(record)
                if mutation == "omit":
                    changed["incarnation"].pop("sources")
                else:
                    changed["incarnation"]["sources"] = []
                destination = tmp_path / "untrusted-reference-read"
                with pytest.raises(ValueError, match="source identity|incarnation"):
                    if operation == "read":
                        store.read(changed, destination)
                    else:
                        getattr(store, operation)(changed)
                assert not destination.exists()
        stored = _execute(
            store.reader, "SELECT CONVERT(NVARCHAR(3750),value) FROM sys.extended_properties "
            "WHERE major_id=OBJECT_ID(?) AND name='autoforecast_sources'", (record["object"],),
        )
        assert json.loads(stored[0][0]) == original
        store.inspect(record)
    finally:
        store.drop_unretained(record)


def test_sqlserver_no_source_capture_stores_an_explicit_empty_binding(
    sqlserver_store, typed_input, tmp_path,
):
    import copy
    import json
    import uuid

    import pyarrow.parquet as pq

    from autoforecast.integrations.sqlserver import _execute

    store = sqlserver_store
    record = store.capture(typed_input[0], capture_id=uuid.uuid4().hex)
    try:
        assert record["incarnation"]["sources"] == []
        stored = _execute(
            store.reader, "SELECT CONVERT(NVARCHAR(3750),value) FROM sys.extended_properties "
            "WHERE major_id=OBJECT_ID(?) AND name='autoforecast_sources'", (record["object"],),
        )
        assert json.loads(stored[0][0]) == []
        changed = copy.deepcopy(record)
        changed["incarnation"].pop("sources")
        with pytest.raises(ValueError, match="source identity binding"):
            store.inspect(changed)
        destination = tmp_path / "valid-no-source"
        store.read(record, destination)
        assert sorted(pq.read_table(destination).to_pylist(), key=repr) == sorted(
            typed_input[1].to_pylist(), key=repr,
        )
        name = record["object"].split(".")[1]
        _execute(
            store.test_admin, "EXEC sys.sp_dropextendedproperty @name=N'autoforecast_sources',"
            "@level0type=N'SCHEMA',@level0name=N'archive',@level1type=N'TABLE',@level1name=?",
            (name,),
        )
        try:
            with pytest.raises(ValueError, match="source identity binding"):
                store.inspect(record)
        finally:
            _execute(
                store.test_admin,
                "EXEC sys.sp_addextendedproperty @name=N'autoforecast_sources',@value=N'[]',"
                "@level0type=N'SCHEMA',@level0name=N'archive',@level1type=N'TABLE',@level1name=?",
                (name,),
            )
    finally:
        store.drop_unretained(record)


@pytest.mark.parametrize("with_source,mutation", [(False, "omit"), (True, "omit"), (True, "empty")])
def test_sqlserver_checkpoint_cannot_remove_source_binding(
    sqlserver_store, typed_input, with_source, mutation,
):
    import uuid

    from autoforecast.integrations.sqlserver import _execute

    store = sqlserver_store
    if with_source:
        store.check_source(store.test_clients["source"])
    capture_id = uuid.uuid4().hex

    def replace_binding(record):
        if mutation == "omit":
            record["incarnation"].pop("sources", None)
        else:
            record["incarnation"]["sources"] = []

    with pytest.raises(ValueError, match="source identity binding|incarnation"):
        store.capture(typed_input[0], capture_id=capture_id, checkpoint=replace_binding)
    assert _execute(store.writer, "SELECT @@TRANCOUNT") == [(0,)]
    assert _execute(store.reader, "SELECT OBJECT_ID(?)", ("archive.af_input_" + capture_id,)) == [
        (None,),
    ]
    assert _execute(store.reader, "SELECT COUNT(*) FROM archive.af_captures WHERE capture_id=?",
                    (capture_id,)) == [(0,)]


def test_sqlserver_worker_reads_carry_source_bindings_into_derived_captures(
    sqlserver_store, typed_input, tmp_path,
):
    import uuid

    from autoforecast.integrations.sqlserver import SQLServerSnapshots, _execute

    store = sqlserver_store
    store.check_source(store.test_clients["source"])
    parent = store.capture(typed_input[0], capture_id=uuid.uuid4().hex)
    store.retain(parent)
    worker = SQLServerSnapshots(store.writer, store.reader, schema=store.schema,
                                releaser=store.releaser)
    assert not worker._source_users
    materialized = tmp_path / "worker-input"
    child = None
    try:
        worker.read(parent, materialized)
        child = worker.capture(materialized, capture_id=uuid.uuid4().hex)
        assert child["incarnation"]["sources"] == parent["incarnation"]["sources"]
        source = store.test_roles["source"]
        _execute(store.test_clients["source"], "USE master")
        _execute(store.test_admin, f"DROP USER [{source}]")
        try:
            with pytest.raises(ValueError, match="source.*mapping"):
                worker.inspect(child)
        finally:
            _execute(store.test_admin, f"CREATE USER [{source}] FOR LOGIN [{source}]")
            _execute(store.test_admin, f"GRANT CONNECT TO [{source}]")
    finally:
        if child is not None:
            worker.drop_unretained(child)
        store.release(parent)


@pytest.mark.parametrize("site,setting,encoding", [
    ("writer", "SET NAMES latin1", None),
    ("writer", "SET SESSION character_set_client='latin1'", None),
    ("writer", "SET SESSION character_set_connection='latin1'", None),
    ("source", "SET SESSION character_set_results='ascii'", None),
    ("source", "SET NAMES ascii", None),
    ("source", "SET CHARACTER SET ascii", None),
    ("reader", "SET SESSION character_set_results='ascii'", None),
    ("reader", "SET NAMES ascii", None),
    ("reader", "SET CHARACTER SET ascii", None),
    ("writer", None, "latin1"),
    ("source", None, "latin1"),
    ("reader", None, "latin1"),
    ("writer", None, "ascii"),
    ("source", None, "ascii"),
    ("reader", None, "ascii"),
])
def test_mysql_transfers_refuse_incompatible_session_and_driver_charsets(
    request, tmp_path, site, setting, encoding,
):
    import uuid

    import pyarrow as pa
    import pyarrow.parquet as pq

    from autoforecast.integrations.mysql import _execute, read_sources

    expected = pa.table({"id": [1], "label": ["café 水"]})
    path = tmp_path / "input.parquet"
    pq.write_table(expected, path)
    record = None
    store = None
    if site == "source":
        admin, connection, database = request.getfixturevalue("mysql_source")
        _execute(admin, f"CREATE TABLE `{database}`.sales "
                 "(id BIGINT,label TEXT CHARACTER SET utf8mb4) ENGINE=InnoDB")
        _execute(admin, f"INSERT INTO `{database}`.sales VALUES(1,%s)", ("café 水",))
    else:
        store = request.getfixturevalue("mysql_store")
        connection = store.writer if site == "writer" else store.reader
        if site == "reader":
            record = store.capture(path, capture_id=uuid.uuid4().hex)
            store.retain(record)
    original_driver = connection.charset, connection.encoding
    session_sql = ("SELECT @@session.character_set_client, @@session.character_set_connection, "
                   "@@session.character_set_results, @@session.sql_mode, @@session.time_zone")
    try:
        if setting is not None:
            _execute(connection, setting)
        if encoding is not None:
            connection.encoding = encoding
        assert connection.charset == "utf8mb4"
        drifted_driver = connection.charset, connection.encoding
        drifted_session = _execute(connection, session_sql)
        output = tmp_path / "output"
        with pytest.raises(ValueError, match="utf8mb4"):
            if site == "source":
                read_sources(connection, queries={"primary": "SELECT id,label FROM sales",
                             "audit": "SELECT id,label FROM sales"},
                             tables={"sales": database + ".sales"}, output_root=output)
            elif site == "writer":
                record = store.capture(path, capture_id=uuid.uuid4().hex)
            else:
                store.read(record, output)
        assert not output.exists()
        assert (connection.charset, connection.encoding) == drifted_driver
        assert _execute(connection, session_sql) == drifted_session
        if site == "writer":
            count = _execute(store.reader,
                             f"SELECT COUNT(*) FROM `{store.database}`.af_captures")[0][0]
            assert count == 0
    finally:
        connection.charset, connection.encoding = original_driver
        _execute(connection, "SET NAMES utf8mb4")
        if record is not None:
            store.release(record)


@pytest.mark.parametrize("site", ["source", "writer", "reader"])
@pytest.mark.parametrize("setup", ["connect", "set_names", "driver_set", "codec_alias"])
def test_mysql_utf8_transport_preserves_unicode_and_caller_settings(
    request, tmp_path, site, setup,
):
    import uuid

    import pyarrow as pa
    import pyarrow.parquet as pq
    from pymysql.constants import SERVER_STATUS

    from autoforecast.integrations.mysql import _execute, read_sources

    expected = pa.table({"id": [1, 2, 3], "label": ["café 水", "emoji 😀 and e\u0301", None]})
    path = tmp_path / "input.parquet"
    pq.write_table(expected, path)
    record = None
    if site == "source":
        admin, connection, database = request.getfixturevalue("mysql_source")
        _execute(admin, f"CREATE TABLE `{database}`.sales "
                 "(id BIGINT,label TEXT CHARACTER SET utf8mb4) ENGINE=InnoDB")
        with admin.cursor() as cursor:
            cursor.executemany(f"INSERT INTO `{database}`.sales VALUES(%s,%s)",
                               [(row["id"], row["label"]) for row in expected.to_pylist()])
    else:
        store = request.getfixturevalue("mysql_store")
        connection = store.writer if site == "writer" else store.reader
        if site == "reader":
            record = store.capture(path, capture_id=uuid.uuid4().hex)
            store.retain(record)
    original_driver = connection.charset, connection.encoding
    try:
        if setup == "set_names":
            _execute(connection, "SET NAMES utf8mb4")
        elif setup == "driver_set":
            connection.set_character_set("utf8mb4", "utf8mb4_unicode_ci")
        elif setup == "codec_alias":
            connection.encoding = "UTF-8"
        before_driver = connection.charset, connection.encoding
        session_sql = ("SELECT @@session.character_set_client, @@session.character_set_connection, "
                       "@@session.character_set_results, @@session.collation_connection, "
                       "@@session.sql_mode, @@session.time_zone")
        before_session = _execute(connection, session_sql)
        assert before_session[0][:3] == ("utf8mb4", "utf8mb4", "utf8mb4")
        if site == "source":
            result = read_sources(connection, queries={"primary": "SELECT id,label FROM sales",
                                  "audit": "SELECT id,label FROM sales"},
                                  tables={"sales": database + ".sales"},
                                  output_root=tmp_path / "output")
            actual = [pq.read_table(result[name][0]) for name in ("primary", "audit")]
        else:
            if site == "writer":
                record = store.capture(path, capture_id=uuid.uuid4().hex)
                store.retain(record)
            store.read(record, tmp_path / "output")
            actual = [pq.read_table(tmp_path / "output")]
        for table in actual:
            assert table.sort_by("id").equals(expected)
        assert (connection.charset, connection.encoding) == before_driver
        assert _execute(connection, session_sql) == before_session
        assert connection.get_autocommit()
        assert not connection.server_status & SERVER_STATUS.SERVER_STATUS_IN_TRANS
    finally:
        connection.charset, connection.encoding = original_driver
        _execute(connection, "SET NAMES utf8mb4")
        if record is not None:
            store.release(record)


def test_mysql_capture_rechecks_charset_after_checkpoint(mysql_store, tmp_path):
    import uuid

    import pyarrow as pa
    import pyarrow.parquet as pq

    from autoforecast.integrations.mysql import _execute

    store = mysql_store
    path = tmp_path / "input.parquet"
    pq.write_table(pa.table({"id": [1], "label": ["café 水"]}), path)
    calls = []
    record = None

    def checkpoint(identity):
        calls.append(identity["capture_id"])
        _execute(store.writer, "SET SESSION character_set_client='latin1'")

    try:
        with pytest.raises(ValueError, match="utf8mb4"):
            record = store.capture(path, capture_id=uuid.uuid4().hex, checkpoint=checkpoint)
        assert len(calls) == 1
        assert _execute(store.writer, "SELECT @@session.character_set_client") == (("latin1",),)
        assert _execute(store.reader, f"SELECT COUNT(*) FROM `{store.database}`.af_captures") == (
            (0,),
        )
    finally:
        _execute(store.writer, "SET NAMES utf8mb4")
        if record is not None:
            store.release(record)


def _native_derivation_registry(store, spark, root, *, mode="table"):
    from autoforecast.study_registration import StudyRegistry

    return StudyRegistry(root, spark=spark, snapshot_mode=mode, connections={"archive": store},
                         snapshots={"connection": "archive", "namespace": store.namespace})


def _native_derivation_options(*, minimum=1):
    return {"series_policy": "use_available", "minimum_history": minimum,
            "metrics": ["mae"], "metric": "mae", "selection_boundaries": []}


def _native_derivation_capture_ids(store):
    from psycopg import sql

    return {row[0] for row in store.test_admin.execute(sql.SQL("SELECT capture_id FROM {}").format(
        sql.Identifier(store.schema, "af_captures"),
    )).fetchall()}


@pytest.mark.parametrize("mode", ["table", "table+hash"], ids=["table", "hashed"])
@pytest.mark.parametrize("policy,minimum", [("use_available", 1), ("exclude_unavailable", 3)],
                         ids=["available", "exclude"])
def test_native_validation_checkpoint_resumes_same_effective_study(
    postgres_store, lifecycle_spark, tmp_path, mode, policy, minimum,
):
    import yaml

    from autoforecast.study_registration import register_study
    from autoforecast.test_study_registration import configuration
    from autoforecast.test_study_tuning import config, implementation, predict

    registry = _native_derivation_registry(postgres_store, lifecycle_spark, tmp_path / "registry",
                                           mode=mode)
    profile = config(tmp_path)
    values = yaml.safe_load(profile.read_text())
    values.update(series_policy=policy, minimum_history=minimum)
    profile.write_text(yaml.safe_dump(values))
    calls = []

    def fit(request):
        calls.append(request.purpose)
        return predict(request)

    try:
        study_id = register_study(configuration(tmp_path / "source"), registry=registry)
        with registry.load_study(study_id) as study:
            checkpoint = study.run_forecast_workflow(
                fit_predict=fit, implementation=implementation(), config=profile,
                fixed_parameters={}, validation_only=True, output_dir=tmp_path / "validation",
            )
            assert checkpoint.testing is None and checkpoint.study_id != study_id
            completed = study.run_forecast_workflow(
                fit_predict=fit, implementation=implementation(), validation_run=checkpoint.path,
                output_dir=tmp_path / "completed",
            )
            assert completed.testing is not None
            assert completed.provenance["study"] == checkpoint.provenance["study"]
            assert calls == ["validation", "refit"]
    finally:
        registry.close()


def test_native_tuning_reuse_does_not_recapture_effective_study(
    postgres_store, lifecycle_spark, tmp_path,
):
    import yaml

    from autoforecast.study_registration import register_study
    from autoforecast.test_study_registration import configuration
    from autoforecast.test_study_tuning import config, implementation, predict

    registry = _native_derivation_registry(postgres_store, lifecycle_spark, tmp_path / "registry")
    profile = config(tmp_path, budget=1)
    values = yaml.safe_load(profile.read_text())
    values.update(series_policy="use_available", minimum_history=1)
    profile.write_text(yaml.safe_dump(values))
    calls = []

    def fit(request):
        calls.append(request.purpose)
        return predict(request)

    try:
        study_id = register_study(configuration(tmp_path / "source"), registry=registry)
        with registry.load_study(study_id) as study:
            first = study.tune(fit_predict=fit, implementation=implementation(), config=profile,
                               output_dir=tmp_path / "first")
            assert calls == ["selection"]
            second = study.tune(fit_predict=fit, implementation=implementation(), config=profile,
                                reuse_tuning=first.path, output_dir=tmp_path / "second")
            assert second.path == first.path
            assert calls == ["selection"]
    finally:
        registry.close()


def test_native_campaign_repeat_reuses_same_frozen_request(
    postgres_store, lifecycle_spark, tmp_path, monkeypatch,
):
    import yaml

    from autoforecast.campaign import CampaignSupervisor
    from autoforecast.study_campaign import StudyModel
    from autoforecast.study_registration import register_study
    from autoforecast.test_study_registration import configuration
    from autoforecast.test_study_tuning import config, implementation

    registry = _native_derivation_registry(postgres_store, lifecycle_spark, tmp_path / "registry")
    profile = config(tmp_path)
    values = yaml.safe_load(profile.read_text())
    values.update(series_policy="use_available", minimum_history=1)
    profile.write_text(yaml.safe_dump(values))
    model = StudyModel("autoforecast.test_study_tuning:predict", implementation(), profile,
                       fixed_parameters={})
    # Capture, derivation, request and campaign-store publication are real. Stop at scheduling.
    monkeypatch.setattr(CampaignSupervisor, "run", lambda self: self.store.state())
    try:
        study_id = register_study(configuration(tmp_path / "source"), registry=registry)
        with registry.load_study(study_id) as study:
            first = study.run_campaign(models=[model], output_dir=tmp_path / "campaign")
            second = study.run_campaign(models=[model], output_dir=tmp_path / "campaign")
            assert first["campaign_path"] == second["campaign_path"]
    finally:
        registry.close()


def test_native_derivation_serializes_reuse_and_separates_changed_decisions(
    postgres_store, lifecycle_spark, tmp_path, monkeypatch,
):
    import os
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    import psycopg
    import pytest
    from psycopg.conninfo import make_conninfo

    from autoforecast import _logical_hash, study_preflight
    from autoforecast.benchmark import read_benchmark_manifest
    from autoforecast.integrations.postgres import PostgresSnapshots
    from autoforecast.study_registration import register_study
    from autoforecast.test_study_registration import configuration

    # Both original series and all 17 observations remain; minimum history changes the decision.
    registry = _native_derivation_registry(postgres_store, lifecycle_spark, tmp_path / "registry")
    monkeypatch.setattr(_logical_hash, "fingerprint", lambda *a, **k: pytest.fail(
        "table-only native derivation must not hash rows"))
    monkeypatch.setattr(study_preflight, "directory_digest", lambda *a, **k: pytest.fail(
        "table-only native derivation must not hash files"))
    barrier = Barrier(2)
    held_reuses = []
    derive_owned = study_preflight._derive_owned

    def require_selected_reader(*args, **kwargs):
        if kwargs.get("reuse") is not None:
            selected = kwargs["reuse"][1]["study_id"]
            with pytest.raises(ValueError, match="active readers"):
                registry.release(selected)
            held_reuses.append(selected)
        return derive_owned(*args, **kwargs)

    monkeypatch.setattr(study_preflight, "_derive_owned", require_selected_reader)
    try:
        study_id = register_study(configuration(tmp_path / "source"), registry=registry)
        _, parent = read_benchmark_manifest(
            registry.root / "benchmarks" / study_id / "benchmark.json",
        )

        def worker(index):
            dsn = os.environ["AUTOFORECAST_TEST_INPUT_POSTGRES_DSN"]
            with psycopg.connect(make_conninfo(
                dsn, user=postgres_store.test_roles["writer"], password="local-test-reader-only",
            ), autocommit=True) as writer, psycopg.connect(make_conninfo(
                dsn, user=postgres_store.test_roles["reader"], password="local-test-reader-only",
            ), autocommit=True) as reader:
                store = PostgresSnapshots(writer, reader, schema=postgres_store.schema)
                local = _native_derivation_registry(
                    store, lifecycle_spark.newSession(), registry.root,
                )
                try:
                    with local.load_study(study_id) as study:
                        barrier.wait(timeout=60)
                        return study_preflight.preflight_study(
                            study, _native_derivation_options(), output_dir=tmp_path / str(index),
                        )
                finally:
                    local.close()

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(worker, range(2)))
        assert results[0].study_id == results[1].study_id
        first = results[0]
        _, child = read_benchmark_manifest(first.study_path, first.study_sha256)
        assert held_reuses == [first.study_id]
        assert _native_derivation_capture_ids(postgres_store) == (
            set(parent["owned_snapshots"]) | set(child["owned_snapshots"])
        )
        with registry.load_study(study_id) as study:
            changed = study_preflight.preflight_study(
                study, _native_derivation_options(minimum=2), output_dir=tmp_path / "changed",
            )
            repeated = study_preflight.preflight_study(
                study, _native_derivation_options(minimum=2), output_dir=tmp_path / "repeat",
            )
            assert changed.study_id != first.study_id
            assert repeated.study_id == changed.study_id
        _, other = read_benchmark_manifest(changed.study_path, changed.study_sha256)
        assert not set(child["owned_snapshots"]) & set(other["owned_snapshots"])
        assert _native_derivation_capture_ids(postgres_store) == (
            set(parent["owned_snapshots"]) | set(child["owned_snapshots"])
            | set(other["owned_snapshots"])
        )
        assert len(list((registry.root / "derivations").glob("*/derivation.json"))) == 2
        registry.release(first.study_id)
        registry.release(changed.study_id)
        registry.release(study_id)
    finally:
        registry.close()


@pytest.mark.parametrize("failure", [
    "lookup_subject", "manifest_subject", "missing_manifest", "released", "rows",
], ids=["lookup", "manifest", "missing", "released", "rows"])
def test_native_derivation_reuse_refuses_invalid_or_released_evidence(
    postgres_store, lifecycle_spark, tmp_path, failure,
):
    import hashlib
    import json

    import pytest
    from psycopg import sql

    from autoforecast.benchmark import read_benchmark_manifest
    from autoforecast.study_preflight import StudyPreflightError, preflight_study
    from autoforecast.study_registration import register_study
    from autoforecast.test_study_registration import configuration

    registry = _native_derivation_registry(postgres_store, lifecycle_spark, tmp_path / "registry")
    row_target = None
    changed_id = None
    try:
        study_id = register_study(configuration(tmp_path / "source"), registry=registry)
        with registry.load_study(study_id) as study:
            first = preflight_study(study, _native_derivation_options(),
                                    output_dir=tmp_path / "first")
        _, child = read_benchmark_manifest(first.study_path, first.study_sha256)
        lookup = next((registry.root / "derivations").glob("*/derivation.json"))
        original_lookup, original_manifest = lookup.read_bytes(), first.study_path.read_bytes()
        try:
            if failure == "lookup_subject":
                parent_path = registry.root / "benchmarks" / study_id / "benchmark.json"
                value = json.loads(original_lookup)
                value.update(study_id=study_id, manifest_sha256=hashlib.sha256(
                    parent_path.read_bytes(),
                ).hexdigest())
                lookup.write_text(json.dumps(value))
            elif failure == "manifest_subject":
                parent_path = registry.root / "benchmarks" / study_id / "benchmark.json"
                first.study_path.write_bytes(parent_path.read_bytes())
                value = json.loads(original_lookup)
                value["manifest_sha256"] = hashlib.sha256(parent_path.read_bytes()).hexdigest()
                lookup.write_text(json.dumps(value))
            elif failure == "missing_manifest":
                first.study_path.unlink()
                lookup.unlink()
            elif failure == "released":
                registry.release(first.study_id)
                lookup.unlink()
            else:
                reference = next(iter(child["grid_identities"].values()))["snapshot"]
                row_target = postgres_store._name(reference)
                postgres_store.test_admin.execute(sql.SQL("UPDATE {} SET date=date+1").format(
                    row_target,
                ))
            before = _native_derivation_capture_ids(postgres_store)
            with registry.load_study(study_id) as study:
                with pytest.raises(StudyPreflightError):
                    preflight_study(study, _native_derivation_options(),
                                    output_dir=tmp_path / "refused")
                assert _native_derivation_capture_ids(postgres_store) == before
                if failure == "released":
                    lookup.write_bytes(original_lookup)
                    with pytest.raises(StudyPreflightError):
                        preflight_study(study, _native_derivation_options(),
                                        output_dir=tmp_path / "released-lookup")
                    assert _native_derivation_capture_ids(postgres_store) == before
                    changed = preflight_study(study, _native_derivation_options(minimum=2),
                                              output_dir=tmp_path / "changed")
                    changed_id = changed.study_id
                    assert changed_id != first.study_id
        finally:
            lookup.write_bytes(original_lookup)
            first.study_path.write_bytes(original_manifest)
            if row_target is not None:
                postgres_store.test_admin.execute(sql.SQL("UPDATE {} SET date=date-1").format(
                    row_target,
                ))
        if changed_id is not None:
            registry.release(changed_id)
        registry.release(first.study_id)
        registry.release(study_id)
    finally:
        registry.close()


def test_native_derivation_recovers_interrupted_lookup_publication(
    postgres_store, lifecycle_spark, tmp_path, monkeypatch,
):
    import pytest
    from mlops import _safe_files

    from autoforecast import study_registration
    from autoforecast.benchmark import read_benchmark_manifest
    from autoforecast.study_preflight import StudyPreflightError, preflight_study
    from autoforecast.study_registration import register_study
    from autoforecast.test_study_registration import configuration

    registry = _native_derivation_registry(postgres_store, lifecycle_spark, tmp_path / "registry")
    publish = _safe_files.publish_bytes
    failed = []

    def fail_lookup_once(root, relative, data, **kwargs):
        if relative == "derivation.json" and not failed:
            failed.append(True)
            raise OSError("synthetic interrupted derivation lookup publication")
        return publish(root, relative, data, **kwargs)

    try:
        study_id = register_study(configuration(tmp_path / "source"), registry=registry)
        with registry.load_study(study_id) as study:
            with monkeypatch.context() as patch:
                patch.setattr(_safe_files, "publish_bytes", fail_lookup_once)
                with pytest.raises(StudyPreflightError, match="interrupted derivation lookup"):
                    preflight_study(study, _native_derivation_options(),
                                    output_dir=tmp_path / "failed")
            assert failed == [True]
            children = []
            for path in (registry.root / "benchmarks").glob("*/benchmark.json"):
                _, record = read_benchmark_manifest(path)
                if record.get("parent", {}).get("study_id") == study_id:
                    children.append(record["study_id"])
            assert len(children) == 1
            before = _native_derivation_capture_ids(postgres_store)
            resumed = preflight_study(study, _native_derivation_options(),
                                      output_dir=tmp_path / "resumed")
            assert resumed.study_id == children[0]
            assert _native_derivation_capture_ids(postgres_store) == before
            assert len(list((registry.root / "derivations").glob("*/derivation.json"))) == 1
            with monkeypatch.context() as patch:
                patch.setattr(study_registration, "_staging_directories", lambda path: pytest.fail(
                    "an authenticated lookup hit must not scan all benchmark directories"))
                repeated = preflight_study(study, _native_derivation_options(),
                                          output_dir=tmp_path / "indexed")
                assert repeated.study_id == resumed.study_id
                assert _native_derivation_capture_ids(postgres_store) == before
        registry.release(resumed.study_id)
        registry.release(study_id)
    finally:
        registry.close()


def test_native_derivation_refuses_replacement_after_selection(
    postgres_store, lifecycle_spark, tmp_path, monkeypatch,
):
    import pytest

    from autoforecast import study_preflight
    from autoforecast.study_registration import register_study
    from autoforecast.test_study_registration import configuration

    registry = _native_derivation_registry(postgres_store, lifecycle_spark, tmp_path / "registry")
    try:
        study_id = register_study(configuration(tmp_path / "source"), registry=registry)
        with registry.load_study(study_id) as study:
            first = study_preflight.preflight_study(study, _native_derivation_options(),
                                                    output_dir=tmp_path / "first")
            original = first.study_path.read_bytes()
            parent = study._path.read_bytes()
            derive = study_preflight._derive

            def replace_selected_manifest(*args, **kwargs):
                selected = derive(*args, **kwargs)
                assert selected == first.study_path
                selected.write_bytes(parent)
                return selected

            before = _native_derivation_capture_ids(postgres_store)
            try:
                with monkeypatch.context() as patch:
                    patch.setattr(study_preflight, "_derive", replace_selected_manifest)
                    with pytest.raises(study_preflight.StudyPreflightError, match="selected"):
                        result = study_preflight.preflight_study(
                            study, _native_derivation_options(), output_dir=tmp_path / "refused",
                        )
                        print("UNSAFE_EFFECTIVE_SELECTION", result.study_id,
                              "SELECTED", first.study_id)
                assert _native_derivation_capture_ids(postgres_store) == before
            finally:
                first.study_path.write_bytes(original)
        registry.release(first.study_id)
        registry.release(study_id)
    finally:
        registry.close()


def _sqlserver_configured_temporary_scope(store, tmp_path, monkeypatch):
    import os

    from mssql_python.connection_string_parser import _ConnectionStringParser

    from autoforecast.snapshots import Connections, NativeReadScope

    base = _ConnectionStringParser._normalize_params(_ConnectionStringParser()._parse(
        os.environ["AUTOFORECAST_TEST_INPUT_SQLSERVER_DSN"],
    ))
    variables = {role: "AF_SQLSERVER_TEMPORARY_" + role.upper()
                 for role in ("reader", "writer", "release", "source")}
    for role, variable in variables.items():
        values = dict(base, Database=store._database(store.reader)["name"],
                      UID=store.test_roles[role], PWD="Local-source-test-only-2026!")
        dsn = ";".join(key + "={" + value.replace("}", "}}") + "}"
                       for key, value in values.items())
        monkeypatch.setenv(variable, dsn)
    connections = Connections({
        "archive": {"kind": "sqlserver", "dsn_env": variables["reader"],
                    "writer_dsn_env": variables["writer"],
                    "release_dsn_env": variables["release"]},
        "source": {"kind": "sqlserver", "dsn_env": variables["source"]},
    })
    scope = NativeReadScope(connections, scratch_root=tmp_path)
    stores = []
    resolve = connections.snapshot_store

    def remember(destination, **kwargs):
        selected = resolve(destination, **kwargs)
        stores.append((dict(destination), selected, selected._source_users))
        return selected

    monkeypatch.setattr(connections, "snapshot_store", remember)
    return connections, scope, stores


def _sqlserver_temporary_parent(store, typed_input):
    import uuid

    store.check_source(store.test_clients["source"])
    parent = store.capture(typed_input[0], capture_id=uuid.uuid4().hex)
    store.retain(parent)
    assert parent["incarnation"]["sources"]
    return dict(parent, schema="autoforecast.native-input.v1", mode="table",
                backend="sqlserver", connection="archive", retention="until_released")


def _assert_sqlserver_temporary_rows(path, expected):
    import pyarrow.parquet as pq

    actual = pq.read_table(path)
    assert actual.schema.equals(expected.schema)
    assert sorted(actual.to_pylist(), key=repr) == sorted(expected.to_pylist(), key=repr)


def test_sqlserver_configured_temporary_capture_preserves_source_binding(
    sqlserver_store, typed_input, tmp_path, monkeypatch,
):
    from autoforecast import _logical_hash
    from autoforecast.integrations.sqlserver import _execute

    store = sqlserver_store
    monkeypatch.setattr(_logical_hash, "fingerprint", lambda *a, **k: pytest.fail(
        "table-only temporary inputs must not hash rows"))
    parent = _sqlserver_temporary_parent(store, typed_input)
    connections, scope, stores = _sqlserver_configured_temporary_scope(store, tmp_path, monkeypatch)
    child = None
    try:
        materialized = scope.materialize(parent)
        reader_store = next(iter(scope._managers.values())).store
        assert reader_store is not store
        assert reader_store.writer is None
        assert reader_store._source_users
        child = scope.capture_temporary(materialized, template=parent,
                                        journal_root=tmp_path / "capture-journal")
        destination, writer_store, initial_sources = stores[-1]
        assert destination == {"connection": "archive", "namespace": store.schema}
        assert writer_store is not reader_store and writer_store is not store
        assert initial_sources == ()
        assert child["capture_id"] != parent["capture_id"]
        assert child["incarnation"]["sources"] == parent["incarnation"]["sources"]
        assert writer_store._source_users == reader_store._source_users
        _assert_sqlserver_temporary_rows(scope.materialize(child), typed_input[1])
        owned_clients = tuple(connections._owned)
        scope.close()
        assert not materialized.exists()
        assert _execute(store.reader, "SELECT OBJECT_ID(?)", (child["object"],)) == [(None,)]
        store.inspect(parent, retained=True)
        connections.close()
        assert all(client.closed for client in owned_clients)
    finally:
        scope.close()
        connections.close()
        if child is not None:
            store.drop_unretained(child)
        store.release(parent)


@pytest.mark.parametrize("failure", ["missing_parent", "database", "seal", "backend"])
def test_sqlserver_temporary_capture_authenticates_destination_before_creating_state(
    sqlserver_store, typed_input, tmp_path, monkeypatch, failure,
):
    import copy

    from autoforecast import _logical_hash
    from autoforecast.integrations.sqlserver import _execute

    store = sqlserver_store
    monkeypatch.setattr(_logical_hash, "fingerprint", lambda *a, **k: pytest.fail(
        "table-only temporary inputs must not hash rows"))
    parent = _sqlserver_temporary_parent(store, typed_input)
    connections, scope, stores = _sqlserver_configured_temporary_scope(store, tmp_path, monkeypatch)
    journal = tmp_path / "capture-journal"
    try:
        materialized = scope.materialize(parent)
        reader_store = next(iter(scope._managers.values())).store
        changed = copy.deepcopy(parent)
        if failure == "missing_parent":
            changed["capture_id"] = "f" * 32
            changed["object"] = store.schema + ".af_input_" + changed["capture_id"]
        elif failure == "database":
            changed["incarnation"]["database"]["guid"] = "00000000-0000-0000-0000-000000000000"
        elif failure == "seal":
            changed["incarnation"]["seal"] = "0" * 32
        else:
            changed["backend"] = "postgresql"
        with pytest.raises((ValueError, FileNotFoundError), match="missing|changed|backend"):
            scope.capture_temporary(materialized, template=changed, journal_root=journal)
        assert len(stores) == 1
        assert stores[0][1] is not reader_store
        assert stores[0][2] == ()
        assert not journal.exists()
        assert _execute(store.reader, "SELECT COUNT(*) FROM archive.af_captures") == [(1,)]
        _assert_sqlserver_temporary_rows(materialized, typed_input[1])
        store.inspect(parent, retained=True)
    finally:
        scope.close()
        connections.close()
        store.release(parent)


def test_sqlserver_configured_temporary_capture_failure_releases_its_owner(
    sqlserver_store, typed_input, tmp_path, monkeypatch,
):
    from autoforecast import _logical_hash
    from autoforecast.integrations.sqlserver import SQLServerSnapshots, _execute
    from autoforecast.study_registration import _publication_lock

    store = sqlserver_store
    monkeypatch.setattr(_logical_hash, "fingerprint", lambda *a, **k: pytest.fail(
        "table-only temporary inputs must not hash rows"))
    parent = _sqlserver_temporary_parent(store, typed_input)
    connections, scope, stores = _sqlserver_configured_temporary_scope(store, tmp_path, monkeypatch)
    journal = tmp_path / "capture-journal"
    operation = SQLServerSnapshots._operation
    failure = ConnectionError("synthetic temporary sealing failure")

    def fail_before_seal(self, connection, action, record, **kwargs):
        if action == "seal" and stores and self is stores[-1][1]:
            assert self._source_users
            raise failure
        return operation(self, connection, action, record, **kwargs)

    monkeypatch.setattr(SQLServerSnapshots, "_operation", fail_before_seal)
    try:
        materialized = scope.materialize(parent)
        with pytest.raises(ConnectionError) as caught:
            scope.capture_temporary(materialized, template=parent, journal_root=journal)
        assert caught.value is failure
        writer_store = stores[-1][1]
        assert writer_store._source_users == tuple(
            tuple(user) for user in parent["incarnation"]["sources"]
        )
        assert _execute(writer_store.writer, "SELECT @@TRANCOUNT") == [(0,)]
        assert _execute(store.reader, "SELECT COUNT(*) FROM archive.af_captures") == [(1,)]
        directories = list(journal.iterdir())
        assert len(directories) == 1
        assert (directories[0] / "intent.json").is_file()
        assert not (directories[0] / "sealed.json").exists()
        assert not (directories[0] / "ready.json").exists()
        with pytest.raises(ValueError, match="active owner"):
            with _publication_lock(directories[0], blocking=False):
                pytest.fail("capture owner was released before its scope closed")
        owned_clients = tuple(connections._owned)
        scope.close()
        assert not materialized.exists()
        with _publication_lock(directories[0], blocking=False):
            pass
        connections.close()
        assert all(client.closed for client in owned_clients)
        store.inspect(parent, retained=True)
    finally:
        scope.close()
        connections.close()
        store.release(parent)


def test_sqlserver_configured_derivation_preserves_parent_source_bindings(
    sqlserver_store, lifecycle_spark, tmp_path, monkeypatch,
):
    import pyarrow.parquet as pq
    import yaml

    from autoforecast import _logical_hash
    from autoforecast.benchmark import read_benchmark_manifest
    from autoforecast.integrations.sqlserver import _execute
    from autoforecast.study_registration import StudyRegistry, register_study
    from autoforecast.test_study_registration import configuration
    from autoforecast.test_study_tuning import config as tuning_config

    store = sqlserver_store
    configured, unused_scope, _ = _sqlserver_configured_temporary_scope(
        store, tmp_path, monkeypatch,
    )
    settings = configured.settings
    unused_scope.close()
    configured.close()
    database = store._database(store.reader)["name"]
    _execute(store.test_admin,
             f"ALTER DATABASE [{database}] SET ALLOW_SNAPSHOT_ISOLATION ON")
    source_config = configuration(tmp_path)
    rows = pq.read_table(tmp_path / "source.parquet").to_pylist()
    assert len(rows) == 17
    # All 17 synthetic observations from two declared series enter the source query.
    _execute(store.test_admin,
             "CREATE TABLE dbo.observations (item NVARCHAR(10),day DATE,value FLOAT)")
    with store.test_admin.cursor() as cursor:
        cursor.executemany("INSERT INTO dbo.observations VALUES(?,?,?)",
                           [(row["item"], row["day"], row["value"]) for row in rows])
    _execute(store.test_admin, "GRANT SELECT ON OBJECT::dbo.observations TO [" +
             store.test_roles["source"] + "]")
    (tmp_path / "source.sql").write_text("SELECT item,day,value FROM observations")
    value = yaml.safe_load(source_config.read_text())
    value["source"] = {"connection": "source", "kind": "sqlserver", "sql": "source.sql",
                       "tables": {"observations": "dbo.observations"}}
    source_config.write_text(yaml.safe_dump(value))
    profile = tuning_config(tmp_path, budget=1)
    value = yaml.safe_load(profile.read_text())
    value.update(series_policy="use_available", minimum_history=1)
    profile.write_text(yaml.safe_dump(value))
    monkeypatch.setattr(_logical_hash, "fingerprint", lambda *a, **k: pytest.fail(
        "table-only native derivation must not hash rows"))
    registry = StudyRegistry(
        tmp_path / "registry", spark=lifecycle_spark, snapshot_mode="table",
        connection_settings=settings,
        snapshots={"connection": "archive", "namespace": store.schema},
    )
    parent_id = derived_id = None
    try:
        parent_id = register_study(source_config, registry=registry)
        assert not registry.connections.supplied
        selected = []
        resolve = registry.connections.snapshot_store

        def remember(destination, **kwargs):
            actual = resolve(destination, **kwargs)
            selected.append((actual, actual._source_users))
            return actual

        monkeypatch.setattr(registry.connections, "snapshot_store", remember)
        with registry.load_study(parent_id) as study:
            parent = study._record(data=True)
            original_sources = next(iter(parent["snapshots"].values()))["incarnation"]["sources"]
            assert original_sources
            readers = [manager.store for manager in study._native_inputs._managers.values()]
            assert readers and all(reader.writer is None for reader in readers)
            result = study.preflight(config=profile, output_dir=tmp_path / "preflight")
            derived_id = result.study_id
            assert derived_id != parent_id
            _, derived = read_benchmark_manifest(result.study_path, result.study_sha256)
            owned = derived["owned_snapshots"]
            assert owned
            assert len(derived["execution_policy"]["retained_ids"]) == 2
            assert selected
            for writer, initial_sources in selected:
                assert writer is not store and all(writer is not reader for reader in readers)
                assert writer.writer is not None
                assert initial_sources == ()
            for reference in owned.values():
                assert reference["incarnation"]["sources"] == original_sources
                store.inspect(reference, retained=True)
    finally:
        if derived_id is not None:
            registry.release(derived_id)
        if parent_id is not None:
            registry.release(parent_id)
        registry.close()


@pytest.mark.parametrize("large", [False, True], ids=["string", "large-string"])
def test_mysql_text_preserves_nul_unicode_and_long_values(mysql_store, tmp_path, large):
    import uuid

    import pyarrow as pa
    import pyarrow.parquet as pq

    from autoforecast.integrations.mysql import read_sources

    values = [None, "", "before\x00after", "\x00", "end\x00\x00", "雪😀e\u0301",
              "a" * 8193 + "\x00"]
    dtype = pa.large_string() if large else pa.string()
    expected = pa.table({"id": list(range(len(values))),
                         "label": pa.array(values, type=dtype)})
    path = tmp_path / "original.parquet"
    pq.write_table(expected, path)
    record = mysql_store.capture(path, capture_id=uuid.uuid4().hex)
    try:
        mysql_store.retain(record)
        mysql_store.read(record, tmp_path / "replay")
        assert pq.read_table(tmp_path / "replay").sort_by("id").equals(expected)
        result = read_sources(mysql_store.reader, queries={"primary": "SELECT id,label FROM saved"},
                              tables={"saved": record["object"]}, output_root=tmp_path / "source")
        assert pq.read_table(result["primary"][0]).sort_by("id").to_pylist() == expected.to_pylist()
    finally:
        mysql_store.release(record)


def test_postgres_nul_text_refusal_leaves_no_capture(postgres_store, tmp_path):
    import uuid

    import psycopg
    import pyarrow as pa
    import pyarrow.parquet as pq
    from psycopg import sql

    path = tmp_path / "original.parquet"
    original = pa.table({"label": ["before\x00after"]})
    pq.write_table(original, path)
    with pytest.raises((ValueError, psycopg.Error), match="NUL|0x00|zero|null"):
        postgres_store.capture(path, capture_id=uuid.uuid4().hex)
    assert pq.read_table(path).equals(original)
    count = postgres_store.test_admin.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(
        sql.Identifier(postgres_store.schema, "af_captures"),
    )).fetchone()[0]
    assert count == 0



def _sqlserver_complete_text_values():
    return [
        None, "", "\x00", "\x00\x00", "before\x00after", "end\x00", "\x00start",
        "café 水🙂\x00", "e\u0301\x00\U0010ffff", "spaces  ", "\x01\x0b\x1f\x7f\x00",
        *("x" * size + "\x00" for size in (3999, 4000, 4001, 4095, 4096, 4097,
                                           7999, 8000, 8001, 8191, 8192, 8193, 16385)),
        "水🙂" * 4097 + "\x00",
    ]


@pytest.mark.parametrize("kind,mode", [
    ("string", "table"), ("string", "table+hash"), ("large_string", "table"),
    ("nested", "table"), ("nested", "table+hash"),
])
@pytest.mark.parametrize("width", ["short", "long"])
def test_sqlserver_complete_native_text(sqlserver_store, tmp_path, monkeypatch, mode, kind, width):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from autoforecast import _logical_hash
    from autoforecast.integrations.sqlserver import _execute
    from autoforecast.snapshots import ColumnLayout, RetainedInputs

    values = _sqlserver_complete_text_values()
    if width == "short":
        values = values[:11]
    dtype = pa.large_string() if kind == "large_string" else pa.string()
    column = values
    if kind == "nested":
        dtype = pa.struct([("leaf", pa.string()), ("child", pa.struct([
            ("large", pa.large_string() if mode == "table" else pa.string()),
        ]))])
        column = [None, {"leaf": None, "child": None}, {"leaf": None, "child": {"large": None}}]
        column.extend({"leaf": value, "child": {"large": value}} for value in values)
    expected = pa.table({"id": pa.array(range(len(column)), type=pa.int32()),
                         "text": pa.array(column, type=dtype)})
    path = tmp_path / "text.parquet"
    pq.write_table(expected, path)
    if mode == "table":
        monkeypatch.setattr(_logical_hash, "fingerprint", lambda *a, **kw: pytest.fail(
            "table-only text transport must not hash dataset rows"))
    store = sqlserver_store
    with RetainedInputs(mode=mode, connection="archive", store=store,
                        scratch_root=tmp_path / "scratch") as manager:
        reference = manager.capture(path)
        record = reference.record()
        try:
            layout = ColumnLayout(expected.schema)
            text_columns = [(index, name) for index, (name, _, present, field_type)
                            in enumerate(layout.entries)
                            if not present and (pa.types.is_string(field_type)
                                                or pa.types.is_large_string(field_type))]
            select = ",".join("CONVERT(VARBINARY(MAX),[" + name + "])"
                              for _, name in text_columns)
            order = layout.entries[0][0]
            stored = _execute(store.reader, "SELECT " + select + " FROM " + store._name(record)
                              + " ORDER BY [" + order + "]")
            encoded = [layout.encode(row) for row in expected.to_pylist()]
            assert stored == [tuple(None if row[index] is None else row[index].encode("utf-16le")
                                    for index, _ in text_columns) for row in encoded]
            store.retain(record)
            with manager.open(reference) as saved:
                actual = pq.read_table(saved).sort_by("id")
                assert actual.schema.equals(expected.schema)
                assert actual.to_pylist() == expected.to_pylist()
        finally:
            store.release(record)


@pytest.mark.parametrize("sql_type,unicode,limit,padded", [
    ("NVARCHAR(100)", True, 100, False),
    ("VARCHAR(100)", False, 100, False),
    ("NCHAR(100)", True, 100, True),
    ("CHAR(100)", False, 100, True),
    ("NVARCHAR(4000)", True, 4000, False),
    ("VARCHAR(8000)", False, 8000, False),
    ("NVARCHAR(MAX)", True, None, False),
    ("VARCHAR(MAX)", False, None, False),
    ("VARCHAR(MAX) COLLATE Latin1_General_100_BIN2_UTF8", True, None, False),
], ids=["nvarchar", "varchar", "nchar", "char", "nvarchar-boundary", "varchar-boundary",
        "nvarchar-max", "varchar-max", "varchar-utf8-max"])
def test_sqlserver_complete_source_text(
    sqlserver_database, tmp_path, sql_type, unicode, limit, padded,
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from autoforecast.integrations.sqlserver import _execute, read_sources

    source, writer, _, _ = sqlserver_database
    values = [value for value in _sqlserver_complete_text_values()
              if value is None or ((unicode or value.isascii())
                                    and (limit is None
                                         or len(value.encode("utf-16le")) // 2 <= limit))]
    _execute(writer, "CREATE TABLE dbo.text_source (id INT NOT NULL,value " + sql_type + ")")
    with writer.cursor() as cursor:
        cursor.executemany("INSERT INTO dbo.text_source VALUES (?,CONVERT(NVARCHAR(MAX),?))", [
            (index, None if value is None else value.encode("utf-16le"))
            for index, value in enumerate(values)
        ])
    if padded:
        values = [None if value is None else value + " " * (
            limit - len(value.encode("utf-16le")) // 2) for value in values]
    binary = _execute(source, "SELECT CONVERT(VARBINARY(MAX),CONVERT(NVARCHAR(MAX),value)) "
                      "FROM dbo.text_source ORDER BY id")
    assert binary == [(None if value is None else value.encode("utf-16le"),) for value in values]
    before = (source.autocommit, _execute(source, "SELECT @@TRANCOUNT"), _execute(
        source, "SELECT transaction_isolation_level FROM sys.dm_exec_sessions "
        "WHERE session_id=@@SPID"))
    queries = {"rows": "SELECT id,value FROM text_source ORDER BY id",
               "empty": "SELECT id,value FROM text_source WHERE id<0",
               "null": "SELECT value FROM text_source WHERE id=0"}
    result = read_sources(source, queries=queries, tables={"text_source": "dbo.text_source"},
                          output_root=tmp_path / "capture")
    actual = pq.read_table(result["rows"][0])
    assert actual.schema.field("value").type == pa.string()
    assert actual.to_pylist() == [{"id": index, "value": value}
                                  for index, value in enumerate(values)]
    empty = pq.read_table(result["empty"][0])
    assert empty.num_rows == 0 and empty.schema.equals(actual.schema)
    assert pq.read_table(result["null"][0]).to_pylist() == [{"value": None}]
    assert (source.autocommit, _execute(source, "SELECT @@TRANCOUNT"), _execute(
        source, "SELECT transaction_isolation_level FROM sys.dm_exec_sessions "
        "WHERE session_id=@@SPID")) == before


def test_sqlserver_text_source_preserves_query_shapes(sqlserver_database, tmp_path):
    import pyarrow.parquet as pq

    from autoforecast.integrations.sqlserver import _execute, read_sources

    source, writer, _, _ = sqlserver_database
    _execute(writer, "CREATE TABLE dbo.text_shapes (id INT NOT NULL,rank INT NOT NULL, "
             "value NVARCHAR(MAX) COLLATE Latin1_General_100_CI_AS)")
    rows = [(1, 30, "same\x00"), (2, 10, "SAME\x00"), (3, 10, "other\x00"),
            (4, 40, "later\x00")]
    with writer.cursor() as cursor:
        cursor.executemany("INSERT INTO dbo.text_shapes VALUES (?,?,CONVERT(NVARCHAR(MAX),?))",
                           [(index, rank, text.encode("utf-16le")) for index, rank, text in rows])
    queries = {
        "plain": "SELECT id,value FROM data",
        "order": "SELECT id,value FROM data ORDER BY rank DESC,id",
        "top": "SELECT TOP (2) id,value FROM data ORDER BY rank,id",
        "ties": "SELECT TOP (1) WITH TIES id,value FROM data ORDER BY rank",
        "offset": "SELECT id,value FROM data ORDER BY rank,id OFFSET 1 ROWS FETCH NEXT 2 ROWS ONLY",
        "union": "SELECT id,value FROM data WHERE id<=2 UNION "
                 "SELECT id,value FROM data WHERE id BETWEEN 2 AND 3 ORDER BY id",
        "union_all": "SELECT id,value FROM data WHERE id<=2 UNION ALL "
                     "SELECT id,value FROM data WHERE id BETWEEN 2 AND 3",
        "distinct": "SELECT DISTINCT value FROM data WHERE id<=2",
        "group": "SELECT MIN(id) AS id,MIN(value) AS value,COUNT_BIG(*) AS total "
                 "FROM data WHERE id<=2 GROUP BY value",
        "aliases": "SELECT id AS [a]]b],value AS [select],value AS [Case],value AS [case],"
                   "value AS [white space 水] FROM data WHERE id=1",
    }
    result = read_sources(source, queries=queries, tables={"data": "dbo.text_shapes"},
                          output_root=tmp_path / "capture")
    expected_ids = {"plain": [1, 2, 3, 4], "order": [1, 2, 3, 4],
                    "top": [2, 3], "ties": [2, 3], "offset": [1, 3],
                    "union": [1, 2, 3], "union_all": [1, 2, 2, 3]}
    by_id = {index: text for index, _, text in rows}
    for name, ids in expected_ids.items():
        actual = pq.read_table(result[name][0]).to_pylist()
        assert sorted(actual, key=lambda row: row["id"]) == [
            {"id": index, "value": by_id[index]} for index in ids]
    distinct = pq.read_table(result["distinct"][0]).to_pylist()
    assert len(distinct) == 1 and distinct[0]["value"] in {"same\x00", "SAME\x00"}
    grouped = pq.read_table(result["group"][0]).to_pylist()
    assert len(grouped) == 1 and grouped[0]["id"] == 1 and grouped[0]["total"] == 2
    assert grouped[0]["value"] in {"same\x00", "SAME\x00"}
    assert pq.read_table(result["aliases"][0]).to_pylist() == [
        {"a]b": 1, "select": "same\x00", "Case": "same\x00", "case": "same\x00",
         "white space 水": "same\x00"}]
    assert all(result[name][1]["sql"] == query for name, query in queries.items())



@pytest.mark.parametrize("query", [
    "WITH chosen AS (SELECT id,value FROM data) SELECT id,value FROM chosen",
    "SELECT id,value FROM data OPTION(MAXDOP 1)",
])
def test_sqlserver_text_transport_preserves_unsupported_query_refusal(tmp_path, query):
    from autoforecast.integrations.sqlserver import read_sources

    with pytest.raises(ValueError, match="unsupported or writable operator"):
        read_sources(None, queries={"rows": query}, tables={"data": "dbo.text_shapes"},
                     output_root=tmp_path / "capture")
    assert not (tmp_path / "capture").exists()


def test_sqlserver_text_batches_are_bounded_and_complete(
    sqlserver_store, sqlserver_database, tmp_path, monkeypatch,
):
    import pyarrow.parquet as pq
    from mssql_python.cursor import Cursor

    from autoforecast import _logical_hash
    from autoforecast.integrations.sqlserver import _execute, read_sources
    from autoforecast.snapshots import RetainedInputs

    source, writer, _, _ = sqlserver_database
    _execute(writer, "CREATE TABLE dbo.text_batches (id INT NOT NULL,value NVARCHAR(MAX))")
    _execute(writer, "INSERT INTO dbo.text_batches "
             "SELECT TOP (8195) ROW_NUMBER() OVER (ORDER BY a.object_id,b.object_id),"
             "CONCAT(N'row',ROW_NUMBER() OVER (ORDER BY a.object_id,b.object_id),NCHAR(0)) "
             "FROM sys.all_objects a CROSS JOIN sys.all_objects b")
    sizes = []
    original = Cursor.arrow_batch

    def measured(cursor, batch_size=8192):
        batch = original(cursor, batch_size)
        sizes.append((batch_size, batch.num_rows))
        return batch

    monkeypatch.setattr(Cursor, "arrow_batch", measured)
    monkeypatch.setattr(_logical_hash, "fingerprint", lambda *a, **kw: pytest.fail(
        "table-only text batches must not hash dataset rows"))
    result = read_sources(source, queries={"rows": "SELECT id,value FROM data ORDER BY id"},
                          tables={"data": "dbo.text_batches"}, output_root=tmp_path / "source")
    expected = [{"id": index, "value": "row" + str(index) + "\x00"}
                for index in range(1, 8196)]
    assert pq.read_table(result["rows"][0]).to_pylist() == expected
    assert sizes == [(0, 0), (8192, 8192), (8192, 3), (8192, 0)]
    sizes.clear()
    store = sqlserver_store
    with RetainedInputs(mode="table", connection="archive", store=store,
                        scratch_root=tmp_path / "scratch") as manager:
        reference = manager.capture(result["rows"][0])
        try:
            with manager.open(reference) as saved:
                assert pq.read_table(saved).sort_by("id").to_pylist() == expected
            assert sizes == [(0, 0), (8192, 8192), (8192, 3), (8192, 0)]
        finally:
            store.drop_unretained(reference.record())


@pytest.mark.parametrize("direction", ["source", "native"])
@pytest.mark.parametrize("failure", ["conversion", "write"])
def test_sqlserver_text_transfer_failure_cleans_up(
    request, tmp_path, monkeypatch, direction, failure,
):
    import uuid

    import pyarrow as pa
    import pyarrow.parquet as pq
    from mssql_python.cursor import Cursor

    from autoforecast.integrations.sqlserver import _execute, read_sources

    store, record = None, None
    if direction == "source":
        source, writer, _, _ = request.getfixturevalue("sqlserver_database")
        _execute(writer, "CREATE TABLE dbo.text_failure (id INT,value NVARCHAR(MAX))")
        _execute(writer, "INSERT INTO dbo.text_failure VALUES (1,N'value'+NCHAR(0))")
    else:
        store = request.getfixturevalue("sqlserver_store")
        path = tmp_path / "input.parquet"
        pq.write_table(pa.table({
            "id": pa.array([1], type=pa.int32()), "value": ["value\x00"],
        }), path)
        record = store.capture(path, capture_id=uuid.uuid4().hex)
        source = store.reader
    before = (source.autocommit, _execute(source, "SELECT @@TRANCOUNT"), _execute(
        source, "SELECT transaction_isolation_level FROM sys.dm_exec_sessions "
        "WHERE session_id=@@SPID"))
    closed = []
    original_reader = Cursor.arrow_reader
    original_writer = pq.ParquetWriter

    class Reader:
        def __init__(self, inner):
            self.inner = inner

        def __enter__(self):
            self.inner.__enter__()
            return self

        def __iter__(self):
            if failure == "conversion":
                yield pa.record_batch([pa.array(["invalid integer"]), pa.array(["value\x00"])],
                                      names=["id", "value"])
            else:
                yield from self.inner

        def __exit__(self, *args):
            try:
                return self.inner.__exit__(*args)
            finally:
                closed.append(True)

    class Writer:
        def __init__(self, *args, **kwargs):
            self.inner = original_writer(*args, **kwargs)

        def __enter__(self):
            self.inner.__enter__()
            return self

        def write_table(self, *args, **kwargs):
            raise RuntimeError("injected text transfer write failure")

        def __exit__(self, *args):
            return self.inner.__exit__(*args)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(Cursor, "arrow_reader", lambda cursor, batch_size=8192: Reader(
                original_reader(cursor, batch_size)))
            if failure == "write":
                patch.setattr(pq, "ParquetWriter", Writer)
            with pytest.raises((pa.ArrowInvalid, RuntimeError, TypeError),
                               match="integer|int32|write failure"):
                if direction == "source":
                    read_sources(
                        source, queries={"rows": "SELECT id,value FROM data"},
                        tables={"data": "dbo.text_failure"}, output_root=tmp_path / "failed",
                    )
                else:
                    store.read(record, tmp_path / "failed")
        assert closed == [True]
        assert (source.autocommit, _execute(source, "SELECT @@TRANCOUNT"), _execute(
            source, "SELECT transaction_isolation_level FROM sys.dm_exec_sessions "
            "WHERE session_id=@@SPID")) == before
        if store is not None:
            store.inspect(record)
            store.read(record, tmp_path / "recovered")
            actual = pq.read_table(tmp_path / "recovered").to_pylist()
        else:
            result = read_sources(
                source, queries={"rows": "SELECT id,value FROM data"},
                tables={"data": "dbo.text_failure"}, output_root=tmp_path / "recovered",
            )
            actual = pq.read_table(result["rows"][0]).to_pylist()
        assert actual == [{"id": 1, "value": "value\x00"}]
    finally:
        if store is not None and record is not None:
            store.release(record)


@pytest.mark.parametrize("operation", ["source", "capture", "read"])
def test_sqlserver_lossy_cursor_is_refused_before_transfer(request, tmp_path, operation):
    import uuid

    import pyarrow as pa
    import pyarrow.parquet as pq

    from autoforecast.integrations.sqlserver import _execute, read_sources

    class Cursor:
        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            if name == "arrow_reader":
                raise AttributeError(name)
            return getattr(self.inner, name)

        def __enter__(self):
            self.inner.__enter__()
            return self

        def __exit__(self, *args):
            return self.inner.__exit__(*args)

    class Connection:
        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def cursor(self):
            return Cursor(self.inner.cursor())

    destination = tmp_path / "transfer"
    if operation == "source":
        source, writer, _, _ = request.getfixturevalue("sqlserver_database")
        _execute(writer, "CREATE TABLE dbo.text_unsupported (value NVARCHAR(MAX))")
        with pytest.raises(ValueError, match="mssql-python 1.17 Arrow streaming"):
            read_sources(Connection(source), queries={"rows": "SELECT value FROM data"},
                         tables={"data": "dbo.text_unsupported"}, output_root=destination)
        assert not destination.exists()
        assert _execute(source, "SELECT @@TRANCOUNT") == [(0,)]
        return
    store = request.getfixturevalue("sqlserver_store")
    path = tmp_path / "input.parquet"
    pq.write_table(pa.table({"value": ["value\x00"]}), path)
    record = store.capture(path, capture_id=uuid.uuid4().hex) if operation == "read" else None
    reader = store.reader
    before = _execute(reader, "SELECT capture_id FROM archive.af_captures")
    store.reader = Connection(reader)
    try:
        with pytest.raises(ValueError, match="mssql-python 1.17 Arrow streaming"):
            if operation == "read":
                store.read(record, destination)
            else:
                store.capture(path, capture_id=uuid.uuid4().hex)
        assert not destination.exists()
        assert _execute(reader, "SELECT capture_id FROM archive.af_captures") == before
        assert _execute(store.writer, "SELECT @@TRANCOUNT") == [(0,)]
    finally:
        store.reader = reader
        if record is not None:
            store.release(record)


@pytest.mark.parametrize("mode", ["hash", "table", "table+hash"])
@pytest.mark.parametrize("source_kind", ["frame", "sql"])
def test_sqlserver_public_text_registration_is_exact(
    request, lifecycle_spark, tmp_path, monkeypatch, mode, source_kind,
):
    import datetime as dt

    import yaml

    from autoforecast import _logical_hash, register_study
    from autoforecast.integrations.sqlserver import _execute
    from autoforecast.study_registration import StudyRegistry
    from autoforecast.test_study_registration import configuration

    store = request.getfixturevalue("sqlserver_store") if mode != "hash" else None
    note = "before\x00after 水🙂\x00"
    connections = {"archive": store} if store is not None else {}
    if mode == "table":
        monkeypatch.setattr(_logical_hash, "fingerprint", lambda *a, **kw: pytest.fail(
            "table-only public text registration must not hash dataset rows"))
    config, data = None, None
    if source_kind == "sql":
        source, writer, _, _ = request.getfixturevalue("sqlserver_database")
        connections["source"] = source
        if store is not None:
            login = _execute(source, "SELECT ORIGINAL_LOGIN()")[0][0]
            _execute(store.test_admin, f"CREATE USER [{login}] FOR LOGIN [{login}]")
            _execute(store.test_admin, f"GRANT CONNECT TO [{login}]")
        # One series: five training, two validation, seven refit and two test rows.
        config = configuration(tmp_path)
        rows = [{"item": "A", "day": dt.date(2024, 1, day), "value": float(day)}
                for day in range(1, 10)]
        _execute(writer, "CREATE TABLE dbo.text_observations "
                 "(item NVARCHAR(10),day DATE,value FLOAT,note NVARCHAR(MAX))")
        with writer.cursor() as cursor:
            cursor.executemany("INSERT INTO dbo.text_observations "
                               "VALUES (?,?,?,CONVERT(NVARCHAR(MAX),?))", [
                (row["item"], row["day"], row["value"], note.encode("utf-16le")) for row in rows
            ])
        (tmp_path / "source.sql").write_text("SELECT item,day,value,note FROM observations")
        settings = yaml.safe_load(config.read_text())
        settings["source"] = {"connection": "source", "sql": "source.sql",
                              "tables": {"observations": "dbo.text_observations"}}
        settings["dataset"]["dataset"]["panel"]["static"] = ["note"]
        config.write_text(yaml.safe_dump(settings))
        training_count, testing_count = 5, 2
    else:
        # One declared synthetic series, nine rows: seven training and two test.
        data = lifecycle_spark.createDataFrame([
            ("A", dt.date(2024, 1, day), float(day), note) for day in range(1, 10)
        ], "item string,day date,value double,note string")
        training_count, testing_count = 7, 2
    with StudyRegistry(
        tmp_path / "registry", spark=lifecycle_spark, snapshot_mode=mode,
        connections=connections,
        snapshots=({"connection": "archive", "namespace": store.schema}
                   if store is not None else None),
    ) as registry:
        study_id = None
        try:
            if config is not None:
                study_id = register_study(config, registry=registry)
                _execute(writer, "UPDATE dbo.text_observations SET note=N'changed'")
            else:
                study_id = register_study(
                    registry=registry, training_data=data.filter('day <= "2024-01-07"'),
                    testing_data=data.filter('day >= "2024-01-08"'), series_id=["item"],
                    date_col="day", target_col="value", target_unit="units", frequency="D",
                    metric="mae", training_date_range=("2024-01-01", "2024-01-07"),
                    testing_date_range=("2024-01-08", "2024-01-09"),
                    feature_roles={"static": ["note"], "season_length": 1},
                )
            with registry.load_study(study_id) as study:
                expected_counts = {"training": training_count, "testing": testing_count}
                if config is not None:
                    expected_counts.update(validation=2, refit=7)
                for stage, count in expected_counts.items():
                    frame = getattr(study, stage + "_data")
                    assert [row["note"] for row in frame.select("note").collect()] == [note] * count
        finally:
            if study_id is not None and store is not None:
                registry.release(study_id)


def test_sqlserver_arrow_source_preserves_scalar_boundaries(sqlserver_database, tmp_path):
    import datetime as dt
    from decimal import Decimal

    import pyarrow as pa
    import pyarrow.parquet as pq

    from autoforecast.integrations.sqlserver import _execute, read_sources

    source, writer, _, _ = sqlserver_database
    fields = [
        ("big_low", "BIGINT", pa.int64(), "-9223372036854775808", -(2**63)),
        ("big_high", "BIGINT", pa.int64(), "9223372036854775807", 2**63 - 1),
        ("tiny", "TINYINT", pa.uint8(), "255", 255),
        ("small_low", "SMALLINT", pa.int16(), "-32768", -32768),
        ("small_high", "SMALLINT", pa.int16(), "32767", 32767),
        ("int_low", "INT", pa.int32(), "-2147483648", -(2**31)),
        ("int_high", "INT", pa.int32(), "2147483647", 2**31 - 1),
        ("real_value", "REAL", pa.float32(), "1.23456789", 1.23456789),
        ("float_value", "FLOAT", pa.float64(), "16777217.125", 16777217.125),
        ("money_low", "MONEY", pa.decimal128(19, 4), "-922337203685477.5808",
         Decimal("-922337203685477.5808")),
        ("money_high", "MONEY", pa.decimal128(19, 4), "922337203685477.5807",
         Decimal("922337203685477.5807")),
        ("smallmoney_low", "SMALLMONEY", pa.decimal128(10, 4), "-214748.3648",
         Decimal("-214748.3648")),
        ("smallmoney_high", "SMALLMONEY", pa.decimal128(10, 4), "214748.3647",
         Decimal("214748.3647")),
        ("decimal_integer", "DECIMAL(38,0)", pa.decimal128(38, 0), "9" * 38,
         Decimal("9" * 38)),
        ("decimal_fraction", "DECIMAL(38,18)", pa.decimal128(38, 18),
         "99999999999999999999.999999999999999999",
         Decimal("99999999999999999999.999999999999999999")),
        ("decimal_scale", "NUMERIC(38,38)", pa.decimal128(38, 38),
         "0.12345678901234567890123456789012345678",
         Decimal("0.12345678901234567890123456789012345678")),
        ("date_low", "DATE", pa.date32(), "'0001-01-01'", dt.date(1, 1, 1)),
        ("date_high", "DATE", pa.date32(), "'9999-12-31'", dt.date(9999, 12, 31)),
        ("at_low", "DATETIME2(6)", pa.timestamp("us"), "'0001-01-01T00:00:00.000001'",
         dt.datetime(1, 1, 1, 0, 0, 0, 1)),
        ("at_high", "DATETIME2(6)", pa.timestamp("us"), "'9999-12-31T23:59:59.999999'",
         dt.datetime(9999, 12, 31, 23, 59, 59, 999999)),
        ("fixed_binary", "BINARY(3)", pa.binary(), "0x00ff00", b"\x00\xff\x00"),
        ("binary_value", "VARBINARY(MAX)", pa.binary(), "0xff0001", b"\xff\x00\x01"),
        ("flag_true", "BIT", pa.bool_(), "1", True),
        ("flag_false", "BIT", pa.bool_(), "0", False),
        ("text_value", "NVARCHAR(MAX)", pa.string(), "N'end'+NCHAR(0)", "end\x00"),
    ]
    _execute(writer, "CREATE TABLE dbo.scalar_boundaries (id INT NOT NULL," + ",".join(
        "[" + name + "] " + declared for name, declared, *_ in fields) + ")")
    _execute(writer, "INSERT INTO dbo.scalar_boundaries VALUES (1," + ",".join(
        expression for _, _, _, expression, _ in fields) + ")")
    _execute(writer, "INSERT INTO dbo.scalar_boundaries VALUES (2," + ",".join(
        "NULL" for _ in fields) + ")")
    schema = pa.schema([pa.field("id", pa.int32(), nullable=False), *(
        pa.field(name, dtype) for name, _, dtype, _, _ in fields)])
    expected = pa.Table.from_pylist([
        dict(id=1, **{name: value for name, _, _, _, value in fields}),
        dict(id=2, **{name: None for name, *_ in fields}),
    ], schema=schema)
    result = read_sources(source, queries={
        "rows": "SELECT * FROM data ORDER BY id",
        "null": "SELECT * FROM data WHERE id=2",
        "empty": "SELECT * FROM data WHERE id<0",
    }, tables={"data": "dbo.scalar_boundaries"}, output_root=tmp_path / "capture")
    actual = pq.read_table(result["rows"][0])
    assert actual.equals(expected)
    assert pq.read_table(result["null"][0]).equals(expected.slice(1))
    empty = pq.read_table(result["empty"][0])
    assert empty.num_rows == 0 and empty.schema.equals(expected.schema)


@pytest.mark.parametrize("suffix", [
    ";", ";\n", "; /* tail */", "; -- tail", "; # tail",
    " -- trailing comment", " # trailing comment", " -- trailing comment\n",
    " /* ordinary tail */", "", ";\r\n",
], ids=["semicolon", "semicolon-newline", "semicolon-block", "semicolon-line",
        "semicolon-hash", "line-comment", "hash-comment", "line-newline",
        "block-comment", "plain", "semicolon-crlf"])
def test_mysql_source_statement_boundaries_preserve_values(mysql_source, tmp_path, suffix):
    from autoforecast.integrations.mysql import read_sources
    from autoforecast.snapshots import bind_source_queries

    admin, reader, namespace = mysql_source
    # One synthetic row checks both exact float transport and quoted SQL punctuation.
    with admin.cursor() as cursor:
        cursor.execute(f"CREATE TABLE `{namespace}`.sales (id INT, small FLOAT) ENGINE=InnoDB")
        cursor.execute(f"INSERT INTO `{namespace}`.sales VALUES (11,1.23456789)")
    label = "semi; slash /* ordinary */ hash# -- "
    query = "-- leading comment\nSELECT id,small,'" + label + "' AS label FROM sales" + suffix
    tables = {"sales": namespace + ".sales"}
    bound = bind_source_queries(
        {"primary": query}, tables, dialect="mysql",
        sources={"sales": f"`{namespace}`.`sales`"},
    )["primary"]
    with reader.cursor() as cursor:
        cursor.execute(bound)
        row = cursor.fetchone()
        assert row[0] == 11 and row[2] == label
        assert cursor.fetchone() is None
    result = read_sources(reader, queries={"primary": query, "audit": query}, tables=tables,
                          output_root=tmp_path / "capture")
    expected_small = pa.array([1.23456789], type=pa.float32())[0].as_py()
    assert set(result) == {"primary", "audit"}
    for name in result:
        actual = pq.read_table(result[name][0])
        assert actual.to_pylist() == [{"id": 11, "small": expected_small, "label": label}]
        assert actual.schema.field("small").type == pa.float32()
        assert result[name][1]["sql"] == query
