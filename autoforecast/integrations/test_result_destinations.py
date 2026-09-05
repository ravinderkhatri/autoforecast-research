"""Live native-client contracts; local PostgreSQL in CI, opt-in BigQuery sandbox."""

import datetime as dt
import itertools
import json
import os
import pathlib
import uuid
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from pyspark.sql import SparkSession

from autoforecast import register_study, run_study
from autoforecast.study_registration import StudyRegistry
from autoforecast.study_results import _search_windows, read_workflow_results
from autoforecast.test_study_registration import configuration
from autoforecast.test_study_tuning import config, implementation, predict


@pytest.fixture(scope="module", params=["D", "H"])
def retained(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("publication")
    spark = (SparkSession.builder.master("local[2]").appName("publication-test")
             .config("spark.sql.shuffle.partitions", "2")
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.ui.enabled", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    try:
        path = configuration(root)
        settings = yaml.safe_load(path.read_text())
        settings["recipe"]["tuning_windows"] = [
            {"origin": "2024-01-03", "horizon": 2},
            {"origin": "2024-01-04", "horizon": 1},
        ]
        if request.param == "H":
            def hourly(value):
                return dt.datetime(2024, 1, 1) + dt.timedelta(
                    hours=dt.date.fromisoformat(str(value)).day - 1,
                )

            source = root / "source.parquet"
            observations = pq.read_table(source).to_pylist()
            for row in observations:
                row["day"] = hourly(row["day"])
            pq.write_table(pa.Table.from_pylist(observations), source)
            settings["dataset"]["dataset"]["panel"]["frequency"] = "H"
            recipe = settings["recipe"]
            recipe["frequency"] = "H"
            for key in ("training_start", "final_train_cutoff"):
                recipe[key] = hourly(recipe[key]).isoformat()
            recipe["validation_origins"] = [hourly(value).isoformat()
                                             for value in recipe["validation_origins"]]
            for window in recipe["tuning_windows"]:
                window["origin"] = hourly(window["origin"]).isoformat()
        path.write_text(yaml.safe_dump(settings))
        registry = StudyRegistry.from_config(path, spark=spark)
        study_id = register_study(path, registry=registry)
        profile = config(root, budget=2)
        settings = yaml.safe_load(profile.read_text())
        settings["selection"] = {"stage": "tuning", "origins": "all", "aggregation": "mean"}
        profile.write_text(yaml.safe_dump(settings))
        calls = []
        partitions = ["series_id", "date"] if request.param == "D" else ["series_id"]

        def model(request):
            calls.append(request.origin)
            output = predict(request)
            part = pathlib.Path(output.forecasts.path) / "part.parquet"
            rows = pq.read_table(part).to_pylist()
            for index, row in enumerate(rows):
                row["yhat"] = (-(request.model_parameters["level"] + 1) * 1e-12 if index % 2
                               else -(request.model_parameters["level"] + 2.5 + index))
            part.unlink()
            pq.write_to_dataset(
                pa.Table.from_pylist(rows), root_path=part.parent,
                partition_cols=partitions,
            )
            return output

        workflow = run_study(study_id, model, registry=registry, model_config=profile,
                             implementation=implementation(), tracking="none", publish=False)
        assert len(calls) == 6
        yield registry, study_id, workflow, calls
    finally:
        spark.stop()


@pytest.fixture(params=["postgres", "bigquery"])
def destination(request):
    prefix = os.environ.get("AUTOFORECAST_TEST_TABLE_PREFIX", "af_test_") + uuid.uuid4().hex
    if request.param == "postgres":
        dsn = os.environ.get("AUTOFORECAST_TEST_POSTGRES_DSN")
        if not dsn:
            pytest.skip("set AUTOFORECAST_TEST_POSTGRES_DSN for the live PostgreSQL contract")
        import psycopg
        from psycopg import sql

        from autoforecast.integrations.postgres import PostgresResults

        with psycopg.connect(dsn, autocommit=True) as connection:
            writer = PostgresResults(connection, results=prefix + "_results",
                                     forecasts=prefix + "_forecasts")
            writer.create_tables()

            def read(kind):
                with connection.cursor() as cursor:
                    cursor.execute(sql.SQL("SELECT * FROM {}").format(writer.tables[kind]))
                    names = [col.name for col in cursor.description]
                    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]

            def corrupt():
                connection.execute(sql.SQL("UPDATE {} SET yhat = yhat + 1").format(
                    writer.tables["forecasts"],
                ))

            try:
                yield writer, read, corrupt
            finally:
                for table in writer.tables.values():
                    connection.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(table))
    else:
        dataset = os.environ.get("AUTOFORECAST_TEST_BIGQUERY_DATASET")
        location = os.environ.get("AUTOFORECAST_TEST_BIGQUERY_LOCATION")
        if not dataset or not location:
            pytest.skip("set BigQuery dataset and location for the live cloud contract")
        from google.cloud import bigquery

        from autoforecast.integrations.bigquery import BigQueryResults

        client = bigquery.Client(project=dataset.split(".")[0])
        writer = BigQueryResults(client, dataset=dataset, location=location,
                                 results=prefix + "_results", forecasts=prefix + "_forecasts")
        try:
            writer.create_tables()

            def read(kind):
                return [dict(row) for row in client.query(
                    f"SELECT * FROM `{writer.tables[kind]}`", location=location,
                ).result()]

            def corrupt():
                query = f"UPDATE `{writer.tables['forecasts']}` SET yhat = yhat + 1 WHERE TRUE"
                client.query(query, location=location).result()

            yield writer, read, corrupt
        finally:
            for table in writer.tables.values():
                client.delete_table(table, not_found_ok=True)
            client.close()


def test_publication_preserves_every_window_and_retries(destination, retained, monkeypatch):
    from autoforecast import study_scoring, study_tuning

    writer, read, corrupt = destination
    registry, study_id, workflow, calls = retained
    monkeypatch.setattr(study_scoring, "score_study", lambda *a, **k: pytest.fail("rescored"))
    monkeypatch.setattr(study_tuning, "score_study", lambda *a, **k: pytest.fail("rescored"))
    windows, _, _, _ = _search_windows(workflow.tuning_path)
    first_windows = [pq.read_table(window["forecasts"].path).to_pylist()
                     for window, trial in windows if trial["window"] == 0]
    assert first_windows[0] != first_windows[1]
    expected = {(trial["trial_id"], trial["attempt_id"], trial["window"]): window
                for window, trial in windows}
    for _ in range(2):
        assert registry.publish_results(study_id, workflow.path,
                                        destination=writer)["status"] == "published"
    rows, forecasts = read("study_results"), read("forecasts")
    # Setup must preserve an already-published run, not recreate its tables.
    writer.create_tables()
    assert Counter(json.dumps(row, sort_keys=True) for row in read("study_results")) == Counter(
        json.dumps(row, sort_keys=True) for row in rows
    )
    assert Counter(json.dumps(row, sort_keys=True) for row in read("forecasts")) == Counter(
        json.dumps(row, sort_keys=True) for row in forecasts
    )
    assert len(rows) == 6 and len(forecasts) == 20 and len(calls) == 6
    for row in rows:
        if row["stage"] == "tuning":
            evidence = expected[(row["trial_id"], row["attempt_id"], row["window_index"])]
        else:
            evidence = workflow.validation if row["stage"] == "validation" else workflow.testing
        assert json.loads(row["evidence_json"]) == evidence["score"]
        assert row["model_sha"] == evidence["model_sha256"]
        assert pathlib.Path(row["model_uri"]).is_dir()
        identity = ("stage", "search_id", "trial_id", "attempt_id", "window_index", "origin",
                    "horizon", "study_id", "study_sha", "run_id")
        actual = Counter((value["series_id"], value["date"], value["yhat"])
                         for value in forecasts if all(value[key] == row[key] for key in identity))
        assert actual == Counter((value["series_id"], str(value["date"]), value["yhat"])
                                 for value in pq.read_table(evidence["forecasts"].path).to_pylist())
    analysis = read_workflow_results(workflow.path)
    assert Counter((r["series_id"], str(r["target_date"]), r["prediction"])
                   for r in analysis.rows.to_pylist()) == Counter(
        (r["series_id"], r["date"], r["yhat"]) for r in forecasts
    )
    corrupt()
    with pytest.raises(ValueError, match="conflicts"):
        registry.publish_results(study_id, workflow.path, destination=writer)
    assert len(read("study_results")) == 6 and len(read("forecasts")) == 20


def test_interrupted_publication_recovers_without_duplicates(destination, retained, monkeypatch):
    writer, read, _ = destination
    registry, study_id, workflow, calls = retained
    module = __import__(type(writer).__module__, fromlist=["_publication_batches"])
    batches = module._publication_batches

    def interrupt(publication, kind):
        if kind == "study_results":
            raise OSError("interrupted before result rows")
        yield from batches(publication, kind)

    with monkeypatch.context() as patch:
        patch.setattr(module, "_publication_batches", interrupt)
        with pytest.raises(OSError, match="interrupted before result rows"):
            registry.publish_results(study_id, workflow.path, destination=writer)
    assert read("study_results") == []
    expected = 0 if type(writer).__name__ == "PostgresResults" else 20
    assert len(read("forecasts")) == expected
    registry.publish_results(study_id, workflow.path, destination=writer)
    assert len(read("forecasts")) == 20 and len(read("study_results")) == 6
    assert len(calls) == 6


def test_concurrent_publishers_keep_one_copy(destination, retained):
    import copy
    from concurrent.futures import ThreadPoolExecutor

    writer, read, _ = destination
    registry, study_id, workflow, calls = retained

    def publish():
        if type(writer).__name__ == "PostgresResults":
            import psycopg

            with psycopg.connect(os.environ["AUTOFORECAST_TEST_POSTGRES_DSN"],
                                 autocommit=True) as connection:
                peer = copy.copy(writer)
                peer.connection = connection
                return registry.publish_results(study_id, workflow.path, destination=peer)
        return registry.publish_results(study_id, workflow.path, destination=writer)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(publish) for _ in range(2)]
        assert all(future.result()["status"] == "published" for future in futures)
    assert len(read("forecasts")) == 20 and len(read("study_results")) == 6
    assert len(calls) == 6


@pytest.mark.parametrize("destination", ["postgres"], indirect=True)
@pytest.mark.parametrize("trigger_kind", ["before", "after", "deferred"])
def test_postgres_refuses_values_changed_by_table_triggers(destination, retained, trigger_kind):
    from psycopg import sql

    writer, read, _ = destination
    registry, study_id, workflow, _ = retained
    connection = writer.connection
    if trigger_kind != "before":
        body = sql.SQL(
            "UPDATE {} SET yhat = yhat + 1 WHERE study_id = NEW.study_id "
            "AND study_sha = NEW.study_sha AND run_id = NEW.run_id; RETURN NEW;",
        ).format(writer.tables["forecasts"])
        timing = sql.SQL("CONSTRAINT" if trigger_kind == "deferred" else "")
        event = sql.SQL("AFTER INSERT ON {}").format(writer.tables["study_results"])
        if trigger_kind == "deferred":
            event += sql.SQL(" DEFERRABLE INITIALLY DEFERRED")
    else:
        body = sql.SQL("NEW.yhat := NEW.yhat + 1; RETURN NEW;")
        timing = sql.SQL("")
        event = sql.SQL("BEFORE INSERT ON {}").format(writer.tables["forecasts"])
    connection.execute(sql.SQL(
        "CREATE FUNCTION pg_temp.change_forecast() RETURNS trigger "
        "LANGUAGE plpgsql AS $$ BEGIN {} END $$",
    ).format(body))
    connection.execute(sql.SQL(
        "CREATE {} TRIGGER change_forecast {} FOR EACH ROW "
        "EXECUTE FUNCTION pg_temp.change_forecast()",
    ).format(timing, event))
    with pytest.raises(ValueError, match="conflicts"):
        registry.publish_results(study_id, workflow.path, destination=writer)
    assert read("forecasts") == [] and read("study_results") == []


@pytest.mark.parametrize("names", [{"schema": ""}, {"results": ""}, {"forecasts": None},
                                  {"results": "same", "forecasts": "same"}])
def test_postgres_destination_rejects_invalid_names(names):
    pytest.importorskip("psycopg")
    from autoforecast.integrations.postgres import PostgresResults

    with pytest.raises(ValueError):
        PostgresResults(object(), **names)


@pytest.mark.parametrize("destination", ["postgres"], indirect=True)
def test_postgres_preserves_caller_transaction(destination, retained):
    import psycopg
    from psycopg.pq import TransactionStatus

    writer, read, _ = destination
    registry, study_id, workflow, _ = retained
    connection = writer.connection
    connection.execute("BEGIN")
    connection.execute("CREATE TEMP TABLE caller_pending (value integer)")
    connection.execute("INSERT INTO caller_pending VALUES (42)")
    for operation in (writer.create_tables, lambda: registry.publish_results(
        study_id, workflow.path, destination=writer,
    )):
        with pytest.raises(ValueError, match="idle connection"):
            operation()
        assert connection.info.transaction_status == TransactionStatus.INTRANS
        assert connection.execute("SELECT value FROM caller_pending").fetchone()[0] == 42
    connection.rollback()
    assert connection.execute("SELECT to_regclass('pg_temp.caller_pending')").fetchone()[0] is None
    connection.execute("BEGIN")
    with pytest.raises(psycopg.errors.DivisionByZero):
        connection.execute("SELECT 1 / 0")
    with pytest.raises(ValueError, match="idle connection"):
        registry.publish_results(study_id, workflow.path, destination=writer)
    assert connection.info.transaction_status == TransactionStatus.INERROR
    connection.rollback()
    assert read("forecasts") == [] and read("study_results") == []


@pytest.mark.parametrize("destination", ["postgres"], indirect=True)
@pytest.mark.parametrize("kind", ["forecasts", "study_results"])
def test_postgres_schema_refusal_leaves_existing_schema_and_rows(destination, retained, kind):
    from psycopg import sql

    writer, read, _ = destination
    registry, study_id, workflow, _ = retained
    writer.connection.execute(sql.SQL("ALTER TABLE {} ADD COLUMN caller_column text").format(
        writer.tables[kind],
    ))
    writer.create_tables()
    with pytest.raises(ValueError, match="schema differs"):
        registry.publish_results(study_id, workflow.path, destination=writer)
    columns = writer.connection.execute(sql.SQL("SELECT * FROM {} LIMIT 0").format(
        writer.tables[kind],
    )).description
    assert columns[-1].name == "caller_column"
    assert read("forecasts") == [] and read("study_results") == []


def test_publication_batches_preserves_multiple_fragments_and_partial_batch(tmp_path):
    from autoforecast.study_results import _publication_batches

    labels = {"study_id": "study", "study_sha": "sha", "run_id": "run", "stage": "tuning",
              "search_id": "search", "trial_id": "trial", "attempt_id": "attempt",
              "window_index": 2, "origin": "2024-01-01", "horizon": 1}
    rows = [{"series_id": str(index), "date": dt.date(2024, 1, 2),
             "yhat": -1e-12 if index % 2 else -2.5} for index in range(65540)]
    pq.write_table(pa.Table.from_pylist(rows[:65537]), tmp_path / "first.parquet")
    nested = tmp_path / "partition"
    nested.mkdir()
    pq.write_table(pa.Table.from_pylist(rows[65537:]), nested / "last.parquet")
    batches = list(_publication_batches({"forecasts": [(str(tmp_path), labels)]}, "forecasts"))
    assert [batch.num_rows for batch in batches] == [65536, 1, 3]
    actual = [row for batch in batches for row in batch.to_pylist()]
    assert actual == [{**labels, **row, "date": str(row["date"])} for row in rows]


def test_workflow_rejects_internally_consistent_search_for_other_configuration(
    retained, monkeypatch,
):
    import copy
    import dataclasses

    from autoforecast import study_results

    _, _, workflow, _ = retained
    load = study_results._search_windows
    for field in ("study", "implementation", "profile_sha256", "seed", "model_id",
                  "trial_id", "winner_config"):
        def changed(path, field=field):
            windows, runs, info, identity = load(path)
            info = copy.deepcopy(info)
            selected = info["selected"]
            if field == "trial_id":
                selected = dataclasses.replace(selected, trial_id="another-trial")
            elif field == "winner_config":
                selected = dataclasses.replace(selected, winner_config=path / "another-config.json")
            else:
                context = info["provenance"]
                if field == "study":
                    context[field] = {**context[field], "sha256": "b" * 64}
                elif field == "implementation":
                    context[field] = {**context[field], "name": "another-model"}
                elif field == "seed":
                    context[field] += 1
                else:
                    context[field] = "b" * 64 if field == "profile_sha256" else "another-model"
                selected = dataclasses.replace(selected, provenance=copy.deepcopy(context))
            info["selected"] = selected
            return windows, runs, info, identity

        with monkeypatch.context() as patch:
            patch.setattr(study_results, "_search_windows", changed)
            with pytest.raises(ValueError, match="different workflow configuration"):
                read_workflow_results(workflow.path)


@pytest.mark.parametrize("options", [
    {"dataset": "project.dataset.extra"}, {"dataset": "project`bad.dataset"},
    {"location": ""}, {"location": None}, {"results": "a.b"},
    {"results": "same", "forecasts": "same"},
])
def test_bigquery_destination_rejects_invalid_configuration(options):
    from autoforecast.integrations.bigquery import BigQueryResults

    settings = {"dataset": "test-project.test_dataset", "location": "EU", **options}
    with pytest.raises(ValueError):
        BigQueryResults(object(), **settings)


class PublicationClient:
    """Drive failure branches with native table/schema types; live tests prove storage behavior."""

    def __init__(self, scenario="fresh", *, created="1000000000000"):
        bigquery = pytest.importorskip("google.cloud.bigquery")
        from autoforecast.integrations.bigquery import BigQueryResults

        self.table = bigquery.Table("test-project.test_dataset.forecasts",
                                    schema=BigQueryResults._schema("forecasts"))
        self.table._properties.update(type="TABLE", creationTime=created)
        self.scenario = scenario
        self.stages, self.deleted, self.lookups, self.insert_ids = [], [], [], []
        self.present = scenario in {"equal", "different"}
        self.identical = scenario == "equal"
        self.comparisons = 0

    def get_table(self, name):
        return self.table

    def create_table(self, table):
        assert table.expires is not None
        lifetime = table.expires - dt.datetime.now(dt.UTC)
        assert dt.timedelta(hours=23) < lifetime <= dt.timedelta(days=1)
        self.stages.append(table)
        return table

    def delete_table(self, name, *, not_found_ok):
        assert not_found_ok
        self.deleted.append(name)

    def load_table_from_file(self, stream, table, **kwargs):
        from types import SimpleNamespace

        assert kwargs["job_config"].create_disposition == "CREATE_NEVER"
        loaded = pq.read_table(stream)
        assert loaded.num_rows == 1 and loaded["yhat"][0].as_py() < 0

        def result():
            if self.scenario == "load_error":
                raise OSError("load failed")
        return SimpleNamespace(result=result)

    def _job(self, *, failed=False):
        from types import SimpleNamespace

        def result():
            if self.scenario == "job_error":
                raise OSError("job failed")
            self.present = True
            self.identical = self.scenario != "competing"
        return SimpleNamespace(done=lambda: failed or self.scenario == "completed",
                               error_result={"reason": "backendError"} if failed else None,
                               result=result)

    def get_job(self, job_id, **kwargs):
        from google.api_core.exceptions import NotFound

        assert kwargs["project"] == "test-project"
        self.lookups.append(job_id)
        if self.scenario == "exhausted" or (self.scenario == "failed" and job_id.endswith("_0")):
            return self._job(failed=True)
        if self.scenario in {"running", "completed", "competing"}:
            return self._job()
        if self.scenario == "race" and len(self.lookups) == 2:
            return self._job()
        raise NotFound("no job")

    def query(self, query, **kwargs):
        from types import SimpleNamespace

        from google.api_core.exceptions import Conflict

        if query.lstrip().startswith("SELECT"):
            self.comparisons += 1
            if self.scenario == "compare_error":
                raise OSError("comparison failed")
            assert len(kwargs["job_config"].query_parameters) == 3
            row = SimpleNamespace(present=self.present, identical=self.identical)
            return SimpleNamespace(result=lambda: iter([row]))
        assert kwargs["project"] == "test-project"
        assert kwargs["job_retry"] is None
        self.insert_ids.append(kwargs["job_id"])
        if self.scenario == "race":
            raise Conflict("another publisher submitted this job")
        return self._job()


def portable_payload(tmp_path, *, yhat=-1e-12):
    folder = tmp_path / uuid.uuid4().hex
    folder.mkdir()
    pq.write_table(pa.Table.from_pylist([
        {"series_id": "A", "date": dt.date(2024, 1, 2), "yhat": yhat},
    ]), folder / "part.parquet")
    scope = {"study_id": "study", "study_sha": "sha", "run_id": "run"}
    labels = {**scope, "stage": "test", "search_id": None, "trial_id": None,
              "attempt_id": None, "window_index": None, "origin": "2024-01-01", "horizon": 1}
    return {"forecasts": [(str(folder), labels)], "scope": scope}


@pytest.mark.parametrize("scenario", [
    "fresh", "equal", "different", "load_error", "compare_error", "job_error",
    "running", "completed", "failed", "race", "exhausted", "competing",
])
def test_bigquery_staging_and_stable_job_recovery(tmp_path, scenario):
    from autoforecast.integrations.bigquery import BigQueryResults

    client = PublicationClient(scenario)
    writer = BigQueryResults(client, dataset="test-project.test_dataset", location="EU")
    expected_error = {"different": ValueError, "load_error": OSError, "compare_error": OSError,
                      "job_error": OSError, "exhausted": RuntimeError, "competing": ValueError}
    if scenario in expected_error:
        with pytest.raises(expected_error[scenario]):
            writer._append("forecasts", portable_payload(tmp_path), tmp_path)
    else:
        writer._append("forecasts", portable_payload(tmp_path), tmp_path)
    assert client.deleted == [
        table.full_table_id or str(table.reference) for table in client.stages
    ]
    assert len(client.stages) == 1
    if scenario == "failed":
        assert len(client.lookups) == 2 and client.insert_ids == [client.lookups[1]]
        assert client.lookups[0].removesuffix("_0") == client.lookups[1].removesuffix("_1")
    elif scenario == "race":
        assert client.lookups == client.insert_ids * 2
    elif scenario == "exhausted":
        assert len(client.lookups) == 100 and client.insert_ids == []
    elif scenario in {"running", "completed", "competing"}:
        assert len(client.lookups) == 1 and client.insert_ids == []


def test_bigquery_job_identity_uses_scope_and_table_incarnation_not_payload(tmp_path):
    from autoforecast.integrations.bigquery import BigQueryResults

    ids = {}
    for change in ("baseline", "payload", "created", "table", "study_id", "study_sha", "run_id"):
        created = "1000000000001" if change == "created" else "1000000000000"
        client = PublicationClient(created=created)
        writer = BigQueryResults(client, dataset="test-project.test_dataset", location="EU",
                                 forecasts="other_forecasts" if change == "table" else "forecasts")
        publication = portable_payload(tmp_path, yhat=-2.0 if change == "payload" else -1.0)
        if change in publication["scope"]:
            publication["scope"][change] = "other"
        writer._append("forecasts", publication, tmp_path)
        ids[change] = client.insert_ids[0]
    assert ids.pop("payload") == ids["baseline"]
    assert len(set(ids.values())) == len(ids)


@pytest.mark.parametrize("change", ["schema", "view"])
def test_bigquery_rejects_incompatible_destination_before_staging(tmp_path, change):
    from autoforecast.integrations.bigquery import BigQueryResults

    client = PublicationClient()
    if change == "schema":
        client.table.schema = client.table.schema[:-1]
    else:
        client.table._properties["type"] = "VIEW"
    writer = BigQueryResults(client, dataset="test-project.test_dataset", location="EU")
    with pytest.raises(ValueError, match="schema differs"):
        writer._append("forecasts", portable_payload(tmp_path), tmp_path)
    assert not client.stages and not client.deleted and not client.lookups


def test_bigquery_concurrent_clients_share_destination_job_project(tmp_path):
    import re
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    from google.api_core.exceptions import Conflict, NotFound
    from google.cloud import bigquery

    from autoforecast.integrations.bigquery import BigQueryResults

    class Backend:
        def __init__(self):
            self.lock = threading.Lock()
            self.before = threading.Barrier(2)
            self.finished = threading.Barrier(2)
            self.stages = {}
            self.jobs = {}
            self.rows = []
            self.namespaces = []


    class Client:
        def __init__(self, backend, project):
            self.backend = backend
            self.project = project
            self.comparisons = 0
            self.expected = None
            self.table = bigquery.Table(
                "shared-project.publication.forecasts",
                schema=BigQueryResults._schema("forecasts"),
            )
            self.table._properties.update(type="TABLE", creationTime="1000000000000")

        def get_table(self, name):
            return self.table

        def create_table(self, table):
            with self.backend.lock:
                self.backend.stages[str(table.reference)] = []
            return table

        def delete_table(self, name, **kwargs):
            with self.backend.lock:
                self.backend.stages.pop(name, None)

        def load_table_from_file(self, stream, table, **kwargs):
            rows = pq.read_table(stream).to_pylist()
            self.expected = rows
            with self.backend.lock:
                self.backend.stages[table] = rows
            return SimpleNamespace(result=lambda: None)

        def get_job(self, job_id, *, location, project=None):
            key = (project or self.project, location, job_id)
            with self.backend.lock:
                if key not in self.backend.jobs:
                    raise NotFound("not in this project namespace")
                return self.backend.jobs[key]

        def query(self, query, *, location, project=None, **kwargs):
            if query.lstrip().startswith("SELECT"):
                self.comparisons += 1
                with self.backend.lock:
                    present = bool(self.backend.rows)
                    existing = Counter(json.dumps(r, sort_keys=True) for r in self.backend.rows)
                    incoming = Counter(json.dumps(r, sort_keys=True) for r in self.expected)
                    identical = existing == incoming
                first = self.comparisons == 1

                def result():
                    if first:
                        self.backend.before.wait(timeout=20)
                    return iter([SimpleNamespace(present=present, identical=identical)])
                return SimpleNamespace(result=result)
            key = (project or self.project, location, kwargs["job_id"])
            stage = re.search(r"SELECT \* FROM `([^`]+)`", query)[1]
            with self.backend.lock:
                if key in self.backend.jobs:
                    raise Conflict("job already exists in this project")
                rows = list(self.backend.stages[stage])
                state = {"done": False}

                def result():
                    with self.backend.lock:
                        if not state["done"]:
                            self.backend.rows.extend(rows)
                            self.backend.namespaces.append(key)
                            state["done"] = True
                    self.backend.finished.wait(timeout=20)
                job = SimpleNamespace(done=lambda: state["done"], error_result=None, result=result)
                self.backend.jobs[key] = job
                return job


    backend = Backend()

    def publish(project):
        directory = tmp_path / project
        directory.mkdir()
        writer = BigQueryResults(Client(backend, project),
                                 dataset="shared-project.publication", location="EU")
        writer._append("forecasts", portable_payload(directory), directory)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(publish, project)
                   for project in ("billing-project-a", "billing-project-b")]
        for future in futures:
            future.result()
    assert len(backend.rows) == 1
    assert len(backend.namespaces) == 1
    assert backend.namespaces[0][0] == "shared-project"
    assert not backend.stages


@pytest.mark.parametrize("layout", [
    state for state in itertools.product(("physical", "partition", "both"), repeat=3)
    if state != ("physical",) * 3
])
def test_publication_reads_hive_columns_before_projection(tmp_path, layout):
    from urllib.parse import quote

    from autoforecast.study_results import _publication_batches

    publication = portable_payload(tmp_path)
    _, labels = publication["forecasts"][0]
    directory = tmp_path / "hive"
    expected = []
    for index in range(2):
        row = {"series_id": f'{{"item":"product {index}"}}',
               "date": f"2024-01-0{index + 2}", "yhat": -2.5 - index}
        path = directory
        physical = {"dummy": index}
        for (field, value), state in zip(row.items(), layout, strict=True):
            if state != "physical":
                path /= f"{field}={quote(str(value), safe='')}"
            if state != "partition":
                physical[field] = str(value) if state == "physical" else "wrong"
                if field == "yhat":
                    physical[field] = str(value) if state == "physical" else "-99.5"
        path.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist([physical]), path / f"part-{index}.parquet")
        expected.append({**labels, **row})
    publication["forecasts"] = [(str(directory), labels)]
    actual = [row for batch in _publication_batches(publication, "forecasts")
              for row in batch.to_pylist()]
    assert sorted(actual, key=lambda row: row["series_id"]) == expected


@pytest.mark.parametrize("missing", ["series_id", "date", "yhat"])
def test_publication_refuses_missing_forecast_columns(tmp_path, missing):
    from autoforecast.study_results import _forecast_batches

    row = {"series_id": "A", "date": dt.date(2024, 1, 2), "yhat": -2.5}
    row.pop(missing)
    pq.write_table(pa.Table.from_pylist([row]), tmp_path / "part.parquet")
    with pytest.raises(ValueError, match="missing required columns"):
        list(_forecast_batches(tmp_path))


@pytest.mark.parametrize("spelling,exact", [
    ("0.5", True), ("-0.5", True), ("0.1", False), ("-0.1", False),
    ("0.10000000000000000001", False), ("-0.10000000000000000001", False),
    ("9007199254740994", True), ("-9007199254740994", True),
    ("9007199254740993", False), ("-9007199254740993", False),
    ("  -2.5\n", True), ("+1.0", True), ("001.0", True), (".5", True),
    ("-.5", True), ("1.", True), ("1e2", True), ("1E+2", True),
])
@pytest.mark.parametrize("storage", ["decimal", "string"])
def test_publication_preserves_only_exact_double_values(tmp_path, spelling, exact, storage):
    from decimal import Decimal

    from autoforecast.study_results import _forecast_batches

    value = Decimal(spelling) if storage == "decimal" else spelling
    pq.write_table(pa.Table.from_pylist([
        {"series_id": "A", "date": dt.date(2024, 1, 2), "yhat": value},
    ]), tmp_path / "part.parquet")
    if exact:
        rows = [row for batch in _forecast_batches(tmp_path) for row in batch.to_pylist()]
        assert rows[0]["yhat"] == Decimal(spelling)
    else:
        with pytest.raises(ValueError, match="without precision loss"):
            list(_forecast_batches(tmp_path))


@pytest.mark.parametrize("value,exact", [(2**53 + 2, True), (2**53 + 1, False),
                                         (-(2**53 + 2), True), (-(2**53 + 1), False)])
def test_publication_integer_precision_boundary(tmp_path, value, exact):
    from autoforecast.study_results import _forecast_batches

    pq.write_table(pa.Table.from_pylist([
        {"series_id": "A", "date": dt.date(2024, 1, 2), "yhat": value},
    ]), tmp_path / "part.parquet")
    if exact:
        assert next(_forecast_batches(tmp_path)).column("yhat")[0].as_py() == value
    else:
        with pytest.raises(ValueError, match="without precision loss"):
            list(_forecast_batches(tmp_path))


@pytest.mark.parametrize("destination", ["postgres"], indirect=True)
@pytest.mark.parametrize("bad_stage", ["tuning", "test"])
def test_publication_refuses_precision_loss_before_any_table_write(
    tmp_path, retained, destination, bad_stage,
):
    from decimal import Decimal

    registry, study_id, original, _ = retained
    writer, read, _ = destination
    origin = (original.testing["forecasts"].origin if bad_stage == "test" else
              _search_windows(original.tuning_path)[0][0][0]["forecasts"].origin)

    def model(request):
        output = predict(request)
        if request.origin == origin:
            part = pathlib.Path(output.forecasts.path) / "part.parquet"
            table = pq.read_table(part)
            table = table.set_column(table.schema.get_field_index("yhat"), "yhat", pa.array(
                [Decimal("0.10000000000000000001")] * table.num_rows,
                type=pa.decimal128(21, 20),
            ))
            pq.write_table(table, part)
        return output

    profile = config(tmp_path)
    settings = yaml.safe_load(profile.read_text())
    settings["selection"] = {"stage": "tuning", "origins": "all", "aggregation": "mean"}
    profile.write_text(yaml.safe_dump(settings))
    workflow = run_study(
        study_id, model, registry=registry, model_config=profile,
        implementation={**implementation(), "name": f"precision-{bad_stage}"},
        fixed_parameters={} if bad_stage == "test" else None,
        tracking="none", publish=False,
    )
    from autoforecast.study_results import _publication

    definition, study_sha = registry.definition(study_id)
    with pytest.raises(ValueError, match="without precision loss"):
        with _publication(definition, study_sha, workflow.path, scratch_root=tmp_path):
            pytest.fail("unsupported values reached the destination before refusal")
    with pytest.raises(ValueError, match="without precision loss"):
        registry.publish_results(study_id, workflow.path, destination=writer)
    assert read("forecasts") == [] and read("study_results") == []
    # Refusing publication must not alter the retained model output.
    forecast = (workflow.testing["forecasts"] if bad_stage == "test" else
                _search_windows(workflow.tuning_path)[0][0][0]["forecasts"])
    assert pq.read_table(forecast.path)["yhat"][0].as_py() == Decimal("0.10000000000000000001")


@pytest.mark.parametrize("dtype", [pa.float32(), pa.float64()])
def test_publication_keeps_floating_point_values(tmp_path, dtype):
    from autoforecast.study_results import _forecast_batches

    values = pa.array([-2.5, -1e-12, 0.0, -0.0, 1e-40], type=dtype)
    pq.write_table(pa.table({"series_id": ["A"] * len(values),
                             "date": [dt.date(2024, 1, 2)] * len(values),
                             "yhat": values}), tmp_path / "part.parquet")
    actual = [value for batch in _forecast_batches(tmp_path)
              for value in batch.column("yhat").to_pylist()]
    assert actual == values.to_pylist()
