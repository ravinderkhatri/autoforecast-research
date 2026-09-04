"""Retained input references and owned reads for native table snapshots.

A reference contains locations and immutable identities, never credentials. Table
mode relies on the database's sealed-data contract. Combined mode also verifies
canonical rows. Model outputs retain their existing, separate integrity checks.
"""

from __future__ import annotations

import base64
import contextvars
import dataclasses
import json
import pathlib
import re
import tempfile
import uuid
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager, nullcontext
from functools import wraps
from typing import Any, Protocol

import pyarrow as pa
import pyarrow.parquet as pq

from autoforecast import _logical_hash

MODES = frozenset({"hash", "table", "table+hash"})


@contextmanager
def _cleanup_on_exit(callback, *, label):
    """Run owned cleanup without replacing a failure from this operation."""
    failure = None
    try:
        yield
    except BaseException as error:
        failure = error
        raise
    finally:
        try:
            callback()
        except BaseException as error:
            if failure is None:
                raise
            failure.add_note(f"{label}: {type(error).__name__}")


def resolve_mode(explicit=None, configured=None, registry=None) -> str:
    """Use the first supplied setting, preserving an explicit hash override."""
    for value in (explicit, configured, registry, "hash"):
        if value is not None:
            if not isinstance(value, str) or value not in MODES:
                raise ValueError("snapshot_mode must be hash, table or table+hash")
            return value
    raise AssertionError("hash is the final default")


def _json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def arrow_schema(value: pa.Schema) -> str:
    """Retain exact typed fields without provider-specific type inference on reload."""
    return base64.b64encode(value.remove_metadata().serialize().to_pybytes()).decode("ascii")


def restore_schema(value: str) -> pa.Schema:
    return pa.ipc.read_schema(pa.BufferReader(base64.b64decode(value, validate=True)))


@dataclasses.dataclass(frozen=True)
class SnapshotReference:
    """An inert, validated native input identity safe to save in a study record."""

    _payload: bytes = dataclasses.field(repr=False)

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> SnapshotReference:
        required = {"schema", "mode", "backend", "connection", "capture_id", "object",
                    "incarnation", "arrow_schema", "rows", "retention"}
        if type(value) is not dict or set(value) != required | (
            {"logical_hash"} if value.get("mode") == "table+hash" else set()
        ):
            raise ValueError("native snapshot fields are missing or unexpected")
        if value["schema"] != "autoforecast.native-input.v1":
            raise ValueError("unsupported native snapshot format")
        if type(value["mode"]) is not str or value["mode"] not in {"table", "table+hash"}:
            raise ValueError("native snapshots require table or table+hash mode")
        if (type(value["backend"]) is not str
                or value["backend"] not in {
                    "delta", "bigquery", "postgresql", "mysql", "sqlserver",
                }):
            raise ValueError("unsupported snapshot backend")
        for name in ("connection", "capture_id"):
            if not isinstance(value[name], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value[name]):
                raise ValueError(f"invalid snapshot {name}")
        if type(value["object"]) is not str or not value["object"]:
            raise ValueError("snapshot object is required")
        if type(value["incarnation"]) is not dict or not value["incarnation"]:
            raise ValueError("snapshot incarnation is required")
        if type(value["rows"]) is not int or value["rows"] < 0:
            raise ValueError("snapshot rows must be a nonnegative integer")
        if value["retention"] != "until_released":
            raise ValueError("native snapshots must remain available until released")
        restore_schema(value["arrow_schema"])
        if value["mode"] == "table+hash":
            logical = value["logical_hash"]
            _logical_hash.validate_metadata(logical)
            if type(logical) is not dict or logical.get("rows") != value["rows"]:
                raise ValueError("snapshot logical identity differs from its row count")
        return cls(_json(value))

    def record(self) -> dict:
        """Return a fresh value; callers cannot edit the retained reference in place."""
        return json.loads(self._payload)


class NativeStore(Protocol):
    """Provider boundary. Implementations must enforce sealed ordinary reads."""

    backend: str

    @property
    def namespace(self) -> str: ...

    def preflight(self) -> None: ...
    def capture(self, path: pathlib.Path, *, capture_id: str, checkpoint=None) -> dict: ...
    def inspect(self, record: dict, *, retained: bool = False) -> None: ...
    def retain(self, record: dict) -> None: ...
    def drop_unretained(self, record: dict) -> None: ...
    def read(self, record: dict, destination: pathlib.Path) -> None: ...
    def release(self, record: dict) -> None: ...


class RetainedInputs:
    """Apply one resolved mode to native artifacts and use explicit connection aliases."""

    def __init__(self, *, mode: str, connection: str, store: NativeStore, scratch_root,
                 journal_root=None):
        self.mode = resolve_mode(mode)
        if self.mode == "hash":
            raise ValueError("hash mode uses the existing portable snapshot writer")
        if not isinstance(connection, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", connection):
            raise ValueError("connection must be an explicit alias")
        self.connection, self.store = connection, store
        self.scratch_root = pathlib.Path(scratch_root)
        self.journal_root = pathlib.Path(journal_root or self.scratch_root / "capture-journal")
        self._locks = ExitStack()
        self._closed = False

    def close(self):
        """End capture ownership after publication or failure cleanup has finished."""
        if not self._closed:
            self._closed = True
            self._locks.close()

    def __enter__(self):
        return self

    def __exit__(self, _kind, failure, _traceback):
        try:
            self.close()
        except BaseException as error:
            if failure is None:
                raise
            failure.add_note(f"Capture owner cleanup needs recovery: {type(error).__name__}")

    def capture(self, path: pathlib.Path, *, spark=None) -> SnapshotReference:
        """Retain already captured typed data once; hash only in combined mode."""
        if self._closed:
            raise ValueError("native capture manager is closed")
        self.store.preflight()
        logical = (_logical_hash.fingerprint(path, spark=spark)
                   if self.mode == "table+hash" else None)
        from mlops._safe_files import create_directory, publish_bytes, read_path_bytes

        capture_id = uuid.uuid4().hex
        namespace = self.store.namespace
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("native destination must expose its configured namespace")
        directory = self.journal_root / capture_id
        intent = {"schema": "autoforecast.capture-intent.v1", "capture_id": capture_id,
                  "connection": self.connection, "backend": self.store.backend,
                  "namespace": namespace, "mode": self.mode}
        # Publish ownership before the provider can create anything. A crash may
        # leave an unresolved intent, but can never make the owned ID disappear.
        from autoforecast.study_registration import _publication_lock

        create_directory(directory)
        self._locks.enter_context(_publication_lock(directory, blocking=False))
        publish_bytes(directory, "intent.json", _json(intent))
        reference = None

        def checkpoint(record):
            nonlocal reference
            value: dict[str, Any] = dict(
                record, schema="autoforecast.native-input.v1", mode=self.mode,
                connection=self.connection, backend=self.store.backend,
                retention="until_released",
            )
            if logical is not None:
                value["logical_hash"] = logical
            reference = SnapshotReference.from_record(value)
            if value["capture_id"] != capture_id:
                raise ValueError("provider returned a different capture ID")
            # PostgreSQL calls this inside its creating transaction, before commit.
            # A lost commit acknowledgement therefore leaves the original identity.
            publish_bytes(directory, "sealed.json", reference._payload)

        record = self.store.capture(path, capture_id=capture_id, checkpoint=checkpoint)
        if reference is None:
            raise ValueError("native provider did not record its sealed capture identity")
        try:
            expected = reference.record()
            if any(expected.get(key) != value for key, value in record.items()):
                raise ValueError("provider changed its capture after checkpointing")
            if read_path_bytes(directory / "sealed.json") != reference._payload:
                raise ValueError("native capture journal changed")
            self.store.inspect(expected)
            if logical is not None:
                with self.open(reference, spark=spark):
                    pass
            publish_bytes(directory, "ready.json", reference._payload)
            return reference
        except BaseException:
            # Keep the operation failure primary. If cleanup fails, its durable
            # identity remains available for a separately authorized recovery.
            import sys

            original = sys.exception()
            try:
                self.store.drop_unretained(reference.record())
                publish_bytes(directory, "removed.json", reference._payload)
            except Exception as error:
                if original is not None:
                    original.add_note(f"Capture cleanup needs recovery: {type(error).__name__}")
            raise

    @contextmanager
    def open(self, reference: SnapshotReference, *, spark=None) -> Iterator[pathlib.Path]:
        """Own a bounded transport copy for the reader lifetime, then remove it."""
        record = SnapshotReference.from_record(reference.record()).record()
        if (record["connection"] != self.connection or record["backend"] != self.store.backend
                or record["mode"] != self.mode):
            raise ValueError("native snapshot does not belong to this resolver")
        self.store.inspect(record)
        resources = ExitStack()
        with _cleanup_on_exit(resources.close, label="Final native input verification failed"):
            root = resources.enter_context(tempfile.TemporaryDirectory(
                dir=self.scratch_root, prefix="native-input-",
            ))
            path = pathlib.Path(root) / "rows"
            self.store.read(record, path)
            files = sorted(path.rglob("*.parquet"))
            if not files:
                raise ValueError("native input transport produced no typed data")
            rows = 0
            expected = restore_schema(record["arrow_schema"])
            for file in files:
                with pq.ParquetFile(file) as reader:
                    if not reader.schema_arrow.equals(expected, check_metadata=False):
                        raise ValueError("native input schema changed during transport")
                    rows += reader.metadata.num_rows
            if rows != record["rows"]:
                raise ValueError("native input row count changed during transport")
            if self.mode == "table+hash":
                if _logical_hash.fingerprint(path, spark=spark) != record["logical_hash"]:
                    raise ValueError("native input content changed")
            self.store.inspect(record)
            resources.callback(self.store.inspect, record)
            yield path


def bind_source_queries(queries, tables, *, dialect, sources):
    """Use the existing read-only SQL checker with explicit captured relations."""
    from autoforecast.integrations.delta import _bound_source_sql

    if (not isinstance(queries, Mapping) or not queries
            or not isinstance(tables, Mapping) or not tables):
        raise ValueError("source queries and declared tables are required")
    aliases = {}
    replacements = {}
    for index, alias in enumerate(tables):
        if (not isinstance(alias, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", alias)
                or alias.casefold() in aliases):
            raise ValueError("source aliases must be distinct simple names")
        slot = "source_" + str(index)
        aliases[alias.casefold()] = slot
        replacements[slot] = sources[alias]
    result = {}
    for name, query in queries.items():
        if not isinstance(query, str) or not query.strip():
            raise ValueError("source queries must contain SQL")
        result[name] = _bound_source_sql(query, aliases, dialect=dialect).format_map(replacements)
    return result


class Connections:
    """Resolve explicit aliases using supplied clients or standard credential providers."""

    def __init__(self, settings=None, supplied=None, *, spark=None):
        self.settings = dict(settings or {})
        self.supplied = dict(supplied or {})
        self.spark = spark
        self._clients = {}
        self._owned = []
        for alias in self.supplied:
            if not isinstance(alias, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", alias):
                raise ValueError("connection aliases must be simple names")
        for alias, value in self.settings.items():
            if not isinstance(alias, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", alias):
                raise ValueError("connection aliases must be simple names")
            if not isinstance(value, dict) or value.get("kind") not in {
                "bigquery", "postgresql", "mysql", "sqlserver", "delta",
            }:
                raise ValueError("connections require an explicit supported kind")
            allowed = {"kind", "project", "location"} if value["kind"] == "bigquery" else (
                {"kind"} if value["kind"] == "delta" else {
                    "kind", "dsn_env", "writer_dsn_env", "release_dsn_env",
                }
            )
            if set(value) - allowed:
                raise ValueError("connection settings contain unknown fields; keep secrets in the "
                                 "credential provider or a named environment variable")
            for key, item in value.items():
                if not isinstance(item, str) or not item.strip():
                    raise ValueError("connection settings must be nonempty strings")
                if key.endswith("_env") and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", item):
                    raise ValueError("DSN settings name environment variables, not credentials")

    def kind(self, alias):
        declared = self.settings.get(alias, {}).get("kind")
        client = self.supplied.get(alias)
        detected = getattr(client, "backend", None)
        supported = {"bigquery", "postgresql", "mysql", "sqlserver", "delta"}
        if detected not in supported:
            detected = None
            # Inspect the hierarchy so a normal user subclass keeps its driver identity.
            for cls in type(client).__mro__:
                for prefix, backend in (("google.cloud.bigquery", "bigquery"),
                                        ("psycopg", "postgresql"), ("pymysql", "mysql"),
                                        ("mssql_python", "sqlserver"),
                                        ("pyspark.sql", "delta")):
                    if cls.__module__.split(".")[:len(prefix.split("."))] == prefix.split("."):
                        detected = backend
                        break
                if detected:
                    break
            # ODBC supports many databases. Its module name alone is not a dialect.
            if client is not None and any(
                cls.__module__ == "pyodbc" for cls in type(client).__mro__
            ):
                import pyodbc

                if client.getinfo(pyodbc.SQL_DBMS_NAME) == "Microsoft SQL Server":
                    detected = "sqlserver"
                else:
                    raise ValueError("ODBC connection is not a supported SQL Server database")
        if declared and detected and declared != detected:
            raise ValueError("configured connection kind conflicts with its supplied client")
        if declared or detected:
            return declared or detected
        if client is None:
            raise ValueError(f"missing connection alias {alias!r}; supply connections= or config")
        raise ValueError("connection kind is ambiguous; declare it explicitly")

    def client(self, alias, *, kind):
        import os

        if self.kind(alias) != kind:
            raise ValueError("source kind conflicts with its connection")
        if alias in self.supplied:
            return self.supplied[alias]
        if alias not in self._clients:
            settings = self.settings[alias]
            if kind == "bigquery":
                from google.cloud import bigquery

                if not settings.get("project") or not settings.get("location"):
                    raise ValueError("BigQuery connections require project and location")
                client = bigquery.Client(project=settings["project"], location=settings["location"])
            elif kind == "postgresql":
                import psycopg

                variable = settings.get("dsn_env")
                if not variable or not os.environ.get(variable):
                    raise ValueError(f"connection {alias!r} needs its DSN environment variable")
                client = psycopg.connect(os.environ[variable], autocommit=True)
            elif kind == "mysql":
                from autoforecast.integrations.mysql import connect

                variable = settings.get("dsn_env")
                if not variable or not os.environ.get(variable):
                    raise ValueError(f"connection {alias!r} needs its DSN environment variable")
                client = connect(os.environ[variable])
            elif kind == "sqlserver":
                from autoforecast.integrations.sqlserver import connect

                variable = settings.get("dsn_env")
                if not variable or not os.environ.get(variable):
                    raise ValueError(f"connection {alias!r} needs its DSN environment variable")
                client = connect(os.environ[variable])
            elif kind == "delta":
                if self.spark is None:
                    raise ValueError("Delta connections need an explicit Spark session")
                return self.spark
            else:
                raise ValueError(f"{kind} connection transport is not yet available")
            self._clients[alias] = client
            self._owned.append(client)
        return self._clients[alias]

    def snapshot_store(self, destination, *, release=False) -> NativeStore:
        """Resolve a separately authorized write destination without guessing it."""
        if (not isinstance(destination, dict)
                or set(destination) != {"connection", "namespace"}
                or any(not isinstance(v, str) or not v.strip() for v in destination.values())):
            raise ValueError("snapshots require an explicit connection and namespace")
        alias, namespace = destination["connection"], destination["namespace"]
        backend = self.kind(alias)
        if backend == "bigquery":
            raise ValueError(
                "BigQuery retained study inputs are not available: restricted writer, "
                "reader and release permissions have not passed the live protection gate"
            )
        supplied = self.supplied.get(alias)
        if supplied is not None and getattr(supplied, "backend", None) == backend:
            actual = supplied.namespace
            if actual != namespace:
                raise ValueError("snapshot destination conflicts with its supplied store")
            if release and backend in {"postgresql", "mysql", "sqlserver"}:
                supplied.release_preflight()
            else:
                supplied.preflight()
            return supplied
        if backend in {"postgresql", "mysql", "sqlserver"}:
            import os

            credential = "release_dsn_env" if release else "writer_dsn_env"
            variable = self.settings.get(alias, {}).get(credential)
            if not variable or not os.environ.get(variable):
                raise ValueError(f"{backend} snapshots need {credential} and a restricted reader")
            key = (alias, "release" if release else "writer")
            if key not in self._clients:
                if backend == "postgresql":
                    import psycopg

                    writer = psycopg.connect(os.environ[variable], autocommit=True)
                elif backend == "mysql":
                    from autoforecast.integrations.mysql import connect

                    writer = connect(os.environ[variable])
                else:
                    from autoforecast.integrations.sqlserver import connect

                    writer = connect(os.environ[variable])
                self._clients[key] = writer
                self._owned.append(writer)
            from autoforecast.integrations.mysql import MySQLSnapshots
            from autoforecast.integrations.postgres import PostgresSnapshots
            from autoforecast.integrations.sqlserver import SQLServerSnapshots

            if backend == "postgresql":
                store = PostgresSnapshots(None if release else self._clients[key],
                                          self.client(alias, kind=backend), schema=namespace,
                                          releaser=self._clients[key] if release else None)
            elif backend == "mysql":
                store = MySQLSnapshots(None if release else self._clients[key],
                                      self.client(alias, kind=backend), database=namespace,
                                      releaser=self._clients[key] if release else None)
            else:
                store = SQLServerSnapshots(None if release else self._clients[key],
                                           self.client(alias, kind=backend), schema=namespace,
                                           releaser=self._clients[key] if release else None)
            if release:
                store.release_preflight()
                return store
        else:
            raise ValueError(f"native {backend} snapshot adapter is not configured")
        store.preflight()
        return store

    def location(self, alias):
        value = self.settings.get(alias, {}).get("location")
        if value is None:
            value = getattr(self.supplied.get(alias), "location", None)
        if not value:
            raise ValueError(f"connection {alias!r} requires an explicit location")
        return value

    def close(self):
        owned, self._owned = self._owned, []
        self._clients.clear()
        with ExitStack() as stack:
            for client in owned:
                stack.callback(client.close)


class NativeReadScope:
    """Own native-to-file transports for one study or worker lifetime.

    The saved reference remains the authority. Cached files are private transport
    copies and are never published as another permanent dataset.
    """

    def __init__(self, connections: Connections, *, scratch_root, spark=None):
        self.connections = connections
        self.scratch_root = pathlib.Path(scratch_root)
        self.spark = spark
        self._stack = ExitStack()
        self._opened = {}
        self._managers = {}
        self._leases = {}
        self._closed = False

    @classmethod
    def from_environment(cls, *, scratch_root=None, spark=None):
        import os

        from autoforecast._exact_yaml import read_record

        existing = _ACTIVE_INPUTS.get()
        if existing is not None:
            return cls(existing.connections, scratch_root=scratch_root or existing.scratch_root,
                       spark=spark if spark is not None else existing.spark)
        settings = {}
        path = os.environ.get("AUTOFORECAST_CONFIG")
        if path:
            configuration = read_record(pathlib.Path(path))
            if not isinstance(configuration, dict):
                raise ValueError("worker configuration requires a mapping")
            settings = configuration.get("connections", {})
        connections = Connections(settings, spark=spark)
        scope = cls(connections, scratch_root=scratch_root or tempfile.gettempdir(), spark=spark)
        scope._stack.callback(connections.close)
        return scope

    def _manager(self, reference):
        record = reference.record()
        alias, backend = record["connection"], record["backend"]
        if backend == "bigquery":
            raise ValueError(
                "BigQuery retained study inputs are not available: restricted writer, "
                "reader and release permissions have not passed the live protection gate"
            )
        key = (alias, backend, record["mode"], record["object"].rsplit(".", 1)[0])
        if key not in self._managers:
            supplied = self.connections.supplied.get(alias)
            if supplied is not None and getattr(supplied, "backend", None) == backend:
                store = supplied
            elif backend == "postgresql":
                from autoforecast.integrations.postgres import PostgresSnapshots

                store = PostgresSnapshots(None, self.connections.client(alias, kind=backend),
                                          schema=key[3])
            elif backend == "mysql":
                from autoforecast.integrations.mysql import MySQLSnapshots

                store = MySQLSnapshots(None, self.connections.client(alias, kind=backend),
                                       database=key[3])
            elif backend == "sqlserver":
                from autoforecast.integrations.sqlserver import SQLServerSnapshots

                store = SQLServerSnapshots(None, self.connections.client(alias, kind=backend),
                                           schema=key[3])
            else:
                raise ValueError(f"native {backend} reads require a configured snapshot store")
            self._managers[key] = RetainedInputs(mode=record["mode"], connection=alias, store=store,
                                                scratch_root=self.scratch_root)
        return self._managers[key]

    def hold_study(self, manifest, sha256):
        """Register this reader before opening any published input objects."""
        if self._closed:
            raise ValueError("native input scope is closed")
        manifest = pathlib.Path(manifest).absolute()
        key = (str(manifest), sha256)
        if key not in self._leases:
            lease = StudyLease(manifest, sha256)
            self._leases[key] = lease
            self._stack.callback(lease.close)

    @staticmethod
    def _files(path):
        """Check private transport ownership without claiming a content hash."""
        result = []
        for file in sorted(path.rglob("*")):
            if file.is_symlink():
                raise ValueError("native transport contains a symbolic link")
            if file.is_file():
                stat = file.stat()
                result.append((str(file.relative_to(path)), stat.st_dev, stat.st_ino,
                               stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_mode))
        return tuple(result)

    def materialize(self, value):
        if self._closed:
            raise ValueError("native input scope is closed")
        reference = SnapshotReference.from_record(value)
        manager = self._manager(reference)
        key = reference._payload
        if key not in self._opened:
            path = self._stack.enter_context(manager.open(reference, spark=self.spark))
            for file in path.rglob("*.parquet"):
                file.chmod(0o400)
            self._opened[key] = path, self._files(path)
        path, original = self._opened[key]
        if self._files(path) != original:
            raise ValueError("owned native input transport changed")
        manager.store.inspect(reference.record())
        return path

    def inspect(self, value, *, retained=False):
        reference = SnapshotReference.from_record(value)
        self._manager(reference).store.inspect(reference.record(), retained=retained)
        if reference._payload in self._opened or reference.record()["mode"] == "table+hash":
            self.materialize(value)

    def capture_temporary(self, path, *, template, journal_root):
        """Seal callback inputs for this scope, releasing them after owned reads close."""
        original = SnapshotReference.from_record(template).record()
        store = self.connections.snapshot_store({
            "connection": original["connection"],
            "namespace": original["object"].rsplit(".", 1)[0],
        })
        if store.backend != original["backend"]:
            raise ValueError("native temporary input backend differs from its template")
        store.inspect(original)
        manager = RetainedInputs(mode=original["mode"], connection=original["connection"],
                                 store=store, scratch_root=self.scratch_root,
                                 journal_root=journal_root)
        self._stack.callback(manager.close)
        reference = manager.capture(path, spark=self.spark).record()
        self._stack.callback(store.drop_unretained, reference)
        return reference

    def close(self):
        if not self._closed:
            self._closed = True
            self._stack.close()

    @contextmanager
    def active(self):
        token = _ACTIVE_INPUTS.set(self)
        try:
            yield self
        finally:
            _ACTIVE_INPUTS.reset(token)


# Contexts propagate only within the current execution. Workers resolve their own
# clients from inert aliases and deployment configuration, never a serialized client.
_ACTIVE_INPUTS: contextvars.ContextVar[NativeReadScope | None] = contextvars.ContextVar(
    "autoforecast_native_inputs", default=None,
)


@contextmanager
def input_scope(scope=None, *, connections=None, scratch_root=None, spark=None):
    existing = _ACTIVE_INPUTS.get()
    if scope is not None or existing is not None:
        selected = scope if scope is not None else existing
        assert selected is not None
        with selected.active():
            yield selected
        return
    if connections is not None and not isinstance(connections, Connections):
        connections = Connections(supplied=connections, spark=spark)
    selected = (NativeReadScope.from_environment(scratch_root=scratch_root, spark=spark)
                if connections is None else NativeReadScope(
                    connections, scratch_root=scratch_root or tempfile.gettempdir(), spark=spark,
                ))
    with _cleanup_on_exit(selected.close, label="Native input cleanup needs recovery"):
        with selected.active():
            yield selected


def study_inputs(function):
    """Use the same retained inputs through method and module-level study entry points."""
    @wraps(function)
    def call(study, *args, **kwargs):
        scope = getattr(study, "_native_inputs", None)
        with scope.active() if scope is not None else nullcontext():
            return function(study, *args, **kwargs)
    return call



def worker_inputs(function):
    """Resolve retained inputs locally when a worker or saved-result reader starts."""
    @wraps(function)
    def call(*args, **kwargs):
        with input_scope():
            return function(*args, **kwargs)
    return call


def current_inputs():
    selected = _ACTIVE_INPUTS.get()
    if selected is None:
        raise ValueError("native input requires an active study or worker read scope")
    return selected


class StudyLease:
    """Durable reader ownership; a crashed process leaves an explicit recovery record."""

    def __init__(self, manifest, sha256):
        from mlops._safe_files import create_directory, path_exists, publish_bytes

        from autoforecast.study_registration import _publication_lock

        manifest = pathlib.Path(manifest).absolute()
        self.study_id = manifest.parent.name
        if (not re.fullmatch(r"[a-f0-9]{64}", self.study_id)
                or not re.fullmatch(r"[a-f0-9]{64}", sha256)):
            raise ValueError("reader ownership requires a published study identity")
        self.root = manifest.parent.parent.parent
        self.directory = self.root / "readers" / self.study_id
        self.identifier = uuid.uuid4().hex
        self.payload = _json({"schema": "autoforecast.study-reader.v1", "study_id": self.study_id,
                              "lease_id": self.identifier, "manifest_sha256": sha256})
        self.closed = False
        with _publication_lock(self.root):
            if path_exists(self.root / "released" / (self.study_id + ".json")):
                raise ValueError("study has been released or is being released")
            lock_root = self.root / "reader-locks" / self.study_id / self.identifier
            create_directory(lock_root)
            self._ownership = ExitStack()
            self._ownership.enter_context(_publication_lock(lock_root, blocking=False))
            try:
                relative = f"readers/{self.study_id}/{self.identifier}.json"
                publish_bytes(self.root, relative, self.payload)
            except BaseException:
                self._ownership.close()
                raise

    def close(self):
        from mlops._safe_files import path_exists, publish_bytes, read_path_bytes

        from autoforecast.study_registration import _publication_lock

        if self.closed:
            return
        with _publication_lock(self.root):
            if read_path_bytes(self.directory / (self.identifier + ".json")) != self.payload:
                raise ValueError("study reader ownership record changed")
            closed = self.directory / (self.identifier + ".closed.json")
            if path_exists(closed):
                if read_path_bytes(closed) != self.payload:
                    raise ValueError("study reader completion record changed")
            else:
                publish_bytes(self.directory, closed.name, self.payload)
            self.closed = True
            self._ownership.close()

    @staticmethod
    def recover(root, study_id, lease_id):
        """Close an abandoned reader only when its original owner lock is available."""
        from mlops._safe_files import path_exists, publish_bytes, read_path_bytes

        from autoforecast.study_registration import _publication_lock

        if (not isinstance(study_id, str) or not re.fullmatch(r"[a-f0-9]{64}", study_id)
                or not isinstance(lease_id, str) or not re.fullmatch(r"[a-f0-9]{32}", lease_id)):
            raise ValueError("reader recovery requires exact study and lease IDs")
        root = pathlib.Path(root)
        directory = root / "readers" / study_id
        record = directory / (lease_id + ".json")
        if not path_exists(record):
            raise FileNotFoundError("reader ownership record is missing")
        with _publication_lock(root):
            active = StudyLease.active(root, study_id)
            if lease_id not in active:
                return lease_id
            lock_root = root / "reader-locks" / study_id / lease_id
            if not path_exists(lock_root):
                raise ValueError("reader has no recoverable owner lock")
            with _publication_lock(lock_root, blocking=False):
                # A local process exit releases flock. On volumes an abandoned
                # lock stays closed until an operator confirms its owner stopped.
                publish_bytes(directory, lease_id + ".closed.json", read_path_bytes(record))
        return lease_id

    @staticmethod
    def active(root, study_id):
        """Call under the registry lock; malformed ownership records fail closed."""
        from mlops._safe_files import path_exists, read_path_bytes, snapshot_tree_types

        directory = pathlib.Path(root) / "readers" / study_id
        if not path_exists(directory):
            return []
        active = []
        for path, kind in snapshot_tree_types(directory).items():
            if kind == "directory":
                continue
            if kind != "file":
                raise ValueError("reader ownership contains a link or special file")
            if path.name.endswith(".closed.json"):
                original = path.with_name(path.name.removesuffix(".closed.json") + ".json")
                if not path_exists(original):
                    raise ValueError("reader completion has no ownership record")
                continue
            raw = read_path_bytes(path)
            record = json.loads(raw)
            if (not isinstance(record, dict)
                    or set(record) != {"schema", "study_id", "lease_id", "manifest_sha256"}
                    or record["schema"] != "autoforecast.study-reader.v1"
                    or record["study_id"] != study_id
                    or not isinstance(record["lease_id"], str)
                    or not re.fullmatch(r"[a-f0-9]{32}", record["lease_id"])
                    or not isinstance(record["manifest_sha256"], str)
                    or not re.fullmatch(r"[a-f0-9]{64}", record["manifest_sha256"])
                    or path.name != record["lease_id"] + ".json"):
                raise ValueError("invalid study reader ownership record")
            closed = path.with_name(path.stem + ".closed.json")
            if not path_exists(closed):
                active.append(record["lease_id"])
            elif read_path_bytes(closed) != raw:
                raise ValueError("study reader completion record changed")
        return active


class ColumnLayout:
    """Store nested structs as typed columns, preserving null structs explicitly.

    Scalar-only data keeps its column names. Struct fields use deterministic column
    positions plus presence flags, so a null struct differs from a struct of nulls.
    No dataset rows are encoded as JSON.
    """

    def __init__(self, schema):
        self.original = schema
        self.entries = []
        nested = any(pa.types.is_struct(field.type) for field in schema)

        def visit(field, path):
            present = pa.types.is_struct(field.type)
            name = f"afc_{len(self.entries)}" if nested else field.name
            self.entries.append((name, path, present, pa.bool_() if present else field.type))
            if present:
                for child in field.type:
                    visit(child, (*path, child.name))

        for field in schema:
            visit(field, (field.name,))
        self.schema = (pa.schema([pa.field(name, dtype) for name, _, _, dtype in self.entries])
                       if nested else schema)

    def encode(self, row):
        values = []
        for _name, path, present, _dtype in self.entries:
            value = row
            for name in path:
                value = value[name] if value is not None else None
            values.append(value is not None if present else value)
        return values

    def decode(self, values):
        columns = iter(values)

        def read(field):
            value = next(columns)
            if pa.types.is_struct(field.type):
                children = {child.name: read(child) for child in field.type}
                if type(value) is not bool:
                    raise ValueError("native struct presence flag is invalid")
                return children if value else None
            return value

        row = {field.name: read(field) for field in self.original}
        if next(columns, self) is not self:
            raise ValueError("native column count differs from the recorded schema")
        return row


    @contextmanager
    def transport(self, files):
        """Flatten structs in bounded temporary Parquet batches only when required."""
        if self.schema.equals(self.original, check_metadata=False):
            yield files
            return
        with tempfile.TemporaryDirectory(prefix="native-columns-") as directory:
            converted = []
            for index, file in enumerate(files):
                destination = pathlib.Path(directory) / f"part-{index}.parquet"
                with pq.ParquetFile(file) as source, pq.ParquetWriter(
                    destination, self.schema,
                ) as writer:
                    for batch in source.iter_batches(batch_size=8192):
                        writer.write_table(pa.Table.from_pylist([
                            dict(zip(self.schema.names, self.encode(row), strict=True))
                            for row in batch.to_pylist()
                        ], schema=self.schema))
                converted.append(destination)
            yield converted


_ACTIVE_CAPTURE: contextvars.ContextVar[CaptureAttempt | None] = contextvars.ContextVar(
    "autoforecast_native_capture", default=None,
)


class CaptureAttempt:
    """Own a registration's native captures until publication chooses retained inputs."""

    def __init__(self, manager, *, spark, connections=None):
        self.manager = manager
        self.spark = spark
        self.scope = NativeReadScope(
            connections or Connections(supplied={manager.connection: manager.store}),
            scratch_root=manager.scratch_root, spark=spark,
        )
        self._paths = {}
        self._references = {}
        self._retained = set()
        self._shared = set()
        self._evidence = {}
        self._published = False

    def bind(self, path):
        path = pathlib.Path(path).absolute()
        if path not in self._paths:
            reference = self.manager.capture(path, spark=self.spark).record()
            self._paths[path] = reference
            self.adopt(reference)
        return {"snapshot": json.loads(_json(self._paths[path]))}

    def reuse(self, path, reference):
        """Read a shared retained capture without claiming cleanup ownership of it."""
        checked = SnapshotReference.from_record(reference).record()
        if checked["mode"] != self.manager.mode:
            raise ValueError("source capture mode differs from this registration")
        path = pathlib.Path(path).absolute()
        if path in self._paths:
            raise ValueError("managed source path is already bound")
        self.scope.inspect(checked, retained=True)
        self._paths[path] = checked
        self._shared.add(checked["capture_id"])

    def path(self, path):
        return self.scope.materialize(self.bind(path)["snapshot"])

    def adopt(self, reference):
        checked = SnapshotReference.from_record(reference).record()
        self._references[checked["capture_id"]] = checked

    def retain(self, reference):
        if reference["capture_id"] in self._shared:
            self.scope.inspect(reference, retained=True)
            return
        self.adopt(reference)
        # Keep the reference even if an acknowledgement is lost after the database
        # has retained it. Recovery may release the orphan with separate authority.
        self._retained.add(reference["capture_id"])
        self.manager.store.retain(reference)

    def retained_snapshots(self):
        """Return the complete owned inventory for a content-addressed publication."""
        return {capture_id: SnapshotReference.from_record(self._references[capture_id]).record()
                for capture_id in sorted(self._retained)}

    def retain_evidence(self, root, relative_path, reference):
        """Publish a small reference document instead of another row-bearing file copy."""
        from mlops._safe_files import publish_bytes

        self.retain(reference)
        payload = _json({"schema": "autoforecast.native-findings.v1", "snapshot": reference})
        path = publish_bytes(pathlib.Path(root), relative_path, payload)
        self._evidence[path] = payload

    def evidence(self, root):
        """Bind the original diagnostic documents into the registered study."""
        import hashlib

        from mlops._safe_files import read_path_bytes

        result = []
        for path, payload in sorted(self._evidence.items()):
            if read_path_bytes(path) != payload:
                raise ValueError("native findings reference changed before publication")
            reference = json.loads(payload)["snapshot"]
            self.manager.store.inspect(reference, retained=True)
            result.append({"path": str(path.relative_to(root)),
                           "sha256": hashlib.sha256(payload).hexdigest(), "snapshot": reference})
        return result

    def published(self):
        """Call only after the registry has durably published the retained references."""
        self._published = True

    @contextmanager
    def active(self):
        token = _ACTIVE_CAPTURE.set(self)
        failure = None
        try:
            with self.scope.active():
                yield self
        except BaseException as error:
            failure = error
            raise
        finally:
            _ACTIVE_CAPTURE.reset(token)
            try:
                with ExitStack() as cleanup:
                    cleanup.callback(self.manager.close)
                    # Only unpublished captures can be dropped by registration.
                    # Retained orphans require separately authorized recovery.
                    for capture_id, reference in self._references.items():
                        if capture_id not in self._retained:
                            cleanup.callback(self.manager.store.drop_unretained, reference)
                    self.scope.close()
            except BaseException as error:
                if failure is None:
                    raise
                failure.add_note(f"Capture cleanup needs recovery: {type(error).__name__}")


def current_capture():
    return _ACTIVE_CAPTURE.get()


def managed_path(path):
    """Route operation-owned registration reads through its resolved storage policy."""
    capture = current_capture()
    return capture.path(path) if capture is not None else pathlib.Path(path)
