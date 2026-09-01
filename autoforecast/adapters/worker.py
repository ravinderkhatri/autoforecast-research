"""Command-line worker used by :mod:`autoforecast.adapters.isolated`.

It intentionally imports no concrete adapter until after the request identifies the one module
needed for this process.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys

from pyspark.sql import SparkSession

from autoforecast import registry, resources
from autoforecast._exact_json import ExactJSONError, load_exact_json
from autoforecast.acceptance_retirement import canonical_json_sha256
from autoforecast.adapters import _ISOLATED_MODULE, _validate_span, register_defaults
from autoforecast.executable_source import validate_executable_source_snapshot
from autoforecast.panel import Panel
from autoforecast.tracking_plan import enable_autologging, tracking_plan

_REQUEST_SCHEMA = "autoforecast.isolated-request.v1"
_RESPONSE_SCHEMA = "autoforecast.isolated-response.v1"
_ARROW_ONLY_ADAPTERS = frozenset(
    {
        "sft",
        "hmfnet",
        "pact",
        "tft_q",
        "tft_hnb",
        "timesfm_hnb",
        "timesfm_v8",
    }
)


def main(request_name: str, response_name: str) -> int:
    request_path = pathlib.Path(request_name)
    try:
        request_envelope = load_exact_json(request_path)
    except ExactJSONError as exc:
        raise ValueError(
            f"isolated request {request_path} is unreadable or ambiguous: {exc}") from exc
    if (
        type(request_envelope) is not dict
        or set(request_envelope) != {"schema", "request_sha256", "request"}
        or request_envelope.get("schema") != _REQUEST_SCHEMA
        or type(request_envelope.get("request_sha256")) is not str
        or len(request_envelope["request_sha256"]) != 64
        or canonical_json_sha256(request_envelope.get("request"))
        != request_envelope["request_sha256"]
    ):
        raise ValueError("isolated request has no exact content-hash envelope")
    request_sha256 = request_envelope["request_sha256"]
    request = request_envelope["request"]
    expected_fields = {
        "adapter", "panel", "origin", "horizon", "kwargs", "worker_profile",
        "worker_runtime", "source_snapshot",
    }
    optional_tracking = {"tracking_context"}
    if type(request) is not dict or set(request) not in (
        expected_fields, expected_fields | optional_tracking,
    ):
        raise ValueError(
            f"isolated request must be one object with fields {sorted(expected_fields)}")
    source = validate_executable_source_snapshot(request["source_snapshot"])
    source_root_value = source["root"]
    if not isinstance(source_root_value, str):  # validated by executable-source contract
        raise ValueError("isolated source snapshot root is invalid")
    source_root = pathlib.Path(source_root_value).resolve()
    if (
        pathlib.Path.cwd().resolve() != source_root
        or pathlib.Path(__file__).resolve()
        != source_root / "autoforecast" / "adapters" / "worker.py"
    ):
        raise ValueError("isolated worker did not import from the authenticated source snapshot")
    adapter = request["adapter"]
    if type(adapter) is not str or not adapter.strip():
        raise ValueError("isolated request adapter must be a non-blank string")
    try:
        module = _ISOLATED_MODULE[adapter]
    except KeyError:
        raise ValueError(f"unknown isolated adapter {adapter!r}") from None

    if type(request["panel"]) is not dict:
        raise ValueError("isolated request panel must be one JSON object")
    raw_panel = dict(request["panel"])
    panel = Panel.from_record(raw_panel)

    if type(request["kwargs"]) is not dict:
        raise ValueError("isolated request kwargs must be one JSON object")
    origin = request["origin"]
    horizon = request["horizon"]
    if type(horizon) is not int or horizon < 1:
        raise ValueError("isolated request horizon must be a positive integer")
    _validate_span(panel, origin, horizon, adapter=adapter)
    kwargs = dict(request["kwargs"])
    tracking_context = request.get("tracking_context")
    if tracking_context is not None:
        required_tracking = {"tracking_uri_sha256", "run_id", "experiment_id"}
        if (type(tracking_context) is not dict or set(tracking_context) != required_tracking
                or not all(
                    isinstance(value, str) and value for value in tracking_context.values()
                )):
            raise ValueError("isolated MLflow tracking context is invalid")
        tracking_uri = os.environ.pop("AUTOFORECAST_WORKER_MLFLOW_TRACKING_URI", None)
        if (
            not isinstance(tracking_uri, str)
            or hashlib.sha256(tracking_uri.encode()).hexdigest()
            != tracking_context["tracking_uri_sha256"]
        ):
            raise ValueError("isolated MLflow tracking URI does not match its retained hash")
        tracking_context = {
            "tracking_uri": tracking_uri,
            "run_id": tracking_context["run_id"],
            "experiment_id": tracking_context["experiment_id"],
        }
    if adapter == "mft" and "resolved_build" in kwargs:
        from spec.resolved import ResolvedBuild

        kwargs["resolved"] = ResolvedBuild.from_record(kwargs.pop("resolved_build"))
    # These are tuple-valued public options; JSON makes them lists.
    for field in ("events", "cycles"):
        if field in kwargs:
            try:
                kwargs[field] = tuple(tuple(value) for value in kwargs[field])
            except TypeError as exc:
                raise ValueError(
                    f"isolated request {field} must be an array of arrays") from exc

    if type(request["worker_profile"]) is not dict:
        raise ValueError("isolated request worker_profile must be one JSON object")
    try:
        profile = resources.SparkProfile(**request["worker_profile"])
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "isolated request has no valid explicit worker profile; deriving one from the whole "
            "machine would double-count the live parent Spark session") from exc
    if profile.cores < 1 or profile.shuffle_partitions < 1 \
            or profile.series_per_partition < 1 or profile.python_worker_budget_gb <= 0:
        raise ValueError(f"invalid isolated worker profile: {profile}")
    runtime = request["worker_runtime"]
    basic_runtime_fields = {"python", "prefix"}
    if (type(runtime) is not dict
            or set(runtime) != basic_runtime_fields
            or type(runtime.get("python")) is not str
            or type(runtime.get("prefix")) is not str):
        raise ValueError(
            "isolated request worker_runtime must contain exact python and prefix strings")
    requested_python = pathlib.Path(runtime["python"])
    requested_prefix = pathlib.Path(runtime["prefix"])
    if (not requested_python.is_absolute() or not requested_prefix.is_absolute()
            or requested_python.parent.parent.resolve() != requested_prefix.resolve()
            or pathlib.Path(sys.executable).resolve() != requested_python.resolve()
            or pathlib.Path(sys.prefix).resolve() != requested_prefix.resolve()):
        raise ValueError(
            "isolated worker did not start in the exact requested Python environment")
    panel = panel.validate()
    from autoforecast.adapter_contracts import descriptor
    from autoforecast.model_runtime import check_runtime
    from autoforecast.project import installed_dependencies
    check = check_runtime(descriptor(adapter).runtime)
    if not check.available:
        raise RuntimeError(check.summary)
    register_defaults(only=(module,))
    spark = None
    if adapter not in _ARROW_ONLY_ADAPTERS or (
        adapter in {"timesfm_hnb", "timesfm_v8"} and panel.plan is not None
    ):
        local_spark_config = resources.configure_local_spark_runtime()
        builder = SparkSession.builder.master(f"local[{profile.cores}]").appName(
            f"forecast-{adapter}")
        for key, value in profile.as_config().items():
            builder = builder.config(key, value)
        for key, value in local_spark_config.items():
            builder = builder.config(key, value)
        spark = (builder.config("spark.ui.showConsoleProgress", "false")
                 .config("spark.sql.execution.pyspark.udf.faulthandler.enabled", "true")
                 .getOrCreate())
        spark.sparkContext.setLogLevel("ERROR")
    try:
        runtime_evidence = {
            "schema": "autoforecast.worker-environment.v1",
            "python": sys.version, "executable": sys.executable,
            "dependencies": installed_dependencies(),
        }
        if spark is not None:
            # This code executes in a Spark Python worker, independently of driver imports.
            def probe(_):
                import importlib
                return {name: importlib.import_module(name).__version__
                        for name in ("pyarrow", "pandas", "numpy")}
            runtime_evidence["spark_python_worker"] = (
                spark.sparkContext.parallelize([0], 1).map(probe).collect()[0]
            )
        if "out_dir" in kwargs:
            evidence_path = pathlib.Path(kwargs["out_dir"]).parent / "worker-runtime.json"
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            with evidence_path.open("x") as stream:
                json.dump(runtime_evidence, stream, indent=2, sort_keys=True)
        # SFT/HMFNet/PACT/TFT-Q/TFT-HNB read Parquet through Arrow, so a JVM only adds
        # startup time and memory pressure. Other adapters keep the isolated Spark session.
        os.environ["AUTOFORECAST_ISOLATED_WORKER"] = "1"
        if tracking_context is not None and tracking_plan(adapter).mode == "autolog":
            try:
                import mlflow
            except ImportError as error:
                raise RuntimeError(
                    "adapter requires MLflow autologging but mlflow-tracking is not installed"
                ) from error
            mlflow.set_tracking_uri(tracking_context["tracking_uri"])
            mlflow.start_run(run_id=tracking_context["run_id"])
            enable_autologging(tracking_plan(adapter))
        forecast = registry.run(adapter, panel, origin, horizon,
                                spark=spark, **kwargs)
        response = {
            "schema": _RESPONSE_SCHEMA,
            "request_sha256": request_sha256,
            "forecast": forecast.as_record(),
        }
        encoded_response = json.dumps(
            response, indent=2, sort_keys=True, allow_nan=False
        ).encode()
        response_descriptor = os.open(
            pathlib.Path(response_name),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
        )
        try:
            view = memoryview(encoded_response)
            while view:
                written = os.write(response_descriptor, view)
                view = view[written:]
        finally:
            os.close(response_descriptor)
    finally:
        os.environ.pop("AUTOFORECAST_ISOLATED_WORKER", None)
        if spark is not None:
            spark.stop()
        validate_executable_source_snapshot(source)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python -m autoforecast.adapters.worker REQUEST.json RESPONSE.json")
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
