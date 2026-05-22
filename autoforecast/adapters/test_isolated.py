"""Strict process-boundary tests for isolated adapter execution."""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import io
import json
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

from autoforecast import resources
from autoforecast.acceptance_retirement import canonical_json_sha256
from autoforecast.adapters import _validate_daily_span, isolated, worker
from autoforecast.panel import Forecast, Panel, PlanSource


def _profile() -> resources.SparkProfile:
    return resources.SparkProfile(
        cores=1, driver_memory="1g", shuffle_partitions=1,
        worker_reuse=False, series_per_partition=1, python_worker_budget_gb=1.0)


def _source_snapshot(root: pathlib.Path) -> dict[str, object]:
    source_root = root / "source" / ("a" * 64)
    return {"root": str(source_root), "manifest_sha256": "a" * 64}


def _controls(root: pathlib.Path, name: str) -> dict[str, object]:
    return {
        "timeout_seconds": 2,
        "heartbeat_seconds": 1,
        "worker_log_path": root / f"{name}.log",
    }


def _request_envelope(request: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "autoforecast.isolated-request.v1",
        "request_sha256": canonical_json_sha256(request),
        "request": request,
    }


def test_daily_compatibility_helper_rejects_a_nonunit_daily_stride() -> None:
    for frequency in ("D", "1D"):
        panel = Panel("/not-read", "id", "date", "qty", frequency=frequency)
        assert _validate_daily_span(
            panel, "2024-01-01", 2, adapter="auto_ets"
        ) == dt.date(2024, 1, 1)
    with pytest.raises(ValueError, match="exact daily"):
        _validate_daily_span(
            Panel("/not-read", "id", "date", "qty", frequency="7D"),
            "2024-01-01",
            2,
            adapter="auto_ets",
        )


def _response_envelope(
    request: dict[str, object], forecast: dict[str, object]
) -> dict[str, object]:
    return {
        "schema": "autoforecast.isolated-response.v1",
        "request_sha256": request["request_sha256"],
        "forecast": forecast,
    }


def _assert_pid_reaped(pid: int, label: str) -> None:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    raise AssertionError(f"{label} {pid} survived isolated-worker cleanup")


def test_terminate_worker_retries_permission_error_after_exiting_leader_is_reaped() -> None:
    process = mock.Mock(pid=424242)
    process.poll.return_value = None
    process.wait.return_value = 0
    permission_error = PermissionError("leader is exiting")
    with mock.patch.object(
        isolated.os,
        "killpg",
        side_effect=(permission_error, None, ProcessLookupError()),
    ) as killpg:
        isolated._terminate_worker(process)

    assert killpg.call_args_list == [
        mock.call(process.pid, signal.SIGTERM),
        mock.call(process.pid, signal.SIGTERM),
        mock.call(process.pid, 0),
    ]
    assert process.wait.call_args_list == [
        mock.call(timeout=isolated._PROCESS_GROUP_PERMISSION_GRACE_SECONDS),
        mock.call(),
    ]


def test_terminate_worker_refuses_permission_error_for_live_unowned_group() -> None:
    process = mock.Mock(pid=434343)
    process.poll.return_value = None
    process.wait.side_effect = subprocess.TimeoutExpired(
        cmd="isolated-worker", timeout=isolated._PROCESS_GROUP_PERMISSION_GRACE_SECONDS
    )
    permission_error = PermissionError("process group is unowned")
    with mock.patch.object(
        isolated.os, "killpg", side_effect=permission_error
    ) as killpg:
        try:
            isolated._terminate_worker(process)
        except PermissionError as exc:
            assert exc is permission_error
        else:
            raise AssertionError("live unowned isolated-worker group was treated as reaped")

    killpg.assert_called_once_with(process.pid, signal.SIGTERM)
    process.wait.assert_called_once_with(
        timeout=isolated._PROCESS_GROUP_PERMISSION_GRACE_SECONDS
    )


def test_worker_refuses_ambiguous_or_nonstandard_request_before_adapter_or_spark() -> None:
    root = pathlib.Path(tempfile.mkdtemp())
    request = root / "request.json"
    response = root / "response.json"
    cases = (
        '{"adapter":"seasonal_naive","adapter":"auto_ets"}',
        '{"adapter":"seasonal_naive","metric":NaN}',
        '["not", "a", "request"]',
        json.dumps(
            {
                "schema": "autoforecast.isolated-request.v1",
                "request_sha256": "0" * 64,
                "request": {"adapter": "seasonal_naive"},
            }
        ),
    )
    for payload in cases:
        request.write_text(payload)
        with mock.patch.object(
                worker, "register_defaults",
                side_effect=AssertionError("invalid request reached adapter registration")):
            try:
                worker.main(str(request), str(response))
            except ValueError as exc:
                assert "isolated request" in str(exc), str(exc)
            else:
                raise AssertionError(f"ambiguous worker request was accepted: {payload}")


def test_worker_echoes_the_exact_consumed_request_hash() -> None:
    root = pathlib.Path(tempfile.mkdtemp())
    request_path = root / "request.json"
    response_path = root / "response.json"
    source_root = pathlib.Path(__file__).resolve().parent.parent.parent
    source = {"root": str(source_root), "manifest_sha256": "a" * 64}
    selected_python = pathlib.Path(sys.executable).absolute()
    panel = Panel(
        path="panel-A",
        series_col="id",
        time_col="date",
        target_col="qty",
        known_future=("price",),
        static=("store",),
        static_reals=("weight",),
        plan=PlanSource("plan-A", "valid", "known", price_col="price"),
    )
    request = {
        "adapter": "seasonal_naive",
        "panel": dataclasses.asdict(panel),
        "origin": "2020-01-01",
        "horizon": 7,
        "kwargs": {},
        "worker_profile": _profile().__dict__,
        "worker_runtime": {
            "python": str(selected_python),
            "prefix": str(selected_python.parent.parent.resolve()),
        },
        "source_snapshot": source,
    }
    envelope = _request_envelope(request)
    request_path.write_text(json.dumps(envelope, sort_keys=True))
    forecast = Forecast(
        adapter="seasonal_naive",
        path=str(root / "forecast"),
        origin="2020-01-01",
        horizon=7,
        n_series=1,
        disposition={"emitted": 7},
    )

    spark = mock.Mock()
    spark.sparkContext.parallelize.return_value.map.return_value.collect.return_value = [
        {"pyarrow": "test", "pandas": "test", "numpy": "test"}
    ]
    builder = mock.Mock()
    builder.master.return_value = builder
    builder.appName.return_value = builder
    builder.config.return_value = builder
    builder.getOrCreate.return_value = spark
    consumed_panels: list[Panel] = []

    def run_adapter(_adapter, consumed, *_args, **_kwargs):
        consumed_panels.append(consumed)
        if os.environ.get("AUTOFORECAST_ISOLATED_WORKER") != "1":
            raise AssertionError("worker did not declare session ownership")
        return forecast

    with mock.patch.object(
        worker, "validate_executable_source_snapshot", return_value=source
    ), mock.patch.object(worker, "register_defaults") as register, mock.patch.object(
        worker, "SparkSession", SimpleNamespace(builder=builder)
    ), mock.patch.object(
        worker.Panel, "validate", lambda self: self
    ), mock.patch.object(
        worker.registry, "run", side_effect=run_adapter
    ) as run:
        assert worker.main(str(request_path), str(response_path)) == 0

    response = json.loads(response_path.read_text())
    assert response == _response_envelope(envelope, forecast.as_record())
    register.assert_called_once()
    run.assert_called_once()
    assert len(consumed_panels) == 1
    consumed = consumed_panels[0]
    assert isinstance(consumed.known_future, tuple)
    assert isinstance(consumed.static, tuple)
    assert isinstance(consumed.static_reals, tuple)
    assert isinstance(consumed.plan, PlanSource)
    spark.stop.assert_called_once_with()
    builder.config.assert_has_calls(
        [
            mock.call("spark.driver.bindAddress", "127.0.0.1"),
            mock.call("spark.driver.host", "127.0.0.1"),
        ]
    )
    assert "AUTOFORECAST_ISOLATED_WORKER" not in os.environ


@pytest.mark.parametrize(
    "adapter",
    ("sft", "hmfnet",
     "pact", "tft_q", "tft_hnb"),
)
def test_arrow_only_worker_does_not_start_an_unused_spark_runtime(adapter: str) -> None:
    root = pathlib.Path(tempfile.mkdtemp())
    request_path = root / "request.json"
    response_path = root / "response.json"
    source_root = pathlib.Path(__file__).resolve().parent.parent.parent
    source = {"root": str(source_root), "manifest_sha256": "a" * 64}
    selected_python = pathlib.Path(sys.executable).absolute()
    request = {
        "adapter": adapter,
        "panel": dataclasses.asdict(Panel("panel-A", "id", "date", "qty")),
        "origin": "2020-01-01",
        "horizon": 7,
        "kwargs": {},
        "worker_profile": _profile().__dict__,
        "worker_runtime": {
            "python": str(selected_python),
            "prefix": str(selected_python.parent.parent.resolve()),
        },
        "source_snapshot": source,
    }
    envelope = _request_envelope(request)
    request_path.write_text(json.dumps(envelope, sort_keys=True))
    forecast = Forecast(
        adapter=adapter,
        path=str(root / "forecast"),
        origin="2020-01-01",
        horizon=7,
        n_series=1,
        disposition={"emitted": 7},
    )

    def run_adapter(_adapter, _panel, *_args, **kwargs):
        assert kwargs["spark"] is None
        assert os.environ.get("AUTOFORECAST_ISOLATED_WORKER") == "1"
        return forecast

    with mock.patch.object(
        worker, "validate_executable_source_snapshot", return_value=source
    ), mock.patch.object(worker, "register_defaults"), mock.patch.object(
        worker.Panel, "validate", lambda self: self
    ), mock.patch.object(
        worker.registry, "run", side_effect=run_adapter
    ), mock.patch.object(
        worker.resources,
        "configure_local_spark_runtime",
        side_effect=AssertionError("revision worker started Spark"),
    ):
        assert worker.main(str(request_path), str(response_path)) == 0

    assert json.loads(response_path.read_text()) == _response_envelope(
        envelope, forecast.as_record()
    )


@pytest.mark.parametrize("frequency", ["W", "7D"])
def test_worker_refuses_unsupported_cadence_before_registration_or_spark(
    frequency: str,
) -> None:
    root = pathlib.Path(tempfile.mkdtemp())
    request_path = root / "request.json"
    response_path = root / "response.json"
    source_root = pathlib.Path(__file__).resolve().parent.parent.parent
    source = {"root": str(source_root), "manifest_sha256": "a" * 64}
    selected_python = pathlib.Path(sys.executable).absolute()
    panel = Panel("panel-A", "id", "date", "qty", frequency=frequency)
    request = {
        "adapter": "tft",
        "panel": panel.__dict__,
        "origin": "2024-01-01",
        "horizon": 2,
        "kwargs": {},
        "worker_profile": _profile().__dict__,
        "worker_runtime": {
            "python": str(selected_python),
            "prefix": str(selected_python.parent.parent.resolve()),
        },
        "source_snapshot": source,
    }
    request_path.write_text(json.dumps(_request_envelope(request), sort_keys=True))
    with mock.patch.object(
            worker, "validate_executable_source_snapshot", return_value=source), \
            mock.patch.object(
                worker, "register_defaults",
                side_effect=AssertionError("unsupported request registered a runtime adapter")):
        try:
            worker.main(str(request_path), str(response_path))
        except ValueError as exc:
            assert "frequency families" in str(exc) and repr(frequency) in str(exc)
            return
    raise AssertionError(f"isolated worker accepted {frequency} for a daily-only adapter")


@pytest.mark.parametrize(
    "runtime_change",
    [
        {"tree_sha256": "a" * 64},
        {"java_home": "/runtime/java"},
        {"python": 7},
        {"prefix": None},
        {"python": None, "prefix": None},
    ],
)
def test_worker_refuses_invalid_runtime_schema(
    tmp_path: pathlib.Path, runtime_change: dict[str, object],
) -> None:
    selected_python = pathlib.Path(sys.executable).absolute()
    runtime = {
        "python": str(selected_python),
        "prefix": str(selected_python.parent.parent.resolve()),
        **runtime_change,
    }
    source = {"root": str(pathlib.Path(__file__).resolve().parents[2])}
    request = {
        "adapter": "seasonal_naive",
        "panel": dataclasses.asdict(Panel("unused", "id", "date", "qty")),
        "origin": "2024-01-01", "horizon": 7, "kwargs": {},
        "worker_profile": dataclasses.asdict(_profile()),
        "worker_runtime": runtime,
        "source_snapshot": source,
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_request_envelope(request)))
    with mock.patch.object(
        worker, "validate_executable_source_snapshot", return_value=source,
    ), mock.patch.object(
        worker.Panel, "validate", side_effect=AssertionError("panel was loaded"),
    ), mock.patch.object(
        worker, "register_defaults", side_effect=AssertionError("runtime was started"),
    ):
        with pytest.raises(ValueError, match="exact python and prefix"):
            worker.main(str(request_path), str(tmp_path / "response.json"))
    assert not (tmp_path / "response.json").exists()


@pytest.mark.parametrize("field", ["python", "prefix"])
def test_worker_refuses_runtime_redirect_before_loading_panel(
    tmp_path: pathlib.Path, field: str,
) -> None:
    selected_python = pathlib.Path(sys.executable).absolute()
    runtime = {
        "python": str(selected_python),
        "prefix": str(selected_python.parent.parent.resolve()),
    }
    runtime[field] = str(tmp_path / "other-runtime")
    source = {"root": str(pathlib.Path(__file__).resolve().parents[2])}
    request = {
        "adapter": "seasonal_naive",
        "panel": dataclasses.asdict(Panel("unused", "id", "date", "qty")),
        "origin": "2024-01-01", "horizon": 7, "kwargs": {},
        "worker_profile": dataclasses.asdict(_profile()),
        "worker_runtime": runtime,
        "source_snapshot": source,
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_request_envelope(request)))
    with mock.patch.object(
        worker, "validate_executable_source_snapshot", return_value=source,
    ), mock.patch.object(
        worker.Panel, "validate", side_effect=AssertionError("panel was loaded"),
    ), mock.patch.object(
        worker, "register_defaults", side_effect=AssertionError("runtime was started"),
    ):
        with pytest.raises(ValueError, match="exact requested Python environment"):
            worker.main(str(request_path), str(tmp_path / "response.json"))
    assert not (tmp_path / "response.json").exists()


@pytest.mark.parametrize("runtime", [None, {}, {"python": sys.executable, "prefix": sys.prefix},
                                      {"java_home": "/runtime/java", "tree_sha256": "a" * 64}])
def test_parent_refuses_removed_runtime_override_before_launch(
    tmp_path: pathlib.Path, runtime: object,
) -> None:
    out = tmp_path / "forecast"
    with mock.patch.object(
        isolated, "validate_executable_source_snapshot",
        side_effect=AssertionError("removed option reached source validation"),
    ), mock.patch.object(
        isolated, "_run_monitored_worker", side_effect=AssertionError("worker was launched"),
    ):
        with pytest.raises(ValueError, match="worker_runtime_identity is unsupported"):
            isolated.run(
                "seasonal_naive", Panel("unused", "id", "date", "qty"), "2024-01-01", 7,
                source_snapshot=_source_snapshot(tmp_path), worker_profile=_profile(),
                worker_runtime_identity=runtime, out_dir=str(out), **_controls(tmp_path, "refusal"),
            )
    assert not out.with_suffix(".request.json").exists()
    assert not out.with_suffix(".response.json").exists()


@pytest.mark.parametrize("explicit_python", [False, True])
def test_parent_runtime_request_is_accepted_by_worker(
    tmp_path: pathlib.Path, explicit_python: bool,
) -> None:
    """Exercise the real child request parser; only adapter execution is substituted."""
    out = tmp_path / "forecast"
    panel_path = tmp_path / "panel.parquet"
    pq.write_table(pa.table({"id": ["a"], "date": [dt.date(2024, 1, 1)], "qty": [1.0]}),
                   panel_path)
    panel = Panel(str(panel_path), "id", "date", "qty")
    source = {"root": str(pathlib.Path(__file__).resolve().parents[2]), "manifest_sha256": "a" * 64}
    selected_python = pathlib.Path(sys.executable).absolute()
    expected_runtime = {
        "python": str(selected_python),
        "prefix": str(selected_python.parent.parent.resolve()),
    }
    forecast = Forecast(
        adapter="sft", path=str(out), origin="2024-01-01", horizon=7,
        n_series=1, disposition={"emitted": 7},
    )

    def launch(argv, *, cwd, env, **_kwargs):
        assert argv[:3] == [str(selected_python), "-m", "autoforecast.adapters.worker"]
        assert cwd == pathlib.Path(source["root"])
        assert env["PYSPARK_PYTHON"] == expected_runtime["python"]
        assert env["PYTHONPATH"] == source["root"]
        request = json.loads(pathlib.Path(argv[-2]).read_text())
        assert request["request"]["worker_runtime"] == expected_runtime
        return SimpleNamespace(returncode=worker.main(argv[-2], argv[-1]), stderr="")

    with mock.patch.object(
        isolated, "validate_executable_source_snapshot", return_value=source,
    ), mock.patch.object(
        worker, "validate_executable_source_snapshot", return_value=source,
    ), mock.patch.object(
        isolated, "_run_monitored_worker", side_effect=launch,
    ), mock.patch.object(
        worker, "register_defaults",
    ), mock.patch.object(
        worker.registry, "run", return_value=forecast,
    ) as run_adapter, mock.patch.object(
        worker.resources, "configure_local_spark_runtime",
        side_effect=AssertionError("Arrow adapter started Spark"),
    ):
        result = isolated.run(
            "sft", panel, "2024-01-01", 7, source_snapshot=source, worker_profile=_profile(),
            worker_python=str(selected_python) if explicit_python else None,
            out_dir=str(out), **_controls(tmp_path, "handoff"),
        )
    assert result == forecast
    run_adapter.assert_called_once_with("sft", panel, "2024-01-01", 7, spark=None, out_dir=str(out))
    request = json.loads(out.with_suffix(".request.json").read_text())
    response = json.loads(out.with_suffix(".response.json").read_text())
    assert response == _response_envelope(request, forecast.as_record())


def test_parent_refuses_forecast_response_for_another_request_and_cleans_private_files() -> None:
    root = pathlib.Path(tempfile.mkdtemp())
    out = root / "forecast"
    panel = Panel(path="not-read", series_col="id", time_col="date", target_col="qty")
    source = _source_snapshot(root)
    private_paths: list[pathlib.Path] = []
    forecast = Forecast(
        adapter="seasonal_naive",
        path=str(out),
        origin="2020-01-01",
        horizon=7,
        n_series=1,
        disposition={"emitted": 7},
    ).as_record()

    def completed(argv, **_kwargs):
        private_request = pathlib.Path(argv[-2])
        private_response = pathlib.Path(argv[-1])
        private_paths.extend((private_request.parent, private_response.parent))
        envelope = json.loads(private_request.read_text())
        response = _response_envelope(envelope, forecast)
        response["request_sha256"] = "f" * 64
        private_response.write_text(json.dumps(response, sort_keys=True))
        return SimpleNamespace(returncode=0)

    with mock.patch.object(isolated, "_run_monitored_worker", side_effect=completed), \
            mock.patch.object(
                isolated, "validate_executable_source_snapshot", return_value=source
            ):
        try:
            isolated.run(
                "seasonal_naive", panel, "2020-01-01", 7,
                source_snapshot=source, worker_profile=_profile(), out_dir=str(out),
                **_controls(root, "foreign-response"))
        except RuntimeError as exc:
            assert "request-bound forecast" in str(exc), str(exc)
        else:
            raise AssertionError("a forecast response for another request was accepted")

    assert private_paths and all(not path.exists() for path in private_paths)
    assert out.with_suffix(".request.json").is_file()
    assert not out.with_suffix(".response.json").exists()


@pytest.mark.parametrize("adapter", ["tft_q", "tft_hnb"])
@pytest.mark.parametrize("explicit_checkpoint", [True, False])
def test_tft_isolated_worker_replays_a_complete_pair_in_the_same_attempt(
    tmp_path: pathlib.Path, adapter: str, explicit_checkpoint: bool,
) -> None:
    out = tmp_path / "forecast"
    checkpoint = (tmp_path / "model.pt" if explicit_checkpoint
                  else tmp_path / "forecast-v6-artifacts" / "model.pt")
    source = _source_snapshot(tmp_path)
    calls: list[tuple[list[str], pathlib.Path]] = []
    record = Forecast(
        adapter=adapter, path=str(out), origin="2024-04-09", horizon=28,
        n_series=1, disposition={"emitted": 28},
    ).as_record()

    def completed(argv, *, worker_log_path, **_kwargs):
        calls.append((list(argv), worker_log_path))
        if len(calls) == 1:
            out.mkdir()
            pq.write_table(
                pa.table({
                    "series_id": ["a"] * 28,
                    "date": pa.array(
                        [dt.date(2024, 4, 9) + dt.timedelta(days=day)
                         for day in range(1, 29)], type=pa.date32(),
                    ),
                    "yhat": [1.0] * 28,
                }),
                out / "part-00000.parquet",
            )
            checkpoint.parent.mkdir(exist_ok=True)
            checkpoint.write_bytes(b"completed model")
            return SimpleNamespace(returncode=1, stderr="worker exited after publication")
        request = json.loads(pathlib.Path(argv[-2]).read_text())
        pathlib.Path(argv[-1]).write_text(
            json.dumps(_response_envelope(request, record), sort_keys=True)
        )
        return SimpleNamespace(returncode=0, stderr="")

    with mock.patch.object(isolated, "_run_monitored_worker", side_effect=completed), \
            mock.patch.object(
                isolated, "validate_executable_source_snapshot", return_value=source
            ):
        result = isolated.run(
            adapter, Panel("/not-read", "id", "date", "qty"), "2024-04-09", 28,
            source_snapshot=source, worker_profile=_profile(), out_dir=str(out),
            checkpoint_out=str(checkpoint) if explicit_checkpoint else None, **_controls(tmp_path, "v6-recovery"),
        )
    assert result.as_record() == record
    assert len(calls) == 2 and calls[0][0] == calls[1][0]
    assert calls[1][1].name == calls[0][1].name + ".recovery"


def test_mlflow_credentials_are_transient_and_only_the_endpoint_hash_is_retained() -> None:
    root = pathlib.Path(tempfile.mkdtemp())
    out = root / "forecast"
    source = _source_snapshot(root)
    secret_uri = "https://user:secret@example.test/mlflow?signature=private"
    forecast_record = Forecast(
        adapter="seasonal_naive",
        path=str(out),
        origin="2020-01-01",
        horizon=7,
        n_series=1,
        disposition={"emitted": 7},
    ).as_record()

    def completed(argv, *, env, **_kwargs):
        assert env["AUTOFORECAST_WORKER_MLFLOW_TRACKING_URI"] == secret_uri
        envelope = json.loads(pathlib.Path(argv[-2]).read_text())
        retained = envelope["request"]["tracking_context"]
        assert retained == {
            "tracking_uri_sha256": hashlib.sha256(secret_uri.encode()).hexdigest(),
            "run_id": "run-1",
            "experiment_id": "experiment-1",
        }
        assert "secret" not in json.dumps(envelope)
        pathlib.Path(argv[-1]).write_text(
            json.dumps(_response_envelope(envelope, forecast_record), sort_keys=True)
        )
        return SimpleNamespace(returncode=0)

    with mock.patch.object(isolated, "_run_monitored_worker", side_effect=completed), \
            mock.patch.object(
                isolated, "validate_executable_source_snapshot", return_value=source
            ):
        isolated.run(
            "seasonal_naive",
            Panel("not-read", "id", "date", "qty"),
            "2020-01-01",
            7,
            source_snapshot=source,
            worker_profile=_profile(),
            tracking_context={
                "tracking_uri": secret_uri,
                "run_id": "run-1",
                "experiment_id": "experiment-1",
            },
            out_dir=str(out),
            **_controls(root, "mlflow-secret"),
        )

    assert "secret" not in out.with_suffix(".request.json").read_text()


def test_parent_refuses_private_request_aba_and_cleans_private_files() -> None:
    root = pathlib.Path(tempfile.mkdtemp())
    out = root / "forecast"
    retained_request = out.with_suffix(".request.json")
    panel = Panel(path="panel-A", series_col="id", time_col="date", target_col="qty")
    source = _source_snapshot(root)
    private_paths: list[pathlib.Path] = []
    consumed_panel: list[str] = []
    forecast = Forecast(
        adapter="seasonal_naive",
        path=str(out),
        origin="2020-01-01",
        horizon=7,
        n_series=1,
        disposition={"emitted": 7},
    ).as_record()

    def completed(argv, **_kwargs):
        private_request = pathlib.Path(argv[-2])
        private_response = pathlib.Path(argv[-1])
        assert private_request != retained_request
        private_paths.extend((private_request.parent, private_response.parent))
        original = private_request.read_bytes()
        envelope = json.loads(original)
        swapped_request = json.loads(json.dumps(envelope["request"]))
        swapped_request["panel"]["path"] = "panel-B"
        swapped = _request_envelope(swapped_request)
        private_request.chmod(0o600)
        private_request.write_text(json.dumps(swapped, indent=2, sort_keys=True))
        consumed = json.loads(private_request.read_text())
        consumed_panel.append(consumed["request"]["panel"]["path"])
        private_response.write_text(
            json.dumps(_response_envelope(consumed, forecast), sort_keys=True)
        )
        private_request.write_bytes(original)
        private_request.chmod(0o400)
        return SimpleNamespace(returncode=0)

    with mock.patch.object(isolated, "_run_monitored_worker", side_effect=completed), \
            mock.patch.object(
                isolated, "validate_executable_source_snapshot", return_value=source
            ):
        try:
            isolated.run(
                "seasonal_naive", panel, "2020-01-01", 7,
                source_snapshot=source, worker_profile=_profile(), out_dir=str(out),
                **_controls(root, "request-aba"))
        except RuntimeError as exc:
            assert "private request changed" in str(exc), str(exc)
        else:
            raise AssertionError("a private request A-to-B-to-A substitution was accepted")

    assert consumed_panel == ["panel-B"]
    assert json.loads(retained_request.read_text())["request"]["panel"]["path"] == "panel-A"
    assert private_paths and all(not path.exists() for path in private_paths)
    assert not out.with_suffix(".response.json").exists()


def test_parent_refuses_ambiguous_or_nonstandard_worker_response() -> None:
    root = pathlib.Path(tempfile.mkdtemp())
    panel = Panel(path="not-read", series_col="id", time_col="date", target_col="qty")
    cases = (
        '{"adapter":"first","adapter":"second"}',
        '{"adapter":"seasonal_naive","metric":Infinity}',
        '["not", "a", "forecast"]',
    )
    for index, payload in enumerate(cases):
        out = root / f"forecast-{index}"

        def completed(argv, payload=payload, **_kwargs):
            pathlib.Path(argv[-1]).write_text(payload)
            return SimpleNamespace(returncode=0)

        source = _source_snapshot(root)
        with mock.patch.object(isolated, "_run_monitored_worker", side_effect=completed), \
                mock.patch.object(
                    isolated, "validate_executable_source_snapshot", return_value=source
                ):
            try:
                isolated.run(
                    "seasonal_naive", panel, "2020-01-01", 7,
                    source_snapshot=source, worker_profile=_profile(), out_dir=str(out),
                    **_controls(root, f"ambiguous-{index}"))
            except RuntimeError as exc:
                assert "isolated adapter" in str(exc), str(exc)
            else:
                raise AssertionError(f"ambiguous worker response was accepted: {payload}")


def test_parent_refuses_nonfinite_request_before_starting_worker() -> None:
    root = pathlib.Path(tempfile.mkdtemp())
    panel = Panel(path="not-read", series_col="id", time_col="date", target_col="qty")
    source = _source_snapshot(root)
    with mock.patch.object(
            isolated, "_run_monitored_worker",
            side_effect=AssertionError("non-finite request started the worker")), \
            mock.patch.object(
                isolated, "validate_executable_source_snapshot", return_value=source
            ):
        try:
            isolated.run(
                "seasonal_naive", panel, "2020-01-01", 7,
                source_snapshot=source, worker_profile=_profile(),
                out_dir=str(root / "forecast"), alpha=float("nan"),
                **_controls(root, "nonfinite"))
        except ValueError as exc:
            assert "strict JSON" in str(exc), str(exc)
        else:
            raise AssertionError("a non-finite isolated request started the worker")

    with mock.patch.object(
            isolated, "_run_monitored_worker",
            side_effect=AssertionError("coerced-key request started the worker")), \
            mock.patch.object(
                isolated, "validate_executable_source_snapshot", return_value=source
            ):
        try:
            isolated.run(
                "seasonal_naive", panel, "2020-01-01", 7,
                source_snapshot=source, worker_profile=_profile(),
                out_dir=str(root / "forecast"),
                options={1: "numeric", "1": "string"},
                **_controls(root, "coerced-key"))
        except TypeError as exc:
            assert "exact string keys" in str(exc), str(exc)
        else:
            raise AssertionError("colliding isolated request keys started the worker")


@pytest.mark.parametrize("frequency", ["W", "7D"])
def test_parent_refuses_unsupported_cadence_before_starting_worker(
    frequency: str,
) -> None:
    root = pathlib.Path(tempfile.mkdtemp())
    source = _source_snapshot(root)
    panel = Panel("/not-read", "id", "date", "qty", frequency=frequency)
    with mock.patch.object(
            isolated, "validate_executable_source_snapshot", return_value=source), \
            mock.patch.object(
                isolated, "_run_monitored_worker",
                side_effect=AssertionError("unsupported request started an isolated worker")):
        try:
            isolated.run(
                "tft",
                panel,
                "2024-01-01",
                2,
                source_snapshot=source,
                worker_profile=_profile(),
                out_dir=str(root / "forecast"),
                **_controls(root, "unsupported-cadence"),
            )
        except ValueError as exc:
            assert "frequency families" in str(exc) and repr(frequency) in str(exc)
            return
    raise AssertionError(f"isolated parent accepted {frequency} for a daily-only adapter")


def _worker_source(root: pathlib.Path, program: str) -> dict[str, object]:
    source = _source_snapshot(root)
    source_root = pathlib.Path(source["root"])
    module = source_root / "autoforecast" / "adapters" / "worker.py"
    module.parent.mkdir(parents=True)
    (source_root / "autoforecast" / "__init__.py").write_text("")
    (module.parent / "__init__.py").write_text("")
    module.write_text(program)
    return source


def _response_program(record: dict[str, object], *, sleep_seconds: float) -> str:
    encoded = json.dumps(record, sort_keys=True, allow_nan=False)
    return f"""
import json
import pathlib
import sys
import time

print('worker-stdout', flush=True)
print('worker-stderr', file=sys.stderr, flush=True)
time.sleep({sleep_seconds!r})
request = json.loads(pathlib.Path(sys.argv[1]).read_text())
response = {{
    'schema': 'autoforecast.isolated-response.v1',
    'request_sha256': request['request_sha256'],
    'forecast': json.loads({encoded!r}),
}}
pathlib.Path(sys.argv[2]).write_text(json.dumps(response, sort_keys=True))
"""


def test_monitored_worker_heartbeats_logs_both_streams_and_returns_real_response() -> None:
    root = pathlib.Path(tempfile.mkdtemp())
    output = root / "forecast"
    record = Forecast(
        adapter="seasonal_naive",
        path=str(output),
        origin="2020-01-01",
        horizon=7,
        n_series=1,
        disposition={"emitted": 7},
    ).as_record()
    source = _worker_source(root, _response_program(record, sleep_seconds=1.2))
    heartbeats: list[str] = []
    log = root / "worker.log"
    with mock.patch.object(
        isolated, "validate_executable_source_snapshot", return_value=source
    ):
        forecast = isolated.run(
            "seasonal_naive",
            Panel("not-read", "id", "date", "qty"),
            "2020-01-01",
            7,
            source_snapshot=source,
            worker_profile=_profile(),
            timeout_seconds=4,
            heartbeat_seconds=1,
            worker_log_path=log,
            heartbeat_sink=heartbeats.append,
            out_dir=str(output),
        )
    assert forecast.as_record() == record
    elapsed = [int(message.rsplit("=", 1)[1]) for message in heartbeats]
    assert elapsed and elapsed[0] == 1
    assert all(left < right for left, right in zip(elapsed, elapsed[1:], strict=False))
    logged = log.read_text()
    assert "[stdout] worker-stdout" in logged
    assert "[stderr] worker-stderr" in logged
    retained_request = json.loads(output.with_suffix(".request.json").read_text())
    retained_response = json.loads(output.with_suffix(".response.json").read_text())
    assert set(retained_request) == {"schema", "request_sha256", "request"}
    assert retained_request["request_sha256"] == canonical_json_sha256(
        retained_request["request"]
    )
    assert retained_response == _response_envelope(retained_request, record)


def test_monitored_worker_relays_live_output_without_log_prefixes() -> None:
    root = pathlib.Path(tempfile.mkdtemp())
    stream = io.StringIO()
    environment = dict(os.environ)
    environment["AUTOFORECAST_LIVE_PROGRESS"] = "1"
    with mock.patch.object(isolated.sys, "stderr", stream):
        completed = isolated._run_monitored_worker(
            [
                sys.executable,
                "-c",
                "import sys; print('clean-out'); print('clean-progress', file=sys.stderr)",
            ],
            cwd=root,
            env=environment,
            timeout_seconds=3,
            heartbeat_seconds=1,
            worker_log_path=root / "worker.log",
            heartbeat_sink=lambda _message: None,
        )
    assert completed.returncode == 0
    assert "clean-out" in stream.getvalue() and "clean-progress" in stream.getvalue()
    assert "[stdout]" not in stream.getvalue() and "[stderr]" not in stream.getvalue()
    retained = (root / "worker.log").read_text()
    assert "[stdout] clean-out" in retained and "[stderr] clean-progress" in retained


def test_monitored_worker_reaps_inherited_descendant_after_success() -> None:
    root = pathlib.Path(tempfile.mkdtemp())
    output = root / "forecast"
    descendant_pid_path = root / "descendant.pid"
    descendant_ready_path = root / "descendant.ready"
    graceful_exit_path = root / "descendant.graceful"
    record = Forecast(
        adapter="seasonal_naive",
        path=str(output),
        origin="2020-01-01",
        horizon=7,
        n_series=1,
        disposition={"emitted": 7},
    ).as_record()
    encoded = json.dumps(record, sort_keys=True, allow_nan=False)
    child_program = f"""
import pathlib
import signal
import time

def stop(_signum, _frame):
    time.sleep(0.2)
    pathlib.Path({str(graceful_exit_path)!r}).write_text('complete')
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop)
pathlib.Path({str(descendant_ready_path)!r}).write_text('ready')
time.sleep(60)
"""
    source = _worker_source(
        root,
        f"""
import json
import pathlib
import subprocess
import sys
import time
child = subprocess.Popen([sys.executable, '-c', {child_program!r}])
while not pathlib.Path({str(descendant_ready_path)!r}).exists():
    time.sleep(0.01)
pathlib.Path({str(descendant_pid_path)!r}).write_text(str(child.pid))
request = json.loads(pathlib.Path(sys.argv[1]).read_text())
response = {{
    'schema': 'autoforecast.isolated-response.v1',
    'request_sha256': request['request_sha256'],
    'forecast': json.loads({encoded!r}),
}}
pathlib.Path(sys.argv[2]).write_text(json.dumps(response, sort_keys=True))
""",
    )
    with mock.patch.object(
        isolated, "validate_executable_source_snapshot", return_value=source
    ):
        forecast = isolated.run(
            "seasonal_naive",
            Panel("not-read", "id", "date", "qty"),
            "2020-01-01",
            7,
            source_snapshot=source,
            worker_profile=_profile(),
            timeout_seconds=4,
            heartbeat_seconds=1,
            worker_log_path=root / "successful-descendant.log",
            out_dir=str(output),
        )
    assert forecast.as_record() == record
    _assert_pid_reaped(int(descendant_pid_path.read_text()), "isolated worker descendant")
    assert graceful_exit_path.read_text() == "complete"


def test_monitored_worker_times_out_terminates_and_retains_log() -> None:
    root = pathlib.Path(tempfile.mkdtemp())
    descendant_pid_path = root / "descendant.pid"
    source = _worker_source(
        root,
        f"""
import pathlib
import subprocess
import sys
import time
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
pathlib.Path({str(descendant_pid_path)!r}).write_text(str(child.pid))
print('before-timeout', file=sys.stderr, flush=True)
time.sleep(60)
""",
    )
    log = root / "timeout.log"
    heartbeats: list[str] = []
    with mock.patch.object(
        isolated, "validate_executable_source_snapshot", return_value=source
    ):
        try:
            isolated.run(
                "seasonal_naive",
                Panel("not-read", "id", "date", "qty"),
                "2020-01-01",
                7,
                source_snapshot=source,
                worker_profile=_profile(),
                timeout_seconds=2,
                heartbeat_seconds=1,
                worker_log_path=log,
                heartbeat_sink=heartbeats.append,
                out_dir=str(root / "forecast"),
            )
        except RuntimeError as exc:
            assert "exceeded timeout_seconds=2" in str(exc), str(exc)
        else:
            raise AssertionError("isolated worker exceeded timeout without termination")
    assert heartbeats and "elapsed_seconds=1" in heartbeats[0]
    assert "before-timeout" in log.read_text()
    _assert_pid_reaped(int(descendant_pid_path.read_text()), "isolated worker descendant")


def test_monitored_worker_retains_full_log_but_bounds_failure_tail() -> None:
    root = pathlib.Path(tempfile.mkdtemp())
    source = _worker_source(
        root,
        """
import sys
sys.stderr.write('Z' * 12000)
sys.stderr.flush()
raise SystemExit(7)
""",
    )
    log = root / "failure.log"
    with mock.patch.object(
        isolated, "validate_executable_source_snapshot", return_value=source
    ):
        try:
            isolated.run(
                "seasonal_naive",
                Panel("not-read", "id", "date", "qty"),
                "2020-01-01",
                7,
                source_snapshot=source,
                worker_profile=_profile(),
                timeout_seconds=4,
                heartbeat_seconds=1,
                worker_log_path=log,
                out_dir=str(root / "forecast"),
            )
        except RuntimeError as exc:
            message = str(exc)
            assert "exit code 7" in message, message
            assert len(message) < 5000, len(message)
        else:
            raise AssertionError("isolated worker nonzero exit was accepted")
    assert log.stat().st_size > 12000


def test_monitored_worker_refuses_relative_log_before_process_start() -> None:
    root = pathlib.Path(tempfile.mkdtemp())
    source = _source_snapshot(root)
    with mock.patch.object(
        isolated, "validate_executable_source_snapshot", return_value=source
    ), mock.patch.object(
        isolated.subprocess,
        "Popen",
        side_effect=AssertionError("relative log started the worker"),
    ):
        try:
            isolated.run(
                "seasonal_naive",
                Panel("not-read", "id", "date", "qty"),
                "2020-01-01",
                7,
                source_snapshot=source,
                worker_profile=_profile(),
                timeout_seconds=4,
                heartbeat_seconds=1,
                worker_log_path="relative.log",
                out_dir=str(root / "forecast"),
            )
        except ValueError as exc:
            assert "worker_log_path must be absolute" in str(exc), str(exc)
        else:
            raise AssertionError("relative isolated worker log was accepted")


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_"):
            function()
            print("PASS " + name)
