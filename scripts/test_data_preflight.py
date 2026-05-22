"""The data-preflight command owns only runtimes it creates itself."""

import os
import sys
import threading
from types import SimpleNamespace
from unittest import mock

import pytest
from autoforecast.model_api import CheckReport, CheckResult, DataFacts

from scripts import run_data_preflight as cli


@pytest.fixture
def command(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [
        "run_data_preflight", str(tmp_path / "dataset.yaml"), str(tmp_path / "data"),
        "--as-of", "2024-01-01", "--output", str(tmp_path / "output"),
    ])
    monkeypatch.setattr(cli.resources, "spark_profile", lambda **_: SimpleNamespace(cores=1))
    return tmp_path


@pytest.mark.parametrize("existing", ["active_session", "default_session", "bare_context"])
def test_preflight_refuses_borrowed_runtime_without_changes(command, monkeypatch, existing):
    borrowed = mock.Mock()
    builder = mock.Mock()
    builder.appName.return_value = builder
    builder.config.return_value = builder
    builder.getOrCreate.return_value = borrowed
    session = SimpleNamespace(
        Builder=SimpleNamespace(_lock=threading.RLock()), builder=builder,
        getActiveSession=lambda: borrowed if existing == "active_session" else None,
        _instantiatedSession=borrowed if existing == "default_session" else None,
    )
    context = SimpleNamespace(
        _lock=threading.RLock(),
        _active_spark_context=borrowed if existing == "bare_context" else None,
    )
    monkeypatch.setattr(cli, "SparkSession", session)
    monkeypatch.setattr(cli, "SparkContext", context)
    monkeypatch.setenv("PYSPARK_PYTHON", "caller-python")
    monkeypatch.setenv("PYSPARK_DRIVER_PYTHON", "caller-driver")
    load = mock.Mock(side_effect=AssertionError("borrowed runtime reached data loading"))
    monkeypatch.setattr(cli, "load_dataset_contract", load)

    with pytest.raises(RuntimeError, match="requires a fresh Spark runtime"):
        cli.main()

    builder.getOrCreate.assert_not_called()
    borrowed.stop.assert_not_called()
    load.assert_not_called()
    assert context._active_spark_context is (borrowed if existing == "bare_context" else None)
    assert os.environ["PYSPARK_PYTHON"] == "caller-python"
    assert os.environ["PYSPARK_DRIVER_PYTHON"] == "caller-driver"


@pytest.mark.parametrize(
    "outcome", ["passed", "rejected", "inspect_error", "load_error", "write_error"]
)
def test_preflight_stops_owned_runtime_on_every_exit(command, monkeypatch, outcome):
    spark = mock.Mock()
    successor = object()
    context = SimpleNamespace(_active_spark_context=spark)
    monkeypatch.setattr(cli, "SparkContext", context)
    # A caller may create its own context as soon as stop releases Spark's creation lock.
    spark.stop.side_effect = lambda: setattr(context, "_active_spark_context", successor)
    acquire = mock.Mock(return_value=spark)
    monkeypatch.setattr(cli.resources, "acquire_local_runtime", acquire)
    contract = mock.Mock()
    load = mock.Mock(return_value=contract)
    if outcome == "load_error":
        load.side_effect = ValueError("bad contract")
    monkeypatch.setattr(cli, "load_dataset_contract", load)
    facts = DataFacts("2024-01-01", "a" * 64, ("2024-01-01", "2024-01-01"),
                      1, 1, 1, 0, 0, {}, {}, None, {})
    report = CheckReport("data", facts, (
        CheckResult("data", "Data", "failed" if outcome == "rejected" else "passed", "result"),
    ))
    inspect = mock.Mock(return_value=report)
    if outcome == "inspect_error":
        inspect.side_effect = ValueError("unreadable panel")
    monkeypatch.setattr(cli, "inspect_panel", inspect)
    write = mock.Mock(return_value=command / "output")
    if outcome == "write_error":
        write.side_effect = OSError("unwritable report")
    monkeypatch.setattr(cli, "write_data_observability_report", write)

    if outcome in {"load_error", "write_error"}:
        with pytest.raises((ValueError, OSError)):
            cli.main()
    else:
        assert cli.main() == (0 if outcome == "passed" else 2)

    acquire.assert_called_once()
    spark.stop.assert_called_once()
    assert context._active_spark_context is successor
    if outcome != "load_error":
        inspect.assert_called_once_with(contract.panel.return_value, spark, as_of="2024-01-01")
