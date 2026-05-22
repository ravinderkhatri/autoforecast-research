# Lifecycle tests for retired experiment runners remain in the internal repository.
"""Pure mocked lifecycle tests for command-line Spark entry points."""

from __future__ import annotations

import pathlib
import sys
import threading
from types import SimpleNamespace
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import run_scenario_contrast as scenario  # noqa: E402
from scripts import score_latent_demand as latent  # noqa: E402


def _runtime(*, active=None, instantiated=None, context=None, log_error=None):
    set_log_level = mock.Mock(name="setLogLevel", side_effect=log_error)
    spark = SimpleNamespace(
        stop=mock.Mock(name="stop"),
        sparkContext=SimpleNamespace(setLogLevel=set_log_level),
    )
    builder = mock.Mock(name="builder")
    builder.master.return_value = builder
    builder.appName.return_value = builder
    builder.config.return_value = builder
    builder.getOrCreate.return_value = spark
    session_type = SimpleNamespace(
        getActiveSession=lambda: active,
        _instantiatedSession=instantiated,
        Builder=SimpleNamespace(_lock=threading.RLock()),
        builder=builder,
    )
    context_type = SimpleNamespace(
        _active_spark_context=context, _lock=threading.RLock())
    return session_type, context_type, builder, spark


def _profile():
    return scenario.resources.SparkProfile(
        cores=1, driver_memory="1g", shuffle_partitions=1,
        worker_reuse=False, series_per_partition=1, python_worker_budget_gb=1.0)


def _raises(call) -> str:
    try:
        call()
    except RuntimeError as exc:
        return str(exc)
    raise AssertionError("expected injected lifecycle failure")


def _scenario_args():
    return SimpleNamespace(campaign_col="campaign", checkpoint="checkpoint.pt")


def _scenario_record():
    return {"run": {"origin": "2026-01-01", "horizon": 1, "series_ids": ["s"]}}


def _latent_argv() -> list[str]:
    return [
        "--actuals", "actuals", "--forecast", "forecast", "--origin", "2026-01-01",
        "--horizon", "1", "--n-series", "1",
    ]






def test_get_or_create_failure_never_enters_body_or_stops_unknown_runtime() -> None:
    profile = _profile()

    session_type, context_type, builder, spark = _runtime()
    builder.getOrCreate.side_effect = RuntimeError("Spark startup failed")
    body = mock.Mock(name="_run_with_spark")
    with mock.patch.object(scenario, "SparkSession", session_type), \
         mock.patch.object(scenario, "SparkContext", context_type), \
         mock.patch.object(scenario, "parse", return_value=_scenario_args()), \
         mock.patch.object(
             scenario.ours, "load_checkpoint", return_value=(object(), _scenario_record())), \
         mock.patch.object(scenario.resources, "spark_profile", return_value=profile), \
         mock.patch.object(scenario, "_run_with_spark", body):
        assert _raises(lambda: scenario.main([])) == "Spark startup failed"
    body.assert_not_called()
    spark.stop.assert_not_called()
    session_type, context_type, builder, spark = _runtime()
    builder.getOrCreate.side_effect = RuntimeError("Spark startup failed")
    body = mock.Mock(name="_run_with_spark")
    with mock.patch.object(latent, "SparkSession", session_type), \
         mock.patch.object(latent, "SparkContext", context_type), \
         mock.patch.object(latent.latent_actuals, "load", return_value=object()), \
         mock.patch.object(latent.resources, "spark_profile", return_value=profile), \
         mock.patch.object(latent, "_run_with_spark", body):
        assert _raises(lambda: latent.main(_latent_argv())) == "Spark startup failed"
    body.assert_not_called()
    spark.stop.assert_not_called()




def test_all_owned_runtimes_stop_after_body_failure() -> None:
    profile = _profile()

    session_type, context_type, _builder, spark = _runtime()
    with mock.patch.object(scenario, "SparkSession", session_type), \
         mock.patch.object(scenario, "SparkContext", context_type), \
         mock.patch.object(scenario, "parse", return_value=_scenario_args()), \
         mock.patch.object(
             scenario.ours, "load_checkpoint", return_value=(object(), _scenario_record())), \
         mock.patch.object(scenario.resources, "spark_profile", return_value=profile), \
         mock.patch.object(
             scenario, "_run_with_spark", side_effect=RuntimeError("body failed")):
        assert _raises(lambda: scenario.main([])) == "body failed"
    spark.stop.assert_called_once_with()

    session_type, context_type, _builder, spark = _runtime()
    with mock.patch.object(latent, "SparkSession", session_type), \
         mock.patch.object(latent, "SparkContext", context_type), \
         mock.patch.object(latent.latent_actuals, "load", return_value=object()), \
         mock.patch.object(latent.resources, "spark_profile", return_value=profile), \
         mock.patch.object(
             latent, "_run_with_spark", side_effect=RuntimeError("body failed")):
        assert _raises(lambda: latent.main(_latent_argv())) == "body failed"
    spark.stop.assert_called_once_with()


def test_all_owned_runtimes_stop_when_set_log_level_fails() -> None:
    profile = _profile()

    session_type, context_type, _builder, spark = _runtime(
        log_error=RuntimeError("log setup failed"))
    with mock.patch.object(scenario, "SparkSession", session_type), \
         mock.patch.object(scenario, "SparkContext", context_type), \
         mock.patch.object(scenario, "parse", return_value=_scenario_args()), \
         mock.patch.object(
             scenario.ours, "load_checkpoint", return_value=(object(), _scenario_record())), \
         mock.patch.object(scenario.resources, "spark_profile", return_value=profile):
        assert _raises(lambda: scenario.main([])) == "log setup failed"
    spark.stop.assert_called_once_with()

    session_type, context_type, _builder, spark = _runtime(
        log_error=RuntimeError("log setup failed"))
    with mock.patch.object(latent, "SparkSession", session_type), \
         mock.patch.object(latent, "SparkContext", context_type), \
         mock.patch.object(latent.latent_actuals, "load", return_value=object()), \
         mock.patch.object(latent.resources, "spark_profile", return_value=profile):
        assert _raises(lambda: latent.main(_latent_argv())) == "log setup failed"
    spark.stop.assert_called_once_with()


def test_embedded_commands_never_stop_any_caller_owned_runtime() -> None:
    profile = _profile()
    for state in ("active", "default", "context"):
        session_type, context_type, builder, spark = _runtime()
        if state == "active":
            session_type.getActiveSession = lambda spark=spark: spark
        elif state == "default":
            session_type._instantiatedSession = spark
        else:
            context_type._active_spark_context = object()
        with mock.patch.object(scenario, "SparkSession", session_type), \
             mock.patch.object(scenario, "SparkContext", context_type), \
             mock.patch.object(scenario, "parse", return_value=_scenario_args()), \
             mock.patch.object(
                 scenario.ours, "load_checkpoint", return_value=(object(), _scenario_record())), \
             mock.patch.object(scenario.resources, "spark_profile", return_value=profile), \
             mock.patch.object(scenario, "_run_with_spark", return_value=0):
            assert scenario.main([]) == 0
        assert builder.getOrCreate.call_count == (1 if state == "context" else 0)
        builder.master.assert_not_called()
        builder.appName.assert_not_called()
        builder.config.assert_not_called()
        spark.sparkContext.setLogLevel.assert_not_called()
        spark.stop.assert_not_called()

        session_type, context_type, builder, spark = _runtime()
        if state == "active":
            session_type.getActiveSession = lambda spark=spark: spark
        elif state == "default":
            session_type._instantiatedSession = spark
        else:
            context_type._active_spark_context = object()
        with mock.patch.object(latent, "SparkSession", session_type), \
             mock.patch.object(latent, "SparkContext", context_type), \
             mock.patch.object(latent.latent_actuals, "load", return_value=object()), \
             mock.patch.object(latent.resources, "spark_profile", return_value=profile), \
             mock.patch.object(latent, "_run_with_spark", return_value=0):
            assert latent.main(_latent_argv()) == 0
        assert builder.getOrCreate.call_count == (1 if state == "context" else 0)
        builder.master.assert_not_called()
        builder.appName.assert_not_called()
        builder.config.assert_not_called()
        spark.sparkContext.setLogLevel.assert_not_called()
        spark.stop.assert_not_called()

    session_type, context_type, _builder, spark = _runtime(active=object())
    with mock.patch.object(scenario, "SparkSession", session_type), \
         mock.patch.object(scenario, "SparkContext", context_type), \
         mock.patch.object(scenario, "parse", return_value=_scenario_args()), \
         mock.patch.object(
             scenario.ours, "load_checkpoint", return_value=(object(), _scenario_record())), \
         mock.patch.object(scenario.resources, "spark_profile", return_value=profile), \
         mock.patch.object(
             scenario, "_run_with_spark", side_effect=RuntimeError("body failed")):
        assert _raises(lambda: scenario.main([])) == "body failed"
    spark.sparkContext.setLogLevel.assert_not_called()
    spark.stop.assert_not_called()

    session_type, context_type, _builder, spark = _runtime(active=object())
    with mock.patch.object(latent, "SparkSession", session_type), \
         mock.patch.object(latent, "SparkContext", context_type), \
         mock.patch.object(latent.latent_actuals, "load", return_value=object()), \
         mock.patch.object(latent.resources, "spark_profile", return_value=profile), \
         mock.patch.object(
             latent, "_run_with_spark", side_effect=RuntimeError("body failed")):
        assert _raises(lambda: latent.main(_latent_argv())) == "body failed"
    spark.sparkContext.setLogLevel.assert_not_called()
    spark.stop.assert_not_called()


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print("PASS " + name)
