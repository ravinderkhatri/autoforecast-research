"""Direct publication and guard tests for the registered revision adapter seam."""

from __future__ import annotations

import datetime as dt
import json
import pathlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from autoforecast.adapters import mft as subject
from autoforecast.model_names import REVISION_MODELS
from autoforecast.panel import Forecast, Panel

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _recipe(variant: str) -> dict[str, object]:
    path = ROOT / f"configs/experiments/multibranch_revisions/{REVISION_MODELS[variant]}_campaign_default.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("variant", ["v2", "v3", "v5"])
@pytest.mark.parametrize("isolated", [False, True])
@pytest.mark.parametrize("precision", ["32-true", "16-mixed"])
@pytest.mark.parametrize("shared_study", [False, True])
def test_revision_adapter_publishes_generic_outputs_and_overlays_controls(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    isolated: bool,
    precision: str,
    shared_study: bool,
) -> None:
    captured = {"spark_released": False}

    def fake_fit(
        panel,
        origin,
        horizon,
        *,
        config,
        levels,
        out_dir,
        training_mode,
        validation_panel,
        fixed_update_budget,
        precision,
        performance_profile,
        performance_profile_dir,
        evaluation_contract_path,
        evaluation_contract_sha256,
    ):
        assert not captured["spark_released"]
        captured["config"] = config
        captured["precision"] = precision
        captured["study_evaluation"] = (evaluation_contract_path, evaluation_contract_sha256)
        captured["levels"] = levels
        captured["training_contract"] = (
            training_mode,
            validation_panel,
            fixed_update_budget,
        )
        captured["performance_profile"] = (
            performance_profile,
            performance_profile_dir,
        )
        native = pathlib.Path(out_dir)
        native.mkdir()
        start = dt.date.fromisoformat(origin)
        pq.write_table(
            pa.table(
                {
                    "series_id": ["series"] * horizon,
                    "date": pa.array(
                        [start + dt.timedelta(days=offset) for offset in range(1, horizon + 1)],
                        type=pa.date32(),
                    ),
                    "yhat": [1.0] * horizon,
                }
            ),
            native / "forecast.parquet",
        )
        captured["forecast_bytes"] = (native / "forecast.parquet").read_bytes()
        (native / "model.pt").write_bytes(b"checkpoint")
        (native / "training.jsonl").write_bytes(b'{"epoch":1}\n')
        return Forecast(
            adapter=REVISION_MODELS[variant],
            path=str(native / "forecast.parquet"),
            origin=origin,
            horizon=horizon,
            n_series=1,
            disposition={"emitted": horizon},
            frequency="D",
            target_estimand="sales_given_availability",
        ).validate()

    spark = None if isolated else object()
    if isolated:
        monkeypatch.setenv("AUTOFORECAST_ISOLATED_WORKER", "1")
    monkeypatch.setattr(subject, "fit_predict_revision", fake_fit)
    monkeypatch.setattr(
        subject,
        "_release_spark_for_torch_training",
        lambda value: captured.update(spark_released=value is spark),
    )
    out = tmp_path / "forecast"
    checkpoint = tmp_path / "generic" / "model.pt"
    metrics = tmp_path / "generic" / "epochs.jsonl"
    rolling_config = _recipe(variant)
    train_stride = 29 if variant == "v5" else 28
    rolling_config.update(encoder_days=1911, train_windows=52, train_stride=train_stride)
    result = subject.fit_predict_revision_adapter(
        Panel("unused.parquet", "id", "date", "target"),
        "2024-01-31",
        7,
        spark=spark,
        out_dir=out,
        rolling_config=rolling_config,
        expected_variant=variant,
        checkpoint_out=checkpoint,
        training_metrics_path=metrics,
        device="cuda",
        batch_size=256,
        gradient_microbatch_size=256,
        patience=7,
        event_max_support=21,
        early_stopping_absolute_min_delta=0.01,
        early_stopping_relative_min_delta=0.02,
        validation_metric="wrmsse",
        precision=precision,
        **({
            "training_mode": "fixed_validation",
            "validation_data_path": str(tmp_path / "validation"),
            "evaluation_contract_path": str(tmp_path / "evaluation.json"),
            "evaluation_contract_sha256": "a" * 64,
        } if shared_study else {}),
    )
    config = captured["config"]
    assert config.variant == variant
    assert config.encoder_days == 1911
    assert config.train_windows == 52 and config.train_stride == train_stride
    assert config.observation_contract == "masked_observed_counts_global_left_padding_v1"
    assert config.device == "cuda" and config.batch_size == 256 and config.patience == 7
    assert config.event_max_support == 21
    assert config.early_stopping_absolute_min_delta == 0.01
    assert config.early_stopping_relative_min_delta == 0.02
    assert config.selection_metric == "wrmsse"
    assert captured["precision"] == precision
    from autoforecast.benchmarks.m5 import M5_LEVELS

    assert captured["levels"] == (() if shared_study else M5_LEVELS)
    if shared_study:
        mode, validation, budget = captured["training_contract"]
        assert mode == "fixed_validation" and budget is None
        assert validation.path == str(tmp_path / "validation")
        assert captured["study_evaluation"] == (str(tmp_path / "evaluation.json"), "a" * 64)
    else:
        assert captured["training_contract"] == ("rolling_backtest", None, None)
        assert captured["study_evaluation"] == (None, None)
    assert captured["performance_profile"] == (None, None)
    assert result.path == str(out)
    assert (out / "part-00000.parquet").read_bytes() == captured["forecast_bytes"]
    assert checkpoint.read_bytes() == b"checkpoint"
    assert metrics.read_bytes() == b'{"epoch":1}\n'
    assert (tmp_path / "forecast-revision-evidence" / "model.pt").is_file()


@pytest.mark.parametrize("variant", ["v2", "v3", "v5"])
@pytest.mark.parametrize(
    "invalid", ["mapping", "variant", "events", "device", "microbatch", "delta", "metric"]
)
def test_revision_adapter_rejects_invalid_controls_before_native_training(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    invalid: str,
) -> None:
    called = False

    def unexpected(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("native training must not start")

    monkeypatch.setattr(subject, "fit_predict_revision", unexpected)
    values = {
        "rolling_config": _recipe(variant),
        "expected_variant": variant,
        "events": (),
        "device": "cpu",
        "batch_size": 32,
        "gradient_microbatch_size": 32,
        "early_stopping_absolute_min_delta": 0.0,
    }
    if invalid == "mapping":
        values["rolling_config"] = []
    elif invalid == "variant":
        values["expected_variant"] = "v3" if variant == "v2" else "v2"
    elif invalid == "events":
        values["events"] = (("event", "2024-01-01", None),)
    elif invalid == "device":
        values["device"] = "mps"
    elif invalid == "microbatch":
        values["gradient_microbatch_size"] = 16
    elif invalid == "metric":
        values["validation_metric"] = "mae"
    else:
        values["early_stopping_absolute_min_delta"] = float("nan")
    with pytest.raises((TypeError, ValueError)):
        subject.fit_predict_revision_adapter(
            Panel("unused.parquet", "id", "date", "target"),
            "2024-01-31",
            7,
            spark=object(),
            out_dir=tmp_path / "forecast",
            checkpoint_out=tmp_path / "model.pt",
            training_metrics_path=tmp_path / "epochs.jsonl",
            **values,
        )
    assert not called


@pytest.mark.parametrize("variant", ["v2", "v3", "v5"])
def test_revision_adapter_passes_validated_profile_to_shared_trainer(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
) -> None:
    from autoforecast.torch_profile_contract import TorchProfileConfig

    captured = {}

    def stop_after_capture(*args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("captured")

    monkeypatch.setattr(subject, "fit_predict_revision", stop_after_capture)
    profile = TorchProfileConfig().as_record()
    with pytest.raises(RuntimeError, match="captured"):
        subject.fit_predict_revision_adapter(
            Panel("unused.parquet", "id", "date", "target"),
            "2024-01-31",
            7,
            spark=object(),
            out_dir=tmp_path / "forecast",
            rolling_config=_recipe(variant),
            expected_variant=variant,
            checkpoint_out=tmp_path / "model.pt",
            training_metrics_path=tmp_path / "epochs.jsonl",
            training_log_dir=tmp_path / "telemetry",
            performance_profile=profile,
            device="cpu",
            batch_size=32,
            gradient_microbatch_size=32,
        )
    assert captured["performance_profile"] == profile
    assert captured["performance_profile_dir"] == str(
        (tmp_path / "telemetry" / "performance-profile").resolve()
    )


@pytest.mark.parametrize(
    "profile,log_dir",
    [
        ({"schema": "unknown"}, pathlib.Path("telemetry")),
        (None, None),
    ],
)
def test_revision_adapter_rejects_invalid_or_incomplete_profile_request(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    profile,
    log_dir,
) -> None:
    from autoforecast.torch_profile_contract import TorchProfileConfig

    if profile is None:
        profile = {**TorchProfileConfig().as_record()}
        log_dir = None
    called = False

    def unexpected(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(subject, "fit_predict_revision", unexpected)
    with pytest.raises(ValueError):
        subject.fit_predict_revision_adapter(
            Panel("unused.parquet", "id", "date", "target"),
            "2024-01-31",
            7,
            spark=object(),
            out_dir=tmp_path / "forecast",
            rolling_config=_recipe("v2"),
            expected_variant="v2",
            checkpoint_out=tmp_path / "model.pt",
            training_metrics_path=tmp_path / "epochs.jsonl",
            training_log_dir=(tmp_path / log_dir) if log_dir is not None else None,
            performance_profile=profile,
            device="cpu",
            batch_size=32,
            gradient_microbatch_size=32,
        )
    assert not called
