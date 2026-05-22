"""Boundary checks for the optional TFT adapter; no model download is required."""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
import pathlib
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from autoforecast.adapters import tft, tft_q
from autoforecast.adapters.tft import fit_predict
from autoforecast.adapters.tft_q import (
    _early_stopping_callback,
    _validate_options,
    required_history,
)
from autoforecast.panel import Panel


def test_mixed_precision_tft_class_retries_same_batch_and_restores_rng() -> None:
    from autoforecast.tft_precision import mixed_precision_tft_class

    counter = {"count": 0, "retries": 0}
    draws = []

    class Base:
        def __init__(self):
            self.training_step_outputs = []
            self.logged = []

        def log(self, name, value):
            self.logged.append((name, value))

        def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
            draws.append(torch.rand(()).item())
            self.training_step_outputs.append(draws[-1])
            self.log("train_loss", draws[-1])
            if len(draws) == 2:
                optimizer.step()

    torch.manual_seed(123)
    model = mixed_precision_tft_class(Base, counter, "cpu")()
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(1.0))])
    model.optimizer_step(0, 0, optimizer, None)
    assert draws[0] == draws[1]
    assert model.training_step_outputs == [draws[1]]
    assert model.logged == [("train_loss", draws[1])]
    assert counter == {"count": 1, "retries": 1}


def test_mixed_precision_tft_class_raises_after_24_skips() -> None:
    from autoforecast.tft_precision import mixed_precision_tft_class

    counter = {"count": 0, "retries": 0}

    class Base:
        def __init__(self):
            self.training_step_outputs = []

        def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
            self.training_step_outputs.append(1)

    model = mixed_precision_tft_class(Base, counter, "cpu")()
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(1.0))])
    with pytest.raises(FloatingPointError, match="24 tries"):
        model.optimizer_step(0, 0, optimizer, None)
    assert counter["retries"] == 24
    assert model.training_step_outputs == []


def test_mixed_precision_tft_class_backs_off_when_mps_scaler_does_not() -> None:
    from autoforecast.tft_precision import mixed_precision_tft_class

    counter = {"count": 0, "retries": 0}
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter])

    class Scaler:
        scale = 65536.0

        def get_scale(self):
            return self.scale

        def update(self, *, new_scale):
            self.scale = new_scale

    scaler = Scaler()

    class Base:
        def __init__(self):
            self.training_step_outputs = []
            self.trainer = SimpleNamespace(
                precision_plugin=SimpleNamespace(scaler=scaler), optimizers=[optimizer]
            )
            self.calls = 0

        def parameters(self):
            return [parameter]

        def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
            self.calls += 1
            self.training_step_outputs.append(self.calls)
            if self.calls == 1:
                parameter.grad = torch.tensor(float("inf"))
            else:
                parameter.grad = torch.tensor(1.0)
                optimizer.step()

    model = mixed_precision_tft_class(Base, counter, "cpu")()
    model.optimizer_step(0, 0, optimizer, None)
    assert scaler.scale == 32768.0
    assert counter == {"count": 1, "retries": 1}
    assert model.training_step_outputs == [2]


def test_clip_gradients_retries_only_real_component_overflow() -> None:
    from autoforecast.amp_gradients import clip_gradients

    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    parameter.grad = torch.tensor([float("inf")])
    _, overflow = clip_gradients([parameter], 1.0, allow_amp_overflow=True)
    assert overflow
    assert torch.isinf(parameter.grad).all()
    with pytest.raises(FloatingPointError, match="nonfinite components"):
        clip_gradients([parameter], 1.0, allow_amp_overflow=False)


def test_clip_gradients_avoids_finite_component_norm_overflow() -> None:
    from autoforecast.amp_gradients import clip_gradients

    first = torch.nn.Parameter(torch.tensor([1.0]))
    second = torch.nn.Parameter(torch.tensor([1.0]))
    first.grad = torch.tensor([1e30])
    second.grad = torch.tensor([1e30])
    norm, overflow = clip_gradients([first, second], 1.0, allow_amp_overflow=True)
    assert not overflow and torch.isfinite(norm)
    assert norm.item() == pytest.approx(2**0.5 * 1e30, rel=1e-6)
    assert first.grad.item() == pytest.approx(2**-0.5, rel=1e-6)
    assert second.grad.item() == pytest.approx(2**-0.5, rel=1e-6)

    half = torch.nn.Parameter(torch.ones(1024, dtype=torch.float16))
    half.grad = torch.full_like(half, 60_000)
    norm, overflow = clip_gradients([half], 1.0, allow_amp_overflow=True)
    assert not overflow and torch.isfinite(norm)
    assert half.grad.float().norm().item() == pytest.approx(1.0, rel=0.01)


def test_clip_gradients_uses_one_aggregate_component_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autoforecast.amp_gradients import clip_gradients

    parameters = [torch.nn.Parameter(torch.ones(1)) for _ in range(100)]
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)

    def reject_tensor_truth(_tensor: torch.Tensor) -> bool:
        raise AssertionError("per-parameter tensor-to-host truth check")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "__bool__", reject_tensor_truth)
        norm, overflow = clip_gradients(parameters, 1.0, allow_amp_overflow=True)
    assert not overflow
    assert norm.item() == pytest.approx(10.0)


def test_fp32_count_output_head_keeps_decoder_projection_full_precision() -> None:
    from autoforecast.tft_precision import fp32_count_output_head

    model = SimpleNamespace(output_layer=torch.nn.Linear(2, 3))
    expected = model.output_layer.state_dict()
    fp32_count_output_head(model)
    for name, value in expected.items():
        torch.testing.assert_close(model.output_layer.state_dict()[name], value)
    with torch.autocast("cpu", dtype=torch.float16):
        result = model.output_layer(torch.ones(2, dtype=torch.float16))
    assert result.dtype == torch.float32
    result.sum().backward()
    assert torch.isfinite(model.output_layer.weight.grad).all()


@pytest.mark.parametrize(
    ("cuda", "mps", "device", "precision", "expected"),
    [
        (False, False, None, "32-true", "cpu"),
        (False, True, None, "16-mixed", "mps"),
        (True, True, None, "16-mixed", "cuda"),
        (True, True, "mps", "16-mixed", "mps"),
        (True, True, "cpu", "32-true", "cpu"),
    ],
)
def test_legacy_tft_resolves_only_automatic_device(
    cuda: bool, mps: bool, device: str | None, precision: str, expected: str
) -> None:
    fake = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )
    assert tft._select_accelerator(fake, precision=precision, device=device) == expected


@pytest.mark.parametrize(
    ("cuda", "mps", "device", "precision"),
    [
        (False, True, "cuda", "16-mixed"),
        (True, False, "mps", "16-mixed"),
        (True, True, "cpu", "16-mixed"),
        (False, False, None, "16-mixed"),
    ],
)
def test_legacy_tft_refuses_unavailable_or_cpu_fp16_device(
    cuda: bool, mps: bool, device: str | None, precision: str
) -> None:
    fake = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )
    with pytest.raises(ValueError):
        tft._select_accelerator(fake, precision=precision, device=device)


def test_tft_mixed_clip_handles_finite_and_component_overflow() -> None:
    from autoforecast.tft_precision import finite_gradient_clip_callback

    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    module = torch.nn.Linear(1, 1)
    module.weight = parameter
    parameter.grad = torch.tensor([2.0])
    callback = finite_gradient_clip_callback(precision="16-mixed")
    callback.on_before_optimizer_step(None, module, None)
    assert parameter.grad.item() == pytest.approx(1.0)
    parameter.grad = torch.tensor([float("inf")])
    callback.on_before_optimizer_step(None, module, None)
    assert not torch.isfinite(parameter.grad).all()


def test_tft_mixed_retries_when_scaler_backs_off_with_finite_visible_gradients() -> None:
    from autoforecast.tft_precision import mixed_precision_tft_class

    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    counter = {"count": 0, "retries": 0}

    class Scaler:
        scale = 65536.0

        def get_scale(self):
            return self.scale

        def update(self, *, new_scale):
            self.scale = new_scale

    scaler = Scaler()

    class Base:
        def __init__(self):
            self.training_step_outputs = []
            self.trainer = SimpleNamespace(precision_plugin=SimpleNamespace(scaler=scaler))
            self.calls = 0

        def parameters(self):
            return [parameter]

        def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
            self.calls += 1
            self.training_step_outputs.append(self.calls)
            parameter.grad = torch.tensor(1.0)
            if self.calls == 1:
                scaler.update(new_scale=scaler.get_scale() * 0.5)
            else:
                optimizer.step()

    model = mixed_precision_tft_class(Base, counter, "cpu")()
    model.optimizer_step(0, 0, optimizer, None)
    assert counter == {"count": 1, "retries": 1}
    assert model.training_step_outputs == [2]


def test_daily_adapter_refuses_weekly_panel_before_optional_runtime_or_spark_io() -> None:
    class ReadSentinel:
        @property
        def read(self):
            raise AssertionError("weekly panel reached TFT/Spark")

    weekly = Panel("/not-read", "id", "date", "qty", frequency="W")
    with pytest.raises(ValueError, match="daily"):
        fit_predict(
            weekly,
            "2024-01-01",
            2,
            spark=ReadSentinel(),
            out_dir="/not-written",
        )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
def test_new_legacy_tft_mixed_precision_fit(tmp_path, monkeypatch) -> None:
    start = dt.date(2024, 1, 1)
    rows = [
        {
            "series_id": sid,
            "d": pd.Timestamp(start + dt.timedelta(days=day)),
            "y": float((day + index) % 5),
            "time_idx": day,
            "dow": (start + dt.timedelta(days=day)).weekday(),
            "dom": (start + dt.timedelta(days=day)).day,
            "month": (start + dt.timedelta(days=day)).month,
        }
        for index, sid in enumerate(("a", "b")) for day in range(202)
    ]
    monkeypatch.setattr(tft, "_training_frame", lambda *args: (pd.DataFrame(rows), 2, 0))
    checkpoint = tmp_path / "model.pt"
    forecast = tft.fit_predict(
        Panel("/unused", "series_id", "d", "y"),
        (start + dt.timedelta(days=201)).isoformat(), 7,
        spark=mock.Mock(), out_dir=tmp_path / "forecast",
        max_epochs=1, batch_size=2, seed=0, hidden_size=8,
        attention_head_size=2, dropout=0.1, learning_rate=0.001,
        precision="16-mixed", checkpoint_out=checkpoint,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert forecast.n_series == 2
    assert payload["hyperparameters"]["precision"] == "16-mixed"


def test_tft_history_requirement_includes_the_longest_target_lag_and_horizon() -> None:
    assert required_history(min_encoder_length=7, lag_long_days=42, horizon=28) == 77


def test_tft_default_checkpoint_is_scoped_to_output(tmp_path) -> None:
    first = tft_q._checkpoint_path(tmp_path / "v6.1", None)
    second = tft_q._checkpoint_path(tmp_path / "v6.2", None)
    assert first != second
    assert first.parent != second.parent
    assert not first.is_relative_to(tmp_path / "v6.1")
    assert tft_q._checkpoint_path(tmp_path / "v6.1", tmp_path / "explicit.pt") == (
        tmp_path / "explicit.pt"
    )


@pytest.mark.parametrize("occupied", ["directory", "forecast_symlink", "checkpoint_symlink"])
def test_tft_refuses_existing_destinations_before_loading_data(tmp_path, occupied) -> None:
    destination = tmp_path / "forecast"
    checkpoint = tmp_path / "selected.pt"
    if occupied == "directory":
        destination.mkdir()
    elif occupied == "forecast_symlink":
        destination.symlink_to(tmp_path / "missing-forecast", target_is_directory=True)
    else:
        checkpoint.symlink_to(tmp_path / "missing-checkpoint")
    with pytest.raises(ValueError, match="destinations must not already exist"):
        tft_q.fit_predict(
            Panel("/not-read", "id", "date", "qty", known_future=("price",),
                  price_col="price"),
            "2024-04-09", 28, spark=None, out_dir=destination,
            checkpoint_out=checkpoint,
            max_epochs=2, batch_size=2, early_stopping_patience=10,
            early_stopping_min_delta=1e-6, device="cpu", seed=0,
            hidden_size=8, hidden_continuous_size=4, attention_head_size=2,
            dropout=0.1, learning_rate=0.001, max_encoder_length=30,
            min_encoder_length=7, lag_short_days=2, lag_long_days=14,
            output_size=4,
        )


@pytest.mark.parametrize("case", [
    "same", "checkpoint_child", "forecast_child",
    "forecast_ancestor_symlink", "checkpoint_ancestor_symlink",
])
def test_tft_refuses_overlapping_or_redirected_destinations_before_loading_data(
    tmp_path, case: str
) -> None:
    destination = tmp_path / "forecast"
    checkpoint = tmp_path / "selected.pt"
    if case == "same":
        checkpoint = destination
    elif case == "checkpoint_child":
        checkpoint = destination / "model.pt"
    elif case == "forecast_child":
        checkpoint = tmp_path / "checkpoint-root"
        destination = checkpoint / "forecast"
    elif case == "forecast_ancestor_symlink":
        (tmp_path / "real").mkdir()
        (tmp_path / "link").symlink_to(tmp_path / "real", target_is_directory=True)
        destination = tmp_path / "link" / "forecast"
    else:
        (tmp_path / "link").symlink_to(tmp_path / "missing", target_is_directory=True)
        checkpoint = tmp_path / "link" / "selected.pt"
    with pytest.raises(ValueError, match="overlap|ancestor"):
        tft_q.fit_predict(
            Panel("/not-read", "id", "date", "qty", known_future=("price",),
                  price_col="price"),
            "2024-04-09", 28, spark=None, out_dir=destination,
            checkpoint_out=checkpoint,
            max_epochs=2, batch_size=2, early_stopping_patience=10,
            early_stopping_min_delta=1e-6, device="cpu", seed=0,
            hidden_size=8, hidden_continuous_size=4, attention_head_size=2,
            dropout=0.1, learning_rate=0.001, max_encoder_length=30,
            min_encoder_length=7, lag_short_days=2, lag_long_days=14,
            output_size=4,
        )


def test_tft_refuses_overlap_across_darwin_tmp_alias_before_loading_data(tmp_path) -> None:
    if pathlib.Path("/tmp").resolve() != pathlib.Path("/private/tmp"):
        pytest.skip("Darwin /tmp alias is not present")
    name = "autoforecast-v6-topology-" + tmp_path.name
    destination = pathlib.Path("/tmp") / name / "forecast"
    checkpoint = pathlib.Path("/private/tmp") / name / "forecast" / "model.pt"
    with pytest.raises(ValueError, match="overlap"):
        tft_q.fit_predict(
            Panel("/not-read", "id", "date", "qty", known_future=("price",),
                  price_col="price"),
            "2024-04-09", 28, spark=None, out_dir=destination,
            checkpoint_out=checkpoint,
            max_epochs=2, batch_size=2, early_stopping_patience=10,
            early_stopping_min_delta=1e-6, device="cpu", seed=0,
            hidden_size=8, hidden_continuous_size=4,
            attention_head_size=2, dropout=0.1, learning_rate=0.001,
            max_encoder_length=30, min_encoder_length=7,
            lag_short_days=2, lag_long_days=14, output_size=4,
        )


def test_tft_refuses_case_alias_overlap_on_case_insensitive_macos(tmp_path) -> None:
    if tft_q.sys.platform != "darwin":
        pytest.skip("macOS path aliases are not present")
    probe = tmp_path / "CaseProbe"
    probe.mkdir()
    if not (tmp_path / "caseprobe").exists():
        pytest.skip("this macOS volume is case-sensitive")
    destination = tmp_path / "FutureForecast"
    checkpoint = tmp_path / "futureforecast" / "model.pt"
    with pytest.raises(ValueError, match="overlap"):
        tft_q._validate_artifact_destinations(destination, checkpoint)
    if pathlib.Path("/Tmp").is_symlink():
        tft_q._validate_artifact_destinations(
            pathlib.Path("/Tmp") / tmp_path.name / "forecast",
            pathlib.Path("/Tmp") / tmp_path.name / "model.pt",
        )


def test_tft_refuses_overlap_across_darwin_users_firmlink(tmp_path) -> None:
    users = pathlib.Path("/Users")
    data_users = pathlib.Path("/System/Volumes/Data/Users")
    if not data_users.exists() or not users.samefile(data_users):
        pytest.skip("Darwin /Users firmlink is unavailable")
    root = pathlib.Path(__file__).resolve().parents[2]
    destination = root / ("v6-firmlink-probe-" + tmp_path.name)
    checkpoint = pathlib.Path("/System/Volumes/Data") / destination.relative_to("/") / "model.pt"
    with pytest.raises(ValueError, match="overlap"):
        tft_q._validate_artifact_destinations(destination, checkpoint)


def test_tft_forecast_publish_leaves_no_partial_destination_on_write_failure(
    tmp_path, monkeypatch
) -> None:
    def failed_write(*_args, **_kwargs):
        raise OSError("simulated Parquet write failure")

    monkeypatch.setattr(tft_q.pq, "write_table", failed_write)
    destination = tmp_path / "forecast"
    with pytest.raises(OSError, match="simulated Parquet write failure"):
        tft_q._write_forecast([("a", dt.date(2024, 4, 10), 1.0)], destination)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failed_step", ["forecast", "checkpoint"])
def test_tft_artifact_publication_rolls_back_and_retries(
    tmp_path, monkeypatch, failed_step: str
) -> None:
    checkpoint = tmp_path / "selected" / "model.pt"
    destination = tmp_path / "forecast"
    payload = {"weight": torch.tensor([1.0])}
    rows = [("a", dt.date(2024, 4, 10), 1.0)]
    write_forecast = tft_q._write_forecast

    with monkeypatch.context() as patch:
        if failed_step == "forecast":
            def fail_after_forecast(rows, output_dir):
                write_forecast(rows, output_dir)
                raise OSError("injected publication failure")

            patch.setattr(tft_q, "_write_forecast", fail_after_forecast)
        else:
            def fail_checkpoint(*_args):
                raise OSError("injected publication failure")

            patch.setattr(tft_q.os, "link", fail_checkpoint)
        with pytest.raises(OSError, match="injected publication failure"):
            tft_q._publish_tft_artifacts(
                payload, rows=rows, checkpoint=checkpoint, output_dir=destination
            )
    assert not checkpoint.exists()
    assert not destination.exists()

    tft_q._publish_tft_artifacts(
        payload, rows=rows, checkpoint=checkpoint, output_dir=destination
    )
    assert torch.load(checkpoint, weights_only=True)["weight"].item() == 1.0
    assert pq.read_table(destination).num_rows == 1


def test_tft_retry_recovers_only_its_own_marked_incomplete_forecast(tmp_path) -> None:
    checkpoint = tmp_path / "selected" / "model.pt"
    destination = tmp_path / "forecast"
    rows = [("a", dt.date(2024, 4, 10), 1.0)]
    tft_q._write_forecast(rows, destination)
    marker = destination / ".v6-pending"
    marker.write_text("/unrelated/model.pt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="destinations must not already exist"):
        tft_q._publish_tft_artifacts(
            {"weight": torch.tensor([1.0])},
            rows=rows, checkpoint=checkpoint, output_dir=destination,
        )
    assert destination.is_dir() and not checkpoint.exists()

    marker.write_text(str(checkpoint.absolute()) + "\n", encoding="utf-8")
    (destination / ".DS_Store").write_bytes(b"finder metadata")
    tft_q._publish_tft_artifacts(
        {"weight": torch.tensor([1.0])},
        rows=rows, checkpoint=checkpoint, output_dir=destination,
    )
    assert not marker.exists()
    assert torch.load(checkpoint, weights_only=True)["weight"].item() == 1.0
    assert pq.read_table(destination).num_rows == 1
    abandoned = list(tmp_path.glob(".forecast-abandoned-*"))
    assert len(abandoned) == 1
    assert (abandoned[0] / ".DS_Store").read_bytes() == b"finder metadata"


def test_tft_retry_adopts_a_complete_post_link_publication(tmp_path, monkeypatch) -> None:
    checkpoint = tmp_path / "selected.pt"
    destination = tmp_path / "forecast"
    input_path = tmp_path / "input.parquet"
    input_path.write_bytes(b"immutable source panel")
    origin = dt.date(2024, 4, 9)
    rows = [("a", origin + dt.timedelta(days=step), 1.0) for step in range(1, 29)]
    panel = Panel(
        str(input_path), "id", "date", "qty", known_future=("price",), price_col="price"
    )
    options = dict(
        panel=panel, origin=origin.isoformat(), horizon=28, spark=None,
        out_dir=destination, checkpoint_out=checkpoint, max_epochs=2, batch_size=2,
        early_stopping_patience=10, early_stopping_min_delta=1e-6,
        device="cpu", seed=0, hidden_size=8, hidden_continuous_size=4,
        attention_head_size=2, dropout=0.1, learning_rate=0.001,
        max_encoder_length=30, min_encoder_length=7,
        lag_short_days=2, lag_long_days=14, output_size=4,
        training_mode="fixed_refit", fixed_update_budget=2,
    )
    run_identity = {
        "panel": dataclasses.asdict(panel),
        **{key: options[key] for key in (
            "origin", "horizon", "seed", "batch_size", "early_stopping_patience",
            "early_stopping_min_delta", "hidden_size", "hidden_continuous_size",
            "attention_head_size", "dropout", "learning_rate", "max_encoder_length",
            "min_encoder_length", "lag_short_days", "lag_long_days", "output_size",
            "training_mode", "fixed_update_budget", "max_epochs",
        )},
        "requested_device": "cpu", "resolved_device": "cpu",
        "loss_kind": "quantile", "precision": "32-true",
        "panel_sha256": tft_q.directory_digest(input_path, suffixes=(".parquet",)),
    }
    payload = {
        "schema": "autoforecast.tft-v6-inference-state.v2",
        "state_dict": {"weight": torch.tensor([1.0])},
        "run_identity": run_identity,
        "training": {
            "maximum_epochs": 2, "n_series": 1, "steps_per_epoch": 1,
            "selected_updates": 2,
            "refit_optimizer_steps": 2, "precision": "32-true",
            "validation_origin": None,
            "training_contract": {"mode": "fixed_refit", "fixed_update_budget": 2},
        },
        "profile": {"loss_kind": "quantile"},
    }
    unlink = pathlib.Path.unlink

    def fail_marker(self, *args, **kwargs):
        if self.name == ".v6-pending":
            raise OSError("injected marker removal failure")
        return unlink(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(pathlib.Path, "unlink", fail_marker)
        with pytest.raises(OSError, match="marker removal failure"):
            tft_q._publish_tft_artifacts(
                payload, rows=rows, checkpoint=checkpoint, output_dir=destination
            )
    assert checkpoint.is_file() and (destination / ".v6-pending").is_file()
    (destination / ".DS_Store").write_bytes(b"finder metadata")
    input_bytes = input_path.read_bytes()
    input_path.write_bytes(b"changed source panel")
    with pytest.raises(ValueError, match="differs from the requested run"):
        tft_q.fit_predict(**options)
    assert (destination / ".v6-pending").is_file()
    input_path.write_bytes(input_bytes)
    part = destination / "part-00000.parquet"
    forecast_bytes = part.read_bytes()
    original = pq.read_table(part)
    for altered in (
        original.set_column(2, "yhat", pa.array([999.0] * 28, type=pa.float64())),
        original.set_column(0, "series_id", pa.array(["wrong"] * 28)),
    ):
        pq.write_table(altered, part)
        with pytest.raises(ValueError, match="differs from the requested run"):
            tft_q.fit_predict(**options)
        assert (destination / ".v6-pending").is_file()
        part.write_bytes(forecast_bytes)
    checkpoint_bytes = checkpoint.read_bytes()
    torch.save({**payload, "forecast_sha256": "0" * 64}, checkpoint)
    with pytest.raises(ValueError, match="differs from the requested run"):
        tft_q.fit_predict(**options)
    assert (destination / ".v6-pending").is_file()
    checkpoint.write_bytes(checkpoint_bytes)
    with pytest.raises(ValueError, match="differs from the requested run"):
        tft_q.fit_predict(**{**options, "seed": 1})
    assert (destination / ".v6-pending").is_file()
    forecast = tft_q.fit_predict(**options)
    assert forecast.n_series == 1 and forecast.disposition == {"emitted": 28}
    assert not (destination / ".v6-pending").exists()
    assert checkpoint.is_file() and pq.read_table(destination).num_rows == 28
    assert tft_q.fit_predict(**options) == forecast
    with pytest.raises(ValueError, match="differs from the requested run"):
        tft_q.fit_predict(**{**options, "seed": 1})


def test_tft_rejects_an_invalid_early_stopping_delta() -> None:
    with pytest.raises(ValueError, match="early_stopping_min_delta"):
        _validate_options(
            max_epochs=2,
            batch_size=32,
            early_stopping_patience=10,
            early_stopping_min_delta=-0.1,
            device="mps",
            hidden_size=50,
            hidden_continuous_size=8,
            attention_head_size=2,
            dropout=0.25,
            learning_rate=0.015,
            max_encoder_length=364,
            min_encoder_length=7,
            lag_short_days=7,
            lag_long_days=42,
            output_size=4,
            horizon=28,
        )


@pytest.mark.parametrize(
    ("horizon", "patience"), ((14, 10), (28, 1))
)
def test_tft_adapter_rejects_non_governing_horizon_or_patience(
    horizon: int, patience: int
) -> None:
    with pytest.raises(ValueError, match="28-day horizon and validation patience 10"):
        _validate_options(
            max_epochs=2, batch_size=2, early_stopping_patience=patience,
            early_stopping_min_delta=1e-6, device="cpu", hidden_size=8,
            hidden_continuous_size=4, attention_head_size=2, dropout=0.1,
            learning_rate=0.001, max_encoder_length=30, min_encoder_length=7,
            lag_short_days=2, lag_long_days=14, output_size=4, horizon=horizon,
        )


def test_tft_uses_validation_quantile_loss_early_stop_rule() -> None:
    callback = _early_stopping_callback(patience=10, min_delta=1e-6)
    assert callback.monitor == "val_loss"
    assert callback.mode == "min"
    assert callback.patience == 10
    assert callback.min_delta == -1e-6
    assert callback._check_on_train_epoch_end is False


def test_tft_epoch_log_uses_weighted_training_epoch_mean_not_last_batch() -> None:
    callback = tft_q._EpochLog()
    trainer = SimpleNamespace(
        num_val_batches=[2],
        sanity_checking=False,
        callback_metrics={"train_loss": torch.tensor(9.0), "val_loss": torch.tensor(5.0)},
        current_epoch=0,
        global_step=2,
    )
    model = SimpleNamespace()
    callback.on_train_epoch_start(trainer, model)
    callback.on_train_batch_end(
        trainer, model, {"loss": torch.tensor(1.0), "n_samples": 2}, None, 0
    )
    callback.on_train_batch_end(
        trainer, model, {"loss": torch.tensor(9.0), "n_samples": 1}, None, 1
    )
    callback.on_train_epoch_end(trainer, model)
    callback.on_validation_epoch_end(trainer, model)
    assert callback.records[0]["train_loss"] == pytest.approx(11 / 3)
    assert callback.records[0]["val_loss"] == 5.0
    assert callback.records[0]["val_loss_dtype"] == "torch.float32"


def test_tft_epoch_log_rejects_nonfinite_validation_loss() -> None:
    callback = tft_q._EpochLog()
    trainer = SimpleNamespace(
        num_val_batches=[1], sanity_checking=False,
        callback_metrics={"val_loss": torch.tensor(float("inf"))},
        current_epoch=0, global_step=1,
    )
    callback.on_train_epoch_start(trainer, None)
    callback.on_train_batch_end(
        trainer, None, {"loss": torch.tensor(1.0), "n_samples": 1}, None, 0
    )
    with pytest.raises(FloatingPointError, match="nonfinite validation loss"):
        callback.on_validation_epoch_end(trainer, None)


def test_tft_gradient_clip_rejects_large_or_nonfinite_norm_before_adam() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    module = SimpleNamespace(parameters=lambda: [parameter])
    optimizer = torch.optim.Adam([parameter])
    callback = tft_q._FiniteGradientClip()
    parameter.grad = torch.tensor([1e21])
    callback.on_before_optimizer_step(None, module, optimizer)
    assert parameter.grad.item() == pytest.approx(1.0)
    parameter.grad = torch.tensor([float("inf")])
    with pytest.raises(FloatingPointError, match="gradient norm is nonfinite"):
        callback.on_before_optimizer_step(None, module, optimizer)
    assert parameter.item() == 1.0
    assert not optimizer.state
    parameter.grad = torch.tensor([2.0])
    callback.on_before_optimizer_step(None, module, optimizer)
    assert parameter.grad.item() == pytest.approx(1.0)


def test_tft_hmfnet_hurdle_loss_matches_exact_count_cases_and_has_gradients() -> None:
    from forecast.models.mft.sft import ShiftedHurdlePrediction

    from autoforecast.adapters.tft_q_loss import HMFNetHurdleLoss

    output = torch.tensor([[[0.0, math.log(3), math.log(2)],
                            [0.5, math.log(3), math.log(2)]]], requires_grad=True)
    target = torch.tensor([[0.0, 2.0]])
    mean_scale = torch.tensor([[1.0, 0.25]])
    metric = HMFNetHurdleLoss()
    cells = metric.loss(output, target, mean_scale)
    reference = ShiftedHurdlePrediction(
        logit=output[..., 0],
        excess=output[..., 1].double().exp(),
        dispersion=output[..., 2].double().exp(),
    )
    torch.testing.assert_close(
        cells,
        reference.nll(target, torch.ones_like(target, dtype=torch.bool))
        + 0.2 * mean_scale * (reference.mean() - target).square(),
    )
    torch.testing.assert_close(metric(output, (target, mean_scale)), cells.mean())
    torch.testing.assert_close(metric.to_prediction(output), reference.mean())
    cells.mean().backward()
    assert output.grad is not None and torch.isfinite(output.grad).all()
    assert output.grad.abs().sum() > 0


def test_tft_hmfnet_hurdle_reduction_matches_hmfnet_uniform_cell_normalization() -> None:
    from forecast.models.mft.sft import ShiftedHurdlePrediction

    from autoforecast.adapters.tft_q_loss import HMFNetHurdleLoss

    batch, horizon = 2, 28
    output = torch.zeros((batch, horizon, 3))
    output[1, :, 0] = 0.4
    output[:, :, 1] = math.log(3)
    output[:, :, 2] = math.log(2)
    target = torch.arange(horizon).repeat(batch, 1).float() % 5
    scale = torch.tensor([1 / 9, 1 / 4]).repeat_interleave(horizon).reshape(batch, horizon)
    prediction = ShiftedHurdlePrediction(
        logit=output[..., 0],
        excess=output[..., 1].double().exp(),
        dispersion=output[..., 2].double().exp(),
    )
    v3_normalization_count = batch * horizon
    v3_likelihood = prediction.nll(target, torch.ones_like(target, dtype=torch.bool)).sum()
    v3_likelihood /= v3_normalization_count
    v3_mean = (((prediction.mean() - target).square()).mean(1) * scale[:, 0]).mean()
    expected = v3_likelihood + 0.2 * v3_mean
    torch.testing.assert_close(HMFNetHurdleLoss()(output, (target, scale)), expected)


def test_tft_hmfnet_nll_guard_matches_series_level_reference(monkeypatch) -> None:
    from autoforecast.adapters.tft_q_loss import HMFNetHurdleLoss

    class RoundedPrediction:
        def __init__(self, cells):
            self.cells = cells

        def nll(self, actual, mask):
            return self.cells

        def mean(self):
            return torch.zeros_like(self.cells)

    cells = torch.tensor([[-1e-16, 1.0]], dtype=torch.float64)
    monkeypatch.setattr(
        HMFNetHurdleLoss, "_decode", staticmethod(lambda _output: RoundedPrediction(cells))
    )
    metric = HMFNetHurdleLoss()
    output = torch.zeros((1, 2, 3))
    actual = torch.zeros((1, 2))
    scale = torch.ones_like(actual)
    torch.testing.assert_close(metric.loss(output, actual, scale), cells)
    cells[:] = -1e-16
    with pytest.raises(FloatingPointError, match="negative or nonfinite"):
        metric.loss(output, actual, scale)


@pytest.mark.parametrize("batch", [1, 2, 32])
def test_tft_hmfnet_hurdle_reduction_stays_finite_for_large_batches(batch: int) -> None:
    from autoforecast.adapters.tft_q_loss import HMFNetHurdleLoss

    output = torch.zeros((batch, 28, 3), dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        output[..., 1] = 353.0
    actual = torch.zeros((batch, 28), dtype=torch.float64)
    scale = torch.ones_like(actual)
    loss = HMFNetHurdleLoss()(output, (actual, scale))
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(2.045852070816995e305, rel=1e-12)
    loss.backward()
    assert torch.isfinite(output.grad).all()


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_tft_hmfnet_hurdle_keeps_finite_float64_loss_until_reduction(device: str) -> None:
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")
    from autoforecast.adapters.tft_q_loss import HMFNetHurdleLoss

    output = torch.zeros((1, 28, 3), device=device, requires_grad=True)
    with torch.no_grad():
        output[0, 0, 1] = 46.0
    target = torch.zeros((1, 28), device=device)
    scale = torch.ones_like(target)
    metric = HMFNetHurdleLoss().to(device)
    expected = metric.loss(output, target, scale).mean()
    value = metric(output, (target, scale))
    assert value.device.type == "cpu" and value.dtype == torch.float64
    assert torch.isfinite(value)
    torch.testing.assert_close(value, expected)
    value.backward()
    assert output.grad is not None and torch.isfinite(output.grad).all()


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_tft_hmfnet_hurdle_rejects_nonfinite_objective(device: str) -> None:
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")
    from autoforecast.adapters.tft_q_loss import HMFNetHurdleLoss

    output = torch.zeros((1, 28, 3), device=device)
    output[:, :, 1] = 356.0
    target = torch.zeros((1, 28), device=device)
    with pytest.raises(FloatingPointError, match="nonfinite"):
        HMFNetHurdleLoss().to(device)(output, (target, torch.ones_like(target)))


def test_tft_hmfnet_mean_scale_uses_only_sales_before_each_decoder_origin() -> None:
    rows = [
        {"series_id": series_id, "time_idx": day, "y": value,
         "history_available": 1.0}
        for series_id in ("a", "b")
        for day in range(84)
        for value in [2.0 if series_id == "a" and day < 28 else 0.0]
    ]
    frame = pd.DataFrame(rows)
    tft_q._assign_hmfnet_mean_scales(frame, decoder_ends=(55, 83), horizon=28)
    early = frame[frame["time_idx"].between(28, 55)]
    late = frame[frame["time_idx"].between(56, 83)]
    assert early.loc[early["series_id"] == "a", "mean_loss_scale"].eq(1 / 9).all()
    assert late.loc[late["series_id"] == "a", "mean_loss_scale"].eq(1).all()
    assert early.loc[early["series_id"] == "b", "mean_loss_scale"].eq(1).all()


def test_tft_hmfnet_mean_scale_accepts_categorical_ids_without_unit_scale() -> None:
    frame = pd.DataFrame([
        {"series_id": series_id, "time_idx": day, "y": sales,
         "history_available": 1.0}
        for series_id, sales in (("a", 1.0), ("b", 2.0))
        for day in range(84)
    ])
    frame["series_id"] = frame["series_id"].astype("category")
    tft_q._assign_hmfnet_mean_scales(frame, decoder_ends=(55, 83), horizon=28)
    decoder = frame[frame["time_idx"].between(28, 55)]
    assert decoder.loc[decoder["series_id"] == "a", "mean_loss_scale"].eq(0.25).all()
    assert decoder.loc[decoder["series_id"] == "b", "mean_loss_scale"].eq(1 / 9).all()


def test_tft_cold_start_is_distinct_from_an_observed_zero() -> None:
    frame = pd.DataFrame(
        {
            "is_history": [True, True, True, False],
            "is_admitted": [False, True, True, True],
            "observed_y": [None, 0.0, 3.0, 99.0],
        }
    )
    tft_q._mark_cold_start(frame)
    assert frame["history_available"].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert frame["y"].tolist() == [0.0, 0.0, 3.0, 0.0]


def test_tft_refuses_censored_counts_in_paired_loss_comparison() -> None:
    frame = pd.DataFrame(
        {"is_history": [True, False], "observation_state": ["lower_bound", None]}
    )
    with pytest.raises(ValueError, match="requires exact or unobserved"):
        tft_q._require_uncensored_m5(frame)


def test_tft_parquet_cold_start_keeps_every_series_and_masks_unavailable_rows(tmp_path) -> None:
    origin = dt.date(2024, 4, 9)
    rows = [
        {
            "id": series_id,
            "date": origin - dt.timedelta(days=104 - day),
            "qty": None if series_id == "b" and day < 40 else float(day % 4),
            "sell_price": 1.0,
            "observation_state": "unobserved" if series_id == "b" and day < 40 else "exact",
        }
        for series_id in ("a", "b")
        for day in range(105)
    ]
    rows.extend(
        {
            "id": series_id,
            "date": origin + dt.timedelta(days=day),
            "qty": None,
            "sell_price": 1.0,
            "observation_state": None,
        }
        for series_id in ("a", "b")
        for day in range(1, 29)
    )
    path = tmp_path / "panel.parquet"
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("date", pa.date32()),
            pa.field("qty", pa.float64()),
            pa.field("sell_price", pa.float64()),
            pa.field("observation_state", pa.string()),
        ]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    panel = Panel(
        str(path), "id", "date", "qty", exposure_col="observation_state",
        known_future=("sell_price",), price_col="sell_price",
    )
    frame, _ = tft_q._tft_frame(
        panel, origin_day=origin, horizon=28, max_encoder_length=30,
        min_encoder_length=7, lag_long_days=42,
    )
    assert set(frame["series_id"]) == {"a", "b"}
    assert len(frame) == 2 * (105 + 28)
    cold = frame[(frame["series_id"] == "b") & (frame["is_history"])]
    assert cold["history_available"].eq(0).sum() == 40
    assert cold.loc[cold["history_available"] == 0, "y"].eq(0).all()
    assert frame.loc[~frame["is_history"], "y"].eq(0).all()


def test_tft_arrow_static_categoricals_keep_spark_string_semantics(tmp_path) -> None:
    origin = dt.date(2024, 4, 9)
    rows = [
        {"id": 7, "date": origin - dt.timedelta(days=104 - day),
         "qty": 1.0, "sell_price": 1.0, "store": 12}
        for day in range(105)
    ] + [
        {"id": 7, "date": origin + dt.timedelta(days=day),
         "qty": None, "sell_price": 1.0, "store": 12}
        for day in range(1, 29)
    ]
    path = tmp_path / "numeric-static.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    panel = Panel(
        str(path), "id", "date", "qty", known_future=("sell_price",),
        price_col="sell_price", static=("store",),
    )
    frame, categoricals = tft_q._tft_frame(
        panel, origin_day=origin, horizon=28, max_encoder_length=30,
        min_encoder_length=7, lag_long_days=42,
    )
    assert categoricals == ("series_id", "store")
    assert frame["series_id"].cat.categories.tolist() == ["7"]
    assert frame["store"].cat.categories.tolist() == ["12"]


def test_tft_arrow_reads_hive_partitioned_static_column(tmp_path) -> None:
    origin = dt.date(2024, 4, 9)
    rows = [
        {"id": "a", "date": origin - dt.timedelta(days=104 - day),
         "qty": 1.0, "sell_price": 1.0, "store_id": "CA_1"}
        for day in range(105)
    ] + [
        {"id": "a", "date": origin + dt.timedelta(days=day),
         "qty": None, "sell_price": 1.0, "store_id": "CA_1"}
        for day in range(1, 29)
    ]
    path = tmp_path / "partitioned"
    pq.write_to_dataset(pa.Table.from_pylist(rows), root_path=path, partition_cols=["store_id"])
    panel = Panel(
        str(path), "id", "date", "qty", known_future=("sell_price",),
        price_col="sell_price", static=("store_id",),
    )
    frame, _ = tft_q._tft_frame(
        panel, origin_day=origin, horizon=28, max_encoder_length=30,
        min_encoder_length=7, lag_long_days=42,
    )
    assert len(frame) == 133
    assert frame["store_id"].cat.categories.tolist() == ["CA_1"]


def test_tft_arrow_ignores_unused_dictionary_series_after_date_filter(tmp_path) -> None:
    origin = dt.date(2024, 4, 9)
    dates = [origin - dt.timedelta(days=104 - day) for day in range(105)]
    dates += [origin + dt.timedelta(days=day) for day in range(1, 29)]
    ids = pa.DictionaryArray.from_arrays(
        pa.array([0] * len(dates), type=pa.int8()), pa.array(["a", "unused"])
    )
    path = tmp_path / "dictionary.parquet"
    pq.write_table(pa.table({
        "id": ids,
        "date": pa.array(dates, type=pa.date32()),
        "qty": pa.array([1.0] * 105 + [None] * 28),
        "sell_price": pa.array([1.0] * len(dates)),
    }), path)
    panel = Panel(
        str(path), "id", "date", "qty", known_future=("sell_price",),
        price_col="sell_price",
    )
    frame, _ = tft_q._tft_frame(
        panel, origin_day=origin, horizon=28, max_encoder_length=30,
        min_encoder_length=7, lag_long_days=42,
    )
    assert set(frame["series_id"]) == {"a"}


def test_tft_arrow_exposure_alias_does_not_duplicate_canonical_columns(tmp_path) -> None:
    origin = dt.date(2024, 4, 9)
    rows = [
        {"id": "a", "day": origin - dt.timedelta(days=104 - day),
         "qty": 1.0, "sell_price": 1.0, "date": "exact"}
        for day in range(105)
    ] + [
        {"id": "a", "day": origin + dt.timedelta(days=day),
         "qty": None, "sell_price": 1.0, "date": None}
        for day in range(1, 29)
    ]
    path = tmp_path / "aliased-exposure.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    panel = Panel(
        str(path), "id", "day", "qty", exposure_col="date",
        known_future=("sell_price",), price_col="sell_price",
    )
    frame, _ = tft_q._tft_frame(
        panel, origin_day=origin, horizon=28, max_encoder_length=30,
        min_encoder_length=7, lag_long_days=42,
    )
    assert len(frame) == 133
    assert frame["date"].dtype == "datetime64[ns]"


@pytest.mark.parametrize(
    "loss_kind,output_size,device,precision,force_overflow",
    [("quantile", 4, "cpu", "32-true", False),
     ("v3_hurdle_nb", 3, "cpu", "32-true", False),
     ("quantile", 4, "mps", "32-true", False),
     ("v3_hurdle_nb", 3, "mps", "32-true", False),
     ("quantile", 4, "mps", "16-mixed", False),
     ("v3_hurdle_nb", 3, "mps", "16-mixed", False),
     ("quantile", 4, "mps", "16-mixed", True)],
)
def test_tft_selects_one_window_per_series_then_refits_exact_updates(
    tmp_path, monkeypatch, loss_kind, output_size, device, precision, force_overflow
) -> None:
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")
    import lightning.pytorch as pl
    from pytorch_forecasting import TemporalFusionTransformer

    original_from_dataset = TemporalFusionTransformer.from_dataset.__func__
    original_seed_everything = pl.seed_everything
    seeds: list[tuple[int, bool]] = []

    def checked_seed(seed, *, workers):
        seeds.append((seed, workers))
        return original_seed_everything(seed, workers=workers)

    monkeypatch.setattr(pl, "seed_everything", checked_seed)

    def checked_from_dataset(cls, dataset, **kwargs):
        assert "history_available" in dataset.time_varying_unknown_reals
        assert "history_available" not in dataset.time_varying_known_reals
        if loss_kind == "v3_hurdle_nb":
            assert len(kwargs["logging_metrics"]) == 0
        return original_from_dataset(cls, dataset, **kwargs)

    monkeypatch.setattr(TemporalFusionTransformer, "from_dataset", classmethod(checked_from_dataset))
    if force_overflow:
        original_training_step = TemporalFusionTransformer.training_step
        attempts = {"count": 0}

        def overflowing_training_step(self, batch, batch_idx):
            result = original_training_step(self, batch, batch_idx)
            attempts["count"] += 1
            if attempts["count"] == 1:
                result["loss"] = result["loss"] * 1e35
            return result

        monkeypatch.setattr(TemporalFusionTransformer, "training_step", overflowing_training_step)
    origin = dt.date(2024, 4, 9)
    start = origin - dt.timedelta(days=99)
    rows = []
    for series_id in ("a", "b", "c"):
        for day in range(128):
            date = start + dt.timedelta(days=day)
            available = date <= origin and not (series_id == "c" and day < 75)
            rows.append(
                {
                    "series_id": series_id,
                    "date": pd.Timestamp(date),
                    "time_idx": day,
                    "is_history": date <= origin,
                    "history_available": float(available),
                    "y": float(day % 7) if available else 0.0,
                    "selling_price": 1.0,
                    "week_of_year": str(date.isocalendar().week),
                    "special_days": "none",
                }
            )
    monkeypatch.setattr(
        tft_q, "_tft_frame", lambda *args, **kwargs: (pd.DataFrame(rows), ("series_id",))
    )
    source_path = tmp_path / "source.parquet"
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False), source_path)

    args = dict(
        panel=Panel(
            str(source_path),
            "series_id",
            "date",
            "y",
            known_future=("selling_price",),
            price_col="selling_price",
        ),
        origin=origin.isoformat(),
        horizon=28,
        spark=None,
        max_epochs=2,
        batch_size=2,
        early_stopping_patience=10,
        early_stopping_min_delta=1e-6,
        device=device,
        seed=0,
        hidden_size=8,
        hidden_continuous_size=4,
        attention_head_size=2,
        dropout=0.1,
        learning_rate=0.001,
        max_encoder_length=30,
        min_encoder_length=7,
        lag_short_days=2,
        lag_long_days=14,
        output_size=output_size,
        loss_kind=loss_kind,
        precision=precision,
    )
    selected = tmp_path / "selection" / "model.pt"
    tft_q.fit_predict(**args, out_dir=tmp_path / "selection" / "forecast", checkpoint_out=selected)
    forecast = pq.read_table(tmp_path / "selection" / "forecast")
    assert forecast.num_rows == 84
    assert forecast.schema == pa.schema([
        pa.field("series_id", pa.string()),
        pa.field("date", pa.date32()),
        pa.field("yhat", pa.float64()),
    ])
    selected_payload = torch.load(selected, map_location="cpu", weights_only=True)
    training = selected_payload["training"]
    assert selected_payload["profile"]["variant"] == (
        "v6.1" if loss_kind == "quantile" else "v6.2"
    )
    assert selected_payload["profile"]["loss_kind"] == loss_kind
    assert training["n_series"] == 3
    assert training["cold_start_rows"] == 75
    assert training["cold_start_series"] == 1
    assert training["steps_per_epoch"] == 2
    assert len(training["epoch_log"]) == 2
    assert {row["val_loss_dtype"] for row in training["epoch_log"]} == {
        "torch.float64" if loss_kind == "v3_hurdle_nb" else "torch.float32"
    }
    assert [row["updates"] for row in training["epoch_log"]] == [2, 4]
    assert all(
        row["train_loss"] is not None
        and row["val_loss"] is not None
        and row["elapsed_seconds"] > 0
        for row in training["epoch_log"]
    )
    assert training["selected_updates"] in {2, 4}
    assert training["refit_optimizer_steps"] is None
    assert training["validation_origin"] == origin.isoformat()
    if force_overflow:
        assert training["amp_overflow_retries"] >= 1
    assert training["selected_epoch"] == min(
        training["epoch_log"], key=lambda row: row["val_loss"]
    )["epoch"]
    selected_state = torch.load(selected, map_location="cpu", weights_only=True)["state_dict"]
    lightning_checkpoint = next((selected.parent / "selection-checkpoints").glob("*.ckpt"))
    lightning_state = torch.load(
        lightning_checkpoint, map_location="cpu", weights_only=False
    )["state_dict"]
    for name in selected_state:
        torch.testing.assert_close(selected_state[name], lightning_state[name])

    refit = tmp_path / "refit" / "model.pt"
    tft_q.fit_predict(
        **{**args, "max_epochs": 1},
        out_dir=tmp_path / "refit" / "forecast",
        checkpoint_out=refit,
        training_mode="fixed_refit",
        fixed_update_budget=training["selected_updates"],
    )
    final = torch.load(refit, map_location="cpu", weights_only=True)["training"]
    assert final["selected_updates"] == training["selected_updates"]
    assert final["refit_optimizer_steps"] == training["selected_updates"]
    assert final["validation_origin"] is None
    assert final["training_contract"]["fixed_update_budget"] == training["selected_updates"]
    assert final["validation"] == []
    assert final["maximum_epochs"] == 1
    assert seeds == [(0, True), (0, True)]
