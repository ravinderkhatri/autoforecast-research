"""Functional checks for platform-owned adapter tracking selection."""

from __future__ import annotations

import pathlib
import tempfile
from unittest import mock

import pytest

from autoforecast.tracking_plan import TrackingPlan, enable_autologging, tracking_plan


def test_lightning_adapter_gets_one_platform_owned_autologger() -> None:
    plan = tracking_plan("tft")
    assert plan.mode == "autolog" and plan.flavor == "lightning"
    with mock.patch("mlflow.pytorch.autolog") as autolog:
        enable_autologging(plan)
    autolog.assert_called_once_with(log_models=False)


def test_custom_vanilla_torch_uses_the_neutral_reporter_path() -> None:
    plan = tracking_plan("mft")
    assert plan.mode == "reporter" and plan.flavor == "none"
    with mock.patch("mlflow.pytorch.autolog") as autolog:
        enable_autologging(plan)
    autolog.assert_not_called()


def test_non_ml_models_do_not_import_or_configure_mlflow() -> None:
    assert tracking_plan("seasonal_naive").mode == "none"
    assert tracking_plan("croston").mode == "none"


def test_invalid_tracking_plan_fails_before_training() -> None:
    with pytest.raises(ValueError, match="requires a framework flavor"):
        TrackingPlan("autolog", "none").validate()


def test_lightning_training_records_loss_without_model_mlflow_code() -> None:
    """The platform plan, not the model, activates MLflow's Lightning integration."""
    mlflow = pytest.importorskip("mlflow")
    pl = pytest.importorskip("lightning.pytorch")
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader, TensorDataset

    class TinyLightningModel(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.layer = torch.nn.Linear(1, 1)

        def training_step(self, batch, _batch_index):
            features, target = batch
            loss = torch.nn.functional.mse_loss(self.layer(features), target)
            self.log("train_loss", loss)
            return loss

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

    with tempfile.TemporaryDirectory() as temporary:
        uri = f"sqlite:///{pathlib.Path(temporary) / 'mlflow.db'}"
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("tracking-plan-functional")
        enable_autologging(tracking_plan("tft"))
        data = DataLoader(
            TensorDataset(torch.ones(4, 1), torch.ones(4, 1)), batch_size=2,
        )
        with mlflow.start_run() as active:
            pl.Trainer(
                max_epochs=1, logger=False, enable_checkpointing=False,
                enable_progress_bar=False, accelerator="cpu",
            ).fit(TinyLightningModel(), data)
        metrics = mlflow.tracking.MlflowClient(uri).get_run(active.info.run_id).data.metrics
    assert "train_loss" in metrics
