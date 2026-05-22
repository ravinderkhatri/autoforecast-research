"""TFT-Q/TFT-HNB M5 loss comparison on one reference-derived PyTorch-Forecasting TFT backbone.

The local reference configuration is weekly (52 encoder periods, four forecast periods).
TFT-Q/TFT-HNB keeps the declared TFT widths, heads, dropout, and learning rate, while translating
those lengths to daily M5 units (364 and 28 days). TFT-Q uses quantiles; TFT-HNB uses HMFNet's
active hurdle-count objective on a three-channel head. Each selects by its own validation loss
and refits for its selected optimizer-update count. The private reference ``models_v2`` trainer is
not present in the available checkout.
"""

from __future__ import annotations

import datetime as dt
import math
import os
import pathlib
import shutil
import sys
import tempfile
import time
import unicodedata
import uuid
from dataclasses import asdict, replace

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pyspark.sql import SparkSession

from autoforecast.adapters import _validate_daily_span
from autoforecast.panel import ELIGIBLE_STATES, Forecast, Panel
from autoforecast.tft_precision import (
    finite_gradient_clip_callback,
    fp32_count_output_head,
    mixed_precision_tft_class,
)
from provenance.lineage import directory_digest

NAME = "tft_q"
QUANTILES = (0.1, 0.25, 0.5, 0.75)
LOSS_KINDS = {"quantile": 4, "v3_hurdle_nb": 3}
LOSS_VARIANTS = {"quantile": "v6.1", "v3_hurdle_nb": "v6.2"}
MODEL_IDS = {"quantile": "tft_q", "v3_hurdle_nb": "tft_hnb"}
_SPECIAL_DAY_DEFAULT = "none"


def fit_predict_quantile(panel, origin, horizon, **kwargs):
    """TFT-Q entrypoint; ``test_tft_named_variants`` covers routing and refusal."""
    if kwargs.get("loss_kind", "quantile") != "quantile" or kwargs.get("output_size", 4) != 4:
        raise ValueError("TFT-Q requires quantile loss and four output channels")
    return fit_predict(
        panel, origin, horizon, **{**kwargs, "loss_kind": "quantile", "output_size": 4}
    )


def fit_predict_hnb(panel, origin, horizon, **kwargs):
    """TFT-HNB entrypoint; ``test_tft_named_variants`` covers routing and refusal."""
    if (
        kwargs.get("loss_kind", "v3_hurdle_nb") != "v3_hurdle_nb"
        or kwargs.get("output_size", 3) != 3
    ):
        raise ValueError("TFT-HNB requires hurdle-NB loss and three output channels")
    return fit_predict(
        panel, origin, horizon, **{**kwargs, "loss_kind": "v3_hurdle_nb", "output_size": 3}
    )


def fit_predict(
    panel: Panel,
    origin: str,
    horizon: int,
    *,
    spark: SparkSession | None,
    out_dir: str | pathlib.Path,
    max_epochs: int,
    batch_size: int,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    device: str,
    seed: int,
    hidden_size: int,
    hidden_continuous_size: int,
    attention_head_size: int,
    dropout: float,
    learning_rate: float,
    max_encoder_length: int,
    min_encoder_length: int,
    lag_short_days: int,
    lag_long_days: int,
    output_size: int,
    loss_kind: str = "quantile",
    precision: str = "32-true",
    checkpoint_out: str | pathlib.Path | None = None,
    training_mode: str = "fixed_validation",
    fixed_update_budget: int | None = None,
    validation_data_path: str | pathlib.Path | None = None,
    evaluation_contract_path: str | pathlib.Path | None = None,
    evaluation_contract_sha256: str | None = None,
) -> Forecast:
    """Select on the preceding 28 days or refit for the sealed update count.

    The source panel must retain the 28 known-future M5 price rows after ``origin``.  Their target
    values are never selected for training or prediction; only the declared future price and
    calendar fields are used. Each fit epoch contains one latest window per series.
    """
    origin_day = _validate_daily_span(panel, origin, horizon, adapter=NAME)
    requested_device = device
    if device == "auto":
        import torch

        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    _validate_options(
        max_epochs=max_epochs,
        batch_size=batch_size,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        device=device,
        hidden_size=hidden_size,
        hidden_continuous_size=hidden_continuous_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        learning_rate=learning_rate,
        max_encoder_length=max_encoder_length,
        min_encoder_length=min_encoder_length,
        lag_short_days=lag_short_days,
        lag_long_days=lag_long_days,
        output_size=output_size,
        loss_kind=loss_kind,
        precision=precision,
        horizon=horizon,
    )
    if panel.price_col is None or panel.price_col not in panel.known_future:
        raise ValueError(f"{NAME} requires a declared known-future price column")
    if training_mode == "rolling_backtest":
        training_mode = "fixed_validation"
    if training_mode not in {"fixed_validation", "fixed_refit"}:
        raise ValueError("TFT-Q/TFT-HNB requires fixed_validation or fixed_refit mode")
    if training_mode == "fixed_refit":
        if type(fixed_update_budget) is not int or fixed_update_budget < 1:
            raise ValueError("fixed_refit requires a positive selected update count")
    elif fixed_update_budget is not None:
        raise ValueError("selection cannot receive a fixed update budget")
    if (evaluation_contract_path is None) != (evaluation_contract_sha256 is None):
        raise ValueError("study evaluation requires both path and digest")
    study = None
    if evaluation_contract_path is not None:
        from autoforecast.study_validation import StudyValidation

        if training_mode != "fixed_validation" or validation_data_path is None:
            raise ValueError("study selection requires a separate fixed validation panel")
        frame, static_categoricals = _tft_frame(
            panel, origin_day=origin_day, horizon=horizon,
            max_encoder_length=max_encoder_length, min_encoder_length=min_encoder_length,
            lag_long_days=lag_long_days,
        )
        study = StudyValidation.read(
            evaluation_contract_path, evaluation_contract_sha256,
            panel=replace(panel, path=str(validation_data_path)), origin=origin, horizon=horizon,
            series_ids=tuple(sorted(frame["series_id"].astype(str).unique())),
        )
    run_identity = {
        "panel": asdict(panel),
        "origin": origin,
        "horizon": horizon,
        "seed": seed,
        "batch_size": batch_size,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "requested_device": requested_device,
        "resolved_device": device,
        "hidden_size": hidden_size,
        "hidden_continuous_size": hidden_continuous_size,
        "attention_head_size": attention_head_size,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "max_encoder_length": max_encoder_length,
        "min_encoder_length": min_encoder_length,
        "lag_short_days": lag_short_days,
        "lag_long_days": lag_long_days,
        "output_size": output_size,
        "loss_kind": loss_kind,
        "precision": precision,
        "training_mode": training_mode,
        "fixed_update_budget": fixed_update_budget,
        "max_epochs": max_epochs,
        **({"study_evaluation": study.as_record()} if study is not None else {}),
    }
    checkpoint = _checkpoint_path(out_dir, checkpoint_out)
    output_dir = pathlib.Path(out_dir)
    _validate_artifact_destinations(output_dir, checkpoint)
    if (
        not output_dir.is_symlink() and output_dir.is_dir()
        and not checkpoint.is_symlink() and checkpoint.is_file()
    ):
        run_identity["panel_sha256"] = directory_digest(panel.path, suffixes=(".parquet",))
    recovered = _adopt_completed_forecast(
        output_dir, checkpoint, panel=panel, origin=origin, origin_day=origin_day,
        horizon=horizon, max_epochs=max_epochs, training_mode=training_mode,
        fixed_update_budget=fixed_update_budget, loss_kind=loss_kind, precision=precision,
        run_identity=run_identity,
    )
    if recovered is not None:
        return recovered
    _discard_incomplete_forecast(output_dir, checkpoint)
    if os.path.lexists(checkpoint) or os.path.lexists(output_dir):
        raise ValueError(
            "TFT-Q/TFT-HNB checkpoint and forecast destinations must not already exist"
        )
    run_identity["panel_sha256"] = directory_digest(panel.path, suffixes=(".parquet",))

    import lightning.pytorch as pl
    import torch
    from lightning.pytorch.callbacks import ModelCheckpoint
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data.encoders import NaNLabelEncoder
    from pytorch_forecasting.metrics import QuantileLoss

    from autoforecast.adapters.tft_q_loss import HMFNetHurdleLoss, RawCountNormalizer
    if loss_kind == "v3_hurdle_nb":
        loss = HMFNetHurdleLoss()
    else:
        loss = QuantileLoss(quantiles=list(QUANTILES))

    pl.seed_everything(seed, workers=True)
    if study is None:
        frame, static_categoricals = _tft_frame(
            panel,
            origin_day=origin_day,
            horizon=horizon,
            max_encoder_length=max_encoder_length,
            min_encoder_length=min_encoder_length,
            lag_long_days=lag_long_days,
        )
    cutoff = int(frame.loc[frame["is_history"], "time_idx"].max())
    internal_validation = training_mode == "fixed_validation" and study is None
    train_cutoff = cutoff - horizon if internal_validation else cutoff
    if loss_kind == "v3_hurdle_nb":
        _assign_hmfnet_mean_scales(
            frame,
            decoder_ends=(train_cutoff, cutoff)
            if internal_validation else (cutoff,),
            horizon=horizon,
        )
    history = frame.loc[frame["is_history"]].copy()
    unavailable_history = history.loc[history["history_available"] == 0.0]
    train = history.loc[history["time_idx"] <= train_cutoff].copy()
    categorical_encoders = {
        name: NaNLabelEncoder(add_nan=True)
        for name in (*static_categoricals, "week_of_year", "special_days")
    }
    dataset = TimeSeriesDataSet(
        train,
        time_idx="time_idx",
        target="y",
        group_ids=["series_id"],
        min_encoder_length=min_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=horizon,
        max_prediction_length=horizon,
        min_prediction_idx=train_cutoff - horizon + 1,
        static_categoricals=list(static_categoricals),
        time_varying_known_categoricals=["week_of_year", "special_days"],
        time_varying_known_reals=["time_idx", "selling_price"],
        time_varying_unknown_reals=["y", "history_available"],
        categorical_encoders=categorical_encoders,
        target_normalizer=RawCountNormalizer(),
        weight="mean_loss_scale" if loss_kind == "v3_hurdle_nb" else None,
        lags={"y": [lag_short_days, lag_long_days]},
        allow_missing_timesteps=False,
        predict_mode=True,
    )
    n_series = int(history["series_id"].nunique())
    _require_one_window(dataset, n_series, "training")
    steps_per_epoch = math.ceil(n_series / batch_size)
    if training_mode == "fixed_refit" and fixed_update_budget < steps_per_epoch:
        raise ValueError(
            "[autoforecast:non-retryable] selected TFT-Q/TFT-HNB refit update budget cannot "
            "cover every training window"
        )
    if internal_validation:
        validation_dataset = TimeSeriesDataSet.from_dataset(
            dataset, history, predict=True, stop_randomization=True
        )
        _require_one_window(validation_dataset, n_series, "validation")
    else:
        validation_dataset = None
    successful_updates = {"count": 0, "retries": 0}
    model_class = (
        mixed_precision_tft_class(TemporalFusionTransformer, successful_updates, device)
        if precision == "16-mixed" else TemporalFusionTransformer
    )
    model = model_class.from_dataset(
        dataset,
        hidden_size=hidden_size,
        hidden_continuous_size=hidden_continuous_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        learning_rate=learning_rate,
        output_size=output_size,
        loss=loss,
        **({"mask_bias": float("-inf")} if precision == "16-mixed" else {}),
        **({"logging_metrics": torch.nn.ModuleList()} if loss_kind == "v3_hurdle_nb" else {}),
    )
    if precision == "16-mixed" and loss_kind == "v3_hurdle_nb":
        fp32_count_output_head(model)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    epoch_log = _EpochLog(successful_updates if precision == "16-mixed" else None)
    callbacks = [epoch_log, _FiniteGradientClip(precision=precision)]
    prediction_dataset = TimeSeriesDataSet.from_dataset(
        dataset, frame.drop(columns=["is_history"]), predict=True, stop_randomization=True
    )
    _require_one_window(prediction_dataset, n_series, "prediction")
    if study is not None:
        import numpy as np

        from autoforecast.study_validation import study_score_callback

        prediction_loader = prediction_dataset.to_dataloader(
            train=False, batch_size=batch_size, num_workers=0,
        )

        def predict_window(module):
            predictions = {}
            for batch in prediction_loader:
                inputs = module.transfer_batch_to_device(batch[0], module.device, 0)
                output = module.to_prediction(module(inputs)).detach().cpu().numpy()
                ids = prediction_dataset.x_to_index(inputs)["series_id"].astype(str)
                for series, values in zip(ids, output, strict=True):
                    if series in predictions:
                        raise ValueError("study TFT prediction duplicated a series")
                    predictions[series] = np.maximum(values, 0.0)
            if set(predictions) != set(study.series_ids):
                raise ValueError("study TFT prediction changed the frozen population")
            return np.stack([predictions[series] for series in study.series_ids])

        callbacks.insert(0, study_score_callback(study, predict_window))
    selected_checkpoint = None
    if training_mode == "fixed_validation":
        monitor = "study_score" if study is not None else "val_loss"
        selected_checkpoint = ModelCheckpoint(
            dirpath=checkpoint.parent / "selection-checkpoints",
            filename="best-{epoch:05d}",
            monitor=monitor,
            mode="min",
            save_top_k=1,
            auto_insert_metric_name=False,
            save_on_train_epoch_end=study is not None,
        )
        callbacks.extend(
            [
                selected_checkpoint,
                _early_stopping_callback(
                    patience=early_stopping_patience, min_delta=early_stopping_min_delta,
                    monitor=monitor, check_on_train_epoch_end=study is not None,
                ),
            ]
        )
    trainer = pl.Trainer(
        max_epochs=(
            max(max_epochs, math.ceil(fixed_update_budget / steps_per_epoch))
            if fixed_update_budget is not None else max_epochs
        ),
        max_steps=fixed_update_budget if fixed_update_budget is not None else -1,
        accelerator=device,
        devices=1,
        precision=precision,
        logger=False,
        enable_checkpointing=selected_checkpoint is not None,
        enable_model_summary=False,
        enable_progress_bar=True,
        callbacks=callbacks,
    )
    trainer.fit(
        model,
        train_dataloaders=dataset.to_dataloader(
            train=True, batch_size=batch_size, num_workers=0, shuffle=False, drop_last=False
        ),
        val_dataloaders=(
            validation_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
            if validation_dataset is not None
            else None
        ),
    )
    if precision == "16-mixed" and successful_updates["count"] != trainer.global_step:
        raise RuntimeError(
            "TFT-Q/TFT-HNB AMP counted a batch without a successful optimizer update"
        )
    if selected_checkpoint is not None:
        if not selected_checkpoint.best_model_path:
            raise RuntimeError("TFT-Q/TFT-HNB validation produced no selected checkpoint")
        if (selected_checkpoint.best_model_score is None
                or not torch.isfinite(selected_checkpoint.best_model_score)):
            raise FloatingPointError(
                "TFT-Q/TFT-HNB selected checkpoint has nonfinite validation loss"
            )
        selected_epoch = (
            int(
                torch.load(
                    selected_checkpoint.best_model_path, map_location="cpu", weights_only=False
                )["epoch"]
            )
            + 1
        )
        selected_updates = (
            int(epoch_log.records[selected_epoch - 1]["updates"])
            if precision == "16-mixed" else selected_epoch * steps_per_epoch
        )
        model = TemporalFusionTransformer.load_from_checkpoint(
            selected_checkpoint.best_model_path, map_location="cpu"
        )
        if precision == "16-mixed" and loss_kind == "v3_hurdle_nb":
            fp32_count_output_head(model)
    else:
        selected_updates = (
            successful_updates["count"] if precision == "16-mixed" else trainer.global_step
        )
        selected_epoch = math.ceil(selected_updates / steps_per_epoch)
        if selected_updates != fixed_update_budget:
            raise RuntimeError("TFT-Q/TFT-HNB refit stopped before the selected update count")
    raw = model.predict(
        prediction_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0),
        mode="quantiles" if loss_kind == "quantile" else "prediction",
        return_index=True,
        trainer_kwargs={
            "accelerator": device,
            "devices": 1,
            "precision": precision,
            "logger": False,
            "enable_progress_bar": True,
        },
    )
    output = raw.output.detach().cpu().numpy()
    index = raw.index
    median_index = QUANTILES.index(0.5)
    rows = [
        (
            str(series_id),
            origin_day + dt.timedelta(days=step + 1),
            float(max(
                output[position, step, median_index]
                if loss_kind == "quantile" else output[position, step],
                0.0,
            )),
        )
        for position, series_id in enumerate(index["series_id"].to_numpy())
        for step in range(horizon)
    ]
    if set(index["series_id"].astype(str)) != set(history["series_id"]):
        raise RuntimeError("TFT-Q/TFT-HNB prediction dropped historical series")
    if directory_digest(panel.path, suffixes=(".parquet",)) != run_identity["panel_sha256"]:
        raise RuntimeError("TFT-Q/TFT-HNB input panel changed during training")
    _publish_tft_artifacts(
        {
            "schema": "autoforecast.tft-v6-inference-state.v2",
            "state_dict": model.state_dict(),
            "run_identity": run_identity,
            "training": {
                "maximum_epochs": max_epochs,
                "n_series": n_series,
                "cold_start_rows": len(unavailable_history),
                "cold_start_series": int(unavailable_history["series_id"].nunique()),
                "selected_epoch": selected_epoch,
                "selected_updates": selected_updates,
                "validation_origin": origin if training_mode == "fixed_validation" else None,
                "refit_optimizer_steps": (
                    selected_updates if training_mode == "fixed_refit" else None
                ),
                "steps_per_epoch": steps_per_epoch,
                "early_stopping_patience": early_stopping_patience,
                "early_stopping_min_delta": early_stopping_min_delta,
                "precision": precision,
                "requested_device": requested_device,
                "resolved_device": device,
                "amp_overflow_retries": successful_updates["retries"],
                "validation": epoch_log.records if training_mode == "fixed_validation" else [],
                "epoch_log": epoch_log.records,
                **({"selection_metric": study.metric} if study is not None else {}),
                "training_contract": {
                    "mode": training_mode,
                    "fixed_update_budget": fixed_update_budget,
                    **({"study_evaluation": study.as_record()} if study is not None else {}),
                },
            },
            "profile": {
                "variant": LOSS_VARIANTS[loss_kind],
                "loss_kind": loss_kind,
                "point_prediction": "median" if loss_kind == "quantile" else "mean",
                "target_normalization": "identity",
                "cold_start": "source_unobserved_rows_zero_with_encoder_mask_v1",
                "mean_loss_weight": 0.2 if loss_kind == "v3_hurdle_nb" else None,
                "quantiles": list(QUANTILES) if loss_kind == "quantile" else None,
                "hidden_size": hidden_size,
                "hidden_continuous_size": hidden_continuous_size,
                "attention_head_size": attention_head_size,
                "dropout": dropout,
                "learning_rate": learning_rate,
                "precision": precision,
                "max_encoder_length": max_encoder_length,
                "min_encoder_length": min_encoder_length,
                "lags": [lag_short_days, lag_long_days],
                "output_size": output_size,
            },
        },
        rows=rows,
        checkpoint=checkpoint,
        output_dir=output_dir,
    )
    return Forecast(
        adapter=MODEL_IDS[loss_kind],
        path=str(out_dir),
        origin=origin,
        horizon=horizon,
        n_series=n_series,
        disposition={"emitted": len(rows)},
        target_estimand=panel.target_estimand,
        frequency=panel.frequency,
    ).validate()


def _write_forecast(
    rows: list[tuple[str, dt.date, float]], output_dir: pathlib.Path
) -> None:
    """Publish the Arrow forecast only after its Parquet part is fully written."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as staged:
        pq.write_table(
            pa.Table.from_pydict(
                {
                    "series_id": [row[0] for row in rows],
                    "date": [row[1] for row in rows],
                    "yhat": [row[2] for row in rows],
                },
                schema=pa.schema([
                    pa.field("series_id", pa.string()),
                    pa.field("date", pa.date32()),
                    pa.field("yhat", pa.float64()),
                ]),
            ),
            pathlib.Path(staged) / "part-00000.parquet",
        )
        os.rename(staged, output_dir)


def _publish_tft_artifacts(
    payload: dict[str, object],
    *,
    rows: list[tuple[str, dt.date, float]],
    checkpoint: pathlib.Path,
    output_dir: pathlib.Path,
) -> None:
    """Publish the checkpoint last; a pending forecast is retryable after interruption."""
    import torch

    _validate_artifact_destinations(output_dir, checkpoint)
    _discard_incomplete_forecast(output_dir, checkpoint)
    if os.path.lexists(checkpoint) or os.path.lexists(output_dir):
        raise ValueError(
            "TFT-Q/TFT-HNB checkpoint and forecast destinations must not already exist"
        )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{checkpoint.name}-", dir=checkpoint.parent,
    ) as staged:
        staged_checkpoint = pathlib.Path(staged) / "model.pt"
        with tempfile.TemporaryDirectory(
            prefix=f".{output_dir.name}-", dir=output_dir.parent,
        ) as forecast_staging:
            staged_forecast = pathlib.Path(forecast_staging) / "forecast"
            _write_forecast(rows, staged_forecast)
            bound_payload = {
                **payload,
                "forecast_sha256": directory_digest(
                    staged_forecast, suffixes=(".parquet",)
                ),
            }
            with staged_checkpoint.open("wb") as handle:
                torch.save(bound_payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            marker = staged_forecast / ".v6-pending"
            with marker.open("x", encoding="utf-8") as handle:
                handle.write(str(checkpoint.absolute()) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.rename(staged_forecast, output_dir)
                os.link(staged_checkpoint, checkpoint)
            except BaseException as error:
                try:
                    _discard_incomplete_forecast(output_dir, checkpoint)
                except OSError as cleanup_error:
                    error.add_note(f"TFT forecast rollback failed: {cleanup_error}")
                raise
            else:
                (output_dir / ".v6-pending").unlink()


def _validate_artifact_destinations(
    output_dir: pathlib.Path, checkpoint: pathlib.Path
) -> None:
    """Reject intersecting paths and redirected ancestors before expensive training."""
    forecast = pathlib.Path(os.path.abspath(output_dir))
    model = pathlib.Path(os.path.abspath(checkpoint))
    if _physical_paths_overlap(forecast, model):
        raise ValueError("TFT-Q/TFT-HNB checkpoint and forecast destinations must not overlap")
    for path in (forecast, model):
        for parent in path.parents:
            try:
                parent.lstat()
            except FileNotFoundError:
                continue
            if parent.is_symlink():
                alias = (
                    unicodedata.normalize("NFC", str(parent)).casefold(),
                    unicodedata.normalize("NFC", str(parent.resolve())).casefold(),
                )
                if sys.platform == "darwin" and alias in {
                    ("/tmp", "/private/tmp"), ("/var", "/private/var")
                }:
                    continue
                raise ValueError("TFT-Q/TFT-HNB destination ancestor must not be a symlink")
            if not parent.is_dir():
                raise ValueError("TFT-Q/TFT-HNB destination ancestor must be a directory")


def _physical_paths_overlap(left: pathlib.Path, right: pathlib.Path) -> bool:
    """Compare unresolved tails under the same inode, including macOS firmlinks."""
    left = pathlib.Path(os.path.realpath(left))
    right = pathlib.Path(os.path.realpath(right))

    def key(parts: tuple[str, ...]) -> tuple[str, ...]:
        if sys.platform != "darwin":
            return parts
        return tuple(unicodedata.normalize("NFD", part).casefold() for part in parts)

    def anchor(path: pathlib.Path) -> tuple[pathlib.Path, tuple[str, ...]]:
        tail: list[str] = []
        while not path.exists() and path != path.parent:
            tail.append(path.name)
            path = path.parent
        return path, key(tuple(reversed(tail)))

    left_anchor, left_tail = anchor(left)
    right_anchor, right_tail = anchor(right)
    try:
        same_anchor = os.path.samefile(left_anchor, right_anchor)
    except OSError:
        same_anchor = False
    if same_anchor and (
        left_tail == right_tail
        or left_tail == right_tail[:len(left_tail)]
        or right_tail == left_tail[:len(right_tail)]
    ):
        return True
    left_key, right_key = key(left.parts), key(right.parts)
    return (
        left_key == right_key
        or left_key == right_key[:len(left_key)]
        or right_key == left_key[:len(right_key)]
    )


def _adopt_completed_forecast(
    output_dir: pathlib.Path,
    checkpoint: pathlib.Path,
    *,
    panel: Panel,
    origin: str,
    origin_day: dt.date,
    horizon: int,
    max_epochs: int,
    training_mode: str,
    fixed_update_budget: int | None,
    loss_kind: str,
    precision: str,
    run_identity: dict[str, object],
) -> Forecast | None:
    """Finish a post-link publication only when its checkpoint and forecast are intact."""
    marker = output_dir / ".v6-pending"
    part = output_dir / "part-00000.parquet"
    pending = os.path.lexists(marker)
    members = (
        _forecast_members(output_dir)
        if not output_dir.is_symlink() and output_dir.is_dir() else None
    )
    if (
        output_dir.is_symlink() or not output_dir.is_dir()
        or checkpoint.is_symlink() or not checkpoint.is_file()
        or part.is_symlink() or not part.is_file()
        or members != ({marker.name, part.name} if pending else {part.name})
        or (pending and (
            marker.is_symlink() or not marker.is_file()
            or marker.read_text(encoding="utf-8") != str(checkpoint.absolute()) + "\n"
        ))
    ):
        return None
    import torch

    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        training = payload["training"]
        profile = payload["profile"]
        contract = training["training_contract"]
        n_series = training["n_series"]
        table = pq.read_table(part)
    except Exception as error:
        raise ValueError("TFT-Q/TFT-HNB pending checkpoint or forecast is unreadable") from error
    expected_schema = pa.schema([
        pa.field("series_id", pa.string()),
        pa.field("date", pa.date32()),
        pa.field("yhat", pa.float64()),
    ])
    if (
        type(payload) is not dict
        or payload.get("schema") != "autoforecast.tft-v6-inference-state.v2"
        or payload.get("run_identity") != run_identity
        or payload.get("forecast_sha256") != directory_digest(
            output_dir, suffixes=(".parquet",)
        )
        or type(training) is not dict or type(profile) is not dict
        or type(contract) is not dict or type(n_series) is not int or n_series < 1
        or training.get("maximum_epochs") != max_epochs
        or training.get("precision") != precision
        or training.get("validation_origin") != (
            origin if training_mode == "fixed_validation" else None
        )
        or type(training.get("selected_updates")) is not int
        or training["selected_updates"] < 1
        or contract.get("mode") != training_mode
        or contract.get("fixed_update_budget") != fixed_update_budget
        or contract.get("study_evaluation") != run_identity.get("study_evaluation")
        or (training_mode == "fixed_refit" and (
            training.get("refit_optimizer_steps") != fixed_update_budget
            or training.get("selected_updates") != fixed_update_budget
            or type(training.get("steps_per_epoch")) is not int
            or training.get("steps_per_epoch") != math.ceil(n_series / run_identity["batch_size"])
            or fixed_update_budget < training["steps_per_epoch"]
        ))
        or profile.get("loss_kind") != loss_kind
        or table.schema != expected_schema
        or table.num_rows != n_series * horizon
        or any(column.null_count for column in table.columns)
        or pc.count_distinct(table["series_id"]).as_py() != n_series
        or pc.min(table["date"]).as_py() != origin_day + dt.timedelta(days=1)
        or pc.max(table["date"]).as_py() != origin_day + dt.timedelta(days=horizon)
        or not pc.all(pc.is_finite(table["yhat"])).as_py()
        or pc.min(table["yhat"]).as_py() < 0
    ):
        raise ValueError(
            "TFT-Q/TFT-HNB pending checkpoint or forecast differs from the requested run"
        )
    forecast = Forecast(
        adapter=MODEL_IDS[loss_kind], path=str(output_dir), origin=origin, horizon=horizon,
        n_series=n_series, disposition={"emitted": table.num_rows},
        target_estimand=panel.target_estimand, frequency=panel.frequency,
    ).validate()
    if pending:
        marker.unlink()
    return forecast


def _forecast_members(output_dir: pathlib.Path) -> set[str] | None:
    """Ignore only Finder's regular metadata file, never an arbitrary extra artifact."""
    entries = {item.name: item for item in output_dir.iterdir()}
    finder = entries.pop(".DS_Store", None)
    if finder is not None and (finder.is_symlink() or not finder.is_file()):
        return None
    return set(entries)


def _discard_incomplete_forecast(
    output_dir: pathlib.Path, checkpoint: pathlib.Path
) -> None:
    """Remove only this adapter's marked, exact-shape orphan before a fresh retry."""
    if checkpoint.exists() or checkpoint.is_symlink() or not output_dir.exists():
        return
    if output_dir.is_symlink() or not output_dir.is_dir():
        return
    marker = output_dir / ".v6-pending"
    part = output_dir / "part-00000.parquet"
    if (
        marker.is_symlink()
        or not marker.is_file()
        or part.is_symlink()
        or not part.is_file()
        or _forecast_members(output_dir) != {marker.name, part.name}
        or marker.read_text(encoding="utf-8") != str(checkpoint.absolute()) + "\n"
    ):
        return
    abandoned = output_dir.with_name(
        f".{output_dir.name}-abandoned-{uuid.uuid4().hex}"
    )
    os.rename(output_dir, abandoned)
    if not os.path.lexists(abandoned / ".DS_Store"):
        shutil.rmtree(abandoned)


def _validate_options(**options: object) -> None:
    """Reject malformed, non-equivalent profile settings before loading data."""
    integer_names = (
        "max_epochs",
        "batch_size",
        "hidden_size",
        "hidden_continuous_size",
        "attention_head_size",
        "max_encoder_length",
        "min_encoder_length",
        "lag_short_days",
        "lag_long_days",
        "output_size",
        "horizon",
        "early_stopping_patience",
    )
    for name in integer_names:
        value = options[name]
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if options["horizon"] != 28 or options["early_stopping_patience"] != 10:
        raise ValueError("TFT-Q/TFT-HNB M5 requires a 28-day horizon and validation patience 10")
    if options["device"] not in ("cpu", "cuda", "mps"):
        raise ValueError("device must be cpu, cuda, or mps")
    precision = options.get("precision", "32-true")
    if precision not in ("32-true", "16-mixed"):
        raise ValueError("precision must be 32-true or 16-mixed")
    if precision == "16-mixed" and options["device"] == "cpu":
        raise ValueError("Lightning substitutes bfloat16 for CPU 16-mixed; use cuda or mps")
    if options["attention_head_size"] > options["hidden_size"]:
        raise ValueError("attention_head_size must not exceed hidden_size")
    if options["min_encoder_length"] > options["max_encoder_length"]:
        raise ValueError("min_encoder_length must not exceed max_encoder_length")
    if options["lag_short_days"] >= options["lag_long_days"]:
        raise ValueError("lag_short_days must be smaller than lag_long_days")
    loss_kind = options.get("loss_kind", "quantile")
    if type(loss_kind) is not str or loss_kind not in LOSS_KINDS:
        raise ValueError(f"unknown TFT loss kind {loss_kind!r}")
    if options["output_size"] != LOSS_KINDS[loss_kind]:
        raise ValueError(f"output_size must equal {LOSS_KINDS[loss_kind]} for {loss_kind}")
    dropout = options["dropout"]
    learning_rate = options["learning_rate"]
    early_stopping_min_delta = options["early_stopping_min_delta"]
    if (
        not isinstance(dropout, int | float)
        or isinstance(dropout, bool)
        or not 0 <= float(dropout) < 1
    ):
        raise ValueError("dropout must be in [0, 1)")
    if (
        not isinstance(learning_rate, int | float)
        or isinstance(learning_rate, bool)
        or not 0 < float(learning_rate) <= 1
    ):
        raise ValueError("learning_rate must be in (0, 1]")
    if (
        not isinstance(early_stopping_min_delta, int | float)
        or isinstance(early_stopping_min_delta, bool)
        or not 0 <= float(early_stopping_min_delta) < 1
    ):
        raise ValueError("early_stopping_min_delta must be in [0, 1)")


def _early_stopping_callback(
    *, patience: int, min_delta: float, monitor: str = "val_loss",
    check_on_train_epoch_end: bool = False,
):
    """Build the explicit TFT-Q/TFT-HNB stopping rule after options have been validated."""
    from lightning.pytorch.callbacks import EarlyStopping

    return EarlyStopping(
        monitor=monitor,
        mode="min",
        min_delta=min_delta,
        patience=patience,
        check_on_train_epoch_end=check_on_train_epoch_end,
    )


def _require_one_window(dataset, expected: int, kind: str) -> None:
    """Refuse a hidden rolling-window expansion or a silently omitted series."""
    if len(dataset) != expected:
        raise ValueError(
            f"TFT {kind} must have exactly one window per series; "
            f"observed={len(dataset)}, expected={expected}"
        )


def _EpochLog(successful_updates: dict[str, int] | None = None, *, label: str = "TFT-Q/TFT-HNB"):
    """Capture elapsed training time and both epoch losses as plain checkpoint evidence."""
    from lightning.pytorch import Callback

    class EpochLog(Callback):
        def __init__(self):
            self.records: list[dict[str, float | int | None]] = []
            self.started = 0.0
            self.loss_sum = None
            self.samples = 0

        def on_train_epoch_start(self, trainer, pl_module):
            self.started = time.monotonic()
            self.loss_sum = None
            self.samples = 0

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            # Validation runs before on_train_epoch_end in Lightning. Accumulate
            # detached scalars on-device, then transfer only once per epoch.
            samples = int(outputs["n_samples"])
            weighted = outputs["loss"].detach() * samples
            self.loss_sum = weighted if self.loss_sum is None else self.loss_sum + weighted
            self.samples += samples

        def on_train_epoch_end(self, trainer, pl_module):
            if not trainer.num_val_batches:
                self._record(trainer)

        def on_validation_epoch_end(self, trainer, pl_module):
            if trainer.sanity_checking:
                return
            self._record(trainer)

        def on_fit_end(self, trainer, pl_module):
            if not self.records and trainer.global_step:
                self._record(trainer)

        def _record(self, trainer):
            if self.samples <= 0 or self.loss_sum is None:
                raise RuntimeError(f"{label} training epoch has no samples to log")
            metrics = trainer.callback_metrics
            val_loss = metrics.get("val_loss")
            if val_loss is not None and not math.isfinite(float(val_loss)):
                raise FloatingPointError(f"{label} epoch has nonfinite validation loss")
            train_loss = float(self.loss_sum / self.samples)
            if not math.isfinite(train_loss):
                raise FloatingPointError(f"{label} epoch has nonfinite training loss")
            self.records.append(
                {
                    "epoch": trainer.current_epoch + 1,
                    "updates": (
                        trainer.global_step
                        if successful_updates is None else successful_updates["count"]
                    ),
                    "train_loss": train_loss,
                    "val_loss": None if val_loss is None else float(val_loss),
                    "val_loss_dtype": None if val_loss is None else str(val_loss.dtype),
                    **({
                        "study_score": float(metrics["study_score"]),
                        "study_score_dtype": str(metrics["study_score"].dtype),
                    } if "study_score" in metrics else {}),
                    "elapsed_seconds": time.monotonic() - self.started,
                }
            )
            print(f"{label} epoch {trainer.current_epoch + 1}: {self.records[-1]}", flush=True)

    return EpochLog()


def _FiniteGradientClip(*, precision: str = "32-true"):
    """Apply HMFNet's norm-one clipping and reject nonfinite gradients before Adam."""
    return finite_gradient_clip_callback(precision=precision)


def required_history(*, min_encoder_length: int, lag_long_days: int, horizon: int) -> int:
    """Minimum eligible observations for one direct TFT window with its longest target lag."""
    return min_encoder_length + lag_long_days + horizon


def _checkpoint_path(
    out_dir: str | pathlib.Path, checkpoint_out: str | pathlib.Path | None
) -> pathlib.Path:
    if checkpoint_out is not None:
        return pathlib.Path(checkpoint_out)
    destination = pathlib.Path(out_dir)
    return destination.with_name(f"{destination.name}-v6-artifacts") / "model.pt"


def _mark_cold_start(frame: pd.DataFrame) -> None:
    """Keep source pre-launch dates as zero-valued, explicitly unavailable encoder context."""
    available = frame["is_history"] & frame["is_admitted"]
    frame["history_available"] = available.astype(float)
    frame["y"] = frame["observed_y"].where(available, 0.0).astype(float)


def _require_uncensored_m5(frame: pd.DataFrame) -> None:
    """The paired objectives must see exact M5 counts, not a censored lower bound."""
    if (frame["is_history"] & frame["observation_state"].eq("lower_bound")).any():
        raise ValueError("TFT-Q/TFT-HNB loss comparison requires exact or unobserved M5 history")


def _assign_hmfnet_mean_scales(
    frame: pd.DataFrame, *, decoder_ends: tuple[int, ...], horizon: int
) -> None:
    """Attach HMFNet's origin-frozen recent-sales mean scaling to decoder targets only."""
    frame["mean_loss_scale"] = 1.0
    for decoder_end in dict.fromkeys(decoder_ends):
        decoder_start = decoder_end - horizon + 1
        recent = frame.loc[
            (frame["time_idx"] >= decoder_start - 28)
            & (frame["time_idx"] < decoder_start)
            & (frame["history_available"] == 1.0)
        ]
        recent_mean = recent.groupby("series_id", sort=False, observed=True)["y"].mean()
        scale = (1.0 + recent_mean).pow(-2)
        decoder = frame["time_idx"].between(decoder_start, decoder_end)
        frame.loc[decoder, "mean_loss_scale"] = (
            frame.loc[decoder, "series_id"].map(scale).astype(float).fillna(1.0).to_numpy()
        )


def _tft_frame(
    panel: Panel,
    *,
    origin_day: dt.date,
    horizon: int,
    max_encoder_length: int,
    min_encoder_length: int,
    lag_long_days: int,
):
    """Build one origin-safe latest-window TFT frame, including only known future price."""
    future_end = origin_day + dt.timedelta(days=horizon)
    history_start = origin_day - dt.timedelta(
        days=max_encoder_length + lag_long_days + 2 * horizon - 1
    )
    static_source = ("series_id", *panel.static)
    if panel.product_col is not None:
        static_source = (*static_source, panel.product_col)
    static_columns = tuple(dict.fromkeys(static_source))
    columns = tuple(dict.fromkeys((
        panel.series_col, panel.time_col, panel.target_col, panel.price_col,
        *((panel.exposure_col,) if panel.has_exposure else ()),
        *static_columns[1:],
    )))
    frame = ds.dataset(panel.path, format="parquet", partitioning="hive").to_table(
        columns=list(columns),
        filter=(ds.field(panel.time_col) >= history_start)
        & (ds.field(panel.time_col) <= future_end),
    ).to_pandas(strings_to_categorical=True, date_as_object=False)
    aliases = {
        panel.series_col: "series_id",
        panel.time_col: "date",
        panel.target_col: "observed_y",
        panel.price_col: "selling_price",
    }
    if panel.has_exposure:
        aliases[panel.exposure_col] = "observation_state"
    frame.rename(columns=aliases, inplace=True)
    if not panel.has_exposure:
        frame["observation_state"] = "exact"
    frame["is_history"] = frame["date"] <= pd.Timestamp(origin_day)
    frame["is_admitted"] = (
        ~frame["is_history"]
        | frame["observation_state"].isin(ELIGIBLE_STATES).fillna(False)
    )
    if frame.empty:
        raise ValueError("TFT-Q/TFT-HNB has no admitted M5 rows through the requested origin")
    frame.sort_values(["series_id", "date"], inplace=True, ignore_index=True)
    frame["date"] = frame["date"].astype("datetime64[ns]")
    frame["time_idx"] = (frame["date"] - frame["date"].min()).dt.days.astype("int64")
    _require_uncensored_m5(frame)
    _mark_cold_start(frame)
    future = frame.loc[~frame["is_history"]]
    expected_future = future.groupby("series_id", sort=False, observed=True).size()
    history_series = set(frame.loc[frame["is_history"], "series_id"])
    missing_series = history_series - set(expected_future.index)
    if missing_series:
        raise ValueError(
            "TFT-Q/TFT-HNB requires known-future rows for every historical series; "
            f"missing_series={len(missing_series)}"
        )
    extra_series = set(expected_future.index) - history_series
    if extra_series:
        raise ValueError(f"TFT future rows contain {len(extra_series)} unknown series")
    missing_future = expected_future[expected_future != horizon]
    if not missing_future.empty:
        raise ValueError(
            "TFT-Q/TFT-HNB requires every series to carry exactly horizon known-future rows; "
            f"violations={len(missing_future)}"
        )
    history_rows = frame.loc[frame["is_history"]]
    history_sizes = history_rows.groupby("series_id", sort=False, observed=True).size()
    history_dates = history_rows.groupby("series_id", sort=False, observed=True)["date"].agg(
        ["min", "max", "nunique"]
    )
    if (history_dates["nunique"] != history_sizes).any() or (
        (history_dates["max"] - history_dates["min"]).dt.days + 1 != history_sizes
    ).any():
        raise ValueError(
            "TFT-Q/TFT-HNB requires one calendar row per historical date, including cold start"
        )
    required = required_history(
        min_encoder_length=min_encoder_length,
        lag_long_days=lag_long_days,
        horizon=2 * horizon,
    )
    short = history_sizes[history_sizes < required]
    if not short.empty:
        raise ValueError(
            "TFT-Q/TFT-HNB refuses to silently drop short-history series; "
            f"required_history={required}, below_requirement={len(short)}"
        )
    last_two_windows = frame.loc[
        (frame["history_available"] == 1.0)
        & (frame["date"] > pd.Timestamp(origin_day - dt.timedelta(days=2 * horizon)))
    ]
    recent_dates = last_two_windows.groupby("series_id", sort=False, observed=True)["date"].agg(
        ["size", "nunique"]
    )
    if (recent_dates.reindex(history_sizes.index).fillna(0) != 2 * horizon).any().any():
        raise ValueError("TFT-Q/TFT-HNB requires two complete 28-day historical target windows")
    if frame.loc[frame["history_available"] == 1.0, "observed_y"].isna().any():
        raise ValueError("TFT-Q/TFT-HNB training and validation targets must be observed")
    expected_dates = set(pd.date_range(origin_day + dt.timedelta(days=1), future_end))
    if any(
        set(group["date"]) != expected_dates or len(group) != horizon
        for _, group in future.groupby("series_id", sort=False, observed=True)
    ):
        raise ValueError("TFT-Q/TFT-HNB requires each exact known-future date once per series")
    frame["selling_price"] = (
        frame.groupby("series_id", sort=False, observed=True)["selling_price"]
        .transform(lambda values: values.ffill().fillna(0.0))
        .astype(float)
    )
    frame["week_of_year"] = frame["date"].dt.isocalendar().week.astype(str).astype("category")
    # M5 has no source-controlled equivalent to reference's special_days field.  Keep the TFT
    # input slot but make its all-none status explicit rather than inventing a holiday feed.
    frame["special_days"] = pd.Categorical([_SPECIAL_DAY_DEFAULT] * len(frame))
    static_categoricals = tuple(
        dict.fromkeys(
            ("series_id", *panel.static, *((panel.product_col,) if panel.product_col else ()))
        )
    )
    for column in static_categoricals:
        values = frame[column].astype("category")
        if not all(isinstance(value, str) for value in values.cat.categories):
            values = values.astype("string").astype("category")
        if values.isna().any():
            if "unknown" not in values.cat.categories:
                values = values.cat.add_categories(["unknown"])
            values = values.fillna("unknown")
        frame[column] = values
    return (
        frame.drop(columns=["observed_y", "observation_state", "is_admitted"]),
        static_categoricals,
    )


__all__ = ["NAME", "QUANTILES", "fit_predict", "required_history"]
