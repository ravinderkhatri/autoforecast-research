"""Train and reload TimesFM-HNB: frozen full TimesFM 3 plus an optional-input count head."""

from __future__ import annotations

import datetime as dt
import io
import math
import os
import pathlib
from dataclasses import asdict, replace

import numpy as np

from autoforecast.adapters import _validate_daily_span
from autoforecast.panel import Forecast, Panel
from autoforecast.timesfm_hnb_data import features, groups, load_data, missing_profiles
from autoforecast.timesfm_hnb_source import TimesFM3Source, file_sha256
from provenance.lineage import directory_digest

NAME = "timesfm_hnb"
SCHEMA = "autoforecast.timesfm-v7-inference.v1"


def _options(
    context_days,
    group_size,
    max_variates,
    head_width,
    learning_rate,
    max_epochs,
    batch_size,
    early_stopping_patience,
    early_stopping_min_delta,
    seed,
    device,
    precision,
    training_mode,
    fixed_update_budget,
):
    """Validate direct callers as well as the catalog bridge before source allocation."""
    for name, value in {
        "context_days": context_days,
        "group_size": group_size,
        "max_variates": max_variates,
        "head_width": head_width,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "patience": early_stopping_patience,
    }.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"TimesFM-HNB {name} must be a positive integer")
    if context_days > 15360 or group_size > max_variates or max_variates > 64:
        raise ValueError("TimesFM-HNB development context/group/channel budget exceeded")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("TimesFM-HNB seed must be a uint32 integer")
    for name, value in {
        "learning_rate": learning_rate,
        "min_delta": early_stopping_min_delta,
    }.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"TimesFM-HNB {name} must be finite and nonnegative")
    if learning_rate == 0 or device not in {"cpu", "cuda", "mps", "auto", None}:
        raise ValueError("TimesFM-HNB learning rate/device is invalid")
    if precision != "32-true":
        raise ValueError("TimesFM-HNB first implementation requires 32-true precision")
    if training_mode not in {"fixed_validation", "fixed_refit"}:
        raise ValueError("TimesFM-HNB requires fixed_validation or fixed_refit")
    if training_mode == "fixed_refit":
        if type(fixed_update_budget) is not int or fixed_update_budget < 1:
            raise ValueError(
                "TimesFM-HNB fresh refit requires a positive sealed selected update count"
            )
    elif fixed_update_budget is not None:
        raise ValueError("TimesFM-HNB selection cannot receive a refit update budget")


def _data_identity(panel: Panel) -> dict[str, str | None]:
    return {
        "panel_sha256": directory_digest(panel.path, suffixes=(".parquet",)),
        "plan_sha256": None
        if panel.plan is None
        else directory_digest(panel.plan.path, suffixes=(".parquet",)),
    }


def _dataset(inputs, target, observed, order=None):
    import torch
    from torch.utils.data import Dataset

    admitted = inputs["admitted"] & observed.any(axis=1)
    if not admitted.any():
        raise ValueError("TimesFM-HNB training window has no observed history/target pairs")
    rows = np.flatnonzero(admitted) if order is None else [i for i in order if admitted[i]]
    tensors = {
        name: torch.from_numpy(value[rows].copy())
        for name, value in inputs.items()
        if name != "admitted"
    }
    tensors.update(
        target=torch.from_numpy(target[rows].copy()),
        observed=torch.from_numpy(observed[rows].copy()),
    )

    class WindowDataset(Dataset):
        def __len__(self):
            return len(tensors["target"])

        def __getitem__(self, index):
            return {name: value[index] for name, value in tensors.items()}

    return WindowDataset()


def _means(head, inputs, batch_size):
    import torch

    head.eval()
    indices = np.flatnonzero(inputs["admitted"])
    result = np.zeros((len(inputs["admitted"]), inputs["features"].shape[1]), np.float64)
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            rows = indices[start : start + batch_size]
            batch = {
                name: torch.from_numpy(value[rows].copy())
                for name, value in inputs.items()
                if name != "admitted"
            }
            result[rows] = head(batch).mean().numpy()
    return result


def _forecast_rows(data, origin, means, admitted):
    day = dt.date.fromisoformat(origin)
    return [
        (sid, day + dt.timedelta(days=step + 1), float(means[i, step]))
        for i, sid in enumerate(data.series)
        if admitted[i]
        for step in range(means.shape[1])
    ]


def _forecast(panel, origin, horizon, out_dir, n_series, emitted, name=NAME):
    disposition = {"emitted": emitted} if emitted else {}
    declined = n_series * horizon - emitted
    if declined:
        disposition["declined_all_unobserved"] = declined
    return Forecast(
        adapter=name,
        path=str(out_dir),
        origin=origin,
        horizon=horizon,
        n_series=n_series,
        disposition=disposition,
        target_estimand=panel.target_estimand,
        frequency=panel.frequency,
    ).validate()


def fit_predict(
    panel: Panel,
    origin: str,
    horizon: int,
    *,
    spark=None,
    out_dir: str | pathlib.Path,
    max_epochs: int = 2,
    batch_size: int = 32,
    early_stopping_patience: int = 10,
    early_stopping_min_delta: float = 1e-6,
    device: str | None = "cpu",
    seed: int = 0,
    context_days: int = 371,
    group_size: int = 4,
    max_variates: int = 32,
    head_width: int = 64,
    learning_rate: float = 0.0003,
    mean_loss_weight: float = 0.0,
    train_windows: int = 1,
    train_stride: int = 28,
    checkpoint_out: str | pathlib.Path | None = None,
    training_mode: str = "fixed_validation",
    fixed_update_budget: int | None = None,
    validation_data_path: str | pathlib.Path | None = None,
    evaluation_contract_path: str | pathlib.Path | None = None,
    evaluation_contract_sha256: str | None = None,
    precision: str = "32-true",
    _adapter: str = NAME,
) -> Forecast:
    """Select by trailing historical count NLL, or freshly refit for exactly selected_updates.

    The count head runs on CPU with float64 likelihood; ``device`` selects frozen-source
    inference. No model or source gradients are trained on post-origin targets. A refit
    restarts head weights from the same seed; it never reuses a selected fitted head.
    """
    if _adapter not in {NAME, "timesfm_v8"}:
        raise ValueError("unknown TimesFM count variant")
    lora = _adapter == "timesfm_v8"
    schema = SCHEMA if not lora else "autoforecast.timesfm-v8-inference.v1"
    if type(mean_loss_weight) not in (int, float) or mean_loss_weight not in (0, 0.2):
        raise ValueError("mean_loss_weight must be the frozen recipe value 0 or 0.2")
    for name, value in (("train_windows", train_windows), ("train_stride", train_stride)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    online = lora or train_windows > 1
    day = _validate_daily_span(panel, origin, horizon, adapter=_adapter)
    if training_mode == "rolling_backtest":
        training_mode = "fixed_validation"
    _options(
        context_days,
        group_size,
        max_variates,
        head_width,
        learning_rate,
        max_epochs,
        batch_size,
        early_stopping_patience,
        early_stopping_min_delta,
        seed,
        device,
        precision,
        training_mode,
        fixed_update_budget,
    )
    if (evaluation_contract_path is None) != (evaluation_contract_sha256 is None):
        raise ValueError("study evaluation requires both path and digest")
    if evaluation_contract_path is not None and (
        training_mode != "fixed_validation" or validation_data_path is None
    ):
        raise ValueError("study selection requires a separate fixed validation panel")
    from autoforecast.adapters.tft_q import (
        _discard_incomplete_forecast,
        _publish_tft_artifacts,
        _validate_artifact_destinations,
    )

    # Reuse the existing architecture-neutral atomic checkpoint/forecast transaction.
    output = pathlib.Path(out_dir)
    checkpoint = (
        pathlib.Path(checkpoint_out)
        if checkpoint_out
        else output.with_name(output.name + f"-{_adapter.rsplit('_', 1)[-1]}-artifacts")
        / "model.pt"
    )
    _validate_artifact_destinations(output, checkpoint)
    import lightning.pytorch as pl
    import torch
    from forecast.models.timesfm_hnb import CountModule
    from forecast.models.timesfm_v8 import EPOCH_DEFINITION, OnePassProfileDataset
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from torch.utils.data import DataLoader

    from autoforecast.adapters.tft_q import _EpochLog
    from autoforecast.tft_precision import finite_gradient_clip_callback
    from autoforecast.training_checkpoint import configure_training_runtime

    runtime = configure_training_runtime("cpu", 4)
    resolved_device = device
    if device in (None, "auto"):
        resolved_device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    identity = {
        "panel": asdict(panel),
        **_data_identity(panel),
        "origin": origin,
        "horizon": horizon,
        "context_days": context_days,
        "group_size": group_size,
        "max_variates": max_variates,
        "head_width": head_width,
        "learning_rate": learning_rate,
        "mean_loss_weight": mean_loss_weight,
        "train_windows": train_windows,
        "train_stride": train_stride,
        "weight_decay": 0.0001,
        "embedding_dim": 8,
        "gradient_clip": 1.0,
        "seed": seed,
        "batch_size": batch_size,
        "maximum_epochs": max_epochs,
        "patience": early_stopping_patience,
        "min_delta": early_stopping_min_delta,
        "source_device": resolved_device,
        "head_device": "cpu",
        "precision": precision,
        "training_mode": training_mode,
        "fixed_update_budget": fixed_update_budget,
        "epoch_definition": EPOCH_DEFINITION,
    }
    data = load_data(
        panel,
        origin,
        horizon,
        context_days,
        history_days=context_days + 2 * horizon + (train_windows - 1) * train_stride,
    )
    study = None
    if evaluation_contract_path is not None:
        from autoforecast.study_validation import StudyValidation

        study = StudyValidation.read(
            evaluation_contract_path, evaluation_contract_sha256,
            panel=replace(panel, path=str(validation_data_path)), origin=origin, horizon=horizon,
            series_ids=tuple(data.series),
        )
        identity["study_evaluation"] = study.as_record()
    internal_validation = training_mode == "fixed_validation" and study is None
    grouping = groups(data, group_size, max_variates)
    pl.seed_everything(seed, workers=True)
    if lora:
        from autoforecast.timesfm_v8_source import LoRASource

        source = LoRASource(resolved_device)
    else:
        source = TimesFM3Source(resolved_device)
    identity["source"] = source.identity
    if checkpoint.exists() and output.exists():
        saved = _load_checkpoint(checkpoint, file_sha256(checkpoint), schema)
        if saved["run_identity"] != identity or saved["forecast_sha256"] != directory_digest(
            output, suffixes=(".parquet",)
        ):
            raise ValueError("TimesFM-HNB completed artifacts differ from the requested run")
        result = _forecast(
            panel,
            origin,
            horizon,
            output,
            saved["training"]["n_series"],
            saved["emitted"],
            _adapter,
        )
        (output / ".v6-pending").unlink(missing_ok=True)
        return result
    _discard_incomplete_forecast(output, checkpoint)
    if os.path.lexists(checkpoint) or os.path.lexists(output):
        raise ValueError("TimesFM-HNB artifact destinations already exist")
    cache_dir = checkpoint.parent / "source-cache"
    feature_records = []
    train_origin = (
        day - dt.timedelta(days=horizon * (2 if internal_validation else 1))
    ).isoformat()
    datasets = []
    profiles = missing_profiles(
        data.fields, tuple(tuple(family) for family in data.schema.get("families", ()))
    )
    order = [row for group in grouping for row in group]
    validation_origin = (day - dt.timedelta(days=horizon)).isoformat()
    if online:
        from forecast.models.timesfm_v8 import OnlineWindows

        train_origins = tuple(
            (dt.date.fromisoformat(train_origin) - dt.timedelta(days=i * train_stride)).isoformat()
            for i in range(train_windows)
        )
        training = OnlineWindows(
            data, train_origins, horizon, context_days, grouping, profiles, batch_size, seed
        )
        validation = (
            OnlineWindows(
                data, validation_origin, horizon, context_days, grouping, ((),), batch_size, seed
            )
            if internal_validation
            else None
        )
    else:
        for absent in profiles:
            print(
                f"TimesFM-HNB frozen source: training origin={train_origin} absent={absent}",
                flush=True,
            )
            inputs, record = features(
                data,
                train_origin,
                horizon,
                context_days,
                grouping,
                source,
                cache_dir,
                absent=absent,
                spark=spark,
            )
            feature_records.append(record)
            position = data.position(train_origin) + 1
            datasets.append(
                _dataset(
                    inputs,
                    data.target[:, position : position + horizon],
                    data.observed[:, position : position + horizon],
                    order,
                )
            )
        training = OnePassProfileDataset(datasets, batch_size, seed)
        validation = None
        if internal_validation:
            inputs, record = features(
                data,
                validation_origin,
                horizon,
                context_days,
                grouping,
                source,
                cache_dir,
                spark=spark,
            )
            feature_records.append(record)
            position = data.position(validation_origin) + 1
            validation = _dataset(
                inputs,
                data.target[:, position : position + horizon],
                data.observed[:, position : position + horizon],
                order,
            )
    if not 0 < len(training) <= len(data.series):
        raise ValueError("TimesFM epoch must visit each eligible series at most once")
    architecture = {
        "n_features": 10 + 2 * len(data.fields),
        "n_static": data.static.shape[-1],
        "cardinalities": [
            len(data.schema["category_maps"][name]) for name in data.schema["static_names"]
        ],
        "n_series": len(data.series),
        "width": head_width,
        "embedding_dim": 8,
    }
    # Source/cache hit patterns must not change head initialization or sample shuffling.
    pl.seed_everything(seed, workers=True)
    if online:
        from forecast.models.timesfm_v8 import LoRACountModule

        model = LoRACountModule(
            architecture,
            learning_rate,
            identity["weight_decay"],
            mean_loss_weight,
            source,
            None if lora else cache_dir,
        )
    else:
        model = CountModule(architecture, learning_rate, identity["weight_decay"], mean_loss_weight)
    class _EpochSampler(pl.Callback):
        def on_train_epoch_start(self, trainer, _model):
            training.set_epoch(trainer.current_epoch)

    epoch_log = _EpochLog(label="V8" if lora else "TimesFM-HNB")
    callbacks = [_EpochSampler(), epoch_log]
    if study is not None:
        from autoforecast.study_validation import study_score_callback

        def predict_window(module):
            if lora:
                source.seal()
            inputs, record = features(
                data, origin, horizon, context_days, grouping, source, cache_dir, spark=spark,
            )
            if not inputs["admitted"].all():
                raise ValueError("study TimesFM prediction changed the frozen population")
            feature_records.append(record)
            return _means(module.head, inputs, batch_size)

        callbacks.insert(0, study_score_callback(study, predict_window))
    if not online:
        callbacks.append(finite_gradient_clip_callback(precision="32-true"))
    selection = None
    if training_mode == "fixed_validation":
        monitor = "study_score" if study is not None else "val_loss"
        selection = ModelCheckpoint(
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
                selection,
                EarlyStopping(
                    monitor=monitor,
                    mode="min",
                    min_delta=early_stopping_min_delta,
                    patience=early_stopping_patience,
                    check_on_train_epoch_end=study is not None,
                ),
            ]
        )
    steps = math.ceil(len(training) / batch_size)
    trainer = pl.Trainer(
        max_epochs=max_epochs
        if fixed_update_budget is None
        else math.ceil(fixed_update_budget / steps),
        max_steps=-1 if fixed_update_budget is None else fixed_update_budget,
        accelerator="cpu",
        devices=1,
        precision="32-true",
        logger=False,
        enable_checkpointing=selection is not None,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        deterministic=True,
    )
    trainer.fit(
        model,
        train_dataloaders=DataLoader(
            training,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=training.collate if online else None,
        ),
        val_dataloaders=None
        if validation is None
        else DataLoader(
            validation, batch_size=batch_size, collate_fn=validation.collate if online else None
        ),
    )
    if selection is not None:
        if (
            not selection.best_model_path
            or selection.best_model_score is None
            or not torch.isfinite(selection.best_model_score)
        ):
            raise RuntimeError("TimesFM-HNB validation failed to select a finite checkpoint")
        best = torch.load(selection.best_model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(best["state_dict"])
        if lora:
            model.on_load_checkpoint(best)
        selected_updates = int(best["global_step"])
        selected_epoch = int(best["epoch"]) + 1
    else:
        selected_updates = int(trainer.global_step)
        selected_epoch = math.ceil(selected_updates / steps)
        if selected_updates != fixed_update_budget:
            raise RuntimeError("TimesFM-HNB refit did not complete the sealed update budget")
    if lora:
        source.seal()
    inputs, record = features(
        data, origin, horizon, context_days, grouping, source, cache_dir, spark=spark
    )
    feature_records.append(record)
    means = _means(model.head, inputs, batch_size)
    rows = _forecast_rows(data, origin, means, inputs["admitted"])
    current_identity = _data_identity(panel)
    if any(identity[name] != value for name, value in current_identity.items()):
        raise RuntimeError("TimesFM-HNB panel or plan changed during training")
    payload = {
        "schema": schema,
        "state_dict": model.head.state_dict(),
        "architecture": architecture,
        "run_identity": identity,
        "feature_schema": data.schema,
        "groups": grouping,
        "features": feature_records,
        "runtime": runtime,
        "emitted": len(rows),
        "training": {
            "maximum_epochs": max_epochs,
            "n_series": len(data.series),
            "n_training_examples": len(training),
            "epoch_definition": EPOCH_DEFINITION,
            "missing_profiles": [list(x) for x in profiles],
            "selected_epoch": selected_epoch,
            "selected_updates": selected_updates,
            "steps_per_epoch": steps,
            "selection_metric": study.metric if study is not None else (
                "count_nll_mean" if mean_loss_weight else "count_nll"
            ),
            "validation_origin": origin if training_mode == "fixed_validation" else None,
            "refit_optimizer_steps": selected_updates if training_mode == "fixed_refit" else None,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
            "precision": precision,
            "validation": epoch_log.records if training_mode == "fixed_validation" else [],
            "epoch_log": epoch_log.records,
            "training_contract": {
                "mode": training_mode,
                "fixed_update_budget": fixed_update_budget,
                **({"study_evaluation": study.as_record()} if study is not None else {}),
            },
        },
    }
    if lora:
        payload["source_lora_state"] = source.adapter_state()
        payload["trained_source_identity"] = source.identity
    _publish_tft_artifacts(payload, rows=rows, checkpoint=checkpoint, output_dir=output)
    return _forecast(panel, origin, horizon, output, len(data.series), len(rows), _adapter)


def _load_checkpoint(path: pathlib.Path, sha256: str, schema=SCHEMA):
    import hashlib

    import torch

    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("TimesFM-HNB checkpoint digest changed")
    saved = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if type(saved) is not dict or saved.get("schema") != schema:
        raise ValueError("not a TimesFM-HNB inference checkpoint")
    return saved


def predict_from_checkpoint(
    panel: Panel,
    origin: str,
    horizon: int,
    *,
    checkpoint: str | pathlib.Path,
    checkpoint_sha256: str,
    out_dir: str | pathlib.Path,
    overrides: dict[str, np.ndarray] | None = None,
    spark=None,
    _adapter: str = NAME,
) -> Forecast:
    """Forecast/scenario replay with the same head and fixed recorded group membership."""
    _validate_daily_span(panel, origin, horizon, adapter=_adapter)
    from forecast.models.timesfm_hnb import CountHead

    from autoforecast.adapters.tft_q import _write_forecast

    path = pathlib.Path(checkpoint)
    lora = _adapter == "timesfm_v8"
    schema = "autoforecast.timesfm-v8-inference.v1" if lora else SCHEMA
    saved = _load_checkpoint(path, checkpoint_sha256, schema)
    identity = saved["run_identity"]
    if origin < identity["origin"] or horizon != identity["horizon"]:
        raise ValueError(
            "TimesFM-HNB replay cannot precede fitted cutoff or change its trained horizon"
        )
    before = _data_identity(panel)
    if lora:
        from autoforecast.timesfm_v8_source import LoRASource

        source = LoRASource(identity["source_device"])
    else:
        source = TimesFM3Source(identity["source_device"])
    if source.identity != identity["source"]:
        raise ValueError("TimesFM-HNB source identity differs from the trained artifact")
    if lora:
        source.load_adapter(saved["source_lora_state"])
        if source.identity != saved["trained_source_identity"]:
            raise ValueError("V8 LoRA digest differs from the trained artifact")
    data = load_data(
        panel, origin, horizon, identity["context_days"], schema=saved["feature_schema"]
    )
    inputs, _ = features(
        data,
        origin,
        horizon,
        identity["context_days"],
        saved["groups"],
        source,
        path.parent / "source-cache",
        overrides=overrides,
        spark=spark,
    )
    head = CountHead(**saved["architecture"])
    head.load_state_dict(saved["state_dict"], strict=True)
    means = _means(head, inputs, identity["batch_size"])
    if before != _data_identity(panel):
        raise RuntimeError("TimesFM-HNB data changed during prediction")
    rows = _forecast_rows(data, origin, means, inputs["admitted"])
    output = pathlib.Path(out_dir)
    if os.path.lexists(output):
        raise ValueError("TimesFM-HNB prediction destination already exists")
    _write_forecast(rows, output)
    return _forecast(panel, origin, horizon, output, len(data.series), len(rows), _adapter)
