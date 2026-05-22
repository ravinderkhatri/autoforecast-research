"""Off-the-shelf Temporal Fusion Transformer, through `pytorch-forecasting`.

The comparator Phase 3 exists to produce. **If stock TFT matches a tuned custom model, the custom
model is not justified** -- a result that is cheap to obtain and expensive to ignore, which is why
this adapter is deliberately thin. Nothing here improves on the reference implementation; the
comparison is only meaningful against it.

WHY THE HANDICAPS ARE SPLIT IN TWO. Phase 3 requires them recorded as STRUCTURAL versus
CONFIGURATION, and the distinction decides what a loss means. A structural handicap is something
the architecture cannot express at all, so beating TFT on it says the architecture matters. A
configuration handicap is a knob we set, so beating TFT on one of those says only that we tuned
better -- and under Phase 3's fairness conditions we are not permitted to.
"""

from __future__ import annotations

import datetime as dt
import pathlib

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from autoforecast.adapters import _validate_daily_span
from autoforecast.panel import ELIGIBLE_STATES, Forecast, Panel
from autoforecast.tft_precision import (
    finite_gradient_clip_callback,
    mixed_precision_tft_class,
)

NAME = "tft"

EMITTED = "emitted"
DECLINED_SHORT_HISTORY = "declined_history_below_encoder_length"

STRUCTURAL_HANDICAPS = (
    "no censoring contract: a stockout day enters the loss as observed demand, so the model is "
    "trained toward the suppressed number",
    "no coherent sample paths: it emits marginal quantiles per cell, so a lead-time sum cannot "
    "be calibrated from its output at all",
    "no learned event window: event effects are whatever the covariate encoding carries, with no "
    "ordered gate deciding how far an effect reaches",
)
"""What the architecture cannot express. Beating TFT on these is evidence about architecture."""

CONFIGURATION_HANDICAPS = (
    "loss: QuantileLoss rather than a count likelihood, because that is the reference default",
    "capacity: reference-scale hidden size and attention heads, not tuned",
)
"""Knobs we set. Beating TFT on these is evidence about tuning, which Phase 3 forbids claiming."""

HANDICAP = "; ".join(
    (
        "STRUCTURAL: " + " | ".join(STRUCTURAL_HANDICAPS),
        "CONFIGURATION: " + " | ".join(CONFIGURATION_HANDICAPS),
    )
)

ENCODER_DAYS = 180
"""History the model reads per series. The reference examples use several multiples of the
horizon; 180 days covers two quarters and every weekday-of-month position."""


def _select_accelerator(torch_module, *, precision: str, device: str | None) -> str:
    """Resolve only automatic requests; explicit device choices never silently fall back."""
    if device not in (None, "cpu", "cuda", "mps"):
        raise ValueError("TFT device must be cpu, cuda, mps, or null")
    accelerator = device or (
        "cuda" if torch_module.cuda.is_available()
        else "mps" if torch_module.backends.mps.is_available() else "cpu"
    )
    if accelerator == "cuda" and not torch_module.cuda.is_available():
        raise ValueError("TFT CUDA device is unavailable")
    if accelerator == "mps" and not torch_module.backends.mps.is_available():
        raise ValueError("TFT MPS device is unavailable")
    if precision == "16-mixed" and accelerator == "cpu":
        raise ValueError("TFT float16 mixed precision needs CUDA or MPS hardware")
    return accelerator


def fit_predict(
    panel: Panel,
    origin: str,
    horizon: int,
    *,
    spark: SparkSession,
    out_dir: str | pathlib.Path,
    max_epochs: int = 10,
    batch_size: int = 128,
    seed: int = 0,
    hidden_size: int = 16,
    attention_head_size: int = 4,
    dropout: float = 0.1,
    learning_rate: float = 0.001,
    precision: str = "32-true",
    device: str | None = None,
    checkpoint_out: str | pathlib.Path | None = None,
) -> Forecast:
    """Train a TFT on eligible history up to `origin` and forecast `horizon` days.

    Args:
        panel: Data. Series with fewer than `ENCODER_DAYS + horizon` eligible observations are
            declined -- `TimeSeriesDataSet` cannot build an encoder window for them.
        origin: Last date the model may see, ISO.
        horizon: Days to forecast.
        spark: Active session, used to read the panel and write the result.
        out_dir: Where the forecast Parquet is written.
        max_epochs: Training epochs. Part of the tuning budget Phase 3 requires to be equal
            across compared models, so it is an argument and gets recorded, never a hidden default.
        batch_size: Training batch size.
        seed: Torch seed, so a gate is reproducible.
        hidden_size: TFT hidden representation width from the declared model profile.
        attention_head_size: Number of attention heads from the declared model profile.
        dropout: Dropout probability from the declared model profile.
        learning_rate: Initial optimizer learning rate from the declared model profile.
        checkpoint_out: Optional empty file path for an inference-only Torch state artifact.

    Returns:
        A validated `Forecast`.

    Raises:
        ValueError: non-positive horizon, or no series long enough to build an encoder window.
        ImportError: `pytorch-forecasting` is not installed. It is an OPTIONAL extra -- this
            adapter is the only thing that needs it, and no gate before Phase 3 depends on it.
    """
    _validate_daily_span(panel, origin, horizon, adapter=NAME)
    _validate_training_options(
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        learning_rate=learning_rate,
    )
    import lightning.pytorch as pl
    import torch
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss

    if precision not in ("32-true", "16-mixed"):
        raise ValueError("TFT precision must be 32-true or 16-mixed")
    accelerator = _select_accelerator(torch, precision=precision, device=device)

    torch.manual_seed(seed)
    frame, n_series, n_declined = _training_frame(panel, spark, origin, horizon)
    if frame.empty:
        raise ValueError(
            f"no series has {ENCODER_DAYS + horizon} eligible observations at or before {origin}, "
            "so no encoder window can be built"
        )

    cutoff = int(frame["time_idx"].max()) - horizon
    training = TimeSeriesDataSet(
        frame[frame.time_idx <= cutoff],
        time_idx="time_idx",
        target="y",
        group_ids=["series_id"],
        max_encoder_length=ENCODER_DAYS,
        max_prediction_length=horizon,
        time_varying_known_reals=["time_idx", "dow", "dom", "month"],
        time_varying_unknown_reals=["y"],
        static_categoricals=["series_id"],
        target_normalizer=GroupNormalizer(groups=["series_id"], transformation="softplus"),
        allow_missing_timesteps=True,
    )
    successful_updates = {"count": 0, "retries": 0}
    model_class = (
        mixed_precision_tft_class(TemporalFusionTransformer, successful_updates, accelerator)
        if precision == "16-mixed" else TemporalFusionTransformer
    )
    model = model_class.from_dataset(
        training,
        loss=QuantileLoss(),
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        learning_rate=learning_rate,
        **({"mask_bias": float("-inf")} if precision == "16-mixed" else {}),
    )
    callbacks = (
        [finite_gradient_clip_callback(precision=precision)]
        if precision == "16-mixed" else []
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
        accelerator=accelerator,
        precision=precision,
        callbacks=callbacks,
    )
    trainer.fit(model, train_dataloaders=training.to_dataloader(train=True, batch_size=batch_size))
    if precision == "16-mixed" and successful_updates["count"] != trainer.global_step:
        raise RuntimeError("TFT AMP counted a batch without a successful optimizer update")

    predict_set = TimeSeriesDataSet.from_dataset(
        training, frame, predict=True, stop_randomization=True
    )
    # ``BaseModel.predict`` otherwise creates a fresh Lightning trainer with its default
    # TensorBoard logger.  The isolated worker intentionally executes from a read-only source
    # snapshot, so that default tries to create ``lightning_logs`` beside source and fails.
    # Training telemetry belongs to the governed attempt directory, not the immutable snapshot.
    raw = model.predict(
        predict_set.to_dataloader(train=False, batch_size=batch_size),
        return_index=True,
        trainer_kwargs={
            "logger": False, "enable_progress_bar": False,
            "accelerator": accelerator, "precision": precision,
        },
    )
    out = _to_forecast_frame(raw, frame, origin, horizon)

    if checkpoint_out is not None:
        target = pathlib.Path(checkpoint_out)
        if target.exists():
            raise ValueError(f"checkpoint path must not already exist: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": "autoforecast.tft-inference-state.v1",
                "state_dict": model.state_dict(),
                "hyperparameters": {
                    "hidden_size": hidden_size,
                    "attention_head_size": attention_head_size,
                    "dropout": dropout,
                    "learning_rate": learning_rate,
                    **({"precision": precision, "amp_overflow_retries":
                        successful_updates["retries"]} if precision != "32-true" else {}),
                    "requested_device": device,
                    "resolved_device": accelerator,
                },
            },
            target,
        )

    spark.createDataFrame(out).write.mode("overwrite").parquet(str(out_dir))
    disposition = {EMITTED: int(len(out))}
    if n_declined:
        disposition[DECLINED_SHORT_HISTORY] = n_declined
    return Forecast(
        adapter=NAME,
        path=str(out_dir),
        origin=origin,
        horizon=horizon,
        n_series=n_series,
        disposition=disposition,
        target_estimand=panel.target_estimand,
        frequency=panel.frequency,
    ).validate()


def _validate_training_options(
    *,
    hidden_size: int,
    attention_head_size: int,
    dropout: float,
    learning_rate: float,
) -> None:
    """Reject malformed profile/HPO settings before constructing the Lightning model."""
    if type(hidden_size) is not int or hidden_size < 1:
        raise ValueError("hidden_size must be a positive integer")
    if type(attention_head_size) is not int or attention_head_size < 1:
        raise ValueError("attention_head_size must be a positive integer")
    if (
        not isinstance(dropout, int | float)
        or isinstance(dropout, bool)
        or not 0 <= float(dropout) < 1
    ):
        raise ValueError("dropout must be a finite number in [0, 1)")
    if (
        not isinstance(learning_rate, int | float)
        or isinstance(learning_rate, bool)
        or not 0 < float(learning_rate) <= 1
    ):
        raise ValueError("learning_rate must be a finite number in (0, 1]")


def _training_frame(panel: Panel, spark: SparkSession, origin: str, horizon: int):
    """Eligible history to `origin` as a pandas frame, plus the series counts.

    pytorch-forecasting is single-machine and takes a DataFrame, so the panel is pulled
    to the driver here. That is the ceiling for this adapter and it is the reference
    implementation's, not ours -- the comparison is only meaningful against the reference.
    Upgrade trigger: a panel whose eligible history does not fit driver memory; at that point the
    comparison must use a stated sample that fits driver memory.
    """
    df = spark.read.parquet(panel.path).filter(F.col(panel.time_col) <= F.to_date(F.lit(origin)))
    if panel.has_exposure:
        df = df.filter(F.col(panel.exposure_col).isin(*sorted(ELIGIBLE_STATES)))
    df = df.select(
        F.col(panel.series_col).alias("series_id"),
        F.col(panel.time_col).alias("d"),
        F.col(panel.target_col).cast("double").alias("y"),
    )

    from pyspark.sql import Window

    w = Window.partitionBy("series_id")
    need = ENCODER_DAYS + horizon
    counted = df.withColumn("_n", F.count(F.lit(1)).over(w))
    total = counted.select("series_id").distinct().count()
    keep = counted.filter(F.col("_n") >= need).drop("_n")
    kept = keep.select("series_id").distinct().count()

    frame = keep.orderBy("series_id", "d").toPandas()
    if not frame.empty:
        start = frame["d"].min()
        frame["time_idx"] = (
            (frame["d"] - start).dt.days.astype("int64")
            if hasattr(frame["d"], "dt")
            else [(d - start).days for d in frame["d"]]
        )
        frame["dow"] = [d.weekday() for d in frame["d"]]
        frame["dom"] = [d.day for d in frame["d"]]
        frame["month"] = [d.month for d in frame["d"]]
        frame["series_id"] = frame["series_id"].astype(str)
    return frame, total, (total - kept) * horizon


def _to_forecast_frame(raw, frame, origin: str, horizon: int):
    """Turn the model's output into `series_id · date · yhat`.

    `predict(..., return_index=True)` gives predictions and the index they belong to separately,
    and pairing them by position is the step where a TFT comparison usually goes silently wrong.
    """
    import pandas as pd

    preds, index = raw.output, raw.index
    values = preds.detach().cpu().numpy() if hasattr(preds, "detach") else preds
    d0 = dt.date.fromisoformat(origin)
    rows = []
    for pos, sid in enumerate(index["series_id"].to_numpy()):
        for step in range(horizon):
            rows.append(
                (str(sid), d0 + dt.timedelta(days=step + 1), float(max(values[pos][step], 0.0)))
            )
    return pd.DataFrame(rows, columns=["series_id", "date", "yhat"])


__all__ = [
    "CONFIGURATION_HANDICAPS",
    "ENCODER_DAYS",
    "HANDICAP",
    "NAME",
    "STRUCTURAL_HANDICAPS",
    "fit_predict",
]
