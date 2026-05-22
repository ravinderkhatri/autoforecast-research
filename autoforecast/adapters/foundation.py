"""TimesFM 2.5 foundation model as a univariate, process-isolated comparator.

The adapter owns only the boundary that TimesFM does not know about: forecast origins,
observation states, the repository's complete-cell account, and Parquet output.  It does not
reimplement the model.  ``timesfm`` remains optional and is imported only when a real provider is
requested; functional tests inject a provider through the same public ``fit_predict`` call.

TimesFM 2.5 has two different point-like outputs.  The first array returned by ``forecast`` is the
median.  The mean is ``quantile_forecast[..., 0]``.  They are deliberately selected by name below
because silently treating the median as the mean would rank a different estimand with WRMSSE.
"""

from __future__ import annotations

import datetime as dt
import importlib
import math
import os
import pathlib
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from typing import Literal, Protocol

import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from autoforecast.adapters import _validate_daily_span
from autoforecast.adapters.timesfm_transformers import (
    TRANSFORMERS_MODEL_ID,
    TRANSFORMERS_REVISION,
    TimesFMTransformersBackend,
)
from autoforecast.benchmarks.m5_contract import M5_HORIZON_DAYS
from autoforecast.benchmarks.m5_timesfm_contract import (
    TIMESFM_2P5_BATCH_SIZE,
    TIMESFM_2P5_CONTEXT_DAYS,
    TIMESFM_2P5_HF_REVISION,
    TIMESFM_2P5_MODEL_ID,
    validate_timesfm_model_policy,
)
from autoforecast.panel import ELIGIBLE_STATES, Forecast, Panel

NAME = "timesfm_standalone"
MODEL_ID = TIMESFM_2P5_MODEL_ID
MODEL_REVISION = TIMESFM_2P5_HF_REVISION
CONTEXT_DAYS = TIMESFM_2P5_CONTEXT_DAYS
BATCH_SIZE = TIMESFM_2P5_BATCH_SIZE

EMITTED = "emitted"
EMITTED_CLIPPED = "emitted_clipped_negative"
DECLINED_ALL_UNOBSERVED = "declined_all_unobserved"

_OUTPUT_SCHEMA = pa.schema([
    pa.field("series_id", pa.string(), nullable=False),
    pa.field("date", pa.date32(), nullable=False),
    pa.field("yhat", pa.float64(), nullable=False),
    pa.field("timesfm_statistic", pa.string(), nullable=False),
])


class _Backend(Protocol):
    """The smallest injectable seam around the maintained provider."""

    def forecast(
        self, inputs: Sequence[Sequence[float]], horizon: int,
    ) -> tuple[object, object]: ...


class _TimesFM25Provider:
    """Thin wrapper over Google's maintained TimesFM 2.5 PyTorch API."""

    def __init__(
        self,
        *,
        model_id: str,
        model_revision: str,
        context_days: int,
        horizon: int,
        batch_size: int,
    ) -> None:
        module = _timesfm_module()
        model = module.TimesFM_2p5_200M_torch.from_pretrained(
            model_id,
            revision=model_revision,
            force_download=False,
            local_files_only=True,
        )
        model.compile(module.ForecastConfig(
            max_context=context_days,
            max_horizon=horizon,
            normalize_inputs=True,
            per_core_batch_size=batch_size,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        ))
        self._model = model

    def forecast(
        self, inputs: Sequence[Sequence[float]], horizon: int,
    ) -> tuple[object, object]:
        import numpy as np

        arrays = [np.asarray(values, dtype=np.float32) for values in inputs]
        return self._model.forecast(horizon=horizon, inputs=arrays)


def _timesfm_module():
    """Load only the current provider API; old package APIs are intentionally unsupported."""
    try:
        module = importlib.import_module("timesfm")
    except ImportError as exc:
        raise ImportError(
            "TimesFM is an optional dependency. Install a pinned TimesFM 2.5 source/runtime "
            "environment for this adapter; the core forecast package does not require it."
        ) from exc
    missing = [name for name in ("TimesFM_2p5_200M_torch", "ForecastConfig")
               if not hasattr(module, name)]
    if missing:
        raise ImportError(
            "the installed timesfm package does not expose the TimesFM 2.5 API "
            f"{missing}; legacy 1.x/2.0 provider APIs are not supported")
    return module


def _provider_module():
    """Preflight either the governed Transformers runtime or parity-only official runtime."""
    try:
        module = importlib.import_module("transformers")
    except ImportError:
        module = None
    if module is not None and hasattr(module, "TimesFm2_5ModelForPrediction"):
        return module
    return _timesfm_module()


def _validate_options(
    *,
    horizon: int,
    context_days: int,
    batch_size: int,
    forecast_statistic: str,
    backend: _Backend | None,
    model_id: str,
    model_revision: str | None,
    provider_family: str,
    provider_checkpoint_dir: str | pathlib.Path | None,
    provider_device: str | None,
    lora_adapter_dir: str | pathlib.Path | None,
    lora_adapter_sha256: str | None,
    runtime_lock_path: str | pathlib.Path | None,
    runtime_lock_file_sha256: str | None,
    runtime_lock_record_path: Sequence[str] | None,
    runtime_lock_sha256: str | None,
    recorded_runtime_lock: Mapping[str, object] | None,
    recorded_adapter_dir: str | pathlib.Path | None,
) -> None:
    for name, value in (("horizon", horizon), ("context_days", context_days),
                        ("batch_size", batch_size)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if forecast_statistic not in ("mean", "median"):
        raise ValueError(
            "forecast_statistic must be 'mean' or 'median'; TimesFM 2.5 returns them in "
            f"different arrays, got {forecast_statistic!r}")
    if backend is not None and not callable(getattr(backend, "forecast", None)):
        raise TypeError("backend must provide forecast(inputs, horizon)")
    governed_provider = backend is None or getattr(backend, "governed_provider", False) is True
    if governed_provider and horizon != M5_HORIZON_DAYS:
        raise ValueError(
            f"a governed TimesFM provider requires horizon={M5_HORIZON_DAYS}; "
            "alternate horizons are diagnostic, injected-backend tests only"
        )
    if governed_provider and forecast_statistic != "mean":
        raise ValueError(
            "a governed TimesFM provider requires forecast_statistic='mean'; median is a "
            "named diagnostic output, not the WRMSSE expected-demand estimand"
        )
    if backend is None:
        if provider_family not in ("transformers", "official_pytorch_reference"):
            raise ValueError(
                "provider_family must be 'transformers' or 'official_pytorch_reference'"
            )
        if context_days != CONTEXT_DAYS or batch_size != BATCH_SIZE:
            raise ValueError(
                "a real TimesFM 2.5 run requires the registered context_days=1024 and "
                "batch_size=32; alternate inference configurations are not evidence")
        if provider_family == "transformers":
            if model_id != TRANSFORMERS_MODEL_ID or model_revision != TRANSFORMERS_REVISION:
                raise ValueError(
                    "the governed Transformers route requires its exact model ID and immutable "
                    "checkpoint revision"
                )
            if (
                provider_checkpoint_dir is None
                or provider_device is None
                or runtime_lock_path is None
                or runtime_lock_file_sha256 is None
                or runtime_lock_record_path is None
                or runtime_lock_sha256 is None
            ):
                raise ValueError(
                    "the Transformers route requires inert provider_checkpoint_dir and "
                    "provider_device plus the complete runtime-lock source identity"
                )
        else:
            validate_timesfm_model_policy(model_id)
            if model_id != MODEL_ID or model_revision != MODEL_REVISION:
                raise ValueError(
                    "the official PyTorch reference route requires the exact registered model "
                    "ID and immutable checkpoint revision"
                )
            if any(
                value is not None
                for value in (
                    provider_checkpoint_dir,
                    provider_device,
                    lora_adapter_dir,
                    lora_adapter_sha256,
                    runtime_lock_path,
                    runtime_lock_file_sha256,
                    runtime_lock_record_path,
                    runtime_lock_sha256,
                    recorded_runtime_lock,
                    recorded_adapter_dir,
                )
            ):
                raise ValueError(
                    "official_pytorch_reference does not accept Transformers/LoRA configuration"
                )


def _canonical_source(
    panel: Panel, spark: SparkSession, origin: str, context_days: int,
) -> tuple[tuple[str, ...], DataFrame]:
    """Return requested identities and the sorted, origin-bounded context rows."""
    if panel.status_prob_cols is not None:
        raise ValueError(
            "TimesFM standalone does not support soft observation-state probabilities")
    try:
        origin_day = dt.date.fromisoformat(origin)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"origin must be ISO YYYY-MM-DD, got {origin!r}") from exc
    if origin_day.isoformat() != origin:
        raise ValueError(f"origin must be canonical ISO YYYY-MM-DD, got {origin!r}")
    start = origin_day - dt.timedelta(days=context_days - 1)

    raw = spark.read.parquet(panel.path)
    dated = raw.withColumn(
        "_timesfm_date", F.expr(f"try_cast(`{panel.time_col}` as date)"))
    if dated.filter(F.col("_timesfm_date").isNull()).limit(1).count():
        raise ValueError("TimesFM input contains a null or invalid calendar date")

    through_origin = dated.filter(F.col("_timesfm_date") <= F.lit(origin))
    identities = (through_origin
                  .select(F.col(panel.series_col).cast("string").alias("series_id"))
                  .distinct().collect())
    series_ids = tuple(sorted(row["series_id"] for row in identities
                              if row["series_id"] is not None
                              and row["series_id"].strip()))
    if len(series_ids) != len(identities):
        raise ValueError("TimesFM input contains a null or blank series identifier")
    if not series_ids:
        raise ValueError(f"no series exists at or before forecast origin {origin}")

    selected = [
        F.col(panel.series_col).cast("string").alias("series_id"),
        F.col("_timesfm_date").alias("date"),
        F.expr(f"try_cast(`{panel.target_col}` as double)").alias("target"),
    ]
    if panel.has_exposure:
        selected.append(F.col(panel.exposure_col).cast("string").alias("state"))
    else:
        selected.append(F.lit("exact").alias("state"))
    source_partitions = int(spark.conf.get("spark.sql.shuffle.partitions"))
    if source_partitions < 1:
        raise ValueError("spark.sql.shuffle.partitions must be a positive integer")
    contexts = (through_origin
                .filter(F.col("_timesfm_date") >= F.lit(start.isoformat()))
                .select(*selected)
                .repartitionByRange(source_partitions, "series_id", "date")
                .sortWithinPartitions("series_id", "date"))
    return series_ids, contexts


def _series_rows(frame: DataFrame, staging_dir: pathlib.Path):
    """Stream sorted contexts through private Parquet, not Spark task results."""
    if staging_dir.exists():
        raise ValueError(f"TimesFM context staging path already exists: {staging_dir}")
    frame.write.mode("errorifexists").parquet(str(staging_dir))
    parts = sorted(staging_dir.glob("part-*.parquet"))
    if not parts or any(not part.is_file() or part.is_symlink() for part in parts):
        raise RuntimeError("TimesFM context staging produced no regular Parquet parts")

    current: str | None = None
    rows = []
    failed = False
    try:
        for part in parts:
            parquet = pq.ParquetFile(part)
            for batch in parquet.iter_batches(
                batch_size=8192,
                columns=["series_id", "date", "target", "state"],
            ):
                for row in batch.to_pylist():
                    sid = row["series_id"]
                    if current is not None and sid != current:
                        yield current, rows
                        rows = []
                    current = sid
                    rows.append(row)
        if current is not None:
            yield current, rows
    except BaseException:
        failed = True
        raise
    finally:
        try:
            shutil.rmtree(staging_dir)
        except OSError:
            if not failed:
                raise


def _context_for_series(
    series_id: str, rows: list[object], origin: dt.date, negative_policy: str,
) -> tuple[float, ...] | None:
    """Trim leading unknowns and reject any unknown or calendar gap after observation begins."""
    seen_dates: set[dt.date] = set()
    first_observed: int | None = None
    allowed_states = {*ELIGIBLE_STATES, "unobserved"}
    for index, row in enumerate(rows):
        day = row["date"]
        if day in seen_dates:
            raise ValueError(f"TimesFM input has duplicate key {(series_id, day)!r}")
        seen_dates.add(day)
        state = row["state"]
        if state not in allowed_states:
            raise ValueError(
                f"TimesFM input has unknown observation state {state!r} for {series_id}")
        if first_observed is None and state in ELIGIBLE_STATES:
            first_observed = index
    if first_observed is None:
        return None

    observed = rows[first_observed:]
    previous: dt.date | None = None
    values: list[float] = []
    for row in observed:
        day, state = row["date"], row["state"]
        if state == "unobserved":
            raise ValueError(
                f"TimesFM input for {series_id} has an unobserved gap after observed history "
                f"begins ({day}); compressing it would change calendar time")
        if previous is not None and day != previous + dt.timedelta(days=1):
            raise ValueError(
                f"TimesFM input for {series_id} is not consecutive daily history after "
                f"observation begins: {previous} then {day}")
        previous = day
        value = row["target"]
        if isinstance(value, bool):
            raise ValueError(f"TimesFM target for {series_id} on {day} is boolean")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"TimesFM target for {series_id} on {day} is not numeric: {value!r}") from exc
        if not math.isfinite(number):
            raise ValueError(
                f"TimesFM target for {series_id} on {day} is non-finite: {value!r}")
        if number < 0.0:
            if negative_policy == "floor":
                number = 0.0
            else:
                raise ValueError(
                    f"TimesFM target for {series_id} on {day} is negative: {number}")
        values.append(number)
    if previous != origin:
        raise ValueError(
            f"TimesFM input for {series_id} ends at {previous}, not forecast origin {origin}; "
            "forecast dates would otherwise be shifted")
    return tuple(values)


def _as_lists(value: object, label: str) -> list:
    """Convert provider arrays to inert nested lists without accepting an opaque result."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list | tuple):
        raise ValueError(f"TimesFM {label} output must be an array, got {type(value).__name__}")
    return list(value)


def _finite_output(value: object, label: str) -> float:
    """Validate one provider scalar without letting coercion hide malformed evidence."""
    if isinstance(value, bool):
        raise ValueError(f"TimesFM {label} output is boolean")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"TimesFM {label} output is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"TimesFM {label} output is non-finite: {value!r}")
    return number


def _select_forecast(
    result: object,
    *,
    n_series: int,
    horizon: int,
    statistic: Literal["mean", "median"],
) -> list[list[float]]:
    """Validate the exact 2.5 output contract and select mean or median by its true route."""
    if not isinstance(result, tuple | list) or len(result) != 2:
        raise ValueError("TimesFM backend must return (median, quantile_forecast)")
    medians = _as_lists(result[0], "median")
    quantiles = _as_lists(result[1], "quantile")
    if len(medians) != n_series or len(quantiles) != n_series:
        raise ValueError(
            "TimesFM output batch dimension does not match its inputs: "
            f"median={len(medians)}, quantile={len(quantiles)}, expected={n_series}")

    selected: list[list[float]] = []
    for index in range(n_series):
        median_row = _as_lists(medians[index], f"median[{index}]")
        quantile_row = _as_lists(quantiles[index], f"quantile[{index}]")
        if len(median_row) != horizon or len(quantile_row) != horizon:
            raise ValueError(
                f"TimesFM output horizon for series {index} is median={len(median_row)}, "
                f"quantile={len(quantile_row)}, expected={horizon}")
        row: list[float] = []
        for step in range(horizon):
            bands = _as_lists(quantile_row[step], f"quantile[{index},{step}]")
            if len(bands) != 10:
                raise ValueError(
                    "TimesFM 2.5 quantile output must have 10 channels "
                    "[mean, q10, ..., q90]")
            median = _finite_output(median_row[step], f"median[{index},{step}]")
            channels = [
                _finite_output(value, f"quantile[{index},{step},{channel}]")
                for channel, value in enumerate(bands)
            ]
            row.append(channels[0] if statistic == "mean" else median)
        selected.append(row)
    return selected


def _write_batch(
    writer: pq.ParquetWriter,
    series_ids: Sequence[str],
    forecasts: Sequence[Sequence[float]],
    origin: dt.date,
    horizon: int,
    statistic: str,
) -> tuple[int, int]:
    ids: list[str] = []
    dates: list[dt.date] = []
    values: list[float] = []
    labels: list[str] = []
    clipped = 0
    for sid, forecast in zip(series_ids, forecasts, strict=True):
        for step in range(horizon):
            value = forecast[step]
            if value < 0.0:
                value = 0.0
                clipped += 1
            ids.append(sid)
            dates.append(origin + dt.timedelta(days=step + 1))
            values.append(value)
            labels.append(statistic)
    writer.write_table(pa.Table.from_arrays([
        pa.array(ids, type=pa.string()),
        pa.array(dates, type=pa.date32()),
        pa.array(values, type=pa.float64()),
        pa.array(labels, type=pa.string()),
    ], schema=_OUTPUT_SCHEMA))
    return len(values) - clipped, clipped


def fit_predict(
    panel: Panel,
    origin: str,
    horizon: int,
    *,
    spark: SparkSession,
    out_dir: str | pathlib.Path,
    context_days: int = CONTEXT_DAYS,
    batch_size: int = BATCH_SIZE,
    forecast_statistic: Literal["mean", "median"] = "mean",
    model_id: str = MODEL_ID,
    model_revision: str | None = MODEL_REVISION,
    provider_family: Literal[
        "transformers", "official_pytorch_reference"
    ] = "official_pytorch_reference",
    provider_checkpoint_dir: str | pathlib.Path | None = None,
    provider_device: Literal["cpu", "mps"] | None = None,
    lora_adapter_dir: str | pathlib.Path | None = None,
    lora_adapter_sha256: str | None = None,
    runtime_lock_path: str | pathlib.Path | None = None,
    runtime_lock_file_sha256: str | None = None,
    runtime_lock_record_path: Sequence[str] | None = None,
    runtime_lock_sha256: str | None = None,
    recorded_runtime_lock: Mapping[str, object] | None = None,
    recorded_adapter_dir: str | pathlib.Path | None = None,
    backend: _Backend | None = None,
) -> Forecast:
    """Forecast with TimesFM 2.5 using only target history available at ``origin``.

    ``backend`` is an injection seam for functional tests and offline providers.  Production
    isolated calls omit it and construct the pinned TimesFM provider inside the worker process.
    No covariate arguments exist on purpose: a covariate-enabled experiment is a different model
    and needs its own point-in-time contract.
    """
    _validate_daily_span(panel, origin, horizon, adapter=NAME)
    _validate_options(
        horizon=horizon,
        context_days=context_days,
        batch_size=batch_size,
        forecast_statistic=forecast_statistic,
        backend=backend,
        model_id=model_id,
        model_revision=model_revision,
        provider_family=provider_family,
        provider_checkpoint_dir=provider_checkpoint_dir,
        provider_device=provider_device,
        lora_adapter_dir=lora_adapter_dir,
        lora_adapter_sha256=lora_adapter_sha256,
        runtime_lock_path=runtime_lock_path,
        runtime_lock_file_sha256=runtime_lock_file_sha256,
        runtime_lock_record_path=runtime_lock_record_path,
        runtime_lock_sha256=runtime_lock_sha256,
        recorded_runtime_lock=recorded_runtime_lock,
        recorded_adapter_dir=recorded_adapter_dir,
    )
    provider = backend
    if provider is None and provider_family == "transformers":
        provider = TimesFMTransformersBackend(
            checkpoint_dir=provider_checkpoint_dir or "",
            model_revision=model_revision or "",
            context_length=context_days,
            device=provider_device or "",
            adapter_dir=lora_adapter_dir,
            adapter_sha256=lora_adapter_sha256,
            forecast_origin=origin if lora_adapter_dir is not None else None,
            runtime_lock_path=runtime_lock_path or "",
            runtime_lock_file_sha256=runtime_lock_file_sha256 or "",
            runtime_lock_record_path=runtime_lock_record_path or (),
            runtime_lock_sha256=runtime_lock_sha256 or "",
            recorded_runtime_lock=recorded_runtime_lock,
            recorded_adapter_dir=recorded_adapter_dir,
        )
    origin_validator = getattr(provider, "validate_origin", None)
    if callable(origin_validator):
        origin_validator(origin)
    if panel.target_estimand != "sales_given_availability":
        raise ValueError(
            "TimesFM standalone supports sales_given_availability only; it has no censored "
            "likelihood for unconstrained demand")
    target = pathlib.Path(out_dir)
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError(
            f"TimesFM output path must be absent or an empty directory, got {target}")
    target.parent.mkdir(parents=True, exist_ok=True)

    requested, contexts = _canonical_source(panel, spark, origin, context_days)
    origin_day = dt.date.fromisoformat(origin)

    temporary = pathlib.Path(tempfile.mkdtemp(
        dir=target.parent, prefix=f".{target.name}.timesfm."))
    writer = pq.ParquetWriter(temporary / "part-00000.parquet", _OUTPUT_SCHEMA)
    emitted = 0
    clipped = 0
    declined_series = 0
    seen: set[str] = set()
    batch_ids: list[str] = []
    batch_contexts: list[tuple[float, ...]] = []

    def flush() -> None:
        nonlocal emitted, clipped, provider
        if not batch_ids:
            return
        if provider is None:
            provider = _TimesFM25Provider(
                model_id=model_id,
                model_revision=model_revision or "",  # exact revision validated above
                context_days=context_days,
                horizon=horizon,
                batch_size=batch_size,
            )
        selected = _select_forecast(
            provider.forecast(batch_contexts, horizon),
            n_series=len(batch_ids),
            horizon=horizon,
            statistic=forecast_statistic,
        )
        ordinary, clipped_now = _write_batch(
            writer, batch_ids, selected, origin_day, horizon, forecast_statistic)
        emitted += ordinary
        clipped += clipped_now
        batch_ids.clear()
        batch_contexts.clear()

    try:
        for sid, rows in _series_rows(contexts, temporary / ".canonical-source"):
            seen.add(sid)
            values = _context_for_series(
                sid, rows, origin_day, panel.negative_target_policy)
            if values is None:
                declined_series += 1
                continue
            batch_ids.append(sid)
            batch_contexts.append(values)
            if len(batch_ids) == batch_size:
                flush()
        declined_series += len(set(requested) - seen)
        flush()
        writer.close()
        if target.exists():
            target.rmdir()  # validated empty before inference
        os.replace(temporary, target)
    except BaseException:
        writer.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    disposition: dict[str, int] = {}
    if emitted:
        disposition[EMITTED] = emitted
    if clipped:
        disposition[EMITTED_CLIPPED] = clipped
    if declined_series:
        disposition[DECLINED_ALL_UNOBSERVED] = declined_series * horizon
    return Forecast(
        adapter=NAME,
        path=str(target),
        origin=origin,
        horizon=horizon,
        n_series=len(requested),
        disposition=disposition,
        target_estimand=panel.target_estimand,
        frequency=panel.frequency,
    ).validate()


__all__ = [
    "BATCH_SIZE", "CONTEXT_DAYS", "MODEL_ID", "MODEL_REVISION", "NAME", "fit_predict",
]
