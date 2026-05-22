"""Cached, model-neutral hierarchy arithmetic for bounded development panels.

Aggregation levels are inputs, not model knowledge. The Spark scorer remains the production
reference. ``test_hierarchy_metric.py`` checks arithmetic, refusal and gradient parity.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import torch

from autoforecast.frequency import canonical_origin, parse_frequency, parse_origin, period_points


def read_forecast_grid(
    path: str,
    series_ids: tuple[str, ...],
    origin: str,
    horizon: int,
    *,
    series_col: str,
    time_col: str,
    value_col: str,
    filter_actuals: bool = False,
    frequency: str = "D",
) -> np.ndarray:
    """Read a complete keyed grid, rejecting duplicates, extras, gaps and invalid values.

    Actuals may contain historical rows; forecasts must contain exactly the requested grid.
    ``test_read_forecast_grid`` covers the ordering/coverage boundary independently of scoring.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    from autoforecast.parquet_time import validate_parquet_time_storage

    if (
        type(horizon) is not int
        or horizon < 1
        or not series_ids
        or any(
            not isinstance(series_id, str) or not series_id.strip() for series_id in series_ids
        )
        or len(set(series_ids)) != len(series_ids)
    ):
        raise ValueError("grid needs a positive horizon and unique nonempty series population")
    try:
        cadence = parse_frequency(frequency)
        start = parse_origin(origin, cadence)
        requested_points = period_points(start, cadence, horizon)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"origin must be canonical for frequency {frequency!r}, got {origin!r}"
        ) from exc
    end = requested_points[-1]
    validate_parquet_time_storage(path, time_col, allow_hive_partition=True)
    # INT96 has no physical unit: read proven-naive values at the same precision as the
    # other readers, preserving wide-year values without nanosecond overflow.
    parquet = ds.ParquetFileFormat(read_options={"coerce_int96_timestamp_unit": "us"})
    source = ds.dataset(path, format=parquet, partitioning="hive")
    date_field = ds.field(time_col)  # pyright: ignore[reportPrivateImportUsage]
    field_type = source.schema.field(time_col).type
    if filter_actuals and (pa.types.is_string(field_type) or pa.types.is_large_string(field_type)):
        # Hive partition keys are strings. Cast only the predicate; keep original keys in
        # the result so strict canonical validation can still reject malformed labels.
        field_type = pa.timestamp("us") if cadence.key_kind == "timestamp" else pa.date32()
        date_field = date_field.cast(field_type)
    if pa.types.is_timestamp(field_type):
        start_bound = start if isinstance(start, dt.datetime) else dt.datetime.combine(
            start, dt.time()
        )
        end_bound = end if isinstance(end, dt.datetime) else dt.datetime.combine(end, dt.time())
    else:
        start_bound, end_bound = start, end
    condition = (
        (date_field > pa.scalar(start_bound, type=field_type))
        & (date_field <= pa.scalar(end_bound, type=field_type))
        if filter_actuals
        else None
    )
    table = source.to_table(columns=[series_col, time_col, value_col], filter=condition)
    if pa.types.is_boolean(table[value_col].type):
        raise ValueError("grid values must be numeric rather than boolean")
    # Arrow's wheel dynamically exports index_in but does not enumerate it in type stubs.
    indices = pc.index_in(  # pyright: ignore[reportAttributeAccessIssue]
        table[series_col], value_set=pa.array(series_ids)
    )
    if (
        table.num_rows != len(series_ids) * horizon
        or indices.null_count
        or table[time_col].null_count
    ):
        raise ValueError("grid contains missing/extra dates or unknown series")
    rows = indices.to_numpy().astype(np.int64)
    point_index = {point: index for index, point in enumerate(requested_points)}
    try:
        points = tuple(
            parse_origin(canonical_origin(value, cadence), cadence)
            for value in table[time_col].to_pylist()
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("grid time keys are malformed or misaligned with the declared cadence") \
            from exc
    days = np.array([point_index.get(point, -1) for point in points], dtype=np.int64)
    flat = rows * horizon + days
    if np.any(days < 0) or np.any(days >= horizon) or len(np.unique(flat)) != len(flat):
        raise ValueError("grid has duplicate or out-of-window cells")
    values = table[value_col].to_numpy().astype(np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("grid values must be finite nonnegative numbers")
    result = np.empty((len(series_ids), horizon), np.float64)
    result[rows, days] = values
    return result


@dataclass(frozen=True)
class HierarchyMetric:
    """Origin-bound scales/weights and complete bottom-to-node maps."""

    codes: tuple[np.ndarray, ...]
    scales: tuple[np.ndarray, ...]
    weights: tuple[np.ndarray, ...]
    names: tuple[str, ...]
    n_bottom: int
    training_surrogate: bool = False
    weighting: str = "revenue"

    @classmethod
    def build(
        cls,
        metadata: dict[str, tuple[str, ...]],
        history: np.ndarray,
        price_history: np.ndarray | None,
        levels: tuple[tuple[str, ...], ...],
        *,
        weight_days: int = 28,
        training_surrogate: bool = False,
        weighting: str = "revenue",
    ) -> HierarchyMetric:
        """Build from history ending at ONE origin; no validation target is accepted here."""
        history = np.asarray(history, dtype=np.float64)
        if weighting not in {"revenue", "uniform"}:
            raise ValueError("hierarchy weighting must be revenue or uniform")
        if weighting == "revenue":
            price_history = np.asarray(price_history, dtype=np.float64)
        if (
            history.ndim != 2
            or history.shape[0] == 0
            or history.shape[1] < 2
            or weighting == "revenue"
            and (price_history is None or price_history.shape != history.shape)
            or not levels
            or type(weight_days) is not int
            or weight_days < 1
            or weighting == "revenue"
            and history.shape[1] < weight_days
        ):
            raise ValueError("invalid history/price shape, levels or weight window")
        if np.any(~np.isfinite(history)) or np.any(history < 0):
            raise ValueError("metric history must be finite nonnegative sales")
        names = ["+".join(level) or "total" for level in levels]
        if (
            type(training_surrogate) is not bool
            or len(set(levels)) != len(levels)
            or len(set(names)) != len(names)
            or any(len(set(level)) != len(level) for level in levels)
        ):
            raise ValueError("hierarchy levels must be unique and unambiguous")
        n = history.shape[0]
        if any(len(values) != n for values in metadata.values()):
            raise ValueError("metadata must cover every bottom series")
        revenue = np.ones(n, dtype=np.float64)
        if weighting == "revenue":
            assert price_history is not None
            sold = history[:, -weight_days:]
            prices = price_history[:, -weight_days:]
            if np.any((sold > 0) & (~np.isfinite(prices) | (prices < 0))):
                raise ValueError("positive sales require finite nonnegative weight prices")
            with np.errstate(over="ignore", invalid="ignore"):
                revenue = np.where(sold > 0, sold * np.nan_to_num(prices, nan=0.0), 0.0).sum(1)
        with np.errstate(over="ignore", invalid="ignore"):
            total_revenue = revenue.sum()
        if not np.isfinite(revenue).all() or not np.isfinite(total_revenue) or total_revenue <= 0:
            raise ValueError("origin has undefined dollar weights")
        all_codes, all_scales, all_weights, names = [], [], [], []
        for level in levels:
            if not set(level) <= set(metadata):
                raise ValueError("hierarchy level names an absent identity column")
            keys = (
                list(zip(*(metadata[name] for name in level), strict=True)) if level else [()] * n
            )
            unique = {key: i for i, key in enumerate(sorted(set(keys)))}
            codes = np.array([unique[key] for key in keys], dtype=np.int64)
            aggregate = np.zeros((len(unique), history.shape[1]), dtype=np.float64)
            with np.errstate(over="ignore", invalid="ignore"):
                np.add.at(aggregate, codes, history)
            if not np.isfinite(aggregate).all():
                raise ValueError("nonfinite aggregate history; numerical overflow")
            started = (aggregate > 0).any(1)
            first = (aggregate > 0).argmax(1)
            with np.errstate(over="ignore", invalid="ignore"):
                differences = np.diff(aggregate, axis=1) ** 2
            if not np.isfinite(differences).all():
                raise ValueError("nonfinite scale differences; numerical overflow")
            eligible = np.arange(1, aggregate.shape[1])[None, :] > first[:, None]
            denominator = aggregate.shape[1] - first - 1
            with np.errstate(over="ignore", invalid="ignore"):
                scales = (differences * eligible).sum(1) / np.maximum(denominator, 1)
            if not np.isfinite(scales).all():
                raise ValueError("nonfinite RMSSE scales; numerical overflow")
            valid = started & (denominator > 0) & np.isfinite(scales) & (scales > 0)
            if not valid.all() and not training_surrogate:
                raise ValueError(f"undefined RMSSE scales at level {level}: {int((~valid).sum())}")
            # Explicit training-only surrogate; never used by score()/official validation.
            if training_surrogate:
                scales = np.maximum(np.where(valid, scales, 0.0), 1.0)
            weights = np.bincount(codes, weights=revenue, minlength=len(unique)) / total_revenue
            if not np.isfinite(weights).all():
                raise ValueError("nonfinite hierarchy weights; numerical overflow")
            all_codes.append(codes)
            all_scales.append(scales)
            all_weights.append(weights)
            names.append("+".join(level) or "total")
        return cls(
            tuple(all_codes),
            tuple(all_scales),
            tuple(all_weights),
            tuple(names),
            n,
            training_surrogate,
            weighting,
        )

    def score(self, prediction: np.ndarray, target: np.ndarray) -> dict:
        """Full-grid RMSSE; only revenue weighting uses the M5 WRMSSE result key."""
        if self.training_surrogate:
            raise ValueError("training-surrogate scales cannot publish an official WRMSSE")
        prediction, target = np.asarray(prediction), np.asarray(target)
        if (
            prediction.ndim != 2
            or prediction.shape != target.shape
            or prediction.shape[0] != self.n_bottom
            or prediction.shape[1] == 0
            or not np.isfinite(prediction).all()
            or not np.isfinite(target).all()
            or np.any(prediction < 0)
            or np.any(target < 0)
        ):
            raise ValueError("score requires complete finite nonnegative matching forecast grids")
        per_level, counts = {}, {}
        error = prediction.astype(np.float64) - target.astype(np.float64)
        for name, codes, scales, weights in zip(
            self.names, self.codes, self.scales, self.weights, strict=True
        ):
            aggregate = np.zeros((len(scales), error.shape[1]), dtype=np.float64)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                np.add.at(aggregate, codes, error)
                rmsse = np.sqrt(np.mean(aggregate**2, 1) / scales)
                value = float(np.sum(weights * rmsse))
            if (
                not np.isfinite(aggregate).all()
                or not np.isfinite(rmsse).all()
                or not np.isfinite(value)
            ):
                raise ValueError("nonfinite hierarchy score; numerical overflow")
            per_level[name] = value
            counts[name] = len(scales)
        with np.errstate(over="ignore", invalid="ignore"):
            overall = float(np.mean(list(per_level.values())))
        if not np.isfinite(overall):
            raise ValueError("nonfinite overall WRMSSE; numerical overflow")
        return {
            "wrmsse" if self.weighting == "revenue" else "uniform_hierarchy_rmsse": overall,
            "weighting": self.weighting,
            "per_level": per_level,
            "n_series_scored": counts,
            "n_series_dropped": dict.fromkeys(counts, 0),
        }

    def loss(
        self, prediction: torch.Tensor, target: torch.Tensor, *, epsilon: float
    ) -> torch.Tensor:
        """Complete-group differentiable benchmark surrogate; not a strictly mean-proper loss."""
        if (
            prediction.ndim != 2
            or prediction.shape != target.shape
            or prediction.shape[0] != self.n_bottom
            or prediction.shape[1] == 0
            or not np.isfinite(epsilon)
            or epsilon <= 0
        ):
            raise ValueError("hierarchy loss requires full grids and positive finite smoothing")
        if (
            not torch.isfinite(prediction).all()
            or not torch.isfinite(target).all()
            or (prediction < 0).any()
            or (target < 0).any()
        ):
            raise ValueError("hierarchy loss needs finite nonnegative inputs")
        error = prediction - target
        terms = []
        for codes, scales, weights in zip(self.codes, self.scales, self.weights, strict=True):
            index = torch.as_tensor(codes, dtype=torch.long, device=error.device)
            aggregate = error.new_zeros((len(scales), error.shape[1])).index_add(0, index, error)
            scale = torch.as_tensor(scales, dtype=error.dtype, device=error.device)
            weight = torch.as_tensor(weights, dtype=error.dtype, device=error.device)
            if not torch.isfinite(scale).all() or not torch.isfinite(weight).all():
                raise ValueError("hierarchy objective constants overflow the prediction dtype")
            terms.append((weight * torch.sqrt(aggregate.square().mean(1) / scale + epsilon)).sum())
        result = torch.stack(terms).mean()
        if not torch.isfinite(result):
            raise ValueError("nonfinite hierarchy objective; numerical overflow")
        return result


class TorchHierarchyMetric:
    """Unsmoothed evaluation RMSSE with cached float64 device-local constants.

    Same origin-bound constants as the NumPy reference, not the training surrogate.
    Only the small per-level result vector leaves the device. See test_torch_hierarchy_metric.
    """

    def __init__(self, metric: HierarchyMetric, device: str | torch.device):
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if self.device.type not in {"cpu", "cuda"} or metric.training_surrogate:
            raise ValueError("Torch scoring requires CPU/CUDA and official, not surrogate, scales")
        if (
            not metric.names
            or metric.n_bottom < 1
            or len(set(metric.names)) != len(metric.names)
            or not (
                len(metric.names) == len(metric.codes) == len(metric.scales) == len(metric.weights)
            )
            or metric.weighting not in {"revenue", "uniform"}
        ):
            raise ValueError("invalid cached hierarchy structure")
        for codes, scales, weights in zip(metric.codes, metric.scales, metric.weights, strict=True):
            if (
                codes.shape != (metric.n_bottom,)
                or codes.dtype.kind not in "iu"
                or scales.ndim != 1
                or not len(scales)
                or scales.shape != weights.shape
                or np.any(codes < 0)
                or np.any(codes >= len(scales))
                or not np.isfinite(scales).all()
                or np.any(scales <= 0)
                or not np.isfinite(weights).all()
                or np.any(weights < 0)
                or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12)
            ):
                raise ValueError("invalid cached hierarchy indices/scales/weights")
        self.names, self.n_bottom, self.weighting = metric.names, metric.n_bottom, metric.weighting
        self.codes = tuple(
            torch.tensor(v, dtype=torch.long, device=self.device) for v in metric.codes
        )
        self.scales = tuple(
            torch.tensor(v, dtype=torch.float64, device=self.device) for v in metric.scales
        )
        self.weights = tuple(
            torch.tensor(v, dtype=torch.float64, device=self.device) for v in metric.weights
        )

    @torch.no_grad()
    def score(self, prediction: torch.Tensor, target: torch.Tensor) -> dict:
        """Full population, float64 reductions, no epsilon and no dropped nodes."""
        if (
            prediction.device != self.device
            or target.device != self.device
            or prediction.ndim != 2
            or prediction.shape != target.shape
            or prediction.shape[0] != self.n_bottom
            or prediction.shape[1] == 0
            or not torch.isfinite(prediction).all()
            or not torch.isfinite(target).all()
            or (prediction < 0).any()
            or (target < 0).any()
        ):
            raise ValueError(
                "Torch score requires matching complete finite nonnegative grids on its device"
            )
        error = prediction.double() - target.double()
        terms = []
        for codes, scale, weight in zip(self.codes, self.scales, self.weights, strict=True):
            aggregate = error.new_zeros((len(scale), error.shape[1])).index_add_(0, codes, error)
            rmsse = (aggregate.square().mean(1) / scale).sqrt()
            terms.append((rmsse * weight).sum())
        values = torch.stack(terms)
        overall = values.mean()
        if not torch.isfinite(values).all() or not torch.isfinite(overall):
            raise ValueError("nonfinite Torch hierarchy score; numerical overflow")
        scalars = torch.cat((overall[None], values)).cpu().tolist()
        counts = {name: len(scale) for name, scale in zip(self.names, self.scales, strict=True)}
        return {
            "wrmsse" if self.weighting == "revenue" else "uniform_hierarchy_rmsse": scalars[0],
            "weighting": self.weighting,
            "per_level": dict(zip(self.names, scalars[1:], strict=True)),
            "n_series_scored": counts,
            "n_series_dropped": dict.fromkeys(counts, 0),
        }
