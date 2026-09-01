"""Origin-safe numeric covariates, bounded groups and cached TimesFM-HNB head features."""

from __future__ import annotations

import datetime as dt
import pathlib
from dataclasses import dataclass
from typing import get_args

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from autoforecast.panel import ELIGIBLE_STATES, ObservationState, Panel
from autoforecast.timesfm_hnb_source import Source, SourceRequest, cached_quantiles


@dataclass
class TimesFMHNBData:
    """One bounded daily grid; missing targets and business fields retain separate masks."""

    panel: Panel
    start: dt.date
    series: tuple[str, ...]
    target: np.ndarray
    observed: np.ndarray
    covariates: np.ndarray
    fields: tuple[str, ...]
    known: tuple[str, ...]
    past: tuple[str, ...]
    static: np.ndarray
    categories: np.ndarray
    metadata: dict[str, list[str]]
    schema: dict[str, object]
    shared_values: np.ndarray | None = None
    shared_present: np.ndarray | None = None

    @property
    def shared(self) -> tuple[str, ...]:
        return tuple(self.schema.get("shared", ()))

    @property
    def local_fields(self) -> tuple[str, ...]:
        return tuple(name for name in self.fields if name not in self.shared)

    def position(self, origin: str) -> int:
        return (dt.date.fromisoformat(origin) - self.start).days


def _load_data(
    panel: Panel,
    origin: str,
    horizon: int,
    context: int,
    *,
    schema: dict[str, object] | None = None,
    history_days: int | None = None,
    series_subset: tuple[str, ...] | None = None,
) -> TimesFMHNBData:
    """Read only needed dates/roles, ignoring all post-origin target values."""
    panel.validate()
    if panel.status_prob_cols is not None or panel.target_estimand != "sales_given_availability":
        raise ValueError("TimesFM-HNB supports hard/missing availability and observed sales only")
    end = dt.date.fromisoformat(origin)
    start = end - dt.timedelta(days=(history_days or context + 2 * horizon) - 1)
    stop = end + dt.timedelta(days=horizon)
    columns = list(panel.role_columns)
    dataset = ds.dataset(panel.path, format="parquet", partitioning="hive")
    time_type = dataset.schema.field(panel.time_col).type
    import pyarrow as pa

    lower_value = (
        dt.datetime.combine(start, dt.time()) if pa.types.is_timestamp(time_type) else start
    )
    # Filter the same calendar dates regardless of the timestamp's time of day.
    next_day = stop + dt.timedelta(days=1)
    if pa.types.is_timestamp(time_type):
        lower_value = pd.Timestamp(start, tz=time_type.tz).to_pydatetime()
        upper_value = pd.Timestamp(next_day, tz=time_type.tz).to_pydatetime()
    else:
        upper_value = next_day
    lower = pa.scalar(lower_value, type=time_type)
    upper = pa.scalar(upper_value, type=time_type)
    row_filter = (ds.field(panel.time_col) >= lower) & (ds.field(panel.time_col) < upper)
    if series_subset is not None:
        row_filter = row_filter & ds.field(panel.series_col).isin(series_subset)
    frame = dataset.to_table(columns=columns, filter=row_filter).to_pandas()
    if frame.empty:
        raise ValueError("TimesFM-HNB panel has no rows in its bounded history window")
    frame[panel.series_col] = frame[panel.series_col].astype(str)
    frame[panel.time_col] = pd.to_datetime(frame[panel.time_col]).dt.date
    if frame.duplicated([panel.series_col, panel.time_col]).any():
        raise ValueError("TimesFM-HNB panel has duplicate series/date rows")
    history = frame.loc[frame[panel.time_col] <= end]
    # Include future-only IDs in disposition; their history stays unavailable.
    series = tuple(sorted(frame[panel.series_col].unique()))
    if schema is not None and tuple(schema["series"]) != series:
        raise ValueError(
            "TimesFM-HNB checkpoint series identities differ; retrain for a new population"
        )
    known = tuple(schema["known"] if schema is not None else panel.known_future)
    declared_past = tuple(
        dict.fromkeys((*panel.past_only, *((panel.oos_col,) if panel.oos_col else ())))
    )
    past = tuple(schema["past"] if schema is not None else declared_past)
    if set(panel.known_future) - set(known) or set(declared_past) - set(past):
        raise ValueError("TimesFM-HNB checkpoint does not contain the new covariate schema")
    if set(past) & set(known):
        raise ValueError("TimesFM-HNB historical availability cannot be a known-future covariate")
    fields = (*past, *known)
    shared = tuple(panel.shared_known_future if schema is None else schema.get("shared", ()))
    if set(panel.shared_known_future) != set(shared) & set(panel.known_future):
        raise ValueError("TimesFM-HNB shared covariate schema changed")
    families = tuple(
        tuple(family)
        for family in (panel.covariate_families if schema is None else schema.get("families", ()))
    )
    active_fields = {*panel.known_future, *declared_past}
    active_families = tuple(
        tuple(name for name in family if name in active_fields) for family in families
    )
    if tuple(family for family in active_families if family) != panel.covariate_families:
        raise ValueError("TimesFM-HNB covariate family schema changed")
    local_fields = tuple(name for name in fields if name not in shared)
    days = (stop - start).days + 1
    if len(series) * days * (5 + 4 * len(local_fields)) > 2_000_000_000:
        raise ValueError("TimesFM-HNB dense history grid exceeds the recorded 2GB development cap")
    target = np.zeros((len(series), days), dtype=np.float32)
    observed = np.zeros(target.shape, dtype=bool)
    covariates = np.full((*target.shape, len(local_fields)), np.nan, dtype=np.float32)
    shared_values = np.full((days, len(shared)), np.nan, dtype=np.float32)
    row_series = pd.Categorical(frame[panel.series_col], categories=series).codes
    row_days = np.asarray([(d - start).days for d in frame[panel.time_col]])
    in_history = np.asarray(frame[panel.time_col] <= end) & (row_series >= 0)
    if panel.exposure_col:
        valid_states = set(get_args(ObservationState))
        states = frame.loc[in_history, panel.exposure_col]
        if not set(states.dropna().unique()) <= valid_states:
            raise ValueError("TimesFM-HNB encountered an unknown observation state")
        in_history &= np.asarray(frame[panel.exposure_col].isin(ELIGIBLE_STATES))
    actual = pd.to_numeric(frame.loc[in_history, panel.target_col], errors="raise").to_numpy(
        dtype=np.float64, na_value=np.nan
    )
    available = ~np.isnan(actual)
    if panel.negative_target_policy == "floor":
        actual = np.maximum(actual, 0)
    if (
        not np.isfinite(actual[available]).all()
        or (actual[available] < 0).any()
        or (actual[available] != np.floor(actual[available])).any()
        or (actual[available] > 2**24).any()
    ):
        raise ValueError("TimesFM-HNB requires nonnegative counts exactly representable in float32")
    rows = row_series[in_history][available]
    dates = row_days[in_history][available]
    target[rows, dates] = actual[available]
    observed[rows, dates] = True
    for index, name in enumerate(local_fields):
        if (
            name not in frame
            or (name in known and name not in panel.known_future)
            or (name in past and name not in declared_past)
        ):
            continue  # A trained family can be omitted; it remains explicitly unavailable.
        values = pd.to_numeric(frame[name], errors="raise").to_numpy(
            dtype=np.float64, na_value=np.nan
        )
        if np.isinf(values).any() or np.any(np.abs(values[np.isfinite(values)]) > 1e20):
            raise ValueError("TimesFM-HNB covariates exceed the source numeric domain")
        admitted = row_series >= 0
        if name in past:
            admitted &= np.asarray(frame[panel.time_col] <= end)
        covariates[row_series[admitted], row_days[admitted], index] = values[admitted]
    for index, name in enumerate(shared):
        if name not in frame or name not in panel.known_future:
            continue
        by_day = frame.groupby(panel.time_col, sort=False)[name]
        if (by_day.nunique(dropna=False) > 1).any():
            raise ValueError(
                f"TimesFM-HNB shared field {name!r} differs between series on one date"
            )
        values = by_day.first()
        numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=np.float64, na_value=np.nan)
        if np.isinf(numeric).any() or np.any(np.abs(numeric[np.isfinite(numeric)]) > 1e20):
            raise ValueError("TimesFM-HNB shared covariates exceed the source numeric domain")
        dates = np.asarray([(day - start).days for day in values.index])
        shared_values[dates, index] = numeric
    metadata: dict[str, list[str]] = {}
    category_maps = {} if schema is None else dict(schema["category_maps"])
    categories = []
    real_values = []
    static_names = tuple(schema["static_names"] if schema is not None else panel.static)
    real_names = tuple(schema["real_names"] if schema is not None else panel.static_reals)
    if set(panel.static) - set(static_names) or set(panel.static_reals) - set(real_names):
        raise ValueError("TimesFM-HNB checkpoint static schema changed")
    meta_names = tuple(
        dict.fromkeys(
            (*static_names, *real_names, *(x for x in (panel.store_col, panel.category_col) if x))
        )
    )
    for name in meta_names:
        if name not in history:
            values = pd.Series(index=series, data=None, dtype=object)
        else:
            if (history.groupby(panel.series_col)[name].nunique(dropna=False) > 1).any():
                raise ValueError(f"TimesFM-HNB static/group field {name!r} changes within a series")
            values = history.drop_duplicates(panel.series_col).set_index(panel.series_col)[name]
            values = values.reindex(series)
        tokens = ["<missing>" if pd.isna(v) else str(v) for v in values]
        metadata[name] = tokens
        if name in static_names:
            if schema is None:
                category_maps[name] = sorted(set(tokens) - {"<missing>"})
            mapping = {v: i + 1 for i, v in enumerate(category_maps[name])}
            categories.append([mapping.get(token, 0) for token in tokens])
        if name in real_names:
            numeric = pd.to_numeric(values, errors="raise").to_numpy(dtype=np.float64)
            if np.isinf(numeric).any():
                raise ValueError("TimesFM-HNB static real values must be finite or missing")
            real_values.extend([np.arcsinh(np.nan_to_num(numeric)), ~np.isnan(numeric)])
    static = (
        np.asarray(real_values, dtype=np.float32).T
        if real_values
        else np.zeros((len(series), 0), np.float32)
    )
    cats = (
        np.asarray(categories, dtype=np.int64).T
        if categories
        else np.zeros((len(series), 0), np.int64)
    )
    saved_schema = {
        "series": list(series),
        "known": list(known),
        "past": list(past),
        "static_names": list(static_names),
        "real_names": list(real_names),
        "category_maps": category_maps,
        "group_columns": [x for x in (panel.store_col, panel.category_col) if x],
    }
    if shared or families:
        saved_schema.update(shared=list(shared), families=[list(family) for family in families])
    if schema is not None and saved_schema != schema:
        raise ValueError("TimesFM-HNB schema/grouping identity changed")
    return TimesFMHNBData(
        panel,
        start,
        series,
        target,
        observed,
        covariates,
        fields,
        known,
        past,
        static,
        cats,
        metadata,
        saved_schema,
        shared_values,
        np.isin(np.arange(days), row_days),
    )


def load_data(panel, origin, horizon, context, *, schema=None, history_days=None):
    """Bound peak Arrow/Pandas memory by reading at most 2,048 series per slice."""
    dataset = ds.dataset(panel.path, format="parquet", partitioning="hive")
    import pyarrow.compute as pc

    end = pd.Timestamp(origin)
    start = end - pd.Timedelta(days=(history_days or context + 2 * horizon) - 1)
    stop = end + pd.Timedelta(days=horizon + 1)
    import pyarrow as pa

    time_type = dataset.schema.field(panel.time_col).type
    if pa.types.is_timestamp(time_type):
        lower = start.tz_localize(time_type.tz).to_pydatetime()
        upper = stop.tz_localize(time_type.tz).to_pydatetime()
    else:
        lower, upper = start.date(), stop.date()
    row_filter = (ds.field(panel.time_col) >= pa.scalar(lower, type=time_type)) & (
        ds.field(panel.time_col) < pa.scalar(upper, type=time_type)
    )
    identities = set()
    for batch in dataset.to_batches(
        columns=[panel.series_col], filter=row_filter, batch_size=262144
    ):
        identities.update(str(value) for value in pc.unique(batch.column(0)).to_pylist())
    if len(identities) <= 2048:
        return _load_data(panel, origin, horizon, context, schema=schema, history_days=history_days)
    from dataclasses import replace

    series = tuple(sorted(identities))
    pieces = []
    for start in range(0, len(series), 2048):
        subset = series[start : start + 2048]
        piece_schema = None if schema is None else {**schema, "series": list(subset)}
        piece = _load_data(
            panel,
            origin,
            horizon,
            context,
            schema=piece_schema,
            history_days=history_days,
            series_subset=subset,
        )
        pieces.append(piece)
        print(
            f"TimesFM panel loaded: {min(start + 2048, len(series))}/{len(series)} series",
            flush=True,
        )
    first = pieces[0]
    shared_values = first.shared_values.copy()
    present = first.shared_present.copy()
    for piece in pieces[1:]:
        overlap = present & piece.shared_present
        if not np.array_equal(shared_values[overlap], piece.shared_values[overlap], equal_nan=True):
            raise ValueError("TimesFM-HNB shared fields differ between series slices")
        shared_values[piece.shared_present] = piece.shared_values[piece.shared_present]
        present |= piece.shared_present
    maps = {
        name: sorted(set().union(*(piece.schema["category_maps"][name] for piece in pieces)))
        for name in first.schema["static_names"]
    }
    if schema is not None:
        maps = schema["category_maps"]
    categories = []
    for piece in pieces:
        remapped = piece.categories.copy()
        for index, name in enumerate(first.schema["static_names"]):
            lookup = {value: i + 1 for i, value in enumerate(maps[name])}
            translate = np.array(
                [0, *(lookup[value] for value in piece.schema["category_maps"][name])]
            )
            remapped[:, index] = translate[piece.categories[:, index]]
        categories.append(remapped)
    combined_series = tuple(sid for piece in pieces for sid in piece.series)
    combined_schema = {**first.schema, "series": list(combined_series), "category_maps": maps}
    if schema is not None and combined_schema != schema:
        raise ValueError("TimesFM-HNB checkpoint series/schema differ from the input")
    return replace(
        first,
        series=combined_series,
        target=np.concatenate([piece.target for piece in pieces]),
        observed=np.concatenate([piece.observed for piece in pieces]),
        covariates=np.concatenate([piece.covariates for piece in pieces]),
        static=np.concatenate([piece.static for piece in pieces]),
        categories=np.concatenate(categories),
        metadata={
            name: [value for piece in pieces for value in piece.metadata[name]]
            for name in first.metadata
        },
        schema=combined_schema,
        shared_values=shared_values,
        shared_present=present,
    )


def groups(data: TimesFMHNBData, group_size: int, max_variates: int) -> tuple[tuple[int, ...], ...]:
    """Bound candidates before grouping; each leaf belongs to exactly one ordered group."""
    per_target = 1 + len(data.local_fields)
    width = min(group_size, (max_variates - len(data.shared)) // per_target)
    if width < 1:
        raise ValueError(
            "TimesFM-HNB channel budget cannot represent one target and declared covariates"
        )
    columns = data.schema["group_columns"]
    partitions: dict[tuple[str, ...], list[int]] = {}
    for index, sid in enumerate(data.series):
        key = tuple(data.metadata[c][index] for c in columns) if columns else (sid,)
        partitions.setdefault(key, []).append(index)
    return tuple(
        tuple(indices[start : start + width])
        for _, indices in sorted(partitions.items())
        for start in range(0, len(indices), width)
    )


def missing_profiles(
    fields: tuple[str, ...],
    families: tuple[tuple[str, ...], ...] = (),
) -> tuple[tuple[str, ...], ...]:
    """Regenerate the source for each missing family and for history-only availability."""
    grouped = {name for family in families for name in family}
    omitted = [*families, *((name,) for name in fields if name not in grouped)]
    canonical = [tuple(name for name in fields if name in absent) for absent in omitted]
    return tuple(dict.fromkeys([(), *canonical, fields]))


def _future_plan(data: TimesFMHNBData, origin: str, horizon: int, spark, work: pathlib.Path):
    panel = data.panel
    if panel.plan is None:
        return None, {"mode": "declared_known_future_snapshot", "origin": origin}
    if spark is None:
        raise ValueError(
            "TimesFM-HNB effective-dated plans require the existing Spark as-of selector"
        )
    from forecast.features.plans import build

    values = {}
    for role in ("price", "promo"):
        source = getattr(panel.plan, f"{role}_col")
        target = getattr(panel, f"{role}_col")
        if source is not None:
            if target not in data.known:
                raise ValueError(
                    "TimesFM-HNB plan roles require corresponding known_future panel fields"
                )
            values[target] = source
    selected = build(
        panel.plan.path,
        origin,
        horizon,
        spark,
        work / origin,
        series_col=panel.series_col,
        valid_date_col=panel.plan.valid_date_col,
        known_at_col=panel.plan.known_at_col,
        snapshot_col=panel.plan.snapshot_col,
        value_cols=values,
        series_ids=data.series,
    )
    return ds.dataset(selected.path, format="parquet").to_table().to_pandas(), selected.as_record()


def features(
    data: TimesFMHNBData,
    origin: str,
    horizon: int,
    context: int,
    grouping: tuple[tuple[int, ...], ...],
    source: Source | None,
    cache_dir: pathlib.Path,
    *,
    absent: tuple[str, ...] = (),
    overrides: dict[str, np.ndarray] | None = None,
    spark=None,
    requests_out: list[SourceRequest] | None = None,
    progress: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Build origin-safe features, applying future-known scenarios before source inference."""
    pos = data.position(origin)
    begin, finish = pos - context + 1, pos + horizon + 1
    if begin < 0 or finish > data.target.shape[1]:
        raise ValueError("TimesFM-HNB origin is outside its materialized history/horizon grid")
    if set(absent) - set(data.fields):
        raise ValueError("unknown absent TimesFM-HNB family")
    history = data.target[:, begin : pos + 1]
    observed = data.observed[:, begin : pos + 1]
    admitted = observed.any(axis=1)
    cov = data.covariates[:, begin:finish].copy()
    shared = (
        data.shared_values[begin:finish].copy()
        if data.shared_values is not None
        else np.zeros((finish - begin, 0), np.float32)
    )
    plan, plan_identity = _future_plan(data, origin, horizon, spark, cache_dir / "plans")
    if plan is not None:
        lookup = {s: i for i, s in enumerate(data.series)}
        for name in (data.panel.price_col, data.panel.promo_col):
            if name is not None and name in plan:
                if name in data.shared:
                    raise ValueError(
                        "TimesFM-HNB effective-dated commercial plans require per-series fields"
                    )
                field = data.local_fields.index(name)
                cov[:, context:, field] = np.nan
                for sid, day, value in plan[["series_id", "date", name]].itertuples(
                    index=False, name=None
                ):
                    step = (pd.Timestamp(day).date() - dt.date.fromisoformat(origin)).days - 1
                    cov[lookup[str(sid)], context + step, field] = value
    for name, values in (overrides or {}).items():
        if name not in data.known or np.shape(values) != (len(data.series), horizon):
            raise ValueError(
                "TimesFM-HNB scenarios may override only declared future-known "
                "[series,horizon] fields"
            )
        if np.isinf(values).any():
            raise ValueError("TimesFM-HNB scenario values must be finite or unavailable")
        if name in data.shared:
            values = np.asarray(values)
            if not np.array_equal(
                values, np.broadcast_to(values[:1], values.shape), equal_nan=True
            ):
                raise ValueError(
                    "TimesFM-HNB shared scenario must agree across the recorded series"
                )
            shared[context:, data.shared.index(name)] = values[0]
        else:
            cov[:, context:, data.local_fields.index(name)] = values
    for name in absent:
        if name in data.shared:
            shared[:, data.shared.index(name)] = np.nan
        else:
            cov[:, :, data.local_fields.index(name)] = np.nan
    n_past = len(data.past)
    cov[:, context:, :n_past] = np.nan
    quantiles = np.zeros((len(data.series), horizon, 9), dtype=np.float32)
    cache_records = []
    group_records = []
    for group_index, group in enumerate(grouping):
        if progress and group_index % 100 == 0:
            print(
                f"TimesFM-HNB source groups: {group_index}/{len(grouping)} "
                f"origin={origin} absent={absent}",
                flush=True,
            )
        indices = [i for i in group if admitted[i]]
        if not indices:
            continue
        group_records.append([data.series[i] for i in indices])
        if source is None and requests_out is None:
            continue
        channels = [f"target:{data.series[i]}" for i in indices]
        tensors = {}
        for label, field_names, selection, length in (
            ("past_only", data.past, slice(0, n_past), context),
            (
                "past_future",
                tuple(n for n in data.known if n not in data.shared),
                slice(n_past, None),
                context + horizon,
            ),
        ):
            if not field_names and not (label == "past_future" and data.shared):
                continue
            values = cov[indices, :length, selection].transpose(0, 2, 1).reshape(-1, length)
            names = [f"{label}:{data.series[i]}:{name}" for i in indices for name in field_names]
            if label == "past_future" and data.shared:
                values = np.concatenate((values, shared.T), axis=0)
                names.extend(f"{label}:shared:{name}" for name in data.shared)
            masks = np.isnan(values)
            retained = ~masks.all(axis=1)
            if retained.any():
                tensors[label] = values[retained]
                tensors[f"{label}_mask"] = masks[retained]
                channels.extend(name for name, keep in zip(names, retained, strict=True) if keep)
        request = SourceRequest(
            history[indices], ~observed[indices], horizon, tuple(channels), **tensors
        )
        if requests_out is not None:
            requests_out.append(request)
        if source is not None:
            quantiles[indices], record = cached_quantiles(source, request, cache_dir)
            cache_records.append(record)
    count = observed.sum(axis=1).clip(1)
    positive = ((history > 0) & observed).sum(axis=1)
    total = np.where(observed, history, 0).sum(axis=1, dtype=np.float64)
    scale = np.maximum(total / count, 1.0)
    probability = (positive + 0.5) / (count + 1)
    excess = np.maximum(total / positive.clip(1) - 1, 1e-6)
    baseline = np.stack([np.log(probability / (1 - probability)), np.log(excess)], axis=-1)
    head = [
        np.arcsinh(quantiles / scale[:, None, None]),
        np.broadcast_to(np.log1p(scale)[:, None, None], (len(scale), horizon, 1)),
    ]
    for name in data.fields:
        if name in data.shared:
            values = np.broadcast_to(
                shared[context:, data.shared.index(name)], (len(scale), horizon)
            )
            mask = np.isnan(values)
        elif name in data.known:
            index = data.local_fields.index(name)
            values = cov[:, context:, index]
            mask = np.isnan(values)
        else:
            index = data.local_fields.index(name)
            previous = cov[:, :context, index]
            available = np.isfinite(previous)
            values = np.where(available, previous, 0).sum(axis=1) / available.sum(axis=1).clip(1)
            mask = ~available.any(axis=1)
            values = np.broadcast_to(values[:, None], (len(scale), horizon))
            mask = np.broadcast_to(mask[:, None], values.shape)
        head.extend([np.arcsinh(np.where(mask, 0, values))[..., None], (~mask)[..., None]])
    inputs = {
        "features": np.concatenate(head, axis=-1).astype(np.float32),
        "baseline": baseline.astype(np.float32),
        "static": data.static,
        "categories": data.categories,
        "series_index": np.arange(len(data.series), dtype=np.int64),
        "admitted": admitted,
        "source_scale": scale.astype(np.float32),
        "mean_scale": (
            1
            + np.where(observed[:, -28:], history[:, -28:], 0).sum(axis=1, dtype=np.float64)
            / observed[:, -28:].sum(axis=1).clip(1)
        )
        ** 2,
    }
    return inputs, {
        "origin": origin,
        "absent": list(absent),
        "groups": group_records,
        "cache": cache_records,
        "plan": plan_identity,
    }
