"""Memory-budgeted rectangular Parquet inputs for chronological development training.

The boundary is a path, not a frame. Tests: ``test_multibranch_revisions.py`` and
``test_revision_portability.py``. This is not the estate-scale Spark data plane.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from spec.forecast_spec import ForecastDataSpec

from autoforecast.panel import Panel, period_days

if TYPE_CHECKING:
    from autoforecast.study_validation import StudyValidation

GLOBAL_LEFT_PADDING_CONTRACT = "masked_observed_counts_global_left_padding_v1"


@dataclass(frozen=True)
class ValidationWindow:
    """One fixed, ordered holdout window; labels are never part of training history."""

    origin: str
    target: np.ndarray
    observed: np.ndarray
    physical_state: np.ndarray | None = None
    objective_mask: np.ndarray | None = None
    study: StudyValidation | None = None

    @classmethod
    def from_study(
        cls, origin: str, study: StudyValidation, *, route_lower_bound_to_exact: bool = False,
    ) -> ValidationWindow:
        """Retain the frozen scoring mask while enforcing the native count-model domain."""
        values = study.target[study.observed]
        if (not np.isfinite(values).all() or np.any(values < 0)
                or np.any(values > 2**24) or np.any(values != np.floor(values))):
            raise ValueError("fixed validation targets must be finite integer counts in [0, 2**24]")
        target = study.target.astype(np.float32)
        physical = (np.where(study.observed, 0, 2).astype(np.int8)
                    if route_lower_bound_to_exact else None)
        for array in (target, physical):
            if array is not None:
                array.setflags(write=False)
        return cls(origin, target, study.observed, physical, study=study)

    @classmethod
    def read(
        cls,
        panel: Panel,
        origin: str,
        horizon: int,
        series_ids: tuple[str, ...],
        *,
        route_lower_bound_to_exact: bool = False,
    ) -> ValidationWindow:
        """Read only the target horizon, aligned to the training panel's series order."""
        try:
            cutoff = dt.date.fromisoformat(origin)
        except (TypeError, ValueError) as exc:
            raise ValueError("validation origin must be canonical ISO YYYY-MM-DD") from exc
        if (
            cutoff.isoformat() != origin
            or type(horizon) is not int
            or horizon < 1
            or not series_ids
            or len(set(series_ids)) != len(series_ids)
            or type(route_lower_bound_to_exact) is not bool
        ):
            raise ValueError("invalid fixed validation window identity")
        panel.validate()
        stride = period_days(panel.frequency)
        end = cutoff + dt.timedelta(days=horizon * stride)
        source = ds.dataset(panel.path, format="parquet", partitioning="hive")
        required = {panel.series_col, panel.time_col, panel.target_col}
        if panel.exposure_col:
            required.add(panel.exposure_col)
        if not required <= set(source.schema.names):
            missing = sorted(required - set(source.schema.names))
            raise ValueError(
                f"fixed validation panel is missing columns: {missing}"
            )
        row_by_id = {series_id: index for index, series_id in enumerate(series_ids)}
        target = np.zeros((len(series_ids), horizon), dtype=np.float32)
        observed = np.zeros((len(series_ids), horizon), dtype=bool)
        physical_state = (
            np.full((len(series_ids), horizon), 2, dtype=np.int8)
            if route_lower_bound_to_exact
            else None
        )
        seen = np.zeros((len(series_ids), horizon), dtype=bool)
        date_field = ds.field(panel.time_col)  # pyright: ignore[reportPrivateImportUsage]
        condition = (date_field > pa.scalar(cutoff)) & (date_field <= pa.scalar(end))
        columns = [panel.series_col, panel.time_col, panel.target_col]
        if panel.exposure_col:
            columns.append(panel.exposure_col)
        for batch in source.scanner(
            columns=columns, filter=condition, batch_size=131072
        ).to_batches():
            ids = batch[panel.series_col].to_pylist()
            dates = batch[panel.time_col].to_pylist()
            try:
                rows = np.fromiter((row_by_id[value] for value in ids), dtype=np.int64)
            except (KeyError, TypeError) as exc:
                raise ValueError("fixed validation panel contains an unknown series") from exc
            positions_list: list[int] = []
            for raw in dates:
                day = raw.date() if isinstance(raw, dt.datetime) else raw
                if not isinstance(day, dt.date):
                    raise ValueError("fixed validation panel contains a non-date value")
                offset = (day - cutoff).days
                if offset <= 0 or offset % stride:
                    raise ValueError(
                        "fixed validation date is outside or misaligned with the horizon"
                    )
                positions_list.append(offset // stride - 1)
            positions = np.asarray(positions_list, dtype=np.int64)
            if np.any(positions < 0) or np.any(positions >= horizon):
                raise ValueError("fixed validation date is outside the requested horizon")
            flat = rows * horizon + positions
            if len(np.unique(flat)) != len(flat) or np.any(seen[rows, positions]):
                raise ValueError("fixed validation panel contains duplicate series/date rows")
            states = (
                batch[panel.exposure_col].to_pylist()
                if panel.exposure_col
                else ["exact"] * len(rows)
            )
            allowed = (
                {"exact", "lower_bound", "unobserved"}
                if route_lower_bound_to_exact
                else {"exact", "unobserved"}
            )
            if any(state not in allowed for state in states):
                raise ValueError("fixed validation panel contains an unsupported observation state")
            state_codes = np.asarray(
                [{"exact": 0, "lower_bound": 1, "unobserved": 2}[state] for state in states],
                dtype=np.int8,
            )
            mask = state_codes != 2 if route_lower_bound_to_exact else state_codes == 0
            values = batch[panel.target_col].to_numpy(zero_copy_only=False).astype(np.float64)
            if (
                not np.isfinite(values[mask]).all()
                or np.any(values[mask] < 0)
                or np.any(values[mask] > 2**24)
                or np.any(values[mask] != np.floor(values[mask]))
            ):
                raise ValueError(
                    "fixed validation targets must be finite integer counts in [0, 2**24]"
                )
            target[rows, positions] = np.where(mask, values, 0).astype(np.float32)
            observed[rows, positions] = mask
            if physical_state is not None:
                physical_state[rows, positions] = state_codes
            seen[rows, positions] = True
        if not seen.all():
            raise ValueError(
                f"fixed validation panel is incomplete: {int((~seen).sum())} missing cells"
            )
        for value in (target, observed, physical_state):
            if value is not None:
                value.setflags(write=False)
        return cls(origin, target, observed, physical_state)


@dataclass(frozen=True)
class PanelArrays:
    """One base copy; optional features are allocated only when declared."""

    series_ids: tuple[str, ...]
    start: dt.date
    cutoff: dt.date
    horizon: int
    target: np.ndarray
    observed: np.ndarray
    price: np.ndarray
    metadata: dict[str, tuple[str, ...]]
    step_days: int = 1
    features: dict[str, np.ndarray] = field(default_factory=dict)
    static_reals: dict[str, np.ndarray] = field(default_factory=dict)
    observation_contract: str = "complete_recorded_sales_prelaunch_zeros_v1"
    physical_state: np.ndarray | None = None

    @classmethod
    def read(
        cls,
        panel: Panel,
        origin: str,
        horizon: int,
        *,
        max_bytes: int,
        data_spec: ForecastDataSpec | None = None,
        metadata_columns: tuple[str, ...] = (),
        observation_contract: str = "complete_recorded_sales_prelaunch_zeros_v1",
        route_lower_bound_to_exact: bool = False,
    ) -> PanelArrays:
        """Read declared inputs; tested by ``test_generic_panel_read`` and real M5 reader tests.

        Future scans never request targets, states, or historical-only covariates. Unknown
        quantities are masked, not interpreted as recorded sales for aggregate supervision.

        The preallocation estimate includes retained arrays and the seen-grid; it excludes
        Arrow/Python metadata and scan/model/metric temporaries. Static-real accounting is
        tested by ``test_panel_arrays_read_budget_counts_static_reals``.
        """
        if (
            type(horizon) is not int
            or horizon < 1
            or type(max_bytes) is not int
            or max_bytes < 1
            or type(route_lower_bound_to_exact) is not bool
        ):
            raise ValueError("horizon and max_bytes must be positive integers")
        try:
            cutoff = dt.date.fromisoformat(origin)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"origin must be canonical ISO YYYY-MM-DD, got {origin!r}") from exc
        if cutoff.isoformat() != origin:
            raise ValueError(f"origin must be canonical ISO YYYY-MM-DD, got {origin!r}")
        panel.validate()
        stride = period_days(panel.frequency)
        if (
            panel.target_estimand not in {"sales_given_availability", "observed_count"}
            or panel.status_prob_cols
            or panel.plan
            or panel.oos_col
            or panel.negative_target_policy != "refuse"
        ):
            raise ValueError("revisions support observed counts, exact/unobserved states, no plans")
        if route_lower_bound_to_exact and panel.target_estimand != "sales_given_availability":
            raise ValueError("lower-bound-to-exact routing is valid only for recorded sales")
        if observation_contract not in {
            "complete_recorded_sales_prelaunch_zeros_v1",
            "masked_observed_counts_v1",
            GLOBAL_LEFT_PADDING_CONTRACT,
        }:
            raise ValueError("unsupported revision observation contract")
        if any(
            name and name not in panel.known_future for name in (panel.price_col, panel.promo_col)
        ):
            raise ValueError("future commercial roles must be explicitly declared known_future")
        try:
            end = cutoff + dt.timedelta(days=horizon * stride)
        except OverflowError as exc:
            raise ValueError("origin/horizon exceeds the representable calendar") from exc
        source = ds.dataset(panel.path, format="parquet", partitioning="hive")
        categorical = set(data_spec.time_varying_known_categoricals if data_spec else ())
        categorical.update(data_spec.time_varying_unknown_categoricals if data_spec else ())
        future = tuple(
            dict.fromkeys(
                (
                    *panel.known_future,
                    *(data_spec.time_varying_known_reals if data_spec else ()),
                    *(data_spec.time_varying_known_categoricals if data_spec else ()),
                )
            )
        )
        historical_only = (
            tuple(
                (
                    *data_spec.time_varying_unknown_reals,
                    *data_spec.time_varying_unknown_categoricals,
                )
            )
            if data_spec
            else ()
        )
        static_reals = tuple(data_spec.static_reals) if data_spec else panel.static_reals
        static_categories = tuple(data_spec.static_categoricals) if data_spec else panel.static
        group_ids = tuple(data_spec.group_ids) if data_spec else ()
        roles = (
            *static_categories,
            *group_ids,
            *metadata_columns,
            panel.store_col,
            panel.product_col,
            panel.category_col,
            panel.factor_group,
        )
        static = tuple(dict.fromkeys(name for name in roles if name and name != panel.series_col))
        dynamic = tuple(dict.fromkeys((*future, *historical_only)))
        protected = {panel.series_col, panel.time_col, panel.target_col, panel.exposure_col}
        if (
            set(dynamic) & protected
            or set(dynamic) & set((*static, *static_reals))
            or set(future) & set(historical_only)
            or set(static) & set(static_reals)
            or set((*static, *static_reals)) & (protected - {panel.series_col})
        ):
            raise ValueError("revision feature roles overlap core, static or future/history roles")
        required = {
            panel.series_col,
            panel.time_col,
            panel.target_col,
            *static,
            *static_reals,
            *dynamic,
            *([panel.exposure_col] if panel.exposure_col else []),
        }
        if not required <= set(source.schema.names):
            raise ValueError(
                f"missing panel columns: {sorted(required - set(source.schema.names))}"
            )
        date_field = ds.field(panel.time_col)  # pyright: ignore[reportPrivateImportUsage]
        earliest = None
        for batch in source.scanner(
            columns=[panel.time_col],
            filter=(date_field <= pa.scalar(cutoff)) | date_field.is_null(),
        ).to_batches():
            column = batch.column(0)
            if column.null_count:
                raise ValueError("null panel date")
            value = pc.min(column).as_py()  # pyright: ignore[reportAttributeAccessIssue]
            if value is not None:
                earliest = value if earliest is None else min(earliest, value)
        if type(earliest) is not dt.date:
            raise ValueError("panel has no date-grain history at or before the origin")
        if (cutoff - earliest).days % stride:
            raise ValueError("origin is misaligned with the declared cadence")
        first = source.to_table(
            columns=[panel.series_col, *static, *static_reals],
            filter=date_field == pa.scalar(earliest),
        )
        first = first.sort_by([(panel.series_col, "ascending")])
        ids = tuple(first[panel.series_col].to_pylist())
        if (
            not ids
            or any(not isinstance(s, str) or not s.strip() for s in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("first date must contain unique nonempty series identifiers")
        metadata = {name: tuple(first[name].to_pylist()) for name in static}
        if any(
            any(not isinstance(v, str) or not v.strip() for v in values)
            for values in metadata.values()
        ):
            raise ValueError("revision identity columns must be nonnull strings")
        metadata[panel.series_col] = ids
        n, steps = len(ids), (cutoff - earliest).days // stride + 1
        widths = sum(264 if name in categorical else 8 for name in dynamic)
        required_bytes = n * (
            steps * (5 + int(route_lower_bound_to_exact))
            + (steps + horizon) * (1 + widths)
            + 8 * len(static_reals)
        )
        if required_bytes > max_bytes:
            raise ValueError(f"panel arrays require {required_bytes} bytes, budget is {max_bytes}")
        numeric_static = {name: first[name].to_numpy().astype(np.float64) for name in static_reals}
        if any(not np.isfinite(value).all() for value in numeric_static.values()):
            raise ValueError("static real covariates must be finite")
        target = np.full((n, steps), np.nan, dtype=np.float32)
        observed = np.zeros((n, steps), dtype=bool)
        physical_state = (
            np.full((n, steps), 2, dtype=np.int8) if route_lower_bound_to_exact else None
        )
        features = {
            name: np.full(
                (n, steps + (horizon if name in future else 0)),
                None if name in categorical else np.nan,
                dtype=object if name in categorical else np.float64,
            )
            for name in dynamic
        }
        seen = np.zeros((n, steps + horizon), dtype=bool)
        id_values = pa.array(ids)
        if not future:
            seen[:, steps:] = True
        for historical, condition in (
            (True, date_field <= pa.scalar(cutoff)),
            (False, (date_field > pa.scalar(cutoff)) & (date_field <= pa.scalar(end))),
        ):
            if not historical and not future:
                continue
            selected_features = dynamic if historical else future
            columns = list(
                dict.fromkeys(
                    [
                        panel.series_col,
                        panel.time_col,
                        *static,
                        *static_reals,
                        *selected_features,
                        *([panel.target_col] if historical else []),
                        *([panel.exposure_col] if historical and panel.exposure_col else []),
                    ]
                )
            )
            for batch in source.scanner(
                columns=columns, filter=condition, batch_size=131072
            ).to_batches():
                by_name = {name: batch.column(name) for name in columns}
                indices = pc.index_in(  # pyright: ignore[reportAttributeAccessIssue]
                    by_name[panel.series_col], value_set=id_values
                )
                if indices.null_count or by_name[panel.time_col].null_count:
                    raise ValueError("unknown series or null date in panel grid")
                rows = indices.to_numpy(zero_copy_only=False).astype(np.int64)
                dates = by_name[panel.time_col].to_numpy(zero_copy_only=False)
                offsets = (dates.astype("datetime64[D]") - np.datetime64(earliest)).astype(int)
                if np.any(offsets % stride):
                    raise ValueError("panel date is misaligned with declared cadence")
                positions = offsets // stride
                if np.any(positions < 0) or np.any(positions >= seen.shape[1]):
                    raise ValueError("date outside the declared panel grid")
                flat = rows * seen.shape[1] + positions
                if len(np.unique(flat)) != len(flat) or np.any(seen[rows, positions]):
                    raise ValueError("duplicate series/date in panel grid")
                for name in (*static, *static_reals):
                    values = metadata[name] if name in metadata else numeric_static[name]
                    expected = pc.take(pa.array(values), indices)
                    equal = pc.fill_null(pc.equal(expected, by_name[name]), False)  # pyright: ignore[reportAttributeAccessIssue]
                    if not pc.all(equal).as_py():  # pyright: ignore[reportAttributeAccessIssue]
                        raise ValueError(f"identity changes within a series: {name}")
                for name in selected_features:
                    if name in categorical:
                        values = by_name[name].to_pylist()
                        if any(
                            v is not None and (not isinstance(v, str) or len(v.encode()) > 256)
                            for v in values
                        ):
                            raise ValueError(
                                "categorical covariates require strings of <=256 bytes"
                            )
                    else:
                        values = by_name[name].to_numpy(zero_copy_only=False).astype(np.float64)
                        if np.isinf(values).any():
                            raise ValueError("covariates must be missing or finite")
                        if name == panel.price_col and np.any(values[np.isfinite(values)] < 0):
                            raise ValueError("price must be missing or finite and nonnegative")
                    features[name][rows, positions] = values
                seen[rows, positions] = True
                if historical:
                    raw_target = by_name[panel.target_col]
                    states = (
                        by_name[panel.exposure_col].to_pylist()
                        if panel.exposure_col
                        else ["exact"] * len(rows)
                    )
                    allowed_states = (
                        {"exact", "lower_bound", "unobserved"}
                        if route_lower_bound_to_exact
                        else {"exact", "unobserved"}
                    )
                    if any(s not in allowed_states for s in states):
                        message = (
                            "PACT received an unsupported physical observation state"
                            if route_lower_bound_to_exact
                            else "rolling revisions cannot fit censored/unknown states"
                        )
                        raise ValueError(message)
                    raw_state = np.array(
                        [{"exact": 0, "lower_bound": 1, "unobserved": 2}[s] for s in states],
                        dtype=np.int8,
                    )
                    mask = raw_state != 2 if route_lower_bound_to_exact else raw_state == 0
                    if pa.types.is_decimal(raw_target.type):
                        validate = (
                            np.ones(len(mask), bool)
                            if observation_contract.startswith("complete_recorded")
                            else mask
                        )
                        if any(
                            include
                            and (
                                value is None
                                or not value.is_finite()
                                or value < 0
                                or value > 2**24
                                or value != value.to_integral_value()
                            )
                            for value, include in zip(raw_target.to_pylist(), validate, strict=True)
                        ):
                            raise ValueError(
                                "observed targets must be finite integer counts in [0, 2**24]"
                            )
                    y = raw_target.to_numpy(zero_copy_only=False).astype(np.float64)
                    checked = y if observation_contract.startswith("complete_recorded") else y[mask]
                    if (
                        not np.isfinite(checked).all()
                        or np.any(checked < 0)
                        or np.any(checked > 2**24)
                        or np.any(checked != np.floor(checked))
                    ):
                        raise ValueError(
                            "observed targets must be finite integer counts in [0, 2**24]"
                        )
                    target[rows, positions] = (
                        y
                        if observation_contract.startswith("complete_recorded")
                        else np.where(mask, y, 0.0)
                    )
                    observed[rows, positions] = mask
                    if physical_state is not None:
                        physical_state[rows, positions] = raw_state
        if not seen.all():
            raise ValueError(f"incomplete panel grid: {int((~seen).sum())} missing cells")
        if observation_contract.startswith("complete_recorded") and (
            np.any(target[~observed] != 0) or np.any(observed[:, :-1] & ~observed[:, 1:])
        ):
            raise ValueError(
                "complete recorded sales required: unobserved cells must be leading "
                "prelaunch zeros, never unknown actuals"
            )
        price = features[panel.price_col] if panel.price_col else np.empty((n, 0), np.float64)
        arrays = [target, observed, price, *features.values(), *numeric_static.values()]
        if physical_state is not None:
            arrays.append(physical_state)
        for array in arrays:
            array.setflags(write=False)
        return cls(
            ids,
            earliest,
            cutoff,
            horizon,
            target,
            observed,
            price,
            metadata,
            stride,
            features,
            numeric_static,
            observation_contract,
            physical_state,
        )

    def position(self, origin: str) -> int:
        """Inclusive date to exclusive history endpoint; tested by ``test_position``."""
        offset = (dt.date.fromisoformat(origin) - self.start).days
        if offset % self.step_days:
            raise ValueError("origin is misaligned with declared cadence")
        position = offset // self.step_days + 1
        if position < 1 or position > self.target.shape[1]:
            raise ValueError("origin is outside available target history")
        return position

    def date_at(self, position: int) -> dt.date:
        """Zero-based period to date; tested by ``test_period_days_and_grid``."""
        return self.start + dt.timedelta(days=int(position) * self.step_days)

    def window(
        self,
        origin: str,
        rows: np.ndarray,
        encoder_days: int,
        *,
        targets: bool,
        reference_window: int = 28,
        left_pad: bool = False,
    ) -> dict[str, np.ndarray]:
        """Slice one origin, optionally masking dates before the global calendar begins."""
        stop = self.position(origin)
        if (
            type(encoder_days) is not int
            or encoder_days < 1
            or type(left_pad) is not bool
            or (encoder_days > stop and not left_pad)
        ):
            raise ValueError("insufficient encoder history")
        rows = np.asarray(rows)
        if (
            rows.ndim != 1
            or rows.dtype.kind not in "iu"
            or len(rows) == 0
            or np.any(rows < 0)
            or np.any(rows >= len(self.series_ids))
        ):
            raise ValueError("rows must be nonempty valid integer series indices")
        start = max(0, stop - encoder_days)
        padding = encoder_days - (stop - start)
        history = self.target[rows, start:stop].copy()
        observed = self.observed[rows, start:stop].copy()
        if padding:
            history = np.pad(history, ((0, 0), (padding, 0)), constant_values=0)
            observed = np.pad(observed, ((0, 0), (padding, 0)), constant_values=False)
        result = {
            "history": history,
            "observed": observed,
            "price_history": self.price[rows, max(0, stop - reference_window) : stop].copy(),
            "price_future": self.price[rows, stop : stop + self.horizon].copy(),
        }
        if left_pad:
            available = np.ones((len(rows), encoder_days), dtype=bool)
            available[:, :padding] = False
            result["available"] = available
        if targets:
            if stop + self.horizon > self.target.shape[1]:
                raise ValueError("training/validation target crosses the data cutoff")
            result["target"] = self.target[rows, stop : stop + self.horizon].copy()
            result["target_observed"] = self.observed[rows, stop : stop + self.horizon].copy()
        return result
