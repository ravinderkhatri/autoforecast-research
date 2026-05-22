"""Pure PACT token grammar and label routing.

The model never sees a future target or future observation state.  This module returns model
inputs and labels as separate dictionaries so that boundary is testable before Torch is imported.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

import numpy as np

from autoforecast.hmfnet_features import HMFNetConfig, history_buckets, summarize_history

EXACT = 0
LOWER_BOUND = 1
UNOBSERVED = 2
PACT_HEAD_LAWS = frozenset({"negbin", "zero_truncated_hurdle_negbin"})
PACT_POSITION_MODES = frozenset({"elapsed_day", "token_index"})


@dataclass(frozen=True)
class PACTConfig:
    """Complete, fingerprinted PACT architecture recipe."""

    recent: int = 56
    middle_end: int = 364
    middle_width: int = 7
    old_width: int = 28
    old_slots: int = 64
    span: int | None = None
    n_layers: int = 4
    ff_multiplier: int = 4
    rope_theta: float = 10_000.0
    position_mode: str = "elapsed_day"
    context_levels: tuple[tuple[str, ...], ...] = ()
    max_streams: int = 4
    head_law: str = "zero_truncated_hurdle_negbin"
    categorical_width: int = 8
    numeric_scale_floor: float = 1e-6
    max_aggregate_bytes: int = 200_000_000
    mask_policy: str = "prefix_causal_future_full_v1"
    token_policy: str = "v5_multiresolution_calendar_segments_v1"

    def __post_init__(self) -> None:
        if not isinstance(self.context_levels, list | tuple) or any(
            not isinstance(level, list | tuple) for level in self.context_levels
        ):
            raise ValueError("PACT context_levels must be a sequence of column sequences")
        levels = tuple(tuple(sorted(level)) for level in self.context_levels)
        if (
            any(
                not level
                or len(set(level)) != len(level)
                or any(not isinstance(name, str) or not name.strip() for name in level)
                for level in levels
            )
            or len({frozenset(level) for level in levels}) != len(levels)
        ):
            raise ValueError("PACT context levels must be unique nonempty column sets")
        object.__setattr__(self, "context_levels", levels)
        positive_ints = (
            self.recent,
            self.middle_end,
            self.middle_width,
            self.old_width,
            self.old_slots,
            self.n_layers,
            self.ff_multiplier,
            self.max_streams,
            self.categorical_width,
            self.max_aggregate_bytes,
        )
        if any(type(value) is not int or value < 1 for value in positive_ints):
            raise ValueError("PACT dimensions and budgets must be positive built-in integers")
        if (
            self.middle_end <= self.recent
            or (self.middle_end - self.recent) % self.middle_width
            or self.old_slots < 2
            or self.span is not None
            and (type(self.span) is not int or self.span < 1)
            or len(levels) + 1 > self.max_streams
            or levels
            and self.n_layers < 2
        ):
            raise ValueError("invalid PACT history bands, span, stream count or layer reachability")
        if (
            type(self.rope_theta) not in (int, float)
            or not math.isfinite(self.rope_theta)
            or self.rope_theta <= 0
            or type(self.numeric_scale_floor) not in (int, float)
            or not math.isfinite(self.numeric_scale_floor)
            or self.numeric_scale_floor <= 0
        ):
            raise ValueError("PACT numerical constants must be finite and positive")
        if self.position_mode not in PACT_POSITION_MODES:
            raise ValueError(f"unsupported PACT position mode {self.position_mode!r}")
        if self.head_law not in PACT_HEAD_LAWS:
            raise ValueError(f"unsupported PACT head law {self.head_law!r}")
        if self.mask_policy != "prefix_causal_future_full_v1":
            raise ValueError("unsupported PACT attention mask policy")
        if self.token_policy != "v5_multiresolution_calendar_segments_v1":
            raise ValueError("unsupported PACT token policy")

    def history_policy(self) -> HMFNetConfig:
        """Reuse HMFNet's validated summary operators without inheriting its model topology."""
        return HMFNetConfig(
            recent=self.recent,
            middle_end=self.middle_end,
            middle_width=self.middle_width,
            old_width=self.old_width,
            old_slots=self.old_slots,
            span=self.span,
            local_blocks=1,
            phase_cycles=((7.0, 3), (365.25, 3)),
            context_levels=self.context_levels,
            expert_levels=(),
            feedback=False,
            categorical_width=self.categorical_width,
            numeric_scale_floor=float(self.numeric_scale_floor),
            max_aggregate_bytes=self.max_aggregate_bytes,
        )


@dataclass(frozen=True)
class CalendarSegment:
    """One Sunday–Saturday forecast segment with day indices in weekday slots."""

    day_indices: tuple[int, ...]
    slots: tuple[int, ...]
    kind: int

    @property
    def active_days(self) -> int:
        return len(self.day_indices)


def route_sales_states(states: np.ndarray, target_estimand: str) -> np.ndarray:
    """Route physical label states without turning an unobserved zero into a sale."""
    states = np.asarray(states)
    if states.dtype.kind not in "iu" or np.any((states < EXACT) | (states > UNOBSERVED)):
        raise ValueError("physical states must use 0=exact, 1=lower_bound, 2=unobserved")
    if target_estimand != "sales_given_availability":
        raise ValueError("core PACT implements sales_given_availability only")
    return np.where(states == LOWER_BOUND, EXACT, states).astype(np.int8, copy=False)


def calendar_segments(origin: dt.date, horizon: int) -> tuple[CalendarSegment, ...]:
    """Partition future days exactly once into Sunday–Saturday calendar segments."""
    if not isinstance(origin, dt.date) or isinstance(origin, dt.datetime):
        raise ValueError("origin must be a datetime.date")
    if type(horizon) is not int or horizon < 1:
        raise ValueError("horizon must be a positive built-in integer")
    grouped: list[list[int]] = []
    keys: list[tuple[int, int]] = []
    for index in range(horizon):
        day = origin + dt.timedelta(days=index + 1)
        sunday = day - dt.timedelta(days=(day.weekday() + 1) % 7)
        key = (sunday.year, sunday.toordinal())
        if not keys or key != keys[-1]:
            keys.append(key)
            grouped.append([])
        grouped[-1].append(index)
    result = []
    for number, indices in enumerate(grouped):
        slots = [-1] * 7
        for index in indices:
            day = origin + dt.timedelta(days=index + 1)
            slots[(day.weekday() + 1) % 7] = index
        full = len(indices) == 7
        leading = number == 0 and slots[0] < 0
        trailing = number == len(grouped) - 1 and slots[-1] < 0
        kind = 3 if full else 6 if leading and trailing else 4 if leading else 5
        result.append(CalendarSegment(tuple(indices), tuple(slots), kind))
    covered = [index for segment in result for index in segment.day_indices]
    if covered != list(range(horizon)):
        raise ValueError("future calendar segmentation is not ordered exact-once")
    return tuple(result)


def _state_array(observed: np.ndarray, physical_state: np.ndarray | None) -> np.ndarray:
    if physical_state is None:
        return np.where(observed, EXACT, UNOBSERVED).astype(np.int8)
    states = np.asarray(physical_state)
    if states.shape != observed.shape or states.dtype.kind not in "iu":
        raise ValueError("physical history states must be integer and target-shaped")
    if np.any((states < EXACT) | (states > UNOBSERVED)):
        raise ValueError("unknown physical history state")
    return states


def build_history_tokens(
    values: np.ndarray,
    observed: np.ndarray,
    positions: np.ndarray,
    *,
    origin_stop: int,
    first_ordinal: int,
    step_days: int,
    config: PACTConfig,
    statistic_windows: tuple[int, ...],
    physical_state: np.ndarray | None = None,
    numeric: tuple[np.ndarray, ...] = (),
    categorical: tuple[np.ndarray, ...] = (),
    categorical_raw: tuple[np.ndarray, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Build chronological PACT history tokens from HMFNet's bounded summary operators."""
    values = np.asarray(values)
    observed = np.asarray(observed)
    positions = np.asarray(positions)
    if values.ndim != 2 or observed.shape != values.shape or observed.dtype != np.bool_:
        raise ValueError("PACT history values and observed mask must be aligned rank-2 arrays")
    if positions.shape != (values.shape[1],) or positions.dtype.kind not in "iu":
        raise ValueError("PACT history positions must be an integer vector")
    states = _state_array(observed, physical_state)
    policy = config.history_policy()
    base = summarize_history(
        values,
        observed,
        positions,
        origin_stop=origin_stop,
        first_ordinal=first_ordinal,
        step_days=step_days,
        policy=policy,
        statistic_windows=statistic_windows,
        numeric=numeric,
        categorical=categorical,
        categorical_raw=categorical_raw,
    )
    ages = origin_stop - 1 - positions
    included = (ages >= 0) & (ages < policy.span if policy.span is not None else True)
    extent = int(ages[included].max()) + 1 if included.any() else 0
    newest = history_buckets(extent, policy)
    if not newest:
        raise ValueError("PACT history has no calendar position before the forecast origin")
    n_rows, n_tokens = values.shape[0], len(newest)
    additions = np.zeros((n_rows, n_tokens, 17), np.float64)
    for token, (low, high, _band, _regular) in enumerate(newest):
        at = np.flatnonzero(included & (ages >= low) & (ages < high))
        if not len(at):
            raise ValueError("PACT history descriptor has no source day")
        token_states = states[:, at]
        additions[:, token, :3] = np.stack(
            [(token_states == code).sum(1) / len(at) for code in range(3)], axis=-1
        )
        day_slots = (first_ordinal + positions[at] * step_days) % 7
        exact = observed[:, at]
        safe = np.where(exact, values[:, at], 0.0)
        scale = np.maximum(safe.sum(1, keepdims=True), 1.0)
        for slot in range(7):
            slot_mask = day_slots == slot
            slot_exact = exact[:, slot_mask]
            additions[:, token, 3 + slot] = (
                np.where(slot_exact, safe[:, slot_mask], 0.0).sum(1) / scale[:, 0]
                if slot_mask.any()
                else 0.0
            )
            additions[:, token, 10 + slot] = (
                slot_exact.sum(1) / int(slot_mask.sum()) if slot_mask.any() else 0.0
            )
    order = np.arange(n_tokens - 1, -1, -1)
    descriptors = [newest[index] for index in order]
    end_positions = np.array([origin_stop - low - 1 for low, _high, _b, _r in descriptors])
    start_positions = np.array([origin_stop - high for _low, high, _b, _r in descriptors])
    if np.any(np.diff(end_positions) <= 0):
        raise ValueError("PACT history tokens are not strictly chronological")
    covered = sorted(
        position
        for low, high, _band, _regular in newest
        for position in range(origin_stop - high, origin_stop - low)
    )
    expected = list(range(origin_stop - extent, origin_stop))
    if covered != expected:
        raise ValueError("PACT history buckets do not cover source days exactly once")
    features = np.concatenate((base["memory_values"], additions.astype(np.float32)), axis=-1)
    end_ordinals = first_ordinal + end_positions * step_days
    origin_ordinal = first_ordinal + (origin_stop - 1) * step_days
    elapsed = end_ordinals - origin_ordinal
    positions_out = elapsed if config.position_mode == "elapsed_day" else np.arange(n_tokens)
    result = {
        "history_features": features[:, order],
        "history_exists": np.ones((n_rows, n_tokens), dtype=bool),
        "history_positions": positions_out.astype(np.int64),
        "history_types": np.array([value[2] for value in descriptors], np.int64),
        "history_widths": (end_positions - start_positions + 1).astype(np.int64),
        "history_regular": np.array([value[3] for value in descriptors], bool),
        "statistics": base["statistics"],
        "quality": base["quality"],
        "history_categorical": base["memory_categorical"][:, order],
        "history_categorical_mask": base["memory_categorical_mask"][:, order],
    }
    if not all(np.isfinite(value).all() for value in (features, result["statistics"])):
        raise ValueError("PACT history construction produced a nonfinite feature")
    return result


def build_future_segments(
    known_future: np.ndarray,
    *,
    origin: dt.date,
    target: np.ndarray | None = None,
    physical_state: np.ndarray | None = None,
    target_estimand: str = "sales_given_availability",
    leading_prefix: np.ndarray | None = None,
    leading_prefix_mask: np.ndarray | None = None,
    position_mode: str = "elapsed_day",
) -> dict[str, np.ndarray]:
    """Build target-free future query inputs and separate complete-case segment labels."""
    known = np.asarray(known_future)
    if known.ndim != 3 or not np.issubdtype(known.dtype, np.number):
        raise ValueError("known_future must be a numeric [series, horizon, features] array")
    if not np.isfinite(known).all():
        raise ValueError("known_future must be canonicalized to finite values before PACT")
    n_rows, horizon, width = known.shape
    segments = calendar_segments(origin, horizon)
    n_segments = len(segments)
    feature_width = 7 * width + 7 + 7 + 4 + 2 + 14
    features = np.zeros((n_rows, n_segments, feature_width), np.float32)
    active = np.zeros((n_segments, 7), bool)
    day_index = np.full((n_segments, 7), -1, np.int64)
    positions = np.zeros(n_segments, np.int64)
    types = np.zeros(n_segments, np.int64)
    prefix = np.zeros((n_rows, 7), np.float32)
    prefix_mask = np.zeros((n_rows, 7), bool)
    if leading_prefix is not None or leading_prefix_mask is not None:
        if leading_prefix is None or leading_prefix_mask is None:
            raise ValueError("leading prefix values and mask must be supplied together")
        prefix = np.asarray(leading_prefix, dtype=np.float32)
        prefix_mask = np.asarray(leading_prefix_mask)
        if prefix.shape != (n_rows, 7) or prefix_mask.shape != prefix.shape:
            raise ValueError("leading prefix must have one Sunday–Saturday row per series")
        if prefix_mask.dtype != np.bool_ or not np.isfinite(prefix[prefix_mask]).all():
            raise ValueError("leading prefix mask/value domain is invalid")
    for segment_index, segment in enumerate(segments):
        slot_values = np.zeros((n_rows, 7, width), np.float32)
        relative = np.zeros(7, np.float32)
        for slot, index in enumerate(segment.slots):
            if index < 0:
                continue
            active[segment_index, slot] = True
            day_index[segment_index, slot] = index
            slot_values[:, slot] = known[:, index]
            relative[slot] = (index + 1) / horizon
        flags = np.array(
            [segment.kind == 3, segment.kind == 4, segment.kind == 5, segment.kind == 6],
            np.float32,
        )
        columns = [
            slot_values.reshape(n_rows, -1),
            np.broadcast_to(active[segment_index], (n_rows, 7)),
            np.broadcast_to(relative, (n_rows, 7)),
            np.broadcast_to(flags, (n_rows, 4)),
            np.full((n_rows, 1), segment.active_days / 7.0, np.float32),
            np.full((n_rows, 1), math.log1p(segment.active_days), np.float32),
            np.stack((np.log1p(np.where(prefix_mask, prefix, 0.0)), prefix_mask), -1).reshape(
                n_rows, -1
            )
            if segment_index == 0
            else np.zeros((n_rows, 14), np.float32),
        ]
        features[:, segment_index] = np.concatenate(columns, axis=1)
        last = segment.day_indices[-1] + 1
        positions[segment_index] = last if position_mode == "elapsed_day" else segment_index
        types[segment_index] = segment.kind
    result = {
        "future_features": features,
        "future_exists": np.ones((n_rows, n_segments), bool),
        "future_active": active,
        "future_day_index": day_index,
        "future_positions": positions,
        "future_types": types,
    }
    if target is None and physical_state is None:
        return result
    if target is None or physical_state is None:
        raise ValueError("future target and physical state must be supplied together")
    target = np.asarray(target)
    states = np.asarray(physical_state)
    if target.shape != (n_rows, horizon) or states.shape != target.shape:
        raise ValueError("future targets and states must match known_future rows/horizon")
    routed = route_sales_states(states, target_estimand)
    exact_values = target[routed == EXACT]
    if (
        not np.isfinite(exact_values).all()
        or np.any(exact_values < 0)
        or np.any(exact_values != np.floor(exact_values))
    ):
        raise ValueError("exact PACT labels must be finite nonnegative integer counts")
    counts = np.zeros((n_rows, n_segments, 7), np.float32)
    exact = np.zeros((n_rows, n_segments, 7), bool)
    for segment_index, segment in enumerate(segments):
        for slot, index in enumerate(segment.slots):
            if index >= 0:
                counts[:, segment_index, slot] = np.where(
                    routed[:, index] == EXACT, target[:, index], 0.0
                )
                exact[:, segment_index, slot] = routed[:, index] == EXACT
    eligible = np.all(~active[None, :, :] | exact, axis=-1)
    result.update(
        target_counts=counts,
        target_exact=exact,
        segment_eligible=eligible,
        objective_day_mask=_objective_day_mask(eligible, active, day_index, horizon),
    )
    return result


def _objective_day_mask(
    eligible: np.ndarray, active: np.ndarray, day_index: np.ndarray, horizon: int
) -> np.ndarray:
    result = np.zeros((eligible.shape[0], horizon), bool)
    for segment in range(active.shape[0]):
        for slot in np.flatnonzero(active[segment]):
            result[:, day_index[segment, slot]] = eligible[:, segment]
    return result


__all__ = [
    "CalendarSegment",
    "EXACT",
    "LOWER_BOUND",
    "UNOBSERVED",
    "PACTConfig",
    "PACT_HEAD_LAWS",
    "PACT_POSITION_MODES",
    "build_future_segments",
    "build_history_tokens",
    "calendar_segments",
    "route_sales_states",
]
