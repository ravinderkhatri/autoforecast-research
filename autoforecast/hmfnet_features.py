"""Deterministic HMFNet data summaries, not a forecasting implementation.

Case tables precede this module in the model regression tests. Structural tests
are in test_hmfnet_features.py; real-data integration is in test_hmfnet_revision.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from autoforecast.rolling_data import PanelArrays


@dataclass(frozen=True)
class HMFNetConfig:
    """Explicit non-claim-bearing recipe; validation is exercised by test_hmfnet_config."""

    recent: int = 56
    middle_end: int = 364
    middle_width: int = 7
    old_width: int = 28
    old_slots: int = 64
    span: int | None = None
    local_blocks: int = 2
    phase_cycles: tuple[tuple[float, int], ...] = ()
    context_levels: tuple[tuple[str, ...], ...] = ()
    expert_levels: tuple[tuple[str, ...], ...] = ()
    feedback: bool = True
    expert_batch_per_group: int = 32
    expert_loss_weight: float = 0.1
    expert_seed: int = 43
    anchor_floor: float = 1e-6
    numeric_scale_floor: float = 1e-6
    categorical_width: int = 8
    # Bounds aggregate ndarray payload and bounded numerical scratch, not caller data or RSS.
    max_aggregate_bytes: int = 200_000_000
    feature_policy: str = "masked_moments_phase_v1"
    aggregate_covariates: str = "none"

    def __post_init__(self) -> None:
        """Reject ambiguous groups, coercions and unsupported recipes; test_hmfnet_config."""
        for name in ("context_levels", "expert_levels"):
            raw = getattr(self, name)
            if not isinstance(raw, list | tuple) or any(
                not isinstance(v, list | tuple) for v in raw
            ):
                raise ValueError("HMFNet grouping sets must be sequences of column sequences")
            groups = tuple(tuple(v) for v in raw)
            if any(
                any(not isinstance(s, str) or not s.strip() for s in g)
                or len(set(g)) != len(g)
                for g in groups
            ):
                raise ValueError("HMFNet grouping columns must be unique nonempty strings")
            if len({frozenset(g) for g in groups}) != len(groups):
                raise ValueError("HMFNet grouping sets contain duplicate aggregations")
            object.__setattr__(self, name, tuple(tuple(sorted(g)) for g in groups))
        if not isinstance(self.phase_cycles, list | tuple) or any(
            not isinstance(v, list | tuple) for v in self.phase_cycles
        ):
            raise ValueError("HMFNet phase cycles must be a sequence")
        cycles = tuple(tuple(v) for v in self.phase_cycles)
        if any(
            len(v) != 2
            or type(v[0]) not in (int, float)
            or not math.isfinite(v[0])
            or v[0] <= 0
            or type(v[1]) is not int
            or v[1] < 1
            for v in cycles
        ):
            raise ValueError("HMFNet phase cycles require positive days and integer harmonics")
        if len({v[0] for v in cycles}) != len(cycles):
            raise ValueError("HMFNet phase periods must be unique")
        object.__setattr__(self, "phase_cycles", cycles)
        counts = (
            self.recent,
            self.middle_end,
            self.middle_width,
            self.old_width,
            self.old_slots,
            self.local_blocks,
            self.expert_batch_per_group,
            self.categorical_width,
            self.max_aggregate_bytes,
        )
        if any(type(v) is not int or v < 1 for v in counts):
            raise ValueError("HMFNet dimensions and budgets must be positive integers")
        if (
            self.middle_end <= self.recent
            or (self.middle_end - self.recent) % self.middle_width
            or self.old_slots < 2
            or self.span is not None
            and (type(self.span) is not int or self.span < 1)
            or type(self.expert_seed) is not int
            or self.expert_seed < 0
            or type(self.feedback) is not bool
        ):
            raise ValueError("invalid HMFNet bands, span, seed or feedback flag")
        if any(
            type(v) not in (int, float) or not math.isfinite(v) or v <= 0
            for v in (self.anchor_floor, self.numeric_scale_floor)
        ):
            raise ValueError("HMFNet numerical floors must be finite positive numbers")
        if (
            type(self.expert_loss_weight) not in (int, float)
            or not math.isfinite(self.expert_loss_weight)
            or self.expert_loss_weight < 0
            or self.feature_policy != "masked_moments_phase_v1"
            or self.aggregate_covariates != "none"
        ):
            raise ValueError("unsupported HMFNet objective or feature policy")


def history_buckets(length: int, policy: HMFNetConfig) -> tuple[tuple[int, int, int, bool], ...]:
    """Newest-first half-open age intervals; test_history_buckets checks exact-once coverage."""
    if type(length) is not int or length < 0:
        raise ValueError("history length must be a nonnegative integer")
    length = min(length, policy.span) if policy.span is not None else length
    result = []
    for band, low, high, width in (
        (0, 0, min(length, policy.recent), 1),
        (1, policy.recent, min(length, policy.middle_end), policy.middle_width),
        (2, policy.middle_end, length, policy.old_width),
    ):
        for number, left in enumerate(range(low, high, width)):
            overflow = band == 2 and number == policy.old_slots - 1 and left + width < high
            right = high if overflow else min(left + width, high)
            result.append((left, right, band, right - left == width and not overflow))
            if overflow:
                break
    return tuple(result)


def summarize_history(
    values: np.ndarray,
    observed: np.ndarray,
    positions: np.ndarray,
    *,
    origin_stop: int,
    first_ordinal: int,
    step_days: int,
    policy: HMFNetConfig,
    statistic_windows: tuple[int, ...],
    position_available: np.ndarray | None = None,
    numeric: tuple[np.ndarray, ...] = (),
    categorical: tuple[np.ndarray, ...] = (),
    categorical_raw: tuple[np.ndarray, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Masked float64 summaries; test_summarize_history and its covariate cases define formulas.

    Numeric fields use NaN for unknowns. Categorical fields are integral IDs, -1 for unknown.
    Position availability is independent of field observation. No learned state is cached here.
    """
    values, observed, positions = np.asarray(values), np.asarray(observed), np.asarray(positions)
    comparisons = categorical if categorical_raw is None else categorical_raw
    available = (
        np.ones(len(positions), bool)
        if position_available is None
        else np.asarray(position_available)
    )
    if (
        values.ndim != 2
        or observed.shape != values.shape
        or observed.dtype != np.bool_
        or positions.shape != (values.shape[1],)
        or positions.dtype.kind not in "iu"
        or available.shape != positions.shape
        or available.dtype != np.bool_
        or type(origin_stop) is not int
        or type(step_days) is not int
        or step_days < 1
        or any(type(w) is not int or w < 1 for w in statistic_windows)
        or len(comparisons) != len(categorical)
    ):
        raise ValueError("invalid HMFNet history shapes, masks, positions or statistic windows")
    if (
        np.any(np.diff(positions) <= 0)
        or np.any(positions[available] >= origin_stop)
        or any(np.shape(x) != values.shape for x in (*numeric, *categorical, *comparisons))
    ):
        raise ValueError("history positions must be ordered, historical and shape-aligned")
    ages = origin_stop - 1 - positions
    included = available & (ages >= 0)
    if policy.span is not None:
        included &= ages < policy.span
    mask = observed & included[None, :]
    checked = values[mask]
    if (
        not np.isfinite(checked).all()
        or np.any(checked < 0)
        or np.any(checked > 2**53)
        or np.any(checked != np.floor(checked))
    ):
        raise ValueError("observed summaries need exact finite nonnegative integer counts")
    value = np.where(mask, values, 0).astype(np.float64)
    n_rows = values.shape[0]
    count = mask.sum(1)
    denom = np.maximum(count, 1)
    mean = value.sum(1) / denom
    scale = np.maximum(mean, 1)
    variance = np.where(mask, (value - mean[:, None]) ** 2, 0).sum(1) / denom
    positive = mask & (value > 0)
    positives = positive.sum(1)
    extent = int(ages[included].max()) + 1 if included.any() else 0
    buckets = history_buckets(extent, policy)
    # One masked placeholder slot keeps the tensor interface nonempty; attention adds its null.
    descriptors = buckets or ((0, 0, 0, False),)
    n_tokens = len(descriptors)
    width = (
        11 + 4 * sum(h for _, h in policy.phase_cycles) + 4 * len(numeric) + 3 * len(categorical)
    )
    output = np.zeros((n_rows, n_tokens, width), np.float64)
    token_mask = np.zeros((n_rows, n_tokens), bool)
    cat_ids = np.zeros((n_rows, n_tokens, len(categorical)), np.int64)
    cat_mask = np.zeros_like(cat_ids, bool)
    bands, regular = [], []
    numeric_fields = []
    for raw in numeric:
        raw = np.asarray(raw, dtype=np.float64)
        if np.isinf(raw).any():
            raise ValueError("numeric covariates must be finite or NaN for unavailable")
        field_mask = np.isfinite(raw) & included[None, :]
        safe = np.where(field_mask, raw, 0)
        d = np.maximum(field_mask.sum(1), 1)
        center = safe.sum(1) / d
        std = np.sqrt(np.where(field_mask, (safe - center[:, None]) ** 2, 0).sum(1) / d)
        numeric_fields.append(
            (raw, field_mask, center, np.maximum(std, policy.numeric_scale_floor))
        )
    for raw in categorical:
        if np.asarray(raw).dtype.kind not in "iu" or np.any(np.asarray(raw) < -1):
            raise ValueError("categorical history must use integer IDs and -1 for missing")
    for token, (low, high, band, is_regular) in enumerate(descriptors):
        at = np.flatnonzero(included & (ages >= low) & (ages < high))
        bands.append(band)
        regular.append(is_regular and len(at) == high - low)
        if not len(at):
            continue
        m, y = mask[:, at], value[:, at]
        n = m.sum(1)
        d = np.maximum(n, 1)
        average = y.sum(1) / d
        std = np.sqrt(np.where(m, (y - average[:, None]) ** 2, 0).sum(1) / d)
        last = np.where(m, np.arange(len(at))[None, :], -1).max(1)
        last_value = y[np.arange(n_rows), np.maximum(last, 0)]
        last_age = np.where(n > 0, ages[at[np.maximum(last, 0)]], 0)
        columns = [
            average / scale,
            std / scale,
            y.max(1) / scale,
            last_value / scale,
            ((y > 0) & m).sum(1) / d,
            np.log1p(n),
            n / len(at),
            (n > 0).astype(float),
            np.full(n_rows, np.log1p(low)),
            np.full(n_rows, np.log1p(high - low)),
            np.log1p(last_age),
        ]
        times = first_ordinal + positions[at] * step_days
        for period, harmonics in policy.phase_cycles:
            for harmonic in range(1, harmonics + 1):
                angle = 2 * np.pi * harmonic * times / period
                for trig in (np.sin(angle), np.cos(angle)):
                    columns.append((y * trig).sum(1) / d / scale)
                for trig in (np.sin(angle), np.cos(angle)):
                    columns.append((m * trig).sum(1) / d)
        active = n > 0
        for raw, field_mask, center, std_scale in numeric_fields:
            fm = field_mask[:, at]
            fn = fm.sum(1)
            fd = np.maximum(fn, 1)
            fy = np.where(fm, raw[:, at], 0)
            last = np.where(fm, np.arange(len(at))[None, :], -1).max(1)
            mean_feature = (fy.sum(1) / fd - center) / std_scale
            last_feature = (fy[np.arange(n_rows), np.maximum(last, 0)] - center) / std_scale
            columns.extend(
                [
                    np.where(fn > 0, mean_feature, 0),
                    np.where(fn > 0, last_feature, 0),
                    fn / len(at),
                    (fn > 0).astype(float),
                ]
            )
            active |= fn > 0
        for column, raw in enumerate(categorical):
            raw = np.asarray(raw)
            ids = raw[:, at]
            fm = ids >= 0
            fn = fm.sum(1)
            last = np.where(fm, np.arange(len(at))[None, :], -1).max(1)
            cat_ids[:, token, column] = np.where(
                fn > 0, ids[np.arange(n_rows), np.maximum(last, 0)], 0
            )
            cat_mask[:, token, column] = fn > 0
            pairs = fm[:, 1:] & fm[:, :-1] & (np.diff(positions[at])[None, :] == 1)
            pair_count = pairs.sum(1)
            original = np.asarray(comparisons[column])[:, at]
            changes = (pairs & (original[:, 1:] != original[:, :-1])).sum(1)
            columns.extend(
                [fn / len(at), changes / np.maximum(pair_count, 1), (pair_count > 0).astype(float)]
            )
            active |= fn > 0
        output[:, token] = np.stack(columns, -1)
        token_mask[:, token] = active
    output = np.where(token_mask[..., None], output, 0)
    last_positive = np.where(positive, ages[None, :], np.inf).min(1, initial=np.inf)
    last_observed = np.where(mask, ages[None, :], np.inf).min(1, initial=np.inf)
    last_index = np.where(mask, np.arange(values.shape[1])[None, :], -1).max(1, initial=-1)
    last_value = (
        value[np.arange(n_rows), np.maximum(last_index, 0)] if value.shape[1] else np.zeros(n_rows)
    )
    maximum = value.max(1, initial=0)
    stats = [
        np.log1p(mean),
        np.log1p(np.sqrt(variance)),
        positives / denom,
        np.log1p(value.sum(1) / np.maximum(positives, 1)),
        count / max(int(included.sum()), 1),
        np.log1p(positives),
        np.log1p(np.where(positives > 0, last_positive, 0)),
        (positives > 0).astype(float),
        (count > 0).astype(float),
    ]
    for window in statistic_windows:
        recent_mask = mask & (ages[None, :] < window)
        recent = np.where(recent_mask, value, 0).sum(1) / np.maximum(recent_mask.sum(1), 1)
        stats.append(np.log1p(recent) - np.log1p(mean))
    stats.extend([np.log1p(last_value), np.log1p(maximum) - np.log1p(mean)])
    statistics = np.stack(stats, -1)
    quality = np.stack(
        (
            count / max(int(included.sum()), 1),
            np.log1p(count),
            np.log1p(np.where(count > 0, last_observed, 0)),
        ),
        -1,
    )
    if not all(np.isfinite(v).all() for v in (output, statistics, quality)):
        raise ValueError("nonfinite HMFNet summary; refusing numerical overflow")
    return dict(
        memory_values=output.astype(np.float32),
        memory_mask=token_mask,
        memory_categorical=cat_ids,
        memory_categorical_mask=cat_mask,
        memory_bands=np.array(bands, np.int64),
        memory_regular=np.array(regular, bool),
        statistics=statistics.astype(np.float32),
        quality=quality.astype(np.float32),
    )


class AggregateHistory:
    """Small declared binary grouping maps and exact totals; test_aggregate_history."""

    def __init__(self, data: PanelArrays, policy: HMFNetConfig) -> None:
        """Freeze only admitted static membership and bound allocations; test_aggregate_history."""
        self.data, self.policy = data, policy
        self.levels = tuple(dict.fromkeys((*policy.context_levels, *policy.expert_levels)))
        index_bytes = (
            len(data.series_ids)
            * np.dtype(np.int64).itemsize
            * (
                len(self.levels)
                + len(policy.context_levels)
                + len(policy.expert_levels)
            )
        )
        if index_bytes > policy.max_aggregate_bytes:
            raise ValueError(
                f"aggregate NumPy arrays need at least {index_bytes} bytes; "
                "configured budget exceeded"
            )
        self.codes, self.slices = {}, {}
        self.node_keys = []
        total = 0
        for group in self.levels:
            if not set(group) <= set(data.metadata):
                raise ValueError("HMFNet aggregate grouping names an absent metadata column")
            keys = (
                list(zip(*(data.metadata[g] for g in group), strict=True))
                if group
                else [()] * len(data.series_ids)
            )
            unique = {key: i for i, key in enumerate(sorted(set(keys)))}
            self.node_keys.extend(
                dict(node_id=total + index, columns=list(group), values=list(key))
                for key, index in unique.items()
            )
            codes = np.array([unique[key] for key in keys], np.int64)
            self.codes[group] = codes
            self.slices[group] = slice(total, total + len(unique))
            total += len(unique)
        matrix_cells = total * data.target.shape[1]
        code_bytes = sum(codes.nbytes for codes in self.codes.values())
        retained_arrays = code_bytes + matrix_cells * 9 + total * 8
        chunk_cells = min(len(data.series_ids), 512) * data.target.shape[1] if total else 0
        # Conservatively reserve the integer/count matrix, its Boolean comparison and bincount,
        # plus bounded masking, floor/domain checks and the int64 chunk conversion. Release each
        # chunk and count matrix before moving to the next allocation stage.
        chunk_scratch = chunk_cells * (2 * data.target.dtype.itemsize + 9)
        aggregation_peak = retained_arrays + matrix_cells * 9 + total * 8 + chunk_scratch
        retained_peak = (
            retained_arrays
            + len(data.series_ids)
            * 8
            * (len(policy.context_levels) + len(policy.expert_levels))
        )
        needed = max(aggregation_peak, retained_peak)
        if needed > policy.max_aggregate_bytes:
            raise ValueError(
                f"aggregate NumPy arrays need {needed} bytes; configured budget exceeded"
            )
        self.target = np.zeros((total, data.target.shape[1]), np.float64)
        self.observed = np.zeros(self.target.shape, bool)
        self.type_index = np.zeros(total, np.int64)
        recorded = data.observation_contract == "complete_recorded_sales_prelaunch_zeros_v1"
        for index, group in enumerate(self.levels):
            rows, codes = self.slices[group], self.codes[group]
            totals = self.target[rows]
            # Use chunks to avoid a full-panel masked copy. Exact integer accumulation also
            # detects 2**53+1, which a float64 addition could round back to the accepted bound.
            integer_totals = np.zeros(totals.shape, np.int64)
            for start in range(0, len(data.series_ids), 512):
                raw = data.target[start : start + 512]
                safe = raw if recorded else np.where(data.observed[start : start + 512], raw, 0)
                if (
                    not np.isfinite(safe).all()
                    or np.any(safe < 0)
                    or np.any(safe > 2**53)
                    or np.any(safe != np.floor(safe))
                ):
                    raise ValueError("aggregate source exceeds the exact-integer domain")
                integer_values = safe.astype(np.int64)
                np.add.at(integer_totals, codes[start : start + 512], integer_values)
                del safe, integer_values
                if np.any(integer_totals > 2**53) or np.any(integer_totals < 0):
                    raise ValueError("aggregate totals exceed the finite exact-integer domain")
            totals[:] = integer_totals
            del integer_totals
            if recorded:
                self.observed[rows] = True
            else:
                counts = np.zeros(totals.shape, np.int64)
                np.add.at(counts, codes, data.observed)
                self.observed[rows] = counts == np.bincount(codes)[:, None]
                totals[~self.observed[rows]] = 0
                del counts
            self.type_index[rows] = index
        self.context = self.memberships(policy.context_levels)
        self.experts = self.memberships(policy.expert_levels)

    def memberships(self, levels: tuple[tuple[str, ...], ...]) -> np.ndarray:
        """Node IDs for each declared membership; test_aggregate_history includes crossed maps."""
        result = np.empty((len(self.data.series_ids), len(levels)), np.int64)
        for index, group in enumerate(levels):
            np.add(self.codes[group], self.slices[group].start, out=result[:, index])
        return result

    def anchors(self, bottom: np.ndarray) -> np.ndarray:
        """Sum origin-known seasonal means, not neural predictions; test_aggregate_history."""
        if (
            bottom.ndim != 2
            or bottom.shape[0] != len(self.data.series_ids)
            or not np.isfinite(bottom).all()
            or np.any(bottom < 0)
        ):
            raise ValueError("expert anchors require finite nonnegative full-population means")
        result = np.zeros((len(self.target), bottom.shape[1]), np.float64)
        for group in self.levels:
            np.add.at(result[self.slices[group]], self.codes[group], bottom)
        if not np.isfinite(result).all():
            raise ValueError("aggregate anchor overflow")
        return np.maximum(result, self.policy.anchor_floor)

    def supervision(self, stop: int, weighting: str, weight_window: int) -> dict[str, np.ndarray]:
        """Separate complete-label eligibility from feedback; test_expert_supervision."""
        data = self.data
        if (
            type(stop) is not int
            or stop < 1
            or stop + data.horizon > data.target.shape[1]
            or weighting not in {"uniform", "revenue"}
            or type(weight_window) is not int
            or weight_window < 1
        ):
            raise ValueError("expert supervision needs a historical origin and complete horizon")
        history = self.target[:, :stop]
        target = self.target[:, stop : stop + data.horizon]
        history_complete = self.observed[:, :stop].all(1)
        target_complete = self.observed[:, stop : stop + data.horizon].all(1)
        eligible = history_complete & target_complete
        first = (history > 0).argmax(1) if len(history) else np.empty(0, np.int64)
        difference = np.diff(history, axis=1) ** 2
        use = np.arange(1, stop)[None, :] > first[:, None]
        raw_scale = (difference * use).sum(1) / np.maximum(stop - first - 1, 1)
        scale = np.maximum(raw_scale, 1)
        if not np.isfinite(scale).all():
            raise ValueError("nonfinite expert scale")
        revenue = np.ones(len(data.series_ids), np.float64)
        price_valid = np.ones(len(data.series_ids), bool)
        if weighting == "revenue":
            if not data.price.shape[1] or stop < weight_window:
                raise ValueError("expert revenue needs prices and a complete weight window")
            sold = data.target[:, stop - weight_window : stop]
            prices = data.price[:, stop - weight_window : stop]
            valid = np.isfinite(prices) & (prices >= 0)
            price_valid = ((sold == 0) | valid).all(1)
            revenue = np.where((sold > 0) & valid, sold * np.where(valid, prices, 0), 0).sum(1)
        weights = np.zeros(len(history), np.float64)
        aggregate_price_valid = np.ones(len(history), bool)
        for group in self.policy.expert_levels:
            rows, codes = self.slices[group], self.codes[group]
            bad_price = np.bincount(
                codes, weights=(~price_valid).astype(float), minlength=rows.stop - rows.start
            )
            eligible[rows] &= bad_price == 0
            aggregate_price_valid[rows] = bad_price == 0
            mass = np.bincount(codes, weights=revenue, minlength=rows.stop - rows.start)
            mass = np.where(eligible[rows], mass, 0)
            if not np.isfinite(mass).all() or not np.isfinite(mass.sum()):
                raise ValueError("nonfinite expert weights")
            if mass.sum() > 0:
                weights[rows] = mass / mass.sum()
        return dict(
            target=target,
            eligible=eligible,
            scale=scale,
            weight=weights,
            history_complete=history_complete,
            target_complete=target_complete,
            price_valid=aggregate_price_valid,
        )

    def sample_experts(
        self, supervision: dict, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Uniform within-group Horvitz–Thompson sample; test_sample_experts enumerates its mean."""
        nodes, coefficients = [], []
        n_groups = len(self.policy.expert_levels)
        for group in self.policy.expert_levels:
            block = self.slices[group]
            candidates = np.arange(block.start, block.stop)[supervision["eligible"][block]]
            if not len(candidates) or supervision["weight"][block].sum() == 0:
                continue
            k = min(self.policy.expert_batch_per_group, len(candidates))
            sample = rng.choice(candidates, k, replace=False)
            nodes.extend(sample.tolist())
            coefficients.extend(
                (len(candidates) / k / n_groups * supervision["weight"][sample]).tolist()
            )
        return np.array(nodes, np.int64), np.array(coefficients, np.float64)
