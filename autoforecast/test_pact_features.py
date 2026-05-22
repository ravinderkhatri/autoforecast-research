"""Structural tests for PACT tokenization and label/input separation."""

from __future__ import annotations

import datetime as dt

import numpy as np

from autoforecast.pact_features import (
    EXACT,
    LOWER_BOUND,
    UNOBSERVED,
    PACTConfig,
    build_future_segments,
    build_history_tokens,
    calendar_segments,
    route_sales_states,
)


def test_pact_config_contract() -> None:
    config = PACTConfig(context_levels=(("store_id",),), n_layers=2)
    assert config.context_levels == (("store_id",),)
    assert config.history_policy().recent == 56
    for change in (
        {"position_mode": "relative_guess"},
        {"head_law": "shifted_hurdle_negbin"},
        {"context_levels": (("store_id",),), "n_layers": 1},
        {"context_levels": (("store_id",),) * 4},
    ):
        try:
            PACTConfig(**change)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid PACT recipe was accepted: {change}")


def test_pact_calendar_segments_all_cutoff_weekdays() -> None:
    for weekday in range(7):
        origin = dt.date(2026, 9, 20) + dt.timedelta(days=weekday)
        segments = calendar_segments(origin, 28)
        covered = [day for segment in segments for day in segment.day_indices]
        assert covered == list(range(28))
        assert all(len(set(segment.slots) - {-1}) == segment.active_days for segment in segments)
        future_starts_sunday = (origin + dt.timedelta(days=1)).weekday() == 6
        assert len(segments) == (4 if future_starts_sunday else 5)


def _history(length: int, *, rows: int = 2) -> dict[str, np.ndarray]:
    values = np.arange(rows * length, dtype=np.float64).reshape(rows, length) % 17
    observed = np.ones_like(values, bool)
    states = np.full_like(values, EXACT, dtype=np.int8)
    return build_history_tokens(
        values,
        observed,
        np.arange(length),
        origin_stop=length,
        first_ordinal=dt.date(2020, 1, 1).toordinal(),
        step_days=1,
        config=PACTConfig(),
        statistic_windows=(7, 28, 56),
        physical_state=states,
    )


def test_pact_history_exact_once_and_bounded_chronology() -> None:
    ordinary = _history(1911)
    assert ordinary["history_features"].shape[1] == 156
    assert np.all(np.diff(ordinary["history_positions"]) > 0)
    assert ordinary["history_positions"][-1] == 0
    assert ordinary["history_widths"].sum() == 1911

    overflow = _history(4000)
    assert overflow["history_features"].shape[1] == 164
    assert overflow["history_widths"].sum() == 4000
    assert overflow["history_widths"][0] > 28


def test_pact_unavailable_history_still_has_calendar_tokens() -> None:
    length = 80
    values = np.zeros((1, length), np.float64)
    observed = np.zeros((1, length), bool)
    result = build_history_tokens(
        values,
        observed,
        np.arange(length),
        origin_stop=length,
        first_ordinal=dt.date(2025, 1, 1).toordinal(),
        step_days=1,
        config=PACTConfig(),
        statistic_windows=(7,),
        physical_state=np.full((1, length), UNOBSERVED, np.int8),
    )
    assert result["history_exists"].all()
    assert np.isfinite(result["history_features"]).all()


def test_pact_future_label_invariance_and_sales_routing() -> None:
    origin = dt.date(2026, 9, 23)
    known = np.arange(2 * 28 * 3, dtype=np.float32).reshape(2, 28, 3) / 100
    target = np.arange(56, dtype=np.float32).reshape(2, 28) % 5
    states = np.full((2, 28), EXACT, np.int8)
    states[0, 2] = LOWER_BOUND
    first = build_future_segments(
        known,
        origin=origin,
        target=target,
        physical_state=states,
    )
    changed_target = target.copy()
    changed_target[:, :] += 100
    changed_states = states.copy()
    changed_states[1, 8] = UNOBSERVED
    second = build_future_segments(
        known,
        origin=origin,
        target=changed_target,
        physical_state=changed_states,
    )
    for name in (
        "future_features",
        "future_exists",
        "future_active",
        "future_day_index",
        "future_positions",
        "future_types",
    ):
        assert np.array_equal(first[name], second[name])
    assert first["segment_eligible"][0].all()  # recorded lower-bound sales route to exact sales
    assert not second["segment_eligible"][1].all()
    assert not np.array_equal(first["target_counts"], second["target_counts"])
    assert route_sales_states(np.array([LOWER_BOUND, UNOBSERVED]), "sales_given_availability").tolist() == [EXACT, UNOBSERVED]


def test_pact_campaign_toggle_changes_only_known_future_inputs() -> None:
    origin = dt.date(2026, 9, 23)
    off = np.zeros((1, 28, 2), np.float32)
    on = off.copy()
    on[:, 5:12, 0] = 1.0
    off_tokens = build_future_segments(off, origin=origin)
    on_tokens = build_future_segments(on, origin=origin)
    assert not np.array_equal(off_tokens["future_features"], on_tokens["future_features"])
    for name in ("future_active", "future_day_index", "future_positions", "future_types"):
        assert np.array_equal(off_tokens[name], on_tokens[name])
