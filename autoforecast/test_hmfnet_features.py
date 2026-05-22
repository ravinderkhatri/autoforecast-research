"""Structural counterexamples only; real-data smoke results are reported separately."""

import dataclasses
import datetime as dt
import itertools
import weakref

import numpy as np
import pytest

from autoforecast import hmfnet_features
from autoforecast.hmfnet_features import (
    AggregateHistory,
    HMFNetConfig,
    history_buckets,
    summarize_history,
)
from autoforecast.rolling_data import PanelArrays


def test_hmfnet_config():
    c = HMFNetConfig(context_levels=[["region", "kind"]], phase_cycles=[[7.0, 1]])
    assert c.context_levels == (("kind", "region"),)
    assert c.phase_cycles == ((7.0, 1),)
    for change in (
        dict(recent=True),
        dict(old_slots=1),
        dict(span=0),
        dict(feedback=1),
        dict(local_blocks=0),
        dict(anchor_floor=float("nan")),
        dict(expert_seed=-1),
        dict(expert_loss_weight=-1),
        dict(context_levels=[["a", "a"]]),
        dict(context_levels=[[" "]]),
        dict(context_levels="a"),
        dict(expert_levels=[["a", "b"], ["b", "a"]]),
        dict(phase_cycles=[[0.0, 1]]),
        dict(phase_cycles=[7]),
        dict(phase_cycles=None),
        dict(phase_cycles=[[7.0, 1], [7.0, 2]]),
        dict(aggregate_covariates="sum"),
        dict(feature_policy="unversioned"),
    ):
        with pytest.raises(ValueError):
            HMFNetConfig(**change)


@pytest.mark.parametrize(
    "length,expected", [(0, 0), (1, 1), (56, 56), (57, 57), (364, 100), (1913, 156), (10000, 164)]
)
def test_history_buckets(length, expected):
    buckets = history_buckets(length, HMFNetConfig())
    assert len(buckets) == expected
    assert list(itertools.chain.from_iterable(range(a, b) for a, b, _, _ in buckets)) == list(
        range(length)
    )
    if length == 1913:
        assert not buckets[-1][-1]
    if length == 10000:
        assert not buckets[-1][-1] and buckets[-1][1] == length
    assert history_buckets(length, HMFNetConfig(span=7)) == history_buckets(min(length, 7), HMFNetConfig())
    with pytest.raises(ValueError):
        history_buckets(-1, HMFNetConfig())


def summarize(values, observed=None, **kw):
    """Convenience inputs for explicitly synthetic structural checks."""
    values = np.asarray(values, np.float64)
    observed = np.ones(values.shape, bool) if observed is None else np.asarray(observed, bool)
    n = values.shape[1]
    defaults = dict(
        origin_stop=n, first_ordinal=0, step_days=1, policy=HMFNetConfig(), statistic_windows=(1, 2, 3)
    )
    defaults.update(kw)
    positions = defaults.pop("positions", np.arange(n))
    return summarize_history(values, observed, positions, **defaults)


def test_summarize_history():
    result = summarize([[0, 2, 4]])
    assert result["memory_values"].shape == (1, 3, 11)
    np.testing.assert_allclose(result["memory_values"][0, :, 0], [2, 1, 0])
    stats = result["statistics"][0]
    assert len(stats) == 14
    np.testing.assert_allclose(
        stats[:4], [np.log(3), np.log1p(np.std([0, 2, 4])), 2 / 3, np.log(4)]
    )
    assert result["memory_mask"].all()
    missing = summarize([[99, 2, 4]], [[False, True, True]])
    changed = summarize([[-1e100, 2, 4]], [[False, True, True]])
    for name in missing:
        np.testing.assert_array_equal(missing[name], changed[name])
    padded = summarize(
        [[1e20, 1e20, 0, 2, 4]],
        [[False, False, True, True, True]],
        positions=np.array([-2, -1, 0, 1, 2]),
        origin_stop=3,
        position_available=np.array([False, False, True, True, True]),
    )
    for name in result:
        np.testing.assert_array_equal(result[name], padded[name])
    empty = summarize(np.zeros((2, 0)))
    assert empty["memory_values"].shape == (2, 1, 11)
    assert not empty["memory_mask"].any() and not empty["statistics"].any()
    all_zero = summarize([[0, 0]])
    assert all_zero["memory_mask"].all() and all_zero["statistics"][0, 8] == 1
    for values in ([[-1]], [[np.inf]], [[0.5]], [[2**54]]):
        with pytest.raises(ValueError, match="counts"):
            summarize(values)
    with pytest.raises(ValueError, match="historical"):
        summarize([[1, 2]], origin_stop=1)
    with pytest.raises(ValueError, match="shapes"):
        summarize_history(
            np.ones((1, 2)),
            np.ones((1, 1), bool),
            np.arange(2),
            origin_stop=2,
            first_ordinal=0,
            step_days=1,
            policy=HMFNetConfig(),
            statistic_windows=(1,),
        )


def test_summarize_history_covariates():
    policy = HMFNetConfig(recent=1, middle_end=4, middle_width=3, phase_cycles=((4.0, 1),))
    result = summarize(
        [[0, 0, 0, 0]],
        [[False] * 4],
        policy=policy,
        numeric=(np.array([[1.0, 3.0, np.nan, 5.0]]),),
        categorical=(np.array([[0, 1, -1, 2]]),),
    )
    assert result["memory_mask"].all()  # Covariates alone make a usable token.
    assert not result["statistics"][0, 8]
    assert result["memory_categorical"].tolist() == [[[2], [1]]]
    assert result["memory_categorical_mask"].all()
    np.testing.assert_allclose(result["memory_values"][0, 1, -3:], [2 / 3, 1, 1])
    gapped = summarize(
        [[0, 0, 0, 0]], [[False] * 4], policy=policy, categorical=(np.array([[0, -1, 1, 2]]),)
    )
    np.testing.assert_allclose(gapped["memory_values"][0, 1, -2:], [0, 0])
    phase = summarize([[1, 2, 3, 4]], policy=policy)
    # Older bucket uses positions 0,1,2 with a=2.5: sine-weighted mean is 2 / 3 / 2.5.
    np.testing.assert_allclose(
        phase["memory_values"][0, 1, 11:15], [2 / 7.5, -2 / 7.5, 1 / 3, 0], atol=1e-7
    )
    with pytest.raises(ValueError, match="categorical"):
        summarize([[0]], categorical=(np.array([[0.5]]),))


def panel_arrays(masked=False):
    """Hand-computed aggregate losses, not forecast-performance data."""
    y = np.array(
        [
            [1, 1, 1, 2, 2, 3, 3],
            [0, 1, 0, 1, 0, 1, 2],
            [2, 2, 2, 2, 2, 3, 3],
            [1, 2, 3, 1, 2, 1, 2],
        ],
        np.float32,
    )
    metadata = dict(group=("a", "a", "b", "b"), kind=("x", "y", "x", "y"))
    return PanelArrays(
        tuple("abcd"),
        dt.date(2020, 1, 1),
        dt.date(2020, 1, 7),
        2,
        y,
        np.ones(y.shape, bool),
        np.ones(y.shape),
        metadata,
        observation_contract="masked_observed_counts_v1"
        if masked
        else "complete_recorded_sales_prelaunch_zeros_v1",
    )


def test_aggregate_history():
    data = panel_arrays()
    policy = HMFNetConfig(context_levels=(("group",),), expert_levels=(("group",), ("kind",)))
    aggregates = AggregateHistory(data, policy)
    assert len(aggregates.target) == 4
    np.testing.assert_array_equal(
        aggregates.target[:2], [data.target[:2].sum(0), data.target[2:].sum(0)]
    )
    assert aggregates.experts.tolist() == [[0, 2], [0, 3], [1, 2], [1, 3]]
    assert aggregates.node_keys == [
        dict(node_id=0, columns=["group"], values=["a"]),
        dict(node_id=1, columns=["group"], values=["b"]),
        dict(node_id=2, columns=["kind"], values=["x"]),
        dict(node_id=3, columns=["kind"], values=["y"]),
    ]
    assert np.array_equal(aggregates.context, aggregates.memberships(policy.context_levels))
    np.testing.assert_array_equal(aggregates.anchors(np.ones((4, 2))), np.full((4, 2), 2))
    with pytest.raises(ValueError, match="anchors"):
        aggregates.anchors(-np.ones((4, 2)))
    with pytest.raises(ValueError, match="absent"):
        AggregateHistory(data, HMFNetConfig(expert_levels=(("missing",),)))
    with pytest.raises(ValueError, match="budget"):
        AggregateHistory(data, dataclasses.replace(policy, max_aggregate_bytes=1))
    incomplete = panel_arrays(masked=True)
    incomplete.observed[0, 1] = False
    aggregate = AggregateHistory(incomplete, policy)
    assert not aggregate.observed[0, 1] and aggregate.target[0, 1] == 0
    assert AggregateHistory(data, HMFNetConfig()).target.shape == (0, 7)
    huge = dataclasses.replace(data, target=np.full(data.target.shape, 2**53, np.float64))
    with pytest.raises(ValueError, match="exact-integer"):
        AggregateHistory(huge, policy)


def test_aggregate_history_releases_integer_scratch_before_mask_counts(monkeypatch):
    """Masked aggregation keeps only one integer scratch matrix live at a time."""
    data = panel_arrays(masked=True)
    policy = HMFNetConfig(context_levels=(("group",),))
    original_zeros = np.zeros
    live_bytes = 0
    peak_bytes = 0

    def release(size):
        nonlocal live_bytes
        live_bytes -= size

    def tracked_zeros(shape, dtype=float, *args, **kwargs):
        nonlocal live_bytes, peak_bytes
        result = original_zeros(shape, dtype, *args, **kwargs)
        if result.shape == (2, 7) and result.dtype == np.int64:
            live_bytes += result.nbytes
            peak_bytes = max(peak_bytes, live_bytes)
            weakref.finalize(result, release, result.nbytes)
        return result

    monkeypatch.setattr(hmfnet_features.np, "zeros", tracked_zeros)
    AggregateHistory(data, policy)
    assert peak_bytes == 2 * 7 * np.dtype(np.int64).itemsize


def test_aggregate_history_budget_covers_indices_and_memberships() -> None:
    """The allocation cap covers all retained NumPy maps, not only target matrices."""
    data = panel_arrays(masked=True)
    policy = HMFNetConfig(
        context_levels=(("group",),),
        expert_levels=(("group",),),
        max_aggregate_bytes=250,
    )
    with pytest.raises(ValueError, match="budget"):
        AggregateHistory(data, policy)


def test_aggregate_history_budget_preflights_indices_and_chunk_temporaries(
    monkeypatch,
) -> None:
    """The cap fires before index allocation and accounts for masked chunk conversion."""
    data = panel_arrays(masked=True)
    policy = HMFNetConfig(
        context_levels=(("group",),),
        expert_levels=(("kind",),),
        max_aggregate_bytes=100,
    )
    original_array = np.array
    allocated_codes = False

    def tracked_array(value, dtype=None, *args, **kwargs):
        nonlocal allocated_codes
        result = original_array(value, dtype, *args, **kwargs)
        if result.shape == (len(data.series_ids),) and result.dtype == np.int64:
            allocated_codes = True
        return result

    monkeypatch.setattr(hmfnet_features.np, "array", tracked_array)
    with pytest.raises(ValueError, match="budget"):
        AggregateHistory(data, policy)
    assert not allocated_codes

    # The retained four-node/seven-period arrays fit this cap, but admitting
    # aggregation must also reserve the bounded numerical chunk workspace.
    with pytest.raises(ValueError, match="budget"):
        AggregateHistory(
            data,
            dataclasses.replace(policy, max_aggregate_bytes=699),
        )


def test_aggregate_history_budget_covers_live_membership_allocations(monkeypatch):
    """Track a lower bound on concurrent arrays, including stack input temporaries."""
    count = 200
    data = dataclasses.replace(
        panel_arrays(masked=True),
        series_ids=tuple(str(index) for index in range(count)),
        cutoff=dt.date(2020, 1, 1),
        horizon=1,
        target=np.ones((count, 1), np.float32),
        observed=np.ones((count, 1), bool),
        price=np.ones((count, 1), np.float64),
        metadata={"group": ("one",) * count},
    )
    policy = HMFNetConfig(
        context_levels=(("group",),),
        expert_levels=(("group",),),
        max_aggregate_bytes=6000,
    )
    live = {}
    peak = 0
    original_stack = np.stack

    def tracked(factory):
        def allocate(*args, **kwargs):
            nonlocal peak
            result = factory(*args, **kwargs)
            identity = id(result)
            live[identity] = result.nbytes
            weakref.finalize(result, live.pop, identity)
            peak = max(peak, sum(live.values()))
            return result
        return allocate

    def stack(values, *args, **kwargs):
        nonlocal peak
        result = tracked(original_stack)(values, *args, **kwargs)
        transient = sum(value.nbytes for value in values if id(value) not in live)
        peak = max(peak, sum(live.values()) + transient)
        return result

    for name in ("array", "zeros", "empty"):
        monkeypatch.setattr(hmfnet_features.np, name, tracked(getattr(np, name)))
    monkeypatch.setattr(hmfnet_features.np, "stack", stack)
    result = AggregateHistory(data, policy)
    assert result.context.shape == result.experts.shape == (count, 1)
    assert peak <= policy.max_aggregate_bytes, (peak, policy.max_aggregate_bytes)


def test_aggregate_history_budget_includes_bounded_chunk_scratch():
    """The retained maps fit, but the numerical chunk workspace also needs admission."""
    policy = HMFNetConfig(
        context_levels=(("group",),),
        expert_levels=(("group",),),
        max_aggregate_bytes=400,
    )
    with pytest.raises(ValueError, match="budget"):
        AggregateHistory(panel_arrays(masked=True), policy)


def test_aggregate_history_releases_count_scratch_before_memberships(monkeypatch):
    """The last group's temporary count matrix must not overlap final map allocation."""
    original_zeros = np.zeros
    original_memberships = AggregateHistory.memberships
    scratch = []

    def zeros(*args, **kwargs):
        result = original_zeros(*args, **kwargs)
        if result.shape == (2, 7) and result.dtype == np.int64:
            scratch.append(weakref.ref(result))
        return result

    def memberships(self, levels):
        assert all(reference() is None for reference in scratch)
        return original_memberships(self, levels)

    monkeypatch.setattr(hmfnet_features.np, "zeros", zeros)
    monkeypatch.setattr(AggregateHistory, "memberships", memberships)
    AggregateHistory(panel_arrays(masked=True), HMFNetConfig(context_levels=(("group",),)))


def test_aggregate_history_releases_masked_chunk_before_count_allocation(monkeypatch):
    """Bounded chunk copies do not remain live during the next workspace stage."""
    data = panel_arrays(masked=True)
    original_where, original_zeros = np.where, np.zeros
    scratch = []

    def where(*args, **kwargs):
        result = original_where(*args, **kwargs)
        if result.shape == data.target.shape and result.dtype == data.target.dtype:
            scratch.append(weakref.ref(result))
        return result

    def zeros(*args, **kwargs):
        result = original_zeros(*args, **kwargs)
        if result.shape == (2, 7) and result.dtype == np.int64:
            assert all(reference() is None for reference in scratch)
        return result

    monkeypatch.setattr(hmfnet_features.np, "where", where)
    monkeypatch.setattr(hmfnet_features.np, "zeros", zeros)
    AggregateHistory(data, HMFNetConfig(context_levels=(("group",),)))


def test_expert_supervision():
    data = panel_arrays(masked=True)
    policy = HMFNetConfig(expert_levels=(("group",), ("kind",)))
    aggregates = AggregateHistory(data, policy)
    supervised = aggregates.supervision(5, "uniform", 2)
    assert supervised["eligible"].all()
    assert supervised["scale"].min() >= 1
    np.testing.assert_array_equal(supervised["weight"], [0.5, 0.5, 0.5, 0.5])
    singleton = AggregateHistory(
        dataclasses.replace(data, metadata={"id": tuple("abcd")}),
        HMFNetConfig(expert_levels=(("id",),)),
    )
    assert singleton.supervision(5, "uniform", 2)["scale"][0] == 1  # raw=.25
    data.observed[0, 5] = False
    changed = AggregateHistory(data, policy)
    np.testing.assert_array_equal(aggregates.target[:, :5], changed.target[:, :5])
    assert not changed.supervision(5, "uniform", 2)["eligible"][0]
    data.price[0, 3] = np.nan
    assert not changed.supervision(5, "revenue", 2)["eligible"][0]
    data.observed[0, 1] = False
    assert not AggregateHistory(data, policy).supervision(5, "uniform", 2)["eligible"][0]
    with pytest.raises(ValueError):
        aggregates.supervision(6, "uniform", 2)
    with pytest.raises(ValueError):
        aggregates.supervision(5, "revenue", 6)


def test_sample_experts():
    aggregates = AggregateHistory(
        panel_arrays(), HMFNetConfig(expert_levels=(("group",),), expert_batch_per_group=1)
    )
    supervision = aggregates.supervision(5, "uniform", 2)
    supervision["weight"][:] = [0.25, 0.75]
    expected = 0.25 * 2 + 0.75 * 10
    losses = np.array([2, 10])
    outcomes = []
    for chosen in (0, 1):

        class EnumeratedRng:
            def choice(self, candidates, k, replace, chosen=chosen):
                assert k == 1 and replace is False
                return np.array([candidates[chosen]])

        indices, coefficients = aggregates.sample_experts(supervision, EnumeratedRng())
        outcomes.append(float((coefficients * losses[indices]).sum()))
    assert np.mean(outcomes) == expected
    one = aggregates.sample_experts(supervision, np.random.default_rng(43))
    two = aggregates.sample_experts(supervision, np.random.default_rng(43))
    for a, b in zip(one, two, strict=True):
        np.testing.assert_array_equal(a, b)
    supervision["eligible"][:] = False
    assert len(aggregates.sample_experts(supervision, np.random.default_rng(43))[0]) == 0


def test_hmfnet_numerical_guards():
    """Explicit overflow/domain faults; none of these arrays is forecasting evidence."""
    for value in (float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="numeric covariates"):
            summarize([[1, 1]], numeric=(np.array([[value, 1.0]]),))
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="nonfinite HMFNet summary"):
            summarize([[1, 1]], numeric=(np.array([[1e308, 1e308]]),))
    data = panel_arrays()
    policy = HMFNetConfig(expert_levels=(("group",),))
    aggregates = AggregateHistory(data, policy)
    for value in (2**54, -1, 0.5, float("nan")):
        target = np.full(data.target.shape, value, np.float64)
        with pytest.raises(ValueError, match="aggregate source"):
            AggregateHistory(dataclasses.replace(data, target=target), policy)
    target = np.zeros(data.target.shape, np.float64)
    target[0, 0], target[1, 0] = 2**53, 1
    with pytest.raises(ValueError, match="aggregate totals"):
        AggregateHistory(dataclasses.replace(data, target=target), policy)
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="anchor overflow"):
            aggregates.anchors(np.full((4, 2), 1e308))
        # Direct cache-corruption fault exercises the numerical boundary after assembly.
        aggregates.target[0, 1] = 1e308
        with pytest.raises(ValueError, match="expert scale"):
            aggregates.supervision(5, "uniform", 2)
        data.price[:] = 1e308
        aggregates = AggregateHistory(data, policy)
        with pytest.raises(ValueError, match="expert weights"):
            aggregates.supervision(5, "revenue", 2)


def test_hmfnet_oov_transitions():
    """OOV is one embedding ID, not one original category value for transition statistics."""
    policy = HMFNetConfig(recent=1, middle_end=3, middle_width=2)
    ids = np.array([[2, 2, 2]])
    raw = np.array([["unseen_a", "unseen_b", "unseen_b"]], object)
    result = summarize([[1, 1, 1]], policy=policy, categorical=(ids,), categorical_raw=(raw,))
    assert result["memory_values"][0, 1, -2] == 1
    assert result["memory_categorical"][0, 1, 0] == 2
    raw[0, 0] = "unseen_b"
    repeated = summarize([[1, 1, 1]], policy=policy, categorical=(ids,), categorical_raw=(raw,))
    assert repeated["memory_values"][0, 1, -2] == 0
    ids[0, 1] = -1
    gap = summarize([[1, 1, 1]], policy=policy, categorical=(ids,), categorical_raw=(raw,))
    assert gap["memory_values"][0, 1, -1] == 0
    with pytest.raises(ValueError, match="shapes"):
        summarize([[1, 1, 1]], categorical=(ids,), categorical_raw=())
