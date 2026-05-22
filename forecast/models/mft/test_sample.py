"""Tests for ancestral sampling, tied back to the likelihood it is supposed to draw from.

The property that matters most is not any single draw -- it is that the SAMPLER and the
LIKELIHOOD describe the same distribution. If they drift apart, every quantile taken from the
paths describes something the model was never fitted to, and nothing downstream can tell.

Run: python3 forecast/models/mft/test_sample.py
"""

from __future__ import annotations

import math
import pathlib
import sys
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent.parent))

import torch
from forecast.models.mft.heads import hurdle_mean, nb_log_pmf
from forecast.models.mft.sample import (
    MAX_SUPPORT,
    SampleError,
    _mu_over_sum,
    _support_sizes_negbin,
    lead_time_totals,
    sample_paths,
    sample_paths_negbin,
    support_size,
    support_sizes,
)
from spec import likelihood as ref


def t(x) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float64)


def gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_the_sampled_mean_matches_the_analytic_mean() -> None:
    """The tie between the sampler and the likelihood. E[Y] = p*mu/(1-P_NB(0)), and if the
    sampler disagreed every quantile drawn from these paths would describe another distribution."""
    for p, mu, k in [(0.35, 3.0, 2.5), (0.8, 1.2, 4.0), (0.1, 8.0, 1.5)]:
        draws = sample_paths(t([p]), t([mu]), t([k]), 200_000, generator=gen(1))
        empirical = float(draws.double().mean())
        analytic = float(hurdle_mean(t([p]), t([mu]), t([k]))[0])
        assert math.isclose(empirical, analytic, rel_tol=0.03), (p, mu, k, empirical, analytic)


def test_the_sampled_zero_rate_matches_one_minus_p() -> None:
    draws = sample_paths(t([0.3]), t([4.0]), t([2.0]), 200_000, generator=gen(2))
    assert abs(float((draws == 0).double().mean()) - 0.7) < 0.005


def test_a_positive_draw_is_never_zero() -> None:
    """It inverts the ZERO-TRUNCATED CDF. Inverting the untruncated one can return 0 and would
    contradict the occurrence draw that just happened -- and on a near-zero mean, often."""
    draws = sample_paths(t([1.0 - 1e-12]), t([0.02]), t([3.0]), 50_000, generator=gen(3))
    assert int((draws == 0).sum()) == 0, "a hurdle that always fires produced a zero"


def test_the_sampled_distribution_matches_the_contract_pmf() -> None:
    """Not just the mean: the whole shape, against the pure-Python contract."""
    p, mu, k = 0.6, 2.0, 3.0
    draws = sample_paths(t([p]), t([mu]), t([k]), 400_000, generator=gen(4)).flatten()
    cell = ref.Hurdle(p=p, mu=mu, k=k)
    for y in range(0, 8):
        empirical = float((draws == y).double().mean())
        want = math.exp(ref.log_likelihood("exact", y, cell))
        assert abs(empirical - want) < 0.004, (y, empirical, want)


def test_the_support_is_derived_from_the_distribution_not_assumed() -> None:
    """A ceiling that suits an intermittent series truncates a fast mover, and the truncation
    shows up as a spike at the ceiling in every quantile drawn from the paths."""
    small = support_size(t([0.5]), t([3.0]))
    large = support_size(t([200.0]), t([2.0]))
    assert large > small * 5, f"support did not scale with the mean: {small} vs {large}"


def test_support_proves_the_requested_omitted_mass_near_poisson() -> None:
    """Regression for a support heuristic that normalised away a material near-Poisson tail."""
    tail_mass = 1e-12
    for k in (100.0, 1_000.0, 8_000.0):
        n = support_size(t([100.0]), t([k]), tail_mass=tail_mass)
        omitted = math.exp(ref.nb_log_sf(n + 1, 100.0, k))
        assert omitted <= tail_mass, (k, n, omitted)
        assert 100 < n <= MAX_SUPPORT, (k, n)


def test_near_poisson_sampling_preserves_the_analytic_mean() -> None:
    """A CDF may condition only on the proved <=1e-12 remainder, not a material missing tail."""
    p, mu = 0.65, 100.0
    for index, k in enumerate((100.0, 1_000.0, 8_000.0)):
        draws = sample_paths(t([p]), t([mu]), t([k]), 100_000, generator=gen(30 + index))
        empirical = float(draws.double().mean())
        analytic = float(hurdle_mean(t([p]), t([mu]), t([k]))[0])
        assert math.isclose(empirical, analytic, rel_tol=0.005), (k, empirical, analytic)


def test_a_distribution_needing_more_support_than_the_cap_is_refused() -> None:
    try:
        support_size(t([1e5]), t([1.0]))
    except SampleError as e:
        assert f"above MAX_SUPPORT={MAX_SUPPORT}" in str(e)
        return
    raise AssertionError("capped the support instead of refusing, creating a ceiling spike")


def test_finite_extreme_parameters_raise_the_sampling_contract_not_raw_overflow() -> None:
    """Both `mu*(k-1)/k` and `mu/(mu+k)` overflowed despite finite operands."""
    assert float(_mu_over_sum(t([1e308]), t([1e308]))[0]) == 0.5
    try:
        support_size(t([1e308]), t([1e308]))
    except SampleError as error:
        assert f"above MAX_SUPPORT={MAX_SUPPORT}" in str(error)
        return
    raise AssertionError("an unrepresentable support escaped as a raw numeric overflow")


def test_invalid_sampling_domains_are_refused_explicitly() -> None:
    calls = [
        lambda: support_size(t([float("inf")]), t([1.0])),
        lambda: support_size(t([1.0]), t([0.0])),
        lambda: support_size(t([1.0]), t([1.0]), tail_mass=1.0),
        lambda: sample_paths(t([float("nan")]), t([1.0]), t([1.0]), 2),
        lambda: sample_paths(t([0.5]), t([1.0]), t([1.0]), 1.5),
    ]
    for call in calls:
        try:
            call()
        except SampleError:
            continue
        raise AssertionError("an invalid sampling parameter was accepted")


def test_every_support_api_rejects_an_invalid_tail_mass() -> None:
    """The scalar, per-cell hurdle, and per-cell plain helpers share one proof contract."""

    class HostileFloat:
        def __float__(self):
            raise RuntimeError("untrusted coercion ran")

    class HostileInt(int):
        def __float__(self):
            raise RuntimeError("integer subclass coercion ran")

    calls = (
        lambda value: support_size(t([1.0]), t([1.0]), tail_mass=value),
        lambda value: support_sizes(t([1.0]), t([1.0]), tail_mass=value),
        lambda value: _support_sizes_negbin(t([1.0]), t([1.0]), tail_mass=value),
    )
    for value in (
        -1.0,
        0.0,
        1.0,
        True,
        float("inf"),
        float("nan"),
        10**400,
        "1e-12",
        HostileFloat(),
        HostileInt(1),
    ):
        for call in calls:
            try:
                call(value)
            except SampleError as error:
                assert "tail_mass must be finite and in (0,1)" in str(error)
                continue
            raise AssertionError(f"a support API accepted invalid tail_mass={value!r}")


def test_exact_zero_hurdle_skips_unreachable_size_support_but_not_domain_validation() -> None:
    """At p=0 the ZTNB component is unreachable, but its tensors must remain well-formed."""
    mu, k = t([5_000.0, 2.0, 100_000.0]), t([1.5, 2.0, 1.0])
    try:
        support_sizes(mu, k)
    except SampleError:
        pass
    else:
        raise AssertionError("the fixture no longer exceeds the ZTNB support cap")

    paths = sample_paths(t([0.0, 1.0, 0.0]), mu, k, 4, generator=gen(12))
    assert paths.shape == (4, 3)
    assert torch.equal(paths[:, [0, 2]], torch.zeros_like(paths[:, [0, 2]]))
    assert bool((paths[:, 1] >= 1).all()), "the reachable ZTNB cell stopped at the hurdle"

    invalid = ((t([float("nan")]), t([1.0])), (t([1.0]), t([0.0])))
    for bad_mu, bad_k in invalid:
        try:
            sample_paths(t([0.0]), bad_mu, bad_k, 1, generator=gen(12))
        except SampleError:
            continue
        raise AssertionError("p=0 bypassed validation of invalid mu or k")


def test_a_reachable_support_refusal_names_the_original_cell_after_zero_hurdles() -> None:
    """Support is certified on the compressed p>0 subset, but diagnostics belong to the caller.

    Two unreachable cells precede the offender here. Reporting ``flat 0`` would identify the
    subset coordinate and send the caller to the wrong series; the original tensor coordinate is
    ``flat 2``.
    """
    p, mu, k = t([0.0, 0.0, 1.0]), t([1.0, 2.0, 5_000.0]), t([1.5, 1.5, 1.5])
    try:
        sample_paths(p, mu, k, 1, generator=gen(0))
    except SampleError as error:
        message = str(error)
        assert "flat 2 (mu=5000, k=1.5)" in message, message
        assert "flat 0 (mu=5000" not in message, message
        return
    raise AssertionError("a reachable cell needing more than MAX_SUPPORT was sampled")


def test_no_draw_lands_on_the_support_ceiling() -> None:
    """The proved support leaves only negligible mass to condition away, so no ceiling spike."""
    p, mu, k = 0.9, 30.0, 1.2
    n = support_size(t([mu]), t([k]))
    draws = sample_paths(t([p]), t([mu]), t([k]), 200_000, generator=gen(5))
    at_ceiling = int((draws == n).sum())
    assert at_ceiling <= 2, f"{at_ceiling} draws landed exactly on the ceiling {n}"


def test_draws_are_reproducible_from_a_seed() -> None:
    a = sample_paths(t([0.4, 0.6]), t([2.0, 3.0]), t([3.0, 2.0]), 50, generator=gen(9))
    b = sample_paths(t([0.4, 0.6]), t([2.0, 3.0]), t([3.0, 2.0]), 50, generator=gen(9))
    assert torch.equal(a, b)
    assert not torch.equal(
        a, sample_paths(t([0.4, 0.6]), t([2.0, 3.0]), t([3.0, 2.0]), 50, generator=gen(10))
    )


def test_the_cdf_is_not_repeated_per_draw() -> None:
    """The defect that killed a real run. Expanding the support table to
    [n_draws * n_cells, support] before searchsorted is 4.3 BILLION floats at 1,000 paths over
    M5's 85,372 cells -- the process was killed with no traceback. searchsorted broadcasts over
    leading dimensions, so the table is built once and the draws go on the last axis.

    Checked by memory rather than by inspection: 1,000 paths over 20,000 cells has to work."""
    cells = 20_000
    p = torch.full((cells,), 0.3, dtype=torch.float64)
    out = sample_paths(p, torch.full_like(p, 2.0), torch.full_like(p, 3.0), 1000, generator=gen(21))
    assert out.shape == (1000, cells)
    assert out.dtype in (torch.int64, torch.int32)


def test_the_shape_is_draws_by_cells() -> None:
    p = t([[0.3, 0.4, 0.5], [0.6, 0.7, 0.8]])
    out = sample_paths(p, torch.full_like(p, 2.0), torch.full_like(p, 3.0), 7, generator=gen())
    assert out.shape == (7, 2, 3)


def test_mismatched_parameter_shapes_are_refused() -> None:
    try:
        sample_paths(t([0.3, 0.4]), t([2.0]), t([3.0]), 5, generator=gen())
    except SampleError as e:
        assert "same cells" in str(e)
        return
    raise AssertionError("broadcast one cell's size onto another cell")


def test_lead_time_totals_sum_the_cover_period_within_each_path() -> None:
    """Why paths exist: marginal quantiles do not determine the aggregate quantile. On this
    positively dependent fixture their sum is far above it; other dependence can reverse that."""
    paths = torch.ones(4, 3, 28, dtype=torch.int64)
    got = lead_time_totals(paths, lead_time=5, review_period=2)
    assert got.shape == (4, 3)
    assert int(got[0, 0]) == 7


def test_a_cover_period_past_the_horizon_is_refused() -> None:
    """Extrapolating would invent dependence past the last day the factor was drawn for."""
    try:
        lead_time_totals(torch.ones(2, 1, 14, dtype=torch.int64), lead_time=10, review_period=10)
    except SampleError as e:
        assert "exceeds the 14-day horizon" in str(e)
        return
    raise AssertionError("summed past the horizon that was drawn")


def test_lead_time_totals_refuses_a_rank_one_tensor_without_a_path_axis() -> None:
    """`[horizon]` has no path axis, so its result cannot preserve one total per sampled path."""
    try:
        lead_time_totals(torch.ones(7, dtype=torch.int64), lead_time=2, review_period=1)
    except SampleError as error:
        assert "path and final horizon dimension" in str(error)
        return
    raise AssertionError("treated one horizon vector as though it contained sampled paths")


def test_lead_time_totals_requires_exact_nonnegative_integer_controls() -> None:
    class HostileInt(int):
        def __add__(self, other):
            raise RuntimeError("hostile addition ran")

    paths = torch.ones(2, 1, 3, dtype=torch.int64)
    for lead_time, review_period in (
        (True, 0),
        (1.0, 0),
        (HostileInt(1), 0),
        (-1, 2),
        (1, -1),
    ):
        try:
            lead_time_totals(paths, lead_time, review_period)
        except SampleError as error:
            assert "built-in non-negative integers" in str(error)
            continue
        raise AssertionError(f"accepted lead_time={lead_time!r}, review_period={review_period!r}")


def test_lead_time_totals_requires_finite_nonnegative_integer_valued_paths() -> None:
    invalid = (
        ([1, 2], "torch tensor"),
        (torch.tensor(1), "horizon dimension"),
        (torch.tensor([[[True, False]]]), "integer-valued"),
        (torch.tensor([[[1.0 + 0.0j]]]), "integer-valued"),
        (torch.tensor([[[-1.0]]]), "non-negative"),
        (torch.tensor([[[0.5]]]), "integer-valued"),
        (torch.tensor([[[float("nan")]]]), "finite"),
        (torch.tensor([[[float("inf")]]]), "finite"),
    )
    for paths, fragment in invalid:
        try:
            lead_time_totals(paths, 1, 0)
        except SampleError as error:
            assert fragment in str(error), (paths, error)
            continue
        raise AssertionError(f"accepted invalid paths {paths!r}")

    integer_valued_float = torch.tensor([[[1.0, 2.0]]])
    assert torch.equal(lead_time_totals(integer_valued_float, 1, 1), torch.tensor([[3.0]]))


def test_lead_time_totals_refuses_numeric_overflow() -> None:
    for paths in (
        torch.tensor([[[1e308, 1e308]]], dtype=torch.float64),
        torch.tensor([[[torch.iinfo(torch.int64).max, torch.iinfo(torch.int64).max]]]),
        torch.tensor([[[torch.iinfo(torch.int64).max] * 3]]),
    ):
        try:
            lead_time_totals(paths, 2, 0)
        except SampleError as error:
            assert "overflowed" in str(error)
            continue
        raise AssertionError("lead-time totals overflowed silently")


def test_the_quantile_of_the_sum_is_below_the_sum_of_the_quantiles() -> None:
    """This fixture's inequality makes marginal quantiles useless for its ordering decision.
    It is not universal; the missing joint dependence is the general defect."""
    p, mu, k, h = 0.5, 3.0, 2.0, 14
    cells = torch.full((h,), p, dtype=torch.float64)
    paths = sample_paths(
        cells, torch.full_like(cells, mu), torch.full_like(cells, k), 40_000, generator=gen(11)
    )
    per_day_q95 = torch.quantile(paths.double(), 0.95, dim=0).sum()
    total_q95 = torch.quantile(lead_time_totals(paths, h, 0).double(), 0.95)
    assert float(total_q95) < float(per_day_q95), (float(total_q95), float(per_day_q95))


def test_plain_nb_paths_include_zero_and_match_the_mean() -> None:
    """The plain sampler's zero is an ordinary outcome of the untruncated CDF -- no occurrence
    stage exists to produce it -- and the empirical mean must sit on mu, because E[Y] = mu with
    nothing truncated."""
    mu = torch.full((40, 7), 1.5, dtype=torch.float64)
    k = torch.full((40, 7), 2.0, dtype=torch.float64)
    paths = sample_paths_negbin(mu, k, 400, generator=torch.Generator().manual_seed(0))
    assert paths.shape == (400, 40, 7)
    assert bool((paths == paths.round()).all()) and bool((paths >= 0).all())
    zero_share = float((paths == 0).double().mean())
    expected_zero = math.exp(
        float(
            __import__(
                "forecast.models.mft.heads", fromlist=["nb_log_pmf"]
            ).nb_log_pmf(
                torch.tensor(0.0, dtype=torch.float64),
                torch.tensor(1.5, dtype=torch.float64),
                torch.tensor(2.0, dtype=torch.float64),
            )
        )
    )
    assert (
        abs(zero_share - expected_zero) < 0.02
    ), f"zero share {zero_share:.3f} vs P(0) {expected_zero:.3f}"
    assert abs(float(paths.double().mean()) - 1.5) < 0.05, "the sampled mean drifted from mu"
    again = sample_paths_negbin(mu, k, 400, generator=torch.Generator().manual_seed(0))
    assert torch.equal(paths, again), "same seed, same paths"


def test_plain_nb_support_proves_absolute_not_zero_truncated_tail_mass() -> None:
    """A mostly-zero plain NB is valid even when its conditional positive tail exceeds the cap.

    This exact fixture was refused by reusing the ZTNB proof: at MAX_SUPPORT its absolute tail is
    safely below 1e-12, while conditioning on the tiny positive mass inflates it above 1e-12.
    """
    mu_value, k_value = 0.18323795010446822, 0.0009118819655545162
    mu, k = t([mu_value]), t([k_value])
    at_cap = math.exp(ref.plain_nb_log_sf(MAX_SUPPORT + 1, mu_value, k_value))
    conditional_at_cap = math.exp(ref.nb_log_sf(MAX_SUPPORT + 1, mu_value, k_value))
    assert at_cap < 1e-12 < conditional_at_cap, (at_cap, conditional_at_cap)

    support = int(_support_sizes_negbin(mu, k)[0])
    omitted = math.exp(ref.plain_nb_log_sf(support + 1, mu_value, k_value))
    assert support == 3_577
    assert omitted <= 1e-12, (support, omitted)
    paths = sample_paths_negbin(mu, k, 3, generator=gen(0))
    assert paths.shape == (3, 1)
    assert bool((paths >= 0).all())


def test_plain_nb_roundoff_above_one_does_not_fake_a_dropped_tail() -> None:
    """The upper CDF guard is arithmetic-only; the certified lower tail is the contract."""
    mu_value, k_value = 9.236708641052246, 4667.02001953125
    mu, k = t([mu_value]), t([k_value])
    support = int(_support_sizes_negbin(mu, k)[0])
    omitted = math.exp(ref.plain_nb_log_sf(support + 1, mu_value, k_value))
    assert support == 38
    assert omitted <= 1e-12, (support, omitted)

    counts = torch.arange(support + 1, dtype=torch.float64)
    cdf_last = float(torch.exp(nb_log_pmf(counts, mu, k)).sum())
    assert cdf_last > 1.0 + 64.0 * torch.finfo(torch.float64).eps, cdf_last

    paths = sample_paths_negbin(mu, k, 3, generator=gen(0))
    assert paths.shape == (3, 1)
    assert bool((paths >= 0).all())


def test_plain_nb_uses_per_cell_power_tiers_without_batch_max_amplification() -> None:
    """A high-volume plain cell may enlarge its own table, never every neighbouring table."""
    import forecast.models.mft.sample as sampler

    mu, k = t([2.0, 100.0]), t([1.5, 1.5])
    supports = _support_sizes_negbin(mu, k)
    assert supports.tolist() == [52, 1978]

    table_shapes: list[tuple[int, ...]] = []
    original = sampler.nb_log_pmf

    def record_shape(
        counts: torch.Tensor, table_mu: torch.Tensor, table_k: torch.Tensor
    ) -> torch.Tensor:
        if counts.ndim == 2:
            table_shapes.append(tuple(counts.shape))
        return original(counts, table_mu, table_k)

    with patch.object(sampler, "nb_log_pmf", side_effect=record_shape):
        paths = sample_paths_negbin(mu, k, 2, generator=gen(0))

    assert paths.shape == (2, 2)
    assert table_shapes == [(1, 65), (1, 2049)], table_shapes
    assert (2, 2049) not in table_shapes, "the loud cell's tier was expanded across the batch"


def test_plain_nb_supports_match_the_canonical_tail_on_a_representative_grid() -> None:
    """Check the proof independently across broad means and nearly the full k clamp."""
    mus = torch.logspace(-2, 2.3, 12, dtype=torch.float64)
    ks = torch.logspace(-2, 3.9, 14, dtype=torch.float64)
    checked = 0
    for mu in mus:
        for k in ks:
            try:
                support = int(_support_sizes_negbin(mu.reshape(1), k.reshape(1))[0])
            except SampleError:
                continue
            omitted = math.exp(ref.plain_nb_log_sf(support + 1, float(mu), float(k)))
            assert omitted <= 1e-12, (float(mu), float(k), support, omitted)
            checked += 1
    assert checked > 100, "too few in-cap cells were exercised by the independent proof sweep"


def test_one_high_volume_cell_does_not_resize_every_other_cell_s_table() -> None:
    """`support_size` proves ONE bound for the whole batch, so the largest mu in it sized the CDF
    table that every cell paid for. Measured before the fix: a single cell moving mu=2 -> mu=100
    took the shared support from 54 to 1,978 and the coexisting float64 PMF+CDF tables from
    0.05 GB to 1.77 GB -- and at
    mu=300 it refused the entire batch on behalf of one cell.

    Per-cell supports make a high-volume cell expensive for itself alone. The neighbours' bounds
    must not move at all when it appears."""
    quiet = t([0.4, 0.8, 1.5, 2.0])
    loud = t([0.4, 0.8, 1.5, 100.0])
    k = t([1.5, 1.5, 1.5, 1.5])
    before = support_sizes(quiet, k)
    after = support_sizes(loud, k)
    assert torch.equal(
        before[:3], after[:3]
    ), "a neighbouring cell's proved support moved because another cell got busier"
    assert int(after[3]) > 10 * int(before[3]), "the loud cell should need a much larger support"
    assert support_size(loud, k) == int(
        after.max()
    ), "the batch bound must still be the worst per-cell bound"


def test_a_cell_beyond_the_cap_is_named_and_its_neighbours_are_not_refused_anonymously() -> None:
    """The cap is a real refusal and stays one -- capping would put a spike at the ceiling in
    every quantile. What changed is attribution: the error must say HOW MANY cells and how large,
    so the caller can find them, instead of refusing 2,000 cells on behalf of one."""
    mu = t([0.5, 1.0, 2.0, 5_000.0])
    k = t([1.5, 1.5, 1.5, 1.5])
    try:
        support_sizes(mu, k)
    except SampleError as e:
        assert "1 cell(s)" in str(e), f"the refusal did not attribute the failure: {e}"
        assert "MAX_SUPPORT" in str(e)
        assert "flat 3" in str(e), f"the refusal did not identify the offending cell: {e}"
        assert "mu=5000" in str(e), f"the refusal rounded away the parameter value: {e}"
        return
    raise AssertionError("a cell needing more than MAX_SUPPORT was sampled instead of refused")


def test_plain_support_refusal_does_not_advance_the_callers_generator() -> None:
    """A deterministic pre-draw refusal must leave every later seeded sample untouched."""
    seed = 123
    generator = torch.Generator().manual_seed(seed)
    before = generator.get_state().clone()
    try:
        sample_paths_negbin(t([300.0]), t([1.0]), 4, generator=generator)
    except SampleError as exc:
        assert "above MAX_SUPPORT" in str(exc)
    else:
        raise AssertionError("the known out-of-cap plain distribution did not refuse")
    assert torch.equal(
        before, generator.get_state()
    ), "support certification consumed uniforms before refusing"
    control = torch.Generator().manual_seed(seed)
    assert torch.equal(torch.rand(4, generator=generator), torch.rand(4, generator=control))


def test_seeded_paths_are_reproducible_and_match_the_hurdle_zero_rate() -> None:
    """A fixed seed must reproduce the same paths, and the zero rate must still track 1-p.

    This is deliberately not evidence for active-occurrence PMF subsetting: the repaired sampler
    certifies and builds tables for every p>0 cell because seed-dependent refusal is invalid.
    """
    p, mu, k = t([0.3] * 200), t([2.0] * 200), t([1.5] * 200)
    first = sample_paths(p, mu, k, 200, generator=gen(11))
    again = sample_paths(p, mu, k, 200, generator=gen(11))
    assert torch.equal(first, again), "the sampler stopped being reproducible from its seed"
    zero_rate = float((first == 0).double().mean())
    assert abs(zero_rate - 0.7) < 0.02, f"zero rate {zero_rate:.3f} does not track 1-p=0.7"
    assert int(first.min()) == 0 and int(first.max()) >= 1


def test_a_concentrated_distribution_is_not_refused_by_its_own_rounding() -> None:
    """Per-term evaluation error, not accumulation, is what bites a SHORT support.

    mu=0.03 with k=316 puts almost all the zero-truncated mass on the count 1, so the CDF is one
    term of size ~1 carrying its own lgamma/exp error -- measured at ~660 ulps, landing the sum at
    1 + 1.47e-13. A tolerance that scaled only with the term count allowed 2.8e-14 at this tier
    and refused the distribution outright, where the pre-change sampler returned a draw. Both
    parameters are ordinary: k is well under the head's own exp(LOG_K_MAX) ~ 8103 ceiling, and a
    small mu is what intermittent demand looks like."""
    p, mu, k = t([1.0]), t([0.03]), t([316.22776601683796])
    drawn = sample_paths(p, mu, k, 8, generator=gen(0))
    assert drawn.shape == (8, 1)
    assert int(drawn.min()) >= 1, "p=1 must always produce a positive count"


def test_refusal_does_not_depend_on_the_coin() -> None:
    """Distribution certification must depend on parameters, not on the sampled occurrence draw,
    random seed or requested draw count.
    """
    p, mu, k = t([0.5]), t([0.03]), t([316.22776601683796])
    outcomes = set()
    for seed in range(8):
        try:
            sample_paths(p, mu, k, 1, generator=gen(seed))
            outcomes.add("returned")
        except SampleError:
            outcomes.add("refused")
    assert outcomes == {"returned"}, (
        f"the same parameters gave {outcomes} across seeds; a refusal that depends on a coin "
        "toss is not a contract"
    )

    for n_draws in (1, 2, 3, 16):
        try:
            sample_paths(p, mu, k, n_draws, generator=gen(0))
        except SampleError as e:
            raise AssertionError(
                f"n_draws={n_draws} refused where other draw counts did not: {e}"
            ) from e


def test_per_cell_supports_do_not_regress_against_the_batch_wide_sampler() -> None:
    """Across the head's own parameter domain, nothing the previous sampler could draw may now
    refuse. An independent review found 47 such regressions from power-of-two tiering; this is
    the guard that keeps the count at zero."""
    mus = torch.logspace(-2, 2.3, 12, dtype=torch.float64)
    ks = torch.logspace(-2, 3.9, 14, dtype=torch.float64)
    refused = []
    for m in mus:
        for kk in ks:
            mu, k = m.reshape(1), kk.reshape(1)
            p = torch.ones(1, dtype=torch.float64)
            try:
                support_sizes(mu, k)
            except SampleError:
                continue  # genuinely past MAX_SUPPORT; refusing is correct
            try:
                sample_paths(p, mu, k, 1, generator=gen(0))
            except SampleError as e:
                refused.append((float(m), float(kk), str(e)[:60]))
    assert not refused, (
        f"{len(refused)} certifiable parameter pairs were refused by the sampler, e.g. "
        f"{refused[:3]}"
    )


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print("PASS " + name)
