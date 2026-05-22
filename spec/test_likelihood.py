"""Tests for the likelihood contract, against HAND-COMPUTED values.

Real data first does not apply here and the reason is worth stating: a
likelihood is a definition, not a property of any dataset. There is no real M5 cell that can tell
you whether the truncation constant belongs in the denominator. What CAN check that is arithmetic
worked by hand, which is what most of this file is -- and a distribution that sums to one, which
is what the rest is.

The numbers below are derived in the test that uses them, so a future change that "fixes" one has
to argue with the derivation rather than with a magic constant.

Run: python3 spec/test_likelihood.py
"""

from __future__ import annotations

import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from spec.likelihood import (
    AR1,
    CENSORED_TREATMENT,
    Hurdle,
    LikelihoodError,
    ancestral_path,
    ancestral_path_plain,
    factor_link,
    iwae_bound,
    log_likelihood,
    nb_log_pmf,
    nb_log_sf,
    plain_log_likelihood,
    soft_informative_weight,
    soft_log_likelihood,
    ztnb_log_pmf,
)

# Worked by hand for mu=2, k=3, used throughout:
#   P_NB(0) = (k/(k+mu))^k                      = (3/5)^3            = 0.216
#   P_NB(1) = [G(4)/(G(3) 1!)] (3/5)^3 (2/5)^1  = 3 * 0.216 * 0.4    = 0.2592
#   P_ZTNB(1) = P_NB(1) / (1 - P_NB(0))         = 0.2592 / 0.784     = 0.33061224489...
MU, K = 2.0, 3.0
P_NB_0 = 0.216
P_NB_1 = 0.2592
P_ZTNB_1 = P_NB_1 / (1 - P_NB_0)


def test_nb_log_pmf_matches_the_hand_computed_value() -> None:
    assert math.isclose(math.exp(nb_log_pmf(0, MU, K)), P_NB_0, rel_tol=1e-12)
    assert math.isclose(math.exp(nb_log_pmf(1, MU, K)), P_NB_1, rel_tol=1e-12)


def test_the_negative_binomial_sums_to_one() -> None:
    total = sum(math.exp(nb_log_pmf(y, MU, K)) for y in range(0, 400))
    assert math.isclose(total, 1.0, rel_tol=1e-10)


def test_the_truncated_component_divides_by_one_minus_p_zero() -> None:
    """THE truncation normalisation. Without it the occurrence head and the size component both
    model P(0), the likelihood double-counts it, and the occurrence head is biased to compensate."""
    assert math.isclose(math.exp(ztnb_log_pmf(1, MU, K)), P_ZTNB_1, rel_tol=1e-12)
    # And the un-normalised value is measurably different -- 0.2592 against 0.3306.
    assert not math.isclose(math.exp(ztnb_log_pmf(1, MU, K)),
                            math.exp(nb_log_pmf(1, MU, K)), rel_tol=1e-3)


def test_the_truncated_component_sums_to_one_over_the_positive_counts() -> None:
    total = sum(math.exp(ztnb_log_pmf(y, MU, K)) for y in range(1, 400))
    assert math.isclose(total, 1.0, rel_tol=1e-10)


def test_finite_extreme_nb_parameters_do_not_fail_in_intermediate_arithmetic() -> None:
    """Finite operands used to produce 0*inf, inf-inf, and a raw OverflowError."""
    tiny = 1e-320
    assert nb_log_pmf(0, tiny, 1.0) == -math.log1p(tiny)
    assert nb_log_pmf(0, 1e-308, 1e308) == -1e-308
    assert math.isfinite(nb_log_pmf(1, 1e-308, 1e308))


def test_a_tiny_but_positive_truncation_constant_remains_defined() -> None:
    """For k=1 the positive component is geometric, so P(Y=1|Y>0)=1/(1+mu)."""
    for mu in (1e-12, 1e-320):
        assert math.isclose(ztnb_log_pmf(1, mu, 1.0), -math.log1p(mu), rel_tol=1e-15)
    # The Poisson-limit expansion is log P(Y=1|Y>0) = -mu/2 + O(mu^2).
    assert ztnb_log_pmf(1, 1e-308, 1e308) == -5e-309


def test_the_extreme_conditional_mean_is_finite_or_explicitly_refused() -> None:
    assert Hurdle(p=0.4, mu=1e-320, k=1.0).mean == 0.4
    try:
        _ = Hurdle(p=0.4, mu=1e308, k=1e-308).mean
    except LikelihoodError as error:
        assert "largest finite binary64" in str(error); return
    raise AssertionError("an unrepresentable conditional mean escaped as raw overflow")


def test_an_unresolvably_broad_extreme_survivor_is_refused_not_made_positive() -> None:
    try:
        nb_log_sf(2, 1e308, 1e-308)
    except LikelihoodError as error:
        assert "broader than binary64" in str(error); return
    raise AssertionError("returned a positive log probability for an extreme conditional tail")


def test_a_nonzero_subnormal_beta_tail_uses_the_certified_log_sum() -> None:
    """Nonzero is not sufficient evidence of relative accuracy in the subnormal range.

    The expected value was evaluated independently at 100 decimal digits. SciPy's direct
    incomplete-beta probability is nonzero here but its binary64 log is wrong by about 1.26e-7;
    the proved log-space tail sum agrees to better than 1e-12.
    """
    got = nb_log_sf(185, 1.2498406625605907, 1132.7068759258516)
    assert math.isclose(got, -729.874127923362751, rel_tol=0.0, abs_tol=2e-12), got


def test_the_whole_hurdle_sums_to_one() -> None:
    """(1-p) + p * sum_{y>=1} P_ZTNB(y) = 1. If the size component were untruncated this would
    exceed 1 by exactly p * P_NB(0)."""
    cell = Hurdle(p=0.4, mu=MU, k=K)
    total = math.exp(log_likelihood("exact", 0, cell))
    total += sum(math.exp(log_likelihood("exact", y, cell)) for y in range(1, 400))
    assert math.isclose(total, 1.0, rel_tol=1e-10)


def test_a_zero_may_not_reach_the_truncated_component() -> None:
    try:
        ztnb_log_pmf(0, MU, K)
    except LikelihoodError as e:
        assert "y >= 1" in str(e); return
    raise AssertionError("the truncated component accepted a zero, which the occurrence head owns")


def test_the_mean_is_not_p_times_mu() -> None:
    """E[Y] = p * mu / (1 - P_NB(0)). Using p*mu under-forecasts every cell, and by more the more
    intermittent the series -- which is exactly where this project's data lives."""
    cell = Hurdle(p=0.5, mu=MU, k=K)
    assert math.isclose(cell.mean, 0.5 * MU / (1 - P_NB_0), rel_tol=1e-12)
    assert cell.mean > 0.5 * MU
    # The gap is 27.6% here, not a rounding difference.
    assert (cell.mean - 0.5 * MU) / (0.5 * MU) > 0.25


def test_the_mean_agrees_with_a_brute_force_expectation() -> None:
    cell = Hurdle(p=0.4, mu=MU, k=K)
    brute = sum(y * math.exp(log_likelihood("exact", y, cell)) for y in range(1, 400))
    assert math.isclose(brute, cell.mean, rel_tol=1e-9)


def test_each_observation_state_contributes_what_the_table_says() -> None:
    cell = Hurdle(p=0.4, mu=MU, k=K)
    assert math.isclose(log_likelihood("exact", 0, cell), math.log(0.6), rel_tol=1e-12)
    assert math.isclose(log_likelihood("exact", 1, cell),
                        math.log(0.4) + math.log(P_ZTNB_1), rel_tol=1e-12)
    assert log_likelihood("unobserved", 7, cell) == 0.0
    assert log_likelihood("unobserved", 0, cell) == 0.0


def test_a_stockout_contributes_the_censored_term_not_the_exact_one() -> None:
    """The decision the blocker exists to force. `lower_bound` at c uses P(Y >= c | Y >= 1), which
    is strictly larger than P(Y = c) -- so a stockout is NOT scored as if that quantity were the
    demand."""
    assert CENSORED_TREATMENT == "right_censored"
    cell = Hurdle(p=0.4, mu=MU, k=K)
    censored = log_likelihood("lower_bound", 3, cell)
    exact = log_likelihood("exact", 3, cell)
    assert censored > exact, "the censored contribution must exceed the point one"
    # And it is not the masked version either, which would drop the magnitude term entirely.
    assert not math.isclose(censored, math.log(0.4), rel_tol=1e-9)


def test_a_stockout_at_one_unit_carries_only_the_occurrence() -> None:
    """P(Y >= 1 | Y >= 1) is 1, so the magnitude term vanishes on its own -- the contract does not
    special-case it, the arithmetic does."""
    cell = Hurdle(p=0.4, mu=MU, k=K)
    assert math.isclose(log_likelihood("lower_bound", 1, cell), math.log(0.4), rel_tol=1e-12)


def test_a_stockout_that_sold_nothing_is_vacuous() -> None:
    """Out of stock all day, sold nothing: "demand >= 0" says nothing and the occurrence is
    unknown too. Treating it as an observed zero would train the model on an assortment fact."""
    cell = Hurdle(p=0.4, mu=MU, k=K)
    assert log_likelihood("lower_bound", 0, cell) == 0.0
    assert log_likelihood("lower_bound", 0, cell) != log_likelihood("exact", 0, cell)


def test_the_censored_survivor_is_monotone_in_the_floor() -> None:
    """P(Y >= c) must fall as c rises. A survivor computed by subtracting a truncated upper tail
    would not be monotone, which is how that bug shows itself."""
    values = [nb_log_sf(c, MU, K) for c in range(1, 12)]
    assert values == sorted(values, reverse=True)
    assert values[0] == 0.0


def test_an_unknown_observation_state_is_refused_rather_than_defaulted() -> None:
    try:
        log_likelihood("probably_fine", 1, Hurdle(p=0.4, mu=MU, k=K))
    except LikelihoodError as e:
        assert "must not be given a default" in str(e); return
    raise AssertionError("a state the contract does not define was given a value anyway")


def test_a_degenerate_occurrence_probability_is_refused() -> None:
    for bad in (0.0, 1.0):
        try:
            Hurdle(p=bad, mu=MU, k=K)
        except LikelihoodError as e:
            assert "(0,1) exclusive" in str(e); continue
        raise AssertionError(f"accepted p={bad}, which makes one outcome's log-likelihood -inf")


def test_the_factor_enters_both_heads_through_separate_loadings() -> None:
    """One drawn factor, two loading vectors. Forcing one loading to serve both heads would make
    them move together by construction rather than by evidence."""
    f = (1.0, -0.5)
    a = factor_link(0.0, 0.0, f, (1.0, 0.0), (0.0, 0.0), log_k=math.log(K))
    b = factor_link(0.0, 0.0, f, (0.0, 0.0), (1.0, 0.0), log_k=math.log(K))
    assert a.p > b.p, "the occurrence loading did not move p"
    assert b.mu > a.mu, "the size loading did not move mu"
    assert math.isclose(a.k, b.k), "the factor must not enter the dispersion"


def test_the_factor_does_not_touch_the_dispersion() -> None:
    """A latent daily factor moving k would let the model explain a bad day by widening the
    distribution instead of shifting its mean -- intervals calibrated on average and useless
    during promotions."""
    for f in ((0.0, 0.0), (3.0, -3.0)):
        cell = factor_link(0.1, 0.2, f, (1.0, 1.0), (1.0, 1.0), log_k=math.log(K))
        assert math.isclose(cell.k, K, rel_tol=1e-12)


def test_a_mismatched_loading_length_is_refused() -> None:
    try:
        factor_link(0.0, 0.0, (1.0, 2.0), (1.0,), (1.0, 2.0), log_k=0.0)
    except LikelihoodError as e:
        assert "silently drops or repeats" in str(e); return
    raise AssertionError("a loading of the wrong rank was accepted")


def test_the_ar1_innovation_scale_holds_the_marginal_variance_at_one() -> None:
    """sqrt(1 - phi^2), not 1. With unit innovations the variance grows as 1 + phi^2 + ..., so
    later horizon days get a wider latent term for no modelled reason."""
    ar = AR1(phi=(0.6, 0.0))
    assert math.isclose(ar.innovation_scale[0], 0.8, rel_tol=1e-12)   # sqrt(1 - 0.36)
    assert math.isclose(ar.innovation_scale[1], 1.0, rel_tol=1e-12)
    # Simulate: the marginal variance must stay at 1 rather than accumulate.
    rng = random.Random(0)
    xs = [[rng.gauss(0, 1)] for _ in range(20000)]
    for _ in range(30):
        for path in xs:
            path.append(0.6 * path[-1] + 0.8 * rng.gauss(0, 1))
    last = [p[-1] for p in xs]
    var = sum(v * v for v in last) / len(last)
    assert 0.93 < var < 1.07, f"marginal variance drifted to {var}"


def test_a_non_stationary_persistence_is_refused() -> None:
    for bad in (1.0, -1.0, 1.2, 10**400, math.nan, math.inf, True, "0.5"):
        try:
            AR1(phi=(bad,))
        except LikelihoodError as e:
            assert "|phi| < 1" in str(e); continue
        raise AssertionError(f"accepted phi={bad}, a random walk with unbounded variance")


def test_the_iwae_bound_is_the_log_mean_not_the_mean_log() -> None:
    """log((1+3)/2) = log 2, whereas the mean of the logs would be (log1 + log3)/2 = 0.549."""
    assert math.isclose(iwae_bound([0.0, math.log(3.0)]), math.log(2.0), rel_tol=1e-12)
    assert not math.isclose(iwae_bound([0.0, math.log(3.0)]), math.log(3.0) / 2, rel_tol=1e-6)


def test_the_iwae_bound_refuses_non_finite_draw_evidence() -> None:
    for bad in (10**400, math.nan, math.inf, -math.inf, True):
        try:
            iwae_bound([bad])
        except LikelihoodError as error:
            assert "finite real number" in str(error)
            continue
        raise AssertionError(f"accepted non-finite/non-numeric draw evidence {bad!r}")


def test_arbitrary_precision_json_numbers_fail_closed_across_likelihood_inputs() -> None:
    from spec.likelihood import _log_sf_tail

    class HostileInt(int):
        def __float__(self):
            raise RuntimeError("untrusted coercion ran")

    for invalid in (10**400, HostileInt(1)):
        calls = (
            lambda invalid=invalid: Hurdle(p=invalid, mu=1.0, k=1.0),
            lambda invalid=invalid: Hurdle(p=0.5, mu=invalid, k=1.0),
            lambda invalid=invalid: Hurdle(p=0.5, mu=1.0, k=invalid),
            lambda invalid=invalid: soft_log_likelihood(
                {"exact": invalid, "lower_bound": 0.0, "unobserved": 0.0},
                1, Hurdle(p=0.5, mu=1.0, k=1.0)),
            lambda invalid=invalid: _log_sf_tail(1, 1.0, 1.0, eps=invalid),
        )
        for call in calls:
            try:
                call()
            except LikelihoodError:
                continue
            raise AssertionError("an invalid numeric object entered likelihood arithmetic")


def test_the_factor_link_never_leaks_math_overflow_or_accepts_nonfinite_inputs() -> None:
    for occurrence in (-1000.0, 1000.0, math.nan):
        try:
            factor_link(occurrence, 0.0, (), (), (), log_k=0.0)
        except LikelihoodError:
            continue
        raise AssertionError(f"accepted an unrepresentable occurrence probability for {occurrence}")
    for size, dispersion in ((1000.0, 0.0), (0.0, 1000.0), (math.nan, 0.0)):
        try:
            factor_link(0.0, size, (), (), (), log_k=dispersion)
        except LikelihoodError:
            continue
        raise AssertionError(f"accepted unrepresentable links size={size}, log_k={dispersion}")


def test_the_bound_is_exact_with_one_draw_and_rises_with_more() -> None:
    assert math.isclose(iwae_bound([-4.2]), -4.2, rel_tol=1e-12)
    assert iwae_bound([-4.2, -1.0]) > iwae_bound([-4.2, -4.2])


def test_the_bound_survives_values_that_would_overflow_a_naive_exp() -> None:
    """Shifting by the max before exponentiating. A naive implementation returns inf here."""
    assert math.isfinite(iwae_bound([-10_000.0, -10_001.0]))


def test_the_bound_needs_at_least_one_draw() -> None:
    try:
        iwae_bound([])
    except LikelihoodError as e:
        assert "at least one factor draw" in str(e); return
    raise AssertionError("produced a bound from no draws")


def test_ancestral_sampling_returns_zero_exactly_when_the_occurrence_draw_fails() -> None:
    cell = Hurdle(p=0.4, mu=MU, k=K)
    assert ancestral_path([cell], [0.99], [0.5]) == [0]
    assert ancestral_path([cell], [0.01], [0.5])[0] >= 1


def test_a_sampled_positive_is_never_zero() -> None:
    """It inverts the ZERO-TRUNCATED CDF. Inverting the untruncated one can return 0 and would
    contradict the occurrence draw that just happened."""
    cell = Hurdle(p=0.9, mu=0.05, k=K)      # P_NB(0) is enormous here
    rng = random.Random(1)
    draws = ancestral_path([cell] * 500, [0.0] * 500, [rng.random() for _ in range(500)])
    assert all(d >= 1 for d in draws)


def test_sampling_reproduces_the_analytic_mean() -> None:
    """The tie between parts 1 and 6: if the sampler and the likelihood disagreed, every quantile
    drawn from the paths would describe a different distribution from the one that was fitted."""
    cell = Hurdle(p=0.35, mu=3.0, k=2.5)
    rng = random.Random(7)
    n = 200_000
    draws = ancestral_path([cell] * n, [rng.random() for _ in range(n)],
                           [rng.random() for _ in range(n)])
    empirical = sum(draws) / n
    assert math.isclose(empirical, cell.mean, rel_tol=0.02), \
        f"sampled mean {empirical} against analytic {cell.mean}"


def test_a_path_needs_one_uniform_per_stage_per_step() -> None:
    cell = Hurdle(p=0.4, mu=MU, k=K)
    try:
        ancestral_path([cell, cell], [0.1], [0.1, 0.2])
    except LikelihoodError as e:
        assert "once per step for each stage" in str(e); return
    raise AssertionError("sampled a path with fewer uniforms than steps")


def test_a_uniform_outside_the_unit_interval_is_refused() -> None:
    cell = Hurdle(p=0.4, mu=MU, k=K)
    for u in (1.0, -0.1):
        try:
            ancestral_path([cell], [u], [0.5])
        except LikelihoodError as e:
            assert "[0,1)" in str(e); continue
        raise AssertionError(f"accepted u={u}")


def test_both_reference_samplers_reject_bool_and_numeric_subclass_uniforms() -> None:
    """`False` is numerically zero, but it is not a recorded U(0,1) draw; exact built-in types
    keep bools and caller-defined comparison behaviour outside the sampling contract."""
    class IntAlias(int):
        pass

    class FloatAlias(float):
        pass

    cell = Hurdle(p=0.4, mu=MU, k=K)
    for invalid in (False, True, IntAlias(0), FloatAlias(0.5)):
        calls = (
            lambda invalid=invalid: ancestral_path_plain([MU], [K], [invalid]),
            lambda invalid=invalid: ancestral_path([cell], [invalid], [0.5]),
            lambda invalid=invalid: ancestral_path([cell], [0.0], [invalid]),
        )
        for call in calls:
            try:
                call()
            except LikelihoodError as error:
                assert "built-in reals" in str(error)
                continue
            raise AssertionError(f"a reference sampler accepted uniform {invalid!r}")




def test_the_plain_head_contributions_match_the_table() -> None:
    """The resolved step-down's likelihood, fixed here so two implementations of it cannot
    disagree the way the hurdle once did. Exact zero is an ORDINARY count -- no hurdle event --
    and the censored term is the unconditional survivor."""
    import math as m
    assert plain_log_likelihood("exact", 0, 2.0, 3.0) == nb_log_pmf(0, 2.0, 3.0)
    assert plain_log_likelihood("exact", 4, 2.0, 3.0) == nb_log_pmf(4, 2.0, 3.0)
    assert plain_log_likelihood("unobserved", 7, 2.0, 3.0) == 0.0
    assert plain_log_likelihood("lower_bound", 0, 2.0, 3.0) == 0.0, "demand >= 0 says nothing"
    censored = plain_log_likelihood("lower_bound", 5, 2.0, 3.0)
    brute = m.log(sum(m.exp(nb_log_pmf(y, 2.0, 3.0)) for y in range(5, 400)))
    assert abs(censored - brute) < 1e-12, f"{censored} vs brute-force {brute}"
    # Relationship to the conditional form: unconditional = conditional + log positive mass.
    conditional = nb_log_sf(5, 2.0, 3.0)
    log_positive = m.log(-m.expm1(nb_log_pmf(0, 2.0, 3.0)))
    assert abs(censored - (conditional + log_positive)) < 1e-12
    try:
        plain_log_likelihood("mystery", 1, 2.0, 3.0)
    except LikelihoodError:
        pass
    else:
        raise AssertionError("an undefined state was given a default")


def test_the_plain_path_samples_zero_as_an_ordinary_outcome() -> None:
    """One uniform per step, untruncated CDF from 0: a u below P(0) IS a zero, with no
    occurrence draw to contradict."""
    import math as m
    p0 = m.exp(nb_log_pmf(0, 0.4, 2.0))
    path = ancestral_path_plain([0.4, 0.4, 5.0], [2.0, 2.0, 1.0],
                                [p0 / 2.0, p0 + 1e-6, 0.9])
    assert path[0] == 0, "a uniform below P(0) must draw zero"
    assert path[1] >= 1, "a uniform above P(0) must draw a positive count"
    assert path[2] >= 1
    again = ancestral_path_plain([0.4, 0.4, 5.0], [2.0, 2.0, 1.0],
                                 [p0 / 2.0, p0 + 1e-6, 0.9])
    assert path == again, "same uniforms, same path -- or no seed can reproduce a gate figure"
    try:
        ancestral_path_plain([1.0], [1.0], [1.5])
    except LikelihoodError:
        pass
    else:
        raise AssertionError("a uniform outside [0,1) was accepted")


def test_soft_states_marginalise_and_degenerate_to_the_hard_contract() -> None:
    """P(s)=1 on one state must reproduce log_likelihood exactly; a genuine mixture must equal
    the hand-computed log of the probability-weighted likelihoods; and a distribution that does
    not sum to one is a feed bug the likelihood refuses to hide."""
    import math as m
    cell = Hurdle(p=0.4, mu=2.5, k=1.5)
    for state in ("exact", "lower_bound", "unobserved"):
        for y in (0, 3):
            hard = log_likelihood(state, y, cell)
            probs = {s: (1.0 if s == state else 0.0)
                     for s in ("exact", "lower_bound", "unobserved")}
            assert abs(soft_log_likelihood(probs, y, cell) - hard) < 1e-12, (state, y)
    mix = {"exact": 0.6, "lower_bound": 0.3, "unobserved": 0.1}
    want = m.log(0.6 * m.exp(log_likelihood("exact", 3, cell))
                 + 0.3 * m.exp(log_likelihood("lower_bound", 3, cell)) + 0.1)
    assert abs(soft_log_likelihood(mix, 3, cell) - want) < 1e-12
    assert abs(soft_informative_weight(mix, 3) - 0.9) < 1e-12
    assert abs(soft_informative_weight(mix, 0) - 0.6) < 1e-12, "a zero floor is vacuous mass"
    try:
        soft_log_likelihood({"exact": 0.7, "lower_bound": 0.7, "unobserved": 0.0}, 1, cell)
    except LikelihoodError:
        pass
    else:
        raise AssertionError("a distribution summing to 1.4 was renormalised into a likelihood")


def test_soft_informative_weight_requires_the_same_complete_distribution_as_likelihood() -> None:
    """The normalisation weight cannot accept a feed the likelihood refuses, including invalid
    lower-bound mass on a zero floor that would otherwise be skipped by the formula."""
    invalid_distributions = (
        {"exact": 1.0, "lower_bound": 0.0},
        {"exact": 1.0, "lower_bound": 0.0, "unobserved": 0.0, "other": 0.0},
        {"exact": 1.0, "lower_bound": False, "unobserved": 0.0},
        {"exact": 1.1, "lower_bound": -0.1, "unobserved": 0.0},
        {"exact": 1.0, "lower_bound": math.nan, "unobserved": 0.0},
        {"exact": 1.0, "lower_bound": math.inf, "unobserved": 0.0},
        {"exact": 0.5, "lower_bound": 0.2, "unobserved": 0.2},
    )
    for probs in invalid_distributions:
        try:
            soft_informative_weight(probs, 0)
        except LikelihoodError:
            continue
        raise AssertionError(f"soft weight accepted an invalid state distribution: {probs}")


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn(); print("PASS " + name)
