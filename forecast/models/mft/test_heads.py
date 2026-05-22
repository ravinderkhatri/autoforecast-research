"""Parity between the torch head and the likelihood contract, plus the head's own guards.

`spec/likelihood.py` is the authority; this file exists to make disagreement fail the build. The
same device is already used between `score.py` and `score_spark.py`, and for the same reason: two
implementations of one definition is a real risk, and picking one and hoping is not a control.

REAL DATA WHERE IT APPLIES. A likelihood is a definition, so most of this is
parity against hand-checkable arithmetic -- but the COUNTS it is evaluated at do not have to be
invented, and real M5 quantities exercise the intermittency and the occasional large day that a
made-up grid would not think to include. `real_counts()` reads them straight from the Parquet.

Run: python3 forecast/models/mft/test_heads.py
"""

from __future__ import annotations

import glob
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent.parent))

import torch
from forecast.models.mft.heads import (
    MAX_CENSOR_COUNT,
    STATE_CODES,
    HeadError,
    HurdleNegBinHead,
    NegBinHead,
    hurdle_log_likelihood,
    hurdle_mean,
    informative_mask,
    log1mexp,
    nb_log_pmf,
    ztnb_log_pmf,
    ztnb_log_sf,
)
from spec import likelihood as ref

TOL = 1e-9
PARAMS = [
    (0.5, 2.0, 3.0),
    (0.1, 0.05, 1.5),
    (0.9, 12.0, 0.4),
    (0.35, 3.0, 2.5),
    (0.02, 0.8, 20.0),
    (0.75, 40.0, 5.0),
]


def t(x) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float64)


def real_counts(limit: int = 400) -> list[int]:
    """Real M5 quantities, straight from the Parquet. Falls back to a stated grid without it."""
    import pyarrow.parquet as pq

    frags = sorted(
        glob.glob(
            str(
                pathlib.Path(__file__).resolve().parent.parent.parent
                / "data"
                / "m5_parquet"
                / "*"
                / "*.parquet"
            )
        )
    )
    if not frags:
        print("    (no M5 parquet; using a declared SYNTHETIC count grid instead)")
        return [0, 1, 2, 3, 5, 8, 13, 21, 40]
    col = pq.read_table(frags[0], columns=["qty"]).column("qty").to_pylist()
    seen, out = set(), []
    for v in col:
        if v is not None and v not in seen:
            seen.add(v)
            out.append(int(v))
        if len(out) >= limit:
            break
    return sorted(out)


# ---------------------------------------------------------------- parity with the contract


def test_nb_log_pmf_matches_the_contract_on_real_m5_counts() -> None:
    counts = real_counts()
    for p, mu, k in PARAMS:
        del p
        got = nb_log_pmf(t([float(c) for c in counts]), t(mu), t(k))
        for i, c in enumerate(counts):
            assert abs(float(got[i]) - ref.nb_log_pmf(c, mu, k)) < TOL, (c, mu, k)


def test_ztnb_log_pmf_matches_the_contract_including_the_truncation() -> None:
    counts = [c for c in real_counts() if c >= 1]
    for _, mu, k in PARAMS:
        got = ztnb_log_pmf(t([float(c) for c in counts]), t(mu), t(k))
        for i, c in enumerate(counts):
            assert abs(float(got[i]) - ref.ztnb_log_pmf(c, mu, k)) < TOL, (c, mu, k)


def test_the_censored_survivor_matches_the_contract_deep_into_the_tail() -> None:
    """The test that found the worst defect in this module, so it is worth saying what it caught.

    Both implementations first computed the survivor as `1 - sum_{y<c} P(y)`. That is the
    difference of two numbers agreeing to fifteen digits, so it loses everything exactly where the
    survivor is small -- the tail, which is large stockouts, which is where the censoring contract
    matters most. Measured at `c=200, mu=0.05, k=1.5`: the true value is **-681.0** and the
    subtraction returned **-34.03**. Off by 647 nats, silently, and the loss would still have gone
    down. The same form also REFUSED cells (c=50, mu=2, k=3) whose survivor is perfectly ordinary.

    Both now sum the upper tail in log space. Worst disagreement across this grid: 2.3e-13, at
    values as small as `log P = -1367`, which neither form could previously represent at all.
    """
    floors = [2, 5, 10, 24, 50, 100, 200, 400]
    checked = 0
    for _, mu, k in PARAMS:
        for c in floors:
            want = ref.nb_log_sf(c, mu, k)
            got = float(ztnb_log_sf(t([float(c)]), t([mu]), t([k]))[0])
            assert abs(got - want) < 1e-10, (c, mu, k, got, want)
            checked += 1
    assert checked == len(PARAMS) * len(floors), "a case was skipped rather than checked"


def test_the_specific_case_that_the_subtraction_form_got_wrong() -> None:
    """Regression, pinned to the measured value. -681.0, not -34.03."""
    got = float(ztnb_log_sf(t([200.0]), t([0.05]), t([1.5]))[0])
    assert abs(got - ref.nb_log_sf(200, 0.05, 1.5)) < 1e-10
    assert got < -600.0, f"got {got}; the subtraction form returned -34.03 here"


def test_the_survivor_never_needs_an_unbounded_tail() -> None:
    """The term count reaches the mode and proves a bound; it is never silently truncated."""
    from forecast.models.mft.heads import TAIL_TERMS_CAP, _tail_terms

    assert _tail_terms(t([200.0]), t([0.05]), t([1.5])) < 20
    assert _tail_terms(t([2.0]), t([30.0]), t([1.2])) > 300
    try:
        _tail_terms(t([2.0]), t([10_000.0]), t([8_000.0]))
    except HeadError as e:
        assert f"above TAIL_TERMS_CAP={TAIL_TERMS_CAP}" in str(e)
        return
    raise AssertionError("a tail that cannot converge inside the cap was truncated silently")


def test_tail_terms_refuses_an_unrepresentable_error_target() -> None:
    from forecast.models.mft.heads import _tail_terms

    class HostileInt(int):
        def __float__(self):
            raise RuntimeError("untrusted coercion ran")

    for invalid in (10**400, HostileInt(1)):
        try:
            _tail_terms(t([2.0]), t([3.0]), t([1.2]), eps=invalid)
        except HeadError as exc:
            assert "finite and in (0,1)" in str(exc)
            continue
        raise AssertionError("an invalid tail tolerance entered head arithmetic")


def test_near_poisson_survivors_are_dense_accurate_and_one_is_exact() -> None:
    """Regression for the invalid `q` tail bound.

    With mu=100 and large k, q=mu/(mu+k) is small but adjacent probabilities around the mode are
    near one. The old code stopped after only a handful of counts and returned log survivors of
    -6, -47 and -64 at c=1, even though P(Y>=1 | Y>=1) is exactly one.
    """
    floors = list(range(1, 251))
    c = t([float(v) for v in floors])
    for k_value in (100.0, 1_000.0, 8_000.0):
        got = ztnb_log_sf(c, torch.full_like(c, 100.0), torch.full_like(c, k_value))
        assert float(got[0]) == 0.0
        want = t([ref.nb_log_sf(v, 100.0, k_value) for v in floors])
        assert float((got - want).abs().max()) < 1e-9, k_value


def test_a_vacuous_floor_cannot_inflate_an_unrelated_batch_tail() -> None:
    """Batching must not make a likelihood undefined.

    The first cell's mode is around ten thousand but c=1 needs no tail. The ordinary second cell
    must give the same answer alone or beside that vacuous floor.
    """
    c = t([1.0, 2.0])
    mu = t([10_000.0, 2.0]).requires_grad_()
    k = t([8_000.0, 3.0]).requires_grad_()
    mixed = ztnb_log_sf(c, mu, k)
    alone = ztnb_log_sf(c[1:], mu[1:], k[1:])
    assert float(mixed[0].detach()) == 0.0
    assert torch.allclose(mixed[1:], alone, atol=0.0, rtol=0.0)
    mixed.sum().backward()
    assert bool(torch.isfinite(mu.grad).all()) and bool(torch.isfinite(k.grad).all())


def test_the_survivor_keeps_parameter_gradients_after_tail_selection() -> None:
    """Choosing an integer support may detach; the likelihood terms themselves must not."""
    mu = t([100.0, 100.0]).requires_grad_()
    k = t([8_000.0, 1_000.0]).requires_grad_()
    got = ztnb_log_sf(t([1.0, 120.0]), mu, k)
    assert float(got[0].detach()) == 0.0
    got.sum().backward()
    assert torch.isfinite(mu.grad).all() and torch.isfinite(k.grad).all()
    assert float(mu.grad[1].abs()) > 0.0 and float(k.grad[1].abs()) > 0.0
    assert float(mu.grad[0]) == 0.0 and float(k.grad[0]) == 0.0

    only_one_mu = t([100.0]).requires_grad_()
    only_one_k = t([8_000.0]).requires_grad_()
    exact_zero = ztnb_log_sf(t([1.0]), only_one_mu, only_one_k)
    exact_zero.sum().backward()
    assert float(exact_zero[0].detach()) == 0.0
    assert float(only_one_mu.grad[0]) == 0.0 and float(only_one_k.grad[0]) == 0.0


def test_the_full_likelihood_matches_the_contract_for_every_state() -> None:
    """The one that matters. Every state, every real count, every parameter set."""
    counts = [c for c in real_counts() if c <= 60]
    for name, code in STATE_CODES.items():
        for p, mu, k in PARAMS:
            got = hurdle_log_likelihood(
                t([float(c) for c in counts]),
                t([p] * len(counts)),
                t([mu] * len(counts)),
                t([k] * len(counts)),
                torch.full((len(counts),), code),
            )
            cell = ref.Hurdle(p=p, mu=mu, k=k)
            for i, c in enumerate(counts):
                want = ref.log_likelihood(name, c, cell)
                assert abs(float(got[i]) - want) < 1e-8, (name, c, p, mu, k)


def test_the_mean_matches_the_contract_and_is_not_p_times_mu() -> None:
    for p, mu, k in PARAMS:
        got = float(hurdle_mean(t(p), t(mu), t(k)))
        assert abs(got - ref.Hurdle(p=p, mu=mu, k=k).mean) < TOL
        assert got > p * mu


# ---------------------------------------------------------------- numerics


def test_log1mexp_is_accurate_where_the_naive_form_is_not() -> None:
    """`log(1 - exp(x))` for x close to 0 -- the truncation constant when the size head predicts
    near-zero demand, which on a 59%-zero panel is not a corner case.

    MEASURED, after a first version of this test asserted the wrong failure. The naive form does
    NOT return -inf in float64; it loses precision to cancellation, because `1 - exp(x)` subtracts
    two numbers that agree to fifteen digits. At x=-1e-16 it is wrong by 0.105 nats and at
    x=-1e-13 by 3.1e-4 -- errors that enter every cell's likelihood, not a corner case that
    crashes and gets noticed."""
    probes = [-1e-16, -1e-13, -1e-12, -1e-8]
    x = t(probes)
    stable = log1mexp(x)
    naive = torch.log1p(-torch.exp(x))
    for i, v in enumerate(probes):
        exact = math.log(-math.expm1(v))
        assert abs(float(stable[i]) - exact) < 1e-12, f"stable form wrong at {v}"
    assert (
        abs(float(naive[0]) - math.log(-math.expm1(probes[0]))) > 1e-2
    ), "the naive form was accurate at 1e-16, so this test proves nothing about the branch"
    assert abs(float(naive[1]) - math.log(-math.expm1(probes[1]))) > 1e-5


def test_log1mexp_refuses_a_non_negative_argument() -> None:
    try:
        log1mexp(t([0.0]))
    except HeadError as e:
        assert "collapsed onto no demand" in str(e)
        return
    raise AssertionError("accepted P_NB(0) >= 1 and would have returned log of a non-positive")


def test_extreme_finite_nb_inputs_match_the_analytic_limits() -> None:
    cases = [(0.0, 1e-320, 1.0), (0.0, 1e-308, 1e308), (1.0, 1e-308, 1e308), (2.0, 1e-308, 1e308)]
    for count, mu, k in cases:
        got = float(nb_log_pmf(t([count]), t([mu]), t([k]))[0])
        want = ref.nb_log_pmf(int(count), mu, k)
        assert math.isfinite(got) and got == want, (count, mu, k, got, want)


def test_tiny_zero_truncation_keeps_value_and_gradient_finite() -> None:
    for mu_value, k_value in ((1e-12, 1.0), (1e-320, 1.0), (1e-308, 1e308)):
        mu = t([mu_value]).requires_grad_()
        k = t([k_value]).requires_grad_()
        got = ztnb_log_pmf(t([1.0]), mu, k)
        assert math.isclose(
            float(got[0].detach()), ref.ztnb_log_pmf(1, mu_value, k_value), rel_tol=1e-14
        )
        got.sum().backward()
        assert bool(torch.isfinite(mu.grad).all()) and bool(torch.isfinite(k.grad).all())


def test_the_full_hurdle_path_uses_the_stable_tiny_truncation() -> None:
    mu = t([1e-320]).requires_grad_()
    k = t([1.0]).requires_grad_()
    value = hurdle_log_likelihood(t([1.0]), t([0.4]), mu, k, torch.tensor([0]))
    value.sum().backward()
    assert math.isfinite(float(value[0].detach()))
    assert bool(torch.isfinite(mu.grad).all()) and bool(torch.isfinite(k.grad).all())


def test_an_unrepresentable_hurdle_mean_raises_instead_of_returning_inf() -> None:
    try:
        hurdle_mean(t([0.4]), t([1e308]), t([1e-308]))
    except HeadError as error:
        assert "largest finite" in str(error)
        return
    raise AssertionError("returned an infinite mean that would corrupt path summaries")


def test_an_unresolvably_broad_extreme_tail_raises_the_head_contract() -> None:
    try:
        ztnb_log_sf(t([2.0]), t([1e308]), t([1e-308]))
    except HeadError as error:
        assert "TAIL_TERMS_CAP" in str(error)
        return
    raise AssertionError("returned an uncertified extreme survivor")


def test_the_survivor_is_monotone_in_the_floor_on_real_counts() -> None:
    floors = sorted(c for c in real_counts() if 1 <= c <= 40)
    got = ztnb_log_sf(t([float(c) for c in floors]), t([2.0] * len(floors)), t([3.0] * len(floors)))
    values = [float(v) for v in got]
    assert values == sorted(values, reverse=True), values


# ---------------------------------------------------------------- guards


def test_a_censored_floor_beyond_the_cap_raises_rather_than_clipping() -> None:
    """Clipping would score a 900-unit stockout as a 512-unit one -- understating demand in
    exactly the cells the censoring contract protects, and silently, since the loss still falls."""
    try:
        ztnb_log_sf(t([float(MAX_CENSOR_COUNT + 1)]), t([2.0]), t([3.0]))
    except HeadError as e:
        assert "Clipping it would" in str(e)
        return
    raise AssertionError("a censored floor past the cap was clipped instead of refused")


def test_invalid_counts_and_parameters_are_refused_before_log_gamma_arithmetic() -> None:
    cases = [
        lambda: nb_log_pmf(t([1.5]), t([2.0]), t([3.0])),
        lambda: nb_log_pmf(t([1.0]), t([float("nan")]), t([3.0])),
        lambda: ztnb_log_pmf(t([0.0]), t([2.0]), t([3.0])),
        lambda: ztnb_log_sf(t([2.5]), t([2.0]), t([3.0])),
    ]
    for call in cases:
        try:
            call()
        except HeadError:
            continue
        raise AssertionError("invalid likelihood input reached gamma/log arithmetic")


def test_an_unknown_state_code_is_refused_rather_than_defaulted() -> None:
    try:
        hurdle_log_likelihood(t([1.0]), t([0.4]), t([2.0]), t([3.0]), torch.tensor([7]))
    except HeadError as e:
        assert "must not be given a default" in str(e)
        return
    raise AssertionError("an undefined state code was given a value")


def test_the_informative_mask_separates_uninformative_zeros_from_predicted_ones() -> None:
    """A contribution of 0 means UNINFORMATIVE, never well predicted. Nothing downstream can tell
    them apart from the value alone, which is why the mask exists."""
    y = t([0.0, 3.0, 0.0, 3.0, 0.0, 3.0])
    state = torch.tensor([0, 0, 1, 1, 2, 2])
    got = informative_mask(y, state).tolist()
    assert got == [True, True, False, True, False, False]


# ---------------------------------------------------------------- the module


def head(d_model: int = 4, n_static: int = 3) -> HurdleNegBinHead:
    return HurdleNegBinHead(d_model=d_model, n_static=n_static).double()


def test_an_untrained_head_predicts_exactly_its_baseline() -> None:
    """Zero BIAS, so the offset is the prediction on step one -- a head that started at zero would
    spend its first epochs climbing back to a baseline it was given for free.

    The weights are small rather than zero, for a reason a companion adapter test found:
    weights of exactly zero send zero gradient upstream, so the whole encoder learns nothing on
    the first step."""
    h = head()
    hidden, static = torch.zeros(5, 4, dtype=torch.float64), torch.zeros(5, 3, dtype=torch.float64)
    logit, log_size = t([-1.0] * 5), t([0.7] * 5)
    p, mu, k = h(hidden, logit, log_size, static)
    assert torch.allclose(p, torch.sigmoid(logit))
    assert torch.allclose(mu, torch.exp(log_size))
    assert torch.allclose(k, torch.ones_like(k))


def test_the_factor_moves_both_heads_and_never_the_dispersion() -> None:
    h = head()
    args = (
        torch.zeros(3, 4, dtype=torch.float64),
        t([0.0] * 3),
        t([0.0] * 3),
        torch.zeros(3, 3, dtype=torch.float64),
    )
    base_p, base_mu, base_k = h(*args)
    p2, mu2, k2 = h(*args, factor_occurrence=t([0.5] * 3), factor_size=t([0.5] * 3))
    assert (p2 > base_p).all() and (mu2 > base_mu).all()
    assert torch.allclose(k2, base_k), "the factor reached the dispersion"


def test_the_loss_divides_by_informative_cells_not_by_tensor_size() -> None:
    """A batch that is mostly never-ranged would otherwise report a small loss simply because most
    cells contribute zero -- and the loss would FALL as coverage got worse."""
    h = head()
    hidden, static = torch.zeros(4, 4, dtype=torch.float64), torch.zeros(4, 3, dtype=torch.float64)
    logit, log_size = t([-0.5] * 4), t([0.5] * 4)
    y = t([2.0, 2.0, 2.0, 2.0])
    all_exact = h.loss(y, torch.tensor([0, 0, 0, 0]), hidden, logit, log_size, static)
    half_unobs = h.loss(y, torch.tensor([0, 0, 2, 2]), hidden, logit, log_size, static)
    assert (
        abs(float(all_exact.detach()) - float(half_unobs.detach())) < 1e-12
    ), "the loss changed when two cells became unobserved, so it is averaging over the tensor"


def test_a_batch_with_nothing_informative_is_refused() -> None:
    h = head()
    hidden, static = torch.zeros(2, 4, dtype=torch.float64), torch.zeros(2, 3, dtype=torch.float64)
    try:
        h.loss(t([0.0, 0.0]), torch.tensor([2, 2]), hidden, t([0.0] * 2), t([0.0] * 2), static)
    except HeadError as e:
        assert "report a loss of zero" in str(e)
        return
    raise AssertionError("averaged over a batch with no informative cell")


def test_the_loss_is_finite_and_its_gradient_flows_on_real_counts() -> None:
    """The whole head is logs and gammas; a NaN anywhere kills a run silently three hours in."""
    h = head()
    counts = [float(c) for c in real_counts() if c <= 60][:32]
    n = len(counts)
    hidden = torch.randn(n, 4, dtype=torch.float64, requires_grad=True)
    static = torch.randn(n, 3, dtype=torch.float64)
    state = torch.tensor([[0, 1, 2][i % 3] for i in range(n)])
    loss = h.loss(t(counts), state, hidden, t([-0.5] * n), t([0.4] * n), static)
    assert torch.isfinite(loss), "the loss is not finite on real counts"
    loss.backward()
    assert torch.isfinite(hidden.grad).all(), "a non-finite gradient reached the trunk"
    assert any(pm.grad is not None and torch.isfinite(pm.grad).all() for pm in h.parameters())


def test_a_saturating_logit_cannot_reach_p_equal_one_or_nan_the_gradient() -> None:
    """In float32 `sigmoid` saturates to exactly 1.0 past a logit of ~16.6, `log1p(-p)` is then
    `-inf`, and `torch.where`'s backward multiplies the unselected branch's local gradient by
    zero -- `0 * inf = NaN`, poisoning every shared parameter. The contract requires p strictly
    inside (0,1); the clamp is how the tensor path holds that."""
    h = HurdleNegBinHead(4, 2)
    hidden = torch.randn(6, 4, requires_grad=True)
    baseline = torch.full((6,), 50.0)  # far past float32 saturation
    p, mu, k = h(hidden, baseline, torch.zeros(6), torch.rand(6, 2))
    assert bool((p < 1.0).all()), "p reached exactly 1.0; log1p(-p) is -inf from here"
    y = torch.zeros(6)  # exact zeros against p ~ 1: worst case
    state = torch.zeros(6, dtype=torch.long)
    loss = h.loss(y, state, hidden, baseline, torch.zeros(6), torch.rand(6, 2))
    assert torch.isfinite(loss), f"loss {loss} is not finite under a saturating logit"
    loss.backward()
    assert torch.isfinite(hidden.grad).all(), "NaN reached the encoder through the where-branch"


def test_the_dispersion_is_clamped_as_the_spec_table_says() -> None:
    """v3: dispersion is 'linear, CLAMPED, partially pooled'. Unclamped, exp(linear) collapses
    toward zero -- where P_NB(0) -> 1 and the truncated component refuses cells -- or explodes
    while contributing nothing but gradient noise."""
    from forecast.models.mft.heads import LOG_K_MAX, LOG_K_MIN

    h = HurdleNegBinHead(4, 2)
    with torch.no_grad():
        h.dispersion.weight.fill_(1e6)  # a head that has gone wrong
    _, _, k = h(torch.randn(3, 4), torch.zeros(3), torch.zeros(3), torch.ones(3, 2))
    assert bool((k <= math.exp(LOG_K_MAX) + 1e-3).all())
    with torch.no_grad():
        h.dispersion.weight.fill_(-1e6)
    _, _, k = h(torch.randn(3, 4), torch.zeros(3), torch.zeros(3), torch.ones(3, 2))
    assert bool((k >= math.exp(LOG_K_MIN) - 1e-9).all())


def test_the_per_series_dispersion_residual_moves_k_and_starts_pooled() -> None:
    h = HurdleNegBinHead(4, 2, n_series=5)
    static = torch.ones(2, 2)
    idx = torch.tensor([0, 3])
    _, _, k0 = h(torch.zeros(2, 4), torch.zeros(2), torch.zeros(2), static, series_index=idx)
    assert torch.allclose(k0[0], k0[1]), "zero-initialised residual must start pooled"
    with torch.no_grad():
        h.dispersion_residual.weight[3] = 1.0
    _, _, k1 = h(torch.zeros(2, 4), torch.zeros(2), torch.zeros(2), static, series_index=idx)
    assert not torch.allclose(k1[0], k1[1]), "the residual did not move this series' dispersion"


def test_a_residual_table_and_its_index_must_travel_together() -> None:
    """A silently ignored index and a silently skipped residual are the same bug from two
    directions: a mechanism that exists on paper and does nothing."""
    with_table = HurdleNegBinHead(4, 2, n_series=5)
    try:
        with_table(torch.zeros(2, 4), torch.zeros(2), torch.zeros(2), torch.ones(2, 2))
    except HeadError as e:
        assert "residual" in str(e)
    else:
        raise AssertionError("a residual table was silently skipped")
    without = HurdleNegBinHead(4, 2)
    try:
        without(
            torch.zeros(2, 4),
            torch.zeros(2),
            torch.zeros(2),
            torch.ones(2, 2),
            series_index=torch.tensor([0, 1]),
        )
    except HeadError as e:
        assert "never built" in str(e)
    else:
        raise AssertionError("an index was accepted by a head with no residual")


def test_heavy_overdispersion_censoring_is_computed_not_refused() -> None:
    """The domain gap the 2026-08-24 review's fuzz found: mu=55, k=0.2 -- a fast mover with
    heavy overdispersion, q ~ 0.996 -- with an ordinary censored floor was REFUSED by the torch
    tail (needs ~8,000 terms) while the contract computes it in closed form (-1.09 at c=2). An
    estate training batch with censoring on can contain exactly this cell, and the refusal
    killed the batch. The complement regime must now agree with the contract."""
    for c in (2, 7, 24, 60, 200, 512):
        want = ref.nb_log_sf(c, 55.0, 0.2)
        got = float(ztnb_log_sf(t([float(c)]), t([55.0]), t([0.2])))
        assert abs(got - want) <= 1e-8 * max(
            abs(want), 1.0
        ), f"c={c}: torch {got} vs contract {want}"


def test_the_survivor_regimes_tile_the_floor_contract() -> None:
    """Dense parity sweep across the decay-rate range up to q ~ 0.999 and floors up to the
    contract's cap, including both regimes and the boundary between them. Neither form covers
    the other's region alone: the certified tail cannot converge at slow decay, and the
    complement loses its digits at small survivors -- the split must be seamless."""
    for mu, k in [
        (55.0, 0.2),
        (200.0, 0.25),
        (999.0, 1.0),
        (30.0, 3.0),
        (2.0, 3.0),
        (0.05, 1.5),
        (5.0, 40.0),
        (120.0, 8.0),
        (0.8, 0.05),
    ]:
        for c in (2, 5, 17, 60, 150, 350, 512):
            want = ref.nb_log_sf(c, mu, k)
            got = float(ztnb_log_sf(t([float(c)]), t([mu]), t([k])))
            assert abs(got - want) <= 1e-8 * max(
                abs(want), 1.0
            ), f"mu={mu} k={k} c={c}: torch {got} vs contract {want}"


def test_the_old_small_survivor_case_did_not_regress() -> None:
    """The measured failure that banished the complement form in the first place: c=200,
    mu=0.05, k=1.5 has a survivor of ~1e-296 whose subtraction returned -34 against a true
    -681. The complement branch must never see it -- it belongs to the certified tail."""
    want = ref.nb_log_sf(200, 0.05, 1.5)
    got = float(ztnb_log_sf(t([200.0]), t([0.05]), t([1.5])))
    assert want < -600 and abs(got - want) <= 1e-8 * abs(want), f"torch {got} vs contract {want}"


def test_gradients_flow_through_both_survivor_regimes() -> None:
    """A mixed batch: one complement-regime cell and one tail-regime cell. Both must carry
    finite gradient to mu and k, or a censored batch trains one regime and freezes the other."""
    mu = torch.tensor([55.0, 0.05], dtype=torch.float64, requires_grad=True)
    k = torch.tensor([0.2, 1.5], dtype=torch.float64, requires_grad=True)
    out = ztnb_log_sf(t([7.0, 200.0]), mu, k)
    out.sum().backward()
    assert torch.isfinite(mu.grad).all() and torch.isfinite(k.grad).all()
    assert (
        float(mu.grad.abs().min()) > 0 and float(k.grad.abs().min()) > 0
    ), "one survivor regime carries no gradient"


def test_the_plain_head_likelihood_matches_the_contract() -> None:
    """`negbin_log_likelihood` against `likelihood.plain_log_likelihood`, every state, both
    survivor regimes -- the step-down head must not drift from the contract any more than the
    hurdle may."""
    from forecast.models.mft.heads import negbin_log_likelihood

    for mu, k in [(2.0, 3.0), (55.0, 0.2), (0.05, 1.5), (12.0, 40.0)]:
        for state_name, state_code, y in [
            ("exact", 0, 0),
            ("exact", 0, 4),
            ("lower_bound", 1, 0),
            ("lower_bound", 1, 6),
            ("lower_bound", 1, 60),
            ("unobserved", 2, 3),
        ]:
            want = ref.plain_log_likelihood(state_name, y, mu, k)
            got = float(
                negbin_log_likelihood(t([float(y)]), t([mu]), t([k]), torch.tensor([state_code]))
            )
            assert abs(got - want) <= 1e-8 * max(
                abs(want), 1.0
            ), f"state={state_name} y={y} mu={mu} k={k}: {got} vs {want}"


def test_the_unconditional_survivor_handles_c_equal_one_exactly() -> None:
    """`P(Y >= 1) = 1 - P(0)` -- the complement over one lower term, with no special case."""
    from forecast.models.mft.heads import nb_log_survivor

    got = float(nb_log_survivor(t([1.0]), t([2.0]), t([3.0])))
    want = ref.plain_nb_log_sf(1, 2.0, 3.0)
    assert abs(got - want) < 1e-10, f"{got} vs {want}"


def test_the_plain_head_module_mirrors_the_hurdle_minus_the_occurrence() -> None:
    """Same baseline offset, same clamped pooled dispersion, same residual pairing rules --
    the head choice must change exactly one thing: whether zeros get their own component."""
    from forecast.models.mft.heads import LOG_K_MAX, NegBinHead

    h = NegBinHead(4, 2, n_series=5)
    mu, k = h(
        torch.zeros(3, 4),
        torch.log(torch.tensor([2.0, 5.0, 9.0])),
        torch.ones(3, 2),
        series_index=torch.tensor([0, 1, 2]),
    )
    assert torch.allclose(
        mu, torch.tensor([2.0, 5.0, 9.0]), rtol=1e-2
    ), "an untrained plain head must start at its size baseline"
    with torch.no_grad():
        h.dispersion.weight.fill_(1e6)
    _, k = h(
        torch.zeros(3, 4), torch.zeros(3), torch.ones(3, 2), series_index=torch.tensor([0, 1, 2])
    )
    assert bool((k <= math.exp(LOG_K_MAX) + 1e-3).all()), "the dispersion clamp is missing"
    try:
        h(torch.zeros(1, 4), torch.zeros(1), torch.ones(1, 2))
    except HeadError:
        pass
    else:
        raise AssertionError("a residual table was silently skipped")


def test_the_soft_tensor_likelihood_matches_the_contract_and_its_hard_degenerate() -> None:
    from forecast.models.mft.heads import (
        hurdle_log_likelihood_soft,
        negbin_log_likelihood_soft,
        soft_informative_weight,
    )

    cell = ref.Hurdle(p=0.4, mu=2.5, k=1.5)
    mix = {"exact": 0.6, "lower_bound": 0.3, "unobserved": 0.1}
    probs = torch.tensor([[0.6, 0.3, 0.1]], dtype=torch.float64)
    for y in (0, 3, 24):
        want = ref.soft_log_likelihood(mix, y, cell)
        got = float(hurdle_log_likelihood_soft(t([float(y)]), t([0.4]), t([2.5]), t([1.5]), probs))
        assert abs(got - want) < 1e-9, (y, got, want)
    onehot = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    hard = float(hurdle_log_likelihood(t([3.0]), t([0.4]), t([2.5]), t([1.5]), torch.tensor([0])))
    soft = float(hurdle_log_likelihood_soft(t([3.0]), t([0.4]), t([2.5]), t([1.5]), onehot))
    assert abs(soft - hard) < 1e-12, "one-hot soft must BE the hard contract"
    from forecast.models.mft.heads import negbin_log_likelihood

    plain_hard = float(negbin_log_likelihood(t([3.0]), t([2.5]), t([1.5]), torch.tensor([0])))
    plain_soft = float(negbin_log_likelihood_soft(t([3.0]), t([2.5]), t([1.5]), onehot))
    assert abs(plain_soft - plain_hard) < 1e-12
    w = soft_informative_weight(
        t([0.0, 3.0]), torch.tensor([[0.6, 0.3, 0.1]] * 2, dtype=torch.float64)
    )
    assert abs(float(w[0]) - 0.6) < 1e-12 and abs(float(w[1]) - 0.9) < 1e-12
    try:
        hurdle_log_likelihood_soft(
            t([1.0]),
            t([0.4]),
            t([2.5]),
            t([1.5]),
            torch.tensor([[0.7, 0.7, 0.0]], dtype=torch.float64),
        )
    except HeadError:
        pass
    else:
        raise AssertionError("an invalid state distribution reached the likelihood")


def test_soft_state_sum_tolerance_matches_the_reference_contract_at_the_boundary() -> None:
    """At source precision both implementations retain the contract's exact 1e-9 boundary."""
    from forecast.models.mft.heads import hurdle_log_likelihood_soft

    cell = ref.Hurdle(p=0.4, mu=2.5, k=1.5)
    for excess, accepted in ((0.9e-9, True), (1.1e-9, False)):
        mix = {"exact": 0.6, "lower_bound": 0.3, "unobserved": 0.1 + excess}
        probs = torch.tensor(
            [[mix[name] for name in ("exact", "lower_bound", "unobserved")]], dtype=torch.float64
        )
        reference_error = tensor_error = None
        try:
            ref.soft_log_likelihood(mix, 3, cell)
        except ref.LikelihoodError as error:
            reference_error = error
        try:
            hurdle_log_likelihood_soft(t([3.0]), t([0.4]), t([2.5]), t([1.5]), probs)
        except HeadError as error:
            tensor_error = error
        assert (reference_error is None) == accepted
        assert (tensor_error is None) == accepted


def test_live_float32_soft_states_allow_only_storage_roundoff() -> None:
    """The builder stores a valid source distribution in float32.  That conversion can move its
    three-term sum beyond 1e-9, but must not open the door to a materially broken feed."""
    from forecast.models.mft.heads import (
        hurdle_log_likelihood_soft,
        soft_informative_weight,
    )

    source = (0.6105326650526257, 0.009026731046928181, 0.3804406039004461)
    assert sum(source) == 1.0
    probs = torch.tensor([source], dtype=torch.float32)
    assert (
        abs(float(probs.sum()) - 1.0) > 1e-9
    ), "the regression no longer distinguishes float32 storage from the source contract"
    y = torch.tensor([3.0], dtype=torch.float32)
    got = hurdle_log_likelihood_soft(
        y,
        torch.tensor([0.4]),
        torch.tensor([2.5]),
        torch.tensor([1.5]),
        probs,
    )
    assert bool(torch.isfinite(got).all())
    weight = soft_informative_weight(y, probs)
    assert 0.0 <= float(weight) <= 1.0

    # These STORED float32 values are 2.25 eps from one, but a native float32 reduction rounds
    # that drift back to exactly 2 eps. Auditing the stored values in float64 is what makes the
    # strict ``> 2 eps`` boundary fail closed. The error is also below the old 1e-6 tolerance.
    malformed = torch.tensor(
        [[0.5999994277954102, 0.30000025033950806, 0.1000000536441803]],
        dtype=torch.float32,
    )
    eps = torch.finfo(torch.float32).eps
    assert abs(float(malformed.sum()) - 1.0) == 2.0 * eps
    audited_drift = abs(float(malformed.to(torch.float64).sum()) - 1.0)
    assert 2.0 * eps < audited_drift < 1e-6
    for call in (
        lambda: hurdle_log_likelihood_soft(
            y, torch.tensor([0.4]), torch.tensor([2.5]), torch.tensor([1.5]), malformed
        ),
        lambda: soft_informative_weight(y, malformed),
    ):
        try:
            call()
        except HeadError as error:
            assert "sum to one" in str(error)
            continue
        raise AssertionError("float32 storage tolerance accepted a sum beyond roundoff")


def test_soft_informative_weight_requires_a_complete_per_cell_tensor_distribution() -> None:
    """Normalisation and likelihood consume the same exact tensor contract.  In particular,
    indexing columns zero and one must not make missing/extra states or a broadcast look valid."""
    from forecast.models.mft.heads import (
        hurdle_log_likelihood_soft,
        negbin_log_likelihood_soft,
        soft_informative_weight,
    )

    y = torch.tensor([[0.0, 3.0], [1.0, 4.0]])
    p = torch.full_like(y, 0.4)
    mu = torch.full_like(y, 2.5)
    k = torch.full_like(y, 1.5)
    invalid = (
        ([0.6, 0.3, 0.1], "torch tensor"),
        (torch.tensor(1.0), "[..., 3]"),
        (torch.full((2, 2, 2), 0.5), "[..., 3]"),
        (torch.full((2, 2, 4), 0.25), "[..., 3]"),
        (torch.tensor([0.6, 0.3, 0.1]), "describe the cells"),
        (torch.ones(2, 2, 3, dtype=torch.long), "floating"),
        (torch.ones(2, 2, 3, dtype=torch.bool), "floating"),
        (torch.ones(2, 2, 3, dtype=torch.complex64), "floating"),
        (torch.tensor([[[math.nan, 0.0, 1.0]] * 2] * 2), "finite"),
        (torch.tensor([[[math.inf, 0.0, 0.0]] * 2] * 2), "finite"),
        (torch.tensor([[[1.1, -0.1, 0.0]] * 2] * 2), "non-negative"),
        (torch.tensor([[[0.5, 0.2, 0.2]] * 2] * 2), "sum to one"),
        (torch.empty(2, 2, 3, device="meta"), "same device"),
    )
    consumers = (
        lambda probs: soft_informative_weight(y, probs),
        lambda probs: hurdle_log_likelihood_soft(y, p, mu, k, probs),
        lambda probs: negbin_log_likelihood_soft(y, mu, k, probs),
    )
    for probs, fragment in invalid:
        for consume in consumers:
            try:
                consume(probs)
            except HeadError as error:
                assert fragment in str(error), (fragment, error)
                continue
            raise AssertionError(f"soft-state consumer accepted invalid probabilities: {probs!r}")


def test_linear_predictors_and_from_predictors_compose_to_forward() -> None:
    """The head split must be exact for BOTH heads.

    `linear_predictors` computes the half that does not depend on a factor draw -- the
    projections of `hidden`, the pooled dispersion, the per-series residual -- and
    `from_predictors` adds the drawn factor and applies the clamps. `forward` composes them, so
    the two routes must agree bit-for-bit. If they ever diverge, integrating over many factor
    draws would silently compute something other than the model."""
    torch.manual_seed(0)
    cells, horizon, d, n_static, n_series = 6, 4, 8, 3, 6
    hidden = torch.randn(cells, horizon, d)
    static = torch.randn(cells, horizon, n_static)
    logit = torch.randn(cells, horizon)
    log_size = torch.randn(cells, horizon)
    series = torch.arange(cells).unsqueeze(-1).expand(cells, horizon)
    f_occ, f_size = torch.randn(cells, horizon) * 0.2, torch.randn(cells, horizon) * 0.2

    hurdle = HurdleNegBinHead(d, n_static, n_series=n_series)
    direct = hurdle(hidden, logit, log_size, static, f_occ, f_size, series)
    eta_occ, eta_size, log_k = hurdle.linear_predictors(hidden, logit, log_size, static, series)
    split = hurdle.from_predictors(eta_occ, eta_size, log_k, f_occ, f_size)
    for name, a, b in zip(("p", "mu", "k"), direct, split, strict=True):
        assert torch.equal(a, b), f"hurdle head {name} differs between forward and the split"

    # The cached half must be REUSABLE: a second factor against the same predictors must match a
    # second full forward. That reuse is the entire point of splitting.
    other_occ, other_size = torch.randn(cells, horizon) * 0.3, torch.randn(cells, horizon) * 0.3
    again = hurdle.from_predictors(eta_occ, eta_size, log_k, other_occ, other_size)
    fresh = hurdle(hidden, logit, log_size, static, other_occ, other_size, series)
    for name, a, b in zip(("p", "mu", "k"), fresh, again, strict=True):
        assert torch.equal(a, b), f"hurdle head {name} differs when predictors are reused"

    plain = NegBinHead(d, n_static, n_series=n_series)
    direct_nb = plain(hidden, log_size, static, f_size, series)
    eta_size_nb, log_k_nb = plain.linear_predictors(hidden, log_size, static, series)
    split_nb = plain.from_predictors(eta_size_nb, log_k_nb, f_size)
    for name, a, b in zip(("mu", "k"), direct_nb, split_nb, strict=True):
        assert torch.equal(a, b), f"plain NB head {name} differs between forward and the split"


def test_the_split_halves_keep_the_residual_contract() -> None:
    """`series_index` and the dispersion residual must still travel together -- a silently
    ignored index and a silently skipped residual are the same bug from two directions, and the
    split must not open a third door to it."""
    torch.manual_seed(0)
    head = HurdleNegBinHead(4, 2, n_series=3)
    hidden, static = torch.randn(3, 2, 4), torch.randn(3, 2, 2)
    base = torch.zeros(3, 2)
    try:
        head.linear_predictors(hidden, base, base, static, None)
    except HeadError as e:
        assert "series_index" in str(e)
    else:
        raise AssertionError("a head with a residual table accepted no series_index")

    no_residual = HurdleNegBinHead(4, 2)
    try:
        no_residual.linear_predictors(
            hidden, base, base, static, torch.zeros(3, 2, dtype=torch.long)
        )
    except HeadError as e:
        assert "no dispersion residual" in str(e)
        return
    raise AssertionError("a head without a residual table accepted a series_index")


def test_both_heads_validate_every_series_index_invariant_before_embedding() -> None:
    """Residual assignment is per cell: dtype, shape, device and range cannot be delegated to
    embedding broadcasting or its backend-specific exception."""
    cell_shape = (2, 3)
    hidden = torch.zeros(*cell_shape, 4)
    static = torch.zeros(*cell_shape, 2)
    baseline = torch.zeros(cell_shape)
    hurdle = HurdleNegBinHead(4, 2, n_series=4)
    plain = NegBinHead(4, 2, n_series=4)
    calls = (
        lambda index, statics=static: hurdle.linear_predictors(
            hidden, baseline, baseline, statics, index
        ),
        lambda index, statics=static: plain.linear_predictors(hidden, baseline, statics, index),
    )
    bad_indices = (
        (0, "torch tensor"),
        (torch.tensor(0), "exactly match"),
        (torch.tensor([0]), "exactly match"),
        (torch.tensor([0, 1]), "exactly match"),
        (torch.zeros(cell_shape), "integral"),
        (torch.zeros(cell_shape, dtype=torch.bool), "integral"),
        (torch.tensor([[-1, 0, 1], [1, 2, 3]]), "lie in"),
        (torch.tensor([[0, 1, 2], [1, 2, 4]]), "lie in"),
        (torch.empty(cell_shape, dtype=torch.long, device="meta"), "device"),
    )
    for call in calls:
        for index, fragment in bad_indices:
            try:
                call(index)
            except HeadError as error:
                assert fragment in str(error), (index, error)
                continue
            raise AssertionError(f"a head accepted invalid series_index {index}")

        call(torch.zeros(cell_shape, dtype=torch.int32))

        try:
            call(torch.zeros(cell_shape, dtype=torch.long), torch.zeros(2, 1, 2))
        except HeadError as error:
            assert "one exact cell shape" in str(error)
        else:
            raise AssertionError("a singleton dispersion predictor broadcast across cells")


def test_both_heads_reject_singleton_baselines_before_they_can_broadcast() -> None:
    """A baseline is one value per predictor cell.  Broadcast-compatible is not enough: a
    singleton row would silently give several series the same baseline before residual checks."""
    cell_shape = (2, 3)
    hidden = torch.zeros(*cell_shape, 4)
    static = torch.zeros(*cell_shape, 2)
    baseline = torch.zeros(cell_shape)
    hurdle = HurdleNegBinHead(4, 2)
    plain = NegBinHead(4, 2)
    invalid_calls = (
        lambda: hurdle.linear_predictors(hidden, torch.zeros(1, 3), baseline, static),
        lambda: hurdle.linear_predictors(hidden, baseline, torch.zeros(1, 3), static),
        lambda: hurdle.linear_predictors(hidden, baseline, baseline, torch.zeros(2, 1, 2)),
        lambda: plain.linear_predictors(hidden, torch.zeros(1, 3), static),
        lambda: plain.linear_predictors(hidden, baseline, torch.zeros(2, 1, 2)),
    )
    for call in invalid_calls:
        try:
            call()
        except HeadError as error:
            assert "one exact cell shape" in str(error)
            continue
        raise AssertionError("a singleton head input broadcast across predictor cells")

    hurdle_predictors = hurdle.linear_predictors(hidden, baseline, baseline, static)
    plain_predictors = plain.linear_predictors(hidden, baseline, static)
    assert all(value.shape == cell_shape for value in (*hurdle_predictors, *plain_predictors))


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print("PASS " + name)
