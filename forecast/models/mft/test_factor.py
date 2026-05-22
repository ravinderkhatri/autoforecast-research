"""Tests for the shared latent factor, held to the AR(1) constraint in the likelihood contract.

The properties here are statistical, so most are checked by simulation with a fixed seed rather
than by a closed form -- but each one has a stated consequence for the forecast, because "the
variance is 1" is only interesting once you know that getting it wrong makes the forecast fan out
with lead time for no modelled reason.

Run: python3 forecast/models/mft/test_factor.py
"""

from __future__ import annotations

import pathlib
import sys
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent.parent))

import torch
from forecast.models.mft.factor import MAX_PHI, FactorError, SharedLatentFactor
from spec.likelihood import AR1

RANK, STORES, HORIZON = 4, 6, 28


def factor(**kw) -> SharedLatentFactor:
    torch.manual_seed(0)
    return SharedLatentFactor(
        rank=kw.pop("rank", RANK), n_stores=kw.pop("n_stores", STORES), **kw
    ).double()


def gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_persistence_cannot_leave_the_stationary_range() -> None:
    """phi = tanh(theta) makes |phi| < 1 UNREACHABLE, not merely checked. At |phi| = 1 the factor
    is a random walk with no stationary marginal and later horizon days acquire unbounded
    variance."""
    f = factor()
    with torch.no_grad():
        f.theta.fill_(50.0)  # an optimiser step that would escape any bound
    assert bool((f.phi.abs() < 1.0).all()), f"phi escaped to {f.phi}"
    AR1(phi=tuple(float(v) for v in f.phi))  # the contract accepts it


def test_init_phi_is_finite_and_strictly_inside_the_reachable_range() -> None:
    invalid = (
        MAX_PHI,
        -MAX_PHI,
        MAX_PHI + 1e-8,
        -MAX_PHI - 1e-8,
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        "0.5",
        None,
    )
    for value in invalid:
        try:
            SharedLatentFactor(rank=2, n_stores=3, init_phi=value)  # type: ignore[arg-type]
        except FactorError as exc:
            assert "strictly inside" in str(exc), (value, exc)
            continue
        raise AssertionError(f"invalid init_phi {value!r} was accepted")

    for value in (MAX_PHI - 1e-8, -MAX_PHI + 1e-8):
        f = factor(init_phi=value)
        assert torch.isfinite(f.theta).all(), (value, f.theta)
        assert torch.isfinite(f.phi).all(), (value, f.phi)
        assert bool((f.phi.abs() < MAX_PHI).all()), (value, f.phi)


def test_default_phi_initializer_preserves_the_historical_parameter_bits() -> None:
    historical = torch.full((RANK,), float(torch.atanh(torch.tensor(0.7) / MAX_PHI)))
    observed = SharedLatentFactor(rank=RANK, n_stores=STORES).theta.detach()
    assert torch.equal(observed, historical), (observed, historical)


def test_finite_native_phi_initializer_result_is_not_recomputed_in_python() -> None:
    historical = torch.atanh(torch.tensor(0.7) / MAX_PHI)
    backend_result = torch.nextafter(historical, torch.tensor(float("inf")))
    with mock.patch.object(torch, "atanh", return_value=backend_result):
        observed = SharedLatentFactor(rank=RANK, n_stores=STORES).theta.detach()
    expected = torch.full((RANK,), float(backend_result))
    assert torch.equal(observed, expected), (observed, expected)


def test_the_innovation_scale_matches_the_contract() -> None:
    f = factor()
    want = AR1(phi=tuple(float(v) for v in f.phi)).innovation_scale
    got = f.innovation_scale
    for i, w in enumerate(want):
        assert abs(float(got[i]) - w) < 1e-12, (i, float(got[i]), w)


def test_the_marginal_variance_stays_at_one_across_the_horizon() -> None:
    """The consequence of getting this wrong: with unit innovations the variance grows as
    1 + phi^2 + phi^4 + ..., so a 28-day forecast fans out with lead time BY CONSTRUCTION rather
    than because uncertainty grows -- and the loadings absorb a scale drifting under them."""
    f = factor()
    path = f.draw(n_groups=1, horizon=HORIZON, n_draws=40_000, generator=gen())
    var = path.var(dim=0).squeeze(0)  # [horizon, rank]
    first, last = float(var[0].mean()), float(var[-1].mean())
    assert 0.94 < first < 1.06, f"step 0 variance {first}"
    assert 0.94 < last < 1.06, f"step {HORIZON-1} variance {last}"
    assert abs(last - first) < 0.06, f"variance drifted from {first} to {last} across the horizon"


def test_consecutive_steps_are_correlated_at_phi() -> None:
    """This is the temporal dependence the factor exists to supply. Without it, drawing
    independently per day and calling the result a sample path does not create the dependence a
    lead-time convolution requires."""
    f = factor()
    path = f.draw(n_groups=1, horizon=HORIZON, n_draws=40_000, generator=gen())
    a, b = path[:, 0, 10, 0], path[:, 0, 11, 0]
    corr = float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std()))
    assert abs(corr - float(f.phi[0])) < 0.03, f"lag-1 correlation {corr} against phi {f.phi[0]}"


def test_iid_draw_is_exactly_the_generators_independent_normals() -> None:
    """IID is a different configured dynamics, not AR(1) with a merely small learned phi."""
    f = factor(factor_dynamics="iid")
    got = f.draw(n_groups=2, horizon=7, n_draws=5, generator=gen(19))
    want = torch.randn(5, 2, 7, RANK, dtype=torch.float64, generator=gen(19))
    assert torch.equal(got, want)
    assert torch.equal(f.phi, torch.zeros(RANK, dtype=torch.float64))
    assert torch.equal(f.innovation_scale, torch.ones(RANK, dtype=torch.float64))


def test_iid_theta_is_state_compatible_but_cannot_reintroduce_persistence() -> None:
    """Strict checkpoint loading keeps working, while a saved AR theta is inert in IID mode."""
    ar1 = factor(factor_dynamics="ar1", init_phi=0.83)
    iid = factor(factor_dynamics="iid")
    assert ar1.state_dict().keys() == iid.state_dict().keys()
    iid.load_state_dict(ar1.state_dict(), strict=True)
    assert not iid.theta.requires_grad
    assert bool((iid.theta != 0).all()), "the compatibility load did not exercise a nonzero theta"
    assert torch.equal(iid.phi, torch.zeros(RANK, dtype=torch.float64))
    assert torch.equal(iid.innovation_scale, torch.ones(RANK, dtype=torch.float64))


def test_unknown_factor_dynamics_is_refused() -> None:
    try:
        SharedLatentFactor(rank=2, n_stores=3, factor_dynamics="random_walk")  # type: ignore[arg-type]
    except FactorError as e:
        assert "'ar1' or 'iid'" in str(e)
        return
    raise AssertionError("an unknown factor_dynamics value was accepted")


def test_stores_in_one_group_share_the_same_drawn_path() -> None:
    """The whole mechanism: cross-store dependence without a chain total. Two stores in one group
    must be correlated; the strength is whatever their loadings agree on."""
    f = factor()
    with torch.no_grad():  # identical loadings -> perfectly correlated
        f.loading_occurrence.weight[0] = f.loading_occurrence.weight[1] = torch.ones(RANK) * 0.3
    path = f.draw(n_groups=2, horizon=4, n_draws=4000, generator=gen())
    same_group = torch.tensor([0, 0])
    stores = torch.tensor([0, 1])
    occ, _ = f.contract(path, same_group, stores)
    a, b = occ[:, 0, 2], occ[:, 1, 2]
    corr = float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std()))
    assert corr > 0.99, f"two stores in one group correlate only {corr}"


def test_stores_in_different_groups_do_not() -> None:
    f = factor()
    with torch.no_grad():
        f.loading_occurrence.weight[0] = f.loading_occurrence.weight[1] = torch.ones(RANK) * 0.3
    path = f.draw(n_groups=2, horizon=4, n_draws=4000, generator=gen())
    occ, _ = f.contract(path, torch.tensor([0, 1]), torch.tensor([0, 1]))
    a, b = occ[:, 0, 2], occ[:, 1, 2]
    corr = float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std()))
    assert abs(corr) < 0.06, f"separate groups correlate at {corr}; they share no factor"


def test_the_two_loadings_are_independent() -> None:
    """A store may be more sensitive in WHETHER it sells than in HOW MUCH. One loading serving
    both heads would make them move together by construction rather than by evidence."""
    f = factor()
    with torch.no_grad():
        f.loading_occurrence.weight[0] = torch.ones(RANK)
        f.loading_size.weight[0] = torch.zeros(RANK)
    path = f.draw(n_groups=1, horizon=3, n_draws=200, generator=gen())
    occ, size = f.contract(path, torch.tensor([0]), torch.tensor([0]))
    assert float(occ.abs().mean()) > 0.1, "the occurrence loading did nothing"
    assert float(size.abs().max()) < 1e-12, "the size term moved with a zero size loading"


def test_the_draw_is_reproducible_from_a_seed() -> None:
    """A gate result that cannot be reproduced is not a gate result."""
    f = factor()
    a = f.draw(n_groups=2, horizon=5, n_draws=3, generator=gen(7))
    b = f.draw(n_groups=2, horizon=5, n_draws=3, generator=gen(7))
    assert torch.equal(a, b)
    assert not torch.equal(a, f.draw(n_groups=2, horizon=5, n_draws=3, generator=gen(8)))


def test_gradients_reach_the_persistence_through_the_path() -> None:
    """Reparameterised, so `theta` learns. A path drawn under `no_grad` would train the loadings
    against a persistence that never moved."""
    f = factor()
    path = f.draw(n_groups=1, horizon=6, n_draws=4, generator=gen())
    occ, size = f.contract(path, torch.tensor([0]), torch.tensor([0]))
    (occ.sum() + size.sum()).backward()
    assert f.theta.grad is not None and torch.isfinite(f.theta.grad).all()
    assert float(f.theta.grad.abs().sum()) > 0, "no gradient reached the persistence"
    assert f.loading_occurrence.weight.grad is not None


def test_the_contracted_shape_is_what_the_head_expects() -> None:
    f = factor()
    path = f.draw(n_groups=3, horizon=HORIZON, n_draws=5, generator=gen())
    occ, size = f.contract(path, torch.tensor([0, 1, 2, 2]), torch.tensor([0, 1, 2, 3]))
    assert occ.shape == (5, 4, HORIZON) and size.shape == (5, 4, HORIZON)


def test_rank_zero_is_refused_rather_than_silently_bottom_up() -> None:
    try:
        SharedLatentFactor(rank=0, n_stores=2)
    except FactorError as e:
        assert "plain bottom-up" in str(e)
        return
    raise AssertionError("rank 0 accepted; that is the topology this factor replaces")


def test_a_mismatched_index_is_refused() -> None:
    f = factor()
    path = f.draw(n_groups=1, horizon=3, n_draws=2, generator=gen())
    try:
        f.contract(path, torch.tensor([0, 0]), torch.tensor([0]))
    except FactorError as e:
        assert "same cells" in str(e)
        return
    raise AssertionError("contracted mismatched group and store indices")


def test_an_out_of_range_index_is_refused() -> None:
    f = factor()
    path = f.draw(n_groups=1, horizon=3, n_draws=2, generator=gen())
    for g, s, msg in (
        (torch.tensor([5]), torch.tensor([0]), "groups were drawn"),
        (torch.tensor([0]), torch.tensor([99]), "loading rows"),
    ):
        try:
            f.contract(path, g, s)
        except FactorError as e:
            assert msg in str(e)
            continue
        raise AssertionError(f"accepted an out-of-range index ({msg})")


def test_negative_indices_are_refused_instead_of_selecting_the_last_row() -> None:
    """PyTorch treats -1 as the last row; that silently rewires the dependence graph."""
    f = factor()
    path = f.draw(n_groups=2, horizon=3, n_draws=2, generator=gen())
    for group_index, store_index, label in (
        (torch.tensor([-1]), torch.tensor([0]), "group"),
        (torch.tensor([0]), torch.tensor([-1]), "store"),
    ):
        try:
            f.contract(path, group_index, store_index)
        except FactorError as exc:
            assert "non-negative" in str(exc), (label, exc)
            continue
        raise AssertionError(f"a negative {label} index selected the final row")


def test_factor_sizes_are_strict_positive_integers() -> None:
    for field, construct in (
        ("rank", lambda value: SharedLatentFactor(rank=value, n_stores=2)),
        ("n_stores", lambda value: SharedLatentFactor(rank=2, n_stores=value)),
    ):
        for value in (True, 1.5, "2"):
            try:
                construct(value)  # type: ignore[arg-type]
            except FactorError as exc:
                assert "positive integer" in str(exc), (field, value, exc)
                continue
            raise AssertionError(f"{field} accepted non-integer {value!r}")


def test_draw_sizes_are_strict_positive_integers() -> None:
    f = factor()
    for field, values in (
        ("n_groups", (True, 1.5, "1")),
        ("horizon", (True, 1.5, "1")),
        ("n_draws", (True, 1.5, "1")),
    ):
        for value in values:
            sizes = {"n_groups": 1, "horizon": 1, "n_draws": 1}
            sizes[field] = value
            try:
                f.draw(**sizes)  # type: ignore[arg-type]
            except FactorError as exc:
                assert "positive integers" in str(exc), (field, value, exc)
                continue
            raise AssertionError(f"draw accepted {field}={value!r}")


def test_contract_requires_rank_one_integer_nonempty_indices_and_a_draw_path() -> None:
    f = factor()
    path = f.draw(n_groups=2, horizon=3, n_draws=2, generator=gen())
    cases = (
        (path, torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long), "cell"),
        (path, torch.tensor([[0]]), torch.tensor([[0]]), "rank-1"),
        (path, torch.tensor([0.0]), torch.tensor([0]), "integer"),
        (path, torch.tensor([0]), torch.tensor([False]), "integer"),
        (path[0], torch.tensor([0]), torch.tensor([0]), "draw path"),
        (path.to(torch.int64), torch.tensor([0]), torch.tensor([0]), "draw path"),
        (path[..., :-1], torch.tensor([0]), torch.tensor([0]), "factor rank"),
    )
    for candidate_path, groups, stores, message in cases:
        try:
            f.contract(candidate_path, groups, stores)
        except FactorError as exc:
            assert message in str(exc), (message, exc)
            continue
        raise AssertionError(f"contract accepted malformed {message}")


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print("PASS " + name)
