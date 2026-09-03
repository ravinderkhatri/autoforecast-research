"""Tests for the encoder stack: branches, fusion, trunk, decoder.

These are structural rather than statistical -- what they check is that each piece does the thing
it was chosen FOR. The architecture picked RevIN over output normalisation, a gate over plain
attention, a distributed-lag kernel over attention, and a direct decoder over an autoregressive
one, each for a stated reason. A test that only checked shapes would pass on a stack that had
quietly lost every one of those properties.

Run: python3 forecast/models/mft/test_stack.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent.parent))

import pytest
import torch
from forecast.models.mft.branches import (
    BranchError,
    CalendarEventBranch,
    CommercialBranch,
    DemandHistoryBranch,
    RevIN,
    StaticIdentityBranch,
    _binary_mask,
)
from forecast.models.mft.decoder import Decoder, DecoderError
from forecast.models.mft.fusion import FusionError, GatedFusion
from forecast.models.mft.trunk import FiLM, RMSNorm, Trunk, TrunkError

D, ENC, HOR = 16, 28, 14


def seed(n: int = 0) -> None:
    torch.manual_seed(n)


# ---------------------------------------------------------------- RevIN


def test_revin_removes_a_2500x_scale_difference() -> None:
    """The reason it exists. Attention is a dot product, so without per-series normalisation the
    fast movers dominate the softmax and the slow movers -- most of the catalogue -- contribute
    almost nothing. The synthetic scale multiplier exercises this invariance."""
    seed()
    r = RevIN(1).double()
    slow = torch.rand(1, ENC, 1, dtype=torch.float64) * 0.2
    fast = slow * 2500.0
    a, _, _ = r(slow)
    b, _, _ = r(fast)
    assert torch.allclose(a, b, atol=1e-9), (
        f"a 2500x rescale moved the normalised input by {float((a-b).abs().max()):.3e}. An "
        "ABSOLUTE variance floor does this -- it does not scale with the series, so a quiet "
        "series is distorted more than a loud one. Measured before the fix: 0.24% on a series "
        "with variance 2.1e-3, and worse the quieter it gets."
    )


def test_revin_excludes_masked_steps_from_its_statistics() -> None:
    """A run of never-ranged days would otherwise drag the mean toward zero and make the series
    look smaller than it is."""
    seed()
    r = RevIN(1).double()
    x = torch.cat(
        [
            torch.full((1, 14, 1), 5.0, dtype=torch.float64),
            torch.zeros(1, 14, 1, dtype=torch.float64),
        ],
        dim=1,
    )
    mask = torch.cat(
        [torch.ones(1, 14, dtype=torch.bool), torch.zeros(1, 14, dtype=torch.bool)], dim=1
    )
    _, mean, _ = r(x, mask)
    assert abs(float(mean) - 5.0) < 1e-9, f"masked zeros pulled the mean to {float(mean)}"
    _, unmasked_mean, _ = r(x)
    assert (
        abs(float(unmasked_mean) - 2.5) < 1e-9
    ), "the unmasked case should differ, or the mask did nothing"


def test_revin_survives_a_series_that_never_moves() -> None:
    """Zero variance over the encoder window. Dividing by it would give inf rather than a
    sensible 'this series does not move' signal."""
    r = RevIN(1).double()
    for level in (3.0, 0.0):  # 0.0 has no scale to be relative to -- the last-resort clamp
        out, _, _ = r(torch.full((1, ENC, 1), level, dtype=torch.float64))
        assert torch.isfinite(out).all(), f"a constant series at {level} produced a non-finite"
        assert float(out.abs().max()) < 1e-6, "a series that never moves should normalise to zero"


def test_revin_refuses_the_wrong_rank() -> None:
    try:
        RevIN(1)(torch.zeros(4, 4))
    except BranchError as e:
        assert "[batch, time, channels]" in str(e)
        return
    raise AssertionError("accepted a rank it cannot interpret")


@pytest.mark.parametrize("fault", [0.0, 1.0, 0.5, float("nan")])
def test_branches_refuse_nonbinary_masks_at_every_public_mask_boundary(fault: float) -> None:
    """A fractional/NaN availability value must not silently become ``True``."""
    history = torch.ones(1, ENC)
    day_mask = torch.ones_like(history)
    day_mask[0, 0] = fault
    with pytest.raises(BranchError, match="binary"):
        RevIN(1)(history.unsqueeze(-1), day_mask)
    with pytest.raises(BranchError, match="binary"):
        DemandHistoryBranch(D, ENC)(history, day_mask)
    with pytest.raises(BranchError, match="binary"):
        DemandHistoryBranch(D, ENC)(history, censored=day_mask)

    covariates = torch.ones(1, ENC, 1)
    covariate_mask = torch.ones_like(covariates)
    covariate_mask[0, 0, 0] = fault
    with pytest.raises(BranchError, match="binary"):
        DemandHistoryBranch(D, ENC, n_covariates=1)(
            history, covariates=covariates, covariate_mask=covariate_mask
        )

    channel_mask = torch.ones(1, 2)
    channel_mask[0, 0] = fault
    with pytest.raises(BranchError, match="binary"):
        CommercialBranch(2, D)(torch.ones(1, 2), channel_mask)


def test_branches_accept_current_boolean_masks() -> None:
    history = torch.ones(1, ENC)
    mask = torch.ones_like(history, dtype=torch.bool)
    RevIN(1)(history.unsqueeze(-1), mask)
    DemandHistoryBranch(D, ENC)(history, mask, censored=torch.zeros_like(mask))
    CommercialBranch(2, D)(torch.ones(1, 2), torch.ones(1, 2, dtype=torch.bool))


def test_binary_mask_rejects_type_shape_and_device_mismatches() -> None:
    reference = torch.ones(2, 3)
    for invalid in (
        [[True] * 3] * 2,
        torch.ones(1, 3, dtype=torch.bool),
        torch.ones(2, 3, dtype=torch.bool, device="meta"),
        torch.ones(2, 3, dtype=torch.complex64),
    ):
        with pytest.raises(BranchError, match="binary 0/1 mask"):
            _binary_mask(invalid, reference, "test availability")
    numeric = torch.tensor([[0, 1, 0], [1, 0, 1]])
    with pytest.raises(BranchError, match="Boolean"):
        _binary_mask(numeric, reference, "test availability")
    boolean = numeric.bool()
    assert _binary_mask(boolean, reference, "test availability") is boolean


def test_demand_covariate_boundaries_and_masked_nonfinite_values() -> None:
    for width in (-1, 0.5, True):
        with pytest.raises(BranchError, match="n_covariates must be a nonnegative integer"):
            DemandHistoryBranch(4, 4, weekly_patch=2, n_heads=1, n_covariates=width)
    layer = DemandHistoryBranch(4, 4, weekly_patch=2, n_heads=1, n_covariates=1).eval()
    history, covariates = torch.ones(2, 4), torch.ones(2, 4, 1)
    mask = torch.ones_like(covariates, dtype=torch.bool)
    for values, present in (
        (None, mask),
        (covariates, None),
        (covariates[:1], mask),
        (covariates, mask[:1]),
    ):
        with pytest.raises(BranchError, match="declared layout"):
            layer(history, covariates=values, covariate_mask=present)
    for value in (float("nan"), float("inf")):
        bad = covariates.clone()
        bad[0, 0, 0] = value
        with pytest.raises(BranchError, match="available historical covariates must be finite"):
            layer(history, covariates=bad, covariate_mask=mask)
        masked = mask.clone()
        masked[0, 0, 0] = False
        expected = layer(history, covariates=covariates, covariate_mask=masked)
        actual = layer(history, covariates=bad, covariate_mask=masked)
        for left, right in zip(actual, expected, strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    disabled = DemandHistoryBranch(4, 4, weekly_patch=2, n_heads=1).eval()
    for values, present in ((covariates, None), (None, mask), (covariates, mask)):
        with pytest.raises(BranchError, match="covariate-free branch"):
            disabled(history, covariates=values, covariate_mask=present)


def test_demand_branch_distinguishes_absent_dates_from_real_unobserved_dates() -> None:
    """A real unobserved zero is a calendar fact; a padded zero is no date at all."""
    seed(7)
    branch = DemandHistoryBranch(
        4,
        4,
        weekly_patch=2,
        n_heads=1,
        dropout=0.0,
        position_availability=True,
    ).eval()
    history = torch.zeros(1, 4, requires_grad=True)
    observed = torch.zeros(1, 4, dtype=torch.bool)
    absent = torch.tensor([[False, False, True, True]])
    real = torch.ones_like(absent)
    absent_output = branch(history, observed, position_mask=absent)[0]
    real_output = branch(history, observed, position_mask=real)[0]
    assert not torch.equal(absent_output, real_output)
    absent_output.sum().backward()
    assert history.grad is not None and not history.grad[:, :2].any()

    changed = history.detach().clone()
    changed[:, :2] = 999
    changed.requires_grad_(True)
    changed_output = branch(changed, observed, position_mask=absent)[0]
    torch.testing.assert_close(changed_output, absent_output.detach(), rtol=0, atol=0)
    changed_output.sum().backward()
    assert changed.grad is not None and not changed.grad[:, :2].any()

    with pytest.raises(BranchError, match="requires position_mask"):
        branch(torch.zeros(1, 4), observed)
    with pytest.raises(BranchError, match="cannot be observed"):
        branch(torch.zeros(1, 4), torch.ones(1, 4, dtype=torch.bool), position_mask=absent)
    covariate_branch = DemandHistoryBranch(
        4,
        4,
        weekly_patch=2,
        n_heads=1,
        dropout=0.0,
        n_covariates=1,
        position_availability=True,
    ).eval()
    with pytest.raises(BranchError, match="cannot contain covariates"):
        covariate_branch(
            torch.zeros(1, 4),
            observed,
            covariates=torch.ones(1, 4, 1),
            covariate_mask=torch.ones(1, 4, 1, dtype=torch.bool),
            position_mask=absent,
        )


def test_deep_demand_attention_excludes_partial_week_padding() -> None:
    seed(13)
    branch = DemandHistoryBranch(
        4,
        14,
        weekly_patch=7,
        n_heads=1,
        dropout=0.0,
        attention_layers=4,
        position_availability=True,
    ).eval()
    with torch.no_grad():
        for refinement in branch.attn_refinements:
            refinement.out_proj.weight.fill_(0.1)
    available = torch.tensor([[False] * 5 + [True] * 9])
    observed = torch.zeros_like(available)
    expected_mask = ~torch.cat(
        (available, available.reshape(1, 2, 7).any(-1)), dim=1
    )
    captured = []
    hooks = [
        attention.register_forward_pre_hook(
            lambda _module, _args, kwargs: captured.append(kwargs["key_padding_mask"].clone()),
            with_kwargs=True,
        )
        for attention in (branch.attn, *branch.attn_refinements)
    ]
    baseline = branch(torch.zeros(1, 14), observed, position_mask=available)[0]
    with torch.no_grad():
        branch.daily_pos[:, :5].add_(999)
    changed = branch(torch.zeros(1, 14), observed, position_mask=available)[0]
    for hook in hooks:
        hook.remove()
    torch.testing.assert_close(changed, baseline, rtol=0, atol=1e-6)
    assert len(captured) == 8
    assert all(torch.equal(mask, expected_mask) for mask in captured)

    whole_patch_padding = torch.tensor([[False] * 7 + [True] * 7])
    _, sequence = branch(
        torch.zeros(1, 14),
        torch.zeros_like(whole_patch_padding),
        position_mask=whole_patch_padding,
    )
    assert not sequence[:, 0].any()


def test_generic_identity_tables_compose_each_role_and_reject_bad_indices() -> None:
    for cardinalities in ((-1,), (0,), (1.5,), (True,)):
        with pytest.raises(BranchError, match="positive sizes and no legacy tables"):
            StaticIdentityBranch(2, 3, identity_cardinalities=cardinalities)
    for legacy in (dict(n_series=2), dict(n_products=2, n_stores=2)):
        with pytest.raises(BranchError, match="positive sizes and no legacy tables"):
            StaticIdentityBranch(2, 3, identity_cardinalities=(2,), **legacy)
    layer = StaticIdentityBranch(2, 3, residual_dim=2, identity_cardinalities=(2, 3))
    with torch.no_grad():
        layer.residual_identities[0].weight.copy_(torch.tensor([[1., 10.], [2., 20.]]))
        layer.residual_identities[1].weight.copy_(torch.tensor([[3., 30.], [4., 40.], [5., 50.]]))
        layer.project.weight.copy_(torch.tensor([[0., 0., 1., 0.], [0., 0., 0., 1.], [0., 0., 1., 1.]]))
        layer.project.bias.zero_()
    attributes = torch.zeros(2, 2)
    actual = layer(attributes, identity_indices=torch.tensor([[0, 2], [1, 0]]))
    torch.testing.assert_close(actual, torch.tensor([[6., 60., 66.], [5., 50., 55.]]), rtol=0, atol=0)
    actual.sum().backward()
    for table in layer.residual_identities:
        assert table.weight.grad is not None and table.weight.grad.abs().sum() > 0
    assert not layer.residual_identities[1].weight.grad[1].any()
    for invalid in (
        None,
        torch.zeros(1, 2, dtype=torch.long),
        torch.zeros(2, 1, dtype=torch.long),
        torch.zeros(2, 2),
        torch.zeros(2, 2, dtype=torch.long, device="meta"),
    ):
        with pytest.raises(BranchError, match="one integral row per series and role"):
            layer(attributes, identity_indices=invalid)
    for invalid in ([[2, 0], [0, 0]], [[0, 3], [0, 0]], [[-1, 0], [0, 0]], [[0, -1], [0, 0]]):
        with pytest.raises(BranchError, match="outside its declared vocabulary"):
            layer(attributes, identity_indices=torch.tensor(invalid))
    without_tables = StaticIdentityBranch(2, 3)
    with pytest.raises(BranchError, match="without generic identity tables"):
        without_tables(attributes, identity_indices=torch.zeros(2, 0, dtype=torch.long))


# ---------------------------------------------------------------- branches


def test_the_demand_branch_returns_a_token_and_the_weekly_sequence() -> None:
    seed()
    b = DemandHistoryBranch(D, ENC)
    token, sequence = b(torch.rand(3, ENC))
    assert token.shape == (3, D)
    assert sequence.shape == (3, ENC // 7, D), "the trunk's time axis is the weekly sequence"
    mask = torch.ones(3, ENC, dtype=torch.bool)
    mask[:, :7] = False
    token, sequence = b(torch.rand(3, ENC), mask)
    assert torch.isfinite(token).all() and torch.isfinite(sequence).all()


def test_masked_days_cannot_reach_the_demand_token() -> None:
    """The leak this replaces, measured: RevIN excluded masked days from its STATISTICS but the
    values still passed through as (0 - mean)/std, and both the weekly summary and any partially
    observed patch read them. Perturbing ONLY masked days moved the output by four orders of
    magnitude. A masked day must now be invisible: same mask, different masked values, same
    output -- and the same for the sequence the trunk consumes."""
    b = DemandHistoryBranch(D, ENC, dropout=0.0).eval()
    hist = torch.zeros(2, ENC)
    hist[:, 14:] = 5.0
    mask = torch.zeros(2, ENC, dtype=torch.bool)
    mask[:, 14:] = True
    base_token, base_seq = b(hist, mask)
    perturbed = hist.clone()
    perturbed[:, :14] = 999.0  # only masked (pre-launch) days
    pert_token, pert_seq = b(perturbed, mask)
    assert torch.allclose(
        base_token, pert_token, atol=1e-5
    ), "masked-day values reached the demand token"
    assert torch.allclose(
        base_seq, pert_seq, atol=1e-5
    ), "masked-day values reached the weekly sequence"


def test_the_censored_flag_is_an_input_the_branch_responds_to() -> None:
    """The encoder-side half of the censoring contract: 'sold 3 then ran out' and 'sold 3' are
    different observations, and before this channel existed they were the same input."""
    b = DemandHistoryBranch(D, ENC, dropout=0.0).eval()
    hist = torch.rand(2, ENC) * 3
    flag = torch.zeros(2, ENC, dtype=torch.bool)
    flag[1, -3:] = True
    plain, _ = b(hist)
    flagged, _ = b(hist, censored=flag)
    assert torch.allclose(plain[0], flagged[0], atol=1e-6), "an unflagged row moved"
    assert not torch.allclose(plain[1], flagged[1]), "the censored flag changed nothing"


def test_exact_zero_oos_zero_and_unobserved_are_three_distinct_inputs() -> None:
    """All three store 0.0, but their information content is different.

    Exact zero says demand was observed and absent; lower-bound zero says the shelf was OOS and
    demand is unknown; unobserved says neither demand nor shelf state is known. Both the pooled
    branch token and the weekly sequence consumed by the trunk must retain those distinctions.
    """
    seed(8)
    b = DemandHistoryBranch(D, ENC, dropout=0.0).eval()
    history = torch.zeros(1, ENC)
    exact = torch.ones(1, ENC, dtype=torch.bool)
    unknown = torch.zeros(1, ENC, dtype=torch.bool)
    oos = torch.zeros(1, ENC, dtype=torch.bool)
    oos[:, -1] = True
    exact_token, exact_seq = b(history, exact)
    unknown_token, unknown_seq = b(history, unknown)
    oos_token, oos_seq = b(history, unknown, censored=oos)
    assert not torch.allclose(exact_token, unknown_token)
    assert not torch.allclose(exact_seq, unknown_seq)
    assert not torch.allclose(
        oos_token, unknown_token
    ), "a lower-bound zero was masked into the same token as no observation"
    assert not torch.allclose(
        oos_seq, unknown_seq
    ), "the trunk sequence lost the lower-bound-zero state"


def test_the_day_of_an_oos_event_is_represented() -> None:
    """Daily attention needs position: OOS yesterday and OOS four weeks ago are not synonyms."""
    seed(9)
    b = DemandHistoryBranch(D, ENC, dropout=0.0).eval()
    history = torch.zeros(1, ENC)
    magnitude = torch.zeros(1, ENC, dtype=torch.bool)
    early = torch.zeros(1, ENC, dtype=torch.bool)
    early[:, 1] = True
    late = torch.zeros(1, ENC, dtype=torch.bool)
    late[:, -2] = True
    early_token, early_seq = b(history, magnitude, censored=early)
    late_token, late_seq = b(history, magnitude, censored=late)
    assert not torch.allclose(early_token, late_token), "daily OOS position was pooled away"
    assert not torch.allclose(early_seq, late_seq), "weekly state inputs lost OOS timing"


def test_the_weekly_sequence_carries_position() -> None:
    """Attention is permutation-invariant, so without positions in the tokens, the same weeks in
    a different order are the same input and a trend has nowhere to live. Probe: roll the
    history by exactly one patch. Content-only tokens would satisfy seq(rolled) == roll(seq);
    positional tokens must not."""
    seed()
    b = DemandHistoryBranch(D, ENC, dropout=0.0).eval()
    hist = torch.rand(1, ENC)
    _, seq = b(hist)
    _, seq_rolled = b(torch.roll(hist, shifts=7, dims=1))
    assert not torch.allclose(
        seq_rolled, torch.roll(seq, shifts=1, dims=1), atol=1e-4
    ), "the sequence is position-free; the same weeks in a different order look identical"


def test_the_demand_branch_refuses_a_ragged_final_patch() -> None:
    try:
        DemandHistoryBranch(D, encoder_days=30, weekly_patch=7)
    except BranchError as e:
        assert "ragged final patch" in str(e)
        return
    raise AssertionError("a final patch covering fewer days than the others was accepted")


def test_the_factorised_residual_gives_a_new_pair_a_vector_built_from_its_parts() -> None:
    """The point of factorisation: a (product, store) pair that has never occurred still has an
    identity, composed from two tables the model already trained on OTHER pairs. A per-series
    table on a product x store panel is the joint lookup `allow_joint_embedding=False` forbids
    -- no row for a new pair, and an assortment change is precisely a new pair."""
    b = StaticIdentityBranch(3, D, n_products=4, n_stores=2, residual_dim=8)
    attrs = torch.rand(1, 3)
    seen = b(attrs, product_index=torch.tensor([2]), store_index=torch.tensor([0]))
    new_pair = b(attrs, product_index=torch.tensor([2]), store_index=torch.tensor([1]))
    assert torch.isfinite(new_pair).all() and not torch.allclose(
        seen, new_pair
    ), "an unseen (product, store) pair must get its own composed identity"
    with torch.no_grad():
        b.residual_product.weight[2] += 1.0  # what the product learned elsewhere
    moved = b(attrs, product_index=torch.tensor([2]), store_index=torch.tensor([1]))
    assert not torch.allclose(
        new_pair, moved
    ), "the product's learned row does not reach a pair it was never trained on"
    try:
        b(attrs, product_index=torch.tensor([2]))
    except BranchError:
        pass
    else:
        raise AssertionError("a factorised residual composed with a missing store index")
    try:
        StaticIdentityBranch(3, D, n_products=4, n_stores=0)
    except BranchError:
        pass
    else:
        raise AssertionError("a factorised residual was built with only one of its two tables")
    try:
        StaticIdentityBranch(3, D, n_products=4, n_stores=2, n_series=8)
    except BranchError:
        pass
    else:
        raise AssertionError("a per-series table was allowed beside the factorised one")


def test_the_text_embedding_is_an_input_the_identity_branch_reads() -> None:
    """Identity source 1 of three: a frozen description embedding, so a new product lands near
    similar ones before it sells anything. The branch accepted a `text` argument for months and
    nothing anywhere produced one -- the mechanism existed as a parameter default."""
    b = StaticIdentityBranch(3, D, text_dim=6)
    attrs = torch.rand(2, 3)
    a = b(attrs, text=torch.zeros(2, 6))
    c = b(attrs, text=torch.randn(2, 6))
    assert not torch.allclose(a, c), "the text input changed nothing"


def test_the_event_response_can_vary_by_series_and_starts_pooled() -> None:
    """`event_pooling='partial'`: a per-event response shared across the catalogue PLUS a
    weight-decayed per-(SKU, event) deviation. The deviation half was never built, so pooling
    only ever ran across event TYPES. Zero-initialised: an untrained model is exactly pooled."""
    b = CalendarEventBranch(2, D, n_series=5)
    with torch.no_grad():
        b.shared.fill_(0.5)
    dist = torch.zeros(2, 2)
    idx = torch.tensor([0, 3])
    pooled = b(dist, series_index=idx)
    assert torch.allclose(pooled[0], pooled[1]), "zero-init deviation must start pooled"
    with torch.no_grad():
        b.series_deviation.weight[3] = 2.0
    varied = b(dist, series_index=idx)
    assert not torch.allclose(varied[0], varied[1]), "the per-series amplitude changed nothing"
    # Index and table must travel together, in both directions.
    try:
        b(dist)
    except BranchError:
        pass
    else:
        raise AssertionError("a per-(series, event) table was silently skipped")
    plain = CalendarEventBranch(2, D)
    try:
        plain(dist, series_index=idx)
    except BranchError:
        pass
    else:
        raise AssertionError("an index was accepted by a branch with no deviation table")


def test_the_event_kernel_evaluates_each_horizon_distance() -> None:
    b = CalendarEventBranch(1, D, n_series=2)
    with torch.no_grad():
        b.shared.copy_(torch.linspace(-1.0, 1.0, b.n_basis))
    distance = torch.tensor([[[-3.0], [0.0], [3.0]], [[-2.0], [1.0], [4.0]]])
    present = torch.ones_like(distance, dtype=torch.bool)
    got = b(distance, present, series_index=torch.tensor([0, 1]))
    assert got.shape == (2, 3, D)
    for step in range(3):
        want = b(distance[:, step], present[:, step], series_index=torch.tensor([0, 1]))
        assert torch.allclose(got[:, step], want)


@pytest.mark.parametrize("fault", [0.0, 1.0, 0.5, float("nan")])
def test_event_and_fusion_refuse_nonbinary_masks(fault: float) -> None:
    event_mask = torch.ones(1, 2)
    event_mask[0, 0] = fault
    with pytest.raises(BranchError, match="binary"):
        CalendarEventBranch(2, D)(torch.zeros(1, 2), event_mask)

    branch_mask = torch.ones(1, 2)
    branch_mask[0, 0] = fault
    with pytest.raises(FusionError, match="binary"):
        GatedFusion(2, D)(torch.ones(1, 2, D), branch_mask)


def test_the_event_branch_refuses_a_broadcastable_wrong_event_width() -> None:
    """A one-event tensor broadcasts against two learned kernels and looks numerically valid.

    That is not a convenience: it assigns one event's distance to two different event identities.
    Both supported ranks must bind their last dimension to the configured event vocabulary.
    """
    branch = CalendarEventBranch(2, D)
    for malformed in (torch.zeros(1, 1), torch.zeros(1, 3, 1)):
        try:
            branch(malformed)
        except BranchError as error:
            assert "expected 2 event channels" in str(error)
            continue
        raise AssertionError(f"broadcast a malformed event tensor {tuple(malformed.shape)}")


def test_a_missing_commercial_channel_is_masked_not_zero_filled() -> None:
    """Zero is a legitimate discount depth, so filling an absent price with zeros teaches 'this
    product is free' rather than 'we do not know'."""
    seed()
    b = CommercialBranch(3, D).double()
    values = torch.tensor([[1.0, 0.5, 0.0]], dtype=torch.float64)
    masked = b(values, torch.tensor([[True, True, False]]))
    # Changing the masked channel must not change the output.
    other = torch.tensor([[1.0, 0.5, 99.0]], dtype=torch.float64)
    assert torch.allclose(masked, b(other, torch.tensor([[True, True, False]])), atol=1e-9)


def test_the_commercial_branch_refuses_a_row_with_no_channels() -> None:
    b = CommercialBranch(2, D)
    try:
        b(torch.zeros(1, 2), torch.tensor([[False, False]]))
    except BranchError as e:
        assert "undefined and returns NaN" in str(e)
        return
    raise AssertionError("attended over an empty key set")


def test_the_commercial_branch_refuses_a_reordered_channel_set() -> None:
    try:
        CommercialBranch(3, D)(torch.zeros(1, 2))
    except BranchError as e:
        assert "wrong column" in str(e)
        return
    raise AssertionError("a different channel count was accepted")


def test_the_event_kernel_is_exactly_zero_outside_its_support() -> None:
    """The property attention could not give. Softmax weights never reach zero, so an event would
    contribute something at every lag however far away."""
    seed()
    b = CalendarEventBranch(2, D, max_support=30).double()
    with torch.no_grad():
        b.shared.fill_(1.0)
    far = b(torch.tensor([[100.0, -100.0]], dtype=torch.float64))
    assert float(far.abs().max()) < 1e-9 or torch.allclose(far, b.project.bias, atol=1e-9)


def test_kernel_weights_shrink_the_per_event_deviation_toward_the_shared_shape() -> None:
    """The partial pooling that makes an N=2 event estimable. A product has seen about two
    Easters; without shrinkage the per-event deviation just memorises them."""
    b = CalendarEventBranch(3, D, n_basis=8, shrink=0.1).double()
    with torch.no_grad():
        b.shared.fill_(1.0)
        b.deviation.fill_(10.0)
    w = b.kernel_weights()
    assert torch.allclose(
        w, torch.full_like(w, 2.0), atol=1e-9
    ), f"expected 1 + 0.1*10 = 2 everywhere, got {w[0, 0]}"


def test_the_static_branch_refuses_a_residual_with_no_series_index() -> None:
    b = StaticIdentityBranch(4, D, n_series=10)
    try:
        b(torch.zeros(2, 4))
    except BranchError as e:
        assert "same residual" in str(e)
        return
    raise AssertionError("every series would have received row 0's residual")


def test_the_static_branch_composes_attributes_text_and_residual() -> None:
    seed()
    b = StaticIdentityBranch(4, D, text_dim=6, residual_dim=3, n_series=5)
    out = b(torch.zeros(2, 4), torch.zeros(2, 6), torch.tensor([0, 1]))
    assert out.shape == (2, D)


# ---------------------------------------------------------------- fusion


def test_the_gate_can_switch_a_branch_off_which_softmax_cannot() -> None:
    """The argument for AoA over plain attention. Softmax weights sum to one, so a branch with
    nothing to say still receives a share; a multiplicative gate can reach zero."""
    seed()
    f = GatedFusion(3, D).double()
    with torch.no_grad():
        f.gate[0].weight.zero_()
        f.gate[0].bias.fill_(-40.0)  # sigmoid(-40) is 0 to double precision
    out = f(torch.randn(2, 3, D, dtype=torch.float64))
    assert (
        float(out.abs().max()) < 1e-6
    ), f"a fully closed gate still passed {float(out.abs().max())}"


def test_a_masked_branch_contributes_nothing() -> None:
    seed()
    f = GatedFusion(3, D).double()
    tokens = torch.randn(1, 3, D, dtype=torch.float64)
    mask = torch.tensor([[True, True, False]])
    a = f(tokens, mask)
    other = tokens.clone()
    other[:, 2, :] = 999.0
    b = f(other, mask)
    assert torch.allclose(a, b, atol=1e-9), "a masked branch changed the fused output"


def test_fusion_refuses_a_row_with_every_branch_masked() -> None:
    f = GatedFusion(2, D)
    try:
        f(torch.randn(1, 2, D), torch.tensor([[False, False]]))
    except FusionError as e:
        assert "refused at the data contract" in str(e)
        return
    raise AssertionError("fused an empty branch set")


def test_fusion_refuses_the_wrong_branch_count() -> None:
    try:
        GatedFusion(5, D)(torch.randn(1, 3, D))
    except FusionError as e:
        assert "expected [batch, 5" in str(e)
        return
    raise AssertionError("a different branch count was accepted")


# ---------------------------------------------------------------- trunk


def test_the_trunk_preserves_its_shape_through_both_axes() -> None:
    seed()
    tr = Trunk(D, d_static=5)
    x = torch.randn(2, 4, 9, D)
    assert tr(x, torch.randn(2, 5)).shape == (2, 4, 9, D)


def test_trunk_masks_global_padding_from_outputs_and_gradients() -> None:
    seed()
    trunk = Trunk(4, d_static=0, n_stages=1, n_heads=1).eval()
    mask = torch.tensor([[False, False, True, True]])
    # Reuse the same real suffix; only the globally padded payload may change.
    x = torch.full((1, 2, 4, 4), 999.0)
    x[:, :, 2:] = torch.randn(1, 2, 2, 4)
    # Compare against an explicit zero-prefix run of this exact real suffix.
    zero = x.clone()
    zero[:, :, :2] = 0
    zero.requires_grad_(True)
    expected = trunk(zero, torch.empty(1, 0), time_mask=mask)
    expected.sum().backward()
    x.requires_grad_(True)
    actual = trunk(x, torch.empty(1, 0), time_mask=mask)
    actual.sum().backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not zero.grad[:, :, :2].any() and not x.grad[:, :, :2].any()
    assert not actual[:, :, :2].any()


@pytest.mark.parametrize("depth", [1, 4])
def test_trunk_masked_prefix_matches_running_only_the_real_suffix(depth: int) -> None:
    seed(11)
    trunk = Trunk(4, d_static=0, n_stages=depth, n_heads=1).eval()
    suffix = torch.randn(1, 2, 2, 4)
    long = torch.cat((torch.randn(1, 2, 2, 4), suffix), dim=2).requires_grad_(True)
    mask = torch.tensor([[False, False, True, True]])
    actual = trunk(long, torch.empty(1, 0), time_mask=mask)[:, :, 2:]
    actual.sum().backward()
    long_gradient = long.grad.detach().clone()
    trunk.zero_grad(set_to_none=True)

    short = suffix.detach().clone().requires_grad_(True)
    expected = trunk(short, torch.empty(1, 0), time_mask=torch.ones(1, 2, dtype=torch.bool))
    expected.sum().backward()
    torch.testing.assert_close(actual.detach(), expected.detach(), rtol=0, atol=1e-6)
    torch.testing.assert_close(long_gradient[:, :, 2:], short.grad, rtol=0, atol=1e-6)
    assert not long_gradient[:, :, :2].any()


def test_film_is_the_identity_at_initialisation() -> None:
    """gamma = 1 + delta with delta at zero. Initialising gamma at zero instead would switch the
    trunk off for the first epochs -- and the loss would still fall, because the head's baselines
    carry the prediction on their own."""
    f = FiLM(5, D).double()
    x = torch.randn(2, 7, D, dtype=torch.float64)
    assert torch.allclose(f(x, torch.randn(2, 5, dtype=torch.float64)), x, atol=1e-12)


def test_rmsnorm_scales_without_centring() -> None:
    n = RMSNorm(D).double()
    x = torch.randn(2, D, dtype=torch.float64) + 10.0
    out = n(x)
    assert abs(float(out.pow(2).mean().sqrt()) - 1.0) < 0.05
    assert float(out.mean().abs()) > 0.1, "the mean was removed; that is LayerNorm, not RMSNorm"


def test_the_trunk_refuses_the_wrong_rank() -> None:
    try:
        Trunk(D, d_static=5)(torch.randn(2, 9, D), torch.randn(2, 5))
    except TrunkError as e:
        assert "silently" in str(e)
        return
    raise AssertionError("a rank that transposes the wrong axes was accepted")


# ---------------------------------------------------------------- decoder


def test_the_decoder_produces_one_representation_per_horizon_step() -> None:
    seed()
    d = Decoder(D, n_known=4)
    out = d(torch.randn(2, 20, D), torch.randn(2, HOR, 4))
    assert out.shape == (2, HOR, D)


def test_decoder_cross_attention_ignores_global_padding() -> None:
    seed()
    decoder = Decoder(4, n_known=1, n_layers=4, n_heads=1, max_horizon=2).eval()
    known = torch.ones(1, 2, 1)
    mask = torch.tensor([[False, False, True, True]])
    real = torch.randn(1, 2, 4)

    def run(prefix: float):
        memory = torch.cat((torch.full((1, 2, 4), prefix), real), dim=1).requires_grad_(True)
        output = decoder(memory, known, memory_mask=mask)
        output.sum().backward()
        return output.detach(), memory.grad.detach()

    zero, zero_gradient = run(0.0)
    actual, actual_gradient = run(999.0)
    decoder.zero_grad(set_to_none=True)
    suffix = real.detach().clone().requires_grad_(True)
    expected = decoder(suffix, known, memory_mask=torch.ones(1, 2, dtype=torch.bool))
    expected.sum().backward()
    torch.testing.assert_close(actual, zero, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected.detach(), rtol=0, atol=1e-6)
    torch.testing.assert_close(actual_gradient[:, 2:], suffix.grad, rtol=0, atol=1e-6)
    assert not zero_gradient[:, :2].any() and not actual_gradient[:, :2].any()


def test_every_horizon_step_sees_every_other_step() -> None:
    """No causal mask, deliberately: this is a DIRECT multi-horizon decoder, so step 28 is
    produced from the encoder and the known future, never from step 27's prediction. An
    autoregressive decoder would consume its own output."""
    seed()
    d = Decoder(D, n_known=2).double()
    memory = torch.randn(1, 10, D, dtype=torch.float64)
    known = torch.zeros(1, 6, 2, dtype=torch.float64)
    base = d(memory, known)
    perturbed = known.clone()
    perturbed[0, 5, :] = 5.0  # change the LAST step
    out = d(memory, perturbed)
    assert (
        float((out[0, 0] - base[0, 0]).abs().max()) > 1e-6
    ), "changing the last horizon step did not affect the first, so the decoder is causal"


def test_a_horizon_past_the_trained_step_embeddings_is_refused() -> None:
    d = Decoder(D, n_known=2, max_horizon=10)
    try:
        d(torch.randn(1, 5, D), torch.randn(1, 11, 2))
    except DecoderError as e:
        assert "produced from noise" in str(e)
        return
    raise AssertionError("forecast past the step embeddings that were trained")


def test_gradients_flow_through_the_whole_stack() -> None:
    """Branches to fusion to trunk to decoder, once, so a shape error anywhere is caught here
    rather than three hours into a training run."""
    seed()
    dem = DemandHistoryBranch(D, ENC)
    com = CommercialBranch(3, D)
    cal = CalendarEventBranch(2, D)
    sta = StaticIdentityBranch(4, D)
    fuse = GatedFusion(4, D)
    trunk = Trunk(D, d_static=4)
    dec = Decoder(D, n_known=3)

    hist = torch.rand(2, ENC, requires_grad=True)
    demand_token, demand_seq = dem(hist)
    tokens = torch.stack(
        [demand_token, com(torch.rand(2, 3)), cal(torch.randn(2, 2) * 10), sta(torch.rand(2, 4))],
        dim=1,
    )
    fused = fuse(tokens)  # [2, 4, D]
    t_enc = demand_seq.shape[1]
    carried = fused.unsqueeze(2).expand(2, 4, t_enc, D).contiguous()
    carried[:, 0] = carried[:, 0] + demand_seq  # the trunk's real time axis
    out = trunk(carried, torch.rand(2, 4))
    memory = out.reshape(2, 4 * t_enc, D)
    loss = dec(memory, torch.rand(2, HOR, 3)).sum()
    loss.backward()
    assert hist.grad is not None and torch.isfinite(hist.grad).all()


def test_the_trunk_time_axis_is_not_inert() -> None:
    """The defect this wiring replaces, verified at the time: the fused tokens were tiled across
    the horizon, attention over identical copies is a fixed average, and the trunk's output was
    bit-identical at every 'time' step -- the time half of every stage was dead arithmetic. With
    the demand sequence on the time axis, the trunk's output must VARY across time, and changing
    one week of history must move that week's trunk column differently from the others."""
    seed()
    dem = DemandHistoryBranch(D, ENC, dropout=0.0).eval()
    sta = StaticIdentityBranch(4, D)
    fuse = GatedFusion(2, D)
    trunk = Trunk(D, d_static=4)
    hist = torch.rand(2, ENC)
    static = torch.rand(2, 4)
    token, seq = dem(hist)
    fused = fuse(torch.stack([token, sta(static)], dim=1))
    carried = fused.unsqueeze(2).expand(2, 2, seq.shape[1], D).contiguous()
    carried[:, 0] = carried[:, 0] + seq
    out = trunk(carried, static)
    spread = (out - out[:, :, :1, :]).abs().max()
    assert (
        float(spread) > 1e-4
    ), f"trunk output is constant across time (spread {float(spread):.2e}); the time axis is inert"


def test_a_series_with_no_observed_day_does_not_produce_nan() -> None:
    """A training window reaching back before a product's launch has EVERY encoder day
    unobserved. The key-padding mask is then all True, attention has no keys to attend over, and
    the result is NaN -- which the IWAE bound's logsumexp spreads across the entire batch.

    Measured on real M5 at a 2014 origin: 3,101 of 30,490 series were in exactly this state, and
    the ladder's third rung reported `val nan` on its first epoch.

    RevIN normalises an all-unobserved window to zeros, so the row is unmasked and attends over
    those zeros. That is the branch's "nothing is known here" output, which is the honest answer
    for a product that did not exist yet."""
    b = DemandHistoryBranch(16, 28)
    history = torch.zeros(3, 28)
    observed = torch.ones(3, 28, dtype=torch.bool)
    observed[1] = False  # the pre-launch row
    out, _ = b(history, observed)
    assert not bool(torch.isnan(out).any()), "an all-unobserved row still produces NaN"
    assert out.shape == (3, 16)
    # And it must still carry gradient, or the batch silently trains on two rows out of three.
    history.requires_grad_(True)
    token, _ = b(history, observed)
    token.sum().backward()
    assert b.query.grad is not None and float(b.query.grad.abs().sum()) > 0


def test_an_all_unobserved_batch_is_still_finite() -> None:
    """The degenerate case: every row pre-launch. Nothing to attend over anywhere."""
    b = DemandHistoryBranch(16, 28)
    out, sequence = b(torch.zeros(4, 28), torch.zeros(4, 28, dtype=torch.bool))
    assert torch.isfinite(out).all(), "a wholly unobserved batch produced non-finite output"
    assert torch.isfinite(sequence).all()


def test_the_kernel_gates_each_event_by_its_own_window() -> None:
    """Per-event hard zero: the same distance can be inside one event's window and outside
    another's, and the branch must not blur them through the shared basis."""
    b = CalendarEventBranch(2, D, max_support=30, event_windows=((-2, 2), (-10, 10)))
    with torch.no_grad():
        b.shared.fill_(1.0)
    d = torch.tensor([[5.0, 5.0]])  # +5 days from both events
    zero_in = b(torch.zeros(1, 2))
    at_five = b(d)
    narrow_moved = not torch.allclose(zero_in, at_five)
    assert narrow_moved  # event B (window 10) contributes at +5
    only_narrow = CalendarEventBranch(1, D, max_support=30, event_windows=((-2, 2),))
    with torch.no_grad():
        only_narrow.shared.fill_(1.0)
    inside = only_narrow(torch.tensor([[1.0]]))
    outside = only_narrow(torch.tensor([[5.0]]))
    baseline = only_narrow(torch.tensor([[100.0]]))  # far outside: pure projection bias
    assert torch.allclose(outside, baseline), "outside its own window must contribute nothing"
    assert not torch.allclose(inside, baseline)
    try:
        CalendarEventBranch(2, D, event_windows=((-2, 2),))
    except BranchError:
        pass
    else:
        raise AssertionError("a positional window mismatch gates one event with another's window")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
