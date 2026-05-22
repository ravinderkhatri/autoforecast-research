"""Tests for the prospective scenario contrast.

Almost every case here is a REFUSAL, and that is the shape of the module: the number itself is
easy to compute and easy to compute in a way that means something other than it appears to. What
the tests defend is the four things that must ship with it -- a pre-treatment origin, a bundle
rather than a lever, a spread from resampled campaigns, and an overlap check.

STRUCTURAL by necessity: there is no real instance of a post-treatment origin or
a one-campaign bootstrap sitting in M5 waiting to be found. The model here is a stub because what
is under test is what the module refuses, not what any forecaster computes.

Run: python3 forecast/scenario/test_contrast.py
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import torch
from forecast.scenario.contrast import (
    Contrast,
    ContrastError,
    InterventionBundle,
    campaign_bootstrap,
    contrast,
)

CELLS, HORIZON, KNOWN = 40, 7, 3
REPRODUCTION = {
    "origin": "2016-05-01",
    "campaign_start": "2016-06-01",
    "n_bootstrap": 200,
    "seed": 0,
    "alpha": 0.1,
    "observed_driver_range": (0.0, 0.5),
    "planned_driver_range": (0.0, 0.3),
    "horizon": HORIZON,
    "window_bounds": (1, HORIZON),
    "requested_window_bounds": None,
}


def model(known_future: torch.Tensor, draws: torch.Tensor) -> torch.Tensor:
    """Demand rises with column 0 (the discount) and with the shared draw. Per cell."""
    return (2.0 + 4.0 * known_future[..., 0].mean(dim=1) + draws).clamp_min(0.0)


def setup(planned: float = 0.3, spread: float = 0.0):
    """`spread` makes campaigns DIFFER from each other. Without it every campaign produces the
    same effect and the bootstrap interval collapses -- which the module now flags, because an
    interval of width 1e-6 reads as extraordinary precision and is its absence."""
    kf = torch.zeros(CELLS, HORIZON, KNOWN)
    kf[..., 0] = planned
    campaigns = torch.arange(CELLS) // 5          # 8 campaigns of 5 cells
    if spread:
        kf[..., 0] += ((campaigns.float() - 3.5) * spread / 3.5).unsqueeze(1)
    draws = torch.randn(CELLS, generator=torch.Generator().manual_seed(0)) * 0.1
    return kf, draws, campaigns


def test_a_post_treatment_origin_is_refused() -> None:
    """Re-running from inside the campaign puts the sales the campaign caused into the encoder,
    so the counterfactual has already seen the effect it is meant to remove -- and the contrast
    understates it SILENTLY."""
    kf, draws, camp = setup()
    try:
        contrast(model, kf, InterventionBundle("promo", (0,)), draws,
                 origin="2016-06-10", campaign_start="2016-06-01", campaign_index=camp,
                 observed_driver_range=(0.0, 0.5), planned_driver_range=(0.0, 0.3))
    except ContrastError as e:
        assert "understates it silently" in str(e); return
    raise AssertionError("computed a contrast from inside the campaign")


def test_a_bundle_must_control_more_than_nothing() -> None:
    try:
        InterventionBundle("empty", ())
    except ContrastError as e:
        assert "controls no columns" in str(e); return
    raise AssertionError("built a bundle that changes nothing")


def test_the_bundle_sets_every_column_together() -> None:
    """'Promo off with the discount left in place' is a world that never existed. Price, display
    and media move together or the contrast describes nothing real."""
    kf, draws, camp = setup()
    kf[..., 1] = 1.0                                   # display on
    out = contrast(model, kf, InterventionBundle("price_display", (0, 1)), draws,
                   origin="2016-05-01", campaign_start="2016-06-01", campaign_index=camp,
                   observed_driver_range=(0.0, 0.5), planned_driver_range=(0.0, 0.3))
    single = contrast(model, kf, InterventionBundle("price_only", (0,)), draws,
                      origin="2016-05-01", campaign_start="2016-06-01", campaign_index=camp,
                      observed_driver_range=(0.0, 0.5), planned_driver_range=(0.0, 0.3))
    assert out.bundle == "price_display" and single.bundle == "price_only"
    assert out.effect != 0.0


def test_the_spread_comes_from_resampling_campaigns_not_cells() -> None:
    """A promotion spans days and stores, so cells within a campaign are anything but
    independent. Resampling cells gives an interval that shrinks with the LENGTH of the campaign
    rather than with the amount of evidence."""
    per_cell = torch.arange(40, dtype=torch.float64)
    camp = torch.arange(40) // 5
    lo_c, hi_c = campaign_bootstrap(per_cell, camp, n_bootstrap=300, seed=1)
    lo_i, hi_i = campaign_bootstrap(per_cell, torch.arange(40), n_bootstrap=300, seed=1)
    assert (hi_c - lo_c) > (hi_i - lo_i), \
        "resampling 8 campaigns gave a NARROWER interval than resampling 40 cells"


def test_a_single_campaign_bootstrap_is_refused() -> None:
    """It returns that campaign's own estimate with zero width, which reads as certainty and is
    the opposite of what the interval is for."""
    kf, draws, _ = setup()
    try:
        contrast(model, kf, InterventionBundle("promo", (0,)), draws,
                 origin="2016-05-01", campaign_start="2016-06-01",
                 campaign_index=torch.zeros(CELLS, dtype=torch.long),
                 observed_driver_range=(0.0, 0.5), planned_driver_range=(0.0, 0.3))
    except ContrastError as e:
        assert "reads as certainty" in str(e); return
    raise AssertionError("produced a zero-width interval from one campaign")


def test_extrapolation_beyond_the_observed_range_is_flagged_not_hidden() -> None:
    """A client asking about a 30% discount when nothing above 15% has run is asking the model to
    invent, and the answer looks identical to one built on evidence."""
    kf, draws, camp = setup(planned=0.30)
    out = contrast(model, kf, InterventionBundle("promo", (0,)), draws,
                   origin="2016-05-01", campaign_start="2016-06-01", campaign_index=camp,
                   observed_driver_range=(0.0, 0.05), planned_driver_range=(0.0, 0.30))
    assert not out.is_quotable
    assert any("extrapolating" in n for n in out.notes)


def test_a_well_supported_contrast_is_quotable() -> None:
    kf, draws, camp = setup(planned=0.1, spread=0.04)
    out = contrast(model, kf, InterventionBundle("promo", (0,)), draws,
                   origin="2016-05-01", campaign_start="2016-06-01", campaign_index=camp,
                   observed_driver_range=(0.0, 0.5), planned_driver_range=(0.0, 0.15),
                   episode_validated=True)
    assert out.is_quotable, f"a well-supported contrast was marked unquotable: {out.notes}"
    assert out.n_campaigns == 8
    assert "NOT QUOTABLE" not in str(out)


def test_campaigns_that_are_all_alike_produce_a_flagged_degenerate_interval() -> None:
    """Found by a test that expected an ordinary interval and got [16.0000041, 16.0000050]. Every
    campaign ran the same thing, so resampling them measures nothing -- and a width of 1e-6 reads
    as extraordinary precision rather than as its absence."""
    kf, draws, camp = setup(planned=0.1, spread=0.0)
    out = contrast(model, kf, InterventionBundle("promo", (0,)), draws,
                   origin="2016-05-01", campaign_start="2016-06-01", campaign_index=camp,
                   observed_driver_range=(0.0, 0.5), planned_driver_range=(0.0, 0.1))
    assert not out.is_quotable
    assert any("indistinguishable" in n for n in out.notes), out.notes


def test_too_few_campaigns_is_flagged_because_count_is_not_identification() -> None:
    """Ten campaigns that all ran the same discount on the same weeks identify nothing about a
    different discount, and the model will happily extrapolate."""
    kf = torch.zeros(15, HORIZON, KNOWN); kf[..., 0] = 0.1
    draws = torch.zeros(15)
    out = contrast(model, kf, InterventionBundle("promo", (0,)), draws,
                   origin="2016-05-01", campaign_start="2016-06-01",
                   campaign_index=torch.arange(15) // 5,          # 3 campaigns
                   observed_driver_range=(0.0, 0.5), planned_driver_range=(0.0, 0.1))
    assert not out.is_quotable
    assert any("not identification" in n for n in out.notes)


def test_the_estimand_is_named_in_the_record_so_it_cannot_be_read_as_causal() -> None:
    """Retailers promote what is already selling, so promotion is confounded with the demand
    state and no re-run of a fitted model undoes that."""
    kf, draws, camp = setup(planned=0.1, spread=0.04)
    rec = contrast(model, kf, InterventionBundle("promo", (0,)), draws,
                   origin="2016-05-01", campaign_start="2016-06-01", campaign_index=camp,
                   observed_driver_range=(0.0, 0.5),
                   planned_driver_range=(0.0, 0.15), n_bootstrap=321, seed=17, alpha=0.2,
                   episode_validated=True).as_record()
    assert rec["estimand"] == "prospective scenario contrast, NOT a causal effect"
    assert rec["origin"] == "2016-05-01" and rec["campaign_start"] == "2016-06-01"
    assert rec["n_bootstrap"] == 321 and rec["seed"] == 17 and rec["alpha"] == 0.2
    assert rec["observed_driver_range"] == [0.0, 0.5]
    assert rec["planned_driver_range"] == [0.0, 0.15]
    assert rec["horizon"] == HORIZON


def test_an_effect_outside_its_own_interval_is_refused() -> None:
    try:
        Contrast(bundle="b", effect=10.0, low=1.0, high=2.0, n_campaigns=8, overlap=1.0,
                 origin_is_pre_treatment=True, **REPRODUCTION)
    except ContrastError as e:
        assert "outside its own interval" in str(e); return
    raise AssertionError("accepted an estimate its interval does not contain")


def test_mismatched_campaign_labels_are_refused() -> None:
    try:
        campaign_bootstrap(torch.zeros(10), torch.zeros(5, dtype=torch.long))
    except ContrastError as e:
        assert "campaign labels" in str(e); return
    raise AssertionError("bootstrapped cells against the wrong labels")


def test_campaign_bootstrap_validates_its_exported_tensor_boundary() -> None:
    """The public helper must refuse malformed calls before tensor operations or broadcasting."""
    effects = torch.arange(4, dtype=torch.float64)
    labels = torch.tensor([0, 0, 1, 1])
    invalid = (
        (effects.tolist(), labels, "per_cell must be a torch.Tensor"),
        (effects, labels.tolist(), "campaign_index must be a torch.Tensor"),
        (effects.reshape(2, 2), labels, "rank-1 tensor"),
        (effects, labels.reshape(2, 2), "rank 1"),
        (effects.to(torch.int64), labels, "floating scenario effects"),
        (torch.empty(4, device="meta"), labels, "must be on"),
    )
    for per_cell, campaign_index, fragment in invalid:
        try:
            campaign_bootstrap(per_cell, campaign_index)
        except ContrastError as error:
            assert fragment in str(error), (fragment, error)
            continue
        raise AssertionError(
            f"accepted malformed bootstrap tensors {per_cell!r}, {campaign_index!r}")


def test_campaign_bootstrap_requires_finite_integer_labels_and_two_episodes() -> None:
    effects = torch.arange(4, dtype=torch.float64)
    for labels in (
        torch.tensor([False, False, True, True]),
        torch.tensor([0, 0, 1, 1], dtype=torch.uint8),
        torch.tensor([0.0, 0.0, 1.0, 1.0]),
        torch.tensor([0.0, 0.0, float("nan"), 1.0]),
        torch.tensor([0.0, 0.0, float("inf"), 1.0]),
        torch.tensor([0.0, 0.0, 1.5, 1.5]),
    ):
        try:
            campaign_bootstrap(effects, labels)
        except ContrastError as error:
            assert "exact int32 or int64 episode labels" in str(error), error
            continue
        raise AssertionError(f"accepted malformed campaign labels {labels!r}")

    low, high = campaign_bootstrap(
        effects, torch.tensor([0, 0, 1, 1], dtype=torch.int32), n_bootstrap=4)
    assert low <= high

    try:
        campaign_bootstrap(effects, torch.zeros(4, dtype=torch.long))
    except ContrastError as error:
        assert "reads as certainty" in str(error)
    else:
        raise AssertionError("the exported bootstrap accepted only one campaign")


def test_campaign_bootstrap_refuses_finite_inputs_whose_totals_overflow() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    for dtype in (torch.float16, torch.float32):
        bound = torch.finfo(dtype).max
        low, high = campaign_bootstrap(
            torch.tensor([bound, bound, -bound, -bound], dtype=dtype),
            labels,
            n_bootstrap=8,
        )
        assert torch.isfinite(torch.tensor([low, high], dtype=torch.float64)).all()

    try:
        campaign_bootstrap(
            torch.full((4,), 1e308, dtype=torch.float64), labels, n_bootstrap=4)
    except ContrastError as error:
        assert "overflowed" in str(error)
        return
    raise AssertionError("the exported bootstrap returned an infinite interval")


def test_is_quotable_requires_all_three_conditions() -> None:
    """Enough campaigns, enough overlap, and no outstanding notes. Any one failing makes the
    number an extrapolation that looks exactly like evidence."""
    good = dict(bundle="b", effect=1.0, low=0.0, high=2.0, origin_is_pre_treatment=True,
                episode_validated=True, **REPRODUCTION)
    assert Contrast(**good, n_campaigns=8, overlap=0.9).is_quotable
    assert not Contrast(**good, n_campaigns=2, overlap=0.9).is_quotable
    assert not Contrast(**good, n_campaigns=8, overlap=0.2).is_quotable
    assert not Contrast(**good, n_campaigns=8, overlap=0.9, notes=["something"]).is_quotable


def test_query_campaign_labels_alone_cannot_make_a_contrast_quotable() -> None:
    kf, draws, camp = setup(planned=0.1, spread=0.04)
    out = contrast(model, kf, InterventionBundle("promo", (0,)), draws,
                   origin="2016-05-01", campaign_start="2016-06-01", campaign_index=camp,
                   observed_driver_range=(0.0, 0.5), planned_driver_range=(0.0, 0.15))
    assert not out.is_quotable and not out.episode_validated
    assert any("held-out whole-episode" in note for note in out.notes)


def test_contrast_record_refuses_non_finite_or_malformed_evidence() -> None:
    for fields in (
        dict(effect=float("nan"), low=0.0, high=1.0, n_campaigns=8, overlap=1.0),
        dict(effect=10**400, low=0.0, high=1.0, n_campaigns=8, overlap=1.0),
        dict(effect=1.0, low=0.0, high=float("inf"), n_campaigns=8, overlap=1.0),
        dict(effect=1.0, low=0.0, high=2.0, n_campaigns=True, overlap=1.0),
        dict(effect=1.0, low=0.0, high=2.0, n_campaigns=8, overlap=1.1),
    ):
        try:
            Contrast(bundle="b", origin_is_pre_treatment=True, **REPRODUCTION, **fields)
        except ContrastError:
            continue
        raise AssertionError(f"accepted malformed contrast evidence: {fields!r}")
    try:
        InterventionBundle("b", (0,), reference=10**400)
    except ContrastError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("accepted an unrepresentable intervention reference")


def test_contrast_record_validates_every_reproduction_control_and_window() -> None:
    class IntAlias(int):
        pass

    class StringAlias(str):
        pass

    base = {
        "bundle": "b",
        "effect": 1.0,
        "low": 0.0,
        "high": 2.0,
        "n_campaigns": 8,
        "overlap": 1.0,
        "origin_is_pre_treatment": True,
        **REPRODUCTION,
    }
    invalid = (
        ("origin", "2016-5-01"),
        ("campaign_start", "2016-05-01"),
        ("n_campaigns", IntAlias(8)),
        ("n_bootstrap", True),
        ("n_bootstrap", 0),
        ("n_bootstrap", IntAlias(2)),
        ("seed", True),
        ("seed", -1),
        ("seed", 2**63),
        ("alpha", True),
        ("alpha", float("nan")),
        ("alpha", 0.0),
        ("alpha", 10**400),
        ("observed_driver_range", [0.0, 0.5]),
        ("observed_driver_range", (0.5, 0.0)),
        ("planned_driver_range", (0.0, float("inf"))),
        ("horizon", True),
        ("horizon", 0),
        ("window", "all_days"),
        ("window", StringAlias("horizon")),
        ("window_bounds", [1, HORIZON]),
        ("window_bounds", (0, HORIZON)),
        ("window_bounds", (1, HORIZON + 1)),
        ("window_bounds", (1, True)),
        ("requested_window_bounds", (1, HORIZON)),
        ("requested_window", "custom"),
        ("requested_window", StringAlias("horizon")),
    )
    for field, value in invalid:
        try:
            Contrast(**(base | {field: value}))
        except ContrastError:
            continue
        raise AssertionError(f"accepted invalid contrast control {field}={value!r}")


def test_contrast_constructor_preserves_requested_and_effective_window_provenance() -> None:
    fields = {
        "bundle": "b",
        "effect": 1.0,
        "low": 0.0,
        "high": 2.0,
        "n_campaigns": 8,
        "overlap": 1.0,
        "origin_is_pre_treatment": True,
        **REPRODUCTION,
        "campaign_start": "2016-05-06",
        "window": "event_support",
        "requested_window": "event_support",
        "requested_window_bounds": (-2, 2),
        "window_bounds": (3, 7),
    }
    record = json.loads(json.dumps(Contrast(**fields).as_record()))
    assert record["horizon"] == HORIZON
    assert record["requested_window_bounds"] == [-2, 2]
    assert record["window_bounds"] == [3, 7]

    for changes in (
        {"requested_window_bounds": None},
        {"requested_window_bounds": (-2, 30), "window_bounds": (3, 6)},
        {"horizon": HORIZON - 1},
    ):
        try:
            Contrast(**(fields | changes))
        except ContrastError:
            continue
        raise AssertionError(f"accepted inconsistent window provenance {changes!r}")


def test_contrast_numeric_controls_refuse_unrepresentable_json_integers() -> None:
    class HostileInt(int):
        def __float__(self):
            raise RuntimeError("untrusted coercion ran")

    kf, draws, camp = setup()
    for invalid in (10**400, HostileInt(1)):
        for call in (
            lambda invalid=invalid: campaign_bootstrap(
                torch.arange(40, dtype=torch.float64), camp, alpha=invalid),
            lambda invalid=invalid: contrast(
                model, kf, InterventionBundle("promo", (0,)), draws,
                origin="2016-05-01", campaign_start="2016-06-01", campaign_index=camp,
                observed_driver_range=(0.0, invalid), planned_driver_range=(0.0, 0.3)),
        ):
            try:
                call()
            except ContrastError:
                continue
            raise AssertionError("an invalid numeric control entered scenario arithmetic")


def test_overlap_stays_exact_when_finite_endpoint_subtraction_would_overflow() -> None:
    kf, draws, camp = setup()
    result = contrast(
        model,
        kf,
        InterventionBundle("promo", (0,)),
        draws,
        origin="2016-05-01",
        campaign_start="2016-06-01",
        campaign_index=camp,
        observed_driver_range=(-1e308, 1e308),
        planned_driver_range=(-1e308, 1e308),
    )
    assert result.overlap == 1.0


def test_model_outputs_have_exact_cell_and_horizon_axes() -> None:
    kf, draws, camp = setup()

    def trailing_axis(known_future: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
        return torch.zeros(CELLS, HORIZON, 1)

    def wrong_horizon(known_future: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
        return torch.zeros(CELLS, HORIZON - 1)

    for model_fn in (trailing_axis, wrong_horizon):
        try:
            contrast(
                model_fn,
                kf,
                InterventionBundle("promo", (0,)),
                draws,
                origin="2016-05-01",
                campaign_start="2016-06-01",
                campaign_index=camp,
                observed_driver_range=(0.0, 0.5),
                planned_driver_range=(0.0, 0.3),
            )
        except ContrastError as exc:
            assert "[cells] or [cells, horizon]" in str(exc)
            continue
        raise AssertionError(f"accepted malformed model output from {model_fn.__name__}")


def test_model_outputs_are_aligned_floating_tensors_on_the_scenario_device() -> None:
    kf, draws, camp = setup()

    def integral(known_future: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
        return torch.zeros(CELLS, dtype=torch.int64)

    def complex_output(known_future: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
        return torch.zeros(CELLS, dtype=torch.complex64)

    def wrong_device(known_future: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
        return torch.empty(CELLS, device="meta")

    n_calls = 0

    def alternating_dtype(
        known_future: torch.Tensor,
        shared: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal n_calls
        n_calls += 1
        dtype = torch.float32 if n_calls == 1 else torch.float64
        return torch.zeros(CELLS, dtype=dtype)

    for model_fn, fragment in (
        (integral, "floating expected-demand"),
        (complex_output, "floating expected-demand"),
        (wrong_device, "same device as known_future"),
        (alternating_dtype, "same floating dtype and device"),
    ):
        try:
            contrast(
                model_fn,
                kf,
                InterventionBundle("promo", (0,)),
                draws,
                origin="2016-05-01",
                campaign_start="2016-06-01",
                campaign_index=camp,
                observed_driver_range=(0.0, 0.5),
                planned_driver_range=(0.0, 0.3),
            )
        except ContrastError as error:
            assert fragment in str(error), (model_fn.__name__, error)
            continue
        raise AssertionError(f"accepted malformed model output from {model_fn.__name__}")


def test_contrast_requires_floating_rank_three_known_future() -> None:
    """Integer storage used to truncate a fractional reference during counterfactual assignment."""
    _, draws, camp = setup()
    malformed = (
        torch.ones(CELLS, HORIZON, KNOWN, dtype=torch.int64),
        torch.ones(CELLS, HORIZON, KNOWN, dtype=torch.bool),
        torch.ones(CELLS, HORIZON, KNOWN, dtype=torch.complex64),
        torch.ones(CELLS, KNOWN, dtype=torch.float32),
    )
    for known_future in malformed:
        try:
            contrast(
                model,
                known_future,
                InterventionBundle("promo", (0,), reference=0.5),
                draws,
                origin="2016-05-01",
                campaign_start="2016-06-01",
                campaign_index=camp,
                observed_driver_range=(0.0, 0.5),
                planned_driver_range=(0.0, 0.3),
            )
        except ContrastError as error:
            assert "floating rank-3" in str(error), error
            continue
        raise AssertionError(
            f"accepted known_future {known_future.dtype} shape={tuple(known_future.shape)}")

    for shape in ((0, HORIZON, KNOWN), (CELLS, 0, KNOWN), (CELLS, HORIZON, 0)):
        try:
            contrast(
                model,
                torch.empty(shape),
                InterventionBundle("promo", (0,)),
                draws,
                origin="2016-05-01",
                campaign_start="2016-06-01",
                campaign_index=camp,
                observed_driver_range=(0.0, 0.5),
                planned_driver_range=(0.0, 0.3),
            )
        except ContrastError as error:
            assert "axes must all be non-empty" in str(error), error
            continue
        raise AssertionError(f"accepted empty known_future axes {shape}")


def test_contrast_refuses_unaligned_references_and_driver_ranges() -> None:
    kf, draws, camp = setup()
    bad_bundle = InterventionBundle("b", (0,), reference_values=torch.zeros(1, 1, 1))
    try:
        contrast(model, kf, InterventionBundle("b", (0,)), draws,
                 origin="2016-05-01", campaign_start="2016-06-01", campaign_index=camp,
                 observed_driver_range=(0.5, 0.0), planned_driver_range=(0.0, 0.3))
    except ContrastError as exc:
        assert "ordered pair" in str(exc)
    else:
        raise AssertionError("accepted a reversed observed driver range")

    try:
        contrast(model, kf, bad_bundle, draws,
                 origin="2016-05-01", campaign_start="2016-06-01", campaign_index=camp,
                 observed_driver_range=(0.0, 0.5), planned_driver_range=(0.0, 0.3))
    except ContrastError as exc:
        assert "does not align" in str(exc)
    else:
        raise AssertionError("broadcast a reference over the wrong scenario cells")


def test_bundle_reference_values_have_one_exact_materialized_scenario_type() -> None:
    class IntAlias(int):
        pass

    for values in (
        [[[0.0]]],
        torch.ones(1, 1, 1, dtype=torch.int64),
        torch.ones(1, 1, 1, dtype=torch.complex64),
        torch.empty(1, 1, 1, device="meta"),
    ):
        try:
            InterventionBundle("b", (0,), reference_values=values)
        except ContrastError as error:
            assert "reference_values" in str(error)
            continue
        raise AssertionError(f"accepted malformed reference values {values!r}")

    try:
        InterventionBundle("b", (IntAlias(0),))
    except ContrastError as error:
        assert "integer indices" in str(error)
    else:
        raise AssertionError("accepted an aliased intervention column index")

    kf, draws, camp = setup()
    wrong_dtype = InterventionBundle(
        "b", (0,), reference_values=torch.zeros(CELLS, HORIZON, 1, dtype=torch.float64))
    try:
        contrast(
            model, kf, wrong_dtype, draws,
            origin="2016-05-01", campaign_start="2016-06-01", campaign_index=camp,
            observed_driver_range=(0.0, 0.5), planned_driver_range=(0.0, 0.3),
        )
    except ContrastError as error:
        assert "same floating dtype and device" in str(error)
    else:
        raise AssertionError("silently converted reference values into the scenario tensor dtype")


def test_the_attribution_window_selects_the_days_it_claims() -> None:
    """`attribution_window` was a spec field nothing read: contrast() summed every horizon day
    regardless -- silently the 'horizon' choice, the one the spec warns counts a promotion's
    spike and none of its payback. With per-day output, event_support must integrate exactly
    the support around the campaign start, label a support that outruns the horizon GROSS, and
    a custom window must slice precisely."""
    kf, draws, camp = setup(spread=0.1)

    def per_day(known_future: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
        base = (2.0 + 4.0 * known_future[..., 0] + shared.unsqueeze(-1)).clamp_min(0.0)
        return base

    implicit = contrast(per_day, kf, InterventionBundle("promo", (0,)), draws,
                        origin="2016-05-01", campaign_start="2016-05-06",
                        campaign_index=camp, observed_driver_range=(0.0, 0.5),
                        planned_driver_range=(0.0, 0.3))
    full = contrast(per_day, kf, InterventionBundle("promo", (0,)), draws,
                    origin="2016-05-01", campaign_start="2016-05-06", campaign_index=camp,
                    observed_driver_range=(0.0, 0.5), planned_driver_range=(0.0, 0.3),
                    window="horizon")
    assert full.window == "horizon"
    assert implicit.horizon == full.horizon == HORIZON
    assert implicit.window_bounds == full.window_bounds == (1, HORIZON)
    implicit_record = json.loads(json.dumps(implicit.as_record()))
    explicit_record = json.loads(json.dumps(full.as_record()))
    assert implicit_record["requested_window"] is None
    assert explicit_record["requested_window"] == "horizon"
    assert implicit_record["horizon"] == explicit_record["horizon"] == HORIZON
    assert implicit_record["window_bounds"] == explicit_record["window_bounds"] == [1, HORIZON]
    assert implicit_record["requested_window_bounds"] is None
    assert explicit_record["requested_window_bounds"] is None
    assert any("BY CHOICE" in note for note in full.notes)
    windowed = contrast(per_day, kf, InterventionBundle("promo", (0,)), draws,
                        origin="2016-05-01", campaign_start="2016-05-06", campaign_index=camp,
                        observed_driver_range=(0.0, 0.5), planned_driver_range=(0.0, 0.3),
                        window="event_support", event_support=(-2, 2))
    assert windowed.window == "event_support"
    assert windowed.window_bounds == (3, 7)
    assert windowed.as_record()["window_bounds"] == [3, 7]
    assert windowed.requested_window_bounds == (-2, 2)
    assert windowed.as_record()["requested_window_bounds"] == [-2, 2]
    # Days 3..7: 5 of 7 days, and the per-day effect is uniform here.
    assert abs(windowed.effect - full.effect * 5 / HORIZON) < 1e-6 * max(abs(full.effect), 1.0)
    gross = contrast(per_day, kf, InterventionBundle("promo", (0,)), draws,
                     origin="2016-05-01", campaign_start="2016-05-06", campaign_index=camp,
                     observed_driver_range=(0.0, 0.5), planned_driver_range=(0.0, 0.3),
                     window="event_support", event_support=(-2, 30))
    assert any("GROSS" in note for note in gross.notes), \
        "a support beyond the horizon must be labelled gross, not presented as net"
    assert gross.window_bounds == (3, HORIZON)
    assert gross.requested_window_bounds == (-2, 30)
    collision = contrast(
        per_day, kf, InterventionBundle("promo", (0,)), draws,
        origin="2016-05-01", campaign_start="2016-05-06", campaign_index=camp,
        observed_driver_range=(0.0, 0.5), planned_driver_range=(0.0, 0.3),
        window="event_support", event_support=(-2, 40),
    )
    gross_record = json.loads(json.dumps(gross.as_record()))
    collision_record = json.loads(json.dumps(collision.as_record()))
    assert gross_record["window_bounds"] == collision_record["window_bounds"] == [3, HORIZON]
    assert gross_record["requested_window_bounds"] == [-2, 30]
    assert collision_record["requested_window_bounds"] == [-2, 40]
    sliced = contrast(per_day, kf, InterventionBundle("promo", (0,)), draws,
                      origin="2016-05-01", campaign_start="2016-05-06", campaign_index=camp,
                      observed_driver_range=(0.0, 0.5), planned_driver_range=(0.0, 0.3),
                      window="custom", custom_window=(1, 7))
    assert sliced.window_bounds == (1, 7)
    assert sliced.requested_window_bounds == (1, 7)
    assert abs(sliced.effect - full.effect * 7 / HORIZON) < 1e-6 * max(abs(full.effect), 1.0)
    try:
        contrast(model, kf, InterventionBundle("promo", (0,)), draws,
                 origin="2016-05-01", campaign_start="2016-05-06", campaign_index=camp,
                 observed_driver_range=(0.0, 0.5), planned_driver_range=(0.0, 0.3),
                 window="event_support", event_support=(-2, 2))
    except ContrastError as e:
        assert "per-day" in str(e)
    else:
        raise AssertionError("a windowed contrast accepted per-cell totals it cannot window")


def test_window_arguments_cannot_be_mixed_or_silently_ignored() -> None:
    kf, draws, camp = setup(spread=0.1)
    invalid = (
        ({"event_support": (-1, 1)}, "only valid with window='event_support'"),
        ({"custom_window": (1, 2)}, "only valid with window='custom'"),
        ({"window": "horizon", "event_support": (-1, 1)}, "only valid"),
        ({"window": "horizon", "custom_window": (1, 2)}, "only valid"),
        ({"window": "event_support", "event_support": (-1, 1),
          "custom_window": (1, 2)}, "only valid with window='custom'"),
        ({"window": "custom", "custom_window": (1, 2),
          "event_support": (-1, 1)}, "only valid with window='event_support'"),
        ({"window": "event_support"}, "needs event_support"),
        ({"window": "custom"}, "needs custom_window"),
        ({"window": "event_support", "event_support": (False, 1)}, "built-in integer"),
    )
    for window_args, fragment in invalid:
        try:
            contrast(
                model,
                kf,
                InterventionBundle("promo", (0,)),
                draws,
                origin="2016-05-01",
                campaign_start="2016-05-06",
                campaign_index=camp,
                observed_driver_range=(0.0, 0.5),
                planned_driver_range=(0.0, 0.3),
                **window_args,
            )
        except ContrastError as error:
            assert fragment in str(error), (window_args, error)
            continue
        raise AssertionError(f"accepted contradictory window arguments {window_args!r}")


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn(); print("PASS " + name)
