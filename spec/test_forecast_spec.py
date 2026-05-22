"""Behavioral checks for data/model specification validation and resolution."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "spec"))

from spec.forecast_spec import (  # noqa: E402
    ExposureSpec,
    ForecastDataSpec,
    ForecastModelSpec,
    ObservationSpec,
)


def _rich() -> ForecastDataSpec:
    """A client with everything: the configuration the design assumes."""
    return ForecastDataSpec(
        time_idx="date", target="qty", group_ids=("store", "sku"),
        observation=ObservationSpec(status_col="oos"),
        exposure=ExposureSpec(ranged_col="is_ranged"),
        hierarchy=("chain", "store"), events=("christmas", "easter"),
        event_known_at_col="announced_on", event_episode_col="campaign_id",
        price_col="unit_price", promo_col="promo", oos_col="oos",
        factor_group_col="chain", lead_time_plus_review=21)


def _bare() -> ForecastDataSpec:
    """The three mandatory columns and nothing else."""
    return ForecastDataSpec(time_idx="date", target="qty", group_ids=("store", "sku"),
                            observation=ObservationSpec())


def _blocking(spec, data):
    return [e for e in spec.validate(data) if not str(e).startswith("NOTE:")]


# ---------------------------------------------------------------- validate
def test_validate_passes_a_fully_specified_client():
    assert _blocking(ForecastModelSpec(), _rich()) == []


def test_validate_refuses_top_down_topology():
    """A chain total is unobservable when a store is censored; summed recorded sales supply a lower
    bound.
    """
    errs = _blocking(ForecastModelSpec(topology="top_down"), _rich())
    assert any("unobserved whenever any store is censored" in str(e) for e in errs)


def test_validate_refuses_a_point_forecast():
    """An expected demand of 0.2 rounds to 0, the store orders nothing, and then stocks out."""
    assert any("emit='point'" in str(e) for e in _blocking(ForecastModelSpec(emit="point"), _rich()))


def test_validate_refuses_a_clamped_baseline():
    """logit(1e-4) is -9.2, and a slow mover's occurrence head never recovers from it."""
    errs = _blocking(ForecastModelSpec(baseline_smoothing="clamp"), _rich())
    assert any("-9.2" in str(e) for e in errs)


def test_validate_refuses_speculative_components_during_first_build():
    """Nothing is measured yet, so the weakly identified parts stay off."""
    errs = _blocking(ForecastModelSpec(cross_product="signed_graph"), _rich())
    assert any("first_build=True" in str(e) for e in errs)


def test_validate_refuses_a_horizon_shorter_than_the_lead_time():
    """The paths exist so replenishment can integrate over lead time plus review period."""
    data = ForecastDataSpec(**{**_rich().__dict__, "max_prediction_length": 14,
                               "lead_time_plus_review": 35})
    assert any("shorter than lead time" in str(e) for e in _blocking(ForecastModelSpec(), data))


def test_validate_refuses_full_path_materialisation():
    """1,000 paths x every cell x horizon is trillions of values at catalogue scale."""
    errs = _blocking(ForecastModelSpec(path_materialisation="full"), _rich())
    assert any("not storable" in str(e) for e in errs)


# ---------------------------------------------------------------- resolve
def test_resolve_degrades_a_bare_client_to_something_buildable():
    """The three-mandatory-inputs promise. It was FALSE twice before this test existed."""
    bare = _bare()
    r = ForecastModelSpec().resolve(bare)
    assert _blocking(r, bare) == []
    assert [e for e in bare.validate() if not str(e).startswith("NOTE:")] == []


def test_resolve_drops_the_factor_without_a_grouping_column():
    r = ForecastModelSpec().resolve(_bare())
    assert r.topology == "bottom_up" and r.factor_rank == 0
    assert r.emit == "parameters", "coherent sample paths are meaningless without the factor"


def test_resolve_disables_price_monotonicity_without_a_price_column():
    r = ForecastModelSpec().resolve(_bare())
    assert r.price_response_mode == "none"
    assert "price" not in r.attribution_drivers


def test_resolve_disables_scenarios_when_no_settable_driver_exists():
    r = ForecastModelSpec().resolve(_bare())
    assert r.attribution is None and not r.attribution_drivers


def test_resolve_keeps_commercial_scenarios_without_an_event_announcement_column():
    """Event announcement metadata cannot gate an unrelated planned-price intervention.

    This client has no event table at all, so requiring `event_known_at_col` disabled a perfectly
    expressible price/promotion contrast because an irrelevant column was absent.
    """
    commercial = ForecastDataSpec(**{
        **_bare().__dict__, "price_col": "unit_price", "promo_col": "promo"})
    r = ForecastModelSpec().resolve(commercial)
    assert r.attribution == "prospective_scenario"
    assert tuple(r.attribution_drivers) == ("price", "promo")
    assert "event" not in r.attribution_drivers and commercial.event_known_at_col is None
    assert any("prospective commercial scenario contrasts" in value
               for value in r.capabilities(commercial)["available"])

    with_unannounced_events = ForecastDataSpec(**{
        **commercial.__dict__, "events": ("campaign",), "event_known_at_col": None})
    r_events = ForecastModelSpec().resolve(with_unannounced_events)
    assert r_events.attribution == "prospective_scenario"
    assert tuple(r_events.attribution_drivers) == ("price", "promo"), \
        "unsafe event contrast should be removed without deleting unrelated commercial drivers"


def test_preannounced_events_preserve_event_capability_and_scenario_driver():
    commercial = ForecastDataSpec(**{
        **_bare().__dict__,
        "price_col": "unit_price",
        "promo_col": "promo",
        "events": ("holiday",),
        "events_preannounced": True,
    })
    resolved = ForecastModelSpec().resolve(commercial)
    assert "event" in resolved.attribution_drivers
    assert not any(
        "without announcement dates" in loss
        for loss in resolved.capabilities(commercial)["skipped"]
    )
    assert not any("nobody had been told" in str(error) for error in resolved.validate(commercial))


def test_events_preannounced_requires_an_actual_boolean():
    for flag in ("false", "true", 1, 0, None):
        data = ForecastDataSpec(**{
            **_bare().__dict__, "events": ("holiday",), "events_preannounced": flag,
        })
        assert any("events_preannounced must be boolean" in error for error in data.validate())
        model = ForecastModelSpec()
        assert "event" not in model.resolve(data).attribution_drivers
        assert any("without announcement dates" in loss
                   for loss in model.capabilities(data)["skipped"])
        assert any("event_known_at_col" in str(error) for error in model.validate(data))


def test_baseline_is_not_an_intervention_driver():
    """A lagged baseline is inferred from history and no scenario API can set it."""
    assert "baseline" not in ForecastModelSpec().attribution_drivers
    errors = _blocking(ForecastModelSpec(attribution_drivers=("baseline",)), _rich())
    assert any("no prospective scenario input can set it" in str(error) for error in errors)


def test_resolve_leaves_a_fully_specified_client_alone():
    rich = _rich()
    r = ForecastModelSpec().resolve(rich)
    assert r.topology == "bottom_up_factor" and r.emit == "sample_paths"


# ---------------------------------------------------------------- capabilities
def test_capabilities_reports_what_a_bare_client_loses():
    caps = ForecastModelSpec().capabilities(_bare())
    assert caps["available"], "a bare client must still build"
    lost = " ".join(caps["skipped"]).lower()
    for expected in ("range eligibility", "censoring", "price", "calendar"):
        assert expected in lost, f"{expected} loss not reported"


def test_capabilities_requires_both_hierarchy_and_factor_group_col():
    """The capability report and resolver must require both pieces of the shared-factor contract."""
    data = ForecastDataSpec(time_idx="d", target="q", group_ids=("b", "s"),
                            observation=ObservationSpec(), hierarchy=("chain", "b"))
    caps = ForecastModelSpec().capabilities(data)
    assert not any("cross-store" in c for c in caps["available"])


def test_capability_report_and_validate_agree():
    """They disagreed for two rounds: the report said BUILDS while validate blocked."""
    bare = _bare()
    r = ForecastModelSpec().resolve(bare)
    assert ("BUILDS: yes" in r.capability_report(bare)) == (not _blocking(r, bare))


# ---------------------------------------------------------------- data contract
def test_data_validate_notes_a_missing_range_column_without_blocking():
    """Absent range eligibility is a capability loss, not a misconfiguration."""
    notes = [e for e in _bare().validate() if str(e).startswith("NOTE:")]
    assert any("ranged_col" in str(n) for n in notes)


def test_the_demand_branch_may_attend_over_time_and_others_may_not():
    """Demand patches can attend over time while other branches retain their declared axes."""
    from forecast_spec import Branch
    def with_branch(br):
        return ForecastDataSpec(time_idx="d", target="q", group_ids=("s",),
                                observation=ObservationSpec(), branches=br)

    def time_axis_errors(ds):
        return [x for x in ds.validate() if "attends over 'time'" in str(x)]

    assert not time_axis_errors(with_branch(
        {"demand": Branch(attend_over="time", is_demand_history=True)}))
    assert time_axis_errors(with_branch({"calendar": Branch(attend_over="time")}))


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
