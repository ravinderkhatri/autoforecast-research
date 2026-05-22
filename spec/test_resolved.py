"""The resolved build: refusal before training, canonical identity, and the head step-down.

Run: python3 spec/test_resolved.py
"""

from __future__ import annotations

import pathlib
import sys
from dataclasses import replace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from spec.forecast_spec import HURDLE_ZERO_SHARE_MIN, ForecastDataSpec, ForecastModelSpec
from spec.resolved import ResolvedBuild, ResolveError

DATA = ForecastDataSpec(
    time_idx="date", target="qty", group_ids=("id",), price_col="price",
    factor_group_col="state", hierarchy=("store", "state"),
    events="holidays", event_known_at_col="known_at",
    max_prediction_length=28, lead_time_plus_review=7)


@pytest.mark.parametrize("flag", ["false", "true", 1, 0, None])
@pytest.mark.parametrize("known_at", [None, "known_at"])
def test_from_specs_and_for_revision_reject_nonboolean_event_flags(flag, known_at):
    data = replace(DATA, events_preannounced=flag, event_known_at_col=known_at)
    model = ForecastModelSpec(factor_rank=0)
    with pytest.raises(ResolveError, match="events_preannounced must be boolean"):
        ResolvedBuild.from_specs(data, model)
    for variant in ("v1_corrected", "v2", "v3"):
        with pytest.raises(ResolveError, match="events_preannounced must be boolean"):
            ResolvedBuild.for_revision(data, model, variant=variant)


def test_from_specs_validates_data_and_model_together():
    data = replace(DATA, events_preannounced="false", event_known_at_col=None)
    with pytest.raises(ResolveError) as failure:
        ResolvedBuild.from_specs(data, ForecastModelSpec(n_sample_paths=100))
    message = str(failure.value)
    assert "events_preannounced must be boolean" in message
    assert "event_known_at_col" in message and "n_sample_paths" in message
    for flag, known_at in ((True, None), (False, "known_at")):
        valid = replace(DATA, events_preannounced=flag, event_known_at_col=known_at)
        build = ResolvedBuild.from_specs(valid, ForecastModelSpec())
        assert build.event_kernel == "fixed_window"
        assert any("exposure.ranged_col" in note for note in build.notes)



def test_the_measured_zero_rate_selects_the_head() -> None:
    """The architecture has always SAID resolve() steps down to a plain count head where the
    zero rate does not justify a hurdle -- and until zero_share existed, the resolver had no
    zero-rate input, so the sentence had no executable path and the plain head was unreachable."""
    hurdle = ResolvedBuild.from_specs(DATA, ForecastModelSpec(), zero_share=0.55)
    assert hurdle.head == "hurdle_negbin"
    plain = ResolvedBuild.from_specs(DATA, ForecastModelSpec(), zero_share=0.03)
    assert plain.head == "negbin", "a 3% zero rate must step the hurdle down"
    boundary = ResolvedBuild.from_specs(DATA, ForecastModelSpec(),
                                        zero_share=HURDLE_ZERO_SHARE_MIN)
    assert boundary.head == "hurdle_negbin", "the threshold itself keeps the hurdle"
    unmeasured = ResolvedBuild.from_specs(DATA, ForecastModelSpec(), zero_share=None)
    assert unmeasured.head == "hurdle_negbin", \
        "an unmeasured rate is not evidence of a low one"


def test_an_invalid_zero_share_is_refused() -> None:
    try:
        ForecastModelSpec().resolve(DATA, zero_share=1.5)
    except ValueError as e:
        assert "zero_share" in str(e); return
    raise AssertionError("a zero share outside [0,1] was accepted")


def test_an_unbuildable_combination_is_refused_before_training() -> None:
    """`validate()` errors must stop the build at resolution, not crash inside an epoch. And the
    refusal carries EVERY error: a builder that reports one of three blockers runs three times."""
    bad = ForecastDataSpec(time_idx="d", target="y", group_ids=("id",),
                           factor_group_col="state", hierarchy=("store",),
                           events="holidays")           # no known_at: a validate() error
    try:
        # Two errors that BOTH survive resolution: the missing announcement column, and a path
        # count too small to resolve a P99 tail. (An error resolution itself removes -- like
        # price-response disabled with no price column -- is degradation working, not a blocker.)
        ResolvedBuild.from_specs(bad, ForecastModelSpec(n_sample_paths=100), zero_share=0.5)
    except ResolveError as e:
        assert "event_known_at_col" in str(e) and "n_sample_paths" in str(e), str(e)
        return
    raise AssertionError("an unbuildable combination produced a build object")


def test_the_build_hash_is_stable_and_sensitive() -> None:
    """Same specs, same hash -- twice. A changed architecture field changes it. Reworded NOTES
    must not: they are advisory prose, and a documentation edit is not a new architecture."""
    a = ResolvedBuild.from_specs(DATA, ForecastModelSpec(), zero_share=0.5)
    b = ResolvedBuild.from_specs(DATA, ForecastModelSpec(), zero_share=0.5)
    assert a.build_hash == b.build_hash and len(a.build_hash) == 64
    changed = ResolvedBuild.from_specs(DATA, ForecastModelSpec(d_model=128), zero_share=0.5)
    assert changed.build_hash != a.build_hash, "a different d_model hashed identically"
    stripped = ResolvedBuild(**{**a.__dict__, "notes": ()})
    assert stripped.build_hash == a.build_hash, "notes leaked into the architecture identity"
    assert "notes" not in a.canonical()


def test_structural_price_response_is_an_explicit_versioned_build_contract() -> None:
    structural_spec = ForecastModelSpec(
        price_response_mode="monotone_raw_v1",
        price_occurrence_slope_max=1.25,
        price_size_slope_max=0.75,
    )
    build = ResolvedBuild.from_specs(DATA, structural_spec, zero_share=0.55)
    assert build.schema == "autoforecast.resolved-build.v2"
    assert build.schema_version == 2
    assert build.price_response_mode == "monotone_raw_v1"
    assert build.price_occurrence_slope_max == 1.25
    assert build.price_size_slope_max == 0.75
    assert build.canonical()["price_response_mode"] == "monotone_raw_v1"

    changed_cap = ResolvedBuild.from_specs(
        DATA,
        ForecastModelSpec(
            price_response_mode="monotone_raw_v1",
            price_occurrence_slope_max=1.5,
            price_size_slope_max=0.75,
        ),
        zero_share=0.55,
    )
    assert changed_cap.build_hash != build.build_hash

    try:
        ResolvedBuild.from_specs(
            DATA,
            ForecastModelSpec(
                price_response_mode="monotone_raw_v1",
                price_occurrence_slope_max=1.25,
            ),
            zero_share=0.55,
        )
    except ResolveError as exc:
        assert "price_size_slope_max" in str(exc)
    else:
        raise AssertionError("a structural build without both registered caps was resolved")


def test_degradation_is_visible_in_the_build() -> None:
    """A client with no factor grouping resolves to factorless marginals, and the build object
    must SAY so -- rank 0, no dynamics, parameters emitted -- because the adapter constructs
    from these fields and nothing else."""
    bare = ForecastDataSpec(time_idx="d", target="y", group_ids=("id",))
    build = ResolvedBuild.from_specs(bare, ForecastModelSpec(), zero_share=0.5)
    assert build.factor_rank == 0 and build.factor_dynamics is None
    assert build.emit == "parameters"
    assert build.event_kernel is None, "no calendar, no kernel"
    assert build.price_response_mode == "none", "no price column, nothing to constrain"


def test_resolved_build_normalizes_float_domain_numeric_aliases() -> None:
    build = ResolvedBuild.from_specs(
        DATA,
        ForecastModelSpec(demand_branch_dropout=0, covariate_dropout=0),
        zero_share=0,
    )
    assert type(build.demand_branch_dropout) is float
    assert type(build.covariate_dropout) is float
    assert type(build.zero_share) is float


def test_invalid_zero_share_is_refused_before_the_model_resolver_runs() -> None:
    class ResolverSentinel:
        called = False

        def resolve(self, data, *, zero_share):
            self.called = True
            raise AssertionError("invalid zero_share reached model.resolve")

    class HostileFloat(float):
        def __float__(self):
            raise AssertionError("invalid zero_share was coerced")

    for invalid in (True, float("nan"), float("inf"), -0.1, 1.1, 10**400,
                    HostileFloat(0.5), "0.5"):
        model = ResolverSentinel()
        try:
            ResolvedBuild.from_specs(DATA, model, zero_share=invalid)
        except ResolveError as exc:
            assert "zero_share" in str(exc)
            assert not model.called
            continue
        raise AssertionError(f"accepted invalid zero_share {invalid!r}")


def test_resolved_record_refuses_every_malformed_typed_contract_field() -> None:
    """The isolated-worker record is an input boundary, not a trusted dataclass dump."""
    build = ResolvedBuild.from_specs(DATA, ForecastModelSpec(), zero_share=0.5)
    corruptions = (
        ("schema", "unregistered.resolved-build.v2"),
        ("schema_version", True),
        ("head", "gaussian"),
        ("d_model", True),
        ("attention_heads", 0),
        ("patch_size", -1),
        ("topology", "top_down"),
        ("factor_rank", True),
        ("factor_dynamics", "random_walk"),
        ("event_kernel", "attention"),
        ("event_max_support", [1, 30]),
        ("event_max_support", [True, 30]),
        ("input_normalisation", None),
        ("demand_branch_dropout", True),
        ("covariate_dropout", float("nan")),
        ("price_response_mode", "unbounded"),
        ("price_occurrence_slope_max", 1.0),
        ("static_variable_selection", 1),
        ("embedding_residual_dim", True),
        ("n_sample_paths", 499),
        ("emit", "point"),
        ("zero_share", True),
        ("notes", [1]),
    )
    for field, value in corruptions:
        record = build.as_record()
        record[field] = value
        with pytest.raises(ResolveError, match=field.split("_")[0]):
            ResolvedBuild.from_record(record)


def test_resolved_constructor_refuses_cross_field_contradictions() -> None:
    build = ResolvedBuild.from_specs(DATA, ForecastModelSpec(), zero_share=0.5)
    invalid = (
        {"schema_version": 1},
        {"d_model": 63},
        {"topology": "bottom_up", "factor_rank": 0},
        {"price_response_mode": "monotone_raw_v1"},
    )
    for changes in invalid:
        with pytest.raises(ResolveError):
            ResolvedBuild(**{**build.__dict__, **changes})


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn(); print("PASS " + name)
