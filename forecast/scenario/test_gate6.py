"""Gate 6: the held-out evidence that makes a contrast quotable, and every way it must not.

Run: python3 forecast/scenario/test_gate6.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import torch
from forecast.scenario.gate6 import (
    Gate6Error,
    Gate6Evidence,
    Gate6Thresholds,
    assignment_imbalance,
    campaign_error_bootstrap,
    evaluate,
    pretrend_gap,
)


def well_behaved():
    torch.manual_seed(0)
    treated_pre = 2.0 + torch.arange(10).float() * 0.1 + torch.rand(6, 10) * 0.2
    untreated_pre = 1.9 + torch.arange(10).float() * 0.1 + torch.rand(6, 10) * 0.2
    campaigns = [(f"e{i}", 100.0 + i, 100.0 + i + (2.0 if i % 2 else -2.0))
                 for i in range(6)]
    # Balance is a STANDARDISED mean difference: at tiny n its sampling noise alone exceeds
    # any strict bar (sqrt(2/n) ~ 0.58 at n=6), so the well-behaved fixture draws both groups
    # from ONE distribution at an n where comparable groups can actually look comparable.
    values = (2.5 + torch.rand(100) * 0.4).sort().values
    treated_base, untreated_base = values[0::2], values[1::2]   # interleaved: comparable by
    return campaigns, treated_pre, untreated_pre, treated_base, untreated_base   # construction


def write_record(record: dict[str, object]) -> pathlib.Path:
    import json
    import tempfile

    path = pathlib.Path(tempfile.mkdtemp()) / "gate6.json"
    path.write_text(json.dumps(record))
    return path


def test_well_behaved_evidence_validates_and_is_deterministic() -> None:
    campaigns, tp, up, tb, ub = well_behaved()
    first = evaluate(campaigns=campaigns, treated_pre=tp, untreated_pre=up,
                     treated_baseline=tb, untreated_baseline=ub, seed=3)
    assert first.episode_validated, first.notes
    assert first.error_low <= first.error_high
    again = evaluate(campaigns=campaigns, treated_pre=tp, untreated_pre=up,
                     treated_baseline=tb, untreated_baseline=ub, seed=3)
    assert first.as_record() == again.as_record(), "evidence is not reproducible"


def test_each_violated_bar_blocks_validation_with_its_own_words() -> None:
    """episode_validated is DERIVED from the registered bars, never set: every violated bar
    must block it and say why in words a reader can act on."""
    campaigns, tp, up, tb, ub = well_behaved()

    few = evaluate(campaigns=campaigns[:3], treated_pre=tp, untreated_pre=up,
                   treated_baseline=tb, untreated_baseline=ub)
    assert not few.episode_validated and any("count is not identification" in n
                                             for n in few.notes)

    bad_model = [(cid, realised, realised * 2.0) for cid, realised, _ in campaigns]
    wrong = evaluate(campaigns=bad_model, treated_pre=tp, untreated_pre=up,
                     treated_baseline=tb, untreated_baseline=ub)
    assert not wrong.episode_validated and any("cannot predict a held-out campaign" in n
                                               for n in wrong.notes)

    trending = tp + torch.arange(10).float() * 0.5      # treated accelerating pre-campaign
    pre = evaluate(campaigns=campaigns, treated_pre=trending, untreated_pre=up,
                   treated_baseline=tb, untreated_baseline=ub)
    assert not pre.episode_validated and any("before the lever moved" in n for n in pre.notes)

    skewed = evaluate(campaigns=campaigns, treated_pre=tp, untreated_pre=up,
                      treated_baseline=torch.full((6,), 9.0),
                      untreated_baseline=torch.full((6,), 1.0))
    assert not skewed.episode_validated and any("promote-what-sells" in n
                                                for n in skewed.notes)


def test_the_diagnostics_measure_what_they_claim() -> None:
    """A pretrend is a difference of TRENDS, not of levels; imbalance is a difference of
    LEVELS. Confusing the two flags every store pairing ever made."""
    parallel = pretrend_gap(1.0 + torch.arange(8).float().repeat(5, 1) * 0.2,
                            3.0 + torch.arange(8).float().repeat(5, 1) * 0.2)
    assert parallel < 1e-6, "parallel trends at different LEVELS must not read as a pretrend"
    diverging = pretrend_gap(torch.arange(8).float().repeat(5, 1) * 0.5,
                             torch.arange(8).float().repeat(5, 1) * 0.1)
    assert diverging > 1.0
    balanced = assignment_imbalance(torch.tensor([2.0, 2.1]), torch.tensor([2.05, 1.95]))
    skewed = assignment_imbalance(torch.tensor([9.0, 9.1]), torch.tensor([1.0, 1.1]))
    assert balanced < skewed and skewed > 1.0


def test_the_bootstrap_refuses_a_single_campaign() -> None:
    try:
        campaign_error_bootstrap([0.1])
    except Gate6Error as e:
        assert "certainty" in str(e); return
    raise AssertionError("a zero-width band over one campaign was produced")


def test_thresholds_ride_the_record_so_the_bar_is_citable() -> None:
    campaigns, tp, up, tb, ub = well_behaved()
    strict = Gate6Thresholds(min_campaigns=10)
    evidence = evaluate(
        campaigns=campaigns, treated_pre=tp, untreated_pre=up,
        treated_baseline=tb, untreated_baseline=ub, thresholds=strict)
    assert not evidence.episode_validated
    assert evidence.as_record()["thresholds"]["min_campaigns"] == 10, \
        "the registered bar must travel with the evidence it judged"


def test_malformed_evidence_is_refused() -> None:
    campaigns, tp, up, tb, ub = well_behaved()
    for bad in ([], [("e1", float("nan"), 1.0)], [("e1", 1.0)]):
        try:
            evaluate(campaigns=bad, treated_pre=tp, untreated_pre=up,
                     treated_baseline=tb, untreated_baseline=ub)
        except Gate6Error:
            continue
        raise AssertionError(f"malformed campaigns {bad!r} produced evidence")
    try:
        pretrend_gap(torch.rand(3, 1), torch.rand(3, 5))
    except Gate6Error:
        pass
    else:
        raise AssertionError("a one-day trend was computed")


def test_file_evidence_is_disabled_until_real_external_authority_exists() -> None:
    """The current runner calls this loader directly. Both a producer-shaped record and the
    original string-bool exploit must refuse until registration/source bytes are governed."""
    from forecast.scenario.gate6 import load_evidence

    campaigns, tp, up, tb, ub = well_behaved()
    evidence = evaluate(campaigns=campaigns, treated_pre=tp, untreated_pre=up,
                        treated_baseline=tb, untreated_baseline=ub)
    for record in (evidence.as_record(), {"episode_validated": "false"}):
        try:
            load_evidence(write_record(record))
        except Gate6Error as error:
            message = str(error)
            assert "committed Gate-6 registration" in message
            assert "pin their immutable identities" in message
            continue
        raise AssertionError("an unauthenticated Gate 6 file reached episode_validated")


def test_episode_verdict_and_failure_text_are_derived_from_stable_codes() -> None:
    campaigns, tp, up, tb, ub = well_behaved()
    bad_model = [(campaign_id, realised, realised * 2.0)
                 for campaign_id, realised, _ in campaigns]
    evidence = evaluate(campaigns=bad_model, treated_pre=tp, untreated_pre=up,
                        treated_baseline=tb, untreated_baseline=ub)
    record = evidence.as_record()
    assert record["failure_codes"] == ["median_absolute_relative_error"]
    assert record["episode_validated"] is False
    assert record["notes"] and "cannot predict" in record["notes"][0]


def test_direct_invalid_evidence_cannot_pass_with_empty_notes() -> None:
    campaigns, tp, up, tb, ub = well_behaved()
    valid = evaluate(campaigns=campaigns, treated_pre=tp, untreated_pre=up,
                     treated_baseline=tb, untreated_baseline=ub)
    try:
        Gate6Evidence(
            campaign_ids=valid.campaign_ids,
            relative_errors=tuple(1.0 for _ in valid.relative_errors),
            error_low=valid.error_low, error_high=valid.error_high,
            pretrend=valid.pretrend, imbalance=valid.imbalance, thresholds=valid.thresholds,
            notes=())
    except Gate6Error as error:
        assert "failure notes" in str(error)
    else:
        raise AssertionError("empty free-form notes made invalid direct evidence pass")


def test_duplicate_campaigns_never_count_as_replication() -> None:
    campaigns, tp, up, tb, ub = well_behaved()
    duplicated = list(campaigns)
    duplicated[-1] = (campaigns[0][0], *campaigns[-1][1:])
    try:
        evaluate(campaigns=duplicated, treated_pre=tp, untreated_pre=up,
                 treated_baseline=tb, untreated_baseline=ub)
    except Gate6Error as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("one campaign id counted as two held-out replications")


def test_nonfinite_huge_and_numeric_alias_measurements_are_refused() -> None:
    campaigns, tp, up, tb, ub = well_behaved()
    for bad_value in (True, "1", float("nan"), float("inf"), 10**400):
        bad_campaigns = list(campaigns)
        bad_campaigns[0] = ("e0", bad_value, 1.0)
        try:
            evaluate(campaigns=bad_campaigns, treated_pre=tp, untreated_pre=up,
                     treated_baseline=tb, untreated_baseline=ub)
        except Gate6Error:
            continue
        raise AssertionError(f"campaign measurement {bad_value!r} was coerced and accepted")

    for constructor in (
        lambda: Gate6Thresholds(min_campaigns=True),
        lambda: Gate6Thresholds(min_campaigns=5.0),
        lambda: Gate6Thresholds(max_pretrend=0),
        lambda: Gate6Thresholds(max_imbalance=float("nan")),
    ):
        try:
            constructor()
        except Gate6Error:
            continue
        raise AssertionError("a threshold bool/int/float alias or non-finite value was accepted")

    valid = evaluate(campaigns=campaigns, treated_pre=tp, untreated_pre=up,
                     treated_baseline=tb, untreated_baseline=ub)
    fields = dict(
        campaign_ids=valid.campaign_ids, relative_errors=valid.relative_errors,
        error_low=valid.error_low, error_high=valid.error_high, pretrend=valid.pretrend,
        imbalance=valid.imbalance, thresholds=valid.thresholds, notes=valid.notes)
    for field, bad_value in (
        ("error_low", 0),
        ("error_high", 10**400),
        ("pretrend", float("nan")),
        ("imbalance", True),
        ("relative_errors", (True, *valid.relative_errors[1:])),
    ):
        try:
            Gate6Evidence(**{**fields, field: bad_value})
        except Gate6Error:
            continue
        raise AssertionError(f"direct evidence accepted invalid {field}={bad_value!r}")


def test_nonfinite_diagnostic_inputs_refuse_instead_of_passing_comparisons() -> None:
    campaigns, tp, up, tb, ub = well_behaved()
    poisoned = tp.clone()
    poisoned[0, 0] = float("nan")
    try:
        evaluate(campaigns=campaigns, treated_pre=poisoned, untreated_pre=up,
                 treated_baseline=tb, untreated_baseline=ub)
    except Gate6Error as error:
        assert "finite real demand" in str(error)
    else:
        raise AssertionError("NaN pretrend compared false against every bar and passed")

    for fn, first, second in (
        (pretrend_gap, torch.ones(2, 3, dtype=torch.bool), torch.ones(2, 3)),
        (assignment_imbalance, torch.tensor([float("inf")]), torch.ones(2)),
    ):
        try:
            fn(first, second)
        except Gate6Error:
            continue
        raise AssertionError("a non-real or non-finite diagnostic measurement was accepted")


def test_loader_does_not_replay_version_sensitive_bootstrap_rng() -> None:
    from unittest import mock

    from forecast.scenario.gate6 import load_evidence

    campaigns, tp, up, tb, ub = well_behaved()
    evidence = evaluate(campaigns=campaigns, treated_pre=tp, untreated_pre=up,
                        treated_baseline=tb, untreated_baseline=ub)
    path = write_record(evidence.as_record())
    with mock.patch(
        "forecast.scenario.gate6.campaign_error_bootstrap",
        side_effect=AssertionError("loader replayed Torch RNG"),
    ) as bootstrap:
        try:
            load_evidence(path)
        except Gate6Error as error:
            assert "consumption is disabled" in str(error)
        else:
            raise AssertionError("Gate 6 file loading was enabled without external authority")
    bootstrap.assert_not_called()


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn(); print("PASS " + name)
