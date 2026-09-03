"""Tests for the protocol artifacts.

Case table per function: the pass path, each distinct failure path, the edge cases. The guards
here are the ones that make a gate a gate, so each is tested independently -- a guard nothing
would notice failing is not a guard.
"""
import contextlib
import json
import multiprocessing as mp
import pathlib
import queue
import sys
import tempfile
from dataclasses import asdict, replace
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import autoforecast.protocol as P

MF = dict(
    name="phase0", target_estimand="sales_given_availability",
    # Temporally separated: development origins' encoder windows AND horizons end before the
    # acceptance origin begins. A same-year split can leak in both directions:
    # forward through overlapping horizons, backward through encoder windows.
    origins=("2024-06-01", "2024-09-01", "2024-12-01", "2026-08-01"),
    acceptance_origins=("2026-08-01",), encoder_days=364, horizon_days=28,
    scoring_grain=("store", "product", "day"), eligible_rule="exposure_daily.is_ranged",
    zero_fill_rule="absent row inside eligible window = true zero",
    source_cutoffs={"sales": "2026-08-01T00:00:00Z"}, clients=("dataset_a",),
    excluded_clients={"dataset_b": "feed gate"},
)


def _tmp(monkey_registry=True):
    d = pathlib.Path(tempfile.mkdtemp())
    P.REGISTRY, P.RESULTS = d / "registered", d / "results"
    return d


# ---------------------------------------------------------------- content_hash
def test_generic_observed_estimand_has_distinct_protocol_identity():
    observed = P.EvaluationManifest(**{**MF, "target_estimand": "observed_nonnegative"})
    sales = P.EvaluationManifest(**MF)
    assert P.content_hash(asdict(observed)) != P.content_hash(asdict(sales))


def test_hash_is_independent_of_key_order():
    assert P.content_hash({"a": 1, "b": 2}) == P.content_hash({"b": 2, "a": 1})


def test_hash_changes_when_a_value_changes():
    assert P.content_hash({"a": 1}) != P.content_hash({"a": 2})


def test_protocol_hash_refuses_non_json_values_and_preserves_json_array_identity() -> None:
    assert P.content_hash({"values": [1, 2]}) == P.content_hash({"values": (1, 2)})
    for value in ({1: "numeric key"}, {"opaque": object()}, {"bad": float("nan")}):
        try:
            P.content_hash(value)
        except (TypeError, ValueError):
            continue
        raise AssertionError(f"protocol content hash accepted non-canonical JSON {value!r}")


# ---------------------------------------------------------------- EvaluationManifest
def test_manifest_pass_path():
    m = P.EvaluationManifest(**MF)
    assert len(m.hash) == 64


def test_manifest_latent_demand_source_is_an_exact_optional_nonblank_string() -> None:
    latent = P.EvaluationManifest(
        **{**MF, "target_estimand": "unconstrained_demand",
           "latent_demand_source": "sealed-latent-actuals.parquet"})
    assert latent.latent_demand_source == "sealed-latent-actuals.parquet"
    for invalid in (True, 1, "", "  "):
        try:
            P.EvaluationManifest(**{**MF, "latent_demand_source": invalid})
        except ValueError as exc:
            assert "latent_demand_source" in str(exc), str(exc)
        else:
            raise AssertionError(f"manifest accepted latent_demand_source={invalid!r}")
    try:
        P.EvaluationManifest(**{**MF, "target_estimand": "unconstrained_demand"})
    except ValueError as exc:
        assert "latent_demand_source" in str(exc), str(exc)
    else:
        raise AssertionError("unconstrained demand accepted no latent actuals source")


def test_manifest_integer_fields_refuse_bool_and_float_aliases() -> None:
    for field in ("horizon_days", "encoder_days", "max_gate_attempts"):
        original = MF.get(field, 3)
        for alias in (float(original), bool(original)):
            try:
                P.EvaluationManifest(**{**MF, field: alias})
            except ValueError:
                continue
            raise AssertionError(f"manifest accepted {field} type alias {alias!r}")


def test_manifest_collection_fields_refuse_shape_aliases() -> None:
    cases = (
        {"origins": "2026-08-01"},
        {"acceptance_origins": "2026-08-01"},
        {"scoring_grain": "store_product_day"},
        {"clients": "m5"},
        {"source_cutoffs": [("sales", "2026-08-01T00:00:00Z")]},
        {"excluded_clients": [("dataset_b", "feed gate")]},
        {"campaign_of_date": [("2026-08-01", "campaign")]},
    )
    for change in cases:
        try:
            P.EvaluationManifest(**{**MF, **change})
        except ValueError:
            continue
        raise AssertionError(f"manifest accepted a collection shape alias: {change!r}")


def test_source_cutoffs_require_timezone_aware_iso_timestamps() -> None:
    for cutoff in ("2026-08-01", "2026-08-01T00:00:00"):
        try:
            P.EvaluationManifest(**{**MF, "source_cutoffs": {"sales": cutoff}})
        except ValueError as exc:
            assert "UTC offset" in str(exc), str(exc)
        else:
            raise AssertionError(f"source cutoff did not identify one instant: {cutoff!r}")
    assert P.EvaluationManifest(
        **{**MF, "source_cutoffs": {"sales": "2026-08-01T05:30:00+05:30"}})


def test_manifest_binds_every_origin_grid_hash_and_changes_with_one_grid() -> None:
    hashes = {origin: "a" * 64 for origin in MF["origins"]}
    bound = P.EvaluationManifest(**MF, eligible_grid_hashes=hashes)
    changed = dict(hashes)
    changed[MF["origins"][0]] = "b" * 64
    assert bound.hash != P.EvaluationManifest(
        **MF, eligible_grid_hashes=changed).hash
    assert bound.hash != P.EvaluationManifest(**MF).hash


def test_manifest_grid_hash_mapping_cannot_be_partial_or_malformed() -> None:
    partial = {origin: "a" * 64 for origin in MF["origins"][:-1]}
    for hashes in (partial, {origin: "NOT-SHA256" for origin in MF["origins"]}):
        try:
            P.EvaluationManifest(**MF, eligible_grid_hashes=hashes)
        except ValueError as exc:
            assert "eligible_grid_hashes" in str(exc)
        else:
            raise AssertionError("manifest accepted incomplete or malformed grid hashes")


def test_manifest_binds_and_validates_missing_forecast_policy() -> None:
    hashes = {origin: "a" * 64 for origin in MF["origins"]}
    strict = P.EvaluationManifest(**MF, eligible_grid_hashes=hashes)
    fallback = P.EvaluationManifest(
        **MF, eligible_grid_hashes=hashes, missing_forecast_policy="zero")
    assert fallback.hash != strict.hash
    assert fallback._record()["missing_forecast_policy"] == "zero"
    for change in (
        {"missing_forecast_policy": "zero"},
        {"eligible_grid_hashes": hashes, "missing_forecast_policy": "mean"},
    ):
        try:
            P.EvaluationManifest(**MF, **change)
        except ValueError as exc:
            assert "missing_forecast_policy" in str(exc)
        else:
            raise AssertionError("manifest accepted an undefined or unbound fallback policy")


def test_pre_grid_manifest_is_rejected_without_migrating_its_hash() -> None:
    """Historical input stays intact and cannot substitute for a complete current manifest."""
    blob = {
        "manifest": {
            "name": "m5_phase2",
            "target_estimand": "sales_given_availability",
            "origins": ["2016-01-03", "2016-01-31", "2016-02-28", "2016-04-24"],
            "horizon_days": 28,
            "scoring_grain": ["store", "product", "day"],
            "eligible_rule": "m5.exposure_from_first_sale (first sale of the product in the store)",
            "zero_fill_rule": "M5 is dense; an absent row is a source failure, not a zero",
            "source_cutoffs": {"m5": "2016-05-22T00:00:00Z"},
            "clients": ["m5"],
            "excluded_clients": {},
            "encoder_days": 365,
            "max_gate_attempts": 3,
            "campaign_of_date": {},
            "latent_demand_source": None,
            "acceptance_origins": ["2016-04-24"],
        },
        "hash": "88b9e55aed6fa5e21f1c722ccdbb10f51e5764e97fd8623d7b7a398f420cfd0f",
    }
    _tmp()
    P.REGISTRY.mkdir(parents=True)
    path = P.REGISTRY / "manifest_m5_phase2.json"
    original = json.dumps(blob)
    path.write_text(original)
    try:
        P._manifest_with_hash(blob["hash"])
    except RuntimeError as exc:
        assert "every current manifest field" in str(exc), str(exc)
    else:
        raise AssertionError("a pre-grid manifest was reconstructed with current defaults")
    assert path.read_text() == original


def test_manifest_serializes_default_fields_and_refuses_their_omission() -> None:
    _tmp()
    manifest = P.EvaluationManifest(**MF)
    path = manifest.save()
    current = json.loads(path.read_text())
    assert current["manifest"]["eligible_grid_hashes"] == {}
    assert current["manifest"]["missing_forecast_policy"] == "error"
    assert P._manifest_with_hash(manifest.hash) == manifest
    for omitted in ("eligible_grid_hashes", "missing_forecast_policy", "campaign_of_date"):
        record = dict(current["manifest"])
        record.pop(omitted)
        digest = P.content_hash(record)
        path.write_text(json.dumps({"manifest": record, "hash": digest}))
        try:
            P._manifest_with_hash(digest)
        except RuntimeError as exc:
            assert "every current manifest field" in str(exc), str(exc)
        else:
            raise AssertionError(f"manifest record filled omitted {omitted}")


def test_manifest_rejects_no_origins():
    try:
        P.EvaluationManifest(**{**MF, "origins": ()})
    except ValueError:
        return
    raise AssertionError("a manifest with no origins must be refused")


def test_manifest_rejects_duplicate_origins():
    try:
        P.EvaluationManifest(**{**MF, "origins": ("2026-05-01", "2026-05-01", "2026-07-01")})
    except ValueError:
        return
    raise AssertionError("duplicate origins silently double-weight a date")


def test_manifest_rejects_empty_scoring_grain():
    try:
        P.EvaluationManifest(**{**MF, "scoring_grain": ()})
    except ValueError:
        return
    raise AssertionError("scoring grain is what makes the metric honest")


def test_manifest_rejects_malformed_claim_bearing_fields():
    """Runtime types are not enforcement: invalid manifests used to hash and score normally."""
    bad = [
        {"target_estimand": "profit"},
        {"origins": ("not-a-date", "2026-06-01", "2026-07-01")},
        {"horizon_days": 0},
        {"clients": ()},
        {"source_cutoffs": {}},
        {"eligible_rule": ""},
        {"zero_fill_rule": ""},
    ]
    for change in bad:
        try:
            P.EvaluationManifest(**{**MF, **change})
        except ValueError:
            continue
        raise AssertionError(f"malformed manifest field was accepted: {change}")


def test_manifest_rejects_client_both_included_and_excluded():
    try:
        P.EvaluationManifest(**{**MF, "clients": ("dataset_a", "dataset_b")})
    except ValueError:
        return
    raise AssertionError("a client cannot be both in the cohort and quarantined")


def test_manifest_refuses_silent_overwrite_with_different_content():
    _tmp()
    P.EvaluationManifest(**MF).save()
    try:
        P.EvaluationManifest(**{**MF, "horizon_days": 14}).save()
    except ValueError:
        return
    raise AssertionError("a frozen manifest must not be editable -- results reference it")


def test_manifest_rewrite_with_identical_content_is_allowed():
    _tmp()
    P.EvaluationManifest(**MF).save()
    P.EvaluationManifest(**MF).save()          # idempotent: same bytes, no error


# ---------------------------------------------------------------- RegisteredGate
GATE = dict(phase="phase2", metric="MASE",
            comparators=("repaired_incumbent", "seasonal_naive"),
            min_improvement=0.05, direction="lower_is_better",
            rationale="below 5% the operational benefit does not cover retraining cost",
            registered_on="2026-08-22")


def test_gate_round_trips():
    _tmp()
    P.RegisteredGate(**GATE).save()
    assert P.RegisteredGate.load("phase2").min_improvement == 0.05


def test_gate_comparator_namespace_refuses_a_bare_string() -> None:
    try:
        P.RegisteredGate(**{**GATE, "comparators": "ab"})
    except ValueError:
        return
    raise AssertionError("a comparator string was split into one-character comparator names")


def test_extra_gate_binds_its_measurement_definition_into_the_hash() -> None:
    criterion = P.ExtraCriterion("coverage_gap", 0.05, "lower_is_better")
    first = P.RegisteredGate(**GATE, extra_criteria=(criterion,), measurement_contract={
        "coverage_gap": {"lead_time_days": 5, "review_period_days": 2,
                         "nominal_levels": [0.5, 0.8, 0.95, 0.99], "n_paths": 1000}})
    changed = P.RegisteredGate(**GATE, extra_criteria=(criterion,), measurement_contract={
        "coverage_gap": {"lead_time_days": 7, "review_period_days": 2,
                         "nominal_levels": [0.5, 0.8, 0.95, 0.99], "n_paths": 1000}})
    assert first.hash != changed.hash
    _tmp()
    first.save()
    assert P.RegisteredGate.load("phase2").measurement_contract == first.measurement_contract


def test_gate_numeric_contracts_refuse_unrepresentable_json_integers() -> None:
    huge = 10**400
    for build in (
        lambda: P.ExtraCriterion("coverage_gap", huge, "lower_is_better"),
        lambda: P.ExtraCriterion("coverage_gap", 0.1, "lower_is_better").evaluate(huge),
        lambda: P.RegisteredGate(**{**GATE, "min_improvement": huge}),
        lambda: P.RegisteredGate(**GATE).passes(_both(1.0), huge),
        lambda: P.RegisteredGate(**GATE).passes(_both(huge), 0.1),
    ):
        try:
            build()
        except ValueError:
            continue
        raise AssertionError("an unrepresentable integer entered gate arithmetic")


def test_extra_gate_cannot_define_only_a_threshold_without_a_measurement() -> None:
    _tmp()
    try:
        P.RegisteredGate(**GATE, extra_criteria=(
            P.ExtraCriterion("coverage_gap", 0.05, "lower_is_better"),))
    except ValueError as exc:
        assert "measurement_contract" in str(exc)
        return
    raise AssertionError("registered an extra threshold whose measurement can still be changed")


def test_gate_save_rechecks_a_measurement_mapping_mutated_after_construction() -> None:
    _tmp()
    contract = {"coverage_gap": {"lead_time_days": 5}}
    gate = P.RegisteredGate(
        **GATE, extra_criteria=(P.ExtraCriterion("coverage_gap", 0.05, "lower_is_better"),),
        measurement_contract=contract)
    contract.clear()
    try:
        gate.save()
    except ValueError as exc:
        assert "measurement_contract" in str(exc), str(exc)
    else:
        raise AssertionError("a gate saved an extra threshold after its measurement was removed")
    assert not (P.REGISTRY / "phase2.json").exists()


def test_gate_records_require_current_default_fields() -> None:
    for extras in (False, True):
        _tmp()
        options = {} if not extras else {
            "extra_criteria": (P.ExtraCriterion("coverage_gap", 0.05, "lower_is_better"),),
            "measurement_contract": {"coverage_gap": {"lead_time_days": 5}},
        }
        gate = P.RegisteredGate(**GATE, **options)
        path = gate.save()
        current = json.loads(path.read_text())
        assert current["gate"]["measurement_contract"] == gate.measurement_contract
        assert P.RegisteredGate.load("phase2") == gate
        for omitted in ("measurement_contract", "extra_criteria", "per_client_policy", "relation"):
            record = dict(current["gate"])
            record.pop(omitted)
            payload = json.dumps({"gate": record, "hash": P.content_hash(record)})
            path.write_text(payload)
            try:
                P.RegisteredGate.load("phase2")
            except ValueError as exc:
                assert "every current registered gate field" in str(exc), str(exc)
            else:
                raise AssertionError(f"gate record filled omitted {omitted}")
            try:
                gate.save()
            except ValueError as exc:
                assert "different threshold" in str(exc), str(exc)
            else:
                raise AssertionError("an old gate record was overwritten with a new identity")
            assert path.read_text() == payload


def test_measurement_contract_must_match_metrics_and_be_canonical_json() -> None:
    criterion = P.ExtraCriterion("coverage_gap", 0.05, "lower_is_better")
    malformed = (
        {"another_metric": {"lead_time_days": 5}},
        {"coverage_gap": {}},
        {"coverage_gap": {"lead_time_days": float("nan")}},
        {"coverage_gap": {"opaque": object()}},
        {"coverage_gap": {1: "numeric keys are not canonical JSON object fields"}},
    )
    for contract in malformed:
        try:
            P.RegisteredGate(**GATE, extra_criteria=(criterion,),
                             measurement_contract=contract)
        except ValueError:
            continue
        raise AssertionError(f"accepted malformed measurement contract {contract!r}")


def test_result_must_attest_the_exact_registered_measurement_contract() -> None:
    _tmp()
    criterion = P.ExtraCriterion("coverage_gap", 0.05, "lower_is_better")
    contract = {"coverage_gap": {"lead_time_days": 5, "n_paths": 1000}}
    P.RegisteredGate(
        **GATE, extra_criteria=(criterion,), measurement_contract=contract).save()
    m = P.EvaluationManifest(**MF); m.save()
    for evidence in (
        None,
        {"coverage_gap": {"lead_time_days": 7, "n_paths": 1000}},
        {"coverage_gap": {"lead_time_days": 5.0, "n_paths": 1000}},
        {"coverage_gap": {"lead_time_days": 5, "n_paths": 1000.0}},
    ):
        try:
            _gate("phase2", m, "MASE", _both(1.0), 0.5,
                  extra_scores={"coverage_gap": 0.01}, measurement_evidence=evidence)
        except ValueError as exc:
            assert "measurement_evidence" in str(exc)
            continue
        raise AssertionError("a result attested a different extra-score measurement")
    result = _gate(
        "phase2", m, "MASE", _both(1.0), 0.5,
        extra_scores={"coverage_gap": 0.01}, measurement_evidence=contract)
    assert result.measurement_evidence == contract


def test_measurement_contract_treats_tuple_and_list_as_the_same_json_array() -> None:
    _tmp()
    criterion = P.ExtraCriterion("coverage_gap", 0.05, "lower_is_better")
    contract = {"coverage_gap": {"nominal_levels": (0.5, 0.8), "n_paths": 1000}}
    P.RegisteredGate(
        **GATE, extra_criteria=(criterion,), measurement_contract=contract).save()
    manifest = P.EvaluationManifest(**MF)
    manifest.save()
    result = _gate(
        "phase2", manifest, "MASE", _both(1.0), 0.5,
        extra_scores={"coverage_gap": 0.01}, measurement_evidence=contract)
    result.save()
    assert P.require_passed("phase2", manifest.hash).passed


def test_gate_result_integer_and_boolean_evidence_is_type_exact_on_reload() -> None:
    aliases = (
        ("passed", 1),
        ("passed", 1.0),
        ("expected_cells", 1000.0),
        ("scored_cells", 1000.0),
        ("scored_origins", ACC + ACC),
        ("scored_origins", dict.fromkeys(ACC)),
        ("extra_scores", [("unregistered_posthoc_metric", 999.0)]),
        ("attempt_number", None),
        ("attempt_number", True),
        ("attempt_number", 1.0),
        ("attempt_number", 0),
    )
    for field, alias in aliases:
        _tmp()
        gate = P.RegisteredGate(**GATE)
        gate.save()
        manifest = P.EvaluationManifest(**MF)
        manifest.save()
        result = _gate("phase2", manifest, "MASE", _both(1.0), 0.5)
        path = result.save()
        values = asdict(result)
        values[field] = alias
        path.write_text(json.dumps(values | {"hash": P.content_hash(values)}))
        try:
            P.require_passed("phase2", manifest.hash)
        except RuntimeError:
            continue
        raise AssertionError(f"gate result accepted {field} type alias {alias!r}")


def test_require_passed_rechecks_measurement_evidence_not_only_the_gate_hash() -> None:
    _tmp()
    criterion = P.ExtraCriterion("coverage_gap", 0.05, "lower_is_better")
    contract = {"coverage_gap": {"lead_time_days": 5, "n_paths": 1000}}
    gate = P.RegisteredGate(
        **GATE, extra_criteria=(criterion,), measurement_contract=contract)
    gate.save()
    m = P.EvaluationManifest(**MF); m.save()
    result = _gate("phase2", m, "MASE", _both(1.0), 0.5,
                   extra_scores={"coverage_gap": 0.01}, measurement_evidence=contract)
    replace(
        result,
        measurement_evidence={"coverage_gap": {"lead_time_days": 7, "n_paths": 1000}},
    ).save()
    try:
        P.require_passed("phase2", m.hash)
    except RuntimeError as exc:
        assert "measurement_contract" in str(exc)
        return
    raise AssertionError("a result with different measurement evidence unlocked the next phase")


def test_loading_an_unregistered_gate_fails():
    _tmp()
    try:
        P.RegisteredGate.load("phase2")
    except FileNotFoundError:
        return
    raise AssertionError("a gate registered after the run proves nothing; absence must raise")


def test_editing_a_registered_gate_is_detected():
    _tmp()
    path = P.RegisteredGate(**GATE).save()
    blob = json.loads(path.read_text())
    blob["gate"]["min_improvement"] = 0.0            # move the bar to whatever happened
    path.write_text(json.dumps(blob))
    try:
        P.RegisteredGate.load("phase2")
    except ValueError:
        return
    raise AssertionError("a tampered threshold must be detected by hash")


def test_re_registering_a_different_threshold_is_refused():
    _tmp()
    P.RegisteredGate(**GATE).save()
    try:
        P.RegisteredGate(**{**GATE, "min_improvement": 0.0}).save()
    except ValueError:
        return
    raise AssertionError("silently re-registering a lower bar is the failure being prevented")


ACC = ("2026-08-01",)                                  # the manifest's acceptance set (O4)
GRID = dict(expected_cells=1000, scored_cells=1000)    # full grid coverage (O7)


def _gate(phase, m, metric, comps, candidate_score=None, **kw):
    """evaluate_gate with the O4 acceptance origins and the O7 coverage proof supplied."""
    if candidate_score is None:
        candidate_score = kw.pop("candidate_score")
    return P.evaluate_gate(phase, m, metric, comps, candidate_score,
                           scored_origins=ACC, **{**GRID, **kw})


def _both(v): return {"repaired_incumbent": v, "seasonal_naive": v}


def _registration_object(kind: str, fields: dict):
    if kind == "manifest":
        return P.EvaluationManifest(**fields)
    if kind == "gate":
        return P.RegisteredGate(**fields)
    raise AssertionError(f"unknown registration kind {kind!r}")


def _registration_path(kind: str, fields: dict) -> pathlib.Path:
    if kind == "manifest":
        return P.REGISTRY / f"manifest_{fields['name']}.json"
    if kind == "gate":
        return P.REGISTRY / f"{fields['phase']}.json"
    raise AssertionError(f"unknown registration kind {kind!r}")


def _registration_document(kind: str, fields: dict) -> dict[str, object]:
    registration = _registration_object(kind, fields)
    record = registration._record()  # noqa: SLF001 - exact producer envelope fixture
    document = {kind: record, "hash": P.content_hash(record)}
    # Match the JSON-native representation returned by ``load_exact_json``: dataclass tuples
    # become arrays on disk even though they have the same canonical protocol identity.
    return json.loads(json.dumps(document, allow_nan=False))


def _registry_save_race_worker(
        label: str,
        kind: str,
        fields: dict,
        registry: str,
        results: str,
        start,
        ready,
        lock_boundary,
        enter_locks,
        write_boundary,
        release_write,
        outcomes) -> None:
    """Pause one real process at the registry lock and atomic-publication boundaries."""
    P.REGISTRY = pathlib.Path(registry)
    P.RESULTS = pathlib.Path(results)
    registration = _registration_object(kind, fields)
    real_lock = P._attempt_ledger_lock  # noqa: SLF001
    real_write = P._atomic_write_text  # noqa: SLF001

    @contextlib.contextmanager
    def synchronized_lock():
        lock_boundary.put(label)
        if not enter_locks.wait(timeout=10):
            raise RuntimeError("registry lock-boundary timeout")
        with real_lock():
            yield

    def paused_write(path: pathlib.Path, payload: str) -> None:
        write_boundary.put(label)
        if not release_write.wait(timeout=10):
            raise RuntimeError("registry write-boundary timeout")
        real_write(path, payload)

    P._attempt_ledger_lock = synchronized_lock
    P._atomic_write_text = paused_write
    ready.put(label)
    if not start.wait(timeout=10):
        outcomes.put((label, "worker-error", "start timeout"))
        return
    try:
        registration.save()
    except ValueError as exc:
        outcomes.put((label, "refused", str(exc)))
    except Exception as exc:  # pragma: no cover - exact child failure is reported to the parent
        outcomes.put((label, "worker-error", f"{type(exc).__name__}: {exc}"))
    else:
        outcomes.put((label, "saved", registration.hash))


def _atomic_registry_writer(
        kind: str,
        fields: dict,
        registry: str,
        results: str,
        replace_ready,
        release_replace,
        outcomes) -> None:
    """Pause after the replacement bytes are fsynced but before the atomic rename."""
    P.REGISTRY = pathlib.Path(registry)
    P.RESULTS = pathlib.Path(results)
    registration = _registration_object(kind, fields)
    target = _registration_path(kind, fields)
    real_replace = P.os.replace

    def paused_replace(source, destination):
        if pathlib.Path(destination) == target:
            replace_ready.put(True)
            if not release_replace.wait(timeout=10):
                raise RuntimeError("registry replace-boundary timeout")
        return real_replace(source, destination)

    P.os.replace = paused_replace
    try:
        registration.save()
    except Exception as exc:  # pragma: no cover - exact child failure is reported to the parent
        outcomes.put(("worker-error", f"{type(exc).__name__}: {exc}"))
    else:
        outcomes.put(("saved", registration.hash))


def _run_registry_save_race(kind: str, fields: tuple[dict, dict]) -> tuple[list[tuple], list[str]]:
    """Run two synchronized registrations and return their outcomes/write order."""
    assert "fork" in mp.get_all_start_methods(), "flock protocol tests require a fork-capable OS"
    context = mp.get_context("fork")
    start = context.Event()
    enter_locks = context.Event()
    release_write = context.Event()
    ready = context.Queue()
    lock_boundary = context.Queue()
    write_boundary = context.Queue()
    outcomes = context.Queue()
    processes = [
        context.Process(
            target=_registry_save_race_worker,
            args=(label, kind, registration_fields, str(P.REGISTRY), str(P.RESULTS),
                  start, ready, lock_boundary, enter_locks, write_boundary, release_write,
                  outcomes),
        )
        for label, registration_fields in zip(("first", "second"), fields, strict=True)
    ]
    for process in processes:
        process.start()
    lock_path = P.RESULTS / ".attempts.jsonl.lock"
    try:
        assert {ready.get(timeout=10) for _ in processes} == {"first", "second"}
        start.set()
        # Both processes reached the same lock call before either may enter it. This forces a
        # check-before-lock mutation to observe the same absent generation in both processes.
        assert {lock_boundary.get(timeout=10) for _ in processes} == {"first", "second"}
        enter_locks.set()
        first_writer = write_boundary.get(timeout=10)

        # A nonblocking cross-process probe proves the persistent inode remains locked THROUGH the
        # atomic writer. No scheduler delay or elapsed-time assertion is used as lock evidence.
        with lock_path.open("a+") as probe:
            try:
                P.fcntl.flock(probe.fileno(), P.fcntl.LOCK_EX | P.fcntl.LOCK_NB)
            except BlockingIOError:
                pass
            else:
                P.fcntl.flock(probe.fileno(), P.fcntl.LOCK_UN)
                raise AssertionError("registry lock was released before atomic publication")
        lock_inode = lock_path.stat().st_ino
        release_write.set()
        observed = [outcomes.get(timeout=10) for _ in processes]
    finally:
        start.set()
        enter_locks.set()
        release_write.set()
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    assert all(process.exitcode == 0 for process in processes), observed
    writers = [first_writer]
    while True:
        try:
            writers.append(write_boundary.get_nowait())
        except queue.Empty:
            break
    assert lock_path.stat().st_ino == lock_inode, "registry publication replaced its lock inode"
    return observed, writers


def test_conflicting_first_registrations_are_serialized_across_processes() -> None:
    """Only one different definition can win the first-write race for each authority type."""
    cases = (
        ("manifest", MF, {**MF, "horizon_days": 14}),
        ("gate", GATE, {**GATE, "min_improvement": 0.10}),
    )
    for kind, first, second in cases:
        _tmp()
        observed, writers = _run_registry_save_race(kind, (first, second))
        assert sorted(item[1] for item in observed) == ["refused", "saved"], observed
        assert len(writers) == 1, (kind, writers)
        stored = P.load_exact_json(_registration_path(kind, first))
        saved_hash = next(item[2] for item in observed if item[1] == "saved")
        assert stored["hash"] == saved_hash


def test_identical_concurrent_registrations_remain_idempotent() -> None:
    """Serialization does not turn an identical concurrent save into a false conflict."""
    for kind, fields in (("manifest", MF), ("gate", GATE)):
        _tmp()
        observed, writers = _run_registry_save_race(kind, (fields, fields))
        assert [item[1] for item in observed].count("saved") == 2, observed
        assert sorted(writers) == ["first", "second"], writers
        assert P.load_exact_json(_registration_path(kind, fields)) == \
            _registration_document(kind, fields)


def test_registry_readers_see_only_complete_old_or_new_json() -> None:
    """An idempotent rewrite atomically replaces distinct formatting, never partial JSON."""
    assert "fork" in mp.get_all_start_methods(), "atomic protocol tests require a fork-capable OS"
    context = mp.get_context("fork")
    for kind, fields in (("manifest", MF), ("gate", GATE)):
        _tmp()
        target = _registration_path(kind, fields)
        target.parent.mkdir(parents=True)
        document = _registration_document(kind, fields)
        old = json.dumps(document, sort_keys=True, separators=(",", ":"))
        new = json.dumps(document, indent=1, allow_nan=False)
        assert old != new
        target.write_text(old)
        replace_ready = context.Queue()
        release_replace = context.Event()
        outcomes = context.Queue()
        process = context.Process(
            target=_atomic_registry_writer,
            args=(kind, fields, str(P.REGISTRY), str(P.RESULTS), replace_ready,
                  release_replace, outcomes),
        )
        process.start()
        try:
            assert replace_ready.get(timeout=10) is True
            observed_before = target.read_text()
            assert observed_before == old
            assert json.loads(observed_before) == document
            release_replace.set()
            assert outcomes.get(timeout=10)[0] == "saved"
        finally:
            release_replace.set()
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        assert process.exitcode == 0
        observed_after = target.read_text()
        assert observed_after == new
        assert json.loads(observed_after) == document
        assert not list(target.parent.glob(f".{target.name}.*.tmp"))


def _concurrent_final_attempt_worker(
        registry: str,
        results: str,
        manifest_fields: dict,
        start,
        ready,
        lock_calls,
        outcomes) -> None:
    """Submit one final-budget attempt from a distinct process."""
    P.REGISTRY = pathlib.Path(registry)
    P.RESULTS = pathlib.Path(results)
    manifest = P.EvaluationManifest(**manifest_fields)
    real_flock = P.fcntl.flock

    def report_exclusive_lock(descriptor: int, operation: int):
        if operation == P.fcntl.LOCK_EX:
            lock_calls.put(True)
        return real_flock(descriptor, operation)

    P.fcntl.flock = report_exclusive_lock
    ready.put(True)
    if not start.wait(timeout=10):
        outcomes.put(("worker-error", "start timeout"))
        return
    try:
        result = _gate("phase2", manifest, "MASE", _both(1.0), 0.5)
    except RuntimeError as exc:
        outcomes.put(("spent", str(exc)))
    except Exception as exc:  # pragma: no cover - reported to the parent with its exact type
        outcomes.put(("worker-error", f"{type(exc).__name__}: {exc}"))
    else:
        outcomes.put(("recorded", result.passed))


def _ordered_attempt_save_worker(
        label: str,
        registry: str,
        results: str,
        manifest_fields: dict,
        candidate_score: float,
        evaluated,
        save_now,
        outcomes,
        save_again=None) -> None:
    """Evaluate in one process, then publish only when the parent releases each save point."""
    P.REGISTRY = pathlib.Path(registry)
    P.RESULTS = pathlib.Path(results)
    manifest = P.EvaluationManifest(**manifest_fields)
    try:
        result = _gate("phase2", manifest, "MASE", _both(1.0), candidate_score)
        evaluated.put((label, result.attempt_number, result.passed))
        if not save_now.wait(timeout=10):
            outcomes.put((label, "worker-error", "first save timeout"))
            return
        try:
            result.save()
        except RuntimeError as exc:
            outcomes.put((label, "refused", str(exc)))
        else:
            outcomes.put((label, "published", result.attempt_number))
        if save_again is not None:
            if not save_again.wait(timeout=10):
                outcomes.put((label, "worker-error", "second save timeout"))
                return
            try:
                result.save()
            except RuntimeError as exc:
                outcomes.put((label, "refused-again", str(exc)))
            else:
                outcomes.put((label, "republished", result.attempt_number))
    except Exception as exc:  # pragma: no cover - exact child failure is reported to the parent
        outcomes.put((label, "worker-error", f"{type(exc).__name__}: {exc}"))


def test_acceptance_origin_order_is_irrelevant_but_duplicates_are_refused() -> None:
    """The reservation is an unordered set of unique dates, not an execution-order contract."""
    _tmp()
    P.RegisteredGate(**GATE).save()
    acceptance = ("2026-08-01", "2027-08-01")
    manifest = P.EvaluationManifest(**{
        **MF,
        "name": "two_acceptance_origins",
        "origins": (*MF["origins"], acceptance[1]),
        "acceptance_origins": acceptance,
    })
    manifest.save()
    forward_result = P.evaluate_gate(
        "phase2", manifest, "MASE", _both(1.0), 0.5,
        scored_origins=acceptance, **GRID)
    reversed_result = P.evaluate_gate(
        "phase2", manifest, "MASE", _both(1.0), 0.5,
        scored_origins=tuple(reversed(acceptance)), **GRID)
    assert forward_result.scored_origins == reversed_result.scored_origins == tuple(
        sorted(acceptance))
    forward_evidence = asdict(forward_result)
    reversed_evidence = asdict(reversed_result)
    forward_evidence.pop("attempt_number")
    reversed_evidence.pop("attempt_number")
    assert P.content_hash(forward_evidence) == P.content_hash(reversed_evidence), \
        "iteration order created two claim identities for one acceptance set"
    reversed_result.save()
    assert P.require_passed("phase2", manifest.hash).passed

    directly_reversed = replace(
        reversed_result, scored_origins=tuple(reversed(reversed_result.scored_origins)))
    directly_reversed.save()
    try:
        P.require_passed("phase2", manifest.hash)
    except RuntimeError as exc:
        assert "reports origins" in str(exc)
    else:
        raise AssertionError("a non-canonical direct result encoding unlocked the next phase")

    invalid_origins = (
        (acceptance[0], acceptance[0]),
        (acceptance[0], 1),
        set(acceptance),
        dict.fromkeys(acceptance),
        (origin for origin in acceptance),
    )
    for invalid in invalid_origins:
        try:
            P.evaluate_gate(
                "phase2", manifest, "MASE", _both(1.0), 0.5,
                scored_origins=invalid, **GRID)
        except ValueError as exc:
            assert "EXACTLY the acceptance origins" in str(exc)
        else:
            raise AssertionError(f"invalid scored origins {invalid!r} were accepted")


def test_direction_is_honoured_both_ways():
    lower = P.RegisteredGate(**GATE)
    higher = P.RegisteredGate(**{**GATE, "direction": "higher_is_better"})
    assert lower.passes(_both(1.00), 0.90)
    assert not lower.passes(_both(1.00), 1.10)
    assert higher.passes(_both(0.80), 0.90)
    assert not higher.passes(_both(0.90), 0.80)


def test_every_comparator_must_be_beaten_not_just_the_weakest():
    """Gates 2 and 4 name several baselines; passing only the easiest was possible before r19."""
    g = P.RegisteredGate(**GATE)
    assert not g.passes({"repaired_incumbent": 1.0, "seasonal_naive": 0.50}, 0.60)
    assert g.passes({"repaired_incumbent": 1.0, "seasonal_naive": 0.90}, 0.50)


def test_a_missing_comparator_is_a_failure_not_an_omission():
    try:
        P.RegisteredGate(**GATE).passes({"repaired_incumbent": 1.0}, 0.1)
    except ValueError:
        return
    raise AssertionError("supplying only one of two comparators must raise")


def test_nan_scores_are_rejected_loudly():
    g = P.RegisteredGate(**GATE)
    for bad in ({"repaired_incumbent":
        float("nan"), "seasonal_naive": 1.0}, None):
        try:
            g.passes(bad or _both(1.0), float("nan") if bad is None else 0.1)
        except ValueError:
            continue
        raise AssertionError("a NaN score means the evaluation is broken and must raise")


def test_non_finite_scores_are_rejected_loudly():
    g = P.RegisteredGate(**GATE)
    for candidate, comparators in [
        (float("inf"), _both(1.0)),
        (float("-inf"), _both(1.0)),
        (0.1, _both(float("inf"))),
    ]:
        try:
            g.passes(comparators, candidate)
        except ValueError:
            continue
        raise AssertionError("an infinite score must not become a claim-bearing PASS/FAIL")


def test_gate_rejects_vacuous_or_malformed_registration():
    """An empty comparator set used to PASS every candidate; negative/NaN margins did too."""
    bad = [
        {"comparators": ()},
        {"comparators": ("x", "x")},
        {"min_improvement": -0.1},
        {"min_improvement": float("nan")},
        {"direction": "sideways"},
        {"relation": "approximately"},
        {"metric": ""},
        {"rationale": ""},
    ]
    for change in bad:
        try:
            P.RegisteredGate(**{**GATE, **change})
        except ValueError:
            continue
        raise AssertionError(f"malformed registered gate was accepted: {change}")


def test_a_forged_pass_is_recomputed_and_rejected():
    """The r19 finding: a hand-authored GateResult with a self-consistent hash was accepted."""
    _tmp()
    P.RegisteredGate(**GATE).save()
    m = P.EvaluationManifest(**MF); m.save()
    result = _gate("phase2", m, "MASE", _both(1.0), 0.5)
    replace(result, candidate_score=99.0).save()
    try:
        P.require_passed("phase2", m.hash)
    except RuntimeError as e:
        assert "recomputes" in str(e)
        return
    raise AssertionError("a forged PASS must be caught by recomputation, not trusted")


def test_require_passed_replays_grid_and_acceptance_structure() -> None:
    """A self-consistent numeric PASS with no cells/origins is still not an evaluation."""
    _tmp()
    P.RegisteredGate(**GATE).save()
    m = P.EvaluationManifest(**MF); m.save()
    result = _gate("phase2", m, "MASE", _both(1.0), 0.5)
    replace(result, expected_cells=0, scored_cells=0, scored_origins=()).save()
    try:
        P.require_passed("phase2", m.hash)
    except RuntimeError as exc:
        assert "complete positive grid" in str(exc); return
    raise AssertionError("zero cells and no acceptance origins unlocked the next phase")


def test_a_result_from_a_different_manifest_is_rejected():
    _tmp()
    P.RegisteredGate(**GATE).save()
    m = P.EvaluationManifest(**MF); m.save()
    _gate("phase2", m, "MASE", _both(1.0), 0.5).save()
    other = P.EvaluationManifest(**{**MF, "name": "other", "horizon_days": 14})
    try:
        P.require_passed("phase2", manifest_hash=other.hash)
    except RuntimeError as e:
        assert "not on the same grid" in str(e)
        return
    raise AssertionError("scores from a different grid must not satisfy this phase")


def test_exact_threshold_passes_and_just_under_fails():
    """The boundary is inclusive: a gain of EXACTLY the registered margin passes.

    Values chosen so the subtraction is exact in binary floating point. An earlier version used
    1.00 and 0.95, where 1.00-0.95 == 0.050000000000000044 -- strictly greater than the margin,
    so the test passed under both `>=` and `>` and did not exercise the boundary at all.
    Mutation testing caught it; the arithmetic is now exact.
    """
    g = P.RegisteredGate(**{**GATE, "min_improvement": 0.5})
    assert (1.0 - 0.5) == 0.5, "test relies on exact binary arithmetic"
    assert g.passes(_both(1.0), 0.5)         # gain is EXACTLY 0.5 -- inclusive bound
    assert not g.passes(_both(1.0), 0.75)    # gain 0.25, under the margin


# ---------------------------------------------------------------- evaluate_gate / require_passed
def test_evaluate_gate_uses_the_registered_bar_not_a_caller_supplied_one():
    _tmp()
    P.RegisteredGate(**GATE).save()
    m = P.EvaluationManifest(**MF); m.save()
    r = _gate("phase2", m, "MASE", _both(1.0), candidate_score=0.80)
    assert r.passed and r.manifest_hash == m.hash and r.gate_hash == P.RegisteredGate.load("phase2").hash


def test_evaluate_gate_requires_a_pre_registered_manifest():
    """An object created after scores were seen is not a frozen evaluation protocol."""
    _tmp()
    P.RegisteredGate(**GATE).save()
    m = P.EvaluationManifest(**MF)
    try:
        _gate("phase2", m, "MASE", _both(1.0), candidate_score=0.80)
    except RuntimeError as e:
        assert "has not been registered" in str(e)
        return
    raise AssertionError("an unsaved in-memory manifest must not produce a result")


def test_evaluate_gate_rejects_a_score_labelled_with_the_wrong_metric():
    """RegisteredGate.metric used to be inert metadata; a WAPE gate accepted a claimed RMSE."""
    _tmp()
    P.RegisteredGate(**GATE).save()
    m = P.EvaluationManifest(**MF); m.save()
    try:
        _gate("phase2", m, "RMSE", _both(1.0), candidate_score=0.80)
    except ValueError as e:
        assert "registered for metric" in str(e)
        return
    raise AssertionError("a gate must refuse a differently labelled metric")


def test_require_passed_blocks_when_no_result_exists():
    _tmp()
    try:
        P.require_passed("phase2", "expected-manifest-hash")
    except RuntimeError:
        return
    raise AssertionError("a phase with no gate result must not be allowed to proceed")


def test_require_passed_blocks_on_a_failed_gate():
    _tmp()
    P.RegisteredGate(**GATE).save()
    m = P.EvaluationManifest(**MF); m.save()
    _gate("phase2", m, "MASE", _both(1.0), candidate_score=0.99).save()
    try:
        P.require_passed("phase2", m.hash)
    except RuntimeError as e:
        assert "FAILED" in str(e)
        return
    raise AssertionError("a failed gate must stop the next phase")


def test_require_passed_detects_an_edited_result():
    _tmp()
    P.RegisteredGate(**GATE).save()
    m = P.EvaluationManifest(**MF); m.save()
    path = _gate("phase2", m, "MASE", _both(1.0), 0.99).save()
    blob = json.loads(path.read_text()); blob["passed"] = True
    path.write_text(json.dumps(blob))
    try:
        P.require_passed("phase2", m.hash)
    except RuntimeError as e:
        assert "edited" in str(e)
        return
    raise AssertionError("flipping passed=True by hand must be detected")


def test_require_passed_allows_a_genuine_pass():
    _tmp()
    P.RegisteredGate(**GATE).save()
    m = P.EvaluationManifest(**MF); m.save()
    _gate("phase2", m, "MASE", _both(1.0), candidate_score=0.80).save()
    assert P.require_passed("phase2", m.hash).passed


def test_require_passed_cannot_omit_the_current_manifest():
    """A caller that omitted the hash used to authorize M2 with a result produced on M1."""
    _tmp()
    P.RegisteredGate(**GATE).save()
    m = P.EvaluationManifest(**MF); m.save()
    _gate("phase2", m, "MASE", _both(1.0), candidate_score=0.80).save()
    try:
        P.require_passed("phase2")
    except TypeError:
        return
    raise AssertionError("the current manifest hash must be a required argument")


def test_require_passed_rejects_a_result_from_a_different_registered_gate():
    """Mutation guard: deleting the current-gate hash check passed all 28 pre-review tests."""
    _tmp()
    gate = P.RegisteredGate(**GATE); gate.save()
    m = P.EvaluationManifest(**MF); m.save()
    result = _gate("phase2", m, "MASE", _both(1.0), 0.5)
    replace(result, gate_hash="0" * 64).save()
    try:
        P.require_passed("phase2", m.hash)
    except RuntimeError as e:
        assert "DIFFERENT gate" in str(e)
        return
    raise AssertionError("a result scored against another registered threshold must not pass")


def test_result_saves_preserve_every_distinct_attempt():
    """A fail followed by a pass used to erase the failure, hiding re-run-until-pass behavior."""
    _tmp()
    P.RegisteredGate(**GATE).save()
    m = P.EvaluationManifest(**MF); m.save()
    _gate("phase2", m, "MASE", _both(1.0), candidate_score=0.99,
                    notes="first attempt").save()
    _gate("phase2", m, "MASE", _both(1.0), candidate_score=0.80,
                    notes="harness defect fixed").save()
    attempts = list((P.RESULTS / "phase2").glob("*.json"))
    assert len(attempts) == 2, "both the failed and passing attempts must remain auditable"
    assert P.require_passed("phase2", m.hash).passed


def test_equivalent_result_key_orders_share_one_canonical_immutable_generation():
    """The content hash ignores object-key order, so its immutable bytes must ignore it too."""
    _tmp()
    P.RegisteredGate(**GATE).save()
    manifest = P.EvaluationManifest(**MF)
    manifest.save()
    first = _gate(
        "phase2", manifest, "MASE", {"repaired_incumbent": 1.0, "seasonal_naive": 1.1}, 0.8)
    second = replace(
        first, comparator_scores={"seasonal_naive": 1.1, "repaired_incumbent": 1.0})

    first_path = first.save()
    first_bytes = first_path.read_bytes()
    second_path = second.save()

    assert second_path.read_bytes() == first_bytes
    assert len(list((P.RESULTS / "phase2").glob("*.json"))) == 1


def test_headroom_gates_are_expressible():
    """r20: Gate 2's residual-headroom test was unpassable -- the oracle always beats the
    candidate, and `passes` could only express 'candidate must win'."""
    g = P.RegisteredGate(**{**GATE, "comparators": ("skill_ceiling",), "relation": "gap_from",
                            "min_improvement": 0.20})
    assert g.passes({"skill_ceiling": 0.50}, 0.90)      # gap 0.40 -> headroom worth pursuing
    assert not g.passes({"skill_ceiling": 0.50}, 0.55)  # gap 0.05 -> not worth a custom model


def test_headroom_requires_the_ceiling_to_be_better_not_merely_different():
    """The first r20 repair used abs(gain), so a worse 'ceiling' passed if far enough away."""
    lower = P.RegisteredGate(**{**GATE, "comparators": ("skill_ceiling",),
                                "relation": "gap_from", "min_improvement": 0.20})
    higher = P.RegisteredGate(**{**GATE, "comparators": ("skill_ceiling",),
                                 "relation": "gap_from", "min_improvement": 0.20,
                                 "direction": "higher_is_better"})
    assert lower.passes({"skill_ceiling": 0.50}, 0.90)
    assert not lower.passes({"skill_ceiling": 0.90}, 0.50)
    assert higher.passes({"skill_ceiling": 0.90}, 0.50)
    assert not higher.passes({"skill_ceiling": 0.50}, 0.90)


def test_beats_and_gap_from_are_not_interchangeable():
    beats = P.RegisteredGate(**{**GATE, "comparators": ("c",), "min_improvement": 0.20})
    gap = P.RegisteredGate(**{**GATE, "comparators": ("c",), "min_improvement": 0.20,
                              "relation": "gap_from"})
    assert not beats.passes({"c": 0.50}, 0.90)   # candidate is worse -> fails a 'beats' gate
    assert gap.passes({"c": 0.50}, 0.90)         # ...and passes a 'gap_from' gate


def test_hand_authored_scores_are_NOT_detectable_and_this_is_documented():
    """The limit of the mechanism, asserted so it cannot be forgotten.

    Caller-supplied scores whose values are internally consistent recompute as a genuine pass.
    Closing this needs access control, not serialisation -- see the module docstring. Recording
    an attempt does not establish that a score was measured.
    """
    _tmp()
    P.RegisteredGate(**GATE).save()
    m = P.EvaluationManifest(**MF); m.save()
    _gate("phase2", m, "MASE", _both(1.0), 0.0).save()
    assert P.require_passed("phase2", m.hash).passed, \
        "documented limitation: unmeasured scores pass"
    assert "unverifiable" in P.__doc__


def test_acceptance_horizon_may_not_sit_inside_a_development_horizon():
    """O4 review, forward leak: origins are start DATES, so an acceptance origin's horizon can
    fall inside a development horizon and the model is judged on days it was tuned on."""
    try:
        P.EvaluationManifest(**{**MF, "origins": ("2026-03-01", "2026-03-15", "2026-04-01"),
                                "acceptance_origins": ("2026-03-15",)})
    except ValueError as e:
        assert "acceptance period" in str(e)
        return
    raise AssertionError("an acceptance horizon inside a development horizon must be refused")


def test_a_later_development_origin_may_not_look_back_over_the_acceptance_period():
    """A development origin AFTER the acceptance set has an encoder window covering it, so tuning
    has already seen the outcomes it will be judged on."""
    try:
        P.EvaluationManifest(**{**MF, "origins": ("2026-03-15", "2026-06-01", "2026-07-01"),
                                "acceptance_origins": ("2026-03-15",)})
    except ValueError as e:
        assert "acceptance period" in str(e)
        return
    raise AssertionError("a development origin whose encoder window covers acceptance must fail")


def test_reserving_acceptance_without_encoder_days_is_refused():
    """An unchecked reservation is bookkeeping, not a control."""
    try:
        P.EvaluationManifest(**{**MF, "encoder_days": 0})
    except ValueError as e:
        assert "encoder_days" in str(e)
        return
    raise AssertionError("temporal separation cannot be checked without encoder_days")


def test_a_campaign_may_not_appear_on_both_sides():
    """Promotions span days, so a date-level split can put one campaign's outcome in both."""
    cal = {"2024-06-05": "xmas_promo", "2026-08-10": "xmas_promo"}
    try:
        P.EvaluationManifest(**{**MF, "campaign_of_date": cal})
    except ValueError as e:
        assert "both sides" in str(e)
        return
    raise AssertionError("a campaign spanning the split must be refused")


def test_the_acceptance_set_is_spent_after_the_attempt_budget():
    """The file-drawer leak: submit, fail, tweak, resubmit turns a holdout into a tuning set."""
    _tmp()
    P.RegisteredGate(**GATE).save()
    m = P.EvaluationManifest(**{**MF, "max_gate_attempts": 2}); m.save()
    for _ in range(2):
        _gate("phase2", m, "MASE", _both(1.0), 0.99)
    try:
        _gate("phase2", m, "MASE", _both(1.0), 0.99)
    except RuntimeError as e:
        assert "spent" in str(e)
        return
    raise AssertionError("a spent acceptance set must refuse further attempts")


def test_a_manifest_needs_at_least_three_rolling_origins():
    """One origin is an anecdote. Three is the minimum that shows whether a result is stable."""
    try:
        P.EvaluationManifest(**{**MF, "origins": ("2024-06-01", "2024-09-01"),
                                "acceptance_origins": ("2024-09-01",)})
    except ValueError as e:
        assert "three rolling origins" in str(e)
        return
    raise AssertionError("fewer than three origins must be refused")


def test_duplicate_origins_are_refused():
    """A repeated origin silently double-weights one date in the average."""
    try:
        P.EvaluationManifest(**{**MF, "origins": ("2024-06-01", "2024-06-01", "2026-08-01")})
    except ValueError as e:
        assert "duplicate" in str(e).lower()
        return
    raise AssertionError("duplicate origins must be refused")


def test_the_gate_refuses_scores_from_development_origins():
    """O4: approving on the periods the model was chosen with is the whole thing being prevented."""
    _tmp()
    P.RegisteredGate(**GATE).save()
    m = P.EvaluationManifest(**MF)
    m.save()
    try:
        P.evaluate_gate("phase2", m, "MASE", _both(1.0), 0.5,
                        scored_origins=m.development_origins, **GRID)
    except ValueError as e:
        assert "acceptance origins" in str(e)
        return
    raise AssertionError("scoring a gate on development origins must be refused")


def test_a_partially_covered_grid_is_refused():
    """O7: the cells that drop out of a partial run are the sparse ones, so the bias is
    systematic rather than random -- a tenth of the grid must not look like a clean result."""
    _tmp()
    P.RegisteredGate(**GATE).save()
    m = P.EvaluationManifest(**MF)
    m.save()
    try:
        P.evaluate_gate("phase2", m, "MASE", _both(1.0), 0.5, scored_origins=ACC,
                        expected_cells=1_000_000, scored_cells=100_000)
    except ValueError as e:
        assert "coverage is incomplete" in str(e)
        return
    raise AssertionError("a partial grid must be refused")


# --------------------------------------------------------------- A1: each helper named directly
def test_require_registered_refuses_an_unsaved_manifest():
    """A result may only cite a manifest that exists on disk; otherwise the grid it claims to
    have used cannot be recovered by anyone checking the number later."""
    _tmp()
    try:
        P.EvaluationManifest(**MF).require_registered()
    except Exception as e:
        assert "regist" in str(e).lower() or "saved" in str(e).lower()
        return
    raise AssertionError("an unsaved manifest must not satisfy require_registered")


def test_attempts_against_records_each_submission():
    """The file-drawer leak is only visible if every attempt is written down."""
    _tmp()
    P.RegisteredGate(**GATE).save()
    m = P.EvaluationManifest(**MF)
    m.save()
    assert P.attempts_against(m.hash) == []
    _gate("phase2", m, "MASE", _both(1.0), 0.99)
    _gate("phase2", m, "MASE", _both(1.0), 0.98)
    assert len(P.attempts_against(m.hash)) == 2


def _assert_attempt_ledger_rejected(payload: bytes, expected: str) -> None:
    _tmp()
    P.RESULTS.mkdir(parents=True)
    (P.RESULTS / "attempts.jsonl").write_bytes(payload)
    try:
        P.attempts_against("a" * 64)
    except RuntimeError as exc:
        assert expected in str(exc), str(exc)
        return
    raise AssertionError(f"attempt ledger accepted invalid bytes {payload!r}")


def test_attempt_ledger_rejects_malformed_json_instead_of_skipping_it() -> None:
    _assert_attempt_ledger_rejected(b'{"phase":"phase2",\n', "invalid JSON")
    _assert_attempt_ledger_rejected(b"   \n", "invalid JSON")
    _assert_attempt_ledger_rejected(b"\n", "empty record")


def test_attempt_ledger_rejects_duplicate_keys_instead_of_skipping_them() -> None:
    payload = (
        b'{"phase":"phase2","phase":"phase3","manifest_hash":"'
        + b"a" * 64
        + b'","passed":false}\n'
    )
    _assert_attempt_ledger_rejected(payload, "duplicate JSON object key")


def test_attempt_ledger_rejects_nonstandard_numeric_constants() -> None:
    payload = (
        b'{"phase":"phase2","manifest_hash":"'
        + b"a" * 64
        + b'","passed":NaN}\n'
    )
    _assert_attempt_ledger_rejected(payload, "non-finite JSON number")


def test_attempt_ledger_rejects_a_truncated_final_line() -> None:
    payload = (
        b'{"phase":"phase2","manifest_hash":"'
        + b"a" * 64
        + b'","passed":false}'
    )
    _assert_attempt_ledger_rejected(payload, "truncated final line")


def test_attempt_ledger_rejects_every_schema_alias() -> None:
    digest = "a" * 64
    invalid_records = (
        [],
        {"phase": "phase2", "manifest_hash": digest},
        {"phase": "phase2", "manifest_hash": digest, "passed": False},
        {"phase": "phase2", "manifest_hash": digest, "passed": False, "extra": 1},
        {"phase": "../phase2", "manifest_hash": digest, "passed": False, "attempt_number": 1},
        {"phase": "phase2", "manifest_hash": "short", "passed": False, "attempt_number": 1},
        {"phase": "phase2", "manifest_hash": digest, "passed": 0, "attempt_number": 1},
        {"phase": "phase2", "manifest_hash": digest, "passed": False,
         "attempt_number": True},
        {"phase": "phase2", "manifest_hash": digest, "passed": False,
         "attempt_number": 1.0},
        {"phase": "phase2", "manifest_hash": digest, "passed": False,
         "attempt_number": 0},
    )
    for record in invalid_records:
        _assert_attempt_ledger_rejected(
            (json.dumps(record) + "\n").encode(), "gate-attempt ledger line 1")


def test_attempt_ledger_rejects_number_gaps_and_unnumbered_records_anywhere() -> None:
    digest = "a" * 64
    numbered_two = json.dumps({
        "phase": "phase2", "manifest_hash": digest, "passed": False,
        "attempt_number": 2,
    }).encode() + b"\n"
    _assert_attempt_ledger_rejected(numbered_two, "expected 1")

    numbered_one = json.dumps({
        "phase": "phase2", "manifest_hash": digest, "passed": False,
        "attempt_number": 1,
    }).encode() + b"\n"
    unnumbered = json.dumps({
        "phase": "phase2", "manifest_hash": "b" * 64, "passed": False,
    }).encode() + b"\n"
    for payload, line_number in ((unnumbered, 1), (unnumbered + numbered_one, 1),
                                 (numbered_one + unnumbered, 2)):
        _assert_attempt_ledger_rejected(payload, f"line {line_number} must be an object")


def test_attempt_reader_validates_the_full_ledger_not_only_matching_records() -> None:
    valid = json.dumps({
        "phase": "phase2", "manifest_hash": "a" * 64, "passed": False,
        "attempt_number": 1,
    }).encode()
    _assert_attempt_ledger_rejected(valid + b'\n{"unrelated":\n', "line 2")


def test_evaluate_gate_refuses_a_corrupt_ledger_without_changing_its_bytes() -> None:
    _tmp()
    P.RegisteredGate(**GATE).save()
    manifest = P.EvaluationManifest(**MF)
    manifest.save()
    P.RESULTS.mkdir(parents=True, exist_ok=True)
    ledger = P.RESULTS / "attempts.jsonl"
    ledger.write_bytes(b'{"phase":"phase2",\n')
    before = ledger.read_bytes()
    try:
        _gate("phase2", manifest, "MASE", _both(1.0), 0.5)
    except RuntimeError as exc:
        assert "attempt ledger line 1 is invalid JSON" in str(exc), str(exc)
    else:
        raise AssertionError("evaluate_gate appended past corrupt attempt evidence")
    assert ledger.read_bytes() == before


def test_attempt_budget_final_slot_is_atomic_across_processes() -> None:
    """Two simultaneous submissions to the last slot produce one record and one refusal."""
    assert "fork" in mp.get_all_start_methods(), "flock protocol tests require a fork-capable OS"
    _tmp()
    P.RegisteredGate(**GATE).save()
    fields = {**MF, "name": "concurrent_final_slot", "max_gate_attempts": 1}
    manifest = P.EvaluationManifest(**fields)
    manifest.save()

    context = mp.get_context("fork")
    start = context.Event()
    ready = context.Queue()
    lock_calls = context.Queue()
    outcomes = context.Queue()
    processes = [
        context.Process(
            target=_concurrent_final_attempt_worker,
            args=(str(P.REGISTRY), str(P.RESULTS), fields,
                  start, ready, lock_calls, outcomes),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    try:
        for _ in processes:
            assert ready.get(timeout=10) is True
        # Both children were forked before this descriptor was opened, so neither inherits the
        # parent's ownership. They must wait on the same stable inode once evaluation begins.
        with P._attempt_ledger_lock():  # noqa: SLF001
            start.set()
            for _ in processes:
                assert lock_calls.get(timeout=10) is True
            try:
                premature = outcomes.get_nowait()
            except queue.Empty:
                pass
            else:
                raise AssertionError(
                    f"a concurrent evaluator bypassed the held ledger lock: {premature!r}")
        observed = [outcomes.get(timeout=10) for _ in processes]
    finally:
        start.set()
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    assert sorted(kind for kind, _ in observed) == ["recorded", "spent"], observed
    assert all(process.exitcode == 0 for process in processes), observed
    assert len(P.attempts_against(manifest.hash)) == 1


def test_canonical_result_cannot_move_backwards_in_concurrent_attempt_order() -> None:
    """A delayed older PASS cannot overwrite a newer FAIL or authorize the next phase."""
    assert "fork" in mp.get_all_start_methods(), "flock protocol tests require a fork-capable OS"
    _tmp()
    P.RegisteredGate(**GATE).save()
    fields = {**MF, "name": "ordered_concurrent_results", "max_gate_attempts": 2}
    manifest = P.EvaluationManifest(**fields)
    manifest.save()

    context = mp.get_context("fork")
    evaluated = context.Queue()
    outcomes = context.Queue()
    older_save = context.Event()
    older_resave = context.Event()
    newer_save = context.Event()
    older = context.Process(
        target=_ordered_attempt_save_worker,
        args=("older", str(P.REGISTRY), str(P.RESULTS), fields, 0.50,
              evaluated, older_save, outcomes, older_resave),
    )
    newer = context.Process(
        target=_ordered_attempt_save_worker,
        args=("newer", str(P.REGISTRY), str(P.RESULTS), fields, 0.99,
              evaluated, newer_save, outcomes),
    )
    processes = (older, newer)
    older.start()
    try:
        assert evaluated.get(timeout=10) == ("older", 1, True)
        older_save.set()
        assert outcomes.get(timeout=10) == ("older", "published", 1)

        newer.start()
        assert evaluated.get(timeout=10) == ("newer", 2, False)
        try:
            P.require_passed("phase2", manifest.hash)
        except RuntimeError as exc:
            assert "not the latest recorded" in str(exc), str(exc)
        else:
            raise AssertionError("an older PASS remained authoritative after a newer FAIL recorded")

        newer_save.set()
        assert outcomes.get(timeout=10) == ("newer", "published", 2)
        older_resave.set()
        stale = outcomes.get(timeout=10)
        assert stale[:2] == ("older", "refused-again"), stale
        assert "no longer the latest" in stale[2]
    finally:
        older_save.set()
        newer_save.set()
        older_resave.set()
        for process in processes:
            if process.pid is None:
                continue
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    assert all(process.exitcode == 0 for process in processes)
    attempts = P.attempts_against(manifest.hash)
    assert [(row["attempt_number"], row["passed"]) for row in attempts] == [
        (1, True), (2, False)]
    canonical = json.loads((P.RESULTS / "phase2.json").read_text())
    assert canonical["attempt_number"] == 2 and canonical["passed"] is False
    assert len(list((P.RESULTS / "phase2").glob("*.json"))) == 2
    try:
        P.require_passed("phase2", manifest.hash)
    except RuntimeError as exc:
        assert "FAILED" in str(exc), str(exc)
    else:
        raise AssertionError("the latest recorded FAIL unlocked the next phase")


def test_phase_canonical_order_spans_different_manifest_hashes() -> None:
    """The canonical path is per phase, so a delayed old manifest cannot replace a newer one."""
    _tmp()
    P.RegisteredGate(**GATE).save()
    older_manifest = P.EvaluationManifest(**{**MF, "name": "older_manifest"})
    newer_manifest = P.EvaluationManifest(**{**MF, "name": "newer_manifest"})
    older_manifest.save()
    newer_manifest.save()
    older = _gate("phase2", older_manifest, "MASE", _both(1.0), 0.50)
    older.save()
    newer = _gate("phase2", newer_manifest, "MASE", _both(1.0), 0.99)
    newer.save()
    try:
        older.save()
    except RuntimeError as exc:
        assert "no longer the latest" in str(exc), str(exc)
    else:
        raise AssertionError("an older manifest moved the phase canonical backwards")
    canonical = json.loads((P.RESULTS / "phase2.json").read_text())
    assert canonical["manifest_hash"] == newer_manifest.hash
    assert canonical["passed"] is False


def test_gate_results_require_a_current_numbered_attempt_and_exact_ledger_record() -> None:
    _tmp()
    values = {
        "phase": "phase2", "manifest_hash": "a" * 64, "gate_hash": "b" * 64,
        "metric": "MASE", "comparator_scores": _both(1.0), "candidate_score": 0.5,
        "passed": True,
    }
    try:
        P.GateResult(**values)
    except TypeError as exc:
        assert "attempt_number" in str(exc), str(exc)
    else:
        raise AssertionError("gate result omitted its current attempt identity")
    for invalid in (None, True, 1.0, 0, -1):
        try:
            P.GateResult(**values, attempt_number=invalid)
        except ValueError as exc:
            assert "positive integer" in str(exc), str(exc)
        else:
            raise AssertionError(f"gate result accepted invalid attempt number {invalid!r}")

    try:
        P.GateResult(**values, attempt_number=1).save()
    except RuntimeError as exc:
        assert "no exact durable ledger record" in str(exc), str(exc)
    else:
        raise AssertionError("an invented attempt number published without a durable ledger row")
    assert not (P.RESULTS / "phase2.json").exists()


def test_result_record_cannot_fill_missing_fields_from_constructor_defaults() -> None:
    _tmp()
    P.RegisteredGate(**GATE).save()
    manifest = P.EvaluationManifest(**MF)
    manifest.save()
    result = _gate("phase2", manifest, "MASE", _both(1.0), 0.5)
    path = result.save()
    for omitted in (
        "attempt_number", "measurement_evidence", "scored_origins", "expected_cells",
        "scored_cells", "per_client", "comparator_per_client", "extra_scores", "reasons", "notes",
    ):
        record = asdict(result)
        record.pop(omitted)
        payload = json.dumps(record | {"hash": P.content_hash(record)})
        path.write_text(payload)
        try:
            P.require_passed("phase2", manifest.hash)
        except RuntimeError as exc:
            assert "every current gate result field" in str(exc), str(exc)
        else:
            raise AssertionError(f"gate result silently filled omitted {omitted}")
        assert path.read_text() == payload


def test_failed_atomic_canonical_replace_cleans_temp_and_is_retryable() -> None:
    """A publication exception retains history, exposes no partial canonical, and releases lock."""
    _tmp()
    P.RegisteredGate(**GATE).save()
    manifest = P.EvaluationManifest(**MF)
    manifest.save()
    result = _gate("phase2", manifest, "MASE", _both(1.0), 0.50)
    real_replace = P.os.replace

    def fail_canonical(source, destination):
        if pathlib.Path(destination) == P.RESULTS / "phase2.json":
            raise OSError("injected canonical replace failure")
        return real_replace(source, destination)

    with mock.patch.object(P.os, "replace", side_effect=fail_canonical):
        try:
            result.save()
        except OSError as exc:
            assert "injected canonical replace failure" in str(exc)
        else:
            raise AssertionError("a failed canonical replace reported successful publication")
    assert not (P.RESULTS / "phase2.json").exists()
    assert not list(P.RESULTS.rglob("*.tmp"))
    assert len(list((P.RESULTS / "phase2").glob("*.json"))) == 1

    result.save()
    assert P.require_passed("phase2", manifest.hash).passed


def test_attempt_append_fsyncs_content_and_new_directory_entries() -> None:
    """A returned attempt survives its data and new-ledger directory durability boundaries."""
    _tmp()
    P.RegisteredGate(**GATE).save()
    manifest = P.EvaluationManifest(**MF)
    manifest.save()
    events: list[tuple[str, pathlib.Path | None]] = []
    with (
        mock.patch.object(
            P.os, "fsync", side_effect=lambda _fd: events.append(("file", None))),
        mock.patch.object(
            P, "_fsync_directory",
            side_effect=lambda path: events.append(("directory", pathlib.Path(path))),
        ),
    ):
        _gate("phase2", manifest, "MASE", _both(1.0), 0.99)
    # Registration has already made RESULTS and the persistent shared lock durable. The first
    # attempt therefore needs to sync only its content and the newly named ledger entry.
    assert events == [("file", None), ("directory", P.RESULTS)]
    assert (P.RESULTS / "attempts.jsonl").read_bytes().endswith(b"\n")
    lock_inode = (P.RESULTS / ".attempts.jsonl.lock").stat().st_ino

    events.clear()
    with (
        mock.patch.object(
            P.os, "fsync", side_effect=lambda _fd: events.append(("file", None))),
        mock.patch.object(
            P, "_fsync_directory",
            side_effect=lambda path: events.append(("directory", pathlib.Path(path))),
        ),
    ):
        _gate("phase2", manifest, "MASE", _both(1.0), 0.98)
    assert events == [("file", None)], "every append flushes and fsyncs its existing ledger"
    assert (P.RESULTS / ".attempts.jsonl.lock").stat().st_ino == lock_inode


def test_check_extras_treats_a_missing_score_as_a_failure():
    """Otherwise Gate 4 passes by simply not measuring stability."""
    g = P.RegisteredGate(**{**GATE, "extra_criteria": (
        P.ExtraCriterion("origin_drift_pct", 0.15, "lower_is_better"),),
        "measurement_contract": {"origin_drift_pct": {"origins": ["2026-07-01", "2026-08-01"]}}})
    ok, why = g.check_extras({})
    assert not ok and "not supplied" in " ".join(why)
    assert g.check_extras({"origin_drift_pct": 0.10})[0]
    assert not g.check_extras({"origin_drift_pct": 0.40})[0]
    for unexpected in (
        {"posthoc": 999.0},
        {"origin_drift_pct": 0.10, "posthoc": 999.0},
    ):
        try:
            g.check_extras(unexpected)
        except ValueError as exc:
            assert "unregistered extra" in str(exc)
            continue
        raise AssertionError("an unregistered post-hoc metric entered the gate artifact")


def test_check_per_client_refuses_a_pooled_pass_that_hides_a_regression():
    """A pooled metric can be dominated by one large dataset."""
    g = P.RegisteredGate(**GATE)
    comparators = {
        "repaired_incumbent": {"a": 0.70, "b": 1.20},
        "seasonal_naive": {"a": 0.90, "b": 1.00},
    }
    # Both candidate scores are below the old pooled reference 1.30, but each is worse than the
    # strongest comparator on that same client. A pooled reference would report the opposite.
    ok, why = g.check_per_client(
        {"a": 0.80, "b": 1.10}, comparators, clients=("a", "b"))
    assert not ok and "hidden" in why
    assert g.check_per_client(
        {"a": 0.60, "b": 0.90}, comparators, clients=("a", "b"))[0]


def test_check_per_client_uses_the_strongest_same_client_comparator_higher_is_better():
    g = P.RegisteredGate(**{**GATE, "direction": "higher_is_better"})
    comparators = {
        "repaired_incumbent": {"a": 0.70, "b": 0.60},
        "seasonal_naive": {"a": 0.80, "b": 0.90},
    }
    ok, why = g.check_per_client(
        {"a": 0.85, "b": 0.85}, comparators, clients=("a", "b"))
    assert not ok and "b" in why
    assert g.check_per_client(
        {"a": 0.85, "b": 0.95}, comparators, clients=("a", "b"))[0]


def test_check_per_client_requires_exact_candidate_and_comparator_coverage():
    g = P.RegisteredGate(**GATE)
    full = {"a": 0.5, "b": 0.5}
    comparators = {
        "repaired_incumbent": {"a": 1.0, "b": 1.0},
        "seasonal_naive": {"a": 1.0, "b": 1.0},
    }
    malformed = [
        ({"a": 0.5}, comparators),
        ({**full, "unregistered": 0.5}, comparators),
        (full, {"repaired_incumbent": comparators["repaired_incumbent"]}),
        (full, {**comparators, "unregistered": {"a": 1.0, "b": 1.0}}),
        (full, {**comparators, "seasonal_naive": {"a": 1.0}}),
    ]
    for candidate, comparator_scores in malformed:
        try:
            g.check_per_client(candidate, comparator_scores, clients=("a", "b"))
        except ValueError:
            continue
        raise AssertionError(
            "partial or unregistered per-client score mappings must be rejected")


def test_check_per_client_rejects_bad_keys_and_non_finite_values():
    g = P.RegisteredGate(**GATE)
    full = {
        "repaired_incumbent": {"a": 1.0, "b": 1.0},
        "seasonal_naive": {"a": 1.0, "b": 1.0},
    }
    malformed = [
        ({"": 0.5, "b": 0.5}, full),
        ({"a": float("nan"), "b": 0.5}, full),
        ({"a": 10**400, "b": 0.5}, full),
        ({"a": 0.5, "b": 0.5},
         {**full, "seasonal_naive": {"a": 1.0, "b": float("inf")}}),
        ({"a": 0.5, "b": 0.5},
         {"": {"a": 1.0, "b": 1.0},
          "seasonal_naive": {"a": 1.0, "b": 1.0}}),
    ]
    for candidate, comparator_scores in malformed:
        try:
            g.check_per_client(candidate, comparator_scores, clients=("a", "b"))
        except ValueError:
            continue
        raise AssertionError("invalid per-client keys and values must not enter a gate artifact")


def test_multi_client_check_requires_names_not_only_a_count():
    """A count of two cannot distinguish {a,b} from the partial or wrong cohort {a,c}."""
    g = P.RegisteredGate(**GATE)
    try:
        g.check_per_client({"a": 0.5, "b": 0.5}, {}, n_clients=2)
    except TypeError as exc:
        assert "n_clients" in str(exc)
        return
    raise AssertionError("a multi-client check without manifest client identities is not strict")


def test_check_per_client_is_vacuous_for_a_single_client():
    """With one client the pooled result IS that client; demanding a breakdown of one would fail
    every single-client phase, M5 included, for no benefit."""
    g = P.RegisteredGate(**GATE)
    assert g.check_per_client({}, {}, clients=("m5",))[0]
    for candidate, comparators in (
        ({"ghost": 0.0}, {}),
        ({}, {"invented": {"ghost": 0.0}}),
        ({"ghost": 0.0}, {name: {"ghost": 0.0} for name in g.comparators}),
    ):
        try:
            g.check_per_client(candidate, comparators, clients=("m5",))
        except ValueError:
            continue
        raise AssertionError("single-client gate retained a wrong or invented evidence namespace")


def test_pooled_only_per_client_check_is_explicitly_vacuous():
    g = P.RegisteredGate(**{**GATE, "per_client_policy": "pooled_only"})
    assert g.check_per_client({}, {}, clients=("a", "b"))[0]
    try:
        g.check_per_client({"a": 0.1, "b": 0.2}, {}, clients=("a", "b"))
    except ValueError as exc:
        assert "pooled_only" in str(exc)
    else:
        raise AssertionError("pooled-only result carried unaudited per-client evidence")


def test_evaluate_gate_persists_complete_comparator_per_client_scores():
    _tmp()
    P.RegisteredGate(**GATE).save()
    m = P.EvaluationManifest(**{**MF, "name": "multi", "clients": ("a", "b")})
    m.save()
    comparator_clients = {
        "repaired_incumbent": {"a": 1.0, "b": 1.0},
        "seasonal_naive": {"a": 0.9, "b": 0.8},
    }
    result = _gate(
        "phase2", m, "MASE", _both(1.0), 0.5,
        per_client={"a": 0.6, "b": 0.7},
        comparator_per_client=comparator_clients)
    assert result.passed
    assert result.comparator_per_client == comparator_clients
    result.save()
    assert P.require_passed("phase2", m.hash).comparator_per_client == comparator_clients


def test_require_passed_recomputes_same_client_comparator_regressions():
    _tmp()
    P.RegisteredGate(**GATE).save()
    m = P.EvaluationManifest(**{**MF, "name": "multi", "clients": ("a", "b")})
    m.save()
    # The pooled primary score passes, but b loses to seasonal_naive on b. A hand-authored PASS
    # must be overturned using the comparator_per_client evidence stored in the result.
    result = _gate(
        "phase2", m, "MASE", _both(1.0), 0.5, per_client={"a": 0.8, "b": 0.8},
        comparator_per_client={name: {"a": 1.0, "b": 1.0} for name in GATE["comparators"]})
    replace(
        result,
        comparator_per_client={
            "repaired_incumbent": {"a": 1.0, "b": 1.0},
            "seasonal_naive": {"a": 1.0, "b": 0.5},
        }).save()
    try:
        P.require_passed("phase2", m.hash)
    except RuntimeError as exc:
        assert "recomputes False" in str(exc)
        return
    raise AssertionError("require_passed trusted a pooled pass hiding a client regression")


if __name__ == "__main__":
    import sys
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
    print(f"\n{len(fns)-failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
