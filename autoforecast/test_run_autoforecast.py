"""Focused checks for score-only evaluation boundaries in the governed runner."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import sys
import tempfile
from types import SimpleNamespace

import pytest
from scripts import run_autoforecast as runner

from autoforecast.model_catalog import discover_candidates
from autoforecast.panel import Forecast, Panel
from autoforecast.train_workflow import EvaluationRequest, RunRequest, TrainWorkflowError
from autoforecast.training_lease import TrainingLeaseError, training_lease


def test_direct_runner_refuses_an_existing_host_training_lease(
    tmp_path: pathlib.Path,
) -> None:
    with training_lease(), pytest.raises(TrainingLeaseError, match="owns the host lease"):
        runner.run_request(
            tmp_path / "request.yaml",
            data_path=tmp_path / "data",
            data_snapshot_id="test",
            data_reference="local://test",
            output_root=tmp_path / "runs",
            epochs=2,
            timeout_seconds=10,
        )


def test_cli_defers_precision_default_to_request_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_request(*args: object, **kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(runner, "run_request", fake_run_request)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_autoforecast.py", "request.yaml", "--data-path", str(tmp_path),
         "--data-snapshot-id", "test", "--data-reference", "local://test"],
    )
    assert runner.main() == 0
    assert captured["precision"] is None
    assert captured["device"] is None
    assert captured["patience"] is None


def test_tft_direct_runner_rejects_explicit_non_governing_patience(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    monkeypatch.setattr(runner, "_runtime_data_path", lambda path: path)
    monkeypatch.setattr(runner, "capture_source_identity", lambda _root: object())
    monkeypatch.setattr(
        runner, "capture_runtime_identity", lambda _root, *, image_digest: object()
    )
    with pytest.raises(TrainWorkflowError, match="TFT-Q/TFT-HNB M5 requires validation patience 10"):
        runner.run_request(
            runner.ROOT / "configs/autoforecast/m5_tft_q_reference_smoke.yaml",
            data_path=tmp_path / "data", data_snapshot_id="test", data_reference="local://test",
            output_root=tmp_path / "runs", epochs=2, timeout_seconds=10, patience=5,
        )


@pytest.mark.parametrize(
    ("declared", "has_parent", "explicit", "expected"),
    [
        ("32-true", False, None, "32-true"),
        ("16-mixed", False, None, "16-mixed"),
        ("16-mixed", True, None, "32-true"),
        ("32-true", False, "16-mixed", "16-mixed"),
        ("16-mixed", False, "32-true", "32-true"),
    ],
)
def test_declared_precision_and_continuation_policy(
    declared: str, has_parent: bool, explicit: str | None, expected: str
) -> None:
    assert runner._resolved_precision(declared, explicit, has_parent=has_parent) == expected


def test_mixed_precision_refuses_checkpoint_continuation_even_when_explicit() -> None:
    with pytest.raises(TrainWorkflowError, match="unsupported for checkpoint continuations"):
        runner._resolved_precision("16-mixed", "16-mixed", has_parent=True)


@pytest.mark.parametrize("declared", ["32-true", "16-mixed"])
def test_runner_uses_precision_from_the_request_file(
    declared: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = tmp_path / "request.yaml"
    request.write_text(
        "schema_version: 3\nrun:\n  name: precision-handoff\n"
        "  dataset_contract: configs/datasets/synthetic_daily.yaml\n"
        "  model_catalog: configs/models\n  models: [seasonal_naive]\n"
        "  origin: '2024-01-31'\n  horizon: 7\n  tracking:\n    mlflow: {enabled: false}\n"
        f"  precision: {declared}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "_runtime_data_path", lambda path: path)
    monkeypatch.setattr(runner, "_inside_repository", lambda path, _label: path)

    def inspect_precision(value: str, explicit: str | None, *, has_parent: bool) -> str:
        assert value == declared
        assert explicit is None and not has_parent
        raise RuntimeError("precision reached the runtime policy")

    monkeypatch.setattr(runner, "_resolved_precision", inspect_precision)
    with pytest.raises(RuntimeError, match="precision reached the runtime policy"):
        runner.run_request(
            request, data_path=tmp_path / "data", data_snapshot_id="test",
            data_reference="local://test", output_root=tmp_path / "runs",
            epochs=2, timeout_seconds=10,
        )


def _request() -> RunRequest:
    return RunRequest(
        "score-only-functional",
        "configs/datasets/m5_daily.yaml",
        "configs/models",
        ("seasonal_naive",),
        "2016-05-22",
        28,
        evaluation=EvaluationRequest(
            "no_data",
            "wrmsse",
            "post_training_score_only",
            "archived_private_replay_not_blind",
        ),
    )


def test_runtime_data_path_is_absolute_before_worker_isolation(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "panel").mkdir()
    monkeypatch.chdir(tmp_path)
    assert runner._runtime_data_path(pathlib.Path("panel")) == tmp_path / "panel"


def test_worker_scratch_parent_is_durable_below_the_governed_output_root(
    tmp_path: pathlib.Path,
) -> None:
    output = tmp_path / "governed-run"
    scratch = runner._durable_worker_scratch_parent(output)
    assert scratch == output / ".autoforecast-worker-scratch"
    assert scratch.is_dir()
    assert scratch != pathlib.Path(tempfile.gettempdir()).resolve()


def test_worker_scratch_parent_rejects_a_symlinked_output_root(
    tmp_path: pathlib.Path,
) -> None:
    real_output = tmp_path / "real-governed-run"
    real_output.mkdir()
    symlinked_output = tmp_path / "aliased-governed-run"
    symlinked_output.symlink_to(real_output, target_is_directory=True)

    with pytest.raises(TrainWorkflowError, match="run output root must not be a symbolic link"):
        runner._durable_worker_scratch_parent(symlinked_output)


def test_owned_preflight_shutdown_terminates_the_gateway_before_worker_training() -> None:
    class FakeSpark:
        def __init__(self) -> None:
            self.stopped = False

        def stop(self) -> None:
            self.stopped = True

    class FakeGateway:
        def __init__(self) -> None:
            self.shutdowns = 0

        def shutdown(self) -> None:
            self.shutdowns += 1

    spark = FakeSpark()
    gateway = FakeGateway()
    session_type = SimpleNamespace(_activeSession=object(), _instantiatedSession=object())
    context_type = SimpleNamespace(
        _active_spark_context=object(),
        _gateway=gateway,
        _jvm=object(),
    )

    runner._shutdown_owned_preflight_runtime(
        spark,
        session_type=session_type,
        context_type=context_type,
    )

    assert spark.stopped
    assert gateway.shutdowns == 1
    assert session_type._activeSession is None
    assert session_type._instantiatedSession is None
    assert context_type._active_spark_context is None
    assert context_type._gateway is None
    assert context_type._jvm is None


@pytest.mark.parametrize("memory_gib", [16, 64])
def test_preflight_preserves_the_isolated_parent_and_worker_budgets(
    monkeypatch: pytest.MonkeyPatch, memory_gib: int
) -> None:
    machine = runner.resources.MachineProfile(memory_gib * 1024**3, 10, "test")
    monkeypatch.setattr(runner.resources, "detect", lambda: machine)
    expected = runner.resources.isolated_spark_profiles()
    assert runner._preflight_spark_profiles((SimpleNamespace(adapter="hmfnet"),)) == expected


def test_parameter_resolution_provenance_uses_private_validated_bytes(
    tmp_path: pathlib.Path,
) -> None:
    original = tmp_path / "caller-resolution.json"
    record = {"schema": "example", "value": 1}
    payload = json.dumps(record, allow_nan=False, indent=2, sort_keys=True) + "\n"
    original.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode()).hexdigest()
    private = tmp_path / "private"
    private.mkdir()

    frozen = runner._freeze_parameter_resolution(private, record, digest)
    original.write_text('{"schema":"changed"}\n', encoding="utf-8")

    assert frozen.read_text(encoding="utf-8") == payload
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize(
    "changes",
    [
        {"id": "changed-model"},
        {"adapter": "changed-adapter"},
        {"runtime": "changed-runtime"},
        {"source_path": pathlib.Path("/changed/model.yaml")},
        {"source_sha256": "0" * 64},
        {"implementation_config_path": pathlib.Path("/changed/config.json")},
        {"implementation_config_sha256": "0" * 64},
    ],
)
def test_hpo_candidate_binding_rejects_every_changed_configuration_identity(
    changes: dict[str, object],
) -> None:
    expected = next(
        item
        for item in discover_candidates(runner.ROOT / "configs/models")
        if item.id == "sft"
    )
    changed = dataclasses.replace(expected, **changes)
    with pytest.raises(TrainWorkflowError, match="changed after the HPO caller"):
        runner._require_expected_candidate((changed,), expected)


def test_raw_hpo_runner_rejects_catalog_swap_before_runtime_admission(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = next(
        item
        for item in discover_candidates(runner.ROOT / "configs/models")
        if item.id == "sft"
    )
    changed = dataclasses.replace(
        expected,
        source_path=runner.ROOT / "configs/models/relocated/model.yaml",
        source_sha256="0" * 64,
        implementation_config_path=runner.ROOT / "configs/experiments/relocated.json",
        implementation_config_sha256="1" * 64,
    )
    request = RunRequest(
        name="candidate-binding",
        dataset_contract="configs/datasets/synthetic_daily.yaml",
        model_catalog="configs/models",
        models=(expected.id,),
        origin="2024-01-31",
        horizon=7,
    )
    monkeypatch.setattr(runner, "load_run_request", lambda _path: request)
    monkeypatch.setattr(runner, "_runtime_data_path", lambda path: path)
    monkeypatch.setattr(runner, "_inside_repository", lambda path, _label: path)
    monkeypatch.setattr(runner, "capture_source_identity", lambda _root: object())
    monkeypatch.setattr(
        runner,
        "capture_runtime_identity",
        lambda _root, *, image_digest: object(),
    )
    monkeypatch.setattr(runner, "discover_candidates", lambda _catalog: (changed,))
    reached_runtime = False

    def check_runtime(_runtime: str):
        nonlocal reached_runtime
        reached_runtime = True
        return SimpleNamespace(available=True, summary="available")

    monkeypatch.setattr(runner, "check_runtime", check_runtime)
    (tmp_path / "request.yaml").write_text("retained request bytes")
    with pytest.raises(TrainWorkflowError, match="changed after the HPO caller"):
        runner.run_request(
            tmp_path / "request.yaml",
            data_path=tmp_path / "data",
            data_snapshot_id="data-v1",
            data_reference="local://data-v1",
            output_root=tmp_path / "runs",
            epochs=2,
            timeout_seconds=10,
            parameter_overrides={
                "model.encoder_days": 28,
                "training.learning_rate": 0.0003,
                "training.train_stride": 7,
                "training.train_windows": 3,
            },
            expected_candidate=expected,
        )
    assert not reached_runtime


@pytest.mark.parametrize(
    ("parameter_overrides", "supply_expected"),
    [({"model.encoder_days": 28}, False), (None, True)],
)
def test_raw_hpo_overrides_and_frozen_candidate_are_atomic(
    tmp_path: pathlib.Path,
    parameter_overrides: dict[str, object] | None,
    supply_expected: bool,
) -> None:
    candidate = next(
        item
        for item in discover_candidates(runner.ROOT / "configs/models")
        if item.id == "sft"
    )
    with pytest.raises(TrainWorkflowError, match="must be supplied together"):
        runner.run_request(
            tmp_path / "request.yaml",
            data_path=tmp_path / "data",
            data_snapshot_id="data-v1",
            data_reference="local://data-v1",
            output_root=tmp_path / "runs",
            epochs=2,
            timeout_seconds=10,
            parameter_overrides=parameter_overrides,
            expected_candidate=candidate if supply_expected else None,
        )


def test_governed_runner_refuses_identical_training_and_score_content_at_separate_paths(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path = tmp_path / "request.yaml"
    request_path.write_text(
        """schema_version: 3
run:
  precision: 32-true
  name: separate-score
  dataset_contract: configs/datasets/synthetic_daily.yaml
  model_catalog: configs/models
  models: [seasonal_naive]
  origin: '2024-01-10'
  horizon: 2
  tracking:
    mlflow: {enabled: false}
  evaluation:
    missing_actual: no_data
    metric: mae
    input_mode: post_training_score_only
    evidence_role: rolling_validation
"""
    )
    training = tmp_path / "training"
    scoring = tmp_path / "scoring"
    training.mkdir()
    scoring.mkdir()
    (training / "part.parquet").write_bytes(b"identical")
    (scoring / "renamed.parquet").write_bytes(b"identical")
    monkeypatch.setattr(runner, "_inside_repository", lambda path, _label: path.resolve())
    with pytest.raises(TrainWorkflowError, match="content distinct"):
        runner.run_request(
            request_path,
            data_path=training,
            data_snapshot_id="training-v1",
            data_reference="local://training-v1",
            output_root=tmp_path / "runs",
            epochs=1,
            timeout_seconds=10,
            evaluation_data_path=scoring,
            evaluation_data_snapshot_id="score-v1",
            evaluation_data_reference="local://score-v1",
        )


def test_post_training_evaluation_refuses_to_open_score_data_without_all_provenance_inputs(
    tmp_path: pathlib.Path,
) -> None:
    forecast = Forecast(
        "seasonal_naive",
        str(tmp_path / "forecast"),
        "2016-05-22",
        28,
        1,
        {"emitted": 28},
    )
    with pytest.raises(TrainWorkflowError, match="requires a separate score panel"):
        runner._evaluate_completed_forecasts(
            request=_request(),
            training_panel=Panel(str(tmp_path / "training"), "id", "date", "qty"),
            forecasts={"seasonal_naive": forecast},
            run_path=tmp_path / "run",
        )


def test_post_training_evaluation_fails_when_the_metric_refuses_any_candidate(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBuilder:
        def master(self, _value: str) -> FakeBuilder:
            return self

        def appName(self, _value: str) -> FakeBuilder:
            return self

        def config(self, _key: str, _value: str) -> FakeBuilder:
            return self

        def getOrCreate(self) -> SimpleNamespace:
            return SimpleNamespace(stop=lambda: None)

    run_path = tmp_path / "run"
    run_path.mkdir()
    forecast = Forecast(
        "seasonal_naive", str(tmp_path / "forecast"), "2016-05-22", 28, 1, {"emitted": 28}
    )

    def compare(*_args: object, output_dir: str | pathlib.Path, **_kwargs: object) -> pathlib.Path:
        output = pathlib.Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        artifact = output / "candidate-comparison.json"
        artifact.write_text(
            json.dumps(
                {
                    "candidates": [{"model_id": "seasonal_naive", "status": "not_scored"}],
                    "ranking": None,
                }
            ),
            encoding="utf-8",
        )
        return artifact

    monkeypatch.setattr(
        runner.resources,
        "spark_profile",
        lambda **_kwargs: SimpleNamespace(cores=1, as_config=lambda: {}),
    )
    monkeypatch.setattr(runner.SparkSession, "builder", FakeBuilder())
    monkeypatch.setattr(runner, "compare_run_forecasts", compare)
    monkeypatch.setattr(runner, "directory_digest", lambda *_args, **_kwargs: "a" * 64)

    assert not runner._evaluate_completed_forecasts(
        request=_request(),
        training_panel=Panel(str(tmp_path / "training"), "id", "date", "qty"),
        score_panel=Panel(str(tmp_path / "score"), "id", "date", "qty"),
        score_data_snapshot_id="m5_private_score_v1",
        score_data_reference="derived://m5-private-score-v1",
        forecasts={"seasonal_naive": forecast},
        run_path=run_path,
    )
    record = json.loads((run_path / "evaluation" / "post-training-evaluation.json").read_text())
    assert record["status"] == "failed"
    assert "seasonal_naive" in record["error"]


def test_archived_m5_replay_refuses_a_pending_selection_lock() -> None:
    root = pathlib.Path(__file__).resolve().parent.parent
    request = runner.load_run_request(
        root / "configs/autoforecast/m5_private_evaluation_deep_v2.yaml"
    )
    assert request.evaluation is not None
    assert request.evaluation.selection_lock is not None
    with pytest.raises(TrainWorkflowError, match="pending: complete and commit"):
        runner._sealed_m5_selection_lock(
            request,
            (SimpleNamespace(id="mft", source_sha256="a" * 64),),
            request_path=root / "configs/autoforecast/m5_private_evaluation_deep_v2.yaml",
            epochs=5000,
            seed=0,
            training_controls={
                "patience": 15,
                "device": "mps",
                "batch_size": 1024,
                "gradient_microbatch_size": 128,
                "early_stopping_absolute_min_delta": 1e-6,
                "early_stopping_relative_min_delta": 0.0,
            },
        )


def test_archived_m5_final_scoring_cannot_bypass_selection_by_relabeling_evidence(
    tmp_path: pathlib.Path,
) -> None:
    request_path = tmp_path / "request.yaml"
    request_path.write_text("request: test\n")
    request = RunRequest(
        "m5-final",
        "configs/datasets/m5_daily.yaml",
        "configs/models",
        ("seasonal_naive",),
        "2016-05-22",
        28,
        evaluation=EvaluationRequest(
            "no_data",
            "wrmsse",
            "post_training_score_only",
            "run_local_exploratory",
        ),
    )
    with pytest.raises(TrainWorkflowError, match="archived M5 final window"):
        runner._sealed_m5_selection_lock(
            request,
            (SimpleNamespace(id="seasonal_naive", source_sha256="a" * 64),),
            request_path=request_path,
            epochs=2,
            seed=0,
            training_controls={},
        )


def test_archived_m5_role_cannot_be_applied_to_an_unrelated_dataset(
    tmp_path: pathlib.Path,
) -> None:
    request_path = tmp_path / "request.yaml"
    request_path.write_text("request: test\n")
    request = RunRequest(
        "fake-m5-final",
        "configs/datasets/synthetic_daily.yaml",
        "configs/models",
        ("seasonal_naive",),
        "2016-05-22",
        28,
        evaluation=EvaluationRequest(
            "no_data",
            "wrmsse",
            "post_training_score_only",
            "archived_private_replay_not_blind",
        ),
    )
    with pytest.raises(TrainWorkflowError, match="canonical M5 contract content"):
        runner._sealed_m5_selection_lock(
            request,
            (SimpleNamespace(id="seasonal_naive", source_sha256="a" * 64),),
            request_path=request_path,
            epochs=2,
            seed=0,
            training_controls={},
        )


def test_training_only_m5_final_refit_does_not_require_the_score_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = SimpleNamespace(
        dataset_contract=pathlib.Path("configs/datasets/m5_daily.yaml"),
        origin="2016-05-22",
        horizon=28,
        evaluation=None,
    )
    monkeypatch.setattr(runner, "_inside_repository", lambda path, _label: path)
    assert (
        runner._sealed_m5_selection_lock(
            request,
            (),
            request_path=runner.ROOT / "configs/autoforecast/m5_private_evaluation_deep_v2.yaml",
            epochs=5000,
            seed=0,
            training_controls={},
        )
        is None
    )


def test_archived_m5_replay_binds_a_sealed_selection_lock(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_path = tmp_path / "request.yaml"
    lock_path = tmp_path / "lock.yaml"
    profile_sha256 = "b" * 64
    request_path.write_text("request: test\n", encoding="utf-8")
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text('{"status":"scored"}\n', encoding="utf-8")
    comparison_sha256 = hashlib.sha256(comparison_path.read_bytes()).hexdigest()
    contract = tmp_path / "configs/datasets/m5_daily.yaml"
    contract.parent.mkdir(parents=True)
    contract.write_bytes((runner.ROOT / "configs/datasets/m5_daily.yaml").read_bytes())
    request_sha256 = hashlib.sha256(request_path.read_bytes()).hexdigest()
    lock_path.write_text(
        f"""schema_version: 1
selection_lock:
  id: m5_accuracy_final_selection_v2
  status: locked
  protocol: m5_accuracy_archived_replay_v1
  public_holdout:
    request: configs/autoforecast/m5_public_holdout_deep_v2.yaml
    origin: '2016-04-24'
    horizon: 28
    candidate_comparison: comparison.json
    candidate_comparison_sha256: {comparison_sha256}
  final_refit:
    request: request.yaml
    origin: '2016-05-22'
    horizon: 28
    model_id: mft
    model_profile_sha256: {profile_sha256}
    run_request_sha256: {request_sha256}
    seed: 0
    maximum_epochs: 5000
    patience: 15
    device: mps
    batch_size: 1024
    gradient_microbatch_size: 128
    early_stopping_absolute_min_delta: 0.000001
    early_stopping_relative_min_delta: 0.0
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    request = RunRequest(
        "sealed-private-replay",
        "configs/datasets/m5_daily.yaml",
        "configs/models",
        ("mft",),
        "2016-05-22",
        28,
        evaluation=EvaluationRequest(
            "no_data",
            "wrmsse",
            "post_training_score_only",
            "archived_private_replay_not_blind",
            "lock.yaml",
        ),
    )
    controls = {
        "patience": 15,
        "device": "mps",
        "batch_size": 1024,
        "gradient_microbatch_size": 128,
        "early_stopping_absolute_min_delta": 1e-6,
        "early_stopping_relative_min_delta": 0.0,
    }
    sealed = runner._sealed_m5_selection_lock(
        request,
        (SimpleNamespace(id="mft", source_sha256=profile_sha256),),
        request_path=request_path,
        epochs=5000,
        seed=0,
        training_controls=controls,
    )
    assert sealed is not None
    assert sealed[0] == lock_path
    assert sealed[2]["id"] == "m5_accuracy_final_selection_v2"
    with pytest.raises(TrainWorkflowError, match="does not match"):
        runner._sealed_m5_selection_lock(
            request,
            (SimpleNamespace(id="mft", source_sha256=profile_sha256),),
            request_path=request_path,
            epochs=5000,
            seed=0,
            training_controls={**controls, "batch_size": 2048},
        )

    comparison_path.write_text('{"status":"changed"}\n', encoding="utf-8")
    with pytest.raises(TrainWorkflowError, match="comparison bytes changed"):
        runner._sealed_m5_selection_lock(
            request,
            (SimpleNamespace(id="mft", source_sha256=profile_sha256),),
            request_path=request_path,
            epochs=5000,
            seed=0,
            training_controls=controls,
        )
    comparison_path.write_text('{"status":"scored"}\n', encoding="utf-8")

    lock_path.write_text("schema_version: 1\n" + lock_path.read_text(), encoding="utf-8")
    with pytest.raises(TrainWorkflowError, match="duplicate YAML mapping key"):
        runner._sealed_m5_selection_lock(
            request,
            (SimpleNamespace(id="mft", source_sha256=profile_sha256),),
            request_path=request_path,
            epochs=5000,
            seed=0,
            training_controls=controls,
        )


@pytest.mark.parametrize("adapter", [
    "sft", "hmfnet", "pact", "tft_q", "tft_hnb",
])
@pytest.mark.parametrize("metric,policy", [("wrmsse", "zero"), ("rmse", "zero"), ("rmse", "mask")])
def test_shared_study_refuses_native_checkpoint_adapters_before_runtime(
    tmp_path, monkeypatch, adapter, metric, policy,
):
    candidates = discover_candidates(runner.ROOT / "configs/models")
    candidate = next(item for item in candidates if item.adapter == adapter)
    request_path = runner.ROOT / "configs/autoforecast/m5_tft_q_reference_smoke.yaml"
    request = dataclasses.replace(runner.load_run_request(request_path),
                                  models=(candidate.id,), evaluation=None)
    monkeypatch.setattr(runner, "load_run_request", lambda *_: request)
    monkeypatch.setattr(runner, "_runtime_data_path", lambda path: path)
    monkeypatch.setattr(runner, "capture_source_identity", lambda *_: object())
    monkeypatch.setattr(runner, "capture_runtime_identity", lambda *_, **kwargs: object())
    def unexpected_runtime(*args, **kwargs):
        pytest.fail("shared study reached the native training runtime")
    monkeypatch.setattr(runner.resources, "acquire_local_runtime", unexpected_runtime)
    from autoforecast import study_scoring
    # Study authentication has its own real-artifact tests. Isolate the worker capability
    # boundary so removing its guard reaches the runtime sentinel, not an incomplete fixture.
    monkeypatch.setattr(study_scoring, "verify_study_evaluation", lambda _: (None, {}))
    fixed = tmp_path / "evaluation.json"
    fixed.write_text(json.dumps({"schema": "autoforecast.fixed-evaluation.v2",
                                 "metric_policy": {"metric": metric, "actual_missing": policy}}))
    with pytest.raises(TrainWorkflowError, match="does not support shared studies"):
        runner.run_request(
            request_path, data_path=tmp_path / "data", data_snapshot_id="test",
            data_reference="local://test", output_root=tmp_path / "runs", epochs=2,
            timeout_seconds=10, evaluation_contract_path=fixed,
            evaluation_contract_sha256=hashlib.sha256(fixed.read_bytes()).hexdigest(),
        )
