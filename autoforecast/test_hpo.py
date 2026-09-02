"""Functional Optuna execution test: trial artifacts, telemetry and test-set isolation."""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import pathlib
import sys
import tempfile
import threading
from types import SimpleNamespace

import pytest

import autoforecast.hpo as hpo_module
from autoforecast.experiment_report import render_benchmark_report
from autoforecast.experiments import (
    ExperimentError,
    require_heldout_evaluation_approval,
    seal_study_selection,
)
from autoforecast.hpo import (
    HeldoutOutcome,
    HeldoutRequest,
    HPOError,
    TrialOutcome,
    TrialRequest,
    load_hpo_study,
    run_heldout_evaluation,
    run_hpo_study,
    sync_hpo_selection_to_mlflow,
)
from autoforecast.observability import open_training_log


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write_study(
    root: pathlib.Path,
    *,
    backend: str = "optuna",
    evaluator: str = "autoforecast.test_hpo:smoke_evaluator",
) -> pathlib.Path:
    config = root / "model.yaml"
    config.write_text(
        "schema_version: 1\nmodel:\n  id: example_model\n"
        "  adapter: mft\n  family: custom\n"
        "  status: candidate\n  tags: [development]\nhyperparameters:\n"
        "  training.learning_rate:\n    default: 0.001\n"
        "    search: {kind: categorical, values: [0.001, 0.003]}\n"
    )
    study = root / "study.yaml"
    fingerprint = _sha("same frozen evaluation contract")
    data_fingerprint = _sha("snapshot")
    study.write_text(
        "schema_version: 1\nstudy:\n  id: hpo-smoke\n"
        f"  model_config: {config.name}\n  backend: {backend}\n  direction: minimize\n"
        "  objective: validation/objective\n  max_trials: 2\n  top_k: 1\n  seed: 9\n"
        f"  comparability_fingerprint: {fingerprint}\n"
        f"  evaluator: {evaluator}\n  data:\n"
        f"    id: generated\n    fingerprint: {data_fingerprint}\n"
        "    observed_date_range: ['2024-01-01', '2024-02-14']\n"
        "    training_date_range: ['2024-01-01', '2024-02-10']\n"
        "    validation_date_range: ['2024-01-01', '2024-02-10']\n"
        "    test_date_range: ['2024-02-13', '2024-02-14']\n"
        "    validation_target_date_range: ['2024-02-11', '2024-02-12']\n"
        "    test_target_date_range: ['2024-02-13', '2024-02-14']\n"
        "    training_series: 5\n    validation_series: 1\n    test: reserved_not_evaluated\n"
    )
    return study


def smoke_evaluator(request: TrialRequest) -> TrialOutcome:
    """Tiny model bridge that writes the same mandatory per-trial evidence as a real trainer."""
    raw_learning_rate = request.training_parameters["learning_rate"]
    assert isinstance(raw_learning_rate, int | float)
    learning_rate = float(raw_learning_rate)
    log = open_training_log(
        None,
        metrics_path=request.attempt.metrics_path,
        epoch_observer=request.epoch_observer,
    )
    for epoch in range(1, 6):
        log.log_epoch(
            epoch,
            train_objective=2.0 / epoch,
            validation_objective=1.0 + learning_rate,
            learning_rate=learning_rate,
            gradient_norm=0.5,
            optimizer_steps=epoch,
            training_examples_seen=epoch * 5,
            seconds=0.01,
            process_peak_rss_bytes=1,
            cuda_peak_memory_bytes=None,
        )
    log.close()
    (request.attempt.path / "model" / "model.bin").write_bytes(b"model")
    return TrialOutcome(validation_objective=1.0 + learning_rate, metrics={"evaluation/mae": 1.0})


def heldout_smoke_evaluator(request: HeldoutRequest) -> HeldoutOutcome:
    """Loads the selected model artifact and emits only fresh held-out evidence."""
    assert (request.source_attempt / "model/model.bin").read_bytes() == b"model"
    return HeldoutOutcome(heldout_objective=0.8, metrics={"heldout/mae": 0.8})


@pytest.mark.parametrize(
    ("outcome", "message"),
    [
        (TrialOutcome(validation_objective=True), "finite TrialOutcome"),
        (TrialOutcome(validation_objective=10**400), "finite TrialOutcome"),
        (
            TrialOutcome(validation_objective=1.0, metrics={"private_test": 1.0}),
            "may not report test metrics",
        ),
        (
            TrialOutcome(validation_objective=1.0, metrics={"validation/mae": True}),
            "must be finite",
        ),
    ],
)
def test_hpo_trial_refuses_boolean_objectives_and_disguised_test_metrics(
    tmp_path: pathlib.Path,
    outcome: TrialOutcome,
    message: str,
) -> None:
    study = load_hpo_study(_write_study(tmp_path), project_root=tmp_path)
    with pytest.raises(HPOError, match=message):
        hpo_module._evaluate(
            study,
            tmp_path / "artifacts",
            "trial-invalid",
            {"training.learning_rate": 0.001},
            lambda _request: outcome,
            mlflow_tracking_uri=None,
            mlflow_experiment="test",
        )


def test_optional_mlflow_start_failure_does_not_discard_a_completed_trial(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = load_hpo_study(_write_study(tmp_path), project_root=tmp_path)
    monkeypatch.setattr(
        hpo_module,
        "start_training_tracking",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("tracking offline")),
    )

    value = hpo_module._evaluate(
        study,
        tmp_path / "artifacts",
        "trial-mlflow-offline",
        {"training.learning_rate": 0.001},
        smoke_evaluator,
        mlflow_tracking_uri="https://token@example.invalid/mlflow?signature=secret",
        mlflow_experiment="test",
    )

    assert value == pytest.approx(1.001)
    attempt = next((tmp_path / "artifacts/studies/hpo-smoke/trials").glob("*/attempts/*"))
    tracking = json.loads((attempt / "tracking/mlflow.json").read_text(encoding="utf-8"))
    assert tracking["status"] == "unavailable"
    assert "token" not in json.dumps(tracking)


def test_hpo_evaluator_cannot_substitute_the_retained_mlflow_run_id(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = load_hpo_study(_write_study(tmp_path), project_root=tmp_path)

    class Live:
        experiment_id = "experiment-real"
        run_id = "run-real"
        log_epoch = None

        def complete(self, _path: pathlib.Path) -> None:
            pass

        def fail(self) -> None:
            pass

    monkeypatch.setattr(hpo_module, "start_training_tracking", lambda **_kwargs: Live())

    def malicious(request: TrialRequest) -> TrialOutcome:
        request.attempt.write_json(
            "tracking/mlflow.json",
            {
                "schema": "autoforecast.mlflow-tracking-link.v2",
                "status": "started",
                "run_id": "victim-run",
            },
        )
        return smoke_evaluator(request)

    with pytest.raises(ExperimentError, match="could not write attempt record"):
        hpo_module._evaluate(
            study,
            tmp_path / "artifacts",
            "trial-forged-link",
            {"training.learning_rate": 0.001},
            malicious,
            mlflow_tracking_uri="sqlite:///tracking.db",
            mlflow_experiment="test",
        )


def test_optuna_runner_creates_complete_validation_only_trial_artifacts() -> None:
    pytest.importorskip("optuna")
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        spec = load_hpo_study(_write_study(root), project_root=root)
        report = run_hpo_study(spec, root / "artifacts", smoke_evaluator)
        assert report.exists()
        study_root = root / "artifacts/studies/hpo-smoke"
        selection = json.loads((study_root / "selection.json").read_text())
        chosen = selection["selected"][0]
        selected = require_heldout_evaluation_approval(
            study_root,
            trial_id=chosen["trial_id"],
            attempt_id=chosen["attempt_id"],
        )
        assert selected == chosen
        attempts = sorted((root / "artifacts/studies/hpo-smoke").glob("trials/*/attempts/*"))
        assert len(attempts) == 2
        for attempt in attempts:
            assert (attempt / "completion.json").exists()
            assert (attempt / "metrics/epochs.jsonl").exists()
            assert (attempt / "model/model.bin").exists()
            assert not list((attempt / "tensorboard").glob("events.out.tfevents.*"))
            validation = (attempt / "evaluation/validation.json").read_text()
            assert "test_metrics" in validation and "test/" not in validation


def test_same_hpo_study_refuses_a_second_concurrent_paid_compute_owner(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = load_hpo_study(_write_study(tmp_path), project_root=tmp_path)
    entered = threading.Event()
    release = threading.Event()
    outcomes: list[object] = []

    def hold_backend(*_args: object, **_kwargs: object) -> None:
        entered.set()
        assert release.wait(timeout=5)

    monkeypatch.setattr(hpo_module, "_run_optuna", hold_backend)
    monkeypatch.setattr(hpo_module, "seal_study_selection", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(hpo_module, "sync_hpo_selection_to_mlflow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        hpo_module,
        "render_study_report",
        lambda *_args, **_kwargs: tmp_path / "report.html",
    )

    def first_owner() -> None:
        try:
            outcomes.append(run_hpo_study(spec, tmp_path / "artifacts", smoke_evaluator))
        except Exception as error:  # pragma: no cover - assertion below exposes the error
            outcomes.append(error)

    owner = threading.Thread(target=first_owner)
    owner.start()
    assert entered.wait(timeout=5)
    try:
        with pytest.raises(HPOError, match="already running"):
            run_hpo_study(spec, tmp_path / "artifacts", smoke_evaluator)
    finally:
        release.set()
        owner.join(timeout=5)
    assert outcomes == [tmp_path / "report.html"]


def test_hpo_config_refuses_test_as_an_objective() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        study = _write_study(root)
        study.write_text(
            study.read_text().replace("objective: validation/objective", "objective: test/wrmsse")
        )
        with pytest.raises(ValueError, match="test is forbidden"):
            load_hpo_study(study, project_root=root)


def test_hpo_ranges_preserve_exact_hour_keys_and_reject_mixed_key_types() -> None:
    assert hpo_module._date_range(  # noqa: SLF001
        ["2024-03-01T22:00:00", "2024-03-02T01:00:00"], "hourly range"
    ) == (dt.datetime(2024, 3, 1, 22), dt.datetime(2024, 3, 2, 1))
    with pytest.raises(HPOError, match="same temporal key type"):
        hpo_module._date_range(  # noqa: SLF001
            ["2024-03-01", "2024-03-02T01:00:00"], "mixed range"
        )


def test_hpo_config_refuses_training_or_test_overlap_with_validation_targets() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        study = _write_study(root)
        study.write_text(
            study.read_text().replace(
                "training_date_range: ['2024-01-01', '2024-02-10']",
                "training_date_range: ['2024-01-01', '2024-02-11']",
            )
        )
        with pytest.raises(ValueError, match="training_date_range must end before"):
            load_hpo_study(study, project_root=root)


def test_hpo_config_accepts_the_governed_builtin_bridge() -> None:
    """A user can select the supported runner without providing executable Python in YAML."""
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        spec = load_hpo_study(_write_study(root, evaluator="builtin"), project_root=root)
        assert spec.evaluator == "builtin"


def test_hpo_study_rejects_duplicate_yaml_keys() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        study = _write_study(root)
        study.write_text(
            study.read_text(encoding="utf-8").replace(
                "  max_trials: 2\n", "  max_trials: 2\n  max_trials: 3\n"
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="duplicate YAML mapping key"):
            load_hpo_study(study, project_root=root)


def test_optuna_hpo_publishes_all_five_epoch_candidates_to_real_mlflow() -> None:
    mlflow = pytest.importorskip("mlflow")
    pytest.importorskip("optuna")
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        spec = load_hpo_study(_write_study(root), project_root=root)
        tracking_uri = f"sqlite:///{root / 'mlflow.db'}"
        run_hpo_study(
            spec,
            root / "artifacts",
            smoke_evaluator,
            mlflow_tracking_uri=tracking_uri,
            mlflow_experiment="autoforecast-hpo-functional",
        )
        client = mlflow.tracking.MlflowClient(tracking_uri)
        experiment = client.get_experiment_by_name("autoforecast-hpo-functional")
        assert experiment is not None
        runs = client.search_runs([experiment.experiment_id])
        assert len(runs) == 2
        for run in runs:
            assert run.info.status == "FINISHED"
            assert run.data.tags["autoforecast.run_kind"] == "hpo_candidate"
            assert run.data.tags["autoforecast.hpo_selection"] in {
                "selected_for_heldout",
                "not_selected",
            }
            assert (
                len(
                    client.get_metric_history(
                        run.info.run_id,
                        "optimization/train_objective",
                    )
                )
                == 5
            )
            assert "evaluation/objective" in run.data.metrics
            assert "model" in {item.path for item in client.list_artifacts(run.info.run_id)}


def test_hpo_config_refuses_heldout_targets_that_overlap_validation_targets() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        study = _write_study(root)
        study.write_text(
            study.read_text().replace(
                "test_target_date_range: ['2024-02-13', '2024-02-14']",
                "test_target_date_range: ['2024-02-12', '2024-02-14']",
            ),
        )
        with pytest.raises(ValueError, match="strictly after"):
            load_hpo_study(study, project_root=root)


def test_selected_candidate_gets_a_separate_heldout_bundle() -> None:
    pytest.importorskip("optuna")
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        spec = load_hpo_study(_write_study(root), project_root=root)
        run_hpo_study(spec, root / "artifacts", smoke_evaluator)
        study_root = root / "artifacts/studies/hpo-smoke"
        selected = json.loads((study_root / "selection.json").read_text())["selected"][0]
        completion = run_heldout_evaluation(
            spec,
            root / "artifacts",
            trial_id=selected["trial_id"],
            attempt_id=selected["attempt_id"],
            evaluator=heldout_smoke_evaluator,
        )
        heldout = json.loads((completion.parent / "evaluation/heldout.json").read_text())
        assert heldout["value"] == 0.8
        assert heldout["selected_source"]["attempt_id"] == selected["attempt_id"]
        assert "heldout/trials" in completion.as_posix()
        benchmark = render_benchmark_report([study_root], root / "benchmark.html")
        page = benchmark.read_text()
        assert "Held-out benchmark" in page
        assert "0.8" in page
        assert "Validation metrics and smoke evidence are absent" in page


def test_ray_runner_creates_the_same_complete_trial_artifacts() -> None:
    ray = pytest.importorskip("ray")
    initially_initialized = ray.is_initialized()
    with tempfile.TemporaryDirectory() as temporary:
        root = pathlib.Path(temporary)
        spec = load_hpo_study(_write_study(root, backend="ray"), project_root=root)
        try:
            report = run_hpo_study(spec, root / "artifacts", smoke_evaluator)
            attempts = sorted((root / "artifacts/studies/hpo-smoke").glob("trials/*/attempts/*"))
            assert report.exists() and len(attempts) == 2
            assert all((attempt / "completion.json").exists() for attempt in attempts)
            assert any((root / "artifacts/ray").rglob("*"))
            assert ray.is_initialized() is initially_initialized
        finally:
            if not initially_initialized and ray.is_initialized():
                ray.shutdown()


def test_mlflow_selection_sync_failure_is_durable_and_retryable(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = load_hpo_study(_write_study(tmp_path), project_root=tmp_path)

    class Live:
        experiment_id = "experiment-1"
        run_id = "run-1"
        log_epoch = None

        def complete(self, _path: pathlib.Path) -> None:
            pass

        def fail(self) -> None:
            pass

    monkeypatch.setattr(hpo_module, "start_training_tracking", lambda **_kwargs: Live())
    root = tmp_path / "artifacts"
    hpo_module._evaluate(
        study,
        root,
        "trial-0000",
        {"training.learning_rate": 0.001},
        smoke_evaluator,
        mlflow_tracking_uri="sqlite:///tracking.db",
        mlflow_experiment="test",
    )
    study_root = root / "studies/hpo-smoke"
    seal_study_selection(study_root, top_k=1, direction="minimize")

    class BrokenClient:
        def __init__(self, _tracking_uri: str) -> None:
            raise RuntimeError("tracking unavailable")

    monkeypatch.setitem(
        sys.modules,
        "mlflow",
        SimpleNamespace(tracking=SimpleNamespace(MlflowClient=BrokenClient)),
    )
    path = sync_hpo_selection_to_mlflow(study_root, tracking_uri="sqlite:///broken.db")
    failed = json.loads(path.read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["error_type"] == "RuntimeError"

    class WorkingClient:
        def __init__(self, _tracking_uri: str) -> None:
            self.tags: list[tuple[str, str, str]] = []

        def get_run(self, run_id: str) -> SimpleNamespace:
            assert run_id == "run-1"
            return SimpleNamespace(
                data=SimpleNamespace(
                    tags={
                        "autoforecast.study_id": "hpo-smoke",
                        "autoforecast.trial_id": "trial-0000",
                        "autoforecast.attempt_id": next(
                            study_root.glob("trials/trial-0000/attempts/*")
                        ).name,
                    }
                ),
                info=SimpleNamespace(experiment_id="experiment-1"),
            )

        def set_tag(self, run_id: str, name: str, value: str) -> None:
            self.tags.append((run_id, name, value))

    monkeypatch.setitem(
        sys.modules,
        "mlflow",
        SimpleNamespace(tracking=SimpleNamespace(MlflowClient=WorkingClient)),
    )
    retried = json.loads(
        sync_hpo_selection_to_mlflow(study_root, tracking_uri="sqlite:///tracking.db").read_text(
            encoding="utf-8"
        )
    )
    assert retried["status"] == "synchronized"
    assert retried["error_type"] is None
    assert retried["attempted_run_ids"] == ["run-1"]
    assert retried["synchronized_run_ids"] == ["run-1"]


@pytest.mark.parametrize("evaluator_name", ["builtin", "external:fit_predict"])
def test_nonexternal_hpo_rejects_unvalidated_external_evidence(tmp_path, evaluator_name):
    study = load_hpo_study(_write_study(tmp_path, evaluator=evaluator_name), project_root=tmp_path)
    def evaluator(request):
        request.attempt.write_bytes("model/model.bin", b"model")
        request.attempt.write_bytes("evaluation/external-run.json", b"not-json")
        return TrialOutcome(1.0)
    with pytest.raises(HPOError, match="epoch metrics"):
        run_hpo_study(study, tmp_path / "artifacts", evaluator)
    assert not list((tmp_path / "artifacts").rglob("selection.json"))


@pytest.mark.parametrize("defect", [None, "missing", "empty", "malformed", "schema", "context", "model", "forecast", "test"])
def test_external_hpo_requires_exact_external_evidence(tmp_path, defect):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from provenance.lineage import directory_digest

    study = load_hpo_study(_write_study(tmp_path), project_root=tmp_path)
    context = {"input": "frozen"}
    fingerprint = hashlib.sha256((json.dumps(context, sort_keys=True) + "\n").encode()).hexdigest()
    study = dataclasses.replace(study, tuning=dataclasses.replace(study.tuning, adapter="external"),
                                evaluator="external:fit_predict", comparability_fingerprint=fingerprint)
    def evaluator(request):
        request.attempt.write_bytes("model/model.bin", b"model")
        forecasts = request.attempt.path / "forecasts"
        forecasts.mkdir()
        pq.write_table(pa.table({"series_id": ["A"], "date": [dt.date(2024, 2, 11)], "yhat": [1.0]}), forecasts / "part.parquet")
        record = {"schema": "autoforecast.external-trial.v1", "test_metrics": "not_evaluated",
                  "context": context, "model_sha256": directory_digest(request.attempt.path / "model"),
                  "forecast_sha256": directory_digest(forecasts, suffixes=(".parquet",))}
        if defect == "missing":
            return TrialOutcome(1.0)
        if defect in {"empty", "malformed"}:
            request.attempt.write_bytes("evaluation/external-run.json", b"" if defect == "empty" else b"not-json")
        else:
            if defect == "schema":
                record["schema"] = "wrong"
            elif defect == "context":
                record["context"] = {"input": "changed"}
            elif defect in {"model", "forecast"}:
                record[defect + "_sha256"] = "0" * 64
            elif defect == "test":
                record["test_metrics"] = "evaluated"
            request.attempt.write_json("evaluation/external-run.json", record)
        return TrialOutcome(1.0)
    if defect:
        with pytest.raises(HPOError, match="external trial evidence"):
            run_hpo_study(study, tmp_path / "artifacts", evaluator)
        assert not list((tmp_path / "artifacts").rglob("selection.json"))
    else:
        run_hpo_study(study, tmp_path / "artifacts", evaluator)
        assert (tmp_path / "artifacts/studies/hpo-smoke/selection.json").is_file()
