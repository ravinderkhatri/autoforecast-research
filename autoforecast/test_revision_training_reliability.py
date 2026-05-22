"""Actual M5 continuation and validation semantics, with explicit structural fault injection."""

import copy
import dataclasses
import datetime as dt
import hashlib
import importlib.util
import io
import json
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from autoforecast import rolling_training
from autoforecast.adapters.mft import (
    HMFNetRevisionFeatures,
    RevisionFeatures,
    fit_predict_revision,
)
from autoforecast.benchmarks.m5 import M5_LEVELS
from autoforecast.hierarchy_metric import HierarchyMetric, TorchHierarchyMetric
from autoforecast.revision_config import verify_validation_ledger
from autoforecast.rolling_data import ValidationWindow
from autoforecast.rolling_training import (
    RollingTrainer,
    RollingTrainingContract,
    rolling_origins,
)
from autoforecast.test_hmfnet_revision import hmfnet_config
from autoforecast.test_multibranch_revisions import small_config
from autoforecast.training_checkpoint import (
    TrainingRun,
    capture_training_rng,
    configure_training_runtime,
)
from autoforecast.training_progress import TrainingProgress

pytest_plugins = [
    "autoforecast.test_multibranch_revisions",
    "autoforecast.test_revision_portability",
]


@pytest.mark.parametrize("tamper", ["oversize", "duplicate", "empty", "label"])
def test_sampler_replay_rejects_producer_impossible_journal(tamper):
    masks = {"2024-01-01": np.ones((3, 28), dtype=bool)}
    coverage = {
        "epoch": 1, "proposed_series": 3, "minimum_row_proposals": 1,
        "maximum_row_proposals": 1, "informative_series": 3,
        "minimum_informative_visits": 1, "maximum_informative_visits": 1,
    }
    rows = [
        {"phase": "selection", "update": 1, "origin": "2024-01-01",
         "sampled_series_indices": [2, 0], "origin_sampling": "coverage_curriculum",
         "likelihood_series_in_batch": 2, "likelihood_informative_series_in_batch": 2},
        {"phase": "selection", "update": 2, "origin": "2024-01-01",
         "sampled_series_indices": [1], "origin_sampling": "coverage_curriculum",
         "likelihood_series_in_batch": 1, "likelihood_informative_series_in_batch": 1,
         "epoch_coverage": coverage},
    ]
    kwargs = dict(
        phase="selection", objective_masks=masks,
        blocks=[np.array([index]) for index in range(3)], batch_size=2,
        seed=0, epochs=50_000, hierarchy_every=1, hierarchy_only_enabled=False,
    )
    assert rolling_training.replay_sampler_journal(rows, **kwargs)["steps_per_epoch"] == 2
    changed = copy.deepcopy(rows)
    if tamper == "oversize":
        changed[0].update(sampled_series_indices=[0, 1, 2], likelihood_series_in_batch=3,
                          likelihood_informative_series_in_batch=3)
        changed[1].update(sampled_series_indices=[], origin_sampling="hierarchy_only",
                          likelihood_series_in_batch=0,
                          likelihood_informative_series_in_batch=0)
    elif tamper == "duplicate":
        changed[1]["sampled_series_indices"] = [0]
    elif tamper == "empty":
        changed[1].update(sampled_series_indices=[], origin_sampling="hierarchy_only",
                          likelihood_series_in_batch=0,
                          likelihood_informative_series_in_batch=0)
    else:
        changed[1]["origin_sampling"] = "uniform_usable_origins"
    with pytest.raises(ValueError, match="governed producer"):
        rolling_training.replay_sampler_journal(changed, **kwargs)


@pytest.mark.parametrize("metric", ["rmse", "mae", "wape", "wrmsse", "uniform_hierarchy_rmsse"])
def test_validation_ledger_can_resume_with_the_selected_metric(metric):
    verify_validation_ledger(
        [{"update": 1, "covered_series": 2, "trainable_series": 2,
          "evaluated": True}],
        [{"update": 1, f"mean_{metric}": 0.25}],
        metric=metric, validation_interval_updates=1,
        budget_updates=2, last_update=1, trainable_series=2,
    )


def test_verified_validation_summary_binds_per_origin_evidence():
    row = {
        "update": 2, "epoch": 2,
        "validation": {
            "2024-01-01": {"wrmsse": 0.25, "validation_nll": 1.0},
            "2024-01-02": {"wrmsse": 0.75, "validation_nll": 3.0},
        },
        "validation_summary": {
            "update": 2, "epoch": 2, "metric_name": "wrmsse",
            "metric": 0.5, "nll": 2.0,
        },
    }
    origins = ("2024-01-01", "2024-01-02")
    assert rolling_training.verified_validation_summary(row, origins, "wrmsse") == 0.5
    changed = copy.deepcopy(row)
    changed["validation_summary"]["metric"] = 0.1
    with pytest.raises(ValueError, match="differs from measurements"):
        rolling_training.verified_validation_summary(changed, origins, "wrmsse")


def recipe(variant, **changes):
    values = dict(epochs=2, checkpoint_every_updates=1, progress="off")
    values.update(changes)
    return hmfnet_config(**values) if variant == "v3" else small_config(variant, **values)


def assembled(revision_data, variant, **changes):
    panel, data, calendar = revision_data
    c = recipe(variant, **changes)
    features = (HMFNetRevisionFeatures if variant == "v3" else RevisionFeatures)(
        panel, data, c, str(calendar)
    )
    trainer = RollingTrainer(
        data,
        c,
        M5_LEVELS,
        features.groups,
        features.make_model,
        features.batch,
        features.training_loss,
        features.mean,
        features.optimizer_groups,
        features.begin_phase,
        features.prediction_context if variant == "v3" else None,
        features.diagnostics,
        validation_values=features.validation_values,
        feature_training_state=features.feature_training_state,
        restore_feature_training_state=features.restore_feature_training_state,
    )
    return features, trainer


def test_feature_state_callbacks_must_be_paired(revision_data):
    _, trainer = assembled(revision_data, "v2")
    trainer.restore_feature_training_state = None
    with pytest.raises(ValueError, match="capture and restore"):
        trainer._fit(lambda _: None, None, TrainingProgress("off"))


@pytest.mark.parametrize("precision", ["32-true", "16-mixed"])
@pytest.mark.parametrize("study_metric", [None, "wrmsse"])
def test_shared_trainer_exposes_model_agnostic_training_and_validation_regions(
    precision, study_metric,
):
    class SpyProfiler:
        def __init__(self):
            self.begun = []
            self.ended = []
            self.regions = []
            self.finished = []
            self.closed = 0

        def begin_step(self, phase):
            self.begun.append(phase)

        @contextmanager
        def record(self, phase, name):
            self.regions.append((phase, name))
            yield

        def end_step(self, phase):
            self.ended.append(phase)

        def finish_phase(self, phase):
            self.finished.append(phase)

        def close(self):
            self.closed += 1

    start = dt.date(2024, 1, 1)
    target = np.arange(60, dtype=np.float32).reshape(2, 30) % 5
    data = rolling_training.PanelArrays(
        ("a", "b"),
        start,
        start + dt.timedelta(days=29),
        2,
        target,
        np.ones_like(target, dtype=bool),
        np.empty((2, 0), dtype=np.float64),
        {"series": ("a", "b")},
    )
    holdout = ValidationWindow(
        data.cutoff.isoformat(),
        np.ones((2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=bool),
    )
    study = None
    if study_metric is not None:
        from autoforecast.study_validation import StudyValidation

        study = StudyValidation(
            json.dumps({
                "study_id": "b" * 64,
                "boundary": f"validation/{data.cutoff.isoformat()}/2/D",
                "metric_policy": {"metric": "wrmsse", "actual_missing": "mask"},
            }), "a" * 64, data.series_ids,
            np.array([[1.0, 0.0], [2.0, 4.0]]),
            np.array([[True, False], [True, True]]),
            np.array([1.0, 3.0]), np.array([1.0, 4.0]),
        )
        holdout = ValidationWindow.from_study(data.cutoff.isoformat(), study)
    config = small_config(
        "v2",
        encoder_days=7,
        train_windows=2,
        train_stride=2,
        epochs=1,
        batch_size=2,
        patience=1,
        hierarchy_loss_weight=0.0,
        aggregate_columns=(),
        event_pool_columns=(),
        progress="off",
    )

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.raw_mean = torch.nn.Parameter(torch.zeros(()))

        def forward(self, values):
            return torch.nn.functional.softplus(self.raw_mean) * torch.ones(
                (len(values), data.horizon), device=values.device
            )

    def batch(day, rows, device, training):
        stop = data.position(day)
        values = torch.ones(len(rows), device=device)
        if not training:
            return values, None, None
        return (
            values,
            torch.tensor(data.target[rows, stop : stop + data.horizon], device=device),
            torch.ones((len(rows), data.horizon), dtype=torch.bool, device=device),
        )

    backward_calls = {"count": 0}

    def training_loss(model, values, labels, observed, denominator, importance):
        del denominator, importance
        backward_calls["count"] += 1
        loss = (model(values)[observed] - labels[observed]).square().mean()
        if precision == "16-mixed" and backward_calls["count"] == 1:
            loss = loss * 1e35
        return loss

    def validation_values(model, values):
        prediction = model(values)
        return prediction, lambda labels, mask: (prediction[mask] - labels[mask]).square()

    trainer = RollingTrainer(
        data,
        config,
        (),
        np.arange(2),
        TinyModel,
        batch,
        training_loss,
        lambda model, values: model(values),
        validation_values=validation_values,
        training_contract=RollingTrainingContract(
            mode="fixed_validation",
            fixed_validation=holdout,
        ),
        precision=precision,
    )
    profiler = SpyProfiler()
    trainer.performance_profiler = profiler
    records = []
    model, summary = trainer.fit(records.append)
    if study is not None:
        with torch.no_grad():
            expected = study.score(model(torch.ones(2)).double().numpy())
        assert summary["selection_metric"] == "wrmsse"
        assert summary["training_contract"]["study_evaluation"] == study.as_record()
        assert summary["validation"][0]["mean_wrmsse"] == pytest.approx(expected["wrmsse"])
        validation = next(record["validation"] for record in records if "validation" in record)
        assert validation[data.cutoff.isoformat()]["observed_cells"] == 3
    if precision == "16-mixed":
        assert summary["precision"] == "16-mixed"
        assert summary["selected_updates"] == summary["steps_per_epoch"]
        assert backward_calls["count"] > summary["selected_updates"]
        assert records[0]["amp_overflow_retries"] >= 1

    assert profiler.closed == 1
    assert profiler.begun.count("training") == profiler.ended.count("training")
    assert profiler.begun.count("validation") == profiler.ended.count("validation")
    assert profiler.finished == ["training", "validation"]
    assert {name for phase, name in profiler.regions if phase == "training"} >= {
        "batch_assembly_and_transfer",
        "forward_and_loss",
        "backward",
        "gradient_clip",
        "diagnostics",
        "optimizer_step",
    }
    assert {name for phase, name in profiler.regions if phase == "validation"} >= {
        "batch_assembly",
        "device_transfer",
        "forward",
        "negative_log_likelihood",
    }


def test_profiler_cleanup_never_masks_the_training_failure():
    class BrokenProfiler:
        def close(self):
            raise KeyboardInterrupt("cleanup interrupted")

    class Progress:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class Subject:
        performance_profiler = BrokenProfiler()

        def _fit(self, *args):
            raise ValueError("training failed")

    progress = Progress()
    with pytest.raises(ValueError, match="training failed") as raised:
        RollingTrainer.fit(Subject(), lambda record: None, progress=progress)
    assert raised.value.__notes__ == [
        "Torch profiler cleanup also failed: KeyboardInterrupt('cleanup interrupted')"
    ]
    assert progress.closed


@pytest.mark.parametrize("precision", ["32-true", "16-mixed"])
def test_fixed_holdout_selects_once_per_epoch_then_refits_exact_updates(
    revision_data, precision
):
    """Default benchmark: fixed holdout selects; fresh train+valid fit never validates."""
    panel, complete, calendar = revision_data
    cutoff = "2016-03-27"
    stop = complete.position(cutoff)
    holdout = ValidationWindow(
        cutoff,
        complete.target[:, stop : stop + complete.horizon].copy(),
        complete.observed[:, stop : stop + complete.horizon].copy(),
    )
    assert holdout.observed.all()
    masked_target = complete.target.copy()
    masked_observed = complete.observed.copy()
    masked_target[:, stop:] = 0
    masked_observed[:, stop:] = False
    training = dataclasses.replace(
        complete,
        cutoff=dt.date.fromisoformat(cutoff),
        target=masked_target,
        observed=masked_observed,
    )
    config = recipe("v2", epochs=2, patience=10)
    selection_features = RevisionFeatures(panel, training, config, str(calendar))
    selection = RollingTrainer(
        training,
        config,
        M5_LEVELS,
        selection_features.groups,
        selection_features.make_model,
        selection_features.batch,
        selection_features.training_loss,
        selection_features.mean,
        selection_features.optimizer_groups,
        selection_features.begin_phase,
        diagnostics=selection_features.diagnostics,
        validation_values=selection_features.validation_values,
        training_contract=RollingTrainingContract(
            mode="fixed_validation", fixed_validation=holdout
        ),
        precision=precision,
    )
    selection_records: list[dict] = []
    _, selected = selection.fit(selection_records.append)

    assert set(record["phase"] for record in selection_records) == {"selection"}
    assert len(selected["validation"]) == 2
    assert [slot["update"] for slot in selected["validation_slots"]] == [
        record["update"] for record in selected["validation"]
    ]
    assert all(slot["evaluated"] for slot in selected["validation_slots"])
    assert [record["validation_slot"] for record in selection_records
            if "validation_slot" in record] == selected["validation_slots"]
    assert all(
        tuple(record["validation"]) == (cutoff,)
        for record in selection_records
        if "validation" in record
    )
    assert selected["selected_updates"] in {
        selected["steps_per_epoch"],
        2 * selected["steps_per_epoch"],
    }
    assert selected["training_contract"] == {
        "mode": "fixed_validation",
        "validation_origin": cutoff,
        "fixed_update_budget": None,
    }
    assert selected["refit_training_origins"] == ()
    assert selected["refit_series_count"] == 0
    assert set(selected["phase_coverage"]) == {"selection"}

    refit_features = RevisionFeatures(panel, complete, config, str(calendar))
    refit = RollingTrainer(
        complete,
        config,
        M5_LEVELS,
        refit_features.groups,
        refit_features.make_model,
        refit_features.batch,
        refit_features.training_loss,
        refit_features.mean,
        refit_features.optimizer_groups,
        refit_features.begin_phase,
        diagnostics=refit_features.diagnostics,
        validation_values=refit_features.validation_values,
        training_contract=RollingTrainingContract(
            mode="fixed_refit", fixed_update_budget=selected["selected_updates"]
        ),
        precision=precision,
    )
    refit_records: list[dict] = []
    _, final = refit.fit(refit_records.append)

    assert len(refit_records) == selected["selected_updates"]
    assert set(record["phase"] for record in refit_records) == {"refit"}
    assert all("validation" not in record for record in refit_records)
    assert final["validation"] == []
    assert final["selected_updates"] == selected["selected_updates"]
    assert final["refit_optimizer_steps"] == selected["selected_updates"]
    assert final["training_contract"] == {
        "mode": "fixed_refit",
        "validation_origin": None,
        "fixed_update_budget": selected["selected_updates"],
    }
    assert final["selection_training_origins"] == ()
    assert set(final["phase_coverage"]) == {"refit"}
    assert final["budget_policy"] == "sealed_fixed_update_budget"


def test_hmfnet_mixed_precision_completes_actual_updates(revision_data):
    _, trainer = assembled(revision_data, "v3", epochs=1)
    trainer.precision = "16-mixed"
    records = []
    _, summary = trainer.fit(records.append)
    assert summary["precision"] == "16-mixed"
    assert summary["selected_updates"] > 0
    assert records[-1]["optimizer_steps"] >= records[-1]["update"]


@pytest.mark.parametrize("stop_update", [0, 1])
def test_fixed_refit_resumes_when_selected_updates_exceed_epoch_ceiling(
    revision_data, tmp_path, monkeypatch, stop_update
):
    configure_training_runtime("cpu", 1)
    selected_updates = 8

    def make():
        _, trainer = assembled(revision_data, "v2", epochs=1)
        trainer.training_contract = RollingTrainingContract(
            mode="fixed_refit", fixed_update_budget=selected_updates
        )
        return trainer

    control_model, control = make().fit(lambda _row: None)
    assert control["steps_per_epoch"] < selected_updates
    assert control["refit_optimizer_steps"] == selected_updates

    root = tmp_path / "resumed-refit"
    binding = {"training_contract": {"mode": "fixed_refit"}}
    original_save = TrainingRun.save

    def stop(run, state, *, reason):
        original_save(run, state, reason=reason)
        if state["phase"] == "refit" and state["update"] == stop_update:
            raise RuntimeError("stop after refit checkpoint")

    with monkeypatch.context() as patch:
        patch.setattr(TrainingRun, "save", stop)
        with TrainingRun(root, binding, every_updates=1) as run:
            with pytest.raises(RuntimeError, match="stop after refit checkpoint"):
                make().fit(run.log, run=run)
    with TrainingRun(root, binding, resume=root / "training_resume.json") as run:
        resumed_model, resumed = make().fit(run.log, run=run)
    assert resumed["refit_optimizer_steps"] == selected_updates
    equal(control_model.state_dict(), resumed_model.state_dict())


def equal(first, second):
    assert type(first) is type(second)
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0, check_stride=True)
    elif isinstance(first, dict):
        assert {(type(key), key) for key in first} == {(type(key), key) for key in second}
        for key in first:
            equal(first[key], second[key])
    elif isinstance(first, list | tuple):
        assert len(first) == len(second)
        for a, b in zip(first, second, strict=True):
            equal(a, b)
    else:
        assert first == second


def numerical(record):
    """Keep all numerical/semantic evidence, remove only explicitly nondeterministic timings/RSS."""
    if isinstance(record, dict):
        return {
            key: numerical(value)
            for key, value in record.items()
            if key != "timing" and "seconds" not in key and key != "peak_rss_bytes"
        }
    if isinstance(record, list):
        return [numerical(value) for value in record]
    return record


@pytest.mark.parametrize(
    "first,second",
    [
        (True, 1),
        (1, 1.0),
        ({0: 1}, {False: 1}),
        ([1], (1,)),
        (torch.tensor([1.0]), torch.tensor([1.0], dtype=torch.float64)),
        (torch.ones(2, 3), torch.ones(3, 2).T),
        (torch.tensor([1e-8]), torch.tensor([0.0])),
    ],
)
def test_equal_state_types(first, second):
    """Strict state comparison is stronger than Python equality or torch.equal."""
    with pytest.raises(AssertionError):
        equal(first, second)


@pytest.mark.parametrize("weighting", ["revenue", "uniform"])
def test_torch_hierarchy_metric(revision_data, weighting):
    _, data, _ = revision_data
    stop = data.position("2016-03-27")
    metric = HierarchyMetric.build(
        data.metadata, data.target[:, :stop], data.price[:, :stop], M5_LEVELS, weighting=weighting
    )
    target = data.target[:, stop : stop + 28].astype(np.float64)
    prediction = data.target[:, stop - 28 : stop].astype(np.float64)
    cached = TorchHierarchyMetric(metric, "cpu")
    actual = cached.score(torch.tensor(prediction), torch.tensor(target))
    expected = metric.score(prediction, target)
    key = "wrmsse" if weighting == "revenue" else "uniform_hierarchy_rmsse"
    assert actual[key] == pytest.approx(expected[key], rel=1e-12, abs=1e-12)
    for level in expected["per_level"]:
        assert actual["per_level"][level] == pytest.approx(
            expected["per_level"][level], rel=1e-12, abs=1e-12
        )
    assert actual["n_series_scored"] == expected["n_series_scored"]
    assert cached.score(torch.tensor(target), torch.tensor(target))[key] == 0
    for bad in (
        torch.tensor(target[:-1]),
        torch.full_like(torch.tensor(target), float("nan")),
        -torch.ones_like(torch.tensor(target)),
        torch.empty(len(target), 0),
    ):
        with pytest.raises(ValueError, match="grids"):
            cached.score(bad, torch.tensor(target))
    with pytest.raises(ValueError, match="overflow"):
        cached.score(torch.full_like(torch.tensor(target), 1e308), torch.tensor(target))
    with pytest.raises(ValueError, match="surrogate"):
        TorchHierarchyMetric(dataclasses.replace(metric, training_surrogate=True), "cpu")
    with pytest.raises(ValueError, match="structure"):
        TorchHierarchyMetric(dataclasses.replace(metric, names=()), "cpu")
    with pytest.raises(ValueError, match="indices/scales"):
        TorchHierarchyMetric(
            dataclasses.replace(metric, scales=tuple(np.zeros_like(v) for v in metric.scales)),
            "cpu",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware not available")
def test_torch_hierarchy_metric_cuda(revision_data):
    _, data, _ = revision_data
    stop = data.position("2016-03-27")
    metric = HierarchyMetric.build(
        data.metadata, data.target[:, :stop], data.price[:, :stop], M5_LEVELS
    )
    target = torch.tensor(data.target[:, stop : stop + 28], device="cuda")
    predicted = torch.tensor(data.target[:, stop - 28 : stop], device="cuda")
    cached = TorchHierarchyMetric(metric, "cuda")
    assert cached.score(predicted, target)["wrmsse"] == pytest.approx(
        metric.score(predicted.cpu().numpy(), target.cpu().numpy())["wrmsse"], rel=1e-12
    )
    with pytest.raises(ValueError, match="device"):
        cached.score(predicted.cpu(), target)


@pytest.mark.parametrize("variant", ["v2", "v3"])
def test_revision_validation(revision_data, variant, monkeypatch):
    configure_training_runtime("cpu", 1)
    features, trainer = assembled(revision_data, variant)
    origin = "2016-03-27"
    origins = rolling_origins(trainer.data, "2016-02-28", trainer.config)
    features.begin_phase(origins, "selection")
    model = features.make_model()
    model.train()
    # A mixed module-mode flag must be restored too.
    list(model.modules())[-1].eval()
    flags = [module.training for module in model.modules()]
    stop = trainer.data.position(origin)
    metric = HierarchyMetric.build(
        trainer.data.metadata,
        trainer.data.target[:, :stop],
        trainer.data.price[:, :stop],
        M5_LEVELS,
    )
    cached = TorchHierarchyMetric(metric, "cpu")
    rng = np.random.default_rng(5)
    before = capture_training_rng(rng)
    feature_before = copy.deepcopy(features.feature_training_state())
    calls = []
    hook = model.register_forward_hook(lambda *args: calls.append(1))
    native_cpu = torch.Tensor.cpu

    def no_grid_cpu(value, *args, **kwargs):
        assert value.ndim < 2, "Torch validation exported a full tensor grid to CPU"
        return native_cpu(value, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", no_grid_cpu)
        actual = trainer.validate(model, origin, cached)
    hook.remove()
    assert len(calls) == len(trainer.chunks())
    equal(before, capture_training_rng(rng))
    equal(feature_before, features.feature_training_state())
    assert [module.training for module in model.modules()] == flags
    trainer.config = dataclasses.replace(trainer.config, validation_backend="numpy")
    reference = trainer.validate(model, origin, metric)
    assert actual["wrmsse"] == pytest.approx(reference["wrmsse"], rel=1e-12)
    assert actual["validation_nll"] == reference["validation_nll"] > 0
    assert actual["timing"]["numpy_metric_seconds"] is None
    assert reference["timing"]["numpy_metric_seconds"] >= 0
    assert actual["timing"]["total_seconds"] >= actual["timing"]["forward_seconds"] > 0
    parts = (
        "feature_seconds",
        "forward_seconds",
        "nll_seconds",
        "transfer_seconds",
        "metric_seconds",
        "prediction_assembly_seconds",
        "other_seconds",
    )
    for score in (actual, reference):
        assert sum(score["timing"][name] for name in parts) == pytest.approx(
            score["timing"]["total_seconds"], abs=1e-9
        )
    for backend in ("torch", "numpy"):
        trainer.config = dataclasses.replace(trainer.config, validation_backend=backend)
        rmse = trainer.validate(model, origin, None)
        target = trainer.data.target[:, stop : stop + 28]
        mask = trainer.data.observed[:, stop : stop + 28]
        assert rmse["rmse"] == pytest.approx(
            float(np.sqrt(np.mean((trainer.predict(model, origin)[mask] - target[mask]) ** 2))),
            rel=1e-12,
        )
        assert (rmse["timing"]["numpy_metric_seconds"] is not None) == (backend == "numpy")
    # Independent masked NLL calculation over the same grid; no auxiliary expert/mean term.
    model.eval()
    sums = []
    with torch.no_grad():
        for rows in trainer.chunks():
            inputs = features.batch(origin, rows, "cpu", False)[0]
            _, nll = features.validation_values(model, inputs)
            sums.append(
                nll(
                    torch.tensor(trainer.data.target[rows, stop : stop + 28]),
                    torch.tensor(trainer.data.observed[rows, stop : stop + 28]),
                )
                .double()
                .sum()
            )
    assert actual["validation_nll"] == pytest.approx(
        float(torch.stack(sums).sum()) / int(trainer.data.observed[:, stop : stop + 28].sum()),
        rel=1e-12,
    )


@pytest.mark.parametrize("variant", ["v2", "v3"])
def test_validation_cleanup_timing(revision_data, variant, monkeypatch):
    """The synchronous pause includes delayed context exit, after restoring module modes."""
    configure_training_runtime("cpu", 1)
    features, trainer = assembled(revision_data, variant)
    features.begin_phase(rolling_origins(trainer.data, "2016-02-28", trainer.config), "selection")
    model = features.make_model()
    model.train()
    list(model.modules())[-1].eval()
    flags = [module.training for module in model.modules()]
    original_context, original_clock = trainer.prediction_context, time.perf_counter
    clock_offset, exited, checked_restoration = 0.0, False, []

    @contextmanager
    def delayed_context(model, origin):
        nonlocal clock_offset, exited
        with original_context(model, origin) if original_context else nullcontext() as timing:
            yield timing
        clock_offset += 1.0  # Deterministic structural delay, no slow test sleep.
        exited = True

    def clock():
        if exited:
            assert [module.training for module in model.modules()] == flags
            checked_restoration.append(True)
        return original_clock() + clock_offset

    trainer.prediction_context = delayed_context
    monkeypatch.setattr(rolling_training.time, "perf_counter", clock)
    result = trainer.validate(model, "2016-03-27", None)
    assert exited and checked_restoration
    assert result["timing"]["total_seconds"] >= 1.0
    assert result["timing"]["other_seconds"] >= 1.0


def test_revision_validation_schedule(revision_data):
    _, trainer = assembled(revision_data, "v2", epochs=3, validation_every_epochs=2)
    assert [u for u in range(1, 7) if trainer.validation_due(u, 6, 2)] == [4, 6]
    trainer.config = dataclasses.replace(trainer.config, epochs=None, evaluate_every=3)
    assert [u for u in range(1, 8) if trainer.validation_due(u, 7, 2)] == [3, 6, 7]
    trainer.config = dataclasses.replace(trainer.config, epochs=2, validation_every_epochs=1)
    assert [u for u in range(1, 5) if trainer.validation_due(u, 4, 2)] == [2, 4]
    for changes in (
        {"validation_every_epochs": 0},
        {"validation_backend": "async"},
        {"checkpoint_every_updates": True},
        {"checkpoint_every_seconds": 0},
        {"checkpoint_keep_last": 1},
        {"progress": "bad"},
    ):
        with pytest.raises(ValueError):
            dataclasses.replace(trainer.config, **changes)


def _coverage_delayed_fixed_trainer(
    *, two_steps_per_epoch: bool = False, validation_every_epochs: int = 1,
) -> RollingTrainer:
    start = dt.date(2024, 1, 1)
    target = np.ones((2, 30), dtype=np.float32)
    data = rolling_training.PanelArrays(
        ("a", "b"), start, start + dt.timedelta(days=29), 2,
        target, np.ones_like(target, dtype=bool),
        np.empty((2, 0), dtype=np.float64), {"series": ("a", "b")},
    )
    config = small_config(
        "v2", encoder_days=7, train_windows=2, train_stride=2,
        epochs=3, batch_size=1 if two_steps_per_epoch else 2, patience=10,
        validation_every_epochs=validation_every_epochs, hierarchy_loss_weight=0.0,
        aggregate_columns=(), event_pool_columns=(), progress="off",
    )
    origins = rolling_origins(data, data.cutoff.isoformat(), config)
    assert len(origins) == 2
    observed = data.observed.copy()
    for series, origin in enumerate(origins):
        stop = data.position(origin)
        observed[1 - series, stop:stop + data.horizon] = False
    data = dataclasses.replace(data, observed=observed)
    holdout = ValidationWindow(
        data.cutoff.isoformat(), np.ones((2, 2), np.float32),
        np.ones((2, 2), bool),
    )

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mean = torch.nn.Parameter(torch.ones(()))

        def forward(self, values):
            return self.mean * torch.ones((len(values), data.horizon))

    def batch(day, rows, device, training):
        values = torch.ones(len(rows), device=device)
        if not training:
            return values, None, None
        stop = data.position(day)
        return (
            values,
            torch.tensor(data.target[rows, stop:stop + data.horizon], device=device),
            torch.tensor(data.observed[rows, stop:stop + data.horizon], device=device),
        )

    def training_loss(model, values, labels, mask, denominator, importance):
        del denominator, importance
        return (model(values)[mask] - labels[mask]).square().mean()

    def validation_values(model, values):
        prediction = model(values)
        return prediction, lambda labels, mask: (prediction[mask] - labels[mask]).square()

    trainer = RollingTrainer(
        data, config, (),
        np.arange(2, dtype=int) if two_steps_per_epoch else np.zeros(2, dtype=int),
        TinyModel, batch,
        training_loss, lambda model, values: model(values),
        validation_values=validation_values,
        training_contract=RollingTrainingContract(
            mode="fixed_validation", fixed_validation=holdout,
        ),
    )
    return trainer


def test_rolling_amp_replays_silent_skipped_step_with_explicit_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_scaler = torch.amp.GradScaler

    class SkipOnce:
        instances: list["SkipOnce"] = []

        def __init__(self, _device: str) -> None:
            self.delegate = original_scaler("cpu")
            self.skipped = False
            self.manual_backoffs = 0
            type(self).instances.append(self)

        def scale(self, loss):
            return self.delegate.scale(loss)

        def unscale_(self, optimizer):
            return self.delegate.unscale_(optimizer)

        def get_scale(self):
            return self.delegate.get_scale()

        def step(self, optimizer):
            if not self.skipped:
                self.skipped = True
                return None
            return self.delegate.step(optimizer)

        def update(self, new_scale=None):
            if new_scale is not None:
                self.manual_backoffs += 1
            return self.delegate.update(new_scale=new_scale)

        def state_dict(self):
            return self.delegate.state_dict()

        def load_state_dict(self, state):
            return self.delegate.load_state_dict(state)

    monkeypatch.setattr(torch.amp, "GradScaler", SkipOnce)
    trainer = _coverage_delayed_fixed_trainer()
    trainer.precision = "16-mixed"
    rows: list[dict] = []
    _, summary = trainer.fit(rows.append)
    scaler = SkipOnce.instances[0]
    assert scaler.skipped and scaler.manual_backoffs == 1
    assert summary["selected_updates"] == len(rows)
    assert rows[0]["amp_overflow_retries"] == 1
    assert [row["optimizer_steps"] for row in rows] == list(range(1, len(rows) + 1))


def test_fixed_holdout_records_coverage_skipped_validation_epoch() -> None:
    trainer = _coverage_delayed_fixed_trainer()
    records: list[dict] = []
    _, summary = trainer.fit(records.append)
    assert [slot["evaluated"] for slot in summary["validation_slots"]] == [
        False, True, True
    ]
    assert [row["update"] for row in summary["validation"]] == [2, 3]
    assert [row["validation_slot"] for row in records if "validation_slot" in row] == (
        summary["validation_slots"]
    )


@pytest.mark.parametrize("layout", ("shared_factor", "singleton_serial", "singleton_packed"))
def test_fixed_refit_rejects_infeasible_selected_budget_before_optimizer_step(
    layout: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _coverage_delayed_fixed_trainer(
        two_steps_per_epoch=layout == "singleton_serial"
    )
    if layout == "singleton_packed":
        trainer.groups = np.arange(2, dtype=int)
    trainer.training_contract = RollingTrainingContract(
        mode="fixed_refit", fixed_update_budget=1
    )

    def forbidden_step(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("infeasible refit reached an optimizer step")

    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden_step)
    with pytest.raises(ValueError, match="non-retryable.*cannot cover every trainable series"):
        trainer.fit(lambda _row: None)


def test_refit_sampler_preflight_skips_hierarchy_only_update_only_when_enabled() -> None:
    masks = {"2024-01-01": np.ones((1, 2), dtype=bool)}
    blocks = [np.array([0])]
    controls = {
        "batch_size": 1, "seed": 0, "budget": 1, "epochs": None,
        "hierarchy_every": 1,
    }
    assert rolling_training._refit_sampler_can_cover(  # noqa: SLF001
        masks, blocks, **controls, hierarchy_only_enabled=False
    )
    assert not rolling_training._refit_sampler_can_cover(  # noqa: SLF001
        masks, blocks, **controls, hierarchy_only_enabled=True
    )


def test_fixed_holdout_resume_rejects_forged_coverage_and_validation(
    tmp_path, monkeypatch,
) -> None:
    configure_training_runtime("cpu", 1)
    root = tmp_path / "coverage-forgery"
    original_save = TrainingRun.save

    def interrupt(run, state, *, reason):
        original_save(run, state, reason=reason)
        if state["phase"] == "selection" and state["update"] == 1:
            raise RuntimeError("interrupt after uncovered slot")

    with monkeypatch.context() as patch:
        patch.setattr(TrainingRun, "save", interrupt)
        with (
            TrainingRun(root, {}, every_updates=1) as run,
            pytest.raises(RuntimeError, match="uncovered slot"),
        ):
            _coverage_delayed_fixed_trainer().fit(run.log, run=run)
    pointer_path = root / "training_resume.json"
    pointer = json.loads(pointer_path.read_text())
    manifest_path = root / "checkpoints" / pointer["manifest"]
    manifest = json.loads(manifest_path.read_text())
    payload_path = root / "checkpoints" / manifest["payload"]
    payload = torch.load(payload_path, weights_only=True)
    state = payload["state"]
    assert state["validation_slots"] == [
        {"update": 1, "covered_series": 1, "trainable_series": 2,
         "evaluated": False}
    ]
    forged_slot = {"update": 1, "covered_series": 2,
                   "trainable_series": 2, "evaluated": True}
    state["validation_slots"] = [forged_slot]
    state["evaluations"] = [{"update": 1, "mean_rmse": 0.0}]
    state["selected_updates"] = 1
    state["best_score"] = 0.0
    state["best_state"] = copy.deepcopy(state["model"])
    state["latest_validation"] = {
        "update": 1, "epoch": 1, "metric": 0.0, "metric_name": "rmse", "nll": 0.0,
    }
    log_path = root / "training.jsonl"
    row = json.loads(log_path.read_text().splitlines()[0])
    row["validation_slot"] = forged_slot
    row["validation_summary"] = {
        **state["latest_validation"], "best_score": 0.0, "stale": 0,
        "evaluations": 1,
    }
    row["validation"] = {"2024-01-30": {"rmse": 0.0, "validation_nll": 0.0}}
    log_path.write_text(json.dumps(row) + "\n")
    manifest["log_bytes"] = len(log_path.read_bytes())
    manifest["log_sha256"] = hashlib.sha256(log_path.read_bytes()).hexdigest()
    torch.save(payload, payload_path)
    manifest["payload_sha256"] = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    pointer["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    pointer_path.write_text(json.dumps(pointer))
    with (
        TrainingRun(root, {}, resume=pointer_path) as run,
        pytest.raises(ValueError, match="coverage differs from objective masks"),
    ):
        _coverage_delayed_fixed_trainer().fit(run.log, run=run)
    assert not (root / "recovery").exists()


def test_fixed_holdout_resume_rejects_forged_sampler_rng(
    tmp_path, monkeypatch,
) -> None:
    configure_training_runtime("cpu", 1)
    root = tmp_path / "sampler-rng-forgery"
    original_save = TrainingRun.save

    def interrupt(run, state, *, reason):
        original_save(run, state, reason=reason)
        if state["phase"] == "selection" and state["update"] == 1:
            raise RuntimeError("interrupt after sampled update")

    with monkeypatch.context() as patch:
        patch.setattr(TrainingRun, "save", interrupt)
        with (
            TrainingRun(root, {}, every_updates=1) as run,
            pytest.raises(RuntimeError, match="sampled update"),
        ):
            _coverage_delayed_fixed_trainer().fit(run.log, run=run)
    pointer_path = root / "training_resume.json"
    pointer = json.loads(pointer_path.read_text())
    manifest_path = root / "checkpoints" / pointer["manifest"]
    manifest = json.loads(manifest_path.read_text())
    payload_path = root / "checkpoints" / manifest["payload"]
    payload = torch.load(payload_path, weights_only=True)
    payload["state"]["rng"]["generator"] = np.random.default_rng(987654321).bit_generator.state
    torch.save(payload, payload_path)
    manifest["payload_sha256"] = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    pointer["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    pointer_path.write_text(json.dumps(pointer))
    with (
        TrainingRun(root, {}, resume=pointer_path, every_updates=1) as run,
        pytest.raises(ValueError, match="sampler state differs"),
    ):
        _coverage_delayed_fixed_trainer().fit(run.log, run=run)
    assert not (root / "recovery").exists()


@pytest.mark.parametrize("study_metric", [None, "mae", "wape"])
def test_fixed_holdout_resume_uses_nondefault_validation_cadence(
    tmp_path, monkeypatch, study_metric,
) -> None:
    configure_training_runtime("cpu", 1)
    def make():
        trainer = _coverage_delayed_fixed_trainer(
            two_steps_per_epoch=True, validation_every_epochs=2,
        )
        if study_metric is not None:
            from autoforecast.study_validation import StudyValidation

            origin = trainer.data.cutoff.isoformat()
            study = StudyValidation(
                json.dumps({"study_id": "b" * 64, "boundary": f"validation/{origin}/2/D",
                            "metric_policy": {"metric": study_metric, "actual_missing": "mask"}}),
                "a" * 64, trainer.data.series_ids,
                np.array([[1.0, 0.0], [2.0, 4.0]]),
                np.array([[True, False], [True, True]]),
            )
            trainer.training_contract = RollingTrainingContract(
                mode="fixed_validation", fixed_validation=ValidationWindow.from_study(origin, study),
            )
        return trainer
    control_model, control = make().fit(lambda _: None)
    root = tmp_path / "nondefault-cadence"
    original_save = TrainingRun.save

    def interrupt(run, state, *, reason):
        original_save(run, state, reason=reason)
        if state["phase"] == "selection" and state["update"] == 5:
            assert [slot["update"] for slot in state["validation_slots"]] == [4]
            raise RuntimeError("interrupt after update five")

    with monkeypatch.context() as patch:
        patch.setattr(TrainingRun, "save", interrupt)
        with (
            TrainingRun(root, {}, every_updates=1) as run,
            pytest.raises(RuntimeError, match="update five"),
        ):
            make().fit(run.log, run=run)
    with TrainingRun(root, {}, resume=root / "training_resume.json") as run:
        resumed_model, resumed = make().fit(run.log, run=run)
    assert resumed["validation"] == control["validation"]
    assert resumed["validation_slots"] == control["validation_slots"]
    assert resumed["selection_metric"] == (study_metric or "rmse")
    equal(control_model.state_dict(), resumed_model.state_dict())


def test_update_budget_resume_before_first_validation_uses_evaluate_every(
    revision_data, tmp_path, monkeypatch,
) -> None:
    configure_training_runtime("cpu", 1)
    _, control_trainer = assembled(
        revision_data, "v2", epochs=None, evaluate_every=3, max_updates=6,
    )
    control_model, control = control_trainer.fit(lambda _: None)
    root = tmp_path / "update-cadence"
    original_save = TrainingRun.save

    def interrupt(run, state, *, reason):
        original_save(run, state, reason=reason)
        if state["phase"] == "selection" and state["update"] == 2:
            assert state["validation_slots"] == []
            raise RuntimeError("interrupt before validation")

    with monkeypatch.context() as patch:
        patch.setattr(TrainingRun, "save", interrupt)
        with (
            TrainingRun(root, {}, every_updates=1) as run,
            pytest.raises(RuntimeError, match="before validation"),
        ):
            assembled(revision_data, "v2", epochs=None, evaluate_every=3,
                      max_updates=6)[1].fit(run.log, run=run)
    with TrainingRun(root, {}, resume=root / "training_resume.json") as run:
        resumed_model, resumed = assembled(
            revision_data, "v2", epochs=None, evaluate_every=3, max_updates=6,
        )[1].fit(run.log, run=run)
    assert resumed["validation"] == control["validation"]
    assert resumed["validation_slots"] == control["validation_slots"]
    equal(control_model.state_dict(), resumed_model.state_dict())


def test_fixed_validation_update_only_resume_accepts_hierarchy_only_row(
    revision_data, tmp_path, monkeypatch,
) -> None:
    configure_training_runtime("cpu", 1)

    def make():
        _, trainer = assembled(
            revision_data, "v2", epochs=None, max_updates=3,
            evaluate_every=3, hierarchy_every=2, hierarchy_loss_weight=0.1,
        )
        count = len(trainer.data.series_ids)
        horizon = trainer.data.horizon
        trainer.training_contract = RollingTrainingContract(
            mode="fixed_validation",
            fixed_validation=ValidationWindow(
                trainer.data.cutoff.isoformat(),
                np.ones((count, horizon), dtype=np.float32),
                np.ones((count, horizon), dtype=bool),
            ),
        )
        return trainer

    control_model, control = make().fit(lambda _: None)
    root = tmp_path / "fixed-hierarchy-resume"
    original_save = TrainingRun.save

    def interrupt(run, state, *, reason):
        original_save(run, state, reason=reason)
        if state["phase"] == "selection" and state["update"] == 2:
            raise RuntimeError("interrupt after hierarchy-only update")

    with monkeypatch.context() as patch:
        patch.setattr(TrainingRun, "save", interrupt)
        with (
            TrainingRun(root, {}, every_updates=1) as run,
            pytest.raises(RuntimeError, match="hierarchy-only update"),
        ):
            make().fit(run.log, run=run)
    rows = [json.loads(line) for line in (root / "training.jsonl").read_text().splitlines()]
    assert rows[1]["origin_sampling"] == "hierarchy_only"
    assert rows[1]["sampled_series_indices"] == []
    with TrainingRun(root, {}, resume=root / "training_resume.json") as run:
        resumed_model, resumed = make().fit(run.log, run=run)
    assert resumed["validation"] == control["validation"]
    equal(control_model.state_dict(), resumed_model.state_dict())


@pytest.mark.parametrize("variant", ["v2", "v3"])
def test_revision_feature_training_state(revision_data, variant):
    features, trainer = assembled(revision_data, variant)
    origins = rolling_origins(trainer.data, "2016-02-28", trainer.config)
    features.begin_phase(origins, "selection")
    saved = features.feature_training_state()
    first = features.batch(origins[0], np.arange(6), "cpu", True)
    features.restore_feature_training_state(saved)
    second = features.batch(origins[0], np.arange(6), "cpu", True)
    equal(first, second)
    with pytest.raises(ValueError, match="feature training state"):
        features.restore_feature_training_state({"schema": "bad"})


@pytest.mark.parametrize(
    "variant,boundary,precision",
    [
        (variant, boundary, "32-true")
        for variant in ("v2", "v3")
        for boundary in (
            "ordinary", "hierarchy", "validation", "validation_checkpoint",
            "selection_complete", "refit", "refit_complete", "export",
        )
    ] + [
        ("v2", "ordinary", "16-mixed"),
        ("v2", "ordinary_after_hierarchy", "16-mixed"),
        ("v2", "after_hierarchy_checkpoint", "16-mixed"),
        ("v2", "refit_after_hierarchy_checkpoint", "16-mixed"),
    ],
)
def test_revision_training_resume(
    revision_data, tmp_path, monkeypatch, variant, boundary, precision
):
    panel, _, calendar = revision_data
    c = recipe(variant)
    control, interrupted = tmp_path / "control", tmp_path / "interrupted"
    kwargs = dict(calendar_path=str(calendar), config=c, levels=M5_LEVELS,
                  precision=precision)
    fit_predict_revision(panel, "2016-04-24", 28, out_dir=str(control), **kwargs)
    original_log, original_save = TrainingRun.log, TrainingRun.save
    original_hierarchy = rolling_training.staged_hierarchy_backward
    original_validate = RollingTrainer.validate
    original_write = pq.write_table
    fired = []

    def fail_once():
        if not fired:
            fired.append(True)
            raise RuntimeError("injected interruption")

    def log(run, record):
        original_log(run, record)
        if (
                boundary in {"ordinary", "ordinary_after_hierarchy"}
                and record["phase"] == "selection"
                and record["update"] == (2 if boundary == "ordinary_after_hierarchy" else 1)
            or boundary == "refit"
            and record["phase"] == "refit"
            and record["update"] == 1
        ):
            fail_once()

    def save(run, state, *, reason):
        original_save(run, state, reason=reason)
        if (
            boundary == "after_hierarchy_checkpoint"
            and state["phase"] == "selection"
            and state["update"] == 2
            or boundary == "refit_after_hierarchy_checkpoint"
            and state["phase"] == "refit"
            and state["update"] == 2
            or boundary == "validation_checkpoint"
            and reason == "validation"
            or boundary == "selection_complete"
            and state["phase"] == "selection"
            and reason == "phase_complete"
            or boundary == "refit_complete"
            and state["phase"] == "refit"
            and reason == "phase_complete"
        ):
            fail_once()

    def hierarchy(*args, **kwargs):
        result = original_hierarchy(*args, **kwargs)
        if boundary == "hierarchy":
            fail_once()
        return result

    def validate(*args, **kwargs):
        result = original_validate(*args, **kwargs)
        if boundary == "validation":
            fail_once()
        return result

    def write(table, path, *args, **kwargs):
        result = original_write(table, path, *args, **kwargs)
        if boundary == "export" and str(path).endswith("forecast.parquet"):
            fail_once()
        return result

    with monkeypatch.context() as patch:
        patch.setattr(TrainingRun, "log", log)
        patch.setattr(TrainingRun, "save", save)
        patch.setattr(rolling_training, "staged_hierarchy_backward", hierarchy)
        patch.setattr(RollingTrainer, "validate", validate)
        patch.setattr(pq, "write_table", write)
        with pytest.raises(RuntimeError, match="injected interruption"):
            fit_predict_revision(panel, "2016-04-24", 28, out_dir=str(interrupted), **kwargs)
    assert fired and not (interrupted / "COMPLETE.json").exists()
    fit_predict_revision(
        panel,
        "2016-04-24",
        28,
        out_dir=str(interrupted),
        resume_training_checkpoint=str(interrupted / "training_resume.json"),
        **kwargs,
    )
    a = torch.load(control / "model.pt", weights_only=True)
    b = torch.load(interrupted / "model.pt", weights_only=True)
    equal(a, b)
    assert (control / "forecast.parquet").read_bytes() == (
        interrupted / "forecast.parquet"
    ).read_bytes()
    logs = [
        [numerical(json.loads(line)) for line in (root / "training.jsonl").read_text().splitlines()]
        for root in (control, interrupted)
    ]
    equal(*logs)
    refit_updates = a["training"]["selected_updates"]
    assert [(r["phase"], r["update"]) for r in logs[1]] == [
        *(("selection", update) for update in range(1, 5)),
        *(("refit", update) for update in range(1, refit_updates + 1)),
    ]
    training_states = []
    for root in (control, interrupted):
        pointer = json.loads((root / "training_resume.json").read_text())
        checkpoint_path = root / "checkpoints" / pointer["manifest"].replace(".json", ".pt")
        saved = torch.load(checkpoint_path, weights_only=True)["state"]
        training_states.append(saved)
        assert (
            saved["phase_done"]
            and saved["phase"] == "refit"
            and saved["optimizer_steps"] == refit_updates
            and saved["optimizer_step_contract"] == "one_step_per_update_v1"
        )
        assert saved["best_state"] is not None and saved["optimizer"]["state"]
    equal(*training_states)  # Includes every optimizer moment, sampler/RNG and coverage tensor.
    assert all(r["sampled_series_indices"] for r in logs[1])
    if boundary in {"ordinary", "refit", "export"}:
        assert list((interrupted / "recovery").rglob("*"))


def test_rolling_resume_state(revision_data, tmp_path):
    features, trainer = assembled(revision_data, "v3")
    root = tmp_path / "run"
    configure_training_runtime("cpu", 1)
    with TrainingRun(root, {}) as run:
        model, _ = trainer.fit(run.log, run=run)
    with TrainingRun(root, {}, resume=root / "training_resume.json") as run:
        saved = run.loaded
    blocks = [np.flatnonzero(trainer.groups == group) for group in np.unique(trainer.groups)]
    optimizer = torch.optim.AdamW(features.optimizer_groups(model), lr=trainer.config.learning_rate)
    trainer._check_resume_state(saved, saved["budget"], blocks, model, optimizer, "wrmsse")
    legacy = copy.deepcopy(saved)
    legacy.pop("optimizer_step_contract")
    with pytest.raises(ValueError, match="legacy FP32 hierarchy checkpoint"):
        trainer._check_resume_state(
            legacy, saved["budget"], blocks, model, optimizer, "wrmsse"
        )
    best = {key: value.clone() for key, value in saved["best_state"].items()}
    with torch.no_grad():
        next(model.parameters()).add_(1)
    equal(best, saved["best_state"])
    mutations = [
        lambda s: s.update(update=-1),
        lambda s: s.update(optimizer_steps=4),
        lambda s: s.update(block_position=99),
        lambda s: s.update(stale=-1),
        lambda s: s.update(selected_updates=99),
        lambda s: s.update(best_score=float("nan")),
        lambda s: s["block_order"].zero_(),
        lambda s: s["likelihood_visits"].fill_(-1),
        lambda s: s["seen_series"].fill_(False),
        lambda s: s["scheduler"].update(completed_updates=99),
        lambda s: s["optimizer"]["param_groups"][0].update(lr=99),
        lambda s: s["optimizer"].update(state={}),
        lambda s: s.update(epoch_visits=s["epoch_visits"].float()),
        lambda s: next(iter(s["optimizer"]["state"].values()))["step"].fill_(1000),
    ]
    for mutate in mutations:
        altered = copy.deepcopy(saved)
        mutate(altered)
        with pytest.raises(ValueError):
            trainer._check_resume_state(
                altered, saved["budget"], blocks, model, optimizer, "wrmsse"
            )
    for mutate, message in [
        (lambda s: s["evaluations"][0].update(update=-1), "selection evaluation history"),
        (lambda s: s.update(best_score=s["best_score"] + 1), "best/selected/stale"),
        (lambda s: s.update(evaluations=[]), "evaluation|completed-phase coverage"),
        (lambda s: s.update(loss_sum=float("nan")), "coverage/loss accumulation"),
        (lambda s: s.update(epoch_loss_count=1), "coverage/loss accumulation"),
    ]:
        altered = copy.deepcopy(saved)
        mutate(altered)
        with pytest.raises(ValueError, match=message):
            trainer._check_resume_state(
                altered, saved["budget"], blocks, model, optimizer, "wrmsse"
            )


@pytest.mark.parametrize("cross_phase", [False, True])
def test_resume_accepts_a_completed_phase_with_excluded_series(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cross_phase: bool
) -> None:
    start = dt.date(2024, 1, 1)
    target = np.tile(np.arange(30, dtype=np.float32) % 5, (2, 1))
    observed = np.ones_like(target, dtype=bool)
    observed[0] = False
    if cross_phase:
        observed[0, -2:] = True
    data = rolling_training.PanelArrays(
        ("excluded", "trainable"), start, start + dt.timedelta(days=29), 2,
        target, observed, np.empty((2, 0), dtype=np.float64),
        {"series": ("excluded", "trainable")},
    )
    holdout = ValidationWindow(
        data.cutoff.isoformat(), np.ones((2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=bool),
    )
    config = small_config(
        "v2", encoder_days=7, train_windows=2, train_stride=2,
        epochs=2, batch_size=1 if cross_phase else 2, patience=10,
        hierarchy_loss_weight=0.0,
        aggregate_columns=(), event_pool_columns=(), progress="off",
    )

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.raw_mean = torch.nn.Parameter(torch.zeros(()))

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.softplus(self.raw_mean) * torch.ones(
                (len(values), data.horizon), device=values.device
            )

    def batch(day, rows, device, training):
        stop = data.position(day)
        values = torch.ones(len(rows), device=device)
        if not training:
            return values, None, None
        return (
            values,
            torch.tensor(data.target[rows, stop : stop + data.horizon], device=device),
            torch.ones((len(rows), data.horizon), dtype=torch.bool, device=device),
        )

    def training_loss(model, values, labels, mask, denominator, importance):
        del denominator, importance
        return (model(values)[mask] - labels[mask]).square().mean()

    def validation_values(model, values):
        prediction = model(values)
        return prediction, lambda labels, mask: (prediction[mask] - labels[mask]).square()

    trainer = RollingTrainer(
        data, config, (), np.arange(2), TinyModel, batch, training_loss,
        lambda model, values: model(values), validation_values=validation_values,
        training_contract=(
            RollingTrainingContract()
            if cross_phase else RollingTrainingContract(
                mode="fixed_validation", fixed_validation=holdout,
            )
        ),
    )
    configure_training_runtime("cpu", config.torch_threads)
    root = tmp_path / "masked-resume"
    original_save = TrainingRun.save
    stop_phase = "refit" if cross_phase else "selection"

    def stop_after_selection(run, state, *, reason):
        original_save(run, state, reason=reason)
        if state["phase"] == stop_phase and reason == "phase_complete":
            raise RuntimeError("injected interruption")

    with monkeypatch.context() as patch:
        patch.setattr(TrainingRun, "save", stop_after_selection)
        with pytest.raises(RuntimeError, match="injected interruption"):
            with TrainingRun(root, {}) as run:
                trainer.fit(run.log, run=run)
    with TrainingRun(root, {}, resume=root / "training_resume.json") as run:
        _, summary = trainer.fit(run.log, run=run)
    assert summary["phase_coverage"]["selection"]["series_seen"] == 1
    if cross_phase:
        assert summary["phase_coverage"]["refit"]["series_seen"] == 2
    assert summary["selected_updates"] > 0


def test_fp32_fixed_refit_raw_optimizer_calls_equal_selected_updates_with_hierarchy(
    revision_data, tmp_path
):
    panel, _, calendar = revision_data
    root = tmp_path / "fit"
    fit_predict_revision(
        panel, "2016-04-24", 28,
        out_dir=str(root), calendar_path=str(calendar),
        config=recipe("v2"), levels=M5_LEVELS, precision="32-true",
    )
    records = [json.loads(line) for line in (root / "training.jsonl").read_text().splitlines()]
    selection = [row for row in records if row["phase"] == "selection"]
    refit = [row for row in records if row["phase"] == "refit"]
    summary = torch.load(root / "model.pt", weights_only=True)["training"]
    selected_updates = summary["selected_updates"]
    assert any("hierarchy_loss" in row for row in refit)
    assert len(refit) == selected_updates
    assert selection[-1]["optimizer_steps"] == len(selection)
    assert refit[-1]["optimizer_steps"] == selected_updates
    assert summary["refit_optimizer_steps"] == selected_updates


def test_revision_resume_cli(tmp_path):
    from scripts.run_multibranch_revision import main

    base = ["--profile", "missing.json", "--out", str(tmp_path / "run"), "--development-only"]
    extras = (
        ["--replay-checkpoint", "model.pt", "--resume-training-checkpoint", "training_resume.json"],
        ["--score-only", "--resume-training-checkpoint", "training_resume.json"],
        ["--resume-training-checkpoint", "model.pt"],
        ["--resume-training-checkpoint", str(tmp_path / "run/training_resume.json")],
    )
    for extra, message in zip(
        extras,
        (
            "mutually exclusive",
            "mutually exclusive",
            "latest training_resume",
            "existing checkpoint pointer",
        ),
        strict=True,
    ):
        with pytest.raises(ValueError, match=message):
            main(base + extra)
    assert not (tmp_path / "run").exists()
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(dataclasses.asdict(recipe("v2"))))
    (tmp_path / "run").mkdir()
    with pytest.raises(ValueError, match="output exists"):
        main(["--profile", str(profile), "--out", str(tmp_path / "run"), "--development-only"])


def test_revision_resume_api_refusals(revision_data, tmp_path, monkeypatch):
    panel, _, calendar = revision_data
    with pytest.raises(ValueError, match="mutually exclusive"):
        fit_predict_revision(
            panel,
            "2016-04-24",
            28,
            config=recipe("v2"),
            out_dir=str(tmp_path / "run"),
            calendar_path=str(calendar),
            replay_checkpoint="model.pt",
            resume_training_checkpoint="training_resume.json",
        )
    (tmp_path / "existing").mkdir()
    with pytest.raises(ValueError, match="already exists"):
        fit_predict_revision(
            panel,
            "2016-04-24",
            28,
            config=recipe("v2"),
            out_dir=str(tmp_path / "existing"),
            calendar_path=str(calendar),
        )
    from provenance import lineage

    original_digest = lineage.directory_digest
    calls = []

    def changed_once(path, **kwargs):
        if str(path) == str(Path(panel.path).resolve()):
            calls.append(1)
            if len(calls) == 2:
                return "changed-input-digest"
        return original_digest(path, **kwargs)

    monkeypatch.setattr(lineage, "directory_digest", changed_once)
    with pytest.raises(ValueError, match="refusing checkpoint/publication"):
        fit_predict_revision(
            panel,
            "2016-04-24",
            28,
            config=recipe("v2"),
            out_dir=str(tmp_path / "source-drift"),
            calendar_path=str(calendar),
        )


@pytest.mark.parametrize("variant", ["v2", "v3"])
def test_revision_training_resume_patience(revision_data, tmp_path, monkeypatch, variant):
    """Controlled flat validation scores exercise nonzero stale and an early phase exit."""
    c = recipe(
        variant,
        epochs=None,
        max_updates=6,
        patience=1,
        evaluate_every=1,
        early_stopping_absolute_min_delta=0.01,
    )
    panel, _, calendar = revision_data
    kwargs = dict(calendar_path=str(calendar), config=c, levels=M5_LEVELS)
    control, resumed = tmp_path / "control", tmp_path / "resumed"
    original_validate, original_save = RollingTrainer.validate, TrainingRun.save
    validation_calls = []

    def flat_validation(*args, **kwargs):
        result = original_validate(*args, **kwargs)
        result["wrmsse"] = 1.0 if not validation_calls else 0.995
        validation_calls.append(1)
        return result

    monkeypatch.setattr(RollingTrainer, "validate", flat_validation)
    fit_predict_revision(panel, "2016-04-24", 28, out_dir=str(control), **kwargs)
    validation_calls.clear()

    def stop(run, state, *, reason):
        original_save(run, state, reason=reason)
        if state["phase"] == "selection" and state["stale"] == 1 and reason == "validation":
            raise RuntimeError("stale checkpoint interruption")

    with monkeypatch.context() as patch:
        patch.setattr(TrainingRun, "save", stop)
        with pytest.raises(RuntimeError, match="stale checkpoint"):
            fit_predict_revision(panel, "2016-04-24", 28, out_dir=str(resumed), **kwargs)
    validation_calls.clear()
    fit_predict_revision(
        panel,
        "2016-04-24",
        28,
        out_dir=str(resumed),
        resume_training_checkpoint=str(resumed / "training_resume.json"),
        **kwargs,
    )
    assert not validation_calls  # Already reached patience; do not take another selection step.
    a, b = (torch.load(path / "model.pt", weights_only=True) for path in (control, resumed))
    equal(a, b)
    assert b["training"]["selected_updates"] == 3
    assert [row["update"] for row in b["training"]["validation"]] == [3, 4]


@pytest.mark.parametrize("variant", ["v2", "v3"])
@pytest.mark.parametrize("ceiling", [5, 50_000])
def test_revision_fixed_epoch_budget_honors_patience(
    revision_data, tmp_path, monkeypatch, variant, ceiling
):
    """Maximum epochs is a ceiling; fixed-epoch selection still stops on validation patience."""
    config = recipe(
        variant,
        epochs=ceiling,
        patience=1,
        validation_every_epochs=1,
        early_stopping_absolute_min_delta=0.01,
    )
    panel, _, calendar = revision_data
    original_validate = RollingTrainer.validate
    validation_calls = []

    def flat_validation(*args, **kwargs):
        result = original_validate(*args, **kwargs)
        result["wrmsse"] = 1.0 if not validation_calls else 0.995
        validation_calls.append(1)
        return result

    monkeypatch.setattr(RollingTrainer, "validate", flat_validation)
    output = tmp_path / variant
    fit_predict_revision(
        panel,
        "2016-04-24",
        28,
        out_dir=str(output),
        calendar_path=str(calendar),
        config=config,
        levels=M5_LEVELS,
    )
    checkpoint = torch.load(output / "model.pt", weights_only=True)
    training = checkpoint["training"]
    steps = training["steps_per_epoch"]
    assert training["selected_updates"] == steps
    assert training["refit_optimizer_steps"] == steps
    assert [item["update"] for item in training["validation"]] == [steps, 2 * steps]
    assert len(training["phase_coverage"]["selection"]["epochs"]) == 2
    assert len(training["phase_coverage"]["refit"]["epochs"]) == 1


def test_revision_min_delta_uses_maximum_threshold_and_strict_boundary() -> None:
    trainer = object.__new__(RollingTrainer)
    trainer.config = recipe(
        "v2",
        early_stopping_absolute_min_delta=0.01,
        early_stopping_relative_min_delta=0.1,
    )
    assert not trainer._improved(0.9, 1.0)  # noqa: SLF001
    assert trainer._improved(0.899, 1.0)  # noqa: SLF001
    assert trainer._improved(100.0, float("inf"))  # noqa: SLF001


@pytest.mark.parametrize(
    "fault",
    [
        "schema",
        "admission",
        "origin",
        "optimizer",
        "premature_refit",
        "selection_counters",
        "scheduler",
        "feature",
        "rng",
        "betas",
        "eps",
        "weight_decay",
        "missing_moment",
        "extra_moment",
        "negative_second_moment",
        "group_mapping",
        "parameter_updates",
        "best_keys",
        "best_shape",
        "best_dtype",
        "nll_nan",
        "nll_missing",
        "empty_optimizer",
        "previous_admission",
        "moment_dtype",
        "second_moment_dtype",
        "moment_shape",
        "second_moment_shape",
        "step_dtype",
        "cuda_rng",
        "block_position",
        "proposal_coverage",
        "likelihood_total",
        "epoch_coverage_none",
        "epoch_coverage_missing",
        "epoch_coverage_values",
        "phase_coverage_none",
        "order_dtype",
        "manifest_empty_frontier",
        "manifest_earlier_frontier",
        "manifest_bad_sequence",
        "manifest_refit_frontier",
        "manifest_update_dtype",
        "previous_coverage_empty",
        "previous_coverage_malformed",
        "previous_coverage_order",
        "previous_coverage_journal",
        "validation_ledger_rewrite",
        "step_cadence_rewrite",
    ],
)
@pytest.mark.parametrize("variant", ["v2", "v3"])
def test_rolling_resume_binding_refusals(revision_data, tmp_path, monkeypatch, fault, variant):
    retained = fault.startswith(("best_", "nll_")) or fault == "validation_ledger_rewrite"
    refit_fault = fault in {
        "premature_refit",
        "previous_admission",
        "manifest_refit_frontier",
    } or fault.startswith("previous_coverage_")
    completed_epoch = fault in {
        "epoch_coverage_missing",
        "epoch_coverage_values",
        "manifest_earlier_frontier",
        "manifest_bad_sequence",
        "likelihood_total",
    }
    changes = dict(epochs=3, validation_every_epochs=2) if retained else {}
    _, trainer = assembled(revision_data, variant, **changes)
    root = tmp_path / "run"
    configure_training_runtime("cpu", 1)
    original_save = TrainingRun.save

    def stop(run, state, *, reason):
        original_save(run, state, reason=reason)
        if state["phase"] == ("refit" if refit_fault else "selection") and state["update"] == (
            0
            if fault == "manifest_refit_frontier"
            else 5
            if retained
            else 1
            if fault == "premature_refit"
            else 2
            if refit_fault or completed_epoch
            else 1
        ):
            raise RuntimeError("stop")

    with monkeypatch.context() as patch:
        patch.setattr(TrainingRun, "save", stop)
        with (
            TrainingRun(root, {}, every_updates=1) as run,
            pytest.raises(RuntimeError, match="stop"),
        ):
            trainer.fit(run.log, run=run)
    # Deliberately rebind digests: semantic refusal must precede log/export recovery.
    pointer_path = root / "training_resume.json"
    pointer = json.loads(pointer_path.read_text())
    manifest_path = root / "checkpoints" / pointer["manifest"]
    manifest = json.loads(manifest_path.read_text())
    payload_path = root / "checkpoints" / manifest["payload"]
    payload = torch.load(payload_path, weights_only=True)
    saved = payload["state"]
    if fault == "schema":
        saved["schema"] = "wrong"
    if fault == "origin":
        saved["validation_origins"] = ("2000-01-01",)
    if fault == "admission":
        saved["phase_admission"]["selection"]["layout_origin"] = "2000-01-01"
    if fault == "previous_admission":
        saved["phase_admission"]["selection"]["layout_origin"] = "2000-01-01"
    if fault.startswith("previous_coverage_"):
        coverage = saved["phase_coverage"]["selection"]
        if fault == "previous_coverage_empty":
            coverage["epochs"] = []
        elif fault == "previous_coverage_malformed":
            coverage["epochs"] = [None, None]
        elif fault == "previous_coverage_order":
            coverage["epochs"][-1]["epoch"] = 99
        else:
            coverage["epochs"][-1]["informative_series"] -= 1
            coverage["epochs"][-1]["minimum_informative_visits"] = 0
            coverage["informative_series_visits"] -= 1
            coverage["minimum_visits_per_series"] -= 1
    if fault == "empty_optimizer":
        saved["optimizer"]["state"] = {}
        saved["parameter_updates"] = [0] * len(saved["parameter_updates"])
    if fault == "optimizer":
        next(iter(saved["optimizer"]["state"].values()))["step"].fill_(1000)
    if fault == "premature_refit":
        saved["phase_done"] = True
        # Keep the newly checked coverage structure valid; exercise terminality itself.
        saved["phase_coverage"]["refit"] = dict(
            epochs=saved["epoch_coverage"],
            series_seen=int(saved["seen_series"].sum()),
            informative_series_visits=int(saved["likelihood_visits"].sum()),
            minimum_visits_per_series=int(saved["likelihood_visits"].min()),
            maximum_visits_per_series=int(saved["likelihood_visits"].max()),
            minimum_row_proposals=int(saved["proposal_visits"].min()),
            maximum_row_proposals=int(saved["proposal_visits"].max()),
        )
    if fault == "selection_counters":
        saved["selected_updates"] = 4
    if fault == "validation_ledger_rewrite":
        assert saved["validation_slots"] and saved["evaluations"]
        saved["validation_slots"][-1]["covered_series"] -= 1
        saved["validation_slots"][-1]["evaluated"] = False
        saved["evaluations"] = []
        saved["selected_updates"] = 0
        saved["best_score"] = None
        saved["best_state"] = None
        saved["latest_validation"] = None
        saved["stale"] = 0
    if fault == "step_cadence_rewrite":
        saved["steps_per_epoch"] += 1
    if fault == "scheduler":
        saved["scheduler"]["completed_updates"] += 1
    if fault == "feature":
        saved["feature_state"] = {"bad": True}
    if fault == "rng":
        saved["rng"] = {}
    if fault == "block_position":
        assert saved["block_position"] > 0
        saved["block_position"] = 0
    if fault == "order_dtype":
        saved["block_order"] = saved["block_order"].to(torch.int32)
    if fault == "proposal_coverage":
        saved["proposal_visits"].fill_(1)
    if fault == "likelihood_total":
        # Keep the pointwise arrays mutually consistent; only epoch-total accounting is wrong.
        assert saved["likelihood_visits"][0] > 0
        saved["likelihood_visits"][0] -= 1
        saved["seen_series"] = saved["likelihood_visits"] > 0
    if fault == "epoch_coverage_none":
        saved["epoch_coverage"] = None
    if fault == "epoch_coverage_missing":
        saved["epoch_coverage"] = []
    if fault == "epoch_coverage_values":
        saved["epoch_coverage"][0]["proposed_series"] -= 1
    if fault == "phase_coverage_none":
        saved["phase_coverage"] = None
    if fault in {"betas", "eps", "weight_decay"}:
        saved["optimizer"]["param_groups"][0][fault] = (0.8, 0.99) if fault == "betas" else 1.0
    if fault == "missing_moment":
        saved["optimizer"]["state"].pop(next(iter(saved["optimizer"]["state"])))
    if fault == "extra_moment":
        saved["optimizer"]["state"][999999] = copy.deepcopy(
            next(iter(saved["optimizer"]["state"].values()))
        )
    if fault == "negative_second_moment":
        next(iter(saved["optimizer"]["state"].values()))["exp_avg_sq"].fill_(-1)
    if fault in {"moment_dtype", "second_moment_dtype", "step_dtype"}:
        key = {
            "moment_dtype": "exp_avg",
            "second_moment_dtype": "exp_avg_sq",
            "step_dtype": "step",
        }[fault]
        moments = next(iter(saved["optimizer"]["state"].values()))
        moments[key] = moments[key].to(torch.float16 if key == "step" else torch.float64)
    if fault in {"moment_shape", "second_moment_shape"}:
        key = "exp_avg" if fault == "moment_shape" else "exp_avg_sq"
        moments = next(iter(saved["optimizer"]["state"].values()))
        moments[key] = moments[key].new_zeros(moments[key].numel() + 1)
    if fault == "group_mapping":
        saved["optimizer"]["param_groups"][0]["params"].reverse()
    if fault == "parameter_updates":
        saved["parameter_updates"][0] = -1
    if fault.startswith("best_"):
        key = next(iter(saved["best_state"]))
        if fault == "best_keys":
            saved["best_state"].pop(key)
        if fault == "best_shape":
            saved["best_state"][key] = saved["best_state"][key].new_zeros(
                saved["best_state"][key].numel() + 1
            )
        if fault == "best_dtype":
            saved["best_state"][key] = saved["best_state"][key].double()
    if fault == "nll_nan":
        saved["latest_validation"]["nll"] = float("nan")
    if fault == "nll_missing":
        saved["latest_validation"].pop("nll")
    if fault == "manifest_update_dtype":
        manifest["update"] = float(manifest["update"])
    elif fault.startswith("manifest_"):
        log_path = root / "training.jsonl"
        lines = log_path.read_bytes().splitlines(keepends=True)
        if fault == "manifest_bad_sequence":
            row = json.loads(lines[0])
            row["update"] = 2
            lines[0] = (json.dumps(row) + "\n").encode()
            prefix = b"".join(lines)
            log_path.write_bytes(prefix)
        else:
            prefix = (
                lines[0]
                if fault in {"manifest_earlier_frontier", "manifest_refit_frontier"}
                else b""
            )
        last = json.loads(prefix.splitlines()[-1]) if prefix else None
        manifest.update(
            log_bytes=len(prefix),
            records=len(prefix.splitlines()),
            last_record=[last["phase"], last["update"]] if last else None,
            log_sha256=hashlib.sha256(prefix).hexdigest(),
        )
    torch.save(payload, payload_path)
    manifest["payload_sha256"] = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    pointer["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    pointer_path.write_text(json.dumps(pointer))
    log_path = root / "training.jsonl"
    preserved = {
        "training.jsonl": log_path.read_bytes() + b'{"uncommitted":true}\n',
        "model.pt": b"partial model",
        "forecast.parquet": b"partial forecast",
    }
    for name, value in preserved.items():
        (root / name).write_bytes(value)
    _, trainer = assembled(revision_data, variant, **changes)
    if fault.startswith("manifest_"):

        def unexpected_step(*args, **kwargs):
            raise AssertionError("optimizer advanced before journal-frontier acceptance")

        monkeypatch.setattr(torch.optim.AdamW, "step", unexpected_step)
    if fault == "cuda_rng":
        # Structural execution-policy injection at the restore boundary, not CUDA computation.
        restore = rolling_training.restore_training_rng
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(
            rolling_training,
            "restore_training_rng",
            lambda state, generator, *, device: restore(state, generator, device="cuda"),
        )
    message = (
        "invalid AdamW checkpoint moments"
        if fault in {"moment_shape", "second_moment_shape"}
        else "sampler/epoch coverage contradicts completed passes"
        if fault == "likelihood_total"
        else "checkpoint|unfinished or uncovered phase"
        if fault == "premature_refit"
        else "feature"
        if fault == "feature"
        else "CUDA RNG device count"
        if fault == "cuda_rng"
        else "checkpoint phase counters differ from completed journal frontier"
        if fault == "manifest_refit_frontier"
        else "checkpoint validation ledger differs from committed journal"
        if fault == "validation_ledger_rewrite"
        else "log|checkpoint"
        if fault.startswith("manifest_")
        else "checkpoint"
    )
    with (
        pytest.raises(ValueError, match=message),
        TrainingRun(root, {}, resume=pointer_path) as run,
    ):
        trainer.fit(run.log, run=run)
    assert all((root / name).read_bytes() == value for name, value in preserved.items())
    assert not (root / "recovery").exists()


@pytest.mark.parametrize("variant", ["v2", "v3"])
@pytest.mark.parametrize(
    "epochs,boundary,position", [(2, 3, 12), (2, 4, 5), (None, 4, 10), (None, 5, 12)]
)
def test_revision_sampler_continuation(
    revision_data, tmp_path, monkeypatch, variant, epochs, boundary, position
):
    """Real twelve-row batches of 5,5,2; hierarchy-only steps do not advance the sampler."""
    configure_training_runtime("cpu", 1)
    changes = dict(epochs=epochs, batch_size=5, max_updates=8, patience=10)
    original_save = TrainingRun.save

    def stop(run, state, *, reason):
        original_save(run, state, reason=reason)
        if state["phase"] == "selection" and state["update"] == boundary:
            assert state["block_position"] == position
            raise RuntimeError("sampler boundary")

    states, traces = [], []
    for name in ("control", "resumed"):
        root = tmp_path / name
        if name == "resumed":
            _, trainer = assembled(revision_data, variant, **changes)
            with monkeypatch.context() as patch:
                patch.setattr(TrainingRun, "save", stop)
                with TrainingRun(root, {}, every_updates=1) as run:
                    with pytest.raises(RuntimeError, match="sampler boundary"):
                        trainer.fit(run.log, run=run)
        _, trainer = assembled(revision_data, variant, **changes)
        resume = root / "training_resume.json" if name == "resumed" else None
        with TrainingRun(root, {}, every_updates=1, resume=resume) as run:
            trainer.fit(run.log, run=run)
        with TrainingRun(root, {}, resume=root / "training_resume.json") as run:
            states.append(run.loaded)
        traces.append(
            numerical(
                [json.loads(row) for row in (root / "training.jsonl").read_text().splitlines()]
            )
        )
    equal(*states)
    equal(*traces)


@pytest.mark.parametrize("variant", ["v2", "v3"])
def test_revision_resume_phase_vocabulary(real_counts, tmp_path, monkeypatch, variant):
    """Real UCI targets with a structural new-category injection between phase layouts."""
    from autoforecast.adapters.mft import revision_data_spec
    from autoforecast.hmfnet_features import HMFNetConfig
    from autoforecast.rolling_data import PanelArrays
    from autoforecast.test_revision_portability import generic_config

    panel, origin, horizon, weekly = real_counts
    c = generic_config("v2", weekly, checkpoint_every_updates=1, progress="off")
    if variant == "v3":
        c = dataclasses.replace(c, variant="v3", v3=HMFNetConfig(span=28), trunk_stages=2)
    spec = revision_data_spec(panel, horizon, c)
    spec.time_varying_unknown_categoricals = ("weather",)
    data = PanelArrays.read(panel, origin, horizon, max_bytes=3_000_000, data_spec=spec)
    validation_origin = data.date_at(data.target.shape[1] - 2 * horizon - 1).isoformat()
    earliest = min(rolling_origins(data, validation_origin, c))
    categories = data.features["weather"].copy()
    categories[:, data.position(earliest) :] = "structural_new_refit_category"
    data = dataclasses.replace(data, features={**data.features, "weather": categories})

    def assembled_phase():
        features = (HMFNetRevisionFeatures if variant == "v3" else RevisionFeatures)(
            panel, data, c, data_spec=spec
        )
        trainer = RollingTrainer(
            data,
            c,
            (),
            features.groups,
            features.make_model,
            features.batch,
            features.training_loss,
            features.mean,
            features.optimizer_groups,
            features.begin_phase,
            features.prediction_context if variant == "v3" else None,
            features.diagnostics,
            validation_values=features.validation_values,
            feature_training_state=features.feature_training_state,
            restore_feature_training_state=features.restore_feature_training_state,
        )
        return trainer

    configure_training_runtime("cpu", 1)
    control, resumed = tmp_path / "control", tmp_path / "resumed"
    trainer = assembled_phase()
    with TrainingRun(control, {}, every_updates=1) as run:
        model, info = trainer.fit(run.log, run=run)
    expected = trainer.predict(model, origin)
    assert (
        info["phase_admission"]["selection"]["dynamic_vocab"]
        != info["phase_admission"]["refit"]["dynamic_vocab"]
    )
    original_save = TrainingRun.save

    def stop(run, state, *, reason):
        original_save(run, state, reason=reason)
        if state["phase"] == "refit" and state["update"] == 1:
            raise RuntimeError("stop at changed refit layout")

    with monkeypatch.context() as patch:
        patch.setattr(TrainingRun, "save", stop)
        with (
            pytest.raises(RuntimeError, match="changed refit"),
            TrainingRun(resumed, {}, every_updates=1) as run,
        ):
            assembled_phase().fit(run.log, run=run)
    trainer = assembled_phase()
    with TrainingRun(resumed, {}, resume=resumed / "training_resume.json") as run:
        continued, _ = trainer.fit(run.log, run=run)
    equal(model.state_dict(), continued.state_dict())
    assert np.array_equal(expected, trainer.predict(continued, origin))


def test_revision_validation_refusals(revision_data, monkeypatch):
    features, trainer = assembled(revision_data, "v2")
    features.begin_phase(rolling_origins(trainer.data, "2016-02-28", trainer.config), "selection")
    model = features.make_model()
    model.train()
    original = trainer.validation_values

    def invalid(*args):
        mean, nll = original(*args)
        return mean * float("nan"), nll

    trainer.validation_values = invalid
    with pytest.raises(ValueError, match="validation prediction or NLL"):
        trainer.validate(model, "2016-03-27", None)
    assert model.training
    trainer._validation_targets.clear()

    def overflow(*args):
        mean, nll = original(*args)
        return torch.full_like(mean.double(), 1e308), nll

    trainer.validation_values = overflow
    with pytest.raises(ValueError, match="nonfinite validation metric"):
        trainer.validate(model, "2016-03-27", None)
    trainer.config = dataclasses.replace(trainer.config, validation_backend="numpy")
    with pytest.raises(ValueError, match="nonfinite NumPy validation metric"):
        trainer.validate(model, "2016-03-27", None)
    trainer._validation_targets.clear()
    trainer.data = dataclasses.replace(trainer.data, observed=np.zeros_like(trainer.data.observed))
    with pytest.raises(ValueError, match="no observed targets"):
        trainer.validate(model, "2016-03-27", None)
    assert model.training


@pytest.mark.parametrize("variant", ["v2", "v3"])
def test_revision_predecessor_training(revision_data, variant):
    """Same real M5 data/cadence: retain the archived implementation as the control oracle."""
    path = Path(__file__).resolve().parents[2] / "v3-work.olz1ND/autoforecast/rolling_training.py"
    if not path.is_file():
        pytest.skip("Retained pre-reliability source is not available in this checkout")
    spec = importlib.util.spec_from_file_location("_pre_reliability_rolling_training", path)
    previous = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = previous
    spec.loader.exec_module(previous)
    outputs, records, batches = [], [], []
    for old in (True, False):
        configure_training_runtime("cpu", 1)
        _, trainer = assembled(revision_data, variant)
        if old:
            fields = {f.name for f in dataclasses.fields(previous.RollingTrainer)}
            trainer = previous.RollingTrainer(**{k: getattr(trainer, k) for k in fields})
            fields = {f.name for f in dataclasses.fields(previous.RollingConfig)}
            trainer.config = previous.RollingConfig(
                **{k: getattr(trainer.config, k) for k in fields}
            )
        original_batch = trainer.batch
        batch_trace = []

        def traced(
            origin, rows, device, targets, batch_trace=batch_trace, original_batch=original_batch
        ):
            if targets:
                batch_trace.append((origin, rows.tolist()))
            return original_batch(origin, rows, device, targets)

        trainer.batch = traced
        log = []
        model, info = trainer.fit(log.append)
        outputs.append((copy.deepcopy(model.state_dict()), info))
        records.append(log)
        batches.append(batch_trace)
    equal(outputs[0][0], outputs[1][0])
    equal(batches[0], batches[1])
    for key in ("selected_updates", "phase_coverage", "phase_admission", "optimizer_groups"):
        equal(outputs[0][1][key], outputs[1][1][key])
    for old, new in zip(records[0], records[1], strict=True):
        for key in (
            "phase",
            "update",
            "origin",
            "loss",
            "learning_rate",
            "origin_sampling",
            "hierarchy_loss",
            "expert_sampled_ids",
            "epoch_coverage",
        ):
            if key in old:
                equal(old[key], new[key])
        for origin, score in old.get("validation", {}).items():
            assert score["wrmsse"] == pytest.approx(new["validation"][origin]["wrmsse"], rel=1e-12)


@pytest.mark.parametrize("variant", ["v2", "v3"])
def test_revision_skipped_epoch_resume(revision_data, tmp_path, monkeypatch, variant):
    """Three real-fixture epochs, interval two, a retained-score interruption and final eval."""
    configure_training_runtime("cpu", 1)
    output = io.StringIO()

    def train(root, resume=False):
        _, trainer = assembled(
            revision_data, variant, epochs=3, validation_every_epochs=2, progress="plain"
        )
        capture = trainer.feature_training_state

        def delayed_capture():
            # Known capture cost must be inside the outer checkpoint pause, not hidden.
            time.sleep(0.005)
            return capture()

        trainer.feature_training_state = delayed_capture
        progress = TrainingProgress("plain", stream=output)
        update = progress.update

        def visible(record, **kwargs):
            progress.last_line = 0.0
            update(record, **kwargs)

        progress.update = visible
        with TrainingRun(
            root, {}, every_updates=1, resume=root / "training_resume.json" if resume else None
        ) as run:
            return trainer.fit(run.log, run=run, progress=progress)

    control, resumed = tmp_path / "control", tmp_path / "resumed"
    control_model, control_info = train(control)
    native_save = TrainingRun.save

    def interrupted(run, state, *, reason):
        native_save(run, state, reason=reason)
        if state["phase"] == "selection" and state["update"] == 5:
            assert state["latest_validation"]["update"] == 4
            assert state["latest_validation"]["epoch"] == 2
            assert state["stale"] == 0 and state["selected_updates"] == 4
            raise RuntimeError("interrupt on skipped validation epoch")

    with monkeypatch.context() as patch:
        patch.setattr(TrainingRun, "save", interrupted)
        with pytest.raises(RuntimeError, match="skipped validation epoch"):
            train(resumed)
    resumed_model, resumed_info = train(resumed, resume=True)
    equal(control_model.state_dict(), resumed_model.state_dict())
    equal(control_info, resumed_info)
    logs, states = [], []
    for root in (control, resumed):
        rows = [json.loads(line) for line in (root / "training.jsonl").read_text().splitlines()]
        assert [row["update"] for row in rows if "validation" in row] == [4, 6]
        assert all("validation" not in row for row in rows if row["epoch"] == 1)
        logs.append(numerical(rows))
        pointer = json.loads((root / "training_resume.json").read_text())
        manifest = json.loads((root / "checkpoints" / pointer["manifest"]).read_text())
        states.append(
            torch.load(root / "checkpoints" / manifest["payload"], weights_only=True)["state"]
        )
        events = [json.loads(p.read_text()) for p in sorted((root / "lifecycle").glob("*.json"))]
        pauses = [event for event in events if event["kind"] == "checkpoint_pause"]
        assert pauses and all(event["seconds"] >= 0.005 for event in pauses)
        publications = {
            (event["session"], event["generation"]): event["seconds"]
            for event in events
            if event["kind"] == "checkpoint"
        }
        assert all(
            event["seconds"] > publications[event["session"], event["generation"]]
            for event in pauses
        )
    equal(*logs)
    equal(*states)
    text = output.getvalue()
    assert "loss:" in text and "learning_rate:" in text
    assert "last_val_" not in text
    summaries = [line for line in text.splitlines() if "complete - loss:" in line]
    assert summaries and all(
        "val_wrmsse:" in line and "best_val_wrmsse:" in line for line in summaries
    )
