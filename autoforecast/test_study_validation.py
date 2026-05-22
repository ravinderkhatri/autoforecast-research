"""Native checkpoint metrics retain the study's population and missing-value semantics."""

import dataclasses
import hashlib
import json
import math

import numpy as np
import pyarrow.parquet as pq
import pytest

from autoforecast import benchmark
from autoforecast.data_admission import load_dataset_contract
from autoforecast.study_validation import StudyValidation
from autoforecast.test_benchmark import _frozen_example


def window(*, metric="rmse", observed=None, weight=None, scale=None):
    return StudyValidation(
        json.dumps({"metric_policy": {"metric": metric, "actual_missing": "mask"}}), "a" * 64, ("a", "b"),
        np.array([[1.0, 0.0], [2.0, 4.0]]),
        np.array([[True, False], [True, True]]) if observed is None else observed,
        weight, scale,
    )


def test_checkpoint_metrics_match_hand_calculated_cell_denominators():
    # Two series, four forecasts, one masked actual: exactly three scored cells.
    score = window().score(np.array([[2.0, 99.0], [4.0, 1.0]]))
    assert score == {
        "rmse": pytest.approx(math.sqrt(14 / 3)), "mae": 2.0,
        "wape": pytest.approx(6 / 7), "observed_cells": 3,
    }
    zero = window(observed=np.ones((2, 2), dtype=bool)).score(
        np.array([[2.0, 99.0], [4.0, 1.0]]),
    )
    assert zero["rmse"] == pytest.approx(math.sqrt((14 + 99**2) / 4))
    assert zero["mae"] == 105 / 4
    assert zero["observed_cells"] == 4


def test_checkpoint_flat_wrmsse_uses_all_declared_weights_and_scales():
    score = window(
        metric="wrmsse", weight=np.array([1.0, 3.0]), scale=np.array([1.0, 4.0]),
    ).score(np.array([[2.0, 99.0], [4.0, 1.0]]))
    assert score["wrmsse"] == pytest.approx((1 + 3 * math.sqrt(6.5) / 2) / 4)


@pytest.mark.parametrize("scale,observed", [
    ([0.0, 4.0], [[True, False], [True, True]]),
    ([float("nan"), 4.0], [[True, False], [True, True]]),
    ([1.0, 4.0], [[False, False], [True, True]]),
])
def test_checkpoint_wrmsse_does_not_redistribute_an_unavailable_series(scale, observed):
    study = window(
        metric="wrmsse", weight=np.array([0.0, 3.0]), scale=np.array(scale),
        observed=np.array(observed),
    )
    if not np.asarray(observed).any(axis=1).all():
        with pytest.raises(ValueError, match="every retained series"):
            study.score(np.zeros((2, 2)))
    else:
        assert study.score(np.zeros((2, 2)))["wrmsse"] is None


def test_checkpoint_empty_coverage_and_zero_volume_remain_unavailable():
    empty = window(observed=np.zeros((2, 2), dtype=bool)).score(np.zeros((2, 2)))
    assert empty == {"rmse": None, "mae": None, "wape": None, "observed_cells": 0}
    zero = dataclasses.replace(window(), target=np.zeros((2, 2)))
    assert zero.score(np.ones((2, 2)))["wape"] is None


@pytest.mark.parametrize("prediction", [
    [[1, -1], [1, 1]], [[1, float("nan")], [1, 1]],
    [[1, float("inf")], [1, 1]], [[1, 1]],
])
def test_checkpoint_requires_every_prediction_even_when_actual_is_masked(prediction):
    with pytest.raises(ValueError, match="finite nonnegative grid"):
        window().score(np.array(prediction))


def test_checkpoint_rejects_overflow_in_errors_or_actual_volume():
    with pytest.raises(ValueError, match="overflowed"):
        window().score(np.full((2, 2), 1e308))
    large = dataclasses.replace(window(), target=np.full((2, 2), 1e308))
    with pytest.raises(ValueError, match="overflowed"):
        large.score(np.full((2, 2), 1e308))


def frozen_window(tmp_path, *, stage="validation"):
    path, record, _ = _frozen_example(tmp_path)
    recipe = record["recipe"]
    origin = (recipe["validation_origins"][0] if stage == "validation"
              else recipe["final_train_cutoff"])
    contract = benchmark.evaluation_contract(path, stage=stage, origin=origin, horizon=7, metric="rmse")
    file = tmp_path / "evaluation.json"
    file.write_text(json.dumps(contract))
    sha = hashlib.sha256(file.read_bytes()).hexdigest()
    panel = load_dataset_contract(path.parent / "dataset.yaml").panel(contract["actuals"]["path"])
    series = tuple(pq.read_table(benchmark.artifact_path(path, record, "source-keys"))[
        "__af_series_id"
    ].to_pylist())
    return file, sha, dict(panel=panel, origin=origin, horizon=7, series_ids=series), contract


def test_checkpoint_reader_authenticates_and_preserves_model_series_order(tmp_path):
    file, sha, args, contract = frozen_window(tmp_path)
    first = StudyValidation.read(file, sha, **args)
    reverse = StudyValidation.read(file, sha, **{**args, "series_ids": args["series_ids"][::-1]})
    np.testing.assert_array_equal(first.target, reverse.target[::-1])
    assert not first.target.flags.writeable and not first.observed.flags.writeable
    assert first.score(first.target)["rmse"] == 0
    assert first.as_record() == {
        "study_id": contract["study_id"], "boundary": contract["boundary"],
        "evaluation_contract_sha256": sha, "metric_policy": contract["metric_policy"],
    }


@pytest.mark.parametrize("defect", ["sha", "boundary", "omitted", "duplicate", "unknown", "actuals"])
def test_checkpoint_reader_rejects_changed_handoff_or_population(tmp_path, defect):
    file, sha, args, contract = frozen_window(tmp_path)
    if defect == "sha":
        sha = "0" * 64
    elif defect == "boundary":
        args["horizon"] = 6
    elif defect == "omitted":
        args["series_ids"] = args["series_ids"][:1]
    elif defect == "duplicate":
        args["series_ids"] = (args["series_ids"][0],) * 2
    elif defect == "unknown":
        args["series_ids"] = (args["series_ids"][0], "unknown")
    else:
        actuals = pq.read_table(contract["actuals"]["path"])
        from pathlib import Path
        pq.write_table(actuals.slice(1), Path(contract["actuals"]["path"]) / "part.parquet")
    with pytest.raises(ValueError):
        StudyValidation.read(file, sha, **args)


def test_checkpoint_reader_never_admits_test_labels_for_selection(tmp_path):
    file, sha, args, _ = frozen_window(tmp_path, stage="test")
    with pytest.raises(ValueError, match="window or population"):
        StudyValidation.read(file, sha, **args)


@pytest.mark.parametrize("physical", [False, True])
def test_native_window_preserves_missing_targets_and_frozen_identity(physical):
    from autoforecast.rolling_data import ValidationWindow
    from autoforecast.rolling_training import RollingTrainingContract

    study = dataclasses.replace(window(), target=np.array([[1.0, np.nan], [2.0, 4.0]]))
    native = ValidationWindow.from_study("2024-01-01", study,
                                         route_lower_bound_to_exact=physical)
    assert np.isnan(native.target[0, 1])
    np.testing.assert_array_equal(native.target, study.target)
    assert not native.target.flags.writeable
    RollingTrainingContract(mode="fixed_validation", fixed_validation=native)
    wrong = dataclasses.replace(native, target=np.nan_to_num(native.target))
    with pytest.raises(ValueError, match="differs from its frozen study"):
        RollingTrainingContract(mode="fixed_validation", fixed_validation=wrong)
    for invalid in (-1.0, 0.5, float("inf"), float("nan"), 2**24 + 1):
        changed = study.target.copy()
        changed[0, 0] = invalid
        with pytest.raises(ValueError, match="finite integer counts"):
            ValidationWindow.from_study("2024-01-01", dataclasses.replace(study, target=changed))
