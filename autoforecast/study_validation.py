"""Verified study windows for native checkpoint selection without a Spark scoring job per epoch."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import tempfile
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds

from autoforecast._exact_json import loads_exact_json
from autoforecast.benchmark import MISSING_COLUMN, artifact_path
from autoforecast.data_admission import _load_dataset_contract_bytes
from autoforecast.frequency import canonical_origin, parse_frequency, period_points
from autoforecast.panel import Panel
from autoforecast.study_scoring import verify_study_evaluation
from provenance.lineage import directory_digest, snapshot_parquet_directory


@dataclass(frozen=True)
class StudyValidation:
    """One authenticated, complete holdout in the model's series order.

    The training loss remains model-owned. This window defines only the study metric used
    to choose a checkpoint. Original missingness controls the metric, not exposure inference.
    """

    contract_json: str
    contract_sha256: str
    series_ids: tuple[str, ...]
    target: np.ndarray
    observed: np.ndarray
    weight: np.ndarray | None = None
    scale: np.ndarray | None = None

    @property
    def metric(self) -> str:
        return json.loads(self.contract_json)["metric_policy"]["metric"]

    def as_record(self) -> dict:
        contract = json.loads(self.contract_json)
        return {
            "study_id": contract["study_id"], "boundary": contract["boundary"],
            "evaluation_contract_sha256": self.contract_sha256,
            "metric_policy": contract["metric_policy"],
        }

    def score(self, prediction: np.ndarray) -> dict[str, float | int | None]:
        """Match study cell errors and flat weighted RMSSE, including unavailable scores."""
        prediction = np.asarray(prediction, dtype=np.float64)
        if (prediction.shape != self.target.shape or not np.isfinite(prediction).all()
                or np.any(prediction < 0)):
            raise ValueError("study checkpoint predictions must cover the finite nonnegative grid")
        count = self.observed.sum(axis=1)
        n = int(count.sum())
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            error = np.where(self.observed, prediction - self.target, 0.0)
            squared = np.square(error).sum(axis=1)
            absolute = float(np.abs(error).sum())
            volume = float(np.where(self.observed, self.target, 0.0).sum())
            if not np.isfinite([squared.sum(), absolute, volume]).all():
                raise ValueError("study checkpoint metric overflowed")
            result: dict[str, float | int | None] = {
                "rmse": float(np.sqrt(squared.sum() / n)) if n else None,
                "mae": absolute / n if n else None,
                "wape": absolute / volume if volume else None,
                "observed_cells": n,
            }
            if self.metric == "wrmsse":
                assert self.weight is not None and self.scale is not None
                available = (count > 0) & np.isfinite(self.scale) & (self.scale > 0)
                result["wrmsse"] = (
                    float(np.sum(self.weight * np.sqrt(squared / count / self.scale))
                          / self.weight.sum())
                    if available.all() else None
                )
        if any(value is not None and not np.isfinite(value) for value in result.values()):
            raise ValueError("study checkpoint metric overflowed")
        return result

    @classmethod
    def read(
        cls, contract_path: str | pathlib.Path, contract_sha256: str, *, panel: Panel,
        origin: str, horizon: int, series_ids: tuple[str, ...],
    ) -> StudyValidation:
        """Authenticate the process handoff and load only its declared validation targets."""
        raw = pathlib.Path(contract_path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != contract_sha256:
            raise ValueError("study checkpoint evaluation contract changed")
        contract = loads_exact_json(raw)
        if type(contract) is not dict:
            raise ValueError("study checkpoint evaluation contract must be an object")
        path, record = verify_study_evaluation(contract)
        cadence = parse_frequency(panel.frequency)
        stage = contract["boundary"].split("/", 1)[0]
        if (stage not in {"tuning", "validation"}
                or contract["boundary"] != f"{stage}/{origin}/{horizon}/{cadence.code}"
                or type(series_ids) is not tuple
                or len(series_ids) != record["cohort_series_count"]
                or len(set(series_ids)) != len(series_ids)):
            raise ValueError("study checkpoint window or population differs from the model")
        dataset_path = path.parent / "dataset.yaml"
        dataset_bytes = dataset_path.read_bytes()
        if hashlib.sha256(dataset_bytes).hexdigest() != record["dataset_sha256"]:
            raise ValueError("study checkpoint dataset contract changed")
        dataset = _load_dataset_contract_bytes(dataset_bytes, dataset_path)
        if dataclasses.asdict(panel) != dataclasses.asdict(dataset.panel(panel.path)):
            raise ValueError("study checkpoint panel roles differ from the frozen dataset")
        if directory_digest(panel.path, suffixes=(".parquet",)) != contract["actuals"]["sha256"]:
            raise ValueError("study checkpoint validation panel differs from its frozen actuals")
        inputs = {
            name: (pathlib.Path(contract[name]["path"]), contract[name]["sha256"])
            for name in ("actuals", "grid", "coverage")
        }
        if contract["metric_policy"]["metric"] == "wrmsse":
            inputs["references"] = (
                artifact_path(path, record, "scoring_references"),
                record["artifacts"]["scoring_references"],
            )
        with tempfile.TemporaryDirectory(prefix="autoforecast-validation-") as directory:
            snapshots = {
                name: snapshot_parquet_directory(
                    source, pathlib.Path(directory) / name, expected_sha256=sha, label=name,
                )
                for name, (source, sha) in inputs.items()
            }
            result = _read_window(
                contract, contract_sha256, snapshots, panel, origin, horizon, series_ids,
            )
        verify_study_evaluation(contract)
        if result.score(result.target)[result.metric] is None:
            raise ValueError("study checkpoint selection metric is unavailable")
        return result


def _read_window(contract, sha, snapshots, panel, origin, horizon, series_ids) -> StudyValidation:
    """Consume operation-owned, verified bytes; no caller-owned table is re-read here."""
    cadence = parse_frequency(panel.frequency)
    points = period_points(origin, cadence, horizon)
    rows = {series: index for index, series in enumerate(series_ids)}
    columns = {canonical_origin(point, cadence): index for index, point in enumerate(points)}
    shape = (len(series_ids), horizon)

    def coordinates(table, series_column, time_column):
        try:
            row = np.array([rows[value] for value in table[series_column].to_pylist()])
            column = np.array([
                columns[canonical_origin(value, cadence)]
                for value in table[time_column].to_pylist()
            ])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("study checkpoint grid has unknown series or time keys") from error
        if (len(row) != len(series_ids) * horizon
                or len(np.unique(row * horizon + column)) != len(row)):
            raise ValueError("study checkpoint grid is incomplete or duplicated")
        return row, column

    grid = ds.dataset(snapshots["grid"], format="parquet").to_table()
    coordinates(grid, "series_id", "date")
    source = ds.dataset(snapshots["actuals"], format="parquet")
    if not pa.types.is_boolean(source.schema.field(MISSING_COLUMN).type):
        raise ValueError("study checkpoint actuals require original missingness")
    field = ds.field(panel.time_col)  # pyright: ignore[reportPrivateImportUsage]
    table = source.to_table(
        columns=[panel.series_col, panel.time_col, panel.target_col, MISSING_COLUMN],
        filter=(field >= pa.scalar(points[0])) & (field <= pa.scalar(points[-1])),
    )
    row, column = coordinates(table, panel.series_col, panel.time_col)
    values = table[panel.target_col].to_numpy().astype(np.float64)
    flags = table[MISSING_COLUMN].to_pylist()
    if (any(type(flag) is not bool for flag in flags) or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values[np.asarray(flags, dtype=bool)] != 0)):
        raise ValueError("study checkpoint actuals or original missingness are invalid")
    target = np.zeros(shape, dtype=np.float64)
    target[row, column] = values
    missing = np.zeros(shape, dtype=bool)
    missing[row, column] = flags
    mask_missing = contract["metric_policy"]["actual_missing"] == "mask"
    observed = ~missing if mask_missing else np.ones(shape, dtype=bool)
    coverage = ds.dataset(snapshots["coverage"], format="parquet").to_table().to_pylist()
    expected = [{
        "series_id": series, "expected": horizon,
        "originally_missing": int(missing[index].sum()),
        "zero_filled": 0 if mask_missing else int(missing[index].sum()),
        "excluded": int(missing[index].sum()) if mask_missing else 0,
        "scored": int(observed[index].sum()),
    } for index, series in enumerate(series_ids)]
    if sorted(coverage, key=lambda item: item["series_id"]) != sorted(
        expected, key=lambda item: item["series_id"],
    ):
        raise ValueError("study checkpoint coverage differs from the frozen denominator")
    weight = scale = None
    if "references" in snapshots:
        references = ds.dataset(snapshots["references"], format="parquet").to_table(
            filter=ds.field("boundary") == contract["boundary"],  # pyright: ignore[reportPrivateImportUsage]
        ).to_pylist()
        by_id = {item["series_id"]: item for item in references}
        if len(references) != len(series_ids) or set(by_id) != set(series_ids):
            raise ValueError("study checkpoint references differ from the frozen population")
        weight = np.array([by_id[series]["weight"] for series in series_ids], dtype=np.float64)
        scale = np.array([by_id[series]["scale"] for series in series_ids], dtype=np.float64)
        if (not np.isfinite(weight).all() or np.any(weight < 0)
                or not 0 < weight.sum() < np.inf
                or np.any(np.isinf(scale)) or np.any(scale < 0)):
            raise ValueError("study checkpoint weights or scales are invalid")
    for array in (target, observed, weight, scale):
        if array is not None:
            array.setflags(write=False)
    return StudyValidation(json.dumps(contract, sort_keys=True), sha, series_ids,
                           target, observed, weight, scale)


def study_score_callback(study: StudyValidation, predict):
    """Supply the CPU float64 study score to Lightning's existing selection callbacks.

    This callback must precede the epoch logger, early stopping and checkpoint callbacks.
    Keep the score on CPU: moving it to MPS would discard its float64 precision. Lightning's
    public callback_metrics mapping is also the input to its checkpoint and stopping rules.
    """
    import torch
    from lightning.pytorch import Callback

    class StudyScore(Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            states = [(module, module.training) for module in pl_module.modules()]
            try:
                pl_module.eval()
                with torch.inference_mode(), trainer.precision_plugin.forward_context():
                    score = study.score(predict(pl_module))[study.metric]
                if score is None or not np.isfinite(score):
                    raise ValueError("study checkpoint selection metric is unavailable")
                trainer.callback_metrics["study_score"] = torch.tensor(score, dtype=torch.float64)
            finally:
                for module, training in states:
                    module.training = training

    return StudyScore()
