"""Functional public-boundary tests for offline Transformers/PEFT TimesFM.

No model is downloaded and no long training runs.  Real Parquet and Spark exercise the cutoff
and fully-observed sampling boundary; deterministic maintained-provider fakes exercise exact
offline loading, expected-demand/median routing, final-only adapter publication, and atomic
failure behavior through the public functions.
"""

from __future__ import annotations

import copy
import csv
import dataclasses
import datetime as dt
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
import pathlib
import re
import shutil
import socket
import stat
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
import torch
from pyspark.sql import SparkSession

from autoforecast import resources
from autoforecast.adapters import foundation as timesfm_boundary
from autoforecast.adapters import isolated
from autoforecast.adapters import timesfm_transformers as subject
from autoforecast.adapters.timesfm_transformers import (
    LoRAContinuationConfig,
    LoRATrainingConfig,
    TimesFMTransformersBackend,
    continue_lora,
    prepare_lora_training_rows,
    train_lora,
)
from autoforecast.panel import Forecast, Panel
from provenance.lineage import directory_digest

CUTOFF = "2020-02-09"
TEST_CONFIG = LoRATrainingConfig(
    context_length=32,
    horizon_length=2,
    epochs=3,
    batch_size=2,
    rank=4,
    alpha=8,
    dropout=0.05,
    learning_rate=1e-4,
    weight_decay=0.01,
    gradient_clip_norm=1.0,
    seed=20260829,
    device="cpu",
)
TEST_CONTINUATION_CONFIG = LoRAContinuationConfig(
    max_additional_epochs=5,
    patience=2,
    restart_learning_rate=1e-5,
)
_SPARK: SparkSession | None = None


def spark() -> SparkSession:
    global _SPARK
    if _SPARK is None:
        if not os.environ.get("JAVA_HOME"):
            raise RuntimeError("JAVA_HOME unset; set JAVA_HOME to a Java 17 installation.")
        _SPARK = (
            SparkSession.builder.master("local[2]")
            .appName("test_timesfm_transformers")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.ui.enabled", "false")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate()
        )
        _SPARK.sparkContext.setLogLevel("ERROR")
    return _SPARK


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _panel(root: pathlib.Path, *, future_value: float = 999.0) -> Panel:
    rows: list[dict[str, object]] = []
    start = dt.date(2020, 1, 1)
    for day_number in range(1, 42):
        day = start + dt.timedelta(days=day_number - 1)
        if day_number <= 40:
            rows.extend(
                [
                    {
                        "id": "a",
                        "date": day,
                        "qty": 0.0 if day_number == 1 else float(day_number),
                        "state": "exact",
                        "price": 1.0,
                    },
                    {
                        "id": "b",
                        "date": day,
                        "qty": float(day_number),
                        "state": "unobserved" if day_number <= 5 else "exact",
                        "price": 2.0,
                    },
                    {
                        "id": "c",
                        "date": day,
                        "qty": float(day_number),
                        "state": "lower_bound" if day_number == 16 else "exact",
                        "price": 3.0,
                    },
                ]
            )
            if day_number <= 33:
                rows.append(
                    {
                        "id": "d",
                        "date": day,
                        "qty": float(day_number),
                        "state": "exact",
                        "price": 4.0,
                    }
                )
        else:
            for series_id in ("a", "b", "c", "d"):
                rows.append(
                    {
                        "id": series_id,
                        "date": day,
                        "qty": future_value,
                        "state": "exact",
                        "price": future_value * 10,
                    }
                )
    path = root / "panel.parquet"
    schema = pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("date", pa.date32(), nullable=False),
            pa.field("qty", pa.float64()),
            pa.field("state", pa.string()),
            pa.field("price", pa.float64()),
        ]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    return Panel(
        path=path,
        series_col="id",
        time_col="date",
        target_col="qty",
        exposure_col="state",
        known_future=("price",),
    ).validate()


def _source_file(root: pathlib.Path) -> tuple[pathlib.Path, str]:
    source = root / "raw-sales.csv"
    source.write_text("sealed raw source fixture\n")
    return source, _sha256(source)


def _forecast_panel(root: pathlib.Path) -> Panel:
    source = _panel(root)
    table = pq.read_table(source.path)
    table = table.filter(pc.is_in(table["id"], value_set=pa.array(["a", "b"])))
    path = root / "forecast-panel.parquet"
    pq.write_table(table, path)
    return Panel(
        path=path,
        series_col="id",
        time_col="date",
        target_col="qty",
        exposure_col="state",
        known_future=("price",),
    ).validate()


def _prepare(root: pathlib.Path, *, future_value: float = 999.0):
    source, source_digest = _source_file(root)
    panel = _panel(root, future_value=future_value)
    return prepare_lora_training_rows(
        panel,
        CUTOFF,
        spark=spark(),
        out_dir=root / "training-rows",
        config=TEST_CONFIG,
        source_data_path=source,
        source_data_sha256=source_digest,
        panel_content_sha256=directory_digest(panel.path, suffixes=(".parquet",)),
    )


def _base_checkpoint(root: pathlib.Path) -> tuple[pathlib.Path, str, str]:
    base = root / "base"
    base.mkdir()
    config = {
        "architectures": ["TimesFm2_5ModelForPrediction"],
        "decode_index": 5,
        "dtype": "float32",
        "force_flip_invariance": True,
        "infer_is_positive": True,
        "use_continuous_quantile_head": True,
        "horizon_length": 128,
        "quantiles": [index / 10 for index in range(1, 10)],
    }
    (base / "config.json").write_text(json.dumps(config, sort_keys=True))
    (base / "model.safetensors").write_bytes(b"functional fake weights")
    return base, _sha256(base / "config.json"), _sha256(base / "model.safetensors")


def _base_identity(config_digest: str, weights_digest: str):
    return mock.patch.multiple(
        subject,
        TRANSFORMERS_CONFIG_SHA256=config_digest,
        TRANSFORMERS_WEIGHTS_SHA256=weights_digest,
    )


def _runtime_lock(root: pathlib.Path) -> tuple[dict[str, object], dict[str, str]]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = re.sub(r"[-_.]+", "-", distribution.metadata["Name"].strip()).lower()
        if name in packages:
            continue
        packages[name] = distribution.version
    for name in (
        "accelerate",
        "huggingface-hub",
        "peft",
        "safetensors",
        "timesfm",
        "transformers",
    ):
        packages.setdefault(name, "functional")
    projection = subject.runtime_lock_projection_sha256(packages)
    path = root / "runtime-lock.json"
    path.write_text(
        json.dumps(
            {
                "python": (
                    f"{sys.version_info.major}.{sys.version_info.minor}."
                    f"{sys.version_info.micro}"
                ),
                "package_versions": packages,
                "runtime_lock_canonicalization": subject.RUNTIME_LOCK_CANONICALIZATION,
                "runtime_lock_sha256": projection,
            },
            sort_keys=True,
        )
    )
    return (
        {
            "runtime_lock_path": path,
            "runtime_lock_file_sha256": _sha256(path),
            "runtime_lock_record_path": (),
            "runtime_lock_sha256": projection,
        },
        packages,
    )


class RecordingTrainingBackend:
    def __init__(self, *, fail_at: int | None = None) -> None:
        self.calls: list[tuple[list[list[float]], list[list[float]]]] = []
        self.save_at: list[int] = []
        self.fail_at = fail_at

    def _train_batch(self, contexts, targets) -> float:
        self.calls.append((list(map(list, contexts)), list(map(list, targets))))
        if self.fail_at == len(self.calls):
            raise RuntimeError("injected functional training failure")
        return float(len(self.calls))

    def _save_adapter(self, path: pathlib.Path) -> None:
        self.save_at.append(len(self.calls))
        (path / "adapter_config.json").write_text('{"peft_type":"LORA"}\n')
        (path / "adapter_model.safetensors").write_bytes(b"functional adapter")

    def _identity(self):
        return {
            "runtime": {
                "torch": "functional",
                "transformers": "functional",
                "peft": "functional",
                "device": "cpu",
                "precision": "float32",
            },
            "parameters": {"trainable": 4, "total": 100},
        }


class RecordingContinuationBackend:
    def __init__(
        self,
        *,
        source_tree_sha256: str,
        source_model_sha256: str,
        continuation_config: LoRAContinuationConfig,
        validation_losses: tuple[float, ...],
        source_model_path: pathlib.Path | None = None,
    ) -> None:
        self.source_tree_sha256 = source_tree_sha256
        self.source_model_sha256 = source_model_sha256
        self.continuation_config = continuation_config
        self.validation_losses = validation_losses
        self.source_model_path = source_model_path
        self.state = 0
        self.train_calls: list[tuple[list[list[float]], list[list[float]]]] = []
        self.validation_calls: list[tuple[list[list[float]], list[list[float]]]] = []
        self.snapshots: list[int] = []
        self.restored: list[int] = []
        self.save_at: list[int] = []
        self.runtime_versions = {
            "torch": "functional",
            "transformers": "functional",
            "peft": "functional",
        }

    def _train_batch(self, contexts, targets) -> float:
        self.train_calls.append((list(map(list, contexts)), list(map(list, targets))))
        self.state += 1
        if self.source_model_path is not None and self.state == 1:
            self.source_model_path.write_bytes(b"mutated warm-start source")
        return float(self.state)

    def _validation_batch(self, contexts, targets) -> float:
        self.validation_calls.append((list(map(list, contexts)), list(map(list, targets))))
        return self.validation_losses[self.state]

    def _snapshot_adapter(self):
        self.snapshots.append(self.state)
        return self.state

    def _restore_adapter(self, state) -> None:
        assert type(state) is int
        self.state = state
        self.restored.append(state)

    def _save_adapter(self, path: pathlib.Path) -> None:
        self.save_at.append(self.state)
        (path / "adapter_config.json").write_text('{"peft_type":"LORA"}\n')
        (path / "adapter_model.safetensors").write_bytes(
            f"continued-state={self.state}\n".encode()
        )

    def _identity(self):
        return {
            "runtime": {
                **self.runtime_versions,
                "device": "cpu",
                "precision": "float32",
            },
            "parameters": {"trainable": 4, "total": 100},
            "source_adapter": {
                "tree_sha256": self.source_tree_sha256,
                "adapter_model_sha256": self.source_model_sha256,
            },
            "optimizer": {
                "optimizer": "torch.optim.AdamW",
                "scheduler": "CosineAnnealingLR_per_batch",
                "optimizer_state_restored": False,
                "scheduler_state_restored": False,
                "restart_learning_rate": (
                    self.continuation_config.restart_learning_rate
                ),
            },
        }


def _trained_source_adapter(root: pathlib.Path):
    rows = _prepare(root)
    base, config_digest, weights_digest = _base_checkpoint(root)
    runtime_args, runtime_packages = _runtime_lock(root)
    with _base_identity(config_digest, weights_digest), mock.patch.object(
        subject,
        "_installed_package_versions",
        return_value=runtime_packages,
    ):
        source = train_lora(
            rows.path,
            rows.rows_sha256,
            rows.manifest_sha256,
            checkpoint_dir=base,
            model_revision=subject.TRANSFORMERS_REVISION,
            out_dir=root / "source-adapter",
            config=TEST_CONFIG,
            backend=RecordingTrainingBackend(),
            **runtime_args,
        )
    source_model_sha256 = _sha256(
        pathlib.Path(source.path) / "adapter_model.safetensors"
    )
    return (
        rows,
        base,
        config_digest,
        weights_digest,
        runtime_args,
        runtime_packages,
        source,
        source_model_sha256,
    )


def _continue_test_adapter(
    *,
    rows,
    base: pathlib.Path,
    config_digest: str,
    weights_digest: str,
    runtime_args: dict[str, object],
    runtime_packages: dict[str, str],
    source,
    source_model_sha256: str,
    out_dir: pathlib.Path,
    continuation_config: LoRAContinuationConfig,
    backend: RecordingContinuationBackend,
):
    backend.runtime_versions = {
        package: runtime_packages[package]
        for package in ("torch", "transformers", "peft")
    }
    with _base_identity(config_digest, weights_digest), mock.patch.object(
        subject,
        "_installed_package_versions",
        return_value=runtime_packages,
    ):
        return continue_lora(
            rows.path,
            rows.rows_sha256,
            rows.manifest_sha256,
            source_adapter_dir=source.path,
            source_adapter_sha256=source.sha256,
            source_adapter_model_sha256=source_model_sha256,
            checkpoint_dir=base,
            model_revision=subject.TRANSFORMERS_REVISION,
            out_dir=out_dir,
            source_config=TEST_CONFIG,
            continuation_config=continuation_config,
            expected_panel_content_sha256=rows.panel_content_sha256,
            expected_training_cutoff=CUTOFF,
            backend=backend,
            **runtime_args,
        )


def test_prepare_lora_training_rows_is_exact_stratified_and_cutoff_safe() -> None:
    first_root = pathlib.Path(tempfile.mkdtemp()).resolve()
    second_root = pathlib.Path(tempfile.mkdtemp()).resolve()
    first = _prepare(first_root, future_value=999.0)
    second = _prepare(second_root, future_value=9_999_999.0)

    assert first.n_series == 2
    assert first.n_rows == 6
    assert first.rows_sha256 == second.rows_sha256
    assert first.panel_content_sha256 != second.panel_content_sha256
    assert first.manifest_sha256 != second.manifest_sha256
    table = pq.read_table(pathlib.Path(first.path) / subject.TRAINING_ROWS_FILE)
    descriptors = table.select(
        ["epoch", "series_id", "start_day", "context_end_day", "target_end_day"]
    ).to_pylist()
    assert [(row["epoch"], row["series_id"]) for row in descriptors] == [
        (epoch, series_id) for epoch in range(3) for series_id in ("a", "b")
    ]
    assert all(row["target_end_day"] <= 40 for row in descriptors)
    for row in descriptors:
        first_day = 2 if row["series_id"] == "a" else 6
        digest = hashlib.sha256(
            f"{TEST_CONFIG.seed}\0{row['epoch']}\0{row['series_id']}".encode()
        ).digest()
        expected_start = first_day + int.from_bytes(digest[:8], "big") % (7 - first_day + 1)
        assert row["start_day"] == expected_start
        assert row["context_end_day"] == expected_start + 31
        assert row["target_end_day"] == expected_start + 33
    index = pathlib.Path(first.path) / subject.TRAINING_ROWS_INDEX
    assert _sha256(index) == first.rows_sha256
    assert json.loads(index.read_text().splitlines()[0]) == {
        "series_id": "a",
        "epoch": 0,
        "start_day": descriptors[0]["start_day"],
        "context_end_day": descriptors[0]["context_end_day"],
        "target_end_day": descriptors[0]["target_end_day"],
    }
    authenticated = subject.validate_training_rows_artifact(
        first.path,
        first.rows_sha256,
        first.manifest_sha256,
        config=TEST_CONFIG,
        expected_cutoff=CUTOFF,
        expected_panel_content_sha256=first.panel_content_sha256,
    )
    assert authenticated == first
    try:
        subject.validate_training_rows_artifact(
            first.path,
            first.rows_sha256,
            first.manifest_sha256,
            config=TEST_CONFIG,
            expected_cutoff="2020-02-08",
            expected_panel_content_sha256=first.panel_content_sha256,
        )
    except ValueError as exc:
        assert "different cutoff" in str(exc)
    else:
        raise AssertionError("training-row authorization accepted a different cutoff")
    try:
        subject.validate_training_rows_artifact(
            second.path,
            second.rows_sha256,
            second.manifest_sha256,
            config=TEST_CONFIG,
            expected_cutoff=CUTOFF,
            expected_panel_content_sha256=first.panel_content_sha256,
        )
    except ValueError as exc:
        assert "different source panel" in str(exc)
    else:
        raise AssertionError("same descriptor rows hid mutated source-panel targets")


def test_optional_m5_source_refuses_corruption_but_discloses_absence(tmp_path, monkeypatch):
    """Only an absent optional source may skip; an installed wrong source must fail."""
    monkeypatch.setitem(
        globals(), "__file__", str(tmp_path / "adapters" / "tests" / "test_timesfm.py")
    )
    with pytest.raises(pytest.skip.Exception, match="public M5 source CSV is not installed"):
        test_real_m5_forward_cutoff_training_row_hashes_are_sealed()
    source = tmp_path / "data" / "m5" / "sales_train_evaluation.csv"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"intentionally invalid structural source fixture\n")
    try:
        with pytest.raises(AssertionError):
            test_real_m5_forward_cutoff_training_row_hashes_are_sealed()
    except pytest.skip.Exception:
        pytest.fail("an installed corrupt source must fail, not skip")


def test_real_m5_forward_cutoff_training_row_hashes_are_sealed() -> None:
    source = pathlib.Path(__file__).resolve().parent.parent.parent / (
        "data/m5/sales_train_evaluation.csv"
    )
    if not source.exists():
        pytest.skip(f"public M5 source CSV is not installed: {source}")
    assert _sha256(source) == subject.M5_SALES_TRAIN_EVALUATION_SHA256
    first_positive_by_series: list[tuple[str, int]] = []
    with source.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        series_index = header.index("id")
        day_indices = tuple(header.index(f"d_{day}") for day in range(1, 1914))
        for source_row in reader:
            first = next(
                (day for day, index in enumerate(day_indices, start=1)
                 if int(source_row[index]) > 0),
                None,
            )
            if first is not None:
                first_positive_by_series.append((source_row[series_index], first))
    config = subject.PROSPECTIVE_M5_LORA_CONFIG
    for identity in subject.M5_LORA_CUTOFF_IDENTITIES:
        eligible = sorted(
            [
                (series_id, first)
                for series_id, first in first_positive_by_series
                if first + config.context_length + config.horizon_length - 1
                <= identity.cutoff_day
            ],
            key=lambda item: item[0].encode("utf-8"),
        )
        digest = hashlib.sha256()
        first_row = None
        for epoch in range(config.epochs):
            for series_id, first in eligible:
                draw = int.from_bytes(
                    hashlib.sha256(
                        f"{config.seed}\0{epoch}\0{series_id}".encode()
                    ).digest()[:8],
                    "big",
                )
                start = first + draw % (identity.last_start_day - first + 1)
                row = {
                    "series_id": series_id,
                    "epoch": epoch,
                    "start_day": start,
                    "context_end_day": start + config.context_length - 1,
                    "target_end_day": (
                        start
                        + config.context_length
                        + config.horizon_length
                        - 1
                    ),
                }
                if first_row is None:
                    first_row = row
                digest.update(
                    (
                        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                    ).encode()
                )
        assert len(eligible) == identity.eligible_series
        assert len(eligible) * config.epochs == identity.windows
        assert digest.hexdigest() == identity.training_rows_sha256
        assert first_row == {
            "series_id": "FOODS_1_001_CA_1_evaluation",
            "epoch": 0,
            "start_day": identity.first_start_day,
            "context_end_day": identity.first_context_end_day,
            "target_end_day": identity.first_target_end_day,
        }


def test_train_lora_saves_once_after_final_epoch_and_failure_publishes_nothing() -> None:
    root = pathlib.Path(tempfile.mkdtemp()).resolve()
    rows = _prepare(root)
    base, config_digest, weights_digest = _base_checkpoint(root)
    runtime_args, runtime_packages = _runtime_lock(root)
    backend = RecordingTrainingBackend()
    with _base_identity(config_digest, weights_digest), mock.patch.object(
        subject,
        "_installed_package_versions",
        return_value=runtime_packages,
    ):
        artifact = train_lora(
            rows.path,
            rows.rows_sha256,
            rows.manifest_sha256,
            checkpoint_dir=base,
            model_revision=subject.TRANSFORMERS_REVISION,
            out_dir=root / "adapter",
            config=TEST_CONFIG,
            backend=backend,
            **runtime_args,
        )
    assert len(backend.calls) == 3
    assert backend.save_at == [3]
    assert artifact.n_training_rows == 6
    manifest = json.loads(pathlib.Path(artifact.manifest_path).read_text())
    assert manifest["publication"] == {
        "saved_after_all_epochs": True,
        "intermediate_checkpoints": 0,
        "evaluation_runs": 0,
    }
    assert manifest["training_rows"]["sha256"] == rows.rows_sha256
    with _base_identity(config_digest, weights_digest), mock.patch.object(
        subject,
        "_installed_package_versions",
        return_value=runtime_packages,
    ):
        authenticated = subject.validate_lora_adapter(
            artifact.path,
            artifact.sha256,
            config=TEST_CONFIG,
            expected_training_rows_sha256=rows.rows_sha256,
            expected_training_rows_manifest_sha256=rows.manifest_sha256,
            expected_panel_content_sha256=rows.panel_content_sha256,
            expected_training_cutoff=CUTOFF,
            forecast_origin=CUTOFF,
            **runtime_args,
        )
    assert authenticated == artifact
    with _base_identity(config_digest, weights_digest), mock.patch.object(
        subject,
        "_installed_package_versions",
        return_value=runtime_packages,
    ):
        try:
            subject.validate_lora_adapter(
                artifact.path,
                artifact.sha256,
                config=TEST_CONFIG,
                expected_training_rows_sha256="0" * 64,
                expected_training_rows_manifest_sha256=rows.manifest_sha256,
                expected_panel_content_sha256=rows.panel_content_sha256,
                expected_training_cutoff=CUTOFF,
                forecast_origin=CUTOFF,
                **runtime_args,
            )
        except ValueError as exc:
            assert "different training rows" in str(exc)
        else:
            raise AssertionError("adapter authorization accepted different training rows")
    failure_target = root / "failed-adapter"
    failure_target.mkdir()
    failing = RecordingTrainingBackend(fail_at=2)
    with _base_identity(config_digest, weights_digest), mock.patch.object(
        subject,
        "_installed_package_versions",
        return_value=runtime_packages,
    ):
        try:
            train_lora(
                rows.path,
                rows.rows_sha256,
                rows.manifest_sha256,
                checkpoint_dir=base,
                model_revision=subject.TRANSFORMERS_REVISION,
                out_dir=failure_target,
                config=TEST_CONFIG,
                backend=failing,
                **runtime_args,
            )
        except RuntimeError as exc:
            assert "injected functional training failure" in str(exc)
        else:
            raise AssertionError("failed LoRA training published an adapter")
    assert failing.save_at == []
    assert list(failure_target.iterdir()) == []


def test_prepare_lora_rows_uses_explicit_range_partitions_without_session_mutation() -> None:
    session = spark()
    coalesce_key = "spark.sql.adaptive.coalescePartitions.enabled"
    previous = session.conf.get(coalesce_key)
    session.conf.set(coalesce_key, "true")
    root = pathlib.Path(tempfile.mkdtemp()).resolve()
    source, source_digest = _source_file(root)
    panel = _panel(root)
    panel_digest = directory_digest(panel.path, suffixes=(".parquet",))
    observed: list[tuple[int, str]] = []
    dataframe_type = type(session.range(1))
    writer_type = type(session.range(1).write)
    original_parquet = writer_type.parquet

    def observed_parquet(writer, *args, **kwargs):
        observed.append(
            (
                writer._df.rdd.getNumPartitions(),
                writer._spark.conf.get(coalesce_key),
            )
        )
        return original_parquet(writer, *args, **kwargs)

    try:
        with (
            mock.patch.object(
                dataframe_type,
                "toLocalIterator",
                side_effect=AssertionError("row preparation used Spark task-result transport"),
            ),
            mock.patch.object(writer_type, "parquet", observed_parquet),
            mock.patch.object(subject, "M5_SALES_TRAIN_EVALUATION_SHA256", source_digest),
            mock.patch.object(subject, "M5_PANEL_CONTENT_SHA256", panel_digest),
            mock.patch.object(subject, "M5_LORA_SOURCE_PARTITIONS", 4),
        ):
            artifact = prepare_lora_training_rows(
                panel,
                CUTOFF,
                spark=session,
                out_dir=root / "training-rows",
                config=TEST_CONFIG,
                source_data_path=source,
                source_data_sha256=source_digest,
                panel_content_sha256=panel_digest,
            )
        assert artifact.n_series == 2 and artifact.n_rows == 6
        assert observed == [(4, "true")], observed
        assert session.conf.get(coalesce_key) == "true", session.conf.get(coalesce_key)
        assert not (pathlib.Path(artifact.path) / ".canonical-source").exists()
    finally:
        session.conf.set(coalesce_key, previous)


def test_timesfm_transformers_backend_forecast_is_offline_and_uses_full_channel_zero() -> None:
    root = pathlib.Path(tempfile.mkdtemp()).resolve()
    base, config_digest, weights_digest = _base_checkpoint(root)
    runtime_args, runtime_packages = _runtime_lock(root)
    observed: dict[str, object] = {}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            observed["base_path"] = path
            observed["base_load"] = kwargs
            return cls()

        def to(self, **kwargs):
            observed["to"] = kwargs
            return self

        def eval(self):
            observed["eval"] = True
            return self

        def __call__(self, **kwargs):
            observed["call"] = kwargs
            full = torch.zeros((len(kwargs["past_values"]), 128, 10), dtype=torch.float32)
            full[..., 0] = 7.0
            full[..., 5] = 11.0
            return SimpleNamespace(mean_predictions=full[..., 5], full_predictions=full)

    original_import = importlib.import_module

    def provider_import(name: str):
        if name == "transformers":
            return SimpleNamespace(TimesFm2_5ModelForPrediction=FakeModel)
        return original_import(name)

    with _base_identity(config_digest, weights_digest), mock.patch.object(
        subject,
        "_installed_package_versions",
        return_value=runtime_packages,
    ):
        locked = subject.validate_runtime_lock(**runtime_args)
        assert locked.projection_sha256 == runtime_args["runtime_lock_sha256"]
        provider = TimesFMTransformersBackend(
            checkpoint_dir=base,
            model_revision=subject.TRANSFORMERS_REVISION,
            context_length=32,
            device="cpu",
            **runtime_args,
        )
        with mock.patch.object(
            subject.importlib,
            "import_module",
            side_effect=provider_import,
        ):
            median, full = provider.forecast([[1.0] * 32, [2.0] * 32], 28)
    assert observed["base_load"] == {
        "local_files_only": True,
        "dtype": torch.float32,
    }
    assert observed["call"]["force_flip_invariance"] is True
    assert observed["call"]["truncate_negative"] is False
    assert observed["call"]["forecast_context_len"] == 32
    assert median.tolist() == [[11.0] * 28, [11.0] * 28]
    assert full[..., 0].tolist() == [[7.0] * 28, [7.0] * 28]

    refused = pathlib.Path(tempfile.mkdtemp()).resolve() / "forecast"
    refused.mkdir()
    try:
        timesfm_boundary.fit_predict(
            _panel(pathlib.Path(tempfile.mkdtemp()).resolve()),
            CUTOFF,
            2,
            spark=spark(),
            out_dir=refused,
            context_days=32,
            batch_size=2,
            backend=provider,
        )
    except ValueError as exc:
        assert "requires horizon=28" in str(exc)
        assert list(refused.iterdir()) == []
    else:
        raise AssertionError("governed Transformers provider accepted a diagnostic horizon")

    runtime_path = pathlib.Path(runtime_args["runtime_lock_path"])
    runtime_path.write_text(runtime_path.read_text() + "\n")
    with _base_identity(config_digest, weights_digest), mock.patch.object(
        subject,
        "_installed_package_versions",
        return_value=runtime_packages,
    ):
        try:
            TimesFMTransformersBackend(
                checkpoint_dir=base,
                model_revision=subject.TRANSFORMERS_REVISION,
                context_length=32,
                device="cpu",
                **runtime_args,
            )
        except ValueError as exc:
            assert "runtime-lock source hash mismatch" in str(exc)
        else:
            raise AssertionError("TimesFM accepted a mutated worker runtime-lock source")


def test_timesfm_fit_predict_constructs_transformers_from_inert_provider_config() -> None:
    root = pathlib.Path(tempfile.mkdtemp()).resolve()
    base, _config_digest, _weights_digest = _base_checkpoint(root)
    runtime_args, _runtime_packages = _runtime_lock(root)
    observed: dict[str, object] = {}

    class BoundaryBackend:
        governed_provider = True

        def forecast(self, inputs, horizon):
            medians = [[11.0] * horizon for _ in inputs]
            full = [[[7.0, *([0.0] * 4), 11.0, *([0.0] * 4)] for _ in range(horizon)]
                    for _ in inputs]
            return medians, full

    def construct(**kwargs):
        observed.update(kwargs)
        return BoundaryBackend()

    output = root / "forecast"
    output.mkdir()
    with mock.patch.object(
        timesfm_boundary,
        "TimesFMTransformersBackend",
        side_effect=construct,
    ):
        forecast = timesfm_boundary.fit_predict(
            _forecast_panel(root),
            CUTOFF,
            28,
            spark=spark(),
            out_dir=output,
            model_id=subject.TRANSFORMERS_MODEL_ID,
            model_revision=subject.TRANSFORMERS_REVISION,
            provider_family="transformers",
            provider_checkpoint_dir=base,
            provider_device="cpu",
            **runtime_args,
        )
    assert observed == {
        "checkpoint_dir": base,
        "model_revision": subject.TRANSFORMERS_REVISION,
        "context_length": 1024,
        "device": "cpu",
        "adapter_dir": None,
        "adapter_sha256": None,
        "forecast_origin": None,
        "recorded_runtime_lock": None,
        "recorded_adapter_dir": None,
        **runtime_args,
    }
    assert forecast.adapter == timesfm_boundary.NAME
    assert {row["yhat"] for row in pq.read_table(output).to_pylist()} == {7.0}


def test_isolated_run_serializes_transformers_and_lora_identity_as_inert_json() -> None:
    root = pathlib.Path(tempfile.mkdtemp()).resolve()
    panel = _forecast_panel(root)
    base = (root / "base-runtime").resolve()
    adapter = (root / "adapter-runtime").resolve()
    runtime_args, _runtime_packages = _runtime_lock(root)
    runtime_json_args = {
        **runtime_args,
        "runtime_lock_path": str(runtime_args["runtime_lock_path"]),
        "runtime_lock_record_path": list(runtime_args["runtime_lock_record_path"]),
    }
    output = root / "isolated-forecast"
    source_snapshot = {
        "root": str(root / "source" / ("a" * 64)),
        "manifest_sha256": "a" * 64,
    }
    profile = resources.SparkProfile(
        cores=1,
        driver_memory="1g",
        shuffle_partitions=1,
        worker_reuse=False,
        series_per_partition=1,
        python_worker_budget_gb=1.0,
    )
    observed: dict[str, object] = {}

    def completed(argv, **_kwargs):
        request_envelope = json.loads(pathlib.Path(argv[-2]).read_text())
        request = request_envelope["request"]
        observed.update(request)
        record = Forecast(
            adapter=timesfm_boundary.NAME,
            path=str(output),
            origin=CUTOFF,
            horizon=28,
            n_series=2,
            disposition={"emitted": 56},
            target_estimand="sales_given_availability",
        ).validate()
        pathlib.Path(argv[-1]).write_text(
            json.dumps(
                {
                    "schema": "autoforecast.isolated-response.v1",
                    "request_sha256": request_envelope["request_sha256"],
                    "forecast": record.as_record(),
                }
            )
        )
        return SimpleNamespace(returncode=0)

    with mock.patch.object(isolated, "_run_monitored_worker", side_effect=completed), \
            mock.patch.object(
                isolated,
                "validate_executable_source_snapshot",
                return_value=source_snapshot,
            ):
        returned = isolated.run(
            timesfm_boundary.NAME,
            panel,
            CUTOFF,
            28,
            source_snapshot=source_snapshot,
            worker_profile=profile,
            worker_python=sys.executable,
            timeout_seconds=4,
            heartbeat_seconds=1,
            worker_log_path=root / "isolated-worker.log",
            out_dir=str(output),
            model_id=subject.TRANSFORMERS_MODEL_ID,
            model_revision=subject.TRANSFORMERS_REVISION,
            provider_family="transformers",
            provider_checkpoint_dir=str(base),
            provider_device="mps",
            lora_adapter_dir=str(adapter),
            lora_adapter_sha256="a" * 64,
            **runtime_json_args,
        )
    assert returned.adapter == timesfm_boundary.NAME
    assert observed["source_snapshot"] == source_snapshot
    assert observed["kwargs"] == {
        "out_dir": str(output),
        "model_id": subject.TRANSFORMERS_MODEL_ID,
        "model_revision": subject.TRANSFORMERS_REVISION,
        "provider_family": "transformers",
        "provider_checkpoint_dir": str(base),
        "provider_device": "mps",
        "lora_adapter_dir": str(adapter),
        "lora_adapter_sha256": "a" * 64,
        **runtime_json_args,
    }


def test_timesfm_transformers_backend_loads_only_exact_hashed_final_lora() -> None:
    root = pathlib.Path(tempfile.mkdtemp()).resolve()
    rows = _prepare(root)
    base, config_digest, weights_digest = _base_checkpoint(root)
    runtime_args, runtime_packages = _runtime_lock(root)
    training_backend = RecordingTrainingBackend()
    with _base_identity(config_digest, weights_digest), mock.patch.object(
        subject,
        "_installed_package_versions",
        return_value=runtime_packages,
    ):
        adapter = train_lora(
            rows.path,
            rows.rows_sha256,
            rows.manifest_sha256,
            checkpoint_dir=base,
            model_revision=subject.TRANSFORMERS_REVISION,
            out_dir=root / "adapter",
            config=TEST_CONFIG,
            backend=training_backend,
            **runtime_args,
        )

    observed: dict[str, object] = {}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, _path, **_kwargs):
            return cls()

        def to(self, **_kwargs):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            full = torch.zeros((len(kwargs["past_values"]), 128, 10), dtype=torch.float32)
            full[..., 5] = 3.0
            return SimpleNamespace(mean_predictions=full[..., 5], full_predictions=full)

    class FakePeftModel:
        @classmethod
        def from_pretrained(cls, model, path, **kwargs):
            observed["adapter_path"] = path
            observed["adapter_load"] = kwargs
            return model

    original_import = importlib.import_module

    def provider_import(name: str):
        if name == "transformers":
            return SimpleNamespace(TimesFm2_5ModelForPrediction=FakeModel)
        if name == "peft":
            return SimpleNamespace(PeftModel=FakePeftModel)
        return original_import(name)

    with _base_identity(config_digest, weights_digest), mock.patch.object(
        subject,
        "_installed_package_versions",
        return_value=runtime_packages,
    ):
        provider = TimesFMTransformersBackend(
            checkpoint_dir=base,
            model_revision=subject.TRANSFORMERS_REVISION,
            context_length=32,
            device="cpu",
            adapter_dir=adapter.path,
            adapter_sha256=adapter.sha256,
            forecast_origin=CUTOFF,
            **runtime_args,
        )
        with mock.patch.object(
            subject.importlib,
            "import_module",
            side_effect=provider_import,
        ):
            provider.forecast([[1.0] * 32], 28)
    loaded_adapter_path = pathlib.Path(str(observed["adapter_path"]))
    assert loaded_adapter_path != pathlib.Path(adapter.path)
    assert loaded_adapter_path.name == "adapter"
    assert not loaded_adapter_path.exists()
    assert observed["adapter_load"] == {
        "is_trainable": False,
        "local_files_only": True,
    }
    provider.validate_origin(CUTOFF)
    try:
        provider.validate_origin("2020-02-08")
    except ValueError as exc:
        assert "may not backcast" in str(exc)
    else:
        raise AssertionError("a later LoRA adapter was allowed to backcast an earlier origin")

    with _base_identity(config_digest, weights_digest), mock.patch.object(
        subject,
        "_installed_package_versions",
        return_value=runtime_packages,
    ):
        backcast = TimesFMTransformersBackend(
            checkpoint_dir=base,
            model_revision=subject.TRANSFORMERS_REVISION,
            context_length=32,
            device="cpu",
            adapter_dir=adapter.path,
            adapter_sha256=adapter.sha256,
            forecast_origin="2020-02-08",
            **runtime_args,
        )
    with _base_identity(config_digest, weights_digest), mock.patch.object(
        subject,
        "_installed_package_versions",
        return_value=runtime_packages,
    ):
        try:
            backcast.forecast([[1.0] * 32], 28)
        except ValueError as exc:
            assert "training cutoff is after its forecast origin" in str(exc)
        else:
            raise AssertionError("a cutoff-40 LoRA adapter produced a cutoff-39 trajectory")

    (pathlib.Path(adapter.path) / "adapter_model.safetensors").write_bytes(b"mutated")
    with _base_identity(config_digest, weights_digest), mock.patch.object(
        subject,
        "_installed_package_versions",
        return_value=runtime_packages,
    ):
        try:
            TimesFMTransformersBackend(
                checkpoint_dir=base,
                model_revision=subject.TRANSFORMERS_REVISION,
                context_length=32,
                device="cpu",
                adapter_dir=adapter.path,
                adapter_sha256=adapter.sha256,
                forecast_origin=CUTOFF,
                **runtime_args,
            )
        except ValueError as exc:
            assert "adapter hash mismatch before snapshot" in str(exc)
        else:
            raise AssertionError("TimesFM loaded a LoRA adapter after its bytes changed")


def test_continue_lora_sha256_split_matches_registered_theta1913_counts() -> None:
    series_ids = [f"series_{index:05d}" for index in range(30_347)]
    table = pa.table({"series_id": series_ids})
    training, validation, record = subject._continuation_series_split(table, TEST_CONFIG)
    ranked = sorted(
        series_ids,
        key=lambda series_id: (
            hashlib.sha256(series_id.encode("utf-8")).digest(),
            series_id.encode("utf-8"),
        ),
    )
    assert validation == tuple(ranked[:3_034])
    assert training == tuple(ranked[3_034:])
    assert record["algorithm"] == "ascending_sha256(series_id_utf8)_then_series_id_utf8"
    assert record["validation_rounding"] == (
        "first_max(1,floor(0.1*n_series))_with_nonempty_training"
    )
    assert record["n_validation_series"] == 3_034
    assert record["n_train_series"] == 27_313


def test_continue_lora_batches_follow_registered_hash_rank_order() -> None:
    series_ids = [f"series_{index:02d}" for index in range(20)]
    rows = [
        {
            "epoch": epoch,
            "series_id": series_id,
            "context": [float(epoch)],
            "target": [float(index)],
        }
        for epoch in (2, 0, 1)
        for index, series_id in enumerate(reversed(series_ids))
    ]
    table = pa.Table.from_pylist(rows)
    training, validation, _record = subject._continuation_series_split(table, TEST_CONFIG)
    expected_ranked = sorted(
        series_ids,
        key=lambda series_id: (
            hashlib.sha256(series_id.encode("utf-8")).digest(),
            series_id.encode("utf-8"),
        ),
    )
    assert validation == tuple(expected_ranked[:2])
    assert training == tuple(expected_ranked[2:])

    batches = list(
        subject._selected_training_batches(
            table,
            selected_series=training,
            batch_size=7,
            source_epoch=0,
        )
    )
    observed_targets = [value for _contexts, targets in batches for value in targets]
    source_target = {
        row["series_id"]: row["target"]
        for row in rows
        if row["epoch"] == 0
    }
    assert observed_targets == [source_target[series_id] for series_id in training]


def test_lora_float_controls_and_training_schemas_are_exact() -> None:
    training_config_aliases = {
        "dropout": 0,
        "learning_rate": 1,
        "weight_decay": 0,
        "gradient_clip_norm": 1,
    }
    for field, alias in training_config_aliases.items():
        changed_config = dataclasses.replace(TEST_CONFIG, **{field: alias})
        try:
            subject._validate_training_config(changed_config)
        except ValueError:
            pass
        else:
            raise AssertionError(f"LoRA training config accepted integer {field}")

    continuation_config_aliases = {
        "restart_learning_rate": 1,
        "absolute_min_delta": 1,
        "relative_min_delta": 1,
    }
    for field, alias in continuation_config_aliases.items():
        changed_config = dataclasses.replace(
            TEST_CONTINUATION_CONFIG,
            **{field: alias},
        )
        try:
            subject._validate_continuation_config(changed_config)
        except ValueError:
            pass
        else:
            raise AssertionError(f"LoRA continuation config accepted integer {field}")

    root = pathlib.Path(tempfile.mkdtemp()).resolve()
    (
        rows,
        base,
        config_digest,
        weights_digest,
        runtime_args,
        runtime_packages,
        source,
        source_model_sha256,
    ) = _trained_source_adapter(root)
    backend = RecordingContinuationBackend(
        source_tree_sha256=source.sha256,
        source_model_sha256=source_model_sha256,
        continuation_config=TEST_CONTINUATION_CONFIG,
        validation_losses=(10.0, 9.0, 8.9995, 8.9995),
    )
    artifact = _continue_test_adapter(
        rows=rows,
        base=base,
        config_digest=config_digest,
        weights_digest=weights_digest,
        runtime_args=runtime_args,
        runtime_packages=runtime_packages,
        source=source,
        source_model_sha256=source_model_sha256,
        out_dir=root / "continued",
        continuation_config=TEST_CONTINUATION_CONFIG,
        backend=backend,
    )
    manifest = json.loads(pathlib.Path(artifact.manifest_path).read_text())
    mutations = {
        "baseline False": lambda value: value["stopping"].__setitem__(
            "baseline_epoch", False
        ),
        "baseline float": lambda value: value["stopping"].__setitem__(
            "baseline_epoch", 0.0
        ),
        "additional epoch bool": lambda value: value["stopping"][
            "validation_trace"
        ][1].__setitem__("additional_epoch", True),
        "additional epoch float": lambda value: value["stopping"][
            "validation_trace"
        ][2].__setitem__("additional_epoch", 2.0),
        "source epoch False": lambda value: value["stopping"][
            "validation_trace"
        ][1].__setitem__("source_row_epoch", False),
        "source epoch True": lambda value: value["stopping"][
            "validation_trace"
        ][2].__setitem__("source_row_epoch", True),
        "source epoch float": lambda value: value["stopping"][
            "validation_trace"
        ][3].__setitem__("source_row_epoch", 2.0),
        "validation pool aliases": lambda value: value["split"].__setitem__(
            "validation_source_epochs", [False, True, 2.0]
        ),
        "fixed boolean alias": lambda value: value["training"].__setitem__(
            "base_weights_frozen", 1
        ),
        "external normalization changed": lambda value: value["training"].__setitem__(
            "external_normalization", "arbitrary"
        ),
        "continuation dropout integer": lambda value: value["training"].__setitem__(
            "dropout", 0
        ),
        "continuation weight decay integer": lambda value: value["training"].__setitem__(
            "weight_decay", 0
        ),
        "continuation gradient clip integer": lambda value: value["training"].__setitem__(
            "gradient_clip_norm", 1
        ),
        "continuation restart learning rate integer": lambda value: value[
            "training"
        ].__setitem__("restart_learning_rate", 1),
        "absolute min delta integer": lambda value: value["stopping"].__setitem__(
            "absolute_min_delta", 1
        ),
        "relative min delta integer": lambda value: value["stopping"].__setitem__(
            "relative_min_delta", 1
        ),
        "patience float": lambda value: value["stopping"].__setitem__(
            "patience", float(value["stopping"]["patience"])
        ),
        "summary loss integer": lambda value: value["stopping"].__setitem__(
            "best_validation_loss", int(value["stopping"]["best_validation_loss"])
        ),
        "trace training loss integer": lambda value: value["stopping"][
            "validation_trace"
        ][1].__setitem__("training_loss", 1),
        "trace validation loss integer": lambda value: value["stopping"][
            "validation_trace"
        ][1].__setitem__("validation_loss", 1),
        "publication aliases": lambda value: value.__setitem__(
            "publication",
            {
                "saved_after_best_adapter_restoration": 1,
                "intermediate_checkpoints": False,
                "accuracy_evaluation_runs": 0.0,
                "validation_runs": 4.0,
            },
        ),
    }
    with _base_identity(config_digest, weights_digest):
        for label, mutate in mutations.items():
            changed = copy.deepcopy(manifest)
            mutate(changed)
            try:
                subject._validate_continuation_lora_manifest(changed)
            except ValueError:
                pass
            else:
                raise AssertionError(f"continuation manifest accepted {label}")

        v1_manifest = json.loads(pathlib.Path(source.manifest_path).read_text())
        v1_mutations = {
            "extra training field": lambda value: value["training"].__setitem__(
                "extra", "forbidden"
            ),
            "missing training field": lambda value: value["training"].pop(
                "external_normalization"
            ),
            "training integer float": lambda value: value["training"].__setitem__(
                "epochs", float(value["training"]["epochs"])
            ),
            "training seed bool": lambda value: value["training"].__setitem__(
                "seed", True
            ),
            "row count float": lambda value: value["training_rows"].__setitem__(
                "n_rows", float(value["training_rows"]["n_rows"])
            ),
            "runtime count float": lambda value: value["runtime"][
                "runtime_lock"
            ].__setitem__(
                "n_distributions",
                float(value["runtime"]["runtime_lock"]["n_distributions"]),
            ),
            "parameter count float": lambda value: value["parameters"].__setitem__(
                "trainable", float(value["parameters"]["trainable"])
            ),
            "publication aliases": lambda value: value.__setitem__(
                "publication",
                {
                    "saved_after_all_epochs": 1,
                    "intermediate_checkpoints": False,
                    "evaluation_runs": 0.0,
                },
            ),
        }
        for field, alias in training_config_aliases.items():
            v1_mutations[f"training {field} integer"] = (
                lambda value, field=field, alias=alias: value["training"].__setitem__(
                    field, alias
                )
            )
        fixed_training_semantics = {
            "precision": "float16",
            "external_normalization": "arbitrary",
            "objective": "arbitrary",
            "optimizer": "arbitrary",
            "scheduler": "arbitrary",
            "target_modules": "arbitrary",
            "bias": "all",
            "base_weights_frozen": False,
            "force_flip_invariance": False,
            "evaluation_during_training": True,
            "early_stopping": True,
            "best_epoch_selection": True,
            "checkpoint_publication": "arbitrary",
        }
        for field, changed_value in fixed_training_semantics.items():
            v1_mutations[f"changed fixed {field}"] = (
                lambda value, field=field, changed_value=changed_value: value[
                    "training"
                ].__setitem__(field, changed_value)
            )
        for label, mutate in v1_mutations.items():
            changed = copy.deepcopy(v1_manifest)
            mutate(changed)
            try:
                subject._validate_v1_lora_manifest(changed)
            except ValueError:
                pass
            else:
                raise AssertionError(f"v1 manifest accepted {label}")


def test_continuation_runtime_and_nonfinite_loss_guards_are_discriminating() -> None:
    runtime_lock = subject.RuntimeLockIdentity(
        source_path="/tmp/runtime.json",
        source_sha256="0" * 64,
        record_path=("registration", "runtime"),
        projection_sha256="1" * 64,
        python="3.12.11",
        package_versions=(("peft", "0.20.0"), ("torch", "2.13.0"), ("transformers", "5.16.1")),
    )
    identity = {
        "runtime": {
            "torch": "wrong",
            "transformers": "5.16.1",
            "peft": "0.20.0",
            "device": "cpu",
            "precision": "float32",
        },
        "parameters": {"trainable": 4, "total": 100},
        "source_adapter": {
            "tree_sha256": "2" * 64,
            "adapter_model_sha256": "3" * 64,
        },
        "optimizer": {
            "optimizer": "torch.optim.AdamW",
            "scheduler": "CosineAnnealingLR_per_batch",
            "optimizer_state_restored": False,
            "scheduler_state_restored": False,
            "restart_learning_rate": 1e-5,
        },
    }
    try:
        subject._validate_continuation_backend_identity(
            identity,
            source_identity={
                "tree_sha256": "2" * 64,
                "adapter_model_sha256": "3" * 64,
            },
            expected_parameters={"trainable": 4, "total": 100},
            runtime_lock=runtime_lock,
            source_config=TEST_CONFIG,
            continuation_config=TEST_CONTINUATION_CONFIG,
        )
    except ValueError as exc:
        assert "runtime identity" in str(exc)
    else:
        raise AssertionError("continuation accepted a backend from another locked runtime")

    try:
        subject._mean_backend_loss(
            [([[1.0]], [[1.0]])],
            lambda _contexts, _targets: float("nan"),
            label="validation",
        )
    except RuntimeError as exc:
        assert "non-finite validation loss" in str(exc)
    else:
        raise AssertionError("continuation treated non-finite validation as no improvement")


def test_continue_lora_cycles_fixed_pool_restores_best_and_is_inference_safe() -> None:
    root = pathlib.Path(tempfile.mkdtemp()).resolve()
    (
        rows,
        base,
        config_digest,
        weights_digest,
        runtime_args,
        runtime_packages,
        source,
        source_model_sha256,
    ) = _trained_source_adapter(root)
    source_path = pathlib.Path(source.path)
    source_manifest_before = (source_path / subject.LORA_MANIFEST).read_bytes()
    source_model_before = (source_path / "adapter_model.safetensors").read_bytes()
    backend = RecordingContinuationBackend(
        source_tree_sha256=source.sha256,
        source_model_sha256=source_model_sha256,
        continuation_config=TEST_CONTINUATION_CONFIG,
        validation_losses=(10.0, 9.0, 8.9995, 8.9995),
    )
    artifact = _continue_test_adapter(
        rows=rows,
        base=base,
        config_digest=config_digest,
        weights_digest=weights_digest,
        runtime_args=runtime_args,
        runtime_packages=runtime_packages,
        source=source,
        source_model_sha256=source_model_sha256,
        out_dir=root / "continued-adapter",
        continuation_config=TEST_CONTINUATION_CONFIG,
        backend=backend,
    )

    manifest = json.loads(pathlib.Path(artifact.manifest_path).read_text())
    assert manifest["schema"] == subject.LORA_CONTINUATION_ARTIFACT_SCHEMA
    assert manifest["source_adapter"] == {
        "path": source.path,
        "tree_sha256": source.sha256,
        "adapter_model_file": "adapter_model.safetensors",
        "adapter_model_sha256": source_model_sha256,
        "manifest_path": source.manifest_path,
        "manifest_sha256": _sha256(pathlib.Path(source.manifest_path)),
        "manifest_schema": subject.LORA_ARTIFACT_SCHEMA,
    }
    assert manifest["stopping"] == {
        "metric": "mean_native_loss_on_fixed_validation_rows",
        "baseline_epoch": 0,
        "baseline_validation_loss": 10.0,
        "best_epoch": 1,
        "best_validation_loss": 9.0,
        "last_training_loss": 3.0,
        "last_validation_loss": 8.9995,
        "actual_additional_epochs": 3,
        "patience": 2,
        "waited_at_stop": 2,
        "stop_reason": "early_stopping_patience",
        "improvement_rule": (
            "validation_loss < best_loss - "
            "max(absolute_min_delta,relative_min_delta*abs(best_loss))"
        ),
        "absolute_min_delta": 1e-6,
        "relative_min_delta": 1e-4,
        "best_adapter_restored": True,
        "validation_trace": [
            {
                "additional_epoch": 0,
                "source_row_epoch": None,
                "training_loss": None,
                "validation_loss": 10.0,
            },
            {
                "additional_epoch": 1,
                "source_row_epoch": 0,
                "training_loss": 1.0,
                "validation_loss": 9.0,
            },
            {
                "additional_epoch": 2,
                "source_row_epoch": 1,
                "training_loss": 2.0,
                "validation_loss": 8.9995,
            },
            {
                "additional_epoch": 3,
                "source_row_epoch": 2,
                "training_loss": 3.0,
                "validation_loss": 8.9995,
            },
        ],
    }
    assert manifest["publication"] == {
        "saved_after_best_adapter_restoration": True,
        "intermediate_checkpoints": 0,
        "accuracy_evaluation_runs": 0,
        "validation_runs": 4,
    }
    assert manifest["parameters"] == {"trainable": 4, "total": 100}
    assert manifest["training"]["max_additional_epochs"] == 5
    assert manifest["training"]["restart_learning_rate"] == 1e-5
    assert manifest["training"]["optimizer_state_restored"] is False
    assert manifest["training"]["scheduler_state_restored"] is False
    assert backend.snapshots == [0, 1]
    assert backend.restored == [1]
    assert backend.save_at == [1]
    assert (
        pathlib.Path(artifact.path) / "adapter_model.safetensors"
    ).read_bytes() == b"continued-state=1\n"

    table_rows = pq.read_table(
        pathlib.Path(rows.path) / subject.TRAINING_ROWS_FILE
    ).to_pylist()
    series_ids = sorted({row["series_id"] for row in table_rows})
    validation_series = min(
        series_ids,
        key=lambda series_id: hashlib.sha256(series_id.encode("utf-8")).digest(),
    )
    train_series = next(series_id for series_id in series_ids if series_id != validation_series)
    expected_train_contexts = [
        row["context"]
        for row in table_rows
        if row["series_id"] == train_series and row["epoch"] in (0, 1, 2)
    ]
    assert [contexts[0] for contexts, _targets in backend.train_calls] == (
        expected_train_contexts
    )
    expected_validation_contexts = [
        row["context"] for row in table_rows if row["series_id"] == validation_series
    ]
    assert len(backend.validation_calls) == 8
    for start in range(0, len(backend.validation_calls), 2):
        one_run = backend.validation_calls[start : start + 2]
        assert [context for contexts, _targets in one_run for context in contexts] == (
            expected_validation_contexts
        )
    assert manifest["split"]["series_disjoint"] is True
    assert manifest["split"]["n_train_series"] == 1
    assert manifest["split"]["n_validation_series"] == 1
    assert manifest["split"]["validation_source_epochs"] == [0, 1, 2]

    with _base_identity(config_digest, weights_digest), mock.patch.object(
        subject,
        "_installed_package_versions",
        return_value=runtime_packages,
    ):
        authenticated = subject.validate_lora_continuation_adapter(
            artifact.path,
            artifact.sha256,
            training_rows_dir=rows.path,
            training_rows_sha256=rows.rows_sha256,
            training_rows_manifest_sha256=rows.manifest_sha256,
            source_adapter_dir=source.path,
            source_adapter_sha256=source.sha256,
            source_adapter_model_sha256=source_model_sha256,
            checkpoint_dir=base,
            model_revision=subject.TRANSFORMERS_REVISION,
            source_config=TEST_CONFIG,
            continuation_config=TEST_CONTINUATION_CONFIG,
            expected_panel_content_sha256=rows.panel_content_sha256,
            expected_training_cutoff=CUTOFF,
            forecast_origin=CUTOFF,
            **runtime_args,
        )
    assert authenticated == artifact

    manifest_path = pathlib.Path(artifact.manifest_path)
    clean_manifest_bytes = manifest_path.read_bytes()
    continued_after_patience = json.loads(clean_manifest_bytes)
    continued_after_patience["stopping"]["validation_trace"].extend(
        [
            {
                "additional_epoch": 4,
                "source_row_epoch": 0,
                "training_loss": 4.0,
                "validation_loss": 8.0,
            },
            {
                "additional_epoch": 5,
                "source_row_epoch": 1,
                "training_loss": 5.0,
                "validation_loss": 8.0,
            },
        ]
    )
    continued_after_patience["stopping"].update(
        {
            "best_epoch": 4,
            "best_validation_loss": 8.0,
            "last_training_loss": 5.0,
            "last_validation_loss": 8.0,
            "actual_additional_epochs": 5,
            "waited_at_stop": 1,
            "stop_reason": "maximum_additional_epochs",
        }
    )
    continued_after_patience["publication"]["validation_runs"] = 6
    manifest_path.write_text(
        json.dumps(continued_after_patience, indent=2, sort_keys=True) + "\n"
    )
    with _base_identity(config_digest, weights_digest):
        try:
            subject._validate_lora_manifest(pathlib.Path(artifact.path))
        except ValueError as exc:
            assert "continues after patience" in str(exc)
        else:
            raise AssertionError("continuation manifest continued optimizing after patience")
    manifest_path.write_bytes(clean_manifest_bytes)

    with _base_identity(config_digest, weights_digest), mock.patch.object(
        subject,
        "_installed_package_versions",
        return_value=runtime_packages,
    ):
        try:
            subject.validate_lora_continuation_adapter(
                artifact.path,
                artifact.sha256,
                training_rows_dir=rows.path,
                training_rows_sha256=rows.rows_sha256,
                training_rows_manifest_sha256=rows.manifest_sha256,
                source_adapter_dir=source.path,
                source_adapter_sha256=source.sha256,
                source_adapter_model_sha256="0" * 64,
                checkpoint_dir=base,
                model_revision=subject.TRANSFORMERS_REVISION,
                source_config=TEST_CONFIG,
                continuation_config=TEST_CONTINUATION_CONFIG,
                expected_panel_content_sha256=rows.panel_content_sha256,
                expected_training_cutoff=CUTOFF,
                forecast_origin=CUTOFF,
                **runtime_args,
            )
        except ValueError as exc:
            assert "adapter_model.safetensors hash differs" in str(exc)
        else:
            raise AssertionError("continuation accepted another source model payload")

    with _base_identity(config_digest, weights_digest), mock.patch.object(
        subject,
        "_installed_package_versions",
        return_value=runtime_packages,
    ):
        try:
            subject.validate_lora_adapter(
                artifact.path,
                artifact.sha256,
                config=TEST_CONFIG,
                expected_training_rows_sha256=rows.rows_sha256,
                expected_training_rows_manifest_sha256=rows.manifest_sha256,
                expected_panel_content_sha256=rows.panel_content_sha256,
                expected_training_cutoff=CUTOFF,
                forecast_origin=CUTOFF,
                **runtime_args,
            )
        except ValueError as exc:
            assert "v1 LoRA validation cannot authenticate" in str(exc)
        else:
            raise AssertionError("the v1 validator accepted a continuation manifest")

    observed: dict[str, object] = {}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, _path, **_kwargs):
            return cls()

        def to(self, **_kwargs):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            full = torch.zeros((len(kwargs["past_values"]), 128, 10), dtype=torch.float32)
            full[..., 5] = 3.0
            return SimpleNamespace(mean_predictions=full[..., 5], full_predictions=full)

    class FakePeftModel:
        @classmethod
        def from_pretrained(cls, model, path, **kwargs):
            observed["adapter_path"] = path
            observed["adapter_load"] = kwargs
            return model

    original_import = importlib.import_module

    def provider_import(name: str):
        if name == "transformers":
            return SimpleNamespace(TimesFm2_5ModelForPrediction=FakeModel)
        if name == "peft":
            return SimpleNamespace(PeftModel=FakePeftModel)
        return original_import(name)

    with _base_identity(config_digest, weights_digest), mock.patch.object(
        subject,
        "_installed_package_versions",
        return_value=runtime_packages,
    ):
        provider = TimesFMTransformersBackend(
            checkpoint_dir=base,
            model_revision=subject.TRANSFORMERS_REVISION,
            context_length=32,
            device="cpu",
            adapter_dir=artifact.path,
            adapter_sha256=artifact.sha256,
            forecast_origin=CUTOFF,
            **runtime_args,
        )
        with mock.patch.object(
            subject.importlib,
            "import_module",
            side_effect=provider_import,
        ):
            provider.forecast([[1.0] * 32], 28)
    loaded_adapter_path = pathlib.Path(str(observed["adapter_path"]))
    assert loaded_adapter_path != pathlib.Path(artifact.path)
    assert loaded_adapter_path.name == "adapter"
    assert not loaded_adapter_path.exists()
    assert observed["adapter_load"] == {
        "is_trainable": False,
        "local_files_only": True,
    }

    epoch_zero_config = LoRAContinuationConfig(
        max_additional_epochs=2,
        patience=1,
        restart_learning_rate=1e-5,
    )
    epoch_zero_backend = RecordingContinuationBackend(
        source_tree_sha256=source.sha256,
        source_model_sha256=source_model_sha256,
        continuation_config=epoch_zero_config,
        validation_losses=(10.0, 9.9995),
    )
    epoch_zero = _continue_test_adapter(
        rows=rows,
        base=base,
        config_digest=config_digest,
        weights_digest=weights_digest,
        runtime_args=runtime_args,
        runtime_packages=runtime_packages,
        source=source,
        source_model_sha256=source_model_sha256,
        out_dir=root / "epoch-zero-best",
        continuation_config=epoch_zero_config,
        backend=epoch_zero_backend,
    )
    epoch_zero_manifest = json.loads(pathlib.Path(epoch_zero.manifest_path).read_text())
    assert epoch_zero_manifest["stopping"]["best_epoch"] == 0
    assert epoch_zero_manifest["stopping"]["actual_additional_epochs"] == 1
    assert epoch_zero_backend.restored == [0]
    assert epoch_zero_backend.save_at == [0]

    assert (source_path / subject.LORA_MANIFEST).read_bytes() == source_manifest_before
    assert (source_path / "adapter_model.safetensors").read_bytes() == source_model_before
    assert subject._tree_sha256(source_path) == source.sha256
    assert subject.lora_adapter_tree_sha256(source_path) == source.sha256
    assert _sha256(source_path / "adapter_model.safetensors") == source_model_sha256

    class FailingSaveBackend(RecordingContinuationBackend):
        def _save_adapter(self, path: pathlib.Path) -> None:
            super()._save_adapter(path)
            raise RuntimeError("injected final-save failure")

    failing_save_config = LoRAContinuationConfig(
        max_additional_epochs=1,
        patience=1,
        restart_learning_rate=1e-5,
    )
    failing_save_backend = FailingSaveBackend(
        source_tree_sha256=source.sha256,
        source_model_sha256=source_model_sha256,
        continuation_config=failing_save_config,
        validation_losses=(10.0, 11.0),
    )
    failing_save_target = root / "failed-final-save"
    failing_save_target.mkdir()
    try:
        _continue_test_adapter(
            rows=rows,
            base=base,
            config_digest=config_digest,
            weights_digest=weights_digest,
            runtime_args=runtime_args,
            runtime_packages=runtime_packages,
            source=source,
            source_model_sha256=source_model_sha256,
            out_dir=failing_save_target,
            continuation_config=failing_save_config,
            backend=failing_save_backend,
        )
    except RuntimeError as exc:
        assert "injected final-save failure" in str(exc)
    else:
        raise AssertionError("continuation published after its final save failed")
    assert list(failing_save_target.iterdir()) == []

    malformed_manifest_path = pathlib.Path(artifact.manifest_path)
    malformed_manifest = json.loads(malformed_manifest_path.read_text())
    malformed_manifest["stopping"]["best_adapter_restored"] = False
    malformed_manifest_path.write_text(
        json.dumps(malformed_manifest, indent=2, sort_keys=True) + "\n"
    )
    malformed_digest = subject._tree_sha256(pathlib.Path(artifact.path))
    with _base_identity(config_digest, weights_digest), mock.patch.object(
        subject,
        "_installed_package_versions",
        return_value=runtime_packages,
    ):
        malformed_provider = TimesFMTransformersBackend(
            checkpoint_dir=base,
            model_revision=subject.TRANSFORMERS_REVISION,
            context_length=32,
            device="cpu",
            adapter_dir=artifact.path,
            adapter_sha256=malformed_digest,
            forecast_origin=CUTOFF,
            **runtime_args,
        )
    with _base_identity(config_digest, weights_digest), mock.patch.object(
        subject,
        "_installed_package_versions",
        return_value=runtime_packages,
    ):
        try:
            malformed_provider.forecast([[1.0] * 32], 28)
        except ValueError as exc:
            assert "early-stopping semantics" in str(exc)
        else:
            raise AssertionError("inference accepted malformed continuation evidence")

    mutation_config = LoRAContinuationConfig(
        max_additional_epochs=2,
        patience=1,
        restart_learning_rate=1e-5,
    )
    mutating_backend = RecordingContinuationBackend(
        source_tree_sha256=source.sha256,
        source_model_sha256=source_model_sha256,
        continuation_config=mutation_config,
        validation_losses=(10.0, 11.0),
        source_model_path=source_path / "adapter_model.safetensors",
    )
    snapshotted_target = root / "source-mutated-after-snapshot"
    snapshotted_target.mkdir()
    snapshotted = _continue_test_adapter(
        rows=rows,
        base=base,
        config_digest=config_digest,
        weights_digest=weights_digest,
        runtime_args=runtime_args,
        runtime_packages=runtime_packages,
        source=source,
        source_model_sha256=source_model_sha256,
        out_dir=snapshotted_target,
        continuation_config=mutation_config,
        backend=mutating_backend,
    )
    assert mutating_backend.save_at == [0]
    assert pathlib.Path(snapshotted.path) == snapshotted_target
    assert (source_path / "adapter_model.safetensors").read_bytes() == (
        b"mutated warm-start source"
    )


def test_continue_lora_uses_private_rows_snapshot_under_aba() -> None:
    root = pathlib.Path(tempfile.mkdtemp()).resolve()
    (
        rows,
        base,
        config_digest,
        weights_digest,
        runtime_args,
        runtime_packages,
        source,
        source_model_sha256,
    ) = _trained_source_adapter(root)
    payload = pathlib.Path(rows.path) / subject.TRAINING_ROWS_FILE
    payload_a = payload.read_bytes()
    table_a = pq.read_table(payload)
    rows_b = table_a.to_pylist()
    for row in rows_b:
        row["context"] = [222.0] * TEST_CONFIG.context_length
        row["target"] = [222.0] * TEST_CONFIG.horizon_length
    payload_b_path = root / "training-rows-b.parquet"
    pq.write_table(pa.Table.from_pylist(rows_b, schema=table_a.schema), payload_b_path)
    payload_b = payload_b_path.read_bytes()
    real_read_table = subject.pq.read_table
    swaps = 0

    def aba_read_table(path, *args, **kwargs):
        nonlocal swaps
        candidate = pathlib.Path(path)
        if candidate == payload:
            swaps += 1
            payload.write_bytes(payload_b)
            try:
                return real_read_table(candidate, *args, **kwargs)
            finally:
                payload.write_bytes(payload_a)
        return real_read_table(path, *args, **kwargs)

    continuation = LoRAContinuationConfig(
        max_additional_epochs=1,
        patience=1,
        restart_learning_rate=1e-5,
    )
    backend = RecordingContinuationBackend(
        source_tree_sha256=source.sha256,
        source_model_sha256=source_model_sha256,
        continuation_config=continuation,
        validation_losses=(10.0, 11.0),
    )
    with mock.patch.object(subject.pq, "read_table", side_effect=aba_read_table):
        artifact = _continue_test_adapter(
            rows=rows,
            base=base,
            config_digest=config_digest,
            weights_digest=weights_digest,
            runtime_args=runtime_args,
            runtime_packages=runtime_packages,
            source=source,
            source_model_sha256=source_model_sha256,
            out_dir=root / "continued-from-private-rows",
            continuation_config=continuation,
            backend=backend,
        )
    consumed = [
        value
        for calls in (backend.train_calls, backend.validation_calls)
        for contexts, targets in calls
        for row in (*contexts, *targets)
        for value in row
    ]
    assert swaps >= 2
    assert 222.0 not in consumed
    assert payload.read_bytes() == payload_a
    assert pathlib.Path(artifact.path).is_dir()

    private_swaps = 0

    def private_aba_read_table(path, *args, **kwargs):
        nonlocal private_swaps
        candidate = pathlib.Path(path)
        if candidate.name == subject.TRAINING_ROWS_FILE and candidate != payload:
            private_swaps += 1
            private_a = candidate.read_bytes()
            candidate.chmod(0o600)
            candidate.write_bytes(payload_b)
            try:
                return real_read_table(candidate, *args, **kwargs)
            finally:
                candidate.write_bytes(private_a)
                candidate.chmod(0o400)
        return real_read_table(path, *args, **kwargs)

    rejected_backend = RecordingContinuationBackend(
        source_tree_sha256=source.sha256,
        source_model_sha256=source_model_sha256,
        continuation_config=continuation,
        validation_losses=(10.0, 11.0),
    )
    rejected_target = root / "private-row-aba-rejected"
    with mock.patch.object(
        subject.pq,
        "read_table",
        side_effect=private_aba_read_table,
    ):
        try:
            _continue_test_adapter(
                rows=rows,
                base=base,
                config_digest=config_digest,
                weights_digest=weights_digest,
                runtime_args=runtime_args,
                runtime_packages=runtime_packages,
                source=source,
                source_model_sha256=source_model_sha256,
                out_dir=rejected_target,
                continuation_config=continuation,
                backend=rejected_backend,
            )
        except RuntimeError as exc:
            assert "private TimesFM continuation inputs changed" in str(exc)
        else:
            raise AssertionError("continuation accepted an ABA-swapped private row snapshot")
    assert private_swaps >= 1
    assert not rejected_target.exists()


def test_peft_continuation_uses_private_base_and_adapter_snapshot_under_aba() -> None:
    root = pathlib.Path(tempfile.mkdtemp()).resolve()
    (
        rows,
        base,
        config_digest,
        weights_digest,
        runtime_args,
        runtime_packages,
        source,
        source_model_sha256,
    ) = _trained_source_adapter(root)
    source_path = pathlib.Path(source.path)
    base_model = base / "model.safetensors"
    source_model = source_path / "adapter_model.safetensors"
    base_a = base_model.read_bytes()
    adapter_a = source_model.read_bytes()
    continuation = LoRAContinuationConfig(
        max_additional_epochs=1,
        patience=1,
        restart_learning_rate=1e-5,
    )
    backend = RecordingContinuationBackend(
        source_tree_sha256=source.sha256,
        source_model_sha256=source_model_sha256,
        continuation_config=continuation,
        validation_losses=(10.0, 11.0),
    )
    backend.runtime_versions = {
        package: runtime_packages[package]
        for package in ("torch", "transformers", "peft")
    }
    observed: dict[str, object] = {}

    def runtime_factory(
        checkpoint_path,
        _model_revision,
        adapter_path,
        *_args,
    ):
        base_model.write_bytes(b"base-B")
        source_model.write_bytes(b"adapter-B")
        try:
            observed["base_path"] = pathlib.Path(checkpoint_path)
            observed["adapter_path"] = pathlib.Path(adapter_path)
            observed["base_bytes"] = (
                pathlib.Path(checkpoint_path) / "model.safetensors"
            ).read_bytes()
            observed["adapter_bytes"] = (
                pathlib.Path(adapter_path) / "adapter_model.safetensors"
            ).read_bytes()
        finally:
            base_model.write_bytes(base_a)
            source_model.write_bytes(adapter_a)
        return backend

    with _base_identity(config_digest, weights_digest), mock.patch.object(
        subject,
        "_installed_package_versions",
        return_value=runtime_packages,
    ), mock.patch.object(
        subject,
        "_PeftContinuationRuntime",
        side_effect=runtime_factory,
    ):
        artifact = continue_lora(
            rows.path,
            rows.rows_sha256,
            rows.manifest_sha256,
            source_adapter_dir=source.path,
            source_adapter_sha256=source.sha256,
            source_adapter_model_sha256=source_model_sha256,
            checkpoint_dir=base,
            model_revision=subject.TRANSFORMERS_REVISION,
            out_dir=root / "continued-from-private-models",
            source_config=TEST_CONFIG,
            continuation_config=continuation,
            expected_panel_content_sha256=rows.panel_content_sha256,
            expected_training_cutoff=CUTOFF,
            **runtime_args,
        )
    assert observed["base_path"] != base
    assert observed["adapter_path"] != source_path
    assert observed["base_bytes"] == base_a
    assert observed["adapter_bytes"] == adapter_a
    assert not pathlib.Path(observed["base_path"]).exists()
    assert not pathlib.Path(observed["adapter_path"]).exists()
    assert base_model.read_bytes() == base_a
    assert source_model.read_bytes() == adapter_a

    assert pathlib.Path(artifact.path).is_dir()

    rejected_backend = RecordingContinuationBackend(
        source_tree_sha256=source.sha256,
        source_model_sha256=source_model_sha256,
        continuation_config=continuation,
        validation_losses=(10.0, 11.0),
    )
    rejected_backend.runtime_versions = dict(backend.runtime_versions)

    def private_aba_runtime_factory(
        checkpoint_path,
        _model_revision,
        adapter_path,
        *_args,
    ):
        private_base_model = pathlib.Path(checkpoint_path) / "model.safetensors"
        private_adapter_model = pathlib.Path(adapter_path) / "adapter_model.safetensors"
        private_base_a = private_base_model.read_bytes()
        private_adapter_a = private_adapter_model.read_bytes()
        private_base_model.chmod(0o600)
        private_adapter_model.chmod(0o600)
        private_base_model.write_bytes(b"private-base-B")
        private_adapter_model.write_bytes(b"private-adapter-B")
        private_base_model.write_bytes(private_base_a)
        private_adapter_model.write_bytes(private_adapter_a)
        private_base_model.chmod(0o400)
        private_adapter_model.chmod(0o400)
        return rejected_backend

    rejected_target = root / "private-model-aba-rejected"
    with _base_identity(config_digest, weights_digest), mock.patch.object(
        subject,
        "_installed_package_versions",
        return_value=runtime_packages,
    ), mock.patch.object(
        subject,
        "_PeftContinuationRuntime",
        side_effect=private_aba_runtime_factory,
    ):
        try:
            continue_lora(
                rows.path,
                rows.rows_sha256,
                rows.manifest_sha256,
                source_adapter_dir=source.path,
                source_adapter_sha256=source.sha256,
                source_adapter_model_sha256=source_model_sha256,
                checkpoint_dir=base,
                model_revision=subject.TRANSFORMERS_REVISION,
                out_dir=rejected_target,
                source_config=TEST_CONFIG,
                continuation_config=continuation,
                expected_panel_content_sha256=rows.panel_content_sha256,
                expected_training_cutoff=CUTOFF,
                **runtime_args,
            )
        except RuntimeError as exc:
            assert "private TimesFM continuation inputs changed" in str(exc)
        else:
            raise AssertionError("continuation accepted private base/adapter ABA")
    assert not rejected_target.exists()


def test_timesfm_inference_uses_private_base_adapter_snapshot_under_aba() -> None:
    root = pathlib.Path(tempfile.mkdtemp()).resolve()
    (
        _rows,
        base,
        config_digest,
        weights_digest,
        runtime_args,
        runtime_packages,
        source,
        _source_model_sha256,
    ) = _trained_source_adapter(root)
    source_path = pathlib.Path(source.path)
    base_model = base / "model.safetensors"
    source_model = source_path / "adapter_model.safetensors"
    base_a = base_model.read_bytes()
    adapter_a = source_model.read_bytes()
    observed: dict[str, object] = {}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, path, **_kwargs):
            base_model.write_bytes(b"base-B")
            source_model.write_bytes(b"adapter-B")
            observed["base_path"] = pathlib.Path(path)
            observed["base_bytes"] = (
                pathlib.Path(path) / "model.safetensors"
            ).read_bytes()
            return cls()

        def to(self, **_kwargs):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            full = torch.zeros((len(kwargs["past_values"]), 128, 10), dtype=torch.float32)
            full[..., 5] = 3.0
            return SimpleNamespace(mean_predictions=full[..., 5], full_predictions=full)

    class FakePeftModel:
        @classmethod
        def from_pretrained(cls, model, path, **_kwargs):
            try:
                observed["adapter_path"] = pathlib.Path(path)
                observed["adapter_bytes"] = (
                    pathlib.Path(path) / "adapter_model.safetensors"
                ).read_bytes()
            finally:
                base_model.write_bytes(base_a)
                source_model.write_bytes(adapter_a)
            return model

    real_import = importlib.import_module

    def provider_import(name: str):
        if name == "transformers":
            return SimpleNamespace(TimesFm2_5ModelForPrediction=FakeModel)
        if name == "peft":
            return SimpleNamespace(PeftModel=FakePeftModel)
        return real_import(name)

    with _base_identity(config_digest, weights_digest), mock.patch.object(
        subject,
        "_installed_package_versions",
        return_value=runtime_packages,
    ):
        provider = TimesFMTransformersBackend(
            checkpoint_dir=base,
            model_revision=subject.TRANSFORMERS_REVISION,
            context_length=32,
            device="cpu",
            adapter_dir=source.path,
            adapter_sha256=source.sha256,
            forecast_origin=CUTOFF,
            **runtime_args,
        )
        with mock.patch.object(
            subject.importlib,
            "import_module",
            side_effect=provider_import,
        ):
            provider.forecast([[1.0] * 32], 28)
    assert observed["base_path"] != base
    assert observed["adapter_path"] != source_path
    assert observed["base_bytes"] == base_a
    assert observed["adapter_bytes"] == adapter_a
    assert not pathlib.Path(observed["base_path"]).exists()
    assert not pathlib.Path(observed["adapter_path"]).exists()
    assert base_model.read_bytes() == base_a
    assert source_model.read_bytes() == adapter_a

    class PrivateAbaModel(FakeModel):
        @classmethod
        def from_pretrained(cls, path, **_kwargs):
            private_model = pathlib.Path(path) / "model.safetensors"
            private_a = private_model.read_bytes()
            private_model.chmod(0o600)
            private_model.write_bytes(b"private-base-B")
            private_model.write_bytes(private_a)
            private_model.chmod(0o400)
            return cls()

    def private_aba_import(name: str):
        if name == "transformers":
            return SimpleNamespace(TimesFm2_5ModelForPrediction=PrivateAbaModel)
        return real_import(name)

    with _base_identity(config_digest, weights_digest), mock.patch.object(
        subject,
        "_installed_package_versions",
        return_value=runtime_packages,
    ):
        rejected = TimesFMTransformersBackend(
            checkpoint_dir=base,
            model_revision=subject.TRANSFORMERS_REVISION,
            context_length=32,
            device="cpu",
            **runtime_args,
        )
        assert rejected._input_snapshots is not None
        rejected_root = rejected._input_snapshots.root
        with mock.patch.object(
            subject.importlib,
            "import_module",
            side_effect=private_aba_import,
        ):
            try:
                rejected.forecast([[1.0] * 32], 28)
            except RuntimeError as exc:
                assert "private TimesFM inference inputs changed" in str(exc)
            else:
                raise AssertionError("inference accepted a private base snapshot ABA")
        assert not rejected_root.exists()

    runtime_swaps: list[pathlib.Path] = []

    class PrivateRuntimeAbaModel(FakeModel):
        @classmethod
        def from_pretrained(cls, path, **_kwargs):
            private_runtime = pathlib.Path(path).parent / "runtime-lock.json"
            runtime_swaps.append(private_runtime)
            payload_a = private_runtime.read_bytes()
            payload_b = payload_a + b"private-runtime-B"
            private_runtime.chmod(0o600)
            private_runtime.write_bytes(payload_b)
            assert private_runtime.read_bytes() == payload_b
            private_runtime.write_bytes(payload_a)
            private_runtime.chmod(0o400)
            return cls()

    def private_runtime_aba_import(name: str):
        if name == "transformers":
            return SimpleNamespace(TimesFm2_5ModelForPrediction=PrivateRuntimeAbaModel)
        return real_import(name)

    with _base_identity(config_digest, weights_digest), mock.patch.object(
        subject,
        "_installed_package_versions",
        return_value=runtime_packages,
    ):
        runtime_rejected = TimesFMTransformersBackend(
            checkpoint_dir=base,
            model_revision=subject.TRANSFORMERS_REVISION,
            context_length=32,
            device="cpu",
            **runtime_args,
        )
        assert runtime_rejected._input_snapshots is not None
        runtime_rejected_root = runtime_rejected._input_snapshots.root
        with mock.patch.object(
            subject.importlib,
            "import_module",
            side_effect=private_runtime_aba_import,
        ):
            try:
                runtime_rejected.forecast([[1.0] * 32], 28)
            except RuntimeError as exc:
                assert "private TimesFM inference inputs changed" in str(exc)
            else:
                raise AssertionError("inference accepted a private runtime-lock ABA")
        assert runtime_swaps
        assert not runtime_rejected_root.exists()


def test_continue_lora_rejects_private_panel_and_runtime_lock_aba() -> None:
    root = pathlib.Path(tempfile.mkdtemp()).resolve()
    (
        rows,
        base,
        config_digest,
        weights_digest,
        runtime_args,
        runtime_packages,
        source,
        source_model_sha256,
    ) = _trained_source_adapter(root)
    continuation = LoRAContinuationConfig(
        max_additional_epochs=1,
        patience=1,
        restart_learning_rate=1e-5,
    )

    panel_source = pathlib.Path(
        json.loads(pathlib.Path(rows.manifest_path).read_text())["panel"]["path"]
    )
    real_directory_digest = subject.directory_digest
    private_panel_swaps: list[pathlib.Path] = []

    def panel_aba_digest(path, *args, **kwargs):
        candidate = pathlib.Path(path)
        is_private_panel = (
            candidate != panel_source
            and candidate.name == panel_source.name
            and any(
                part.startswith("autoforecast-timesfm-continuation.")
                for part in candidate.parts
            )
        )
        if not is_private_panel:
            return real_directory_digest(path, *args, **kwargs)
        private_panel_swaps.append(candidate)
        payload_a = candidate.read_bytes()
        payload_b = payload_a + b"private-panel-B"
        candidate.chmod(0o600)
        candidate.write_bytes(payload_b)
        try:
            assert candidate.read_bytes() == payload_b
            return rows.panel_content_sha256
        finally:
            candidate.write_bytes(payload_a)
            candidate.chmod(0o400)

    panel_target = root / "private-panel-aba-rejected"
    panel_backend = RecordingContinuationBackend(
        source_tree_sha256=source.sha256,
        source_model_sha256=source_model_sha256,
        continuation_config=continuation,
        validation_losses=(10.0, 11.0),
    )
    panel_backend.runtime_versions = {
        package: runtime_packages[package]
        for package in ("torch", "transformers", "peft")
    }
    with _base_identity(config_digest, weights_digest), mock.patch.object(
        subject,
        "_installed_package_versions",
        return_value=runtime_packages,
    ), mock.patch.object(
        subject,
        "directory_digest",
        side_effect=panel_aba_digest,
    ):
        try:
            continue_lora(
                rows.path,
                rows.rows_sha256,
                rows.manifest_sha256,
                source_adapter_dir=source.path,
                source_adapter_sha256=source.sha256,
                source_adapter_model_sha256=source_model_sha256,
                checkpoint_dir=base,
                model_revision=subject.TRANSFORMERS_REVISION,
                out_dir=panel_target,
                source_config=TEST_CONFIG,
                continuation_config=continuation,
                expected_panel_content_sha256=rows.panel_content_sha256,
                expected_training_cutoff=CUTOFF,
                backend=panel_backend,
                **runtime_args,
            )
        except RuntimeError as exc:
            assert "private TimesFM continuation inputs changed" in str(exc)
        else:
            raise AssertionError("continuation accepted a private-panel ABA")
    assert private_panel_swaps
    assert not panel_target.exists()

    runtime_target = root / "private-runtime-lock-aba-rejected"
    runtime_backend = RecordingContinuationBackend(
        source_tree_sha256=source.sha256,
        source_model_sha256=source_model_sha256,
        continuation_config=continuation,
        validation_losses=(10.0, 11.0),
    )
    runtime_backend.runtime_versions = dict(panel_backend.runtime_versions)
    private_runtime_swaps: list[pathlib.Path] = []

    def runtime_aba_factory(checkpoint_path, _revision, _adapter_path, *_args):
        private_runtime = pathlib.Path(checkpoint_path).parent / "runtime-lock.json"
        private_runtime_swaps.append(private_runtime)
        payload_a = private_runtime.read_bytes()
        payload_b = payload_a + b"private-runtime-B"
        private_runtime.chmod(0o600)
        private_runtime.write_bytes(payload_b)
        assert private_runtime.read_bytes() == payload_b
        private_runtime.write_bytes(payload_a)
        private_runtime.chmod(0o400)
        return runtime_backend

    with _base_identity(config_digest, weights_digest), mock.patch.object(
        subject,
        "_installed_package_versions",
        return_value=runtime_packages,
    ), mock.patch.object(
        subject,
        "_PeftContinuationRuntime",
        side_effect=runtime_aba_factory,
    ):
        try:
            continue_lora(
                rows.path,
                rows.rows_sha256,
                rows.manifest_sha256,
                source_adapter_dir=source.path,
                source_adapter_sha256=source.sha256,
                source_adapter_model_sha256=source_model_sha256,
                checkpoint_dir=base,
                model_revision=subject.TRANSFORMERS_REVISION,
                out_dir=runtime_target,
                source_config=TEST_CONFIG,
                continuation_config=continuation,
                expected_panel_content_sha256=rows.panel_content_sha256,
                expected_training_cutoff=CUTOFF,
                backend=None,
                **runtime_args,
            )
        except RuntimeError as exc:
            assert "private TimesFM continuation inputs changed" in str(exc)
        else:
            raise AssertionError("continuation accepted a private runtime-lock ABA")
    assert private_runtime_swaps
    assert not runtime_target.exists()


def test_timesfm_outer_snapshots_preserve_original_adapter_runtime_provenance() -> None:
    root = pathlib.Path(tempfile.mkdtemp()).resolve()
    (
        rows,
        base,
        config_digest,
        weights_digest,
        runtime_args,
        runtime_packages,
        source,
        source_model_sha256,
    ) = _trained_source_adapter(root)
    with mock.patch.object(
        subject,
        "_installed_package_versions",
        return_value=runtime_packages,
    ):
        recorded_runtime = subject.validate_runtime_lock(
            runtime_args["runtime_lock_path"],
            runtime_args["runtime_lock_file_sha256"],
            runtime_lock_record_path=runtime_args["runtime_lock_record_path"],
            runtime_lock_sha256=runtime_args["runtime_lock_sha256"],
        )
    recorded_runtime_json = {
        "source_path": recorded_runtime.source_path,
        "source_sha256": recorded_runtime.source_sha256,
        "record_path": list(recorded_runtime.record_path),
        "projection_sha256": recorded_runtime.projection_sha256,
        "python": recorded_runtime.python,
        "package_versions": [list(pair) for pair in recorded_runtime.package_versions],
    }

    outer = root / "outer-input-snapshot"
    outer.mkdir()
    outer_rows = pathlib.Path(shutil.copytree(rows.path, outer / "training-rows"))
    outer_source = pathlib.Path(shutil.copytree(source.path, outer / "source-adapter"))
    outer_base = pathlib.Path(shutil.copytree(base, outer / "base"))
    outer_runtime = pathlib.Path(
        shutil.copy2(runtime_args["runtime_lock_path"], outer / "runtime-lock.json")
    )
    recorded_panel = pathlib.Path(
        json.loads(pathlib.Path(rows.manifest_path).read_text())["panel"]["path"]
    )
    outer_panel = pathlib.Path(shutil.copy2(recorded_panel, outer / recorded_panel.name))

    continuation = LoRAContinuationConfig(
        max_additional_epochs=1,
        patience=1,
        restart_learning_rate=1e-5,
    )
    backend = RecordingContinuationBackend(
        source_tree_sha256=source.sha256,
        source_model_sha256=source_model_sha256,
        continuation_config=continuation,
        validation_losses=(10.0, 11.0),
    )
    backend.runtime_versions = {
        package: runtime_packages[package]
        for package in ("torch", "transformers", "peft")
    }
    outer_runtime_args = {
        **runtime_args,
        "runtime_lock_path": outer_runtime,
    }
    with _base_identity(config_digest, weights_digest), mock.patch.object(
        subject,
        "_installed_package_versions",
        return_value=runtime_packages,
    ):
        artifact = continue_lora(
            outer_rows,
            rows.rows_sha256,
            rows.manifest_sha256,
            source_adapter_dir=outer_source,
            source_adapter_sha256=source.sha256,
            source_adapter_model_sha256=source_model_sha256,
            checkpoint_dir=outer_base,
            model_revision=subject.TRANSFORMERS_REVISION,
            out_dir=root / "continued-from-outer-snapshot",
            source_config=TEST_CONFIG,
            continuation_config=continuation,
            expected_panel_content_sha256=rows.panel_content_sha256,
            expected_training_cutoff=CUTOFF,
            panel_validation_path=outer_panel,
            recorded_source_adapter_dir=source.path,
            recorded_runtime_lock=recorded_runtime_json,
            backend=backend,
            **outer_runtime_args,
        )
        authenticated = subject.validate_lora_continuation_adapter(
            artifact.path,
            artifact.sha256,
            training_rows_dir=outer_rows,
            training_rows_sha256=rows.rows_sha256,
            training_rows_manifest_sha256=rows.manifest_sha256,
            source_adapter_dir=outer_source,
            source_adapter_sha256=source.sha256,
            source_adapter_model_sha256=source_model_sha256,
            checkpoint_dir=outer_base,
            model_revision=subject.TRANSFORMERS_REVISION,
            source_config=TEST_CONFIG,
            continuation_config=continuation,
            expected_panel_content_sha256=rows.panel_content_sha256,
            expected_training_cutoff=CUTOFF,
            forecast_origin=CUTOFF,
            panel_validation_path=outer_panel,
            recorded_source_adapter_dir=source.path,
            recorded_runtime_lock=recorded_runtime_json,
            **outer_runtime_args,
        )
    assert authenticated == artifact
    manifest = json.loads(pathlib.Path(artifact.manifest_path).read_text())
    assert manifest["source_adapter"]["path"] == source.path
    assert manifest["source_adapter"]["manifest_path"] == source.manifest_path
    assert manifest["runtime"]["runtime_lock"]["source_path"] == str(
        runtime_args["runtime_lock_path"]
    )

    outer_adapter = pathlib.Path(
        shutil.copytree(artifact.path, outer / "continued-adapter")
    )
    observed: dict[str, pathlib.Path] = {}

    class FakeModel:
        @classmethod
        def from_pretrained(cls, path, **_kwargs):
            observed["base"] = pathlib.Path(path)
            return cls()

        def to(self, **_kwargs):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            full = torch.zeros((len(kwargs["past_values"]), 128, 10), dtype=torch.float32)
            full[..., 5] = 3.0
            return SimpleNamespace(mean_predictions=full[..., 5], full_predictions=full)

    class FakePeftModel:
        @classmethod
        def from_pretrained(cls, model, path, **_kwargs):
            observed["adapter"] = pathlib.Path(path)
            return model

    original_import = importlib.import_module

    def provider_import(name: str):
        if name == "transformers":
            return SimpleNamespace(TimesFm2_5ModelForPrediction=FakeModel)
        if name == "peft":
            return SimpleNamespace(PeftModel=FakePeftModel)
        return original_import(name)

    with _base_identity(config_digest, weights_digest), mock.patch.object(
        subject,
        "_installed_package_versions",
        return_value=runtime_packages,
    ):
        provider = TimesFMTransformersBackend(
            checkpoint_dir=outer_base,
            model_revision=subject.TRANSFORMERS_REVISION,
            context_length=32,
            device="cpu",
            adapter_dir=outer_adapter,
            adapter_sha256=artifact.sha256,
            forecast_origin=CUTOFF,
            recorded_runtime_lock=recorded_runtime_json,
            recorded_adapter_dir=artifact.path,
            **outer_runtime_args,
        )
        assert provider._input_snapshots is not None
        private_root = provider._input_snapshots.root
        with mock.patch.object(
            subject.importlib,
            "import_module",
            side_effect=provider_import,
        ):
            provider.forecast([[1.0] * 32], 28)
    assert observed["base"].parent == private_root
    assert observed["adapter"].parent == private_root
    assert observed["base"] != outer_base
    assert observed["adapter"] != outer_adapter
    assert provider._runtime_lock.source_path == str(runtime_args["runtime_lock_path"])
    assert provider._recorded_adapter_dir == pathlib.Path(artifact.path)
    assert not private_root.exists()


def test_lora_tree_rejects_alias_empty_hardlink_and_special_nodes() -> None:
    short_tmp = pathlib.Path("/private/tmp" if pathlib.Path("/private/tmp").is_dir() else "/tmp")
    root = pathlib.Path(tempfile.mkdtemp(dir=short_tmp)).resolve()

    def tree(name: str) -> pathlib.Path:
        path = root / name
        path.mkdir()
        (path / "adapter_model.safetensors").write_bytes(b"weights")
        return path

    def refuses(path: pathlib.Path, fragment: str) -> None:
        try:
            subject._tree_sha256(path)
        except ValueError as exc:
            assert fragment in str(exc)
        else:
            raise AssertionError(f"LoRA tree accepted {fragment}: {path}")

    real_parent = root / "real-parent"
    real_parent.mkdir()
    aliased_tree = real_parent / "adapter"
    aliased_tree.mkdir()
    (aliased_tree / "adapter_model.safetensors").write_bytes(b"weights")
    ancestor_alias = root / "ancestor-alias"
    ancestor_alias.symlink_to(real_parent, target_is_directory=True)
    refuses(ancestor_alias / "adapter", "symlink-free")

    hardlinked = tree("hardlinked")
    external = root / "external.bin"
    external.write_bytes(b"alias")
    os.link(external, hardlinked / "alias.bin")
    refuses(hardlinked, "hard-linked")

    empty = tree("empty-directory")
    (empty / "unbound").mkdir()
    refuses(empty, "empty directory")

    fifo = tree("fifo")
    os.mkfifo(fifo / "special")
    refuses(fifo, "special filesystem nodes")

    socket_tree = tree("socket")
    socket_path = socket_tree / "special"
    unix_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        unix_socket.bind(str(socket_path))
        refuses(socket_tree, "special filesystem nodes")
    finally:
        unix_socket.close()

    for mode in (stat.S_IFBLK | 0o600, stat.S_IFCHR | 0o600):
        metadata = SimpleNamespace(st_mode=mode, st_nlink=1)
        try:
            subject._source_tree_entry_flags(metadata, label="device")
        except ValueError as exc:
            assert "special filesystem nodes" in str(exc)
        else:
            raise AssertionError(f"LoRA tree classifier accepted device mode {mode:o}")


def test_timesfm_inference_snapshot_cleanup_covers_failure_and_unused_backend() -> None:
    root = pathlib.Path(tempfile.mkdtemp()).resolve()
    base, config_digest, weights_digest = _base_checkpoint(root)
    runtime_args, runtime_packages = _runtime_lock(root)
    with _base_identity(config_digest, weights_digest), mock.patch.object(
        subject,
        "_installed_package_versions",
        return_value=runtime_packages,
    ):
        unused = TimesFMTransformersBackend(
            checkpoint_dir=base,
            model_revision=subject.TRANSFORMERS_REVISION,
            context_length=32,
            device="cpu",
            **runtime_args,
        )
        unused_root = unused._input_snapshots.root
        assert unused_root.exists()
        del unused
        gc.collect()
        assert not unused_root.exists()

        failed = TimesFMTransformersBackend(
            checkpoint_dir=base,
            model_revision=subject.TRANSFORMERS_REVISION,
            context_length=32,
            device="cpu",
            **runtime_args,
        )
        failed_root = failed._input_snapshots.root
        with mock.patch.object(
            subject.importlib,
            "import_module",
            side_effect=RuntimeError("injected provider-load failure"),
        ):
            try:
                failed.forecast([[1.0] * 32], 28)
            except RuntimeError as exc:
                assert "injected provider-load failure" in str(exc)
            else:
                raise AssertionError("injected provider-load failure was ignored")
        assert not failed_root.exists()


def test_peft_continuation_runtime_warm_loads_exact_adapter_as_trainable() -> None:
    root = pathlib.Path(tempfile.mkdtemp()).resolve()
    base, config_digest, weights_digest = _base_checkpoint(root)
    source = root / "warm-source"
    source.mkdir()
    (source / "adapter_config.json").write_text('{"peft_type":"LORA"}\n')
    (source / "adapter_model.safetensors").write_bytes(b"warm source weights")
    source_tree_sha256 = subject._tree_sha256(source)
    source_model_sha256 = _sha256(source / "adapter_model.safetensors")
    observed: dict[str, object] = {}

    class FakeBase(torch.nn.Module):
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            observed["base_path"] = path
            observed["base_load"] = kwargs
            return cls()

    class FakeWarmAdapter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_weight = torch.nn.Parameter(
                torch.zeros(2, dtype=torch.float32), requires_grad=False
            )
            self.lora_weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

        def save_pretrained(self, _path, **_kwargs):
            raise AssertionError("warm-load construction must not publish")

    class FakePeftModel:
        @classmethod
        def from_pretrained(cls, model, path, **kwargs):
            observed["base"] = model
            observed["adapter_path"] = path
            observed["adapter_load"] = kwargs
            return FakeWarmAdapter()

    original_import = importlib.import_module

    def training_import(name: str, package: str | None = None):
        if name == "transformers":
            return SimpleNamespace(
                __version__="functional-transformers",
                TimesFm2_5ModelForPrediction=FakeBase,
            )
        if name == "peft":
            return SimpleNamespace(
                __version__="functional-peft",
                PeftModel=FakePeftModel,
            )
        return original_import(name, package)

    with _base_identity(config_digest, weights_digest), mock.patch.object(
        subject.importlib,
        "import_module",
        side_effect=training_import,
    ):
        runtime = subject._PeftContinuationRuntime(
            base,
            subject.TRANSFORMERS_REVISION,
            source,
            source_tree_sha256,
            source_model_sha256,
            TEST_CONFIG,
            TEST_CONTINUATION_CONFIG,
            total_steps=2,
            expected_parameters={"trainable": 1, "total": 3},
        )
    assert observed["base_path"] == str(base)
    assert observed["base_load"] == {
        "local_files_only": True,
        "dtype": torch.float32,
    }
    assert observed["adapter_path"] == str(source)
    assert observed["adapter_load"] == {
        "is_trainable": True,
        "local_files_only": True,
    }
    identity = runtime._identity()
    assert identity["parameters"] == {"trainable": 1, "total": 3}
    assert identity["optimizer"] == {
        "optimizer": "torch.optim.AdamW",
        "scheduler": "CosineAnnealingLR_per_batch",
        "optimizer_state_restored": False,
        "scheduler_state_restored": False,
        "restart_learning_rate": 1e-5,
    }
    assert subject._tree_sha256(source) == source_tree_sha256
    assert _sha256(source / "adapter_model.safetensors") == source_model_sha256


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
