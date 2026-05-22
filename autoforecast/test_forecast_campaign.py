"""Protocol tests for the high-level forecasting campaign planner."""

from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import hashlib
import io
import json
import pathlib
import sys
from types import SimpleNamespace
from typing import cast

import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import pytest
import torch
import yaml

import autoforecast.forecast_campaign as subject
from autoforecast.model_catalog import discover_candidates
from autoforecast.model_names import REVISION_MODELS
from autoforecast.tuning import loads_tuning_spec

ROOT = pathlib.Path(__file__).resolve().parent.parent
M5_DEPTH10_PARAMETERS = (
    {
        "model.encoder_days": 1911,
        "model.d_model": 80,
        "model.n_heads": 10,
        "model.patch_size": 7,
        "model.trunk_stages": 10,
        "model.decoder_layers": 10,
        "model.history_attention_layers": 10,
        "model.commercial_attention_layers": 10,
        "model.event_kernel_layers": 10,
        "model.commercial_attention_heads": 10,
        "model.demand_dropout": 0.1,
        "model.covariate_dropout": 0.2,
        "training.learning_rate": 0.0003,
        "training.train_stride": 28,
        "training.train_windows": 52,
        "training.series_loss_weighting": "uniform",
    },
    {
        "model.encoder_days": 1911,
        "model.d_model": 80,
        "model.n_heads": 10,
        "model.trunk_stages": 10,
        "model.covariate_dropout": 0.2,
        "training.learning_rate": 0.0003,
        "training.train_stride": 28,
        "training.train_windows": 52,
        "training.series_loss_weighting": "uniform",
    },
)


def test_campaign_precision_policy_is_versioned_and_legacy_safe(tmp_path: pathlib.Path) -> None:
    original = (
        ROOT / "configs/autoforecast/examples/synthetic_sft_single_model_2epochs.yaml"
    ).read_text(encoding="utf-8")
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text(original, encoding="utf-8")
    assert subject.load_forecast_campaign_request(legacy).compute.precision == "32-true"
    fresh = tmp_path / "fresh.yaml"
    fresh.write_text(original.replace("schema_version: 5", "schema_version: 6", 1),
                     encoding="utf-8")
    assert subject.load_forecast_campaign_request(fresh).compute.precision == "16-mixed"
    explicit = tmp_path / "explicit.yaml"
    explicit.write_text(
        original.replace("schema_version: 5", "schema_version: 6", 1).replace(
            "  compute:\n", "  compute:\n    precision: 32-true\n", 1
        ),
        encoding="utf-8",
    )
    assert subject.load_forecast_campaign_request(explicit).compute.precision == "32-true"
    for index, bad in enumerate(("[]", "{}", "true", "16", "null")):
        malformed = tmp_path / f"bad-{index}.yaml"
        malformed.write_text(
            original.replace("schema_version: 5", "schema_version: 6", 1).replace(
                "  compute:\n", f"  compute:\n    precision: {bad}\n", 1
            ),
            encoding="utf-8",
        )
        with pytest.raises(subject.ForecastCampaignError, match="compute.precision"):
            subject.load_forecast_campaign_request(malformed)


@pytest.mark.parametrize("mode", ["defaults", "manual"])
def test_schema6_static_variant_identity_includes_effective_precision(mode: str) -> None:
    candidates = {candidate.adapter: candidate for candidate in discover_candidates(ROOT / "configs/models")}
    resolution = {"effective_parameters": {"model.encoder_days": 28}}
    trainable = candidates["sft"]
    fp32 = subject._variant_spec_identity(  # noqa: SLF001
        trainable, mode, resolution, None, {"precision": "32-true"}
    )
    fp16 = subject._variant_spec_identity(  # noqa: SLF001
        trainable, mode, resolution, None, {"precision": "16-mixed"}
    )
    assert fp32 != fp16
    native = candidates["seasonal_naive"]
    from autoforecast.precision_policy import effective_precision

    assert subject._variant_spec_identity(  # noqa: SLF001
        native, mode, resolution, None,
        {"precision": effective_precision(native.adapter, "32-true")},
    ) == subject._variant_spec_identity(  # noqa: SLF001
        native, mode, resolution, None,
        {"precision": effective_precision(native.adapter, "16-mixed")},
    )


def test_schema6_child_result_authenticates_training_precision(tmp_path: pathlib.Path) -> None:
    model_id = "sft"
    metadata = tmp_path / "attempts" / model_id / "attempt-0001" / "metadata"
    metadata.mkdir(parents=True)
    path = metadata / "training-controls.json"
    controls = {
        "schema": "autoforecast.training-controls.v2",
        "maximum_epochs": 2,
        "patience": 10,
        "device": "cpu",
        "batch_size": 2,
        "gradient_microbatch_size": 2,
        "early_stopping_absolute_min_delta": 1e-6,
        "early_stopping_relative_min_delta": 0.0,
        "requested_precision": "16-mixed",
        "effective_precision": "16-mixed",
    }
    path.write_text(subject._json(controls), encoding="utf-8")  # noqa: SLF001
    resolved = {"compute": {"device": "cpu", "batch_size": 2,
                            "gradient_microbatch_size": 2, "precision": "16-mixed"}}
    child = {"id": "v0fs0o0", "model_id": model_id,
             "adapter": model_id,
             "budget": {"maximum_epochs": 2, "patience": 10,
                        "early_stopping_absolute_min_delta": 1e-6,
                        "early_stopping_relative_min_delta": 0.0}}
    digest = subject._sha_file(path)  # noqa: SLF001
    subject._verify_child_precision_controls(tmp_path, resolved, child, digest)  # noqa: SLF001

    controls["effective_precision"] = "native"
    path.write_text(subject._json(controls), encoding="utf-8")  # noqa: SLF001
    with pytest.raises(subject.ForecastCampaignError, match="precision evidence"):
        subject._verify_child_precision_controls(  # noqa: SLF001
            tmp_path, resolved, child, subject._sha_file(path)  # noqa: SLF001
        )
    with pytest.raises(subject.ForecastCampaignError, match="identity differs"):
        subject._verify_child_precision_controls(  # noqa: SLF001
            tmp_path, resolved, child, digest
        )


def test_mixed_campaign_rejects_mft_weights_only_arm_before_building_children(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = yaml.safe_load(
        (ROOT / "configs/autoforecast/examples/synthetic_sft_single_model_2epochs.yaml")
        .read_text(encoding="utf-8")
    )
    source["schema_version"] = 6
    body = source["forecast_campaign"]
    body["purpose"] = "forecast_campaign"
    body["models"]["candidates"] = [
        {"model_id": "mft", "parameters": {"mode": "defaults"}}
    ]
    body["initialization_arms"] = ["fresh", "weights_only_finetune"]
    body["temporal"]["initialization_origin"] = "2024-03-01"
    request = tmp_path / "request.yaml"
    request.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    for offset, name in enumerate(("selection", "final-training", "final-score")):
        _panel(tmp_path / name, days=130, target_offset=float(offset * 100))
    monkeypatch.setattr(subject, "_git_source_identity", lambda _path: {"git_commit": "a" * 40})
    with pytest.raises(subject.ForecastCampaignError, match="MFT weights-only continuation"):
        _compile(request, tmp_path)


def test_forecast_campaign_numbers_reject_values_outside_binary64() -> None:
    with pytest.raises(subject.ForecastCampaignError, match="finite non-negative number"):
        subject._nonnegative_number(10**400, "budget")  # noqa: SLF001

    comparison = {
        "candidates": [
            {
                "model_id": "model",
                "status": "scored",
                "score": {"mae": 10**400},
            }
        ]
    }
    with pytest.raises(subject.ForecastCampaignError, match="absent or non-finite"):
        subject._score_value(comparison, "mae", "model")  # noqa: SLF001


def test_public_parameter_and_score_stage_entrypoints_refuse_invalid_artifact_sha(
    tmp_path: pathlib.Path,
) -> None:
    with pytest.raises(subject.ForecastCampaignError, match="SHA-256 is invalid"):
        subject.run_tune_parameters(
            tmp_path / "missing-resolved.json",
            "invalid",
            model_id="seasonal_naive",
            variant_id="default",
            campaign_root=tmp_path / "campaign",
            attempt_dir=tmp_path / "attempt",
        )
    with pytest.raises(subject.ForecastCampaignError, match="SHA-256 is invalid"):
        subject.score_final(
            tmp_path / "missing-resolved.json",
            "invalid",
            tmp_path / "missing-score.json",
            "invalid",
            campaign_root=tmp_path / "campaign",
            attempt_dir=tmp_path / "attempt",
        )


def _panel(
    path: pathlib.Path,
    *,
    start: dt.date = dt.date(2024, 1, 1),
    days: int = 32,
    target_offset: float = 0.0,
) -> None:
    rows = [
        (
            series,
            start + dt.timedelta(days=offset),
            float(offset + series_index) + target_offset,
        )
        for series_index, series in enumerate(("a", "b"))
        for offset in range(days)
    ]
    path.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            [dict(zip(("series_id", "date", "target"), row, strict=True)) for row in rows]
        ),
        path / "part.parquet",
    )


def _request(
    tmp_path: pathlib.Path,
    *,
    models: list[str] | None = None,
    arms: list[str] | None = None,
) -> pathlib.Path:
    selection = tmp_path / "selection"
    final_training = tmp_path / "final-training"
    final_score = tmp_path / "final-score"
    for offset, path in enumerate((selection, final_training, final_score)):
        _panel(path, target_offset=float(offset * 100))
    value = {
        "schema_version": 1,
        "forecast_campaign": {
            "name": "synthetic-campaign",
            "dataset": {
                "contract": "configs/datasets/synthetic_daily.yaml",
                "run_template": "configs/autoforecast/examples/synthetic_seasonal_naive.yaml",
                "selection": {
                    "snapshot_id": "selection-v1",
                    "reference": "local://selection-v1",
                },
                "final_training": {
                    "snapshot_id": "training-v1",
                    "reference": "local://training-v1",
                },
                "final_score": {
                    "snapshot_id": "score-v1",
                    "reference": "local://score-v1",
                },
            },
            "models": {
                "catalog": "configs/models",
                "ids": models or ["seasonal_naive", "croston"],
            },
            "temporal": {
                "training_start": "2024-01-01",
                "initialization_origin": "2024-01-10",
                "validation_origins": ["2024-01-14", "2024-01-21"],
                "validation_horizon": 2,
                "final_train_cutoff": "2024-01-25",
                "final_test_horizon": 2,
            },
            "seeds": [0],
            "initialization_arms": arms or ["fresh", "weights_only_finetune"],
            "selection": {
                "metric": "mae",
                "missing_actual": "no_data",
                "direction": "minimize",
                "tie_break": [
                    "mean_score",
                    "score_stddev",
                    "model_id",
                    "initialization",
                ],
            },
            "final_evidence_role": "blind_test",
            "budgets": {
                name: {
                    "maximum_epochs": 2,
                    "patience": 1,
                    "timeout_seconds": 120,
                    "max_attempts": 2,
                    "early_stopping_absolute_min_delta": 1e-6,
                    "early_stopping_relative_min_delta": 0.0,
                }
                for name in ("initialization", "validation", "final_fit")
            },
            "compute": {
                "device": "cpu",
                "batch_size": 32,
                "gradient_microbatch_size": 16,
                "memory_gib": 0.25,
                "min_free_disk_gib": 0.1,
            },
            "tracking": {"mlflow": False},
        },
    }
    path = tmp_path / "request.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _compile(path: pathlib.Path, tmp_path: pathlib.Path) -> subject.CompiledForecastCampaign:
    return subject.compile_forecast_campaign(
        path,
        selection_data_path=tmp_path / "selection",
        final_training_data_path=tmp_path / "final-training",
        final_score_data_path=tmp_path / "final-score",
        compiled_root=tmp_path / "compiled",
    )


def test_wrmsse_without_price_is_rejected_before_campaign_training(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    value = yaml.safe_load(request.read_text())
    value["forecast_campaign"]["selection"]["metric"] = "wrmsse"
    request.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(subject, "_git_source_identity", lambda *_args: {})
    with pytest.raises(subject.ForecastCampaignError, match="price column before training"):
        _compile(request, tmp_path)


def test_wrmsse_uniform_revision_weighting_is_rejected_before_training(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_id = "sft"
    request = _request(tmp_path, models=[model_id], arms=["fresh"])
    value = yaml.safe_load(request.read_text())
    value["forecast_campaign"]["selection"]["metric"] = "wrmsse"
    request.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(subject, "_git_source_identity", lambda *_args: {})
    monkeypatch.setattr(
        subject, "_load_dataset_contract_bytes",
        lambda *_args: SimpleNamespace(
            panel=lambda *_panel_args: SimpleNamespace(frequency="D", price_col="price")
        ),
    )
    discover = subject.discover_candidates

    def uniform_candidate(root):
        candidates = discover(root)
        result = []
        for candidate in candidates:
            if candidate.id == model_id:
                recipe = json.loads(candidate.implementation_config_bytes)
                recipe["hierarchy_weighting"] = "uniform"
                candidate = dataclasses.replace(
                    candidate, implementation_config_bytes=json.dumps(recipe).encode()
                )
            result.append(candidate)
        return tuple(result)

    monkeypatch.setattr(subject, "discover_candidates", uniform_candidate)
    with pytest.raises(subject.ForecastCampaignError, match="uniform hierarchy weighting"):
        _compile(request, tmp_path)


def test_compiled_artifact_publication_cannot_be_redirected_by_parent_swap(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "compiled"
    parent.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    real_publish = subject.publish_bytes_set

    def swap_then_publish(root: pathlib.Path, payloads: dict[str, bytes]) -> pathlib.Path:
        root.rmdir()
        root.symlink_to(outside, target_is_directory=True)
        return real_publish(root, payloads)

    monkeypatch.setattr(subject, "publish_bytes_set", swap_then_publish)
    with pytest.raises(subject.ForecastCampaignError, match="could not publish"):
        subject._write_immutable_set(  # noqa: SLF001
            parent, {"request-source.yaml": b"sealed\n"}
        )

    assert not (outside / "request-source.yaml").exists()


def _request_v2(
    tmp_path: pathlib.Path,
    candidates: list[dict[str, object]],
) -> pathlib.Path:
    path = _request(tmp_path, models=[cast(str, item["id"]) for item in candidates])
    value = yaml.safe_load(path.read_text())
    value["schema_version"] = 2
    value["forecast_campaign"]["models"] = {
        "catalog": "configs/models",
        "candidates": candidates,
    }
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _request_v3(
    tmp_path: pathlib.Path,
    candidates: list[dict[str, object]],
) -> pathlib.Path:
    path = _request(tmp_path, models=[cast(str, item["model_id"]) for item in candidates])
    value = yaml.safe_load(path.read_text())
    value["schema_version"] = 3
    value["forecast_campaign"]["models"] = {
        "catalog": "configs/models",
        "candidates": candidates,
    }
    value["forecast_campaign"]["selection"]["tie_break"] = list(subject._TIE_BREAK)  # noqa: SLF001
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _request_v4(
    tmp_path: pathlib.Path,
    candidates: list[dict[str, object]],
    *,
    run_template: str | None = None,
) -> pathlib.Path:
    path = _request(tmp_path, models=[cast(str, item["model_id"]) for item in candidates])
    value = yaml.safe_load(path.read_text())
    value["schema_version"] = 4
    dataset = value["forecast_campaign"]["dataset"]
    if run_template is None:
        dataset.pop("run_template")
    else:
        dataset["run_template"] = run_template
    for role in ("selection", "final_training", "final_score"):
        dataset[role] = {"source_uri": dataset[role]["reference"]}
    value["forecast_campaign"]["models"] = {
        "catalog": "configs/models",
        "candidates": candidates,
    }
    value["forecast_campaign"]["compute"]["gradient_microbatch_size"] = value["forecast_campaign"][
        "compute"
    ]["batch_size"]
    value["forecast_campaign"]["selection"]["tie_break"] = list(subject._TIE_BREAK)  # noqa: SLF001
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_schema_four_derives_variant_and_data_identities(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _request_v4(
        tmp_path,
        [
            {
                "model_id": "sft",
                "parameters": {
                    "mode": "manual",
                    "values": {
                        "model.encoder_days": 28,
                        "training.train_windows": 3,
                        "training.train_stride": 7,
                    },
                },
            },
            {
                "model_id": "hmfnet",
                "parameters": {"mode": "defaults"},
            },
        ],
    )
    request = subject.load_forecast_campaign_request(path)
    assert request.schema_version == 4
    assert request.variant_ids == (None, None)
    assert request.candidate_ids == (None, None)
    assert request.template is None
    assert request.selection_data.snapshot_id is None
    assert request.selection_data.reference == "local://selection-v1"

    monkeypatch.setattr(subject, "_git_source_identity", lambda _path: {"git_commit": "a" * 40})
    compiled = _compile(path, tmp_path)
    resolved = json.loads(compiled.resolved_path.read_text())
    profiles = resolved["parameter_profiles"]
    assert [item["mode"] for item in profiles] == ["manual", "defaults"]
    assert profiles[0]["variant_id"].startswith("manual-")
    assert profiles[1]["variant_id"].startswith("defaults-")
    assert all(len(item["variant_spec_sha256"]) == 64 for item in profiles)
    assert [item["candidate_id"] for item in resolved["models"]] == [
        item["candidate_id"] for item in profiles
    ]
    for role in ("selection", "final_training"):
        record = resolved["data"][role]
        assert record["snapshot_id"] == f"{role.replace('_', '-')}-{record['sha256'][:16]}"
    changed_data = tmp_path / "changed-selection-bytes"
    _panel(changed_data, target_offset=1.0)
    changed_data_identity = subject._data_identity(  # noqa: SLF001
        request.selection_data, changed_data, "selection data"
    )
    assert changed_data_identity["snapshot_id"] != resolved["data"]["selection"]["snapshot_id"]
    assert any("parameter_mode=manual (HPO off" in line for line in compiled.preflight_lines)
    assert any("parameter_mode=defaults (HPO off" in line for line in compiled.preflight_lines)

    repeated = subject.compile_forecast_campaign(
        path,
        selection_data_path=tmp_path / "selection",
        final_training_data_path=tmp_path / "final-training",
        final_score_data_path=tmp_path / "final-score",
        compiled_root=tmp_path / "compiled-again",
    )
    repeated_profiles = json.loads(repeated.resolved_path.read_text())["parameter_profiles"]
    assert repeated_profiles[0]["variant_id"] == profiles[0]["variant_id"]

    changed_root = tmp_path / "changed"
    changed_path = _request_v4(
        changed_root,
        [
            {
                "model_id": "sft",
                "parameters": {
                    "mode": "manual",
                    "values": {
                        "model.encoder_days": 56,
                        "training.train_windows": 3,
                        "training.train_stride": 7,
                    },
                },
            }
        ],
    )
    changed = _compile(changed_path, changed_root)
    changed_profile = json.loads(changed.resolved_path.read_text())["parameter_profiles"][0]
    assert changed_profile["variant_id"] != profiles[0]["variant_id"]


@pytest.mark.parametrize(
    ("role", "replacement_offset"),
    [("selection", 1.0), ("final-training", 101.0), ("final-score", 201.0)],
)
def test_compiled_directory_identity_covers_every_runtime_dataset(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    replacement_offset: float,
) -> None:
    path = _request_v4(
        tmp_path,
        [{"model_id": "seasonal_naive", "parameters": {"mode": "defaults"}}],
    )
    monkeypatch.setattr(subject, "_git_source_identity", lambda _path: {"git_commit": "a" * 40})
    first = _compile(path, tmp_path)
    first_resolved = first.resolved_path.read_bytes()
    first_score_input = first.score_input_path.read_bytes()

    _panel(tmp_path / role, target_offset=replacement_offset)
    changed = _compile(path, tmp_path)
    repeated = _compile(path, tmp_path)

    assert changed.directory != first.directory
    assert repeated.directory == changed.directory
    assert first.resolved_path.read_bytes() == first_resolved
    assert first.score_input_path.read_bytes() == first_score_input


def test_reused_tuned_variant_identity_ignores_machine_paths_but_not_evidence() -> None:
    candidate = next(
        item
        for item in discover_candidates(ROOT / "configs/models")
        if item.id == "sft"
    )
    resolution: dict[str, object] = {
        "source": {
            "kind": "reuse_selection",
            "study_id": "study",
            "study_config_path": "configs/hpo/study.yaml",
            "study_config_sha256": "a" * 64,
            "selection_path": "/machine-a/study/selection.json",
            "selection_sha256": "b" * 64,
            "selected_trial_id": "trial_0",
            "selected_attempt_id": "attempt_0",
            "selected_attempt_path": "/machine-a/study/trial_0/attempt_0",
            "selected_attempt_manifest_sha256": "c" * 64,
            "protocol": {"metric": "wrmsse", "training_seed": 0},
            "selected_source_commit": "d" * 40,
        },
        "effective_parameters": {"model.encoder_days": 364},
    }
    study = {"sha256": "a" * 64}
    original = subject._variant_spec_identity(  # noqa: SLF001
        candidate, "reuse_selection", resolution, study, {}
    )
    relocated = json.loads(subject._json(resolution))  # noqa: SLF001
    relocated["source"]["selection_path"] = "/machine-b/study/selection.json"
    relocated["source"]["selected_attempt_path"] = "/machine-b/study/trial_0/attempt_0"
    assert (
        subject._variant_spec_identity(  # noqa: SLF001
            candidate, "reuse_selection", relocated, study, {}
        )
        == original
    )

    changed_evidence = json.loads(subject._json(resolution))  # noqa: SLF001
    changed_evidence["source"]["selection_sha256"] = "e" * 64
    assert (
        subject._variant_spec_identity(  # noqa: SLF001
            candidate, "reuse_selection", changed_evidence, study, {}
        )
        != original
    )
    changed_parameters = json.loads(subject._json(resolution))  # noqa: SLF001
    changed_parameters["effective_parameters"]["model.encoder_days"] = 728
    assert (
        subject._variant_spec_identity(  # noqa: SLF001
            candidate, "reuse_selection", changed_parameters, study, {}
        )
        != original
    )


def test_schema_four_rejects_duplicate_resolved_candidate_specs(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = {
        "model_id": "sft",
        "parameters": {"mode": "defaults"},
    }
    path = _request_v4(
        tmp_path,
        [
            candidate,
            {
                "model_id": "sft",
                "parameters": {"mode": "defaults"},
            },
        ],
    )
    monkeypatch.setattr(subject, "_git_source_identity", lambda _path: {"git_commit": "a" * 40})
    with pytest.raises(subject.ForecastCampaignError, match="same resolved model"):
        _compile(path, tmp_path)


def test_schema_four_rejects_authored_variant_snapshot_and_boolean_modes(
    tmp_path: pathlib.Path,
) -> None:
    path = _request_v4(
        tmp_path,
        [{"model_id": "sft", "parameters": {"mode": "defaults"}}],
    )
    value = yaml.safe_load(path.read_text())
    value["forecast_campaign"]["models"]["candidates"][0]["variant_id"] = "stale"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    with pytest.raises(subject.ForecastCampaignError, match="requires exactly"):
        subject.load_forecast_campaign_request(path)

    value = yaml.safe_load(path.read_text())
    value["forecast_campaign"]["models"]["candidates"][0].pop("variant_id")
    value["forecast_campaign"]["dataset"]["selection"]["snapshot_id"] = "manual-id"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    with pytest.raises(subject.ForecastCampaignError, match="dataset.selection requires exactly"):
        subject.load_forecast_campaign_request(path)

    value["forecast_campaign"]["dataset"]["selection"].pop("snapshot_id")
    value["forecast_campaign"]["models"]["candidates"][0]["parameters"] = {
        "hpo": False,
        "custom": True,
    }
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    with pytest.raises(subject.ForecastCampaignError, match="parameters.mode"):
        subject.load_forecast_campaign_request(path)


def test_schema_three_separates_models_from_parameter_variants(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _request_v3(
        tmp_path,
        [
            {
                "model_id": "sft",
                "variant_id": "short_context",
                "parameters": {
                    "source": "manual",
                    "base": "profile_defaults",
                    "values": {
                        "model.encoder_days": 28,
                        "training.train_windows": 3,
                        "training.train_stride": 7,
                    },
                },
            },
            {
                "model_id": "sft",
                "variant_id": "annual_context",
                "parameters": {"source": "defaults"},
            },
            {
                "model_id": "hmfnet",
                "variant_id": "annual_context",
                "parameters": {"source": "defaults"},
            },
        ],
    )
    body = yaml.safe_load(path.read_text())
    body["forecast_campaign"]["compute"]["gradient_microbatch_size"] = 32
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    request = subject.load_forecast_campaign_request(path)
    assert request.model_ids == (
        "sft",
        "sft",
        "hmfnet",
    )
    assert request.candidate_ids == (
        "@3:sft:short_context",
        "@3:sft:annual_context",
        "@6:hmfnet:annual_context",
    )
    monkeypatch.setattr(subject, "_git_source_identity", lambda _path: {"git_commit": "a" * 40})
    compiled = _compile(path, tmp_path)
    resolved = json.loads(compiled.resolved_path.read_text())
    assert [item["candidate_id"] for item in resolved["parameter_profiles"]] == list(
        request.candidate_ids
    )
    assert {
        (item["candidate_id"], item["model_id"], item["variant_id"])
        for item in resolved["children"]
        if item["kind"] == "validation"
    } == {
        ("@3:sft:short_context", "sft", "short_context"),
        (
            "@3:sft:annual_context",
            "sft",
            "annual_context",
        ),
        (
            "@6:hmfnet:annual_context",
            "hmfnet",
            "annual_context",
        ),
    }


def test_resolved_v3_replays_while_resolved_v4_requires_variant_identity(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _request_v3(
        tmp_path,
        [
            {
                "model_id": "seasonal_naive",
                "variant_id": "default",
                "parameters": {"source": "defaults"},
            }
        ],
    )
    monkeypatch.setattr(subject, "_git_source_identity", lambda _path: {"git_commit": "a" * 40})
    compiled = _compile(path, tmp_path)
    resolved = json.loads(compiled.resolved_path.read_text())
    monkeypatch.setattr(subject, "_verify_resolved_source", lambda _resolved: None)

    historical = json.loads(subject._json(resolved))  # noqa: SLF001
    historical["schema"] = subject.LEGACY_RESOLVED_SCHEMA
    historical.pop("request_schema_version")
    historical["parameter_profiles"][0].pop("variant_spec_sha256")
    historical_path = tmp_path / "resolved-v3.json"
    historical_path.write_text(subject._json(historical), encoding="utf-8")  # noqa: SLF001
    assert (
        subject._load_resolved(  # noqa: SLF001
            historical_path,
            subject._sha_file(historical_path),  # noqa: SLF001
        )["schema"]
        == subject.LEGACY_RESOLVED_SCHEMA
    )

    for index, removed in enumerate(
        [("variant_spec_sha256",), ("request_schema_version", "variant_spec_sha256")]
    ):
        invalid = json.loads(subject._json(resolved))  # noqa: SLF001
        if "request_schema_version" in removed:
            invalid.pop("request_schema_version")
        invalid["parameter_profiles"][0].pop("variant_spec_sha256")
        invalid_path = tmp_path / f"invalid-resolved-v4-{index}.json"
        invalid_path.write_text(subject._json(invalid), encoding="utf-8")  # noqa: SLF001
        with pytest.raises(subject.ForecastCampaignError):
            subject._load_resolved(  # noqa: SLF001
                invalid_path,
                subject._sha_file(invalid_path),  # noqa: SLF001
            )

    downgraded = json.loads(subject._json(resolved))  # noqa: SLF001
    downgraded["schema"] = subject.LEGACY_RESOLVED_SCHEMA
    downgraded["parameter_profiles"][0].pop("variant_spec_sha256")
    downgraded_path = tmp_path / "downgraded-resolved-v4.json"
    downgraded_path.write_text(subject._json(downgraded), encoding="utf-8")  # noqa: SLF001
    with pytest.raises(subject.ForecastCampaignError, match="legacy resolved artifact"):
        subject._load_resolved(  # noqa: SLF001
            downgraded_path,
            subject._sha_file(downgraded_path),  # noqa: SLF001
        )


def test_checked_in_m5_sft_hmfnet_campaign_is_archival_not_executable(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = ROOT / "configs/autoforecast/m5_sft_hmfnet_50k.yaml"
    request = subject.load_forecast_campaign_request(source)
    assert request.model_ids == (
        "sft",
        "hmfnet",
    )
    assert request.schema_version == 4
    assert request.variant_ids == (None, None)
    assert request.candidate_ids == (None, None)
    assert [source.mode for source in request.parameter_sources] == ["manual", "manual"]
    assert [source.values for source in request.parameter_sources] == list(M5_DEPTH10_PARAMETERS)
    candidates = {item.id: item for item in discover_candidates(ROOT / "configs/models")}
    for model_id, parameter_source in zip(
        request.model_ids, request.parameter_sources, strict=True
    ):
        candidate = candidates[model_id]
        spec = loads_tuning_spec(candidate.source_bytes, source_path=candidate.source_path)
        assert set(parameter_source.values) == {parameter.name for parameter in spec.parameters}
    assert request.compute.device == "cpu"
    assert request.compute.batch_size == request.compute.gradient_microbatch_size == 256
    assert request.template is None
    assert request.initialization_origin == dt.date(2016, 2, 28)
    assert request.validation_origins == (dt.date(2016, 3, 27), dt.date(2016, 4, 24))
    assert request.final_train_cutoff == dt.date(2016, 5, 22)
    assert request.validation_horizon == request.final_test_horizon == 28
    assert request.seeds == (0, 1, 2)
    assert request.initialization_arms == ("fresh",)
    assert request.metric == "wrmsse" and request.missing_actual == "no_data"
    assert request.evidence_role == "archived_private_replay_not_blind"
    assert request.mlflow_enabled is True
    for budget in (
        request.initialization_budget,
        request.validation_budget,
        request.final_fit_budget,
    ):
        assert budget.maximum_epochs == 50_000 and budget.patience == 10
        assert budget.timeout_seconds == 86_400 and budget.max_attempts == 2
        assert budget.absolute_min_delta == 1e-6 and budget.relative_min_delta == 0

    with pytest.raises(subject.ForecastCampaignError, match="separate requests and output roots"):
        subject.compile_forecast_campaign(
            source,
            selection_data_path=tmp_path / "selection",
            final_training_data_path=tmp_path / "training",
            final_score_data_path=tmp_path / "score",
        )
    short_source = tmp_path / "archival-two-epoch-structure.yaml"
    short_request = yaml.safe_load(source.read_text(encoding="utf-8"))
    for budget in short_request["forecast_campaign"]["budgets"].values():
        budget["maximum_epochs"] = 2
    short_source.write_text(yaml.safe_dump(short_request), encoding="utf-8")

    data_paths = [tmp_path / name for name in ("selection", "training", "score")]
    for offset, path in enumerate(data_paths):
        _panel(path, target_offset=float(offset * 100))
    monkeypatch.setattr(subject, "_git_source_identity", lambda _path: {"git_commit": "a" * 40})
    compiled = subject.compile_forecast_campaign(
        short_source,
        selection_data_path=data_paths[0],
        final_training_data_path=data_paths[1],
        final_score_data_path=data_paths[2],
        compiled_root=tmp_path / "compiled",
    )
    resolved = json.loads(compiled.resolved_path.read_text())
    assert [item["model_id"] for item in resolved["models"]] == list(request.model_ids)
    assert [item["variant_id"].split("-", maxsplit=1)[0] for item in resolved["models"]] == [
        "manual",
        "manual",
    ]
    assert all(
        item["resolution"]["effective_parameters"]["model.encoder_days"] == 1911
        for item in resolved["parameter_profiles"]
    )
    assert [
        (
            item["resolution"]["effective_parameters"]["model.d_model"],
            item["resolution"]["effective_parameters"]["model.n_heads"],
            item["resolution"]["effective_parameters"]["model.trunk_stages"],
        )
        for item in resolved["parameter_profiles"]
    ] == [(80, 10, 10), (80, 10, 10)]
    assert (
        resolved["parameter_profiles"][0]["resolution"]["effective_parameters"][
            "model.event_kernel_layers"
        ]
        == 10
    )
    assert (
        "model.event_kernel_layers"
        not in resolved["parameter_profiles"][1]["resolution"]["effective_parameters"]
    )
    validation_models = [
        item["model_id"] for item in resolved["children"] if item["kind"] == "validation"
    ]
    assert validation_models == ["sft"] * 6 + ["hmfnet"] * 6


def test_checked_in_m5_sft_hmfnet_b32_campaign_preserves_early_stopping_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = subject.load_forecast_campaign_request(
        ROOT / "configs/autoforecast/m5_sft_hmfnet_50k_b32.yaml"
    )

    assert request.model_ids == ("sft", "hmfnet")
    assert request.compute.batch_size == request.compute.gradient_microbatch_size == 32
    for budget in (
        request.initialization_budget,
        request.validation_budget,
        request.final_fit_budget,
    ):
        assert budget.maximum_epochs == 50_000 and budget.patience == 10
        assert budget.timeout_seconds == 259_200 and budget.max_attempts == 2
    with pytest.raises(subject.ForecastCampaignError, match="separate requests and output roots"):
        subject.compile_forecast_campaign(
            ROOT / "configs/autoforecast/m5_sft_hmfnet_50k_b32.yaml",
            selection_data_path="/not-read",
            final_training_data_path="/not-read",
            final_score_data_path="/not-read",
        )
    with pytest.raises(subject.ForecastCampaignError, match="separate requests and output roots"):
        monkeypatch.setattr(
            subject, "_resolved_catalog_candidates",
            lambda _resolved: {
                model_id: SimpleNamespace(adapter=model_id) for model_id in request.model_ids
            },
        )
        subject._run_generic_child(  # noqa: SLF001
            {"models": [{"model_id": model_id} for model_id in request.model_ids],
             "budgets": {"validation": {"maximum_epochs": 50_000}}},
            {}, campaign_root=pathlib.Path("/not-read"),
            attempt_dir=pathlib.Path("/not-read"), evaluate=False,
        )


@pytest.mark.parametrize("aliases", [
    ("v2_alias", "hmfnet"),
    ("sft", "v3_alias"),
    ("v2_alias", "v3_alias"),
])
def test_long_sft_hmfnet_isolation_uses_adapter_not_catalog_alias(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
    aliases: tuple[str, str],
) -> None:
    source = ROOT / "configs/autoforecast/m5_sft_hmfnet_50k.yaml"
    request = subject.load_forecast_campaign_request(source)
    catalog = {candidate.id: candidate for candidate in discover_candidates(ROOT / "configs/models")}
    monkeypatch.setattr(
        subject, "load_forecast_campaign_request",
        lambda _path: dataclasses.replace(request, model_ids=aliases),
    )
    monkeypatch.setattr(subject, "_git_source_identity", lambda _path: {"git_commit": "a" * 40})
    monkeypatch.setattr(
        subject, "discover_candidates",
        lambda _path: tuple(
            dataclasses.replace(catalog[model_id], id=alias)
            for model_id, alias in zip(request.model_ids, aliases, strict=True)
        ),
    )
    with pytest.raises(subject.ForecastCampaignError, match="separate requests and output roots"):
        subject.compile_forecast_campaign(
            source, selection_data_path=tmp_path / "selection",
            final_training_data_path=tmp_path / "training",
            final_score_data_path=tmp_path / "score",
        )
    monkeypatch.setattr(
        subject, "_resolved_catalog_candidates",
        lambda _resolved: {
            alias: SimpleNamespace(adapter=model_id)
            for model_id, alias in zip(request.model_ids, aliases, strict=True)
        },
    )
    with pytest.raises(subject.ForecastCampaignError, match="separate requests and output roots"):
        subject._run_generic_child(  # noqa: SLF001
            {"models": [{"model_id": alias} for alias in aliases],
             "budgets": {"validation": {"maximum_epochs": 50_000}}},
            {}, campaign_root=tmp_path, attempt_dir=tmp_path, evaluate=False,
        )


def test_campaign_path_guard_allows_case_spelling_of_darwin_tmp_alias() -> None:
    if sys.platform != "darwin" or not pathlib.Path("/Tmp").is_symlink():
        pytest.skip("case-insensitive Darwin /tmp alias is unavailable")
    path = pathlib.Path("/Tmp/autoforecast-no-write-test")
    assert subject._path_without_links(path, "test path") == path  # noqa: SLF001


def test_checked_in_m5_hmfnet_b32_campaign_preserves_early_stopping_capacity() -> None:
    request = subject.load_forecast_campaign_request(
        ROOT / "configs/autoforecast/m5_hmfnet_50k_b32.yaml"
    )

    assert request.model_ids == ("hmfnet",)
    assert request.compute.batch_size == request.compute.gradient_microbatch_size == 32
    for budget in (
        request.initialization_budget,
        request.validation_budget,
        request.final_fit_budget,
    ):
        assert budget.maximum_epochs == 50_000 and budget.patience == 10
        assert budget.timeout_seconds == 259_200 and budget.max_attempts == 2


@pytest.mark.parametrize(
    ("filename", "model_id", "batch_size", "timeout_seconds"),
    (
        ("m5_sft_b64_3epoch_capacity_soak.yaml", "sft", 64, 86_400),
        ("m5_sft_b32_3epoch_capacity_soak.yaml", "sft", 32, 172_800),
        ("m5_hmfnet_b32_3epoch_capacity_soak.yaml", "hmfnet", 32, 172_800),
    ),
)
def test_checked_in_capacity_soaks_allow_the_complete_three_epoch_worker(
    filename: str,
    model_id: str,
    batch_size: int,
    timeout_seconds: int,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = ROOT / "configs/autoforecast" / filename
    request = subject.load_forecast_campaign_request(source)

    assert request.schema_version == 5
    assert request.purpose == "capacity_diagnostic"
    assert request.model_ids == (model_id,)
    assert request.seeds == (0,)
    assert request.initialization_arms == ("fresh",)
    assert request.compute.batch_size == request.compute.gradient_microbatch_size == batch_size
    for budget in (
        request.initialization_budget,
        request.validation_budget,
        request.final_fit_budget,
    ):
        assert budget.maximum_epochs == 3 and budget.patience == 10
        assert budget.timeout_seconds == timeout_seconds and budget.max_attempts == 1

    data_paths = [tmp_path / filename / name for name in ("selection", "training", "score")]
    for offset, path in enumerate(data_paths):
        _panel(path, target_offset=float(offset * 100))
    monkeypatch.setattr(subject, "_git_source_identity", lambda _path: {"git_commit": "a" * 40})
    compiled = subject.compile_forecast_campaign(
        source,
        selection_data_path=data_paths[0],
        final_training_data_path=data_paths[1],
        final_score_data_path=data_paths[2],
        compiled_root=tmp_path / filename / "compiled",
    )
    resolved = json.loads(compiled.resolved_path.read_text(encoding="utf-8"))
    stages = yaml.safe_load(compiled.plan_path.read_text(encoding="utf-8"))["campaign"]["stages"]
    assert [(stage["id"], stage["timeout_seconds"], stage["max_attempts"]) for stage in stages] == [
        ("v0fs0o0", timeout_seconds, 1)
    ]

    class StopBeforeWorkerLaunch(Exception):
        pass

    class CapturingProcess:
        def wait(self) -> int:
            raise StopBeforeWorkerLaunch

    captured: list[list[str]] = []

    def capture(
        command: list[str], *, cwd: pathlib.Path, pass_fds: tuple[int, ...]
    ) -> CapturingProcess:
        assert cwd == ROOT
        assert pass_fds == ()
        captured.append(command)
        return CapturingProcess()

    monkeypatch.setattr(subject, "_verify_data", lambda *_args: data_paths[0])
    monkeypatch.setattr(subject, "_materialize_training_panel", lambda *_args, **_kwargs: "d" * 64)
    monkeypatch.setattr(subject, "_write_child_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(subject, "_peak_rss", lambda _process: 0)
    monkeypatch.setattr(subject.subprocess, "Popen", capture)
    attempt_dir = tmp_path / filename / "attempt"
    attempt_dir.mkdir(parents=True)
    with pytest.raises(StopBeforeWorkerLaunch):
        subject._run_generic_child(  # noqa: SLF001
            resolved,
            resolved["children"][0],
            campaign_root=tmp_path / filename / "campaign",
            attempt_dir=attempt_dir,
            evaluate=False,
        )
    command = captured[0]
    assert command[command.index("--timeout-seconds") + 1] == str(timeout_seconds)


def test_checked_in_m5_depth10_smoke_is_full_data_and_compiles(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    full = subject.load_forecast_campaign_request(ROOT / "configs/autoforecast/m5_sft_hmfnet_50k.yaml")
    source = ROOT / "configs/autoforecast/m5_sft_hmfnet_depth10_2epoch_smoke.yaml"
    smoke = subject.load_forecast_campaign_request(source)

    assert smoke.model_ids == full.model_ids
    assert [item.values for item in smoke.parameter_sources] == list(M5_DEPTH10_PARAMETERS)
    assert [item.values for item in full.parameter_sources] == list(M5_DEPTH10_PARAMETERS)
    assert (
        smoke.contract,
        smoke.selection_data,
        smoke.final_training_data,
        smoke.final_score_data,
    ) == (
        full.contract,
        full.selection_data,
        full.final_training_data,
        full.final_score_data,
    )
    assert smoke.seeds == (0,)
    assert smoke.validation_origins == (dt.date(2016, 4, 24),)
    assert smoke.compute.device == "cpu"
    assert smoke.compute.batch_size == smoke.compute.gradient_microbatch_size == 512
    assert smoke.compute.memory_gib == 6.7 and smoke.compute.min_free_disk_gib == 10
    for budget in (
        smoke.initialization_budget,
        smoke.validation_budget,
        smoke.final_fit_budget,
    ):
        assert budget.maximum_epochs == 2
        assert budget.timeout_seconds == 1_800 and budget.max_attempts == 1

    data_paths = [tmp_path / name for name in ("selection", "training", "score")]
    for offset, path in enumerate(data_paths):
        _panel(path, target_offset=float(offset * 100))
    monkeypatch.setattr(subject, "_git_source_identity", lambda _path: {"git_commit": "a" * 40})
    compiled = subject.compile_forecast_campaign(
        source,
        selection_data_path=data_paths[0],
        final_training_data_path=data_paths[1],
        final_score_data_path=data_paths[2],
        compiled_root=tmp_path / "compiled",
    )
    assert compiled.plan_path.is_file() and compiled.resolved_path.is_file()


def test_schema_three_rejects_only_duplicate_model_variant_pairs(tmp_path: pathlib.Path) -> None:
    path = _request_v3(
        tmp_path,
        [
            {
                "model_id": "sft",
                "variant_id": "same",
                "parameters": {"source": "defaults"},
            },
            {
                "model_id": "sft",
                "variant_id": "same",
                "parameters": {"source": "defaults"},
            },
        ],
    )
    with pytest.raises(subject.ForecastCampaignError, match="model_id, variant_id"):
        subject.load_forecast_campaign_request(path)


def test_candidate_identity_encoding_is_injective_for_valid_identifiers() -> None:
    assert subject._candidate_id("foo--bar", "default") == "foo--bar"  # noqa: SLF001
    assert subject._candidate_id("foo", "bar") == "@3:foo:bar"  # noqa: SLF001
    assert subject._candidate_id("foo--bar", "default") != subject._candidate_id(  # noqa: SLF001
        "foo", "bar"
    )


@pytest.mark.parametrize("model_id", ["sft", "hmfnet"])
@pytest.mark.parametrize("invalid", ["mps", "microbatch", "events"])
def test_revision_controls_fail_during_campaign_compilation(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    model_id: str,
    invalid: str,
) -> None:
    path = _request_v3(
        tmp_path,
        [{"model_id": model_id, "variant_id": "candidate", "parameters": {"source": "defaults"}}],
    )
    value = yaml.safe_load(path.read_text())
    compute = value["forecast_campaign"]["compute"]
    compute["device"] = "cpu"
    compute["batch_size"] = 32
    compute["gradient_microbatch_size"] = 32
    if invalid == "mps":
        compute["device"] = "mps"
    elif invalid == "microbatch":
        compute["gradient_microbatch_size"] = 16
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    if invalid == "events":
        original = subject.load_run_request

        def with_event(request_path):
            return dataclasses.replace(
                original(request_path),
                calendar_events=(("event", "2024-01-01", None),),
            )

        monkeypatch.setattr(subject, "load_run_request", with_event)
    monkeypatch.setattr(subject, "_git_source_identity", lambda _path: {"git_commit": "a" * 40})
    with pytest.raises(subject.ForecastCampaignError, match="not executable"):
        _compile(path, tmp_path)


def test_tuned_revision_recipe_is_validated_before_hpo_or_worker_launch(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _request_v3(
        tmp_path,
        [
            {
                "model_id": "sft",
                "variant_id": "tuned",
                "parameters": {
                    "source": "tune",
                    "study_config": "configs/hpo/does-not-need-to-exist.yaml",
                },
            }
        ],
    )
    value = yaml.safe_load(path.read_text())
    value["forecast_campaign"]["compute"]["gradient_microbatch_size"] = 32
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    discovered = discover_candidates(ROOT / "configs/models")
    candidate = next(item for item in discovered if item.id == "sft")
    recipe = json.loads(candidate.implementation_config_bytes)
    recipe["gradient_clip"] = 0
    payload = (json.dumps(recipe, allow_nan=False) + "\n").encode()
    invalid = dataclasses.replace(
        candidate,
        implementation_config_bytes=payload,
        implementation_config_sha256=hashlib.sha256(payload).hexdigest(),
    )
    monkeypatch.setattr(
        subject,
        "discover_candidates",
        lambda _catalog: tuple(invalid if item.id == invalid.id else item for item in discovered),
    )
    monkeypatch.setattr(subject, "_git_source_identity", lambda _path: {"git_commit": "a" * 40})
    with pytest.raises(subject.ForecastCampaignError, match="baseline execution recipe"):
        _compile(path, tmp_path)


@pytest.mark.parametrize("changed_field", ["source_sha256", "implementation_config_sha256"])
def test_resolved_candidate_second_read_must_match_compiled_profile_identity(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_field: str,
) -> None:
    path = _request_v3(
        tmp_path,
        [
            {
                "model_id": "sft",
                "variant_id": "candidate",
                "parameters": {"source": "defaults"},
            }
        ],
    )
    value = yaml.safe_load(path.read_text())
    value["forecast_campaign"]["compute"]["gradient_microbatch_size"] = 32
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(subject, "_git_source_identity", lambda _path: {"git_commit": "a" * 40})
    compiled = _compile(path, tmp_path)
    resolved = json.loads(compiled.resolved_path.read_text())
    discovered = discover_candidates(ROOT / "configs/models")
    changed = tuple(
        dataclasses.replace(item, **{changed_field: "0" * 64})
        if item.id == "sft"
        else item
        for item in discovered
    )
    monkeypatch.setattr(subject, "discover_candidates", lambda _catalog: changed)
    with pytest.raises(subject.ForecastCampaignError, match="changed after campaign compilation"):
        subject._verify_resolved_parameter_profiles(resolved)  # noqa: SLF001


@pytest.mark.parametrize(
    "parameters",
    [
        {"source": "defaults"},
        {
            "source": "manual",
            "base": "profile_defaults",
            "values": {"model.d_model": 40},
        },
        {"source": "tune", "study_config": "configs/hpo/example.yaml"},
        {
            "source": "reuse_selection",
            "study_config": "configs/hpo/example.yaml",
            "selection": ".autoforecast/hpo/studies/example/selection.json",
        },
    ],
)
def test_schema_two_exposes_one_unambiguous_parameter_source(
    tmp_path: pathlib.Path, parameters: dict[str, object]
) -> None:
    path = _request_v2(tmp_path, [{"id": "mft", "parameters": parameters}])
    request = subject.load_forecast_campaign_request(path)
    assert request.parameter_sources[0].mode == parameters["source"]


@pytest.mark.parametrize(
    ("parameters", "expected_mode"),
    [
        ({"mode": "defaults"}, "defaults"),
        ({"mode": "manual", "values": {"model.d_model": 40}}, "manual"),
        ({"mode": "tune", "study_config": "configs/hpo/example.yaml"}, "tune"),
        (
            {
                "mode": "reuse_tuned",
                "study_config": "configs/hpo/example.yaml",
                "selection": ".autoforecast/hpo/studies/example/selection.json",
            },
            "reuse_selection",
        ),
    ],
)
def test_schema_four_exposes_one_tagged_parameter_mode(
    tmp_path: pathlib.Path,
    parameters: dict[str, object],
    expected_mode: str,
) -> None:
    path = _request_v4(
        tmp_path,
        [{"model_id": "mft", "parameters": parameters}],
    )
    request = subject.load_forecast_campaign_request(path)
    assert request.parameter_sources[0].mode == expected_mode


@pytest.mark.parametrize(
    "parameters",
    [
        {"mode": "defaults", "values": {}},
        {"mode": "manual"},
        {"mode": "manual", "base": "profile_defaults", "values": {}},
        {
            "mode": "tune",
            "study_config": "configs/hpo/example.yaml",
            "selection": "selection.json",
        },
        {"mode": "reuse_tuned", "selection": "selection.json"},
    ],
)
def test_schema_four_parameter_modes_reject_inactive_fields(
    tmp_path: pathlib.Path, parameters: dict[str, object]
) -> None:
    path = _request_v4(
        tmp_path,
        [{"model_id": "mft", "parameters": parameters}],
    )
    with pytest.raises(subject.ForecastCampaignError):
        subject.load_forecast_campaign_request(path)


def test_parameter_source_rejects_inactive_or_contradictory_fields(
    tmp_path: pathlib.Path,
) -> None:
    cases = [
        {"source": "defaults", "values": {}},
        {"source": "manual", "base": "profile_defaults"},
        {
            "source": "tune",
            "study_config": "configs/hpo/example.yaml",
            "selection": "selection.json",
        },
        {"source": "reuse_selection", "selection": "selection.json"},
    ]
    for index, parameters in enumerate(cases):
        path = _request_v2(
            tmp_path / str(index),
            [{"id": "mft", "parameters": parameters}],
        )
        with pytest.raises(subject.ForecastCampaignError):
            subject.load_forecast_campaign_request(path)


def test_compile_freezes_complete_manual_values_and_keeps_models_sequential(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _request_v2(
        tmp_path,
        [
            {
                "id": "mft",
                "parameters": {
                    "source": "manual",
                    "base": "profile_defaults",
                    "values": {"model.d_model": 40},
                },
            },
            {"id": "seasonal_naive", "parameters": {"source": "defaults"}},
        ],
    )
    monkeypatch.setattr(subject, "_git_source_identity", lambda _path: {"git_commit": "a" * 40})
    compiled = _compile(path, tmp_path)
    resolved = json.loads(compiled.resolved_path.read_text())
    manual = resolved["parameter_profiles"][0]["resolution"]
    assert manual["effective_parameters"]["model.d_model"] == 40
    assert manual["effective_parameters"]["model.encoder_days"] == 28
    assert resolved["parameter_profiles"][1]["resolution"]["effective_parameters"] == {}
    plan = yaml.safe_load(compiled.plan_path.read_text())["campaign"]["stages"]
    validation_ids = [stage["id"] for stage in plan if stage["id"].startswith("v")]
    assert validation_ids[:2] == ["v0fs0o0", "v0fs0o1"]
    initialization_parent = next(stage for stage in plan if stage["id"].startswith("p"))
    assert "evaluation-panel" not in initialization_parent["required_outputs"]


def test_tune_mode_compiles_as_a_required_durable_stage(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _request_v2(
        tmp_path,
        [
            {
                "id": "shallow_global",
                "parameters": {
                    "source": "tune",
                    "study_config": "configs/hpo/study.yaml",
                },
            }
        ],
    )
    monkeypatch.setattr(subject, "_git_source_identity", lambda _path: {"git_commit": "a" * 40})
    monkeypatch.setattr(
        subject,
        "_campaign_hpo_study",
        lambda *_args, **_kwargs: SimpleNamespace(
            id="study",
            source_path=ROOT / "configs/models/global/shallow_global/model.yaml",
            source_sha256="b" * 64,
            max_trials=2,
        ),
    )
    compiled = _compile(path, tmp_path)
    plan = yaml.safe_load(compiled.plan_path.read_text())["campaign"]["stages"]
    tuning = plan[0]
    assert tuning["id"] == "t00"
    assert tuning["max_attempts"] == 1
    assert tuning["required_outputs"] == [
        "parameter-resolution.json",
        "hpo",
        "hpo-training-panel",
        "hpo-validation-panel",
        "hpo-run-request.yaml",
    ]
    validation = next(stage for stage in plan if stage["id"].startswith("v"))
    assert validation["needs"] == ["t00"]
    assert validation["required_outputs"] == [
        "child-result.json",
        "run",
        "training-panel",
        "child-request.yaml",
        "parameter-resolution.json",
        "evaluation-panel",
    ]


def test_scoring_snapshot_supports_one_parquet_file_and_excludes_sidecars(
    tmp_path: pathlib.Path,
) -> None:
    source = tmp_path / "panel.parquet"
    source.write_bytes(b"parquet-bytes")
    sidecar = tmp_path / "credentials.txt"
    sidecar.write_text("must not be copied", encoding="utf-8")
    expected = subject.directory_digest(source, suffixes=(".parquet",))

    snapshot = subject._snapshot_parquet_directory(  # noqa: SLF001
        source,
        tmp_path / "snapshot",
        expected_sha256=expected,
        label="score data",
    )

    assert (snapshot / "panel.parquet").read_bytes() == b"parquet-bytes"
    assert not (snapshot / sidecar.name).exists()


def test_request_is_strict_temporal_and_requires_fresh(tmp_path: pathlib.Path) -> None:
    path = _request(tmp_path)
    value = yaml.safe_load(path.read_text())
    value["forecast_campaign"]["surprise"] = True
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(subject.ForecastCampaignError, match="requires exactly"):
        subject.load_forecast_campaign_request(path)

    path = _request(tmp_path / "second", arms=["weights_only_finetune"])
    with pytest.raises(subject.ForecastCampaignError, match="include fresh"):
        subject.load_forecast_campaign_request(path)


def test_campaign_request_preserves_exact_hour_temporal_identity(tmp_path: pathlib.Path) -> None:
    path = _request(tmp_path, models=["seasonal_naive"], arms=["fresh"])
    value = yaml.safe_load(path.read_text())
    temporal = value["forecast_campaign"]["temporal"]
    temporal.update(
        {
            "training_start": "2024-03-01T20:00:00",
            "initialization_origin": "2024-03-01T21:00:00",
            "validation_origins": ["2024-03-01T22:00:00", "2024-03-02T00:00:00"],
            "final_train_cutoff": "2024-03-02T03:00:00",
        }
    )
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    request = subject.load_forecast_campaign_request(path)
    assert request.validation_origins == (
        dt.datetime(2024, 3, 1, 22),
        dt.datetime(2024, 3, 2, 0),
    )
    assert request.final_train_cutoff.isoformat() == "2024-03-02T03:00:00"


@pytest.mark.parametrize(
    "frequency,start,end,span",
    [
        ("H", "2024-03-01T22:00:00", "2024-03-02T01:00:00", 3),
        ("W", "2024-01-03", "2024-01-24", 3),
        ("M", "2024-01-31", "2024-04-30", 3),
        ("Y", "2023-12-31", "2027-12-31", 4),
    ],
)
def test_campaign_window_spans_use_declared_periods(frequency, start, end, span):
    cadence = subject.parse_frequency(frequency)
    assert subject._period_span(  # noqa: SLF001
        subject.parse_origin(start, cadence),
        subject.parse_origin(end, cadence),
        cadence,
        "test window",
    ) == span


@pytest.mark.parametrize("frequency", ["W", "7D"])
def test_campaign_rejects_a_daily_only_adapter_before_unsupported_cadence_execution(
    frequency: str,
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _request(tmp_path, models=["croston"], arms=["fresh"])
    value = yaml.safe_load(path.read_text())
    value["forecast_campaign"]["temporal"].update(
        {
            "training_start": "2024-01-03",
            "initialization_origin": "2024-01-10",
            "validation_origins": ["2024-01-17", "2024-01-24"],
            "validation_horizon": 1,
            "final_train_cutoff": "2024-01-31",
            "final_test_horizon": 1,
        }
    )
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    unsupported = SimpleNamespace(
        panel=lambda _path: SimpleNamespace(frequency=frequency, price_col=None)
    )
    monkeypatch.setattr(subject, "_load_dataset_contract_bytes", lambda *_args: unsupported)
    with pytest.raises(
        subject.ForecastCampaignError, match=f"do not support {frequency} cadence"
    ):
        _compile(path, tmp_path)


@pytest.mark.parametrize(
    "frequency,start",
    [
        ("H", dt.datetime(2024, 1, 1)),
        ("W", dt.date(2024, 1, 1)),
        ("M", dt.date(2024, 1, 31)),
        ("MS", dt.date(2024, 1, 1)),
        ("Y", dt.date(2024, 12, 31)),
        ("YS", dt.date(2024, 1, 1)),
    ],
)
def test_compile_propagates_supported_dataset_cadence(
    frequency, start, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _request(tmp_path, models=["seasonal_naive"], arms=["fresh"])
    cadence = subject.parse_frequency(frequency)
    points = [subject.advance(start, cadence, index) for index in range(7)]
    value = yaml.safe_load(path.read_text())
    value["forecast_campaign"]["temporal"].update(
        {
            "training_start": points[0].isoformat(),
            "initialization_origin": points[1].isoformat(),
            "validation_origins": [points[2].isoformat(), points[4].isoformat()],
            "validation_horizon": 1,
            "final_train_cutoff": points[5].isoformat(),
            "final_test_horizon": 1,
        }
    )
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    contract = SimpleNamespace(panel=lambda _path: SimpleNamespace(frequency=frequency))
    monkeypatch.setattr(subject, "_load_dataset_contract_bytes", lambda *_args: contract)
    monkeypatch.setattr(subject, "_git_source_identity", lambda _path: {})

    compiled = _compile(path, tmp_path)
    resolved = json.loads(compiled.resolved_path.read_text())
    temporal = resolved["temporal"]
    assert temporal["frequency"] == frequency
    assert "step_days" not in temporal
    assert {child["origin"] for child in resolved["children"]} == {
        points[2].isoformat(),
        points[4].isoformat(),
    }


def test_request_refuses_ambiguous_yaml_zero_memory_and_symlink_source(
    tmp_path: pathlib.Path,
) -> None:
    path = _request(tmp_path)
    duplicate = path.read_text() + "schema_version: 1\n"
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(subject.ForecastCampaignError, match="duplicate YAML mapping key"):
        subject.load_forecast_campaign_request(path)

    path = _request(tmp_path / "zero")
    value = yaml.safe_load(path.read_text())
    value["forecast_campaign"]["compute"]["memory_gib"] = 0
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    with pytest.raises(subject.ForecastCampaignError, match="strictly positive"):
        subject.load_forecast_campaign_request(path)

    real = _request(tmp_path / "linked")
    link = tmp_path / "request-link.yaml"
    link.symlink_to(real)
    with pytest.raises(subject.ForecastCampaignError, match="may not traverse a symlink"):
        subject.load_forecast_campaign_request(link)


def test_compiler_rejects_request_outside_repository_as_campaign_error(
    tmp_path: pathlib.Path,
) -> None:
    request = _request(tmp_path)

    with pytest.raises(subject.ForecastCampaignError, match="committed source input"):
        _compile(request, tmp_path)


@pytest.mark.parametrize(
    "field",
    (
        "dataset.contract",
        "dataset.run_template",
        "models.catalog",
    ),
)
def test_request_rejects_non_string_source_paths_as_campaign_errors(
    tmp_path: pathlib.Path, field: str
) -> None:
    path = _request(tmp_path)
    value = yaml.safe_load(path.read_text())
    section, name = field.split(".")
    value["forecast_campaign"][section][name] = []
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    with pytest.raises(subject.ForecastCampaignError, match="path string"):
        subject.load_forecast_campaign_request(path)


def test_child_stage_ids_are_unambiguous_above_two_digit_indexes() -> None:
    first = subject._child_stage_id(0, "fresh", 10, 101)  # noqa: SLF001
    second = subject._child_stage_id(0, "fresh", 101, 1)  # noqa: SLF001
    assert first != second


def test_compile_applies_dataset_frequency_to_validation_window(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _request(tmp_path)
    monkeypatch.setattr(subject, "_git_source_identity", lambda _path: {})
    weekly = SimpleNamespace(panel=lambda _path: SimpleNamespace(frequency="W"))
    monkeypatch.setattr(subject, "_load_dataset_contract_bytes", lambda *_args: weekly)
    with pytest.raises(subject.ForecastCampaignError, match="at dataset frequency"):
        _compile(path, tmp_path)


def test_peak_rss_tolerates_a_child_that_exits_before_psutil_attaches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ProcessGone(Exception):
        pass

    class MissingProcess:
        def __init__(self, _pid: int) -> None:
            raise ProcessGone

    child = SimpleNamespace(pid=123, wait=lambda: 0)
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(Process=MissingProcess, Error=ProcessGone),
    )
    assert subject._peak_rss(child) == 0  # noqa: SLF001


def test_capability_matrix_never_falls_back_between_initialization_lifecycles() -> None:
    candidates = {item.id: item for item in discover_candidates(ROOT / "configs/models")}
    custom = candidates["mft"]
    baseline = candidates["seasonal_naive"]
    assert (
        subject.initialization_capability(custom, "weights_only_finetune")["status"] == "supported"
    )
    assert (
        subject.initialization_capability(custom, "full_state_continuation")["status"]
        == "not_compatible"
    )
    assert (
        subject.initialization_capability(baseline, "weights_only_finetune")["status"]
        == "not_compatible"
    )


def test_compiler_expands_only_supported_arms_and_isolates_final_label_argument(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _request(tmp_path)
    monkeypatch.setattr(
        subject,
        "_git_source_identity",
        lambda _path: {"git_commit": "a" * 40, "request_path": "request.yaml"},
    )
    compiled = _compile(path, tmp_path)
    resolved = json.loads(compiled.resolved_path.read_text())
    assert [item["model_id"] for item in resolved["source"]["model_profiles"]] == [
        "seasonal_naive",
        "croston",
    ]
    assert all(len(item["sha256"]) == 64 for item in resolved["source"]["model_profiles"])
    validations = [item for item in resolved["children"] if item["kind"] == "validation"]
    assert len(validations) == 4
    assert {item["initialization"] for item in validations} == {"fresh"}
    incompatible = [
        item
        for item in resolved["initialization_capabilities"]
        if item["initialization"] == "weights_only_finetune"
    ]
    assert len(incompatible) == 2
    assert all(item["status"] == "not_compatible" for item in incompatible)

    plan = yaml.safe_load(compiled.plan_path.read_text())
    stages = plan["campaign"]["stages"]
    score_input = str(compiled.score_input_path)
    final_score_path = str((tmp_path / "final-score").resolve())
    assert final_score_path not in compiled.resolved_path.read_text()
    assert final_score_path not in (compiled.directory / "request-source.yaml").read_text()
    assert final_score_path in compiled.score_input_path.read_text()
    assert score_input in next(item for item in stages if item["id"] == "final_score")["command"]
    assert all(score_input not in item["command"] for item in stages if item["id"] != "final_score")
    assert all(
        final_score_path not in " ".join(item["command"])
        for item in stages
        if item["id"] != "final_score"
    )
    assert [item["id"] for item in stages[-4:]] == [
        "aggregate_selection",
        "final_fit",
        "final_score",
        "replay_report",
    ]


def _make_single_model_benchmark(
    path: pathlib.Path, *, models: list[str] | None = None
) -> dict[str, object]:
    value = yaml.safe_load(path.read_text())
    value["schema_version"] = 5
    campaign = value["forecast_campaign"]
    campaign["purpose"] = "single_model_benchmark"
    for role in ("selection", "final_training", "final_score"):
        campaign["dataset"][role] = {"source_uri": f"local://{role}"}
    campaign["models"] = {
        "catalog": "configs/models",
        "candidates": [
            {"model_id": model, "parameters": {"mode": "defaults"}}
            for model in (models or ["seasonal_naive"])
        ],
    }
    campaign["selection"]["tie_break"] = list(subject._TIE_BREAK)  # noqa: SLF001
    campaign["temporal"]["validation_origins"] = ["2024-01-21"]
    campaign["seeds"] = [0]
    campaign["initialization_arms"] = ["fresh"]
    return value


def test_single_model_benchmark_is_one_validation_plus_four_governed_handoffs(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _request(
        tmp_path,
        models=["seasonal_naive"],
        arms=["fresh"],
    )
    value = _make_single_model_benchmark(path)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(
        subject,
        "_git_source_identity",
        lambda _path: {"git_commit": "a" * 40, "request_path": "request.yaml"},
    )
    compiled = _compile(path, tmp_path)
    stages = yaml.safe_load(compiled.plan_path.read_text())["campaign"]["stages"]
    assert [item["id"] for item in stages] == [
        "v0fs0o0",
        "aggregate_selection",
        "final_fit",
        "final_score",
        "replay_report",
    ]


@pytest.mark.parametrize("metric", ["mae", "wape"])
def test_fixed_budget_benchmark_rejects_unwired_selection_metric(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, metric: str
) -> None:
    path = _request(tmp_path, models=["sft"], arms=["fresh"])
    value = _make_single_model_benchmark(path, models=["sft"])
    value["forecast_campaign"]["selection"]["metric"] = metric
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(
        subject, "_git_source_identity",
        lambda _path: {"git_commit": "a" * 40, "request_path": "request.yaml"},
    )
    with pytest.raises(subject.ForecastCampaignError, match="RMSE or WRMSSE"):
        _compile(path, tmp_path)


@pytest.mark.parametrize(
    "adapter",
    ["sft", "hmfnet", "pact"],
)
def test_revision_selection_budget_is_authenticated_and_final_refit_is_verified(
    tmp_path: pathlib.Path,
    adapter: str,
) -> None:
    selection_checkpoint = tmp_path / "selection.pt"
    training = {
        "config": {
            "epochs": 50_000,
            "patience": 10,
            "validation_every_epochs": 1,
            "selection_metric": "auto",
            "early_stopping_absolute_min_delta": 1e-6,
            "early_stopping_relative_min_delta": 0.0,
        },
        "epochs_per_phase": 50_000,
        "steps_per_epoch": 1,
        "selected_updates": 17,
        "selection_metric": "wrmsse",
        "validation_origins": ("2024-01-21",),
        "training_contract": {
            "mode": "fixed_validation",
            "validation_origin": "2024-01-21",
            "fixed_update_budget": None,
        },
        "validation": [
            {"update": update, "mean_wrmsse": 1.0 / update if update <= 17 else 1.0}
            for update in range(1, 28)
        ],
        "validation_slots": [
            {"update": update, "covered_series": 2, "trainable_series": 2,
             "evaluated": True}
            for update in range(1, 28)
        ],
    }
    torch.save({"training": training}, selection_checkpoint)
    result = {
        "adapter": adapter,
        "origin": "2024-01-21",
        "budget": {
            "maximum_epochs": 50_000, "patience": 10,
            "early_stopping_absolute_min_delta": 1e-6,
            "early_stopping_relative_min_delta": 0.0,
        },
        "checkpoint": {
            "path": str(selection_checkpoint),
            "sha256": subject._sha_file(selection_checkpoint),  # noqa: SLF001
        },
    }
    assert subject._selected_update_budget(result) == 17  # noqa: SLF001
    delayed = {
        **training,
        "selected_updates": 2,
        "validation": [
            {"update": update, "mean_wrmsse": 0.5 if update == 2 else 1.0}
            for update in range(2, 13)
        ],
        "validation_slots": [
            {"update": 1, "covered_series": 1, "trainable_series": 2,
             "evaluated": False},
            *[
                {"update": update, "covered_series": 2, "trainable_series": 2,
                 "evaluated": True}
                for update in range(2, 13)
            ],
        ],
    }
    torch.save({"training": delayed}, selection_checkpoint)
    result["checkpoint"]["sha256"] = subject._sha_file(selection_checkpoint)  # noqa: SLF001
    assert subject._selected_update_budget(result) == 2  # noqa: SLF001
    torch.save({"training": training}, selection_checkpoint)
    result["checkpoint"]["sha256"] = subject._sha_file(selection_checkpoint)  # noqa: SLF001
    for changed in (
        {**result, "budget": {**result["budget"], "early_stopping_absolute_min_delta": 0.1}},
        {**result, "budget": {**result["budget"], "patience": 9}},
    ):
        with pytest.raises(subject.ForecastCampaignError, match="selection"):
            subject._selected_update_budget(changed, "wrmsse")  # noqa: SLF001
    with pytest.raises(subject.ForecastCampaignError, match="selection"):
        subject._selected_update_budget(result, "rmse")  # noqa: SLF001
    for altered in (
        {**training, "validation": []},
        {**training, "validation": training["validation"][:17]},
        {**delayed, "validation_slots": delayed["validation_slots"][1:]},
        {**training, "config": {**training["config"], "validation_every_epochs": 2},
         "selected_updates": 2,
         "validation": [
             {"update": update, "mean_wrmsse": 0.5 if update == 2 else 1.0}
             for update in range(2, 24, 2)
         ],
         "validation_slots": [
             {"update": update, "covered_series": 2, "trainable_series": 2,
              "evaluated": True}
             for update in range(2, 24, 2)
         ]},
        {**training, "refit_optimizer_steps": 999},
        {**training, "validation": [{"update": 16, "mean_wrmsse": 0.9}]},
        {**training, "validation": [
            {"update": 17, "mean_wrmsse": 0.9},
            {"update": 18, "mean_wrmsse": 0.8},
        ]},
        {**training, "validation_origins": ("2024-01-20",)},
        {**training, "training_contract": {
            **training["training_contract"], "validation_origin": "2024-01-20",
        }},
    ):
        torch.save({"training": altered}, selection_checkpoint)
        result["checkpoint"]["sha256"] = subject._sha_file(  # noqa: SLF001
            selection_checkpoint
        )
        with pytest.raises(subject.ForecastCampaignError, match="selection|validation witness|slot cadence"):
            subject._selected_update_budget(result)  # noqa: SLF001
    torch.save({"training": training}, selection_checkpoint)
    result["checkpoint"]["sha256"] = subject._sha_file(selection_checkpoint)  # noqa: SLF001

    final_checkpoint = tmp_path / "final.pt"
    final_training = {
        "config": {"epochs": 50_000},
        "epochs_per_phase": 50_000,
        "steps_per_epoch": 1,
        "selected_updates": 17,
        "refit_optimizer_steps": 17,
        "budget_policy": "sealed_fixed_update_budget",
        "validation_origins": (),
        "training_contract": {
            "mode": "fixed_refit",
            "validation_origin": None,
            "fixed_update_budget": 17,
        },
        "validation": [],
        "validation_slots": [],
    }
    torch.save({"training": final_training}, final_checkpoint)
    final_run = tmp_path / "final-run"
    final_metrics = (
        final_run / "attempts" / adapter / "attempt" / "metrics" / "epochs.jsonl"
    )
    final_metrics.parent.mkdir(parents=True)
    final_metrics.write_text("".join(
        json.dumps({"phase": "refit", "update": update,
                    "optimizer_steps": update}) + "\n"
        for update in range(1, 18)
    ))
    final_result = {
        "adapter": adapter,
        "model_id": adapter,
        "run_path": str(final_run),
        "training_metrics_sha256": subject._sha_file(final_metrics),  # noqa: SLF001
        "budget": {"maximum_epochs": 50_000},
        "checkpoint": {
            "path": str(final_checkpoint),
            "sha256": subject._sha_file(final_checkpoint),  # noqa: SLF001
        },
    }
    subject._verify_fixed_refit_budget(final_result, 17)  # noqa: SLF001
    final_training["config"] = {"epochs": 1}
    final_training["epochs_per_phase"] = 1
    final_result["budget"] = {"maximum_epochs": 1}
    torch.save({"training": final_training}, final_checkpoint)
    final_result["checkpoint"]["sha256"] = subject._sha_file(final_checkpoint)  # noqa: SLF001
    subject._verify_fixed_refit_budget(final_result, 17)  # noqa: SLF001
    final_training["config"] = {"epochs": 50_000}
    final_training["epochs_per_phase"] = 50_000
    final_result["budget"] = {"maximum_epochs": 50_000}
    torch.save({"training": final_training}, final_checkpoint)
    final_result["checkpoint"]["sha256"] = subject._sha_file(final_checkpoint)  # noqa: SLF001
    for rows in (
        [json.dumps({"phase": "refit", "update": update,
                     "optimizer_steps": update}) for update in range(1, 17)],
        [json.dumps({"phase": "refit", "update": update,
                     "optimizer_steps": update}) for update in range(1, 19)],
        [json.dumps({"phase": "refit", "update": update,
                     **({} if update == 17 else {"optimizer_steps": update})})
         for update in range(1, 18)],
        [json.dumps({"phase": "refit", "update": update,
                     "optimizer_steps": update,
                     **({"validation_slot": {"update": update}} if update == 17 else {})})
         for update in range(1, 18)],
    ):
        final_metrics.write_text("\n".join(rows) + "\n")
        final_result["training_metrics_sha256"] = subject._sha_file(final_metrics)  # noqa: SLF001
        with pytest.raises(subject.ForecastCampaignError, match="refit journal"):
            subject._verify_fixed_refit_budget(final_result, 17)  # noqa: SLF001
    final_metrics.write_text("".join(
        json.dumps({"phase": "refit", "update": update,
                    "optimizer_steps": update}) + "\n"
        for update in range(1, 18)
    ))
    final_result["training_metrics_sha256"] = subject._sha_file(final_metrics)  # noqa: SLF001
    with pytest.raises(subject.ForecastCampaignError, match="sealed update budget"):
        subject._verify_fixed_refit_budget(final_result, 16)  # noqa: SLF001
    for changed in (
        {"refit_optimizer_steps": 16},
        {"refit_optimizer_steps": None},
        {"budget_policy": "validation_selected_updates"},
        {"epochs_per_phase": 49_999},
        {"config": {"epochs": 49_999}},
        {"validation_origins": ("2024-01-21",)},
        {"validation_slots": [{"update": 1, "covered_series": 2,
                               "trainable_series": 2, "evaluated": True}]},
        {"training_contract": {
            **final_training["training_contract"], "validation_origin": "2024-01-21",
        }},
    ):
        torch.save({"training": {**final_training, **changed}}, final_checkpoint)
        final_result["checkpoint"]["sha256"] = subject._sha_file(final_checkpoint)  # noqa: SLF001
        with pytest.raises(subject.ForecastCampaignError, match="ceiling|sealed update budget"):
            subject._verify_fixed_refit_budget(final_result, 17)  # noqa: SLF001
    with pytest.raises(
        subject.ForecastCampaignError, match="cannot prove the sealed update budget"
    ):
        subject._verify_fixed_refit_budget({**final_result, "adapter": "unknown_adapter"}, 17)  # noqa: SLF001


def test_rolling_final_refit_rejects_counter_only_journal(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    model_id = "sft"
    run = tmp_path / "run"
    metrics = run / "attempts" / model_id / "attempt" / "metrics" / "epochs.jsonl"
    metrics.parent.mkdir(parents=True)
    coverage = [
        {"epoch": update, "proposed_series": 1, "minimum_row_proposals": 1,
         "maximum_row_proposals": 1, "informative_series": 1,
         "minimum_informative_visits": 1, "maximum_informative_visits": 1}
        for update in (1, 2)
    ]
    rows = [
        {"phase": "refit", "update": update, "optimizer_steps": update,
         "origin": "2024-01-01", "sampled_series_indices": [0],
         "origin_sampling": (
             "coverage_curriculum" if update == 1 else "uniform_usable_origins"
         ),
         "likelihood_series_in_batch": 1,
         "likelihood_informative_series_in_batch": 1,
         "epoch_coverage": coverage[update - 1]}
        for update in (1, 2)
    ]
    metrics.write_text("".join(json.dumps(row) + "\n" for row in rows))
    checkpoint = tmp_path / "final.pt"
    torch.save({"training": {
        "config": {"epochs": 1, "seed": 0, "factor_rank": 0, "batch_size": 1},
        "epochs_per_phase": 1, "steps_per_epoch": 1, "selected_updates": 2,
        "refit_optimizer_steps": 2, "refit_series_count": 1,
        "refit_training_origins": ("2024-01-01",),
        "phase_coverage": {"refit": {
            "epochs": coverage, "series_seen": 1,
            "informative_series_visits": 2,
            "minimum_visits_per_series": 2, "maximum_visits_per_series": 2,
            "minimum_row_proposals": 2, "maximum_row_proposals": 2,
        }},
        "budget_policy": "sealed_fixed_update_budget",
        "training_contract": {"mode": "fixed_refit", "validation_origin": None,
                              "fixed_update_budget": 2},
        "validation": [], "validation_slots": [], "validation_origins": (),
    }}, checkpoint)
    result = {
        "adapter": model_id, "model_id": model_id, "run_path": str(run),
        "seed": 0, "budget": {"maximum_epochs": 1},
        "checkpoint": {"path": str(checkpoint),
                       "sha256": subject._sha_file(checkpoint)},  # noqa: SLF001
        "training_metrics_sha256": subject._sha_file(metrics),  # noqa: SLF001
    }
    monkeypatch.setattr(
        subject, "_governed_rolling_selection_origins",
        lambda *_args, **_kwargs: ("2024-01-01",),
    )
    monkeypatch.setattr(
        subject, "_rolling_selection_objective_masks",
        lambda *_args: {"2024-01-01": np.ones((1, 28), dtype=bool)},
    )
    subject._verify_fixed_refit_budget(  # noqa: SLF001
        result, 2, batch_size=1, resolved={}
    )
    saved = torch.load(checkpoint, weights_only=True)
    saved["training"]["phase_coverage"]["refit"]["informative_series_visits"] = 3
    torch.save(saved, checkpoint)
    result["checkpoint"]["sha256"] = subject._sha_file(checkpoint)  # noqa: SLF001
    with pytest.raises(subject.ForecastCampaignError, match="population or cadence"):
        subject._verify_fixed_refit_budget(  # noqa: SLF001
            result, 2, batch_size=1, resolved={}
        )
    saved["training"]["phase_coverage"]["refit"]["informative_series_visits"] = 2
    torch.save(saved, checkpoint)
    result["checkpoint"]["sha256"] = subject._sha_file(checkpoint)  # noqa: SLF001
    metrics.write_text("".join(json.dumps({
        "phase": "refit", "update": update, "optimizer_steps": update,
    }) + "\n" for update in (1, 2)))
    result["training_metrics_sha256"] = subject._sha_file(metrics)  # noqa: SLF001
    with pytest.raises(subject.ForecastCampaignError, match="refit sampler"):
        subject._verify_fixed_refit_budget(  # noqa: SLF001
            result, 2, batch_size=1, resolved={}
        )


@pytest.mark.parametrize("version", ("v2", "v3", "v5"))
def test_rolling_final_refit_rejects_incomplete_replayed_population(
    version: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    model_id = REVISION_MODELS[version]
    masks = {
        "2024-01-01": np.array([[True], [False]], dtype=bool),
        "2024-01-02": np.array([[False], [True]], dtype=bool),
    }
    rng = np.random.default_rng(0)
    sampled = rng.permutation(2).tolist()
    origin = tuple(masks)[int(rng.choice(np.array([0, 1])))]
    epoch = {
        "epoch": 1, "proposed_series": 2, "minimum_row_proposals": 1,
        "maximum_row_proposals": 1, "informative_series": 1,
        "minimum_informative_visits": 0, "maximum_informative_visits": 1,
    }
    row = {
        "phase": "refit", "update": 1, "optimizer_steps": 1, "origin": origin,
        "sampled_series_indices": sampled, "origin_sampling": "coverage_curriculum",
        "likelihood_series_in_batch": 2,
        "likelihood_informative_series_in_batch": 1,
        "epoch_coverage": epoch,
    }
    coverage = {
        "epochs": [epoch], "series_seen": 1, "informative_series_visits": 1,
        "minimum_visits_per_series": 0, "maximum_visits_per_series": 1,
        "minimum_row_proposals": 1, "maximum_row_proposals": 1,
    }
    run = tmp_path / "run"
    metrics = run / "attempts" / model_id / "attempt" / "metrics" / "epochs.jsonl"
    metrics.parent.mkdir(parents=True)
    metrics.write_text(json.dumps(row) + "\n", encoding="utf-8")
    checkpoint = tmp_path / "final.pt"
    torch.save({"training": {
        "config": {"epochs": 1, "seed": 0, "factor_rank": 0, "batch_size": 2},
        "epochs_per_phase": 1, "steps_per_epoch": 1, "selected_updates": 1,
        "refit_optimizer_steps": 1, "refit_series_count": 2,
        "refit_training_origins": tuple(masks), "phase_coverage": {"refit": coverage},
        "budget_policy": "sealed_fixed_update_budget",
        "training_contract": {"mode": "fixed_refit", "validation_origin": None,
                              "fixed_update_budget": 1},
        "validation": [], "validation_slots": [], "validation_origins": (),
    }}, checkpoint)
    result = {
        "adapter": model_id, "model_id": model_id, "run_path": str(run),
        "seed": 0, "budget": {"maximum_epochs": 1},
        "checkpoint": {"path": str(checkpoint),
                       "sha256": subject._sha_file(checkpoint)},  # noqa: SLF001
        "training_metrics_sha256": subject._sha_file(metrics),  # noqa: SLF001
    }
    monkeypatch.setattr(
        subject, "_governed_rolling_selection_origins", lambda *_args, **_kwargs: tuple(masks)
    )
    monkeypatch.setattr(subject, "_rolling_selection_objective_masks", lambda *_args: masks)
    with pytest.raises(subject.ForecastCampaignError, match="population or cadence"):
        subject._verify_fixed_refit_budget(  # noqa: SLF001
            result, 1, batch_size=2, resolved={}
        )


def test_tft_refit_uses_selected_updates_beyond_final_epoch_field_and_checks_log(
    tmp_path: pathlib.Path,
) -> None:
    checkpoint = tmp_path / "final.pt"
    forecast_record = tmp_path / "forecast.json"
    forecast_record.write_text('{"n_series": 3}\n')
    epoch_log = [
        {"epoch": epoch, "updates": min(epoch * 2, 5), "train_loss": 1.0,
         "val_loss": None, "val_loss_dtype": None, "elapsed_seconds": 0.1}
        for epoch in range(1, 4)
    ]
    training = {
        "maximum_epochs": 1, "steps_per_epoch": 2, "n_series": 3,
        "selected_updates": 5, "selected_epoch": 3,
        "refit_optimizer_steps": 5, "validation_origin": None,
        "validation": [], "epoch_log": epoch_log,
        "training_contract": {"mode": "fixed_refit", "fixed_update_budget": 5},
    }
    torch.save({"training": training}, checkpoint)
    result = {
        "adapter": "tft_q", "budget": {"maximum_epochs": 1},
        "checkpoint": {"path": str(checkpoint),
                       "sha256": subject._sha_file(checkpoint)},  # noqa: SLF001
        "forecast_record_path": str(forecast_record),
        "forecast_record_sha256": subject._sha_file(forecast_record),  # noqa: SLF001
    }
    subject._verify_fixed_refit_budget(result, 5, batch_size=2)  # noqa: SLF001
    for changed in (
        {**training, "epoch_log": [{**row, "updates": 4} if row["epoch"] == 3 else row
                                    for row in epoch_log]},
        {**training, "epoch_log": epoch_log[:1]},
        {**training, "selected_epoch": 2},
        {**training, "n_series": 2},
    ):
        torch.save({"training": changed}, checkpoint)
        result["checkpoint"]["sha256"] = subject._sha_file(checkpoint)  # noqa: SLF001
        with pytest.raises(subject.ForecastCampaignError, match="refit epoch telemetry"):
            subject._verify_fixed_refit_budget(result, 5, batch_size=2)  # noqa: SLF001
    too_short = {
        **training,
        "selected_updates": 1,
        "refit_optimizer_steps": 1,
        "selected_epoch": 1,
        "training_contract": {"mode": "fixed_refit", "fixed_update_budget": 1},
        "epoch_log": [{**epoch_log[0], "updates": 1}],
    }
    torch.save({"training": too_short}, checkpoint)
    result["checkpoint"]["sha256"] = subject._sha_file(checkpoint)  # noqa: SLF001
    with pytest.raises(subject.ForecastCampaignError, match="refit epoch telemetry"):
        subject._verify_fixed_refit_budget(result, 1, batch_size=2)  # noqa: SLF001


def test_single_model_selection_requires_one_unique_fixed_update_budget() -> None:
    resolved = {"purpose": "single_model_benchmark"}
    chosen = {"candidate_id": "candidate", "initialization": "fresh"}
    with pytest.raises(subject.ForecastCampaignError, match="no unique child budget"):
        subject._locked_selected_update_budget(resolved, chosen, [])  # noqa: SLF001

    duplicate = [
        {"candidate_id": "candidate", "initialization": "fresh"},
        {"candidate_id": "candidate", "initialization": "fresh"},
    ]
    with pytest.raises(subject.ForecastCampaignError, match="no unique child budget"):
        subject._locked_selected_update_budget(resolved, chosen, duplicate)  # noqa: SLF001


@pytest.mark.parametrize(
    ("external_metric", "frontier_metric"),
    (("wrmsse", "wrmsse"), ("mae", "rmse"), ("wape", "rmse")),
)
def test_research_selection_locks_lower_median_of_verified_update_frontiers(
    external_metric: str,
    frontier_metric: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved = {"purpose": "research_campaign", "selection": {"metric": external_metric},
                "compute": {"batch_size": 2}}
    chosen = {"candidate_id": "candidate", "initialization": "fresh"}
    results = [
        {**chosen, "adapter": "sft", "model_id": "v2",
         "selected": updates, "parameter_resolution": {"effective_parameters": {}}}
        for updates in (9, 3, 5, 7)
    ]
    from types import SimpleNamespace

    monkeypatch.setattr(
        subject, "_campaign_candidate",
        lambda *_args: SimpleNamespace(
            implementation_config_bytes=b'{"validation_every_epochs":1}'
        ),
    )
    selected_metrics: list[str] = []

    def selected_budget(result: dict, metric: str, *_args: object, **_kwargs: object) -> int:
        selected_metrics.append(metric)
        return result["selected"]

    monkeypatch.setattr(subject, "_selected_update_budget", selected_budget)
    monkeypatch.setattr(subject, "_governed_rolling_selection_origins", lambda *_args: ("day",))
    monkeypatch.setattr(subject, "_rolling_selection_objective_masks", lambda *_args: {"day": True})
    verified: list[tuple[int, str]] = []
    monkeypatch.setattr(
        subject, "_verify_rolling_selection_journal",
        lambda result, metric, *_args, **_kwargs: verified.append((result["selected"], metric)),
    )
    assert subject._locked_selected_update_budget(resolved, chosen, results) == 5  # noqa: SLF001
    assert selected_metrics == [frontier_metric] * 4
    assert verified == [(updates, frontier_metric) for updates in (9, 3, 5, 7)]


def test_final_result_rechecks_locked_refit_at_scoring_and_replay_boundary(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This fixture isolates refit semantics; journal authentication is exercised separately.
    monkeypatch.setattr(subject, "_load_completed_json",
                        lambda path: subject._load_json_with_identity(path, "fixture output"))
    lock_path = tmp_path / "selection-lock.json"
    lock_path.write_text("{}\n")
    lock = {"chosen": {"selected_update_budget": 17}}
    result = {"selection_lock_sha256": subject._sha_file(lock_path)}  # noqa: SLF001
    monkeypatch.setattr(subject, "_resolved_final_child", lambda *_args: {})
    monkeypatch.setattr(
        subject, "_validated_child_result", lambda *_args, **_kwargs: result
    )
    calls = []
    monkeypatch.setattr(
        subject, "_verify_fixed_refit_budget",
        lambda child, updates, **kwargs: calls.append((child, updates, kwargs)),
    )
    assert subject._validated_locked_final_result(  # noqa: SLF001
        {"compute": {"batch_size": 2}}, lock, lock_path, tmp_path / "child-result.json"
    ) is result
    assert calls == [(result, 17, {"batch_size": 2, "resolved": {"compute": {"batch_size": 2}}})]
    result["selection_lock_sha256"] = "0" * 64
    with pytest.raises(subject.ForecastCampaignError, match="selection lock"):
        subject._validated_locked_final_result(  # noqa: SLF001
            {"compute": {"batch_size": 2}}, lock, lock_path, tmp_path / "child-result.json"
        )


def test_rolling_selection_accepts_governed_two_epoch_validation_cadence(
    tmp_path: pathlib.Path,
) -> None:
    import numpy as np

    model_id = "sft"
    run = tmp_path / "run"
    metrics = run / "attempts" / model_id / "attempt" / "metrics" / "epochs.jsonl"
    metrics.parent.mkdir(parents=True)
    epoch_coverage = [
        {"epoch": update, "proposed_series": 1, "minimum_row_proposals": 1,
         "maximum_row_proposals": 1, "informative_series": 1,
         "minimum_informative_visits": 1, "maximum_informative_visits": 1}
        for update in range(1, 4)
    ]
    slots = [
        {"update": update, "covered_series": 1, "trainable_series": 1,
         "evaluated": True}
        for update in (2, 3)
    ]
    validation = [
        {"update": update, "mean_wrmsse": score}
        for update, score in ((2, 0.5), (3, 1.0))
    ]
    journal = []
    for update in range(1, 4):
        row = {
            "phase": "selection", "update": update, "optimizer_steps": update,
            "epoch": update,
            "origin": "2024-01-01", "origin_sampling": (
                "coverage_curriculum" if update == 1 else "uniform_usable_origins"
            ),
            "sampled_series_indices": [0], "likelihood_series_in_batch": 1,
            "likelihood_informative_series_in_batch": 1,
            "epoch_coverage": epoch_coverage[update - 1],
        }
        if update > 1:
            row["validation_slot"] = slots[update - 2]
            row["validation"] = {"2024-01-21": {
                "wrmsse": validation[update - 2]["mean_wrmsse"],
                "validation_nll": None,
            }}
            row["validation_summary"] = {
                "update": update, "epoch": update, "metric_name": "wrmsse",
                "metric": validation[update - 2]["mean_wrmsse"], "nll": None,
            }
        journal.append(row)
    metrics.write_text("".join(json.dumps(row) + "\n" for row in journal))
    forecast_record = tmp_path / "forecast.json"
    forecast_record.write_text('{"n_series":1}\n', encoding="utf-8")
    checkpoint = tmp_path / "model.pt"
    training = {
        "config": {"epochs": 3, "patience": 10, "validation_every_epochs": 2,
                   "seed": 0, "factor_rank": 0, "batch_size": 1,
                   "selection_metric": "wrmsse",
                   "early_stopping_absolute_min_delta": 1e-6,
                   "early_stopping_relative_min_delta": 0.0},
        "epochs_per_phase": 3, "steps_per_epoch": 1, "selected_updates": 2,
        "selection_metric": "wrmsse", "validation_origins": ("2024-01-21",),
        "selection_training_origins": ("2024-01-01",),
        "training_contract": {"mode": "fixed_validation",
                              "validation_origin": "2024-01-21",
                              "fixed_update_budget": None},
        "validation": validation, "validation_slots": slots,
        "phase_coverage": {"selection": {
            "epochs": epoch_coverage, "series_seen": 1,
            "informative_series_visits": 3, "minimum_visits_per_series": 3,
            "maximum_visits_per_series": 3, "minimum_row_proposals": 3,
            "maximum_row_proposals": 3,
        }},
        "excluded_series_count": 0,
    }
    torch.save({"training": training}, checkpoint)
    result = {
        "adapter": model_id, "model_id": model_id, "seed": 0,
        "run_path": str(run),
        "training_metrics_sha256": subject._sha_file(metrics),  # noqa: SLF001
        "forecast_record_path": str(forecast_record),
        "forecast_record_sha256": subject._sha_file(forecast_record),  # noqa: SLF001
        "checkpoint": {"path": str(checkpoint),
                       "sha256": subject._sha_file(checkpoint)},  # noqa: SLF001
        "origin": "2024-01-21",
        "budget": {"maximum_epochs": 3, "patience": 10,
                   "early_stopping_absolute_min_delta": 1e-6,
                   "early_stopping_relative_min_delta": 0.0},
    }
    assert subject._selected_update_budget(  # noqa: SLF001
        result, "wrmsse", expected_validation_every=2
    ) == 2
    subject._verify_rolling_selection_journal(  # noqa: SLF001
        result, "wrmsse", 1, {"2024-01-01": np.ones((1, 28), dtype=bool)},
        expected_validation_every=2,
    )
    with pytest.raises(subject.ForecastCampaignError, match="selection metric controls"):
        subject._selected_update_budget(result, "wrmsse")  # noqa: SLF001


def test_locked_rolling_selection_binds_validation_cadence_to_source_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = {
        "adapter": "sft", "model_id": "sft",
        "candidate_id": "v2", "initialization": "fresh",
        "parameter_resolution": {"effective_parameters": {}},
    }
    resolved = {
        "purpose": "single_model_benchmark",
        "selection": {"metric": "wrmsse"},
        "compute": {"batch_size": 2},
    }
    calls = []
    monkeypatch.setattr(subject, "_campaign_candidate", lambda *_args: SimpleNamespace(
        implementation_config_bytes=b'{"validation_every_epochs":2}'
    ))
    monkeypatch.setattr(subject, "_selected_update_budget", lambda *_args, **kwargs: (
        calls.append(("selection", kwargs["expected_validation_every"])) or 7
    ))
    monkeypatch.setattr(subject, "_governed_rolling_selection_origins", lambda *_args: (
        "2024-01-01",
    ))
    monkeypatch.setattr(subject, "_rolling_selection_objective_masks", lambda *_args: {
        "2024-01-01": True,
    })
    monkeypatch.setattr(subject, "_verify_rolling_selection_journal", lambda *_args, **kwargs: (
        calls.append(("journal", kwargs["expected_validation_every"]))
    ))
    assert subject._locked_selected_update_budget(  # noqa: SLF001
        resolved, {"candidate_id": "v2", "initialization": "fresh"}, [result]
    ) == 7
    assert calls == [("selection", 2), ("journal", 2)]


@pytest.mark.parametrize("model_id", [
    "sft", "hmfnet", "pact",
])
def test_rolling_selection_journal_rejects_coherently_rewritten_epoch_cadence(
    model_id: str,
    tmp_path: pathlib.Path,
) -> None:
    import numpy as np

    from autoforecast.revision_config import verify_validation_ledger

    run = tmp_path / "run"
    metrics = run / "attempts" / model_id / "attempt" / "metrics" / "epochs.jsonl"
    metrics.parent.mkdir(parents=True)
    slots = [
        {"update": update, "covered_series": 2, "trainable_series": 2,
         "evaluated": True}
        for update in range(1, 12)
    ]
    evaluations = [
        {"update": update, "mean_wrmsse": 0.5 if update == 1 else 1.0}
        for update in range(1, 12)
    ]
    epoch_coverage = [
        {"epoch": update, "proposed_series": 2,
         "minimum_row_proposals": 1, "maximum_row_proposals": 1,
         "informative_series": 2, "minimum_informative_visits": 1,
         "maximum_informative_visits": 1}
        for update in range(1, 12)
    ]
    rng = np.random.default_rng(0)
    journal = []
    for update in range(1, 12):
        sampled = rng.permutation(2).tolist()
        rng.choice(np.array([0]))
        journal.append({
            "phase": "selection", "update": update, "optimizer_steps": update,
            "epoch": update,
            "origin": "2024-01-01", "origin_sampling": (
                "coverage_curriculum" if update == 1 else "uniform_usable_origins"
            ),
            "sampled_series_indices": sampled, "likelihood_series_in_batch": 2,
            "likelihood_informative_series_in_batch": 2,
            "epoch_coverage": epoch_coverage[update - 1],
            "validation_slot": slots[update - 1],
            "validation": {"2024-01-21": {
                "wrmsse": evaluations[update - 1]["mean_wrmsse"],
                "validation_nll": None,
            }},
            "validation_summary": {
                "update": update, "epoch": update, "metric_name": "wrmsse",
                "metric": evaluations[update - 1]["mean_wrmsse"], "nll": None,
            },
        })
    metrics.write_text("".join(json.dumps(row) + "\n" for row in journal))
    forecast_record = tmp_path / "forecast.json"
    forecast_record.write_text('{"n_series":2}\n', encoding="utf-8")
    checkpoint = tmp_path / "model.pt"
    training = {
        "config": {"epochs": 50_000, "patience": 10,
                   "seed": 0, "factor_rank": 0, "batch_size": 2,
                   "train_windows": 1, "train_stride": 1, "encoder_days": 7,
                   "validation_every_epochs": 1, "selection_metric": "wrmsse",
                   "early_stopping_absolute_min_delta": 1e-6,
                   "early_stopping_relative_min_delta": 0.0},
        "epochs_per_phase": 50_000,
        "steps_per_epoch": 1,
        "selected_updates": 1,
        "selection_metric": "wrmsse",
        "validation_origins": ("2024-01-21",),
        "selection_training_origins": ("2024-01-01",),
        "training_contract": {"mode": "fixed_validation",
                              "validation_origin": "2024-01-21",
                              "fixed_update_budget": None},
        "validation": evaluations,
        "validation_slots": slots,
        "phase_coverage": {"selection": {
            "epochs": epoch_coverage, "series_seen": 2,
            "informative_series_visits": 22,
            "minimum_visits_per_series": 11, "maximum_visits_per_series": 11,
            "minimum_row_proposals": 11, "maximum_row_proposals": 11,
        }},
        "excluded_series_count": 0,
    }
    torch.save({"training": training}, checkpoint)
    result = {
        "adapter": model_id, "model_id": model_id, "seed": 0,
        "run_path": str(run),
        "training_metrics_sha256": subject._sha_file(metrics),  # noqa: SLF001
        "forecast_record_path": str(forecast_record),
        "forecast_record_sha256": subject._sha_file(forecast_record),  # noqa: SLF001
        "checkpoint": {"path": str(checkpoint),
                       "sha256": subject._sha_file(checkpoint)},  # noqa: SLF001
        "origin": "2024-01-21",
        "horizon": 20,
        "parameter_resolution": {"effective_parameters": {
            "training.train_windows": 1, "training.train_stride": 1,
            "model.encoder_days": 7,
        }},
        "budget": {"maximum_epochs": 50_000, "patience": 10,
                   "early_stopping_absolute_min_delta": 1e-6,
                   "early_stopping_relative_min_delta": 0.0},
    }
    verify_validation_ledger(
        [{"update": 1, "covered_series": 1, "trainable_series": 2,
          "evaluated": False}, *slots[1:3]], evaluations[1:3],
        metric="wrmsse", validation_interval_updates=1, budget_updates=50_000,
        last_update=3, trainable_series=2,
    )
    objective_masks = {"2024-01-01": np.ones((2, 28), dtype=bool)}
    governed = {"temporal": {"step_days": 1}}
    assert subject._governed_rolling_selection_origins(  # noqa: SLF001
        governed, result
    ) == ("2024-01-01",)
    assert subject._selected_update_budget(result, "wrmsse") == 1  # noqa: SLF001
    subject._verify_rolling_selection_journal(  # noqa: SLF001
        result, "wrmsse", 2, objective_masks
    )
    forged_rows = [{**journal[0], "validation_summary": {
        **journal[0]["validation_summary"], "metric": 0.1,
    }}, *journal[1:]]
    metrics.write_text("".join(json.dumps(row) + "\n" for row in forged_rows))
    result["training_metrics_sha256"] = subject._sha_file(metrics)  # noqa: SLF001
    forged_training = {**training, "validation": [
        {**evaluations[0], "mean_wrmsse": 0.1}, *evaluations[1:]
    ]}
    torch.save({"training": forged_training}, checkpoint)
    result["checkpoint"]["sha256"] = subject._sha_file(checkpoint)  # noqa: SLF001
    with pytest.raises(subject.ForecastCampaignError, match="validation summary"):
        subject._verify_rolling_selection_journal(  # noqa: SLF001
            result, "wrmsse", 2, objective_masks
        )
    metrics.write_text("".join(json.dumps(row) + "\n" for row in journal))
    result["training_metrics_sha256"] = subject._sha_file(metrics)  # noqa: SLF001
    torch.save({"training": training}, checkpoint)
    result["checkpoint"]["sha256"] = subject._sha_file(checkpoint)  # noqa: SLF001
    impossible = {"2024-01-01": np.array([[True] * 28, [False] * 28])}
    with pytest.raises(subject.ForecastCampaignError, match="journal sampler"):
        subject._verify_rolling_selection_journal(  # noqa: SLF001
            result, "wrmsse", 2, impossible
        )
    omitted_rows = [
        {**row, "sampled_series_indices": [0], "likelihood_series_in_batch": 1,
         "likelihood_informative_series_in_batch": 1,
         "epoch_coverage": {
             **row["epoch_coverage"], "proposed_series": 1,
             "informative_series": 1,
         },
         "validation_slot": {**row["validation_slot"], "covered_series": 1,
                             "trainable_series": 1}}
        for row in journal
    ]
    omitted_training = {
        **training,
        "selection_training_origins": ("2024-01-01", "2023-12-31"),
        "excluded_series_count": 1,
        "validation_slots": [row["validation_slot"] for row in omitted_rows],
        "phase_coverage": {"selection": {
            **training["phase_coverage"]["selection"],
            "epochs": [row["epoch_coverage"] for row in omitted_rows],
            "series_seen": 1, "informative_series_visits": 11,
        }},
    }
    torch.save({"training": omitted_training}, checkpoint)
    result["checkpoint"]["sha256"] = subject._sha_file(checkpoint)  # noqa: SLF001
    metrics.write_text("".join(json.dumps(row) + "\n" for row in omitted_rows))
    result["training_metrics_sha256"] = subject._sha_file(metrics)  # noqa: SLF001
    full_masks = {
        "2024-01-01": np.array([[True] * 28, [False] * 28]),
        "2023-12-31": np.array([[False] * 28, [True] * 28]),
    }
    with pytest.raises(subject.ForecastCampaignError, match="journal sampler|epoch coverage|validation coverage"):
        subject._verify_rolling_selection_journal(  # noqa: SLF001
            result, "wrmsse", 2, full_masks
        )
    with pytest.raises(subject.ForecastCampaignError, match="governed schedule"):
        subject._governed_rolling_selection_origins(governed, result)  # noqa: SLF001
    torch.save({"training": training}, checkpoint)
    result["checkpoint"]["sha256"] = subject._sha_file(checkpoint)  # noqa: SLF001
    metrics.write_text("".join(json.dumps(row) + "\n" for row in journal))
    result["training_metrics_sha256"] = subject._sha_file(metrics)  # noqa: SLF001
    forged_epoch = [{**journal[0], "epoch_coverage": {"epoch": 999}}, *journal[1:]]
    metrics.write_text("".join(json.dumps(row) + "\n" for row in forged_epoch))
    result["training_metrics_sha256"] = subject._sha_file(metrics)  # noqa: SLF001
    with pytest.raises(subject.ForecastCampaignError, match="epoch coverage"):
        subject._verify_rolling_selection_journal(  # noqa: SLF001
            result, "wrmsse", 2, objective_masks
        )
    metrics.write_text("".join(json.dumps(row) + "\n" for row in journal))
    result["training_metrics_sha256"] = subject._sha_file(metrics)  # noqa: SLF001
    for changed in (
        {**journal[0], "optimizer_steps": 0},
        {key: value for key, value in journal[0].items() if key != "optimizer_steps"},
    ):
        metrics.write_text("".join(
            json.dumps(changed if index == 0 else row) + "\n"
            for index, row in enumerate(journal)
        ))
        result["training_metrics_sha256"] = subject._sha_file(metrics)  # noqa: SLF001
        with pytest.raises(subject.ForecastCampaignError, match="committed journal"):
            subject._verify_rolling_selection_journal(  # noqa: SLF001
                result, "wrmsse", 2, objective_masks
            )
    metrics.write_text("".join(json.dumps(row) + "\n" for row in journal))
    result["training_metrics_sha256"] = subject._sha_file(metrics)  # noqa: SLF001
    rounded_type = {**training, "validation": [
        {**row, "mean_wrmsse": 1} if index == 1 else row
        for index, row in enumerate(evaluations)
    ]}
    torch.save({"training": rounded_type}, checkpoint)
    result["checkpoint"]["sha256"] = subject._sha_file(checkpoint)  # noqa: SLF001
    with pytest.raises(subject.ForecastCampaignError, match="sampler epoch cadence|epoch coverage|committed journal"):
        subject._verify_rolling_selection_journal(  # noqa: SLF001
            result, "wrmsse", 2, objective_masks
        )
    torch.save({"training": training}, checkpoint)
    result["checkpoint"]["sha256"] = subject._sha_file(checkpoint)  # noqa: SLF001
    doubled_slots = [{**slot, "update": slot["update"] * 2} for slot in slots]
    doubled_evaluations = [{**row, "update": row["update"] * 2} for row in evaluations]
    rewritten = {**training, "steps_per_epoch": 2, "selected_updates": 2,
                 "validation_slots": doubled_slots, "validation": doubled_evaluations}
    torch.save({"training": rewritten}, checkpoint)
    result["checkpoint"]["sha256"] = subject._sha_file(checkpoint)  # noqa: SLF001
    assert subject._selected_update_budget(result, "wrmsse") == 2  # noqa: SLF001
    with pytest.raises(subject.ForecastCampaignError, match="sampler epoch cadence|epoch coverage|committed journal"):
        subject._verify_rolling_selection_journal(  # noqa: SLF001
            result, "wrmsse", 2, objective_masks
        )
    coherent_journal = []
    for update in range(1, 23):
        row = {"phase": "selection", "update": update, "optimizer_steps": update,
               "epoch": (update + 1) // 2,
               "origin": "2024-01-01", "origin_sampling": "coverage_curriculum",
               "sampled_series_indices": [0, 1], "likelihood_series_in_batch": 2,
               "likelihood_informative_series_in_batch": 2}
        if update % 2 == 0:
            index = update // 2 - 1
            row.update({
                "epoch_coverage": {
                    **epoch_coverage[index],
                    "minimum_row_proposals": 2, "maximum_row_proposals": 2,
                    "minimum_informative_visits": 2,
                    "maximum_informative_visits": 2,
                },
                "validation_slot": doubled_slots[index],
                "validation": {"2024-01-21": {
                    "wrmsse": doubled_evaluations[index]["mean_wrmsse"],
                    "validation_nll": None,
                }},
                "validation_summary": {
                    "update": update, "epoch": (update + 1) // 2,
                    "metric_name": "wrmsse",
                    "metric": doubled_evaluations[index]["mean_wrmsse"],
                    "nll": None,
                },
            })
        coherent_journal.append(row)
    metrics.write_text("".join(json.dumps(row) + "\n" for row in coherent_journal))
    result["training_metrics_sha256"] = subject._sha_file(metrics)  # noqa: SLF001
    with pytest.raises(subject.ForecastCampaignError, match="epoch cadence|epoch coverage"):
        subject._verify_rolling_selection_journal(  # noqa: SLF001
            result, "wrmsse", 2, objective_masks
        )


def test_rolling_objective_masks_are_rebuilt_from_sealed_training_panel(
    tmp_path: pathlib.Path,
) -> None:
    import numpy as np

    attempt = tmp_path / "attempt"
    training_panel = attempt / "training-panel"
    _panel(training_panel, days=10)
    run = attempt / "run" / "group"
    run.mkdir(parents=True)
    result = {
        "run_path": str(run), "horizon": 2,
        "training_data_sha256": subject.directory_digest(
            training_panel, suffixes=(".parquet",)
        ),
        "adapter": "sft",
    }
    resolved = {"source": {"dataset_contract": "configs/datasets/synthetic_daily.yaml"}}
    masks = subject._rolling_selection_objective_masks(  # noqa: SLF001
        resolved, result, ("2024-01-05",)
    )
    assert np.array_equal(masks["2024-01-05"], np.ones((2, 2), dtype=bool))
    pq.write_table(pa.table({"series_id": ["a"], "date": [dt.date(2024, 1, 11)],
                             "target": [1.0]}), training_panel / "extra.parquet")
    with pytest.raises(subject.ForecastCampaignError, match="training panel changed"):
        subject._rolling_selection_objective_masks(  # noqa: SLF001
            resolved, result, ("2024-01-05",)
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("models", ["seasonal_naive", "croston"], "exactly one model"),
        ("seeds", [0, 1], "exactly one seed"),
        ("validation_origins", ["2024-01-14", "2024-01-21"], "exactly one validation"),
        ("initialization_arms", ["fresh", "weights_only_finetune"], "requires initialization"),
    ),
)
def test_single_model_benchmark_rejects_research_campaign_multipliers(
    tmp_path: pathlib.Path,
    field: str,
    value: object,
    message: str,
) -> None:
    path = _request(tmp_path, models=["seasonal_naive"], arms=["fresh"])
    request = _make_single_model_benchmark(path)
    campaign = request["forecast_campaign"]
    if field == "models":
        campaign["models"] = _make_single_model_benchmark(path, models=value)["forecast_campaign"][
            "models"
        ]
    elif field == "validation_origins":
        campaign["temporal"][field] = value
    else:
        campaign[field] = value
    path.write_text(yaml.safe_dump(request, sort_keys=False), encoding="utf-8")

    with pytest.raises(subject.ForecastCampaignError, match=message):
        subject.load_forecast_campaign_request(path)


@pytest.mark.parametrize(
    ("filename", "model_id"),
    (
        ("m5_sft_competition_benchmark.yaml", "sft"),
        ("m5_hmfnet_competition_benchmark.yaml", "hmfnet"),
    ),
)
def test_checked_in_competition_benchmarks_are_independent_single_model_requests(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    model_id: str,
) -> None:
    path = ROOT / "configs" / "autoforecast" / filename
    request = subject.load_forecast_campaign_request(path)
    assert request.purpose == "single_model_benchmark"
    assert request.model_ids == (model_id,)
    assert request.seeds == (0,)
    assert tuple(day.isoformat() for day in request.validation_origins) == ("2016-04-24",)
    assert request.initialization_arms == ("fresh",)
    assert request.validation_budget.maximum_epochs == 50000
    assert request.validation_budget.patience == 10
    assert request.compute.batch_size == 32
    for offset, role in enumerate(("selection", "final-training", "final-score")):
        _panel(tmp_path / role, target_offset=float(offset * 100))
    monkeypatch.setattr(
        subject,
        "_git_source_identity",
        lambda _path: {"git_commit": "a" * 40, "request_path": filename},
    )
    compiled = _compile(path, tmp_path)
    stages = yaml.safe_load(compiled.plan_path.read_text())["campaign"]["stages"]
    assert [stage["id"] for stage in stages] == [
        "v0fs0o0",
        "aggregate_selection",
        "final_fit",
        "final_score",
        "replay_report",
    ]


def test_sft_two_epoch_smoke_changes_only_name_epochs_and_model_tens(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_root = ROOT / "configs" / "autoforecast"
    full_path = config_root / "m5_sft_competition_benchmark.yaml"
    smoke_path = config_root / "m5_sft_competition_benchmark_2epoch_smoke.yaml"
    full = yaml.safe_load(full_path.read_text())
    smoke = yaml.safe_load(smoke_path.read_text())
    smoke_campaign = smoke["forecast_campaign"]
    assert smoke_campaign["name"] == "m5-sft-competition-benchmark-2epoch-smoke"
    smoke_campaign["name"] = full["forecast_campaign"]["name"]
    for name in ("initialization", "validation", "final_fit"):
        assert smoke_campaign["budgets"][name]["maximum_epochs"] == 2
        smoke_campaign["budgets"][name]["maximum_epochs"] = 50000
    full_values = full["forecast_campaign"]["models"]["candidates"][0]["parameters"]["values"]
    smoke_values = smoke_campaign["models"]["candidates"][0]["parameters"]["values"]
    reduced_model_keys = {
        "model.n_heads",
        "model.trunk_stages",
        "model.decoder_layers",
        "model.history_attention_layers",
        "model.commercial_attention_layers",
        "model.event_kernel_layers",
        "model.commercial_attention_heads",
    }
    assert {key for key, value in smoke_values.items() if value == 5} == reduced_model_keys
    for key in reduced_model_keys:
        assert full_values[key] == 10
        smoke_values[key] = 10
    assert smoke == full

    request = subject.load_forecast_campaign_request(smoke_path)
    assert request.purpose == "single_model_benchmark"
    assert request.validation_budget.maximum_epochs == 2
    assert request.final_fit_budget.maximum_epochs == 2
    for offset, role in enumerate(("selection", "final-training", "final-score")):
        _panel(tmp_path / role, target_offset=float(offset * 100))
    monkeypatch.setattr(
        subject,
        "_git_source_identity",
        lambda _path: {
            "git_commit": "a" * 40,
            "request_path": smoke_path.name,
        },
    )
    compiled = _compile(smoke_path, tmp_path)
    stages = yaml.safe_load(compiled.plan_path.read_text())["campaign"]["stages"]
    assert [stage["id"] for stage in stages] == [
        "v0fs0o0",
        "aggregate_selection",
        "final_fit",
        "final_score",
        "replay_report",
    ]


def test_hmfnet_two_epoch_smoke_matches_full_config_except_name_and_epochs() -> None:
    config_root = ROOT / "configs" / "autoforecast"
    full = yaml.safe_load((config_root / "m5_hmfnet_competition_benchmark.yaml").read_text())
    smoke = yaml.safe_load(
        (config_root / "m5_hmfnet_competition_benchmark_2epoch_smoke.yaml").read_text()
    )
    smoke_campaign = smoke["forecast_campaign"]
    assert smoke_campaign["name"] == "m5-hmfnet-competition-benchmark-2epoch-smoke"
    smoke_campaign["name"] = full["forecast_campaign"]["name"]
    for name in ("initialization", "validation", "final_fit"):
        assert smoke_campaign["budgets"][name]["maximum_epochs"] == 2
        smoke_campaign["budgets"][name]["maximum_epochs"] = 50000
    full_values = full["forecast_campaign"]["models"]["candidates"][0]["parameters"]["values"]
    smoke_values = smoke_campaign["models"]["candidates"][0]["parameters"]["values"]
    for name in ("model.n_heads", "model.trunk_stages"):
        assert full_values[name] == 5
        assert smoke_values[name] == 5
    assert smoke == full


def test_pact_competition_benchmark_uses_requested_full_run_contract() -> None:
    path = ROOT / "configs" / "autoforecast" / "m5_pact_competition_benchmark.yaml"
    campaign = yaml.safe_load(path.read_text())["forecast_campaign"]
    candidate = campaign["models"]["candidates"][0]
    values = candidate["parameters"]["values"]

    assert candidate["model_id"] == "pact"
    assert values["model.encoder_days"] == 371
    assert values["model.d_model"] == 64
    assert values["model.n_heads"] == 4
    assert values["model.trunk_stages"] == 4
    assert campaign["selection"]["metric"] == "wrmsse"
    assert campaign["selection"]["direction"] == "minimize"
    assert campaign["compute"]["batch_size"] == 32
    for name in ("initialization", "validation", "final_fit"):
        assert campaign["budgets"][name]["maximum_epochs"] == 50000
        assert campaign["budgets"][name]["patience"] == 10


@pytest.mark.parametrize(
    ("seeds", "origins"),
    (
        ([0, 1], ["2016-04-24"]),
        ([0], ["2016-03-27", "2016-04-24"]),
        ([0, 1], ["2016-03-27", "2016-04-24"]),
    ),
)
def test_tft_rejects_unsupported_fold_count_before_training(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
    seeds: list[int], origins: list[str],
) -> None:
    source = yaml.safe_load(
        (ROOT / "configs/autoforecast/m5_tft_loss_2epoch.yaml").read_text()
    )
    campaign = source["forecast_campaign"]
    campaign["seeds"] = seeds
    campaign["temporal"]["validation_origins"] = origins
    request = tmp_path / "request.yaml"
    request.write_text(yaml.safe_dump(source), encoding="utf-8")
    monkeypatch.setattr(
        subject, "_git_source_identity", lambda _path: {"git_commit": "a" * 40}
    )
    with pytest.raises(subject.ForecastCampaignError, match="one seed and one validation origin"):
        _compile(request, tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("contract", "configs/datasets/synthetic_daily.yaml"),
        ("metric", "mae"),
        ("validation_horizon", 14),
        ("final_test_horizon", 14),
        ("patience", 1),
        ("relative_min_delta", 0.01),
    ),
)
def test_tft_profile_rejects_non_governing_campaign_settings(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    source = yaml.safe_load(
        (ROOT / "configs/autoforecast/m5_tft_loss_2epoch.yaml").read_text()
    )
    campaign = source["forecast_campaign"]
    if field == "contract":
        campaign["dataset"]["contract"] = value
    elif field == "metric":
        campaign["selection"]["metric"] = value
    elif field == "patience":
        campaign["budgets"]["validation"]["patience"] = value
    elif field == "relative_min_delta":
        campaign["budgets"]["validation"]["early_stopping_relative_min_delta"] = value
    else:
        campaign["temporal"][field] = value
    request = tmp_path / "request.yaml"
    request.write_text(yaml.safe_dump(source), encoding="utf-8")
    monkeypatch.setattr(
        subject, "_git_source_identity", lambda _path: {"git_commit": "a" * 40}
    )
    with pytest.raises(subject.ForecastCampaignError, match="TFT-Q/TFT-HNB M5 requires"):
        _compile(request, tmp_path)


@pytest.mark.parametrize("role", ("selection", "final_training", "final_score"))
def test_tft_profile_requires_a_pinned_digest_for_every_m5_data_role(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, role: str
) -> None:
    source = yaml.safe_load(
        (ROOT / "configs/autoforecast/m5_tft_loss_2epoch.yaml").read_text()
    )
    del source["forecast_campaign"]["dataset"][role]["sha256"]
    request = tmp_path / "request.yaml"
    request.write_text(yaml.safe_dump(source), encoding="utf-8")
    monkeypatch.setattr(subject, "_git_source_identity", lambda _path: {"git_commit": "a" * 40})
    with pytest.raises(subject.ForecastCampaignError, match="pinned data-role digests"):
        _compile(request, tmp_path)


def test_pinned_data_role_rejects_different_parquet_bytes(tmp_path: pathlib.Path) -> None:
    intended = tmp_path / "intended"
    changed = tmp_path / "changed"
    _panel(intended)
    _panel(changed, target_offset=1.0)
    expected = subject.directory_digest(intended, suffixes=(".parquet",))
    input_role = subject.DataInput(None, "local://m5-public-selection-v1", expected)
    assert subject._data_identity(input_role, intended, "selection data")["sha256"] == expected  # noqa: SLF001
    with pytest.raises(subject.ForecastCampaignError, match="pinned SHA-256"):
        subject._data_identity(input_role, changed, "selection data")  # noqa: SLF001


def test_data_role_refuses_malformed_pinned_sha256() -> None:
    with pytest.raises(subject.ForecastCampaignError, match="lowercase SHA-256"):
        subject._data_input(  # noqa: SLF001
            {"source_uri": "local://m5-public-selection-v1", "sha256": "invalid"},
            "dataset.selection",
            schema_version=5,
        )


def test_tft_profile_alias_locks_refit_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        subject, "_selected_update_budget",
        lambda _result, _metric, _batch, _loss: 6,
    )
    chosen = {"candidate_id": "variant", "initialization": "fresh", "model_id": "my_tft"}
    result = {
        "candidate_id": "variant", "initialization": "fresh", "adapter": "tft_q"
    }
    resolved = {
        "purpose": "forecast_campaign",
        "selection": {"metric": "wrmsse"},
        "compute": {"batch_size": 2},
        "children": [{"kind": "validation", **result}],
    }
    assert subject._locked_selected_update_budget(
        resolved, chosen, [result]
    ) == 6
    chosen["model_id"] = "tft_q"
    result["adapter"] = "seasonal_naive"
    resolved["children"] = [{"kind": "validation", **result}]
    assert subject._locked_selected_update_budget(
        resolved, chosen, [result, result]
    ) is None


@pytest.mark.parametrize("loss_kind", ("quantile", "v3_hurdle_nb"))
def test_tft_checkpoint_recipe_is_bound_to_governed_parameters(
    loss_kind: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclasses.dataclass(frozen=True)
    class FakePanel:
        path: str

    panel = FakePanel(str(tmp_path / "training-panel"))
    monkeypatch.setattr(
        subject, "load_dataset_contract",
        lambda _path: SimpleNamespace(panel=lambda _training_path: panel),
    )
    model = {
        "hidden_size": 50, "hidden_continuous_size": 8, "attention_head_size": 2,
        "dropout": 0.25, "max_encoder_length": 364, "min_encoder_length": 7,
        "lag_short_days": 7, "lag_long_days": 42,
        "output_size": 4 if loss_kind == "quantile" else 3,
    }
    parameters = {"effective_parameters": {
        **{f"model.{name}": value for name, value in model.items()},
        "model.loss_kind": loss_kind, "training.learning_rate": 0.015,
    }}
    child = {
        "adapter": "tft_q", "kind": "validation", "origin": "2024-01-21",
        "horizon": 28, "seed": 0,
        "budget": {"maximum_epochs": 2, "patience": 10,
                   "early_stopping_absolute_min_delta": 1e-6},
    }
    resolved = {
        "source": {"dataset_contract": "contract.yaml"},
        "selection": {"metric": "wrmsse"},
        "compute": {"device": "cpu", "batch_size": 2, "precision": "32-true"},
    }
    identity = {
        "panel": dataclasses.asdict(panel), "origin": child["origin"],
        "horizon": 28, "seed": 0, "batch_size": 2,
        "early_stopping_patience": 10, "early_stopping_min_delta": 1e-6,
        "requested_device": "cpu", "resolved_device": "cpu", **model,
        "learning_rate": 0.015, "loss_kind": loss_kind, "precision": "32-true",
        "training_mode": "fixed_validation", "fixed_update_budget": None,
        "max_epochs": 2, "panel_sha256": "a" * 64,
    }
    profile = {
        "variant": "v6.1" if loss_kind == "quantile" else "v6.2",
        "loss_kind": loss_kind,
        "point_prediction": "median" if loss_kind == "quantile" else "mean",
        "target_normalization": "identity",
        "cold_start": "source_unobserved_rows_zero_with_encoder_mask_v1",
        "mean_loss_weight": None if loss_kind == "quantile" else 0.2,
        "quantiles": [0.1, 0.25, 0.5, 0.75] if loss_kind == "quantile" else None,
        **{name: model[name] for name in (
            "hidden_size", "hidden_continuous_size", "attention_head_size", "dropout",
            "max_encoder_length", "min_encoder_length", "output_size",
        )},
        "learning_rate": 0.015, "precision": "32-true", "lags": [7, 42],
    }
    saved = {
        "schema": "autoforecast.tft-v6-inference-state.v2",
        "run_identity": identity,
        "profile": profile,
        "training": {"training_contract": {
            "mode": "fixed_validation", "fixed_update_budget": None,
        }},
    }
    subject._verify_revision_checkpoint_binding(  # noqa: SLF001
        saved, resolved, child, parameters, tmp_path / "training-panel", "a" * 64
    )
    for section, field, value in (
        ("run_identity", "learning_rate", 123.0),
        ("run_identity", "hidden_size", 99),
        ("run_identity", "panel_sha256", "b" * 64),
        ("profile", "loss_kind", "quantile" if loss_kind != "quantile" else "v3_hurdle_nb"),
    ):
        changed = copy.deepcopy(saved)
        changed[section][field] = value
        with pytest.raises(subject.ForecastCampaignError, match="checkpoint recipe"):
            subject._verify_revision_checkpoint_binding(  # noqa: SLF001
                changed, resolved, child, parameters, tmp_path / "training-panel", "a" * 64
            )


@pytest.mark.parametrize("version", ("v2", "v3", "v5"))
@pytest.mark.parametrize("purpose", ("forecast_campaign", "capacity_diagnostic"))
def test_rolling_checkpoint_recipe_rejects_coherent_ungoverned_changes(
    version: str, purpose: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autoforecast.model_execution import (
        REVISION_MODEL_PARAMETERS,
        REVISION_TRAINING_PARAMETERS,
        resolve_revision_rolling_config,
    )
    from autoforecast.rolling_training import RollingConfig

    adapter = REVISION_MODELS[version]
    candidate = next(item for item in discover_candidates(ROOT / "configs/models")
                     if item.adapter == adapter)
    defaults = {
        name: record["default"]
        for name, record in yaml.safe_load(candidate.source_bytes)["hyperparameters"].items()
    }
    model = {name: defaults[f"model.{name}"] for name in REVISION_MODEL_PARAMETERS[adapter]}
    tuning = {name: defaults[f"training.{name}"] for name in REVISION_TRAINING_PARAMETERS}
    recipe = resolve_revision_rolling_config(
        candidate, model_parameters=model, training_parameters=tuning, epochs=2, seed=0
    )
    mode = "rolling_backtest" if purpose == "capacity_diagnostic" else "fixed_validation"
    expected = dataclasses.asdict(RollingConfig(**{
        **recipe, "device": "cpu", "batch_size": 2, "patience": 10,
        "event_max_support": 14, "early_stopping_absolute_min_delta": 1e-6,
        "early_stopping_relative_min_delta": 0.0,
        **({"selection_metric": "wrmsse"} if mode == "fixed_validation" else {}),
    }))
    contract = {"mode": mode,
                "validation_origin": "2024-01-21" if mode == "fixed_validation" else None,
                "fixed_update_budget": None}
    saved = {
        "binding": {"config": json.loads(subject._json(expected)),  # noqa: SLF001
                    "origin": "2024-01-21", "horizon": 28,
                    "training_contract": contract},
        "training": {"config": expected, "training_contract": contract},
    }
    parameters = {"effective_parameters": defaults}
    child = {"adapter": adapter, "model_id": candidate.id, "kind": "validation",
             "origin": "2024-01-21", "horizon": 28, "seed": 0,
             "budget": {"maximum_epochs": 2, "patience": 10,
                        "early_stopping_absolute_min_delta": 1e-6,
                        "early_stopping_relative_min_delta": 0.0}}
    resolved = {"purpose": purpose, "selection": {"metric": "wrmsse"},
                "compute": {"device": "cpu", "batch_size": 2},
                "template_features": {"event_max_support": 14}}
    monkeypatch.setattr(subject, "_campaign_candidate", lambda *_args: candidate)
    subject._verify_revision_checkpoint_binding(  # noqa: SLF001
        saved, resolved, child, parameters, tmp_path, "a" * 64
    )
    mixed = copy.deepcopy(saved)
    mixed["binding"]["precision"] = "16-mixed"
    mixed["training"]["precision"] = "16-mixed"
    mixed_resolved = copy.deepcopy(resolved)
    mixed_resolved["compute"]["precision"] = "16-mixed"
    subject._verify_revision_checkpoint_binding(  # noqa: SLF001
        mixed, mixed_resolved, child, parameters, tmp_path, "a" * 64
    )
    for record in ("binding", "training"):
        changed = copy.deepcopy(mixed)
        changed[record].pop("precision")
        with pytest.raises(subject.ForecastCampaignError, match="checkpoint recipe"):
            subject._verify_revision_checkpoint_binding(  # noqa: SLF001
                changed, mixed_resolved, child, parameters, tmp_path, "a" * 64
            )
    with pytest.raises(subject.ForecastCampaignError, match="checkpoint recipe"):
        subject._verify_revision_checkpoint_binding(  # noqa: SLF001
            mixed, resolved, child, parameters, tmp_path, "a" * 64
        )
    fields = ["learning_rate", "gradient_clip", "d_model", "mean_loss_weight"]
    for field in fields:
        changed = copy.deepcopy(saved)
        changed["training"]["config"][field] = 123.0
        changed["binding"]["config"][field] = 123.0
        with pytest.raises(subject.ForecastCampaignError, match="checkpoint recipe"):
            subject._verify_revision_checkpoint_binding(  # noqa: SLF001
                changed, resolved, child, parameters, tmp_path, "a" * 64
            )
    if version in {"v3", "v5"}:
        field = "v3" if version == "v3" else "v5"
        nested = "local_blocks" if version == "v3" else "n_layers"
        changed = copy.deepcopy(saved)
        changed["training"]["config"][field][nested] = 123
        changed["binding"]["config"][field][nested] = 123
        with pytest.raises(subject.ForecastCampaignError, match="checkpoint recipe"):
            subject._verify_revision_checkpoint_binding(  # noqa: SLF001
                changed, resolved, child, parameters, tmp_path, "a" * 64
            )


def test_tft_campaign_locks_and_verifies_selected_updates(tmp_path: pathlib.Path) -> None:
    path = ROOT / "configs/autoforecast/m5_tft_q_reference_benchmark.yaml"
    request = subject.load_forecast_campaign_request(path)
    assert request.model_ids == ("tft_q", "tft_hnb")
    assert tuple(source.mode for source in request.parameter_sources) == ("defaults", "manual")
    assert request.parameter_sources[1].values == {
        "model.loss_kind": "v3_hurdle_nb", "model.output_size": 3
    }
    assert request.metric == "wrmsse"
    assert request.validation_budget.maximum_epochs == 50000
    assert request.validation_budget.patience == 10
    assert request.initialization_origin is None

    smoke = subject.load_forecast_campaign_request(
        ROOT / "configs/autoforecast/m5_tft_loss_2epoch.yaml"
    )
    assert smoke.model_ids == request.model_ids
    assert smoke.parameter_sources == request.parameter_sources
    assert smoke.metric == "wrmsse"
    assert smoke.initialization_origin is None
    assert smoke.validation_budget.maximum_epochs == 2
    assert smoke.validation_budget.patience == 10
    catalog = {
        candidate.id: candidate
        for candidate in discover_candidates(ROOT / "configs/models")
    }
    profiles = subject._parameter_profiles(
        smoke, catalog, {"sha256": "a" * 64}, frequency="D",
        dataset_contract_sha256="b" * 64, features=None,
    )
    assert len(profiles) == 2
    assert profiles[0]["candidate_id"] != profiles[1]["candidate_id"]
    assert profiles[0]["variant_id"] != profiles[1]["variant_id"]

    checkpoint = tmp_path / "model.pt"
    forecast_record = tmp_path / "forecast.json"
    forecast_record.write_text('{"n_series": 6}\n', encoding="utf-8")
    training = {
        "maximum_epochs": 50000, "n_series": 6,
        "steps_per_epoch": 3, "selected_updates": 6,
        "selected_epoch": 2,
        "early_stopping_patience": 10, "early_stopping_min_delta": 1e-6,
        "validation_origin": "2024-01-21",
        "validation": [
            {"epoch": 1, "updates": 3, "val_loss": 2.0,
             "val_loss_dtype": "torch.float32"},
            {"epoch": 2, "updates": 6, "val_loss": 1.0,
             "val_loss_dtype": "torch.float32"},
        ] + [
            {"epoch": epoch, "updates": epoch * 3, "val_loss": 1.25,
             "val_loss_dtype": "torch.float32"}
            for epoch in range(3, 13)
        ],
        "training_contract": {"mode": "fixed_validation", "fixed_update_budget": None},
    }
    training["epoch_log"] = training["validation"]
    torch.save({"training": training}, checkpoint)
    result = {
        "adapter": "tft_q", "checkpoint": {"path": str(checkpoint),
                                           "sha256": subject._sha_file(checkpoint)},
        "parameter_resolution": {"effective_parameters": {"model.loss_kind": "quantile"}},
        "forecast_record_path": str(forecast_record),
        "forecast_record_sha256": subject._sha_file(forecast_record),
        "origin": "2024-01-21",
        "budget": {
            "maximum_epochs": 50000, "patience": 10,
            "early_stopping_absolute_min_delta": 1e-6,
            "early_stopping_relative_min_delta": 0.0,
        },
    }
    assert subject._selected_update_budget(result, "wrmsse", 2) == 6
    with pytest.raises(subject.ForecastCampaignError, match="selection controls"):
        subject._selected_update_budget(result, "wrmsse", 3)
    threshold_edge = {
        **training,
        "validation": [
            {"epoch": 1, "updates": 3, "val_loss": 1.0,
             "val_loss_dtype": "torch.float32"},
            *[
                {"epoch": epoch, "updates": epoch * 3,
                 "val_loss": 0.9999989867210388,
                 "val_loss_dtype": "torch.float32"}
                for epoch in range(2, 12)
            ],
        ],
    }
    threshold_edge["epoch_log"] = threshold_edge["validation"]
    torch.save({"training": threshold_edge}, checkpoint)
    result["checkpoint"]["sha256"] = subject._sha_file(checkpoint)  # noqa: SLF001
    assert subject._selected_update_budget(result, "wrmsse", 2) == 6
    torch.save({"training": training}, checkpoint)
    result["checkpoint"]["sha256"] = subject._sha_file(checkpoint)  # noqa: SLF001
    double_loss = {
        **training,
        "validation": [
            {**row, "val_loss_dtype": "torch.float64"}
            for row in training["validation"]
        ],
    }
    double_loss["epoch_log"] = double_loss["validation"]
    torch.save({"training": double_loss}, checkpoint)
    result["checkpoint"]["sha256"] = subject._sha_file(checkpoint)  # noqa: SLF001
    with pytest.raises(subject.ForecastCampaignError, match="selection|validation"):
        subject._selected_update_budget(result, "wrmsse", 2)
    result["parameter_resolution"]["effective_parameters"]["model.loss_kind"] = "v3_hurdle_nb"
    assert subject._selected_update_budget(result, "wrmsse", 2) == 6
    result["parameter_resolution"]["effective_parameters"]["model.loss_kind"] = "quantile"
    rounded_loss = [
        training["validation"][0],
        {**training["validation"][1], "val_loss": 0.99999999},
        *training["validation"][2:],
    ]
    relabeled_boundary = [
        {**row, "val_loss_dtype": "torch.float64"}
        for row in [
            *threshold_edge["validation"],
            {"epoch": 12, "updates": 36, "val_loss": 0.9999989867210388,
             "val_loss_dtype": "torch.float32"},
        ]
    ]
    for altered in (
        {**training, "validation": rounded_loss, "epoch_log": rounded_loss},
        {**training, "epoch_log": []},
        {**training, "validation": relabeled_boundary,
         "epoch_log": relabeled_boundary},
        {**training, "validation": [
            {**row, "val_loss": 70000.0, "val_loss_dtype": "torch.float16"}
            for row in training["validation"]
        ], "epoch_log": [
            {**row, "val_loss": 70000.0, "val_loss_dtype": "torch.float16"}
            for row in training["validation"]
        ]},
    ):
        torch.save({"training": altered}, checkpoint)
        result["checkpoint"]["sha256"] = subject._sha_file(checkpoint)  # noqa: SLF001
        with pytest.raises(subject.ForecastCampaignError, match="selection|validation"):
            subject._selected_update_budget(result, "wrmsse", 2)
    torch.save({"training": training}, checkpoint)
    result["checkpoint"]["sha256"] = subject._sha_file(checkpoint)  # noqa: SLF001
    for changed in (
        {**result, "budget": {**result["budget"], "early_stopping_absolute_min_delta": 0.1}},
        {**result, "budget": {**result["budget"], "patience": 9}},
    ):
        with pytest.raises(subject.ForecastCampaignError, match="selection"):
            subject._selected_update_budget(changed, "wrmsse", 2)
    for altered in (
        {**training, "validation": []},
        {**training, "validation": training["validation"][:2]},
        {**training, "steps_per_epoch": 1, "selected_updates": 2,
         "validation": [{**row, "updates": row["epoch"]}
                        for row in training["validation"]]},
        {**training, "selected_updates": 5, "validation": [
            training["validation"][0],
            {"epoch": 2, "updates": 5, "val_loss": 1.0,
             "val_loss_dtype": "torch.float32"},
            *training["validation"][2:],
        ]},
        {**training, "refit_optimizer_steps": 999},
        {**training, "validation": [{"epoch": 2, "updates": 5, "val_loss": 1.0}]},
        {**training, "validation": [
            {"epoch": 1, "updates": 3, "val_loss": 0.5},
            {"epoch": 2, "updates": 6, "val_loss": 1.0},
        ]},
        {**training, "validation_origin": "2024-01-20"},
    ):
        torch.save({"training": altered}, checkpoint)
        result["checkpoint"]["sha256"] = subject._sha_file(checkpoint)  # noqa: SLF001
        with pytest.raises(subject.ForecastCampaignError, match="selection|validation witness"):
            subject._selected_update_budget(result, "wrmsse", 2)  # noqa: SLF001
    torch.save({"training": training}, checkpoint)
    result["checkpoint"]["sha256"] = subject._sha_file(checkpoint)  # noqa: SLF001
    chosen = {"candidate_id": "v6-1", "model_id": "tft_q", "initialization": "fresh"}
    result.update({"candidate_id": "v6-1", "initialization": "fresh"})
    assert subject._locked_selected_update_budget(  # noqa: SLF001
        {"purpose": "forecast_campaign", "selection": {"metric": "wrmsse"},
         "compute": {"batch_size": 2},
         "children": [{"kind": "validation", **result}]},
        chosen, [result],
    ) == 6
    training["maximum_epochs"] = 2
    torch.save({"training": training}, checkpoint)
    result["checkpoint"]["sha256"] = subject._sha_file(checkpoint)
    with pytest.raises(subject.ForecastCampaignError, match="ceiling"):
        subject._selected_update_budget(result, "wrmsse", 2)


@pytest.mark.parametrize(
    ("purpose", "adapter", "kind", "evaluate", "expected_budget", "expected_mode"),
    tuple(
        (purpose, adapter, kind, evaluate, budget, mode)
        for purpose, adapter in (
            ("forecast_campaign", "tft_q"),
            ("forecast_campaign", "sft"),
            ("research_campaign", "sft"),
        )
        for kind, evaluate, budget, mode in (
            ("validation", True, None, "fixed_validation"),
            ("final_fit", False, 6, "fixed_refit"),
        )
    ) + (("capacity_diagnostic", "sft",
          "validation", False, None, None),),
)
def test_revision_children_use_fixed_validation_and_locked_refit_modes(
    kind: str,
    evaluate: bool,
    expected_budget: int | None,
    purpose: str,
    adapter: str,
    expected_mode: str | None,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path, models=["seasonal_naive"], arms=["fresh"])
    monkeypatch.setattr(subject, "_git_source_identity", lambda _path: {"git_commit": "a" * 40})
    compiled = _compile(request, tmp_path)
    resolved = json.loads(compiled.resolved_path.read_text())
    resolved["purpose"] = purpose
    child = {**resolved["children"][0], "kind": kind, "adapter": adapter}

    class StopBeforeWorkerLaunch(Exception):
        pass

    class CapturingProcess:
        def wait(self) -> int:
            raise StopBeforeWorkerLaunch

    captured: list[list[str]] = []

    def capture(
        command: list[str], *, cwd: pathlib.Path, pass_fds: tuple[int, ...]
    ) -> CapturingProcess:
        assert cwd == ROOT
        assert pass_fds == ()
        captured.append(command)
        return CapturingProcess()

    monkeypatch.setattr(subject, "_verify_data", lambda *_args: tmp_path)
    monkeypatch.setattr(subject, "_materialize_training_panel", lambda *_a, **_k: "d" * 64)
    monkeypatch.setattr(subject, "_materialize_panel_range", lambda *_a, **_k: None)
    monkeypatch.setattr(subject, "_write_child_request", lambda *_a, **_k: None)
    monkeypatch.setattr(subject, "_peak_rss", lambda _process: 0)
    monkeypatch.setattr(subject.subprocess, "Popen", capture)
    attempt = tmp_path / kind
    attempt.mkdir()
    with pytest.raises(StopBeforeWorkerLaunch):
        subject._run_generic_child(  # noqa: SLF001
            resolved, child, campaign_root=tmp_path / "campaign", attempt_dir=attempt,
            evaluate=evaluate, fixed_update_budget=expected_budget,
        )
    command = captured[0]
    if expected_mode is None:
        assert "--training-mode" not in command
    else:
        assert command[command.index("--training-mode") + 1] == expected_mode
    if expected_budget is not None:
        assert command[command.index("--fixed-update-budget") + 1] == str(expected_budget)


def test_research_campaign_expands_each_seed_across_every_validation_origin(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _request(tmp_path, models=["seasonal_naive"], arms=["fresh"])
    value = _make_single_model_benchmark(path)
    campaign = value["forecast_campaign"]
    campaign["purpose"] = "research_campaign"
    campaign["temporal"]["validation_origins"] = [
        "2024-01-04",
        "2024-01-10",
        "2024-01-16",
        "2024-01-22",
    ]
    campaign["temporal"]["initialization_origin"] = None
    campaign["seeds"] = [0, 1, 2]
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(
        subject,
        "_git_source_identity",
        lambda _path: {"git_commit": "a" * 40, "request_path": "request.yaml"},
    )

    compiled = _compile(path, tmp_path)
    stages = yaml.safe_load(compiled.plan_path.read_text())["campaign"]["stages"]

    assert len(stages) == 16
    assert [stage["id"] for stage in stages[:12]] == [
        "v0fs0o0",
        "v0fs0o1",
        "v0fs0o2",
        "v0fs0o3",
        "v0fs1o0",
        "v0fs1o1",
        "v0fs1o2",
        "v0fs1o3",
        "v0fs2o0",
        "v0fs2o1",
        "v0fs2o2",
        "v0fs2o3",
    ]
    assert [stage["id"] for stage in stages[12:]] == [
        "aggregate_selection",
        "final_fit",
        "final_score",
        "replay_report",
    ]


def test_research_campaign_rejects_only_one_validation_origin(tmp_path: pathlib.Path) -> None:
    path = _request(tmp_path, models=["seasonal_naive"], arms=["fresh"])
    value = _make_single_model_benchmark(path)
    campaign = value["forecast_campaign"]
    campaign["purpose"] = "research_campaign"
    campaign["seeds"] = [0, 1, 2]
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")

    with pytest.raises(subject.ForecastCampaignError, match="at least two validation origins"):
        subject.load_forecast_campaign_request(path)


@pytest.mark.parametrize(
    ("filename", "model_id"),
    (
        ("m5_sft_research_campaign.yaml", "sft"),
        ("m5_hmfnet_research_campaign.yaml", "hmfnet"),
    ),
)
def test_checked_in_research_campaigns_are_separate_sixteen_stage_model_requests(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    model_id: str,
) -> None:
    path = ROOT / "configs" / "autoforecast" / filename
    request = subject.load_forecast_campaign_request(path)
    assert request.purpose == "research_campaign"
    assert request.model_ids == (model_id,)
    assert request.seeds == (0, 1, 2)
    assert tuple(day.isoformat() for day in request.validation_origins) == (
        "2016-01-31",
        "2016-02-28",
        "2016-03-27",
        "2016-04-24",
    )
    assert request.initialization_origin is None
    assert request.initialization_arms == ("fresh",)
    for offset, role in enumerate(("selection", "final-training", "final-score")):
        _panel(tmp_path / role, target_offset=float(offset * 100))
    monkeypatch.setattr(
        subject,
        "_git_source_identity",
        lambda _path: {"git_commit": "a" * 40, "request_path": filename},
    )
    compiled = _compile(path, tmp_path)
    stages = yaml.safe_load(compiled.plan_path.read_text())["campaign"]["stages"]
    assert len(stages) == 16
    assert [stage["id"] for stage in stages[-4:]] == [
        "aggregate_selection",
        "final_fit",
        "final_score",
        "replay_report",
    ]


def test_resolved_source_requires_the_compiled_commit_and_model_profile_bytes(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _request(tmp_path)
    monkeypatch.setattr(
        subject,
        "_git_source_identity",
        lambda _path: {"git_commit": "a" * 40, "request_path": "request.yaml"},
    )
    compiled = _compile(path, tmp_path)
    resolved = json.loads(compiled.resolved_path.read_text())
    request_source = ROOT / "configs/autoforecast/examples/synthetic_seasonal_naive.yaml"
    resolved["source"]["request_path"] = request_source.relative_to(ROOT).as_posix()
    resolved["source"]["request_sha256"] = subject._sha_file(request_source)  # noqa: SLF001
    monkeypatch.setattr(
        subject,
        "capture_source_identity",
        lambda _root: SimpleNamespace(commit="a" * 40),
    )

    subject._verify_resolved_source(resolved)  # noqa: SLF001

    original_sha = resolved["source"]["model_profiles"][0]["sha256"]
    resolved["source"]["model_profiles"][0]["sha256"] = "0" * 64
    with pytest.raises(subject.ForecastCampaignError, match="model profile .* bytes changed"):
        subject._verify_resolved_source(resolved)  # noqa: SLF001
    resolved["source"]["model_profiles"][0]["sha256"] = original_sha

    monkeypatch.setattr(
        subject,
        "capture_source_identity",
        lambda _root: SimpleNamespace(commit="b" * 40),
    )
    with pytest.raises(subject.ForecastCampaignError, match="source commit changed"):
        subject._verify_resolved_source(resolved)  # noqa: SLF001


def test_compiler_refuses_final_labels_at_a_training_path(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _request(tmp_path)
    monkeypatch.setattr(subject, "_git_source_identity", lambda _path: {})
    with pytest.raises(subject.ForecastCampaignError, match="separate"):
        subject.compile_forecast_campaign(
            path,
            selection_data_path=tmp_path / "selection",
            final_training_data_path=tmp_path / "final-training",
            final_score_data_path=tmp_path / "final-training",
            compiled_root=tmp_path / "compiled",
        )


def test_compiler_refuses_symlinked_runtime_data(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _request(tmp_path)
    link = tmp_path / "selection-link"
    link.symlink_to(tmp_path / "selection", target_is_directory=True)
    monkeypatch.setattr(subject, "_git_source_identity", lambda _path: {})
    with pytest.raises(subject.ForecastCampaignError, match="may not traverse a symlink"):
        subject.compile_forecast_campaign(
            path,
            selection_data_path=link,
            final_training_data_path=tmp_path / "final-training",
            final_score_data_path=tmp_path / "final-score",
            compiled_root=tmp_path / "compiled",
        )


def test_compiler_refuses_separate_paths_with_identical_final_label_bytes(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    duplicate = tmp_path / "duplicate-final-score"
    duplicate.mkdir()
    duplicate.joinpath("renamed.parquet").write_bytes(
        (tmp_path / "selection/part.parquet").read_bytes()
    )
    monkeypatch.setattr(subject, "_git_source_identity", lambda _path: {})
    with pytest.raises(subject.ForecastCampaignError, match="content-distinct"):
        subject.compile_forecast_campaign(
            request,
            selection_data_path=tmp_path / "selection",
            final_training_data_path=tmp_path / "final-training",
            final_score_data_path=duplicate,
            compiled_root=tmp_path / "compiled",
        )


def test_compiler_never_labels_the_revealed_m5_private_window_as_blind(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    value = yaml.safe_load(request.read_text())
    campaign = value["forecast_campaign"]
    campaign["dataset"]["contract"] = "configs/datasets/m5_daily.yaml"
    campaign["temporal"]["final_train_cutoff"] = "2016-05-22"
    campaign["temporal"]["final_test_horizon"] = 28
    campaign["temporal"]["training_start"] = "2015-01-01"
    campaign["temporal"]["validation_origins"] = ["2016-04-24"]
    campaign["temporal"]["validation_horizon"] = 28
    campaign["temporal"]["initialization_origin"] = "2016-03-01"
    request.write_text(yaml.safe_dump(value, sort_keys=False))
    monkeypatch.setattr(subject, "_git_source_identity", lambda _path: {})
    with pytest.raises(subject.ForecastCampaignError, match="revealed M5 private window"):
        _compile(request, tmp_path)


def test_training_materializer_truncates_a_panel_without_known_future_rows(
    tmp_path: pathlib.Path,
) -> None:
    source = tmp_path / "source"
    _panel(source, days=10)
    destination = tmp_path / "training"
    subject._materialize_training_panel(  # noqa: SLF001 - physical leakage boundary is behavior
        source,
        destination,
        contract_path=ROOT / "configs/datasets/synthetic_daily.yaml",
        origin="2024-01-05",
        horizon=2,
        training_start=None,
        expected_source_sha256=subject.directory_digest(source, suffixes=(".parquet",)),
    )
    table = pads.dataset(destination, format="parquet").to_table()
    assert max(table["date"].to_pylist()) == dt.date(2024, 1, 5)
    assert table.num_rows == 10


def test_hourly_training_materializer_excludes_later_same_day_rows(
    tmp_path: pathlib.Path,
) -> None:
    contract = tmp_path / "hourly.yaml"
    contract.write_text(
        """schema_version: 1
dataset:
  id: hourly
  panel:
    series_col: series_id
    time_col: date
    target_col: target
    frequency: H
    season_length: 24
""",
        encoding="utf-8",
    )
    source = tmp_path / "source"
    source.mkdir()
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"series_id": "a", "date": dt.datetime(2024, 3, 1, hour), "target": float(hour)}
                for hour in (21, 22, 23)
            ]
        ),
        source / "part.parquet",
    )
    destination = tmp_path / "training"
    subject._materialize_training_panel(  # noqa: SLF001
        source,
        destination,
        contract_path=contract,
        origin="2024-03-01T22:00:00",
        horizon=2,
        training_start=None,
        expected_source_sha256=subject.directory_digest(source, suffixes=(".parquet",)),
    )
    points = pads.dataset(destination, format="parquet").to_table()["date"].to_pylist()
    assert points == [dt.datetime(2024, 3, 1, 21), dt.datetime(2024, 3, 1, 22)]


def test_snapshot_boundaries_reject_timezone_bearing_int96(
    tmp_path: pathlib.Path,
) -> None:
    contract = tmp_path / "hourly.yaml"
    contract.write_text(
        """schema_version: 1
dataset:
  id: aware-int96
  panel:
    series_col: series_id
    time_col: date
    target_col: target
    frequency: H
    season_length: 2
""",
        encoding="utf-8",
    )
    source = tmp_path / "source"
    source.mkdir()
    pq.write_table(
        pa.table({
            "series_id": ["a", "a", "a"],
            "date": pa.array(
                [dt.datetime(2024, 1, 1, hour) for hour in range(3)],
                type=pa.timestamp("us", tz="Asia/Kathmandu"),
            ),
            "target": [1.0, 2.0, 3.0],
        }),
        source / "part.parquet",
        use_deprecated_int96_timestamps=True,
    )
    expected = subject.directory_digest(source, suffixes=(".parquet",))

    with pytest.raises(ValueError, match="timezone-bearing"):
        subject._panel_range_facts(source, contract_path=contract)  # noqa: SLF001

    training = tmp_path / "training"
    with pytest.raises(ValueError, match="timezone-bearing"):
        subject._materialize_training_panel(  # noqa: SLF001
            source,
            training,
            contract_path=contract,
            origin="2024-01-01T01:00:00",
            horizon=1,
            training_start=None,
            expected_source_sha256=expected,
        )
    assert not training.exists()

    evaluation = tmp_path / "evaluation"
    with pytest.raises(ValueError, match="timezone-bearing"):
        subject._materialize_panel_range(  # noqa: SLF001
            source,
            evaluation,
            contract_path=contract,
            start="2024-01-01T00:00:00",
            end="2024-01-01T01:00:00",
            expected_source_sha256=expected,
        )
    assert not evaluation.exists()


def test_training_materializer_keeps_known_future_features_but_masks_labels(
    tmp_path: pathlib.Path,
) -> None:
    contract = tmp_path / "dataset.yaml"
    contract.write_text(
        """schema_version: 1
dataset:
  id: masked_daily
  panel:
    series_col: series_id
    time_col: date
    target_col: target
    exposure_col: state
    known_future: [price]
    price_col: price
    frequency: D
    season_length: 7
""",
        encoding="utf-8",
    )
    source = tmp_path / "source"
    source.mkdir()
    rows = [
        {
            "series_id": "a",
            "date": dt.date(2024, 1, day),
            "target": float(day),
            "state": "exact",
            "price": float(10 + day),
        }
        for day in range(1, 8)
    ]
    pq.write_table(pa.Table.from_pylist(rows), source / "part.parquet")
    destination = tmp_path / "training"
    subject._materialize_training_panel(  # noqa: SLF001
        source,
        destination,
        contract_path=contract,
        origin="2024-01-05",
        horizon=2,
        training_start=None,
        expected_source_sha256=subject.directory_digest(source, suffixes=(".parquet",)),
    )
    values = pads.dataset(destination, format="parquet").to_table().to_pylist()
    future = [row for row in values if row["date"] > dt.date(2024, 1, 5)]
    assert [row["price"] for row in future] == [16.0, 17.0]
    assert all(row["target"] == 0 and row["state"] == "unobserved" for row in future)


def test_evaluation_materializer_physically_excludes_later_reserved_rows(
    tmp_path: pathlib.Path,
) -> None:
    source = tmp_path / "source"
    _panel(source, days=10)
    destination = tmp_path / "evaluation"

    subject._materialize_panel_range(  # noqa: SLF001
        source,
        destination,
        contract_path=ROOT / "configs/datasets/synthetic_daily.yaml",
        start=None,
        end="2024-01-07",
        expected_source_sha256=subject.directory_digest(source, suffixes=(".parquet",)),
    )

    table = pads.dataset(destination, format="parquet").to_table()
    assert min(table["date"].to_pylist()) == dt.date(2024, 1, 1)
    assert max(table["date"].to_pylist()) == dt.date(2024, 1, 7)
    assert table.num_rows == 14


def test_training_materializer_removes_output_if_source_changes_during_scan(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    _panel(source, days=10)
    destination = tmp_path / "training"
    real_digest = subject.directory_digest
    expected = real_digest(source, suffixes=(".parquet",))
    source_calls = 0

    def changing_digest(path: pathlib.Path, *, suffixes: tuple[str, ...] = ()) -> str:
        nonlocal source_calls
        if pathlib.Path(path) == source:
            source_calls += 1
            if source_calls > 1:
                return "0" * 64
        return real_digest(path, suffixes=suffixes)

    monkeypatch.setattr(subject, "directory_digest", changing_digest)
    with pytest.raises(subject.ForecastCampaignError, match="changed during"):
        subject._materialize_training_panel(  # noqa: SLF001
            source,
            destination,
            contract_path=ROOT / "configs/datasets/synthetic_daily.yaml",
            origin="2024-01-05",
            horizon=2,
            training_start=None,
            expected_source_sha256=expected,
        )
    assert not destination.exists()


def test_run_group_path_is_bound_to_the_current_stage_attempt(
    tmp_path: pathlib.Path,
) -> None:
    attempt = tmp_path / "campaign/stages/train/attempt-0001"
    expected = attempt / "run/group"
    expected.mkdir(parents=True)
    assert subject._run_group_path(attempt, expected, "child run path") == expected  # noqa: SLF001

    foreign = tmp_path / "other/run/group"
    foreign.mkdir(parents=True)
    with pytest.raises(subject.ForecastCampaignError, match="outside the current stage"):
        subject._run_group_path(attempt, foreign, "child run path")  # noqa: SLF001

    expected.rmdir()
    expected.symlink_to(foreign, target_is_directory=True)
    with pytest.raises(subject.ForecastCampaignError, match="may not traverse a symlink"):
        subject._run_group_path(attempt, expected, "child run path")  # noqa: SLF001


def test_published_child_run_groups_excludes_only_private_worker_scratch(
    tmp_path: pathlib.Path,
) -> None:
    run_root = tmp_path / "attempt/run"
    published = run_root / "run-20260922T000000Z"
    scratch = run_root / ".autoforecast-worker-scratch"
    published.mkdir(parents=True)
    scratch.mkdir()

    assert subject._published_child_run_groups(run_root) == [published]  # noqa: SLF001

    unexpected = run_root / ".unexpected-child-directory"
    unexpected.mkdir()
    assert subject._published_child_run_groups(run_root) == [  # noqa: SLF001
        unexpected,
        published,
    ]


def test_checkpoint_path_is_bound_to_the_governed_child_run(
    tmp_path: pathlib.Path,
) -> None:
    run = tmp_path / "attempt/run/group"
    expected = run / "attempts/model-a/attempt-0001/model/model.pt"
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"checkpoint")
    assert subject._run_checkpoint_path(run, "model-a", expected) == expected  # noqa: SLF001

    foreign = tmp_path / "foreign/model.pt"
    foreign.parent.mkdir()
    foreign.write_bytes(b"checkpoint")
    with pytest.raises(subject.ForecastCampaignError, match="outside the governed child run"):
        subject._run_checkpoint_path(run, "model-a", foreign)  # noqa: SLF001


def test_child_result_rejects_failed_or_numerically_invalid_evidence(
    tmp_path: pathlib.Path,
) -> None:
    child = {
        "id": "v0fs0o0",
        "kind": "validation",
        "candidate_id": "seasonal_naive",
        "model_id": "seasonal_naive",
        "variant_id": "default",
        "adapter": "seasonal_naive",
        "initialization": "fresh",
        "seed": 0,
        "origin": "2024-01-14",
        "horizon": 2,
        "budget": {},
        "parent_stage": None,
    }
    record = {field: None for field in subject._CHILD_RESULT_FIELDS}  # noqa: SLF001
    record.update(
        {
            "schema": subject.CHILD_RESULT_SCHEMA,
            "status": "failed",
            "kind": child["kind"],
            "child_id": child["id"],
            "model_id": child["model_id"],
            "adapter": child["adapter"],
            "initialization": child["initialization"],
            "seed": child["seed"],
            "origin": child["origin"],
            "horizon": child["horizon"],
            "budget": child["budget"],
            "training_data_sha256": "a" * 64,
            "score": -1.0,
            "runtime_seconds": -2.0,
        }
    )
    path = tmp_path / "child-result.json"
    path.write_text(subject._json(record), encoding="utf-8")  # noqa: SLF001
    with pytest.raises(subject.ForecastCampaignError, match="identity/status"):
        subject._validated_child_result(  # noqa: SLF001
            {}, child, path, require_score=True, completed=False
        )


@pytest.mark.parametrize("field", ["candidate_id", "model_id", "variant_id"])
def test_child_result_rejects_each_candidate_identity_field_independently(
    tmp_path: pathlib.Path, field: str
) -> None:
    child = {
        "id": "v0fs0o0",
        "kind": "validation",
        "candidate_id": "@3:sft:short",
        "model_id": "sft",
        "variant_id": "short",
        "adapter": "sft",
        "parameter_mode": "defaults",
        "initialization": "fresh",
        "seed": 0,
        "origin": "2024-01-14",
        "horizon": 2,
        "budget": {},
        "parent_stage": None,
    }
    record = {name: None for name in subject._CHILD_RESULT_FIELDS}  # noqa: SLF001
    record["training_metrics_sha256"] = "a" * 64
    record.update(
        {
            "schema": subject.CHILD_RESULT_SCHEMA,
            "status": "completed",
            "kind": child["kind"],
            "child_id": child["id"],
            "candidate_id": child["candidate_id"],
            "model_id": child["model_id"],
            "variant_id": child["variant_id"],
            "adapter": child["adapter"],
            "parameter_mode": child["parameter_mode"],
            "initialization": child["initialization"],
            "seed": child["seed"],
            "origin": child["origin"],
            "horizon": child["horizon"],
            "budget": child["budget"],
            "training_data_sha256": "a" * 64,
            "score": 1.0,
            "runtime_seconds": 1.0,
        }
    )
    record[field] = f"wrong-{field}"
    path = tmp_path / "child-result.json"
    path.write_text(subject._json(record), encoding="utf-8")  # noqa: SLF001
    with pytest.raises(subject.ForecastCampaignError, match="identity/status"):
        subject._validated_child_result({}, child, path, require_score=True, completed=False)  # noqa: SLF001


def test_selection_lock_status_and_values_are_recomputed_before_final_fit(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child = {
        "id": "v0fs0o0",
        "kind": "validation",
        "model_id": "seasonal_naive",
        "adapter": "seasonal_naive",
        "initialization": "fresh",
        "seed": 0,
        "origin": "2024-01-14",
        "horizon": 2,
        "budget": {},
        "parent_stage": None,
    }
    candidate = {
        "candidate_id": "seasonal_naive",
        "model_id": "seasonal_naive",
        "variant_id": "default",
        "initialization": "fresh",
        "mean_score": 1.0,
        "score_stddev": 0.0,
    }
    resolved = {
        "children": [child],
        "seeds": [0],
        "selection": {"metric": "mae"},
        "initialization_capabilities": [],
        "source": {},
        "data": {"selection": {}},
        "temporal": {"validation_origins": ["2024-01-14"], "validation_horizon": 2},
        "budgets": {"final_fit": {}},
    }
    summary = {
        "schema": subject.VALIDATION_SUMMARY_SCHEMA,
        "resolved_sha256": "a" * 64,
        "selection": resolved["selection"],
        "capabilities": [],
        "candidates": [candidate],
        "paired_initialization_deltas": [],
        "ranking": [
            {
                "candidate_id": "seasonal_naive",
                "model_id": "seasonal_naive",
                "variant_id": "default",
                "initialization": "fresh",
            }
        ],
    }
    summary_path = tmp_path / "validation-summary.json"
    summary_path.write_text(subject._json(summary), encoding="utf-8")  # noqa: SLF001
    lock = {
        "schema": subject.SELECTION_LOCK_SCHEMA,
        "status": "pending",
        "resolved_sha256": "a" * 64,
        "source": {},
        "selection_data": {},
        "metric": "mae",
        "direction": "minimize",
        "tie_break": list(subject._TIE_BREAK),  # noqa: SLF001
        "validation_origins": ["2024-01-14"],
        "validation_horizon": 2,
        "validation_summary_sha256": subject._sha_file(summary_path),  # noqa: SLF001
        "chosen": {
            **candidate,
            "final_seed": 99,
            "final_fit_budget": {"maximum_epochs": 999},
        },
    }
    lock_path = tmp_path / "selection-lock.json"
    lock_path.write_text(subject._json(lock), encoding="utf-8")  # noqa: SLF001
    child_path = tmp_path / "child-result.json"
    child_path.write_text("{}\n")

    def completed(_root: pathlib.Path, stage: str, relative: str) -> pathlib.Path:
        if (stage, relative) == ("aggregate_selection", "validation-summary.json"):
            return summary_path
        if (stage, relative) == ("aggregate_selection", "selection-lock.json"):
            return lock_path
        return child_path

    monkeypatch.setattr(subject, "_completed_output", completed)
    monkeypatch.setattr(subject, "_load_completed_json",
                        lambda path: subject._load_json_with_identity(path, "fixture output"))
    monkeypatch.setattr(
        subject,
        "_validated_child_result",
        lambda *_args, **_kwargs: {"score": 1.0},
    )
    monkeypatch.setattr(
        subject,
        "_selection_statistics",
        lambda *_args: ([candidate], [], [candidate]),
    )
    with pytest.raises(subject.ForecastCampaignError, match="selection lock differs"):
        subject._validated_selection_evidence(  # noqa: SLF001
            resolved, "a" * 64, tmp_path / "campaign"
        )


def test_aggregation_requires_every_child_and_applies_the_predeclared_tie_break(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    monkeypatch.setattr(subject, "_git_source_identity", lambda _path: {"git_commit": "a" * 40})
    compiled = _compile(request, tmp_path)
    resolved = json.loads(compiled.resolved_path.read_text())
    outputs: dict[tuple[str, str], pathlib.Path] = {}
    for child in (item for item in resolved["children"] if item["kind"] == "validation"):
        directory = tmp_path / "upstream" / child["id"]
        directory.mkdir(parents=True)
        forecast = directory / "forecast.json"
        forecast.write_text("{}\n", encoding="utf-8")
        # Both candidates have the same mean; seasonal_naive has zero variance and must win
        # before the lexical model-id tie-break is considered.
        if child["model_id"] == "seasonal_naive":
            score = 1.0
        else:
            score = 0.5 if child["origin"] == "2024-01-14" else 1.5
        result = {
            "schema": subject.CHILD_RESULT_SCHEMA,
            "child_id": child["id"],
            "candidate_id": child["candidate_id"],
            "model_id": child["model_id"],
            "variant_id": child["variant_id"],
            "parameter_mode": child["parameter_mode"],
            "parameter_resolution": next(
                profile["resolution"]
                for profile in resolved["parameter_profiles"]
                if profile["candidate_id"] == child["candidate_id"]
            ),
            "initialization": child["initialization"],
            "seed": child["seed"],
            "origin": child["origin"],
            "horizon": child["horizon"],
            "score": score,
            "runtime_seconds": 2.0,
            "peak_rss_bytes": 100,
            "replay": {"status": "verified_artifact_bytes"},
            "forecast_record_path": str(forecast),
            "forecast_record_sha256": subject._sha_file(forecast),  # noqa: SLF001
        }
        result["parameter_resolution_sha256"] = subject.parameter_resolution_sha256(
            result["parameter_resolution"]
        )
        result_path = directory / "child-result.json"
        result_path.write_text(subject._json(result), encoding="utf-8")  # noqa: SLF001
        outputs[(child["id"], "child-result.json")] = result_path

    def completed(_root: pathlib.Path, stage_id: str, relative: str) -> pathlib.Path:
        try:
            return outputs[(stage_id, relative)]
        except KeyError as error:
            raise subject.ForecastCampaignError("missing required child") from error

    monkeypatch.setattr(subject, "_completed_output", completed)
    monkeypatch.setattr(subject, "_load_completed_json",
                        lambda path: subject._load_json_with_identity(path, "fixture output"))
    monkeypatch.setattr(
        subject,
        "_validated_child_result",
        lambda _resolved, _child, path, *, require_score: json.loads(path.read_text()),
    )
    monkeypatch.setattr(subject, "_verify_resolved_source", lambda _resolved: None)
    attempt = tmp_path / "aggregate"
    attempt.mkdir()
    subject.aggregate_selection(
        compiled.resolved_path,
        subject._sha_file(compiled.resolved_path),  # noqa: SLF001
        campaign_root=tmp_path / "campaign",
        attempt_dir=attempt,
    )
    lock = json.loads((attempt / "selection-lock.json").read_text())
    assert lock["chosen"]["model_id"] == "seasonal_naive"
    assert lock["chosen"]["score_stddev"] == 0.0

    outputs.pop((resolved["children"][0]["id"], "child-result.json"))
    with pytest.raises(subject.ForecastCampaignError, match="missing required child"):
        subject.aggregate_selection(
            compiled.resolved_path,
            subject._sha_file(compiled.resolved_path),  # noqa: SLF001
            campaign_root=tmp_path / "campaign",
            attempt_dir=tmp_path / "must-not-publish",
        )


def test_final_fit_uses_the_locked_winner_and_final_training_identity_only(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    monkeypatch.setattr(subject, "_git_source_identity", lambda _path: {"git_commit": "a" * 40})
    compiled = _compile(request, tmp_path)
    resolved = json.loads(compiled.resolved_path.read_text())
    lock = {
        "schema": subject.SELECTION_LOCK_SCHEMA,
        "resolved_sha256": subject._sha_file(compiled.resolved_path),  # noqa: SLF001
        "chosen": {
            "candidate_id": "seasonal_naive",
            "model_id": "seasonal_naive",
            "variant_id": "default",
            "initialization": "fresh",
            "final_seed": 0,
            "final_fit_budget": resolved["budgets"]["final_fit"],
        },
    }
    # This fixture isolates refit semantics; journal authentication is exercised separately.
    monkeypatch.setattr(subject, "_load_completed_json",
                        lambda path: subject._load_json_with_identity(path, "fixture output"))
    lock_path = tmp_path / "selection-lock.json"
    lock_path.write_text(subject._json(lock), encoding="utf-8")  # noqa: SLF001
    monkeypatch.setattr(
        subject,
        "_validated_selection_evidence",
        lambda _resolved, _sha, _root: (lock, {}, lock_path, tmp_path / "summary.json"),
    )
    seen: dict[str, object] = {}

    def run_child(
        current: dict[str, object],
        child: dict[str, object],
        **_kwargs: object,
    ) -> dict[str, object]:
        seen["child"] = dict(child)
        seen["training_path"] = current["data"]["selection"]["path"]  # type: ignore[index]
        return {"schema": subject.CHILD_RESULT_SCHEMA, **child}

    monkeypatch.setattr(subject, "_run_generic_child", run_child)
    monkeypatch.setattr(subject, "_verify_resolved_source", lambda _resolved: None)
    attempt = tmp_path / "final"
    attempt.mkdir()
    subject.run_final_fit(
        compiled.resolved_path,
        subject._sha_file(compiled.resolved_path),  # noqa: SLF001
        campaign_root=tmp_path / "campaign",
        attempt_dir=attempt,
    )
    child = seen["child"]
    assert isinstance(child, dict)
    assert child["origin"] == "2024-01-25"
    assert child["horizon"] == 2
    assert seen["training_path"] == resolved["data"]["final_training"]["path"]
    assert json.loads(compiled.resolved_path.read_text())["data"] == resolved["data"]


def test_validation_child_trains_on_selection_but_scores_from_final_training(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path, models=["seasonal_naive"], arms=["fresh"])
    monkeypatch.setattr(subject, "_git_source_identity", lambda _path: {"git_commit": "a" * 40})
    compiled = _compile(request, tmp_path)
    resolved = json.loads(compiled.resolved_path.read_text())
    child = resolved["children"][0]
    verified: list[tuple[str, str]] = []
    ranged: list[pathlib.Path] = []

    def verify(record: dict[str, object], label: str) -> pathlib.Path:
        verified.append((label, str(record["snapshot_id"])))
        return pathlib.Path(str(record["path"]))

    def materialize_range(source: pathlib.Path, *_args: object, **_kwargs: object) -> None:
        ranged.append(source)

    class StopBeforeWorkerLaunch(Exception):
        pass

    class CapturingProcess:
        def wait(self) -> int:
            raise StopBeforeWorkerLaunch

    monkeypatch.setattr(subject, "_verify_data", verify)
    monkeypatch.setattr(subject, "_materialize_training_panel", lambda *_a, **_k: "d" * 64)
    monkeypatch.setattr(subject, "_materialize_panel_range", materialize_range)
    monkeypatch.setattr(subject, "_write_child_request", lambda *_a, **_k: None)
    monkeypatch.setattr(subject, "_peak_rss", lambda _process: 0)
    monkeypatch.setattr(subject.subprocess, "Popen", lambda *_a, **_k: CapturingProcess())
    attempt = tmp_path / "validation-child"
    attempt.mkdir()

    with pytest.raises(StopBeforeWorkerLaunch):
        subject._run_generic_child(  # noqa: SLF001
            resolved,
            child,
            campaign_root=tmp_path / "campaign",
            attempt_dir=attempt,
            evaluate=True,
        )

    assert verified == [
        ("selection data", resolved["data"]["selection"]["snapshot_id"]),
        ("final training data", resolved["data"]["final_training"]["snapshot_id"]),
    ]
    assert ranged == [pathlib.Path(resolved["data"]["final_training"]["path"])]


def test_hash_bound_campaign_artifact_is_parsed_from_the_hashed_read(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "artifact.json"
    original = b'{"schema":"probe.v1","value":"original"}\n'
    replacement = b'{"schema":"probe.v1","value":"replacement"}\n'
    path.write_bytes(original)
    real_read = pathlib.Path.read_bytes
    reads = 0

    def adversarial_read(source: pathlib.Path) -> bytes:
        nonlocal reads
        if source == path:
            reads += 1
            return original if reads == 1 else replacement
        return real_read(source)

    monkeypatch.setattr(pathlib.Path, "read_bytes", adversarial_read)
    record = subject._load_exact(  # noqa: SLF001
        path, hashlib.sha256(original).hexdigest(), "probe.v1"
    )
    assert record["value"] == "original"
    assert reads == 1


def test_revision_journal_is_parsed_from_its_hashed_read(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "epochs.jsonl"
    original = b'{"phase":"selection","update":1}\n'
    replacement = b'{"phase":"selection","update":99}\n'
    path.write_bytes(original)
    monkeypatch.setattr(subject, "_rolling_metrics_path", lambda *_args: path)
    real_read = pathlib.Path.read_bytes
    reads = 0

    def adversarial_read(source: pathlib.Path) -> bytes:
        nonlocal reads
        if source == path:
            reads += 1
            return original if reads == 1 else replacement
        return real_read(source)

    monkeypatch.setattr(pathlib.Path, "read_bytes", adversarial_read)
    rows = subject._rolling_metrics_rows({  # noqa: SLF001
        "run_path": str(tmp_path), "model_id": "probe",
        "training_metrics_sha256": hashlib.sha256(original).hexdigest(),
    })
    assert rows == [{"phase": "selection", "update": 1}]
    assert reads == 1


def test_revision_checkpoint_is_parsed_from_its_hashed_read(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model.pt"
    original_buffer = io.BytesIO()
    torch.save({"training": {
        "config": {"epochs": 2}, "epochs_per_phase": 2,
        "steps_per_epoch": 1, "selected_updates": 1,
    }}, original_buffer)
    original = original_buffer.getvalue()
    path.write_bytes(original)
    real_read = pathlib.Path.read_bytes
    reads = 0

    def adversarial_read(source: pathlib.Path) -> bytes:
        nonlocal reads
        if source == path:
            reads += 1
            return original if reads == 1 else b"invalid checkpoint"
        return real_read(source)

    monkeypatch.setattr(pathlib.Path, "read_bytes", adversarial_read)
    training = subject._revision_training_record({  # noqa: SLF001
        "adapter": "sft",
        "checkpoint": {"path": str(path), "sha256": hashlib.sha256(original).hexdigest()},
        "budget": {"maximum_epochs": 2},
    })
    assert training["selected_updates"] == 1
    assert reads == 1
