"""Functional data-agnostic admission check; deliberately independent of M5 artifacts."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import pathlib
import sys
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from autoforecast import adapters
from autoforecast.adapter_contracts import DESCRIPTORS, descriptor
from autoforecast.adapters import adapter_capabilities, known_adapter_names
from autoforecast.data_admission import (
    get_model_input_contract,
    load_dataset_contract,
    preflight_candidates,
    validate_model_compatibility,
)
from autoforecast.model_catalog import discover_candidates, load_selection_policy
from autoforecast.model_names import REVISION_MODELS
from autoforecast.model_readiness import create_resolved_run, model_registration
from autoforecast.panel import Panel, PlanSource
from autoforecast.profiling import Profile
from autoforecast.tracking_plan import tracking_plan
from autoforecast.tuning_support import tuning_support


def _profile(minimum: int, maximum: int) -> Profile:
    return Profile(
        1000,
        2,
        500,
        "2026-01-01",
        1.0,
        0.2,
        0.2,
        3.0,
        2.0,
        min_eligible_history=minimum,
        max_eligible_history=maximum,
    )


def _write_client_panel(directory: pathlib.Path) -> pathlib.Path:
    path = directory / "client_panel.parquet"
    pq.write_table(
        pa.table(
            {
                "client_sku": ["a", "b"],
                "trading_day": pa.array([dt.date(2025, 1, 1), dt.date(2025, 1, 2)]),
                "units_sold": [3, 4],
                "observed_state": ["exact", "exact"],
                "competitor_price": [3.0, 4.0],
            }
        ),
        path,
    )
    return path


def test_dataset_contract_maps_client_columns_to_generic_panel_roles() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        directory = pathlib.Path(temporary)
        path = _write_client_panel(directory)
        contract_path = directory / "client.yaml"
        contract_path.write_text(
            "schema_version: 1\ndataset:\n  id: client_daily\n  panel:\n"
            "    series_col: client_sku\n    time_col: trading_day\n"
            "    target_col: units_sold\n    frequency: D\n"
            "    past_only: [competitor_price]\n"
        )
        panel = load_dataset_contract(contract_path).panel(path).validate()
        assert panel.past_only == ("competitor_price",)
        assert (panel.series_col, panel.time_col, panel.target_col, panel.frequency) == (
            "client_sku",
            "trading_day",
            "units_sold",
            "D",
        )


def test_dataset_authoring_defaults_and_plan_survive_persisted_round_trip(tmp_path) -> None:
    path = tmp_path / "dataset.yaml"
    path.write_text(
        "schema_version: 1\ndataset:\n  id: daily\n  panel:\n"
        "    series_col: id\n    time_col: date\n    target_col: qty\n"
        "    known_future: [price]\n    price_col: price\n"
        "    plan:\n      path: /plans\n      valid_date_col: valid_date\n"
        "      known_at_col: known_at\n      price_col: price\n"
    )
    contract = load_dataset_contract(path)
    panel = contract.panel("/panel")
    assert panel.frequency == "D" and panel.season_length == 7
    assert panel.static_reals == () and panel.past_only == ()
    assert panel.plan == PlanSource("/plans", "valid_date", "known_at", price_col="price")
    assert Panel.from_record(json.loads(json.dumps(dataclasses.asdict(panel)))) == panel
    # Preparation serializes the authoring roles back to YAML. They must stay plain values.
    import yaml

    prepared_path = tmp_path / "prepared-dataset.yaml"
    prepared_path.write_text(yaml.safe_dump({
        "schema_version": 2,
        "dataset": {"id": contract.id, "panel": contract.panel_roles,
                    "required_series_count": 1},
    }))
    assert load_dataset_contract(prepared_path).panel("/panel") == panel


def test_revision_adapter_registration_contracts_are_exact() -> None:
    panel = Panel("unused.parquet", "id", "date", "target")
    for variant in ("v2", "v3"):
        adapter = REVISION_MODELS[variant]
        assert adapters._ISOLATED_MODULE[adapter] == "mft"  # noqa: SLF001
        assert adapter_capabilities(adapter) == {
            "consumes_exposure": True,
            "handles_right_censoring": True,
            "supported_estimands": frozenset({"sales_given_availability", "unconstrained_demand"}),
            "supported_frequency_families": frozenset({"D"}),
            "supports_strided_daily": False,
            "tracking": tracking_plan(adapter),
            "descriptor": descriptor(adapter),
        }
        contract = get_model_input_contract(adapter)
        assert contract.frequency == "D"
        assert contract.minimum_history(season_length=7, horizon=28) == 56
        run = create_resolved_run(
            model_id=adapter,
            adapter=adapter,
            origin="2024-01-31",
            horizon=7,
            settings={"encoder_days": 35},
        )
        registration = model_registration(adapter)
        assert registration.required_history(panel, run) == 42
        padded = create_resolved_run(
            model_id=adapter,
            adapter=adapter,
            origin="2024-01-31",
            horizon=28,
            settings={"required_history": 1457},
        )
        assert registration.required_history(panel, padded) == 1457
        assert registration.requires_complete_window
        assert tracking_plan(adapter).mode == "reporter"
        assert tracking_plan(adapter).flavor == "none"
        assert tuning_support(adapter).supported


def test_dataset_contract_refuses_duplicate_yaml_keys() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = pathlib.Path(temporary) / "ambiguous.yaml"
        path.write_text(
            "schema_version: 1\nschema_version: 1\ndataset: {}\n",
            encoding="utf-8",
        )
        try:
            load_dataset_contract(path)
        except ValueError as error:
            assert "duplicate YAML mapping key" in str(error)
            return
        raise AssertionError("dataset contract accepted an ambiguous YAML mapping")


def test_m5_contract_explicitly_declares_the_released_future_price_schedule() -> None:
    """The M5 replay must not silently read future price rows as realised data."""
    root = pathlib.Path(__file__).resolve().parents[1]
    contract = load_dataset_contract(root / "configs/datasets/m5_daily.yaml")
    assert contract.panel_roles["price_col"] == "sell_price"
    assert contract.panel_roles["known_future"] == ("sell_price",)


def test_model_compatibility_admits_only_data_that_meets_each_model_contract() -> None:
    assert all(get_model_input_contract(name) for name in known_adapter_names())
    assert adapter_capabilities("tft")["supported_estimands"]
    assert get_model_input_contract("tft").minimum_history(season_length=7, horizon=28) == 208
    assert get_model_input_contract("tft_q").minimum_history(season_length=7, horizon=28) == 105
    with tempfile.TemporaryDirectory() as temporary:
        directory = pathlib.Path(temporary)
        path = _write_client_panel(directory)
        panel = Panel(str(path), "client_sku", "trading_day", "units_sold")
        eligible = validate_model_compatibility("tft", panel, _profile(208, 208), horizon=28)
        assert eligible.status == "eligible" and eligible.may_run
        assert validate_model_compatibility("tft_q", panel, _profile(105, 105), horizon=28).may_run
        assert not validate_model_compatibility(
            "tft_q", panel, _profile(104, 104), horizon=28
        ).may_run
        assert (
            validate_model_compatibility("tft", panel, _profile(100, 100), horizon=28).status
            == "ineligible"
        )
        assert (
            validate_model_compatibility(
                "tft",
                Panel(str(path), "client_sku", "trading_day", "units_sold", frequency="M"),
                _profile(208, 208),
                horizon=28,
            ).status
            == "ineligible"
        )
        assert (
            validate_model_compatibility(
                "autogluon_zoo",
                Panel(
                    str(path),
                    "client_sku",
                    "trading_day",
                    "units_sold",
                    exposure_col="observed_state",
                    target_estimand="unconstrained_demand",
                ),
                _profile(1, 1),
                horizon=28,
            ).status
            == "ineligible"
        )
        bad_path = directory / "bad_panel.parquet"
        pq.write_table(
            pa.table(
                {
                    "client_sku": ["a"],
                    "trading_day": pa.array([dt.date(2025, 1, 1)]),
                    "units_sold": ["three"],
                }
            ),
            bad_path,
        )
        assert (
            validate_model_compatibility(
                "seasonal_naive",
                Panel(str(bad_path), "client_sku", "trading_day", "units_sold"),
                _profile(7, 7),
                horizon=28,
            ).status
            == "ineligible"
        )
        invalid_frequency = validate_model_compatibility(
            "seasonal_naive",
            Panel(
                str(path), "client_sku", "trading_day", "units_sold",
                frequency="hourly", season_length=7,
            ),
            _profile(7, 7),
            horizon=1,
        )
        assert invalid_frequency.status == "ineligible"
        assert any("supported frequencies" in reason for reason in invalid_frequency.reasons)


def test_every_adapter_explicitly_declares_family_and_daily_stride_support() -> None:
    all_families = ("H", "D", "W", "M", "Y")
    for name in DESCRIPTORS:
        declared = descriptor(name).supported_frequency_families
        if name in {"seasonal_naive", "last_value"}:
            assert declared == all_families
            assert descriptor(name).supports_strided_daily
            assert all(get_model_input_contract(name).supports(code) for code in all_families)
            assert get_model_input_contract(name).supports("MS")
            assert get_model_input_contract(name).supports("YS")
            assert get_model_input_contract(name).supports("7D")
        else:
            assert declared == ("D",)
            assert not descriptor(name).supports_strided_daily
            assert get_model_input_contract(name).supports("D")
            assert not get_model_input_contract(name).supports("7D")
            assert not any(
                get_model_input_contract(name).supports(code) for code in ("H", "W", "M", "Y")
            )


def test_preflight_selects_data_compatible_catalog_candidates_before_fitting() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = _write_client_panel(pathlib.Path(temporary))
        panel = Panel(str(path), "client_sku", "trading_day", "units_sold")
        root = pathlib.Path(__file__).resolve().parents[1]
        preflight = preflight_candidates(
            discover_candidates(root / "configs/models"),
            load_selection_policy(root / "configs/autoforecast/selection/development_top_k.yaml"),
            panel,
            _profile(1024, 1024),
            horizon=28,
        )
        assert len(preflight.eligible) == 16
        assert "timesfm_hnb" in {candidate.id for candidate in preflight.eligible}


if __name__ == "__main__":
    test_dataset_contract_maps_client_columns_to_generic_panel_roles()
    test_m5_contract_explicitly_declares_the_released_future_price_schedule()
    test_model_compatibility_admits_only_data_that_meets_each_model_contract()
    test_preflight_selects_data_compatible_catalog_candidates_before_fitting()
    print("PASS test_data_admission")


def test_dataset_contract_refuses_incomplete_and_nonstring_role_names(tmp_path):
    import pytest
    import yaml

    from autoforecast.data_admission import DatasetContractError

    path = tmp_path / "dataset.yaml"
    for roles in ({}, {"series_col": "id"}, {1: "id", "time_col": "date"}):
        path.write_text(yaml.safe_dump({
            "schema_version": 1, "dataset": {"id": "measurements", "panel": roles},
        }))
        with pytest.raises(DatasetContractError, match="dataset.panel"):
            load_dataset_contract(path)


def test_dataset_authoring_refuses_malformed_plan_sources(tmp_path) -> None:
    import pytest
    import yaml

    from autoforecast.data_admission import DatasetContractError

    path = tmp_path / "dataset.yaml"
    valid = {"path": "/plans", "valid_date_col": "date", "known_at_col": "known_at",
             "price_col": "price"}
    for plan in ("not-a-mapping", {}, {**valid, "unknown": "unsupported"},
                 {**valid, "price_col": None}):
        path.write_text(yaml.safe_dump({
            "schema_version": 1, "dataset": {"id": "daily", "panel": {
                "series_col": "id", "time_col": "date", "target_col": "qty", "plan": plan,
            }},
        }))
        with pytest.raises(DatasetContractError):
            load_dataset_contract(path)
