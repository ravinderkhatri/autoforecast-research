"""Functional check for the archived M5 private-evaluation panel boundary."""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from autoforecast.benchmarks.m5 import (
    build_feature_complete_forecast_parquet,
    build_origin_masked_forecast_parquet,
    build_private_evaluation_parquet,
)


def test_origin_masked_builder_refuses_noncanonical_span_before_writing(tmp_path) -> None:
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="canonical"):
        build_origin_masked_forecast_parquet(
            output,
            origin="20110130",
            horizon=2,
            m5_dir=tmp_path / "not-read",
        )
    assert not output.exists()


def test_private_replay_panel_appends_exact_targets_without_future_exposure_leak(
    tmp_path: pathlib.Path,
) -> None:
    """The d_1942..d_1969 labels become score-only rows and keep a pre-origin exposure state."""
    m5_dir = tmp_path / "m5"
    labels_dir = tmp_path / "private-labels"
    m5_dir.mkdir()
    labels_dir.mkdir()
    private_days = [f"d_{day}" for day in range(1942, 1970)]
    calendar = pd.DataFrame(
        [("d_1", "2011-01-29", 1), ("d_2", "2011-01-30", 1)]
        + [(day, f"2016-06-{index + 1:02d}", 2) for index, day in enumerate(private_days)],
        columns=["d", "date", "wm_yr_wk"],
    )
    calendar.to_csv(m5_dir / "calendar.csv", index=False)
    pd.DataFrame([{
        "id": "FOODS_1_001_CA_1_evaluation",
        "item_id": "FOODS_1_001",
        "dept_id": "FOODS_1",
        "cat_id": "FOODS",
        "store_id": "CA_1",
        "state_id": "CA",
        "d_1": 0,
        "d_2": 0,
    }]).to_csv(m5_dir / "sales_train_evaluation.csv", index=False)
    pd.DataFrame([
        {"store_id": "CA_1", "item_id": "FOODS_1_001", "wm_yr_wk": 1, "sell_price": 2.0},
        {"store_id": "CA_1", "item_id": "FOODS_1_001", "wm_yr_wk": 2, "sell_price": 3.0},
    ]).to_csv(m5_dir / "sell_prices.csv", index=False)
    private = {
        "item_id": "FOODS_1_001",
        "dept_id": "FOODS_1",
        "cat_id": "FOODS",
        "store_id": "CA_1",
        "state_id": "CA",
    }
    private.update({day: 1 for day in private_days})
    pd.DataFrame([private]).to_csv(labels_dir / "sales_test_evaluation.csv", index=False)

    out = build_private_evaluation_parquet(
        tmp_path / "panel", m5_dir=m5_dir, private_labels_dir=labels_dir
    )
    panel = pd.read_parquet(out)
    assert len(panel) == 30
    assert set(panel.loc[panel["d"].isin(private_days), "id"]) == {
        "FOODS_1_001_CA_1_evaluation"
    }
    # The first sale occurs only in released private labels, so it must not establish eligibility
    # at the d_1941 origin. It remains unavailable to the model as an exposure signal.
    assert set(panel.loc[panel["d"].isin(private_days), "observation_state"]) == {"unobserved"}
    assert panel.loc[panel["d"] == "d_1942", "revenue"].iloc[0] == 3.0


def test_feature_complete_panel_keeps_published_future_prices_without_inventing_targets(
    tmp_path: pathlib.Path,
) -> None:
    m5_dir = tmp_path / "m5"
    m5_dir.mkdir()
    pd.DataFrame(
        [("d_1", "2011-01-29", 1), ("d_2", "2011-01-30", 1), ("d_3", "2011-01-31", 2)],
        columns=["d", "date", "wm_yr_wk"],
    ).to_csv(m5_dir / "calendar.csv", index=False)
    pd.DataFrame([{
        "id": "FOODS_1_001_CA_1_evaluation", "item_id": "FOODS_1_001",
        "dept_id": "FOODS_1", "cat_id": "FOODS", "store_id": "CA_1", "state_id": "CA",
        "d_1": 1, "d_2": 0,
    }]).to_csv(m5_dir / "sales_train_evaluation.csv", index=False)
    pd.DataFrame([
        {"store_id": "CA_1", "item_id": "FOODS_1_001", "wm_yr_wk": 1, "sell_price": 2.0},
        {"store_id": "CA_1", "item_id": "FOODS_1_001", "wm_yr_wk": 2, "sell_price": 3.0},
    ]).to_csv(m5_dir / "sell_prices.csv", index=False)

    panel = pd.read_parquet(build_feature_complete_forecast_parquet(tmp_path / "panel", m5_dir=m5_dir))
    future = panel.loc[panel["d"] == "d_3"].iloc[0]
    assert pd.isna(future["qty"])
    assert future["observation_state"] == "unobserved"
    assert future["sell_price"] == 3.0


def test_origin_masked_panel_never_exposes_public_holdout_targets_to_training(
    tmp_path: pathlib.Path,
) -> None:
    m5_dir = tmp_path / "m5"
    m5_dir.mkdir()
    pd.DataFrame(
        [
            ("d_1", "2011-01-29", 1),
            ("d_2", "2011-01-30", 1),
            ("d_3", "2011-01-31", 2),
            ("d_4", "2011-02-01", 2),
        ],
        columns=["d", "date", "wm_yr_wk"],
    ).to_csv(m5_dir / "calendar.csv", index=False)
    pd.DataFrame([{
        "id": "FOODS_1_001_CA_1_evaluation", "item_id": "FOODS_1_001",
        "dept_id": "FOODS_1", "cat_id": "FOODS", "store_id": "CA_1", "state_id": "CA",
        "d_1": 1, "d_2": 2, "d_3": 99,
    }]).to_csv(m5_dir / "sales_train_evaluation.csv", index=False)
    pd.DataFrame([
        {"store_id": "CA_1", "item_id": "FOODS_1_001", "wm_yr_wk": 1, "sell_price": 2.0},
        {"store_id": "CA_1", "item_id": "FOODS_1_001", "wm_yr_wk": 2, "sell_price": 3.0},
    ]).to_csv(m5_dir / "sell_prices.csv", index=False)

    panel = pd.read_parquet(build_origin_masked_forecast_parquet(
        tmp_path / "panel", origin="2011-01-30", horizon=2, m5_dir=m5_dir,
    ))
    holdout = panel.loc[panel["d"].isin(["d_3", "d_4"])]
    assert set(panel.loc[panel["d"].isin(["d_1", "d_2"]), "qty"]) == {1, 2}
    assert holdout["qty"].isna().all()
    assert set(holdout["observation_state"]) == {"unobserved"}
    assert set(holdout["sell_price"]) == {3.0}
