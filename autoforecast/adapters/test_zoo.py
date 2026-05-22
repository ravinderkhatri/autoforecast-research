"""Boundary checks for the optional AutoGluon adapter; no model download is required."""

from __future__ import annotations

import pytest

from autoforecast.adapters.zoo import fit_predict
from autoforecast.panel import Panel


def test_daily_adapter_refuses_weekly_panel_before_optional_runtime_or_spark_io() -> None:
    class ReadSentinel:
        @property
        def read(self):
            raise AssertionError("weekly panel reached AutoGluon/Spark")

    weekly = Panel("/not-read", "id", "date", "qty", frequency="W")
    with pytest.raises(ValueError, match="daily"):
        fit_predict(
            weekly,
            "2024-01-01",
            2,
            spark=ReadSentinel(),
            out_dir="/not-written",
        )
