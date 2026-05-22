"""Direct contract checks for the framework-independent model API records."""

from __future__ import annotations

from autoforecast.model_api import CheckResult, CheckStatus


def test_check_result_blocks_run_only_for_failed_status() -> None:
    statuses: tuple[CheckStatus, ...] = (
        "passed",
        "failed",
        "warning",
        "not_checked",
        "not_applicable",
    )

    decisions = {
        status: CheckResult("check", "Check", status, "result").blocks_run
        for status in statuses
    }

    assert decisions == {
        "passed": False,
        "failed": True,
        "warning": False,
        "not_checked": False,
        "not_applicable": False,
    }
