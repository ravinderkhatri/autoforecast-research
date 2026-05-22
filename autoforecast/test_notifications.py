"""Functional checks for the opt-in terminal notification boundary."""

from __future__ import annotations

from unittest import mock

from autoforecast.model_api import CheckReport, CheckResult, DataFacts
from autoforecast.notifications import (
    campaign_notification_from_environment,
    notifier_from_environment,
)
from autoforecast.train_workflow import ModelOutcome, RunSummary


def _summary(tmp_path):
    facts = DataFacts(
        "2026-09-11", "a" * 64, ("2026-01-01", "2026-09-11"), 1, 1, 1, 0, 0, {}, {}, None, {}
    )
    report = CheckReport("data", facts, (CheckResult("data_ready", "Data ready", "passed", "ok"),))
    return RunSummary(
        "functional-notify",
        tmp_path,
        (ModelOutcome("seasonal_naive", "completed", "done"),),
        report,
        None,
    )


def test_environment_notifier_delivers_a_compact_slack_summary(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("AUTOFORECAST_SLACK_WEBHOOK_URL", "https://notify.example.test/hook")
    notifier = notifier_from_environment()
    assert notifier is not None
    response = mock.MagicMock(status=200)
    response.__enter__.return_value = response
    with mock.patch("urllib.request.urlopen", return_value=response) as post:
        notifier(_summary(tmp_path))
    payload = post.call_args.args[0].data.decode("utf-8")
    assert "seasonal_naive" in payload and "functional-notify" in payload


def test_no_environment_configuration_means_no_notification() -> None:
    assert notifier_from_environment() is None


def test_campaign_notification_records_disabled_delivery_without_network(
    monkeypatch,
) -> None:
    monkeypatch.delenv("AUTOFORECAST_SLACK_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("AUTOFORECAST_EMAIL_TO", raising=False)
    result = campaign_notification_from_environment(
        campaign_name="overnight-run", status="completed", path="/local/campaign"
    )
    assert result["status"] == "disabled"
    assert result["configured_sinks"] == []


def test_campaign_notification_attempts_every_sink_and_returns_failures(monkeypatch) -> None:
    monkeypatch.setenv("AUTOFORECAST_SLACK_WEBHOOK_URL", "https://notify.example.test/hook")
    monkeypatch.setenv("AUTOFORECAST_EMAIL_TO", "owner@example.test")
    with (
        mock.patch("autoforecast.notifications._slack", side_effect=RuntimeError("offline")),
        mock.patch("autoforecast.notifications._email") as email,
    ):
        result = campaign_notification_from_environment(
            campaign_name="overnight-run", status="failed", path="/local/campaign"
        )
    email.assert_called_once()
    assert result["status"] == "failed"
    assert result["delivered_sinks"] == ["email"]
    assert result["failures"] == ["Slack delivery failed (RuntimeError)"]


def test_campaign_notification_never_persists_secret_bearing_exception_text(monkeypatch) -> None:
    secret = "https://notify.example.test/secret-token"
    monkeypatch.setenv("AUTOFORECAST_SLACK_WEBHOOK_URL", secret)
    monkeypatch.delenv("AUTOFORECAST_EMAIL_TO", raising=False)
    with mock.patch(
        "autoforecast.notifications._slack",
        side_effect=RuntimeError(secret),
    ):
        result = campaign_notification_from_environment(
            campaign_name="overnight-run", status="failed", path="/local/campaign"
        )
    assert secret not in str(result)
    assert result["failures"] == ["Slack delivery failed (RuntimeError)"]
