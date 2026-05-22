"""Opt-in terminal notifications for local AutoForecast run groups.

Secrets and recipient addresses stay in the environment; run YAML and immutable bundles contain
only the factual outcome.  Delivery is deliberately a best-effort side effect.
"""

from __future__ import annotations

import json
import os
import smtplib
import urllib.request
from collections.abc import Callable
from email.message import EmailMessage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autoforecast.train_workflow import RunSummary

__all__ = [
    "NotificationError",
    "campaign_notification_from_environment",
    "notifier_from_environment",
]


class NotificationError(RuntimeError):
    """One or more configured notification deliveries failed."""


def _delivery_failure(sink: str, error: Exception) -> str:
    """Describe a sink failure without persisting credential-bearing exception text."""
    return f"{sink} delivery failed ({type(error).__name__})"


def _summary(summary: RunSummary) -> tuple[str, str]:
    completed = ", ".join(summary.trained_models) or "none"
    failed = (
        ", ".join(
            outcome.model_id
            for outcome in summary.outcomes
            if outcome.status in {"blocked", "not_ready", "failed"}
        )
        or "none"
    )
    outcome = "completed" if summary.trained_models else "no model trained"
    subject = f"AutoForecast {summary.run_id}: {outcome}"
    body = (
        f"Run: {summary.run_id}\n"
        f"Outcome: {'completed' if summary.trained_models else 'no model trained'}\n"
        f"Completed models: {completed}\n"
        f"Blocked, not ready, or failed: {failed}\n"
        f"Local report: {summary.path / 'report.html'}\n"
    )
    return subject, body


def _slack(webhook_url: str, subject: str, body: str) -> None:
    request = urllib.request.Request(
        webhook_url,
        data=json.dumps({"text": f"{subject}\n{body}"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310 - opt-in URL
        if not 200 <= response.status < 300:
            raise NotificationError(f"Slack returned HTTP {response.status}")


def _email(recipient: str, subject: str, body: str) -> None:
    host = os.environ.get("AUTOFORECAST_SMTP_HOST")
    sender = os.environ.get("AUTOFORECAST_SMTP_FROM")
    if not host or not sender:
        raise NotificationError(
            "email delivery requires AUTOFORECAST_SMTP_HOST and AUTOFORECAST_SMTP_FROM"
        )
    raw_port = os.environ.get("AUTOFORECAST_SMTP_PORT", "587")
    try:
        port = int(raw_port)
    except ValueError as error:
        raise NotificationError("AUTOFORECAST_SMTP_PORT must be an integer") from error
    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)
    with smtplib.SMTP(host, port, timeout=10) as client:
        if os.environ.get("AUTOFORECAST_SMTP_STARTTLS", "true").lower() != "false":
            client.starttls()
        username = os.environ.get("AUTOFORECAST_SMTP_USERNAME")
        password = os.environ.get("AUTOFORECAST_SMTP_PASSWORD")
        if username or password:
            if not username or not password:
                raise NotificationError(
                    "SMTP authentication needs both AUTOFORECAST_SMTP_USERNAME and "
                    "AUTOFORECAST_SMTP_PASSWORD"
                )
            client.login(username, password)
        client.send_message(message)


def notifier_from_environment() -> Callable[[RunSummary], None] | None:
    """Return a terminal notifier when Slack and/or email are explicitly configured.

    ``AUTOFORECAST_SLACK_WEBHOOK_URL`` enables Slack. ``AUTOFORECAST_EMAIL_TO`` enables
    SMTP email.  Both may be enabled; each is attempted so one failed sink does not prevent the
    other from receiving the outcome.  The workflow catches the resulting error after it has
    already published the immutable run report.
    """
    webhook_url = os.environ.get("AUTOFORECAST_SLACK_WEBHOOK_URL")
    email_to = os.environ.get("AUTOFORECAST_EMAIL_TO")
    if not webhook_url and not email_to:
        return None

    def notify(summary: RunSummary) -> None:
        subject, body = _summary(summary)
        failures: list[str] = []
        if webhook_url:
            try:
                _slack(webhook_url, subject, body)
            except Exception as error:
                failures.append(_delivery_failure("Slack", error))
        if email_to:
            try:
                _email(email_to, subject, body)
            except Exception as error:
                failures.append(_delivery_failure("email", error))
        if failures:
            raise NotificationError("; ".join(failures))

    return notify


def campaign_notification_from_environment(
    *, campaign_name: str, status: str, path: str
) -> dict[str, object]:
    """Attempt each opt-in terminal sink and return a durable, non-governing result.

    The caller persists this result exactly once while it owns the campaign lease. Delivery can
    fail without changing the research result, but a restart can then see that delivery was
    already attempted and must not send a duplicate terminal message.
    """
    if status not in {"completed", "failed"}:
        raise NotificationError("campaign notification requires a terminal status")
    webhook_url = os.environ.get("AUTOFORECAST_SLACK_WEBHOOK_URL")
    email_to = os.environ.get("AUTOFORECAST_EMAIL_TO")
    configured = [name for name, value in (("slack", webhook_url), ("email", email_to)) if value]
    subject = f"AutoForecast campaign {campaign_name}: {status}"
    body = f"Campaign: {campaign_name}\nOutcome: {status}\nLocal artifacts: {path}\n"
    failures: list[str] = []
    delivered: list[str] = []
    if webhook_url:
        try:
            _slack(webhook_url, subject, body)
        except Exception as error:
            failures.append(_delivery_failure("Slack", error))
        else:
            delivered.append("slack")
    if email_to:
        try:
            _email(email_to, subject, body)
        except Exception as error:
            failures.append(_delivery_failure("email", error))
        else:
            delivered.append("email")
    return {
        "schema": "autoforecast.campaign-terminal-notification.v1",
        "status": "disabled" if not configured else ("failed" if failures else "sent"),
        "campaign_name": campaign_name,
        "campaign_status": status,
        "configured_sinks": configured,
        "delivered_sinks": delivered,
        "failures": failures,
        "artifact_path": path,
    }
