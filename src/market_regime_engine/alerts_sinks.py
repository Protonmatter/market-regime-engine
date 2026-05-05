# SPDX-License-Identifier: Apache-2.0
"""Live transports for routed alerts (v1.3 item E).

The historical :func:`market_regime_engine.alerts.route_alerts` writes
structured rows to the ``routed_alerts`` warehouse table but never
forwards them anywhere. v1.3 adds three live sinks (Slack, Email,
PagerDuty) so a production deployment can wire the engine into its
on-call rotation without writing custom code:

- :class:`SlackSink` — POSTs an Incoming Webhook payload.
- :class:`EmailSink` — sends an SMTP message via ``smtplib``.
- :class:`PagerDutySink` — POSTs a v2 Events API trigger.

Each sink's transport is gated behind an env var. When the env var is
unset the sink is a no-op (returns ``status="skipped"``) so importing
this module is always safe — there is no startup network IO. The
:func:`dispatch_alerts` helper iterates the live sinks for every alert
in a frame and returns a long-format dispatch outcome dataframe ready
to be written via ``Warehouse.write_alert_dispatches``.

Sinks are deliberately stateless and constructed fresh per dispatch so
unit tests can mock ``requests`` / ``smtplib`` without monkeypatching
module globals.
"""

from __future__ import annotations

import json
import os
import smtplib
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.message import EmailMessage
from typing import Protocol

import pandas as pd

try:  # ``requests`` is already a hard dep but keep an import guard so
    # this module is import-safe in pruned environments.
    import requests  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    requests = None  # type: ignore[assignment]


@dataclass(frozen=True)
class _SinkResult:
    sink: str
    status: str  # "ok" | "error" | "skipped"
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"sink": self.sink, "status": self.status, "detail": self.detail}


class _Sink(Protocol):
    name: str

    def send(self, alert: dict) -> _SinkResult: ...


@dataclass
class SlackSink:
    """Slack Incoming Webhook sink.

    The env var ``MRE_SLACK_WEBHOOK_URL`` is consulted at construction
    time (so a unit test can construct one with an explicit URL).
    """

    webhook_url: str | None = None
    timeout: float = 10.0
    name: str = field(default="slack", init=False)

    def __post_init__(self) -> None:
        if self.webhook_url is None:
            self.webhook_url = os.getenv("MRE_SLACK_WEBHOOK_URL")

    def send(self, alert: dict) -> _SinkResult:
        if not self.webhook_url:
            return _SinkResult(self.name, "skipped", "MRE_SLACK_WEBHOOK_URL not set")
        if requests is None:
            return _SinkResult(self.name, "error", "requests not installed")
        # Slack accepts a plain ``{"text": "..."}`` for the simplest
        # incoming-webhook contract; we attach the alert metadata as a
        # fenced JSON block so on-call sees the structured payload.
        text = (
            f":rotating_light: *MRE alert* `{alert.get('alert_type', 'unknown')}` "
            f"({alert.get('severity', 'info')}) — {alert.get('message', '')}"
        )
        try:
            resp = requests.post(
                self.webhook_url,
                json={"text": text},
                timeout=self.timeout,
            )
        except Exception as exc:
            return _SinkResult(self.name, "error", f"slack post failed: {exc}")
        if resp.status_code >= 300:
            return _SinkResult(self.name, "error", f"slack returned {resp.status_code}: {resp.text[:200]}")
        return _SinkResult(self.name, "ok", f"slack 200 ({len(resp.content)}B)")


@dataclass
class EmailSink:
    """SMTP email sink.

    Configurable via ``MRE_SMTP_HOST``, ``MRE_SMTP_PORT``,
    ``MRE_SMTP_FROM``, ``MRE_SMTP_TO`` (comma separated),
    ``MRE_SMTP_USER``, ``MRE_SMTP_PASSWORD``. STARTTLS is used when
    ``MRE_SMTP_PORT == 587`` or when ``MRE_SMTP_TLS`` is truthy.
    """

    smtp_host: str | None = None
    smtp_port: int | None = None
    from_addr: str | None = None
    to_addrs: list[str] | None = None
    smtp_user: str | None = None
    smtp_password: str | None = None
    starttls: bool = False
    timeout: float = 15.0
    name: str = field(default="email", init=False)

    def __post_init__(self) -> None:
        if self.smtp_host is None:
            self.smtp_host = os.getenv("MRE_SMTP_HOST")
        if self.smtp_port is None:
            raw = os.getenv("MRE_SMTP_PORT")
            self.smtp_port = int(raw) if raw and raw.isdigit() else None
        if self.from_addr is None:
            self.from_addr = os.getenv("MRE_SMTP_FROM")
        if self.to_addrs is None:
            raw_to = os.getenv("MRE_SMTP_TO", "")
            self.to_addrs = [a.strip() for a in raw_to.split(",") if a.strip()]
        if self.smtp_user is None:
            self.smtp_user = os.getenv("MRE_SMTP_USER")
        if self.smtp_password is None:
            self.smtp_password = os.getenv("MRE_SMTP_PASSWORD")
        if not self.starttls:
            tls_env = os.getenv("MRE_SMTP_TLS", "").lower()
            self.starttls = tls_env in {"1", "true", "yes"} or self.smtp_port == 587

    def _configured(self) -> bool:
        return bool(self.smtp_host and self.smtp_port and self.from_addr and self.to_addrs)

    def send(self, alert: dict) -> _SinkResult:
        if not self._configured():
            return _SinkResult(self.name, "skipped", "MRE_SMTP_* not fully configured")
        msg = EmailMessage()
        msg["Subject"] = f"[MRE {alert.get('severity', 'info').upper()}] {alert.get('alert_type', 'unknown')}"
        msg["From"] = self.from_addr
        msg["To"] = ", ".join(self.to_addrs or [])
        body = (
            f"Alert: {alert.get('alert_type', 'unknown')}\n"
            f"Severity: {alert.get('severity', 'info')}\n"
            f"Channel: {alert.get('channel', 'unknown')}\n"
            f"Date: {alert.get('date', 'unknown')}\n"
            f"Message: {alert.get('message', '')}\n\n"
            f"Metadata:\n{alert.get('metadata_json', '{}')}\n"
        )
        msg.set_content(body)
        try:
            host = str(self.smtp_host or "")
            with smtplib.SMTP(host, int(self.smtp_port or 0), timeout=self.timeout) as smtp:
                if self.starttls:
                    smtp.starttls()
                if self.smtp_user and self.smtp_password:
                    smtp.login(self.smtp_user, self.smtp_password)
                smtp.send_message(msg)
        except Exception as exc:
            return _SinkResult(self.name, "error", f"smtp send failed: {exc}")
        return _SinkResult(self.name, "ok", f"smtp delivered to {len(self.to_addrs or [])} recipients")


@dataclass
class PagerDutySink:
    """PagerDuty v2 Events API sink.

    Configurable via ``MRE_PAGERDUTY_INTEGRATION_KEY``. Severities map
    to PagerDuty levels:

      - ``high`` / ``critical`` → ``critical``
      - ``medium``              → ``error``
      - ``low``                 → ``warning``
      - everything else         → ``info``
    """

    integration_key: str | None = None
    timeout: float = 10.0
    events_url: str = "https://events.pagerduty.com/v2/enqueue"
    name: str = field(default="pagerduty", init=False)

    def __post_init__(self) -> None:
        if self.integration_key is None:
            self.integration_key = os.getenv("MRE_PAGERDUTY_INTEGRATION_KEY")

    @staticmethod
    def _map_severity(severity: str) -> str:
        s = (severity or "").lower()
        if s in {"high", "critical"}:
            return "critical"
        if s == "medium":
            return "error"
        if s == "low":
            return "warning"
        return "info"

    def send(self, alert: dict) -> _SinkResult:
        if not self.integration_key:
            return _SinkResult(self.name, "skipped", "MRE_PAGERDUTY_INTEGRATION_KEY not set")
        if requests is None:
            return _SinkResult(self.name, "error", "requests not installed")
        payload = {
            "routing_key": self.integration_key,
            "event_action": "trigger",
            "dedup_key": f"mre-{alert.get('alert_type', 'unknown')}-{alert.get('date', 'unknown')}",
            "payload": {
                "summary": alert.get("message", "MRE alert"),
                "severity": self._map_severity(str(alert.get("severity", "info"))),
                "source": "market-regime-engine",
                "component": str(alert.get("channel", "model_risk")),
                "class": str(alert.get("alert_type", "unknown")),
                "custom_details": alert,
            },
        }
        try:
            resp = requests.post(self.events_url, json=payload, timeout=self.timeout)
        except Exception as exc:
            return _SinkResult(self.name, "error", f"pagerduty post failed: {exc}")
        if resp.status_code >= 300:
            return _SinkResult(self.name, "error", f"pagerduty returned {resp.status_code}: {resp.text[:200]}")
        return _SinkResult(self.name, "ok", f"pagerduty 202 ({len(resp.content)}B)")


def configured_sinks() -> list[_Sink]:
    """Return the live sinks visible from the environment.

    Each sink is constructed unconditionally; the ones whose env vars
    are unset will report ``status="skipped"`` on send. Tests can pass
    an explicit list to :func:`dispatch_alerts` to bypass env-driven
    construction.
    """
    return [SlackSink(), EmailSink(), PagerDutySink()]


def dispatch_alerts(
    alerts: pd.DataFrame,
    *,
    sinks: Iterable[_Sink] | None = None,
) -> pd.DataFrame:
    """Forward every alert through every configured sink.

    Returns a long-format frame matching the ``alert_dispatches``
    warehouse table (``date, alert_type, sink, status, detail,
    dispatched_at_utc, metadata_json``). The function never raises;
    transport-level errors are recorded as ``status="error"`` rows so a
    flapping sink never blocks the orchestration pipeline.
    """
    if alerts is None or alerts.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "alert_type",
                "sink",
                "status",
                "detail",
                "dispatched_at_utc",
                "metadata_json",
            ]
        )
    live_sinks = list(sinks) if sinks is not None else configured_sinks()
    rows: list[dict] = []
    now = datetime.now(UTC).isoformat(timespec="seconds")
    for _, row in alerts.iterrows():
        alert = row.to_dict()
        for sink in live_sinks:
            try:
                result = sink.send(alert)
            except Exception as exc:  # pragma: no cover - defensive
                result = _SinkResult(getattr(sink, "name", "unknown"), "error", str(exc))
            rows.append(
                {
                    "date": str(alert.get("date", "")),
                    "alert_type": str(alert.get("alert_type", "unknown")),
                    "sink": result.sink,
                    "status": result.status,
                    "detail": result.detail,
                    "dispatched_at_utc": now,
                    "metadata_json": json.dumps(
                        {"severity": alert.get("severity", "info")},
                        sort_keys=True,
                    ),
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "EmailSink",
    "PagerDutySink",
    "SlackSink",
    "configured_sinks",
    "dispatch_alerts",
]
