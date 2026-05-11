# SPDX-License-Identifier: Apache-2.0
"""Correlation-ID middleware + log integration (PR-7 §I).

Per plan §7 §I + REVIEW.md §3.6 P2: every inbound HTTP request gets
an ``X-Request-ID`` header (generated as UUID4 if absent), threaded
through a :mod:`contextvars` context so every log line emitted from
the request handler carries ``request_id`` automatically.

The middleware is FastAPI-compatible (Starlette ``BaseHTTPMiddleware``)
but the underlying primitive — :data:`_REQUEST_ID_CTX` — is framework-
agnostic so the same correlation id propagates through CLI commands
that opt in via :func:`set_request_id`.

Logging integration: :class:`CorrelationIdLogFilter` injects the
current request_id onto every :class:`logging.LogRecord`. Install via
:func:`install_correlation_id_log_filter` once at process start.
"""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar
from typing import Any

__all__ = [
    "CorrelationIdLogFilter",
    "CorrelationIdMiddleware",
    "current_request_id",
    "install_correlation_id_log_filter",
    "set_request_id",
]


_REQUEST_ID_HEADER = "x-request-id"
_REQUEST_ID_RESPONSE_HEADER = "X-Request-ID"
_REQUEST_ID_CTX: ContextVar[str | None] = ContextVar("mre_request_id", default=None)


def current_request_id() -> str | None:
    """Return the request_id for the current async / sync context."""
    return _REQUEST_ID_CTX.get()


def set_request_id(value: str | None) -> None:
    """Override the request_id for the current context.

    CLI commands or background workers can call this to propagate a
    correlation id captured from an upstream system.
    """
    _REQUEST_ID_CTX.set(value)


class CorrelationIdMiddleware:
    """Starlette / FastAPI middleware that threads ``X-Request-ID``.

    - Reads ``X-Request-ID`` from the inbound request; falls back to a
      fresh UUID4 hex string if absent or empty.
    - Sets the value on :data:`_REQUEST_ID_CTX` for the lifetime of
      the request handler so log lines emitted in nested coroutines
      see the same correlation id.
    - Echoes the resolved id in the response under ``X-Request-ID``.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = scope.get("headers") or []
        existing = self._header_value(headers, _REQUEST_ID_HEADER.encode("ascii"))
        if existing:
            try:
                request_id = existing.decode("ascii")
            except Exception:
                request_id = uuid.uuid4().hex
        else:
            request_id = uuid.uuid4().hex
        token = _REQUEST_ID_CTX.set(request_id)
        try:
            async def _send_with_header(message: dict[str, Any]) -> None:
                if message.get("type") == "http.response.start":
                    response_headers = list(message.get("headers") or [])
                    response_headers = [
                        (k, v)
                        for k, v in response_headers
                        if k.decode("latin-1").lower() != _REQUEST_ID_RESPONSE_HEADER.lower()
                    ]
                    response_headers.append(
                        (
                            _REQUEST_ID_RESPONSE_HEADER.encode("latin-1"),
                            request_id.encode("latin-1"),
                        )
                    )
                    message["headers"] = response_headers
                await send(message)

            await self.app(scope, receive, _send_with_header)
        finally:
            _REQUEST_ID_CTX.reset(token)

    @staticmethod
    def _header_value(
        headers: list[tuple[bytes, bytes]], target: bytes
    ) -> bytes | None:
        target_lower = target.lower()
        for raw_key, raw_value in headers:
            if raw_key.lower() == target_lower:
                return raw_value
        return None


class CorrelationIdLogFilter(logging.Filter):
    """Attach the current request_id (or empty) onto every log record.

    Once :func:`install_correlation_id_log_filter` has been called, the
    JSON / human formatters in :mod:`logging_setup` see ``request_id``
    on the record attribute namespace and emit it alongside the rest of
    the structured log payload.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        request_id = current_request_id()
        record.request_id = request_id if request_id is not None else ""
        return True


def install_correlation_id_log_filter() -> CorrelationIdLogFilter:
    """Idempotently install :class:`CorrelationIdLogFilter` on root.

    Returns the installed filter so callers can detach it later if
    they need a custom logging configuration.
    """
    root = logging.getLogger()
    for existing in root.filters:
        if isinstance(existing, CorrelationIdLogFilter):
            return existing
    correlation_filter = CorrelationIdLogFilter()
    root.addFilter(correlation_filter)
    for handler in root.handlers:
        handler.addFilter(correlation_filter)
    return correlation_filter
