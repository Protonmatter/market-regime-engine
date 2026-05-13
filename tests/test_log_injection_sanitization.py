# SPDX-License-Identifier: Apache-2.0
"""v1.6.0 — F8 / Finding §3.12 regression tests.

Pin the contract that caller-controlled string fields (``request_id``,
``cusip``, ``model_run_id``) are sanitized before being interpolated
into log lines so a caller cannot split a single log record into two
by smuggling ``\\n`` or ``\\r`` into the payload.

The fix is centralised in :func:`fixed_income.correlation.log_safe`,
applied at:

- :class:`CorrelationIdMiddleware` (boundary between caller HTTP
  header and the contextvar that the log filter reads).
- :func:`fixed_income.api.build_router` log call sites that
  interpolate ``body.request_id``.
- :func:`fixed_income.tca_segmentation.materialize_tca_segments_for_day`
  log warning that interpolates a caller request id.
"""

from __future__ import annotations

import logging
from io import StringIO

import pytest

from market_regime_engine.fixed_income.correlation import log_safe


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("plain-id", "plain-id"),
        ("with-\nnewline", "with-\\nnewline"),
        ("carriage\rret", "carriage\\rret"),
        ("crlf\r\nstamp", "crlf\\r\\nstamp"),
        ("multi\nline\npayload", "multi\\nline\\npayload"),
        ("", ""),
        # Non-string payloads stringify first, then sanitise.
        (42, "42"),
        (None, "None"),
    ],
)
def test_log_safe_strips_newlines_and_carriage_returns(
    payload: object, expected: str
) -> None:
    assert log_safe(payload) == expected


def test_log_safe_preserves_safe_unicode() -> None:
    """Non-control unicode passes through unchanged."""
    assert log_safe("\u00e9") == "\u00e9"
    assert log_safe("emoji-test") == "emoji-test"


def test_log_call_with_injected_newline_produces_single_line() -> None:
    """End-to-end: a log call wrapping a caller-controlled value with
    ``log_safe`` must produce a single log record."""
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s|%(message)s"))
    logger = logging.getLogger("mre.test.f8")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.INFO)

    request_id = "abc\nINFO|synthetic-second-record"
    logger.info("audit (request_id=%s)", log_safe(request_id))

    text = stream.getvalue()
    # The injected fake INFO line MUST NOT appear as its own record.
    assert text.count("\n") == 1, (
        f"expected one trailing newline, got {text.count(chr(10))} lines: {text!r}"
    )
    assert "synthetic-second-record" in text  # payload preserved
    assert "\\n" in text  # newline rendered as escape


def test_correlation_middleware_sanitises_inbound_request_id_header() -> None:
    """The CorrelationIdMiddleware decodes the inbound X-Request-ID
    header as ASCII (which permits \\n / \\r) and writes the value into
    the contextvar. The fix sanitises here so the rest of the codebase
    sees a log-safe string."""
    import asyncio
    import inspect

    from market_regime_engine.fixed_income.correlation import (
        CorrelationIdMiddleware,
        current_request_id,
    )

    # Confirm the middleware source calls log_safe — pinning the
    # contract via source inspection so a refactor that drops the
    # sanitise step fails this test.
    src = inspect.getsource(CorrelationIdMiddleware.__call__)
    assert "log_safe" in src, (
        "CorrelationIdMiddleware no longer sanitises caller-controlled "
        "request_id before binding to the contextvar"
    )

    async def _scenario() -> str | None:
        sent: list[dict] = []

        async def _app(scope, receive, send) -> None:
            # Capture the request_id stored on the contextvar at the
            # point the wrapped app runs.
            sent.append({"context": current_request_id()})
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [],
                }
            )
            await send({"type": "http.response.body", "body": b""})

        mw = CorrelationIdMiddleware(_app)
        scope = {
            "type": "http",
            "headers": [
                (b"x-request-id", b"foo\ninjected"),
            ],
        }

        async def _receive() -> dict:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def _send(message: dict) -> None:
            pass

        await mw(scope, _receive, _send)
        return sent[0]["context"]

    rid = asyncio.run(_scenario())
    assert rid is not None
    assert "\n" not in rid
    assert rid == "foo\\ninjected"
