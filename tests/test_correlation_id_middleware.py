# SPDX-License-Identifier: Apache-2.0
"""PR-7 §I — correlation-ID middleware acceptance tests."""

from __future__ import annotations

import logging
import re
import uuid

from fastapi import FastAPI
from fastapi.testclient import TestClient

from market_regime_engine.fixed_income.correlation import (
    CorrelationIdLogFilter,
    CorrelationIdMiddleware,
    current_request_id,
    install_correlation_id_log_filter,
    set_request_id,
)


_UUID4_REGEX = re.compile(r"^[0-9a-f]{32}$")


def _build_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(CorrelationIdMiddleware)

    @app.get("/echo")
    def echo() -> dict[str, str | None]:
        return {"request_id": current_request_id()}

    return app


def test_middleware_generates_uuid_when_absent() -> None:
    client = TestClient(_build_app())
    resp = client.get("/echo")
    assert resp.status_code == 200
    body = resp.json()
    assert body["request_id"] is not None
    assert _UUID4_REGEX.match(body["request_id"]) is not None


def test_middleware_propagates_x_request_id() -> None:
    client = TestClient(_build_app())
    custom_id = "test-corr-id-abc-123"
    resp = client.get("/echo", headers={"X-Request-ID": custom_id})
    assert resp.status_code == 200
    assert resp.json()["request_id"] == custom_id


def test_response_includes_x_request_id_header() -> None:
    client = TestClient(_build_app())
    custom_id = "test-resp-id-456"
    resp = client.get("/echo", headers={"X-Request-ID": custom_id})
    assert resp.headers.get("X-Request-ID") == custom_id


def test_response_includes_generated_uuid_header_when_absent() -> None:
    client = TestClient(_build_app())
    resp = client.get("/echo")
    sent = resp.headers.get("X-Request-ID") or resp.headers.get("x-request-id")
    assert sent is not None
    assert _UUID4_REGEX.match(sent) is not None


def test_log_lines_include_request_id_when_set(caplog) -> None:
    """The log filter populates ``record.request_id`` with the
    current context's correlation id."""
    install_correlation_id_log_filter()
    log = logging.getLogger(f"test_corr_{uuid.uuid4().hex}")
    set_request_id("explicit-test-id")
    try:
        with caplog.at_level(logging.INFO, logger=log.name):
            log.info("hello")
        # Find the record we just emitted.
        for record in caplog.records:
            if record.name == log.name:
                assert getattr(record, "request_id", "") == "explicit-test-id"
                return
        raise AssertionError("expected at least one matching log record")
    finally:
        set_request_id(None)


def test_install_correlation_id_log_filter_is_idempotent() -> None:
    f1 = install_correlation_id_log_filter()
    f2 = install_correlation_id_log_filter()
    assert f1 is f2
    assert isinstance(f1, CorrelationIdLogFilter)


def test_set_request_id_is_context_local() -> None:
    set_request_id("ctx-1")
    assert current_request_id() == "ctx-1"
    set_request_id(None)
    assert current_request_id() is None
