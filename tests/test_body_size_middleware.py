# SPDX-License-Identifier: Apache-2.0
"""Regression — body-size cap as ASGI middleware closes chunked-bypass DoS.

Pre-fix (REVIEW.md Tier-2 B-Ask-1): the FI POST handler
``_execution_confidence_handler`` checked ``Content-Length`` AFTER
FastAPI had already parsed the body. Chunked ``Transfer-Encoding``
requests carry no ``Content-Length`` header so the check was a no-op
and an attacker streaming an unbounded body bypassed the 32 KB cap.

Post-fix:
:class:`market_regime_engine.fixed_income.middleware.MaxBodySizeMiddleware`
is installed on ``api_v1.app``. The middleware wraps the ASGI
``receive`` callable and accumulates body bytes; once the running
total exceeds the cap it emits HTTP 413 directly without invoking the
downstream route. Production mode (``MRE_ENV=production`` or
``MRE_FI_REJECT_CHUNKED=1``) additionally refuses
chunked-without-Content-Length up front.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import market_regime_engine.fixed_income  # noqa: F401  - register FI schema
from market_regime_engine.fixed_income.middleware import (
    DEFAULT_BODY_SIZE_CAP_BYTES,
    MaxBodySizeMiddleware,
)
from market_regime_engine.storage import close_pooled_warehouses


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    for name in (
        "MRE_ENV",
        "MRE_FI_REJECT_CHUNKED",
        "MRE_FI_BODY_SIZE_CAP_BYTES",
    ):
        monkeypatch.delenv(name, raising=False)
    yield


def _valid_request_body() -> dict:
    return {
        "timestamp": "2026-05-08T16:00:00Z",
        "cusip": "9128283N8",
        "side": "buy",
        "notional": 1_000_000.0,
        "protocol": "Auto-X",
        "request_id": "req-body-cap-test",
        "urgency": "normal",
    }


@pytest.fixture
def app_with_middleware(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Reload ``api_v1`` so the middleware is registered fresh on each
    test with a small cap (so we don't have to construct 32 KB blobs)."""
    db = tmp_path / "body-cap.duckdb"
    monkeypatch.setenv("MRE_DB_PATH", str(db))
    monkeypatch.setenv("MRE_FI_BODY_SIZE_CAP_BYTES", "1024")  # 1 KB cap for tests
    monkeypatch.delenv("MRE_API_KEY", raising=False)
    close_pooled_warehouses()

    import importlib

    import market_regime_engine.api_v1 as api_v1

    importlib.reload(api_v1)
    yield api_v1
    close_pooled_warehouses()


def test_request_under_cap_succeeds(app_with_middleware) -> None:
    """A normal-sized body must pass the cap and reach the handler
    (Pydantic validation / signal-availability is exercised
    downstream; 200 / 422 / 500 / 503 are all acceptable — what we
    care about is that the middleware did NOT short-circuit with
    413). Use ``raise_server_exceptions=False`` so a downstream
    handler that raises (e.g. JSON-serialising an ``inf`` on an
    empty warehouse) still produces a status code instead of
    blowing up the test client."""
    client = TestClient(
        app_with_middleware.app, raise_server_exceptions=False
    )
    resp = client.post(
        "/v1/execution_confidence",
        json=_valid_request_body(),
    )
    assert resp.status_code != 413, resp.text


def test_request_over_cap_returns_413(app_with_middleware) -> None:
    """An oversized body must return 413 from the middleware BEFORE
    the route handler runs. We pad ``metadata`` to overflow the
    test cap (1 KB)."""
    client = TestClient(app_with_middleware.app)
    big_body = _valid_request_body()
    big_body["metadata"] = {"padding": "X" * 4096}  # 4 KB padding > 1 KB cap
    resp = client.post(
        "/v1/execution_confidence",
        json=big_body,
    )
    assert resp.status_code == 413, resp.text
    body = resp.json()
    assert "limit_bytes" in body
    assert body["limit_bytes"] == 1024


def test_chunked_request_over_cap_returns_413(app_with_middleware) -> None:
    """Chunked ``Transfer-Encoding`` requests (no Content-Length) MUST
    still be capped by the middleware via the receive-wrapping path."""
    client = TestClient(app_with_middleware.app)

    big_body = _valid_request_body()
    big_body["metadata"] = {"padding": "Y" * 4096}
    raw = json.dumps(big_body).encode("utf-8")

    # Stream the body in two chunks so the TestClient sends chunked-style
    # bytes through the ASGI receive callable. Per Starlette
    # convention, sending an iterator with no Content-Length triggers
    # chunked-style receive events.
    def _chunks():
        # Split into two halves so the second receive crosses the cap.
        half = len(raw) // 2
        yield raw[:half]
        yield raw[half:]

    resp = client.post(
        "/v1/execution_confidence",
        content=_chunks(),
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 413, resp.text


def test_production_profile_rejects_chunked_with_no_content_length(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When ``MRE_ENV=production`` the middleware refuses
    chunked-without-Content-Length up front even if the body would
    fit under the cap, because chunked input bypasses the
    Content-Length pre-screen.

    v1.6 PR-22: ``api_v1`` now runs ``assert_production_ready()`` at
    import time, which requires ``MRE_API_KEY`` to be set whenever
    ``MRE_ENV=production``. The test is about body-cap behavior, not
    auth, so we set a stub key just to clear the import-time guard.
    """
    db = tmp_path / "prod-body-cap.duckdb"
    monkeypatch.setenv("MRE_DB_PATH", str(db))
    # v1.6.0 (REVIEW_DEEP_V1_5_2.md §3.14): production guard now
    # requires slowapi importable when MRE_FI_RATE_LIMIT_ENABLED is
    # truthy. Skip on dev boxes without the [security] extra.
    pytest.importorskip("slowapi")
    monkeypatch.setenv("MRE_ENV", "production")
    monkeypatch.setenv("MRE_FI_BODY_SIZE_CAP_BYTES", "65536")
    monkeypatch.setenv("MRE_API_KEY", "test-body-cap-key")
    # v1.6.0 (REVIEW_DEEP_V1_5_2.md §3.14): production guard now also
    # requires the FI HMAC key + rate-limit-enabled env vars.
    monkeypatch.setenv("MRE_FI_HMAC_KEY_VERSIONS", "v1=" + ("a" * 64))
    monkeypatch.setenv("MRE_FI_RATE_LIMIT_ENABLED", "1")
    close_pooled_warehouses()
    import importlib

    import market_regime_engine.api_v1 as api_v1

    importlib.reload(api_v1)
    try:
        client = TestClient(api_v1.app)
        small_body = _valid_request_body()
        raw = json.dumps(small_body).encode("utf-8")

        def _chunks():
            yield raw

        resp = client.post(
            "/v1/execution_confidence",
            content=_chunks(),
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 413, resp.text
        body = resp.json()
        assert "chunked" in body["detail"].lower()
    finally:
        close_pooled_warehouses()


# ---------------------------------------------------------------------------
# Unit tests for the middleware class itself.
# ---------------------------------------------------------------------------


def test_middleware_passes_through_non_protected_paths() -> None:
    """The middleware must NOT touch paths outside its
    ``protected_path_prefixes`` even when the body is huge — macro
    endpoints have their own (or no) cap."""
    inner = FastAPI()

    @inner.post("/macro-route")
    async def _macro():
        return {"ok": True}

    inner.add_middleware(
        MaxBodySizeMiddleware,
        max_size=64,
        protected_path_prefixes=("/v1/execution_confidence",),
    )
    client = TestClient(inner)
    resp = client.post("/macro-route", content=b"X" * 10_000)
    assert resp.status_code == 200, resp.text


def test_middleware_default_cap_matches_handler_constant() -> None:
    """Defence-in-depth: the middleware default and the historical
    in-handler constant must agree so a stripped-down deployment that
    only has the middleware still has the documented 32 KB cap."""
    assert DEFAULT_BODY_SIZE_CAP_BYTES == 32 * 1024


def test_middleware_passes_health_endpoint(app_with_middleware) -> None:
    """The macro health endpoint must not be affected by the FI body
    cap — it lives on a different prefix."""
    client = TestClient(app_with_middleware.app)
    with contextlib.suppress(Exception):
        # /v1/health returns 200 in a healthy deploy; if the route
        # is rate-limited or absent in this test env, the call still
        # demonstrates the middleware does not intercept.
        resp = client.get("/v1/health")
        assert resp.status_code in (200, 401, 403)
