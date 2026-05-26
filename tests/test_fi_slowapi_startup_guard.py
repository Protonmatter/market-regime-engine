# SPDX-License-Identifier: Apache-2.0
"""v1.5.1 (PR-9 FIX 1): startup guard for the slowapi rate limiter.

When ``MRE_FI_RATE_LIMIT_ENABLED=1`` is set the FI router MUST raise a
``RuntimeError`` at module-load time if ``import slowapi`` fails.
Production deployments that forget the ``[security]`` extra must not
silently mount an unlimited handler.
"""

from __future__ import annotations

import sys

import pytest


def test_assert_slowapi_available_noop_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset env var => no-op even if slowapi is missing."""

    from market_regime_engine.fixed_income import api as fi_api

    monkeypatch.delenv("MRE_FI_RATE_LIMIT_ENABLED", raising=False)
    # Even when slowapi is forcibly absent the helper returns cleanly
    # because the env var is unset.
    monkeypatch.setitem(sys.modules, "slowapi", None)
    fi_api.assert_slowapi_available()


def test_assert_slowapi_available_raises_when_enabled_and_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``MRE_FI_RATE_LIMIT_ENABLED=1`` + missing slowapi => RuntimeError."""

    from market_regime_engine.fixed_income import api as fi_api

    monkeypatch.setenv("MRE_FI_RATE_LIMIT_ENABLED", "1")
    # Simulate slowapi missing by inserting ``None`` into ``sys.modules``
    # so ``importlib.import_module`` cannot find it.
    monkeypatch.setitem(sys.modules, "slowapi", None)
    with pytest.raises(RuntimeError, match="slowapi required when MRE_FI_RATE_LIMIT_ENABLED=1"):
        fi_api.assert_slowapi_available()


def test_assert_slowapi_available_succeeds_when_enabled_and_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """env=1 + slowapi installed => no error."""

    from market_regime_engine.fixed_income import api as fi_api

    monkeypatch.setenv("MRE_FI_RATE_LIMIT_ENABLED", "1")
    # Restore the real slowapi module so the importlib path succeeds.
    if "slowapi" in sys.modules and sys.modules["slowapi"] is None:
        sys.modules.pop("slowapi")
    pytest.importorskip("slowapi")
    fi_api.assert_slowapi_available()


def test_rate_limit_enabled_accepts_truthy_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """``rate_limit_enabled`` accepts ``1``, ``true``, ``yes`` (case-insensitive)."""

    from market_regime_engine.fixed_income import api as fi_api

    for value in ("1", "true", "TRUE", "yes", "On"):
        monkeypatch.setenv("MRE_FI_RATE_LIMIT_ENABLED", value)
        assert fi_api.rate_limit_enabled()

    for value in ("", "0", "false", "no", "off", "random-text"):
        monkeypatch.setenv("MRE_FI_RATE_LIMIT_ENABLED", value)
        assert not fi_api.rate_limit_enabled()


def test_burst_beyond_limit_returns_429_when_slowapi_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Integration test: with the limiter wired, burst > limit => 429.

    Builds a router-only FastAPI app with a 2/second limit and confirms
    the 3rd request inside the same second gets 429. Uses a temp warehouse
    so the score path returns 503 ``no_data`` rather than blowing up on the
    DuckDB read; we only care about the rate-limit response semantics.
    """

    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient

    from market_regime_engine.fixed_income.api import build_router
    from market_regime_engine.storage import Warehouse

    monkeypatch.setenv("MRE_FI_EXEC_CONF_RATE_LIMIT", "2/second")
    pytest.importorskip("slowapi")
    from slowapi import Limiter
    from slowapi.errors import RateLimitExceeded

    db = tmp_path / "rate.duckdb"
    wh = Warehouse(path=str(db))

    def _factory():
        return wh

    def _key_func(request) -> str:  # slowapi inspects the parameter name
        return "test-key"

    limiter = Limiter(key_func=_key_func, default_limits=["2/second"])
    app = FastAPI()
    app.state.limiter = limiter

    async def _rate_handler(request, exc):
        return JSONResponse(
            {"detail": f"rate limit exceeded: {exc.detail}"},
            status_code=429,
            headers={"Retry-After": "1"},
        )

    app.add_exception_handler(RateLimitExceeded, _rate_handler)
    router = build_router(warehouse_factory=_factory, limiter=limiter)
    app.include_router(router)

    client = TestClient(app, raise_server_exceptions=False)
    payload = {
        "timestamp": "2026-01-01T00:00:00Z",
        "cusip": "037833100",
        "side": "buy",
        "notional": 1000.0,
        "protocol": "RFQ",
        "urgency": "normal",
        "request_id": "req-001",
    }
    statuses = []
    for _ in range(5):
        r = client.post("/v1/execution_confidence", json=payload)
        statuses.append(r.status_code)
    # We don't care whether the score path itself succeeds (an empty
    # warehouse routes through fail-closed branches); we only require
    # that the slowapi limiter fires for at least one burst response.
    assert 429 in statuses, f"expected 429 in {statuses!r}"
