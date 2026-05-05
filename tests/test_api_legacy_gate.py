"""Regression tests for the v1.2.1 legacy-API gate.

Pre-v1.2.1, ``market_regime_engine.api:app`` could be mounted by uvicorn
without any authentication. The gate added in v1.2.1 raises ``RuntimeError``
at module import time unless ``MRE_LEGACY_API_ALLOW_UNAUTH=1`` is set.

The check is intentionally placed at import (not request) time so a
misconfigured deployment fails fast: uvicorn will refuse to start rather
than silently exposing the legacy surface.
"""

from __future__ import annotations

import importlib
import sys

import pytest

_API_MODULE = "market_regime_engine.api"


def _purge_api_module() -> None:
    """Remove the legacy ``api`` module from ``sys.modules`` so the next
    import re-runs the module-level gate."""
    sys.modules.pop(_API_MODULE, None)


def test_importing_legacy_api_without_env_var_raises(monkeypatch) -> None:
    monkeypatch.delenv("MRE_LEGACY_API_ALLOW_UNAUTH", raising=False)
    _purge_api_module()
    with pytest.raises(RuntimeError, match="legacy /api is unauthenticated"):
        importlib.import_module(_API_MODULE)
    # The aborted import must NOT leave a half-initialized module in
    # ``sys.modules`` that a later import could pick up by accident.
    assert _API_MODULE not in sys.modules


def test_importing_legacy_api_with_env_var_one_works(monkeypatch) -> None:
    monkeypatch.setenv("MRE_LEGACY_API_ALLOW_UNAUTH", "1")
    _purge_api_module()
    api_module = importlib.import_module(_API_MODULE)
    assert hasattr(api_module, "app")


def test_importing_legacy_api_with_env_var_other_value_raises(monkeypatch) -> None:
    """Only the literal string ``"1"`` opts in. A truthy value like
    ``"true"`` or ``"yes"`` must NOT bypass the gate — operators are
    notoriously bad at spelling truthy strings consistently across shells
    and we want exactly one accepted value."""
    monkeypatch.setenv("MRE_LEGACY_API_ALLOW_UNAUTH", "true")
    _purge_api_module()
    with pytest.raises(RuntimeError):
        importlib.import_module(_API_MODULE)


def test_legacy_api_health_returns_runtime_version(monkeypatch) -> None:
    monkeypatch.setenv("MRE_LEGACY_API_ALLOW_UNAUTH", "1")
    monkeypatch.setenv("MRE_DB_PATH", "data/test_legacy_health.db")
    _purge_api_module()
    api_module = importlib.import_module(_API_MODULE)

    from fastapi.testclient import TestClient

    from market_regime_engine import __version__ as runtime_version

    client = TestClient(api_module.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"] == runtime_version


def test_legacy_api_regime_route_works_when_gate_open(tmp_path, monkeypatch) -> None:
    """End-to-end: with the gate open, an actual data route should still
    function (returns 404 against an empty warehouse, not 500)."""
    monkeypatch.setenv("MRE_LEGACY_API_ALLOW_UNAUTH", "1")
    db_path = tmp_path / "legacy_smoke.db"
    monkeypatch.setenv("MRE_DB_PATH", str(db_path))
    _purge_api_module()
    api_module = importlib.import_module(_API_MODULE)

    from fastapi.testclient import TestClient

    client = TestClient(api_module.app)
    # Empty warehouse -> 404 from the explain layer. The fact that we get
    # a structured 404 rather than an import-time error proves the gate
    # opened cleanly.
    resp = client.get("/regime/latest")
    assert resp.status_code == 404, resp.text


def test_legacy_gate_message_is_instructive(monkeypatch) -> None:
    """The error message must point operators at the safe mount and the
    env-var override; bare ``RuntimeError`` would leave them flailing."""
    monkeypatch.setenv("MRE_LEGACY_API_ALLOW_UNAUTH", "1")
    _purge_api_module()
    api_module = importlib.import_module(_API_MODULE)

    msg = api_module._LEGACY_GATE_MESSAGE  # type: ignore[attr-defined]
    assert "api_v1" in msg
    assert "MRE_LEGACY_API_ALLOW_UNAUTH" in msg
