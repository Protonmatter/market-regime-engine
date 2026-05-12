# SPDX-License-Identifier: Apache-2.0
"""Regression — api_v1 cache hit/miss counters route through OTel.

Pre-fix (REVIEW.md Tier-2 C-AUTO-4): ``api_v1._read`` called
``metrics().incr("mre_api_cache_hits_total"/..._misses_total, ...)``
which writes only to the legacy in-process ``MetricsRegistry``
(``_GLOBAL``). When ``configure_otel(enabled=True)`` was active at
boot the OTel meter received nothing — so OTel-backed dashboards
showed zero cache activity in production.

Post-fix: routes through the module-level
:func:`market_regime_engine.observability.incr` which mirrors to BOTH
backends per ``observability.py:393-409``.
"""

from __future__ import annotations

from importlib import util as importlib_util
from pathlib import Path

import pytest

import market_regime_engine.fixed_income  # noqa: F401  - register FI tables
from market_regime_engine import observability
from market_regime_engine.storage import close_pooled_warehouses, get_pooled_warehouse


def _has_otel() -> bool:
    return importlib_util.find_spec("opentelemetry") is not None


@pytest.fixture
def reset_otel_state():
    yield
    observability.configure_otel(enabled=False)


def _legacy_counter_value(name: str, **labels: str) -> float:
    snap = observability.metrics().snapshot()
    if labels:
        label_part = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        key = f"{name}{{{label_part}}}"
    else:
        key = name
    return float(snap["counters"].get(key, 0.0))


@pytest.fixture
def api_v1_module(monkeypatch, tmp_path: Path):
    """Reload ``api_v1`` so it picks up a temp ``MRE_DB_PATH`` for each test."""
    import pandas as pd

    db = tmp_path / "cache_counter.duckdb"
    monkeypatch.setenv("MRE_DB_PATH", str(db))
    monkeypatch.delenv("MRE_API_KEY", raising=False)
    close_pooled_warehouses()
    # Seed a release-gate row so the endpoint returns 200 + the value
    # actually populates the per-process cache (the handler raises 404
    # on empty tables and the cache never fills, which would defeat the
    # hit-counter test).
    wh = get_pooled_warehouse(db)
    wh.write_release_gates(
        pd.DataFrame(
            [
                {
                    "date": "2026-05-08",
                    "approved": True,
                    "decision": "release",
                    "confidence": 0.8,
                    "confidence_grade": "high",
                    "severe_drift": 0,
                    "major_drift": 0,
                    "max_psi": 0.05,
                    "high_invalidation_triggers": 0,
                    "active_trigger_names": "[]",
                    "reasons": "ok",
                    "metadata_json": "{}",
                    "resolved_profile": "production",
                }
            ]
        )
    )

    import importlib

    import market_regime_engine.api_v1 as api_v1

    importlib.reload(api_v1)
    yield api_v1
    close_pooled_warehouses()


def test_api_v1_cache_hits_increment_via_legacy_global(api_v1_module) -> None:
    """The legacy ``MetricsRegistry`` snapshot must still reflect cache
    hits — post-fix routing through ``observability.incr`` MUST NOT
    silently break the legacy backend."""
    from fastapi.testclient import TestClient

    api_v1 = api_v1_module
    client = TestClient(api_v1.app)
    # First request → cache miss.
    before_miss = _legacy_counter_value("mre_api_cache_misses_total", endpoint="release_gate_latest")
    resp = client.get("/v1/release-gate/latest")
    assert resp.status_code in (200, 404, 503)
    after_miss = _legacy_counter_value("mre_api_cache_misses_total", endpoint="release_gate_latest")
    assert after_miss >= before_miss + 1.0

    # Second request → cache hit.
    before_hit = _legacy_counter_value("mre_api_cache_hits_total", endpoint="release_gate_latest")
    resp = client.get("/v1/release-gate/latest")
    assert resp.status_code in (200, 404, 503)
    after_hit = _legacy_counter_value("mre_api_cache_hits_total", endpoint="release_gate_latest")
    assert after_hit >= before_hit + 1.0


def test_api_v1_cache_counters_increment_via_otel_when_configured(reset_otel_state, api_v1_module) -> None:
    """When ``configure_otel(enabled=True)`` is active, the cache
    counter call sites must register OTel counter instruments so the
    OTel exporter pipeline sees the increments."""
    if not _has_otel():
        pytest.skip("OpenTelemetry SDK not installed")
    assert observability.configure_otel(enabled=True) is True
    api_v1 = api_v1_module

    from fastapi.testclient import TestClient

    client = TestClient(api_v1.app)
    client.get("/v1/release-gate/latest")
    # Second request → cache hit.
    client.get("/v1/release-gate/latest")

    miss_counter = observability._otel_counter(  # type: ignore[attr-defined]
        "mre_api_cache_misses_total"
    )
    hit_counter = observability._otel_counter(  # type: ignore[attr-defined]
        "mre_api_cache_hits_total"
    )
    assert miss_counter is not None, (
        "OTel counter for mre_api_cache_misses_total must be created when "
        "the call site routes through observability.incr"
    )
    assert hit_counter is not None, (
        "OTel counter for mre_api_cache_hits_total must be created when the call site routes through observability.incr"
    )
