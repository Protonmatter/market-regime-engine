# SPDX-License-Identifier: Apache-2.0
"""PR-7 §L (REVIEW.md §3.6 PR-8) — versioned cache key acceptance tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import market_regime_engine.fixed_income  # noqa: F401  - register FI tables
from market_regime_engine.fixed_income.api import build_router, reset_fi_cache
from market_regime_engine.storage import Warehouse


def _row(ts_iso: str, score: float = 47.0) -> dict:
    return {
        "model_run_id": f"run-{ts_iso}",
        "timestamp": ts_iso,
        "regime_score": score,
        "regime_label": "Watch / Transition",
        "confidence": 0.85,
        "drivers_json": "[]",
        "component_scores_json": "{}",
        "release_gate": 1,
        "artifact_hash": "sha256:" + "a" * 64,
        "metadata_json": "{}",
    }


def _build_app_fresh(db_path: str) -> FastAPI:
    """App with a factory that opens a fresh Warehouse per request.

    The FI router closes the warehouse in its finally block (mirror of
    the production pooled-warehouse contract), so a multi-request test
    must open a fresh handle each time. In production
    ``get_pooled_warehouse`` returns the singleton that survives close.
    """
    app = FastAPI()

    def _factory() -> Warehouse:
        return Warehouse(db_path)

    app.include_router(build_router(_factory))
    return app


@pytest.fixture(autouse=True)
def _clean_cache():
    reset_fi_cache()
    yield
    reset_fi_cache()


def test_cache_hit_when_latest_score_unchanged(tmp_path: Path) -> None:
    """Two calls in a row with the same latest_ts go through the cache."""
    db_path = tmp_path / "vc-1.duckdb"
    wh = Warehouse(str(db_path))
    try:
        wh.write_credit_regime_score(pd.DataFrame([_row("2026-05-08T16:00:00Z")]))
    finally:
        wh.close()

    from market_regime_engine.fixed_income import api as fi_api

    with patch.object(
        fi_api,
        "latest_credit_regime_score",
        wraps=fi_api.latest_credit_regime_score,
    ) as spy:
        client = TestClient(_build_app_fresh(str(db_path)))
        r1 = client.get("/v1/regime_index/latest")
        r2 = client.get("/v1/regime_index/latest")
        r3 = client.get("/v1/regime_index/latest")
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r3.status_code == 200
        # First request computes; second + third hit the cache.
        assert spy.call_count == 1


def test_cache_invalidates_when_latest_regime_score_advances(tmp_path: Path) -> None:
    """A newer row in credit_regime_scores must invalidate the cache."""
    db_path = tmp_path / "vc-2.duckdb"
    wh = Warehouse(str(db_path))
    try:
        wh.write_credit_regime_score(pd.DataFrame([_row("2026-05-08T16:00:00Z", 47.0)]))
    finally:
        wh.close()

    from market_regime_engine.fixed_income import api as fi_api

    client = TestClient(_build_app_fresh(str(db_path)))
    with patch.object(
        fi_api,
        "latest_credit_regime_score",
        wraps=fi_api.latest_credit_regime_score,
    ) as spy:
        r1 = client.get("/v1/regime_index/latest")
        assert r1.status_code == 200
        r2 = client.get("/v1/regime_index/latest")
        assert r2.status_code == 200
        # Cache hit: only one compute call across both requests.
        assert spy.call_count == 1

    wh_writer = Warehouse(str(db_path))
    try:
        wh_writer.write_credit_regime_score(
            pd.DataFrame([_row("2026-05-08T17:00:00Z", 60.0)])
        )
    finally:
        wh_writer.close()

    with patch.object(
        fi_api,
        "latest_credit_regime_score",
        wraps=fi_api.latest_credit_regime_score,
    ) as spy2:
        r3 = client.get("/v1/regime_index/latest")
        assert r3.status_code == 200
        # Cache invalidated by the newer timestamp → compute fires once.
        assert spy2.call_count == 1
        # And the returned payload reflects the new score.
        body = r3.json()
        assert body["regime_score"] == pytest.approx(60.0)
