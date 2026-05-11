# SPDX-License-Identifier: Apache-2.0
"""``GET /v1/regime_index/latest`` acceptance tests (PR-3 task H)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import market_regime_engine.fixed_income  # noqa: F401  - register FI schema
from market_regime_engine.fixed_income.api import build_router
from market_regime_engine.fixed_income.credit_spread_regime import (
    score_credit_regime,
    write_credit_regime_score,
)
from market_regime_engine.storage import Warehouse

_ASOF = pd.Timestamp("2026-05-08T16:00:00+00:00")


def _row(date: pd.Timestamp, feature_name: str, value: float) -> dict:
    return {
        "date": date,
        "feature_name": feature_name,
        "value": float(value),
        "source_timestamp": date,
        "vintage_date": None,
    }


def _features(asof: pd.Timestamp, n: int = 30) -> pd.DataFrame:
    dates = pd.date_range(end=asof, periods=n, freq="D", tz="UTC")
    rows: list[dict] = []
    for ts in dates:
        rows.append(_row(ts, "ust_slope", 0.5))
        rows.append(_row(ts, "ust_curvature", 0.1))
        rows.append(_row(ts, "cdx_ig_5y", 65.0))
        rows.append(_row(ts, "cdx_hy_5y", 350.0))
        rows.append(_row(ts, "vix", 18.0))
        rows.append(_row(ts, "move", 100.0))
        rows.append(_row(ts, "etf_prem_disc", 0.10))
    return pd.DataFrame(rows)


def _app_with_warehouse(wh: Warehouse) -> FastAPI:
    """Build a FastAPI app with the FI router pointed at the given warehouse."""
    app = FastAPI()
    app.include_router(build_router(warehouse_factory=lambda: wh))
    return app


@pytest.fixture
def populated_warehouse(tmp_path: Path) -> Warehouse:
    wh = Warehouse(str(tmp_path / "fi-api.duckdb"))
    out = score_credit_regime(_features(_ASOF), asof=_ASOF, model_run_id="run-api-1")
    write_credit_regime_score(wh, out)
    yield wh
    wh.close()


@pytest.fixture
def empty_warehouse(tmp_path: Path) -> Warehouse:
    wh = Warehouse(str(tmp_path / "fi-api-empty.duckdb"))
    yield wh
    wh.close()


def test_get_regime_index_latest_returns_200_with_full_payload(
    populated_warehouse: Warehouse,
) -> None:
    client = TestClient(_app_with_warehouse(populated_warehouse))
    resp = client.get("/v1/regime_index/latest")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Required AGENT.md §6.1 fields.
    for key in (
        "timestamp",
        "regime_score",
        "regime_label",
        "confidence",
        "drivers",
        "component_scores",
        "model_run_id",
        "release_gate",
        "artifact_hash",
        "metadata",
    ):
        assert key in body, f"missing field {key!r} in response"
    assert body["timestamp"].endswith("Z")
    assert 0.0 <= body["regime_score"] <= 100.0
    assert body["release_gate"] is True
    assert body["artifact_hash"].startswith("sha256:")


def test_get_regime_index_latest_returns_503_when_no_data(empty_warehouse: Warehouse) -> None:
    client = TestClient(_app_with_warehouse(empty_warehouse))
    resp = client.get("/v1/regime_index/latest")
    assert resp.status_code == 503
    body = resp.json()
    assert body["detail"] == "no_data"
    assert body["release_gate"] is False


def test_get_regime_index_latest_returns_release_gate_false_passthrough(
    tmp_path: Path,
) -> None:
    wh = Warehouse(str(tmp_path / "fi-api-rg.duckdb"))
    try:
        out = score_credit_regime(
            _features(_ASOF),
            asof=_ASOF,
            model_run_id="run-api-rg",
            release_gate=False,
        )
        write_credit_regime_score(wh, out)
        client = TestClient(_app_with_warehouse(wh))
        resp = client.get("/v1/regime_index/latest")
        assert resp.status_code == 200
        body = resp.json()
        # Consumers must see the row + the gate so they can fail closed
        # downstream (AGENT.md non-negotiable 8).
        assert body["release_gate"] is False
        assert body["confidence"] <= 0.5
    finally:
        wh.close()


def test_get_regime_index_latest_response_includes_governance_fields(
    populated_warehouse: Warehouse,
) -> None:
    client = TestClient(_app_with_warehouse(populated_warehouse))
    body = client.get("/v1/regime_index/latest").json()
    assert body["model_run_id"]
    assert isinstance(body["release_gate"], bool)
    assert body["artifact_hash"].startswith("sha256:")


def test_fi_router_mounted_on_api_v1_app() -> None:
    """The PR-3 router is mounted on the shared ``api_v1.app`` at import time."""
    from market_regime_engine.api_v1 import app as v1_app

    paths = {route.path for route in v1_app.routes}
    assert "/v1/regime_index/latest" in paths
    # Stub routes still present for the PR-4..PR-7 endpoints.
    assert "/v1/liquidity_index/latest" in paths
    assert "/v1/execution_confidence" in paths


def test_all_fi_endpoints_live_in_pr7(tmp_path: Path) -> None:
    """After PR-7, every FI router endpoint is live.

    PR-3 made ``/v1/regime_index/latest`` live; PR-4 made
    ``/v1/liquidity_index/*`` live; PR-5 made
    ``POST /v1/execution_confidence`` live; PR-6 made
    ``/v1/tca/regime-segments/latest`` live; PR-7 makes
    ``GET /v1/evidence-pack/{model_run_id}`` live (returns 404 for an
    unknown run_id, 200 with the pack JSON otherwise).
    """
    db_path = tmp_path / "fi-pr7.duckdb"
    out = score_credit_regime(_features(_ASOF), asof=_ASOF, model_run_id="run-pr7-1")
    wh = Warehouse(str(db_path))
    try:
        write_credit_regime_score(wh, out)
    finally:
        wh.close()
    # PR-7: GET /v1/evidence-pack/{run_id} is live; unknown run_id → 404.
    wh1 = Warehouse(str(db_path))
    try:
        client = TestClient(_app_with_warehouse(wh1))
        resp = client.get("/v1/evidence-pack/run-x")
        assert resp.status_code == 404, resp.status_code
        assert resp.json()["detail"] == "evidence_pack_not_found"
    finally:
        # Handler may have closed the warehouse already; tolerate it.
        try:
            wh1.close()
        except Exception:
            pass
    # PR-6: /v1/tca/regime-segments/latest is live; empty warehouse → 503.
    wh2 = Warehouse(str(db_path))
    try:
        client = TestClient(_app_with_warehouse(wh2))
        resp = client.get("/v1/tca/regime-segments/latest")
        assert resp.status_code == 503, resp.status_code
        assert resp.json()["detail"] == "no_data"
    finally:
        try:
            wh2.close()
        except Exception:
            pass
    # PR-5: POST /v1/execution_confidence is live; an empty body should
    # 422 (Pydantic validation), not 501.
    resp = client.post("/v1/execution_confidence", json={})
    assert resp.status_code == 422
