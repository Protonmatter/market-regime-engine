# SPDX-License-Identifier: Apache-2.0
"""``GET /v1/liquidity_index/*`` API acceptance tests (PR-4 task G / H.7)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import market_regime_engine.fixed_income  # noqa: F401  - register FI schema
from market_regime_engine.fixed_income.api import build_router
from market_regime_engine.fixed_income.liquidity_stress import (
    score_liquidity_stress,
    write_liquidity_stress_score,
)
from market_regime_engine.frontier.data_cleaning import NanPolicy
from market_regime_engine.storage import Warehouse

_ASOF = pd.Timestamp("2026-05-08T16:00:00+00:00")


def _features(asof: pd.Timestamp = _ASOF, n: int = 20) -> pd.DataFrame:
    dates = pd.date_range(end=asof, periods=n, freq="D", tz="UTC")
    rows = []
    for i, ts in enumerate(dates):
        rows.append({"date": ts, "feature_name": "bid_ask_width", "value": 0.5 + 0.01 * i, "source_timestamp": ts, "vintage_date": None})
        rows.append({"date": ts, "feature_name": "quote_dispersion", "value": 0.05 + 0.001 * i, "source_timestamp": ts, "vintage_date": None})
        rows.append({"date": ts, "feature_name": "trade_count_velocity", "value": 5.0 + 0.1 * i, "source_timestamp": ts, "vintage_date": None})
        rows.append({"date": ts, "feature_name": "dealers_requested", "value": 5.0, "source_timestamp": ts, "vintage_date": None})
        rows.append({"date": ts, "feature_name": "quotes_received", "value": 3.0, "source_timestamp": ts, "vintage_date": None})
        rows.append({"date": ts, "feature_name": "amihud_illiquidity", "value": 1e-9 + 1e-12 * i, "source_timestamp": ts, "vintage_date": None})
        rows.append({"date": ts, "feature_name": "time_since_last_trade", "value": 5.0, "source_timestamp": ts, "vintage_date": None})
    frame = pd.DataFrame(rows)
    frame.attrs["nan_policy"] = NanPolicy.NAN_FAILS_PIT_AUDIT.value
    return frame


def _app_with_warehouse(wh: Warehouse) -> FastAPI:
    app = FastAPI()
    app.include_router(build_router(warehouse_factory=lambda: wh))
    return app


@pytest.fixture
def populated_warehouse(tmp_path: Path) -> Warehouse:
    wh = Warehouse(str(tmp_path / "fi-liq-api.duckdb"))
    market_out = score_liquidity_stress(
        _features(),
        scope_type="market",
        scope_id="ALL",
        asof=_ASOF,
        model_run_id="api-market-1",
    )
    cusip_out = score_liquidity_stress(
        _features(),
        scope_type="cusip",
        scope_id="9128283N8",
        asof=_ASOF,
        model_run_id="api-cusip-1",
    )
    write_liquidity_stress_score(wh, market_out)
    write_liquidity_stress_score(wh, cusip_out)
    yield wh
    wh.close()


@pytest.fixture
def empty_warehouse(tmp_path: Path) -> Warehouse:
    wh = Warehouse(str(tmp_path / "fi-liq-api-empty.duckdb"))
    yield wh
    wh.close()


# ---------------------------------------------------------------------------
# /v1/liquidity_index/latest
# ---------------------------------------------------------------------------


def test_get_liquidity_index_latest(populated_warehouse: Warehouse) -> None:
    client = TestClient(_app_with_warehouse(populated_warehouse))
    resp = client.get("/v1/liquidity_index/latest")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    for key in (
        "timestamp",
        "scope_type",
        "scope_id",
        "liquidity_index",
        "liquidity_label",
        "confidence",
        "drivers",
        "model_run_id",
        "release_gate",
        "artifact_hash",
        "metadata",
    ):
        assert key in body, f"missing {key!r} in response"
    assert body["timestamp"].endswith("Z")
    assert 0.0 <= body["liquidity_index"] <= 100.0
    assert body["artifact_hash"].startswith("sha256:")


def test_get_liquidity_index_returns_503_when_no_data(empty_warehouse: Warehouse) -> None:
    client = TestClient(_app_with_warehouse(empty_warehouse))
    resp = client.get("/v1/liquidity_index/latest")
    assert resp.status_code == 503
    body = resp.json()
    assert body["detail"] == "no_data"
    assert body["release_gate"] is False


# ---------------------------------------------------------------------------
# /v1/liquidity_index/{scope_type}/{scope_id}
# ---------------------------------------------------------------------------


def test_get_liquidity_index_by_scope_market(populated_warehouse: Warehouse) -> None:
    client = TestClient(_app_with_warehouse(populated_warehouse))
    resp = client.get("/v1/liquidity_index/market/ALL")
    assert resp.status_code == 200
    body = resp.json()
    assert body["scope_type"] == "market"
    assert body["scope_id"] == "ALL"


def test_get_liquidity_index_by_scope_cusip(populated_warehouse: Warehouse) -> None:
    client = TestClient(_app_with_warehouse(populated_warehouse))
    resp = client.get("/v1/liquidity_index/cusip/9128283N8")
    assert resp.status_code == 200
    body = resp.json()
    assert body["scope_type"] == "cusip"
    assert body["scope_id"] == "9128283N8"


def test_get_liquidity_index_by_scope_invalid_type_returns_404(
    populated_warehouse: Warehouse,
) -> None:
    client = TestClient(_app_with_warehouse(populated_warehouse))
    resp = client.get("/v1/liquidity_index/garbage/ALL")
    assert resp.status_code == 404
    body = resp.json()
    assert body["detail"] == "invalid_scope_type"
    assert set(body["valid_scope_types"]) == {"market", "sector", "rating", "cusip"}


def test_get_liquidity_index_by_scope_503_when_scope_missing(
    populated_warehouse: Warehouse,
) -> None:
    client = TestClient(_app_with_warehouse(populated_warehouse))
    resp = client.get("/v1/liquidity_index/sector/UNKNOWN_SECTOR")
    assert resp.status_code == 503
    body = resp.json()
    assert body["detail"] == "no_data"
    assert body["release_gate"] is False


def test_get_liquidity_index_release_gate_false_passthrough(tmp_path: Path) -> None:
    """A row with release_gate=False is still returned so consumers can fail closed."""
    wh = Warehouse(str(tmp_path / "fi-liq-api-rg.duckdb"))
    try:
        out = score_liquidity_stress(
            _features(),
            scope_type="market",
            scope_id="ALL",
            asof=_ASOF,
            model_run_id="api-rg",
            release_gate=False,
        )
        write_liquidity_stress_score(wh, out)
        client = TestClient(_app_with_warehouse(wh))
        resp = client.get("/v1/liquidity_index/latest")
        assert resp.status_code == 200
        body = resp.json()
        assert body["release_gate"] is False
        assert body["confidence"] <= 0.5
    finally:
        wh.close()
