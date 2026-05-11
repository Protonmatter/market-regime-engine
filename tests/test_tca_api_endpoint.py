# SPDX-License-Identifier: Apache-2.0
"""PR-6 §G.3 — GET /v1/tca/regime-segments/latest end-to-end tests."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

import market_regime_engine.fixed_income  # noqa: F401 — register FI schema
from market_regime_engine.fixed_income.schemas import TcaRegimeSegment
from market_regime_engine.fixed_income.tca_segmentation import (
    write_tca_regime_segment,
)
from market_regime_engine.storage import (
    Warehouse,
    close_pooled_warehouses,
)


def _seed_segments(wh: Warehouse) -> list[TcaRegimeSegment]:
    base_ts = pd.Timestamp("2026-05-01T00:00:00Z")
    segs = [
        TcaRegimeSegment(
            timestamp=base_ts,
            regime_label="Normal Liquidity",
            liquidity_label="Normal",
            execution_confidence_bucket=None,
            protocol=None,
            side=None,
            sector=None,
            rating=None,
            maturity_bucket=None,
            notional_bucket=None,
            metric_name="arrival_cost_bps",
            metric_value=2.5,
            sample_count=10,
            model_run_id="rt-1",
            metadata_json=json.dumps({"dim": "regime_label,liquidity_label"}, sort_keys=True),
        ),
        TcaRegimeSegment(
            timestamp=base_ts,
            regime_label="Watch / Transition",
            liquidity_label="Mild Stress",
            execution_confidence_bucket=None,
            protocol=None,
            side=None,
            sector=None,
            rating=None,
            maturity_bucket=None,
            notional_bucket=None,
            metric_name="vwap_slippage_bps",
            metric_value=3.5,
            sample_count=12,
            model_run_id="rt-1",
            metadata_json=json.dumps({"dim": "regime_label,liquidity_label"}, sort_keys=True),
        ),
        TcaRegimeSegment(
            timestamp=base_ts,
            regime_label="Normal Liquidity",
            liquidity_label=None,
            execution_confidence_bucket=None,
            protocol="Auto-X",
            side=None,
            sector=None,
            rating=None,
            maturity_bucket=None,
            notional_bucket=None,
            metric_name="arrival_cost_bps",
            metric_value=1.5,
            sample_count=4,
            model_run_id="rt-1",
            metadata_json=json.dumps({"dim": "regime_label,protocol"}, sort_keys=True),
        ),
    ]
    for s in segs:
        write_tca_regime_segment(wh, s)
    return segs


@pytest.fixture
def client_seeded(tmp_path: Path, monkeypatch):
    db = tmp_path / "tca_api.duckdb"
    monkeypatch.setenv("MRE_DB_PATH", str(db))
    monkeypatch.delenv("MRE_API_KEY", raising=False)
    close_pooled_warehouses()
    wh = Warehouse(db)
    _seed_segments(wh)
    wh.close()

    sys.modules.pop("market_regime_engine.api_v1", None)
    import market_regime_engine.api_v1 as api_v1

    importlib.reload(api_v1)
    from fastapi.testclient import TestClient

    with TestClient(api_v1.app) as testclient:
        yield testclient, db
    close_pooled_warehouses()


@pytest.fixture
def client_empty(tmp_path: Path, monkeypatch):
    db = tmp_path / "tca_api_empty.duckdb"
    monkeypatch.setenv("MRE_DB_PATH", str(db))
    monkeypatch.delenv("MRE_API_KEY", raising=False)
    close_pooled_warehouses()
    # Pre-create the warehouse so the schema is registered even though
    # no rows are seeded.
    wh = Warehouse(db)
    wh.close()

    sys.modules.pop("market_regime_engine.api_v1", None)
    import market_regime_engine.api_v1 as api_v1

    importlib.reload(api_v1)
    from fastapi.testclient import TestClient

    with TestClient(api_v1.app) as testclient:
        yield testclient
    close_pooled_warehouses()


def test_get_tca_regime_segments_latest_returns_200(client_seeded) -> None:
    testclient, _ = client_seeded
    resp = testclient.get("/v1/tca/regime-segments/latest")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "segments" in body
    assert "count" in body
    assert body["count"] >= 1
    # Each segment carries the required fields.
    segment = body["segments"][0]
    for required in (
        "timestamp",
        "regime_label",
        "liquidity_label",
        "metric_name",
        "metric_value",
        "sample_count",
        "model_run_id",
    ):
        assert required in segment


def test_get_tca_regime_segments_latest_filters_by_dimensions(client_seeded) -> None:
    testclient, _ = client_seeded
    resp = testclient.get(
        "/v1/tca/regime-segments/latest?dimensions=regime_label,liquidity_label"
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Only the (regime_label, liquidity_label) rows qualify.
    for s in body["segments"]:
        assert s["regime_label"] is not None
        assert s["liquidity_label"] is not None


def test_get_tca_regime_segments_latest_503_when_no_data(client_empty) -> None:
    resp = client_empty.get("/v1/tca/regime-segments/latest")
    assert resp.status_code == 503, resp.text
    body = resp.json()
    assert body["detail"] == "no_data"


def test_get_tca_regime_segments_latest_rejects_invalid_dimension(
    client_seeded,
) -> None:
    testclient, _ = client_seeded
    resp = testclient.get(
        "/v1/tca/regime-segments/latest?dimensions=bogus_dim"
    )
    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert "bogus_dim" in str(body)


def test_get_tca_regime_segments_latest_respects_limit_query_param(
    client_seeded,
) -> None:
    testclient, _ = client_seeded
    resp = testclient.get("/v1/tca/regime-segments/latest?limit=1")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 1
