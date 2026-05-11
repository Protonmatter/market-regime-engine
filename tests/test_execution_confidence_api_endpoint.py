# SPDX-License-Identifier: Apache-2.0
"""PR-5 §C.4: end-to-end POST /v1/execution_confidence handler.

Verifies the route is mounted, persistence works, the response carries
the governance triple, and the row lands in
``execution_confidence_predictions`` keyed by the inbound request_id.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

import market_regime_engine.fixed_income  # noqa: F401  registers FI schema
from market_regime_engine.fixed_income import (
    score_credit_regime,
    score_liquidity_stress,
    write_credit_regime_score,
    write_liquidity_stress_score,
)
from market_regime_engine.storage import (
    Warehouse,
    close_pooled_warehouses,
    get_pooled_warehouse,
)


def _seed(wh: Warehouse, ts: pd.Timestamp) -> None:
    rows = [
        {
            "date": ts - pd.Timedelta(days=i),
            "feature_name": "cdx_ig_5y",
            "value": float(i),
            "source_timestamp": ts - pd.Timedelta(days=i),
            "vintage_date": None,
        }
        for i in range(100, -1, -1)
    ]
    features = pd.DataFrame(rows)
    features.attrs["nan_policy"] = "NAN_TO_LAST_VALID"
    write_credit_regime_score(wh, score_credit_regime(features, asof=ts, release_gate=True))

    rows = [
        {
            "date": ts - pd.Timedelta(days=i),
            "feature_name": "bid_ask_width",
            "value": float(i),
            "source_timestamp": ts - pd.Timedelta(days=i),
            "vintage_date": None,
        }
        for i in range(100, -1, -1)
    ]
    features = pd.DataFrame(rows)
    features.attrs["nan_policy"] = "NAN_TO_LAST_VALID"
    write_liquidity_stress_score(
        wh,
        score_liquidity_stress(
            features,
            scope_type="cusip",
            scope_id="00206RGB6",
            asof=ts,
            release_gate=True,
        ),
    )


@pytest.fixture
def client(monkeypatch, tmp_path: Path):
    db = tmp_path / "endpoint.duckdb"
    monkeypatch.setenv("MRE_DB_PATH", str(db))
    monkeypatch.delenv("MRE_API_KEY", raising=False)
    close_pooled_warehouses()
    wh = Warehouse(db)
    _seed(wh, pd.Timestamp("2026-05-01T16:00:00Z"))
    wh.close()

    sys.modules.pop("market_regime_engine.api_v1", None)
    import market_regime_engine.api_v1 as api_v1

    importlib.reload(api_v1)
    from fastapi.testclient import TestClient

    with TestClient(api_v1.app) as testclient:
        yield testclient, db
    close_pooled_warehouses()


def _payload(request_id: str = "req-endpoint-1") -> dict:
    return {
        "timestamp": "2026-05-01T16:00:30Z",
        "cusip": "00206RGB6",
        "side": "buy",
        "notional": 1_000_000,
        "protocol": "Auto-X",
        "urgency": "normal",
        "request_id": request_id,
    }


def test_endpoint_returns_full_response_shape(client) -> None:
    testclient, _ = client
    resp = testclient.post("/v1/execution_confidence", json=_payload())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    expected_keys = {
        "timestamp",
        "cusip",
        "side",
        "notional",
        "protocol",
        "confidence_score",
        "expected_slippage_bps",
        "confidence_interval_low",
        "confidence_interval_high",
        "recommended_action",
        "human_review_required",
        "model_run_id",
        "release_gate",
        "artifact_hash",
        "metadata",
    }
    assert expected_keys.issubset(body)


def test_endpoint_persists_prediction_to_warehouse(client) -> None:
    testclient, db = client
    resp = testclient.post("/v1/execution_confidence", json=_payload("req-persist"))
    assert resp.status_code == 200, resp.text
    wh = get_pooled_warehouse(db)
    df = wh.read_execution_confidence_predictions()
    assert not df.empty
    assert "req-persist" in df["request_id"].astype(str).tolist()


def test_endpoint_returns_signal_age_seconds_in_metadata(client) -> None:
    testclient, _ = client
    resp = testclient.post("/v1/execution_confidence", json=_payload("req-age"))
    body = resp.json()
    assert "metadata" in body
    assert "signal_age_seconds_credit_regime" in body["metadata"]
    assert "signal_age_seconds_liquidity" in body["metadata"]
    assert "max_signal_age_seconds" in body["metadata"]


def test_endpoint_returns_stale_signal_payload_when_signals_are_old(client, monkeypatch) -> None:
    testclient, _ = client
    payload = _payload("req-stale")
    payload["timestamp"] = "2030-01-01T16:00:00Z"  # 4 years in the future
    resp = testclient.post("/v1/execution_confidence", json=payload)
    # 4 years > 15 min default threshold → soft fail to stale.
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["recommended_action"] == "Unavailable — stale signal"
    assert body["release_gate"] is False
