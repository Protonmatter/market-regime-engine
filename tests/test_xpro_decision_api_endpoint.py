# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

from tests.test_protocol_recommendation import _seed

from market_regime_engine.storage import Warehouse, close_pooled_warehouses


@pytest.fixture
def client(monkeypatch, tmp_path: Path):
    db = tmp_path / "endpoint.duckdb"
    monkeypatch.setenv("MRE_DB_PATH", str(db))
    monkeypatch.setenv("MRE_FI_HMAC_KEY_VERSIONS", '{"v1":"secret"}')
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
        yield testclient
    close_pooled_warehouses()


def _payload(request_id: str = "req-xpro-api") -> dict:
    return {
        "timestamp": "2026-05-01T16:00:30Z",
        "cusip": "00206RGB6",
        "side": "buy",
        "notional": 1_000_000,
        "protocol": "Auto-X",
        "urgency": "normal",
        "request_id": request_id,
        "candidate_protocols": ["Auto-X", "RFQ", "Manual"],
    }


def test_xpro_decision_endpoint_returns_signed_artifact(client) -> None:
    resp = client.post("/v1/xpro/decision", json=_payload())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["artifact_version"] == "xpro_decision_artifact_v1"
    assert body["decision"]["recommended_protocol"] in {"Auto-X", "RFQ", "Manual"}
    assert "hmac" in body["evidence"]
    get_resp = client.get(f"/v1/xpro/decision/{body['decision_id']}")
    assert get_resp.status_code == 200
    assert get_resp.json()["decision_id"] == body["decision_id"]


def test_xpro_decision_verify_endpoint_detects_tamper(client) -> None:
    body = client.post("/v1/xpro/decision", json=_payload("req-xpro-verify")).json()
    assert client.post("/v1/xpro/decision/verify", json=body).json()["verified"] is True
    body["decision"]["recommended_protocol"] = "Manual"
    assert client.post("/v1/xpro/decision/verify", json=body).json()["verified"] is False
