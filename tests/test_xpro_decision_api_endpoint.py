# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from market_regime_engine.fixed_income.api_handlers import build_router
from market_regime_engine.storage import Warehouse, close_pooled_warehouses
from tests.test_protocol_recommendation import _seed


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


def _minimal_artifact(decision_id: str = "dec-xpro-api") -> dict:
    return {
        "artifact_version": "xpro_decision_artifact_v1",
        "decision_id": decision_id,
        "request_id": "req-xpro-api",
        "asof_epoch_ns": "1777636830000000000",
        "numeric_policy": {"prob_scale": 1000000, "bps_scale": 10000},
        "input": {"cusip": "00206RGB6"},
        "model_outputs": {},
        "decision": {"recommended_protocol": "RFQ", "release_gate": True},
        "lineage": {},
        "evidence": {"artifact_hash": "sha256:test"},
    }


class _CloseTrackingWarehouse:
    def __init__(
        self,
        *,
        latest_payload: dict | None = None,
        read_error: Exception | None = None,
    ) -> None:
        self.closed = False
        self.writes: list[dict] = []
        self.latest_payload = latest_payload
        self.read_error = read_error

    def write_xpro_decision_artifact(self, artifact: dict) -> int:
        self.writes.append(artifact)
        return 1

    def latest_xpro_decision_artifact(self, decision_id: str) -> pd.DataFrame | None:
        if self.read_error is not None:
            raise self.read_error
        if self.latest_payload is None:
            return None
        return pd.DataFrame([{"decision_id": decision_id, "payload_json": json.dumps(self.latest_payload)}])

    def close(self) -> None:
        self.closed = True


def _client_with_factory(factory) -> TestClient:
    app = FastAPI()
    app.include_router(build_router(warehouse_factory=factory))
    return TestClient(app)


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


def test_xpro_decision_post_uses_write_context(monkeypatch) -> None:
    import market_regime_engine.fixed_income.api_handlers as api_handlers
    import market_regime_engine.fixed_income.xpro_decision as xpro_decision

    events: list[str] = []
    artifact = _minimal_artifact()
    wh = _CloseTrackingWarehouse()

    @contextlib.contextmanager
    def _write_context(warehouse):
        events.append("enter")
        yield
        events.append("exit")

    def _write(artifact_to_write: dict) -> int:
        assert events == ["enter"]
        wh.writes.append(artifact_to_write)
        return 1

    wh.write_xpro_decision_artifact = _write  # type: ignore[method-assign]
    monkeypatch.setattr(api_handlers, "_xpro_write_context", _write_context)
    monkeypatch.setattr(
        xpro_decision,
        "build_xpro_decision_artifact",
        lambda *args, **kwargs: artifact,
    )

    resp = _client_with_factory(lambda: wh).post("/v1/xpro/decision", json=_payload("req-xpro-write-context"))

    assert resp.status_code == 200, resp.text
    assert events == ["enter", "exit"]
    assert wh.writes == [artifact]
    assert wh.closed is True


def test_xpro_write_context_uses_pooled_warehouse_lock(monkeypatch, tmp_path: Path) -> None:
    import market_regime_engine.fixed_income.api_handlers as api_handlers
    import market_regime_engine.storage as storage

    db = tmp_path / "pooled-xpro-lock.duckdb"
    wh = storage.get_pooled_warehouse(db)
    events: list[Path] = []

    @contextlib.contextmanager
    def _lock(path):
        events.append(Path(path).resolve())
        yield

    monkeypatch.setattr(storage, "pooled_warehouse_write_lock", _lock)

    with api_handlers._xpro_write_context(wh):
        events.append(Path("inside"))

    assert events == [db.resolve(), Path("inside")]


def test_xpro_decision_post_closes_non_pooled_warehouse(monkeypatch) -> None:
    import market_regime_engine.fixed_income.xpro_decision as xpro_decision

    artifact = _minimal_artifact("dec-post-close")
    wh = _CloseTrackingWarehouse()
    monkeypatch.setattr(
        xpro_decision,
        "build_xpro_decision_artifact",
        lambda *args, **kwargs: artifact,
    )

    resp = _client_with_factory(lambda: wh).post("/v1/xpro/decision", json=_payload("req-xpro-close"))

    assert resp.status_code == 200, resp.text
    assert wh.writes == [artifact]
    assert wh.closed is True


def test_xpro_decision_get_closes_non_pooled_warehouse() -> None:
    artifact = _minimal_artifact("dec-get-close")
    wh = _CloseTrackingWarehouse(latest_payload=artifact)

    resp = _client_with_factory(lambda: wh).get("/v1/xpro/decision/dec-get-close")

    assert resp.status_code == 200, resp.text
    assert resp.json()["decision_id"] == "dec-get-close"
    assert wh.closed is True


def test_xpro_decision_get_read_failure_returns_503_fail_closed() -> None:
    wh = _CloseTrackingWarehouse(read_error=RuntimeError("simulated warehouse outage"))

    resp = _client_with_factory(lambda: wh).get("/v1/xpro/decision/dec-read-fails")

    assert resp.status_code == 503, resp.text
    assert resp.json() == {"detail": "xpro_decision_read_failed", "release_gate": False}
    assert wh.closed is True
