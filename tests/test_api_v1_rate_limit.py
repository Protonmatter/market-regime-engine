# SPDX-License-Identifier: Apache-2.0
"""PR-5 §C.3: per-API-key rate limit on POST /v1/execution_confidence."""

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
from market_regime_engine.storage import Warehouse, close_pooled_warehouses


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
    db = tmp_path / "rate.duckdb"
    monkeypatch.setenv("MRE_DB_PATH", str(db))
    monkeypatch.delenv("MRE_API_KEY", raising=False)
    # Strict 3/second so the test runs in < 5s.
    monkeypatch.setenv("MRE_FI_EXEC_CONF_RATE_LIMIT", "3/second")
    close_pooled_warehouses()
    wh = Warehouse(db)
    _seed(wh, pd.Timestamp("2026-05-01T16:00:00Z"))
    wh.close()

    sys.modules.pop("market_regime_engine.api_v1", None)
    import market_regime_engine.api_v1 as api_v1

    importlib.reload(api_v1)
    from fastapi.testclient import TestClient

    with TestClient(api_v1.app) as client:
        yield client
    close_pooled_warehouses()


def _payload(request_id: str) -> dict:
    return {
        "timestamp": "2026-05-01T16:00:30Z",
        "cusip": "00206RGB6",
        "side": "buy",
        "notional": 1_000_000,
        "protocol": "Auto-X",
        "urgency": "normal",
        "request_id": request_id,
    }


def test_rate_limit_fires_after_quota_exceeded(client) -> None:
    """3/second budget — the 4th rapid request must 429."""
    last_status = None
    for i in range(8):
        resp = client.post("/v1/execution_confidence", json=_payload(f"req-{i}"))
        last_status = resp.status_code
        if last_status == 429:
            # Retry-After header should be present.
            assert resp.headers.get("retry-after") is not None
            return
    pytest.fail(f"expected 429 after the 3/second budget was exceeded; last_status={last_status}")


def test_rate_limit_is_per_api_key(client) -> None:
    """Distinct ``X-API-Key`` values get separate buckets — sending 3
    requests from each of two keys should NOT 429 on the 4th overall
    request when each key is below its own budget."""
    for i in range(3):
        resp_a = client.post(
            "/v1/execution_confidence",
            json=_payload(f"a-{i}"),
            headers={"X-API-Key": "key-a"},
        )
        assert resp_a.status_code == 200, resp_a.text
    # A request from a *different* key still has full budget.
    resp_b = client.post(
        "/v1/execution_confidence",
        json=_payload("b-1"),
        headers={"X-API-Key": "key-b"},
    )
    assert resp_b.status_code == 200, resp_b.text
