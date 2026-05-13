# SPDX-License-Identifier: Apache-2.0
"""PR-5 §C.2: 32 KB body cap on POST /v1/execution_confidence."""

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
    db = tmp_path / "cap.duckdb"
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

    with TestClient(api_v1.app) as client:
        yield client
    close_pooled_warehouses()


def test_endpoint_rejects_oversized_body_with_413(client) -> None:
    # Stuff metadata with a 50 KB blob.
    big_blob = "x" * (50 * 1024)
    payload = {
        "timestamp": "2026-05-01T16:00:30Z",
        "cusip": "00206RGB6",
        "side": "buy",
        "notional": 1_000_000,
        "protocol": "Auto-X",
        "urgency": "normal",
        "request_id": "req-large",
        "metadata": {"blob": big_blob},
    }
    resp = client.post("/v1/execution_confidence", json=payload)
    assert resp.status_code == 413, (resp.status_code, resp.text[:200])
    body = resp.json()
    assert "32 KB cap" in str(body)


def test_endpoint_accepts_body_just_under_cap(client) -> None:
    """v1.6.0 (REVIEW_DEEP_V1_5_2.md F9 / Finding §3.13):
    metadata is now capped at 8192 bytes (canonical-JSON-encoded).
    Send 4 KB metadata which is well under both the metadata cap
    (8 KB) and the body cap (32 KB)."""
    payload = {
        "timestamp": "2026-05-01T16:00:30Z",
        "cusip": "00206RGB6",
        "side": "buy",
        "notional": 1_000_000,
        "protocol": "Auto-X",
        "urgency": "normal",
        "request_id": "req-under-cap",
        "metadata": {"blob": "y" * 4096},
    }
    resp = client.post("/v1/execution_confidence", json=payload)
    assert resp.status_code == 200, resp.text


def test_endpoint_rejects_metadata_above_8kb(client) -> None:
    """v1.6.0 (REVIEW_DEEP_V1_5_2.md F9 / Finding §3.13):
    metadata must pass the 8192-byte cap. The full body is still
    under the 32 KB body cap, so the rejection must come from the
    Pydantic metadata validator (422) rather than the body-size
    middleware (413)."""
    payload = {
        "timestamp": "2026-05-01T16:00:30Z",
        "cusip": "00206RGB6",
        "side": "buy",
        "notional": 1_000_000,
        "protocol": "Auto-X",
        "urgency": "normal",
        "request_id": "req-large-metadata",
        "metadata": {"blob": "y" * (10 * 1024)},
    }
    resp = client.post("/v1/execution_confidence", json=payload)
    assert resp.status_code == 422, resp.text
    assert "metadata too large" in resp.text


def test_endpoint_rejects_deeply_nested_metadata(client) -> None:
    """v1.6.0 F9: nesting depth > 5 must be rejected by the
    validator."""
    nested: dict = {"v": 1}
    for _ in range(7):
        nested = {"k": nested}
    payload = {
        "timestamp": "2026-05-01T16:00:30Z",
        "cusip": "00206RGB6",
        "side": "buy",
        "notional": 1_000_000,
        "protocol": "Auto-X",
        "urgency": "normal",
        "request_id": "req-deep-metadata",
        "metadata": nested,
    }
    resp = client.post("/v1/execution_confidence", json=payload)
    assert resp.status_code == 422, resp.text
    assert "depth" in resp.text.lower()
