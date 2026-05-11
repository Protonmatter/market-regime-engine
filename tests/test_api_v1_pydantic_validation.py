# SPDX-License-Identifier: Apache-2.0
"""PR-5 §C.1: Pydantic v2 validation on POST /v1/execution_confidence."""

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
from market_regime_engine.fixed_income.api import ExecutionConfidenceRequestModel
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
    db = tmp_path / "validate.duckdb"
    monkeypatch.setenv("MRE_DB_PATH", str(db))
    monkeypatch.delenv("MRE_API_KEY", raising=False)
    close_pooled_warehouses()
    wh = Warehouse(db)
    _seed(wh, pd.Timestamp("2026-05-01T16:00:00Z"))
    wh.close()

    # Re-import api_v1 so it picks up MRE_DB_PATH.
    sys.modules.pop("market_regime_engine.api_v1", None)
    import market_regime_engine.api_v1 as api_v1

    importlib.reload(api_v1)
    from fastapi.testclient import TestClient

    with TestClient(api_v1.app) as client:
        yield client
    close_pooled_warehouses()


def _valid_payload(request_id: str = "req-1") -> dict:
    return {
        "timestamp": "2026-05-01T16:00:30Z",
        "cusip": "00206RGB6",
        "side": "buy",
        "notional": 1_000_000,
        "protocol": "Auto-X",
        "urgency": "normal",
        "request_id": request_id,
    }


# --------------------------------------------------------------------------
# pydantic model unit tests
# --------------------------------------------------------------------------


def test_pydantic_model_accepts_valid_body() -> None:
    body = ExecutionConfidenceRequestModel(**_valid_payload())
    assert body.cusip == "00206RGB6"
    assert body.notional == 1_000_000


def test_pydantic_model_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="explicit tz info"):
        ExecutionConfidenceRequestModel(**{**_valid_payload(), "timestamp": "2026-05-01T16:00:30"})


def test_pydantic_model_rejects_non_alphanumeric_cusip() -> None:
    with pytest.raises(ValueError, match="alphanumeric"):
        ExecutionConfidenceRequestModel(**{**_valid_payload(), "cusip": "00206RGB!"})


def test_pydantic_model_rejects_oversized_notional() -> None:
    with pytest.raises(ValueError):
        ExecutionConfidenceRequestModel(**{**_valid_payload(), "notional": 1_000_000_000})


def test_pydantic_model_rejects_unknown_side() -> None:
    with pytest.raises(ValueError):
        ExecutionConfidenceRequestModel(**{**_valid_payload(), "side": "short"})


def test_pydantic_model_rejects_extra_fields() -> None:
    """``extra="forbid"`` is the v1.5 boundary contract — unknown fields
    indicate a stale client."""
    with pytest.raises(ValueError):
        ExecutionConfidenceRequestModel(**{**_valid_payload(), "rogue": "field"})


# --------------------------------------------------------------------------
# end-to-end FastAPI tests
# --------------------------------------------------------------------------


def test_endpoint_accepts_valid_body_and_returns_200(client) -> None:
    resp = client.post("/v1/execution_confidence", json=_valid_payload())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["cusip"] == "00206RGB6"
    assert "confidence_score" in body
    assert "release_gate" in body
    assert "metadata" in body
    assert "signal_age_seconds_credit_regime" in body["metadata"]


def test_endpoint_rejects_naive_timestamp_with_422(client) -> None:
    bad = {**_valid_payload(), "timestamp": "2026-05-01T16:00:30"}
    resp = client.post("/v1/execution_confidence", json=bad)
    assert resp.status_code == 422


def test_endpoint_rejects_negative_notional_with_422(client) -> None:
    bad = {**_valid_payload(), "notional": -10}
    resp = client.post("/v1/execution_confidence", json=bad)
    assert resp.status_code == 422
