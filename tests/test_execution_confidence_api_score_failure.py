# SPDX-License-Identifier: Apache-2.0
"""v1.6.0 — A10 / Finding §3.5 regression tests.

Pin the contract that when ``score_execution_confidence`` raises an
exception other than :class:`PitViolationError`, the POST
``/v1/execution_confidence`` handler MUST translate it to a 503
fail-closed envelope. Previously the handler only caught
``PitViolationError``; any other exception left the local ``response``
variable unbound and the trailing ``JSONResponse(...)`` line raised
``UnboundLocalError`` — surfacing a 500 with no governance envelope to
the operator.
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
    db = tmp_path / "score_fail.duckdb"
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
        yield testclient, api_v1
    close_pooled_warehouses()


def _payload(request_id: str = "req-score-fail") -> dict:
    return {
        "timestamp": "2026-05-01T16:00:30Z",
        "cusip": "00206RGB6",
        "side": "buy",
        "notional": 1_000_000,
        "protocol": "Auto-X",
        "urgency": "normal",
        "request_id": request_id,
    }


def test_unexpected_scorer_exception_returns_503_fail_closed_envelope(client, monkeypatch) -> None:
    """A10: a ``ValueError`` raised inside ``score_execution_confidence``
    must NOT propagate as an UnboundLocalError; the handler maps it to
    503 with the canonical fail-closed body."""
    testclient, _ = client

    def _boom(*args, **kwargs):
        raise ValueError("simulated scorer regression in component coefficients")

    monkeypatch.setattr(
        "market_regime_engine.fixed_income.api.score_execution_confidence",
        _boom,
    )

    resp = testclient.post("/v1/execution_confidence", json=_payload())
    assert resp.status_code == 503, resp.text
    body = resp.json()
    assert body["detail"]["detail"] == "score_failed"
    assert body["detail"]["release_gate"] is False


def test_unexpected_scorer_exception_does_not_leak_unbound_local(client, monkeypatch) -> None:
    """A10: an arbitrary RuntimeError must also produce the fail-closed
    envelope — pinning that the handler does not fall through to a
    bare ``return JSONResponse(execution_confidence_response_to_dict(
    response), ...)`` with an unbound name."""
    testclient, _ = client

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated downstream warehouse glitch")

    monkeypatch.setattr(
        "market_regime_engine.fixed_income.api.score_execution_confidence",
        _boom,
    )

    resp = testclient.post("/v1/execution_confidence", json=_payload("req-unbound"))
    assert resp.status_code == 503, resp.text
    body = resp.json()
    assert body["detail"]["detail"] == "score_failed"
    assert body["detail"]["release_gate"] is False
    # Specifically: there is NO mention of UnboundLocalError or
    # `response` in the error envelope. The handler swallowed the
    # internal exception and rendered the fail-closed shape.
    text = resp.text
    assert "UnboundLocalError" not in text
    assert "cannot access local variable" not in text
