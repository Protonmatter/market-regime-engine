# SPDX-License-Identifier: Apache-2.0
"""PR-5 §L (REVIEW.md §3.6 PR-1): load smoke for POST /v1/execution_confidence.

Spins up the FastAPI TestClient against an in-memory DuckDB seeded with
synthetic regime + liquidity signals, then issues 1000 sequential POSTs
varying the cusip. Asserts p99 latency < 500 ms with the pooled warehouse
in play — the slow marker keeps the test out of the default ``pytest -m
"not slow"`` run.
"""

from __future__ import annotations

import importlib
import statistics
import sys
import time
import uuid
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

_CUSIPS = tuple(f"00206RG{idx:02X}" for idx in range(10))


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

    for cusip in _CUSIPS:
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
                scope_id=cusip,
                asof=ts,
                release_gate=True,
            ),
        )


@pytest.mark.slow
def test_load_smoke_1000_posts_p99_under_500ms(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / "load.duckdb"
    monkeypatch.setenv("MRE_DB_PATH", str(db))
    monkeypatch.delenv("MRE_API_KEY", raising=False)
    # Use a generous rate limit so the load smoke is not throttled.
    monkeypatch.setenv("MRE_FI_EXEC_CONF_RATE_LIMIT", "100000/second")
    close_pooled_warehouses()
    wh = Warehouse(db)
    _seed(wh, pd.Timestamp("2026-05-01T16:00:00Z"))
    wh.close()

    sys.modules.pop("market_regime_engine.api_v1", None)
    import market_regime_engine.api_v1 as api_v1

    importlib.reload(api_v1)
    from fastapi.testclient import TestClient

    durations_ms: list[float] = []
    n_requests = 1000
    with TestClient(api_v1.app) as client:
        # warm-up — first request pays the import cost.
        client.post(
            "/v1/execution_confidence",
            json={
                "timestamp": "2026-05-01T16:00:30Z",
                "cusip": _CUSIPS[0],
                "side": "buy",
                "notional": 1_000_000,
                "protocol": "Auto-X",
                "urgency": "normal",
                "request_id": uuid.uuid4().hex,
            },
        )
        for i in range(n_requests):
            payload = {
                "timestamp": "2026-05-01T16:00:30Z",
                "cusip": _CUSIPS[i % len(_CUSIPS)],
                "side": "buy" if i % 2 == 0 else "sell",
                "notional": 500_000 + (i % 100) * 1_000,
                "protocol": ["Auto-X", "RFQ", "Manual"][i % 3],
                "urgency": ["low", "normal", "high"][i % 3],
                "request_id": uuid.uuid4().hex,
            }
            t0 = time.perf_counter()
            resp = client.post("/v1/execution_confidence", json=payload)
            durations_ms.append((time.perf_counter() - t0) * 1000.0)
            assert resp.status_code == 200, resp.text
    close_pooled_warehouses()

    durations_ms.sort()
    p50 = statistics.median(durations_ms)
    p99 = durations_ms[int(0.99 * (len(durations_ms) - 1))]
    print(f"\nLoad smoke: n={n_requests} p50={p50:.2f}ms p99={p99:.2f}ms")
    assert p99 < 500.0, f"p99 latency {p99:.2f}ms exceeded 500ms budget"
