# SPDX-License-Identifier: Apache-2.0
"""v1.5.1 (PR-9 FIX 2): performance + parity tests for the indexed-SQL
fast path on the credit-regime read.

Pre-PR-9 ``latest_credit_regime_score`` did a full table scan +
``.iloc[-1]``. The fast path issues a parameterised ``LIMIT 1`` against
the backend. We assert both:

1. **Performance.** p99 of the SQL path on a 100k-row fixture must be ≤
   5 ms (p50 ≈ 1 ms on dev hardware). The legacy full-table path is
   ~80 ms on the same input. Threshold widened to 25 ms for shared CI
   runners — anything over that means the indexed path regressed.
2. **Parity.** The two paths must return the same
   :class:`CreditRegimeOutput` (identical to the legacy ``.iloc[-1]``).
"""

from __future__ import annotations

import json
import time

import pandas as pd
import pytest

from market_regime_engine.fixed_income.credit_spread_regime import (
    latest_credit_regime_score,
)
from market_regime_engine.storage import Warehouse


@pytest.fixture(scope="module")
def warehouse_100k(tmp_path_factory: pytest.TempPathFactory) -> Warehouse:
    """Build a DuckDB warehouse with 100k credit_regime_scores rows.

    Module scope keeps the build cost amortised across the perf and
    parity tests in this file.
    """
    db_dir = tmp_path_factory.mktemp("credit-perf")
    wh = Warehouse(path=str(db_dir / "credit.duckdb"))
    n = 100_000
    base_ts = pd.Timestamp("2024-01-01T00:00:00Z")
    timestamps = [(base_ts + pd.Timedelta(minutes=int(i))).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(n)]
    rows = pd.DataFrame(
        {
            "model_run_id": [f"run-{i % 100}" for i in range(n)],
            "timestamp": timestamps,
            "regime_score": [50.0 + (i % 20) for i in range(n)],
            "regime_label": ["neutral"] * n,
            "confidence": [0.75] * n,
            "drivers_json": [json.dumps([])] * n,
            "component_scores_json": [json.dumps({})] * n,
            "release_gate": [1] * n,
            "artifact_hash": [f"sha256:{i:064x}" for i in range(n)],
            "metadata_json": [json.dumps({})] * n,
        }
    )
    wh.write_credit_regime_score(rows)
    return wh


def test_latest_credit_regime_score_indexed_sql_p99_under_25ms(
    warehouse_100k: Warehouse,
) -> None:
    """Indexed SQL fast path: p99 of 50 sequential reads ≤ 25 ms.

    The PR-9 target is 5 ms on dev hardware; we widen to 25 ms here to
    avoid flakes on shared CI. A regression that brings the full-table
    pandas scan back (~80 ms) would still be caught.
    """
    timings_ms: list[float] = []
    for _ in range(50):
        t0 = time.perf_counter()
        out = latest_credit_regime_score(warehouse_100k)
        timings_ms.append((time.perf_counter() - t0) * 1000.0)
        assert out is not None
    timings_ms.sort()
    p50 = timings_ms[25]
    p99 = timings_ms[-1]
    assert p99 <= 25.0, (
        f"latest_credit_regime_score regressed: p99={p99:.2f}ms, p50={p50:.2f}ms; "
        "expected p99 ≤ 25 ms after PR-9 indexed-SQL fast path"
    )


def test_latest_credit_regime_score_indexed_matches_legacy(
    warehouse_100k: Warehouse,
) -> None:
    """Parity: indexed-SQL output equals legacy full-table ``.iloc[-1]``."""
    fast_output = latest_credit_regime_score(warehouse_100k)
    assert fast_output is not None
    legacy_df = warehouse_100k.read_credit_regime_scores()
    legacy_row = legacy_df.iloc[-1]
    assert fast_output.model_run_id == str(legacy_row["model_run_id"])
    assert fast_output.regime_score == float(legacy_row["regime_score"])
    assert fast_output.confidence == float(legacy_row["confidence"])
    assert fast_output.artifact_hash == str(legacy_row["artifact_hash"])


def test_warehouse_latest_credit_regime_score_asof(
    warehouse_100k: Warehouse,
) -> None:
    """``asof`` argument caps the result at the given timestamp."""
    asof_mid = "2024-01-01T01:00:00Z"
    df = warehouse_100k.latest_credit_regime_score(asof=asof_mid)
    assert df is not None
    assert not df.empty
    # All rows in the fixture have lexicographically sortable
    # timestamps with the ``Z`` suffix; the row we get back must have
    # ``timestamp <= asof_mid``.
    assert str(df.iloc[0]["timestamp"]) <= asof_mid


def test_warehouse_latest_credit_regime_score_returns_none_on_empty(
    tmp_path: pytest.TempPathFactory,
) -> None:
    wh = Warehouse(path=str(tmp_path / "empty.duckdb"))
    assert wh.latest_credit_regime_score() is None


def test_warehouse_latest_liquidity_stress_score_uses_indexed_path(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """``latest_liquidity_stress_score`` hits ``idx_liquidity_scope_ts``."""
    wh = Warehouse(path=str(tmp_path / "liq.duckdb"))
    rows = pd.DataFrame(
        {
            "model_run_id": ["run-1", "run-2", "run-3"],
            "scope_type": ["cusip", "cusip", "sector"],
            "scope_id": ["AAA", "AAA", "FIN"],
            "timestamp": [
                "2024-01-01T00:00:00Z",
                "2024-01-02T00:00:00Z",
                "2024-01-03T00:00:00Z",
            ],
            "liquidity_score": [50.0, 60.0, 70.0],
            "liquidity_label": ["normal", "stressed", "normal"],
            "confidence": [0.8, 0.7, 0.9],
            "drivers_json": ["[]", "[]", "[]"],
            "release_gate": [1, 1, 1],
            "artifact_hash": ["sha256:0", "sha256:1", "sha256:2"],
            "metadata_json": ["{}", "{}", "{}"],
        }
    )
    wh.write_liquidity_stress_score(rows)

    df = wh.latest_liquidity_stress_score(scope_type="cusip", scope_id="AAA")
    assert df is not None
    got_ts = pd.Timestamp(df.iloc[0]["timestamp"])
    expected_ts = pd.Timestamp("2024-01-02T00:00:00")
    if got_ts.tzinfo is not None:
        got_ts = got_ts.tz_convert("UTC").tz_localize(None)
    assert got_ts == expected_ts

    df_all = wh.latest_liquidity_stress_score()
    assert df_all is not None
    assert df_all.iloc[0]["scope_type"] in {"cusip", "sector"}


def test_enrich_execution_requests_asof_duckdb_path(tmp_path: pytest.TempPathFactory) -> None:
    """DuckDB-only ASOF LEFT JOIN: annotates exec requests with credit + liquidity labels."""
    wh = Warehouse(path=str(tmp_path / "asof.duckdb"))
    credit = pd.DataFrame(
        {
            "model_run_id": ["c-1"],
            "timestamp": ["2024-01-01T00:00:00Z"],
            "regime_score": [50.0],
            "regime_label": ["risk_off"],
            "confidence": [0.9],
            "drivers_json": ["[]"],
            "component_scores_json": ["{}"],
            "release_gate": [1],
            "artifact_hash": ["sha256:c"],
            "metadata_json": ["{}"],
        }
    )
    liquidity = pd.DataFrame(
        {
            "model_run_id": ["l-1"],
            "scope_type": ["cusip"],
            "scope_id": ["AAAAAAAA"],
            "timestamp": ["2024-01-01T00:00:00Z"],
            "liquidity_score": [80.0],
            "liquidity_label": ["liquid"],
            "confidence": [0.95],
            "drivers_json": ["[]"],
            "release_gate": [1],
            "artifact_hash": ["sha256:l"],
            "metadata_json": ["{}"],
        }
    )
    wh.write_credit_regime_score(credit)
    wh.write_liquidity_stress_score(liquidity)

    requests = pd.DataFrame(
        {
            "request_id": ["r-1"],
            "timestamp": [pd.Timestamp("2024-01-02T00:00:00Z")],
            "cusip": ["AAAAAAAA"],
            "side": ["buy"],
            "notional": [100000.0],
        }
    )
    out = wh.enrich_execution_requests_asof(requests)
    assert "regime_label" in out.columns
    assert "liquidity_label" in out.columns
    assert out.iloc[0]["regime_label"] == "risk_off"
    assert out.iloc[0]["liquidity_label"] == "liquid"
