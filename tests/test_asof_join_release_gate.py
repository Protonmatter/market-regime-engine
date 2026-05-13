# SPDX-License-Identifier: Apache-2.0
"""v1.6.0 — A14 / Finding §3.8 governance regression test.

Pin the contract that ``Warehouse.enrich_execution_requests_asof``
restricts the ASOF JOIN to ``release_gate = 1`` rows on both the
credit-regime and liquidity-stress sides. A not-yet-promoted candidate
score (``release_gate = 0``) must NEVER colour an execution decision —
treating ungated candidate output as a governance-approved label is
the kind of silent fail-open the v1.5 governance triple was designed
to prevent.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

from market_regime_engine.storage import Warehouse


def test_asof_join_sql_filters_on_release_gate() -> None:
    """A14: the source must encode the governance contract. Inspect
    the function source to confirm both ``c.release_gate = 1`` and
    ``l.release_gate = 1`` predicates are present in the ASOF JOIN
    SQL — independent of which backend executes the join."""
    src = inspect.getsource(Warehouse.enrich_execution_requests_asof)
    assert "c.release_gate = 1" in src, (
        "ASOF LEFT JOIN credit_regime_scores is missing the release_gate filter"
    )
    assert "l.release_gate = 1" in src, (
        "ASOF LEFT JOIN liquidity_stress_scores is missing the release_gate filter"
    )


@pytest.fixture
def duckdb_warehouse(tmp_path: Path) -> Warehouse:
    """DuckDB-backed warehouse — skipped at fixture level when duckdb
    isn't installed so the source-inspection test above still runs."""
    pytest.importorskip("duckdb")
    return Warehouse(tmp_path / "asof.duckdb")


def _seed_credit_regime(wh: Warehouse, rows: list[dict]) -> None:
    frame = pd.DataFrame(
        [
            {
                "model_run_id": r.get("model_run_id", "credit-prod-1"),
                "timestamp": r["timestamp"],
                "regime_score": float(r.get("regime_score", 50.0)),
                "regime_label": r.get("regime_label", "NORMAL_LIQUIDITY"),
                "confidence": float(r.get("confidence", 0.8)),
                "drivers_json": json.dumps(r.get("drivers", [])),
                "component_scores_json": "{}",
                "release_gate": int(r["release_gate"]),
                "artifact_hash": r.get("artifact_hash", "h"),
                "metadata_json": "{}",
            }
            for r in rows
        ]
    )
    wh.write_credit_regime_score(frame)


def _seed_liquidity(wh: Warehouse, rows: list[dict]) -> None:
    frame = pd.DataFrame(
        [
            {
                "model_run_id": r.get("model_run_id", "liq-prod-1"),
                "scope_type": r.get("scope_type", "cusip"),
                "scope_id": r["scope_id"],
                "timestamp": r["timestamp"],
                "liquidity_score": float(r.get("liquidity_score", 30.0)),
                "liquidity_label": r["liquidity_label"],
                "confidence": 0.8,
                "drivers_json": "[]",
                "release_gate": int(r["release_gate"]),
                "artifact_hash": "h",
                "metadata_json": "{}",
            }
            for r in rows
        ]
    )
    wh.write_liquidity_stress_score(frame)


def _exec_request(timestamp: str, cusip: str) -> pd.DataFrame:
    return pd.DataFrame(
        [{"request_id": "r1", "timestamp": timestamp, "cusip": cusip}]
    )


def test_asof_join_excludes_release_gate_false_credit_row(
    duckdb_warehouse: Warehouse,
) -> None:
    """A14: a credit row with release_gate=0 immediately before the
    request timestamp must NOT be joined onto the request, even though
    it satisfies ``e.timestamp >= c.timestamp``."""
    wh = duckdb_warehouse
    _seed_credit_regime(
        wh,
        [
            {
                "timestamp": "2026-05-01T10:00:00Z",
                "regime_label": "GATED_OLDER",
                "release_gate": 1,
            },
            {
                "timestamp": "2026-05-01T15:00:00Z",
                "regime_label": "UNGATED_CANDIDATE",
                "release_gate": 0,
            },
        ],
    )
    _seed_liquidity(
        wh,
        [
            {
                "scope_id": "CUSIP1",
                "timestamp": "2026-05-01T10:00:00Z",
                "liquidity_label": "NORMAL",
                "release_gate": 1,
            }
        ],
    )

    out = wh.enrich_execution_requests_asof(
        _exec_request("2026-05-01T16:00:00Z", "CUSIP1")
    )
    assert len(out) == 1
    assert out.iloc[0]["regime_label"] == "GATED_OLDER"


def test_asof_join_excludes_release_gate_false_liquidity_row(
    duckdb_warehouse: Warehouse,
) -> None:
    """A14: same governance contract for the liquidity side."""
    wh = duckdb_warehouse
    _seed_credit_regime(
        wh,
        [
            {
                "timestamp": "2026-05-01T10:00:00Z",
                "regime_label": "GATED",
                "release_gate": 1,
            }
        ],
    )
    _seed_liquidity(
        wh,
        [
            {
                "scope_id": "CUSIP1",
                "timestamp": "2026-05-01T10:00:00Z",
                "liquidity_label": "GATED_OLDER",
                "release_gate": 1,
            },
            {
                "scope_id": "CUSIP1",
                "timestamp": "2026-05-01T15:00:00Z",
                "liquidity_label": "UNGATED_CANDIDATE",
                "release_gate": 0,
            },
        ],
    )
    out = wh.enrich_execution_requests_asof(
        _exec_request("2026-05-01T16:00:00Z", "CUSIP1")
    )
    assert len(out) == 1
    assert out.iloc[0]["liquidity_label"] == "GATED_OLDER"


def test_asof_join_returns_null_when_no_gated_row_exists(
    duckdb_warehouse: Warehouse,
) -> None:
    """A14 edge case: when EVERY candidate row is ungated, the join
    yields ``None`` — fail-closed at the SQL layer rather than
    surfacing a candidate label as if it had cleared the gate."""
    wh = duckdb_warehouse
    _seed_credit_regime(
        wh,
        [
            {
                "timestamp": "2026-05-01T10:00:00Z",
                "regime_label": "UNGATED_ONLY",
                "release_gate": 0,
            }
        ],
    )
    _seed_liquidity(
        wh,
        [
            {
                "scope_id": "CUSIP1",
                "timestamp": "2026-05-01T10:00:00Z",
                "liquidity_label": "UNGATED_ONLY",
                "release_gate": 0,
            }
        ],
    )
    out = wh.enrich_execution_requests_asof(
        _exec_request("2026-05-01T16:00:00Z", "CUSIP1")
    )
    assert len(out) == 1
    assert pd.isna(out.iloc[0]["regime_label"])
    assert pd.isna(out.iloc[0]["liquidity_label"])
