# SPDX-License-Identifier: Apache-2.0
"""v1.6.0 — A12 / Finding §3.6 regression test.

Pin the contract that the new ``idx_credit_regime_ts_run`` secondary
index on ``credit_regime_scores(timestamp DESC, model_run_id DESC)`` is

1. registered in the FI schema (covered by an integration test that
   queries SQLite's index list), and
2. honoured by the planner for the
   ``latest_credit_regime_score`` SQL fast path.

The v1.5.x implementation relied on the PK ``(model_run_id, timestamp)``
which cannot serve a leading ``ORDER BY timestamp DESC`` sort. EXPLAIN
QUERY PLAN should mention the new ``idx_credit_regime_ts_run`` (or
otherwise demonstrate the planner is no longer doing a full
table scan + sort).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import market_regime_engine.fixed_income  # noqa: F401  (table registration)
from market_regime_engine.storage import Warehouse


@pytest.fixture
def wh_sqlite(tmp_path: Path) -> Warehouse:
    """SQLite-backed warehouse — EXPLAIN QUERY PLAN works there too."""
    return Warehouse(tmp_path / "idx.sqlite")


def _seed(wh: Warehouse, n: int) -> None:
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    for i in range(n):
        rows.append(
            {
                "model_run_id": f"credit_spread_regime-production-{i % 3}",
                "timestamp": (base + pd.Timedelta(seconds=i * 10)).isoformat(),
                "regime_score": float(i % 100),
                "regime_label": "RISK_ON_COMPRESSION",
                "confidence": 0.9,
                "drivers_json": "[]",
                "component_scores_json": "{}",
                "release_gate": 1,
                "artifact_hash": f"h{i}",
                "metadata_json": "{}",
            }
        )
    wh.write_credit_regime_score(pd.DataFrame(rows))


def test_idx_credit_regime_ts_run_is_created(wh_sqlite: Warehouse) -> None:
    """The secondary index must exist on the credit_regime_scores table
    after the warehouse boots its FI schema."""
    _seed(wh_sqlite, n=10)
    backend = wh_sqlite._backend  # noqa: SLF001  inspect-only
    rows = backend.read_sql(
        "SELECT name FROM sqlite_master WHERE type='index' "
        "AND tbl_name='credit_regime_scores'"
    )
    index_names = set(rows["name"].astype(str).tolist())
    assert "idx_credit_regime_ts_run" in index_names, (
        f"missing idx_credit_regime_ts_run; got: {index_names}"
    )


def test_latest_credit_regime_score_plan_hits_secondary_index(
    wh_sqlite: Warehouse,
) -> None:
    """EXPLAIN QUERY PLAN must reference ``idx_credit_regime_ts_run`` for
    the ``latest_credit_regime_score`` SQL fast path."""
    _seed(wh_sqlite, n=200)
    backend = wh_sqlite._backend  # noqa: SLF001  inspect-only
    plan = backend.read_sql(
        "EXPLAIN QUERY PLAN "
        "SELECT * FROM credit_regime_scores "
        "ORDER BY timestamp DESC, model_run_id DESC "
        "LIMIT 1"
    )
    plan_text = " ".join(plan["detail"].astype(str).tolist())
    assert "idx_credit_regime_ts_run" in plan_text, (
        f"planner did NOT pick idx_credit_regime_ts_run; plan={plan_text!r}"
    )


def test_latest_credit_regime_score_returns_max_timestamp_row(
    wh_sqlite: Warehouse,
) -> None:
    """End-to-end smoke: the indexed-read returns the row with the
    maximum timestamp \u2014 the same answer the legacy full-scan would
    have produced."""
    _seed(wh_sqlite, n=200)
    out = wh_sqlite.latest_credit_regime_score()
    assert out is not None
    assert len(out) == 1
    legacy = wh_sqlite.read_credit_regime_scores()
    legacy = legacy.sort_values(
        ["timestamp", "model_run_id"], ascending=[False, False]
    )
    expected_ts = legacy.iloc[0]["timestamp"]
    assert out.iloc[0]["timestamp"] == expected_ts
