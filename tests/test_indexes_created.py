# SPDX-License-Identifier: Apache-2.0
"""Acceptance test for PR-2 task F / ASK-11 — per-table FI indexes are
created during ``Warehouse.init_schema`` on both DuckDB and SQLite.

Indexes verified (REVIEW.md §3.2 ASK-11 + plan §2):

- ``idx_trace_trades_cusip_ts`` on ``trace_trades(cusip, timestamp)``
- ``idx_rfq_events_cusip_ts`` on ``rfq_events(cusip, timestamp)``
- ``idx_dealer_quotes_cusip_ts`` on ``dealer_quotes(cusip, timestamp)``
- ``idx_liquidity_scope_ts`` on ``liquidity_stress_scores(scope_type,
  scope_id, timestamp)``
- ``idx_exec_conf_action_ts`` on
  ``execution_confidence_predictions(timestamp, recommended_action)``
- ``idx_evidence_packs_run_id`` on
  ``fixed_income_evidence_packs(model_run_id)``
- ``idx_exec_outcomes_request_id`` on
  ``execution_outcomes(request_id)``

Plus the ``idx_bond_reference_valid_window`` index that supports the
temporal versioning helpers from PR-2 task C.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

import market_regime_engine.fixed_income  # noqa: F401 - register FI tables
from market_regime_engine.storage import Warehouse

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


_EXPECTED_INDEXES: dict[str, str] = {
    "idx_trace_trades_cusip_ts": "trace_trades",
    "idx_rfq_events_cusip_ts": "rfq_events",
    "idx_dealer_quotes_cusip_ts": "dealer_quotes",
    "idx_liquidity_scope_ts": "liquidity_stress_scores",
    "idx_exec_conf_action_ts": "execution_confidence_predictions",
    "idx_evidence_packs_run_id": "fixed_income_evidence_packs",
    "idx_exec_outcomes_request_id": "execution_outcomes",
    "idx_bond_reference_valid_window": "bond_reference",
}


def test_fi_indexes_exist_after_init_duckdb(tmp_path: Path) -> None:
    """Every expected FI index appears in ``duckdb_indexes()`` after a
    fresh Warehouse init on DuckDB."""

    pytest.importorskip("duckdb")
    wh = Warehouse(str(tmp_path / "idx.duckdb"), backend="duckdb")
    try:
        rows = wh._backend.conn.execute(
            "SELECT index_name, table_name FROM duckdb_indexes()"
        ).fetchall()
        actual = {name: table for name, table in rows}
        for idx_name, expected_table in _EXPECTED_INDEXES.items():
            assert idx_name in actual, f"missing index {idx_name}; have {sorted(actual)!r}"
            assert (
                actual[idx_name] == expected_table
            ), f"index {idx_name!r} expected on {expected_table!r}, found on {actual[idx_name]!r}"
    finally:
        wh.close()


def test_fi_indexes_exist_after_init_sqlite(tmp_path: Path) -> None:
    """SQLite parity: the same index names appear in
    ``sqlite_master`` after a fresh Warehouse init."""

    wh = Warehouse(str(tmp_path / "idx.db"), backend="sqlite")
    try:
        rows = wh._backend.conn.execute(
            "SELECT name, tbl_name FROM sqlite_master WHERE type = 'index'"
        ).fetchall()
        actual = {name: table for name, table in rows}
        for idx_name, expected_table in _EXPECTED_INDEXES.items():
            assert idx_name in actual, f"missing index {idx_name}; have {sorted(actual)!r}"
            assert (
                actual[idx_name] == expected_table
            ), f"index {idx_name!r} expected on {expected_table!r}, found on {actual[idx_name]!r}"
    finally:
        wh.close()


def test_index_creation_is_idempotent(tmp_path: Path) -> None:
    """Re-initialising the warehouse against an existing DB file is a
    no-op because every CREATE INDEX uses ``IF NOT EXISTS``."""

    pytest.importorskip("duckdb")
    path = tmp_path / "idemp.duckdb"
    Warehouse(str(path), backend="duckdb").close()
    Warehouse(str(path), backend="duckdb").close()  # second init must not raise
    wh = Warehouse(str(path), backend="duckdb")
    try:
        rows = wh._backend.conn.execute(
            "SELECT index_name FROM duckdb_indexes() WHERE index_name LIKE 'idx_%'"
        ).fetchall()
        names = sorted({row[0] for row in rows})
        assert len(names) == len(set(names)), f"duplicate index names: {names!r}"
    finally:
        wh.close()
