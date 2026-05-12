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

Plus the ``idx_bond_reference_valid_window`` index that supports the
temporal versioning helpers from PR-2 task C.

v1.5 PR-8 (Tier-4 FLAG F-A1, REVIEW.md):
``idx_exec_outcomes_request_id`` was DROPPED — the PRIMARY KEY on
``execution_outcomes.request_id`` already creates an automatic
B-tree index on both DuckDB and SQLite, so the explicit secondary
index was redundant. The assertion below now verifies the index is
NOT present after fresh init.
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
    # v1.5 PR-8 (Tier-4 FLAG F-A1): ``idx_exec_outcomes_request_id``
    # is intentionally OMITTED — the PK on ``execution_outcomes(request_id)``
    # auto-creates the equivalent index; the explicit secondary one
    # was redundant.
    "idx_bond_reference_valid_window": "bond_reference",
}


_DROPPED_INDEXES_F_A1 = ("idx_exec_outcomes_request_id",)


def test_fi_indexes_exist_after_init_duckdb(tmp_path: Path) -> None:
    """Every expected FI index appears in ``duckdb_indexes()`` after a
    fresh Warehouse init on DuckDB."""

    pytest.importorskip("duckdb")
    wh = Warehouse(str(tmp_path / "idx.duckdb"), backend="duckdb")
    try:
        rows = wh._backend.conn.execute(
            "SELECT index_name, table_name FROM duckdb_indexes()"
        ).fetchall()
        actual = dict(rows)
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
        actual = dict(rows)
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


def test_dropped_redundant_index_not_in_duckdb_indexes(tmp_path: Path) -> None:
    """v1.5 PR-8 (Tier-4 FLAG F-A1): the redundant
    ``idx_exec_outcomes_request_id`` index MUST NOT appear in a fresh
    DuckDB warehouse — the PRIMARY KEY on ``request_id`` already
    creates the auto-index, so the explicit secondary one was
    consuming extra space for no benefit."""
    pytest.importorskip("duckdb")
    wh = Warehouse(str(tmp_path / "dropped.duckdb"), backend="duckdb")
    try:
        rows = wh._backend.conn.execute(
            "SELECT index_name FROM duckdb_indexes()"
        ).fetchall()
        actual = {row[0] for row in rows}
        for dropped in _DROPPED_INDEXES_F_A1:
            assert dropped not in actual, (
                f"expected dropped index {dropped!r} to be absent; "
                f"present in {sorted(actual)!r}"
            )
    finally:
        wh.close()


def test_dropped_redundant_index_not_in_sqlite_master(tmp_path: Path) -> None:
    """SQLite parity for the F-A1 drop: the redundant secondary index
    must be absent from ``sqlite_master``."""
    wh = Warehouse(str(tmp_path / "dropped.db"), backend="sqlite")
    try:
        rows = wh._backend.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index'"
        ).fetchall()
        actual = {row[0] for row in rows}
        for dropped in _DROPPED_INDEXES_F_A1:
            assert dropped not in actual, (
                f"expected dropped index {dropped!r} to be absent; "
                f"present in {sorted(actual)!r}"
            )
    finally:
        wh.close()
