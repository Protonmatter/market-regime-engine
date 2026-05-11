# SPDX-License-Identifier: Apache-2.0
"""PR-7 §K.3 — TRACE :class:`IngestContract` acceptance tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import market_regime_engine.fixed_income  # noqa: F401  - register FI tables
from market_regime_engine.fixed_income.ingest.trace import (
    TRACE_CONTRACT,
    ingest_trace,
)
from market_regime_engine.storage import Warehouse


def _valid_trace_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trade_id": "T1",
                "timestamp": pd.Timestamp("2026-05-01T10:00:00"),
                "cusip": "AAA111111",
                "price": 100.0,
                "size": 1_000_000.0,
                "side": "buy",
                "yield_pct": 4.5,
            },
            {
                "trade_id": "T2",
                "timestamp": pd.Timestamp("2026-05-01T10:01:00"),
                "cusip": "BBB222222",
                "price": 99.5,
                "size": 2_000_000.0,
                "side": "sell",
                "yield_pct": 4.6,
            },
        ]
    )


def test_trace_contract_accepts_valid_df(tmp_path: Path) -> None:
    db = Warehouse(str(tmp_path / "trace.duckdb"))
    try:
        report = ingest_trace(db, _valid_trace_frame())
        assert report.passed is True
        assert report.errors == ()
        assert report.dropped_count == 0
        assert report.rows_in == 2
        assert report.rows_out == 2
        df = db.read_trace_trades()
        assert len(df) == 2
    finally:
        db.close()


def test_trace_contract_rejects_missing_required_column(tmp_path: Path) -> None:
    db = Warehouse(str(tmp_path / "trace-missing.duckdb"))
    try:
        bad = _valid_trace_frame().drop(columns=["price"])
        report = ingest_trace(db, bad)
    finally:
        db.close()
    assert report.passed is False
    assert any("missing required columns" in err for err in report.errors)
    assert "price" in str(report.errors)


def test_trace_contract_warns_on_unknown_column_default_mode(tmp_path: Path) -> None:
    db = Warehouse(str(tmp_path / "trace-warn.duckdb"))
    try:
        df = _valid_trace_frame()
        df["random_field"] = 1
        report = ingest_trace(db, df)
    finally:
        db.close()
    assert report.passed is True
    assert any("unknown columns" in w for w in report.warnings)


def test_trace_contract_errors_on_unknown_column_strict_mode(
    tmp_path: Path,
) -> None:
    db = Warehouse(str(tmp_path / "trace-strict.duckdb"))
    try:
        df = _valid_trace_frame()
        df["random_field"] = 1
        with pytest.raises(ValueError, match="unknown columns"):
            ingest_trace(db, df, strict_unknown=True)
    finally:
        db.close()


def test_trace_contract_rejects_non_monotonic_timestamps(tmp_path: Path) -> None:
    db = Warehouse(str(tmp_path / "trace-nm.duckdb"))
    try:
        df = _valid_trace_frame()
        df.loc[1, "timestamp"] = pd.Timestamp("2025-01-01T00:00:00")
        report = ingest_trace(db, df)
    finally:
        db.close()
    assert report.passed is False
    assert any("monotonic" in err for err in report.errors)


def test_trace_contract_rejects_notional_out_of_bounds(tmp_path: Path) -> None:
    """Per the contract, notional > 500M is dropped from the persisted
    feed. TRACE contract uses ``size`` as the notional; we approximate
    via the contract's notional check by adding a notional column."""
    db = Warehouse(str(tmp_path / "trace-nb.duckdb"))
    try:
        df = _valid_trace_frame()
        df["notional"] = [100.0, 1_000_000_000.0]  # second row out of bounds
        report = ingest_trace(db, df, strict_unknown=False)
    finally:
        db.close()
    # First row passes, second row dropped.
    assert report.dropped_count >= 1
    assert any("out of bounds" in w for w in report.warnings)


def test_trace_contract_required_and_optional_columns() -> None:
    """The exported contract must list the AGENT.md PR-7 §K columns."""
    assert {"timestamp", "cusip", "price", "size", "side", "trade_id"} <= set(TRACE_CONTRACT.required_columns)
    assert {"yield_pct"} <= set(TRACE_CONTRACT.optional_columns)
