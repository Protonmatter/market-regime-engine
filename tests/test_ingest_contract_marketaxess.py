# SPDX-License-Identifier: Apache-2.0
"""PR-7 §K.3 — MarketAxess RFQ :class:`IngestContract` acceptance tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import market_regime_engine.fixed_income  # noqa: F401  - register FI tables
from market_regime_engine.fixed_income.ingest.marketaxess_rfq import (
    MARKETAXESS_RFQ_CONTRACT,
    ingest_marketaxess_rfq,
)
from market_regime_engine.storage import Warehouse


def _valid_rfq_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rfq_id": "RFQ1",
                "timestamp": pd.Timestamp("2026-05-01T10:00:00"),
                "cusip": "AAA111111",
                "side": "buy",
                "notional": 1_000_000.0,
                "protocol": "Auto-X",
                "dealers_requested": 5,
                "quotes_received": 3,
                "status": "filled",
            },
            {
                "rfq_id": "RFQ2",
                "timestamp": pd.Timestamp("2026-05-01T10:05:00"),
                "cusip": "BBB222222",
                "side": "sell",
                "notional": 2_500_000.0,
                "protocol": "RFQ",
                "dealers_requested": 8,
                "quotes_received": 5,
                "status": "open",
            },
        ]
    )


def test_marketaxess_contract_accepts_valid_df(tmp_path: Path) -> None:
    db = Warehouse(str(tmp_path / "rfq.duckdb"))
    try:
        report = ingest_marketaxess_rfq(db, _valid_rfq_frame())
        assert report.passed is True
        assert report.errors == ()
        assert report.dropped_count == 0
        df = db.read_rfq_events()
        assert len(df) == 2
    finally:
        db.close()


def test_marketaxess_contract_rejects_missing_required_column(tmp_path: Path) -> None:
    db = Warehouse(str(tmp_path / "rfq-miss.duckdb"))
    try:
        bad = _valid_rfq_frame().drop(columns=["status"])
        report = ingest_marketaxess_rfq(db, bad)
    finally:
        db.close()
    assert report.passed is False
    assert any("status" in err for err in report.errors)


def test_marketaxess_contract_drops_invalid_status(tmp_path: Path) -> None:
    db = Warehouse(str(tmp_path / "rfq-stat.duckdb"))
    try:
        df = _valid_rfq_frame()
        df.loc[1, "status"] = "ERR"
        report = ingest_marketaxess_rfq(db, df)
    finally:
        db.close()
    assert report.dropped_count == 1
    assert report.rows_out == 1


def test_marketaxess_contract_drops_invalid_side(tmp_path: Path) -> None:
    db = Warehouse(str(tmp_path / "rfq-side.duckdb"))
    try:
        df = _valid_rfq_frame()
        df.loc[0, "side"] = "long"  # not in {buy, sell}
        report = ingest_marketaxess_rfq(db, df)
    finally:
        db.close()
    assert report.dropped_count == 1


def test_marketaxess_contract_warns_on_unknown_column_default_mode(
    tmp_path: Path,
) -> None:
    db = Warehouse(str(tmp_path / "rfq-warn.duckdb"))
    try:
        df = _valid_rfq_frame()
        df["random_extra"] = "x"
        report = ingest_marketaxess_rfq(db, df)
    finally:
        db.close()
    assert report.passed is True
    assert any("unknown columns" in w for w in report.warnings)


def test_marketaxess_contract_errors_on_unknown_column_strict_mode(
    tmp_path: Path,
) -> None:
    db = Warehouse(str(tmp_path / "rfq-strict.duckdb"))
    try:
        df = _valid_rfq_frame()
        df["random_extra"] = "x"
        with pytest.raises(ValueError, match="unknown columns"):
            ingest_marketaxess_rfq(db, df, strict_unknown=True)
    finally:
        db.close()


def test_marketaxess_contract_rejects_notional_out_of_bounds(tmp_path: Path) -> None:
    db = Warehouse(str(tmp_path / "rfq-nb.duckdb"))
    try:
        df = _valid_rfq_frame()
        df.loc[0, "notional"] = 1_000_000_000.0
        report = ingest_marketaxess_rfq(db, df)
    finally:
        db.close()
    assert report.dropped_count == 1


def test_marketaxess_required_and_optional_columns_exported() -> None:
    expected_required = {
        "rfq_id",
        "timestamp",
        "cusip",
        "side",
        "notional",
        "protocol",
        "dealers_requested",
        "quotes_received",
        "status",
    }
    assert expected_required <= set(MARKETAXESS_RFQ_CONTRACT.required_columns)
    expected_optional = {"best_bid", "best_ask", "mid_price", "execution_price"}
    assert expected_optional <= set(MARKETAXESS_RFQ_CONTRACT.optional_columns)
