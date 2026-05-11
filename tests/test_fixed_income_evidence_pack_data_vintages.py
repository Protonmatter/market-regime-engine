# SPDX-License-Identifier: Apache-2.0
"""PR-7 §A.2 — data_vintages capture acceptance tests.

Per review §4.3 point 1: every evidence pack must record the latest
vintage timestamp for every FI source table at decision time so an
auditor can replay against the same vintages even if more rows have
landed since.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import market_regime_engine.fixed_income  # noqa: F401  - register FI tables
from market_regime_engine.fixed_income.evidence_pack import (
    capture_data_vintages,
    write_evidence_pack,
    build_evidence_pack,
)
from market_regime_engine.storage import Warehouse

# Source table names captured by ``capture_data_vintages`` (PR-7 §A.2).
_FI_SOURCE_TABLES = {
    "trace_trades",
    "rfq_events",
    "curve_snapshots",
    "cds_curve_snapshots",
    "bond_reference",
    "dealer_quotes",
    "dealer_response_stats",
}


@pytest.fixture()
def warehouse(tmp_path: Path) -> Warehouse:
    db_path = tmp_path / "vintage.duckdb"
    wh = Warehouse(str(db_path))
    yield wh
    wh.close()


_T1 = pd.Timestamp("2026-05-01T10:00:00")
_T2 = pd.Timestamp("2026-05-02T10:00:00")
_TC = pd.Timestamp("2026-05-03T16:00:00")
_T2_ISO = "2026-05-02T10:00:00Z"
_TC_ISO = "2026-05-03T16:00:00Z"
_T1_ISO = "2026-05-01T10:00:00Z"


def _seed_trace_trades(warehouse: Warehouse) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "trade_id": "T1",
                "timestamp": _T1,
                "cusip": "AAA111111",
                "price": 100.0,
                "yield_pct": 4.5,
                "size": 1_000_000.0,
                "side": "buy",
                "protocol": "Auto-X",
                "venue": "TRACE",
                "source": "vendor1",
                "reported_at": _T1,
                "metadata_json": "{}",
            },
            {
                "trade_id": "T2",
                "timestamp": _T2,
                "cusip": "AAA111111",
                "price": 100.5,
                "yield_pct": 4.4,
                "size": 2_000_000.0,
                "side": "sell",
                "protocol": "RFQ",
                "venue": "TRACE",
                "source": "vendor1",
                "reported_at": _T2,
                "metadata_json": "{}",
            },
        ]
    )
    warehouse.write_trace_trades(df)
    return df


def _seed_curve_snapshots(warehouse: Warehouse) -> None:
    df = pd.DataFrame(
        [
            {
                "timestamp": _TC,
                "curve_type": "UST",
                "tenor": "10Y",
                "rate": 4.20,
                "source": "fed_h15",
                "metadata_json": "{}",
            }
        ]
    )
    warehouse.write_curve_snapshots(df)


def test_capture_data_vintages_includes_all_fi_tables(warehouse: Warehouse) -> None:
    """Every recognised FI source table must appear in the result dict.

    Missing tables produce the epoch sentinel rather than dropping the
    key — the dict shape is invariant so downstream consumers can
    safely look up ``vintages["trace_trades"]`` without ``KeyError``.
    """
    vintages = capture_data_vintages(warehouse)
    assert _FI_SOURCE_TABLES <= set(vintages.keys())
    for value in vintages.values():
        assert value.endswith("Z")
    assert vintages["trace_trades"] == "1970-01-01T00:00:00Z"


def test_capture_data_vintages_returns_latest_timestamp(warehouse: Warehouse) -> None:
    _seed_trace_trades(warehouse)
    _seed_curve_snapshots(warehouse)
    vintages = capture_data_vintages(warehouse)
    assert vintages["trace_trades"] == _T2_ISO
    assert vintages["curve_snapshots"] == _TC_ISO


def test_capture_data_vintages_uses_asof_filter(warehouse: Warehouse) -> None:
    _seed_trace_trades(warehouse)
    early_asof = pd.Timestamp("2026-05-01T12:00:00")
    vintages = capture_data_vintages(warehouse, asof=early_asof)
    assert vintages["trace_trades"] == _T1_ISO
    later_asof = pd.Timestamp("2026-04-30T00:00:00")
    vintages2 = capture_data_vintages(warehouse, asof=later_asof)
    assert vintages2["trace_trades"] == "1970-01-01T00:00:00Z"


def test_evidence_pack_persists_data_vintages_dict(
    warehouse: Warehouse, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``write_evidence_pack`` round-trips the dict through DuckDB."""
    _seed_trace_trades(warehouse)
    vintages = capture_data_vintages(warehouse)
    pack = build_evidence_pack(
        model_run_id="run-vintage-1",
        component_name="credit_regime",
        model_version="0.1.0",
        code_sha=None,
        model_hash="sha256:m",
        input_features_hash="sha256:in",
        output_hash="sha256:out",
        release_gate=True,
        data_vintages=vintages,
    )
    monkeypatch.delenv("MRE_FI_HMAC_KEY_VERSIONS", raising=False)
    monkeypatch.delenv("MRE_FI_HMAC_KEY", raising=False)
    monkeypatch.delenv("MRE_FI_REQUIRE_HMAC", raising=False)
    monkeypatch.delenv("MRE_ENV", raising=False)

    write_evidence_pack(warehouse, pack, request_id="req-vintage-1")

    df = warehouse.read_evidence_packs()
    assert not df.empty
    persisted = json.loads(df.iloc[-1]["data_vintages_json"])
    assert persisted == vintages
    assert persisted["trace_trades"] == _T2_ISO


def test_capture_data_vintages_handles_missing_reader(warehouse: Warehouse) -> None:
    """A warehouse-like object that lacks a reader returns the epoch.

    Used by the in-memory mock path in unit tests; production
    Warehouse always exposes the reader.
    """

    class _PartialWh:
        def read_trace_trades(self) -> pd.DataFrame:
            return pd.DataFrame()

        def read_rfq_events(self) -> pd.DataFrame:
            return pd.DataFrame()

    vintages = capture_data_vintages(_PartialWh())
    assert vintages["trace_trades"] == "1970-01-01T00:00:00Z"
    assert vintages["rfq_events"] == "1970-01-01T00:00:00Z"
    assert vintages["bond_reference"] == "1970-01-01T00:00:00Z"
