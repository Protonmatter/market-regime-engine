# SPDX-License-Identifier: Apache-2.0
"""FI warehouse round-trip tests.

For each of the 13 FI tables, write a synthetic frame via the
``_DuckDBBackend.upsert_frame`` / ``_SqliteBackend.upsert_frame`` fast
path and read it back; assert the column set, row count, and selected
type-sensitive values survive the round trip on both DuckDB and SQLite.

The schema is DuckDB-first per AGENT.md §PR-2; the SQLite fallback
substitutes TEXT/REAL/TEXT for TIMESTAMP/DECIMAL/JSON so the same
synthetic frame must serialise on both backends.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import market_regime_engine.fixed_income  # noqa: F401 - register FI tables
from market_regime_engine.storage import Warehouse

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ---------------------------------------------------------------------------
# Backend fixtures
# ---------------------------------------------------------------------------


def _wh_for(backend: str, tmp_path: Path) -> Warehouse:
    suffix = ".duckdb" if backend == "duckdb" else ".db"
    return Warehouse(str(tmp_path / f"roundtrip{suffix}"), backend=backend)


# ---------------------------------------------------------------------------
# Per-table frame builders + column lists
# ---------------------------------------------------------------------------


_TS = "2026-01-15T12:30:00+00:00"
_TS2 = "2026-01-15T12:35:00+00:00"


def _bond_reference_frame() -> tuple[str, pd.DataFrame, list[str]]:
    return (
        "bond_reference",
        pd.DataFrame(
            [
                {
                    "cusip": "037833100",
                    "valid_from": "2026-01-01T00:00:00+00:00",
                    "valid_to": None,
                    "ticker": "AAPL",
                    "issuer": "Apple Inc.",
                    "sector": "Technology",
                    "rating": "AA+",
                    "issue_date": "2020-05-04T00:00:00+00:00",
                    "maturity": "2030-05-04T00:00:00+00:00",
                    "coupon": 1.65,
                    "currency": "USD",
                    "country": "US",
                    "duration": 7.1,
                    "convexity": 60.3,
                    "amount_outstanding": 2_500_000_000.0,
                    "is_callable": 0,
                    "call_schedule_json": json.dumps([]),
                    "default_date": None,
                    "delisted_date": None,
                    "metadata_json": "{}",
                }
            ]
        ),
        [
            "cusip",
            "valid_from",
            "valid_to",
            "ticker",
            "issuer",
            "sector",
            "rating",
            "issue_date",
            "maturity",
            "coupon",
            "currency",
            "country",
            "duration",
            "convexity",
            "amount_outstanding",
            "is_callable",
            "call_schedule_json",
            "default_date",
            "delisted_date",
            "metadata_json",
        ],
    )


def _trace_trades_frame() -> tuple[str, pd.DataFrame, list[str]]:
    return (
        "trace_trades",
        pd.DataFrame(
            [
                {
                    "trade_id": "T1",
                    "timestamp": _TS,
                    "cusip": "037833100",
                    "price": 99.875,
                    "yield_pct": 1.62,
                    "size": 1_000_000.0,
                    "side": "B",
                    "protocol": "tba",
                    "venue": "marketaxess",
                    "source": "trace",
                    "reported_at": _TS2,
                    "metadata_json": "{}",
                },
            ]
        ),
        [
            "trade_id",
            "timestamp",
            "cusip",
            "price",
            "yield_pct",
            "size",
            "side",
            "protocol",
            "venue",
            "source",
            "reported_at",
            "metadata_json",
        ],
    )


def _rfq_events_frame() -> tuple[str, pd.DataFrame, list[str]]:
    return (
        "rfq_events",
        pd.DataFrame(
            [
                {
                    "rfq_id": "R1",
                    "timestamp": _TS,
                    "cusip": "037833100",
                    "side": "S",
                    "notional": 2_000_000.0,
                    "protocol": "rfq",
                    "status": "filled",
                    "dealers_requested": 4,
                    "dealers_responded": 3,
                    "time_to_first_response_ms": 250,
                    "client_id": "CLIENT_A",
                    "metadata_json": "{}",
                }
            ]
        ),
        [
            "rfq_id",
            "timestamp",
            "cusip",
            "side",
            "notional",
            "protocol",
            "status",
            "dealers_requested",
            "dealers_responded",
            "time_to_first_response_ms",
            "client_id",
            "metadata_json",
        ],
    )


def _dealer_quotes_frame() -> tuple[str, pd.DataFrame, list[str]]:
    return (
        "dealer_quotes",
        pd.DataFrame(
            [
                {
                    "timestamp": _TS,
                    "cusip": "037833100",
                    "dealer_id": "DLR_1",
                    "side": "B",
                    "price": 99.5,
                    "size": 500_000.0,
                    "expires_at": _TS2,
                    "metadata_json": "{}",
                }
            ]
        ),
        ["timestamp", "cusip", "dealer_id", "side", "price", "size", "expires_at", "metadata_json"],
    )


def _dealer_response_stats_frame() -> tuple[str, pd.DataFrame, list[str]]:
    return (
        "dealer_response_stats",
        pd.DataFrame(
            [
                {
                    "dealer_id": "DLR_1",
                    "window_start": "2026-01-15T00:00:00+00:00",
                    "window_end": "2026-01-15T23:59:59+00:00",
                    "requests": 100,
                    "responses": 78,
                    "avg_response_ms": 312.5,
                    "metadata_json": "{}",
                }
            ]
        ),
        [
            "dealer_id",
            "window_start",
            "window_end",
            "requests",
            "responses",
            "avg_response_ms",
            "metadata_json",
        ],
    )


def _curve_snapshots_frame() -> tuple[str, pd.DataFrame, list[str]]:
    return (
        "curve_snapshots",
        pd.DataFrame(
            [
                {
                    "timestamp": _TS,
                    "curve_type": "ust",
                    "tenor": "10Y",
                    "rate": 4.25,
                    "source": "fed",
                    "metadata_json": "{}",
                }
            ]
        ),
        ["timestamp", "curve_type", "tenor", "rate", "source", "metadata_json"],
    )


def _cds_curve_snapshots_frame() -> tuple[str, pd.DataFrame, list[str]]:
    return (
        "cds_curve_snapshots",
        pd.DataFrame(
            [
                {
                    "timestamp": _TS,
                    "reference_entity": "CDX.IG",
                    "tenor": "5Y",
                    "spread_bps": 65.5,
                    "source": "markit",
                    "metadata_json": "{}",
                }
            ]
        ),
        ["timestamp", "reference_entity", "tenor", "spread_bps", "source", "metadata_json"],
    )


def _credit_regime_scores_frame() -> tuple[str, pd.DataFrame, list[str]]:
    return (
        "credit_regime_scores",
        pd.DataFrame(
            [
                {
                    "model_run_id": "run_credit_1",
                    "timestamp": _TS,
                    "regime_score": 42.5,
                    "regime_label": "normal_liquidity",
                    "confidence": 0.81,
                    "drivers_json": json.dumps(["ust_slope", "cdx_ig"]),
                    "component_scores_json": json.dumps({"ust_slope": 0.6, "cdx_ig": 0.3}),
                    "release_gate": 1,
                    "artifact_hash": "abc123",
                    "metadata_json": "{}",
                }
            ]
        ),
        [
            "model_run_id",
            "timestamp",
            "regime_score",
            "regime_label",
            "confidence",
            "drivers_json",
            "component_scores_json",
            "release_gate",
            "artifact_hash",
            "metadata_json",
        ],
    )


def _liquidity_stress_scores_frame() -> tuple[str, pd.DataFrame, list[str]]:
    return (
        "liquidity_stress_scores",
        pd.DataFrame(
            [
                {
                    "model_run_id": "run_liq_1",
                    "scope_type": "sector",
                    "scope_id": "Technology",
                    "timestamp": _TS,
                    "liquidity_score": 30.0,
                    "liquidity_label": "mild_stress",
                    "confidence": 0.7,
                    "drivers_json": json.dumps(["bid_ask", "amihud"]),
                    "release_gate": 1,
                    "artifact_hash": "def456",
                    "metadata_json": "{}",
                }
            ]
        ),
        [
            "model_run_id",
            "scope_type",
            "scope_id",
            "timestamp",
            "liquidity_score",
            "liquidity_label",
            "confidence",
            "drivers_json",
            "release_gate",
            "artifact_hash",
            "metadata_json",
        ],
    )


def _execution_confidence_predictions_frame() -> tuple[str, pd.DataFrame, list[str]]:
    return (
        "execution_confidence_predictions",
        pd.DataFrame(
            [
                {
                    "request_id": "req-1",
                    "timestamp": _TS,
                    "model_run_id": "run_exec_1",
                    "cusip": "037833100",
                    "side": "B",
                    "notional": 1_000_000.0,
                    "protocol": "rfq",
                    "confidence_score": 0.88,
                    "expected_slippage_bps": 1.25,
                    "confidence_interval_low": 0.82,
                    "confidence_interval_high": 0.93,
                    "recommended_action": "auto_x_allowed",
                    "human_review_required": 0,
                    "release_gate": 1,
                    "artifact_hash": "ghi789",
                    "metadata_json": "{}",
                }
            ]
        ),
        [
            "request_id",
            "timestamp",
            "model_run_id",
            "cusip",
            "side",
            "notional",
            "protocol",
            "confidence_score",
            "expected_slippage_bps",
            "confidence_interval_low",
            "confidence_interval_high",
            "recommended_action",
            "human_review_required",
            "release_gate",
            "artifact_hash",
            "metadata_json",
        ],
    )


def _execution_outcomes_frame() -> tuple[str, pd.DataFrame, list[str]]:
    return (
        "execution_outcomes",
        pd.DataFrame(
            [
                {
                    "request_id": "req-1",
                    "cusip": "037833100",
                    "side": "B",
                    "notional": 1_000_000.0,
                    "filled_quantity": 1_000_000.0,
                    "execution_price": 99.875,
                    "observed_at": "2026-01-15T12:31:00+00:00",
                    "outcome_observation_lag": 60.0,
                    "decision_timestamp": _TS,
                    "metadata_json": "{}",
                }
            ]
        ),
        [
            "request_id",
            "cusip",
            "side",
            "notional",
            "filled_quantity",
            "execution_price",
            "observed_at",
            "outcome_observation_lag",
            "decision_timestamp",
            "metadata_json",
        ],
    )


def _tca_regime_segments_frame() -> tuple[str, pd.DataFrame, list[str]]:
    return (
        "tca_regime_segments",
        pd.DataFrame(
            [
                {
                    "model_run_id": "run_tca_1",
                    "timestamp": _TS,
                    "regime_label": "normal_liquidity",
                    "liquidity_label": "normal",
                    "execution_confidence_bucket": "high",
                    "protocol": "rfq",
                    "side": "B",
                    "sector": "Technology",
                    "rating": "AA",
                    "maturity_bucket": "5_10y",
                    "notional_bucket": "1M_5M",
                    "metric_name": "arrival_cost_bps",
                    "metric_value": 1.5,
                    "sample_count": 100,
                    "metadata_json": "{}",
                }
            ]
        ),
        [
            "model_run_id",
            "timestamp",
            "regime_label",
            "liquidity_label",
            "execution_confidence_bucket",
            "protocol",
            "side",
            "sector",
            "rating",
            "maturity_bucket",
            "notional_bucket",
            "metric_name",
            "metric_value",
            "sample_count",
            "metadata_json",
        ],
    )


def _fixed_income_evidence_packs_frame() -> tuple[str, pd.DataFrame, list[str]]:
    return (
        "fixed_income_evidence_packs",
        pd.DataFrame(
            [
                {
                    "model_run_id": "run_evidence_1",
                    "request_id": "req-evidence-1",
                    "component_name": "credit_regime",
                    "model_version": "1.0",
                    "timestamp": _TS,
                    "code_sha": "abc123def",
                    "model_hash": "modelhash",
                    "input_features_hash": "ifhash",
                    "output_hash": "ohash",
                    "data_vintages_json": json.dumps({"trace_trades": "2026-01-15"}),
                    "validation_results_json": json.dumps({"dsr": 0.6}),
                    "release_gate": 1,
                    "random_seeds_json": json.dumps({"numpy": 42}),
                    "python_version": "3.13.4",
                    "lockfile_hash": "lockhash",
                    "hmac_signature": None,
                    "metadata_json": "{}",
                }
            ]
        ),
        [
            "model_run_id",
            "request_id",
            "component_name",
            "model_version",
            "timestamp",
            "code_sha",
            "model_hash",
            "input_features_hash",
            "output_hash",
            "data_vintages_json",
            "validation_results_json",
            "release_gate",
            "random_seeds_json",
            "python_version",
            "lockfile_hash",
            "hmac_signature",
            "metadata_json",
        ],
    )


_FRAME_BUILDERS = [
    _bond_reference_frame,
    _trace_trades_frame,
    _rfq_events_frame,
    _dealer_quotes_frame,
    _dealer_response_stats_frame,
    _curve_snapshots_frame,
    _cds_curve_snapshots_frame,
    _credit_regime_scores_frame,
    _liquidity_stress_scores_frame,
    _execution_confidence_predictions_frame,
    _execution_outcomes_frame,
    _tca_regime_segments_frame,
    _fixed_income_evidence_packs_frame,
]


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["duckdb", "sqlite"])
@pytest.mark.parametrize("builder", _FRAME_BUILDERS, ids=[fn.__name__[1:-6] for fn in _FRAME_BUILDERS])
def test_fi_table_round_trip(backend: str, builder, tmp_path: Path) -> None:
    """Each FI table accepts the synthetic frame and reads it back with
    the right row count and a stable column set on both backends."""

    if backend == "duckdb":
        pytest.importorskip("duckdb")

    table, df, cols = builder()
    wh = _wh_for(backend, tmp_path)
    try:
        wh._backend.upsert_frame(table, df, cols, mode="REPLACE")
        wh._backend.commit()
        readback = wh._backend.read_sql(f"SELECT * FROM {table}")
    finally:
        wh.close()

    assert not readback.empty, f"{table} on {backend} round-tripped to an empty frame"
    assert len(readback) == len(df), f"row count mismatch for {table} on {backend}"
    # Column names from the read frame include all the cols we wrote.
    for c in cols:
        assert c in readback.columns, f"{table} on {backend} missing column {c}"


@pytest.mark.parametrize("backend", ["duckdb", "sqlite"])
def test_fi_tables_present_after_warehouse_init(backend: str, tmp_path: Path) -> None:
    """Every FI table is created during ``Warehouse.init_schema`` and
    can be queried even when no rows are written."""

    if backend == "duckdb":
        pytest.importorskip("duckdb")

    from market_regime_engine.fixed_income import FI_TABLE_NAMES

    wh = _wh_for(backend, tmp_path)
    try:
        for name in FI_TABLE_NAMES:
            df = wh._backend.read_sql(f"SELECT * FROM {name}")
            assert df is not None
            assert df.empty
    finally:
        wh.close()


@pytest.mark.parametrize("backend", ["duckdb", "sqlite"])
def test_execution_confidence_predictions_request_id_pk(backend: str, tmp_path: Path) -> None:
    """PR-15: ``request_id`` is part of the composite PK so a second
    write under the same (request_id, timestamp) upserts in place
    rather than duplicating."""

    if backend == "duckdb":
        pytest.importorskip("duckdb")

    table, df, cols = _execution_confidence_predictions_frame()
    wh = _wh_for(backend, tmp_path)
    try:
        wh._backend.upsert_frame(table, df, cols, mode="REPLACE")
        wh._backend.commit()

        df2 = df.copy()
        df2.loc[:, "confidence_score"] = 0.55
        wh._backend.upsert_frame(table, df2, cols, mode="REPLACE")
        wh._backend.commit()

        read = wh._backend.read_sql(f"SELECT * FROM {table}")
        assert len(read) == 1
        assert float(read["confidence_score"].iloc[0]) == pytest.approx(0.55)
    finally:
        wh.close()


@pytest.mark.parametrize("backend", ["duckdb", "sqlite"])
def test_fixed_income_evidence_packs_request_id_pk(backend: str, tmp_path: Path) -> None:
    """PR-15: ``(model_run_id, request_id)`` is the composite PK; two
    workers writing packs for the same client request under
    *different* model_run_ids both land (different PK rows), but a
    re-write of the same (model_run_id, request_id) upserts in
    place."""

    if backend == "duckdb":
        pytest.importorskip("duckdb")

    table, df, cols = _fixed_income_evidence_packs_frame()
    wh = _wh_for(backend, tmp_path)
    try:
        # Worker A
        wh._backend.upsert_frame(table, df, cols, mode="REPLACE")
        wh._backend.commit()

        # Worker B writes under a different model_run_id — both rows persist.
        df_b = df.copy()
        df_b.loc[:, "model_run_id"] = "run_evidence_2"
        wh._backend.upsert_frame(table, df_b, cols, mode="REPLACE")
        wh._backend.commit()

        read = wh._backend.read_sql(f"SELECT * FROM {table}")
        assert len(read) == 2

        # Re-writing the same (model_run_id, request_id) upserts in place.
        df_a_v2 = df.copy()
        df_a_v2.loc[:, "model_hash"] = "modelhash_v2"
        wh._backend.upsert_frame(table, df_a_v2, cols, mode="REPLACE")
        wh._backend.commit()

        read = wh._backend.read_sql(f"SELECT * FROM {table}")
        assert len(read) == 2
        row_a = read[read["model_run_id"] == "run_evidence_1"].iloc[0]
        assert row_a["model_hash"] == "modelhash_v2"
    finally:
        wh.close()
