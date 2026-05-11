# SPDX-License-Identifier: Apache-2.0
"""FI appender tests (PR-2 task J.4 / FLAG F-18).

Mirror the shape of ``tests/test_warehouse_duckdb_appender.py`` for the
seven FI write helpers called out by the user prompt:

- write_trace_trades
- write_rfq_events
- write_credit_regime_score
- write_liquidity_stress_score
- write_execution_confidence_prediction
- write_execution_outcome
- write_evidence_pack

The contracts verified are:

1. Each write helper returns the row count it inserted.
2. The matching read helper round-trips the same number of rows.
3. ON CONFLICT semantics: re-writing the same PK upserts in place
   (no row duplication).
4. write_execution_outcome enforces the Q-2 inequality
   ``observed_at > decision_timestamp``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import market_regime_engine.fixed_income  # noqa: F401 - register FI tables
from market_regime_engine.storage import Warehouse

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _require_duckdb() -> None:
    pytest.importorskip("duckdb")


def _wh(tmp_path: Path) -> Warehouse:
    return Warehouse(str(tmp_path / "fi_appender.duckdb"), backend="duckdb")


# ---------------------------------------------------------------------------
# Per-helper appender tests
# ---------------------------------------------------------------------------


def test_write_trace_trades_round_trip(tmp_path: Path) -> None:
    _require_duckdb()
    wh = _wh(tmp_path)
    try:
        df = pd.DataFrame(
            [
                {
                    "trade_id": f"T{i}",
                    "timestamp": f"2026-01-15T12:{30 + i}:00+00:00",
                    "cusip": "037833100",
                    "price": 99.875 + 0.001 * i,
                    "yield_pct": 1.62,
                    "size": 1_000_000.0,
                    "side": "B",
                    "protocol": "tba",
                    "venue": "marketaxess",
                    "source": "trace",
                    "reported_at": f"2026-01-15T12:{30 + i}:05+00:00",
                    "metadata_json": "{}",
                }
                for i in range(5)
            ]
        )
        n = wh.write_trace_trades(df)
        assert n == 5
        readback = wh.read_trace_trades()
        assert len(readback) == 5

        # ON CONFLICT semantics: rewrite the same PK with a new price.
        df2 = df.iloc[:1].copy()
        df2.loc[:, "price"] = 100.0
        n2 = wh.write_trace_trades(df2)
        assert n2 == 1
        readback2 = wh.read_trace_trades()
        assert len(readback2) == 5
        first = readback2[readback2["trade_id"] == "T0"].iloc[0]
        assert float(first["price"]) == pytest.approx(100.0)
    finally:
        wh.close()


def test_write_rfq_events_round_trip(tmp_path: Path) -> None:
    _require_duckdb()
    wh = _wh(tmp_path)
    try:
        df = pd.DataFrame(
            [
                {
                    "rfq_id": "R1",
                    "timestamp": "2026-01-15T12:30:00+00:00",
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
        )
        assert wh.write_rfq_events(df) == 1
        readback = wh.read_rfq_events()
        assert len(readback) == 1
        assert readback["status"].iloc[0] == "filled"

        df_update = df.copy()
        df_update.loc[:, "status"] = "cancelled"
        assert wh.write_rfq_events(df_update) == 1
        readback2 = wh.read_rfq_events()
        assert len(readback2) == 1
        assert readback2["status"].iloc[0] == "cancelled"
    finally:
        wh.close()


def test_write_credit_regime_score_round_trip(tmp_path: Path) -> None:
    _require_duckdb()
    wh = _wh(tmp_path)
    try:
        df = pd.DataFrame(
            [
                {
                    "model_run_id": "run_credit_1",
                    "timestamp": "2026-01-15T12:30:00+00:00",
                    "regime_score": 42.5,
                    "regime_label": "normal_liquidity",
                    "confidence": 0.81,
                    "drivers_json": json.dumps(["ust_slope"]),
                    "component_scores_json": json.dumps({"ust_slope": 0.6}),
                    "release_gate": 1,
                    "artifact_hash": "abc",
                    "metadata_json": "{}",
                }
            ]
        )
        assert wh.write_credit_regime_score(df) == 1
        readback = wh.read_credit_regime_scores()
        assert len(readback) == 1
        assert int(readback["release_gate"].iloc[0]) == 1
    finally:
        wh.close()


def test_write_liquidity_stress_score_round_trip(tmp_path: Path) -> None:
    _require_duckdb()
    wh = _wh(tmp_path)
    try:
        df = pd.DataFrame(
            [
                {
                    "model_run_id": "run_liq_1",
                    "scope_type": "sector",
                    "scope_id": "Technology",
                    "timestamp": "2026-01-15T12:30:00+00:00",
                    "liquidity_score": 30.0,
                    "liquidity_label": "mild_stress",
                    "confidence": 0.7,
                    "drivers_json": json.dumps(["bid_ask"]),
                    "release_gate": 1,
                    "artifact_hash": "def",
                    "metadata_json": "{}",
                }
            ]
        )
        assert wh.write_liquidity_stress_score(df) == 1
        readback = wh.read_liquidity_stress_scores()
        assert len(readback) == 1
        assert readback["liquidity_label"].iloc[0] == "mild_stress"
    finally:
        wh.close()


def test_write_execution_confidence_prediction_round_trip(tmp_path: Path) -> None:
    _require_duckdb()
    wh = _wh(tmp_path)
    try:
        df = pd.DataFrame(
            [
                {
                    "request_id": "req-1",
                    "timestamp": "2026-01-15T12:30:00+00:00",
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
                    "artifact_hash": "ghi",
                    "metadata_json": "{}",
                }
            ]
        )
        assert wh.write_execution_confidence_prediction(df) == 1
        readback = wh.read_execution_confidence_predictions()
        assert len(readback) == 1
        assert readback["recommended_action"].iloc[0] == "auto_x_allowed"
    finally:
        wh.close()


def test_write_execution_outcome_round_trip(tmp_path: Path) -> None:
    _require_duckdb()
    wh = _wh(tmp_path)
    try:
        df = pd.DataFrame(
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
                    "decision_timestamp": "2026-01-15T12:30:00+00:00",
                    "metadata_json": "{}",
                }
            ]
        )
        assert wh.write_execution_outcome(df) == 1
        readback = wh.read_execution_outcomes()
        assert len(readback) == 1
    finally:
        wh.close()


def test_write_execution_outcome_rejects_invalid_observation_lag(tmp_path: Path) -> None:
    """Q-2: ``observed_at <= decision_timestamp`` is a writer-side
    constraint because DuckDB / SQLite disagree on CHECK semantics."""

    _require_duckdb()
    wh = _wh(tmp_path)
    try:
        df = pd.DataFrame(
            [
                {
                    "request_id": "req-bad",
                    "cusip": "037833100",
                    "side": "B",
                    "notional": 1_000_000.0,
                    "filled_quantity": 1_000_000.0,
                    "execution_price": 99.875,
                    # observed_at is BEFORE decision_timestamp by 60s.
                    "observed_at": "2026-01-15T12:30:00+00:00",
                    "outcome_observation_lag": -60.0,
                    "decision_timestamp": "2026-01-15T12:31:00+00:00",
                    "metadata_json": "{}",
                }
            ]
        )
        with pytest.raises(ValueError, match="observed_at > decision_timestamp"):
            wh.write_execution_outcome(df)
    finally:
        wh.close()


def test_write_evidence_pack_round_trip(tmp_path: Path) -> None:
    _require_duckdb()
    wh = _wh(tmp_path)
    try:
        df = pd.DataFrame(
            [
                {
                    "model_run_id": "run_evidence_1",
                    "request_id": "req-evidence-1",
                    "component_name": "credit_regime",
                    "model_version": "1.0",
                    "timestamp": "2026-01-15T12:30:00+00:00",
                    "code_sha": "abc123",
                    "model_hash": "mhash",
                    "input_features_hash": "ifhash",
                    "output_hash": "ohash",
                    "data_vintages_json": json.dumps({"trace_trades": "2026-01-15"}),
                    "validation_results_json": json.dumps({"dsr": 0.6}),
                    "release_gate": 1,
                    "random_seeds_json": json.dumps({"numpy": 42}),
                    "python_version": "3.13.4",
                    "lockfile_hash": "lhash",
                    "hmac_signature": None,
                    "metadata_json": "{}",
                }
            ]
        )
        assert wh.write_evidence_pack(df) == 1
        readback = wh.read_evidence_packs()
        assert len(readback) == 1
        assert readback["component_name"].iloc[0] == "credit_regime"
    finally:
        wh.close()


def test_write_evidence_pack_request_id_pk_prevents_race(tmp_path: Path) -> None:
    """PR-15: composite PK on (model_run_id, request_id) means two
    workers writing under different model_run_ids both land — but a
    re-write of the same (model_run_id, request_id) upserts in
    place."""

    _require_duckdb()
    wh = _wh(tmp_path)
    try:
        base = pd.DataFrame(
            [
                {
                    "model_run_id": "run_evidence_1",
                    "request_id": "req-1",
                    "component_name": "credit_regime",
                    "model_version": "1.0",
                    "timestamp": "2026-01-15T12:30:00+00:00",
                    "code_sha": "abc",
                    "model_hash": "mhash_v1",
                    "input_features_hash": "ifh",
                    "output_hash": "oh",
                    "data_vintages_json": "{}",
                    "validation_results_json": "{}",
                    "release_gate": 1,
                    "random_seeds_json": "{}",
                    "python_version": "3.13.4",
                    "lockfile_hash": "lh",
                    "hmac_signature": None,
                    "metadata_json": "{}",
                }
            ]
        )
        wh.write_evidence_pack(base)
        worker_b = base.copy()
        worker_b.loc[:, "model_run_id"] = "run_evidence_2"
        worker_b.loc[:, "model_hash"] = "mhash_v2"
        wh.write_evidence_pack(worker_b)
        # Both packs persist.
        readback = wh.read_evidence_packs()
        assert len(readback) == 2

        # Re-writing same (run_id, request_id) upserts in place.
        amended = base.copy()
        amended.loc[:, "model_hash"] = "mhash_v3"
        wh.write_evidence_pack(amended)
        readback2 = wh.read_evidence_packs()
        assert len(readback2) == 2
        row_a = readback2[readback2["model_run_id"] == "run_evidence_1"].iloc[0]
        assert row_a["model_hash"] == "mhash_v3"
    finally:
        wh.close()


def test_fi_appender_bulk_write_under_2s(tmp_path: Path) -> None:
    """Throughput regression guard: 5000 trace_trades writes complete
    in under 2s on the v1.4 bulk-load path. Mirrors the shape of the
    v1.4 ``test_duckdb_bulk_write_10k_rows_under_2s`` test against the
    FI table set."""

    import time

    _require_duckdb()
    wh = _wh(tmp_path)
    try:
        n = 5000
        df = pd.DataFrame(
            [
                {
                    "trade_id": f"T{i:06d}",
                    "timestamp": f"2026-01-{(i % 28) + 1:02d}T12:{(i % 60):02d}:00+00:00",
                    "cusip": f"CUS{i % 200:04d}",
                    "price": 99.0 + (i % 200) * 0.001,
                    "yield_pct": 1.5,
                    "size": 1_000_000.0,
                    "side": "B" if i % 2 == 0 else "S",
                    "protocol": "tba",
                    "venue": "marketaxess",
                    "source": "trace",
                    "reported_at": f"2026-01-{(i % 28) + 1:02d}T12:{(i % 60):02d}:30+00:00",
                    "metadata_json": "{}",
                }
                for i in range(n)
            ]
        )
        t0 = time.perf_counter()
        rows = wh.write_trace_trades(df)
        elapsed = time.perf_counter() - t0
        assert rows == n
        # Generous 2s budget: v1.4's 6600x speedup put 10k rows under
        # 2s; 5k rows on the FI table set must fit the same envelope.
        assert elapsed < 2.0, f"FI bulk-write took {elapsed:.2f}s, exceeds 2s budget"
    finally:
        wh.close()
