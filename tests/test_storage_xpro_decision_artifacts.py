# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json

import pandas as pd

import market_regime_engine.fixed_income  # noqa: F401
from market_regime_engine.storage import Warehouse


def test_xpro_decision_artifact_storage_roundtrip(tmp_path) -> None:
    wh = Warehouse(tmp_path / "xpro.duckdb")
    try:
        payload = {
            "artifact_version": "xpro_decision_artifact_v1",
            "decision_id": "dec-1",
            "decision": {"recommended_protocol": "RFQ", "release_gate": True},
            "evidence": {"artifact_hash": "sha256:abc"},
        }
        rows = pd.DataFrame(
            [
                {
                    "decision_id": "dec-1",
                    "request_id": "req-1",
                    "timestamp": "2026-05-26T12:31:00Z",
                    "model_run_id": "run-1",
                    "recommended_protocol": "RFQ",
                    "release_gate": 1,
                    "artifact_hash": "sha256:abc",
                    "hmac_signature": "v1:abc",
                    "payload_json": json.dumps(payload, sort_keys=True),
                    "metadata_json": "{}",
                }
            ]
        )
        assert wh.write_xpro_decision_artifact(rows) == 1
        latest = wh.latest_xpro_decision_artifact("dec-1")
        assert latest is not None
        assert latest.iloc[0]["recommended_protocol"] == "RFQ"
        assert "xpro_decision_artifacts" in wh.table_names()
    finally:
        wh.close()


def test_execution_confidence_prediction_quantized_columns_exist(tmp_path) -> None:
    wh = Warehouse(tmp_path / "quantized_cols.duckdb")
    try:
        cols = wh._backend.column_names("execution_confidence_predictions")
        assert {
            "notional_cents",
            "confidence_score_ppm",
            "expected_slippage_bps_q4",
            "confidence_interval_low_ppm",
            "confidence_interval_high_ppm",
        } <= cols
    finally:
        wh.close()


def test_execution_confidence_prediction_quantized_backfill_populates_legacy_nulls(tmp_path) -> None:
    db = tmp_path / "quantized_backfill.duckdb"
    wh = Warehouse(db)
    try:
        wh._backend.execute(
            """
            INSERT INTO execution_confidence_predictions (
                request_id,
                timestamp,
                model_run_id,
                cusip,
                side,
                notional,
                protocol,
                confidence_score,
                expected_slippage_bps,
                confidence_interval_low,
                confidence_interval_high,
                recommended_action,
                human_review_required,
                release_gate,
                artifact_hash,
                metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-req",
                "2026-05-26T12:31:00Z",
                "legacy-run",
                "123456AB7",
                "buy",
                1_000_000.0,
                "RFQ",
                0.61244,
                12.125,
                0.51244,
                0.71244,
                "Auto-X caution / trader confirm",
                0,
                1,
                "sha256:legacy",
                "{}",
            ),
        )
        wh._backend.commit()
        before = wh.read_execution_confidence_predictions().iloc[0]
        assert pd.isna(before["notional_cents"])
        assert wh.backfill_execution_confidence_prediction_quantized_columns() == 1
        after = wh.read_execution_confidence_predictions().iloc[0]
        assert int(after["notional_cents"]) == 100000000
        assert int(after["confidence_score_ppm"]) == 612440
        assert int(after["expected_slippage_bps_q4"]) == 121250
        assert int(after["confidence_interval_low_ppm"]) == 512440
        assert int(after["confidence_interval_high_ppm"]) == 712440
        assert wh.backfill_execution_confidence_prediction_quantized_columns() == 0
    finally:
        wh.close()


def test_execution_confidence_prediction_quantized_backfill_runs_on_init(tmp_path) -> None:
    db = tmp_path / "quantized_init_backfill.duckdb"
    wh = Warehouse(db)
    try:
        wh._backend.execute(
            """
            INSERT INTO execution_confidence_predictions (
                request_id,
                timestamp,
                model_run_id,
                cusip,
                side,
                notional,
                protocol,
                confidence_score,
                expected_slippage_bps,
                confidence_interval_low,
                confidence_interval_high,
                recommended_action,
                human_review_required,
                release_gate,
                artifact_hash,
                metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-init",
                "2026-05-26T12:32:00Z",
                "legacy-run",
                "123456AB7",
                "buy",
                2_000_000.0,
                "RFQ",
                0.5,
                1.25,
                0.4,
                0.6,
                "Auto-X caution / trader confirm",
                0,
                1,
                "sha256:legacy-init",
                "{}",
            ),
        )
        wh._backend.commit()
    finally:
        wh.close()

    reopened = Warehouse(db)
    try:
        row = reopened.read_execution_confidence_predictions().iloc[0]
        assert int(row["notional_cents"]) == 200000000
        assert int(row["confidence_score_ppm"]) == 500000
        assert int(row["expected_slippage_bps_q4"]) == 12500
    finally:
        reopened.close()
