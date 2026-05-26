# SPDX-License-Identifier: Apache-2.0
"""Empirical execution-confidence calibration tests.

These tests pin the v1.7 calibration contract:

- training data comes only from joined prediction/outcome rows;
- future/unobserved outcomes are excluded by calibration as-of;
- fitted calibrators are persisted with audit metadata;
- scoring applies a calibrator only when its training cutoff is PIT-usable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import market_regime_engine.fixed_income  # noqa: F401 - registers FI schema
from market_regime_engine.fixed_income.execution_calibration import (
    FILL_SUCCESS_METHOD,
    FILL_SUCCESS_TARGET,
    SLIPPAGE_METHOD,
    SLIPPAGE_TARGET,
    build_execution_calibration_dataset,
    calibrate_execution_confidence_from_outcomes,
)
from market_regime_engine.fixed_income.schemas import ExecutionConfidenceRequest
from market_regime_engine.fixed_income.execution_confidence import score_execution_confidence
from market_regime_engine.storage import Warehouse


def _warehouse(tmp_path: Path) -> Warehouse:
    return Warehouse(tmp_path / "exec_calibration.sqlite", backend="sqlite")


def _prediction_row(
    request_id: str,
    ts: str,
    *,
    confidence_score: float,
    expected_slippage_bps: float = 10.0,
    release_gate: int = 1,
) -> dict:
    return {
        "request_id": request_id,
        "timestamp": ts,
        "model_run_id": f"pred-{request_id}",
        "cusip": "00206RGB6",
        "side": "buy",
        "notional": 1_000_000.0,
        "protocol": "Auto-X",
        "confidence_score": confidence_score,
        "expected_slippage_bps": expected_slippage_bps,
        "confidence_interval_low": max(0.0, confidence_score - 0.1),
        "confidence_interval_high": min(1.0, confidence_score + 0.1),
        "recommended_action": "Auto-X caution / trader confirm",
        "human_review_required": 0,
        "release_gate": release_gate,
        "artifact_hash": f"hash-{request_id}",
        "metadata_json": json.dumps({"request_id": request_id}),
    }


def _outcome_row(
    request_id: str,
    decision_ts: str,
    observed_at: str,
    *,
    success: bool,
    slippage_bps: float = 5.0,
) -> dict:
    return {
        "request_id": request_id,
        "cusip": "00206RGB6",
        "side": "buy",
        "notional": 1_000_000.0,
        "filled_quantity": 1_000_000.0 if success else 0.0,
        "execution_price": 100.0 + slippage_bps / 100.0,
        "observed_at": observed_at,
        "outcome_observation_lag": 300.0,
        "decision_timestamp": decision_ts,
        "metadata_json": json.dumps({"observed_slippage_bps": slippage_bps}),
    }


def _seed_prediction_outcome_history(wh: Warehouse) -> None:
    base = pd.Timestamp("2026-01-02T14:00:00Z")
    raw_scores = [0.15, 0.25, 0.35, 0.65, 0.78, 0.90]
    successes = [False, False, False, True, True, True]
    prediction_rows = []
    outcome_rows = []
    for idx, (score, success) in enumerate(zip(raw_scores, successes)):
        decision = base + pd.Timedelta(minutes=idx)
        observed = decision + pd.Timedelta(minutes=5)
        request_id = f"req-{idx}"
        prediction_rows.append(
            _prediction_row(
                request_id,
                decision.isoformat(),
                confidence_score=score,
                expected_slippage_bps=20.0 - 10.0 * score,
            )
        )
        outcome_rows.append(
            _outcome_row(
                request_id,
                decision.isoformat(),
                observed.isoformat(),
                success=success,
                slippage_bps=18.0 - 8.0 * score,
            )
        )

    # Future/unobserved by the calibration cutoff; must be ignored.
    prediction_rows.append(
        _prediction_row(
            "future-req",
            "2026-01-02T15:00:00+00:00",
            confidence_score=0.99,
        )
    )
    outcome_rows.append(
        _outcome_row(
            "future-req",
            "2026-01-02T15:00:00+00:00",
            "2026-01-03T15:05:00+00:00",
            success=True,
        )
    )

    # Prediction generated after its decision timestamp; must be dropped as a
    # lookahead/warehouse-corruption row.
    prediction_rows.append(
        _prediction_row(
            "bad-prediction-time",
            "2026-01-02T16:01:00+00:00",
            confidence_score=0.99,
        )
    )
    outcome_rows.append(
        _outcome_row(
            "bad-prediction-time",
            "2026-01-02T16:00:00+00:00",
            "2026-01-02T16:05:00+00:00",
            success=True,
        )
    )

    wh.write_execution_confidence_prediction(pd.DataFrame(prediction_rows))
    wh.write_execution_outcome(pd.DataFrame(outcome_rows))


def _seed_live_signals(wh: Warehouse, signal_ts: str = "2026-01-02T14:20:00+00:00") -> None:
    wh.write_credit_regime_score(
        pd.DataFrame(
            [
                {
                    "model_run_id": "credit-live",
                    "timestamp": signal_ts,
                    "regime_score": 20.0,
                    "regime_label": "Normal Liquidity",
                    "confidence": 0.95,
                    "drivers_json": "[]",
                    "component_scores_json": "{}",
                    "release_gate": 1,
                    "artifact_hash": "credit-hash",
                    "metadata_json": "{}",
                }
            ]
        )
    )
    wh.write_liquidity_stress_score(
        pd.DataFrame(
            [
                {
                    "model_run_id": "liquidity-live",
                    "scope_type": "cusip",
                    "scope_id": "00206RGB6",
                    "timestamp": signal_ts,
                    "liquidity_score": 20.0,
                    "liquidity_label": "Normal",
                    "confidence": 0.95,
                    "drivers_json": "[]",
                    "release_gate": 1,
                    "artifact_hash": "liquidity-hash",
                    "metadata_json": "{}",
                }
            ]
        )
    )


def test_build_execution_calibration_dataset_is_pit_safe(tmp_path: Path) -> None:
    wh = _warehouse(tmp_path)
    _seed_prediction_outcome_history(wh)

    dataset = build_execution_calibration_dataset(
        wh, asof="2026-01-02T14:30:00Z", fill_ratio_threshold=0.999
    )

    assert len(dataset) == 6
    assert set(dataset["request_id"]) == {f"req-{i}" for i in range(6)}
    assert dataset["fill_success"].sum() == 3.0
    assert dataset["observed_slippage_bps"].notna().all()


def test_calibrate_execution_confidence_persists_probability_and_slippage_models(tmp_path: Path) -> None:
    wh = _warehouse(tmp_path)
    _seed_prediction_outcome_history(wh)

    results = calibrate_execution_confidence_from_outcomes(
        wh,
        asof="2026-01-02T14:30:00Z",
        min_observations=6,
        run_id="exec-cal-test",
    )

    assert {r.target for r in results} == {FILL_SUCCESS_TARGET, SLIPPAGE_TARGET}
    calibration_rows = wh.read_calibration_models()
    assert set(calibration_rows["target"]) >= {FILL_SUCCESS_TARGET, SLIPPAGE_TARGET}
    fill_row = calibration_rows.loc[
        (calibration_rows["target"] == FILL_SUCCESS_TARGET)
        & (calibration_rows["method"] == FILL_SUCCESS_METHOD)
    ].iloc[0]
    fill_meta = json.loads(fill_row["metadata_json"])
    assert int(fill_row["observations"]) == 6
    assert fill_meta["training_cutoff_utc"] == "2026-01-02T14:30:00Z"
    assert fill_meta["artifact_hash"]
    assert "metrics" in fill_meta

    slippage_row = calibration_rows.loc[
        (calibration_rows["target"] == SLIPPAGE_TARGET)
        & (calibration_rows["method"] == SLIPPAGE_METHOD)
    ].iloc[0]
    assert int(slippage_row["observations"]) == 6

    model_runs = wh.read_model_runs()
    assert "exec-cal-test" in set(model_runs["run_id"])


def test_score_execution_confidence_applies_pit_usable_empirical_calibrator(tmp_path: Path) -> None:
    wh = _warehouse(tmp_path)
    _seed_prediction_outcome_history(wh)
    _seed_live_signals(wh)
    calibrate_execution_confidence_from_outcomes(
        wh,
        asof="2026-01-02T14:15:00Z",
        min_observations=6,
        run_id="exec-cal-live",
    )

    request = ExecutionConfidenceRequest(
        timestamp="2026-01-02T14:21:00Z",
        cusip="00206RGB6",
        side="buy",
        notional=1_000_000.0,
        protocol="Auto-X",
        urgency="normal",
        rating="BBB+",
    )
    response = score_execution_confidence(request, warehouse=wh)

    assert response.metadata["probability_calibration_applied"] is True
    assert response.metadata["slippage_calibration_applied"] is True
    assert response.metadata["raw_confidence_score"] != response.confidence_score
    assert response.metadata["probability_calibration_artifact_hash"]


def test_score_execution_confidence_skips_future_calibrator_for_historical_decision(tmp_path: Path) -> None:
    wh = _warehouse(tmp_path)
    _seed_prediction_outcome_history(wh)
    _seed_live_signals(wh, signal_ts="2026-01-02T14:05:00+00:00")
    calibrate_execution_confidence_from_outcomes(
        wh,
        asof="2026-01-02T14:30:00Z",
        min_observations=6,
        run_id="exec-cal-future",
    )

    request = ExecutionConfidenceRequest(
        timestamp="2026-01-02T14:06:00Z",
        cusip="00206RGB6",
        side="buy",
        notional=1_000_000.0,
        protocol="Auto-X",
        urgency="normal",
        rating="BBB+",
    )
    response = score_execution_confidence(request, warehouse=wh)

    assert response.metadata["probability_calibration_applied"] is False
    assert response.metadata["slippage_calibration_applied"] is False
    assert response.metadata["probability_calibration_skip_reason"] == "calibrator_not_pit_usable_for_decision_timestamp"
    assert response.metadata["raw_confidence_score"] == response.confidence_score
