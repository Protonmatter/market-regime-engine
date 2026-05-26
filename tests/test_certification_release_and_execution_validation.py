# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json

import pandas as pd

from market_regime_engine.fixed_income.execution_validation import (
    certification_confidence_row,
    validate_execution_confidence_realized_outcomes,
)
from market_regime_engine.release_gates import certification_profile, evaluate_release_gate


class _Warehouse:
    def __init__(self, predictions: pd.DataFrame, outcomes: pd.DataFrame) -> None:
        self._predictions = predictions
        self._outcomes = outcomes

    def read_execution_confidence_predictions(self) -> pd.DataFrame:
        return self._predictions.copy()

    def read_execution_outcomes(self) -> pd.DataFrame:
        return self._outcomes.copy()


def _synthetic_execution_frames(n: int = 90) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.Timestamp("2026-01-01T14:30:00Z")
    preds = []
    outs = []
    for i in range(n):
        regime = "calm" if i < n // 2 else "stressed"
        score = 0.25 + 0.70 * (i / max(1, n - 1))
        success = score >= 0.50
        request_id = f"req-{i:03d}"
        decision = base + pd.Timedelta(minutes=i)
        expected_slippage = 40.0 - 25.0 * score
        observed_slippage = 35.0 - 28.0 * score + (0.5 if regime == "stressed" else 0.0)
        preds.append(
            {
                "request_id": request_id,
                "timestamp": decision.isoformat(),
                "model_run_id": "run-1",
                "cusip": "000000AA0",
                "side": "buy",
                "notional": 1_000_000.0,
                "protocol": "Auto-X",
                "confidence_score": score,
                "expected_slippage_bps": expected_slippage,
                "confidence_interval_low": max(0.0, score - 0.1),
                "confidence_interval_high": min(1.0, score + 0.1),
                "recommended_action": "AUTO_X_ALLOWED" if score >= 0.8 else "AUTO_X_CAUTION",
                "human_review_required": 0,
                "release_gate": 1,
                "artifact_hash": f"hash-{i}",
                "metadata_json": json.dumps({"regime_label": regime, "liquidity_label": "Normal"}),
            }
        )
        outs.append(
            {
                "request_id": request_id,
                "cusip": "000000AA0",
                "side": "buy",
                "notional": 1_000_000.0,
                "filled_quantity": 1_000_000.0 if success else 0.0,
                "execution_price": 100.0,
                "observed_at": (decision + pd.Timedelta(minutes=10)).isoformat(),
                "outcome_observation_lag": 600.0,
                "decision_timestamp": decision.isoformat(),
                "metadata_json": json.dumps({"observed_slippage_bps": observed_slippage}),
            }
        )
    return pd.DataFrame(preds), pd.DataFrame(outs)


def _baseline_gate_inputs() -> dict:
    return {
        "drift": pd.DataFrame([{"date": "2026-01-01", "feature_name": "x", "psi": 0.0, "status": "ok"}]),
        "invalidation": pd.DataFrame(
            [{"date": "2026-01-01", "trigger": "none", "severity": "low", "status": "inactive"}]
        ),
        "promotion": pd.DataFrame([{"date": "2026-01-01", "promoted": True, "mcs_evidence": "in_set"}]),
        "coverage_report": pd.DataFrame([{"coverage": 0.95, "bucket": "all", "n": 90}]),
    }


def test_certification_profile_requires_machine_auditable_artifacts() -> None:
    assert certification_profile()["require_validation_artifacts"] is True
    conf = pd.DataFrame([{"date": "2026-01-01", "confidence": 0.99, "grade": "A"}])
    out = evaluate_release_gate(confidence=conf, profile="certification", **_baseline_gate_inputs())
    reasons = str(out.iloc[0]["reasons"])
    assert bool(out.iloc[0]["approved"]) is False
    assert "certification_missing_dsr" in reasons
    assert "certification_missing_validation_artifact_hash" in reasons
    assert "certification_missing_model_card" in reasons


def test_execution_validation_report_feeds_certification_release_gate() -> None:
    preds, outs = _synthetic_execution_frames()
    wh = _Warehouse(preds, outs)
    report = validate_execution_confidence_realized_outcomes(
        wh,
        asof="2026-01-02T00:00:00Z",
        min_observations=60,
        min_regime_sample_size=30,
        max_brier=0.30,
        max_ece=0.25,
    )
    assert report.observations == 90
    assert report.artifact_hash
    assert report.calibration_by_regime
    assert report.lift_by_decile
    assert report.tca_lift_by_regime

    conf = certification_confidence_row(
        report,
        dsr=0.75,
        pbo=0.01,
        evidence_pack_hmac="v1:abcdef",
    )
    gate = evaluate_release_gate(confidence=conf, profile="certification", **_baseline_gate_inputs())
    reasons = str(gate.iloc[0]["reasons"])
    assert "certification_missing_" not in reasons
    assert "certification_regime_sample_size_below_floor" not in reasons


def test_certification_tca_lift_requires_positive_direction() -> None:
    gate_inputs = _baseline_gate_inputs()
    # Statistically significant but adverse: high-confidence executions are worse.
    conf = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "confidence": 0.99,
                "grade": "A",
                "dsr": 0.80,
                "pbo": 0.01,
                "brier": 0.05,
                "ece": 0.01,
                "tca_lift": {"calm": {"p_value": 0.001, "effect_size": -0.9, "n": 60}},
                "pit_leakage_passed": True,
                "walk_forward_passed": True,
                "validation_artifact_hash": "sha256:abc",
                "model_card_path": "docs/method_cards/execution_confidence.md",
                "evidence_pack_hmac": "v1:hmac",
                "min_regime_sample_size": 60,
            }
        ]
    )
    out = evaluate_release_gate(confidence=conf, profile="certification", **gate_inputs)
    assert bool(out.iloc[0]["approved"]) is False
    assert "tca_lift_no_positive_significant_segment" in str(out.iloc[0]["reasons"])


def test_malformed_tca_payload_fails_closed_without_exception() -> None:
    conf = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "confidence": 0.99,
                "grade": "A",
                "dsr": 0.80,
                "pbo": 0.01,
                "brier": 0.05,
                "ece": 0.01,
                "tca_lift": {"calm": "not-a-dict"},
                "pit_leakage_passed": True,
                "walk_forward_passed": True,
                "validation_artifact_hash": "sha256:abc",
                "model_card_path": "docs/method_cards/execution_confidence.md",
                "evidence_pack_hmac": "v1:hmac",
                "min_regime_sample_size": 60,
            }
        ]
    )
    out = evaluate_release_gate(confidence=conf, profile="certification", **_baseline_gate_inputs())
    assert bool(out.iloc[0]["approved"]) is False
    assert "tca_lift_missing_or_invalid" in str(out.iloc[0]["reasons"])


def test_execution_validation_rejects_invalid_probability_scores() -> None:
    preds, outs = _synthetic_execution_frames(40)
    preds.loc[0, "confidence_score"] = 1.25
    report = validate_execution_confidence_realized_outcomes(
        _Warehouse(preds, outs),
        asof="2026-01-02T00:00:00Z",
        min_observations=30,
        min_regime_sample_size=10,
        max_brier=0.50,
        max_ece=0.50,
    )
    assert report.passed is False
    assert "invalid_probability_score_rows" in report.reasons


def test_empty_execution_validation_fails_on_nonfinite_metrics() -> None:
    preds, outs = _synthetic_execution_frames(10)
    # Outcomes are after as-of, so PIT-safe join should produce no validation rows.
    report = validate_execution_confidence_realized_outcomes(
        _Warehouse(preds, outs),
        asof="2025-12-31T00:00:00Z",
        min_observations=1,
        min_regime_sample_size=1,
    )
    assert report.passed is False
    assert "missing_or_nonfinite_validation_metrics" in report.reasons
