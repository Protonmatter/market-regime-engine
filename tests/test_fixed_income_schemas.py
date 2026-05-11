# SPDX-License-Identifier: Apache-2.0
"""PR-1 acceptance tests for the FI data contracts (AGENT.md "Data contracts")."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from market_regime_engine.fixed_income.schemas import (
    CreditRegimeOutput,
    ExecutionConfidenceResponse,
    ExecutionRecommendation,
    FixedIncomeEvidencePack,
    LiquidityLabel,
    LiquidityStressOutput,
    RegimeLabel,
    liquidity_label_from_score,
    regime_label_from_score,
)


def _credit_output() -> CreditRegimeOutput:
    return CreditRegimeOutput(
        timestamp="2026-05-10T16:00:00Z",
        regime_score=42.5,
        regime_label=RegimeLabel.WATCH_TRANSITION.label,
        confidence=0.81,
        drivers=("OAS", "MOVE"),
        component_scores={"oas": 45.0, "move": 40.0},
        model_run_id="abc12345",
        release_gate=True,
        artifact_hash="sha256:deadbeef",
    )


def _liquidity_output() -> LiquidityStressOutput:
    return LiquidityStressOutput(
        timestamp="2026-05-10T16:00:00Z",
        scope_type="cusip",
        scope_id="9128283N8",
        liquidity_index=22.7,
        liquidity_label=LiquidityLabel.NORMAL.label,
        confidence=0.75,
        drivers=("RFQ fill-rate", "bid_ask_width"),
        model_run_id="run-1",
        release_gate=True,
        artifact_hash="sha256:cafebabe",
    )


def _execution_response() -> ExecutionConfidenceResponse:
    return ExecutionConfidenceResponse(
        timestamp="2026-05-10T14:05:23Z",
        cusip="9128283N8",
        side="buy",
        notional=5_000_000.0,
        protocol="Auto-X",
        confidence_score=0.74,
        expected_slippage_bps=15.0,
        confidence_interval_low=0.68,
        confidence_interval_high=0.80,
        recommended_action=ExecutionRecommendation.AUTO_X_ALLOWED.label,
        human_review_required=False,
        model_run_id="run-2",
        release_gate=True,
        artifact_hash="sha256:0xdeadbeef",
    )


def _evidence_pack() -> FixedIncomeEvidencePack:
    return FixedIncomeEvidencePack(
        model_run_id="run-3",
        component_name="credit_regime",
        model_version="0.1.0",
        timestamp="2026-05-10T16:00:00Z",
        code_sha="abcdef0",
        model_hash="sha256:m",
        input_features_hash="sha256:i",
        output_hash="sha256:o",
        data_vintages={"trace_trades": "2026-05-10T15:00:00Z"},
        validation_results={"calibration_error": 0.05},
        release_gate=True,
        random_seeds={"numpy": 7},
        python_version="3.13.4",
        lockfile_hash="sha256:lock",
        hmac_signature=None,
    )


def test_credit_regime_output_freezes() -> None:
    out = _credit_output()
    with pytest.raises(FrozenInstanceError):
        out.regime_score = 99.0  # type: ignore[misc]


def test_liquidity_stress_output_freezes() -> None:
    out = _liquidity_output()
    with pytest.raises(FrozenInstanceError):
        out.liquidity_index = 99.0  # type: ignore[misc]


def test_execution_confidence_response_freezes() -> None:
    resp = _execution_response()
    with pytest.raises(FrozenInstanceError):
        resp.confidence_score = 0.99  # type: ignore[misc]


def test_evidence_pack_freezes() -> None:
    pack = _evidence_pack()
    with pytest.raises(FrozenInstanceError):
        pack.model_run_id = "different"  # type: ignore[misc]


@pytest.mark.parametrize(
    "score,expected",
    [
        (0.0, RegimeLabel.RISK_ON_COMPRESSION),
        (10.0, RegimeLabel.RISK_ON_COMPRESSION),
        (19.99, RegimeLabel.RISK_ON_COMPRESSION),
        (20.0, RegimeLabel.NORMAL_LIQUIDITY),
        (30.0, RegimeLabel.NORMAL_LIQUIDITY),
        (40.0, RegimeLabel.WATCH_TRANSITION),
        (50.0, RegimeLabel.WATCH_TRANSITION),
        (60.0, RegimeLabel.RISK_OFF_HIGH_RISK_AVERSION),
        (70.0, RegimeLabel.RISK_OFF_HIGH_RISK_AVERSION),
        (80.0, RegimeLabel.CRISIS_SEVERE_DISLOCATION),
        (95.0, RegimeLabel.CRISIS_SEVERE_DISLOCATION),
        (100.0, RegimeLabel.CRISIS_SEVERE_DISLOCATION),
    ],
)
def test_regime_label_bucket_mapping(score: float, expected: RegimeLabel) -> None:
    """All 5 buckets plus the explicit boundary scores 20/40/60/80."""
    assert regime_label_from_score(score) is expected


@pytest.mark.parametrize(
    "score,expected",
    [
        (0.0, LiquidityLabel.NORMAL),
        (19.99, LiquidityLabel.NORMAL),
        (20.0, LiquidityLabel.MILD_STRESS),
        (40.0, LiquidityLabel.ELEVATED_STRESS),
        (60.0, LiquidityLabel.SEVERE_STRESS),
        (80.0, LiquidityLabel.CRISIS_LIQUIDITY),
        (100.0, LiquidityLabel.CRISIS_LIQUIDITY),
    ],
)
def test_liquidity_label_bucket_mapping(score: float, expected: LiquidityLabel) -> None:
    assert liquidity_label_from_score(score) is expected


def test_label_human_form_strings() -> None:
    """``.label`` strings must match AGENT.md "Recommended labels" verbatim."""
    assert RegimeLabel.RISK_ON_COMPRESSION.label == "Risk-On / Compression"
    assert RegimeLabel.NORMAL_LIQUIDITY.label == "Normal Liquidity"
    assert RegimeLabel.WATCH_TRANSITION.label == "Watch / Transition"
    assert RegimeLabel.RISK_OFF_HIGH_RISK_AVERSION.label == "Risk-Off / High Risk Aversion"
    assert RegimeLabel.CRISIS_SEVERE_DISLOCATION.label == "Crisis / Severe Dislocation"
    assert LiquidityLabel.NORMAL.label == "Normal"
    assert LiquidityLabel.MILD_STRESS.label == "Mild Stress"
    assert LiquidityLabel.ELEVATED_STRESS.label == "Elevated Stress"
    assert LiquidityLabel.SEVERE_STRESS.label == "Severe Stress"
    assert LiquidityLabel.CRISIS_LIQUIDITY.label == "Crisis Liquidity"
    assert ExecutionRecommendation.AUTO_X_ALLOWED.label == "Auto-X allowed"
    assert ExecutionRecommendation.AUTO_X_CAUTION.label == "Auto-X caution / trader confirm"
    assert ExecutionRecommendation.MANUAL_REVIEW_REQUIRED.label == "Manual review required"
    assert ExecutionRecommendation.UNAVAILABLE_GOVERNANCE.label == "Unavailable — governance gate failed"
    assert ExecutionRecommendation.UNAVAILABLE_STALE_SIGNAL.label == "Unavailable — stale signal"


def test_score_out_of_bounds_raises() -> None:
    with pytest.raises(ValueError):
        regime_label_from_score(-0.1)
    with pytest.raises(ValueError):
        regime_label_from_score(100.1)
    with pytest.raises(ValueError):
        liquidity_label_from_score(150.0)
