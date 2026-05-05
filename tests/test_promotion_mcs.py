"""Tests for the Hansen-MCS-aware promotion + release-gate paths."""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_regime_engine.forecast_compare import mcs_promotion_filter
from market_regime_engine.promotion import PromotionGate
from market_regime_engine.release_gates import evaluate_release_gate


def _candidate_row(
    brier: float = 0.10, log_loss: float = 0.30, ece: float = 0.05, *, model: str = "candidate_logistic"
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "target": "drawdown_gt_10pct",
                "horizon": "3m",
                "model": model,
                "observations": 100,
                "event_rate": 0.30,
                "brier": brier,
                "log_loss": log_loss,
                "ece": ece,
            }
        ]
    )


def _benchmark_row(brier: float = 0.18, log_loss: float = 0.42) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "target": "drawdown_gt_10pct",
                "horizon": "3m",
                "model": "expanding_event_rate",
                "observations": 100,
                "event_rate": 0.30,
                "brier": brier,
                "log_loss": log_loss,
                "ece": 0.20,
            }
        ]
    )


def test_mcs_promotion_filter_returns_surviving_models() -> None:
    rng = np.random.default_rng(0)
    losses = pd.DataFrame(
        {
            "best": rng.normal(loc=0.20, scale=0.05, size=200),
            "tied": rng.normal(loc=0.205, scale=0.05, size=200),
            "worst": rng.normal(loc=0.80, scale=0.05, size=200),
        }
    )
    surviving = mcs_promotion_filter(losses, confidence=0.10, bootstrap=200, block_size=10, seed=0)
    assert isinstance(surviving, set)
    assert "best" in surviving
    assert "worst" not in surviving


def test_mcs_promotion_filter_returns_empty_for_empty_frame() -> None:
    assert mcs_promotion_filter(pd.DataFrame()) == set()


def test_promotion_marks_mcs_evidence_absent_when_no_membership() -> None:
    out = PromotionGate().evaluate_binary(_candidate_row(), _benchmark_row())
    assert "mcs_evidence" in out.columns
    assert (out["mcs_evidence"] == "absent").all()


def test_promotion_marks_mcs_evidence_in_set_when_candidate_in_set() -> None:
    out = PromotionGate().evaluate_binary(
        _candidate_row(),
        _benchmark_row(),
        mcs_membership={"candidate_logistic", "expanding_event_rate"},
    )
    assert (out["mcs_evidence"] == "in_set").all()


def test_promotion_marks_mcs_evidence_out_of_set_when_excluded() -> None:
    out = PromotionGate().evaluate_binary(
        _candidate_row(),
        _benchmark_row(),
        mcs_membership={"expanding_event_rate"},
    )
    assert (out["mcs_evidence"] == "out_of_set").all()


def test_release_gate_blocks_when_require_mcs_membership_and_evidence_absent() -> None:
    promotion = pd.DataFrame(
        [
            {
                "target": "drawdown_gt_10pct",
                "horizon": "3m",
                "promoted": True,
                "mcs_evidence": "absent",
            }
        ]
    )
    # v1.4.1 (item F): pass profile="default" so we exercise the v1.2.1
    # looser baseline this test was originally written for. The
    # ``require_mcs_membership=True`` kwarg still wins over the profile.
    confidence = pd.DataFrame([{"date": "2024-01-31", "confidence": 0.7, "grade": "B"}])
    drift = pd.DataFrame()
    invalidation = pd.DataFrame()
    gate = evaluate_release_gate(
        confidence=confidence,
        drift=drift,
        invalidation=invalidation,
        promotion=promotion,
        require_mcs_membership=True,
        profile="default",
    )
    assert bool(gate.iloc[0]["approved"]) is False
    assert "mcs_evidence_absent" in str(gate.iloc[0]["reasons"])


def test_release_gate_passes_when_require_mcs_membership_and_evidence_in_set() -> None:
    promotion = pd.DataFrame(
        [
            {
                "target": "drawdown_gt_10pct",
                "horizon": "3m",
                "promoted": True,
                "mcs_evidence": "in_set",
            }
        ]
    )
    # v1.4.1 (item F): profile="default" preserves the v1.2.1 looser
    # confidence threshold (0.55) so confidence=0.7 still passes.
    confidence = pd.DataFrame([{"date": "2024-01-31", "confidence": 0.7, "grade": "B"}])
    gate = evaluate_release_gate(
        confidence=confidence,
        drift=pd.DataFrame(),
        invalidation=pd.DataFrame(),
        promotion=promotion,
        require_mcs_membership=True,
        profile="default",
    )
    assert bool(gate.iloc[0]["approved"]) is True


def test_release_gate_default_does_not_require_mcs_membership() -> None:
    """Backward compatibility: with the v1.2.1-baseline profile ("default")
    the gate must not block on absent MCS evidence.

    v1.4.1 (item F) flipped the no-flags default to ``production``; this
    test exercises the explicit-default opt-in path.
    """
    promotion = pd.DataFrame(
        [
            {
                "target": "drawdown_gt_10pct",
                "horizon": "3m",
                "promoted": True,
                "mcs_evidence": "absent",
            }
        ]
    )
    confidence = pd.DataFrame([{"date": "2024-01-31", "confidence": 0.7, "grade": "B"}])
    gate = evaluate_release_gate(
        confidence=confidence,
        drift=pd.DataFrame(),
        invalidation=pd.DataFrame(),
        promotion=promotion,
        profile="default",
    )
    assert bool(gate.iloc[0]["approved"]) is True
