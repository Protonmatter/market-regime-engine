# SPDX-License-Identifier: Apache-2.0
"""PR-6 §C — outcome observation lag (REVIEW.md §3.4 Q-2 / §3.6 PR-10).

Pins the strict ``observed_at > decision_timestamp`` inequality at three
boundaries:

1. ``Warehouse.write_execution_outcome`` (storage writer).
2. :func:`fixed_income.tca_outcome_lag.assert_outcome_after_decision`
   (canonical guard).
3. :func:`fixed_income.tca_segmentation.compute_tca_metrics_for_outcome`
   (PR-6 task A boundary).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import market_regime_engine.fixed_income  # noqa: F401 — register FI schema
from market_regime_engine.fixed_income import ExecutionConfidenceRequest
from market_regime_engine.fixed_income.pit_guard import PitViolationError
from market_regime_engine.fixed_income.tca_outcome_lag import (
    assert_outcome_after_decision,
)
from market_regime_engine.storage import Warehouse


def test_execution_outcomes_observe_after_decision_timestamp_strict(
    tmp_path: Path,
) -> None:
    """``observed_at > decision_timestamp`` is strict (one nanosecond later passes)."""
    wh = Warehouse(tmp_path / "lag.duckdb")
    df = pd.DataFrame(
        [
            {
                "request_id": "req-1",
                "cusip": "00206RGB6",
                "side": "buy",
                "notional": 1_000_000.0,
                "filled_quantity": 1_000_000.0,
                "execution_price": 100.0,
                "observed_at": "2026-05-01T16:00:00.000000001Z",
                "outcome_observation_lag": 0.000000001,
                "decision_timestamp": "2026-05-01T16:00:00Z",
                "metadata_json": "{}",
            }
        ]
    )
    assert wh.write_execution_outcome(df) == 1


def test_write_execution_outcome_rejects_same_timestamp(tmp_path: Path) -> None:
    """A trade fill that reports at the same nanosecond as the decision must be rejected."""
    wh = Warehouse(tmp_path / "lag.duckdb")
    df = pd.DataFrame(
        [
            {
                "request_id": "req-same",
                "cusip": "00206RGB6",
                "side": "buy",
                "notional": 1_000_000.0,
                "filled_quantity": 1_000_000.0,
                "execution_price": 100.0,
                "observed_at": "2026-05-01T16:00:00Z",
                "outcome_observation_lag": 0.0,
                "decision_timestamp": "2026-05-01T16:00:00Z",
                "metadata_json": "{}",
            }
        ]
    )
    with pytest.raises(ValueError, match="strictly greater|strict|must be strictly|observed_at > decision_timestamp"):
        wh.write_execution_outcome(df)


def test_write_execution_outcome_rejects_observed_before_decision(
    tmp_path: Path,
) -> None:
    """observed_at < decision_timestamp is also rejected (PIT violation)."""
    wh = Warehouse(tmp_path / "lag.duckdb")
    df = pd.DataFrame(
        [
            {
                "request_id": "req-back",
                "cusip": "00206RGB6",
                "side": "buy",
                "notional": 1_000_000.0,
                "filled_quantity": 1_000_000.0,
                "execution_price": 100.0,
                "observed_at": "2026-05-01T15:00:00Z",
                "outcome_observation_lag": -3600.0,
                "decision_timestamp": "2026-05-01T16:00:00Z",
                "metadata_json": "{}",
            }
        ]
    )
    with pytest.raises(ValueError):
        wh.write_execution_outcome(df)


def test_compute_tca_metrics_rejects_outcome_before_decision() -> None:
    """compute_tca_metrics_for_outcome enforces the same strict inequality."""
    from market_regime_engine.fixed_income.schemas import (
        ExecutionConfidenceResponse,
    )
    from market_regime_engine.fixed_income.tca_segmentation import (
        compute_tca_metrics_for_outcome,
    )

    request = ExecutionConfidenceRequest(
        timestamp="2026-05-01T16:00:00Z",
        cusip="00206RGB6",
        side="buy",
        notional=1_000_000.0,
        protocol="Auto-X",
    )
    response = ExecutionConfidenceResponse(
        timestamp="2026-05-01T16:00:00Z",
        cusip="00206RGB6",
        side="buy",
        notional=1_000_000.0,
        protocol="Auto-X",
        confidence_score=0.7,
        expected_slippage_bps=10.0,
        confidence_interval_low=0.6,
        confidence_interval_high=0.8,
        recommended_action="Auto-X allowed",
        human_review_required=False,
        model_run_id="m-1",
        release_gate=True,
        artifact_hash="sha256:test",
    )
    outcome = {
        # observed_at == request.timestamp → strict violation
        "observed_at": "2026-05-01T16:00:00Z",
        "arrival_price": 100.0,
        "execution_price": 100.0,
    }
    with pytest.raises(PitViolationError):
        compute_tca_metrics_for_outcome(
            request, response, outcome, warehouse=None
        )


def test_assert_outcome_after_decision_returns_utc_normalised() -> None:
    decision_utc, observed_utc = assert_outcome_after_decision(
        decision_timestamp="2026-05-01T16:00:00-04:00",  # ET
        observed_at="2026-05-01T20:01:00Z",  # 1 minute later in UTC
    )
    assert str(decision_utc.tzinfo) == "UTC"
    assert str(observed_utc.tzinfo) == "UTC"
    assert observed_utc > decision_utc


def test_assert_outcome_after_decision_strict_equality_raises() -> None:
    with pytest.raises(PitViolationError):
        assert_outcome_after_decision(
            decision_timestamp="2026-05-01T16:00:00Z",
            observed_at="2026-05-01T16:00:00Z",
        )


def test_assert_outcome_after_decision_naive_input_normalises_to_utc() -> None:
    """Naive timestamps coerce to UTC; the helper does not silently fail open."""
    # decision is naive (treated as UTC); observed is 1s later (also UTC) — passes.
    decision_utc, observed_utc = assert_outcome_after_decision(
        decision_timestamp="2026-05-01T16:00:00",
        observed_at="2026-05-01T16:00:01",
    )
    assert (observed_utc - decision_utc).total_seconds() == pytest.approx(1.0)
