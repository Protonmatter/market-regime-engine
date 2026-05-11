# SPDX-License-Identifier: Apache-2.0
"""PR-6 §C.3 / §3.6 PR-10 — label-construction guard.

Pins :func:`compute_execution_success_label` to the strict outcome-lag
rail and verifies the threshold / unobservable semantics documented in
INSTRUCTIONS.md §6.4.
"""

from __future__ import annotations

import pytest

from market_regime_engine.fixed_income import ExecutionConfidenceRequest
from market_regime_engine.fixed_income.pit_guard import PitViolationError
from market_regime_engine.fixed_income.tca_outcome_lag import (
    EXECUTION_SUCCESS_DEFAULT_THRESHOLD_BPS,
    SUCCESS_THRESHOLD_ENV,
    compute_execution_success_label,
)


def _request(ts: str = "2026-05-01T16:00:00Z") -> ExecutionConfidenceRequest:
    return ExecutionConfidenceRequest(
        timestamp=ts,
        cusip="00206RGB6",
        side="buy",
        notional=1_000_000.0,
        protocol="Auto-X",
    )


def test_compute_execution_success_label_returns_none_for_unobservable_outcome() -> None:
    request = _request()
    # ``observed_at`` absent → unobservable, return None (not False, not raise).
    label = compute_execution_success_label(request, outcome={})
    assert label is None


def test_compute_execution_success_label_returns_none_when_prices_missing() -> None:
    request = _request()
    label = compute_execution_success_label(
        request,
        outcome={"observed_at": "2026-05-01T16:30:00Z"},
    )
    assert label is None


def test_compute_execution_success_label_uses_strict_inequality() -> None:
    """observed_at == decision_timestamp → PitViolationError (strict)."""
    request = _request()
    with pytest.raises(PitViolationError):
        compute_execution_success_label(
            request,
            outcome={
                "observed_at": "2026-05-01T16:00:00Z",
                "arrival_price": 100.0,
                "execution_price": 100.0,
            },
        )


def test_compute_execution_success_label_returns_true_within_threshold() -> None:
    request = _request()
    # 5 bps slippage; default threshold is 25 bps.
    label = compute_execution_success_label(
        request,
        outcome={
            "observed_at": "2026-05-01T16:30:00Z",
            "arrival_price": 100.0,
            "execution_price": 100.05,  # 5 bps cost
        },
    )
    assert label is True


def test_compute_execution_success_label_returns_false_outside_threshold() -> None:
    request = _request()
    label = compute_execution_success_label(
        request,
        outcome={
            "observed_at": "2026-05-01T16:30:00Z",
            "arrival_price": 100.0,
            "execution_price": 100.50,  # 50 bps cost > 25 bps threshold
        },
    )
    assert label is False


def test_compute_execution_success_label_explicit_threshold_override() -> None:
    request = _request()
    label = compute_execution_success_label(
        request,
        outcome={
            "observed_at": "2026-05-01T16:30:00Z",
            "arrival_price": 100.0,
            "execution_price": 100.05,  # 5 bps cost
        },
        success_threshold_bps=3.0,  # tighter threshold
    )
    assert label is False


def test_compute_execution_success_label_env_override(monkeypatch) -> None:
    """MRE_FI_TCA_SUCCESS_THRESHOLD_BPS env overrides the default."""
    request = _request()
    monkeypatch.setenv(SUCCESS_THRESHOLD_ENV, "3.0")
    label = compute_execution_success_label(
        request,
        outcome={
            "observed_at": "2026-05-01T16:30:00Z",
            "arrival_price": 100.0,
            "execution_price": 100.05,  # 5 bps
        },
    )
    assert label is False


def test_compute_execution_success_label_zero_arrival_price_returns_none() -> None:
    """Division-by-zero guard: a zero arrival price yields None, not raise."""
    request = _request()
    label = compute_execution_success_label(
        request,
        outcome={
            "observed_at": "2026-05-01T16:30:00Z",
            "arrival_price": 0.0,
            "execution_price": 100.0,
        },
    )
    assert label is None


def test_compute_execution_success_default_threshold_matches_spec() -> None:
    """Pin the AGENT.md / INSTRUCTIONS.md §6.4 default threshold."""
    assert EXECUTION_SUCCESS_DEFAULT_THRESHOLD_BPS == 25.0
