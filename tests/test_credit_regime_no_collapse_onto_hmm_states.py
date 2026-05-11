# SPDX-License-Identifier: Apache-2.0
"""F-9 guard: FI credit regime labels are domain-specific (not HMM state names)."""

from __future__ import annotations

from market_regime_engine.fixed_income.schemas import RegimeLabel
from market_regime_engine.hmm import REGIME_STATES


def test_regime_label_distinct_from_hmm_regime_states() -> None:
    """Per REVIEW.md §3.3 F-9 / plan §3: the FI credit-regime label states
    are domain-specific (Risk-On / Normal Liquidity / Watch / Risk-Off /
    Crisis) and must NOT collapse onto ``hmm.REGIME_STATES`` strings.
    """
    hmm_states = set(REGIME_STATES)
    fi_states = {label.value for label in RegimeLabel}
    fi_labels = {label.label for label in RegimeLabel}
    # No overlap on enum values.
    assert fi_states.isdisjoint(hmm_states), f"FI regime values overlap HMM: {fi_states & hmm_states!r}"
    # And no overlap on the human-readable labels either.
    assert fi_labels.isdisjoint(hmm_states), f"FI regime labels collide with HMM names: {fi_labels & hmm_states!r}"
    # Sanity: enum values are the snake_case FI labels we expect.
    assert "risk_on_compression" in fi_states
    assert "crisis_severe_dislocation" in fi_states
    # HMM had its own ``risk_on_expansion`` token; ensure the FI side
    # uses a distinct ``risk_on_compression`` token.
    assert "risk_on_expansion" not in fi_states
    assert "risk_on_compression" not in hmm_states
