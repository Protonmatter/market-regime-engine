# SPDX-License-Identifier: Apache-2.0
"""PR-5 Q-5: Minimum Track Record Length (Bailey–López de Prado)."""

from __future__ import annotations

import math

import pytest

from market_regime_engine.validation import minimum_track_record_length


def test_mtrl_inf_when_observed_does_not_exceed_target() -> None:
    """If the observed Sharpe is at or below the target, the inequality
    can never be defended → ``inf``."""
    assert math.isinf(minimum_track_record_length(0.5, 1.0))
    assert math.isinf(minimum_track_record_length(0.0, 0.0))


def test_mtrl_decreases_with_larger_observed_sharpe() -> None:
    weak = minimum_track_record_length(0.6, 0.5)
    strong = minimum_track_record_length(1.5, 0.5)
    assert strong < weak


def test_mtrl_increases_with_negative_skew_and_high_kurtosis() -> None:
    normal = minimum_track_record_length(1.0, 0.0, skew=0.0, excess_kurt=0.0)
    fat_tail = minimum_track_record_length(1.0, 0.0, skew=-1.0, excess_kurt=8.0)
    assert fat_tail > normal


def test_mtrl_rejects_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence must be in"):
        minimum_track_record_length(1.0, 0.0, confidence=1.5)


def test_mtrl_matches_blp_closed_form_for_gaussian_returns() -> None:
    """For Gaussian returns (skew=0, kurt=0), MTRL reduces to
    ``1 + (Φ⁻¹(C) / (SR − SR_target))²``."""
    sr = 1.0
    sr_target = 0.0
    # Φ⁻¹(0.95) ≈ 1.6449.
    expected = 1.0 + (1.6449 / (sr - sr_target)) ** 2
    out = minimum_track_record_length(sr, sr_target, confidence=0.95)
    assert abs(out - expected) < 0.01
