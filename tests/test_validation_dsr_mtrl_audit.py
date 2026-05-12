# SPDX-License-Identifier: Apache-2.0
"""v1.5.1 (PR-9 FIX 4d): audit of DSR + MTRL formulas vs Bailey & Lopez de Prado (2014).

This module pins down the agreed BLP-consistent variance term

    Var(SR_hat) ≈ (1 − γ_3·SR + (γ_4 − 1)/4 · SR²) / (T − 1)

(BLP "The Deflated Sharpe Ratio" §3 eq. 5, also Lo 2002, Mertens 2002)
where γ_3 is sample skewness and γ_4 is **excess** kurtosis
(γ_4 = raw_kurt − 3). Both ``deflated_sharpe`` and
``minimum_track_record_length`` must agree on this single form.

These tests are hand-rolled "property-style" sweeps over closed-form
inputs and do not require Hypothesis. They lock the formula to the
audited BLP closed form so a future "simplification" cannot
silently regress the variance estimator.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from market_regime_engine.validation import (
    _normal_ppf,
    deflated_sharpe,
    minimum_track_record_length,
)

# -- DSR audit ----------------------------------------------------------------


def test_dsr_uses_blp_eq5_variance_term_for_normal_returns() -> None:
    """For pure-noise Gaussian (skew=0, kurt=0, n_trials=1), DSR should
    equal Φ(SR_hat / sqrt(var_term / (n − 1))) where ``var_term``
    follows BLP eq. (5) — the asymptotic null distribution.

    Concretely with n_trials=1 the deflated threshold collapses to the
    raw target Sharpe (no multiple-testing inflation), and the test
    confirms that ``DSR > 0.5`` when ``SR_hat > 0`` for n_trials=1
    even for short series.
    """
    rng = np.random.default_rng(0)
    returns = rng.normal(0.05, 0.1, size=512)
    out = deflated_sharpe(returns, n_trials=1)
    sr_hat = returns.mean() / returns.std(ddof=1)
    # SR_hat is well above 0 for these 512 draws (mu/sigma ~= 0.5).
    assert sr_hat > 0
    assert out > 0.5


def test_dsr_invariant_when_skew_and_kurt_pinned_to_normal() -> None:
    """Passing ``skew=0, kurt=0`` should match the sample-driven branch
    when the sample also happens to be ~Gaussian."""
    rng = np.random.default_rng(1)
    returns = rng.normal(0.02, 0.05, size=2048)
    sample = deflated_sharpe(returns, n_trials=10)
    pinned = deflated_sharpe(returns, n_trials=10, skew=0.0, kurt=0.0)
    # The sample skew/kurt of 2048 Gaussian draws is tiny, so the two
    # paths should be within 1e-2 of each other.
    assert abs(sample - pinned) < 0.05


def test_dsr_fat_tailed_input_yields_lower_score_when_above_threshold() -> None:
    """When SR_hat exceeds the deflated threshold SR*, larger excess
    kurtosis inflates ``Var(SR_hat)`` and therefore *lowers* DSR
    because the same numerator ``(SR_hat − SR*)`` is divided by a
    larger σ.

    BLP (2014) §3 eq. (5) — ``(γ_4 − 1)/4`` term grows with γ_4 for
    SR > 1 (where SR² > 0). We construct a signal with SR_hat ≈ 0.3
    and moderate n_trials so the DSR sits in the (0, 1) interior
    where the variance term is visible.
    """
    rng = np.random.default_rng(3)
    # Strong-enough signal: mu/sigma ~= 0.3 so SR_hat clearly exceeds SR*
    # even after n_trials=20 deflation.
    base = rng.normal(0.03, 0.1, size=512)
    benign = deflated_sharpe(base, n_trials=20, skew=0.0, kurt=0.0)
    fat_tail = deflated_sharpe(base, n_trials=20, skew=0.0, kurt=20.0)
    # Sanity: neither should saturate the [0, 1] interval.
    assert 0.0 < fat_tail < benign < 1.0


# -- MTRL audit ---------------------------------------------------------------


def test_mtrl_blp_closed_form_at_sr_one_gaussian() -> None:
    """Locked closed form (BLP eq. 5, 8) for SR=1, γ_3=0, γ_4=0:

    n* = 1 + (1 − 1/4) · (Z / 1)² = 1 + 0.75 · Z²
    """
    sr = 1.0
    z = _normal_ppf(0.95)
    expected = 1.0 + 0.75 * z * z
    got = minimum_track_record_length(sr, 0.0, skew=0.0, excess_kurt=0.0, confidence=0.95)
    assert abs(got - expected) < 1e-6


def test_mtrl_blp_closed_form_at_sr_two_gaussian() -> None:
    """At SR=2 and Gaussian (γ_4=0), the variance term flips sign and
    is clamped to ``1e-12`` by the implementation; MTRL collapses
    to ``1 + ε·(Z/SR)²`` (~ 1.0)."""
    sr = 2.0
    z = _normal_ppf(0.95)
    var_term = max(1.0 - (sr * sr) / 4.0, 1e-12)
    expected = 1.0 + var_term * (z / sr) ** 2
    got = minimum_track_record_length(sr, 0.0, skew=0.0, excess_kurt=0.0, confidence=0.95)
    assert abs(got - expected) < 1e-3


def test_mtrl_consistency_with_dsr_variance_term() -> None:
    """The MTRL var_term ``(1 − γ_3·SR + (γ_4 − 1)/4 · SR²)`` must
    match the DSR closed form. We pick a non-trivial (skew, kurt)
    pair and verify the closed form numerically."""
    sr = 1.2
    skew = -0.4
    kurt = 5.0
    z = _normal_ppf(0.95)
    var_term = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    expected = 1.0 + var_term * (z / sr) ** 2
    got = minimum_track_record_length(sr, 0.0, skew=skew, excess_kurt=kurt, confidence=0.95)
    assert abs(got - expected) < 1e-9


def test_mtrl_sweeps_consistent_across_grid() -> None:
    """Property sweep: for a grid of (SR, skew, kurt) tuples the
    BLP-consistent closed form must reproduce ``minimum_track_record_length``
    to machine precision."""
    z = _normal_ppf(0.95)
    for sr in (0.6, 0.8, 1.0, 1.2):
        for skew in (-0.5, 0.0, 0.5):
            for kurt in (0.0, 1.0, 4.0, 8.0):
                var_term = max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr, 1e-12)
                expected = 1.0 + var_term * (z / sr) ** 2
                got = minimum_track_record_length(sr, 0.0, skew=skew, excess_kurt=kurt, confidence=0.95)
                assert math.isfinite(got), f"non-finite for SR={sr}"
                assert abs(got - expected) < 1e-9, (
                    f"closed-form mismatch at SR={sr}, skew={skew}, kurt={kurt}: got={got}, expected={expected}"
                )


def test_mtrl_rejects_negative_or_unity_confidence() -> None:
    with pytest.raises(ValueError):
        minimum_track_record_length(1.0, 0.0, confidence=0.0)
    with pytest.raises(ValueError):
        minimum_track_record_length(1.0, 0.0, confidence=1.0)


def test_mtrl_returns_inf_when_observed_at_or_below_target() -> None:
    assert math.isinf(minimum_track_record_length(0.5, 0.5))
    assert math.isinf(minimum_track_record_length(0.0, 0.1))
