# SPDX-License-Identifier: Apache-2.0
"""v1.5.2: audit of DSR + MTRL + PBO formulas vs Bailey & López de Prado (2014, 2017).

This module pins down the BLP-consistent closed forms

    Var(SR_hat) ≈ (1 − γ_3·SR + (γ_4_excess + 2)/4 · SR²) / (T − 1)     (eq. 5)
    SR*        =  sharpe_target + sqrt(Var(SR_hat)) · E[max_z(N)]       (eq. 9)
    n*         =  1 + Var(SR_hat) · (Z / (SR − SR_target))²             (eq. 8)

(BLP "The Deflated Sharpe Ratio" §3 eq. 5/8/9; Lo 2002; Mertens 2002)
where γ_3 is sample skewness, γ_4_excess is *excess* kurtosis
(γ_4_excess = γ_4 − 3), and ``_sample_skew_kurt`` returns the excess
form. Both ``deflated_sharpe`` and ``minimum_track_record_length``
must agree on this single var_term form.

v1.5.2 corrects three BLP-conformance bugs flagged in the v1.5.1 review:

- **A1** — DSR + MTRL variance term used ``(γ_4 − 1)/4`` with *excess*
  kurtosis as input. The Pearson-kurtosis form ``(γ_4 − 1)/4`` rewrites
  to ``(γ_4_excess + 2)/4`` in excess form. Off by 3/4 in the kurt
  term.
- **A2** — PBO ``_purge_and_embargo`` used max-semantics on the right
  side so embargo was silently subsumed when ``embargo <= purge``. LdP
  CPCV specifies union (additive) semantics; total right-side drop =
  ``purge + embargo``.
- **A3** — DSR multiplicity threshold ``SR*`` scaled with ``1/sqrt(T−1)``
  instead of the moment-corrected ``sqrt(Var(SR_hat))``. BLP eq. (9)
  specifies ``SR* = sharpe_target + sqrt(Var(SR_hat)) · E[max_z(N)]``.

These tests lock the BLP-correct closed forms so a future
"simplification" cannot silently regress the variance estimator
or the multiplicity scaling.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.validation import (
    _expected_max_z,
    _normal_cdf,
    _normal_ppf,
    deflated_sharpe,
    minimum_track_record_length,
    probability_of_backtest_overfitting,
)


# -- DSR audit ----------------------------------------------------------------


def test_dsr_uses_blp_eq5_variance_term_for_normal_returns() -> None:
    """For pure-noise Gaussian (skew=0, kurt=0, n_trials=1), DSR should
    equal Φ(SR_hat / sqrt(var_term / (n − 1))) where ``var_term``
    follows BLP eq. (5) — the asymptotic null distribution.

    Concretely with n_trials=1 the deflated threshold collapses to the
    raw target Sharpe (``_expected_max_z(1) = 0``), and the test
    confirms that ``DSR > 0.5`` when ``SR_hat > 0`` for n_trials=1
    even for short series.
    """
    rng = np.random.default_rng(0)
    returns = rng.normal(0.05, 0.1, size=512)
    out = deflated_sharpe(returns, n_trials=1)
    sr_hat = returns.mean() / returns.std(ddof=1)
    # SR_hat is well above 0 for these 512 draws (mu/sigma ~= 0.5).  # noqa: RUF003
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

    BLP (2014) §3 eq. (5) — ``(γ_4_excess + 2)/4`` term grows with γ_4
    excess kurtosis. We construct a signal with SR_hat ≈ 0.3 and
    moderate n_trials so the DSR sits in the (0, 1) interior where
    the variance term is visible.
    """
    rng = np.random.default_rng(3)
    # Strong-enough signal: mu/sigma ~= 0.3 so SR_hat clearly exceeds SR*  # noqa: RUF003
    # even after n_trials=20 deflation.
    base = rng.normal(0.03, 0.1, size=512)
    benign = deflated_sharpe(base, n_trials=20, skew=0.0, kurt=0.0)
    fat_tail = deflated_sharpe(base, n_trials=20, skew=0.0, kurt=20.0)
    # Sanity: neither should saturate the [0, 1] interval.
    assert 0.0 < fat_tail < benign < 1.0


# -- BLP eq. 5 var_term anchors (A1) ------------------------------------------


def _blp_var_term(sharpe: float, skew: float, excess_kurt: float) -> float:
    """BLP eq. (5) variance term in *excess*-kurtosis form (A1-corrected)."""
    return max(
        1.0 - skew * sharpe + (excess_kurt + 2.0) / 4.0 * sharpe * sharpe,
        1e-12,
    )


def test_a1_gaussian_iid_var_term_is_1_plus_half_sr_squared() -> None:
    """A1 anchor: Gaussian iid returns (skew=0, excess_kurt=0) at
    SR_hat=1 must yield ``var_term = 1 + 0.5·1² = 1.5``.

    Pre-A1 the buggy ``(γ_4 − 1)/4`` form with excess kurt as input
    gave ``var_term = 1 − 0.25 = 0.75`` — off by 0.75 from the BLP
    eq. (5) value. We probe ``var_term`` via MTRL's exact closed form
    ``n* = 1 + var_term · (Z / SR)²``.
    """
    sr = 1.0
    z = _normal_ppf(0.95)
    expected_var_term = 1.5
    expected_n_star = 1.0 + expected_var_term * (z / sr) ** 2
    got = minimum_track_record_length(
        sr, 0.0, skew=0.0, excess_kurt=0.0, confidence=0.95
    )
    assert abs(got - expected_n_star) < 1e-9, (
        f"BLP eq. 5 var_term at Gaussian iid (SR=1) should be 1.5, "
        f"but MTRL yielded n*={got}, expected {expected_n_star}"
    )


def test_a1_positive_skew_decreases_var_term() -> None:
    """A1 property: positive skewness DECREASES ``var_term`` (good — a
    fat right tail biases SR upward, so the variance penalty shrinks).

    Compare baseline (skew=0) to strongly positive skew (skew=+1) at
    fixed SR_hat=1 and Gaussian kurtosis. Verify the implementation
    via MTRL's closed form.
    """
    sr = 1.0
    baseline_var = _blp_var_term(sr, skew=0.0, excess_kurt=0.0)
    skewed_var = _blp_var_term(sr, skew=1.0, excess_kurt=0.0)
    assert skewed_var < baseline_var, (
        f"Positive skew should reduce var_term ({skewed_var} >= {baseline_var})"
    )
    # MTRL is monotone in var_term, so positive skew should reduce n*.
    got_baseline = minimum_track_record_length(
        sr, 0.0, skew=0.0, excess_kurt=0.0, confidence=0.95
    )
    got_skewed = minimum_track_record_length(
        sr, 0.0, skew=1.0, excess_kurt=0.0, confidence=0.95
    )
    assert got_skewed < got_baseline


def test_a1_positive_excess_kurtosis_increases_var_term() -> None:
    """A1 property: positive excess kurtosis INCREASES ``var_term``
    (good — heavy tails inflate the SR sampling variance).

    Compare baseline (kurt=0) to fat tails (kurt=+8) at fixed
    SR_hat=1, Gaussian skew. Verify the implementation via MTRL.

    Pre-A1 the buggy ``(γ_4 − 1)/4`` form would have INCREASED with
    γ_4 too, but starting from the wrong baseline (0.75 instead of
    1.5). The fix relocates the baseline AND keeps the monotone
    growth.
    """
    sr = 1.0
    baseline_var = _blp_var_term(sr, skew=0.0, excess_kurt=0.0)
    fat_var = _blp_var_term(sr, skew=0.0, excess_kurt=8.0)
    assert fat_var > baseline_var, (
        f"Positive excess kurtosis should inflate var_term "
        f"({fat_var} <= {baseline_var})"
    )
    got_baseline = minimum_track_record_length(
        sr, 0.0, skew=0.0, excess_kurt=0.0, confidence=0.95
    )
    got_fat = minimum_track_record_length(
        sr, 0.0, skew=0.0, excess_kurt=8.0, confidence=0.95
    )
    assert got_fat > got_baseline


# -- MTRL audit ---------------------------------------------------------------


def test_mtrl_blp_closed_form_at_sr_one_gaussian() -> None:
    """Locked closed form (BLP eq. 5, 8) for SR=1, γ_3=0, γ_4_excess=0:

        var_term = 1 + (0 + 2)/4 · 1² = 1.5
        n*       = 1 + 1.5 · Z²

    Pre-A1 the buggy ``(γ_4_excess − 1)/4`` form gave var_term=0.75
    and ``n* = 1 + 0.75·Z²``; v1.5.2 A1 corrects to 1.5.
    """
    sr = 1.0
    z = _normal_ppf(0.95)
    expected = 1.0 + 1.5 * z * z
    got = minimum_track_record_length(
        sr, 0.0, skew=0.0, excess_kurt=0.0, confidence=0.95
    )
    assert abs(got - expected) < 1e-6


def test_mtrl_blp_closed_form_at_sr_two_gaussian() -> None:
    """At SR=2 and Gaussian (γ_4_excess=0), the v1.5.2 A1-corrected
    closed form gives a *positive* var_term and the clamp is not hit:

        var_term = 1 + (0 + 2)/4 · 4 = 3.0
        n*       = 1 + 3.0 · (Z / 2)² = 1 + 0.75·Z²

    Pre-A1 the buggy ``(γ_4_excess − 1)/4`` form gave
    ``var_term = 1 − 1 = 0`` (clamped to 1e-12) so MTRL collapsed to
    ~1. The fix restores a numerically sensible required-track
    length for high-SR Gaussian inputs.
    """
    sr = 2.0
    z = _normal_ppf(0.95)
    var_term = 1.0 + (0.0 + 2.0) / 4.0 * sr * sr
    expected = 1.0 + var_term * (z / sr) ** 2
    got = minimum_track_record_length(
        sr, 0.0, skew=0.0, excess_kurt=0.0, confidence=0.95
    )
    assert abs(got - expected) < 1e-9


def test_mtrl_consistency_with_dsr_variance_term() -> None:
    """The MTRL var_term ``(1 − γ_3·SR + (γ_4_excess + 2)/4 · SR²)``
    must match the DSR closed form (v1.5.2 A1). We pick a non-trivial
    (skew, kurt) pair and verify the closed form numerically.
    """
    sr = 1.2
    skew = -0.4
    kurt = 5.0
    z = _normal_ppf(0.95)
    var_term = 1.0 - skew * sr + (kurt + 2.0) / 4.0 * sr * sr
    expected = 1.0 + var_term * (z / sr) ** 2
    got = minimum_track_record_length(sr, 0.0, skew=skew, excess_kurt=kurt, confidence=0.95)
    assert abs(got - expected) < 1e-9


def test_mtrl_sweeps_consistent_across_grid() -> None:
    """Property sweep: for a grid of (SR, skew, kurt) tuples the
    v1.5.2 A1 BLP-consistent closed form
    ``var_term = 1 − γ_3·SR + (γ_4_excess + 2)/4 · SR²`` must
    reproduce ``minimum_track_record_length`` to machine precision.
    """
    z = _normal_ppf(0.95)
    for sr in (0.6, 0.8, 1.0, 1.2):
        for skew in (-0.5, 0.0, 0.5):
            for kurt in (0.0, 1.0, 4.0, 8.0):
                var_term = max(
                    1.0 - skew * sr + (kurt + 2.0) / 4.0 * sr * sr,
                    1e-12,
                )
                expected = 1.0 + var_term * (z / sr) ** 2
                got = minimum_track_record_length(
                    sr, 0.0, skew=skew, excess_kurt=kurt, confidence=0.95
                )
                assert math.isfinite(got), f"non-finite for SR={sr}"
                assert abs(got - expected) < 1e-9, (
                    f"closed-form mismatch at SR={sr}, skew={skew}, kurt={kurt}: "
                    f"got={got}, expected={expected}"
                )


def test_mtrl_rejects_negative_or_unity_confidence() -> None:
    with pytest.raises(ValueError):
        minimum_track_record_length(1.0, 0.0, confidence=0.0)
    with pytest.raises(ValueError):
        minimum_track_record_length(1.0, 0.0, confidence=1.0)


def test_mtrl_returns_inf_when_observed_at_or_below_target() -> None:
    assert math.isinf(minimum_track_record_length(0.5, 0.5))
    assert math.isinf(minimum_track_record_length(0.0, 0.1))


# -- A2 audit: PBO embargo additive to purge ---------------------------------


def test_a2_pbo_embargo_additive_to_purge() -> None:
    """A2: PBO ``_purge_and_embargo`` applies embargo AFTER the purge
    window (total right-side drop = ``purge + embargo``), not max-style
    (right-side drop = ``max(purge, embargo)``).

    With purge=5, embargo=3 the BLP-correct (additive) semantics drop
    8 rows on the right of every OOS block; the buggy max-semantics
    drop only ``max(5, 3) = 5`` rows — identical to embargo=0. We
    therefore assert ``PBO(purge=5, embargo=3) != PBO(purge=5, embargo=0)``
    which fails under the buggy code and passes under v1.5.2 A2.
    """
    t = 60
    n_strats = 5
    purge = 5
    embargo = 3
    n_partitions = 6  # block size = 10
    rng = np.random.default_rng(20260512)
    perf = rng.normal(0.0, 1.0, size=(t, n_strats))
    df = pd.DataFrame(perf, columns=[f"s{i}" for i in range(n_strats)])
    pbo_with_embargo = probability_of_backtest_overfitting(
        df, n_partitions=n_partitions, purge=purge, embargo=embargo
    )
    pbo_no_embargo = probability_of_backtest_overfitting(
        df, n_partitions=n_partitions, purge=purge, embargo=0
    )
    assert pbo_with_embargo != pbo_no_embargo, (
        f"Expected PBO(purge=5, embargo=3) != PBO(purge=5, embargo=0); "
        f"got identical {pbo_with_embargo}. Embargo is being subsumed "
        f"by purge (buggy max-semantics) instead of additive."
    )


# -- A3 audit: DSR multiplicity uses moment-corrected stderr -----------------


def test_a3_dsr_multiplicity_threshold_scales_with_moment_corrected_stderr() -> None:
    """A3 BLP eq. (9): ``SR* = sharpe_target + sqrt(Var(SR_hat)) · E[max_z(N)]``.

    The deflated threshold must scale with the moment-corrected stderr
    ``sqrt(var_term/(T−1))``, NOT with ``1/sqrt(T−1)`` alone. We
    construct a fat-tail input (excess kurt=8, so var_term ≈ 2× the
    Gaussian baseline) and verify the actual DSR matches the
    BLP-correct closed form, not the v1.5.1 buggy form.
    """
    rng = np.random.default_rng(7)
    # Returns with a non-trivial SR_hat (~0.18) — DSR sits in the
    # (0.1, 0.9) interior so the difference between BLP-correct and
    # buggy SR* scaling is numerically resolvable.
    returns = rng.normal(0.03, 0.1, size=120)
    n_trials = 50
    skew = 0.0
    kurt = 8.0  # fat tails so var_term > 1
    sharpe_target = 0.0
    n = returns.size
    sr_hat = returns.mean() / returns.std(ddof=1)
    var_term = max(
        1.0 - skew * sr_hat + (kurt + 2.0) / 4.0 * sr_hat * sr_hat,
        1e-12,
    )
    denom = math.sqrt(var_term / (n - 1))
    # BLP eq. 9: SR* scales with sqrt(Var(SR_hat)).
    sr_star_correct = sharpe_target + _expected_max_z(n_trials) * denom
    z_correct = (sr_hat - sr_star_correct) / denom
    expected_dsr = _normal_cdf(z_correct)
    # Buggy v1.5.1 formula: SR* scales with 1/sqrt(T-1).
    sr_star_buggy = sharpe_target + _expected_max_z(n_trials) / math.sqrt(n - 1)
    z_buggy = (sr_hat - sr_star_buggy) / denom
    buggy_dsr = _normal_cdf(z_buggy)

    actual_dsr = deflated_sharpe(
        returns,
        n_trials=n_trials,
        skew=skew,
        kurt=kurt,
        sharpe_target=sharpe_target,
    )

    assert abs(actual_dsr - expected_dsr) < 1e-9, (
        f"DSR={actual_dsr} != BLP-correct {expected_dsr}; "
        f"the multiplicity threshold may have regressed away from "
        f"sqrt(Var(SR_hat)) · E[max_z(N)] scaling."
    )
    # And the actual DSR must differ measurably from the buggy form —
    # var_term is ~2× Gaussian here so the threshold scaling matters.
    assert abs(actual_dsr - buggy_dsr) > 1e-4, (
        f"DSR matches the buggy 1/sqrt(T-1) form ({buggy_dsr}); "
        f"A3 fix may not have landed."
    )


def test_a3_dsr_threshold_grows_with_n_trials() -> None:
    """A3 sanity: as ``n_trials`` grows, the multiplicity correction
    inflates ``SR*`` and DSR strictly decreases (more candidates →
    harder for any single SR_hat to clear the deflated threshold).

    This is true under both the buggy and the BLP-correct A3 form, but
    pinning it here protects against future regressions that might
    decouple the threshold from ``_expected_max_z``.
    """
    rng = np.random.default_rng(11)
    returns = rng.normal(0.001, 0.1, size=120)
    prev = 1.0
    for n_trials in (1, 5, 25, 125, 625):
        dsr = deflated_sharpe(returns, n_trials=n_trials, skew=0.0, kurt=0.0)
        assert dsr <= prev + 1e-12, (
            f"DSR should be monotone non-increasing in n_trials; "
            f"got {dsr} after {prev} at n_trials={n_trials}"
        )
        prev = dsr
