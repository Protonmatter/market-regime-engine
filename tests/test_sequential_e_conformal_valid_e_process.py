# SPDX-License-Identifier: Apache-2.0
"""Empirical e-process validity tests for :class:`SequentialEConformal`.

REVIEW_DEEP_V1_5_2.md §1.7 / Finding #3 (GA blocker): the prior
``SequentialEConformal._increment_from_score`` implementation
``2 * (1 - score)`` is only expectation-1 at ``p_hat = 0.5``. For any
non-balanced binary forecaster ``E[E_t / E_{t-1} | F_{t-1}] > 1`` under
H_0, so the Ville-inequality rejection ``E_t >= 1/alpha`` had no
anytime-valid type-I error control.

The fix replaces the increment with a betting e-process per
Ramdas-Manole 2023 §3:

    E_t = E_{t-1} * (1 + lambda_t * (y_t - p_hat_t))

with ``lambda_t = 1`` (GROW-conservative; admissible for any
``p_hat in [EPS, 1-EPS]``). Under H_0 the increment satisfies
``E[1 + lambda * (y - p_hat) | F_{t-1}] = 1``, so ``E_t`` is a
non-negative martingale and the Ville-inequality control holds.

This test pins both the algebraic identity (per ``p_hat``) and the
empirical expectation across 5000 iid Bernoulli draws.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.frontier.conformal_ts import EPS, SequentialEConformal


@pytest.mark.parametrize("p_hat", [0.05, 0.10, 0.30, 0.50, 0.70, 0.90, 0.95])
def test_increment_is_expectation_1_under_h0_algebraically(p_hat: float) -> None:
    """E[1 + 1 * (y - p_hat) | y ~ Bernoulli(p_hat)] = 1."""
    layer = SequentialEConformal()
    inc_y_eq_1 = layer._increment(p_hat, 1)
    inc_y_eq_0 = layer._increment(p_hat, 0)
    expected = p_hat * inc_y_eq_1 + (1.0 - p_hat) * inc_y_eq_0
    assert math.isclose(expected, 1.0, abs_tol=1e-12)


@pytest.mark.parametrize("p_hat", [0.05, 0.10, 0.30, 0.50, 0.70, 0.90, 0.95])
def test_increment_is_strictly_positive(p_hat: float) -> None:
    """The betting e-process increment must be non-negative for E_t to
    remain a non-negative martingale."""
    layer = SequentialEConformal()
    assert layer._increment(p_hat, 0) > 0.0
    assert layer._increment(p_hat, 1) > 0.0


@pytest.mark.parametrize("p_hat", [0.10, 0.30, 0.50, 0.70, 0.90])
def test_empirical_e_variable_property_under_calibrated_forecaster(p_hat: float) -> None:
    """Empirical mean of increments under H_0 (calibrated y ~ Bern(p_hat))
    must be close to 1 within Monte Carlo noise.

    With n = 5000 draws and increment SD bounded by max(p_hat, 1-p_hat) <= 1,
    the SE of the empirical mean is at most 1/sqrt(5000) ~ 0.014. A 3-sigma
    band around 1.0 is ~ +/- 0.05.
    """
    rng = np.random.default_rng(2024)
    n = 5000
    ys = rng.binomial(1, p_hat, size=n)
    layer = SequentialEConformal()
    increments = np.array([layer._increment(p_hat, int(y)) for y in ys])
    emp_mean = float(increments.mean())
    assert abs(emp_mean - 1.0) < 0.05, f"Empirical mean {emp_mean} too far from 1 for p_hat={p_hat}"


def test_e_process_is_martingale_under_h0() -> None:
    """For a calibrated stream (y_t ~ Bernoulli(p_hat_t) drawn fresh each step
    with varying p_hat_t), the running E_t should hover around 1 — its
    geometric drift under H_0 is ``E[log(increment)] <= 0`` (Jensen).

    We verify ``E_t / E_0 = E_t`` does not systematically explode (or
    collapse) over a long stream of varying p_hat_t.
    """
    rng = np.random.default_rng(7)
    n = 5000
    p_hats = rng.uniform(0.1, 0.9, size=n)
    ys = rng.binomial(1, p_hats)
    layer = SequentialEConformal()
    e = 1.0
    e_history = []
    for p_hat_val, y_val in zip(p_hats, ys, strict=True):
        e *= layer._increment(p_hat_val, int(y_val))
        e = max(e, layer.e_floor)
        e_history.append(e)
    # Under H_0, E_t is a non-negative supermartingale (in fact martingale)
    # so by Doob's optional-stopping or LLN log E_t / t -> -KL(p||p_hat) <= 0.
    # For a calibrated forecaster on iid streams this means the e-process
    # should NOT exceed 1/alpha = 10 in n=5000 with high probability.
    max_e = max(e_history)
    # Ville's inequality: P(sup E_t >= 10) <= 0.10. This is a soft check.
    # We only assert that the trajectory stays bounded (no runaway growth).
    assert max_e < 1e6, f"e-process exploded under H_0: max_e = {max_e}"


def test_fit_consumes_y_and_p_columns_for_e_process() -> None:
    """The new fit() path requires both y and p in the calibration frame
    so the e-process can be initialised correctly. (Regression test for
    the signature change in v1.6.0.)"""
    rng = np.random.default_rng(0)
    n = 200
    p_hats = rng.uniform(0.2, 0.8, size=n)
    ys = rng.binomial(1, p_hats)
    df = pd.DataFrame({"y": ys, "p": p_hats, "regime_bucket": ["a"] * n})
    layer = SequentialEConformal(alpha=0.10).fit(df)
    assert layer.bucket_counts["a"] == n
    assert layer.e_per_bucket["a"] > 0.0
    # The fit-time e-statistic should reflect the betting e-process; for a
    # calibrated forecaster the running product should be close to 1
    # (geometric mean of expectation-1 increments).
    assert layer.e_per_bucket["a"] < 10.0, f"Calibrated fit produced runaway e-statistic: {layer.e_per_bucket['a']}"


def test_update_uses_outcome_not_just_score() -> None:
    """The update path uses both pred and y (the betting e-process needs
    both). For a calibrated forecaster, repeated updates should keep the
    e-statistic near 1; for a miscalibrated forecaster (predicting 0.5
    when the true rate is 0.95), the e-statistic should grow."""
    rng = np.random.default_rng(0)

    # Calibrated case: e-stat hovers around 1.
    layer_cal = SequentialEConformal(alpha=0.05)
    layer_cal.fit(pd.DataFrame({"y": rng.binomial(1, 0.5, size=100), "p": [0.5] * 100, "regime_bucket": ["a"] * 100}))
    e_initial = layer_cal.e_per_bucket["a"]
    for _ in range(500):
        y_val = int(rng.binomial(1, 0.5))
        layer_cal.update("a", y_val, 0.5)
    e_final_cal = layer_cal.e_per_bucket["a"]

    # Miscalibrated case: predict 0.5 but truth is 0.95 → e-stat grows
    # because each y=1 increment is 1 + (1 - 0.5) = 1.5 with prob 0.95.
    layer_mis = SequentialEConformal(alpha=0.05)
    layer_mis.fit(pd.DataFrame({"y": rng.binomial(1, 0.95, size=100), "p": [0.5] * 100, "regime_bucket": ["a"] * 100}))
    for _ in range(500):
        y_val = int(rng.binomial(1, 0.95))
        layer_mis.update("a", y_val, 0.5)
    e_final_mis = layer_mis.e_per_bucket["a"]

    # Calibrated case: e-stat stays bounded.
    assert e_final_cal < 100.0
    # Miscalibrated case: e-stat should be detectably larger (large H_1 signal).
    assert e_final_mis > e_final_cal
    # And miscalibrated should breach the rejection threshold 1/alpha = 20
    # with high probability after 500 updates with ~0.95 success rate
    # (alpha=0.05). This is the Ville-inequality contract: H_1 detection
    # is fast, H_0 type-I control is anytime-valid.
    assert e_final_mis > 1.0 / 0.05, f"Miscalibrated H_1 not detected: e = {e_final_mis}"


def test_p_hat_clipped_to_eps_keeps_increment_admissible() -> None:
    """``p_hat = 0`` or ``p_hat = 1`` would make lambda = 1 inadmissible;
    the implementation clips to ``[EPS, 1-EPS]`` so the increment stays
    bounded and positive."""
    layer = SequentialEConformal()
    # p_hat = 0 → clipped to EPS, increment is in [1-EPS, 2-EPS]
    inc_zero_y0 = layer._increment(0.0, 0)
    inc_zero_y1 = layer._increment(0.0, 1)
    assert inc_zero_y0 > 0.0
    assert inc_zero_y1 > 0.0
    # p_hat = 1 → clipped to 1-EPS, increment is in [EPS, 1+EPS]
    inc_one_y0 = layer._increment(1.0, 0)
    inc_one_y1 = layer._increment(1.0, 1)
    assert inc_one_y0 > 0.0
    assert inc_one_y1 > 0.0
    # Floor is positive so e_floor never zeros out E_t.
    assert layer.e_floor > 0.0
    assert EPS > 0.0
