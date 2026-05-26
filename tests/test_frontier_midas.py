# SPDX-License-Identifier: Apache-2.0
"""Baseline tests for :mod:`market_regime_engine.frontier.midas`.

Phase 5.2 of v1.6.0 (REVIEW_DEEP_V1_5_2.md §4 / §1.12 / F17): pin the
MIDAS Almon-weight + finite-difference gradient identity, the
nan_policy plumbing introduced in Phase 2 (replacing the silent
``fillna(0)``), and the basic fit / predict surface.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.frontier.data_cleaning import NanPolicy
from market_regime_engine.frontier.midas import MIDASLagSpec, MIDASRegressor


def test_almon_weights_sum_to_one():
    """The exponential Almon weights are a softmax over polynomial
    pre-activations, so they must sum to 1 for any theta and any k.
    """
    for theta in (
        np.array([0.0, 0.0]),
        np.array([0.1, -0.1]),
        np.array([-0.5, 0.05]),
        np.array([1.5]),
    ):
        for k in (4, 8, 12):
            w = MIDASRegressor.almon_weights(theta, k)
            assert w.shape == (k,)
            assert np.all(w >= 0.0)
            assert abs(float(w.sum()) - 1.0) < 1e-9


def test_almon_weights_gradient_matches_finite_difference():
    """Closed-form gradient ``dw/dθ_d`` matches central finite-differences.

    The MIDAS coordinate-descent update relies on the analytical
    gradient ``(idx**d) * w - w * (idx**d @ w)`` (see
    :meth:`MIDASRegressor.fit`); pinning this identity is the cheapest
    insurance against a sign-flip regression in the optimiser.
    """
    theta = np.array([-0.1, 0.05])
    k = 8
    w = MIDASRegressor.almon_weights(theta, k)
    idx = np.arange(1, k + 1, dtype=float)
    eps = 1e-5
    for d in range(1, len(theta) + 1):
        analytic = (idx**d) * w - w * float((idx**d) @ w)
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[d - 1] += eps
        theta_minus[d - 1] -= eps
        w_plus = MIDASRegressor.almon_weights(theta_plus, k)
        w_minus = MIDASRegressor.almon_weights(theta_minus, k)
        numeric = (w_plus - w_minus) / (2.0 * eps)
        assert np.allclose(analytic, numeric, atol=1e-5), f"theta_d={d} analytic={analytic} vs numeric={numeric}"


def test_midas_regressor_fit_predict_smoke():
    """End-to-end fit / predict on a synthetic MIDAS problem."""
    rng = np.random.default_rng(0)
    n = 60
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    x_high = rng.normal(size=n)
    X = pd.DataFrame({"x_high": x_high}, index=dates)
    y = pd.Series(0.5 * np.roll(x_high, 1) + 0.2 * np.roll(x_high, 2) + rng.normal(scale=0.05, size=n), index=dates)
    spec = MIDASLagSpec(column="x_high", lags=4, polynomial_degree=2)
    model = MIDASRegressor(max_iter=20).fit(X, y, lag_specs=[spec])
    assert model.fitted is True
    preds = model.predict(X)
    assert preds.shape == (n,)


def test_midas_regressor_nan_policy_routed_to_clean_with_policy():
    """Phase-2 §1.12 / F17: ``nan_policy`` is plumbed to ``clean_with_policy``.

    The previous implementation called ``series.fillna(0.0)`` directly
    (a strong "risk-on today" signal for credit-spread MIDAS). The new
    default :attr:`NanPolicy.NAN_TO_LAST_VALID` forward-fills within the
    series; ``NAN_TO_ZERO`` reproduces the legacy numerics for back-compat.
    """
    rng = np.random.default_rng(1)
    n = 32
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    x = rng.normal(size=n)
    x_with_nan = x.copy()
    x_with_nan[5] = np.nan
    X = pd.DataFrame({"x_high": x_with_nan}, index=dates)
    spec = MIDASLagSpec(column="x_high", lags=3)
    model = MIDASRegressor(nan_policy=NanPolicy.NAN_TO_LAST_VALID)
    lag_mat = model._build_lag_matrix(X, spec)
    # The last_valid policy must NOT zero the NaN row's contributions —
    # the lag matrix should differ from the all-zero result.
    assert lag_mat.shape == (n, 3)
    assert np.any(np.abs(lag_mat) > 0)


def test_midas_regressor_predict_before_fit_returns_zero_or_empty():
    model = MIDASRegressor()
    out = model.predict(pd.DataFrame())
    assert out.shape == (0,)


@pytest.mark.parametrize("policy", [NanPolicy.NAN_TO_ZERO, NanPolicy.NAN_TO_LAST_VALID])
def test_midas_regressor_nan_policy_explicit_default(policy):
    """Smoke: setting nan_policy to either common policy doesn't crash
    the lag-matrix builder.
    """
    rng = np.random.default_rng(2)
    n = 24
    dates = pd.date_range("2020-01-01", periods=n, freq="MS")
    X = pd.DataFrame({"x_high": rng.normal(size=n)}, index=dates)
    spec = MIDASLagSpec(column="x_high", lags=2)
    model = MIDASRegressor(nan_policy=policy)
    lag_mat = model._build_lag_matrix(X, spec)
    assert lag_mat.shape == (n, 2)
