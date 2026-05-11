# SPDX-License-Identifier: Apache-2.0
"""PR-5 Q-5: Deflated Sharpe Ratio (Bailey–López de Prado 2014)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_regime_engine.validation import deflated_sharpe


def test_deflated_sharpe_rejects_pure_noise_signal_under_multiple_trials() -> None:
    """A zero-mean noise return stream should land near 0 DSR after
    selecting from many candidates (BLP eq. 9 example)."""
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(loc=0.0, scale=0.01, size=2520))  # 10y daily
    dsr = deflated_sharpe(returns, n_trials=100)
    # Multiple testing punishes a zero-edge strategy hard — DSR < 0.5.
    assert 0.0 <= dsr <= 0.5


def test_deflated_sharpe_accepts_clear_signal_under_few_trials() -> None:
    """A strong real Sharpe (~2.0) should survive a small number of trials
    with DSR > 0.95."""
    rng = np.random.default_rng(1)
    # ~Sharpe = 2.0 on daily cadence.
    returns = pd.Series(rng.normal(loc=0.10 / 252, scale=0.005, size=2520))
    dsr = deflated_sharpe(returns, n_trials=5)
    assert dsr > 0.95


def test_deflated_sharpe_returns_nan_on_short_input() -> None:
    out = deflated_sharpe(pd.Series([0.0]), n_trials=10)
    assert np.isnan(out)


def test_deflated_sharpe_returns_one_on_constant_positive_return() -> None:
    out = deflated_sharpe(pd.Series([0.01] * 100), n_trials=1, sharpe_target=0.0)
    assert out == 1.0


def test_deflated_sharpe_respects_explicit_skew_kurt_arguments() -> None:
    rng = np.random.default_rng(2)
    returns = pd.Series(rng.normal(loc=0.001, scale=0.01, size=1000))
    base = deflated_sharpe(returns, n_trials=5, skew=0.0, kurt=0.0)
    # Heavy left tail (negative skew, large kurt) should drag DSR down.
    heavy_tail = deflated_sharpe(returns, n_trials=5, skew=-1.0, kurt=8.0)
    assert heavy_tail < base


def test_deflated_sharpe_higher_trial_count_lowers_score() -> None:
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.normal(loc=0.001, scale=0.01, size=1500))
    few = deflated_sharpe(returns, n_trials=5)
    many = deflated_sharpe(returns, n_trials=10_000)
    assert many <= few
