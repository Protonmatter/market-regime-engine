# SPDX-License-Identifier: Apache-2.0
"""PR-5 Q-5: Probability of Backtest Overfitting (Bailey et al. 2017)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.validation import probability_of_backtest_overfitting


def test_pbo_on_pure_noise_is_high() -> None:
    """When every strategy is a random walk, the in-sample best is
    expected to revert to median (or below) out-of-sample. PBO should
    sit well above 0.5."""
    rng = np.random.default_rng(0)
    t, s = 256, 50
    noise = rng.normal(size=(t, s))
    pbo = probability_of_backtest_overfitting(pd.DataFrame(noise), n_partitions=16)
    assert pbo > 0.4


def test_pbo_on_persistent_signal_is_low() -> None:
    """When strategy 0 has a real edge across every period, the IS best
    will be 0 and OOS best will be 0 too — PBO should sit at the floor."""
    rng = np.random.default_rng(1)
    t, s = 512, 20
    base = rng.normal(size=(t, s)) * 0.1
    base[:, 0] += 1.0  # strategy 0 dominates every period
    pbo = probability_of_backtest_overfitting(pd.DataFrame(base), n_partitions=8)
    assert pbo < 0.05


def test_pbo_returns_nan_on_empty_or_single_column() -> None:
    assert np.isnan(probability_of_backtest_overfitting(pd.DataFrame(), n_partitions=4))
    single = pd.DataFrame({"only": np.zeros(100)})
    assert np.isnan(probability_of_backtest_overfitting(single, n_partitions=4))


def test_pbo_rejects_odd_n_partitions() -> None:
    with pytest.raises(ValueError, match="even integer"):
        probability_of_backtest_overfitting(pd.DataFrame(np.zeros((20, 5))), n_partitions=3)


def test_pbo_handles_short_history_gracefully() -> None:
    """fewer rows than partitions → NaN, not a crash."""
    short = pd.DataFrame(np.zeros((4, 10)))
    assert np.isnan(probability_of_backtest_overfitting(short, n_partitions=16))
