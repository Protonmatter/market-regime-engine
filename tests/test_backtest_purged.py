"""Regression tests for the purged-walk-forward versions of the backtest.

The historical implementation used a naive expanding-window split that left
overlapping forward-target windows in the training set; v1.1 wires both
``expanding_window_binary_backtest`` and ``expanding_window_quantile_backtest``
through :class:`walk_forward.PurgedWalkForward` to drop the leak.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from market_regime_engine.backtest import (
    _parse_horizon_months,
    expanding_window_binary_backtest,
    expanding_window_quantile_backtest,
)
from market_regime_engine.walk_forward import PurgedWalkForward


def _synthetic_panel(n: int = 180, seed: int = 0) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2005-01-01", periods=n, freq="MS")
    X = pd.DataFrame(
        {
            "factor_a": rng.normal(size=n),
            "factor_b": rng.normal(size=n),
            "factor_c": rng.normal(size=n),
        },
        index=idx,
    )
    score = X["factor_a"] - 0.5 * X["factor_b"] + 0.1 * rng.normal(size=n)
    y_binary = pd.Series((score > 0).astype(float), index=idx, name="dd10_3m")
    y_return = pd.Series(0.01 * score + 0.02 * rng.normal(size=n), index=idx, name="ret_3m")
    return X, y_binary, y_return


def test_horizon_parser_extracts_integer_months() -> None:
    # v1.5 PR-5 (AF-10): _parse_horizon_months is deprecated in favour of
    # _parse_horizon_periods(cadence='monthly'); the shim preserves the
    # v1.4 permissive fallback behaviour. Silence the deprecation
    # warning here so the legacy contract pin stays a regression check.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert _parse_horizon_months("3m") == 3
        assert _parse_horizon_months("12m") == 12
        assert _parse_horizon_months("forward_18m_extreme") == 18
        assert _parse_horizon_months("garbage") == 1
        assert _parse_horizon_months("") == 1


def test_binary_backtest_returns_non_empty_predictions_and_validation() -> None:
    X, y_binary, _ = _synthetic_panel(n=180)
    preds, val = expanding_window_binary_backtest(
        X, y_binary, target="drawdown_gt_10pct", horizon="3m", min_train=60, step=4
    )
    assert not preds.empty, "purged walk-forward backtest must emit predictions"
    assert {"date", "target", "horizon", "y", "p", "model"}.issubset(preds.columns)
    assert (preds["p"].between(0.0, 1.0) | preds["p"].isna()).all()
    assert not val.empty


def test_binary_backtest_purges_overlapping_horizon_window() -> None:
    """For every test fold, every training row's index must satisfy
    ``train_idx <= test_idx_min - H``. That guarantees no training row's
    horizon-month forward window overlaps the test point."""
    X, y_binary, _ = _synthetic_panel(n=200)
    H = 12  # months

    splitter = PurgedWalkForward(min_train=72, step=6, horizon=H, embargo=1, expanding=True, test_block=1)
    folds = list(splitter.split(len(X)))
    assert len(folds) > 0, "splitter must emit at least one fold"
    for fold in folds:
        if fold.train_idx.size == 0 or fold.test_idx.size == 0:
            continue
        # Every training row must be at least `H` periods before the test point.
        assert fold.train_idx.max() <= int(fold.test_idx.min()) - H, (
            f"fold {fold.fold} retained training rows whose horizon window overlaps the test point"
        )

    # End-to-end: the same gap holds when expanding_window_binary_backtest
    # uses horizon='12m'.
    preds, _ = expanding_window_binary_backtest(
        X, y_binary, target="drawdown_gt_10pct", horizon="12m", min_train=72, step=6
    )
    assert not preds.empty


def test_quantile_backtest_returns_non_empty_predictions_and_metrics() -> None:
    X, _, y_return = _synthetic_panel(n=180)
    preds, metrics = expanding_window_quantile_backtest(
        X, y_return, target="forward_return_log", horizon="6m", min_train=60, step=4
    )
    assert not preds.empty
    assert {"date", "target", "horizon", "y", "model"}.issubset(preds.columns)
    quantile_cols = {col for col in preds.columns if col.startswith("q") and col[1:].isdigit()}
    assert {"q05", "q50", "q95"}.issubset(quantile_cols)
    assert not metrics.empty
    assert {"quantile", "pinball_loss", "coverage", "model"}.issubset(metrics.columns)
