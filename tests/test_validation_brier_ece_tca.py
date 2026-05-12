# SPDX-License-Identifier: Apache-2.0
"""v1.5.1 (PR-9 FIX 4b/4c): Brier / ECE / TCA-lift validation primitives."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.validation import (
    brier_score,
    expected_calibration_error,
    reliability_diagram_bins,
    tca_lift_test,
)


def test_brier_perfect_forecaster_returns_zero() -> None:
    y_true = [0.0, 1.0, 0.0, 1.0]
    y_prob = [0.0, 1.0, 0.0, 1.0]
    assert brier_score(y_true, y_prob) == pytest.approx(0.0, abs=1e-9)


def test_brier_worst_forecaster_returns_one() -> None:
    """Brier score is MSE of probabilities; the worst case is 1.0."""
    y_true = [0.0, 1.0]
    # Worst predictions: probability of TRUE event = 0; of FALSE event = 1.
    y_prob = [1.0, 0.0]
    # With our _clip_prob() the worst-case score gets clipped close to 1
    # but not exactly equal — accept anywhere in [0.99, 1.0].
    assert 0.99 <= brier_score(y_true, y_prob) <= 1.0


def test_brier_uniform_05_returns_quarter() -> None:
    """Uniform 0.5 prediction over a 50/50 outcome series → 0.25 Brier."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=10_000).astype(float)
    y_prob = np.full_like(y_true, 0.5)
    assert brier_score(y_true, y_prob) == pytest.approx(0.25, abs=1e-3)


def test_ece_perfect_forecaster_returns_zero() -> None:
    y_true = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    y_prob = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    assert expected_calibration_error(y_true, y_prob) == pytest.approx(0.0, abs=1e-9)


def test_ece_n_bins_kwarg_overrides_positional() -> None:
    """``n_bins=15`` is the PR-9 canonical contract; ``bins`` stays as alias."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=2000).astype(float)
    p = rng.uniform(0.0, 1.0, size=2000)
    via_positional = expected_calibration_error(y, p, bins=15)
    via_kw = expected_calibration_error(y, p, n_bins=15)
    assert via_positional == pytest.approx(via_kw, abs=1e-9)


def test_reliability_diagram_bins_returns_required_columns() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=500).astype(float)
    p = rng.uniform(0.0, 1.0, size=500)
    table = reliability_diagram_bins(y, p, n_bins=10)
    assert {"bin_idx", "mean_pred", "mean_obs", "count"}.issubset(table.columns)
    assert table["count"].sum() == 500


def test_reliability_diagram_bins_empty_input() -> None:
    """Empty / all-NaN input returns an empty frame with the required schema."""
    table = reliability_diagram_bins([], [], n_bins=15)
    assert table.empty
    assert {"bin_idx", "mean_pred", "mean_obs", "count"}.issubset(table.columns)


def test_tca_lift_test_passes_for_significant_regime() -> None:
    """One regime with a clear shift over baseline → significant p, large d."""
    rng = np.random.default_rng(0)
    n_per = 200
    df = pd.DataFrame(
        {
            "regime_label": ["calm"] * n_per + ["stressed"] * n_per,
            "slippage_bps": np.concatenate(
                [
                    rng.normal(loc=10.0, scale=1.0, size=n_per),  # ~ baseline
                    rng.normal(loc=20.0, scale=1.0, size=n_per),  # +10 bps shift
                ]
            ),
        }
    )
    baseline = rng.normal(loc=10.0, scale=1.0, size=400)
    out = tca_lift_test(df, pd.Series(baseline))
    assert "stressed" in out
    assert out["stressed"]["p_value"] < 0.001
    assert abs(out["stressed"]["effect_size"]) >= 2.0
    assert "calm" in out
    # The calm regime is near baseline — small effect size, p > 0.05.
    assert abs(out["calm"]["effect_size"]) < 0.3


def test_tca_lift_test_empty_returns_empty_dict() -> None:
    assert tca_lift_test(pd.DataFrame(), pd.Series([10.0, 11.0])) == {}


def test_tca_lift_test_handles_singleton_segment() -> None:
    """A regime with fewer than 2 observations is skipped (no test possible)."""
    df = pd.DataFrame(
        {
            "regime_label": ["calm", "alone"],
            "slippage_bps": [10.0, 20.0],
        }
    )
    baseline = [10.0, 11.0, 12.0, 13.0]
    out = tca_lift_test(df, pd.Series(baseline))
    assert "alone" not in out
    assert "calm" not in out  # also singleton


def test_brier_handles_nan_inputs_gracefully() -> None:
    y = [0.0, float("nan"), 1.0]
    p = [0.1, 0.5, 0.9]
    score = brier_score(y, p)
    assert math.isfinite(score)
