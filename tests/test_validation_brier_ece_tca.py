# SPDX-License-Identifier: Apache-2.0
"""v1.5.1 (PR-9 FIX 4b/4c): Brier / ECE / TCA-lift validation primitives."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.validation import (
    brier_score,
    calibration_table,
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


# ---------------------------------------------------------------------------
# v1.6.0 (REVIEW_DEEP_V1_5_2.md §2.2 S4): ECE bin_strategy parameter
# ---------------------------------------------------------------------------


def test_ece_equal_width_honest_on_clustered_probabilities() -> None:
    """All predicted probs cluster in [0.45, 0.55] but outcomes are bimodal.

    Under v1.5.x ``pd.cut(..., duplicates="drop")`` the calibration table
    would collapse to a single bin and the reported ECE would silently
    average across that one bucket — understating miscalibration.
    v1.6.0 ``equal_width`` keeps all 10 fixed bins so the count-weighted
    mean of ``|pred − obs|`` reflects honest dispersion across the
    populated buckets only. The honest ECE for a forecaster that
    predicts ~0.5 when the true label rate is also ~0.8 is large
    (~ 0.3 in the surviving bins).
    """
    rng = np.random.default_rng(42)
    n = 1000
    p = rng.uniform(0.45, 0.55, size=n)
    y = (rng.uniform(0, 1, size=n) < 0.8).astype(float)
    with pytest.warns(RuntimeWarning, match=r"ECE bin collapse"):
        ece = expected_calibration_error(y, p, n_bins=10, bin_strategy="equal_width")
    assert ece >= 0.25, "equal_width ECE should reflect honest miscalibration on clustered probability inputs"


def test_ece_bin_collapse_warning_fires_when_triggered() -> None:
    """A forecaster that lands only in 2 of 10 bins triggers the warning."""
    p = [0.05] * 100 + [0.95] * 100
    y = [0.0] * 100 + [1.0] * 100
    with pytest.warns(RuntimeWarning, match=r"ECE bin collapse: only \d+/10"):
        _ = expected_calibration_error(y, p, n_bins=10)


def test_ece_no_collapse_warning_on_uniform_probabilities() -> None:
    """Probabilities spread evenly populate all bins; no warning."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=2000)
    y = (rng.uniform(0, 1, size=2000) < 0.5).astype(float)
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error", RuntimeWarning)
        ece = expected_calibration_error(y, p, n_bins=10)
    assert 0.0 <= ece <= 1.0


def test_ece_equal_mass_documented_collapse_with_small_n() -> None:
    """``equal_mass`` with small N and tied probabilities collapses bins;
    warning fires but the function still returns a defined ECE."""
    p = [0.5] * 50 + [0.51] * 50
    y = [0.0, 1.0] * 50
    with pytest.warns(RuntimeWarning, match=r"ECE bin collapse"):
        ece = expected_calibration_error(y, p, n_bins=10, bin_strategy="equal_mass")
    assert 0.0 <= ece <= 1.0


def test_ece_equal_mass_balanced_bins_no_warning() -> None:
    """``equal_mass`` with a smooth distribution populates all bins."""
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, size=1000)
    y = (rng.uniform(0, 1, size=1000) < p).astype(float)
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error", RuntimeWarning)
        ece = expected_calibration_error(y, p, n_bins=10, bin_strategy="equal_mass")
    assert 0.0 <= ece <= 1.0


def test_ece_invalid_bin_strategy_raises() -> None:
    with pytest.raises(ValueError, match="bin_strategy must be"):
        expected_calibration_error(
            [0.0, 1.0],
            [0.1, 0.9],
            n_bins=10,
            bin_strategy="exotic",  # type: ignore[arg-type]
        )


def test_calibration_table_equal_width_vs_equal_mass_differ() -> None:
    """The two strategies produce different bin layouts and counts."""
    rng = np.random.default_rng(0)
    n = 1000
    p = rng.beta(2, 5, size=n)
    y = (rng.uniform(0, 1, size=n) < p).astype(float)
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)
        table_ew = calibration_table(y, p, bins=10, bin_strategy="equal_width")
        table_em = calibration_table(y, p, bins=10, bin_strategy="equal_mass")
    counts_ew = table_ew["count"].to_numpy()
    counts_em = table_em["count"].to_numpy()
    assert counts_em.std() < counts_ew.std()


def test_release_gate_default_routes_to_equal_width() -> None:
    """Default ``bin_strategy`` matches explicit ``equal_width`` — preserves
    the v1.5.x ECE contract for callers that do not opt in."""
    rng = np.random.default_rng(11)
    p = rng.uniform(0.05, 0.95, size=500)
    y = (rng.uniform(0, 1, size=500) < p).astype(float)
    default = expected_calibration_error(y, p, n_bins=10)
    explicit = expected_calibration_error(y, p, n_bins=10, bin_strategy="equal_width")
    assert default == pytest.approx(explicit, abs=1e-12)
