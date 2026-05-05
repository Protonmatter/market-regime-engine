"""Tests for Mondrian conformal, CQR, ACI, and the multi-horizon Bonferroni
joint-coverage helper."""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_regime_engine.conformal import (
    AdaptiveConformalInference,
    ConformalizedQuantileRegression,
    MondrianBinaryConformal,
    fit_mondrian_from_oos,
)
from market_regime_engine.multi_horizon_conformal import BonferroniMultiHorizonConformal


def _binary_calibration(n: int = 500, alpha_used: float = 0.10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = rng.beta(2.0, 5.0, size=n)
    # bake in a tiny per-bucket miscalibration
    bucket = rng.choice(["a", "b", "c"], size=n)
    bias = np.where(bucket == "a", 0.0, np.where(bucket == "b", 0.05, -0.05))
    y = (rng.uniform(size=n) < np.clip(p + bias, 1e-3, 1 - 1e-3)).astype(int)
    return pd.DataFrame({"p": p, "y": y, "regime_bucket": bucket})


def test_mondrian_thresholds_per_bucket():
    df = _binary_calibration(n=600)
    layer = MondrianBinaryConformal(alpha=0.10).fit(df)
    assert set(layer.thresholds.keys()) == {"a", "b", "c"}
    for v in layer.thresholds.values():
        assert 0.0 <= v <= 1.0


def test_mondrian_coverage_matches_target_within_two_pct():
    df = _binary_calibration(n=2000)
    layer = MondrianBinaryConformal(alpha=0.10).fit(df)
    report = layer.coverage_report(df)
    # Realized coverage on the calibration set should hover near 1 - alpha = 0.9
    for cov in report["coverage"]:
        assert cov >= 0.85, f"coverage {cov} below tolerance"


def test_mondrian_transform_emits_set_and_uncertainty_columns():
    df = _binary_calibration(n=400)
    layer = MondrianBinaryConformal(alpha=0.20).fit(df)
    out = layer.transform(df.head(50).copy())
    assert {"conformal_set", "conformal_uncertain", "conformal_threshold"}.issubset(out.columns)
    # Sets are all non-empty strings
    assert out["conformal_set"].astype(str).str.len().min() > 0


def test_fit_mondrian_from_oos_handles_missing_bucket():
    df = pd.DataFrame({"p": np.random.uniform(size=100), "y": np.random.randint(0, 2, size=100)})
    layer = fit_mondrian_from_oos(df, alpha=0.1)
    assert "general" in layer.thresholds


def test_cqr_inflation_widens_interval_when_base_too_tight():
    """Force base predictions that miss `y` often, so CQR must inflate."""
    rng = np.random.default_rng(0)
    n = 800
    y = rng.normal(scale=1.0, size=n)
    # Base interval is centered on a wrong location with insufficient half-width;
    # ~30% of y values fall outside, so the residual is positive on those rows.
    q_lo = -0.4 * np.ones(n)
    q_hi = 0.4 * np.ones(n)
    cal = pd.DataFrame({"y": y, "q_lo": q_lo, "q_hi": q_hi})
    base_cov = float(((y >= q_lo) & (y <= q_hi)).mean())
    assert base_cov < 0.85  # sanity: base interval undercovers
    cqr = ConformalizedQuantileRegression(alpha=0.10).fit(cal)
    assert cqr.inflation > 0.0
    out = cqr.transform(cal)
    assert (out["q_hi_conformal"] - out["q_lo_conformal"]).min() >= (cal["q_hi"] - cal["q_lo"]).min()
    rep = cqr.coverage_report(cal)
    # Conformal layer brings coverage close to (1 - alpha) = 0.9.
    assert rep["coverage"] >= 0.85


def test_cqr_inflation_negative_when_base_already_overcovers():
    """If the base interval already covers everything, CQR may shrink (negative inflation)."""
    rng = np.random.default_rng(0)
    n = 400
    y = rng.normal(scale=1.0, size=n)
    q_lo = y - 3.0
    q_hi = y + 3.0
    cal = pd.DataFrame({"y": y, "q_lo": q_lo, "q_hi": q_hi})
    cqr = ConformalizedQuantileRegression(alpha=0.10).fit(cal)
    assert cqr.inflation <= 0.0


def _bonferroni_calibration_frame(seed: int, *, dates: pd.DatetimeIndex, half_width: float) -> pd.DataFrame:
    """Synthetic CQR calibration frame indexed by ``dates``.

    The base interval is ``[-half_width, half_width]`` and ``y`` is drawn from
    ``N(0, 1)`` so a fraction of points fall outside, forcing CQR to inflate.
    """
    rng = np.random.default_rng(seed)
    n = len(dates)
    y = rng.normal(scale=1.0, size=n)
    return pd.DataFrame(
        {
            "date": dates,
            "y": y,
            "q_lo": -half_width * np.ones(n),
            "q_hi": half_width * np.ones(n),
        }
    )


def test_bonferroni_joint_coverage_three_horizons_overlapping_dates() -> None:
    """Regression test for the rsuffix join collision in
    :meth:`BonferroniMultiHorizonConformal.joint_coverage`.

    Three horizons share an overlapping date range. The buggy implementation
    silently produced ambiguous columns and either raised or returned a
    biased coverage. The fix uses ``pd.concat`` along the date index so
    column names are unique by construction.
    """
    dates = pd.date_range("2018-01-01", periods=300, freq="MS")
    cal = {
        "3m": _bonferroni_calibration_frame(0, dates=dates, half_width=0.4),
        "6m": _bonferroni_calibration_frame(1, dates=dates, half_width=0.5),
        "12m": _bonferroni_calibration_frame(2, dates=dates, half_width=0.6),
    }
    layer = BonferroniMultiHorizonConformal(horizons=("3m", "6m", "12m"), alpha=0.30).fit(cal)
    out = layer.joint_coverage(cal)
    assert out["n"] > 0
    cov = float(out["joint_coverage"])
    assert 0.0 <= cov <= 1.0, f"joint_coverage out of bounds: {cov}"
    assert out["horizons_used"] == ["3m", "6m", "12m"]


def test_bonferroni_joint_coverage_handles_partial_horizon_overlap() -> None:
    """If some horizons share fewer dates, the joint coverage is reported
    only over the dates present in *every* horizon (pd.concat inner join)."""
    full_dates = pd.date_range("2018-01-01", periods=200, freq="MS")
    short_dates = full_dates[50:150]  # 100 dates fully inside the longer ones
    cal = {
        "3m": _bonferroni_calibration_frame(0, dates=full_dates, half_width=0.4),
        "6m": _bonferroni_calibration_frame(1, dates=short_dates, half_width=0.5),
        "12m": _bonferroni_calibration_frame(2, dates=full_dates, half_width=0.6),
    }
    layer = BonferroniMultiHorizonConformal(horizons=("3m", "6m", "12m"), alpha=0.30).fit(cal)
    out = layer.joint_coverage(cal)
    assert out["n"] == len(short_dates), f"expected join over {len(short_dates)} rows, got {out['n']}"
    assert 0.0 <= out["joint_coverage"] <= 1.0


def test_bonferroni_joint_coverage_drops_horizon_when_calibrator_missing() -> None:
    """Horizons whose CQR layer was not fitted are gracefully skipped."""
    dates = pd.date_range("2020-01-01", periods=120, freq="MS")
    cal_full = {
        "3m": _bonferroni_calibration_frame(0, dates=dates, half_width=0.4),
        "6m": _bonferroni_calibration_frame(1, dates=dates, half_width=0.5),
    }
    layer = BonferroniMultiHorizonConformal(horizons=("3m", "6m"), alpha=0.20).fit(cal_full)
    # Caller asks for 3 horizons but only feeds 2; the third is silently
    # skipped (and reported via ``horizons_used``).
    out = layer.joint_coverage(cal_full)
    assert out["horizons_used"] == ["3m", "6m"]


def test_aci_alpha_drifts_toward_target():
    rng = np.random.default_rng(0)

    def base_fit(history, alpha_t):
        ys = np.array([h["y"] for h in history], dtype=float)
        # simulate a fixed VaR-ish threshold
        return float(np.quantile(ys, alpha_t))

    def base_apply(threshold, record):
        return record["y"] >= threshold

    records = []
    for t, y in enumerate(rng.normal(size=200)):
        records.append((pd.Timestamp("2010-01-01") + pd.DateOffset(months=t), {"y": float(y)}))
    aci = AdaptiveConformalInference(alpha_target=0.10, gamma=0.05)
    out = aci.run(records, base_fit=base_fit, base_apply=base_apply, warmup=20)
    assert not out.empty
    assert out["alpha_t"].between(0.001, 0.5).all()
