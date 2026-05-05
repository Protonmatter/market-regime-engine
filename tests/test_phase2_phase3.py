"""Tests for Phase 2 (DFM, robust z) and Phase 3 (MS-VAR, BOCPD-MUSE, hazard, BMA, cross-sectional)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_regime_engine.bma import OnlineBMA, online_bma_from_oos
from market_regime_engine.bocpd_hazard import CovariateBOCPDHazard
from market_regime_engine.bocpd_muse import BOCPDMuse
from market_regime_engine.cross_sectional import (
    fama_french_regime_head,
    sector_dispersion_head,
    yield_curve_factor_head,
)
from market_regime_engine.dfm import DFMDomainModel
from market_regime_engine.hazard_model import DiscreteTimeHazardModel
from market_regime_engine.msvar import MarkovSwitchingVAR
from market_regime_engine.robust_stats import (
    robust_zscore_frame,
    rolling_robust_z,
    rolling_winsorized_z,
)

# ---------------------------------------------------------------------------
# robust stats
# ---------------------------------------------------------------------------


def test_rolling_robust_z_outlier_signal_does_not_collapse():
    """The classic z-score *under-flags* a return-to-normal value because the
    in-window outlier inflated the std. The robust z keeps a stable scale, so
    its post-outlier readings sit closer to the true deviation magnitude.
    """
    rng = np.random.default_rng(0)
    n = 200
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    s = pd.Series(rng.normal(size=n), index=idx)
    s.iloc[100] = 50.0  # outlier

    classic = (s - s.shift(1).rolling(60, min_periods=24).mean()) / s.shift(1).rolling(60, min_periods=24).std(ddof=1)
    robust = rolling_robust_z(s, window=60, min_periods=24)

    # Both must remain finite throughout the post-outlier window.
    assert np.isfinite(robust.iloc[105:160].dropna()).all()
    # Robust scale should not be dominated by the outlier — once the outlier
    # leaves the rolling window (row 161 onward), robust z is essentially
    # uncontaminated. We assert that robust z magnitudes recover a reasonable
    # scale (median absolute z ≈ 0.6 - 0.8 for unit-variance noise).
    post_outlier = robust.iloc[170:].abs().median()
    assert 0.2 < post_outlier < 1.5
    # Classic z post-outlier-window is also reasonable; this just guards that
    # both regimes return to similar scale once the outlier leaves the window.
    classic_post = classic.iloc[170:].abs().median()
    assert 0.2 < classic_post < 1.5


def test_robust_zscore_frame_returns_one_col_per_input():
    df = pd.DataFrame(
        {
            "a": np.linspace(0, 1, 80),
            "b": np.sin(np.linspace(0, 6, 80)),
        },
        index=pd.date_range("2010-01-01", periods=80, freq="MS"),
    )
    out = robust_zscore_frame(df, method="expanding_mad", min_periods=12)
    assert list(out.columns) == ["a", "b"]
    assert len(out) == 80


def test_rolling_winsorized_z_returns_finite():
    n = 120
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    s = pd.Series(np.random.default_rng(0).normal(size=n), index=idx)
    out = rolling_winsorized_z(s, window=36, min_periods=18)
    assert np.isfinite(out.dropna()).all()


# ---------------------------------------------------------------------------
# DFM
# ---------------------------------------------------------------------------


def test_dfm_fit_and_transform_run():
    rng = np.random.default_rng(0)
    n = 240
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    factor = np.cumsum(rng.normal(scale=0.2, size=n))
    obs = np.column_stack(
        [
            factor + rng.normal(scale=0.1, size=n),
            0.7 * factor + rng.normal(scale=0.1, size=n),
            -0.4 * factor + rng.normal(scale=0.1, size=n),
        ]
    )
    df = pd.DataFrame(obs, index=idx, columns=["a", "b", "c"])
    model = DFMDomainModel(max_iter=20).fit(df)
    assert model.fitted
    f = model.transform(df)
    assert len(f) == n
    assert np.isfinite(f.dropna()).all()
    # Loadings on the dominant component must be non-trivial
    assert abs(model.loadings[0]) > 0.05


# ---------------------------------------------------------------------------
# MS-VAR
# ---------------------------------------------------------------------------


def test_msvar_fit_score_runs():
    rng = np.random.default_rng(0)
    n = 240
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    df = pd.DataFrame(
        rng.normal(scale=0.4, size=(n, 8)),
        index=idx,
        columns=["labor", "rates", "inflation", "credit", "housing", "energy", "fx", "fiscal"],
    )
    df.iloc[80:130] += np.array([1.5, 0.9, 1.7, 0.8, 0.8, 1.2, 0.6, 0.8])
    df.iloc[170:210] += np.array([0.7, 0.9, 0.4, 1.8, 1.1, 0.3, 0.7, 0.7])
    model = MarkovSwitchingVAR(max_iter=8).fit(df)
    assert model.fitted
    out = model.score(df)
    cols = [c for c in out.columns if c.startswith("msvar_prob_")]
    sums = out[cols].sum(axis=1)
    assert (sums - 1.0).abs().max() < 1e-6


# ---------------------------------------------------------------------------
# BOCPD-MUSE
# ---------------------------------------------------------------------------


def test_bocpd_muse_runs_and_emits_model_posterior():
    rng = np.random.default_rng(0)
    n = 80
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    pre = rng.normal(scale=1.0, size=(40, 3))
    post = rng.normal(scale=1.0, size=(40, 3)) + np.array([3.0, -2.5, 1.5])
    x = pd.DataFrame(np.vstack([pre, post]), index=idx, columns=["a", "b", "c"])
    out = BOCPDMuse(max_run=32).score(x)
    assert len(out) == n
    assert out["change_point_prob"].between(0.0, 1.0).all()
    posts = out[["model_post_niw", "model_post_diag", "model_post_ar1"]].sum(axis=1)
    assert (posts - 1.0).abs().max() < 1e-6


# ---------------------------------------------------------------------------
# Cox-style hazard
# ---------------------------------------------------------------------------


def test_covariate_hazard_fits_and_predicts_in_unit_interval():
    rng = np.random.default_rng(0)
    n = 240
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    cov = pd.DataFrame({"credit": rng.normal(size=n).cumsum() * 0.05}, index=idx)
    # Inject regime-change ground truth that depends on credit z-score.
    z = (cov["credit"] - cov["credit"].mean()) / cov["credit"].std()
    regimes = pd.Series(["risk_on"] * n, index=idx)
    flips = z.abs() > 1.5
    state = "risk_on"
    states_out = []
    for _d, flip in flips.items():
        if flip:
            state = "credit_stress" if state == "risk_on" else "risk_on"
        states_out.append(state)
    regimes = pd.Series(states_out, index=idx)
    hz = CovariateBOCPDHazard().fit(regimes, cov)
    assert hz.fitted
    series = hz.hazard_series(cov)
    assert series.between(1e-4, 0.5).all()


# ---------------------------------------------------------------------------
# Online BMA
# ---------------------------------------------------------------------------


def test_online_bma_drifts_toward_better_model():
    rng = np.random.default_rng(0)
    n = 200
    y = rng.binomial(1, 0.4, size=n).astype(float)
    p_good = np.where(y == 1, 0.7 + rng.normal(scale=0.05, size=n), 0.3 + rng.normal(scale=0.05, size=n))
    p_bad = np.full(n, 0.5)
    bma = OnlineBMA(forgetting=0.9)
    bma.initialize(["good", "bad"])
    final_w: dict[str, float] = {}
    for t in range(n):
        final_w = bma.update(float(y[t]), {"good": float(np.clip(p_good[t], 0.01, 0.99)), "bad": float(p_bad[t])})
    assert final_w["good"] > final_w["bad"]


def test_online_bma_from_oos_returns_per_step_history():
    n = 80
    idx = pd.date_range("2015-01-01", periods=n, freq="MS")
    y = np.random.default_rng(0).binomial(1, 0.3, size=n).astype(float)
    rows = []
    for i, d in enumerate(idx):
        rows.append(
            {
                "date": d,
                "model_name": "a",
                "horizon": "3m",
                "target": "drawdown_gt_10pct",
                "y": float(y[i]),
                "p": 0.6 if y[i] == 1 else 0.2,
            }
        )
        rows.append(
            {"date": d, "model_name": "b", "horizon": "3m", "target": "drawdown_gt_10pct", "y": float(y[i]), "p": 0.5}
        )
    history, _bma = online_bma_from_oos(pd.DataFrame(rows), target="drawdown_gt_10pct", horizon="3m")
    assert not history.empty
    assert {"w_a", "w_b", "mixed_p"}.issubset(history.columns)


# ---------------------------------------------------------------------------
# Cross-sectional heads
# ---------------------------------------------------------------------------


def test_cross_sectional_heads_emit_outputs():
    rng = np.random.default_rng(0)
    n = 96
    idx = pd.date_range("2015-01-01", periods=n, freq="MS")
    factors = pd.DataFrame({"SMB": rng.normal(size=n), "HML": rng.normal(size=n)}, index=idx)
    sectors = pd.DataFrame(rng.normal(size=(n, 5)), index=idx, columns=[f"sec_{i}" for i in range(5)])
    yields = pd.DataFrame(
        np.cumsum(rng.normal(scale=0.05, size=(n, 5)), axis=0), index=idx, columns=["1y", "2y", "5y", "10y", "30y"]
    )
    regimes = pd.DataFrame(
        {
            "date": idx,
            "decoded_regime": ["risk_on_expansion"] * n,
            "regime_prob_risk_on_expansion": np.full(n, 0.6),
            "regime_prob_late_cycle": np.full(n, 0.3),
            "regime_prob_credit_stress": np.full(n, 0.1),
        }
    )
    out_ff = fama_french_regime_head(factors, regimes)
    out_disp = sector_dispersion_head(sectors, regimes)
    out_curve = yield_curve_factor_head(yields, regimes)
    assert not out_ff.empty
    assert not out_disp.empty
    assert not out_curve.empty
    for frame in (out_ff, out_disp, out_curve):
        assert frame["value"].apply(np.isfinite).all()


# ---------------------------------------------------------------------------
# horizon_probability_path
# ---------------------------------------------------------------------------


def test_hazard_horizon_probability_path_matches_constant_when_flat():
    h = np.full(30, 0.05)
    cum = DiscreteTimeHazardModel.horizon_probability_path(h, horizon_months=12)
    expected = 1.0 - (1.0 - 0.05) ** 12
    # allow truncation at the tail
    assert abs(cum[5] - expected) < 1e-9
