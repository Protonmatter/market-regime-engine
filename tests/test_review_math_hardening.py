# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for the end-to-end code/math review hardening pass."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.frontier.bayesian_msvar import BayesianMSVAR
from market_regime_engine.frontier.dfm_mq import MQDynamicFactorModel, build_synthetic_panel
from market_regime_engine.hmm import _optimal_assignment
from market_regime_engine.msvar import MarkovSwitchingVAR


def test_bayesian_msvar_accepts_p_gt_1_and_scores_full_lag_stack() -> None:
    """Bayesian AR(p) must expose a real p-lag coefficient tensor, not reject p>1."""
    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    panel = pd.DataFrame({"x": [0.0, 1.0, 0.2, 1.4, 0.3, 1.6, 0.4, 1.8]}, index=idx)
    model = BayesianMSVAR(states=["lag2_positive", "lag2_negative"], domains=["x"], p=2)
    model._posterior_intercepts = np.array([[0.0], [0.0]])
    model._posterior_coefficients = np.array(
        [
            [[[0.0]], [[1.0]]],
            [[[0.0]], [[-1.0]]],
        ],
        dtype=float,
    )
    model._posterior_covariances = np.array([[[0.05]], [[0.05]]], dtype=float)
    model._posterior_prior = np.array([0.5, 0.5], dtype=float)
    model._posterior_transition = np.array([[0.95, 0.05], [0.05, 0.95]], dtype=float)
    model.fitted = True

    scored = model.score(panel)
    assert len(scored) == len(panel)
    assert model._posterior_coefficients.shape == (2, 2, 1, 1)
    assert np.isfinite(scored["msvar_confidence"].iloc[2:]).all()


def test_exact_assignment_beats_greedy_counterexample() -> None:
    """A global minimum assignment is required for stable regime pinning."""
    cost = np.array(
        [
            [10.0, 1.0, 1.0],
            [1.0, 10.0, 1.0],
            [1.0, 1.0, 10.0],
        ]
    )
    assignment = _optimal_assignment(cost)
    assert sum(cost[i, assignment[i]] for i in range(3)) == pytest.approx(3.0)


def test_mq_dfm_custom_state_space_supports_daily_weekly_monthly_layout() -> None:
    """D/W/M panels must use the custom Kalman mixed-frequency backend."""
    idx = pd.date_range("2024-01-01", periods=75, freq="D")
    t = np.arange(len(idx), dtype=float)
    daily = np.sin(t / 8.0)
    weekly = pd.Series(np.nan, index=idx)
    weekly.iloc[6::7] = pd.Series(daily, index=idx).rolling(7, min_periods=1).mean().iloc[6::7]
    monthly = pd.Series(np.nan, index=idx)
    month_end_mask = idx.is_month_end
    monthly.loc[month_end_mask] = (
        pd.Series(daily, index=idx).groupby(idx.to_period("M")).transform("mean").loc[month_end_mask]
    )
    panel = pd.DataFrame({"daily": daily, "weekly": weekly, "monthly": monthly}, index=idx)

    model = MQDynamicFactorModel().fit(panel, frequencies={"daily": "D", "weekly": "W", "monthly": "M"})
    assert model.fitted is True
    assert model.backend == "custom_state_space"
    early = model.nowcast(idx[20])
    late = model.nowcast(idx[-1])
    assert early["backend"] == "custom_state_space"
    assert late["backend"] == "custom_state_space"
    assert np.isfinite(early["factor"])
    assert np.isfinite(late["factor_se"])


def test_mq_dfm_rejects_unsupported_quarterly_daily_mix() -> None:
    panel, _ = build_synthetic_panel(n_months=24, n_series=2, seed=7)
    with pytest.raises(ValueError, match=r"quarterly .* daily/weekly"):
        MQDynamicFactorModel().fit(panel, frequencies={panel.columns[0]: "D", panel.columns[1]: "Q"})


def test_msvar_stabilizes_explosive_var_coefficients() -> None:
    """Explosive M-step coefficients should be shrunk before promotion."""
    model = MarkovSwitchingVAR(states=["a", "b"], domains=["x"], p=1, stability_radius=0.95)
    coeffs = np.array([[[1.25]]], dtype=float)
    stabilized, original_radius, was_stabilized = model._stabilize_coefficients(coeffs)
    assert original_radius > 1.0
    assert was_stabilized is True
    assert model._companion_radius(stabilized) <= model.stability_radius + 1e-12


def test_msvar_covariance_shrinkage_handles_collinearity() -> None:
    """Perfectly collinear domains should not produce singular regime covariances."""
    idx = pd.date_range("2023-01-01", periods=80, freq="D")
    x = np.sin(np.arange(80) / 6.0)
    panel = pd.DataFrame({"x1": x, "x2": x}, index=idx)
    model = MarkovSwitchingVAR(states=["a", "b"], domains=["x1", "x2"], p=1, max_iter=5).fit(panel)
    assert model.fitted is True
    assert np.isfinite(model.covariances).all()
    eig_min = min(float(np.linalg.eigvalsh(cov).min()) for cov in model.covariances)
    assert eig_min > 0.0
    assert "max_covariance_shrinkage_intensity" in model.fit_log


def test_giacomini_white_conditional_path_returns_full_hac_sandwich_result() -> None:
    from market_regime_engine.forecast_compare import giacomini_white

    rng = np.random.default_rng(123)
    n = 80
    z = pd.DataFrame({"state": rng.normal(size=n)})
    loss_a = 1.0 + 0.15 * z["state"].to_numpy() + rng.normal(scale=0.1, size=n)
    loss_b = 1.0 + rng.normal(scale=0.1, size=n)
    out = giacomini_white(loss_a, loss_b, z=z, h=2)
    assert out["n"] == n
    assert out["df"] == 2
    assert out["covariance_estimator"] == "hac_sandwich"
    assert out["lag"] == 1
    cov = np.asarray(out["covariance"], dtype=float)
    assert cov.shape == (2, 2)
    assert np.isfinite(out["statistic"])
    assert 0.0 <= out["pvalue"] <= 1.0


def test_retrospective_frontier_paths_require_explicit_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    from market_regime_engine.frontier.dfm_mq import MQDynamicFactorModel
    from market_regime_engine.release_gates import evaluate_release_gate

    class _Factors:
        smoothed = pd.DataFrame({"f": [1.0, 2.0]})

    class _Results:
        factors = _Factors()

    monkeypatch.delenv("MRE_ENABLE_EXPERIMENTAL_FRONTIER", raising=False)
    with pytest.raises(RuntimeError, match="experimental frontier path disabled"):
        MQDynamicFactorModel._extract_factor_series(_Results(), filtered=False)

    confidence = pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "confidence": [0.99], "grade": ["A"]})
    promotion = pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "promoted": [True]})
    e_log = pd.DataFrame({"e_value": [100.0], "decision": ["promote"]})
    with pytest.raises(RuntimeError, match="experimental frontier path disabled"):
        evaluate_release_gate(
            confidence=confidence,
            promotion=promotion,
            coverage_report=pd.DataFrame({"coverage": [0.99]}),
            promotion_method="e_values",
            e_value_log=e_log,
            profile="default",
        )
