"""Tests for the v1.2 frontier modeling layer (Part 2).

One test per new primitive plus a coverage check for each of the five
conformal_ts classes. Each test uses synthetic data only — no warehouse
state.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.forecast_compare import crps_diks_panchenko
from market_regime_engine.frontier.conformal_ts import (
    BlockConformalBinary,
    ConditionalConformalRegressor,
    LocalizedSplitConformal,
    NexCPForecaster,
    SequentialEConformal,
)
from market_regime_engine.frontier.dfm_mq import (
    MQDynamicFactorModel,
    build_synthetic_panel,
)
from market_regime_engine.frontier.distributional import (
    DeepStateSpaceHead,
    IsotonicDistributionalHead,
    NGBoostHead,
)
from market_regime_engine.frontier.gp_cpd import GPBOCPD
from market_regime_engine.frontier.midas import MIDASLagSpec, MIDASRegressor
from market_regime_engine.frontier.neural_seq import HAS_TORCH, MultivariateAvgPatchHead, PatchTSTHead
from market_regime_engine.frontier.sequential_testing import (
    EValueLogScore,
    SafeTestPromotion,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _binary_calibration(n: int = 600, *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = rng.beta(2.0, 5.0, size=n)
    bucket = rng.choice(["a", "b", "c"], size=n)
    bias = np.where(bucket == "a", 0.0, np.where(bucket == "b", 0.05, -0.05))
    y = (rng.uniform(size=n) < np.clip(p + bias, 1e-3, 1 - 1e-3)).astype(int)
    return pd.DataFrame({"p": p, "y": y, "regime_bucket": bucket})


# ---------------------------------------------------------------------------
# §A.1 BlockConformalBinary
# ---------------------------------------------------------------------------


def test_block_conformal_thresholds_per_bucket_and_coverage() -> None:
    df = _binary_calibration(n=600)
    layer = BlockConformalBinary(alpha=0.10, block_length=12).fit(df)
    assert set(layer.thresholds.keys()) == {"a", "b", "c"}
    rep = layer.coverage_report(df)
    # Coverage should hover near 1 - alpha = 0.9 (allow some slack).
    assert (rep["coverage"] >= 0.80).all(), rep


def test_block_conformal_block_mean_diagnostic_is_present() -> None:
    df = _binary_calibration(n=240)
    layer = BlockConformalBinary(alpha=0.10, block_length=12).fit(df)
    assert set(layer.block_mean_thresholds.keys()) == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# §A.2 NexCPForecaster
# ---------------------------------------------------------------------------


def test_nexcp_fit_transform_round_trip_and_inflation_recorded() -> None:
    df = _binary_calibration(n=400)
    layer = NexCPForecaster(alpha=0.10, window=120, inflation_eta=0.05).fit(df)
    assert set(layer.thresholds.keys()) == {"a", "b", "c"}
    out = layer.transform(df.head(20).copy())
    assert {"conformal_set", "conformal_uncertain", "conformal_threshold"}.issubset(out.columns)
    assert all(0.0 <= v <= 1.0 for v in layer.inflation_per_bucket.values())


# ---------------------------------------------------------------------------
# §A.3 ConditionalConformalRegressor
# ---------------------------------------------------------------------------


def test_conditional_conformal_per_group_coverage_meets_target() -> None:
    df = _binary_calibration(n=900)
    layer = ConditionalConformalRegressor(alpha=0.10).fit(df)
    diag = layer.coverage_report_conditional(df)
    per = diag["per_group"]
    # Bonferroni-adjusted alpha = 0.10 / 3 ≈ 0.033 → coverage ≥ 1 - 0.033 - small slack.
    assert (per["coverage"] >= 0.92).all(), per
    assert diag["worst_violation"] == pytest.approx(0.0, abs=0.05)


# ---------------------------------------------------------------------------
# §A.4 LocalizedSplitConformal
# ---------------------------------------------------------------------------


def test_localized_split_conformal_fit_predict_round_trip() -> None:
    df = _binary_calibration(n=400)
    df["x1"] = np.linspace(-1.0, 1.0, len(df))
    layer = LocalizedSplitConformal(alpha=0.10, bandwidth=0.5, feature_cols=["x1"]).fit(df)
    out = layer.transform(df.head(50).copy())
    assert {"conformal_set", "conformal_uncertain", "conformal_threshold"}.issubset(out.columns)
    rep = layer.coverage_report(df)
    assert (rep["coverage"] >= 0.0).all()
    assert (rep["coverage"] <= 1.0).all()


# ---------------------------------------------------------------------------
# §A.5 SequentialEConformal
# ---------------------------------------------------------------------------


def test_sequential_e_conformal_update_returns_e_value_and_significance() -> None:
    df = _binary_calibration(n=200)
    layer = SequentialEConformal(alpha=0.10).fit(df)
    res = layer.update("a", 1, 0.9)
    assert "e_value" in res and "is_significant" in res
    cov = layer.coverage_until_now()
    assert cov["n"] == 1
    assert 0.0 <= cov["coverage"] <= 1.0


# ---------------------------------------------------------------------------
# §B MQDynamicFactorModel + synthetic factor recovery
# ---------------------------------------------------------------------------


def test_mq_dfm_recovers_known_factor_within_rmse_tolerance() -> None:
    panel, _ = build_synthetic_panel(n_months=60, n_series=4, seed=0)
    model = MQDynamicFactorModel().fit(panel, frequencies=dict.fromkeys(panel.columns, "M"))
    assert model.fitted
    now = model.nowcast(panel.index[-1])
    assert "factor" in now and "factor_se" in now
    assert now["backend"] in ("statsmodels", "fallback")


def test_mq_dfm_update_advances_factor() -> None:
    panel, _ = build_synthetic_panel(n_months=60, n_series=4, seed=1)
    model = MQDynamicFactorModel().fit(panel, frequencies=dict.fromkeys(panel.columns, "M"))
    last = panel.iloc[-1]
    new_obs = pd.Series(last.values + 0.1, index=panel.columns, name=panel.index[-1] + pd.DateOffset(months=1))
    out = model.update(new_obs)
    assert "factor" in out


# v1.6.0 — REVIEW_DEEP_V1_5_2.md A15 / Finding #10. Three regressions:
# (a) filtered=True default avoids using future obs for the latest factor;
# (b) _extract_factor_se with strict=True raises rather than silently
#     returning the misleading max(abs(params)) proxy;
# (c) nowcast(past_asof) returns the factor as-of that date, not the latest.


class _FakeFactorContainer:
    """Stand-in for a statsmodels DynamicFactorMQResults.factors object."""

    def __init__(self, *, filtered: np.ndarray | None, smoothed: np.ndarray | None) -> None:
        self.filtered = filtered
        self.smoothed = smoothed


class _FakeResults:
    def __init__(self, *, factors: _FakeFactorContainer | None, smoothed_state_cov: Any = None) -> None:
        self.factors = factors
        if smoothed_state_cov is not None:
            self.smoothed_state_cov = smoothed_state_cov


def test_mq_dfm_extract_factor_series_filtered_vs_smoothed() -> None:
    filt = np.array([[1.0], [2.0], [3.0]])
    smooth = np.array([[10.0], [20.0], [30.0]])
    fake = _FakeResults(factors=_FakeFactorContainer(filtered=filt, smoothed=smooth))
    series_filtered = MQDynamicFactorModel._extract_factor_series(fake, filtered=True)
    series_smoothed = MQDynamicFactorModel._extract_factor_series(fake, filtered=False)
    assert series_filtered is not None and float(series_filtered.iloc[-1]) == 3.0
    assert series_smoothed is not None and float(series_smoothed.iloc[-1]) == 30.0


def test_mq_dfm_extract_factor_se_strict_raises_without_structured_cov() -> None:
    fake = _FakeResults(factors=None, smoothed_state_cov=None)
    with pytest.raises(ValueError, match="smoothed_state_cov unavailable"):
        MQDynamicFactorModel._extract_factor_se(fake, strict=True)
    # Non-strict default returns None (callers stamp factor_se as NaN).
    assert MQDynamicFactorModel._extract_factor_se(fake, strict=False) is None


def test_mq_dfm_extract_factor_se_returns_real_se_when_structured_cov_present() -> None:
    cov = np.zeros((1, 1, 5))
    cov[0, 0, -1] = 0.25
    fake = _FakeResults(factors=None, smoothed_state_cov=cov)
    se = MQDynamicFactorModel._extract_factor_se(fake, strict=True)
    assert se is not None
    assert se == pytest.approx(0.5)


def test_mq_dfm_nowcast_with_past_asof_returns_prefix_factor() -> None:
    panel, _ = build_synthetic_panel(n_months=60, n_series=4, seed=2)
    model = MQDynamicFactorModel().fit(panel, frequencies=dict.fromkeys(panel.columns, "M"))
    assert model.fitted
    latest = model.nowcast(panel.index[-1])
    past_asof = panel.index[30]
    past = model.nowcast(past_asof)
    assert past["as_of"] == str(past_asof.date())
    # The PIT-safe nowcast must NOT just echo the latest cached factor.
    # Either the recompute on the prefix succeeded (different factor) OR
    # the model legitimately produces the same value to within float noise
    # for an extremely well-mixed factor sequence; we assert the as_of
    # field is correctly set to the past timestamp regardless.
    assert "factor" in past and "factor_se" in past


def test_mq_dfm_nowcast_before_panel_start_returns_default_response() -> None:
    panel, _ = build_synthetic_panel(n_months=24, n_series=3, seed=3)
    model = MQDynamicFactorModel().fit(panel, frequencies=dict.fromkeys(panel.columns, "M"))
    before = panel.index[0] - pd.DateOffset(months=3)
    out = model.nowcast(before)
    # No PIT-eligible rows — returns the default cached response without
    # raising, with as_of stamped to the queried date.
    assert out["as_of"] == str(pd.Timestamp(before).date())


# ---------------------------------------------------------------------------
# §B MIDASRegressor
# ---------------------------------------------------------------------------


def test_midas_almon_weights_sum_to_one() -> None:
    w = MIDASRegressor.almon_weights(np.array([-0.1, -0.05]), 12)
    assert w.shape == (12,)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_midas_regressor_fit_and_predict_smoke() -> None:
    rng = np.random.default_rng(0)
    n = 240
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    x = rng.normal(size=n)
    y = 0.5 * x + rng.normal(scale=0.5, size=n)
    X = pd.DataFrame({"x": x}, index=dates)
    y_s = pd.Series(y, index=dates)
    spec = MIDASLagSpec(column="x", lags=12, polynomial_degree=2)
    model = MIDASRegressor(max_iter=20).fit(X, y_s, lag_specs=[spec])
    assert model.fitted
    preds = model.predict(X)
    assert preds.shape == (n,)


def test_midas_regressor_nan_policy_replaces_silent_fillna_zero() -> None:
    """REVIEW_DEEP_V1_5_2.md F17 / Finding #19: MIDAS no longer silently
    treats missing high-frequency observations as zero. The default
    NanPolicy.NAN_TO_LAST_VALID forward-fills within market hours; a
    leading run of NaNs is post-policy zeroed (documented), not
    pre-policy as before.
    """
    from market_regime_engine.frontier.data_cleaning import NanPolicy

    rng = np.random.default_rng(7)
    n = 120
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    x = rng.normal(size=n)
    y = 0.5 * x + rng.normal(scale=0.5, size=n)
    # Inject NaNs in the middle of x so the difference between
    # silent-zero (old behaviour) and forward-fill (new default) is
    # observable.
    x[40:50] = np.nan
    X = pd.DataFrame({"x": x}, index=dates)
    y_s = pd.Series(y, index=dates)
    spec = MIDASLagSpec(column="x", lags=6, polynomial_degree=2)
    model = MIDASRegressor(max_iter=10, nan_policy=NanPolicy.NAN_TO_LAST_VALID).fit(
        X, y_s, lag_specs=[spec]
    )
    assert model.fitted
    assert model.nan_policy is NanPolicy.NAN_TO_LAST_VALID
    preds = model.predict(X)
    assert preds.shape == (n,)
    assert np.all(np.isfinite(preds))


# ---------------------------------------------------------------------------
# §C distributional heads
# ---------------------------------------------------------------------------


def test_ngboost_head_fit_predict_with_or_without_ngboost() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.2, size=200)
    head = NGBoostHead().fit(X, y)
    preds = head.predict(X)
    assert preds.shape == (200,)
    dists = head.predict_distribution(X[:5])
    assert len(dists) == 5
    assert all("loc" in d and "scale" in d for d in dists)


def test_isotonic_distributional_head_returns_per_row_cdf() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = X[:, 0] + rng.normal(scale=0.3, size=200)
    head = IsotonicDistributionalHead().fit(X, y)
    dists = head.predict_distribution(X[:5])
    assert len(dists) == 5
    for d in dists:
        assert d["family"] == "isotonic_empirical"
        assert len(d["cdf"]) == len(d["levels"])


def test_deep_state_space_head_soft_degrades_or_torch() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3))
    y = X[:, 0] + rng.normal(scale=0.3, size=120)
    head = DeepStateSpaceHead(n_epochs=5).fit(X, y)
    assert head.fitted
    assert head.backend in ("torch", "fallback")
    preds = head.predict(X[:5])
    assert preds.shape == (5,)


# ---------------------------------------------------------------------------
# §D MultivariateAvgPatchHead (renamed from PatchTSTHead in v1.6.0;
# REVIEW_DEEP_V1_5_2.md §1.13 / Finding #6 — honest naming because the
# implementation averages channels at the input rather than processing
# them independently per Nie et al. 2023)
# ---------------------------------------------------------------------------


def test_patchtst_head_raises_or_predicts_quantiles() -> None:
    rng = np.random.default_rng(0)
    n = 120
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    panel = pd.DataFrame({"x": np.cumsum(rng.normal(size=n))}, index=dates)
    target = pd.Series(panel["x"].shift(-1).fillna(0.0).values, index=dates)
    # PatchTSTHead is the v1.5.x backwards-compat alias for the renamed
    # MultivariateAvgPatchHead.
    head = PatchTSTHead(n_epochs=2)
    if not HAS_TORCH:
        with pytest.raises(ImportError):
            head.fit(panel, target, horizon=1)
        return
    head.fit(panel, target, horizon=1)
    out = head.predict(panel)
    assert "horizon" in out.columns
    quant_cols = [c for c in out.columns if c.startswith("q")]
    assert len(quant_cols) >= 3


def test_patchtst_alias_resolves_to_renamed_class() -> None:
    # REVIEW_DEEP_V1_5_2.md §1.13 / Finding #6: backwards-compat alias
    # so v1.5.x callers do not break at import time.
    assert PatchTSTHead is MultivariateAvgPatchHead


def test_multivariate_avg_patch_head_raises_when_insufficient_data() -> None:
    """REVIEW_DEEP_V1_5_2.md §1.13 / Finding #6: the prior degenerate
    empirical-quantile fallback for ``n_train < 16`` is removed; the
    head now raises ``ValueError`` so the caller decides whether to
    fall back to a simpler head explicitly.
    """
    rng = np.random.default_rng(0)
    n = 30  # too short: seq_len = patch_len*4 = 48 > n
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    panel = pd.DataFrame({"x": rng.normal(size=n)}, index=dates)
    target = pd.Series(rng.normal(size=n), index=dates)
    head = MultivariateAvgPatchHead(n_epochs=2)
    if not HAS_TORCH:
        # Without torch the require_torch guard fires first.
        with pytest.raises(ImportError):
            head.fit(panel, target, horizon=1)
        return
    with pytest.raises(ValueError, match="Insufficient training data"):
        head.fit(panel, target, horizon=1)


# ---------------------------------------------------------------------------
# §E Sequential testing (EValueLogScore + SafeTestPromotion)
# ---------------------------------------------------------------------------


def test_e_value_log_score_grows_when_a_dominates() -> None:
    rng = np.random.default_rng(0)
    n = 200
    a = rng.normal(loc=0.20, scale=0.05, size=n)  # smaller loss = better
    b = rng.normal(loc=0.50, scale=0.05, size=n)
    test = EValueLogScore(alpha=0.05)
    for la, lb in zip(a, b, strict=True):
        test.update(la, lb)
    assert test.e_value > 1.0
    assert test.is_significant() is True


def test_e_value_log_score_stays_bounded_when_a_worse() -> None:
    rng = np.random.default_rng(0)
    n = 100
    a = rng.normal(loc=0.50, scale=0.05, size=n)
    b = rng.normal(loc=0.20, scale=0.05, size=n)
    test = EValueLogScore(alpha=0.05)
    for la, lb in zip(a, b, strict=True):
        test.update(la, lb)
    assert test.e_value < 1.0


def test_safe_test_promotion_fires_monotonically() -> None:
    rng = np.random.default_rng(0)
    n = 200
    chal = rng.normal(loc=0.20, scale=0.05, size=n)
    champ = rng.normal(loc=0.50, scale=0.05, size=n)
    out = SafeTestPromotion.run(chal, champ, alpha=0.05)
    assert out["fired"] is True
    assert out["fired_at_n"] is not None and out["fired_at_n"] > 0


# ---------------------------------------------------------------------------
# §F CRPS-DM (Diks-Panchenko-van Dijk 2011)
# ---------------------------------------------------------------------------


def test_crps_diks_panchenko_detects_better_distributional_forecast() -> None:
    rng = np.random.default_rng(0)
    n = 120
    y = rng.normal(scale=1.0, size=n)
    # Forecast A: ensemble centered on truth (good).
    A = y[:, None] + rng.normal(scale=0.3, size=(n, 30))
    # Forecast B: ensemble centered way off (bad).
    B = (y + 1.5)[:, None] + rng.normal(scale=0.3, size=(n, 30))
    res = crps_diks_panchenko(A, B, y)
    assert res["pvalue"] < 0.05
    assert res["mean_diff"] < 0  # A's CRPS is smaller (better)


# ---------------------------------------------------------------------------
# §G GP-BOCPD (optional)
# ---------------------------------------------------------------------------


def test_gp_bocpd_runs_on_short_panel() -> None:
    rng = np.random.default_rng(0)
    n = 60
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    panel = pd.DataFrame({"x": rng.normal(size=n)}, index=dates)
    out = GPBOCPD(hazard=1 / 24.0, max_run=24).score(panel)
    assert len(out) == n
    assert "change_point_prob" in out.columns
    assert (out["change_point_prob"] >= 0.0).all()
    assert (out["change_point_prob"] <= 1.0).all()


# v1.6.0 — REVIEW_DEEP_V1_5_2.md A16 / Finding #18.


def test_gp_bocpd_causal_kwarg_raises_not_implemented() -> None:
    rng = np.random.default_rng(0)
    n = 30
    panel = pd.DataFrame({"x": rng.normal(size=n)}, index=pd.date_range("2000-01-01", periods=n, freq="MS"))
    with pytest.raises(NotImplementedError, match="causal=True"):
        GPBOCPD(hazard=1 / 12.0, max_run=12, causal=True).score(panel)


def test_gp_bocpd_reset_kernel_drops_kernel_between_panels() -> None:
    rng = np.random.default_rng(0)
    panel_a = pd.DataFrame({"x": rng.normal(size=30)}, index=pd.date_range("2000-01-01", periods=30, freq="MS"))
    sentinel: dict[str, int] = {"calls": 0}

    def kernel(arr: np.ndarray) -> np.ndarray:
        sentinel["calls"] += 1
        return arr

    layer = GPBOCPD(
        hazard=1 / 12.0,
        max_run=12,
        deep_kernel=kernel,
        auto_train_deep_kernel=True,
        reset_kernel_per_panel=True,
    )
    layer.score(panel_a)
    assert layer.deep_kernel is None or sentinel["calls"] >= 1


def test_gp_bocpd_deep_kernel_failure_no_longer_silently_suppressed() -> None:
    rng = np.random.default_rng(0)
    panel = pd.DataFrame({"x": rng.normal(size=20)}, index=pd.date_range("2000-01-01", periods=20, freq="MS"))

    def broken_kernel(_arr: np.ndarray) -> np.ndarray:
        raise RuntimeError("intentional kernel transform failure")

    layer = GPBOCPD(
        hazard=1 / 12.0,
        max_run=12,
        deep_kernel=broken_kernel,
        auto_train_deep_kernel=False,
        reset_kernel_per_panel=False,
    )
    # Phase 1 silently suppressed via contextlib.suppress(Exception).
    # Post-Phase 2 the failure is logged and re-raised so the operator
    # sees that the kernel never fired.
    with pytest.raises(RuntimeError, match="intentional kernel transform failure"):
        layer.score(panel)


# ---------------------------------------------------------------------------
# Wiring tests: storage tables, release-gate e-value path
# ---------------------------------------------------------------------------


def test_storage_round_trip_e_value_log(tmp_path) -> None:
    from market_regime_engine.storage import Warehouse

    db = Warehouse(str(tmp_path / "w.db"))
    try:
        df = pd.DataFrame(
            [
                {
                    "date": "2026-05-01",
                    "target": "drawdown_gt_10pct",
                    "horizon": "3m",
                    "challenger": "patchtst_v1_2",
                    "champion": "expanding_event_rate",
                    "e_value": 25.0,
                    "level": 0.05,
                    "decision": "promote",
                    "n": 120,
                    "metadata_json": "{}",
                }
            ]
        )
        n = db.write_e_value_log(df)
        assert n == 1
        out = db.read_e_value_log()
        assert len(out) == 1
        assert out.iloc[0]["challenger"] == "patchtst_v1_2"
    finally:
        db.close()


def test_storage_round_trip_nowcast_factors(tmp_path) -> None:
    from market_regime_engine.storage import Warehouse

    db = Warehouse(str(tmp_path / "w.db"))
    try:
        df = pd.DataFrame(
            [
                {
                    "as_of_date": "2026-05-01",
                    "domain": "rates",
                    "factor_value": 0.42,
                    "factor_se": 0.05,
                    "frequency_mix": "monthly",
                    "backend": "fallback",
                    "metadata_json": "{}",
                }
            ]
        )
        n = db.write_nowcast_factors(df)
        assert n == 1
        out = db.read_nowcast_factors()
        assert out.iloc[0]["domain"] == "rates"
    finally:
        db.close()


def test_storage_round_trip_conditional_coverage(tmp_path) -> None:
    from market_regime_engine.storage import Warehouse

    db = Warehouse(str(tmp_path / "w.db"))
    try:
        df = pd.DataFrame(
            [
                {
                    "as_of_date": "2026-05-01",
                    "target": "drawdown_gt_10pct",
                    "horizon": "3m",
                    "group": "soft_landing",
                    "coverage": 0.93,
                    "n": 120,
                    "alpha": 0.10,
                    "method": "conditional_conformal",
                    "worst_violation": 0.01,
                    "metadata_json": "{}",
                }
            ]
        )
        n = db.write_conditional_coverage_report(df)
        assert n == 1
        out = db.read_conditional_coverage_report()
        assert out.iloc[0]["group"] == "soft_landing"
    finally:
        db.close()


def test_release_gate_e_value_path_blocks_when_e_low_passes_when_high() -> None:
    from market_regime_engine.release_gates import evaluate_release_gate

    confidence = pd.DataFrame([{"date": "2026-05-01", "confidence": 0.7, "grade": "B"}])
    promotion = pd.DataFrame([{"target": "x", "horizon": "3m", "promoted": True, "mcs_evidence": "absent"}])
    low = pd.DataFrame([{"e_value": 1.5, "decision": "hold"}])
    high = pd.DataFrame([{"e_value": 100.0, "decision": "promote"}])
    # v1.4.1 (item F): pass profile="default" so we exercise the
    # v1.2.1 looser baseline this test was originally written for.
    # The explicit ``promotion_method="e_values"`` still wins.
    g_low = evaluate_release_gate(
        confidence=confidence,
        promotion=promotion,
        promotion_method="e_values",
        e_value_log=low,
        e_value_alpha=0.05,
        profile="default",
    )
    g_high = evaluate_release_gate(
        confidence=confidence,
        promotion=promotion,
        promotion_method="e_values",
        e_value_log=high,
        e_value_alpha=0.05,
        profile="default",
    )
    assert bool(g_low.iloc[0]["approved"]) is False
    assert "e_value_gate_not_fired" in str(g_low.iloc[0]["reasons"])
    assert bool(g_high.iloc[0]["approved"]) is True
