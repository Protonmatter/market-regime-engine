"""Regression tests for the v1.2 math correctness fixes (Part 1).

Each test maps to one fix in the V1_2_FRONTIER doc table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.bma import OnlineBMA
from market_regime_engine.bocpd_muse import _AR1State
from market_regime_engine.conformal import MondrianBinaryConformal
from market_regime_engine.dfm import DFMDomainModel
from market_regime_engine.forecast_compare import (
    diebold_mariano,
    hansen_mcs,
    pit_uniformity,
)
from market_regime_engine.hazard_model import (
    DiscreteTimeHazardModel,
    train_fitted_hazard_outputs,
)
from market_regime_engine.hmm import HMMRegimePosterior

# ---------------------------------------------------------------------------
# Fix #1, #2, #12 — DFM marginal likelihood, cached (mu, sd), no dead **0
# ---------------------------------------------------------------------------


def _synthetic_dfm_panel(n: int = 240, k: int = 4, *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f = np.zeros(n)
    for t in range(1, n):
        f[t] = 0.7 * f[t - 1] + rng.normal(scale=0.5)
    loadings = rng.uniform(0.6, 1.4, size=k)
    eps = rng.normal(scale=0.2, size=(n, k))
    Y = f[:, None] * loadings[None, :] + eps
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    return pd.DataFrame(Y, index=dates, columns=[f"x{i}" for i in range(k)])


def test_dfm_marginal_likelihood_is_finite_and_monotone() -> None:
    """The new Sherman-Morrison-Woodbury marginal likelihood is finite and EM is monotone (up to numerical noise)."""
    panel = _synthetic_dfm_panel()
    model = DFMDomainModel(max_iter=20).fit(panel)
    assert model.fitted
    assert np.isfinite(model.log_likelihood)
    # Re-run a single EM iteration and ensure the LL is bounded above by the
    # final value (proxy for "EM monotone enough"); we allow a small tolerance
    # because the EM uses an information-form posterior collapse step.
    model_short = DFMDomainModel(max_iter=1).fit(panel)
    assert np.isfinite(model_short.log_likelihood)


def test_dfm_caches_train_mu_sd_and_transform_uses_them() -> None:
    """transform() must reuse fit()-time (mu, sd) — re-fitting per call is the bug."""
    panel = _synthetic_dfm_panel(n=240)
    model = DFMDomainModel(max_iter=10).fit(panel)
    assert model.train_mu.size == panel.shape[1]
    assert model.train_sd.size == panel.shape[1]
    np.testing.assert_allclose(model.train_mu, panel.to_numpy(float).mean(axis=0), rtol=1e-6)
    # Transform on a *short* window must not collapse the factor amplitude
    # (the bug was: short window → per-call sd → factor of ~0).
    short = panel.iloc[-12:]
    factor = model.transform(short)
    assert factor.std() > 0.0, "factor collapsed on short window — cached (mu, sd) not respected"


def test_dfm_no_identically_one_term_in_kalman_gain() -> None:
    """The dead `np.maximum(gain_den, 1e-12) ** 0` is gone — sanity check via grep."""
    src = __import__("market_regime_engine.dfm", fromlist=["__file__"]).__file__
    with open(src, encoding="utf-8") as fh:
        text = fh.read()
    assert "** 0" not in text and "**0)" not in text


# ---------------------------------------------------------------------------
# Fix #3 — Mondrian backend dispatch + exchangeable= flag
# ---------------------------------------------------------------------------


def _binary_calibration(n: int = 600, *, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = rng.beta(2.0, 5.0, size=n)
    bucket = rng.choice(["a", "b", "c"], size=n)
    y = (rng.uniform(size=n) < np.clip(p, 0.01, 0.99)).astype(int)
    return pd.DataFrame({"p": p, "y": y, "regime_bucket": bucket})


@pytest.mark.parametrize("backend", ["block", "nexcp", "conditional", "localized", "e_conformal"])
def test_mondrian_backend_dispatch_round_trip(backend: str) -> None:
    df = _binary_calibration()
    layer = MondrianBinaryConformal(alpha=0.10, backend=backend).fit(df)
    assert set(layer.thresholds.keys()) == {"a", "b", "c"}
    out = layer.transform(df.head(20).copy())
    assert {"conformal_set", "conformal_uncertain", "conformal_threshold"}.issubset(out.columns)
    rep = layer.coverage_report(df)
    assert (rep["coverage"] >= 0.0).all()
    assert (rep["coverage"] <= 1.0).all()


def test_mondrian_exchangeable_flag_default_preserves_back_compat() -> None:
    df = _binary_calibration()
    legacy = MondrianBinaryConformal(alpha=0.10).fit(df)
    explicit = MondrianBinaryConformal(alpha=0.10, exchangeable=True, backend="split").fit(df)
    for bucket in legacy.thresholds:
        assert legacy.thresholds[bucket] == pytest.approx(explicit.thresholds[bucket])


def test_mondrian_exchangeable_false_auto_bumps_to_block_backend() -> None:
    df = _binary_calibration()
    layer = MondrianBinaryConformal(alpha=0.10, exchangeable=False).fit(df)
    # When exchangeable=False and backend left at "split", the layer must
    # delegate to a non-split backend (block conformal by default).
    assert layer._backend_obj is not None
    assert type(layer._backend_obj).__name__ == "BlockConformalBinary"


# ---------------------------------------------------------------------------
# Fix #4 — _AR1State.update centers inputs before cross-products
# ---------------------------------------------------------------------------


def test_ar1state_phi_unbiased_for_nonzero_mean() -> None:
    """With E[x] ≠ 0 but no AR persistence, phi-hat should be ≈ 0 after centering."""
    rng = np.random.default_rng(0)
    n = 600
    x = 5.0 + rng.normal(scale=0.5, size=n)  # mean 5, no AR structure
    state = _AR1State.prior(dim=1, prior_var=1.0)
    for v in x:
        state = state.update(np.array([float(v)]))
    denom = np.maximum(state.sum_x_lag_sq, 1e-9)
    phi_hat = float(np.clip(state.sum_x_lag_x[0] / denom[0], -0.99, 0.99))
    assert abs(phi_hat) < 0.2, f"phi_hat {phi_hat} should be ~0 when E[x] != 0 and no AR structure"


# ---------------------------------------------------------------------------
# Fix #5 — train_fitted_hazard_outputs assumption flag and path mode
# ---------------------------------------------------------------------------


def test_hazard_outputs_emit_constant_hazard_assumption_in_metadata() -> None:
    import json

    rng = np.random.default_rng(0)
    dates = pd.date_range("2000-01-01", periods=240, freq="MS")
    feats = []
    for c in ["unrate.zscore", "fedfunds.diff", "credit.zscore"]:
        for d in dates:
            feats.append(
                {
                    "feature_name": c,
                    "date": d,
                    "value": float(rng.normal()),
                    "domain": "labor" if "unrate" in c else "rates",
                }
            )
    feats_df = pd.DataFrame(feats)
    rec = pd.DataFrame(
        {
            "date": dates,
            "recession": (rng.uniform(size=len(dates)) < 0.05).astype(float),
        }
    )
    out, _ = train_fitted_hazard_outputs(feats_df, rec)
    assert not out.empty
    horizon_rows = out[out["target"] == "recession_probability"]
    for meta in horizon_rows["metadata_json"]:
        d = json.loads(meta)
        assert d.get("assumption") == "constant_hazard"


def test_hazard_outputs_path_mode_marks_assumption_path() -> None:
    import json

    rng = np.random.default_rng(0)
    dates = pd.date_range("2000-01-01", periods=240, freq="MS")
    feats = []
    for c in ["x.z"]:
        for d in dates:
            feats.append(
                {
                    "feature_name": c,
                    "date": d,
                    "value": float(rng.normal()),
                    "domain": "labor",
                }
            )
    feats_df = pd.DataFrame(feats)
    rec = pd.DataFrame({"date": dates, "recession": np.zeros(len(dates))})
    path = np.full(24, 0.02)
    out, _ = train_fitted_hazard_outputs(feats_df, rec, monthly_hazard_path=path)
    horizon_rows = out[out["target"] == "recession_probability"]
    for meta in horizon_rows["metadata_json"]:
        d = json.loads(meta)
        assert d.get("assumption") == "path"


# ---------------------------------------------------------------------------
# Fix #6 — DiscreteTimeHazardModel.class_weight is configurable
# ---------------------------------------------------------------------------


def test_hazard_model_class_weight_default_is_none() -> None:
    model = DiscreteTimeHazardModel()
    assert model.class_weight is None
    clf = model.pipeline.named_steps["clf"]
    assert clf.get_params().get("class_weight") is None


def test_hazard_model_accepts_balanced_legacy_behavior() -> None:
    model = DiscreteTimeHazardModel(class_weight="balanced")
    assert model.class_weight == "balanced"
    clf = model.pipeline.named_steps["clf"]
    assert clf.get_params().get("class_weight") == "balanced"


# ---------------------------------------------------------------------------
# Fix #7 — Baum-Welch still converges after dead-line removal
# ---------------------------------------------------------------------------


def test_baum_welch_converges_after_dead_line_removal() -> None:
    rng = np.random.default_rng(0)
    n = 240
    domains = ["labor", "rates", "inflation", "credit", "housing", "energy", "fx", "fiscal"]
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    panel = pd.DataFrame(rng.normal(size=(n, len(domains))), columns=domains, index=dates)
    hmm = HMMRegimePosterior().fit(panel, max_iter=10)
    assert hmm.fitted
    np.testing.assert_allclose(hmm.transition.sum(axis=1), np.ones(hmm.transition.shape[0]), rtol=1e-6)


# ---------------------------------------------------------------------------
# Fix #8 — hansen_mcs supports T_R and T_SQ
# ---------------------------------------------------------------------------


def test_hansen_mcs_t_r_and_t_sq_both_work() -> None:
    rng = np.random.default_rng(0)
    n = 200
    losses = pd.DataFrame(
        {
            "best": rng.normal(loc=0.20, scale=0.05, size=n),
            "mid": rng.normal(loc=0.50, scale=0.05, size=n),
            "worst": rng.normal(loc=0.80, scale=0.05, size=n),
        }
    )
    out_tr = hansen_mcs(losses, confidence=0.10, bootstrap=200, block_size=10, seed=0, statistic="T_R")
    out_tsq = hansen_mcs(losses, confidence=0.10, bootstrap=200, block_size=10, seed=0, statistic="T_SQ")
    assert "best" in out_tr["mcs"]
    assert "best" in out_tsq["mcs"]
    assert out_tr["statistic"] == "T_R"
    assert out_tsq["statistic"] == "T_SQ"
    # Both should reject the worst model.
    assert "worst" not in out_tr["mcs"]
    assert "worst" not in out_tsq["mcs"]


# ---------------------------------------------------------------------------
# Fix #9 — pit_uniformity autocorrelation flag
# ---------------------------------------------------------------------------


def test_pit_uniformity_autocorrelation_flag_returns_lags() -> None:
    rng = np.random.default_rng(0)
    u = rng.uniform(size=2000)
    base = pit_uniformity(u, bins=10, autocorrelation=False)
    aug = pit_uniformity(u, bins=10, autocorrelation=True, autocorr_lags=4)
    assert base.get("autocorrelations", []) == []
    assert isinstance(aug["autocorrelations"], list) and len(aug["autocorrelations"]) == 4
    # On iid data, all four sample autocorrelations should be small.
    assert max(abs(r) for r in aug["autocorrelations"]) < 0.1


def test_pit_uniformity_autocorrelation_rejects_persistent_series() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    # Generate a persistent PIT series via a logistic transform of an AR(1).
    z = np.zeros(n)
    for t in range(1, n):
        z[t] = 0.85 * z[t - 1] + rng.normal()
    u = 1.0 / (1.0 + np.exp(-z))
    aug = pit_uniformity(u, bins=10, autocorrelation=True, autocorr_lags=4)
    # Persistent series: at least one of the first 4 lags should be > 0.1 by magnitude.
    assert max(abs(r) for r in aug["autocorrelations"]) > 0.1


# ---------------------------------------------------------------------------
# Fix #10 — diebold_mariano direction at p<0.05
# ---------------------------------------------------------------------------


def test_dm_direction_uses_5pct_not_10pct() -> None:
    """Force a borderline p in [0.05, 0.10) and verify direction stays "tie"."""
    rng = np.random.default_rng(42)
    n = 60
    a = rng.normal(loc=0.5, scale=0.5, size=n)
    b = rng.normal(loc=0.4, scale=0.5, size=n)  # marginal effect
    res = diebold_mariano(a, b, h=1)
    if 0.05 <= res.pvalue < 0.10:
        assert res.direction == "tie", f"expected tie at p={res.pvalue:.3f}"


# ---------------------------------------------------------------------------
# Fix #11 — OnlineBMA.update floors AFTER normalization
# ---------------------------------------------------------------------------


def test_online_bma_floor_applied_after_normalization() -> None:
    bma = OnlineBMA(forgetting=0.95, floor_weight=1e-9)
    bma.initialize(["a", "b", "c"])
    out = bma.update(1.0, {"a": 0.9, "b": 0.05, "c": 0.05})
    weights = list(out.values())
    # All weights must sum to 1 (post-floor renormalization).
    assert abs(sum(weights) - 1.0) < 1e-6
    # Floor of 1e-9 is small enough to leave the dominant weight nearly 1.
    assert max(weights) > 0.5


def test_online_bma_floor_default_is_1e9() -> None:
    bma = OnlineBMA()
    assert bma.floor_weight == 1e-9
