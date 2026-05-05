"""Tests for orchestration, observability, scenarios, counterfactual, multi-horizon conformal, and api_v1."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.config import load_catalog
from market_regime_engine.counterfactual import (
    counterfactual_delta,
    permutation_attribution,
    shap_attribution_if_available,
)
from market_regime_engine.multi_horizon_conformal import (
    AdaptiveMultiHorizonConformal,
    BonferroniMultiHorizonConformal,
)
from market_regime_engine.observability import metrics, prometheus_text, time_block
from market_regime_engine.orchestration import daily_flow
from market_regime_engine.sample import generate_sample_observations
from market_regime_engine.scenarios import SCENARIOS, replay_all, replay_scenario
from market_regime_engine.storage import Warehouse

# ---------------------------------------------------------------------------
# observability
# ---------------------------------------------------------------------------


def test_metrics_record_counter_and_histogram():
    metrics().incr("test_counter_total", value=2.0, tag="a")
    with time_block("test_block_seconds", tag="a"):
        pass
    snap = metrics().snapshot()
    assert any("test_counter_total" in k for k in snap["counters"])
    assert any("test_block_seconds" in k for k in snap["histograms"])


def test_prometheus_text_emits_some_lines():
    metrics().incr("mre_test_metric_total", value=1.0)
    text = prometheus_text()
    assert "mre_test_metric_total" in text


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------


def test_scenarios_replay_runs_on_sample():
    obs = generate_sample_observations()
    catalog = load_catalog()
    results = replay_all(obs, catalog, only=["gfc", "covid_shock"])
    assert len(results) == 2
    for r in results:
        d = r.to_dict()
        assert d["rows"] > 0
        assert d["cp_max"] >= 0.0


def test_replay_scenario_handles_out_of_range_dates():
    obs = generate_sample_observations()
    catalog = load_catalog()
    res = replay_scenario(obs, catalog, SCENARIOS[0])  # 1973 oil shock — sample data starts 1990
    assert res.rows == 0


# ---------------------------------------------------------------------------
# counterfactual
# ---------------------------------------------------------------------------


def test_counterfactual_delta_returns_per_feature_rows():
    rng = np.random.default_rng(0)
    n = 60
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)}, index=idx)
    coefs = np.array([0.5, -0.3])

    def predict_fn(df: pd.DataFrame) -> np.ndarray:
        return df.to_numpy(float) @ coefs

    out = counterfactual_delta(predict_fn, X, columns=["a", "b"])
    assert not out.empty
    assert set(out["feature"].unique()) == {"a", "b"}


def test_permutation_attribution_returns_per_feature_rows():
    rng = np.random.default_rng(0)
    n = 40
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)}, index=idx)

    def predict_fn(df: pd.DataFrame) -> np.ndarray:
        return df["a"].to_numpy() * 0.5 + df["b"].to_numpy() * 0.1

    out = permutation_attribution(predict_fn, X, n_samples=4)
    assert not out.empty
    assert set(out["feature"].unique()) == {"a", "b"}


def test_shap_attribution_if_available_handles_missing_dependency():
    # If shap isn't installed it should return an empty frame (no exception).
    out = shap_attribution_if_available(estimator=None, X=pd.DataFrame())
    assert out.empty


# ---------------------------------------------------------------------------
# multi-horizon conformal
# ---------------------------------------------------------------------------


def test_bonferroni_multi_horizon_alpha_per_horizon():
    layer = BonferroniMultiHorizonConformal(horizons=("3m", "6m", "12m"), alpha=0.12)
    assert abs(layer.per_horizon_alpha - 0.04) < 1e-9


def test_bonferroni_multi_horizon_fits_each_horizon():
    rng = np.random.default_rng(0)
    n = 400
    cal = {}
    for h in ("3m", "6m", "12m"):
        y = rng.normal(scale=1.0, size=n)
        cal[h] = pd.DataFrame({"y": y, "q_lo": -0.5 * np.ones(n), "q_hi": 0.5 * np.ones(n)})
    layer = BonferroniMultiHorizonConformal(alpha=0.30).fit(cal)
    for h in ("3m", "6m", "12m"):
        assert h in layer.cqrs
        assert layer.cqrs[h].fitted_n > 0


def test_adaptive_multi_horizon_initialises_and_steps():
    layer = AdaptiveMultiHorizonConformal(horizons=("3m", "6m"), alpha=0.10)
    layer.initialize()
    rng = np.random.default_rng(0)
    history = {h: pd.DataFrame({"y": rng.normal(size=80), "q_lo": -0.5, "q_hi": 0.5}) for h in ("3m", "6m")}
    realized = {h: (-0.5, 0.5, float(rng.normal())) for h in ("3m", "6m")}
    out = layer.step(history, realized)
    assert "alpha" in out and "inflation" in out
    assert set(out["alpha"].keys()) == {"3m", "6m"}


# ---------------------------------------------------------------------------
# orchestration end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_daily_flow_runs_on_synthetic_pipeline(tmp_path):
    db_path = tmp_path / "mre.db"
    db = Warehouse(db_path)
    try:
        db.write_observations(generate_sample_observations())
        # Seed the vintage tables so the asof step can run.
        from market_regime_engine.alfred_real import seed_vintage_observations_from_latest

        vintages, vintage_obs = seed_vintage_observations_from_latest(db.read_observations())
        db.write_series_vintages(vintages)
        db.write_vintage_observations(vintage_obs)
    finally:
        db.close()
    summary = daily_flow(db_path=str(db_path), validation_dir=str(tmp_path / "validation"), enforce_audit=False)
    assert summary["regime_rows"] > 0
    assert "run_id" in summary
    assert "envelope" in summary


# ---------------------------------------------------------------------------
# api_v1 sanity check
# ---------------------------------------------------------------------------


def test_api_v1_health_endpoint_does_not_require_key(monkeypatch):
    monkeypatch.delenv("MRE_API_KEY", raising=False)
    from fastapi.testclient import TestClient

    from market_regime_engine.api_v1 import app

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "ok"
    assert "version" in payload


def test_api_v1_metrics_endpoint_returns_text(monkeypatch):
    monkeypatch.delenv("MRE_API_KEY", raising=False)
    from fastapi.testclient import TestClient

    from market_regime_engine.api_v1 import app

    client = TestClient(app)
    resp = client.get("/v1/metrics")
    assert resp.status_code == 200
    assert "counter" in resp.text or "histogram" in resp.text or "#" in resp.text
