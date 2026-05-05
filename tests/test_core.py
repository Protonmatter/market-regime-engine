import numpy as np
import pandas as pd

from market_regime_engine.changepoint import RollingMultivariateChangePoint
from market_regime_engine.ensemble import dynamic_weights
from market_regime_engine.features import build_features, monthly_panel
from market_regime_engine.wfst import RegimeWFST


def test_monthly_panel_and_features():
    obs = pd.DataFrame(
        [
            {"series_id": "A", "date": "2020-01-01", "value": 1.0, "vintage_date": "2020-01-10", "source": "x"},
            {"series_id": "A", "date": "2020-02-01", "value": 2.0, "vintage_date": "2020-02-10", "source": "x"},
        ]
    )
    panel = monthly_panel(obs)
    features = build_features(panel, [{"series_id": "A", "domain": "test"}])
    assert "A.level" in set(features["feature_name"])


def test_changepoint_runs():
    idx = pd.date_range("2020-01-01", periods=60, freq="MS")
    x = pd.DataFrame({"a": np.r_[np.zeros(40), np.ones(20) * 5], "b": np.r_[np.zeros(40), np.ones(20) * -3]}, index=idx)
    out = RollingMultivariateChangePoint().score(x)
    assert len(out) == 60
    assert out["change_point_prob"].between(0, 1).all()


def test_wfst_decode():
    dec = RegimeWFST()
    obs = ["risk_on_expansion", "late_cycle", "sticky_inflation", "energy_shock"]
    assert len(dec.decode(obs)) == len(obs)


def test_wfst_learns_costs_from_transition_matrix():
    """Empirical P(dst|src) should narrow gaps between probable arcs."""
    from market_regime_engine.wfst import PRIOR_ARCS

    dec = RegimeWFST()
    states = sorted(dec.states)
    K = len(states)
    idx = {s: i for i, s in enumerate(states)}
    counts = np.full((K, K), 1.0)
    counts[idx["risk_on_expansion"], idx["late_cycle"]] += 50
    counts[idx["late_cycle"], idx["sticky_inflation"]] += 40
    counts[idx["sticky_inflation"], idx["recessionary_bear"]] += 30
    transition = counts / counts.sum(axis=1, keepdims=True)
    prior_cost_late_to_sticky = next(
        a.cost for a in PRIOR_ARCS if a.src == "late_cycle" and a.dst == "sticky_inflation"
    )
    dec.fit_costs_from_transition_matrix(transition, states)
    fitted_cost_late_to_sticky = dec.transition_cost("late_cycle", "sticky_inflation")
    # Transitions we drove up in the empirical counts should not be more
    # expensive than the prior.
    assert fitted_cost_late_to_sticky <= prior_cost_late_to_sticky + 1e-9
    assert dec.fitted is True


def test_wfst_event_bonus_grid_picks_a_value():
    dec = RegimeWFST()
    obs = ["risk_on_expansion", "late_cycle", "credit_stress", "recessionary_bear"]
    gold = ["risk_on_expansion", "late_cycle", "credit_stress", "recessionary_bear"]
    labels = [set(), set(), {"credit_break"}, {"credit_break"}]
    dec.fit_event_bonus(obs, gold, event_labels=labels)
    assert "event_bonus" in dec.fit_log


def test_dynamic_weights():
    prior = pd.Series({"a": 0.5, "b": 0.5})
    losses = pd.Series({"a": 0.1, "b": 2.0})
    w = dynamic_weights(prior, losses=losses)
    assert abs(w.sum() - 1.0) < 1e-9
    assert w["a"] > w["b"]


from market_regime_engine.model_registry import create_model_card
from market_regime_engine.point_in_time import assert_no_future_vintages, observations_as_of
from market_regime_engine.validation import (
    brier_score,
    calibration_table,
    log_loss_score,
    pinball_loss,
    quantile_coverage,
)


def test_validation_metrics_run():
    y = [0, 1, 1, 0]
    p = [0.1, 0.8, 0.6, 0.2]
    assert brier_score(y, p) < 0.2
    assert log_loss_score(y, p) > 0
    table = calibration_table(y, p, bins=2)
    assert not table.empty
    assert pinball_loss([1, 2, 3], [1, 2, 2], 0.5) >= 0
    assert 0 <= quantile_coverage([1, 2, 3], [1, 2, 2]) <= 1


def test_point_in_time_filter():
    obs = pd.DataFrame(
        [
            {"series_id": "A", "date": "2020-01-01", "value": 1.0, "vintage_date": "2020-02-01", "source": "x"},
            {"series_id": "A", "date": "2020-01-01", "value": 2.0, "vintage_date": "2020-03-01", "source": "x"},
        ]
    )
    assert_no_future_vintages(obs)
    out = observations_as_of(obs, "2020-02-15")
    assert float(out.iloc[0]["value"]) == 1.0


def test_model_card_hash():
    card = create_model_card(
        model_name="m",
        version="v",
        target="t",
        horizon="1m",
        training_start="2020-01-01",
        training_end="2021-01-01",
        feature_count=3,
        observations=10,
        objective="test",
        known_limitations=["none"],
        validation_metrics={"brier": 0.1},
    )
    assert len(card.artifact_hash) == 64


from market_regime_engine.baselines import expanding_event_rate_baseline, expanding_quantile_baseline
from market_regime_engine.bocpd import DiagonalStudentTBOCPD
from market_regime_engine.hmm import HMMRegimePosterior
from market_regime_engine.point_in_time import apply_release_lag
from market_regime_engine.promotion import PromotionGate


def test_bocpd_runs_and_detects_probability_bounds():
    idx = pd.date_range("2020-01-01", periods=80, freq="MS")
    x = pd.DataFrame({"a": np.r_[np.zeros(50), np.ones(30) * 4], "b": np.r_[np.zeros(50), np.ones(30) * -2]}, index=idx)
    out = DiagonalStudentTBOCPD(max_run=48).score(x)
    assert len(out) == 80
    assert out["change_point_prob"].between(0, 1).all()
    assert "bocpd_map_run_length" in out.columns


def test_multivariate_niw_bocpd_detects_jump():
    from market_regime_engine.bocpd import MultivariateNIWBOCPD, learned_constant_hazard

    rng = np.random.default_rng(0)
    n_pre, n_post = 60, 40
    idx = pd.date_range("2010-01-01", periods=n_pre + n_post, freq="MS")
    pre = rng.normal(0.0, 1.0, size=(n_pre, 3))
    post = rng.normal(0.0, 1.0, size=(n_post, 3)) + np.array([4.0, -3.0, 2.5])
    x = pd.DataFrame(np.vstack([pre, post]), index=idx, columns=["a", "b", "c"])
    out = MultivariateNIWBOCPD(max_run=64).score(x)
    assert len(out) == len(idx)
    assert out["change_point_prob"].between(0, 1).all()
    # The jump is at index n_pre. The detector should put more posterior mass on
    # a recent change-point in the post-jump window than in the pre-jump window.
    pre_mean = float(out["change_point_prob"].iloc[5:n_pre].mean())
    post_mean = float(out["change_point_prob"].iloc[n_pre : n_pre + 10].mean())
    assert post_mean > pre_mean

    # Hazard learning helper sanity check.
    h = learned_constant_hazard(["a", "a", "a", "b", "b", "a", "a"])
    assert 0 < h <= 0.5


def test_multivariate_niw_bocpd_handles_singular_inputs():
    """A constant column should not crash the Cholesky path."""
    from market_regime_engine.bocpd import MultivariateNIWBOCPD

    idx = pd.date_range("2010-01-01", periods=20, freq="MS")
    x = pd.DataFrame({"a": np.linspace(0, 1, 20), "b": np.zeros(20)}, index=idx)
    out = MultivariateNIWBOCPD(max_run=16).score(x)
    assert out["change_point_prob"].between(0, 1).all()


def test_hmm_regime_posterior_runs():
    idx = pd.date_range("2020-01-01", periods=12, freq="MS")
    scores = pd.DataFrame(
        {
            "labor": 0.1,
            "rates": 1.5,
            "inflation": 1.6,
            "credit": 0.3,
            "housing": 0.2,
            "energy": 0.1,
            "fx": 0.2,
            "fiscal": 0.4,
        },
        index=idx,
    )
    out = HMMRegimePosterior().score(scores)
    assert len(out) == 12
    assert out["hmm_confidence"].between(0, 1).all()
    assert any(c.startswith("regime_prob_") for c in out.columns)


def test_hmm_baum_welch_fit_preserves_regime_names():
    """Baum-Welch fit should learn from data but keep the named regime ordering
    via the post-fit pinning step."""
    rng = np.random.default_rng(0)
    n = 240
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    x = pd.DataFrame(
        rng.normal(size=(n, len(["labor", "rates", "inflation", "credit", "housing", "energy", "fx", "fiscal"]))) * 0.6,
        index=idx,
        columns=["labor", "rates", "inflation", "credit", "housing", "energy", "fx", "fiscal"],
    )
    x.iloc[80:120] += np.array([1.5, 0.9, 1.7, 0.8, 0.8, 1.2, 0.6, 0.8])  # stagflation-like block
    x.iloc[160:200] += np.array([0.7, 0.9, 0.4, 1.8, 1.1, 0.3, 0.7, 0.7])  # credit-stress-like block

    hmm = HMMRegimePosterior().fit(x, max_iter=20)
    assert hmm.fitted is True
    assert "log_likelihood" in hmm.fit_log
    out = hmm.score(x)
    assert len(out) == n
    # Regime names are preserved via pinning even after EM.
    assert set(out["hmm_regime"]) <= set(REGIME_STATES)


from market_regime_engine.hmm import REGIME_STATES


def test_quantile_model_non_crossing():
    """The HGBR-quantile head must produce monotone-in-tau quantiles."""
    from market_regime_engine.models import QuantileReturnModel

    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = pd.Series(0.5 * X["a"] + 0.1 * X["b"] + rng.normal(scale=0.3, size=n))
    model = QuantileReturnModel(n_estimators=120, learning_rate=0.05, min_train=50).fit(X, y)
    pred = model.predict(X.head(50))
    cols = ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]
    assert list(pred.columns) == cols
    arr = pred.to_numpy()
    diffs = np.diff(arr, axis=1)
    assert (diffs >= -1e-9).all(), "non-crossing repair must produce monotone quantiles"


def test_quantile_model_falls_back_to_linear_then_empirical():
    from market_regime_engine.models import QuantileReturnModel

    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=10), "b": rng.normal(size=10)})
    y = pd.Series(rng.normal(size=10))
    model = QuantileReturnModel(min_train=200).fit(X, y)
    # 10 samples is below min_train // 2 = 100, so empirical quantile fallback.
    pred = model.predict(X.head(2))
    assert pred["q50"].nunique() == 1


def test_benchmarks_and_promotion_gate_run():
    idx = pd.date_range("2020-01-01", periods=20, freq="MS")
    y = pd.Series([0, 1] * 10, index=idx, dtype=float)
    b = expanding_event_rate_baseline(y, min_train=5)
    assert not b.empty
    q = expanding_quantile_baseline(pd.Series(np.arange(20.0), index=idx), min_train=5)
    assert not q.empty
    cand = pd.DataFrame(
        [{"target": "x", "horizon": "1m", "observations": 30, "brier": 0.10, "log_loss": 0.30, "ece": 0.05}]
    )
    bench = pd.DataFrame(
        [{"target": "x", "horizon": "1m", "observations": 30, "brier": 0.20, "log_loss": 0.50, "ece": 0.08}]
    )
    result = PromotionGate().evaluate_binary(cand, bench)
    assert bool(result.iloc[0]["promoted"])


def test_apply_release_lag_moves_known_macro_forward():
    obs = pd.DataFrame(
        [{"series_id": "UNRATE", "date": "2020-01-01", "value": 4.0, "vintage_date": "2020-01-01", "source": "x"}]
    )
    out = apply_release_lag(obs)
    assert pd.to_datetime(out.iloc[0]["vintage_date"]) >= pd.Timestamp("2020-02-01")


from market_regime_engine.analogs import HistoricalAnalogEngine, analog_summary
from market_regime_engine.attribution import feature_driver_attribution
from market_regime_engine.ensemble_v2 import dynamic_model_weights, mix_binary_probabilities
from market_regime_engine.nber import add_forward_recession_targets, label_recession_months


def test_nber_labels_run():
    dates = pd.date_range("2007-01-01", periods=36, freq="MS")
    labels = label_recession_months(dates)
    assert labels["recession"].sum() > 0
    fwd = add_forward_recession_targets(labels)
    assert "recession_next_12m" in fwd.columns


def test_analogs_and_attribution_run():
    idx = pd.date_range("2000-01-01", periods=160, freq="MS")
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "UNRATE.diff_3m": rng.normal(size=len(idx)),
            "BAA10Y.level": rng.normal(size=len(idx)),
            "PERMIT.log_yoy": rng.normal(size=len(idx)),
            "CPIAUCSL.log_yoy": rng.normal(size=len(idx)),
        },
        index=idx,
    )
    targets = pd.DataFrame({"ret_12m": rng.normal(size=len(idx)), "dd_12m": rng.normal(size=len(idx))}, index=idx)
    analogs = HistoricalAnalogEngine(min_history=30, top_n=5).score(X, targets=targets)
    assert len(analogs) == 5
    assert "weighted_forward_returns" in analog_summary(analogs)

    features = (
        X.reset_index()
        .melt(id_vars="index", var_name="feature_name", value_name="value")
        .rename(columns={"index": "date"})
    )
    features["domain"] = "test"
    fa = feature_driver_attribution(features, top_n=3)
    assert len(fa) == 3


def test_dynamic_model_weights_run():
    weights = dynamic_model_weights(
        losses={"a": 0.1, "b": 1.0},
        calibration_errors={"a": 0.02, "b": 0.2},
        regime_fit={"a": 0.5, "b": 0.1},
        change_point_prob=0.2,
        staleness={"a": 0.1, "b": 0.9},
    )
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert weights["a"] > weights["b"]
    mixed = mix_binary_probabilities({"a": 0.2, "b": 0.8}, weights)
    assert 0 <= mixed <= 1


from market_regime_engine.calibration import PlattCalibrator, apply_binary_calibration
from market_regime_engine.confidence import compute_model_confidence
from market_regime_engine.invalidation import forecast_invalidation_triggers
from market_regime_engine.model_runs import create_model_run, model_run_frame
from market_regime_engine.release_calendar import audit_release_calendar, enforce_release_calendar


def test_release_calendar_and_calibration_run():
    obs = pd.DataFrame(
        [{"series_id": "UNRATE", "date": "2020-01-01", "value": 4.0, "vintage_date": "2020-01-01", "source": "x"}]
    )
    fixed = enforce_release_calendar(obs)
    audit = audit_release_calendar(fixed)
    assert audit.iloc[0]["violations"] == 0

    cal = PlattCalibrator().fit(pd.Series([0, 1, 0, 1] * 10), pd.Series([0.2, 0.8, 0.3, 0.7] * 10))
    out = cal.transform(pd.Series([0.5]))
    assert 0 <= float(out[0]) <= 1

    raw = pd.DataFrame(
        [
            {
                "model_name": "m",
                "date": "2020-01-01",
                "horizon": "3m",
                "target": "drawdown_gt_10pct",
                "value": 0.5,
                "metadata_json": "{}",
            }
        ]
    )
    cals = pd.DataFrame(
        [
            {
                "horizon": "3m",
                "target": "drawdown_gt_10pct",
                "method": "platt_logit",
                "intercept": 0.0,
                "slope": 1.0,
                "fallback_rate": np.nan,
                "observations": 40,
                "raw_mean": 0.5,
                "calibrated_mean": 0.5,
                "metadata_json": "{}",
            }
        ]
    )
    calibrated = apply_binary_calibration(raw, cals)
    assert not calibrated.empty


def test_invalidation_confidence_and_model_run():
    dates = pd.date_range("2020-01-01", periods=8, freq="MS")
    features = pd.DataFrame(
        [
            {"date": d, "feature_name": "UNRATE.level", "value": 4 + i * 0.2, "domain": "labor"}
            for i, d in enumerate(dates)
        ]
        + [{"date": d, "feature_name": "BAA10Y.level", "value": 3.0, "domain": "credit"} for d in dates]
    )
    regimes = pd.DataFrame(
        [
            {
                "date": dates[-1],
                "regime": "credit_stress",
                "decoded_regime": "credit_stress",
                "score": 2.0,
                "change_point_prob": 0.7,
                "metadata_json": "{}",
            }
        ]
    )
    triggers = forecast_invalidation_triggers(features, regimes)
    assert "change_point_spike" in set(triggers["trigger"])
    conf = compute_model_confidence(
        regimes=regimes,
        validation=pd.DataFrame([{"ece": 0.1, "brier": 0.2}]),
        analogs=pd.DataFrame([{"as_of_date": dates[-1], "similarity": 0.4}]),
        release_audit=pd.DataFrame([{"rows": 10, "violations": 0}]),
    )
    assert 0 <= float(conf.iloc[0]["confidence"]) <= 1
    run = create_model_run(engine_version="0.5.0", purpose="test", features=features, model_outputs=pd.DataFrame())
    assert len(run.run_id) == 16
    assert not model_run_frame(run).empty


from market_regime_engine.analytics_warehouse import export_sqlite_to_lake, warehouse_health
from market_regime_engine.drift import compute_feature_drift
from market_regime_engine.release_calendar_exact import (
    audit_exact_release_calendar,
    build_exact_release_calendar,
    enforce_exact_release_calendar,
)
from market_regime_engine.release_gates import evaluate_release_gate
from market_regime_engine.stacking import optimize_binary_stacking
from market_regime_engine.storage import Warehouse
from market_regime_engine.survival import recession_hazard_scores


def test_exact_release_calendar_runs():
    obs = pd.DataFrame(
        [{"series_id": "UNRATE", "date": "2020-01-01", "value": 4.0, "vintage_date": "2020-01-01", "source": "x"}]
    )
    cal = build_exact_release_calendar(obs, [{"series_id": "UNRATE", "domain": "labor"}])
    assert not cal.empty
    fixed = enforce_exact_release_calendar(obs, cal)
    audit = audit_exact_release_calendar(fixed, cal)
    assert int(audit.iloc[0]["violations"]) == 0


def test_survival_stacking_drift_gate_run():
    idx = pd.date_range("2000-01-01", periods=180, freq="MS")
    features = []
    for i, d in enumerate(idx):
        features.append({"date": d, "feature_name": "UNRATE.level", "value": 4 + np.sin(i / 8), "domain": "labor"})
        features.append({"date": d, "feature_name": "BAA10Y.level", "value": 2 + np.cos(i / 9), "domain": "credit"})
        features.append({"date": d, "feature_name": "PERMIT.level", "value": 1000 - i * 0.3, "domain": "housing"})
    f = pd.DataFrame(features)
    survival = recession_hazard_scores(f)
    assert not survival.empty
    assert survival["value"].between(0, 1).all()

    preds = pd.DataFrame(
        [
            {
                "date": d.strftime("%Y-%m-%d"),
                "model_name": "a",
                "horizon": "12m",
                "target": "drawdown_gt_10pct",
                "value": 0.2 + 0.1 * (i % 2),
            }
            for i, d in enumerate(idx[:80])
        ]
        + [
            {
                "date": d.strftime("%Y-%m-%d"),
                "model_name": "b",
                "horizon": "12m",
                "target": "drawdown_gt_10pct",
                "value": 0.6 - 0.1 * (i % 2),
            }
            for i, d in enumerate(idx[:80])
        ]
    )
    y = pd.Series([(i % 3) == 0 for i in range(80)], index=[d.strftime("%Y-%m-%d") for d in idx[:80]], dtype=float)
    stack = optimize_binary_stacking(preds, y, "drawdown_gt_10pct", "12m", step=0.5)
    assert not stack.weights.empty
    assert abs(stack.weights["weight"].sum() - 1.0) < 1e-9

    drift = compute_feature_drift(f, baseline_months=120, recent_months=12, top_n=5)
    assert "psi" in drift.columns
    # v1.4.1 (item F): the test only asserts that the decision is one
    # of {release, hold}; both profiles keep that surface unchanged.
    gate = evaluate_release_gate(
        confidence=pd.DataFrame([{"date": "2020-01-01", "confidence": 0.8, "grade": "B"}]),
        drift=drift,
        invalidation=pd.DataFrame(),
        promotion=pd.DataFrame([{"promoted": True}]),
        profile="default",
    )
    assert gate.iloc[0]["decision"] in {"release", "hold"}


def test_export_sqlite_to_lake_csv_fallback(tmp_path):
    db_path = tmp_path / "mre.db"
    wh = Warehouse(db_path)
    try:
        wh.write_observations(
            pd.DataFrame(
                [{"series_id": "A", "date": "2020-01-01", "value": 1.0, "vintage_date": "2020-01-02", "source": "x"}]
            )
        )
    finally:
        wh.close()
    manifest = export_sqlite_to_lake(db_path, tmp_path / "lake", prefer_parquet=False)
    assert not manifest.empty
    health = warehouse_health(tmp_path / "lake")
    assert bool(health["exists"].all())


from market_regime_engine.alerts import route_alerts
from market_regime_engine.alfred import build_alfred_request_matrix, vintage_grid
from market_regime_engine.hazard_model import train_fitted_hazard_outputs
from market_regime_engine.promotion_workflow import evaluate_promotion_workflow
from market_regime_engine.stacking_v2 import regime_conditioned_stacking


def test_alfred_request_matrix_runs():
    grid = vintage_grid("2020-01-01", "2020-03-15", "MS")
    assert grid[0] == "2020-01-01"
    assert "2020-03-15" in grid
    matrix = build_alfred_request_matrix(["UNRATE", "CPIAUCSL"], vintage_start="2020-01-01", vintage_end="2020-02-01")
    assert not matrix.empty
    assert set(matrix["series_id"]) == {"UNRATE", "CPIAUCSL"}


def test_fitted_hazard_outputs_run():
    idx = pd.date_range("2000-01-01", periods=180, freq="MS")
    features = []
    for i, d in enumerate(idx):
        features.append({"date": d, "feature_name": "UNRATE.level", "value": 4 + np.sin(i / 8), "domain": "labor"})
        features.append({"date": d, "feature_name": "BAA10Y.level", "value": 2 + np.cos(i / 9), "domain": "credit"})
    labels = pd.DataFrame(
        {
            "date": idx,
            "recession": [1.0 if 80 <= i <= 90 or 140 <= i <= 145 else 0.0 for i in range(len(idx))],
            "source": "test",
            "metadata_json": "{}",
        }
    )
    outputs, diag = train_fitted_hazard_outputs(pd.DataFrame(features), labels)
    assert not outputs.empty
    assert not diag.empty
    assert outputs["value"].between(0, 1).all()


def test_regime_conditioned_stacking_and_alerts(tmp_path):
    valdir = tmp_path / "validation"
    valdir.mkdir()
    rows = []
    idx = pd.date_range("2020-01-01", periods=40, freq="MS")
    for i, d in enumerate(idx):
        y = float(i % 3 == 0)
        rows.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "model_name": "candidate",
                "horizon": "12m",
                "target": "drawdown_gt_10pct",
                "p": 0.7 if y else 0.2,
                "y": y,
            }
        )
        rows.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "model_name": "benchmark",
                "horizon": "12m",
                "target": "drawdown_gt_10pct",
                "p": 0.4,
                "y": y,
            }
        )
    pd.DataFrame(rows).to_csv(valdir / "binary_predictions_12m.csv", index=False)
    regimes = pd.DataFrame(
        {"date": [d.strftime("%Y-%m-%d") for d in idx], "decoded_regime": ["credit_stress"] * len(idx)}
    )
    out = regime_conditioned_stacking(valdir, regimes, step=0.5)
    assert not out["oos_predictions"].empty
    assert not out["ensemble_weights"].empty

    alerts = route_alerts(
        release_gates=pd.DataFrame([{"date": "2020-01-01", "approved": 0, "reasons": "test hold"}]),
        drift=pd.DataFrame([{"feature_name": "x", "status": "severe", "psi": 0.5}]),
        invalidation=pd.DataFrame([{"trigger": "x", "status": "active", "severity": "high"}]),
        confidence=pd.DataFrame([{"date": "2020-01-01", "confidence": 0.4}]),
        promotion=pd.DataFrame([{"promoted": False}]),
    )
    assert not alerts.empty
    assert "release_gate_hold" in set(alerts["alert_type"])

    workflow = evaluate_promotion_workflow(
        promotion=pd.DataFrame([{"promoted": True}]),
        release_gate=pd.DataFrame([{"approved": 1}]),
        confidence=pd.DataFrame([{"date": "2020-01-01", "confidence": 0.8}]),
        drift=pd.DataFrame([{"status": "ok"}]),
    )
    assert workflow.iloc[0]["decision"] == "promote"


def test_v08_vintage_asof_pipeline(tmp_path):
    from market_regime_engine.alfred_real import seed_vintage_observations_from_latest
    from market_regime_engine.asof import (
        audit_feature_asof_lineage,
        audit_vintage_observations,
        materialize_feature_asof_values,
    )
    from market_regime_engine.config import load_catalog
    from market_regime_engine.sample import generate_sample_observations
    from market_regime_engine.storage import Warehouse

    db = Warehouse(tmp_path / "v08.db")
    obs = generate_sample_observations()
    db.write_observations(obs)
    vintages, vintage_obs = seed_vintage_observations_from_latest(obs)
    db.write_series_vintages(vintages)
    db.write_vintage_observations(vintage_obs)
    feature_asof = materialize_feature_asof_values(
        db.read_vintage_observations(), load_catalog(), min_history_months=36
    )
    assert not feature_asof.empty
    db.write_feature_asof_values(feature_asof)
    vintage_audit = audit_vintage_observations(db.read_vintage_observations())
    feature_audit = audit_feature_asof_lineage(db.read_feature_asof_values())
    assert int(vintage_audit["violations"].iloc[0]) == 0
    assert int(feature_audit["violations"].iloc[0]) == 0
    db.close()
