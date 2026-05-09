# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.models import (
    ElasticNetLogisticClassifier,
    HistGradientBoostingProbabilityModel,
    HistGradientBoostingQuantileRegressor,
    HistoricalQuantileRegressor,
    LinearQuantileRegressor,
    LogisticRegressionClassifier,
    PersistenceClassifier,
    RandomForestClassifierModel,
    RandomForestQuantileRegressor,
    RollingBaseRateClassifier,
    available_models,
    make_model,
    model_cards,
    normalize_model_name,
)


def _features(n: int = 32) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    x1 = np.linspace(-2.0, 2.0, n)
    x2 = rng.normal(0.0, 0.5, n)
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n, freq="D"),
            "target": "drawdown_gt_10pct",
            "horizon": "1m",
            "x1": x1,
            "x2": x2,
            "y": (x1 + x2 > 0).astype(int),
        }
    )


def _binary_y(n: int = 32) -> np.ndarray:
    x = np.linspace(-2.0, 2.0, n)
    return (x > 0).astype(int)


def _regression_y(n: int = 32) -> np.ndarray:
    x = np.linspace(-2.0, 2.0, n)
    return 0.25 + (1.5 * x) + np.sin(x)


def test_registry_normalizes_aliases_and_returns_model_cards() -> None:
    names = available_models()

    assert "persistence" in names
    assert "rolling_base_rate" in names
    assert "elastic_net_logistic" in names
    assert "linear_quantile" in names
    assert "hist_gradient_boosting_quantile" in names
    assert normalize_model_name("rf") == "random_forest"
    assert normalize_model_name("quantile-regression") == "linear_quantile"
    assert make_model("lr").model_name == "logistic_regression"
    assert any(card["model_name"] == "linear_quantile" for card in model_cards("quantile"))


@pytest.mark.parametrize(
    "model",
    [
        PersistenceClassifier(),
        RollingBaseRateClassifier(window=5),
        LogisticRegressionClassifier(max_iter=200),
        ElasticNetLogisticClassifier(max_iter=200),
        RandomForestClassifierModel(n_estimators=12, min_samples_leaf=1, random_state=0, n_jobs=1),
        HistGradientBoostingProbabilityModel(max_iter=12, random_state=0),
    ],
)
def test_binary_baselines_emit_prediction_evidence_frame(model) -> None:
    X = _features()
    y = _binary_y(len(X))

    pred = model.fit(X, y).predict(X)

    assert len(pred) == len(X)
    assert {"date", "target", "horizon", "model_name", "y", "p"}.issubset(pred.columns)
    assert pred["model_name"].nunique() == 1
    assert pred["p"].between(0.0, 1.0).all()


@pytest.mark.parametrize(
    "model",
    [
        HistoricalQuantileRegressor(),
        LinearQuantileRegressor(alpha=0.0),
        RandomForestQuantileRegressor(n_estimators=12, min_samples_leaf=1, random_state=0, n_jobs=1),
        HistGradientBoostingQuantileRegressor(max_iter=12, random_state=0),
    ],
)
def test_quantile_baselines_emit_prediction_evidence_frame(model) -> None:
    X = _features()
    y = _regression_y(len(X))

    pred = model.fit(X, y).predict(X)

    assert len(pred) == len(X)
    assert {"date", "target", "horizon", "model_name", "y", "q_lo", "q_hi"}.issubset(pred.columns)
    assert pred["model_name"].nunique() == 1
    assert (pred["q_lo"] <= pred["q_hi"]).all()


def test_single_class_binary_training_falls_back_to_constant_probability() -> None:
    X = _features()
    y = np.zeros(len(X), dtype=int)

    pred = LogisticRegressionClassifier().fit(X, y).predict(X)

    assert pred["p"].nunique() == 1
    assert pred["p"].between(0.0, 1.0).all()
