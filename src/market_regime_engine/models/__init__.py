# SPDX-License-Identifier: Apache-2.0
"""Forecast model zoo exports."""

from market_regime_engine.models.base import ForecastModel, ModelCard
from market_regime_engine.models.baselines import ElasticNetLogisticClassifier, RollingBaseRateClassifier
from market_regime_engine.models.classification import (
    LogisticRegressionClassifier,
    PersistenceClassifier,
    RandomForestClassifierModel,
)
from market_regime_engine.models.gradient_boosting import (
    HistGradientBoostingProbabilityModel,
    HistGradientBoostingQuantileRegressor,
)
from market_regime_engine.models.linear_quantile import LinearQuantileRegressor
from market_regime_engine.models.registry import (
    available_models,
    get_model_class,
    make_model,
    model_cards,
    normalize_model_name,
)
from market_regime_engine.models.regression import (
    HistoricalQuantileRegressor,
    RandomForestQuantileRegressor,
)

__all__ = [
    "ElasticNetLogisticClassifier",
    "ForecastModel",
    "HistGradientBoostingProbabilityModel",
    "HistGradientBoostingQuantileRegressor",
    "HistoricalQuantileRegressor",
    "LinearQuantileRegressor",
    "LogisticRegressionClassifier",
    "ModelCard",
    "PersistenceClassifier",
    "RandomForestClassifierModel",
    "RandomForestQuantileRegressor",
    "RollingBaseRateClassifier",
    "available_models",
    "get_model_class",
    "make_model",
    "model_cards",
    "normalize_model_name",
]
