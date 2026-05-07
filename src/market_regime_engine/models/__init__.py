# SPDX-License-Identifier: Apache-2.0
"""Forecast model zoo exports."""

from market_regime_engine.models.base import ForecastModel, ModelCard
from market_regime_engine.models.classification import (
    LogisticRegressionClassifier,
    PersistenceClassifier,
    RandomForestClassifierModel,
)
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
    "ForecastModel",
    "HistoricalQuantileRegressor",
    "LogisticRegressionClassifier",
    "ModelCard",
    "PersistenceClassifier",
    "RandomForestClassifierModel",
    "RandomForestQuantileRegressor",
    "available_models",
    "get_model_class",
    "make_model",
    "model_cards",
    "normalize_model_name",
]
