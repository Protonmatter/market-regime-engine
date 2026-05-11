# SPDX-License-Identifier: Apache-2.0
"""Forecast model zoo exports.

v1.5 (PR-2 task I.1): the historical ``market_regime_engine.models``
module (a single file at ``models.py`` exposing :class:`ProbabilityModel`
and :class:`QuantileReturnModel`) was shadowed by the PR #9 baseline
model-zoo package directory, so any caller doing
``from market_regime_engine.models import ProbabilityModel`` resolved
the package ``__init__`` instead of the module — and the package did
not re-export the legacy names. We rename the file to
``models_legacy.py`` and re-export the public surface here so both
import paths continue to work.
"""

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
from market_regime_engine.models_legacy import (
    DEFAULT_QUANTILES,
    ProbabilityModel,
    QuantileReturnModel,
    train_latest_outputs,
)

__all__ = [
    "DEFAULT_QUANTILES",
    "ElasticNetLogisticClassifier",
    "ForecastModel",
    "HistGradientBoostingProbabilityModel",
    "HistGradientBoostingQuantileRegressor",
    "HistoricalQuantileRegressor",
    "LinearQuantileRegressor",
    "LogisticRegressionClassifier",
    "ModelCard",
    "PersistenceClassifier",
    "ProbabilityModel",
    "QuantileReturnModel",
    "RandomForestClassifierModel",
    "RandomForestQuantileRegressor",
    "RollingBaseRateClassifier",
    "available_models",
    "get_model_class",
    "make_model",
    "model_cards",
    "normalize_model_name",
    "train_latest_outputs",
]
