# SPDX-License-Identifier: Apache-2.0
"""Registry for baseline forecast models."""

from __future__ import annotations

from typing import Any

from market_regime_engine.models.base import ForecastModel
from market_regime_engine.models.classification import (
    LogisticRegressionClassifier,
    PersistenceClassifier,
    RandomForestClassifierModel,
)
from market_regime_engine.models.regression import (
    HistoricalQuantileRegressor,
    RandomForestQuantileRegressor,
)

_MODEL_REGISTRY: dict[str, type[ForecastModel]] = {
    "persistence": PersistenceClassifier,
    "logistic_regression": LogisticRegressionClassifier,
    "random_forest": RandomForestClassifierModel,
    "historical_quantile": HistoricalQuantileRegressor,
    "random_forest_quantile": RandomForestQuantileRegressor,
}

_ALIASES: dict[str, str] = {
    "prior": "persistence",
    "lr": "logistic_regression",
    "rf": "random_forest",
    "hist_quantile": "historical_quantile",
    "rf_quantile": "random_forest_quantile",
}


def normalize_model_name(name: str) -> str:
    """Normalize aliases into canonical model names."""

    key = name.strip().lower().replace("-", "_")
    return _ALIASES.get(key, key)


def available_models(output_type: str | None = None) -> list[str]:
    """Return sorted canonical model names."""

    names: list[str] = []
    for name, cls in _MODEL_REGISTRY.items():
        if output_type is None or cls().output_type == output_type:
            names.append(name)
    return sorted(names)


def get_model_class(name: str) -> type[ForecastModel]:
    """Return a model class by canonical name or alias."""

    canonical = normalize_model_name(name)
    try:
        return _MODEL_REGISTRY[canonical]
    except KeyError as exc:
        available = ", ".join(available_models())
        raise KeyError(f"Unknown model {name!r}. Available models: {available}") from exc


def make_model(name: str, **params: Any) -> ForecastModel:
    """Instantiate a model by name."""

    return get_model_class(name)(**params)


def model_cards(output_type: str | None = None) -> list[dict[str, Any]]:
    """Return model cards for the registered baselines."""

    return [make_model(name).model_card() for name in available_models(output_type=output_type)]


__all__ = ["available_models", "get_model_class", "make_model", "model_cards", "normalize_model_name"]
