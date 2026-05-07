# SPDX-License-Identifier: Apache-2.0
"""Quantile baseline forecast models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as SklearnRandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from market_regime_engine.models.base import (
    ModelCard,
    check_is_fitted,
    numeric_feature_frame,
    quantile_prediction_frame,
)


def _regression_target(y: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("y must contain at least one row")
    if not np.all(np.isfinite(arr)):
        raise ValueError("y must contain only finite numeric values")
    return arr


def _validate_quantiles(lo: float, hi: float) -> None:
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError("quantile bounds must satisfy 0 <= lo < hi <= 1")


@dataclass
class HistoricalQuantileRegressor:
    """Constant historical quantile interval model."""

    lo: float = 0.1
    hi: float = 0.9
    default_target: str = "target"
    default_horizon: str = "1m"
    model_name: str = "historical_quantile"
    output_type: str = "quantile"

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray, **_: Any) -> HistoricalQuantileRegressor:
        del X
        target = _regression_target(y)
        _validate_quantiles(self.lo, self.hi)
        self.q_lo_ = float(np.quantile(target, self.lo))
        self.q50_ = float(np.quantile(target, 0.5))
        self.q_hi_ = float(np.quantile(target, self.hi))
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray, **_: Any) -> pd.DataFrame:
        check_is_fitted(self)
        n = len(X)
        return quantile_prediction_frame(
            X,
            q_lo=np.full(n, self.q_lo_, dtype=float),
            q50=np.full(n, self.q50_, dtype=float),
            q_hi=np.full(n, self.q_hi_, dtype=float),
            model_name=self.model_name,
            default_target=self.default_target,
            default_horizon=self.default_horizon,
        )

    def get_params(self) -> dict[str, Any]:
        return {
            "lo": self.lo,
            "hi": self.hi,
            "default_target": self.default_target,
            "default_horizon": self.default_horizon,
        }

    def model_card(self) -> dict[str, Any]:
        return ModelCard(
            model_name=self.model_name,
            output_type=self.output_type,
            family="baseline",
            description="Constant historical quantile interval model.",
            params=self.get_params(),
        ).to_dict()


@dataclass
class RandomForestQuantileRegressor:
    """Random forest interval model using empirical per-tree quantiles."""

    lo: float = 0.1
    hi: float = 0.9
    n_estimators: int = 200
    max_depth: int | None = None
    min_samples_leaf: int = 5
    random_state: int | None = 0
    n_jobs: int | None = None
    default_target: str = "target"
    default_horizon: str = "1m"
    model_name: str = "random_forest_quantile"
    output_type: str = "quantile"

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray, **_: Any) -> RandomForestQuantileRegressor:
        target = _regression_target(y)
        _validate_quantiles(self.lo, self.hi)
        features = numeric_feature_frame(X)
        self.feature_columns_ = list(features.columns)
        self.pipeline_ = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    SklearnRandomForestRegressor(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        min_samples_leaf=self.min_samples_leaf,
                        random_state=self.random_state,
                        n_jobs=self.n_jobs,
                    ),
                ),
            ]
        )
        self.pipeline_.fit(features, target)
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray, **_: Any) -> pd.DataFrame:
        check_is_fitted(self)
        features = numeric_feature_frame(X)
        imputer = self.pipeline_.named_steps["imputer"]
        forest = self.pipeline_.named_steps["model"]
        transformed = imputer.transform(features)
        tree_predictions = np.vstack([tree.predict(transformed) for tree in forest.estimators_])
        return quantile_prediction_frame(
            X,
            q_lo=np.quantile(tree_predictions, self.lo, axis=0),
            q50=np.quantile(tree_predictions, 0.5, axis=0),
            q_hi=np.quantile(tree_predictions, self.hi, axis=0),
            model_name=self.model_name,
            default_target=self.default_target,
            default_horizon=self.default_horizon,
        )

    def get_params(self) -> dict[str, Any]:
        return {
            "lo": self.lo,
            "hi": self.hi,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "default_target": self.default_target,
            "default_horizon": self.default_horizon,
        }

    def model_card(self) -> dict[str, Any]:
        return ModelCard(
            model_name=self.model_name,
            output_type=self.output_type,
            family="tree_ensemble",
            description="Random forest interval model using empirical per-tree quantiles.",
            params=self.get_params(),
            dependencies=("scikit-learn",),
        ).to_dict()


__all__ = ["HistoricalQuantileRegressor", "RandomForestQuantileRegressor"]
