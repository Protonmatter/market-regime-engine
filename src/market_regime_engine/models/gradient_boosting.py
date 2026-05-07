# SPDX-License-Identifier: Apache-2.0
"""Histogram gradient boosting baseline models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from market_regime_engine.models.base import (
    ModelCard,
    binary_prediction_frame,
    check_is_fitted,
    numeric_feature_frame,
    quantile_prediction_frame,
)
from market_regime_engine.models.classification import _binary_target, _smooth_rate
from market_regime_engine.models.regression import _regression_target, _validate_quantiles


@dataclass
class HistGradientBoostingProbabilityModel:
    """Histogram gradient boosting probability baseline."""

    max_iter: int = 100
    learning_rate: float = 0.1
    max_leaf_nodes: int | None = 31
    l2_regularization: float = 0.0
    random_state: int | None = 0
    default_target: str = "target"
    default_horizon: str = "1m"
    model_name: str = "hist_gradient_boosting_probability"
    output_type: str = "binary"

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        **_: Any,
    ) -> HistGradientBoostingProbabilityModel:
        target = _binary_target(y)
        features = numeric_feature_frame(X)
        self.constant_probability_ = _smooth_rate(target, alpha=1.0)
        self.pipeline_ = None
        if np.unique(target).size >= 2:
            self.pipeline_ = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            learning_rate=self.learning_rate,
                            l2_regularization=self.l2_regularization,
                            max_iter=self.max_iter,
                            max_leaf_nodes=self.max_leaf_nodes,
                            random_state=self.random_state,
                        ),
                    ),
                ]
            )
            self.pipeline_.fit(features, target)
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray, **_: Any) -> pd.DataFrame:
        check_is_fitted(self)
        if self.pipeline_ is None:
            p = np.full(len(X), self.constant_probability_, dtype=float)
        else:
            p = self.pipeline_.predict_proba(numeric_feature_frame(X))[:, 1]
        return binary_prediction_frame(
            X,
            p,
            model_name=self.model_name,
            default_target=self.default_target,
            default_horizon=self.default_horizon,
        )

    def get_params(self) -> dict[str, Any]:
        return {
            "max_iter": self.max_iter,
            "learning_rate": self.learning_rate,
            "max_leaf_nodes": self.max_leaf_nodes,
            "l2_regularization": self.l2_regularization,
            "random_state": self.random_state,
        }

    def model_card(self) -> dict[str, Any]:
        return ModelCard(
            model_name=self.model_name,
            output_type=self.output_type,
            family="tree_ensemble",
            description="Histogram gradient boosting probability baseline.",
            params=self.get_params(),
            dependencies=("scikit-learn",),
        ).to_dict()


@dataclass
class HistGradientBoostingQuantileRegressor:
    """Histogram gradient boosting quantile interval baseline."""

    lo: float = 0.1
    hi: float = 0.9
    max_iter: int = 100
    learning_rate: float = 0.1
    max_leaf_nodes: int | None = 31
    l2_regularization: float = 0.0
    random_state: int | None = 0
    default_target: str = "target"
    default_horizon: str = "1m"
    model_name: str = "hist_gradient_boosting_quantile"
    output_type: str = "quantile"

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        **_: Any,
    ) -> HistGradientBoostingQuantileRegressor:
        target = _regression_target(y)
        _validate_quantiles(self.lo, self.hi)
        features = numeric_feature_frame(X)
        self.q_lo_model_ = self._make_pipeline(self.lo)
        self.q50_model_ = self._make_pipeline(0.5)
        self.q_hi_model_ = self._make_pipeline(self.hi)
        self.q_lo_model_.fit(features, target)
        self.q50_model_.fit(features, target)
        self.q_hi_model_.fit(features, target)
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray, **_: Any) -> pd.DataFrame:
        check_is_fitted(self)
        features = numeric_feature_frame(X)
        return quantile_prediction_frame(
            X,
            q_lo=self.q_lo_model_.predict(features),
            q50=self.q50_model_.predict(features),
            q_hi=self.q_hi_model_.predict(features),
            model_name=self.model_name,
            default_target=self.default_target,
            default_horizon=self.default_horizon,
        )

    def get_params(self) -> dict[str, Any]:
        return {
            "lo": self.lo,
            "hi": self.hi,
            "max_iter": self.max_iter,
            "learning_rate": self.learning_rate,
            "max_leaf_nodes": self.max_leaf_nodes,
            "l2_regularization": self.l2_regularization,
            "random_state": self.random_state,
        }

    def model_card(self) -> dict[str, Any]:
        return ModelCard(
            model_name=self.model_name,
            output_type=self.output_type,
            family="tree_ensemble",
            description="Histogram gradient boosting quantile interval baseline.",
            params=self.get_params(),
            dependencies=("scikit-learn",),
        ).to_dict()

    def _make_pipeline(self, quantile: float) -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        loss="quantile",
                        quantile=quantile,
                        learning_rate=self.learning_rate,
                        l2_regularization=self.l2_regularization,
                        max_iter=self.max_iter,
                        max_leaf_nodes=self.max_leaf_nodes,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )


__all__ = ["HistGradientBoostingProbabilityModel", "HistGradientBoostingQuantileRegressor"]
