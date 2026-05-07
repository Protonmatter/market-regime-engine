# SPDX-License-Identifier: Apache-2.0
"""Additional baseline models and compatibility exports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from market_regime_engine.models.base import (
    ModelCard,
    binary_prediction_frame,
    check_is_fitted,
    numeric_feature_frame,
)
from market_regime_engine.models.classification import (
    LogisticRegressionClassifier,
    PersistenceClassifier,
    RandomForestClassifierModel,
    _binary_target,
    _smooth_rate,
)


@dataclass
class RollingBaseRateClassifier:
    """Smoothed rolling base-rate model."""

    window: int = 60
    alpha: float = 1.0
    default_target: str = "target"
    default_horizon: str = "1m"
    model_name: str = "rolling_base_rate"
    output_type: str = "binary"

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray, **_: Any) -> RollingBaseRateClassifier:
        del X
        target = _binary_target(y)
        if self.window < 1:
            raise ValueError("window must be >= 1")
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0")
        self.probability_ = _smooth_rate(target[-self.window :], self.alpha)
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray, **_: Any) -> pd.DataFrame:
        check_is_fitted(self)
        p = np.full(len(X), self.probability_, dtype=float)
        return binary_prediction_frame(
            X,
            p,
            model_name=self.model_name,
            default_target=self.default_target,
            default_horizon=self.default_horizon,
        )

    def get_params(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "alpha": self.alpha,
            "default_target": self.default_target,
            "default_horizon": self.default_horizon,
        }

    def model_card(self) -> dict[str, Any]:
        return ModelCard(
            model_name=self.model_name,
            output_type=self.output_type,
            family="baseline",
            description="Smoothed rolling historical base-rate model.",
            params=self.get_params(),
        ).to_dict()


@dataclass
class ElasticNetLogisticClassifier:
    """Elastic-net logistic probability model."""

    C: float = 1.0
    l1_ratio: float = 0.5
    max_iter: int = 1000
    class_weight: str | dict[int, float] | None = None
    random_state: int | None = 0
    default_target: str = "target"
    default_horizon: str = "1m"
    model_name: str = "elastic_net_logistic"
    output_type: str = "binary"

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray, **_: Any) -> ElasticNetLogisticClassifier:
        target = _binary_target(y)
        features = numeric_feature_frame(X)
        self.feature_columns_ = list(features.columns)
        self.constant_probability_ = _smooth_rate(target, alpha=1.0)
        self.pipeline_ = None
        if np.unique(target).size >= 2:
            self.pipeline_ = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=self.C,
                            class_weight=self.class_weight,
                            l1_ratio=self.l1_ratio,
                            max_iter=self.max_iter,
                            penalty="elasticnet",
                            random_state=self.random_state,
                            solver="saga",
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
            "C": self.C,
            "l1_ratio": self.l1_ratio,
            "max_iter": self.max_iter,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            "default_target": self.default_target,
            "default_horizon": self.default_horizon,
        }

    def model_card(self) -> dict[str, Any]:
        return ModelCard(
            model_name=self.model_name,
            output_type=self.output_type,
            family="linear",
            description="Median-imputed standardized elastic-net logistic probability model.",
            params=self.get_params(),
            dependencies=("scikit-learn",),
        ).to_dict()


__all__ = [
    "ElasticNetLogisticClassifier",
    "LogisticRegressionClassifier",
    "PersistenceClassifier",
    "RandomForestClassifierModel",
    "RollingBaseRateClassifier",
]
