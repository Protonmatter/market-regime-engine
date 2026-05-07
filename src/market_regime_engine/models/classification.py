# SPDX-License-Identifier: Apache-2.0
"""Binary baseline forecast models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
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


def _binary_target(y: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("y must contain at least one row")
    return (arr > 0).astype(int)


def _smooth_rate(y: np.ndarray, alpha: float) -> float:
    return float((float(np.sum(y)) + alpha) / (float(len(y)) + (2.0 * alpha)))


@dataclass
class PersistenceClassifier:
    """Smoothed prior-rate binary model."""

    alpha: float = 1.0
    default_target: str = "target"
    default_horizon: str = "1m"
    model_name: str = "persistence"
    output_type: str = "binary"

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray, **_: Any) -> PersistenceClassifier:
        del X
        target = _binary_target(y)
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0")
        self.probability_ = _smooth_rate(target, self.alpha)
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
            "alpha": self.alpha,
            "default_target": self.default_target,
            "default_horizon": self.default_horizon,
        }

    def model_card(self) -> dict[str, Any]:
        return ModelCard(
            model_name=self.model_name,
            output_type=self.output_type,
            family="baseline",
            description="Smoothed historical positive-rate binary model.",
            params=self.get_params(),
        ).to_dict()


@dataclass
class LogisticRegressionClassifier:
    """Median-imputed, standardized logistic regression model."""

    C: float = 1.0
    max_iter: int = 1000
    class_weight: str | dict[int, float] | None = None
    random_state: int | None = 0
    default_target: str = "target"
    default_horizon: str = "1m"
    model_name: str = "logistic_regression"
    output_type: str = "binary"

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray, **_: Any) -> LogisticRegressionClassifier:
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
                            max_iter=self.max_iter,
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
            "C": self.C,
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
            description="Median-imputed standardized logistic regression binary model.",
            params=self.get_params(),
            dependencies=("scikit-learn",),
        ).to_dict()


@dataclass
class RandomForestClassifierModel:
    """Random forest probability model."""

    n_estimators: int = 200
    max_depth: int | None = None
    min_samples_leaf: int = 5
    random_state: int | None = 0
    n_jobs: int | None = None
    default_target: str = "target"
    default_horizon: str = "1m"
    model_name: str = "random_forest"
    output_type: str = "binary"

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray, **_: Any) -> RandomForestClassifierModel:
        target = _binary_target(y)
        features = numeric_feature_frame(X)
        self.feature_columns_ = list(features.columns)
        self.constant_probability_ = _smooth_rate(target, alpha=1.0)
        self.pipeline_ = None
        if np.unique(target).size >= 2:
            self.pipeline_ = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        SklearnRandomForestClassifier(
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
            description="Median-imputed random forest probability model.",
            params=self.get_params(),
            dependencies=("scikit-learn",),
        ).to_dict()


__all__ = ["LogisticRegressionClassifier", "PersistenceClassifier", "RandomForestClassifierModel"]
