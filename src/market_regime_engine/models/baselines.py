# SPDX-License-Identifier: Apache-2.0
"""Additional baseline models and compatibility exports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from market_regime_engine.models.base import ModelCard, binary_prediction_frame, check_is_fitted
from market_regime_engine.models.classification import (
    LogisticRegressionClassifier,
    PersistenceClassifier,
    RandomForestClassifierModel,
    _binary_target,
    _smooth_rate,
)


@dataclass
class RollingBaseRateClassifier:
    """Smoothed rolling base-rate model.

    Walk-forward callers should refit at each origin for a true rolling path.
    """

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
        return {"window": self.window, "alpha": self.alpha}

    def model_card(self) -> dict[str, Any]:
        return ModelCard(
            model_name=self.model_name,
            output_type=self.output_type,
            family="baseline",
            description="Smoothed rolling historical base-rate model.",
            params=self.get_params(),
        ).to_dict()


@dataclass
class ElasticNetLogisticClassifier(LogisticRegressionClassifier):
    """Elastic-net logistic model using scikit-learn's saga solver."""

    l1_ratio: float = 0.5
    model_name: str = "elastic_net_logistic"

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray, **kwargs: Any) -> ElasticNetLogisticClassifier:
        super().fit(X, y, **kwargs)
        if self.pipeline_ is not None:
            model = self.pipeline_.named_steps["model"]
            model.set_params(penalty="elasticnet", solver="saga", l1_ratio=self.l1_ratio)
            model.fit(self.pipeline_.named_steps["scaler"].transform(self.pipeline_.named_steps["imputer"].transform(X.drop(columns=[c for c in ("date", "target", "horizon", "regime", "y") if c in X.columns], errors="ignore"))), _binary_target(y))
        return self

    def get_params(self) -> dict[str, Any]:
        params = super().get_params()
        params["l1_ratio"] = self.l1_ratio
        return params

    def model_card(self) -> dict[str, Any]:
        return ModelCard(
            model_name=self.model_name,
            output_type=self.output_type,
            family="linear",
            description="Elastic-net logistic probability model.",
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
