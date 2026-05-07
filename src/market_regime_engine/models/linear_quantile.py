# SPDX-License-Identifier: Apache-2.0
"""Linear quantile-regression baseline model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from market_regime_engine.models.base import ModelCard, check_is_fitted, numeric_feature_frame, quantile_prediction_frame
from market_regime_engine.models.regression import _regression_target, _validate_quantiles


@dataclass
class LinearQuantileRegressor:
    """Linear quantile interval baseline using scikit-learn QuantileRegressor."""

    lo: float = 0.1
    hi: float = 0.9
    alpha: float = 0.0
    solver: str = "highs"
    default_target: str = "target"
    default_horizon: str = "1m"
    model_name: str = "linear_quantile"
    output_type: str = "quantile"

    def fit(self, X: pd.DataFrame, y: pd.Series, **_: Any) -> LinearQuantileRegressor:
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

    def predict(self, X: pd.DataFrame, **_: Any) -> pd.DataFrame:
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
            "alpha": self.alpha,
            "solver": self.solver,
            "default_target": self.default_target,
            "default_horizon": self.default_horizon,
        }

    def model_card(self) -> dict[str, Any]:
        return ModelCard(
            model_name=self.model_name,
            output_type=self.output_type,
            family="linear",
            description="Median-imputed standardized linear quantile-regression interval model.",
            params=self.get_params(),
            dependencies=("scikit-learn",),
        ).to_dict()

    def _make_pipeline(self, quantile: float) -> Pipeline:
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", QuantileRegressor(alpha=self.alpha, quantile=quantile, solver=self.solver)),
            ]
        )


__all__ = ["LinearQuantileRegressor"]
