# SPDX-License-Identifier: Apache-2.0
"""Backward-compatible model names used by the legacy backtest CLI.

The v1.6 model zoo introduced evidence-frame oriented model classes. Older
backtest paths still expect ``ProbabilityModel.predict_proba`` and a
``QuantileReturnModel`` that emits q05/q10/q25/q50/q75/q90/q95 columns. These
wrappers preserve that public surface while delegating to scikit-learn models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from market_regime_engine.models.base import check_is_fitted, numeric_feature_frame


@dataclass
class ProbabilityModel:
    """Legacy binary probability model with ``predict_proba`` output."""

    n_estimators: int = 100
    learning_rate: float = 0.1
    min_train: int = 50
    random_state: int | None = 0

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray, **_: Any) -> ProbabilityModel:
        target = pd.Series(y).astype(float)
        mask = target.notna()
        target = target.loc[mask]
        features = numeric_feature_frame(X).loc[mask.to_numpy()]
        self.fallback_probability_ = float(np.clip((target.sum() + 1.0) / (len(target) + 2.0), 1e-6, 1.0 - 1e-6))
        self.pipeline_ = None
        if len(target) >= max(4, self.min_train) and target.nunique() >= 2:
            self.pipeline_ = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            max_iter=self.n_estimators,
                            learning_rate=self.learning_rate,
                            random_state=self.random_state,
                        ),
                    ),
                ]
            )
        elif target.nunique() >= 2:
            self.pipeline_ = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", LogisticRegression(max_iter=500, random_state=self.random_state)),
                ]
            )
        if self.pipeline_ is not None:
            self.pipeline_.fit(features, target.astype(int))
        self.is_fitted_ = True
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        if self.pipeline_ is None:
            return np.full(len(X), self.fallback_probability_, dtype=float)
        p = self.pipeline_.predict_proba(numeric_feature_frame(X))[:, 1]
        return np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.predict_proba(X)


@dataclass
class QuantileReturnModel:
    """Legacy quantile-return model emitting q05..q95 columns."""

    n_estimators: int = 100
    learning_rate: float = 0.1
    min_train: int = 120
    random_state: int | None = 0

    quantiles: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    columns: tuple[str, ...] = ("q05", "q10", "q25", "q50", "q75", "q90", "q95")

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray, **_: Any) -> QuantileReturnModel:
        target = pd.Series(y).astype(float).dropna()
        if target.empty:
            target = pd.Series([0.0])
        self.empirical_quantiles_ = np.quantile(target.to_numpy(float), self.quantiles)
        self.models_: dict[str, Pipeline] = {}
        if len(target) >= max(4, self.min_train // 2):
            features = numeric_feature_frame(X).loc[target.index]
            for col, tau in zip(self.columns, self.quantiles, strict=True):
                model = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "model",
                            HistGradientBoostingRegressor(
                                loss="quantile",
                                quantile=tau,
                                max_iter=self.n_estimators,
                                learning_rate=self.learning_rate,
                                random_state=self.random_state,
                            ),
                        ),
                    ]
                )
                model.fit(features, target)
                self.models_[col] = model
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        check_is_fitted(self)
        index = X.index if isinstance(X, pd.DataFrame) else pd.RangeIndex(len(X))
        if not self.models_:
            arr = np.tile(self.empirical_quantiles_, (len(index), 1))
            return pd.DataFrame(arr, index=index, columns=self.columns)
        features = numeric_feature_frame(X)
        raw = np.column_stack([self.models_[col].predict(features) for col in self.columns])
        repaired = np.maximum.accumulate(raw, axis=1)
        return pd.DataFrame(repaired, index=index, columns=self.columns)


__all__ = ["ProbabilityModel", "QuantileReturnModel"]
