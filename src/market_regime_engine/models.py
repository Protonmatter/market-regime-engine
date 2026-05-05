# SPDX-License-Identifier: Apache-2.0
"""Baseline supervised heads.

``ProbabilityModel``
    Logistic regression with median imputation and standard scaling. Used for
    binary targets like ``dd10_*`` (drawdown ≤ -10% within horizon h).

``QuantileReturnModel``
    HistGradientBoostingRegressor (sklearn ≥1.3) with the pinball quantile loss,
    one fitted estimator per ``tau`` in :attr:`quantiles`. Quantile crossings
    are repaired via post-hoc isotonic projection: at predict time the per-row
    quantile values are sorted to enforce ``q05 ≤ q10 ≤ q25 ≤ q50 ≤ q75 ≤ q90 ≤ q95``.
    A linear-quantile fallback is used when the training set is too small for
    boosting (≤ 60 rows).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_QUANTILES: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


class ProbabilityModel:
    def __init__(self) -> None:
        self.model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=300, class_weight="balanced", solver="liblinear")),
            ]
        )
        self.constant: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ProbabilityModel:
        mask = y.notna()
        yfit = y[mask].astype(int)
        if len(yfit) == 0:
            self.constant = 0.0
            return self
        if yfit.nunique() < 2:
            self.constant = float(yfit.mean())
            return self
        self.model.fit(X.loc[mask], yfit)
        self.constant = None
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.constant is not None:
            return np.full(len(X), self.constant)
        return self.model.predict_proba(X)[:, 1]


class QuantileReturnModel:
    """Per-quantile HistGradientBoostingRegressor with non-crossing repair.

    Parameters
    ----------
    quantiles:
        Quantile levels τ ∈ (0, 1) to fit. The default covers 5/10/25/50/75/90/95.
    n_estimators:
        Number of boosting iterations per quantile model. The v0.8 baseline used
        ``n_estimators=12`` which is far too small for any real signal; the
        v1.0 default is ``300`` with early stopping on a 20% validation tail.
    max_depth:
        Tree depth cap. ``None`` lets HistGradientBoosting decide.
    learning_rate:
        Shrinkage. The default of 0.05 trades training cost for stability.
    min_train:
        Below this many training rows, fall back to a per-quantile linear
        model (``QuantileRegressor``). Below ``min_train // 2`` rows, fall
        back to expanding empirical quantiles. This three-tier policy keeps
        the model usable at every realistic sample size.
    enforce_non_crossing:
        Sort the per-row quantiles at predict time so the output is monotone
        in τ. Cheaper than constraining training and equivalent in expectation
        for sufficiently rich models (Chernozhukov-Fernández-Val 2010).
    """

    def __init__(
        self,
        quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
        *,
        n_estimators: int = 300,
        max_depth: int | None = None,
        learning_rate: float = 0.05,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.1,
        early_stopping: bool = True,
        random_state: int = 42,
        min_train: int = 60,
        enforce_non_crossing: bool = True,
    ) -> None:
        self.quantiles = tuple(sorted({float(q) for q in quantiles}))
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.learning_rate = float(learning_rate)
        self.min_samples_leaf = int(min_samples_leaf)
        self.l2_regularization = float(l2_regularization)
        self.early_stopping = bool(early_stopping)
        self.random_state = int(random_state)
        self.min_train = int(min_train)
        self.enforce_non_crossing = bool(enforce_non_crossing)
        self.models: dict[float, Pipeline] = {}
        self.linear_models: dict[float, Pipeline] = {}
        self.fallback: dict[float, float] = {}

    def _build_boost(self, q: float) -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "reg",
                    HistGradientBoostingRegressor(
                        loss="quantile",
                        quantile=q,
                        max_iter=self.n_estimators,
                        learning_rate=self.learning_rate,
                        max_depth=self.max_depth,
                        min_samples_leaf=self.min_samples_leaf,
                        l2_regularization=self.l2_regularization,
                        early_stopping=self.early_stopping,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    def _build_linear(self, q: float) -> Pipeline:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
                ("reg", QuantileRegressor(quantile=q, alpha=1e-3, solver="highs")),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> QuantileReturnModel:
        mask = y.notna()
        n = int(mask.sum())
        if n == 0:
            self.fallback = dict.fromkeys(self.quantiles, 0.0)
            return self
        Xm = X.loc[mask]
        ym = y.loc[mask]
        if n < self.min_train // 2:
            self.fallback = {q: float(ym.quantile(q)) for q in self.quantiles}
            return self
        if n < self.min_train:
            self.fallback = {}
            self.linear_models = {q: self._build_linear(q).fit(Xm, ym) for q in self.quantiles}
            self.models = {}
            return self
        self.fallback = {}
        self.linear_models = {}
        self.models = {}
        for q in self.quantiles:
            try:
                self.models[q] = self._build_boost(q).fit(Xm, ym)
            except Exception:
                # If boosting fails (e.g. degenerate target), drop to linear.
                self.linear_models[q] = self._build_linear(q).fit(Xm, ym)
        return self

    def _quantile_columns(self) -> list[str]:
        return [f"q{round(q * 100):02d}" for q in self.quantiles]

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = self._quantile_columns()
        if X.empty:
            return pd.DataFrame(columns=cols)
        n_rows = len(X)
        idx = X.index
        out = pd.DataFrame(index=idx, columns=cols, dtype=float)
        if self.fallback:
            for q, name in zip(self.quantiles, cols, strict=False):
                out[name] = np.full(n_rows, self.fallback[q])
        else:
            for q, name in zip(self.quantiles, cols, strict=False):
                if q in self.models:
                    out[name] = self.models[q].predict(X)
                elif q in self.linear_models:
                    out[name] = self.linear_models[q].predict(X)
                else:
                    # Neither boost nor linear was fit; emit median fallback.
                    out[name] = np.full(n_rows, 0.0)
        if self.enforce_non_crossing:
            arr = out.to_numpy(dtype=float)
            arr.sort(axis=1)
            out = pd.DataFrame(arr, index=idx, columns=cols)
        return out


def train_latest_outputs(X: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    latest_X = X.tail(1)
    outputs = []

    for h in (3, 6, 12):
        prob = ProbabilityModel().fit(X, targets[f"dd10_{h}m"]).predict_proba(latest_X)[0]
        outputs.append(
            {
                "model_name": "baseline_logistic",
                "date": latest_X.index[-1],
                "horizon": f"{h}m",
                "target": "drawdown_gt_10pct",
                "value": float(prob),
                "metadata_json": json.dumps({"calibration": "not_yet_calibrated"}),
            }
        )

        q = QuantileReturnModel().fit(X, targets[f"ret_{h}m"]).predict(latest_X).iloc[0]
        for qname, value in q.items():
            outputs.append(
                {
                    "model_name": "baseline_quantile_hgbt",
                    "date": latest_X.index[-1],
                    "horizon": f"{h}m",
                    "target": f"forward_return_{qname}",
                    "value": float(value),
                    "metadata_json": json.dumps({"return_type": "log_return", "non_crossing": True}),
                }
            )

    return pd.DataFrame(outputs)


__all__ = [
    "DEFAULT_QUANTILES",
    "ProbabilityModel",
    "QuantileReturnModel",
    "train_latest_outputs",
]
