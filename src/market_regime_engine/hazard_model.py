# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from market_regime_engine.features import feature_matrix


def _monthly_hazard_labels(recession_labels: pd.DataFrame) -> pd.Series:
    if recession_labels is None or recession_labels.empty:
        return pd.Series(dtype=float)
    frame = recession_labels.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").set_index("date")
    rec = frame["recession"].astype(float)
    # Event starts when next month enters recession while current month is not in recession.
    next_rec = rec.shift(-1)
    y = ((rec <= 0) & (next_rec > 0)).astype(float)
    # Months already in recession are not at risk of starting a recession.
    y[rec > 0] = np.nan
    return y


class DiscreteTimeHazardModel:
    """Fitted discrete-time recession-start hazard model.

    This is intentionally simple: a calibrated logistic-style monthly hazard fitted on
    point-in-time features. Horizon probabilities are computed as survival complements.
    """

    def __init__(self, *, class_weight: dict | str | None = None) -> None:
        # ``class_weight`` is now configurable; the v1.0 default was
        # ``"balanced"`` which up-weights minority-class observations and
        # systematically biases probability estimates upward. The v1.2 default
        # is ``None`` because Platt + isotonic + conformal layers downstream
        # already correct calibration without that bias term. Callers that
        # want the legacy behavior can pass ``class_weight="balanced"``
        # explicitly.
        self.class_weight = class_weight
        self.pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=400, class_weight=class_weight, solver="liblinear")),
            ]
        )
        self.constant: float | None = None
        self.feature_columns: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> DiscreteTimeHazardModel:
        y = y.copy()
        y.index = pd.to_datetime(y.index)
        joined = X.copy()
        joined.index = pd.to_datetime(joined.index)
        joined = joined.join(y.rename("hazard"), how="inner")
        joined = joined.dropna(subset=["hazard"])
        self.feature_columns = list(X.columns)
        if joined.empty:
            self.constant = 0.01
            return self
        yy = joined["hazard"].astype(int)
        if yy.nunique() < 2 or len(yy) < 36:
            self.constant = float(np.clip(yy.mean() if len(yy) else 0.01, 0.005, 0.25))
            return self
        self.pipeline.fit(joined[self.feature_columns], yy)
        self.constant = None
        return self

    def predict_monthly_hazard(self, X: pd.DataFrame) -> np.ndarray:
        if X.empty:
            return np.array([])
        Xp = X.reindex(columns=self.feature_columns, fill_value=np.nan) if self.feature_columns else X
        if self.constant is not None:
            return np.full(len(Xp), self.constant)
        return self.pipeline.predict_proba(Xp)[:, 1]

    @staticmethod
    def horizon_probability(monthly_hazard: np.ndarray, horizon_months: int) -> np.ndarray:
        """Constant-hazard horizon survival.

        Equivalent to ``1 - (1 - h0)^H``. Use this as the *fallback* for live
        forecasts where future feature paths are unknown.
        """
        h = np.clip(monthly_hazard.astype(float), 0.0001, 0.95)
        return 1.0 - np.power(1.0 - h, horizon_months)

    @staticmethod
    def horizon_probability_path(
        monthly_hazard_path: np.ndarray | pd.Series,
        horizon_months: int,
    ) -> np.ndarray:
        """Path-aware horizon survival.

        Given a *time-indexed* monthly hazard series, the cumulative recession
        probability at row ``t`` over ``horizon_months`` periods is
        ``1 - prod_{k=1..H}(1 - h_{t+k})``. This is the correct aggregation
        for backtests where the hazard is re-estimated each month.

        Rows whose forward window extends past the last observation are
        truncated to whatever periods are available; callers should treat those
        truncated estimates as biased downwards and surface that fact in the
        validation layer.
        """
        h = np.clip(np.asarray(monthly_hazard_path, dtype=float), 0.0001, 0.95)
        n = len(h)
        out = np.full(n, np.nan, dtype=float)
        for t in range(n):
            window = h[t + 1 : t + 1 + horizon_months]
            if window.size == 0:
                out[t] = h[t]
            else:
                out[t] = float(1.0 - np.prod(1.0 - window))
        return out


def train_fitted_hazard_outputs(
    features: pd.DataFrame,
    recession_labels: pd.DataFrame,
    horizons: tuple[int, ...] = (3, 6, 12),
    *,
    monthly_hazard_path: pd.Series | np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train the hazard model and emit per-horizon recession probabilities.

    By default the caller is in *live forecast* mode: only the latest fitted
    monthly hazard is known. The function then falls back to the constant-
    hazard horizon survival ``1 - (1 - h0)^H`` and marks each emitted row's
    ``metadata_json["assumption"] = "constant_hazard"`` so downstream
    consumers can tell that the multi-month path was synthesized rather than
    backtested.

    When ``monthly_hazard_path`` is supplied (the OOS backtest path), the
    function calls :meth:`DiscreteTimeHazardModel.horizon_probability_path` on
    the trailing window so the horizon probability uses the actual
    re-estimated hazard at each future month.
    """
    if features is None or features.empty:
        return pd.DataFrame(), pd.DataFrame()
    X = feature_matrix(features)
    if X.empty:
        return pd.DataFrame(), pd.DataFrame()
    y = _monthly_hazard_labels(recession_labels)
    model = DiscreteTimeHazardModel().fit(X, y)
    monthly = model.predict_monthly_hazard(X)
    latest_date = X.index[-1]
    latest_hazard = float(monthly[-1]) if len(monthly) else 0.01

    rows = []
    rows.append(
        {
            "model_name": "hazard_logit_v0_7",
            "date": latest_date,
            "horizon": "1m",
            "target": "monthly_recession_hazard",
            "value": latest_hazard,
            "metadata_json": json.dumps(
                {"model": "discrete_time_logit", "fitted": model.constant is None},
                sort_keys=True,
            ),
        }
    )
    use_path = monthly_hazard_path is not None and len(np.asarray(monthly_hazard_path)) > 0
    if use_path:
        path_arr = np.asarray(monthly_hazard_path, dtype=float)
        for horizon in horizons:
            p_path = DiscreteTimeHazardModel.horizon_probability_path(path_arr, horizon)
            p = float(p_path[-1]) if len(p_path) else float("nan")
            rows.append(
                {
                    "model_name": "hazard_logit_v0_7",
                    "date": latest_date,
                    "horizon": f"{horizon}m",
                    "target": "recession_probability",
                    "value": p,
                    "metadata_json": json.dumps(
                        {
                            "model": "discrete_time_logit",
                            "monthly_hazard": latest_hazard,
                            "assumption": "path",
                            "path_length": len(path_arr),
                        },
                        sort_keys=True,
                    ),
                }
            )
    else:
        for horizon in horizons:
            p = float(DiscreteTimeHazardModel.horizon_probability(np.array([latest_hazard]), horizon)[0])
            rows.append(
                {
                    "model_name": "hazard_logit_v0_7",
                    "date": latest_date,
                    "horizon": f"{horizon}m",
                    "target": "recession_probability",
                    "value": p,
                    "metadata_json": json.dumps(
                        {
                            "model": "discrete_time_logit",
                            "monthly_hazard": latest_hazard,
                            "assumption": "constant_hazard",
                        },
                        sort_keys=True,
                    ),
                }
            )
    diag = pd.DataFrame(
        [
            {
                "date": latest_date,
                "model_name": "hazard_logit_v0_7",
                "observations": int(y.dropna().shape[0]),
                "events": int(y.dropna().sum()) if not y.empty else 0,
                "feature_count": int(X.shape[1]),
                "constant_fallback": bool(model.constant is not None),
                "latest_monthly_hazard": latest_hazard,
                "metadata_json": json.dumps(
                    {"target_definition": "next-month recession start while currently not in recession"}, sort_keys=True
                ),
            }
        ]
    )
    return pd.DataFrame(rows), diag


def hazard_backtest_matrix(
    features: pd.DataFrame, recession_labels: pd.DataFrame, min_train: int = 120, step: int = 3
) -> pd.DataFrame:
    X = feature_matrix(features)
    y = _monthly_hazard_labels(recession_labels)
    joined = X.join(y.rename("actual"), how="inner")
    rows = []
    if joined.empty:
        return pd.DataFrame()
    # Re-use the latest fit's monthly hazards as a forward-looking path for
    # ``horizon_probability_path``. Each as-of date we score also gets the
    # forward window ``[t+1, t+H]`` of monthly hazards from the same model,
    # which is the correct aggregation under the v1.2 fix.
    for i in range(min_train, len(joined), step):
        train = joined.iloc[:i].dropna(subset=["actual"])
        if train.empty:
            continue
        model = DiscreteTimeHazardModel().fit(train[X.columns], train["actual"])
        # Score the entire panel from this fit so we can build a forward
        # monthly-hazard path. The path is a cheap by-product (ndarray of
        # length ``len(joined) - i``).
        forward_X = joined.iloc[i:][X.columns]
        if forward_X.empty:
            continue
        forward_h = model.predict_monthly_hazard(forward_X)
        monthly = float(forward_h[0])
        for horizon in (3, 6, 12):
            path_p = DiscreteTimeHazardModel.horizon_probability_path(forward_h, horizon)
            p_val = (
                float(path_p[0])
                if len(path_p)
                else float(DiscreteTimeHazardModel.horizon_probability(np.array([monthly]), horizon)[0])
            )
            rows.append(
                {
                    "date": joined.index[i],
                    "model_name": "hazard_logit_v0_7_oos",
                    "horizon": f"{horizon}m",
                    "target": "recession_probability",
                    "value": p_val,
                    "actual": np.nan,  # horizon labels can be joined by the stacking layer
                    "metadata_json": json.dumps({"assumption": "path"}, sort_keys=True),
                }
            )
    return pd.DataFrame(rows)
