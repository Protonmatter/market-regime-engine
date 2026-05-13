# SPDX-License-Identifier: Apache-2.0
"""Covariate-conditioned BOCPD hazard.

The default BOCPD hazard is constant (or empirical-mean-of-runs). That fails
near regime boundaries where the prior probability of a change-point should
spike with macro stress. This module fits a small logistic regression of
"is the next month a change-point?" on a vector of covariates (e.g. credit
spread, change-point probability of the previous BOCPD pass, VIX-like proxy)
and exposes a callable ``hazard(t, covariates_t)`` that the BOCPD recursion
can consume in place of the constant rate.

The training set is built from a labelled or self-labelled regime path:

- If ``decoded_regime`` changes at ``t``, ``y_t = 1`` (change-point).
- Otherwise ``y_t = 0``.

The model is a calibrated logistic regression on standardized covariates
with an L2 prior. ``hazard(...)`` returns a clipped probability in
``[1e-4, 0.5]`` so it remains a sane BOCPD hazard.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class CovariateBOCPDHazard:
    """Logistic-regression hazard with covariates.

    The class deliberately follows the same calling convention as the
    constant ``hazard`` parameter of :class:`market_regime_engine.bocpd.MultivariateNIWBOCPD`,
    so the BOCPD recursion can plug it in without code changes::

        h = CovariateBOCPDHazard().fit(regime_path, covariates).hazard

    where ``hazard`` is a callable ``(timestamp, covariate_vector) -> float``.
    """

    floor: float = 1e-4
    ceiling: float = 0.5
    fallback_hazard: float = 1.0 / 48.0

    pipeline: Pipeline | None = None
    feature_columns: list[str] = field(default_factory=list)
    fitted: bool = False

    @staticmethod
    def _changepoint_labels(regime_path: pd.Series) -> pd.Series:
        idx = regime_path.index
        prev = regime_path.shift(1)
        y = (regime_path != prev).astype(int)
        y.iloc[0] = 0
        return pd.Series(y.values, index=idx, name="cp")

    def fit(self, regime_path: pd.Series, covariates: pd.DataFrame) -> CovariateBOCPDHazard:
        if regime_path is None or covariates is None or covariates.empty:
            return self
        y = self._changepoint_labels(regime_path)
        df = covariates.copy()
        df.index = pd.to_datetime(df.index)
        y.index = pd.to_datetime(y.index)
        joined = df.join(y, how="inner").dropna()
        if joined.empty or joined["cp"].nunique() < 2 or len(joined) < 36:
            return self
        self.feature_columns = list(covariates.columns)
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, solver="liblinear", class_weight="balanced", max_iter=400)),
            ]
        )
        self.pipeline.fit(joined[self.feature_columns], joined["cp"].astype(int))
        self.fitted = True
        return self

    def hazard_at(self, covariates_row: pd.Series | np.ndarray) -> float:
        if not self.fitted or self.pipeline is None:
            return float(self.fallback_hazard)
        if isinstance(covariates_row, pd.Series):
            x = covariates_row.reindex(self.feature_columns).fillna(0.0).to_numpy(float).reshape(1, -1)
        else:
            x = np.asarray(covariates_row, dtype=float).reshape(1, -1)
        prob = float(self.pipeline.predict_proba(x)[0, 1])
        return float(np.clip(prob, self.floor, self.ceiling))

    def hazard_series(self, covariates: pd.DataFrame) -> pd.Series:
        if not self.fitted or covariates is None or covariates.empty:
            return pd.Series([], dtype=float)
        # v1.6.0 (REVIEW_DEEP_V1_5_2.md §4.2 — safety-critical):
        # ``self.fitted`` does not statically imply ``self.pipeline is not
        # None`` so mypy correctly flags the bare ``.predict_proba`` call.
        # More importantly a caller that mutates ``fitted=True`` without
        # ever calling ``.fit(...)`` (e.g. unpickling a partially
        # constructed object, or a test fixture) would hit a confusing
        # ``AttributeError`` deep inside sklearn instead of a clean
        # contract violation. Raise an explicit ``RuntimeError`` so the
        # failure mode is loud and pinned to the API boundary.
        if self.pipeline is None:
            raise RuntimeError(
                "hazard classifier not fitted; call .fit(regime_path, covariates) first"
            )
        Xp = covariates.reindex(columns=self.feature_columns, fill_value=0.0)
        prob = self.pipeline.predict_proba(Xp.to_numpy(float))[:, 1]
        return pd.Series(np.clip(prob, self.floor, self.ceiling), index=covariates.index, name="hazard_t")


__all__ = ["CovariateBOCPDHazard"]
