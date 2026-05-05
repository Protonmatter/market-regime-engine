# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RollingMultivariateChangePoint:
    window: int = 36
    min_periods: int = 18
    ridge: float = 1e-4
    threshold: float = 3.0

    def score(self, x: pd.DataFrame) -> pd.DataFrame:
        x = x.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
        rows = []
        arr = x.to_numpy(float)
        for i, date in enumerate(x.index):
            if i < self.min_periods:
                rows.append({"date": date, "change_point_prob": 0.0, "mahalanobis": 0.0})
                continue
            hist = arr[max(0, i - self.window) : i]
            mu = hist.mean(axis=0)
            cov = np.cov(hist, rowvar=False)
            cov = np.atleast_2d(cov) + np.eye(arr.shape[1]) * self.ridge
            diff = arr[i] - mu
            md = float(np.sqrt(max(diff @ np.linalg.pinv(cov) @ diff.T, 0.0)))
            prob = 1.0 / (1.0 + np.exp(-(md - self.threshold)))
            rows.append({"date": date, "change_point_prob": prob, "mahalanobis": md})
        return pd.DataFrame(rows)
