# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pandas as pd


def expanding_event_rate_baseline(y: pd.Series, *, min_train: int = 96, step: int = 1) -> pd.DataFrame:
    """Naive binary benchmark: historical expanding event rate.

    A probabilistic model has no right to promotion if it cannot beat this boring creature.
    """
    y = y.sort_index()
    rows = []
    for i in range(min_train, len(y), step):
        hist = y.iloc[:i].dropna()
        obs = y.iloc[i]
        if hist.empty or pd.isna(obs):
            continue
        rows.append({"date": y.index[i], "y": float(obs), "p": float(hist.mean()), "benchmark": "expanding_event_rate"})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(columns=["y", "p", "benchmark"])


def previous_event_baseline(y: pd.Series, *, min_train: int = 96, step: int = 1) -> pd.DataFrame:
    """Naive binary benchmark: previous observed event, shrunk toward expanding mean."""
    y = y.sort_index()
    rows = []
    for i in range(min_train, len(y), step):
        hist = y.iloc[:i].dropna()
        obs = y.iloc[i]
        if hist.empty or pd.isna(obs):
            continue
        last = float(hist.iloc[-1])
        mean = float(hist.mean())
        p = 0.75 * last + 0.25 * mean
        rows.append(
            {
                "date": y.index[i],
                "y": float(obs),
                "p": float(np.clip(p, 0.01, 0.99)),
                "benchmark": "previous_event_shrunk",
            }
        )
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(columns=["y", "p", "benchmark"])


def expanding_quantile_baseline(
    y: pd.Series,
    *,
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.50, 0.75, 0.95),
    min_train: int = 120,
    step: int = 1,
) -> pd.DataFrame:
    """Naive return distribution benchmark: expanding historical quantiles."""
    y = y.sort_index()
    rows = []
    for i in range(min_train, len(y), step):
        hist = y.iloc[:i].dropna()
        obs = y.iloc[i]
        if hist.empty or pd.isna(obs):
            continue
        row = {"date": y.index[i], "y": float(obs), "benchmark": "expanding_return_quantiles"}
        for q in quantiles:
            row[f"q{int(q * 100):02d}"] = float(hist.quantile(q))
        rows.append(row)
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
