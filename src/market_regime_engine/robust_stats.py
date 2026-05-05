# SPDX-License-Identifier: Apache-2.0
"""PIT-respecting robust statistics.

These helpers replace the legacy ``rolling_z`` (mean / std-based) with
median- and MAD-based versions. Heavy-tailed macro series and one-off
outliers (Volcker tightening, GFC, COVID) make the standard z-score
unreliable as an out-of-sample measure of "how unusual is the current
print?". The robust counterparts reproduce most of the desirable z-score
properties while shrugging off a single 5σ outlier.

All functions are *strictly point-in-time*: every statistic at row ``t`` is
computed from rows ``< t`` (``shift(1)`` after the rolling/expanding window).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_MAD_TO_STD = 1.4826  # Gaussian-consistent MAD scaling


def rolling_robust_z(
    s: pd.Series,
    window: int = 60,
    min_periods: int = 24,
) -> pd.Series:
    """PIT-respecting robust z-score using rolling median and MAD."""
    median = s.rolling(window, min_periods=min_periods).median().shift(1)
    deviation = (s - median).abs()
    mad = deviation.rolling(window, min_periods=min_periods).median().shift(1)
    scale = (mad * _MAD_TO_STD).replace(0, np.nan)
    return ((s - median) / scale).replace([np.inf, -np.inf], np.nan)


def expanding_robust_z(
    s: pd.Series,
    min_periods: int = 24,
) -> pd.Series:
    """PIT-respecting expanding-window robust z-score."""
    median = s.expanding(min_periods=min_periods).median().shift(1)
    deviation = (s - median).abs()
    mad = deviation.expanding(min_periods=min_periods).median().shift(1)
    scale = (mad * _MAD_TO_STD).replace(0, np.nan)
    return ((s - median) / scale).replace([np.inf, -np.inf], np.nan)


def rolling_winsorized_z(
    s: pd.Series,
    window: int = 60,
    min_periods: int = 24,
    *,
    quantile_lo: float = 0.01,
    quantile_hi: float = 0.99,
) -> pd.Series:
    """Rolling z-score with winsorized inputs (caps at the 1/99 percentiles).

    Useful when MAD is too aggressive — particularly for series with very fat
    tails like oil shocks where MAD-rescaling deflates real signals.
    """
    lo = s.rolling(window, min_periods=min_periods).quantile(quantile_lo).shift(1)
    hi = s.rolling(window, min_periods=min_periods).quantile(quantile_hi).shift(1)
    clipped = s.clip(lower=lo, upper=hi)
    mu = clipped.rolling(window, min_periods=min_periods).mean().shift(1)
    sd = clipped.rolling(window, min_periods=min_periods).std(ddof=1).shift(1).replace(0, np.nan)
    return ((s - mu) / sd).replace([np.inf, -np.inf], np.nan)


def robust_zscore_frame(
    frame: pd.DataFrame,
    *,
    method: str = "mad",
    window: int = 60,
    min_periods: int = 24,
) -> pd.DataFrame:
    """Apply a robust z-score per column using the requested method."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    out_cols: dict[str, pd.Series] = {}
    for col in frame.columns:
        s = frame[col].astype(float)
        if method == "mad":
            out_cols[col] = rolling_robust_z(s, window=window, min_periods=min_periods)
        elif method == "expanding_mad":
            out_cols[col] = expanding_robust_z(s, min_periods=min_periods)
        elif method == "winsorized":
            out_cols[col] = rolling_winsorized_z(s, window=window, min_periods=min_periods)
        else:
            raise ValueError(f"unknown robust z method: {method}")
    return pd.DataFrame(out_cols, index=frame.index)


__all__ = [
    "expanding_robust_z",
    "robust_zscore_frame",
    "rolling_robust_z",
    "rolling_winsorized_z",
]
