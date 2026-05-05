# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pandas as pd


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.nanmax(x)
    e = np.exp(x)
    return e / e.sum() if e.sum() > 0 else np.ones_like(x) / len(x)


def dynamic_weights(
    base_prior: pd.Series,
    losses: pd.Series | None = None,
    calibration_error: pd.Series | None = None,
    regime_fit: pd.Series | None = None,
    corr_penalty: pd.Series | None = None,
    instability: pd.Series | None = None,
) -> pd.Series:
    names = list(base_prior.index)
    score = np.log(base_prior.reindex(names).fillna(1.0 / len(names)).to_numpy())
    if regime_fit is not None:
        score += 0.5 * regime_fit.reindex(names).fillna(0.0).to_numpy()
    if losses is not None:
        score -= losses.reindex(names).fillna(losses.mean()).to_numpy()
    if calibration_error is not None:
        score -= 0.5 * calibration_error.reindex(names).fillna(calibration_error.mean()).to_numpy()
    if corr_penalty is not None:
        score -= 0.2 * corr_penalty.reindex(names).fillna(0.0).to_numpy()
    if instability is not None:
        score -= 0.2 * instability.reindex(names).fillna(0.0).to_numpy()
    return pd.Series(softmax(score), index=names)
