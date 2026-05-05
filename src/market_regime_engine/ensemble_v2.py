# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pandas as pd


def softmax(scores: pd.Series | dict[str, float], temperature: float = 1.0) -> dict[str, float]:
    s = pd.Series(scores, dtype=float)
    if s.empty:
        return {}
    vals = (s / max(temperature, 1e-9)).to_numpy()
    vals = vals - np.nanmax(vals)
    ex = np.exp(vals)
    denom = float(np.nansum(ex))
    if denom <= 0 or not np.isfinite(denom):
        return {k: 1.0 / len(s) for k in s.index}
    return {k: float(v / denom) for k, v in zip(s.index, ex, strict=False)}


def dynamic_model_weights(
    *,
    losses: dict[str, float],
    calibration_errors: dict[str, float] | None = None,
    regime_fit: dict[str, float] | None = None,
    correlation_penalty: dict[str, float] | None = None,
    change_point_prob: float = 0.0,
    staleness: dict[str, float] | None = None,
) -> dict[str, float]:
    calibration_errors = calibration_errors or {}
    regime_fit = regime_fit or {}
    correlation_penalty = correlation_penalty or {}
    staleness = staleness or {}
    names = set(losses) | set(calibration_errors) | set(regime_fit) | set(correlation_penalty) | set(staleness)
    scores = {}
    for n in names:
        scores[n] = (
            -1.8 * float(losses.get(n, 0.0))
            - 0.9 * float(calibration_errors.get(n, 0.0))
            + 0.8 * float(regime_fit.get(n, 0.0))
            - 0.4 * float(correlation_penalty.get(n, 0.0))
            - float(change_point_prob) * float(staleness.get(n, 0.0))
        )
    return softmax(scores)


def mix_binary_probabilities(probs: dict[str, float], weights: dict[str, float]) -> float:
    total = 0.0
    denom = 0.0
    for name, p in probs.items():
        w = float(weights.get(name, 0.0))
        total += w * float(p)
        denom += w
    return float(total / denom) if denom > 0 else float(np.nan)
