# SPDX-License-Identifier: Apache-2.0
"""Online conformal calibration primitives for nonstationary time series."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def _inflated_quantile(values: Sequence[float], alpha: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return float("inf")
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), n)
    return float(np.sort(arr)[rank - 1])


@dataclass
class EnbPIInterval:
    """Ensemble batch prediction intervals via residual quantiles.

    This is a lightweight EnbPI-style conformal wrapper. It expects one or more
    base-model predictions and calibrates absolute residuals against realized
    observations, then applies the residual quantile to future ensemble means.
    """

    alpha: float = 0.10
    residual_quantile: float = 0.0
    fitted_n: int = 0

    def fit(self, predictions: pd.DataFrame, y: Sequence[float]) -> EnbPIInterval:
        if predictions is None or predictions.empty:
            self.residual_quantile = float("inf")
            self.fitted_n = 0
            return self
        pred = predictions.astype(float).mean(axis=1).to_numpy()
        obs = np.asarray(y, dtype=float)[: len(pred)]
        mask = np.isfinite(pred) & np.isfinite(obs)
        residuals = np.abs(obs[mask] - pred[mask])
        self.residual_quantile = _inflated_quantile(residuals, self.alpha)
        self.fitted_n = int(residuals.size)
        return self

    def predict_interval(self, predictions: pd.DataFrame | pd.Series) -> pd.DataFrame:
        if isinstance(predictions, pd.Series):
            center = predictions.astype(float)
        else:
            center = predictions.astype(float).mean(axis=1)
        return pd.DataFrame(
            {
                "center": center,
                "lower": center - self.residual_quantile,
                "upper": center + self.residual_quantile,
                "alpha": self.alpha,
            },
            index=center.index,
        )


@dataclass
class StronglyAdaptiveACI:
    """Strongly adaptive online conformal controller using expert gammas."""

    alpha_target: float = 0.10
    gammas: tuple[float, ...] = (0.001, 0.005, 0.01, 0.05)
    alpha_min: float = 0.001
    alpha_max: float = 0.5
    expert_weights: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    expert_alphas: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)

    def __post_init__(self) -> None:
        if self.expert_weights.size == 0:
            self.expert_weights = np.ones(len(self.gammas), dtype=float) / len(self.gammas)
        if self.expert_alphas.size == 0:
            self.expert_alphas = np.ones(len(self.gammas), dtype=float) * self.alpha_target

    def update(self, covered: bool) -> float:
        err = 0.0 if covered else 1.0
        losses = np.abs(self.expert_alphas - err)
        self.expert_weights *= np.exp(-0.5 * losses)
        self.expert_weights /= max(float(self.expert_weights.sum()), 1e-12)
        for i, gamma in enumerate(self.gammas):
            self.expert_alphas[i] = float(
                np.clip(
                    self.expert_alphas[i] + gamma * (self.alpha_target - err),
                    self.alpha_min,
                    self.alpha_max,
                )
            )
        return float(np.dot(self.expert_weights, self.expert_alphas))

    def run(self, covered: Iterable[bool]) -> pd.DataFrame:
        rows = []
        for t, flag in enumerate(covered):
            alpha_t = self.update(bool(flag))
            rows.append({"t": t, "covered": bool(flag), "alpha_t": alpha_t})
        return pd.DataFrame(rows)


@dataclass
class AgACI:
    """AgACI-style aggregation over several ACI controllers."""

    alpha_target: float = 0.10
    gammas: tuple[float, ...] = (0.001, 0.005, 0.01, 0.05)

    def run(self, covered: Iterable[bool]) -> pd.DataFrame:
        controller = StronglyAdaptiveACI(alpha_target=self.alpha_target, gammas=self.gammas)
        return controller.run(covered)


__all__ = ["AgACI", "EnbPIInterval", "StronglyAdaptiveACI"]
