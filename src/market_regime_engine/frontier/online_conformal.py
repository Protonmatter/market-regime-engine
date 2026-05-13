# SPDX-License-Identifier: Apache-2.0

"""Online conformal calibration primitives for nonstationary time series.

v1.6.0 honest-naming refactor (REVIEW_DEEP_V1_5_2.md §1.9 / Findings #5,
#23 — both classes had misleading names that overclaim the cited
algorithm fidelity):

- ``EnbPIInterval`` (v1.5.x) → :class:`EnsembleMeanSplitConformal`. The
  shipped implementation is plain split conformal on the ensemble mean;
  Xu-Xie 2021 EnbPI uses leave-one-out residuals from a bootstrap
  ensemble (asymptotic conditional coverage under serial dependence).
  The v1.5.x name is preserved as an alias.
- ``StronglyAdaptiveACI`` (v1.5.x) → :class:`MultiplicativeWeightsACI`.
  The shipped implementation uses a quadratic-loss surrogate, not the
  log-loss multiplicative-weights expert update of Bhatt-Foster-Bobu-
  Russell 2023 *Strongly adaptive online learning*. Functional but not
  faithful; alias preserved.
- ``AgACI`` (v1.5.x) — wraps :class:`MultiplicativeWeightsACI` with no
  cross-stream aggregation, so it is not faithful Zaffran et al. 2022
  AgACI either. Removed as a separate class; the alias points at the
  underlying multiplicative-weights controller for v1.5.x compat.

Faithful reimplementations of all three primitives are tracked as
v1.7.0 TODOs.
"""

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
class EnsembleMeanSplitConformal:
    """Plain split conformal on the ensemble mean.

    NOT faithful Xu-Xie 2021 EnbPI: that estimator uses leave-one-out
    residuals from a bootstrap ensemble and provides asymptotic
    conditional coverage under serial dependence. This class instead
    averages the ensemble columns into a single point prediction and
    runs split conformal on the resulting residuals — marginal coverage
    only, no serial-dependence guarantee.

    Renamed in v1.6.0 from ``EnbPIInterval`` per
    REVIEW_DEEP_V1_5_2.md §1.9 / Finding #5. The v1.5.x name is
    preserved as a backwards-compat alias.

    TODO(v1.7.0): implement true EnbPI per Xu-Xie 2021 (ICML).
    """

    alpha: float = 0.10
    residual_quantile: float = 0.0
    fitted_n: int = 0

    def fit(self, predictions: pd.DataFrame, y: Sequence[float]) -> EnsembleMeanSplitConformal:
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
class MultiplicativeWeightsACI:
    """Multiplicative-weights ACI controller with quadratic-loss surrogate.

    NOT faithful Bhatt-Foster-Bobu-Russell 2023 *Strongly adaptive
    online learning*: that estimator uses log-loss for the expert
    update, not the absolute-error / quadratic surrogate used here. The
    shipped implementation will converge slower (and to slightly
    different alphas) than canonical SAOL.

    Renamed in v1.6.0 from ``StronglyAdaptiveACI`` per
    REVIEW_DEEP_V1_5_2.md §1.9 / Finding #23. The v1.5.x name is
    preserved as a backwards-compat alias.

    TODO(v1.7.0): switch the expert update to log-loss to match SAOL.
    """

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


# v1.5.x backwards-compat aliases. Tagged for removal in v1.7.0
# alongside faithful reimplementations.
EnbPIInterval = EnsembleMeanSplitConformal
StronglyAdaptiveACI = MultiplicativeWeightsACI
# AgACI is NOT a separate class in v1.6.0 — it was a no-op wrapper that
# delegated everything to StronglyAdaptiveACI without aggregating across
# multiple controllers (which is what canonical AgACI does, per Zaffran
# et al. 2022). Aliased to the underlying controller so existing imports
# continue to work; flagged for v1.7.0 reimplementation.
AgACI = MultiplicativeWeightsACI


__all__ = [
    "AgACI",
    "EnbPIInterval",
    "EnsembleMeanSplitConformal",
    "MultiplicativeWeightsACI",
    "StronglyAdaptiveACI",
]
