# SPDX-License-Identifier: Apache-2.0
"""Online Bayesian model averaging for binary forecasts.

The simplex-grid stacking in :mod:`market_regime_engine.stacking` is fine for
small ``K`` but does not scale and does not adapt online. :class:`OnlineBMA`
maintains a posterior over models using an *exponential forgetting factor*::

    log w_{i, t+1} = (1 - lambda) * log w_{i, t} + log f_i(y_t | history_t)

where ``f_i`` is the predictive likelihood of model ``i`` at time ``t``. The
forgetting factor ``lambda`` controls how fast the weights adapt; small
values (~0.95) keep memory long, large values (~0.5) react quickly to
regime change.

The class is target-agnostic: it exposes ``update(y_t, p_t_per_model)`` and
``mix(p_t_per_model)``. Coupled with the conformal layer in
:mod:`conformal`, this gives a calibrated, regime-aware probabilistic
forecast.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EPS = 1e-9


def _log_score(y: float, p: float) -> float:
    p = float(np.clip(p, EPS, 1.0 - EPS))
    return float(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


@dataclass
class OnlineBMA:
    """Bates-Granger / exponentially-discounted log-score weights."""

    forgetting: float = 0.96
    # v1.2 fix #11: ``floor_weight`` is applied *after* normalization, not
    # before. Previously the floor was applied to the raw exp(arr) values and
    # then the result was renormalized, which silently inflated the smallest
    # model's weight by a factor proportional to ``1 / (sum(w) + K * floor)``.
    # The new default 1e-9 matches the EPS used by the log-score so the
    # post-normalization floor is a numerical safety net rather than a
    # smoothing prior.
    floor_weight: float = 1e-9
    log_weights: dict[str, float] = field(default_factory=dict)

    def initialize(self, models: list[str]) -> None:
        if not models:
            return
        u = np.log(np.full(len(models), 1.0 / len(models)))
        self.log_weights = {m: float(u[i]) for i, m in enumerate(models)}

    def update(self, y: float, predictions: dict[str, float]) -> dict[str, float]:
        if not self.log_weights:
            self.initialize(list(predictions.keys()))
        for name, p in predictions.items():
            prev = self.log_weights.get(name, 0.0)
            self.log_weights[name] = float(self.forgetting) * prev + _log_score(y, float(p))
        keys = list(self.log_weights.keys())
        arr = np.array([self.log_weights[k] for k in keys], dtype=float)
        arr -= arr.max()
        w = np.exp(arr)
        w = w / w.sum()
        # Floor AFTER normalization so the prior probability mass is allocated
        # by the log-score and the floor is purely a numerical safety net.
        w = np.maximum(w, self.floor_weight)
        w = w / w.sum()
        out = {k: float(v) for k, v in zip(keys, w, strict=True)}
        for k, v in out.items():
            self.log_weights[k] = float(np.log(max(v, EPS)))
        return out

    def mix(self, predictions: dict[str, float]) -> float:
        if not self.log_weights:
            return float(np.mean(list(predictions.values()))) if predictions else float("nan")
        arr = np.array([self.log_weights[k] for k in self.log_weights], dtype=float)
        arr -= arr.max()
        w = np.exp(arr)
        w = w / w.sum()
        keys = list(self.log_weights.keys())
        return float(sum(float(predictions.get(k, 0.0)) * float(w[i]) for i, k in enumerate(keys)))


def online_bma_from_oos(
    oos: pd.DataFrame,
    *,
    target: str,
    horizon: str,
    forgetting: float = 0.96,
) -> tuple[pd.DataFrame, OnlineBMA]:
    """Run :class:`OnlineBMA` over an OOS prediction frame.

    The frame must contain ``date, model_name, y, p`` and is widened on
    ``model_name`` so each row carries a vector of model predictions. Returns
    the per-step weight history and the fitted ``OnlineBMA`` instance.
    """
    if oos is None or oos.empty:
        return pd.DataFrame(), OnlineBMA(forgetting=forgetting)
    sub = oos[(oos["target"] == target) & (oos["horizon"] == horizon)].copy()
    if sub.empty:
        return pd.DataFrame(), OnlineBMA(forgetting=forgetting)
    pivot = sub.pivot_table(index="date", columns="model_name", values="p", aggfunc="last").sort_index()
    realized = sub.drop_duplicates("date").set_index("date")["y"].astype(float).reindex(pivot.index)
    bma = OnlineBMA(forgetting=forgetting)
    bma.initialize(list(pivot.columns))
    rows = []
    for date, row in pivot.iterrows():
        y = float(realized.loc[date]) if pd.notna(realized.loc[date]) else None
        if y is None:
            continue
        weights = bma.update(y, {k: float(v) for k, v in row.items() if pd.notna(v)})
        mix = bma.mix({k: float(v) for k, v in row.items() if pd.notna(v)})
        out = {"date": date, "target": target, "horizon": horizon, "mixed_p": mix}
        out.update({f"w_{k}": v for k, v in weights.items()})
        rows.append(out)
    return pd.DataFrame(rows), bma


__all__ = ["OnlineBMA", "online_bma_from_oos"]
