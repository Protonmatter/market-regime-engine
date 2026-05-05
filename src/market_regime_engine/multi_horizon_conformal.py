# SPDX-License-Identifier: Apache-2.0
"""Multi-horizon coherent conformal prediction.

Standard conformal layers (split, CQR, Mondrian) calibrate one horizon at a
time and yield independently-valid intervals at each horizon. For a multi-
horizon trajectory, that double-counts coverage error: at any point at least
one horizon's interval may miss with probability ``H * alpha``.

Stankevičiūtė, Alaa & van der Schaar (NeurIPS 2021) introduce *conformal
time-series prediction* with a Bonferroni adjustment: calibrating each
horizon at miscoverage ``alpha / H`` yields a joint coverage guarantee of at
least ``1 - alpha`` over the full trajectory.

This module implements two estimators:

- :class:`BonferroniMultiHorizonConformal` — the Stankevičiūtė et al. construction
  applied to CQR-style intervals.
- :class:`AdaptiveMultiHorizonConformal` — the same construction wrapped in
  the Gibbs-Candès online update so the per-horizon ``alpha_h`` adapt over
  time. Useful for production where the realized coverage drifts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from market_regime_engine.conformal import ConformalizedQuantileRegression


@dataclass
class BonferroniMultiHorizonConformal:
    """Per-horizon CQR with a Bonferroni-adjusted alpha.

    Parameters
    ----------
    horizons:
        Sequence of integer or string horizon labels matching the keys in the
        per-horizon prediction frames.
    alpha:
        Joint miscoverage budget. Each horizon is calibrated at
        ``alpha / len(horizons)``.
    """

    horizons: tuple[str, ...] = field(default_factory=lambda: ("3m", "6m", "12m"))
    alpha: float = 0.10
    cqrs: dict[str, ConformalizedQuantileRegression] = field(default_factory=dict)

    @property
    def per_horizon_alpha(self) -> float:
        return float(self.alpha) / max(len(self.horizons), 1)

    def fit(self, calibration: dict[str, pd.DataFrame]) -> BonferroniMultiHorizonConformal:
        for h in self.horizons:
            df = calibration.get(h, pd.DataFrame())
            cqr = ConformalizedQuantileRegression(alpha=self.per_horizon_alpha).fit(df)
            self.cqrs[h] = cqr
        return self

    def transform(self, predictions: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for h in self.horizons:
            cqr = self.cqrs.get(h)
            df = predictions.get(h, pd.DataFrame())
            if cqr is None or df.empty:
                out[h] = df
                continue
            out[h] = cqr.transform(df)
        return out

    def joint_coverage(self, calibration: dict[str, pd.DataFrame]) -> dict:
        """Realized joint coverage over all horizons on a held-out frame.

        Caller is responsible for aligning the frames by date so that joint
        coverage means "the simultaneous interval covered y at every horizon
        on the same date".

        The implementation aligns every horizon's per-row outputs by date
        using ``pd.concat([... add_suffix(f"_{h}")])`` along the ``date``
        index. The previous implementation used a chained ``DataFrame.join``
        with ``rsuffix=f"_{h}"`` which collided with the renamed
        ``q_lo_{h}``/``q_hi_{h}`` columns and produced ambiguous lookups for
        any horizon past the first.
        """
        if not calibration:
            return {"joint_coverage": float("nan"), "n": 0}

        per_horizon: list[pd.DataFrame] = []
        present_horizons: list[str] = []
        for h in self.horizons:
            df = calibration.get(h)
            if df is None or df.empty:
                continue
            cqr = self.cqrs.get(h)
            if cqr is None:
                continue
            adj = cqr.transform(df).copy()
            if "date" not in adj.columns:
                # Use the existing index if "date" is not a column.
                adj = adj.reset_index().rename(columns={adj.index.name or "index": "date"})
            # Keep only the columns we need then add a per-horizon suffix.
            keep = adj[["date", "y", "q_lo_conformal", "q_hi_conformal"]].copy()
            keep = keep.set_index("date")
            keep = keep.rename(
                columns={
                    "y": "y",
                    "q_lo_conformal": "q_lo",
                    "q_hi_conformal": "q_hi",
                }
            )
            keep = keep.add_suffix(f"_{h}")
            per_horizon.append(keep)
            present_horizons.append(h)

        if not per_horizon:
            return {"joint_coverage": float("nan"), "n": 0}

        # Inner-join every horizon on the date index. concat(axis=1) keeps the
        # union of the indexes; .dropna() drops any date that is missing in
        # any horizon, so the returned coverage is computed only over dates
        # where all horizons agree.
        merged = pd.concat(per_horizon, axis=1, join="inner")
        if merged.empty:
            return {"joint_coverage": float("nan"), "n": 0}

        joint = pd.Series(True, index=merged.index)
        for h in present_horizons:
            ylo = merged[f"q_lo_{h}"]
            yhi = merged[f"q_hi_{h}"]
            yh = merged[f"y_{h}"]
            joint &= (yh >= ylo) & (yh <= yhi)
        return {
            "joint_coverage": float(joint.mean()),
            "n": len(joint),
            "alpha": self.alpha,
            "horizons_used": list(present_horizons),
        }


@dataclass
class AdaptiveMultiHorizonConformal:
    """Adaptive analogue of the Bonferroni multi-horizon construction.

    Each horizon's ``alpha_h`` is updated online via Gibbs-Candès::

        alpha_h_{t+1} = alpha_h_t + gamma * (alpha_target_h - I[y_h not covered])

    where ``alpha_target_h = alpha / H``. Only the running per-horizon
    inflations are persisted; the implementation reuses
    :class:`ConformalizedQuantileRegression` per horizon as the calibrator.
    """

    horizons: tuple[str, ...] = field(default_factory=lambda: ("3m", "6m", "12m"))
    alpha: float = 0.10
    gamma: float = 0.01
    alpha_min: float = 1e-3
    alpha_max: float = 0.5
    state: dict[str, float] = field(default_factory=dict)

    def initialize(self) -> None:
        per = self.alpha / max(len(self.horizons), 1)
        self.state = {h: float(per) for h in self.horizons}

    def step(self, history: dict[str, pd.DataFrame], realized: dict[str, tuple[float, float, float]]) -> dict:
        """Update per-horizon alphas using realized miscoverage.

        ``realized`` maps horizon → (q_lo, q_hi, y) for the most recent
        realized observation. ``history`` maps horizon → calibration frame
        used to refit CQR each step.
        """
        if not self.state:
            self.initialize()
        new_inflations: dict[str, float] = {}
        for h in self.horizons:
            cqr = ConformalizedQuantileRegression(alpha=self.state[h]).fit(history.get(h, pd.DataFrame()))
            q_lo, q_hi, y = realized.get(h, (float("nan"),) * 3)
            covered = (
                np.isfinite(q_lo)
                and np.isfinite(q_hi)
                and np.isfinite(y)
                and (q_lo - cqr.inflation) <= y <= (q_hi + cqr.inflation)
            )
            err = 0.0 if covered else 1.0
            new_alpha = float(
                np.clip(
                    self.state[h] + self.gamma * ((self.alpha / max(len(self.horizons), 1)) - err),
                    self.alpha_min,
                    self.alpha_max,
                )
            )
            self.state[h] = new_alpha
            new_inflations[h] = float(cqr.inflation)
        return {"alpha": dict(self.state), "inflation": new_inflations}


__all__ = ["AdaptiveMultiHorizonConformal", "BonferroniMultiHorizonConformal"]
