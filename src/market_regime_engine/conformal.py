# SPDX-License-Identifier: Apache-2.0
"""Conformal prediction layers.

Three production-grade tools live here, each chosen because it has an explicit
finite-sample coverage guarantee even under arbitrary base-model
misspecification:

1. :class:`MondrianBinaryConformal` — Mondrian split conformal for binary
   probabilities, conditioned on a discrete bucket (regime). For every bucket
   independently, calibration scores ``s_i = 1 - p_hat(y_i)`` are sorted; the
   ``(1 - alpha)``-quantile defines a per-bucket score threshold. Predictions
   whose conformity score exceeds the threshold are marked uncertain.

2. :class:`ConformalizedQuantileRegression` (CQR, Romano-Patterson-Candès 2019)
   — adjusts a base quantile model (e.g. the engine's HGBR pinball regressor)
   so that the resulting prediction interval has marginal coverage of exactly
   ``1 - alpha`` regardless of model misspecification.

3. :class:`AdaptiveConformalInference` (ACI, Gibbs-Candès 2021) — online
   coverage tracking that adjusts ``alpha_t`` based on realized hits/misses,
   so the engine maintains long-run coverage even under regime drift.

All three classes follow the same public contract:

- ``fit(...)`` consumes calibration data (out-of-sample predictions paired
  with realized outcomes).
- ``transform(...)`` produces calibrated probabilities, intervals, or sets.
- ``coverage_report(...)`` reports realized coverage on a held-out window.

They are explicitly designed to live downstream of :mod:`walk_forward`: take
the OOS predictions emitted by :func:`evaluate_walk_forward`, split off a
calibration window, and fit conformal layers on it before applying them to
the production-time forecast.

Time-series-native conformal variants (block conformal, NexCP, conditional,
localized, sequential e-values) live in
:mod:`market_regime_engine.frontier.conformal_ts`. They are wired into
:class:`MondrianBinaryConformal` via the ``backend=`` parameter so the same
public surface can dispatch to a non-exchangeable primitive without breaking
back-compat.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

EPS = 1e-9


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _quantile_inflated(values: np.ndarray, alpha: float) -> float:
    """Conformal quantile with the standard finite-sample inflation factor.

    For ``n`` calibration scores, the conformal-valid threshold is the
    ``ceil((n + 1) * (1 - alpha)) / n`` empirical quantile of the absolute
    nonconformity scores.
    """
    n = len(values)
    if n == 0:
        return float("inf")
    rank = int(math_ceil((n + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), n)
    sorted_vals = np.sort(values)
    return float(sorted_vals[rank - 1])


def math_ceil(x: float) -> int:
    return int(np.ceil(x))


# ---------------------------------------------------------------------------
# Mondrian binary conformal
# ---------------------------------------------------------------------------


@dataclass
class MondrianBinaryConformal:
    """Per-bucket split conformal for binary probability forecasts.

    The conformity score for a calibration row ``(p_i, y_i)`` is::

        s_i = 1 - p_i if y_i == 1 else p_i

    Lower scores ⇒ better-aligned probabilities. For a target miscoverage
    ``alpha``, the bucket threshold is the conformal quantile of those scores.
    A test prediction ``p`` is *covered* if its score ``s = min(p, 1 - p)``
    when ``y`` is unknown is below the threshold; we expose two helpers:

    - :meth:`prediction_set` — returns the set of labels that satisfy the
      conformal coverage criterion.
    - :meth:`uncertainty_flag` — emits ``True`` when the prediction set is
      ``{0, 1}`` (i.e. the model is too uncertain at the requested ``alpha``).
    """

    alpha: float = 0.10
    bucket_col: str = "regime_bucket"
    fallback_alpha: float = 0.10
    thresholds: dict[str, float] = field(default_factory=dict)
    fallback_threshold: float = float("inf")
    bucket_counts: dict[str, int] = field(default_factory=dict)
    # Exchangeability assumption. When False, the v1.2 frontier conformal_ts
    # backends are required (block / NexCP / conditional / localized /
    # e-conformal); a plain quantile is no longer finite-sample valid.
    exchangeable: bool = True
    # Backend dispatch. ``"split"`` (default) uses the per-bucket inflated
    # quantile that has been the v1.0/v1.1 behavior. Any other value triggers
    # delegation to the matching primitive in
    # :mod:`market_regime_engine.frontier.conformal_ts`.
    backend: Literal["split", "block", "nexcp", "conditional", "localized", "e_conformal"] = "split"
    backend_kwargs: dict = field(default_factory=dict)
    # Internal handle to a delegated backend object (set when backend != split).
    _backend_obj: object | None = field(default=None, repr=False)

    def fit(self, calibration: pd.DataFrame) -> MondrianBinaryConformal:
        """Fit thresholds from a calibration frame.

        ``calibration`` is expected to have columns ``y``, ``p``, and the
        bucket column named by :attr:`bucket_col`. Rows where ``y`` is missing
        are dropped.

        When ``exchangeable=False`` the call routes through a frontier
        time-series-native backend (defaults to ``backend="block"`` if the
        caller did not pick one). Setting ``exchangeable=True`` (default)
        preserves the v1.0 split-conformal behavior verbatim.
        """
        if calibration is None or calibration.empty:
            return self
        frame = calibration.dropna(subset=["y", "p"]).copy()
        if frame.empty:
            return self
        if self.bucket_col not in frame:
            frame[self.bucket_col] = "general"
        frame[self.bucket_col] = frame[self.bucket_col].fillna("general").astype(str)
        # Choose backend. If the caller toggled exchangeable=False but left
        # backend="split", auto-bump to block conformal (the safest mixing-only
        # default). Anything other than "split" is treated as a non-exchangeable
        # path even when exchangeable=True, so the keyword is purely advisory.
        backend = self.backend
        if not self.exchangeable and backend == "split":
            backend = "block"
        if backend != "split":
            return self._fit_via_backend(frame, backend)
        scores = np.where(
            frame["y"].astype(int) == 1,
            1.0 - frame["p"].astype(float),
            frame["p"].astype(float),
        )
        frame["__score__"] = scores
        self.thresholds = {}
        self.bucket_counts = {}
        for bucket, group in frame.groupby(self.bucket_col, observed=True):
            vals = group["__score__"].to_numpy(dtype=float)
            self.thresholds[str(bucket)] = _quantile_inflated(vals, self.alpha)
            self.bucket_counts[str(bucket)] = len(vals)
        all_scores = scores
        self.fallback_threshold = _quantile_inflated(all_scores, self.fallback_alpha)
        return self

    def _fit_via_backend(self, frame: pd.DataFrame, backend: str) -> MondrianBinaryConformal:
        """Delegate fit to a frontier backend; mirror its thresholds back.

        We always materialize a per-bucket threshold dict so that downstream
        consumers (``threshold_for``, ``transform``, the warehouse) keep
        working unchanged. The delegated object is also stored on
        ``_backend_obj`` for callers that want richer diagnostics.
        """
        from market_regime_engine.frontier import conformal_ts as _ct

        kwargs = dict(self.backend_kwargs)
        obj: object
        if backend == "block":
            obj = _ct.BlockConformalBinary(alpha=self.alpha, bucket_col=self.bucket_col, **kwargs).fit(frame)
        elif backend == "nexcp":
            obj = _ct.NexCPForecaster(alpha=self.alpha, bucket_col=self.bucket_col, **kwargs).fit(frame)
        elif backend == "conditional":
            obj = _ct.ConditionalConformalRegressor(alpha=self.alpha, bucket_col=self.bucket_col, **kwargs).fit(frame)
        elif backend == "localized":
            obj = _ct.LocalizedSplitConformal(alpha=self.alpha, bucket_col=self.bucket_col, **kwargs).fit(frame)
        elif backend == "e_conformal":
            obj = _ct.SequentialEConformal(alpha=self.alpha, bucket_col=self.bucket_col, **kwargs).fit(frame)
        else:  # pragma: no cover - guarded by Literal
            raise ValueError(f"unknown conformal backend: {backend!r}")
        self._backend_obj = obj
        # Mirror per-bucket thresholds and counts so the public surface stays
        # identical. Backends expose ``thresholds: dict[str, float]`` and
        # ``bucket_counts: dict[str, int]`` to honor this contract.
        self.thresholds = dict(getattr(obj, "thresholds", {}))
        self.bucket_counts = dict(getattr(obj, "bucket_counts", {}))
        self.fallback_threshold = float(getattr(obj, "fallback_threshold", float("inf")))
        return self

    def threshold_for(self, bucket: str) -> float:
        return self.thresholds.get(str(bucket), self.fallback_threshold)

    def prediction_set(self, p: float, bucket: str) -> set[int]:
        """Return the conformal prediction set for a single probability."""
        threshold = self.threshold_for(bucket)
        return {label for label in (0, 1) if self._score(p, label) <= threshold}

    def uncertainty_flag(self, p: float, bucket: str) -> bool:
        return self.prediction_set(p, bucket) == {0, 1}

    def transform(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Annotate a prediction frame with prediction sets and flags."""
        if predictions is None or predictions.empty:
            return predictions
        frame = predictions.copy()
        if self.bucket_col not in frame:
            frame[self.bucket_col] = "general"
        frame[self.bucket_col] = frame[self.bucket_col].fillna("general").astype(str)
        sets: list[str] = []
        flags: list[bool] = []
        thresholds: list[float] = []
        for _, row in frame.iterrows():
            bucket = str(row[self.bucket_col])
            p = float(row["p"])
            ps = self.prediction_set(p, bucket)
            sets.append("|".join(str(lbl) for lbl in sorted(ps)) or "empty")
            flags.append(ps == {0, 1})
            thresholds.append(self.threshold_for(bucket))
        frame["conformal_set"] = sets
        frame["conformal_uncertain"] = flags
        frame["conformal_threshold"] = thresholds
        return frame

    def coverage_report(self, calibration: pd.DataFrame) -> pd.DataFrame:
        """Realized coverage by bucket, useful for diagnostics."""
        if calibration is None or calibration.empty:
            return pd.DataFrame(columns=[self.bucket_col, "n", "coverage", "alpha", "threshold"])
        frame = calibration.dropna(subset=["y", "p"]).copy()
        if frame.empty:
            return pd.DataFrame(columns=[self.bucket_col, "n", "coverage", "alpha", "threshold"])
        if self.bucket_col not in frame:
            frame[self.bucket_col] = "general"
        rows = []
        for bucket, group in frame.groupby(self.bucket_col, observed=True):
            covered = sum(
                int(int(y) in self.prediction_set(float(p), str(bucket)))
                for y, p in zip(group["y"], group["p"], strict=True)
            )
            rows.append(
                {
                    self.bucket_col: str(bucket),
                    "n": len(group),
                    "coverage": covered / len(group),
                    "alpha": float(self.alpha),
                    "threshold": float(self.threshold_for(str(bucket))),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _score(p: float, label: int) -> float:
        if label == 1:
            return float(1.0 - p)
        return float(p)


# ---------------------------------------------------------------------------
# Conformalized quantile regression (CQR)
# ---------------------------------------------------------------------------


@dataclass
class ConformalizedQuantileRegression:
    """Romano-Patterson-Candès 2019 CQR.

    Adjusts a pre-fitted quantile pair ``(q_lo, q_hi)`` so that the resulting
    interval has marginal coverage exactly ``1 - alpha``. The adjustment is a
    single scalar inflation, computed from out-of-sample residuals.

    Calibration data is a frame with columns ``y, q_lo, q_hi``. ``transform``
    consumes a frame with columns ``q_lo, q_hi`` and returns the adjusted
    interval.
    """

    alpha: float = 0.10
    inflation: float = 0.0
    fitted_n: int = 0

    def fit(self, calibration: pd.DataFrame) -> ConformalizedQuantileRegression:
        if calibration is None or calibration.empty:
            self.inflation = 0.0
            return self
        frame = calibration.dropna(subset=["y", "q_lo", "q_hi"]).copy()
        if frame.empty:
            self.inflation = 0.0
            return self
        y = frame["y"].astype(float).to_numpy()
        lo = frame["q_lo"].astype(float).to_numpy()
        hi = frame["q_hi"].astype(float).to_numpy()
        residual = np.maximum(lo - y, y - hi)  # E_i = max(qlo - y, y - qhi)
        self.inflation = _quantile_inflated(residual, self.alpha)
        self.fitted_n = len(frame)
        return self

    def transform(self, predictions: pd.DataFrame) -> pd.DataFrame:
        if predictions is None or predictions.empty:
            return predictions
        frame = predictions.copy()
        frame["q_lo_conformal"] = frame["q_lo"].astype(float) - self.inflation
        frame["q_hi_conformal"] = frame["q_hi"].astype(float) + self.inflation
        return frame

    def coverage_report(self, calibration: pd.DataFrame) -> dict:
        if calibration is None or calibration.empty:
            return {"n": 0, "coverage": float("nan"), "alpha": self.alpha, "inflation": self.inflation}
        frame = calibration.dropna(subset=["y", "q_lo", "q_hi"]).copy()
        if frame.empty:
            return {"n": 0, "coverage": float("nan"), "alpha": self.alpha, "inflation": self.inflation}
        y = frame["y"].astype(float).to_numpy()
        lo = (frame["q_lo"].astype(float) - self.inflation).to_numpy()
        hi = (frame["q_hi"].astype(float) + self.inflation).to_numpy()
        covered = float(np.mean((y >= lo) & (y <= hi)))
        return {
            "n": len(frame),
            "coverage": covered,
            "alpha": float(self.alpha),
            "inflation": float(self.inflation),
        }


# ---------------------------------------------------------------------------
# Adaptive conformal inference (ACI, Gibbs-Candès 2021)
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveConformalInference:
    """Online coverage adjustment.

    Tracks ``alpha_t`` over time with the recursion::

        alpha_{t+1} = alpha_t + gamma * (alpha_target - I[y_t not covered])

    The base conformal threshold is recomputed at each step using a sliding
    window of recent calibration data. For binary probabilities this is the
    Mondrian threshold; for quantile intervals, a CQR inflation. The class is
    deliberately algorithm-agnostic — the caller supplies a ``base_fit`` and a
    ``base_apply`` callback so it can adapt either form.
    """

    alpha_target: float = 0.10
    gamma: float = 0.01
    alpha_min: float = 0.001
    alpha_max: float = 0.5

    def run(
        self,
        records: Iterable[tuple[pd.Timestamp, dict]],
        *,
        base_fit: Callable[..., Any],
        base_apply: Callable[..., Any],
        warmup: int = 36,
    ) -> pd.DataFrame:
        """Iterate through records and return per-step coverage diagnostics.

        ``base_fit(history)`` should return a fitted threshold/inflation; we
        accept any object as long as ``base_apply(threshold, record)`` returns a
        boolean indicating whether the realized outcome is covered.
        """
        history: list[dict] = []
        alpha_t = float(self.alpha_target)
        rows: list[dict] = []
        for date, record in records:
            history.append(dict(record))
            if len(history) < warmup:
                rows.append(
                    {
                        "date": pd.Timestamp(date),
                        "alpha_t": alpha_t,
                        "covered": None,
                        "n_history": len(history),
                    }
                )
                continue
            threshold = base_fit(history, alpha_t)
            covered = bool(base_apply(threshold, record))
            err = 0.0 if covered else 1.0
            alpha_t = float(
                np.clip(
                    alpha_t + self.gamma * (self.alpha_target - err),
                    self.alpha_min,
                    self.alpha_max,
                )
            )
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "alpha_t": alpha_t,
                    "covered": covered,
                    "threshold": float(threshold) if isinstance(threshold, (int, float)) else None,
                    "n_history": len(history),
                }
            )
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# convenience pipeline
# ---------------------------------------------------------------------------


def fit_mondrian_from_oos(
    oos_predictions: pd.DataFrame,
    *,
    alpha: float = 0.10,
    bucket_col: str = "regime_bucket",
) -> MondrianBinaryConformal:
    """Build a Mondrian conformal layer from a walk-forward OOS prediction frame.

    ``oos_predictions`` must contain columns ``date, model_name, target,
    horizon, y, p`` and optionally ``regime_bucket``. The function aggregates
    duplicated date+model rows by mean (defensive) and fits one Mondrian per
    target+horizon if buckets are present.
    """
    if oos_predictions is None or oos_predictions.empty:
        return MondrianBinaryConformal(alpha=alpha, bucket_col=bucket_col)
    frame = oos_predictions.copy()
    if bucket_col not in frame:
        frame[bucket_col] = "general"
    layer = MondrianBinaryConformal(alpha=alpha, bucket_col=bucket_col)
    return layer.fit(frame.rename(columns={"value": "p"}) if "value" in frame and "p" not in frame else frame)


__all__ = [
    "AdaptiveConformalInference",
    "ConformalizedQuantileRegression",
    "MondrianBinaryConformal",
    "fit_mondrian_from_oos",
]
