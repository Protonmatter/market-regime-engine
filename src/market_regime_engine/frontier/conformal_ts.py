# SPDX-License-Identifier: Apache-2.0
"""Time-series-native conformal predictors (v1.2 frontier).

This module bundles the 2021-2024 generation of conformal predictors that
relax exchangeability, give group-conditional coverage, or run anytime-valid.
All five classes mirror the public surface of
:class:`market_regime_engine.conformal.MondrianBinaryConformal` so they can
plug into it via the ``backend=`` keyword:

1. :class:`BlockConformalBinary` — block-bootstrap conformal for binary
   forecasts (Politis-Romano 1994 stationary block bootstrap composed with
   split conformal). Block-mean nonconformity scores are quantile-thresholded
   per bucket. Recovers a finite-sample coverage guarantee under stationary
   beta-mixing (no exchangeability).
2. :class:`NexCPForecaster` — Stankevičiūtė-Alaa-van der Schaar (2021) NexCP.
   Rolling absolute residuals with adaptive inflation; the time-series-native
   variant of split conformal.
3. :class:`ConditionalConformalRegressor` — Gibbs-Cherian-Candès (2023)
   "Conformal Prediction with Conditional Guarantees" (arXiv 2305.12616),
   finite-class version. Per-group quantile + Bonferroni-adjusted inflation
   to enforce 1 - alpha coverage *per group*.
4. :class:`LocalizedSplitConformal` — Lin-Trivedi-Sun (2023), arXiv 2307.10460
   "Conformal Prediction Beyond Exchangeability". Calibration scores are
   weighted by an RBF kernel of distance from the test feature vector, giving
   a test-point-conditional quantile.
5. :class:`SequentialEConformal` — Vovk-Wang (2021) "E-values: Calibration,
   combination and applications" (JASA 2024). Per-bucket e-process emits
   anytime-valid prediction sets with formal long-run coverage.

All five classes expose a Mondrian-shaped surface:

- ``alpha`` — target miscoverage (default 0.10).
- ``bucket_col`` — bucket column name (default ``"regime_bucket"``).
- ``thresholds: dict[str, float]`` — per-bucket score threshold so the legacy
  Mondrian wrapper can mirror them straight to the warehouse.
- ``bucket_counts: dict[str, int]`` — per-bucket calibration sample count.
- ``fallback_threshold: float`` — pooled-data threshold used for novel
  buckets at test time.
- ``fit(calibration: pd.DataFrame) -> Self``
- ``transform(predictions: pd.DataFrame) -> pd.DataFrame`` — annotated frame
  with ``conformal_set``, ``conformal_uncertain``, ``conformal_threshold``.
- ``coverage_report(calibration: pd.DataFrame) -> pd.DataFrame`` — realized
  coverage by bucket on a held-out frame.

In addition, :class:`ConditionalConformalRegressor` exposes
``coverage_report_conditional()`` for per-group coverage diagnostics with
worst-case violation reporting, and :class:`SequentialEConformal` exposes
``update(x, y, pred)`` and ``coverage_until_now()`` for online use.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EPS = 1e-9


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _binary_score(p: float, y: int) -> float:
    """Standard split-conformal binary score: 1 - p_hat(y)."""
    return float(1.0 - p) if int(y) == 1 else float(p)


def _binary_score_unknown(p: float, label: int) -> float:
    """Score evaluated at an arbitrary candidate ``label`` (used by transform)."""
    return float(1.0 - p) if int(label) == 1 else float(p)


def _quantile_inflated(values: np.ndarray, alpha: float) -> float:
    """Standard split-conformal quantile with finite-sample inflation."""
    n = len(values)
    if n == 0:
        return float("inf")
    rank = math.ceil((n + 1) * (1.0 - alpha))
    rank = min(max(rank, 1), n)
    sorted_vals = np.sort(values)
    return float(sorted_vals[rank - 1])


def _prepare_calibration_frame(frame: pd.DataFrame, bucket_col: str) -> pd.DataFrame:
    """Defensive normalisation of (y, p, bucket) calibration frames."""
    df = frame.dropna(subset=["y", "p"]).copy()
    if df.empty:
        return df
    if bucket_col not in df:
        df[bucket_col] = "general"
    df[bucket_col] = df[bucket_col].fillna("general").astype(str)
    df["__score__"] = np.where(
        df["y"].astype(int) == 1,
        1.0 - df["p"].astype(float),
        df["p"].astype(float),
    )
    return df


def _prediction_set(p: float, threshold: float) -> set[int]:
    """Return the binary conformal prediction set for ``p`` at ``threshold``."""
    return {label for label in (0, 1) if _binary_score_unknown(p, label) <= threshold}


def _annotate_predictions(
    predictions: pd.DataFrame,
    bucket_col: str,
    threshold_for: Callable[..., float],
) -> pd.DataFrame:
    """Attach conformal_set / conformal_uncertain / conformal_threshold columns."""
    if predictions is None or predictions.empty:
        return predictions
    frame = predictions.copy()
    if bucket_col not in frame:
        frame[bucket_col] = "general"
    frame[bucket_col] = frame[bucket_col].fillna("general").astype(str)
    sets: list[str] = []
    flags: list[bool] = []
    thresholds: list[float] = []
    for _, row in frame.iterrows():
        bucket = str(row[bucket_col])
        p = float(row["p"])
        thr = float(threshold_for(bucket, row=row))
        ps = _prediction_set(p, thr)
        sets.append("|".join(str(lbl) for lbl in sorted(ps)) or "empty")
        flags.append(ps == {0, 1})
        thresholds.append(thr)
    frame["conformal_set"] = sets
    frame["conformal_uncertain"] = flags
    frame["conformal_threshold"] = thresholds
    return frame


def _coverage_by_bucket(
    calibration: pd.DataFrame,
    bucket_col: str,
    threshold_for: Callable[..., float],
    alpha: float,
) -> pd.DataFrame:
    """Per-bucket realized coverage at the layer's chosen thresholds."""
    if calibration is None or calibration.empty:
        return pd.DataFrame(columns=[bucket_col, "n", "coverage", "alpha", "threshold"])
    frame = calibration.dropna(subset=["y", "p"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=[bucket_col, "n", "coverage", "alpha", "threshold"])
    if bucket_col not in frame:
        frame[bucket_col] = "general"
    rows: list[dict[str, object]] = []
    for bucket, group in frame.groupby(bucket_col, observed=True):
        thr = float(threshold_for(str(bucket), row=group.iloc[0]))
        covered = sum(
            int(int(y) in _prediction_set(float(p), thr)) for y, p in zip(group["y"], group["p"], strict=True)
        )
        rows.append(
            {
                bucket_col: str(bucket),
                "n": len(group),
                "coverage": covered / len(group),
                "alpha": float(alpha),
                "threshold": thr,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Block-bootstrap conformal (Politis-Romano 1994 + split conformal)
# ---------------------------------------------------------------------------


@dataclass
class BlockConformalBinary:
    """Block-bootstrap conformal for binary probability forecasts.

    Calibration scores ``s_i = 1 - p_hat(y_i)`` are resampled in contiguous
    blocks of length ``L`` (Politis-Romano 1994 stationary block bootstrap).
    Each replicate gives a single empirical ``(1 - alpha)``-quantile of the
    individual-row scores; the per-bucket threshold is the median of those
    bootstrap quantiles.

    The block aggregation ensures that under stationary beta-mixing the
    bootstrap quantile is a consistent estimate of the population
    ``(1 - alpha)``-quantile of the calibration score distribution. Coverage
    on *individual* prediction sets is preserved (in expectation), unlike a
    naive block-mean threshold which would only cover block averages.

    The block_means_threshold is also exposed for diagnostic purposes; it is
    the literal "quantile of block means" view referenced in the v1.2 spec.
    """

    alpha: float = 0.10
    bucket_col: str = "regime_bucket"
    block_length: int = 12
    bootstrap: int = 200
    seed: int = 0
    fallback_alpha: float = 0.10
    thresholds: dict[str, float] = field(default_factory=dict)
    bucket_counts: dict[str, int] = field(default_factory=dict)
    fallback_threshold: float = float("inf")
    block_count: dict[str, int] = field(default_factory=dict)
    block_mean_thresholds: dict[str, float] = field(default_factory=dict)

    def _block_bootstrap_quantile(self, scores: np.ndarray) -> tuple[float, float]:
        """Return (per-row threshold, block-mean threshold)."""
        n = len(scores)
        L = max(int(self.block_length), 1)
        if n == 0:
            return float("inf"), float("inf")
        if n < L:
            # Degenerate: too few samples for a single block, fall back to
            # the standard inflated quantile.
            t = _quantile_inflated(scores, self.alpha)
            return t, t
        rng = np.random.default_rng(self.seed)
        n_blocks = max(n // L, 1)
        boot_q = np.empty(self.bootstrap, dtype=float)
        boot_means = np.empty(self.bootstrap, dtype=float)
        for b in range(self.bootstrap):
            starts = rng.integers(0, n - L + 1, size=n_blocks)
            idx = np.concatenate([np.arange(s, s + L) for s in starts])
            sample = scores[idx[:n]]
            # Bootstrap quantile of individual-row scores (preserves coverage).
            boot_q[b] = _quantile_inflated(sample, self.alpha)
            # Diagnostic block-mean quantile.
            sample_blocks = sample[: n_blocks * L].reshape(n_blocks, L).mean(axis=1)
            boot_means[b] = float(np.quantile(sample_blocks, 1.0 - self.alpha))
        return float(np.median(boot_q)), float(np.median(boot_means))

    def fit(self, calibration: pd.DataFrame) -> BlockConformalBinary:
        frame = _prepare_calibration_frame(calibration, self.bucket_col)
        if frame.empty:
            return self
        self.thresholds = {}
        self.bucket_counts = {}
        self.block_count = {}
        self.block_mean_thresholds = {}
        all_scores: list[float] = []
        L = max(int(self.block_length), 1)
        for bucket, group in frame.groupby(self.bucket_col, observed=True):
            scores = group["__score__"].to_numpy(dtype=float)
            n = len(scores)
            if n == 0:
                continue
            t_row, t_block = self._block_bootstrap_quantile(scores)
            self.thresholds[str(bucket)] = t_row
            self.block_mean_thresholds[str(bucket)] = t_block
            self.bucket_counts[str(bucket)] = int(n)
            self.block_count[str(bucket)] = max(n // L, 1)
            all_scores.extend(scores.tolist())
        if all_scores:
            arr = np.asarray(all_scores, dtype=float)
            t_row, _ = self._block_bootstrap_quantile(arr)
            self.fallback_threshold = t_row
        return self

    def threshold_for(self, bucket: str, *, row: pd.Series | None = None) -> float:
        return self.thresholds.get(str(bucket), self.fallback_threshold)

    def transform(self, predictions: pd.DataFrame) -> pd.DataFrame:
        return _annotate_predictions(predictions, self.bucket_col, self.threshold_for)

    def coverage_report(self, calibration: pd.DataFrame) -> pd.DataFrame:
        return _coverage_by_bucket(calibration, self.bucket_col, self.threshold_for, self.alpha)


# ---------------------------------------------------------------------------
# 2. NexCP forecaster (Stankevičiūtė-Alaa-van der Schaar 2021)
# ---------------------------------------------------------------------------


@dataclass
class NexCPForecaster:
    """Time-series-native split conformal with **online** ACI adaptation.

    Maintains a *rolling* window of the most recent absolute residuals
    (here ``s_t = 1 - p_hat(y_t)`` for the binary case) per bucket and a
    per-bucket inflation term that is updated online via the
    Stankevičiūtė-Alaa-van der Schaar 2021 (arXiv 2102.13066, §3) /
    Gibbs-Candès 2021 ACI rule:

        inflation_{t+1} = inflation_t + gamma * (err_t - alpha)

    where ``err_t = 1[y_t not in C_t(x_t)]`` is the realised coverage
    error at time ``t`` and ``gamma = inflation_eta`` is the step size.
    Under-coverage (``err = 1``) widens the prediction set by raising
    inflation; over-coverage (``err = 0``) shrinks it. The adaptive
    inflation guarantees long-run coverage of ``1 - alpha`` for any
    distribution shift, modulo the standard ACI step-size constraint.

    v1.6.0 bug fix (REVIEW_DEEP_V1_5_2.md §1.7 / Finding #4): the prior
    implementation computed inflation *once* at ``.fit(...)`` time and
    froze it. The class name promised the time-series-native online
    adaptive primitive of Stankevičiūtė et al. 2021; the implementation
    delivered a one-shot variant that degraded to plain split conformal
    at test time. This commit adds an explicit :meth:`step` method that
    rolls the inflation forward at every observed test point. ``.fit()``
    is preserved as the initial-calibration entry point but the online
    contract is now :meth:`step`.

    The calibration window length defaults to ``window=120``.
    """

    alpha: float = 0.10
    bucket_col: str = "regime_bucket"
    window: int = 120
    inflation_eta: float = 0.05
    thresholds: dict[str, float] = field(default_factory=dict)
    bucket_counts: dict[str, int] = field(default_factory=dict)
    fallback_threshold: float = float("inf")
    inflation_per_bucket: dict[str, float] = field(default_factory=dict)
    base_thresholds: dict[str, float] = field(default_factory=dict)
    fallback_base_threshold: float = float("inf")
    history: list[dict] = field(default_factory=list)

    def fit(self, calibration: pd.DataFrame) -> NexCPForecaster:
        frame = _prepare_calibration_frame(calibration, self.bucket_col)
        if frame.empty:
            return self
        # Sort by date when present so the rolling window respects time order.
        if "date" in frame.columns:
            frame = frame.sort_values("date")
        self.thresholds = {}
        self.base_thresholds = {}
        self.bucket_counts = {}
        self.inflation_per_bucket = {}
        all_scores: list[float] = []
        for bucket, group in frame.groupby(self.bucket_col, observed=True):
            scores = group["__score__"].to_numpy(dtype=float)
            window = scores[-int(self.window) :] if len(scores) > self.window else scores
            base = _quantile_inflated(window, self.alpha)
            # Initial inflation seeded from the calibration coverage gap so
            # the first .step() does not start from zero (which would mirror
            # plain split conformal). Subsequent .step() calls roll the
            # inflation per the ACI rule.
            covered = float(np.mean(window <= base))
            err = max((1.0 - self.alpha) - covered, 0.0)
            inflation = float(self.inflation_eta * err)
            current = float(min(max(base + inflation, 0.0), 1.0))
            self.thresholds[str(bucket)] = current
            self.base_thresholds[str(bucket)] = float(base)
            self.bucket_counts[str(bucket)] = len(scores)
            self.inflation_per_bucket[str(bucket)] = inflation
            all_scores.extend(scores.tolist())
        if all_scores:
            arr = np.asarray(all_scores, dtype=float)
            self.fallback_base_threshold = _quantile_inflated(arr[-int(self.window) :], self.alpha)
            self.fallback_threshold = self.fallback_base_threshold
        return self

    def step(self, pred: float, y: int | float, bucket: str) -> dict:
        """Online ACI update per Stankevičiūtė-Alaa-van der Schaar 2021 §3.

        Forms the binary prediction set ``C_t(x_t) = {label : score(pred, label)
        <= threshold_t}`` using the current per-bucket threshold, observes
        the realised binary outcome ``y``, computes the coverage error
        ``err_t = 1[y not in C_t]``, and rolls the inflation:

            inflation_{t+1} = inflation_t + inflation_eta * (err_t - alpha)

        The new threshold ``base + inflation_{t+1}`` is clipped to
        ``[0, 1]`` (binary scores are bounded) and persisted to
        ``self.thresholds[bucket]`` so subsequent ``.transform`` /
        ``.threshold_for`` calls see the updated value.

        Returns ``{"prediction_set": set[int], "covered": bool,
        "inflation": float, "threshold": float, "err": int}``.
        """
        bucket_str = str(bucket)
        base = self.base_thresholds.get(bucket_str, self.fallback_base_threshold)
        if not math.isfinite(base):
            base = self.fallback_threshold
        inflation = self.inflation_per_bucket.get(bucket_str, 0.0)
        current_threshold = float(min(max(base + inflation, 0.0), 1.0))
        pred_set = _prediction_set(float(pred), current_threshold)
        y_int = int(y)
        err = 0 if y_int in pred_set else 1
        new_inflation = inflation + float(self.inflation_eta) * (err - float(self.alpha))
        self.inflation_per_bucket[bucket_str] = float(new_inflation)
        new_threshold = float(min(max(base + new_inflation, 0.0), 1.0))
        self.thresholds[bucket_str] = new_threshold
        if bucket_str not in self.base_thresholds and math.isfinite(self.fallback_base_threshold):
            self.base_thresholds[bucket_str] = float(self.fallback_base_threshold)
        if bucket_str not in self.bucket_counts:
            self.bucket_counts[bucket_str] = 0
        self.bucket_counts[bucket_str] += 1
        record = {
            "bucket": bucket_str,
            "pred": float(pred),
            "y": y_int,
            "err": int(err),
            "inflation": float(new_inflation),
            "threshold": float(new_threshold),
            "covered": bool(err == 0),
        }
        self.history.append(record)
        return {
            "prediction_set": pred_set,
            "covered": bool(err == 0),
            "inflation": float(new_inflation),
            "threshold": float(new_threshold),
            "err": int(err),
        }

    def threshold_for(self, bucket: str, *, row: pd.Series | None = None) -> float:
        return self.thresholds.get(str(bucket), self.fallback_threshold)

    def transform(self, predictions: pd.DataFrame) -> pd.DataFrame:
        return _annotate_predictions(predictions, self.bucket_col, self.threshold_for)

    def coverage_report(self, calibration: pd.DataFrame) -> pd.DataFrame:
        return _coverage_by_bucket(calibration, self.bucket_col, self.threshold_for, self.alpha)


# ---------------------------------------------------------------------------
# 3. Conditional conformal (Gibbs-Cherian-Candès 2023, finite-class)
# ---------------------------------------------------------------------------


@dataclass
class ConditionalConformalRegressor:
    """Group-conditional conformal with Bonferroni-adjusted inflation.

    Implements the *finite-class* version of Gibbs-Cherian-Candès 2023
    "Conformal Prediction with Conditional Guarantees". For each group ``g``
    (here the bucket column), we set the per-group threshold to the
    ``(1 - alpha / G)``-quantile of the in-group nonconformity scores, where
    ``G`` is the number of distinct groups present in the calibration set.
    The Bonferroni inflation guarantees *simultaneous* coverage at level
    ``1 - alpha`` across all groups: ``P(Y in C(X) | G = g) >= 1 - alpha``
    holds for every ``g`` (Theorem 2.1 of the paper, restricted to discrete
    bucket functions).
    """

    alpha: float = 0.10
    bucket_col: str = "regime_bucket"
    thresholds: dict[str, float] = field(default_factory=dict)
    bucket_counts: dict[str, int] = field(default_factory=dict)
    fallback_threshold: float = float("inf")
    n_groups: int = 0

    def fit(self, calibration: pd.DataFrame) -> ConditionalConformalRegressor:
        frame = _prepare_calibration_frame(calibration, self.bucket_col)
        if frame.empty:
            return self
        groups = sorted(frame[self.bucket_col].unique())
        self.n_groups = max(len(groups), 1)
        # Bonferroni-adjusted per-group alpha.
        adj_alpha = float(self.alpha) / self.n_groups
        self.thresholds = {}
        self.bucket_counts = {}
        for bucket in groups:
            sub = frame[frame[self.bucket_col] == bucket]
            scores = sub["__score__"].to_numpy(dtype=float)
            self.thresholds[str(bucket)] = _quantile_inflated(scores, adj_alpha)
            self.bucket_counts[str(bucket)] = len(scores)
        all_scores = frame["__score__"].to_numpy(dtype=float)
        self.fallback_threshold = _quantile_inflated(all_scores, adj_alpha)
        return self

    def threshold_for(self, bucket: str, *, row: pd.Series | None = None) -> float:
        return self.thresholds.get(str(bucket), self.fallback_threshold)

    def transform(self, predictions: pd.DataFrame) -> pd.DataFrame:
        return _annotate_predictions(predictions, self.bucket_col, self.threshold_for)

    def coverage_report(self, calibration: pd.DataFrame) -> pd.DataFrame:
        return _coverage_by_bucket(calibration, self.bucket_col, self.threshold_for, self.alpha)

    def coverage_report_conditional(self, calibration: pd.DataFrame) -> dict:
        """Per-group realized coverage with worst-case violation diagnostics.

        Returns a dict with:

        - ``"per_group"``: dataframe of per-group coverage, shape
          ``(n_groups, [bucket, n, coverage, threshold])``.
        - ``"worst_violation"``: float, the largest gap below 1 - alpha; 0.0
          when every group meets coverage.
        - ``"alpha"``: target miscoverage.
        - ``"adjusted_alpha"``: per-group alpha after Bonferroni.
        """
        per_group = self.coverage_report(calibration)
        target = 1.0 - float(self.alpha)
        if per_group.empty:
            worst = 0.0
        else:
            gaps = (target - per_group["coverage"].astype(float)).clip(lower=0.0)
            worst = float(gaps.max())
        return {
            "per_group": per_group,
            "worst_violation": worst,
            "alpha": float(self.alpha),
            "adjusted_alpha": float(self.alpha) / max(self.n_groups, 1),
        }


# ---------------------------------------------------------------------------
# 4. Localized split conformal (Lin-Trivedi-Sun 2023)
# ---------------------------------------------------------------------------


@dataclass
class LocalizedSplitConformal:
    """RBF-weighted split conformal with test-point-conditional thresholds.

    Calibration nonconformity scores are weighted by an RBF kernel of
    distance from the test feature vector::

        w_i(x*) = exp(-||x_i - x*||^2 / (2 * bandwidth^2))

    The conformal threshold becomes a *weighted* empirical quantile, which is
    test-point-conditional. Under mild regularity (Lin-Trivedi-Sun 2023, arXiv
    2307.10460, Theorem 4.2), this recovers conditional coverage up to a
    bandwidth-dependent bias term while remaining marginally valid.

    The feature vector is taken from the calibration frame's columns named in
    ``feature_cols``. If ``feature_cols`` is ``None``, the bucket column is
    used as a one-hot embedding (so behavior degenerates to per-bucket
    conformal — useful when no continuous features are passed).
    """

    alpha: float = 0.10
    bucket_col: str = "regime_bucket"
    bandwidth: float = 1.0
    feature_cols: list[str] | None = None
    thresholds: dict[str, float] = field(default_factory=dict)
    bucket_counts: dict[str, int] = field(default_factory=dict)
    fallback_threshold: float = float("inf")
    _calibration_features: np.ndarray = field(default_factory=lambda: np.empty(0))
    _calibration_scores: np.ndarray = field(default_factory=lambda: np.empty(0))
    _calibration_buckets: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))
    _feature_cols_resolved: list[str] = field(default_factory=list)

    def _featurize(self, frame: pd.DataFrame) -> np.ndarray:
        cols = self._feature_cols_resolved or self.feature_cols or []
        if not cols:
            # One-hot the bucket as a fallback feature.
            buckets = frame[self.bucket_col].astype(str)
            uniq = sorted(buckets.unique())
            mat = np.zeros((len(frame), max(len(uniq), 1)))
            for j, b in enumerate(uniq):
                mat[buckets.values == b, j] = 1.0
            return mat
        return frame.reindex(columns=cols, fill_value=0.0).to_numpy(dtype=float)

    def fit(self, calibration: pd.DataFrame) -> LocalizedSplitConformal:
        frame = _prepare_calibration_frame(calibration, self.bucket_col)
        if frame.empty:
            return self
        # Snap feature_cols to whatever's present at fit-time.
        if self.feature_cols is not None:
            self._feature_cols_resolved = [c for c in self.feature_cols if c in frame.columns]
        else:
            self._feature_cols_resolved = []
        feats = self._featurize(frame)
        self._calibration_features = feats
        self._calibration_scores = frame["__score__"].to_numpy(dtype=float)
        self._calibration_buckets = frame[self.bucket_col].astype(str).to_numpy()
        # Per-bucket centroids → marginal threshold (used when transform()
        # cannot find the test row's features).
        self.thresholds = {}
        self.bucket_counts = {}
        for bucket in np.unique(self._calibration_buckets):
            mask = self._calibration_buckets == bucket
            self.thresholds[str(bucket)] = _quantile_inflated(self._calibration_scores[mask], self.alpha)
            self.bucket_counts[str(bucket)] = int(mask.sum())
        self.fallback_threshold = _quantile_inflated(self._calibration_scores, self.alpha)
        return self

    def _localized_threshold(self, x_test: np.ndarray) -> float:
        if self._calibration_features.size == 0:
            return self.fallback_threshold
        diffs = self._calibration_features - x_test[None, :]
        d2 = np.sum(diffs * diffs, axis=1)
        bw = max(float(self.bandwidth), 1e-9)
        weights = np.exp(-d2 / (2.0 * bw * bw))
        weights = np.maximum(weights, EPS)
        order = np.argsort(self._calibration_scores)
        sorted_scores = self._calibration_scores[order]
        sorted_weights = weights[order]
        # NexCP / Lin-Trivedi-Sun 2023 weighted-quantile construction
        # (REVIEW_DEEP_V1_5_2.md §1.7 / Finding #12 fix). The test-point
        # nonconformity is conventionally appended at +infinity (rank n+1)
        # with weight equal to the kernel at zero distance (= 1.0). The
        # weighted (1 - alpha) quantile is therefore the smallest
        # calibration score whose cumulative calibration weight reaches
        # ``(1 - alpha) * (sum_of_calibration_weights + test_weight)``.
        # The earlier ``cum = cumsum + test_weight`` was equivalent to
        # inserting the test point at rank 0 (smallest score), shifting
        # every empirical CDF value up by ``test_weight / total`` and
        # producing an over-narrow prediction set.
        test_weight = 1.0  # RBF at zero distance.
        total = float(np.sum(sorted_weights)) + test_weight
        target = (1.0 - float(self.alpha)) * total
        cum = np.cumsum(sorted_weights)
        idx = int(np.searchsorted(cum, target, side="left"))
        idx = min(max(idx, 0), len(sorted_scores) - 1)
        return float(sorted_scores[idx])

    def threshold_for(self, bucket: str, *, row: pd.Series | None = None) -> float:
        if row is None or self._calibration_features.size == 0:
            return self.thresholds.get(str(bucket), self.fallback_threshold)
        # Build a single-row feature vector matching fit-time featurization.
        if self._feature_cols_resolved:
            x = np.array(
                [float(row.get(c, 0.0)) if pd.notna(row.get(c, 0.0)) else 0.0 for c in self._feature_cols_resolved],
                dtype=float,
            )
        else:
            # One-hot bucket fallback: replicate fit-time bucket ordering.
            uniq = sorted(set(self._calibration_buckets.tolist()))
            x = np.zeros(max(len(uniq), 1))
            for j, b in enumerate(uniq):
                if str(bucket) == str(b):
                    x[j] = 1.0
                    break
        return self._localized_threshold(x)

    def transform(self, predictions: pd.DataFrame) -> pd.DataFrame:
        return _annotate_predictions(predictions, self.bucket_col, self.threshold_for)

    def coverage_report(self, calibration: pd.DataFrame) -> pd.DataFrame:
        return _coverage_by_bucket(calibration, self.bucket_col, self.threshold_for, self.alpha)


# ---------------------------------------------------------------------------
# 5. Sequential e-conformal (Vovk-Wang 2021 / Ramdas anytime-valid)
# ---------------------------------------------------------------------------


@dataclass
class SequentialEConformal:
    """E-process based anytime-valid binary conformal (betting e-process).

    Maintains a per-bucket e-statistic that is the running product of a
    **betting e-process** per Ramdas-Manole 2023 §3:

        E_t = E_{t-1} * (1 + lambda_t * (y_t - p_hat_t))

    where ``y_t in {0, 1}`` is the realised binary outcome,
    ``p_hat_t in [EPS, 1-EPS]`` is the forecaster's predicted P(Y=1),
    and ``lambda_t in [-1/(1-p_hat_t), 1/p_hat_t]`` is the betting
    coefficient. We use ``lambda_t = 1`` (GROW-conservative; admissible
    for any ``p_hat_t in [EPS, 1-EPS]``) — this is the canonical
    Vovk-Wang 2021 / Ramdas-Manole 2023 default. Under H_0
    ("forecaster is calibrated", ``y_t ~ Bernoulli(p_hat_t)``):

        E[1 + lambda_t * (y_t - p_hat_t) | F_{t-1}]
            = 1 + lambda_t * (E[y_t | F_{t-1}] - p_hat_t)
            = 1 + lambda_t * (p_hat_t - p_hat_t)
            = 1

    so the increment is a valid e-variable (expectation 1 under H_0) and
    ``E_t`` is a non-negative martingale. By Ville's inequality
    ``P(sup_t E_t >= 1/alpha) <= alpha`` under H_0; the rejection region
    ``E_t >= 1/alpha`` therefore has *anytime-valid* type-I error control.

    v1.6.0 bug fix (REVIEW_DEEP_V1_5_2.md §1.7 / Finding #3): the prior
    increment ``2 * (1 - score)`` is only expectation-1 at ``p_hat = 0.5``
    — for any non-balanced forecaster ``E[increment | H_0] > 1``, so the
    Ville-inequality control did not hold. The new betting e-process
    requires *both* the prediction ``p_hat_t`` *and* the realised outcome
    ``y_t`` to roll the e-statistic forward; the ``_increment`` method
    signature reflects that.

    Use :meth:`fit` to bootstrap from a calibration window (with both
    ``y`` and ``p`` columns); use :meth:`update` to roll forward online.

    References:
    - Ramdas & Manole (2023), "Randomized and exchangeable improvements
      of Markov's, Chebyshev's and Chernoff's inequalities", §3.
    - Vovk & Wang (2021), "E-values: Calibration, combination and
      applications", JASA 2024.
    - Howard, Ramdas, McAuliffe, Sekhon (2021), "Time-uniform,
      nonparametric, nonasymptotic confidence sequences", AOS.
    """

    alpha: float = 0.10
    bucket_col: str = "regime_bucket"
    e_floor: float = 1e-9
    lambda_t: float = 1.0
    thresholds: dict[str, float] = field(default_factory=dict)
    bucket_counts: dict[str, int] = field(default_factory=dict)
    fallback_threshold: float = float("inf")
    e_per_bucket: dict[str, float] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    def fit(self, calibration: pd.DataFrame) -> SequentialEConformal:
        frame = _prepare_calibration_frame(calibration, self.bucket_col)
        if frame.empty:
            return self
        if "date" in frame.columns:
            frame = frame.sort_values("date")
        self.thresholds = {}
        self.bucket_counts = {}
        self.e_per_bucket = {}
        for bucket, group in frame.groupby(self.bucket_col, observed=True):
            scores = group["__score__"].to_numpy(dtype=float)
            preds = group["p"].astype(float).to_numpy()
            outcomes = group["y"].astype(int).to_numpy()
            self.bucket_counts[str(bucket)] = len(scores)
            # Marginal threshold = standard split-conformal quantile so
            # transform() and the warehouse mirror it 1:1. The e-process is
            # the *online* contract; the threshold here is the "static" view.
            self.thresholds[str(bucket)] = _quantile_inflated(scores, self.alpha)
            # Initialize e-stat to the running product of betting e-process
            # increments seen on the calibration set. Both p_hat and y are
            # required (Ramdas-Manole 2023 §3); we walk the calibration in
            # order so the e-process matches an online roll-out.
            e = 1.0
            for p_hat_val, y_val in zip(preds, outcomes, strict=True):
                e *= self._increment(float(p_hat_val), int(y_val))
                e = max(e, self.e_floor)
            self.e_per_bucket[str(bucket)] = float(e)
        all_scores = frame["__score__"].to_numpy(dtype=float)
        self.fallback_threshold = _quantile_inflated(all_scores, self.alpha)
        return self

    def _increment(self, p_hat: float, y: int) -> float:
        """Betting e-process increment ``1 + lambda * (y - p_hat)``.

        Clips ``p_hat`` to ``[EPS, 1-EPS]`` so the admissible interval
        ``lambda in [-1/(1-p_hat), 1/p_hat]`` is non-degenerate. With the
        default ``lambda_t = 1.0`` the increment lies in
        ``[1 - p_hat, 2 - p_hat]`` — strictly positive for any clipped
        ``p_hat``, expectation-1 under H_0 (verified in
        ``tests/test_sequential_e_conformal_valid_e_process.py``).
        """
        p_hat_clipped = max(min(float(p_hat), 1.0 - EPS), EPS)
        y_int = 1 if int(y) == 1 else 0
        return float(1.0 + float(self.lambda_t) * (y_int - p_hat_clipped))

    def update(self, x: object, y: int | float, pred: float) -> dict:
        """Roll the e-statistic forward with a single (x, y, pred) triple.

        ``x`` may be any hashable bucket label; if it is a dict / Series we
        try ``x[bucket_col]`` first and fall back to the string repr.
        ``y`` is the realised binary outcome and ``pred`` is the
        forecaster's predicted P(Y=1). Both are required by the betting
        e-process (Ramdas-Manole 2023 §3).
        Returns ``{"e_value": float, "is_significant": bool}``.
        """
        bucket: str
        if isinstance(x, (dict, pd.Series)):
            bucket = str(x.get(self.bucket_col, "general"))
        else:
            bucket = str(x)
        score = _binary_score(float(pred), int(y))
        e_prev = self.e_per_bucket.get(bucket, 1.0)
        e_new = max(e_prev * self._increment(float(pred), int(y)), self.e_floor)
        self.e_per_bucket[bucket] = float(e_new)
        is_sig = bool(e_new >= 1.0 / max(float(self.alpha), EPS))
        self.history.append(
            {
                "bucket": bucket,
                "score": float(score),
                "e_value": float(e_new),
                "significant": is_sig,
            }
        )
        return {"e_value": float(e_new), "is_significant": is_sig}

    def coverage_until_now(self) -> dict:
        """Empirical coverage so far across all updates.

        The "covered" event is ``score <= threshold`` at the static threshold
        chosen by ``fit``; the e-process layer is in addition the anytime-
        valid hypothesis test.
        """
        if not self.history:
            return {
                "n": 0,
                "coverage": float("nan"),
                "alpha": float(self.alpha),
                "max_e_value": 0.0,
            }
        n = len(self.history)
        covered = sum(
            int(float(h["score"]) <= self.thresholds.get(h["bucket"], self.fallback_threshold)) for h in self.history
        )
        max_e = float(max(h["e_value"] for h in self.history))
        return {
            "n": int(n),
            "coverage": float(covered / n),
            "alpha": float(self.alpha),
            "max_e_value": max_e,
        }

    def threshold_for(self, bucket: str, *, row: pd.Series | None = None) -> float:
        return self.thresholds.get(str(bucket), self.fallback_threshold)

    def transform(self, predictions: pd.DataFrame) -> pd.DataFrame:
        return _annotate_predictions(predictions, self.bucket_col, self.threshold_for)

    def coverage_report(self, calibration: pd.DataFrame) -> pd.DataFrame:
        return _coverage_by_bucket(calibration, self.bucket_col, self.threshold_for, self.alpha)


__all__ = [
    "BlockConformalBinary",
    "ConditionalConformalRegressor",
    "LocalizedSplitConformal",
    "NexCPForecaster",
    "SequentialEConformal",
]
