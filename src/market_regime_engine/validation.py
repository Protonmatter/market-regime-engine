# SPDX-License-Identifier: Apache-2.0
"""Validation primitives.

v1.5 PR-5 (deep research §"Validation & metrics", REVIEW.md §3.4 Q-5)
extends the historical Brier / log-loss / pinball / coverage toolkit with
three primitives the FI execution-confidence release gate needs:

- :func:`deflated_sharpe` — Bailey–López de Prado (2014) DSR adjusts the
  observed Sharpe for multiple-trial selection bias and non-normality of
  returns.
- :func:`probability_of_backtest_overfitting` — Bailey et al. (2017) PBO
  via combinatorial-purged-cross-validation; the fraction of in-sample
  top-ranked strategies that rank below median out-of-sample is the
  signed strength of overfit.
- :func:`minimum_track_record_length` — companion to DSR; minimum
  observations needed to claim ``sharpe_observed > sharpe_target`` at the
  requested confidence under the same non-normality correction.

The release gate consumes the DSR + PBO columns (when present in the
``confidence`` frame) per the ``production`` profile defaults.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 1e-9


def _clip_prob(p: np.ndarray | pd.Series) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def brier_score(y_true: Iterable[float], p_pred: Iterable[float]) -> float:
    """Mean squared probability error for binary outcomes."""
    y = np.asarray(list(y_true), dtype=float)
    p = _clip_prob(np.asarray(list(p_pred), dtype=float))
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean((y[mask] - p[mask]) ** 2))


def log_loss_score(y_true: Iterable[float], p_pred: Iterable[float]) -> float:
    """Binary log loss, clipped for numerical stability."""
    y = np.asarray(list(y_true), dtype=float)
    p = _clip_prob(np.asarray(list(p_pred), dtype=float))
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() == 0:
        return float("nan")
    return float(-np.mean(y[mask] * np.log(p[mask]) + (1.0 - y[mask]) * np.log(1.0 - p[mask])))


def calibration_table(y_true: Iterable[float], p_pred: Iterable[float], bins: int = 10) -> pd.DataFrame:
    """Reliability table: predicted probability bucket vs realized frequency."""
    y = np.asarray(list(y_true), dtype=float)
    p = _clip_prob(np.asarray(list(p_pred), dtype=float))
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() == 0:
        return pd.DataFrame(columns=["bin", "count", "pred_mean", "actual_rate", "calibration_error"])
    frame = pd.DataFrame({"y": y[mask], "p": p[mask]})
    frame["bin"] = pd.cut(frame["p"], np.linspace(0, 1, bins + 1), include_lowest=True, duplicates="drop")
    out = (
        frame.groupby("bin", observed=True)
        .agg(count=("y", "size"), pred_mean=("p", "mean"), actual_rate=("y", "mean"))
        .reset_index()
    )
    out["calibration_error"] = out["actual_rate"] - out["pred_mean"]
    out["bin"] = out["bin"].astype(str)
    return out


def expected_calibration_error(y_true: Iterable[float], p_pred: Iterable[float], bins: int = 10) -> float:
    table = calibration_table(y_true, p_pred, bins=bins)
    if table.empty or table["count"].sum() == 0:
        return float("nan")
    weights = table["count"] / table["count"].sum()
    return float(np.sum(weights * table["calibration_error"].abs()))


def pinball_loss(y_true: Iterable[float], q_pred: Iterable[float], tau: float) -> float:
    """Quantile/pinball loss for return quantile forecasts."""
    y = np.asarray(list(y_true), dtype=float)
    q = np.asarray(list(q_pred), dtype=float)
    mask = np.isfinite(y) & np.isfinite(q)
    if mask.sum() == 0:
        return float("nan")
    e = y[mask] - q[mask]
    return float(np.mean(np.maximum(tau * e, (tau - 1.0) * e)))


def quantile_coverage(y_true: Iterable[float], q_pred: Iterable[float]) -> float:
    """Observed share of outcomes below the predicted quantile."""
    y = np.asarray(list(y_true), dtype=float)
    q = np.asarray(list(q_pred), dtype=float)
    mask = np.isfinite(y) & np.isfinite(q)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(y[mask] <= q[mask]))


@dataclass(frozen=True)
class BinaryValidationResult:
    target: str
    horizon: str
    observations: int
    event_rate: float
    brier: float
    log_loss: float
    ece: float


def validate_binary_forecast(target: str, horizon: str, y_true: pd.Series, p_pred: pd.Series) -> BinaryValidationResult:
    aligned = pd.concat([y_true.rename("y"), p_pred.rename("p")], axis=1).dropna()
    if aligned.empty:
        return BinaryValidationResult(target, horizon, 0, float("nan"), float("nan"), float("nan"), float("nan"))
    return BinaryValidationResult(
        target=target,
        horizon=horizon,
        observations=len(aligned),
        event_rate=float(aligned["y"].mean()),
        brier=brier_score(aligned["y"], aligned["p"]),
        log_loss=log_loss_score(aligned["y"], aligned["p"]),
        ece=expected_calibration_error(aligned["y"], aligned["p"]),
    )


def validation_frame(results: list[BinaryValidationResult]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in results])


# ---------------------------------------------------------------------------
# v1.5 PR-5: Bailey–López de Prado validation primitives (Q-5, deep research)
# ---------------------------------------------------------------------------


def _normal_cdf(z: float) -> float:
    """Numerically stable standard-normal CDF; ``math.erf`` only."""
    return 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))


def _normal_ppf(p: float) -> float:
    """Beasley–Springer–Moro inverse-CDF approximation for the standard
    normal. We avoid a hard dependency on scipy here — ``validation.py``
    must work without the ``[frontier]`` extra.

    Reference: Moro (1995) accuracy is ~7 decimal places on ``p ∈ (1e-7, 1 - 1e-7)``
    which is far better than the DSR confidence pipeline needs.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0,1); got {p!r}")
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


_EULER_MASCHERONI = 0.5772156649015328606
_GAMMA_AT_INFINITY = 1.0  # placeholder for the closed-form below


def _expected_max_z(n_trials: int) -> float:
    """E[max of N i.i.d. standard normals] using the Bailey–López de Prado
    closed-form approximation (eq. 2 in BLP 2014).

    For ``n_trials == 1`` returns 0 (the observed estimator is the only
    candidate so multiple-testing inflation is zero). The BLP closed form
    is:

        E[max] ≈ (1 − γ) · Φ⁻¹(1 − 1/N) + γ · Φ⁻¹(1 − 1/(N·e))

    where ``γ`` is the Euler–Mascheroni constant.
    """
    n = int(n_trials)
    if n <= 1:
        return 0.0
    term1 = _normal_ppf(1.0 - 1.0 / n)
    term2 = _normal_ppf(1.0 - 1.0 / (n * math.e))
    return (1.0 - _EULER_MASCHERONI) * term1 + _EULER_MASCHERONI * term2


def _sample_skew_kurt(returns: np.ndarray) -> tuple[float, float]:
    """Sample skew and excess kurtosis without scipy."""
    if returns.size < 4:
        return 0.0, 0.0
    mean = returns.mean()
    std = returns.std(ddof=0)
    if std <= 0:
        return 0.0, 0.0
    centered = returns - mean
    skew = float(np.mean((centered / std) ** 3))
    kurt = float(np.mean((centered / std) ** 4) - 3.0)
    return skew, kurt


def deflated_sharpe(
    strategy_returns: pd.Series | Iterable[float],
    *,
    n_trials: int,
    skew: float | None = None,
    kurt: float | None = None,
    sharpe_target: float = 0.0,
) -> float:
    """Bailey–López de Prado (2014) Deflated Sharpe Ratio.

    Returns the probability that the *true* Sharpe exceeds
    ``sharpe_target`` given the observed Sharpe, after deflating for:

    1. Multiple-testing selection bias (``n_trials`` candidate strategies),
    2. Non-normality of returns via sample ``skew`` + excess ``kurt``.

    Inputs:

    - ``strategy_returns``: per-period (e.g. daily) return series.
    - ``n_trials``: number of candidate strategies the operator attempted
      before reporting this one (the multiple-testing correction).
    - ``skew`` / ``kurt``: pre-computed from elsewhere (e.g. an external
      analytics pipeline). When ``None``, the sample skew / excess
      kurtosis are computed from ``strategy_returns``.
    - ``sharpe_target``: the null Sharpe; DSR is the posterior probability
      that ``sharpe_true > sharpe_target``. Defaults to 0.

    Returns DSR in ``[0, 1]``. Higher is stronger evidence of true edge.

    Reference: López de Prado, "The Deflated Sharpe Ratio" (2014),
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
    """
    arr = np.asarray(
        list(strategy_returns) if not isinstance(strategy_returns, pd.Series) else strategy_returns,
        dtype=float,
    )
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 2:
        return float("nan")
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    if std <= 0:
        # Zero variance — degenerate. Defensive: return 1.0 when mean > target
        # (deterministic edge), 0.0 otherwise.
        return 1.0 if mean > sharpe_target else 0.0
    sharpe_hat = mean / std

    if skew is None or kurt is None:
        sample_skew, sample_kurt = _sample_skew_kurt(arr)
        skew = sample_skew if skew is None else skew
        kurt = sample_kurt if kurt is None else kurt
    skew = float(skew)
    kurt = float(kurt)

    # Deflated threshold: SR* such that the multiple-testing-inflated
    # null distribution gives p(SR_max >= SR*) = 0.5.
    sr_star_threshold = float(sharpe_target) + (_expected_max_z(n_trials) / math.sqrt(max(n - 1, 1)))

    # Non-normality adjustment per BLP eq. (5).
    # Var(SR_hat) ≈ (1 − skew·SR + ((kurt − 1)/4)·SR^2) / (n − 1).
    var_term = 1.0 - skew * sharpe_hat + (kurt - 1.0) / 4.0 * sharpe_hat * sharpe_hat
    var_term = max(var_term, 1e-12)
    denom = math.sqrt(var_term / max(n - 1, 1))

    z = (sharpe_hat - sr_star_threshold) / denom
    return float(_normal_cdf(z))


def probability_of_backtest_overfitting(
    performance_matrix: pd.DataFrame,
    *,
    n_partitions: int = 16,
) -> float:
    """Bailey, Borwein, López de Prado, Zhu (2017) PBO.

    The combinatorial-purged-cross-validation form: split the
    performance time-axis into ``n_partitions`` equal halves taken at a
    time, rank strategies in-sample, and measure how often the in-sample
    best ranks below median out-of-sample. PBO is the share of those
    (overfit) outcomes across all combinatorial splits.

    Inputs:

    - ``performance_matrix``: ``T × S`` frame; rows are time periods,
      columns are candidate strategies, values are per-period
      performance (e.g. Sharpe in that period).
    - ``n_partitions``: the ``2N`` halves are produced from
      ``n_partitions`` blocks. ``n_partitions=16`` matches the BBLZ
      reference; lower values reduce variance but raise the floor on
      detectable overfitting.

    Returns PBO in ``[0, 1]``. ``< 0.05`` is the production-profile bar.

    Reference: Bailey, Borwein, López de Prado, Zhu, "The probability of
    backtest overfitting" (2017),
    https://www.researchgate.net/publication/271215436
    """
    if performance_matrix is None or performance_matrix.empty:
        return float("nan")
    n_partitions = int(n_partitions)
    if n_partitions < 2 or n_partitions % 2 != 0:
        raise ValueError("n_partitions must be an even integer ≥ 2")
    mat = performance_matrix.to_numpy(dtype=float)
    t, s = mat.shape
    if s < 2 or t < n_partitions:
        return float("nan")

    block_edges = np.linspace(0, t, n_partitions + 1, dtype=int)
    block_indices: list[np.ndarray] = [
        np.arange(block_edges[i], block_edges[i + 1]) for i in range(n_partitions)
    ]

    # Each combination picks half the blocks as IS, the other half as OOS.
    from itertools import combinations

    n_overfit = 0
    n_total = 0
    for combo in combinations(range(n_partitions), n_partitions // 2):
        is_blocks = np.concatenate([block_indices[i] for i in combo])
        oos_blocks = np.concatenate(
            [block_indices[i] for i in range(n_partitions) if i not in combo]
        )
        if is_blocks.size == 0 or oos_blocks.size == 0:
            continue
        is_perf = mat[is_blocks].mean(axis=0)
        oos_perf = mat[oos_blocks].mean(axis=0)

        is_best = int(np.argmax(is_perf))
        oos_ranks = pd.Series(oos_perf).rank(method="average", ascending=False)
        n_strategies = oos_ranks.size
        median_rank = (n_strategies + 1) / 2.0
        if float(oos_ranks.iloc[is_best]) > median_rank:
            n_overfit += 1
        n_total += 1

    if n_total == 0:
        return float("nan")
    return float(n_overfit) / float(n_total)


def minimum_track_record_length(
    sharpe_observed: float,
    sharpe_target: float = 0.0,
    *,
    skew: float = 0.0,
    excess_kurt: float = 0.0,
    confidence: float = 0.95,
) -> float:
    """Minimum Track Record Length (Bailey–López de Prado).

    Returns the minimum number of observations ``n*`` such that the
    observed Sharpe ``sharpe_observed`` is statistically greater than the
    target Sharpe ``sharpe_target`` at the requested confidence under the
    same non-normality correction as the DSR.

    Closed form (BLP 2014, eq. 8):

        n* = 1 + (1 − skew·SR + (kurt − 1)/4 · SR²) · (Φ⁻¹(C) / (SR − SR_target))²

    where ``Φ⁻¹`` is the inverse standard-normal CDF and ``C`` is the
    requested confidence (default 0.95).

    Returns ``inf`` when ``sharpe_observed <= sharpe_target`` (the
    inequality can never be defended).
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0,1); got {confidence!r}")
    if sharpe_observed <= sharpe_target:
        return float("inf")
    z = _normal_ppf(confidence)
    var_term = 1.0 - skew * sharpe_observed + (excess_kurt) / 4.0 * sharpe_observed * sharpe_observed
    var_term = max(var_term, 1e-12)
    diff = sharpe_observed - sharpe_target
    return 1.0 + var_term * (z / diff) ** 2


__all__ = [
    "BinaryValidationResult",
    "EPS",
    "brier_score",
    "calibration_table",
    "deflated_sharpe",
    "expected_calibration_error",
    "log_loss_score",
    "minimum_track_record_length",
    "pinball_loss",
    "probability_of_backtest_overfitting",
    "quantile_coverage",
    "validate_binary_forecast",
    "validation_frame",
]
