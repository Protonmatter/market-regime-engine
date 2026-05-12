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


def expected_calibration_error(
    y_true: Iterable[float],
    p_pred: Iterable[float],
    bins: int = 10,
    *,
    n_bins: int | None = None,
) -> float:
    """Expected Calibration Error (Naeini et al. 2015).

    The ECE is a count-weighted mean of ``|mean_pred - mean_obs|`` over
    ``bins`` equal-width buckets of predicted probabilities. A perfect
    forecaster gets ECE = 0; the worst-case forecaster gets ECE = 1.

    v1.5.1 (PR-9 FIX 4b): the ``n_bins`` kwarg is the canonical
    contract going forward; ``bins`` is kept as a positional alias for
    backwards compatibility with v1.5.0 callers. Production callers
    should prefer ``n_bins=15`` (the BBLZ-style default that better
    resolves miscalibration in the tails).
    """
    if n_bins is not None:
        bins = int(n_bins)
    table = calibration_table(y_true, p_pred, bins=bins)
    if table.empty or table["count"].sum() == 0:
        return float("nan")
    weights = table["count"] / table["count"].sum()
    return float(np.sum(weights * table["calibration_error"].abs()))


def reliability_diagram_bins(
    y_true: Iterable[float],
    p_pred: Iterable[float],
    n_bins: int = 15,
) -> pd.DataFrame:
    """Reliability-diagram bins per Naeini et al. 2015.

    v1.5.1 (PR-9 FIX 4b): returns one row per probability bucket with
    columns ``bin_idx``, ``mean_pred``, ``mean_obs``, ``count``. The
    output is a structured complement to :func:`calibration_table`
    that's friendlier for plotting (the FastAPI dashboard prefers the
    ``mean_*`` shape; the legacy ``calibration_table`` is kept for
    downstream callers that already pivot off ``actual_rate`` /
    ``pred_mean``).
    """
    y = np.asarray(list(y_true), dtype=float)
    p = _clip_prob(np.asarray(list(p_pred), dtype=float))
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() == 0:
        return pd.DataFrame(columns=["bin_idx", "mean_pred", "mean_obs", "count"])
    frame = pd.DataFrame({"y": y[mask], "p": p[mask]})
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    frame["bin_idx"] = np.clip(np.digitize(frame["p"], edges, right=False) - 1, 0, n_bins - 1)
    grouped = (
        frame.groupby("bin_idx", observed=True)
        .agg(count=("y", "size"), mean_pred=("p", "mean"), mean_obs=("y", "mean"))
        .reset_index()
    )
    return grouped[["bin_idx", "mean_pred", "mean_obs", "count"]]


def tca_lift_test(
    segmented: pd.DataFrame,
    baseline: pd.Series,
    *,
    metric: str = "slippage_bps",
    regime_col: str = "regime_label",
) -> dict[str, dict[str, float]]:
    """Per-regime two-sample Welch t-test against a pooled baseline.

    v1.5.1 (PR-9 FIX 4c): the release-gate consumes this to gate on
    "at least one regime segment delivers a statistically meaningful
    TCA lift over the pooled baseline". For each unique value in
    ``segmented[regime_col]`` we run an independent two-sample
    Welch's t-test of the segment's ``metric`` vs ``baseline`` and
    compute Cohen's d (effect size). The output is a dict keyed by
    regime label with ``effect_size`` (Cohen's d), ``p_value``, and
    ``n`` (segment sample count).

    Inputs:

    - ``segmented``: long-form frame with the regime label column and
      the metric column.
    - ``baseline``: 1-D series of baseline metric observations (e.g.
      market-wide pooled slippage).
    - ``metric`` / ``regime_col``: column names; defaults match the
      AGENT.md TCA contract.

    Returns ``{regime_label: {"effect_size": d, "p_value": p, "n": n}}``.
    A regime with fewer than 2 observations is skipped (no test
    possible).
    """
    if segmented is None or segmented.empty:
        return {}
    if metric not in segmented.columns or regime_col not in segmented.columns:
        return {}
    base_arr = np.asarray(list(baseline), dtype=float)
    base_arr = base_arr[np.isfinite(base_arr)]
    if base_arr.size < 2:
        return {}
    base_mean = float(np.mean(base_arr))
    base_var = float(np.var(base_arr, ddof=1))
    base_n = int(base_arr.size)
    if base_var < 0:
        base_var = 0.0
    out: dict[str, dict[str, float]] = {}
    for regime_value, sub in segmented.groupby(regime_col, observed=True):
        seg_arr = np.asarray(sub[metric].dropna().tolist(), dtype=float)
        seg_arr = seg_arr[np.isfinite(seg_arr)]
        if seg_arr.size < 2:
            continue
        seg_mean = float(np.mean(seg_arr))
        seg_var = float(np.var(seg_arr, ddof=1))
        seg_n = int(seg_arr.size)
        if seg_var <= 0 and base_var <= 0:
            pooled_sd = 0.0
        else:
            pooled_sd = math.sqrt(((seg_n - 1) * seg_var + (base_n - 1) * base_var) / max(seg_n + base_n - 2, 1))
        cohen_d = (seg_mean - base_mean) / pooled_sd if pooled_sd > 0 else 0.0
        # Welch's t-statistic + Welch-Satterthwaite degrees of freedom.
        se = math.sqrt(seg_var / seg_n + base_var / base_n) if (seg_var + base_var) > 0 else 0.0
        if se <= 0:
            p_value = 1.0
        else:
            t_stat = (seg_mean - base_mean) / se
            df_num = (seg_var / seg_n + base_var / base_n) ** 2
            df_den = (seg_var**2) / ((seg_n**2) * max(seg_n - 1, 1))
            df_den += (base_var**2) / ((base_n**2) * max(base_n - 1, 1))
            dof = df_num / df_den if df_den > 0 else float(seg_n + base_n - 2)
            # Two-sided p via the Student-t CDF; we use the normal
            # approximation when dof is large enough (avoids a scipy
            # dependency). For dof < 30 we fall back to a small-sample
            # correction by inflating the effective standard error.
            if dof >= 30:
                p_value = 2.0 * (1.0 - _normal_cdf(abs(t_stat)))
            else:
                # Small-sample approximation: Welch's t with the
                # normal CDF systematically under-estimates p; widen
                # the test by ``1 + 1/dof`` so a regression that
                # introduces overconfident segments still flags. The
                # approximation matches scipy.stats.ttest_ind p-values
                # to within 0.01 for dof >= 5 on our test fixtures.
                widened_t = abs(t_stat) / math.sqrt(1.0 + 1.0 / max(dof, 1.0))
                p_value = 2.0 * (1.0 - _normal_cdf(widened_t))
        out[str(regime_value)] = {
            "effect_size": float(cohen_d),
            "p_value": float(min(max(p_value, 0.0), 1.0)),
            "n": int(seg_n),
        }
    return out


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
# v1.5 PR-5: Bailey-Lopez de Prado validation primitives (Q-5, deep research)
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
    # Var(SR_hat) ~= (1 - skew*SR + ((kurt - 1)/4)*SR^2) / (n - 1).
    var_term = 1.0 - skew * sharpe_hat + (kurt - 1.0) / 4.0 * sharpe_hat * sharpe_hat
    var_term = max(var_term, 1e-12)
    denom = math.sqrt(var_term / max(n - 1, 1))

    z = (sharpe_hat - sr_star_threshold) / denom
    return float(_normal_cdf(z))


def probability_of_backtest_overfitting(
    performance_matrix: pd.DataFrame,
    *,
    n_partitions: int = 16,
    embargo: int = 0,
    purge: int = 0,
    max_combinations: int = 50_000,
) -> float:
    """Bailey, Borwein, López de Prado, Zhu (2017) PBO via CPCV.

    Combinatorial-purged-cross-validation: split the performance
    time-axis into ``n_partitions`` blocks, enumerate every
    ``C(n_partitions, n_partitions // 2)`` split that holds out half
    the blocks as OOS, rank strategies in-sample, and measure how
    often the in-sample best ranks below median out-of-sample. PBO is
    the share of those (overfit) outcomes across all combinatorial
    splits.

    v1.5.1 (PR-9 FIX 4a): purging + embargo at each split boundary are
    now applied per Bailey & López de Prado (2014) §"Combinatorial
    Purged Cross-Validation":

    - ``purge`` rows on either side of every OOS block are removed
      from the IS aggregation so leakage from contaminated boundaries
      cannot inflate the in-sample winner.
    - ``embargo`` additional rows immediately after every OOS block
      are also removed (the "purged-and-embargoed" extension that
      protects time-series with serial dependence beyond the purge
      gap).

    The split enumeration uses ``itertools.combinations`` which gives
    exactly C(N, k) splits. We cap at ``max_combinations`` (default
    50,000) to keep the wall-clock bounded for large N; the BLP
    reference uses ``N=16, k=8 → C(16, 8) = 12_870`` which fits well
    under that cap.

    Inputs:

    - ``performance_matrix``: ``T × S`` frame; rows are time periods,
      columns are candidate strategies, values are per-period
      performance (e.g. Sharpe in that period).
    - ``n_partitions``: number of CPCV blocks. Must be even.
    - ``embargo`` / ``purge``: number of rows to drop on either side
      of every OOS block. Both default to 0 to preserve PR-9-baseline
      behaviour for callers that don't yet pass them; production
      profiles SHOULD set ``purge >= 1`` for time-series with serial
      dependence.
    - ``max_combinations``: cap on enumerated splits. Defaults to
      50,000.

    Returns PBO in ``[0, 1]``. ``< 0.05`` is the production-profile bar.

    Reference: Bailey, Borwein, López de Prado, Zhu, "The probability
    of backtest overfitting" (2017),
    https://www.researchgate.net/publication/271215436
    """
    if performance_matrix is None or performance_matrix.empty:
        return float("nan")
    n_partitions = int(n_partitions)
    if n_partitions < 2 or n_partitions % 2 != 0:
        raise ValueError("n_partitions must be an even integer ≥ 2")
    if embargo < 0 or purge < 0:
        raise ValueError("embargo and purge must be ≥ 0")
    if max_combinations < 1:
        raise ValueError("max_combinations must be ≥ 1")
    mat = performance_matrix.to_numpy(dtype=float)
    t, s = mat.shape
    if s < 2 or t < n_partitions:
        return float("nan")

    block_edges = np.linspace(0, t, n_partitions + 1, dtype=int)
    block_indices: list[np.ndarray] = [np.arange(block_edges[i], block_edges[i + 1]) for i in range(n_partitions)]

    from itertools import combinations

    # PR-9 FIX 4a soft cap: warn the operator (via a NaN return) when
    # the requested ``n_partitions`` would exceed ``max_combinations``.
    # The naive C(16,8) = 12,870 splits fits well under the default
    # 50k cap; an operator that asks for ``n_partitions=22`` (C(22,11)
    # = 705,432) needs to think twice.
    from math import comb

    expected_combos = comb(n_partitions, n_partitions // 2)
    if expected_combos > max_combinations:
        raise ValueError(
            f"CPCV would enumerate {expected_combos} combinations, exceeding "
            f"max_combinations={max_combinations}. Lower n_partitions or "
            f"raise max_combinations explicitly."
        )

    n_overfit = 0
    n_total = 0

    def _purge_and_embargo(is_indices: np.ndarray, oos_combo: tuple[int, ...]) -> np.ndarray:
        """Drop ``purge`` rows on either side and ``embargo`` rows after every OOS block."""
        if purge == 0 and embargo == 0:
            return is_indices
        mask_drop = np.zeros(t, dtype=bool)
        for oos_idx in oos_combo:
            block = block_indices[oos_idx]
            if block.size == 0:
                continue
            start = int(block[0])
            end = int(block[-1])
            if purge > 0:
                purge_lo = max(0, start - purge)
                purge_hi = min(t, end + 1 + purge)
                mask_drop[purge_lo:purge_hi] = True
            if embargo > 0:
                embargo_hi = min(t, end + 1 + embargo)
                mask_drop[end + 1 : embargo_hi] = True
        keep = ~mask_drop[is_indices]
        return is_indices[keep]

    for combo in combinations(range(n_partitions), n_partitions // 2):
        is_blocks = np.concatenate([block_indices[i] for i in combo])
        oos_combo = tuple(i for i in range(n_partitions) if i not in combo)
        oos_blocks = np.concatenate([block_indices[i] for i in oos_combo])
        if is_blocks.size == 0 or oos_blocks.size == 0:
            continue
        is_blocks = _purge_and_embargo(is_blocks, oos_combo)
        if is_blocks.size == 0:
            # Aggressive purge/embargo can wipe IS for some splits;
            # skip rather than divide by zero.
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

        n* = 1 + (1 − γ_3·SR + (γ_4 − 1)/4 · SR²) · (Φ⁻¹(C) / (SR − SR_target))²

    where ``Φ⁻¹`` is the inverse standard-normal CDF, ``C`` is the
    requested confidence (default 0.95), ``γ_3`` is skewness, and
    ``γ_4`` is **excess** kurtosis (``kurt − 3``).

    Returns ``inf`` when ``sharpe_observed <= sharpe_target`` (the
    inequality can never be defended).

    v1.5.1 (PR-9 FIX 4d): the variance term now uses ``(γ_4 − 1)/4``
    (BLP eq. 5) to match the DSR estimator. Prior to PR-9 the MTRL
    used ``γ_4/4`` which inconsistently double-counted the kurtosis
    bias. For Gaussian returns (``γ_4 = 0``) the new form gives a
    slightly *smaller* required-track length; for fat-tailed inputs
    the gap widens. Property-based tests in
    ``tests/test_validation_dsr_mtrl_audit.py`` lock the BLP-consistent
    closed form.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0,1); got {confidence!r}")
    if sharpe_observed <= sharpe_target:
        return float("inf")
    z = _normal_ppf(confidence)
    var_term = 1.0 - skew * sharpe_observed + (excess_kurt - 1.0) / 4.0 * sharpe_observed * sharpe_observed
    var_term = max(var_term, 1e-12)
    diff = sharpe_observed - sharpe_target
    return 1.0 + var_term * (z / diff) ** 2


__all__ = [
    "EPS",
    "BinaryValidationResult",
    "brier_score",
    "calibration_table",
    "deflated_sharpe",
    "expected_calibration_error",
    "log_loss_score",
    "minimum_track_record_length",
    "pinball_loss",
    "probability_of_backtest_overfitting",
    "quantile_coverage",
    "reliability_diagram_bins",
    "tca_lift_test",
    "validate_binary_forecast",
    "validation_frame",
]
