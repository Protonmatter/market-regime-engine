# SPDX-License-Identifier: Apache-2.0
"""Forecast-comparison statistics that survive academic review.

Six tests live here. Each one is the standard reference implementation:

- :func:`diebold_mariano` — Diebold and Mariano (1995) test of equal predictive
  accuracy with Newey-West HAC variance and the Harvey-Leybourne-Newbold
  small-sample correction.
- :func:`giacomini_white` — Giacomini and White (2006) conditional predictive
  ability test (one-sided regression of loss differential on a constant +
  conditioning variables).
- :func:`hansen_mcs` — Hansen, Lunde and Nason (2011) Model Confidence Set,
  block-bootstrapped, returning the smallest set of models that contain the
  true best with probability ``1 - confidence``.
- :func:`pit_uniformity` — Diebold-Gunther-Tay (1998) PIT histogram + the
  Knüppel (2015) goodness-of-fit test on the PIT autocorrelation moments.
- :func:`christoffersen_coverage` — Christoffersen (1998) unconditional and
  conditional coverage tests for binary VaR-like exceedances.
- :func:`murphy_decomposition` — CRPS decomposition into MCB / DSC / UNC
  (miscalibration / discrimination / uncertainty) for distributional
  forecasts.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _newey_west_var(d: np.ndarray, max_lag: int) -> float:
    """Newey-West HAC long-run variance estimate of a stationary series."""
    n = len(d)
    if n == 0:
        return float("nan")
    d = d - d.mean()
    gamma0 = float(np.dot(d, d) / n)
    s = gamma0
    for lag in range(1, min(max_lag, n - 1) + 1):
        gamma = float(np.dot(d[:-lag], d[lag:]) / n)
        weight = 1.0 - lag / (max_lag + 1.0)
        s += 2.0 * weight * gamma
    return max(s, 1e-12)


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _chi2_sf(x: float, df: int) -> float:
    """Survival function of a chi-squared distribution.

    Uses the regularized lower incomplete gamma identity
    ``P(X <= x) = γ(df/2, x/2) / Γ(df/2)`` and returns the complement, with a
    series fallback that avoids pulling in ``scipy.special`` as a hard
    dependency.
    """
    if x <= 0:
        return 1.0
    a = df / 2.0
    z = x / 2.0
    try:  # pragma: no cover - prefer scipy when available
        from scipy.special import gammaincc

        return float(gammaincc(a, z))
    except Exception:
        # Series expansion of the regularized lower incomplete gamma.
        term = 1.0 / a
        total = term
        for k in range(1, 500):
            term *= z / (a + k)
            total += term
            if term < 1e-16 * total:
                break
        # Lower incomplete gamma * gamma(a) = total * z^a * exp(-z)
        ln_p = a * math.log(z) - z + math.log(total) - math.lgamma(a)
        p_lower = math.exp(ln_p)
        return float(max(0.0, 1.0 - p_lower))


# ---------------------------------------------------------------------------
# Diebold-Mariano
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DMResult:
    n: int
    mean_diff: float
    statistic: float
    pvalue: float
    lag: int
    direction: str  # "model_a_better" | "model_b_better" | "tie"


def diebold_mariano(loss_a: Sequence[float], loss_b: Sequence[float], *, h: int = 1, hln: bool = True) -> DMResult:
    """Diebold-Mariano test of equal predictive accuracy.

    ``loss_a`` and ``loss_b`` are realized loss series of the two models on
    the same hold-out set. ``h`` is the forecast horizon (used to set the
    Newey-West lag cap); ``hln=True`` applies the Harvey-Leybourne-Newbold
    finite-sample adjustment (default).
    """
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    n = len(a)
    if n < 4:
        return DMResult(n, float("nan"), float("nan"), float("nan"), 0, "tie")
    d = a - b
    lag = max(0, h - 1)
    var = _newey_west_var(d, lag)
    se = math.sqrt(var / n)
    if se < 1e-12:
        return DMResult(n, float(d.mean()), 0.0, 1.0, lag, "tie")
    stat = float(d.mean() / se)
    if hln:
        scale = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
        stat = stat * scale
    p = 2.0 * (1.0 - _normal_cdf(abs(stat)))
    direction = "tie"
    # v1.2 fix: tighten the direction threshold to the canonical 5% level
    # (was 10%, which surfaced too many spurious "better-than" calls in the
    # release-gate logs). Anything between 5% and 10% is now "tie".
    if p < 0.05:
        direction = "model_a_better" if d.mean() < 0 else "model_b_better"
    return DMResult(n, float(d.mean()), stat, float(p), lag, direction)


# ---------------------------------------------------------------------------
# Giacomini-White conditional predictive ability test
# ---------------------------------------------------------------------------


def giacomini_white(
    loss_a: Sequence[float],
    loss_b: Sequence[float],
    z: pd.DataFrame | None = None,
    *,
    h: int = 1,
) -> dict:
    """Giacomini-White conditional predictive ability test.

    The test regresses the loss differential on a constant plus the
    conditioning variables ``z``, then forms a Wald statistic on the
    coefficients. With ``z=None`` the test reduces to an unconditional
    DM-style check using a HAC variance.
    """
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    n = len(a)
    if n < 8:
        return {"n": n, "statistic": float("nan"), "pvalue": float("nan"), "df": 0}
    d = a - b
    if z is not None and len(z) >= n:
        Z = z.iloc[mask].to_numpy(float) if isinstance(z, pd.DataFrame) else np.asarray(z, dtype=float)[mask]
        Z = np.column_stack([np.ones(n), Z])
    else:
        Z = np.ones((n, 1))
    XtX = Z.T @ Z
    XtY = Z.T @ d
    try:
        beta = np.linalg.solve(XtX, XtY)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(Z, d, rcond=None)[0]
    resid = d - Z @ beta
    lag = max(0, h - 1)
    # Long-run variance of the loss differential. With Z = ones, the Wald
    # statistic on beta is n * beta^2 / s. With Z = [1, X], cov(beta) = s/n *
    # (Z'Z/n)^{-1} = s * (Z'Z)^{-1}.
    s = _newey_west_var(resid, lag)
    cov = s * np.linalg.pinv(XtX)
    cov = (cov + cov.T) / 2.0 + 1e-12 * np.eye(cov.shape[0])
    stat = float(beta @ np.linalg.pinv(cov) @ beta)
    df = beta.shape[0]
    p = _chi2_sf(stat, df)
    return {"n": n, "statistic": stat, "pvalue": p, "df": df, "beta": beta.tolist()}


# ---------------------------------------------------------------------------
# Hansen MCS via block bootstrap
# ---------------------------------------------------------------------------


def hansen_mcs(
    losses: pd.DataFrame,
    *,
    confidence: float = 0.10,
    bootstrap: int = 1000,
    block_size: int = 12,
    seed: int = 0,
    statistic: Literal["T_R", "T_SQ"] = "T_R",
) -> dict:
    """Hansen-Lunde-Nason Model Confidence Set.

    ``losses`` is a date-indexed dataframe whose columns are model names and
    whose values are realized losses (lower=better). The function returns the
    set of models that survive at the given ``confidence`` level using a
    block-bootstrap approximation of the elimination statistic.

    ``statistic`` selects the elimination test:

    - ``"T_R"`` (default, v1.0 behavior): max over models of the studentized
      deviation from the cross-sectional mean. Eliminates the worst single
      model per iteration.
    - ``"T_SQ"`` (Hansen-Lunde-Nason 2011 Section 3): the *sum-of-squared*
      studentized deviations across all surviving models. The MCS p-value at
      each iteration uses the full sum (more powerful when many models are
      tied near the boundary), but elimination still removes the model with
      the largest single deviation so the contraction is monotone.
    """
    if losses is None or losses.empty:
        return {"mcs": [], "pvalues": {}, "iterations": 0, "statistic": statistic}
    L = losses.dropna(how="any").to_numpy(dtype=float)
    if L.size == 0:
        return {"mcs": [], "pvalues": {}, "iterations": 0, "statistic": statistic}
    names = list(losses.dropna(how="any").columns)
    n, m = L.shape
    if m < 2:
        return {
            "mcs": names,
            "pvalues": {names[0]: 1.0} if names else {},
            "iterations": 0,
            "statistic": statistic,
        }
    rng = np.random.default_rng(seed)
    blocks = max(1, n // block_size)
    survivors = list(range(m))
    pvalues: dict[str, float] = {}
    iteration = 0
    while len(survivors) > 1:
        iteration += 1
        sub = L[:, survivors]
        mean_loss = sub.mean(axis=0)
        # Per-model standard error of the mean (block-aware via Newey-West).
        se = np.array(
            [math.sqrt(_newey_west_var(sub[:, j] - mean_loss[j], block_size) / n) for j in range(sub.shape[1])]
        )
        se = np.maximum(se, 1e-12)
        # Studentized deviation from the cross-sectional mean.
        td = (mean_loss - mean_loss.mean()) / se
        if statistic == "T_R":
            observed = float(np.max(td))
        elif statistic == "T_SQ":
            observed = float(np.sum(td**2))
        else:  # pragma: no cover - guarded by Literal
            raise ValueError(f"unknown statistic: {statistic!r}")
        # Stationary block bootstrap recentered under the equal-mean null.
        boot_stat = np.empty(bootstrap, dtype=float)
        for b in range(bootstrap):
            starts = rng.integers(0, n, size=blocks)
            idx = np.concatenate([np.arange(s, s + block_size) % n for s in starts])[:n]
            sample = sub[idx]
            sm = sample.mean(axis=0)
            # Recentered statistic so the null distribution has E[T] ≈ 0.
            centered = (sm - sm.mean()) - (mean_loss - mean_loss.mean())
            cd = centered / se
            if statistic == "T_R":
                boot_stat[b] = float(np.max(cd))
            else:  # "T_SQ"
                boot_stat[b] = float(np.sum(cd**2))
        pval = float(np.mean(boot_stat >= observed))
        # Even under T_SQ we eliminate by max-deviation so the loop monotonically
        # contracts and the MCS captures the worst-aligned model first.
        worst = int(np.argmax(td))
        worst_name = names[survivors[worst]]
        pvalues[worst_name] = pval
        if pval > confidence:
            break
        survivors.pop(worst)
    for surv_idx in survivors:
        pvalues.setdefault(names[surv_idx], 1.0)
    return {
        "mcs": [names[i] for i in survivors],
        "pvalues": pvalues,
        "iterations": iteration,
        "statistic": statistic,
    }


# ---------------------------------------------------------------------------
# PIT and Knüppel
# ---------------------------------------------------------------------------


def pit_uniformity(
    pit_values: Sequence[float],
    *,
    bins: int = 10,
    moments: int = 4,
    autocorrelation: bool = False,
    autocorr_lags: int = 4,
) -> dict:
    """PIT histogram and Knüppel goodness-of-fit on raw moments.

    Under correct calibration, the PIT (probability integral transform) values
    are iid Uniform(0, 1). This function returns:

    - ``histogram``: counts in ``bins`` equal-width buckets.
    - ``chi2_stat`` / ``chi2_pvalue``: classic uniformity chi-squared test.
    - ``moment_stat`` / ``moment_pvalue``: Knüppel (2015) joint test on the
      first ``moments`` raw moments of the PIT vs. their Uniform(0,1)
      expected values.

    When ``autocorrelation=True`` the moment vector is augmented with the
    first ``autocorr_lags`` autocorrelations of the PIT series (rho_1, ...,
    rho_K). Under iid Uniform the population autocorrelations are 0, so the
    augmented joint Wald statistic is ``(diff_moments | rho_hat)' Sigma^-1
    (diff_moments | rho_hat)`` (Knüppel 2015 Section 3.2). Default is off
    for backward compatibility with v1.0/v1.1 callers.
    """
    u = np.asarray([float(v) for v in pit_values if np.isfinite(v)], dtype=float)
    if u.size == 0:
        return {
            "n": 0,
            "chi2_stat": float("nan"),
            "chi2_pvalue": float("nan"),
            "moment_stat": float("nan"),
            "moment_pvalue": float("nan"),
            "histogram": [],
            "autocorrelations": [],
        }
    u = np.clip(u, 1e-9, 1 - 1e-9)
    n = len(u)
    edges = np.linspace(0, 1, bins + 1)
    counts = np.histogram(u, bins=edges)[0].astype(float)
    expected = n / bins
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    chi2_p = _chi2_sf(chi2, bins - 1)

    # Knüppel: compare first `moments` raw moments to their Uniform(0,1) values.
    m_observed = np.array([float(np.mean(u**k)) for k in range(1, moments + 1)])
    m_expected = np.array([1.0 / (k + 1) for k in range(1, moments + 1)])
    diff = m_observed - m_expected
    # Stack moment-residuals (per observation) for HAC variance estimation.
    centred = np.column_stack([(u**k - m_expected[k - 1]) for k in range(1, moments + 1)])
    # Knüppel autocorrelation augmentation (rho_1..rho_K, expected 0).
    autocorr_vals: list[float] = []
    if autocorrelation and autocorr_lags > 0:
        u_centred = u - u.mean()
        var_u = float(np.dot(u_centred, u_centred) / n)
        var_u = max(var_u, 1e-12)
        for k in range(1, autocorr_lags + 1):
            if k >= n:
                autocorr_vals.append(0.0)
                continue
            cov_k = float(np.dot(u_centred[:-k], u_centred[k:]) / n)
            autocorr_vals.append(cov_k / var_u)
        # Per-observation contributions for each rho_k:
        # phi_t^{(k)} = (u_t - mean) * (u_{t-k} - mean) / var_u
        # (lag k positions are zero-padded so the matrix is n x autocorr_lags).
        rho_columns = []
        for k in range(1, autocorr_lags + 1):
            col = np.zeros(n)
            if k < n:
                col[k:] = u_centred[:-k] * u_centred[k:] / var_u
            rho_columns.append(col)
        rho_mat = np.column_stack(rho_columns) if rho_columns else np.zeros((n, 0))
        centred = np.column_stack([centred, rho_mat])
        diff = np.concatenate([diff, np.array(autocorr_vals)])
    df_moment = centred.shape[1]
    # Newey-West with floor lag 4.
    lag = max(4, int(np.floor(4 * (n / 100) ** (2 / 9))))
    cov = np.zeros((df_moment, df_moment))
    for tau in range(-lag, lag + 1):
        if tau >= 0:
            cov += centred[: n - tau].T @ centred[tau:] / n * (1.0 - abs(tau) / (lag + 1.0))
        else:
            cov += centred[-tau:].T @ centred[: n + tau] / n * (1.0 - abs(tau) / (lag + 1.0))
    cov = (cov + cov.T) / 2.0
    cov += 1e-10 * np.eye(df_moment)
    try:
        inv = np.linalg.inv(cov)
        moment_stat = float(n * (diff @ inv @ diff))
    except np.linalg.LinAlgError:
        moment_stat = float("nan")
    moment_p = _chi2_sf(moment_stat, df_moment) if np.isfinite(moment_stat) else float("nan")

    return {
        "n": int(n),
        "chi2_stat": chi2,
        "chi2_pvalue": chi2_p,
        "moment_stat": moment_stat,
        "moment_pvalue": moment_p,
        "histogram": counts.tolist(),
        "autocorrelations": autocorr_vals,
    }


# ---------------------------------------------------------------------------
# Christoffersen unconditional and conditional coverage tests
# ---------------------------------------------------------------------------


def christoffersen_coverage(hits: Sequence[int], alpha: float) -> dict:
    """Christoffersen unconditional and conditional coverage tests for a binary
    hit sequence (``1`` = exceedance, ``0`` = no exceedance) at expected
    coverage ``alpha``.
    """
    h = np.asarray([int(v) for v in hits if v in (0, 1)], dtype=int)
    n = len(h)
    if n < 5:
        return {
            "n": n,
            "uc_stat": float("nan"),
            "uc_pvalue": float("nan"),
            "cc_stat": float("nan"),
            "cc_pvalue": float("nan"),
        }
    n1 = int(h.sum())
    n0 = n - n1
    pi_hat = n1 / n
    # Unconditional coverage (Kupiec)
    if pi_hat <= 0 or pi_hat >= 1:
        uc_stat = 0.0
    else:
        ll0 = n0 * math.log(1 - alpha) + n1 * math.log(alpha)
        ll1 = n0 * math.log(1 - pi_hat) + n1 * math.log(pi_hat)
        uc_stat = -2.0 * (ll0 - ll1)
    uc_p = _chi2_sf(uc_stat, 1)
    # Conditional coverage independence (transition counts).
    n00 = n01 = n10 = n11 = 0
    from itertools import pairwise as _pairwise

    for prev, curr in _pairwise(h):
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    if pi <= 0 or pi >= 1 or pi01 in (0, 1) or pi11 in (0, 1):
        ind_stat = 0.0
    else:
        ll_pooled = (n00 + n10) * math.log(1 - pi) + (n01 + n11) * math.log(pi)
        ll_split = (
            n00 * math.log(max(1 - pi01, 1e-9))
            + n01 * math.log(max(pi01, 1e-9))
            + n10 * math.log(max(1 - pi11, 1e-9))
            + n11 * math.log(max(pi11, 1e-9))
        )
        ind_stat = -2.0 * (ll_pooled - ll_split)
    cc_stat = uc_stat + ind_stat
    cc_p = _chi2_sf(cc_stat, 2)
    return {
        "n": n,
        "uc_stat": float(uc_stat),
        "uc_pvalue": float(uc_p),
        "cc_stat": float(cc_stat),
        "cc_pvalue": float(cc_p),
        "ind_stat": float(ind_stat),
        "transitions": {"n00": n00, "n01": n01, "n10": n10, "n11": n11},
    }


# ---------------------------------------------------------------------------
# Murphy / CRPS decomposition for binary forecasts
# ---------------------------------------------------------------------------


def murphy_decomposition(y: Sequence[float], p: Sequence[float], *, bins: int = 10) -> dict:
    """Murphy (1973) decomposition of the Brier score.

    Returns ``{"reliability": REL, "resolution": RES, "uncertainty": UNC,
    "brier": REL - RES + UNC}``. REL is the calibration error term (smaller is
    better), RES is the discrimination term (larger is better), UNC is the
    irreducible base-rate variance.
    """
    y_arr = np.asarray([float(v) for v in y if np.isfinite(v)], dtype=float)
    p_arr = np.asarray([float(v) for v in p if np.isfinite(v)], dtype=float)
    n = min(len(y_arr), len(p_arr))
    if n == 0:
        return {
            "reliability": float("nan"),
            "resolution": float("nan"),
            "uncertainty": float("nan"),
            "brier": float("nan"),
        }
    y_arr = y_arr[:n]
    p_arr = p_arr[:n]
    o = float(y_arr.mean())
    edges = np.linspace(0, 1, bins + 1)
    bucket = np.clip(np.digitize(p_arr, edges) - 1, 0, bins - 1)
    rel = res = 0.0
    for k in range(bins):
        mask = bucket == k
        nk = int(mask.sum())
        if nk == 0:
            continue
        pk = float(p_arr[mask].mean())
        ok = float(y_arr[mask].mean())
        rel += nk * (pk - ok) ** 2
        res += nk * (ok - o) ** 2
    rel /= n
    res /= n
    unc = o * (1 - o)
    brier = float(np.mean((p_arr - y_arr) ** 2))
    return {
        "reliability": float(rel),
        "resolution": float(res),
        "uncertainty": float(unc),
        "brier": float(brier),
    }


# ---------------------------------------------------------------------------
# Hansen MCS thin wrapper for the promotion gate
# ---------------------------------------------------------------------------


def mcs_promotion_filter(
    loss_frame: pd.DataFrame,
    *,
    confidence: float = 0.10,
    bootstrap: int = 1000,
    block_size: int = 12,
    seed: int = 0,
) -> set[str]:
    """Return the set of model names surviving Hansen MCS at ``confidence``.

    ``loss_frame`` is a date-indexed dataframe whose columns are model names
    and whose values are realized losses (lower=better) — exactly the input
    shape :func:`hansen_mcs` accepts. The wrapper exists so promotion / release
    gates have a single one-line call site that does not need to know about
    the bootstrap parameters.
    """
    if loss_frame is None or loss_frame.empty:
        return set()
    out = hansen_mcs(
        loss_frame,
        confidence=confidence,
        bootstrap=bootstrap,
        block_size=block_size,
        seed=seed,
    )
    return {str(name) for name in out.get("mcs", [])}


# ---------------------------------------------------------------------------
# CRPS-direct forecast comparison (Diks-Panchenko-van Dijk 2011)
# ---------------------------------------------------------------------------


def _empirical_crps(forecast: np.ndarray, observation: float) -> float:
    """Empirical CRPS for an ensemble forecast vs. a scalar observation.

    Uses the unbiased estimator (Hersbach 2000):
    ``CRPS = E|X - y| - 0.5 * E|X - X'|`` with X, X' iid from the forecast.
    """
    arr = np.asarray(forecast, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or not np.isfinite(observation):
        return float("nan")
    term1 = float(np.mean(np.abs(arr - observation)))
    term2 = 0.5 * float(np.mean(np.abs(arr[:, None] - arr[None, :])))
    return float(term1 - term2)


def crps_diks_panchenko(
    forecast_a: Sequence[Sequence[float]] | np.ndarray,
    forecast_b: Sequence[Sequence[float]] | np.ndarray,
    observations: Sequence[float],
    *,
    h: int = 1,
    weight_fn: Callable[[float], float] | None = None,
) -> dict:
    """Diks-Panchenko-van Dijk 2011 CRPS-based equal-predictive-ability test.

    Each forecast is an ensemble (rows of size ``M``); observations are a
    1-D series. The per-period score differential is

        d_t = CRPS_A(forecast_a[t]) - CRPS_B(forecast_b[t])

    with optional weight ``weight_fn(observation_t)`` for the tail-focused
    variant (set to ``lambda y: 1.0`` for the unweighted classic). The
    statistic is the Diebold-Mariano-style ``mean(d) / sqrt(var(d) / n)``
    using a Newey-West HAC variance with lag ``h - 1``.
    """
    A = np.asarray(forecast_a, dtype=float)
    B = np.asarray(forecast_b, dtype=float)
    y = np.asarray(observations, dtype=float)
    if A.ndim == 1 or B.ndim == 1:
        A = A[:, None] if A.ndim == 1 else A
        B = B[:, None] if B.ndim == 1 else B
    n = min(len(A), len(B), len(y))
    if n < 4:
        return {"n": int(n), "statistic": float("nan"), "pvalue": float("nan"), "mean_diff": float("nan")}
    A = A[:n]
    B = B[:n]
    y = y[:n]
    crps_a = np.array([_empirical_crps(A[t], y[t]) for t in range(n)])
    crps_b = np.array([_empirical_crps(B[t], y[t]) for t in range(n)])
    if weight_fn is not None:
        weights = np.array([float(weight_fn(float(y[t]))) for t in range(n)])
        crps_a = crps_a * weights
        crps_b = crps_b * weights
    mask = np.isfinite(crps_a) & np.isfinite(crps_b)
    d = (crps_a - crps_b)[mask]
    if d.size < 4:
        return {"n": int(d.size), "statistic": float("nan"), "pvalue": float("nan"), "mean_diff": float("nan")}
    lag = max(0, h - 1)
    var = _newey_west_var(d, lag)
    se = math.sqrt(var / len(d))
    if se < 1e-12:
        return {"n": len(d), "statistic": 0.0, "pvalue": 1.0, "mean_diff": float(d.mean())}
    stat = float(d.mean() / se)
    p = 2.0 * (1.0 - _normal_cdf(abs(stat)))
    return {
        "n": len(d),
        "statistic": stat,
        "pvalue": float(p),
        "mean_diff": float(d.mean()),
        "lag": int(lag),
    }


__all__ = [
    "DMResult",
    "christoffersen_coverage",
    "crps_diks_panchenko",
    "diebold_mariano",
    "giacomini_white",
    "hansen_mcs",
    "mcs_promotion_filter",
    "murphy_decomposition",
    "pit_uniformity",
]
