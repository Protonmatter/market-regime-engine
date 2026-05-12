# SPDX-License-Identifier: Apache-2.0
"""Online Bayesian change-point detection (Adams and MacKay 2007).

Two production-ready cores:

- ``MultivariateNIWBOCPD`` is the v1.0 default. It uses a Normal-Inverse-Wishart
  conjugate prior on the Gaussian emission, which yields a closed-form
  multivariate Student-t posterior predictive density (Murphy 2007). This
  captures cross-domain covariance, which the diagonal-Student-t fallback misses
  when, for example, credit and labor stress co-move during a recession.

- ``DiagonalStudentTBOCPD`` is kept as a numerically conservative fallback for
  tiny windows where the full NIW posterior is poorly conditioned. It is also
  what the existing tests pin so it must remain importable and behaviorally
  identical to the v0.8 implementation.

Both classes return a frame with ``change_point_prob``, ``bocpd_run_length_mean``,
``bocpd_map_run_length`` and ``predictive_log_likelihood``.

The hazard can be supplied directly or learned from an empirical regime path
via ``learned_constant_hazard``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from market_regime_engine.frontier.data_cleaning import NanPolicy, clean_with_policy

# ---------------------------------------------------------------------------
# numerical helpers
# ---------------------------------------------------------------------------


def _logsumexp(values: Iterable[float]) -> float:
    vals = np.asarray(list(values), dtype=float)
    if vals.size == 0:
        return -np.inf
    m = np.nanmax(vals)
    if not np.isfinite(m):
        return -np.inf
    return float(m + np.log(np.nansum(np.exp(vals - m))))


def _multigammaln(a: float, d: int) -> float:
    """Log of the multivariate gamma function ``Γ_d(a)``."""
    s = 0.25 * d * (d - 1) * math.log(math.pi)
    for j in range(d):
        s += math.lgamma(a - 0.5 * j)
    return s


# ---------------------------------------------------------------------------
# diagonal Student-t fallback (kept identical to v0.8 behavior)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunningDiagState:
    """Running diagonal mean/variance state used by the diagonal Student-t fallback.

    This is intentionally diagonal. Macro domain scores are low-dimensional but
    correlated; the diagonal predictive density is much more numerically stable
    for tiny run lengths than the full NIW posterior. The diagonal fallback is
    retained so callers who explicitly want it (or who run on degenerate
    pre-warmup windows) can opt in.
    """

    n: int
    mean: np.ndarray
    m2: np.ndarray
    prior_var: float = 1.0

    @classmethod
    def prior(cls, dim: int, prior_var: float = 1.0) -> RunningDiagState:
        return cls(n=0, mean=np.zeros(dim, dtype=float), m2=np.zeros(dim, dtype=float), prior_var=prior_var)

    def update(self, x: np.ndarray) -> RunningDiagState:
        x = np.asarray(x, dtype=float)
        if self.n == 0:
            return RunningDiagState(1, x.copy(), np.zeros_like(x), self.prior_var)
        n = self.n + 1
        delta = x - self.mean
        mean = self.mean + delta / n
        delta2 = x - mean
        m2 = self.m2 + delta * delta2
        return RunningDiagState(n, mean, m2, self.prior_var)

    @property
    def variance(self) -> np.ndarray:
        if self.n < 2:
            return np.full_like(self.mean, self.prior_var, dtype=float)
        return np.maximum(self.m2 / max(self.n - 1, 1), 1e-8)


def _student_t_logpdf_diag(x: np.ndarray, state: RunningDiagState, min_df: float = 3.0) -> float:
    """Independent Student-t predictive log density."""
    x = np.asarray(x, dtype=float)
    df = max(min_df, float(state.n + 1))
    var = state.variance
    scale = np.sqrt(np.maximum(var * (1.0 + 1.0 / max(state.n + 1, 1)), 1e-8))
    z = (x - state.mean) / scale
    c = math.lgamma((df + 1.0) / 2.0) - math.lgamma(df / 2.0) - 0.5 * math.log(df * math.pi)
    logp = c - np.log(scale) - ((df + 1.0) / 2.0) * np.log1p((z * z) / df)
    return float(np.sum(logp))


@dataclass
class DiagonalStudentTBOCPD:
    """Multivariate online change-point detector using BOCPD with a diagonal
    Student-t predictive density.

    Kept as the v0.8 fallback. New code paths should default to
    ``MultivariateNIWBOCPD``.
    """

    hazard: float = 1.0 / 48.0
    max_run: int = 96
    prior_var: float = 1.0
    min_prob: float = 1e-12

    def score(
        self,
        x: pd.DataFrame,
        *,
        nan_policy: NanPolicy = NanPolicy.NAN_TO_ZERO,
        column_policies: Mapping[str, NanPolicy] | None = None,
    ) -> pd.DataFrame:
        """Score the BOCPD posterior.

        v1.5 (PR-3 ASK-5/AF-8): the legacy ``ffill().fillna(0.0)``
        cleaner is now a per-column NaN policy. The default
        ``NanPolicy.NAN_TO_ZERO`` is bit-for-bit identical to the v1.4
        cleaner so existing macro fixtures keep passing.
        FI callers override to :attr:`NanPolicy.NAN_FAILS_PIT_AUDIT` so
        a missing CUSIP-level feature trips release_gate=false instead
        of silently zero-filling.
        """
        if x.empty:
            return pd.DataFrame(
                columns=[
                    "date",
                    "change_point_prob",
                    "bocpd_run_length_mean",
                    "bocpd_map_run_length",
                    "predictive_log_likelihood",
                ]
            )

        frame = clean_with_policy(
            x, default_policy=nan_policy, column_policies=column_policies
        ).astype(float)
        arr = frame.to_numpy(float)
        dim = arr.shape[1]

        prior = RunningDiagState.prior(dim, self.prior_var)
        states: list[RunningDiagState] = [prior]
        log_joint = np.array([0.0], dtype=float)
        rows = []

        h = min(max(float(self.hazard), self.min_prob), 1.0 - self.min_prob)
        log_h = math.log(h)
        log_1mh = math.log(1.0 - h)

        for i, date in enumerate(frame.index):
            xt = arr[i]
            pred_logs = np.array([_student_t_logpdf_diag(xt, st) for st in states], dtype=float)
            pred_norm = _logsumexp(log_joint + pred_logs)

            cp_log = _logsumexp(log_joint + pred_logs + log_h)
            growth_logs = log_joint + pred_logs + log_1mh

            new_log_joint = np.empty(min(len(growth_logs) + 1, self.max_run + 1), dtype=float)
            new_log_joint[0] = cp_log
            kept_growth = growth_logs[: self.max_run]
            new_log_joint[1 : 1 + len(kept_growth)] = kept_growth

            norm = _logsumexp(new_log_joint)
            new_log_joint = new_log_joint - norm
            probs = np.exp(new_log_joint)
            probs = probs / probs.sum()

            cp_prob = float(probs[0])
            run_lengths = np.arange(len(probs), dtype=float)
            rows.append(
                {
                    "date": date,
                    "change_point_prob": cp_prob,
                    "bocpd_run_length_mean": float(np.sum(run_lengths * probs)),
                    "bocpd_map_run_length": int(np.argmax(probs)),
                    "predictive_log_likelihood": float(pred_norm),
                }
            )

            new_states: list[RunningDiagState] = [prior.update(xt)]
            new_states.extend([st.update(xt) for st in states[: self.max_run]])
            states = new_states[: len(probs)]
            log_joint = np.log(np.maximum(probs, self.min_prob))

        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# multivariate NIW core
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NIWState:
    """Sufficient statistics of a Normal-Inverse-Wishart posterior.

    Hyperparameters follow Murphy 2007 ``Conjugate Bayesian analysis of the
    Gaussian distribution``::

        mu_0   prior mean (length d)
        kappa  prior strength on the mean
        nu     prior degrees of freedom (>= d for a proper prior)
        psi    prior scale matrix (d x d, SPD)
        n      number of observations absorbed so far
        sum_x  running sum of observations
        ssq    running sum of x x^T outer products

    All updates are O(d^2). The posterior predictive is multivariate Student-t.
    """

    n: int
    sum_x: np.ndarray
    ssq: np.ndarray
    mu0: np.ndarray
    kappa0: float
    nu0: float
    psi0: np.ndarray

    @classmethod
    def prior(cls, dim: int, *, kappa0: float = 1.0, nu0: float | None = None, psi_scale: float = 1.0) -> NIWState:
        eye = np.eye(dim, dtype=float)
        return cls(
            n=0,
            sum_x=np.zeros(dim, dtype=float),
            ssq=np.zeros((dim, dim), dtype=float),
            mu0=np.zeros(dim, dtype=float),
            kappa0=float(kappa0),
            nu0=float(nu0 if nu0 is not None else dim + 2.0),
            psi0=psi_scale * eye,
        )

    def update(self, x: np.ndarray) -> NIWState:
        x = np.asarray(x, dtype=float).reshape(-1)
        return NIWState(
            n=self.n + 1,
            sum_x=self.sum_x + x,
            ssq=self.ssq + np.outer(x, x),
            mu0=self.mu0,
            kappa0=self.kappa0,
            nu0=self.nu0,
            psi0=self.psi0,
        )

    def posterior(self) -> tuple[np.ndarray, float, float, np.ndarray]:
        """Return ``(mu_n, kappa_n, nu_n, psi_n)`` for the current sufficient stats."""
        n = self.n
        kappa_n = self.kappa0 + n
        nu_n = self.nu0 + n
        if n == 0:
            mu_n = self.mu0
            psi_n = self.psi0
            return mu_n, kappa_n, nu_n, psi_n
        xbar = self.sum_x / n
        mu_n = (self.kappa0 * self.mu0 + self.sum_x) / kappa_n
        # Within-cluster scatter S = sum (x_i - xbar)(x_i - xbar)^T = ssq - n xbar xbar^T
        S = self.ssq - n * np.outer(xbar, xbar)
        delta = (xbar - self.mu0).reshape(-1, 1)
        psi_n = self.psi0 + S + (self.kappa0 * n / kappa_n) * (delta @ delta.T)
        return mu_n, kappa_n, nu_n, psi_n

    def predictive_logpdf(self, x: np.ndarray, *, ridge: float = 1e-6) -> float:
        """Log of the multivariate Student-t posterior predictive at ``x``.

        Murphy 2007, Eq. 232. Uses Cholesky for stability.
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        d = x.shape[0]
        mu_n, kappa_n, nu_n, psi_n = self.posterior()
        df = nu_n - d + 1.0
        if df <= 0:
            df = max(df, 1.0)
        scale = psi_n * (kappa_n + 1.0) / (kappa_n * df)
        scale = scale + ridge * np.eye(d)
        try:
            L = np.linalg.cholesky(scale)
        except np.linalg.LinAlgError:
            scale = scale + (ridge * 100.0) * np.eye(d)
            try:
                L = np.linalg.cholesky(scale)
            except np.linalg.LinAlgError:
                scale = np.diag(np.maximum(np.diag(scale), ridge * 1000.0))
                L = np.linalg.cholesky(scale)
        log_det_scale = 2.0 * float(np.sum(np.log(np.diag(L))))
        diff = x - mu_n
        # Cholesky-solve y = L^{-1} diff, so quad = ||y||^2 = diff^T scale^{-1} diff
        z = _cholesky_solve_lower(L, diff)
        quad = float(np.dot(z, z))
        log_norm = (
            math.lgamma((df + d) / 2.0) - math.lgamma(df / 2.0) - 0.5 * (d * math.log(df * math.pi) + log_det_scale)
        )
        return log_norm - 0.5 * (df + d) * math.log1p(quad / df)


def _cholesky_solve_lower(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve ``L y = b`` for a lower-triangular ``L``.

    Uses scipy.linalg.solve_triangular when available, otherwise a numpy
    forward-substitution loop. We do not pull scipy as a hard dependency since
    the rest of the engine is sklearn/pandas/numpy only, but we use it when
    present because it is roughly an order of magnitude faster.
    """
    try:  # pragma: no cover - import speed not worth measuring
        from scipy.linalg import solve_triangular

        return solve_triangular(L, b, lower=True)
    except Exception:
        b = np.asarray(b, dtype=float).reshape(-1)
        n = b.shape[0]
        y = np.zeros_like(b)
        for i in range(n):
            s = b[i] - np.dot(L[i, :i], y[:i])
            y[i] = s / L[i, i]
        return y


@dataclass
class MultivariateNIWBOCPD:
    """Online Bayesian change-point detector with a NIW conjugate prior.

    Parameters
    ----------
    hazard:
        Constant per-step changepoint hazard. ``learned_constant_hazard`` can
        seed this from an empirical regime path.
    max_run:
        Maximum retained run length. The run-length posterior is truncated at
        this depth to keep inference O(max_run * d^2) per step.
    prior_kappa:
        Prior strength on the mean. Higher = stronger pull to ``mu0=0``. The
        default of 1.0 makes the prior nearly uninformative once a few samples
        accumulate.
    prior_nu_offset:
        Prior degrees of freedom offset above ``d``; the actual ``nu0`` is
        ``d + prior_nu_offset``. Must be > 0 for a proper prior.
    prior_psi_scale:
        Prior scale-matrix multiplier on the identity. Larger values mean wider
        prior uncertainty about Σ.
    min_prob:
        Numerical floor for the run-length posterior.
    """

    hazard: float = 1.0 / 48.0
    max_run: int = 96
    prior_kappa: float = 1.0
    prior_nu_offset: float = 2.0
    prior_psi_scale: float = 1.0
    min_prob: float = 1e-12

    def score(
        self,
        x: pd.DataFrame,
        *,
        nan_policy: NanPolicy = NanPolicy.NAN_TO_ZERO,
        column_policies: Mapping[str, NanPolicy] | None = None,
    ) -> pd.DataFrame:
        """Score the NIW BOCPD posterior.

        v1.5 (PR-3 ASK-5/AF-8): same ``nan_policy`` plumbing as
        :class:`DiagonalStudentTBOCPD`. Default ``NAN_TO_ZERO``
        preserves v1.4 numerics; FI callers pass
        :attr:`NanPolicy.NAN_FAILS_PIT_AUDIT`.
        """
        if x.empty:
            return pd.DataFrame(
                columns=[
                    "date",
                    "change_point_prob",
                    "bocpd_run_length_mean",
                    "bocpd_map_run_length",
                    "predictive_log_likelihood",
                ]
            )

        frame = clean_with_policy(
            x, default_policy=nan_policy, column_policies=column_policies
        ).astype(float)
        arr = frame.to_numpy(float)
        dim = arr.shape[1]

        prior = NIWState.prior(
            dim,
            kappa0=self.prior_kappa,
            nu0=dim + self.prior_nu_offset,
            psi_scale=self.prior_psi_scale,
        )
        states: list[NIWState] = [prior]
        log_joint = np.array([0.0], dtype=float)
        rows: list[dict] = []

        h = min(max(float(self.hazard), self.min_prob), 1.0 - self.min_prob)
        log_h = math.log(h)
        log_1mh = math.log(1.0 - h)

        for i, date in enumerate(frame.index):
            xt = arr[i]
            pred_logs = np.array([st.predictive_logpdf(xt) for st in states], dtype=float)
            pred_norm = _logsumexp(log_joint + pred_logs)

            cp_log = _logsumexp(log_joint + pred_logs + log_h)
            growth_logs = log_joint + pred_logs + log_1mh

            new_log_joint = np.empty(min(len(growth_logs) + 1, self.max_run + 1), dtype=float)
            new_log_joint[0] = cp_log
            kept_growth = growth_logs[: self.max_run]
            new_log_joint[1 : 1 + len(kept_growth)] = kept_growth

            norm = _logsumexp(new_log_joint)
            new_log_joint = new_log_joint - norm
            probs = np.exp(new_log_joint)
            probs = probs / probs.sum()

            cp_prob = float(probs[0])
            run_lengths = np.arange(len(probs), dtype=float)
            rows.append(
                {
                    "date": date,
                    "change_point_prob": cp_prob,
                    "bocpd_run_length_mean": float(np.sum(run_lengths * probs)),
                    "bocpd_map_run_length": int(np.argmax(probs)),
                    "predictive_log_likelihood": float(pred_norm),
                }
            )

            new_states: list[NIWState] = [prior.update(xt)]
            new_states.extend([st.update(xt) for st in states[: self.max_run]])
            states = new_states[: len(probs)]
            log_joint = np.log(np.maximum(probs, self.min_prob))

        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# hazard learning helpers
# ---------------------------------------------------------------------------


def learned_constant_hazard(states: pd.Series | Iterable, *, min_runs: int = 3, fallback: float = 1.0 / 48.0) -> float:
    """Estimate a constant per-step BOCPD hazard from an observed regime path.

    The hazard is approximated as ``1 / mean_run_length`` of contiguous identical
    states. When the path is too short to support an estimate, ``fallback`` is
    returned. The estimate is clipped to ``[1e-4, 0.5]`` so it remains a sane
    BOCPD hazard.
    """
    seq = list(states)
    if len(seq) < 2:
        return float(fallback)
    runs: list[int] = []
    cur = 1
    for prev, nxt in itertools.pairwise(seq):
        if prev == nxt:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    if len(runs) < min_runs:
        return float(fallback)
    mean_len = float(np.mean(runs))
    if mean_len <= 1.0:
        return float(np.clip(1.0 - 1e-3, 1e-4, 0.5))
    h = 1.0 / mean_len
    return float(np.clip(h, 1e-4, 0.5))


__all__ = [
    "DiagonalStudentTBOCPD",
    "MultivariateNIWBOCPD",
    "NIWState",
    "RunningDiagState",
    "learned_constant_hazard",
]
