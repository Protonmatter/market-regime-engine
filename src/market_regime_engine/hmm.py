# SPDX-License-Identifier: Apache-2.0
"""Hidden Markov Model regime posterior.

Two operating modes are supported:

1. **Hand-prior mode** (the v0.8 behavior). Centroids and transitions come from
   ``DEFAULT_CENTROIDS`` and a hand-set ``stay_prob``. ``score()`` runs a
   filtered forward pass for the regime posterior. This stays as the
   zero-warmup default so callers without enough history still get an answer.
2. **EM-fitted mode**. ``fit(panel)`` runs Baum-Welch (Rabiner 1989) with
   full-covariance Gaussian emissions on a domain-score panel and replaces the
   centroids, emission covariances, and transition matrix with learned
   estimates. After fitting, each learned state is **label-pinned** back to the
   nearest hand-prior centroid by exact minimum-cost assignment so that the
   reporting names (``risk_on_expansion``, ``stagflation``, ...) remain stable
   downstream. ``score()`` then runs a forward pass with the learned emissions.

The pinning step is the bridge between unsupervised state learning and the
named-regime contract that the rest of the engine, the WFST decoder, and the
report writer all rely on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import permutations
from typing import overload

import numpy as np
import pandas as pd

REGIME_STATES = [
    "risk_on_expansion",
    "late_cycle",
    "soft_landing",
    "sticky_inflation",
    "credit_stress",
    "energy_shock",
    "stagflation",
    "recessionary_bear",
    "liquidity_meltup",
]

DOMAIN_COLUMNS = ["labor", "rates", "inflation", "credit", "housing", "energy", "fx", "fiscal"]

DEFAULT_CENTROIDS: dict[str, dict[str, float]] = {
    "risk_on_expansion": {
        "labor": -0.2,
        "rates": 0.2,
        "inflation": 0.2,
        "credit": 0.2,
        "housing": -0.2,
        "energy": 0.1,
        "fx": 0.1,
        "fiscal": 0.2,
    },
    "late_cycle": {
        "labor": 0.2,
        "rates": 1.0,
        "inflation": 0.8,
        "credit": 0.5,
        "housing": 0.4,
        "energy": 0.4,
        "fx": 0.3,
        "fiscal": 0.5,
    },
    "soft_landing": {
        "labor": 0.2,
        "rates": 0.7,
        "inflation": 0.4,
        "credit": 0.3,
        "housing": 0.2,
        "energy": 0.1,
        "fx": 0.1,
        "fiscal": 0.4,
    },
    "sticky_inflation": {
        "labor": 0.1,
        "rates": 1.4,
        "inflation": 1.6,
        "credit": 0.4,
        "housing": 0.4,
        "energy": 0.5,
        "fx": 0.4,
        "fiscal": 0.6,
    },
    "credit_stress": {
        "labor": 0.7,
        "rates": 0.9,
        "inflation": 0.4,
        "credit": 1.8,
        "housing": 1.1,
        "energy": 0.3,
        "fx": 0.7,
        "fiscal": 0.7,
    },
    "energy_shock": {
        "labor": 0.3,
        "rates": 1.0,
        "inflation": 1.4,
        "credit": 0.5,
        "housing": 0.4,
        "energy": 2.0,
        "fx": 0.5,
        "fiscal": 0.5,
    },
    "stagflation": {
        "labor": 1.0,
        "rates": 1.2,
        "inflation": 1.7,
        "credit": 0.8,
        "housing": 0.8,
        "energy": 1.2,
        "fx": 0.6,
        "fiscal": 0.8,
    },
    "recessionary_bear": {
        "labor": 1.6,
        "rates": 0.8,
        "inflation": 0.5,
        "credit": 1.6,
        "housing": 1.4,
        "energy": 0.6,
        "fx": 0.7,
        "fiscal": 0.9,
    },
    "liquidity_meltup": {
        "labor": -0.3,
        "rates": -0.2,
        "inflation": 0.2,
        "credit": -0.2,
        "housing": -0.1,
        "energy": 0.1,
        "fx": -0.2,
        "fiscal": 0.3,
    },
}


def _normalize(v: np.ndarray) -> np.ndarray:
    s = float(v.sum())
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(v) / len(v)
    return v / s


@overload
def _logsumexp(a: np.ndarray, axis: None = None) -> float: ...


@overload
def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray: ...


def _logsumexp(a: np.ndarray, axis: int | None = None) -> float | np.ndarray:
    """Numerically stable ``log(sum(exp(a)))``.

    v1.6.0 (REVIEW_DEEP_V1_5_2.md §4.2): typed as overloaded so callers
    that pass ``axis=None`` (the scalar branch) receive ``float`` and
    callers that pass ``axis=int`` (the array branch) receive
    ``np.ndarray``. Without the overloads the union return type leaks
    into every consumer (Hamilton filter, Kim smoother) and triggers
    cascading mypy ``return-value`` errors.
    """
    if axis is None:
        a = np.asarray(a, dtype=float).ravel()
        m = float(np.max(a)) if a.size else float("-inf")
        if not np.isfinite(m):
            return float(m)
        return float(m + np.log(np.sum(np.exp(a - m))))
    m = np.max(a, axis=axis, keepdims=True)
    safe_m = np.where(np.isfinite(m), m, 0.0)
    out = safe_m + np.log(np.sum(np.exp(a - safe_m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def _optimal_assignment(cost: np.ndarray) -> list[int]:
    """Return the minimum-cost one-to-one assignment for a square cost matrix.

    ``assignment[i] = j`` means row ``i`` is assigned to column ``j``.
    The implementation uses SciPy's linear-sum assignment when available and
    otherwise falls back to an exact brute-force solver. The fallback is
    acceptable here because regime counts are small (K <= 9 in the production
    configuration), while exactness matters for stable regime naming.
    """
    arr = np.asarray(cost, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("assignment cost matrix must be square")
    K = int(arr.shape[0])
    if K == 0:
        return []
    try:  # pragma: no cover - optional dependency branch
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(arr)
        assignment = [-1] * K
        for r, c in zip(rows, cols, strict=True):
            assignment[int(r)] = int(c)
        return assignment
    except Exception:
        best_perm: tuple[int, ...] | None = None
        best_cost = float("inf")
        for perm in permutations(range(K)):
            total = float(sum(arr[i, perm[i]] for i in range(K)))
            if total < best_cost:
                best_cost = total
                best_perm = perm
        if best_perm is None:  # defensive; only possible for malformed input
            return list(range(K))
        return [int(x) for x in best_perm]


def _mvn_logpdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray, *, ridge: float = 1e-6) -> float:
    """Multivariate normal log density. Uses Cholesky for stability."""
    d = x.shape[-1]
    cov = cov + ridge * np.eye(d)
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov = cov + (ridge * 100.0) * np.eye(d)
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            cov = np.diag(np.maximum(np.diag(cov), ridge * 1000.0))
            L = np.linalg.cholesky(cov)
    diff = x - mean
    z = np.linalg.solve(L, diff)
    log_det = 2.0 * float(np.sum(np.log(np.diag(L))))
    return -0.5 * (d * math.log(2.0 * math.pi) + log_det + float(np.dot(z, z)))


def _mvn_logpdf_batch(X: np.ndarray, mean: np.ndarray, cov: np.ndarray, *, ridge: float = 1e-6) -> np.ndarray:
    _n, d = X.shape
    cov = cov + ridge * np.eye(d)
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov = cov + (ridge * 100.0) * np.eye(d)
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            cov = np.diag(np.maximum(np.diag(cov), ridge * 1000.0))
            L = np.linalg.cholesky(cov)
    diff = X - mean
    z = np.linalg.solve(L, diff.T).T
    log_det = 2.0 * float(np.sum(np.log(np.diag(L))))
    quad = np.einsum("ij,ij->i", z, z)
    return -0.5 * (d * math.log(2.0 * math.pi) + log_det + quad)


@dataclass
class HMMRegimePosterior:
    states: list[str] = field(default_factory=lambda: REGIME_STATES.copy())
    emission_scale: float = 0.9
    stay_prob: float = 0.82
    min_prob: float = 1e-12
    fitted: bool = False

    def __post_init__(self) -> None:
        self.domains = DOMAIN_COLUMNS.copy()
        # Hand-prior centroids: shape (K, D). When fit() runs, these and
        # ``self.covariances`` are replaced by the learned values, but the
        # ordering of self.states (and therefore the labels) is preserved.
        self.centroids = np.array(
            [[DEFAULT_CENTROIDS[s].get(d, 0.0) for d in self.domains] for s in self.states], dtype=float
        )
        K, D = self.centroids.shape
        # Hand-prior covariances are isotropic at emission_scale^2 * I.
        self.covariances = np.tile((self.emission_scale**2) * np.eye(D)[None, :, :], (K, 1, 1))
        self.transition = self._build_transition()
        self.prior = np.ones(len(self.states), dtype=float) / len(self.states)
        self.prior[self.states.index("risk_on_expansion")] += 0.15
        self.prior = _normalize(self.prior)
        # Original priors retained for label-pinning even after fit().
        self._prior_centroids = self.centroids.copy()
        self.fit_log: dict[str, float] = {}

    def _build_transition(self) -> np.ndarray:
        n = len(self.states)
        mat = np.full((n, n), (1.0 - self.stay_prob) / (n - 1), dtype=float)
        np.fill_diagonal(mat, self.stay_prob)

        preferred = {
            "risk_on_expansion": ["late_cycle", "soft_landing", "liquidity_meltup"],
            "late_cycle": ["sticky_inflation", "soft_landing", "credit_stress"],
            "sticky_inflation": ["energy_shock", "stagflation", "recessionary_bear"],
            "credit_stress": ["recessionary_bear"],
            "energy_shock": ["stagflation", "recessionary_bear"],
            "stagflation": ["recessionary_bear", "soft_landing"],
            "recessionary_bear": ["soft_landing"],
            "soft_landing": ["risk_on_expansion", "late_cycle"],
            "liquidity_meltup": ["late_cycle", "risk_on_expansion"],
        }
        for src, dsts in preferred.items():
            i = self.states.index(src)
            for dst in dsts:
                j = self.states.index(dst)
                mat[i, j] += 0.08 / max(len(dsts), 1)
            mat[i, :] = _normalize(mat[i, :])
        return mat

    # ------------------------------------------------------------------
    # Hand-prior emission (used when fitted == False; matches v0.8 output)
    # ------------------------------------------------------------------

    def _emission_logprob_handprior(self, x: np.ndarray) -> np.ndarray:
        diff = self.centroids - x.reshape(1, -1)
        sq = np.sum((diff / self.emission_scale) ** 2, axis=1)
        return -0.5 * sq - len(self.domains) * math.log(max(self.emission_scale, 1e-6))

    # ------------------------------------------------------------------
    # Fitted full-covariance Gaussian emission
    # ------------------------------------------------------------------

    def _emission_logprob_fitted(self, x: np.ndarray) -> np.ndarray:
        K = len(self.states)
        out = np.empty(K, dtype=float)
        for k in range(K):
            out[k] = _mvn_logpdf(x, self.centroids[k], self.covariances[k])
        return out

    def _emission_logprob_batch(self, X: np.ndarray) -> np.ndarray:
        K = len(self.states)
        if not self.fitted:
            n = X.shape[0]
            out = np.empty((n, K), dtype=float)
            for i in range(n):
                out[i] = self._emission_logprob_handprior(X[i])
            return out
        out = np.empty((X.shape[0], K), dtype=float)
        for k in range(K):
            out[:, k] = _mvn_logpdf_batch(X, self.centroids[k], self.covariances[k])
        return out

    # ------------------------------------------------------------------
    # Baum-Welch
    # ------------------------------------------------------------------

    def fit(
        self,
        panel: pd.DataFrame,
        *,
        max_iter: int = 50,
        tol: float = 1e-4,
        cov_ridge: float = 1e-3,
        prior_pseudocount: float = 1.0,
        transition_pseudocount: float = 1.0,
    ) -> HMMRegimePosterior:
        """Fit emissions and transitions on a domain-score panel via Baum-Welch.

        The hand-prior centroids and transition matrix are used as starting
        points so EM converges to a local optimum that is recognizable; the
        post-fit pinning step then re-aligns the learned state ordering to the
        named regime ordering by an exact minimum-cost assignment.

        Parameters
        ----------
        panel:
            Wide-format dataframe indexed by date with one column per element of
            ``self.domains``. Missing columns are filled with 0.
        max_iter:
            Cap on EM iterations.
        tol:
            Convergence tolerance on relative log-likelihood improvement.
        cov_ridge:
            Diagonal regularization added to the M-step covariance to keep it
            SPD even when a state captures few samples.
        prior_pseudocount, transition_pseudocount:
            Dirichlet pseudocounts (Laplace smoothing) on the M-step initial
            distribution and transition matrix. They keep all states reachable
            in subsequent decoding even if EM never visits them.

        Returns
        -------
        self, with ``self.centroids``, ``self.covariances``, ``self.transition``,
        ``self.prior`` updated and ``self.fitted = True``.
        """
        if panel is None or panel.empty:
            return self
        frame = panel.copy()
        for d in self.domains:
            if d not in frame:
                frame[d] = 0.0
        frame = frame[self.domains].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
        X = frame.to_numpy(float)
        n, d = X.shape
        K = len(self.states)
        if n < K * 2:
            return self

        log_pi = np.log(np.maximum(self.prior, 1e-12))
        log_A = np.log(np.maximum(self.transition, 1e-12))

        prev_ll = -np.inf
        for it in range(max_iter):
            log_b = self._emission_logprob_batch(X)  # (n, K)

            # ---- forward pass (log-space) ----
            log_alpha = np.full((n, K), -np.inf)
            log_alpha[0] = log_pi + log_b[0]
            for t in range(1, n):
                # log_alpha[t, j] = log_b[t, j] + logsumexp_i (log_alpha[t-1, i] + log_A[i, j])
                m = log_alpha[t - 1][:, None] + log_A
                log_alpha[t] = log_b[t] + _logsumexp(m, axis=0)

            ll = float(_logsumexp(log_alpha[-1]))
            if not np.isfinite(ll):
                break

            # ---- backward pass (log-space) ----
            log_beta = np.full((n, K), -np.inf)
            log_beta[-1] = 0.0
            for t in range(n - 2, -1, -1):
                m = log_A + (log_b[t + 1] + log_beta[t + 1])[None, :]
                log_beta[t] = _logsumexp(m, axis=1)

            # ---- posterior gamma and xi ----
            log_gamma = log_alpha + log_beta
            log_gamma -= _logsumexp(log_gamma, axis=1)[:, None]
            gamma = np.exp(log_gamma)

            # xi summed over t (n-1, K, K)
            log_xi_sum = np.full((K, K), -np.inf)
            denom = ll
            for t in range(n - 1):
                # log_xi[t, i, j] = log_alpha[t, i] + log_A[i, j] + log_b[t+1, j] + log_beta[t+1, j] - ll
                term = log_alpha[t][:, None] + log_A + (log_b[t + 1] + log_beta[t + 1])[None, :] - denom
                # accumulate in log-space
                m = np.maximum(log_xi_sum, term)
                log_xi_sum = m + np.log(np.exp(log_xi_sum - m) + np.exp(term - m))

            # ---- M step ----
            new_prior = gamma[0] + prior_pseudocount / K
            new_prior = _normalize(new_prior)
            log_pi = np.log(np.maximum(new_prior, 1e-12))

            xi_sum = np.exp(log_xi_sum)
            xi_sum_smoothed = xi_sum + transition_pseudocount / K
            new_A = xi_sum_smoothed / xi_sum_smoothed.sum(axis=1, keepdims=True)
            log_A = np.log(np.maximum(new_A, 1e-12))

            gamma_sum = gamma.sum(axis=0)
            new_centroids = np.empty_like(self.centroids)
            new_covs = np.empty_like(self.covariances)
            eye = np.eye(d) * cov_ridge
            for k in range(K):
                gk = gamma[:, k]
                wk = float(max(gamma_sum[k], 1e-9))
                mu = (gk[:, None] * X).sum(axis=0) / wk
                diff = X - mu
                cov = (gk[:, None, None] * diff[:, :, None] * diff[:, None, :]).sum(axis=0) / wk
                new_centroids[k] = mu
                new_covs[k] = cov + eye

            self.centroids = new_centroids
            self.covariances = new_covs
            self.transition = new_A
            self.prior = new_prior

            if abs(ll - prev_ll) < tol * max(abs(prev_ll), 1.0):
                self.fit_log = {"log_likelihood": ll, "iterations": it + 1, "converged": True}
                self.fitted = True
                self._pin_to_handprior_labels()
                return self
            prev_ll = ll

        self.fit_log = {"log_likelihood": prev_ll, "iterations": max_iter, "converged": False}
        self.fitted = True
        self._pin_to_handprior_labels()
        return self

    # ------------------------------------------------------------------
    # Label pinning
    # ------------------------------------------------------------------

    def _pin_to_handprior_labels(self) -> None:
        """Reorder learned states so that index ``i`` is the closest learned
        match to the ``self.states[i]`` hand-prior centroid.

        Uses an exact minimum-cost one-to-one assignment. Greedy matching is
        not equivalent to the Hungarian / linear-sum assignment problem and can
        mislabel regimes when distances are close or partially symmetric.

        After this step, ``self.centroids[i]`` is the learned mean of the
        regime named ``self.states[i]``, ``self.transition[i, j]`` is the
        learned transition from regime i to regime j, etc.
        """
        # cost[i, k] = ||prior_centroid_i - learned_centroid_k||
        diffs = self._prior_centroids[:, None, :] - self.centroids[None, :, :]
        cost = np.linalg.norm(diffs, axis=2)
        order = _optimal_assignment(cost)
        self.centroids = self.centroids[order]
        self.covariances = self.covariances[order]
        # Reorder transition rows and columns.
        self.transition = self.transition[np.ix_(order, order)]
        self.prior = self.prior[order]

    # ------------------------------------------------------------------
    # Online filtering (used by score)
    # ------------------------------------------------------------------

    def _emission_logprob(self, x: np.ndarray) -> np.ndarray:
        if self.fitted:
            return self._emission_logprob_fitted(x)
        return self._emission_logprob_handprior(x)

    def score(self, domain_score_pivot: pd.DataFrame) -> pd.DataFrame:
        if domain_score_pivot.empty:
            return pd.DataFrame(columns=["date", "hmm_regime", "hmm_confidence"])
        frame = domain_score_pivot.copy()
        for d in self.domains:
            if d not in frame:
                frame[d] = 0.0
        frame = frame[self.domains].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

        alpha = self.prior.copy()
        rows = []
        for date, row in frame.iterrows():
            x = row.to_numpy(float)
            loge = self._emission_logprob(x)
            emit = np.exp(loge - np.max(loge))
            alpha = _normalize((alpha @ self.transition) * emit)
            best_idx = int(np.argmax(alpha))
            out = {
                "date": date,
                "hmm_regime": self.states[best_idx],
                "hmm_confidence": float(alpha[best_idx]),
            }
            out.update({f"regime_prob_{s}": float(alpha[i]) for i, s in enumerate(self.states)})
            rows.append(out)
        return pd.DataFrame(rows)


__all__ = [
    "DEFAULT_CENTROIDS",
    "DOMAIN_COLUMNS",
    "REGIME_STATES",
    "HMMRegimePosterior",
]
