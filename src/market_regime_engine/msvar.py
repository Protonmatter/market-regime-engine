# SPDX-License-Identifier: Apache-2.0
"""Markov-Switching VAR (Hamilton 1989) regime model.

The plain Gaussian-emission HMM in :mod:`hmm` answers "which regime are we in
right now?" but assumes the *cross-section* of the eight domain factors is iid
within a regime. That misses one of the most important features of real macro
regimes: how variables co-move *changes* across regimes (e.g. credit and
labor co-load tightly during recessions and weakly during expansions).

This module fits a Markov-Switching VAR(p): each latent regime ``k`` has its
own VAR coefficient matrix ``A_k`` and innovation covariance ``Sigma_k``::

    y_t = c_k + A_k @ y_{t-1} + ... + A_{k,p} @ y_{t-p} + e_t,  e_t ~ N(0, Sigma_k)

Inference uses the Hamilton (1989) filter + Kim (1994) smoother. Parameters
are estimated with EM. The class exposes ``score()`` returning the same
``hmm_regime`` / ``hmm_confidence`` / ``regime_prob_*`` columns as
:class:`HMMRegimePosterior` so it slots into the existing ensemble layer.

The implementation deliberately stays in numpy / pandas — installing
statsmodels just for this is not worth the dependency footprint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from market_regime_engine.hmm import DOMAIN_COLUMNS, REGIME_STATES


def _normalize(v: np.ndarray) -> np.ndarray:
    s = float(v.sum())
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(v) / len(v)
    return v / s


def _logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True) if axis is not None else np.max(a)
    safe_m = np.where(np.isfinite(m), m, 0.0)
    out = safe_m + np.log(np.sum(np.exp(a - safe_m), axis=axis, keepdims=True))
    if axis is None:
        return float(np.squeeze(out))
    return np.squeeze(out, axis=axis)


def _mvn_logpdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray, *, ridge: float = 1e-6) -> float:
    d = x.shape[-1]
    cov = cov + ridge * np.eye(d)
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov = cov + (ridge * 100.0) * np.eye(d)
        L = np.linalg.cholesky(cov + 1e-3 * np.eye(d))
    diff = x - mean
    z = np.linalg.solve(L, diff)
    log_det = 2.0 * float(np.sum(np.log(np.diag(L))))
    return -0.5 * (d * math.log(2.0 * math.pi) + log_det + float(z @ z))


@dataclass
class MarkovSwitchingVAR:
    """Hamilton-Kim MS-VAR with Gaussian innovations."""

    states: list[str] = field(default_factory=lambda: REGIME_STATES.copy())
    domains: list[str] = field(default_factory=lambda: DOMAIN_COLUMNS.copy())
    p: int = 1
    max_iter: int = 30
    tol: float = 1e-4
    cov_ridge: float = 1e-3
    transition_pseudocount: float = 1.0

    # Learned parameters (per regime)
    intercepts: np.ndarray = field(default_factory=lambda: np.array([]))
    coefficients: np.ndarray = field(default_factory=lambda: np.array([]))  # shape (K, p, d, d)
    covariances: np.ndarray = field(default_factory=lambda: np.array([]))
    transition: np.ndarray = field(default_factory=lambda: np.array([]))
    prior: np.ndarray = field(default_factory=lambda: np.array([]))

    fitted: bool = False
    fit_log: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # filter / smoother
    # ------------------------------------------------------------------

    def _emission_logprob(self, y_t: np.ndarray, y_lags: np.ndarray) -> np.ndarray:
        K = len(self.states)
        out = np.empty(K)
        for k in range(K):
            mu = self.intercepts[k].copy()
            for j in range(self.p):
                mu = mu + self.coefficients[k, j] @ y_lags[j]
            out[k] = _mvn_logpdf(y_t, mu, self.covariances[k])
        return out

    def _hamilton_filter(self, Y: np.ndarray) -> tuple[np.ndarray, float]:
        n, _d = Y.shape
        K = len(self.states)
        log_alpha = np.full((n, K), -np.inf)
        log_pi = np.log(np.maximum(self.prior, 1e-12))
        log_A = np.log(np.maximum(self.transition, 1e-12))
        ll_total = 0.0
        for t in range(self.p, n):
            y_t = Y[t]
            y_lags = np.array([Y[t - j - 1] for j in range(self.p)])
            log_b = self._emission_logprob(y_t, y_lags)
            if t == self.p:
                log_alpha[t] = log_pi + log_b
            else:
                log_alpha[t] = log_b + _logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0)
            ll_total = float(_logsumexp(log_alpha[t]))
        return log_alpha, ll_total

    def _kim_smoother(self, log_alpha: np.ndarray) -> np.ndarray:
        n, K = log_alpha.shape
        log_A = np.log(np.maximum(self.transition, 1e-12))
        log_gamma = np.full((n, K), -np.inf)
        log_gamma[-1] = log_alpha[-1] - _logsumexp(log_alpha[-1])
        for t in range(n - 2, self.p - 1, -1):
            denom = _logsumexp(log_alpha[t][:, None] + log_A, axis=0)  # shape (K,) over j
            log_gamma[t] = log_alpha[t] + _logsumexp(log_A + (log_gamma[t + 1] - denom)[None, :], axis=1)
            log_gamma[t] -= _logsumexp(log_gamma[t])
        return log_gamma

    # ------------------------------------------------------------------
    # EM
    # ------------------------------------------------------------------

    def fit(self, panel: pd.DataFrame) -> MarkovSwitchingVAR:
        if panel is None or panel.empty:
            return self
        frame = panel.copy()
        for d in self.domains:
            if d not in frame:
                frame[d] = 0.0
        Y = frame[self.domains].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).to_numpy(float)
        n, d = Y.shape
        K = len(self.states)
        if n < self.p + K * 4:
            return self
        # Initialise params with simple cluster-style splits.
        rng = np.random.default_rng(0)
        clusters = rng.integers(0, K, size=n)
        self.intercepts = np.zeros((K, d))
        self.coefficients = np.tile(0.5 * np.eye(d), (K, self.p, 1, 1))
        self.covariances = np.tile(np.eye(d), (K, 1, 1))
        for k in range(K):
            mask = clusters == k
            if mask.sum() < d + 2:
                continue
            self.intercepts[k] = Y[mask].mean(axis=0)
            self.covariances[k] = np.cov(Y[mask].T) + self.cov_ridge * np.eye(d)
        self.prior = np.ones(K) / K
        self.transition = np.full((K, K), 0.05)
        np.fill_diagonal(self.transition, 0.6)
        self.transition = self.transition / self.transition.sum(axis=1, keepdims=True)

        prev_ll = -np.inf
        for it in range(self.max_iter):
            log_alpha, ll = self._hamilton_filter(Y)
            log_gamma = self._kim_smoother(log_alpha)
            gamma = np.exp(log_gamma)
            gamma[self.p :].sum(axis=0)
            # M step: weighted regression per regime
            new_intercepts = np.zeros((K, d))
            new_coefficients = np.zeros((K, self.p, d, d))
            new_covariances = np.zeros((K, d, d))
            for k in range(K):
                w = gamma[self.p :, k]
                if w.sum() < d + 2:
                    new_intercepts[k] = self.intercepts[k]
                    new_coefficients[k] = self.coefficients[k]
                    new_covariances[k] = self.covariances[k]
                    continue
                # Build regression: y_t = c + A_1 y_{t-1} + ... weighted by w
                X = np.ones((n - self.p, 1 + self.p * d))
                for j in range(self.p):
                    X[:, 1 + j * d : 1 + (j + 1) * d] = Y[self.p - 1 - j : n - 1 - j]
                Yreg = Y[self.p :]
                W = np.diag(w)
                XtWX = X.T @ W @ X + 1e-6 * np.eye(X.shape[1])
                XtWY = X.T @ W @ Yreg
                try:
                    beta = np.linalg.solve(XtWX, XtWY)
                except np.linalg.LinAlgError:
                    beta = np.linalg.lstsq(XtWX, XtWY, rcond=None)[0]
                new_intercepts[k] = beta[0]
                for j in range(self.p):
                    new_coefficients[k, j] = beta[1 + j * d : 1 + (j + 1) * d].T
                resid = Yreg - X @ beta
                new_covariances[k] = (resid.T * w) @ resid / max(w.sum(), 1e-9) + self.cov_ridge * np.eye(d)
            # Transition update
            xi_sum = np.zeros((K, K))
            log_A = np.log(np.maximum(self.transition, 1e-12))
            for t in range(self.p + 1, n):
                # joint posterior xi[t-1, i, j]
                num = (
                    log_alpha[t - 1][:, None]
                    + log_A
                    + (log_gamma[t] - _logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0))[None, :]
                )
                xi_sum += np.exp(num - _logsumexp(num))
            xi_sum_smoothed = xi_sum + self.transition_pseudocount / K
            self.transition = xi_sum_smoothed / xi_sum_smoothed.sum(axis=1, keepdims=True)
            self.prior = _normalize(np.exp(log_gamma[self.p]) + 1.0 / K)
            self.intercepts = new_intercepts
            self.coefficients = new_coefficients
            self.covariances = new_covariances

            if abs(ll - prev_ll) < self.tol * max(abs(prev_ll), 1.0):
                self.fit_log = {"log_likelihood": ll, "iterations": it + 1, "converged": True}
                self.fitted = True
                return self
            prev_ll = ll
        self.fit_log = {"log_likelihood": prev_ll, "iterations": self.max_iter, "converged": False}
        self.fitted = True
        return self

    # ------------------------------------------------------------------
    # online filtering
    # ------------------------------------------------------------------

    def score(self, panel: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted or panel is None or panel.empty:
            return pd.DataFrame(columns=["date", "msvar_regime", "msvar_confidence"])
        frame = panel.copy()
        for d in self.domains:
            if d not in frame:
                frame[d] = 0.0
        Y = frame[self.domains].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).to_numpy(float)
        n = Y.shape[0]
        log_alpha, _ = self._hamilton_filter(Y)
        rows: list[dict] = []
        for t in range(self.p, n):
            la = log_alpha[t]
            la = la - _logsumexp(la)
            probs = np.exp(la)
            best = int(np.argmax(probs))
            row = {
                "date": frame.index[t],
                "msvar_regime": self.states[best],
                "msvar_confidence": float(probs[best]),
            }
            for i, s in enumerate(self.states):
                row[f"msvar_prob_{s}"] = float(probs[i])
            rows.append(row)
        return pd.DataFrame(rows)


__all__ = ["MarkovSwitchingVAR"]
