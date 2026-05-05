"""Dynamic factor model for domain stress scores.

The legacy ``regimes.domain_scores`` collapses a domain's transforms into a
single number using hand-tuned coefficients (e.g.
``25 * CPIAUCSL.log_yoy + 30 * CPILFESL.log_yoy``). That is opaque to model
risk review and impossible to validate. The replacement is a small
state-space model:

    f_t  = phi * f_{t-1} + eta_t,   eta_t ~ N(0, sigma_eta^2)        # latent factor
    y_kt = lambda_k * f_t + eps_kt, eps_kt ~ N(0, sigma_k^2)         # observed transforms

Each domain gets its own DFM (one factor) fit with EM (Watson-Engle 1983) on
the *standardized* domain transforms. The factor loadings ``lambda_k`` are
identified up to sign by anchoring the dominant transform's loading positive.

This module deliberately implements EM in numpy / pandas without pulling in
statsmodels. Production deployments that need MQ-mixed-frequency or factor
regularization can swap in ``statsmodels.tsa.DynamicFactorMQ`` behind the same
``DFMDomainModel`` interface; the engine's downstream code only uses the
factor scores.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DFMDomainModel:
    """Single-factor Watson-Engle DFM on a domain's standardized transforms.

    Parameters
    ----------
    max_iter:
        EM iteration cap.
    tol:
        Relative log-likelihood tolerance for convergence.
    ridge:
        Diagonal regularization on the observation covariance during the
        M-step (keeps the model identifiable when one transform dominates).
    """

    max_iter: int = 50
    tol: float = 1e-4
    ridge: float = 1e-4

    # Learned parameters
    phi: float = 0.0
    sigma_eta2: float = 1.0
    loadings: np.ndarray = field(default_factory=lambda: np.array([]))
    sigma_eps2: np.ndarray = field(default_factory=lambda: np.array([]))
    columns: list[str] = field(default_factory=list)
    # Training-time standardization stats. Cached so transform() reuses the
    # exact (mu, sd) used during fit instead of re-fitting per-call (which
    # produces inconsistent factor scales across windows).
    train_mu: np.ndarray = field(default_factory=lambda: np.array([]))
    train_sd: np.ndarray = field(default_factory=lambda: np.array([]))

    fitted: bool = False
    log_likelihood: float = float("nan")
    iterations: int = 0
    # v1.3 (item B1): per-iteration log-likelihood path so the EM
    # convergence check is auditable. The Sherman-Morrison-Woodbury
    # marginal likelihood loses precision in the rare degenerate case
    # where ``1 + pp * sum(lambda^2 / sigma_eps^2)`` collapses; the
    # fitter detects non-monotonicity within ``tol``, falls back to a
    # diagonal-conditional log-likelihood for the convergence check
    # only, and records the fallback in ``fit_log["likelihood_path"]``.
    fit_log: dict = field(default_factory=dict)

    def fit(self, panel: pd.DataFrame) -> DFMDomainModel:
        if panel is None or panel.empty:
            return self
        Y = panel.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).to_numpy(float)
        n, k = Y.shape
        if n < max(24, k * 3):
            return self
        # Standardize columns (PIT not enforced here — the caller is expected to
        # pass robust-z transformed inputs).
        mu = Y.mean(axis=0)
        sd = Y.std(axis=0, ddof=0)
        sd[sd < 1e-9] = 1.0
        Z = (Y - mu) / sd
        self.columns = list(panel.columns)

        # Initial guesses: PCA-style.
        try:
            cov = np.cov(Z, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            loadings = eigvecs[:, order[0]] * math.sqrt(max(eigvals[order[0]], 1e-3))
        except np.linalg.LinAlgError:
            loadings = np.ones(k) / math.sqrt(k)
        # Anchor sign on first column's loading.
        if loadings[0] < 0:
            loadings = -loadings
        sigma_eps2 = np.maximum(np.var(Z, axis=0) - loadings**2, 1e-3)
        phi = 0.6
        sigma_eta2 = 1.0 - phi**2

        prev_ll = -np.inf
        # v1.3 (item B1): track the full likelihood path AND the per-step
        # diagonal surrogate. When the marginal likelihood is non-monotone
        # (rare numeric edge case where SMW loses precision) we fall back
        # to the surrogate for the convergence check only — the fitted
        # params still come from the marginal-likelihood E-step.
        likelihood_path: list[float] = []
        surrogate_path: list[float] = []
        fallback_iters: list[int] = []
        for it in range(self.max_iter):
            # Kalman filter / RTS smoother
            f_filt = np.zeros(n)
            p_filt = np.ones(n)
            f_pred = np.zeros(n)
            p_pred = np.ones(n)
            f_filt[0] = 0.0
            p_filt[0] = 1.0
            ll = 0.0
            ll_surrogate = 0.0
            for t in range(n):
                if t == 0:
                    fp = 0.0
                    pp = 1.0
                else:
                    fp = phi * f_filt[t - 1]
                    pp = phi**2 * p_filt[t - 1] + sigma_eta2
                f_pred[t] = fp
                p_pred[t] = pp
                # innovation
                resid = Z[t] - loadings * fp
                # True marginal Z_t ~ N(0, lambda lambda^T pp + diag(sigma_eps^2))
                # via the Sherman-Morrison-Woodbury identity for log|S| and S^-1
                # so we don't form the k x k matrix explicitly:
                #   log|S| = sum log(sigma_eps^2) + log(1 + pp * sum(lambda^2/sigma_eps^2))
                #   resid' S^-1 resid =
                #       sum(resid^2/sigma_eps^2)
                #     - pp * (sum(lambda*resid/sigma_eps^2))^2
                #       / (1 + pp * sum(lambda^2/sigma_eps^2))
                inv_eps = 1.0 / np.maximum(sigma_eps2, 1e-12)
                lam_sq_inv_eps = float(np.sum((loadings**2) * inv_eps))
                denom_term = 1.0 + pp * lam_sq_inv_eps
                quad_a = float(np.sum((resid**2) * inv_eps))
                quad_b_num = float(np.sum(loadings * resid * inv_eps))
                quad = quad_a - pp * (quad_b_num**2) / max(denom_term, 1e-12)
                log_det = float(np.sum(np.log(np.maximum(sigma_eps2, 1e-12)))) + math.log(max(denom_term, 1e-12))
                ll += -0.5 * (k * math.log(2 * math.pi) + log_det + quad)
                # v1.3 (item B1): diagonal-conditional surrogate likelihood.
                # Treat the factor as known at its predicted value; the
                # observation density factorises across components and is
                # numerically stable even when SMW loses precision.
                ll_surrogate += -0.5 * float(
                    k * math.log(2 * math.pi) + np.sum(np.log(np.maximum(sigma_eps2, 1e-12))) + quad_a
                )
                # Information-form posterior (joint update across all observation dims):
                # 1/p_filt = 1/pp + sum(loadings^2 / sigma_eps2)
                info = 1.0 / max(pp, 1e-12) + np.sum((loadings**2) * inv_eps)
                p_filt[t] = 1.0 / info
                f_filt[t] = p_filt[t] * (fp / max(pp, 1e-12) + float(np.sum(loadings * Z[t] * inv_eps)))

            # RTS smoother
            f_smooth = np.zeros(n)
            p_smooth = np.zeros(n)
            f_smooth[-1] = f_filt[-1]
            p_smooth[-1] = p_filt[-1]
            for t in range(n - 2, -1, -1):
                C = phi * p_filt[t] / max(p_pred[t + 1], 1e-12)
                f_smooth[t] = f_filt[t] + C * (f_smooth[t + 1] - f_pred[t + 1])
                p_smooth[t] = p_filt[t] + C**2 * (p_smooth[t + 1] - p_pred[t + 1])

            # M-step
            E_ff = float(np.sum(f_smooth**2 + p_smooth))
            new_loadings = np.zeros(k)
            new_sigma_eps2 = np.zeros(k)
            for j in range(k):
                num = float(np.sum(Z[:, j] * f_smooth))
                new_loadings[j] = num / max(E_ff, 1e-12)
                resid = Z[:, j] - new_loadings[j] * f_smooth
                new_sigma_eps2[j] = float(np.mean(resid**2 + (new_loadings[j] ** 2) * p_smooth))
                new_sigma_eps2[j] = max(new_sigma_eps2[j], self.ridge)
            # phi update
            num = float(np.sum(f_smooth[1:] * f_smooth[:-1]))
            den = float(np.sum(f_smooth[:-1] ** 2 + p_smooth[:-1]))
            new_phi = num / max(den, 1e-12)
            new_phi = float(np.clip(new_phi, -0.99, 0.99))
            # sigma_eta2 update (innovation variance of factor)
            innov = f_smooth[1:] - new_phi * f_smooth[:-1]
            new_sigma_eta2 = max(float(np.mean(innov**2 + p_smooth[1:])), self.ridge)

            loadings = new_loadings
            sigma_eps2 = new_sigma_eps2
            phi = new_phi
            sigma_eta2 = new_sigma_eta2

            likelihood_path.append(float(ll))
            surrogate_path.append(float(ll_surrogate))

            # Detect a non-monotone marginal step (the rare SMW-precision
            # edge case from item B1). The current iteration's ``ll`` is
            # below the previous one *despite* an EM step that is
            # mathematically guaranteed to weakly increase the objective.
            non_monotone = (
                math.isfinite(prev_ll) and ll < prev_ll - 1e-9 and (prev_ll - ll) < self.tol * max(abs(prev_ll), 1.0)
            )

            check_ll = ll
            check_prev = prev_ll
            if non_monotone:
                # Fall back to the diagonal-conditional surrogate for the
                # convergence check only. The fitted params still come
                # from the marginal-likelihood E-step above. Record the
                # fallback so ``fit_log["likelihood_path"]`` exposes it.
                fallback_iters.append(it)
                check_ll = ll_surrogate
                check_prev = surrogate_path[-2] if len(surrogate_path) >= 2 else -np.inf

            if abs(check_ll - check_prev) < self.tol * max(abs(check_prev), 1.0):
                self.iterations = it + 1
                # The reported log-likelihood is always the marginal LL
                # (the surrogate is a diagnostic, not a replacement).
                self.log_likelihood = float(ll)
                self.fitted = True
                break
            prev_ll = ll
        else:
            self.iterations = self.max_iter
            self.log_likelihood = float(ll)
            self.fitted = True

        self.fit_log = {
            "likelihood_path": likelihood_path,
            "surrogate_path": surrogate_path,
            "fallback_iters": fallback_iters,
            "fallback_used": bool(fallback_iters),
        }

        # Anchor loading sign on first column (so factor direction is identifiable).
        if loadings[0] < 0:
            loadings = -loadings
        self.phi = float(phi)
        self.sigma_eta2 = float(sigma_eta2)
        self.loadings = loadings
        self.sigma_eps2 = sigma_eps2
        # Cache training-time standardization so transform() reproduces the
        # same scale across calls (vs. refitting per window, which collapses
        # short windows to z=0 and silently shrinks the factor amplitude).
        self.train_mu = mu.astype(float).copy()
        self.train_sd = sd.astype(float).copy()
        return self

    def transform(self, panel: pd.DataFrame) -> pd.Series:
        """Return the smoothed factor series for the supplied panel.

        For convenience this re-runs the Kalman filter on the new data using
        the learned parameters, so the same model can score historical and
        live windows. The factor is reported on a standardized scale so it
        slots straight into the existing domain-score pipeline.
        """
        if not self.fitted or panel is None or panel.empty:
            return pd.Series(0.0, index=panel.index if panel is not None else None)
        Y = panel.reindex(columns=self.columns).fillna(0.0).to_numpy(float)
        # Reuse the (mu, sd) captured at fit() time so the factor scale is
        # consistent across calls. Falls back to per-call stats only if the
        # cache is empty (e.g. legacy pickled models without train_mu).
        if self.train_mu.size == self.loadings.size and self.train_sd.size == self.loadings.size:
            mu = self.train_mu
            sd = np.where(self.train_sd < 1e-9, 1.0, self.train_sd)
        else:
            mu = Y.mean(axis=0)
            sd = Y.std(axis=0, ddof=0)
            sd[sd < 1e-9] = 1.0
        Z = (Y - mu) / sd
        n = Y.shape[0]
        f = np.zeros(n)
        pf = np.ones(n)
        for t in range(n):
            if t == 0:
                fp = 0.0
                pp = 1.0
            else:
                fp = self.phi * f[t - 1]
                pp = self.phi**2 * pf[t - 1] + self.sigma_eta2
            info = 1.0 / max(pp, 1e-12) + float(np.sum((self.loadings**2) / np.maximum(self.sigma_eps2, 1e-12)))
            pf[t] = 1.0 / info
            f[t] = pf[t] * (
                fp / max(pp, 1e-12) + float(np.sum(self.loadings * Z[t] / np.maximum(self.sigma_eps2, 1e-12)))
            )
        return pd.Series(f, index=panel.index)


def fit_domain_dfm(
    domain_features: pd.DataFrame,
    *,
    domain_columns: dict[str, Iterable[str]],
) -> dict[str, DFMDomainModel]:
    """Fit one DFM per domain. ``domain_columns`` maps domain → ordered list of
    feature columns to feed into that domain's model.
    """
    models: dict[str, DFMDomainModel] = {}
    for domain, cols in domain_columns.items():
        cols = [c for c in cols if c in domain_features.columns]
        if not cols:
            continue
        sub = domain_features[cols].dropna(how="all")
        if sub.empty:
            continue
        models[domain] = DFMDomainModel().fit(sub)
    return models


__all__ = ["DFMDomainModel", "fit_domain_dfm"]
