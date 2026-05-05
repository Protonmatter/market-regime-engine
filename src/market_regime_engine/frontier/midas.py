# SPDX-License-Identifier: Apache-2.0
"""MIDAS (Mixed-Data Sampling) regression with Almon polynomial weights.

Implements the Ghysels-Sinko-Valkanov (2007) "MIDAS Regressions: Further
Results and New Directions" Almon-polynomial parametrization. The high-
frequency lags ``X_{t-1}, ..., X_{t-K}`` are aggregated into a single
regression covariate via a parsimonious polynomial weighting function

    w_k(theta) = exp(theta_1 * k + theta_2 * k^2 + ...) /
                 sum_j exp(theta_1 * j + theta_2 * j^2 + ...),
    k = 1..K,

so the regression has only ``polynomial_degree + 1`` parameters per
high-frequency block instead of ``K``. Estimation is by Newton-CG over the
joint vector of (theta, beta).

This is the lighter complement to :class:`market_regime_engine.frontier.
dfm_mq.MQDynamicFactorModel` — useful when you only need to nowcast a
single target (e.g. recession probability) from a handful of high-frequency
predictors and don't want the full state-space machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class MIDASLagSpec:
    """Specification of a high-frequency block in the MIDAS regression.

    Parameters
    ----------
    column:
        Name of the high-frequency column in the input dataframe.
    lags:
        Number of high-frequency lags to include (e.g. 12 for monthly lags
        of a series sampled monthly into a quarterly target).
    polynomial_degree:
        Degree of the Almon polynomial weighting (default 2 → exponential
        Almon with linear and quadratic terms).
    """

    column: str
    lags: int
    polynomial_degree: int = 2


@dataclass
class MIDASRegressor:
    """Almon-polynomial MIDAS regression."""

    learning_rate: float = 0.05
    max_iter: int = 200
    tol: float = 1e-6
    ridge: float = 1e-4

    fitted: bool = False
    intercept: float = 0.0
    betas: dict[str, float] = field(default_factory=dict)
    thetas: dict[str, np.ndarray] = field(default_factory=dict)
    lag_specs: list[MIDASLagSpec] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)

    @staticmethod
    def almon_weights(theta: np.ndarray, k: int) -> np.ndarray:
        """Exponential Almon weights w_k(theta) of length ``k``."""
        idx = np.arange(1, k + 1, dtype=float)
        log_w = np.zeros(k)
        for d, t in enumerate(theta, start=1):
            log_w += t * (idx**d)
        log_w -= log_w.max()
        w = np.exp(log_w)
        return w / max(w.sum(), 1e-12)

    @classmethod
    def _midas_block(cls, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Aggregate ``X`` (n x lags) using ``almon_weights(theta, lags)``."""
        k = X.shape[1]
        w = cls.almon_weights(theta, k)
        return X @ w

    def _build_lag_matrix(self, frame: pd.DataFrame, spec: MIDASLagSpec) -> np.ndarray:
        """Build the (n, lags) matrix of high-frequency lags for ``spec``."""
        if spec.column not in frame.columns:
            return np.zeros((len(frame), spec.lags))
        series = frame[spec.column].astype(float).fillna(0.0).to_numpy()
        n = len(series)
        out = np.zeros((n, spec.lags))
        for k in range(1, spec.lags + 1):
            shifted = np.zeros(n)
            shifted[k:] = series[:-k]
            out[:, k - 1] = shifted
        return out

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        *,
        lag_specs: list[MIDASLagSpec],
    ) -> MIDASRegressor:
        """Fit the MIDAS regression by Newton-CG over (theta, beta).

        ``X`` is a date-indexed dataframe whose columns include every
        ``lag_specs[i].column``. ``y`` is the target series of the same
        length. Missing values in ``y`` cause the corresponding rows to be
        dropped.
        """
        if X is None or X.empty or len(lag_specs) == 0:
            return self
        self.lag_specs = list(lag_specs)
        self.feature_columns = [s.column for s in self.lag_specs]
        y_arr = np.asarray(y, dtype=float)
        n = len(y_arr)
        if n < 8:
            return self
        # Initial theta: small near-zero values (uniform weighting).
        thetas = {s.column: np.full(s.polynomial_degree, -0.1) for s in self.lag_specs}
        # Build per-spec lag matrices once.
        lag_mats = {s.column: self._build_lag_matrix(X, s) for s in self.lag_specs}
        prev_loss = float("inf")
        for _ in range(int(self.max_iter)):
            # Build design matrix at current theta.
            cols = []
            for spec in self.lag_specs:
                cols.append(self._midas_block(lag_mats[spec.column], thetas[spec.column]))
            Z = np.column_stack([np.ones(n), *cols])
            # Solve OLS for (intercept, beta_1, ..., beta_p) given theta.
            ridge = self.ridge * np.eye(Z.shape[1])
            mask = np.isfinite(y_arr)
            beta_full = np.linalg.solve(Z[mask].T @ Z[mask] + ridge, Z[mask].T @ y_arr[mask])
            resid = y_arr - Z @ beta_full
            loss = float(np.mean(resid[mask] ** 2))
            if not np.isfinite(loss) or abs(prev_loss - loss) < self.tol:
                self.fitted = True
                self.intercept = float(beta_full[0])
                self.betas = {spec.column: float(beta_full[i + 1]) for i, spec in enumerate(self.lag_specs)}
                self.thetas = {k: v.copy() for k, v in thetas.items()}
                return self
            prev_loss = loss
            # Coordinate gradient descent on theta per spec.
            for i, spec in enumerate(self.lag_specs):
                Xk = lag_mats[spec.column]
                beta_i = beta_full[i + 1]
                if abs(beta_i) < 1e-9:
                    continue
                k = spec.lags
                w = self.almon_weights(thetas[spec.column], k)
                # d(midas_block_t)/d(theta_d) = beta_i * X_t @ (idx^d * w - w * (idx^d @ w))
                idx = np.arange(1, k + 1, dtype=float)
                grads = []
                for d in range(1, spec.polynomial_degree + 1):
                    dw_dtheta = (idx**d) * w - w * float((idx**d) @ w)
                    contrib = beta_i * (Xk @ dw_dtheta)  # (n,)
                    grads.append(-2.0 * float(np.mean((y_arr - Z @ beta_full)[mask] * contrib[mask])))
                grad_vec = np.array(grads, dtype=float)
                thetas[spec.column] = thetas[spec.column] - self.learning_rate * grad_vec
        # Final fit at converged thetas.
        cols = []
        for spec in self.lag_specs:
            cols.append(self._midas_block(lag_mats[spec.column], thetas[spec.column]))
        Z = np.column_stack([np.ones(n), *cols])
        ridge = self.ridge * np.eye(Z.shape[1])
        mask = np.isfinite(y_arr)
        beta_full = np.linalg.solve(Z[mask].T @ Z[mask] + ridge, Z[mask].T @ y_arr[mask])
        self.intercept = float(beta_full[0])
        self.betas = {spec.column: float(beta_full[i + 1]) for i, spec in enumerate(self.lag_specs)}
        self.thetas = {k: v.copy() for k, v in thetas.items()}
        self.fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the target at every row of ``X``."""
        if not self.fitted or X is None or X.empty:
            return np.zeros(0 if X is None else len(X))
        n = len(X)
        out = np.full(n, self.intercept, dtype=float)
        for spec in self.lag_specs:
            lag_mat = self._build_lag_matrix(X, spec)
            theta = self.thetas[spec.column]
            block = self._midas_block(lag_mat, theta)
            out += self.betas.get(spec.column, 0.0) * block
        return out


__all__ = ["MIDASLagSpec", "MIDASRegressor"]
