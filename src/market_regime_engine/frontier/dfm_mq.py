# SPDX-License-Identifier: Apache-2.0
"""Mixed-frequency dynamic factor model (Bańbura-Modugno + native D/W/M).

The statsmodels ``DynamicFactorMQ`` backend is used for the monthly/quarterly
(M/Q) contract when the optional ``[nowcast]`` dependencies are installed. When
daily or weekly series are present, this module switches to a native single-
factor Kalman state-space backend whose observation matrix maps a latent daily
factor into daily, weekly, and monthly releases. Unsupported layouts fail closed
rather than being silently relabeled as monthly data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from market_regime_engine.dfm import DFMDomainModel
from market_regime_engine.frontier.experimental import require_frontier_experimental

log = logging.getLogger(__name__)


def _statsmodels_available() -> tuple[bool, Any]:
    """Return ``(installed, DynamicFactorMQ_class_or_None)``."""
    try:  # pragma: no cover - exercised when optional dependency exists
        from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

        return True, DynamicFactorMQ
    except Exception:
        return False, None


@dataclass
class _CustomMixedFrequencyStateSpace:
    """Single-factor daily state-space model for true D/W/M ragged panels.

    State vector: ``[f_t, f_{t-1}, ..., f_{t-L+1}]`` on a daily calendar.
    Transition: AR(1) for the current factor plus deterministic lag shifting.
    Observation: each observed release uses ``lambda_i`` times an average of
    latent daily factors: D=1 day, W=7 calendar days, M=month-to-date capped at
    ``max_lag``. This is a real mixed-frequency observation equation.
    """

    max_lag: int = 31
    phi: float = 0.85
    factor_var: float = 1.0
    obs_ridge: float = 1e-3

    columns: list[str] = field(default_factory=list)
    frequencies: dict[str, str] = field(default_factory=dict)
    train_mu: pd.Series = field(default_factory=pd.Series)
    train_sd: pd.Series = field(default_factory=pd.Series)
    loadings: np.ndarray = field(default_factory=lambda: np.array([]))
    obs_var: np.ndarray = field(default_factory=lambda: np.array([]))
    daily_index: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))
    filtered_factor: pd.Series = field(default_factory=pd.Series)
    filtered_var: pd.Series = field(default_factory=pd.Series)
    fitted: bool = False
    fit_log: dict[str, Any] = field(default_factory=dict)

    def _window(self, ts: pd.Timestamp, freq: str) -> int:
        freq = str(freq).upper()
        if freq == "D":
            return 1
        if freq == "W":
            return min(7, self.max_lag)
        if freq == "M":
            return min(max(int(pd.Timestamp(ts).day), 1), self.max_lag)
        raise ValueError(f"custom mixed-frequency backend supports only D/W/M; received {freq!r}")

    def _transition(self, dim: int) -> tuple[np.ndarray, np.ndarray]:
        F = np.zeros((dim, dim), dtype=float)
        F[0, 0] = float(np.clip(self.phi, -0.995, 0.995))
        if dim > 1:
            F[1:, :-1] = np.eye(dim - 1)
        Q = np.zeros((dim, dim), dtype=float)
        Q[0, 0] = max(float(self.factor_var), 1e-8)
        return F, Q

    def _standardize(self, panel: pd.DataFrame) -> pd.DataFrame:
        aligned = panel.reindex(columns=self.columns).astype(float)
        return (aligned - self.train_mu) / self.train_sd.replace(0.0, 1.0)

    def _initial_pca(self, z: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float, float]:
        filled = z.ffill().bfill().fillna(0.0).to_numpy(dtype=float)
        n, k = filled.shape
        if n < 3 or k == 0:
            return np.ones(k), np.ones(k), 0.85, 1.0
        cov = np.cov(filled, rowvar=False)
        if np.ndim(cov) == 0:
            loadings = np.array([float(np.sqrt(max(float(cov), 1e-3)))])
            factor = filled[:, 0] / max(loadings[0], 1e-6)
        else:
            try:
                eigvals, eigvecs = np.linalg.eigh(cov + self.obs_ridge * np.eye(k))
                idx = int(np.argmax(eigvals))
                loadings = eigvecs[:, idx] * np.sqrt(max(float(eigvals[idx]), 1e-3))
                factor = filled @ eigvecs[:, idx]
            except np.linalg.LinAlgError:
                loadings = np.ones(k) / max(np.sqrt(k), 1.0)
                factor = filled @ loadings
        if loadings.size and loadings[0] < 0:
            loadings = -loadings
            factor = -factor
        den = float(np.dot(factor[:-1], factor[:-1])) if len(factor) > 1 else 0.0
        phi = float(np.clip(np.dot(factor[1:], factor[:-1]) / max(den, 1e-12), -0.95, 0.95)) if len(factor) > 1 else 0.85
        innov = factor[1:] - phi * factor[:-1] if len(factor) > 1 else factor
        factor_var = max(float(np.nanvar(innov)), self.obs_ridge)
        residual = filled - np.outer(factor, loadings)
        obs_var = np.maximum(np.nanvar(residual, axis=0), self.obs_ridge)
        return loadings.astype(float), obs_var.astype(float), phi, factor_var

    def fit(self, panel: pd.DataFrame, frequencies: dict[str, str]) -> _CustomMixedFrequencyStateSpace:
        if panel is None or panel.empty:
            return self
        panel = panel.sort_index().copy()
        self.columns = list(panel.columns)
        self.frequencies = dict(frequencies)
        self.train_mu = panel.mean(skipna=True).fillna(0.0)
        sd = panel.std(skipna=True, ddof=0).replace(0.0, np.nan).fillna(1.0)
        self.train_sd = sd.where(sd.abs() > 1e-9, 1.0)
        z = self._standardize(panel)
        self.loadings, self.obs_var, self.phi, self.factor_var = self._initial_pca(z)

        start = pd.Timestamp(panel.index.min()).normalize()
        end = pd.Timestamp(panel.index.max()).normalize()
        self.daily_index = pd.date_range(start, end, freq="D")
        dim = int(self.max_lag)
        F, Q = self._transition(dim)
        x = np.zeros(dim, dtype=float)
        P = np.eye(dim, dtype=float)
        factors: list[float] = []
        variances: list[float] = []
        z_by_date = {pd.Timestamp(idx).normalize(): row for idx, row in z.iterrows()}
        update_count = 0
        for ts in self.daily_index:
            x = F @ x
            P = F @ P @ F.T + Q
            row = z_by_date.get(pd.Timestamp(ts).normalize())
            if row is not None:
                H_rows: list[np.ndarray] = []
                y_vals: list[float] = []
                r_vals: list[float] = []
                for col_idx, col in enumerate(self.columns):
                    val = row.get(col)
                    if val is None or not np.isfinite(float(val)):
                        continue
                    w = self._window(ts, self.frequencies.get(col, "M"))
                    h = np.zeros(dim, dtype=float)
                    h[:w] = float(self.loadings[col_idx]) / float(w)
                    H_rows.append(h)
                    y_vals.append(float(val))
                    r_vals.append(float(self.obs_var[col_idx]))
                if H_rows:
                    H = np.vstack(H_rows)
                    y = np.asarray(y_vals, dtype=float)
                    R = np.diag(np.maximum(np.asarray(r_vals, dtype=float), self.obs_ridge))
                    innov = y - H @ x
                    S = H @ P @ H.T + R
                    try:
                        K_gain = P @ H.T @ np.linalg.inv(S)
                    except np.linalg.LinAlgError:
                        K_gain = P @ H.T @ np.linalg.pinv(S)
                    x = x + K_gain @ innov
                    P = (np.eye(dim) - K_gain @ H) @ P
                    P = (P + P.T) / 2.0 + 1e-10 * np.eye(dim)
                    update_count += 1
            factors.append(float(x[0]))
            variances.append(float(max(P[0, 0], 1e-12)))
        self.filtered_factor = pd.Series(factors, index=self.daily_index, name="factor")
        self.filtered_var = pd.Series(variances, index=self.daily_index, name="factor_var")
        self.fitted = True
        self.fit_log = {
            "backend": "custom_state_space",
            "state_dim": dim,
            "updates": update_count,
            "phi": float(self.phi),
            "factor_var": float(self.factor_var),
            "frequency_contract": sorted(set(self.frequencies.values())),
        }
        return self

    def nowcast(self, asof: pd.Timestamp) -> dict[str, Any]:
        asof_ts = pd.Timestamp(asof).normalize()
        if not self.fitted or self.filtered_factor.empty:
            return {"as_of": str(asof_ts.date()), "factor": 0.0, "factor_se": float("nan"), "backend": "custom_state_space", "fitted": False}
        prefix = self.filtered_factor.loc[self.filtered_factor.index <= asof_ts]
        vprefix = self.filtered_var.loc[self.filtered_var.index <= asof_ts]
        if prefix.empty:
            val = float(self.filtered_factor.iloc[0])
            var = float(self.filtered_var.iloc[0])
        else:
            val = float(prefix.iloc[-1])
            var = float(vprefix.iloc[-1])
        return {"as_of": str(asof_ts.date()), "factor": val, "factor_se": float(np.sqrt(max(var, 1e-12))), "backend": "custom_state_space", "fitted": self.fitted}

    def update(self, panel: pd.DataFrame, frequencies: dict[str, str]) -> _CustomMixedFrequencyStateSpace:
        return self.fit(panel, frequencies)


@dataclass
class MQDynamicFactorModel:
    """Mixed-frequency dynamic-factor facade with soft-degrade.

    ``D/W/M`` panels use the native custom Kalman state-space backend. ``M/Q``
    panels use statsmodels ``DynamicFactorMQ`` when available, otherwise the
    legacy single-frequency ``DFMDomainModel`` fallback.
    """

    n_factors: int = 1
    factor_orders: int = 1
    enforce_stationarity: bool = True

    fitted: bool = False
    backend: str = "fallback"
    columns: list[str] = field(default_factory=list)
    frequencies: dict[str, str] = field(default_factory=dict)
    _model: Any = None
    _results: Any = None
    _fallback_model: DFMDomainModel | None = None
    _custom_model: _CustomMixedFrequencyStateSpace | None = None
    _panel: pd.DataFrame | None = None
    _last_factor: float = 0.0
    _last_factor_se: float = 1.0

    @staticmethod
    def _normalize_frequencies(panel: pd.DataFrame, frequencies: dict[str, str] | None) -> dict[str, str]:
        freq_map = {col: "M" for col in panel.columns}
        if frequencies:
            unknown = set(frequencies).difference(panel.columns)
            if unknown:
                raise ValueError(f"frequencies contains columns not present in panel: {sorted(unknown)}")
            for col, freq in frequencies.items():
                norm = str(freq).upper()
                if norm not in {"D", "W", "M", "Q"}:
                    raise ValueError(
                        "MQDynamicFactorModel supports only daily ('D'), weekly ('W'), monthly ('M'), "
                        f"and quarterly ('Q') frequencies; column {col!r} was marked {freq!r}."
                    )
                freq_map[col] = norm
        values = set(freq_map.values())
        if values and values.issubset({"Q"}):
            raise ValueError("at least one monthly ('M') series is required when quarterly series are supplied")
        if values.intersection({"D", "W"}) and "Q" in values:
            raise ValueError("unsupported frequency layout: quarterly ('Q') cannot be mixed with daily/weekly custom D/W/M state-space inputs")
        return freq_map

    @staticmethod
    def _statsmodels_kwargs(panel: pd.DataFrame, frequencies: dict[str, str], *, n_factors: int, factor_orders: int) -> dict[str, Any]:
        monthly_cols = [col for col in panel.columns if frequencies.get(col, "M") == "M"]
        quarterly_cols = [col for col in panel.columns if frequencies.get(col, "M") == "Q"]
        kwargs: dict[str, Any] = {"endog": panel[monthly_cols] if monthly_cols else panel, "factors": n_factors, "factor_orders": factor_orders}
        if quarterly_cols and monthly_cols:
            kwargs["endog_quarterly"] = panel[quarterly_cols]
        elif quarterly_cols and not monthly_cols:
            raise ValueError("at least one monthly ('M') series is required when quarterly series are supplied")
        return kwargs

    def fit(self, panel: pd.DataFrame, *, frequencies: dict[str, str] | None = None) -> MQDynamicFactorModel:
        if panel is None or panel.empty:
            return self
        self.columns = list(panel.columns)
        self.frequencies = self._normalize_frequencies(panel, frequencies)
        self._panel = panel.copy().sort_index()
        freq_values = set(self.frequencies.values())
        if freq_values.intersection({"D", "W"}):
            custom = _CustomMixedFrequencyStateSpace().fit(self._panel, self.frequencies)
            self._custom_model = custom
            self.backend = "custom_state_space"
            latest = custom.nowcast(pd.Timestamp(self._panel.index.max()))
            self._last_factor = float(latest["factor"])
            self._last_factor_se = float(latest["factor_se"])
            self.fitted = True
            return self

        installed, dfm_cls = _statsmodels_available()
        if installed:
            try:
                model = dfm_cls(**self._statsmodels_kwargs(self._panel, self.frequencies, n_factors=self.n_factors, factor_orders=self.factor_orders))
                results = model.fit(disp=False)
                self._model = model
                self._results = results
                self.backend = "statsmodels"
                factor_series = self._extract_factor_series(results, filtered=True)
                if factor_series is not None and len(factor_series) > 0:
                    self._last_factor = float(factor_series.iloc[-1])
                    se = self._extract_factor_se(results, strict=False)
                    self._last_factor_se = float(se) if se is not None else float("nan")
                self.fitted = True
                return self
            except Exception:
                pass
        fallback = DFMDomainModel().fit(self._panel)
        if fallback.fitted:
            self._fallback_model = fallback
            self.backend = "fallback"
            factor_series = fallback.transform(self._panel)
            if not factor_series.empty:
                self._last_factor = float(factor_series.iloc[-1])
                self._last_factor_se = 1.0
            self.fitted = True
        return self

    @staticmethod
    def _extract_factor_series(results: Any, *, filtered: bool = True) -> pd.Series | None:
        if not filtered:
            require_frontier_experimental("DFM-MQ smoothed latent factors are retrospective-only and use future observations")
        preferred = "filtered" if filtered else "smoothed"
        fallback_attr = "smoothed" if filtered else "filtered"
        for attr in ("factors", "smoothed_state"):
            try:
                candidate = getattr(results, attr, None)
                if candidate is None:
                    continue
                for state_attr in (preferred, fallback_attr):
                    state = getattr(candidate, state_attr, None)
                    if state is None:
                        continue
                    if isinstance(state, pd.DataFrame):
                        return state.iloc[:, 0]
                    if isinstance(state, np.ndarray):
                        return pd.Series(state[:, 0])
                if isinstance(candidate, pd.DataFrame):
                    return candidate.iloc[:, 0]
                if isinstance(candidate, np.ndarray) and candidate.ndim == 2:
                    return pd.Series(candidate[:, 0])
            except Exception:
                continue
        return None

    @staticmethod
    def _extract_factor_se(results: Any, *, strict: bool = False) -> float | None:
        try:
            cov = getattr(results, "smoothed_state_cov", None)
            if cov is not None and isinstance(cov, np.ndarray) and cov.ndim == 3:
                return float(np.sqrt(max(cov[0, 0, -1], 1e-12)))
        except Exception:
            pass
        if strict:
            raise ValueError("factor_se: structured smoothed_state_cov unavailable; refusing to return params-based proxy")
        log.warning("MQDynamicFactorModel: structured smoothed_state_cov unavailable; factor_se will be reported as NaN")
        return None

    def nowcast(self, asof: pd.Timestamp) -> dict[str, Any]:
        asof_ts = pd.Timestamp(asof)
        response = {"as_of": str(asof_ts.date()), "factor": float(self._last_factor), "factor_se": float(self._last_factor_se) if self._last_factor_se is not None else float("nan"), "backend": self.backend, "fitted": self.fitted}
        if not self.fitted or self._panel is None:
            return response
        prefix = self._panel[self._panel.index <= asof_ts]
        if prefix.empty:
            return response
        if self.backend == "custom_state_space" and self._custom_model is not None:
            return self._custom_model.nowcast(asof_ts)
        if self.backend == "statsmodels" and self._results is not None:
            try:
                installed, dfm_cls = _statsmodels_available()
                if installed:
                    pit_model = dfm_cls(**self._statsmodels_kwargs(prefix, self.frequencies, n_factors=self.n_factors, factor_orders=self.factor_orders))
                    pit_results = pit_model.fit(disp=False)
                    series = self._extract_factor_series(pit_results, filtered=True)
                    if series is not None and len(series) > 0:
                        response["factor"] = float(series.iloc[-1])
                        se = self._extract_factor_se(pit_results, strict=False)
                        response["factor_se"] = float(se) if se is not None else float("nan")
            except Exception:
                pass
        elif self.backend == "fallback" and self._fallback_model is not None:
            try:
                series = self._fallback_model.transform(prefix)
                if not series.empty:
                    response["factor"] = float(series.iloc[-1])
                    response["factor_se"] = 1.0
            except Exception:
                pass
        return response

    def update(self, new_observation: pd.Series) -> dict[str, Any]:
        obs_ts = pd.Timestamp(new_observation.name) if new_observation.name is not None else pd.Timestamp.today()
        if not self.fitted:
            return self.nowcast(obs_ts)
        if self._panel is not None and new_observation.name is not None:
            new_row = pd.DataFrame([new_observation], columns=self.columns, index=[obs_ts])
            self._panel = pd.concat([self._panel, new_row]).sort_index()
        if self.backend == "custom_state_space" and self._custom_model is not None:
            if self._panel is not None:
                self._custom_model.update(self._panel, self.frequencies)
                latest = self._custom_model.nowcast(obs_ts)
                self._last_factor = float(latest["factor"])
                self._last_factor_se = float(latest["factor_se"])
                return latest
            return self._custom_model.nowcast(obs_ts)
        if self.backend == "statsmodels" and self._results is not None:
            try:
                row = pd.DataFrame([new_observation], columns=self.columns, index=[obs_ts])
                applied = self._results.append(row)
                self._results = applied
                factor_series = self._extract_factor_series(applied, filtered=True)
                if factor_series is not None and len(factor_series) > 0:
                    self._last_factor = float(factor_series.iloc[-1])
                    se = self._extract_factor_se(applied, strict=False)
                    self._last_factor_se = float(se) if se is not None else float("nan")
            except Exception:
                pass
        elif self.backend == "fallback" and self._fallback_model is not None:
            row = pd.DataFrame([new_observation], columns=self.columns, index=[obs_ts])
            try:
                f = self._fallback_model.transform(row)
                if not f.empty:
                    self._last_factor = float(f.iloc[-1])
                    self._last_factor_se = 1.0
            except Exception:
                pass
        return self.nowcast(obs_ts)


def build_synthetic_panel(*, n_months: int = 60, seed: int = 0, n_series: int = 4, factor_persistence: float = 0.7) -> tuple[pd.DataFrame, np.ndarray]:
    """Build a synthetic monthly panel and latent factor for tests."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_months, freq="MS")
    f = np.zeros(n_months)
    for t in range(1, n_months):
        f[t] = factor_persistence * f[t - 1] + rng.normal(scale=0.5)
    loadings = rng.uniform(0.6, 1.4, size=n_series)
    eps = rng.normal(scale=0.3, size=(n_months, n_series))
    Y = f[:, None] * loadings[None, :] + eps
    cols = [f"series_{i}" for i in range(n_series)]
    return pd.DataFrame(Y, index=dates, columns=cols), f


__all__ = ["MQDynamicFactorModel", "build_synthetic_panel"]
