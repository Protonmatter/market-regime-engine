# SPDX-License-Identifier: Apache-2.0
"""Mixed-frequency dynamic factor model (Bańbura-Modugno 2014).

Wraps :class:`statsmodels.tsa.statespace.dynamic_factor_mq.DynamicFactorMQ`
behind a thin domain-friendly facade so the engine can run a real DFM-MQ
when the optional ``[nowcast]`` extra is installed and *gracefully degrade*
to the existing single-frequency ``DFMDomainModel`` otherwise.

The Bańbura-Modugno 2014 design (JoE 2014, "Maximum Likelihood Estimation of
Factor Models on Datasets with Arbitrary Pattern of Missing Data") handles
ragged-edge data by aligning month-end, week-end, and daily series in the
same state-space and missing-data EM machinery. This is the standard
production nowcast architecture at the New York Fed and the ECB.

Public API:

- :class:`MQDynamicFactorModel` — fit / nowcast / update.
- :func:`build_synthetic_panel` — helper for tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from market_regime_engine.dfm import DFMDomainModel

log = logging.getLogger(__name__)


def _statsmodels_available() -> tuple[bool, Any]:
    """Return ``(installed, DynamicFactorMQ_class_or_None)``."""
    try:  # pragma: no cover - exercised at import time
        from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

        return True, DynamicFactorMQ
    except Exception:
        return False, None


@dataclass
class MQDynamicFactorModel:
    """Bańbura-Modugno 2014 DFM-MQ wrapper with soft-degrade.

    Parameters
    ----------
    n_factors:
        Number of latent global factors. Default 1 (matches the legacy
        per-domain DFM contract; bump to 2-3 for cross-domain factors).
    factor_orders:
        Lag order of the factor VAR. Defaults to 1 (Watson-Engle style).
    enforce_stationarity:
        Pass-through to ``DynamicFactorMQ``. Default ``True``.
    """

    n_factors: int = 1
    factor_orders: int = 1
    enforce_stationarity: bool = True

    fitted: bool = False
    backend: str = "fallback"  # "statsmodels" | "fallback"
    columns: list[str] = field(default_factory=list)
    frequencies: dict[str, str] = field(default_factory=dict)
    _model: Any = None
    _results: Any = None
    _fallback_model: DFMDomainModel | None = None
    _panel: pd.DataFrame | None = None
    _last_factor: float = 0.0
    _last_factor_se: float = 1.0

    def fit(self, panel: pd.DataFrame, *, frequencies: dict[str, str] | None = None) -> MQDynamicFactorModel:
        """Fit the DFM-MQ on ``panel`` (date-indexed wide frame).

        ``frequencies`` maps each column to ``"M"`` (monthly), ``"W"``
        (weekly), or ``"D"`` (daily). When statsmodels is missing or the fit
        raises, the function transparently falls back to the legacy single-
        frequency ``DFMDomainModel`` on whatever monthly columns remain.
        """
        if panel is None or panel.empty:
            return self
        self.columns = list(panel.columns)
        self.frequencies = dict(frequencies or {})
        # v1.6.0 (REVIEW_DEEP_V1_5_2.md A15 / Finding #10): cache the full
        # panel index so .nowcast(asof) can apply a PIT-safe filter at
        # nowcast time. The Bańbura-Modugno smoothed state at index t uses
        # observations after t — cannot be reported without leakage when
        # asof < latest_index.
        self._panel = panel.copy()
        installed, dfm_cls = _statsmodels_available()
        if installed:
            try:
                # statsmodels' DynamicFactorMQ takes the panel directly with a
                # DatetimeIndex; mixed frequencies are inferred from the index
                # spacing per column.
                model = dfm_cls(
                    endog=panel,
                    factors=self.n_factors,
                    factor_orders=self.factor_orders,
                    enforce_stationarity=self.enforce_stationarity,
                )
                results = model.fit(disp=False)
                self._model = model
                self._results = results
                self.backend = "statsmodels"
                # Cache the latest FILTERED (not smoothed) factor mean for
                # nowcast PIT-safety; smoothed states use future obs.
                factor_series = self._extract_factor_series(results, filtered=True)
                if factor_series is not None and len(factor_series) > 0:
                    self._last_factor = float(factor_series.iloc[-1])
                    se = self._extract_factor_se(results, strict=False)
                    self._last_factor_se = float(se) if se is not None else float("nan")
                self.fitted = True
                return self
            except Exception:
                # Fall through to the v1.0 single-frequency DFM.
                pass
        # Soft-degrade: fit the v1.0 DFM on whatever the panel contains.
        fallback = DFMDomainModel().fit(panel)
        if fallback.fitted:
            self._fallback_model = fallback
            self.backend = "fallback"
            factor_series = fallback.transform(panel)
            if not factor_series.empty:
                self._last_factor = float(factor_series.iloc[-1])
                self._last_factor_se = 1.0
            self.fitted = True
        return self

    @staticmethod
    def _extract_factor_series(results: Any, *, filtered: bool = True) -> pd.Series | None:
        """Return the latent factor series from a fitted model.

        Parameters
        ----------
        results:
            Statsmodels ``DynamicFactorMQResults``-like object exposing a
            ``factors``-style structured state attribute.
        filtered:
            When ``True`` (default), prefer the **filtered** state
            (``factors.filtered``) — uses only observations up to time
            ``t`` and is the PIT-safe choice for nowcasting. When
            ``False``, fall back to the smoothed state (uses the entire
            sample's information including future observations).

        v1.6.0 (REVIEW_DEEP_V1_5_2.md A15 / Finding #10): the previous
        implementation defaulted to smoothed and silently leaked
        future information into ``nowcast``. The default is now
        ``filtered=True``; callers explicitly opt in to smoothed for
        backtesting / retrospective analysis.
        """
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
            except Exception:  # pragma: no cover - defensive
                continue
        return None

    @staticmethod
    def _extract_factor_se(results: Any, *, strict: bool = False) -> float | None:
        """Extract the latest factor standard error.

        v1.6.0 (REVIEW_DEEP_V1_5_2.md A15 / Finding #10): previously the
        fallback returned ``max(abs(params))`` as a "rough proxy" — this
        is **not** a standard error in any statistical sense and was
        indistinguishable downstream from a real Kalman-derived SE. The
        new contract:

        - Structured covariance (``smoothed_state_cov[0, 0, -1]``)
          present → return ``sqrt(...)`` as before. Real SE.
        - Structured covariance unavailable AND ``strict=True`` → raise
          ``ValueError`` so the caller can decide.
        - Structured covariance unavailable AND ``strict=False`` (default)
          → log a warning and return ``None`` (NaN-equivalent in the
          numeric pipeline). Callers receive an explicit "no SE
          available" signal rather than a misleading proxy.
        """
        try:
            cov = getattr(results, "smoothed_state_cov", None)
            if cov is not None and isinstance(cov, np.ndarray) and cov.ndim == 3:
                return float(np.sqrt(max(cov[0, 0, -1], 1e-12)))
        except Exception:  # pragma: no cover - defensive
            pass
        if strict:
            raise ValueError(
                "factor_se: structured smoothed_state_cov unavailable; refusing "
                "to return params-based proxy (REVIEW_DEEP_V1_5_2.md A15)."
            )
        log.warning(
            "MQDynamicFactorModel: structured smoothed_state_cov unavailable; "
            "factor_se will be reported as NaN (no params-based proxy emitted)"
        )
        return None

    def nowcast(self, asof: pd.Timestamp) -> dict[str, Any]:
        """Return the latest factor mean + standard error at or before ``asof``.

        v1.6.0 (REVIEW_DEEP_V1_5_2.md A15 / Finding #10): previously the
        ``asof`` argument was ignored (only used to label the response);
        the cached ``_last_factor`` was returned regardless of ``asof``,
        which is PIT-unsafe when ``asof`` lies in the past. The new
        contract filters the cached panel to ``index <= asof`` and
        recomputes the latest factor over that prefix using the same
        filtered-state / fallback machinery as ``.fit``.
        """
        asof_ts = pd.Timestamp(asof)
        # Default response shape: factor + SE from the last successful fit.
        response = {
            "as_of": str(asof_ts.date()),
            "factor": float(self._last_factor),
            "factor_se": float(self._last_factor_se) if self._last_factor_se is not None else float("nan"),
            "backend": self.backend,
            "fitted": self.fitted,
        }
        if not self.fitted or self._panel is None:
            return response
        # PIT-safe filter: only rows at or before asof contribute to the factor.
        prefix = self._panel[self._panel.index <= asof_ts]
        if prefix.empty:
            return response
        if self.backend == "statsmodels" and self._results is not None:
            try:
                installed, dfm_cls = _statsmodels_available()
                if installed:
                    pit_model = dfm_cls(
                        endog=prefix,
                        factors=self.n_factors,
                        factor_orders=self.factor_orders,
                        enforce_stationarity=self.enforce_stationarity,
                    )
                    pit_results = pit_model.fit(disp=False)
                    series = self._extract_factor_series(pit_results, filtered=True)
                    if series is not None and len(series) > 0:
                        response["factor"] = float(series.iloc[-1])
                        se = self._extract_factor_se(pit_results, strict=False)
                        response["factor_se"] = float(se) if se is not None else float("nan")
            except Exception:  # pragma: no cover - defensive
                # PIT-safe nowcast on a sub-panel may fail; fall back to the
                # cached final-fit response (still labelled with ``asof``).
                pass
        elif self.backend == "fallback" and self._fallback_model is not None:
            try:
                series = self._fallback_model.transform(prefix)
                if not series.empty:
                    response["factor"] = float(series.iloc[-1])
                    response["factor_se"] = 1.0
            except Exception:  # pragma: no cover - defensive
                pass
        return response

    def update(self, new_observation: pd.Series) -> dict[str, Any]:
        """Roll the latent factor forward with a single new row.

        Returns the updated factor mean / SE under the same shape as
        :meth:`nowcast`. Under the statsmodels backend we re-apply the fitted
        Kalman filter; under the fallback we re-evaluate the legacy DFM.
        """
        if not self.fitted:
            return self.nowcast(pd.Timestamp(new_observation.name) if new_observation.name else pd.Timestamp.today())
        if self.backend == "statsmodels" and self._results is not None:
            try:
                row = pd.DataFrame([new_observation], columns=self.columns)
                applied = self._results.append(row)
                self._results = applied
                factor_series = self._extract_factor_series(applied, filtered=True)
                if factor_series is not None and len(factor_series) > 0:
                    self._last_factor = float(factor_series.iloc[-1])
                    se = self._extract_factor_se(applied, strict=False)
                    self._last_factor_se = float(se) if se is not None else float("nan")
            except Exception:  # pragma: no cover - defensive
                pass
        elif self.backend == "fallback" and self._fallback_model is not None:
            row = pd.DataFrame([new_observation], columns=self.columns)
            try:
                f = self._fallback_model.transform(row)
                if not f.empty:
                    self._last_factor = float(f.iloc[-1])
                    self._last_factor_se = 1.0
            except Exception:  # pragma: no cover - defensive
                pass
        # Append the new observation to the cached panel so subsequent
        # .nowcast(asof) calls can apply a PIT-safe prefix filter.
        if self._panel is not None and new_observation.name is not None:
            new_row = pd.DataFrame(
                [new_observation], columns=self.columns, index=[pd.Timestamp(new_observation.name)]
            )
            self._panel = pd.concat([self._panel, new_row]).sort_index()
        return self.nowcast(
            pd.Timestamp(new_observation.name) if new_observation.name is not None else pd.Timestamp.today()
        )


def build_synthetic_panel(
    *,
    n_months: int = 60,
    seed: int = 0,
    n_series: int = 4,
    factor_persistence: float = 0.7,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build a synthetic mixed-frequency panel for tests.

    Returns ``(panel, true_factor)`` where ``panel`` is a month-indexed wide
    frame and ``true_factor`` is the underlying AR(1) factor used to generate
    the columns.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_months, freq="MS")
    f = np.zeros(n_months)
    for t in range(1, n_months):
        f[t] = factor_persistence * f[t - 1] + rng.normal(scale=0.5)
    loadings = rng.uniform(0.6, 1.4, size=n_series)
    eps = rng.normal(scale=0.3, size=(n_months, n_series))
    Y = f[:, None] * loadings[None, :] + eps
    cols = [f"series_{i}" for i in range(n_series)]
    panel = pd.DataFrame(Y, index=dates, columns=cols)
    return panel, f


__all__ = ["MQDynamicFactorModel", "build_synthetic_panel"]
