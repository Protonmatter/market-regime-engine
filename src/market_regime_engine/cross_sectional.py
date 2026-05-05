# SPDX-License-Identifier: Apache-2.0
"""Cross-sectional factor / sector / curve heads conditioned on regime state.

The base engine forecasts the SP-style aggregate. Most institutional uses of
a regime engine want sector / style / curve dispersion *conditional* on the
current regime. This module exposes three light-weight heads that consume the
regime posterior ``gamma_t`` and the change-point probability ``CP_t``:

- :func:`fama_french_regime_head` regresses the four canonical equity factors
  (SMB / HML / MOM / QMJ) on the regime posterior.
- :func:`sector_dispersion_head` regresses sector return spreads on regime
  state.
- :func:`yield_curve_factor_head` decomposes yield-curve principal components
  (level / slope / curvature) per regime.

Each head returns a tidy frame keyed by ``(model_name, date, horizon, target,
value)`` so the warehouse's ``model_outputs`` table accepts them without
schema migration.

The implementation is deliberately *thin* — the goal is to give a place to
land the cross-sectional layer cleanly. Real production use would swap in a
proper factor-loading regression with regime interactions and HAC standard
errors; the API here supports that without callers needing to change.
"""

from __future__ import annotations

import json

import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _build_regime_design(regimes: pd.DataFrame) -> pd.DataFrame:
    """Pull the regime posterior into a clean wide design matrix."""
    if regimes is None or regimes.empty:
        return pd.DataFrame()
    cols = [c for c in regimes.columns if c.startswith("regime_prob_")]
    if not cols:
        # Fall back to one-hot on decoded_regime.
        if "decoded_regime" not in regimes:
            return pd.DataFrame()
        idx = pd.to_datetime(regimes["date"])
        return pd.get_dummies(regimes.set_index(idx)["decoded_regime"], prefix="regime").astype(float)
    idx = pd.to_datetime(regimes["date"])
    return regimes.set_index(idx)[cols].astype(float)


def fama_french_regime_head(
    factor_returns: pd.DataFrame,
    regimes: pd.DataFrame,
    *,
    horizon_months: int = 3,
) -> pd.DataFrame:
    """Regress factor returns on regime probabilities.

    ``factor_returns`` is a date-indexed dataframe whose columns are factor
    short names (e.g. ``SMB``, ``HML``, ``MOM``, ``QMJ``).
    """
    design = _build_regime_design(regimes)
    if design.empty or factor_returns is None or factor_returns.empty:
        return pd.DataFrame()
    df = factor_returns.copy()
    df.index = pd.to_datetime(df.index)
    rows: list[dict] = []
    for factor in df.columns:
        y = df[factor].dropna()
        # Forward h-month sum return as the target.
        forward = y.rolling(horizon_months).sum().shift(-horizon_months + 1)
        joined = design.join(forward.rename("y"), how="inner").dropna()
        if len(joined) < 24:
            continue
        model = Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                ("ridge", RidgeCV(alphas=(0.1, 1.0, 10.0))),
            ]
        )
        Xfit = joined[design.columns].to_numpy(float)
        yfit = joined["y"].to_numpy(float)
        model.fit(Xfit, yfit)
        latest = joined[design.columns].iloc[[-1]]
        pred = float(model.predict(latest.to_numpy(float))[0])
        rows.append(
            {
                "model_name": "cross_sectional_ff_v1",
                "date": joined.index[-1].strftime("%Y-%m-%d"),
                "horizon": f"{horizon_months}m",
                "target": f"factor_return_{factor}",
                "value": pred,
                "metadata_json": json.dumps({"head": "fama_french", "factor": factor}, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def sector_dispersion_head(
    sector_returns: pd.DataFrame,
    regimes: pd.DataFrame,
    *,
    horizon_months: int = 3,
) -> pd.DataFrame:
    """Forecast sector return dispersion (cross-sectional std) by regime."""
    design = _build_regime_design(regimes)
    if design.empty or sector_returns is None or sector_returns.empty:
        return pd.DataFrame()
    sr = sector_returns.copy()
    sr.index = pd.to_datetime(sr.index)
    horizon_returns = sr.rolling(horizon_months).sum().shift(-horizon_months + 1)
    dispersion = horizon_returns.std(axis=1).rename("dispersion").dropna()
    joined = design.join(dispersion, how="inner").dropna()
    if len(joined) < 24:
        return pd.DataFrame()
    model = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("ridge", RidgeCV(alphas=(0.1, 1.0, 10.0))),
        ]
    )
    Xfit = joined[design.columns].to_numpy(float)
    yfit = joined["dispersion"].to_numpy(float)
    model.fit(Xfit, yfit)
    latest = joined[design.columns].iloc[[-1]]
    pred = float(model.predict(latest.to_numpy(float))[0])
    return pd.DataFrame(
        [
            {
                "model_name": "cross_sectional_dispersion_v1",
                "date": joined.index[-1].strftime("%Y-%m-%d"),
                "horizon": f"{horizon_months}m",
                "target": "sector_dispersion",
                "value": pred,
                "metadata_json": json.dumps({"head": "sector_dispersion"}, sort_keys=True),
            }
        ]
    )


def yield_curve_factor_head(
    yields: pd.DataFrame,
    regimes: pd.DataFrame,
    *,
    horizon_months: int = 3,
) -> pd.DataFrame:
    """Decompose the yield curve into level/slope/curvature and forecast each."""
    design = _build_regime_design(regimes)
    if design.empty or yields is None or yields.empty or yields.shape[1] < 3:
        return pd.DataFrame()
    df = yields.copy()
    df.index = pd.to_datetime(df.index)
    df = df.dropna(how="any")
    if df.empty:
        return pd.DataFrame()
    # Crude principal-component-style projections.
    level = df.mean(axis=1)
    slope = df.iloc[:, -1] - df.iloc[:, 0]
    curvature = df.iloc[:, -1] - 2.0 * df.iloc[:, df.shape[1] // 2] + df.iloc[:, 0]
    factors = pd.DataFrame({"level": level, "slope": slope, "curvature": curvature}).dropna()
    rows: list[dict] = []
    for col in factors.columns:
        y = factors[col].rolling(horizon_months).mean().shift(-horizon_months + 1)
        joined = design.join(y.rename("y"), how="inner").dropna()
        if len(joined) < 24:
            continue
        model = Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                ("ridge", RidgeCV(alphas=(0.1, 1.0, 10.0))),
            ]
        )
        Xfit = joined[design.columns].to_numpy(float)
        yfit = joined["y"].to_numpy(float)
        model.fit(Xfit, yfit)
        latest = joined[design.columns].iloc[[-1]]
        pred = float(model.predict(latest.to_numpy(float))[0])
        rows.append(
            {
                "model_name": "cross_sectional_curve_v1",
                "date": joined.index[-1].strftime("%Y-%m-%d"),
                "horizon": f"{horizon_months}m",
                "target": f"curve_{col}",
                "value": pred,
                "metadata_json": json.dumps({"head": "yield_curve", "component": col}, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


__all__ = ["fama_french_regime_head", "sector_dispersion_head", "yield_curve_factor_head"]
