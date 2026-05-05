# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _zscore_frame(X: pd.DataFrame) -> pd.DataFrame:
    if X.empty:
        return X.copy()
    mu = X.expanding(min_periods=24).mean().shift(1)
    sd = X.expanding(min_periods=24).std(ddof=0).shift(1).replace(0, np.nan)
    return ((X - mu) / sd).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _domain_columns(X: pd.DataFrame, keyword: str) -> list[str]:
    return [c for c in X.columns if keyword.lower() in c.lower()]


def recession_hazard_scores(
    features: pd.DataFrame, recession_labels: pd.DataFrame | None = None, horizons: tuple[int, ...] = (3, 6, 12)
) -> pd.DataFrame:
    """Create a deterministic survival-style recession timing forecast.

    This is a transparent hazard model. It is intentionally simple and stable:
    it uses rolling standardized labor/rates/credit/housing/inflation stress and
    converts those stresses into monthly hazard rates, then into horizon survival
    probabilities. A later v0.7 can replace the coefficients with fitted Cox or
    discrete-time logit coefficients.
    """
    if features is None or features.empty:
        return pd.DataFrame(columns=["model_name", "date", "horizon", "target", "value", "metadata_json"])
    from market_regime_engine.features import feature_matrix

    X = feature_matrix(features)
    if X.empty:
        return pd.DataFrame()
    Z = _zscore_frame(X)

    def block_score(names: list[str], sign: float = 1.0) -> pd.Series:
        cols = [c for c in names if c in Z.columns]
        if not cols:
            return pd.Series(0.0, index=Z.index)
        return sign * Z[cols].mean(axis=1)

    labor = block_score(_domain_columns(Z, "UNRATE") + _domain_columns(Z, "U6"), 1.0)
    credit = block_score(_domain_columns(Z, "BAA") + _domain_columns(Z, "spread"), 1.0)
    rates = block_score(
        _domain_columns(Z, "FEDFUNDS") + _domain_columns(Z, "DGS10") + _domain_columns(Z, "MORTGAGE"), 1.0
    )
    housing = block_score(_domain_columns(Z, "PERMIT") + _domain_columns(Z, "HOUST"), -1.0)
    inflation = block_score(_domain_columns(Z, "CPI") + _domain_columns(Z, "PCE"), 1.0)
    energy = block_score(_domain_columns(Z, "OIL") + _domain_columns(Z, "GAS"), 1.0)

    raw = -3.05 + 0.42 * labor + 0.36 * credit + 0.24 * rates + 0.30 * housing + 0.18 * inflation + 0.12 * energy
    hazard = raw.map(_sigmoid).clip(0.005, 0.45)

    rows = []
    for dt, h0 in hazard.items():
        for horizon in horizons:
            p = 1.0 - float(np.prod([(1.0 - h0)] * horizon))
            rows.append(
                {
                    "model_name": "survival_hazard_v0_6",
                    "date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                    "horizon": f"{horizon}m",
                    "target": "recession_probability",
                    "value": float(max(0.0, min(1.0, p))),
                    "metadata_json": "{}",
                }
            )
            rows.append(
                {
                    "model_name": "survival_hazard_v0_6",
                    "date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                    "horizon": "1m",
                    "target": "monthly_recession_hazard",
                    "value": float(h0),
                    "metadata_json": "{}",
                }
            )
    out = pd.DataFrame(rows).drop_duplicates(["model_name", "date", "horizon", "target"], keep="last")
    return out.sort_values(["date", "target", "horizon"])


def survival_summary(outputs: pd.DataFrame) -> str:
    if outputs is None or outputs.empty:
        return "No survival outputs."
    latest = outputs[outputs["date"] == outputs["date"].max()]
    lines = [f"Survival hazard summary as of {latest['date'].max()}"]
    for _, row in latest.sort_values(["target", "horizon"]).iterrows():
        lines.append(f"- {row['target']} {row['horizon']}: {float(row['value']):.1%}")
    return "\n".join(lines)
