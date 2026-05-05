# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json

import pandas as pd

from market_regime_engine.features import feature_matrix


def _latest_value(X: pd.DataFrame, name: str) -> float | None:
    if X.empty or name not in X.columns:
        return None
    v = X[name].dropna()
    if v.empty:
        return None
    return float(v.iloc[-1])


def _latest_change(X: pd.DataFrame, name: str, months: int = 3) -> float | None:
    if X.empty or name not in X.columns:
        return None
    v = X[name].dropna()
    if len(v) <= months:
        return None
    return float(v.iloc[-1] - v.iloc[-1 - months])


def forecast_invalidation_triggers(features: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    """Generate simple forecast invalidation triggers based on current state.

    These are not forecasts. They are conditions that should force reweighting/revalidation.
    """
    X = feature_matrix(features)
    if X.empty:
        return pd.DataFrame(columns=["date", "trigger", "severity", "status", "value", "threshold", "metadata_json"])
    date = X.index[-1]
    latest_cp = 0.0
    latest_regime = "unknown"
    if regimes is not None and not regimes.empty:
        r = regimes.copy()
        r["date"] = pd.to_datetime(r["date"])
        rr = r.iloc[-1]
        latest_cp = float(rr.get("change_point_prob", 0.0) or 0.0)
        latest_regime = str(rr.get("decoded_regime", rr.get("regime", "unknown")))

    checks = [
        ("change_point_spike", latest_cp, 0.65, "high", ">="),
        ("unemployment_3m_jump", _latest_change(X, "UNRATE.level", 3), 0.5, "high", ">="),
        ("u6_3m_jump", _latest_change(X, "U6RATE.level", 3), 0.7, "high", ">="),
        ("credit_spread_stress", _latest_value(X, "BAA10Y.level"), 2.5, "high", ">="),
        ("housing_permit_yoy_drop", _latest_value(X, "PERMIT.log_yoy"), -0.15, "medium", "<="),
        ("oil_yoy_shock", _latest_value(X, "DCOILWTICO.log_yoy"), 0.30, "medium", ">="),
        ("dollar_yoy_surge", _latest_value(X, "DTWEXBGS.log_yoy"), 0.08, "medium", ">="),
        ("yield_curve_deep_inversion", _latest_value(X, "T10Y3M.level"), -1.0, "medium", "<="),
    ]
    rows = []
    for name, value, threshold, severity, op in checks:
        if value is None or not pd.notna(value):
            status = "unobservable"
            breached = False
        elif op == ">=":
            breached = float(value) >= float(threshold)
            status = "breached" if breached else "clear"
        else:
            breached = float(value) <= float(threshold)
            status = "breached" if breached else "clear"
        rows.append(
            {
                "date": date,
                "trigger": name,
                "severity": severity if breached else "watch",
                "status": status,
                "value": None if value is None else float(value),
                "threshold": float(threshold),
                "metadata_json": json.dumps({"operator": op, "latest_regime": latest_regime}, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)
