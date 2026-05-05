# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from market_regime_engine.features import feature_matrix
from market_regime_engine.regimes import domain_scores


def feature_driver_attribution(
    features: pd.DataFrame, *, as_of: str | pd.Timestamp | None = None, top_n: int = 20
) -> pd.DataFrame:
    X = feature_matrix(features)
    if X.empty:
        return pd.DataFrame(
            columns=["date", "rank", "feature_name", "domain", "value", "zscore", "abs_zscore", "direction"]
        )
    X = X.replace([np.inf, -np.inf], np.nan).ffill()
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else X.index.max()
    if as_of_ts not in X.index:
        as_of_ts = X.index[X.index <= as_of_ts].max()
    hist = X[X.index < as_of_ts].tail(120)
    mu = hist.mean()
    sd = hist.std(ddof=1).replace(0, np.nan)
    z = ((X.loc[as_of_ts] - mu) / sd).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    latest_features = features.copy()
    domain_map = (
        latest_features.drop_duplicates("feature_name").set_index("feature_name")["domain"].to_dict()
        if not latest_features.empty
        else {}
    )
    rows = []
    for rank, (fname, _zval) in enumerate(z.abs().sort_values(ascending=False).head(top_n).items(), start=1):
        raw_z = float(z[fname])
        rows.append(
            {
                "date": as_of_ts,
                "rank": rank,
                "feature_name": fname,
                "domain": domain_map.get(fname, "unknown"),
                "value": float(X.loc[as_of_ts, fname]) if pd.notna(X.loc[as_of_ts, fname]) else 0.0,
                "zscore": raw_z,
                "abs_zscore": abs(raw_z),
                "direction": "risk_up" if raw_z > 0 else "risk_down",
                "metadata_json": json.dumps({"window_months": len(hist)}),
            }
        )
    return pd.DataFrame(rows)


def domain_driver_attribution(features: pd.DataFrame, *, as_of: str | pd.Timestamp | None = None) -> pd.DataFrame:
    ds = domain_scores(features)
    if ds.empty:
        return pd.DataFrame(columns=["date", "rank", "domain", "score", "zscore", "change_3m", "metadata_json"])
    piv = ds.pivot(index="date", columns="domain", values="score").sort_index().ffill().fillna(0.0)
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else piv.index.max()
    if as_of_ts not in piv.index:
        as_of_ts = piv.index[piv.index <= as_of_ts].max()
    hist = piv[piv.index < as_of_ts].tail(120)
    mu = hist.mean()
    sd = hist.std(ddof=1).replace(0, np.nan)
    z = ((piv.loc[as_of_ts] - mu) / sd).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    prev = piv.shift(3).loc[as_of_ts] if as_of_ts in piv.shift(3).index else piv.loc[as_of_ts] * 0
    rows = []
    for rank, domain in enumerate(z.abs().sort_values(ascending=False).index, start=1):
        rows.append(
            {
                "date": as_of_ts,
                "rank": rank,
                "domain": domain,
                "score": float(piv.loc[as_of_ts, domain]),
                "zscore": float(z[domain]),
                "change_3m": float(piv.loc[as_of_ts, domain] - prev.get(domain, 0.0)),
                "metadata_json": json.dumps({"window_months": len(hist)}),
            }
        )
    return pd.DataFrame(rows)
