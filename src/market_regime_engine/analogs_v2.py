# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json

import pandas as pd

from market_regime_engine.analogs import HistoricalAnalogEngine


def regime_weighted_analogs(
    X: pd.DataFrame, targets: pd.DataFrame | None, regimes: pd.DataFrame, *, top_n: int = 10, as_of: str | None = None
) -> pd.DataFrame:
    """Post-process historical analogs so same/near regimes get higher similarity.

    v0.5 keeps the base distance engine but reweights by decoded-regime agreement and HMM posterior overlap
    when posterior metadata is available.
    """
    base = HistoricalAnalogEngine(top_n=max(top_n * 3, top_n), min_history=60).score(X, targets, regimes, as_of=as_of)
    if base.empty or regimes is None or regimes.empty:
        return base.head(top_n)
    r = regimes.copy()
    r["date"] = pd.to_datetime(r["date"])
    regime_by_date = dict(zip(r["date"], r.get("decoded_regime", r.get("regime")), strict=False))
    meta_by_date = {}
    for _, row in r.iterrows():
        try:
            meta_by_date[pd.Timestamp(row["date"])] = json.loads(row.get("metadata_json", "{}") or "{}")
        except Exception:
            meta_by_date[pd.Timestamp(row["date"])] = {}
    asof_date = pd.Timestamp(base["as_of_date"].iloc[0])
    current_regime = regime_by_date.get(asof_date)
    current_post = (meta_by_date.get(asof_date) or {}).get("hmm_posterior", {})

    weights = []
    metas = []
    for _, row in base.iterrows():
        adate = pd.Timestamp(row["analog_date"])
        areg = regime_by_date.get(adate)
        boost = 1.0
        if current_regime and areg == current_regime:
            boost += 0.30
        apost = (meta_by_date.get(adate) or {}).get("hmm_posterior", {})
        overlap = 0.0
        if isinstance(current_post, dict) and isinstance(apost, dict):
            keys = set(current_post).intersection(apost)
            overlap = sum(min(float(current_post.get(k, 0.0)), float(apost.get(k, 0.0))) for k in keys)
            boost += 0.20 * max(0.0, min(1.0, overlap))
        weights.append(float(row["similarity"]) * boost)
        try:
            meta = json.loads(row.get("metadata_json", "{}") or "{}")
        except Exception:
            meta = {}
        meta["regime_weighting"] = {
            "current_regime": current_regime,
            "analog_regime": areg,
            "posterior_overlap": overlap,
            "boost": boost,
        }
        metas.append(json.dumps(meta, sort_keys=True))
    out = base.copy()
    out["similarity"] = weights
    out["similarity"] = out["similarity"] / out["similarity"].sum()
    out["metadata_json"] = metas
    out = out.sort_values("similarity", ascending=False).head(top_n).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    return out
