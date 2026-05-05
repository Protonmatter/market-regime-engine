# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _robust_zscore(frame: pd.DataFrame, window: int = 120, min_periods: int = 36) -> pd.DataFrame:
    mu = frame.rolling(window, min_periods=min_periods).mean().shift(1)
    sd = frame.rolling(window, min_periods=min_periods).std(ddof=1).shift(1)
    return (frame - mu) / sd.replace(0, np.nan)


def _safe_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)


@dataclass
class HistoricalAnalogEngine:
    max_lookback: int | None = None
    min_history: int = 60
    top_n: int = 10
    temperature: float = 1.0

    def score(
        self,
        feature_matrix: pd.DataFrame,
        targets: pd.DataFrame | None = None,
        regimes: pd.DataFrame | None = None,
        as_of: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        if feature_matrix.empty:
            return pd.DataFrame(
                columns=["as_of_date", "analog_date", "rank", "distance", "similarity", "metadata_json"]
            )
        X = feature_matrix.copy().sort_index()
        X = _safe_numeric(X)
        Z = _robust_zscore(X).dropna(how="all")
        if Z.empty:
            Z = X.apply(lambda s: (s - s.expanding().mean().shift(1)) / s.expanding().std(ddof=1).shift(1)).fillna(0.0)
        Z = _safe_numeric(Z)
        as_of_ts = pd.Timestamp(as_of) if as_of is not None else Z.index.max()
        if as_of_ts not in Z.index:
            as_of_ts = Z.index[Z.index <= as_of_ts].max()
        hist = Z[Z.index < as_of_ts].copy()
        if self.max_lookback:
            hist = hist.tail(self.max_lookback)
        if len(hist) < self.min_history:
            return pd.DataFrame(
                columns=["as_of_date", "analog_date", "rank", "distance", "similarity", "metadata_json"]
            )
        current = Z.loc[as_of_ts]

        # Domain-aware column weights via feature prefix. Keep intentionally simple and testable.
        weights = pd.Series(1.0, index=Z.columns)
        for col in Z.columns:
            if any(token in col for token in ["BAA", "credit", "T10Y3M"]):
                weights[col] = 1.4
            if any(token in col for token in ["UNRATE", "U6RATE", "PAYEMS"]):
                weights[col] = 1.25
            if any(token in col for token in ["PERMIT", "HOUST", "MORTGAGE"]):
                weights[col] = 1.20
            if any(token in col for token in ["CPI", "DCOIL", "FEDFUNDS", "DGS10"]):
                weights[col] = 1.15

        diffs = hist.subtract(current, axis=1)
        dist = np.sqrt(((diffs**2).multiply(weights, axis=1)).mean(axis=1).replace(0, np.nan)).replace(np.nan, 0.0)
        selected = dist.sort_values().head(self.top_n)
        if selected.empty:
            return pd.DataFrame()
        raw_sim = np.exp(-selected / max(self.temperature, 1e-9))
        sim = raw_sim / raw_sim.sum() if raw_sim.sum() else raw_sim

        target_lookup = targets.copy() if targets is not None and not targets.empty else pd.DataFrame()
        if not target_lookup.empty:
            target_lookup.index = pd.to_datetime(target_lookup.index)
        regime_lookup = {}
        if regimes is not None and not regimes.empty:
            r = regimes.copy()
            r["date"] = pd.to_datetime(r["date"])
            regime_lookup = dict(zip(r["date"], r.get("decoded_regime", r.get("regime")), strict=False))

        rows = []
        for rank, (date, d) in enumerate(selected.items(), start=1):
            meta: dict[str, object] = {}
            if not target_lookup.empty and date in target_lookup.index:
                vals = target_lookup.loc[date]
                meta["forward_returns"] = {
                    k: float(v) for k, v in vals.items() if str(k).startswith("ret_") and pd.notna(v)
                }
                meta["drawdowns"] = {k: float(v) for k, v in vals.items() if str(k).startswith("dd_") and pd.notna(v)}
            if date in regime_lookup:
                meta["analog_regime"] = str(regime_lookup[date])
            rows.append(
                {
                    "as_of_date": as_of_ts,
                    "analog_date": date,
                    "rank": rank,
                    "distance": float(d),
                    "similarity": float(sim.loc[date]),
                    "metadata_json": json.dumps(meta, sort_keys=True),
                }
            )
        return pd.DataFrame(rows)


def analog_summary(analogs: pd.DataFrame) -> dict:
    if analogs.empty:
        return {"status": "no_analogs"}
    regimes: dict[str, float] = {}
    ret: dict[str, float] = {}
    dd: dict[str, float] = {}
    for _, row in analogs.iterrows():
        w = float(row.get("similarity", 0.0))
        try:
            meta = json.loads(row.get("metadata_json", "{}"))
        except Exception:
            meta = {}
        reg = meta.get("analog_regime")
        if reg:
            regimes[reg] = regimes.get(reg, 0.0) + w
        for k, v in meta.get("forward_returns", {}).items():
            ret[k] = ret.get(k, 0.0) + w * float(v)
        for k, v in meta.get("drawdowns", {}).items():
            dd[k] = dd.get(k, 0.0) + w * float(v)
    return {
        "regime_mix": dict(sorted(regimes.items(), key=lambda kv: kv[1], reverse=True)),
        "weighted_forward_returns": ret,
        "weighted_forward_drawdowns": dd,
    }
