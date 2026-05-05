# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pandas as pd


def population_stability_index(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    e = pd.to_numeric(expected, errors="coerce").dropna()
    a = pd.to_numeric(actual, errors="coerce").dropna()
    if len(e) < bins or len(a) < bins:
        return 0.0
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    e_counts = np.histogram(e, bins=edges)[0].astype(float)
    a_counts = np.histogram(a, bins=edges)[0].astype(float)
    e_pct = np.clip(e_counts / max(e_counts.sum(), 1.0), 1e-6, 1.0)
    a_pct = np.clip(a_counts / max(a_counts.sum(), 1.0), 1e-6, 1.0)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def drift_status(psi: float) -> str:
    if psi >= 0.50:
        return "severe"
    if psi >= 0.25:
        return "major"
    if psi >= 0.10:
        return "moderate"
    return "stable"


def compute_feature_drift(
    features: pd.DataFrame, baseline_months: int = 120, recent_months: int = 12, top_n: int = 50
) -> pd.DataFrame:
    if features is None or features.empty:
        return pd.DataFrame(columns=["date", "feature_name", "psi", "mean_shift", "status", "metadata_json"])
    f = features.copy()
    f["date"] = pd.to_datetime(f["date"])
    as_of = f["date"].max()
    baseline_start = as_of - pd.DateOffset(months=baseline_months + recent_months)
    recent_start = as_of - pd.DateOffset(months=recent_months)
    base = f[(f["date"] >= baseline_start) & (f["date"] < recent_start)]
    recent = f[f["date"] >= recent_start]
    rows = []
    for name in sorted(set(f["feature_name"])):
        b = base.loc[base["feature_name"] == name, "value"]
        r = recent.loc[recent["feature_name"] == name, "value"]
        if len(b) < 12 or len(r) < 3:
            continue
        psi = population_stability_index(b, r)
        bstd = float(pd.to_numeric(b, errors="coerce").std(ddof=0) or 0.0)
        shift = (
            0.0
            if bstd == 0
            else float((pd.to_numeric(r, errors="coerce").mean() - pd.to_numeric(b, errors="coerce").mean()) / bstd)
        )
        rows.append(
            {
                "date": as_of.strftime("%Y-%m-%d"),
                "feature_name": name,
                "psi": psi,
                "mean_shift": shift,
                "status": drift_status(psi),
                "metadata_json": "{}",
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["abs_mean_shift"] = out["mean_shift"].abs()
    out = out.sort_values(["psi", "abs_mean_shift"], ascending=False).head(top_n).drop(columns=["abs_mean_shift"])
    return out.reset_index(drop=True)


def drift_summary(drift: pd.DataFrame) -> pd.DataFrame:
    if drift is None or drift.empty:
        return pd.DataFrame(
            [{"date": None, "severe": 0, "major": 0, "moderate": 0, "stable": 0, "max_psi": 0.0, "drift_score": 0.0}]
        )
    latest = drift[drift["date"] == drift["date"].max()]
    counts = latest["status"].value_counts().to_dict()
    max_psi = float(latest["psi"].max())
    score = min(1.0, max_psi / 0.50)
    return pd.DataFrame(
        [
            {
                "date": latest["date"].iloc[0],
                "severe": int(counts.get("severe", 0)),
                "major": int(counts.get("major", 0)),
                "moderate": int(counts.get("moderate", 0)),
                "stable": int(counts.get("stable", 0)),
                "max_psi": max_psi,
                "drift_score": score,
            }
        ]
    )
