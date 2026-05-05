# SPDX-License-Identifier: Apache-2.0
"""Diagnostic: compare the v1.2.1 vectorised path to a faithful recreation
of the pre-v1.2.1 per-asof loop on the synthetic sample.

Run::

    .venv\\Scripts\\python.exe scripts/compare_asof_implementations.py

Exits 0 when the two outputs match within float tolerance, non-zero
otherwise. Used to drive the correctness regression test in
``tests/test_asof_perf.py``.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from market_regime_engine.alfred_real import seed_vintage_observations_from_latest
from market_regime_engine.asof import (
    latest_vintage_observations_asof,
    materialize_feature_asof_values,
)
from market_regime_engine.config import load_catalog
from market_regime_engine.features import build_features, monthly_panel
from market_regime_engine.sample import generate_sample_observations


def _date_str(x: object) -> str:
    return pd.Timestamp(x).strftime("%Y-%m-%d")


def legacy_materialize(
    vintage_observations: pd.DataFrame,
    catalog: list[dict],
    *,
    asof_dates: list[str] | None = None,
    min_history_months: int = 36,
) -> pd.DataFrame:
    """Verbatim copy of the pre-v1.2.1 per-asof loop, retained here for
    correctness comparison only."""
    if vintage_observations is None or vintage_observations.empty:
        return pd.DataFrame()
    df = vintage_observations.copy()
    df["vintage_date"] = pd.to_datetime(df["vintage_date"], errors="coerce")
    df["observation_date"] = pd.to_datetime(df["observation_date"], errors="coerce")
    if asof_dates is None:
        vmin = df["vintage_date"].min()
        vmax = df["vintage_date"].max()
        if pd.isna(vmin) or pd.isna(vmax):
            return pd.DataFrame()
        start = max(vmin, df["observation_date"].min() + pd.DateOffset(months=min_history_months))
        asof_idx = pd.date_range(start=start, end=vmax, freq="MS")
        asof_dates = [d.strftime("%Y-%m-%d") for d in asof_idx]
    created = "fixed-utc-for-comparison"
    rows: list[pd.DataFrame] = []
    series_by_feature = {f"{item['series_id']}.": item["series_id"] for item in catalog}
    for asof in asof_dates:
        legal_obs = latest_vintage_observations_asof(df, asof)
        if legal_obs.empty:
            continue
        panel = monthly_panel(legal_obs)
        feats = build_features(panel, catalog)
        if feats.empty:
            continue
        feats["date"] = pd.to_datetime(feats["date"])
        cutoff = pd.Timestamp(asof)
        latest_feats = (
            feats[feats["date"] <= cutoff]
            .sort_values(["feature_name", "date"])
            .drop_duplicates("feature_name", keep="last")
        )
        if latest_feats.empty:
            continue
        lineage_rows = []
        for _, r in latest_feats.iterrows():
            fname = str(r["feature_name"])
            sid = next(
                (v for prefix, v in series_by_feature.items() if fname.startswith(prefix)),
                fname.split(".")[0],
            )
            obs_date = pd.Timestamp(r["date"])
            source_candidates = legal_obs[
                (legal_obs["series_id"] == sid)
                & (pd.to_datetime(legal_obs["date"]) <= obs_date)
            ].copy()
            if source_candidates.empty:
                continue
            source_row = source_candidates.sort_values("date").iloc[-1]
            transform = fname.split(".", 1)[1] if "." in fname else "unknown"
            lineage_rows.append(
                {
                    "as_of_date": _date_str(asof),
                    "feature_name": fname,
                    "source_series_id": sid,
                    "observation_date": _date_str(source_row["date"]),
                    "vintage_date": _date_str(source_row["vintage_date"]),
                    "value": float(r["value"]),
                    "transform_name": transform,
                    "created_at_utc": created,
                    "metadata_json": json.dumps(
                        {"feature_date": _date_str(r["date"]), "domain": r.get("domain")},
                        sort_keys=True,
                    ),
                }
            )
        if lineage_rows:
            rows.append(pd.DataFrame(lineage_rows))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop the timestamp-stamped ``created_at_utc`` and sort
    deterministically so the two implementations can be compared row-wise."""
    cols = [c for c in frame.columns if c != "created_at_utc"]
    out = frame[cols].copy()
    out["value"] = out["value"].astype(float).round(9)
    out = out.sort_values(["as_of_date", "feature_name"]).reset_index(drop=True)
    return out


def main() -> int:
    obs = generate_sample_observations()
    _, vintage_obs = seed_vintage_observations_from_latest(obs)
    catalog = load_catalog()

    t0 = time.perf_counter()
    legacy = legacy_materialize(vintage_obs, catalog)
    legacy_elapsed = time.perf_counter() - t0
    print(f"legacy elapsed: {legacy_elapsed:.3f}s   rows: {len(legacy)}")

    t0 = time.perf_counter()
    fast = materialize_feature_asof_values(vintage_obs, catalog)
    fast_elapsed = time.perf_counter() - t0
    print(f"fast   elapsed: {fast_elapsed:.3f}s   rows: {len(fast)}")
    print(f"speedup: {legacy_elapsed / max(fast_elapsed, 1e-9):.1f}x")

    if len(legacy) != len(fast):
        print(f"ROW COUNT MISMATCH: legacy={len(legacy)} fast={len(fast)}")
        return 1

    a = _normalize(legacy)
    b = _normalize(fast)
    if not a.equals(b):
        diffs = a.compare(b)
        print("DIFFS (first 50 rows):")
        print(diffs.head(50).to_string())
        return 2
    print("OK: vectorised output matches legacy per-loop output exactly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
