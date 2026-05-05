# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json

import pandas as pd


def latest_explanation(regimes: pd.DataFrame) -> dict:
    if regimes.empty:
        return {"status": "no_regime_scores"}
    row = regimes.sort_values("date").iloc[-1]
    try:
        meta = json.loads(row.get("metadata_json", "{}"))
    except Exception:
        meta = {}
    scores = meta.get("domain_scores", {})
    top = sorted(scores.items(), key=lambda kv: abs(float(kv[1])), reverse=True)[:5]
    return {
        "date": str(row["date"]),
        "regime": row["decoded_regime"],
        "raw_regime": row["regime"],
        "score": float(row["score"]),
        "change_point_prob": float(row["change_point_prob"]) if pd.notna(row["change_point_prob"]) else None,
        "top_drivers": [{"domain": k, "score": float(v)} for k, v in top],
    }
