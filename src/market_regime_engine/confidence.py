# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json

import pandas as pd


def _clip01(x: float) -> float:
    if not pd.notna(x):
        return 0.0
    return max(0.0, min(1.0, float(x)))


def compute_model_confidence(
    *,
    regimes: pd.DataFrame,
    validation: pd.DataFrame | None = None,
    analogs: pd.DataFrame | None = None,
    release_audit: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if regimes.empty:
        return pd.DataFrame(columns=["date", "confidence", "grade", "metadata_json"])
    latest = regimes.copy()
    latest["date"] = pd.to_datetime(latest["date"])
    r = latest.iloc[-1]
    cp = _clip01(float(r.get("change_point_prob", 0.0) or 0.0))

    validation_score = 0.5
    if validation is not None and not validation.empty:
        v = validation.copy()
        if "ece" in v and v["ece"].notna().any():
            validation_score = 1.0 - _clip01(float(v["ece"].mean()) / 0.25)
        elif "brier" in v and v["brier"].notna().any():
            validation_score = 1.0 - _clip01(float(v["brier"].mean()) / 0.30)

    analog_score = 0.3
    if analogs is not None and not analogs.empty and "similarity" in analogs:
        latest_asof = analogs["as_of_date"].max() if "as_of_date" in analogs else None
        a = analogs[analogs["as_of_date"] == latest_asof] if latest_asof is not None else analogs
        analog_score = _clip01(float(a["similarity"].head(5).sum()))

    pit_score = 0.8
    if release_audit is not None and not release_audit.empty and "violations" in release_audit:
        rows = max(1, int(release_audit["rows"].sum()))
        violations = int(release_audit["violations"].sum())
        pit_score = 1.0 - _clip01(violations / rows)

    cp_penalty = 0.25 * cp
    confidence = _clip01(
        0.35 * validation_score + 0.25 * analog_score + 0.25 * pit_score + 0.15 * (1.0 - cp) - cp_penalty
    )
    if confidence >= 0.75:
        grade = "high"
    elif confidence >= 0.50:
        grade = "medium"
    elif confidence >= 0.30:
        grade = "low"
    else:
        grade = "unstable"

    meta = {
        "validation_score": validation_score,
        "analog_score": analog_score,
        "point_in_time_score": pit_score,
        "change_point_penalty": cp_penalty,
        "change_point_prob": cp,
    }
    return pd.DataFrame(
        [
            {
                "date": r["date"],
                "confidence": confidence,
                "grade": grade,
                "metadata_json": json.dumps(meta, sort_keys=True),
            }
        ]
    )
