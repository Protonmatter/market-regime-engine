# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json

import pandas as pd


def route_alerts(
    *,
    release_gates: pd.DataFrame | None = None,
    drift: pd.DataFrame | None = None,
    invalidation: pd.DataFrame | None = None,
    confidence: pd.DataFrame | None = None,
    promotion: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    latest_date = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")

    if release_gates is not None and not release_gates.empty:
        row = release_gates.iloc[-1]
        latest_date = str(row.get("date", latest_date))
        if not bool(row.get("approved", 0)):
            rows.append(
                {
                    "date": latest_date,
                    "alert_type": "release_gate_hold",
                    "severity": "high",
                    "channel": "model_risk",
                    "message": f"Release gate held: {row.get('reasons', '')}",
                    "metadata_json": json.dumps(row.to_dict(), default=str),
                }
            )

    if drift is not None and not drift.empty:
        major = drift[drift["status"].isin(["major", "severe"])] if "status" in drift else pd.DataFrame()
        if not major.empty:
            rows.append(
                {
                    "date": latest_date,
                    "alert_type": "feature_drift",
                    "severity": "medium",
                    "channel": "quant_research",
                    "message": f"{len(major)} features show major/severe PSI drift",
                    "metadata_json": json.dumps({"features": major.head(20).to_dict(orient="records")}, default=str),
                }
            )

    if invalidation is not None and not invalidation.empty:
        active = (
            invalidation[(invalidation.get("status", "") == "active") & (invalidation.get("severity", "") == "high")]
            if {"status", "severity"}.issubset(invalidation.columns)
            else pd.DataFrame()
        )
        if not active.empty:
            rows.append(
                {
                    "date": latest_date,
                    "alert_type": "forecast_invalidation",
                    "severity": "high",
                    "channel": "portfolio_risk",
                    "message": f"High severity invalidation triggers active: {', '.join(active['trigger'].astype(str).head(10))}",
                    "metadata_json": json.dumps({"triggers": active.to_dict(orient="records")}, default=str),
                }
            )

    if confidence is not None and not confidence.empty:
        c = confidence.iloc[-1]
        val = float(c.get("confidence", 1.0))
        if val < 0.55:
            rows.append(
                {
                    "date": str(c.get("date", latest_date)),
                    "alert_type": "low_model_confidence",
                    "severity": "medium",
                    "channel": "model_risk",
                    "message": f"Model confidence below release threshold: {val:.2f}",
                    "metadata_json": json.dumps(c.to_dict(), default=str),
                }
            )

    if (
        promotion is not None
        and not promotion.empty
        and "promoted" in promotion
        and not bool(promotion["promoted"].fillna(False).any())
    ):
        rows.append(
            {
                "date": latest_date,
                "alert_type": "no_promoted_challenger",
                "severity": "low",
                "channel": "quant_research",
                "message": "No challenger model passed promotion gates.",
                "metadata_json": json.dumps({"rows": promotion.tail(10).to_dict(orient="records")}, default=str),
            }
        )

    if not rows:
        rows.append(
            {
                "date": latest_date,
                "alert_type": "all_clear",
                "severity": "info",
                "channel": "model_risk",
                "message": "No active model-risk alerts.",
                "metadata_json": "{}",
            }
        )
    return pd.DataFrame(rows)
