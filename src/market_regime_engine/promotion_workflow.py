# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json

import pandas as pd


def evaluate_promotion_workflow(
    *,
    promotion: pd.DataFrame | None = None,
    release_gate: pd.DataFrame | None = None,
    confidence: pd.DataFrame | None = None,
    drift: pd.DataFrame | None = None,
) -> pd.DataFrame:
    latest_date = pd.Timestamp.now("UTC").strftime("%Y-%m-%d")
    promoted = bool(
        promotion is not None
        and not promotion.empty
        and "promoted" in promotion
        and promotion["promoted"].fillna(False).any()
    )
    gate_ok = bool(
        release_gate is not None and not release_gate.empty and bool(release_gate.iloc[-1].get("approved", 0))
    )
    conf_val = None
    conf_ok = False
    if confidence is not None and not confidence.empty:
        latest_date = str(confidence.iloc[-1].get("date", latest_date))
        conf_val = float(confidence.iloc[-1].get("confidence", 0.0))
        conf_ok = conf_val >= 0.55
    drift_ok = True
    if drift is not None and not drift.empty and "status" in drift:
        drift_ok = not drift["status"].isin(["severe"]).any()

    approved = promoted and gate_ok and conf_ok and drift_ok
    reasons = []
    if not promoted:
        reasons.append("no_challenger_promoted")
    if not gate_ok:
        reasons.append("release_gate_not_approved")
    if not conf_ok:
        reasons.append("confidence_below_threshold")
    if not drift_ok:
        reasons.append("severe_feature_drift")
    return pd.DataFrame(
        [
            {
                "date": latest_date,
                "workflow": "champion_challenger_v0_7",
                "approved": bool(approved),
                "decision": "promote" if approved else "hold",
                "confidence": conf_val,
                "promoted_challenger_present": promoted,
                "release_gate_approved": gate_ok,
                "drift_ok": drift_ok,
                "reasons": ";".join(reasons) if reasons else "approved",
                "metadata_json": json.dumps(
                    {"gate_ok": gate_ok, "conf_ok": conf_ok, "drift_ok": drift_ok, "promoted": promoted}, sort_keys=True
                ),
            }
        ]
    )
