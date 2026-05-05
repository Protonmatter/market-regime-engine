# SPDX-License-Identifier: Apache-2.0
"""Legacy v0.8 read-only API mount.

This surface predates the hardened ``api_v1`` mount and ships **no**
authentication. v1.2.1 closes the deployment-mistake loophole that let
operators run::

    uvicorn market_regime_engine.api:app

without realizing they were exposing the same governance / model-output
surface as ``/v1`` to the public internet.

Importing this module now requires the operator to set the env var
``MRE_LEGACY_API_ALLOW_UNAUTH=1``. Anything else raises ``RuntimeError``
at module import time, so a misconfigured uvicorn deployment fails fast
rather than silently serving artifacts.

Production deployments should mount ``market_regime_engine.api_v1:app``
which honors ``MRE_API_KEY``.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException

from market_regime_engine import __version__
from market_regime_engine.analogs import analog_summary
from market_regime_engine.explain import latest_explanation
from market_regime_engine.storage import Warehouse

_LEGACY_GATE_ENV = "MRE_LEGACY_API_ALLOW_UNAUTH"
_LEGACY_GATE_MESSAGE = (
    "legacy /api is unauthenticated and exposes governance artifacts. "
    "Mount market_regime_engine.api_v1:app instead, or set "
    f"{_LEGACY_GATE_ENV}=1 to acknowledge the security trade-off."
)


def _enforce_legacy_gate() -> None:
    """Raise unless the operator opts in to the unauthenticated mount.

    Triggered at module import time so a misconfigured ``uvicorn`` deploy
    fails immediately. The check looks at the environment when the module
    is loaded; tests can mutate ``MRE_LEGACY_API_ALLOW_UNAUTH`` and re-import
    via ``importlib.reload(market_regime_engine.api)``.
    """
    if os.getenv(_LEGACY_GATE_ENV) != "1":
        raise RuntimeError(_LEGACY_GATE_MESSAGE)


_enforce_legacy_gate()


app = FastAPI(title="Market Regime Engine (legacy)", version=__version__)


def wh() -> Warehouse:
    return Warehouse(os.getenv("MRE_DB_PATH", "data/mre.db"))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/regime/latest")
def latest_regime() -> dict:
    db = wh()
    try:
        out = latest_explanation(db.read_regimes())
        if out.get("status"):
            raise HTTPException(status_code=404, detail="No regime scores found")
        return out
    finally:
        db.close()


@app.get("/model-outputs/latest")
def latest_outputs() -> dict:
    db = wh()
    try:
        df = db.read_model_outputs()
        if df.empty:
            raise HTTPException(status_code=404, detail="No model outputs found")
        latest = df[df["date"] == df["date"].max()]
        return {"date": str(latest["date"].iloc[0]), "outputs": latest.to_dict(orient="records")}
    finally:
        db.close()


@app.get("/calibrated-outputs/latest")
def latest_calibrated_outputs() -> dict:
    db = wh()
    try:
        df = db.read_calibrated_outputs()
        if df.empty:
            raise HTTPException(status_code=404, detail="No calibrated outputs found")
        latest = df[df["date"] == df["date"].max()]
        return {"date": str(latest["date"].iloc[0]), "outputs": latest.to_dict(orient="records")}
    finally:
        db.close()


@app.get("/analogs/latest")
def latest_analogs() -> dict:
    db = wh()
    try:
        df = db.read_historical_analogs()
        if df.empty:
            raise HTTPException(status_code=404, detail="No analogs found")
        latest_date = df["as_of_date"].max()
        latest = df[df["as_of_date"] == latest_date]
        return {
            "as_of_date": latest_date,
            "summary": analog_summary(latest),
            "analogs": latest.to_dict(orient="records"),
        }
    finally:
        db.close()


@app.get("/attribution/latest")
def latest_attribution() -> dict:
    db = wh()
    try:
        df = db.read_driver_attribution()
        if df.empty:
            raise HTTPException(status_code=404, detail="No attribution found")
        latest_date = df["date"].max()
        latest = df[df["date"] == latest_date]
        return {"date": latest_date, "drivers": latest.to_dict(orient="records")}
    finally:
        db.close()


@app.get("/confidence/latest")
def latest_confidence() -> dict:
    db = wh()
    try:
        df = db.read_confidence_scores()
        if df.empty:
            raise HTTPException(status_code=404, detail="No confidence score found")
        latest = df[df["date"] == df["date"].max()]
        return latest.iloc[0].to_dict()
    finally:
        db.close()


@app.get("/invalidation/latest")
def latest_invalidation() -> dict:
    db = wh()
    try:
        df = db.read_invalidation_triggers()
        if df.empty:
            raise HTTPException(status_code=404, detail="No invalidation triggers found")
        latest_date = df["date"].max()
        latest = df[df["date"] == latest_date]
        return {"date": latest_date, "triggers": latest.to_dict(orient="records")}
    finally:
        db.close()


@app.get("/model-runs/latest")
def latest_model_run() -> dict:
    db = wh()
    try:
        df = db.read_model_runs()
        if df.empty:
            raise HTTPException(status_code=404, detail="No model runs found")
        return df.iloc[-1].to_dict()
    finally:
        db.close()


@app.get("/drift/latest")
def latest_drift() -> dict:
    db = wh()
    try:
        df = db.read_model_drift()
        if df.empty:
            raise HTTPException(status_code=404, detail="No drift data found")
        latest_date = df["date"].max()
        latest = df[df["date"] == latest_date]
        return {"date": latest_date, "drift": latest.to_dict(orient="records")}
    finally:
        db.close()


@app.get("/release-gate/latest")
def latest_release_gate() -> dict:
    db = wh()
    try:
        df = db.read_release_gates()
        if df.empty:
            raise HTTPException(status_code=404, detail="No release gate found")
        return df.iloc[-1].to_dict()
    finally:
        db.close()


@app.get("/ensemble-weights/latest")
def latest_ensemble_weights() -> dict:
    db = wh()
    try:
        df = db.read_ensemble_weights()
        if df.empty:
            raise HTTPException(status_code=404, detail="No ensemble weights found")
        return {"weights": df.to_dict(orient="records")}
    finally:
        db.close()


@app.get("/alerts/latest")
def latest_alerts() -> dict:
    db = wh()
    try:
        df = db.read_routed_alerts()
        if df.empty:
            raise HTTPException(status_code=404, detail="No routed alerts found")
        latest_date = df["date"].max()
        return {"date": latest_date, "alerts": df[df["date"] == latest_date].to_dict(orient="records")}
    finally:
        db.close()


@app.get("/promotion-workflow/latest")
def latest_promotion_workflow() -> dict:
    db = wh()
    try:
        df = db.read_promotion_workflow()
        if df.empty:
            raise HTTPException(status_code=404, detail="No promotion workflow found")
        return df.iloc[-1].to_dict()
    finally:
        db.close()


@app.get("/hazard/latest")
def latest_hazard() -> dict:
    db = wh()
    try:
        df = db.read_hazard_diagnostics()
        if df.empty:
            raise HTTPException(status_code=404, detail="No hazard diagnostics found")
        return df.iloc[-1].to_dict()
    finally:
        db.close()


@app.get("/vintage-audit/latest")
def latest_vintage_audit() -> dict:
    db = wh()
    try:
        df = db.read_vintage_audits()
        if df.empty:
            raise HTTPException(status_code=404, detail="No vintage audit found")
        latest_ts = df["run_at_utc"].max()
        return {"run_at_utc": latest_ts, "audits": df[df["run_at_utc"] == latest_ts].to_dict(orient="records")}
    finally:
        db.close()


@app.get("/feature-asof/latest")
def latest_feature_asof() -> dict:
    db = wh()
    try:
        df = db.read_feature_asof_values()
        if df.empty:
            raise HTTPException(status_code=404, detail="No feature as-of values found")
        latest_date = df["as_of_date"].max()
        latest = df[df["as_of_date"] == latest_date]
        return {"as_of_date": latest_date, "feature_count": len(latest), "features": latest.to_dict(orient="records")}
    finally:
        db.close()


@app.get("/vintage-observations/coverage")
def vintage_observation_coverage() -> dict:
    db = wh()
    try:
        df = db.read_vintage_observations()
        if df.empty:
            raise HTTPException(status_code=404, detail="No vintage observations found")
        return {
            "rows": len(df),
            "series": int(df["series_id"].nunique()),
            "min_observation_date": str(df["observation_date"].min()),
            "max_observation_date": str(df["observation_date"].max()),
            "min_vintage_date": str(df["vintage_date"].min()),
            "max_vintage_date": str(df["vintage_date"].max()),
        }
    finally:
        db.close()
