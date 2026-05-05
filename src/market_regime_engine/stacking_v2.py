# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from market_regime_engine.stacking import optimize_binary_stacking


def load_oos_prediction_matrix(validation_dir: str | Path) -> pd.DataFrame:
    """Load candidate and benchmark OOS prediction CSVs from validate output."""
    root = Path(validation_dir)
    frames = []
    for path in sorted(root.glob("binary*_predictions_*m.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        if "model" in df and "model_name" not in df:
            df = df.rename(columns={"model": "model_name"})
        required = {"date", "model_name", "horizon", "target", "p"}
        if required.issubset(df.columns):
            frames.append(df.rename(columns={"p": "value"})[["date", "model_name", "horizon", "target", "value", "y"]])
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out


def _regime_bucket(regime: str) -> str:
    r = str(regime).lower()
    if "credit" in r:
        return "credit_stress"
    if "energy" in r or "inflation" in r or "stag" in r:
        return "inflation_energy"
    if "recession" in r or "bear" in r:
        return "recessionary"
    if "soft" in r:
        return "soft_landing"
    return "general"


def regime_conditioned_stacking(
    validation_dir: str | Path, regimes: pd.DataFrame | None = None, step: float = 0.1
) -> dict[str, pd.DataFrame]:
    """Optimize binary stacking weights by decoded regime bucket where possible."""
    oos = load_oos_prediction_matrix(validation_dir)
    if oos.empty:
        return {
            "oos_predictions": pd.DataFrame(),
            "ensemble_weights": pd.DataFrame(),
            "stacked_outputs": pd.DataFrame(),
            "stacking_diagnostics": pd.DataFrame(),
        }
    oos["regime_bucket"] = "general"
    if regimes is not None and not regimes.empty:
        reg = regimes.copy()
        reg["date"] = pd.to_datetime(reg["date"]).dt.strftime("%Y-%m-%d")
        reg["regime_bucket"] = reg["decoded_regime"].map(_regime_bucket)
        oos = oos.merge(reg[["date", "regime_bucket"]], on="date", how="left", suffixes=("", "_r"))
        oos["regime_bucket"] = oos["regime_bucket_r"].fillna(oos["regime_bucket"])
        oos = oos.drop(columns=[c for c in ["regime_bucket_r"] if c in oos])

    weights, preds, diags = [], [], []
    for (target, horizon, bucket), g in oos.groupby(["target", "horizon", "regime_bucket"], dropna=False):
        if g["y"].dropna().nunique() < 2 or g["model_name"].nunique() < 1 or len(g) < 20:
            continue
        realized = g.drop_duplicates("date").set_index("date")["y"].astype(float)
        res = optimize_binary_stacking(
            g.rename(columns={"regime_bucket": "metadata_regime_bucket"})[
                ["date", "model_name", "horizon", "target", "value"]
            ],
            realized,
            target=str(target),
            horizon=str(horizon),
            step=step,
        )
        if not res.weights.empty:
            w = res.weights.copy()
            w["method"] = "regime_grid_logloss"
            w["metadata_json"] = [json.dumps({"regime_bucket": bucket}, sort_keys=True)] * len(w)
            weights.append(w)
        if not res.predictions.empty:
            p = res.predictions.copy()
            p["model_name"] = f"stacked_binary_v0_7_{bucket}"
            p["metadata_json"] = [json.dumps({"regime_bucket": bucket}, sort_keys=True)] * len(p)
            preds.append(p)
        if not res.diagnostics.empty:
            d = res.diagnostics.copy()
            d["method"] = "regime_grid_logloss"
            d["metadata_json"] = [json.dumps({"regime_bucket": bucket}, sort_keys=True)] * len(d)
            diags.append(d)
    return {
        "oos_predictions": oos,
        "ensemble_weights": pd.concat(weights, ignore_index=True) if weights else pd.DataFrame(),
        "stacked_outputs": pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(),
        "stacking_diagnostics": pd.concat(diags, ignore_index=True) if diags else pd.DataFrame(),
    }
