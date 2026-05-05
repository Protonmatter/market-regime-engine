# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 1e-6


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p.astype(float), EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((y.astype(float) - p.astype(float)) ** 2))


def _simplex_grid(n: int, step: float = 0.1) -> list[np.ndarray]:
    if n == 1:
        return [np.array([1.0])]
    units = round(1.0 / step)
    out = []
    for cuts in itertools.product(range(units + 1), repeat=n - 1):
        s = sum(cuts)
        if s <= units:
            vals = [*list(cuts), units - s]
            out.append(np.array(vals, dtype=float) / units)
    return out


@dataclass
class StackingResult:
    horizon: str
    target: str
    weights: pd.DataFrame
    predictions: pd.DataFrame
    diagnostics: pd.DataFrame


def optimize_binary_stacking(
    predictions: pd.DataFrame, realized: pd.DataFrame | pd.Series, target: str, horizon: str, step: float = 0.1
) -> StackingResult:
    """Constrained stacking optimizer over a simplex grid.

    predictions columns: date, model_name, horizon, target, value.
    realized: date-indexed Series or frame with target column.
    """
    preds = predictions.copy()
    preds = preds[(preds["target"] == target) & (preds["horizon"] == horizon)]
    if preds.empty:
        empty = pd.DataFrame()
        return StackingResult(horizon, target, empty, empty, empty)
    piv = preds.pivot_table(index="date", columns="model_name", values="value", aggfunc="last").sort_index()
    if isinstance(realized, pd.DataFrame):
        if target in realized.columns:
            y = realized[target].copy()
        else:
            y = realized.iloc[:, 0].copy()
    else:
        y = realized.copy()
    y.index = pd.to_datetime(y.index).strftime("%Y-%m-%d")
    joined = piv.join(y.rename("actual"), how="inner").dropna()
    if joined.empty or joined["actual"].nunique() < 2:
        models = list(piv.columns)
        w = np.ones(len(models)) / max(1, len(models))
        weights = pd.DataFrame(
            [
                {
                    "horizon": horizon,
                    "target": target,
                    "model_name": m,
                    "weight": float(wi),
                    "method": "uniform_fallback",
                }
                for m, wi in zip(models, w, strict=False)
            ]
        )
        return StackingResult(horizon, target, weights, pd.DataFrame(), pd.DataFrame())

    models = [c for c in joined.columns if c != "actual"]
    X = joined[models].clip(EPS, 1 - EPS).to_numpy(float)
    yy = joined["actual"].to_numpy(float)
    best_w = None
    best_loss = float("inf")
    for w in _simplex_grid(len(models), step=step):
        p = X @ w
        loss = _log_loss(yy, p) + 0.02 * float(np.sum(w**2))
        if loss < best_loss:
            best_loss = loss
            best_w = w
    if best_w is None:
        best_w = np.ones(len(models)) / len(models)
    ensemble_p = X @ best_w
    weights = pd.DataFrame(
        [
            {"horizon": horizon, "target": target, "model_name": m, "weight": float(wi), "method": "grid_logloss"}
            for m, wi in zip(models, best_w, strict=False)
        ]
    )
    pred_out = pd.DataFrame(
        {
            "date": joined.index,
            "model_name": "stacked_binary_v0_6",
            "horizon": horizon,
            "target": target,
            "value": ensemble_p,
            "metadata_json": "{}",
        }
    )
    diagnostics = pd.DataFrame(
        [
            {
                "horizon": horizon,
                "target": target,
                "observations": len(joined),
                "log_loss": _log_loss(yy, ensemble_p),
                "brier": _brier(yy, ensemble_p),
                "model_count": len(models),
                "method": "grid_logloss",
            }
        ]
    )
    return StackingResult(horizon, target, weights, pred_out, diagnostics)


def optimize_from_model_outputs(
    outputs: pd.DataFrame, targets: pd.DataFrame, target_map: dict[str, str] | None = None, step: float = 0.1
) -> dict[str, pd.DataFrame]:
    if outputs is None or outputs.empty or targets is None or targets.empty:
        return {
            "ensemble_weights": pd.DataFrame(),
            "stacked_outputs": pd.DataFrame(),
            "stacking_diagnostics": pd.DataFrame(),
        }
    target_map = target_map or {"drawdown_gt_10pct": "drawdown_gt_10pct", "recession_probability": "recession_next_12m"}
    weights, preds, diags = [], [], []
    for out_target, realized_col in target_map.items():
        if realized_col not in targets.columns:
            continue
        for horizon in sorted(outputs.loc[outputs["target"] == out_target, "horizon"].dropna().unique()):
            res = optimize_binary_stacking(outputs, targets[realized_col], out_target, str(horizon), step=step)
            if not res.weights.empty:
                weights.append(res.weights)
            if not res.predictions.empty:
                preds.append(res.predictions)
            if not res.diagnostics.empty:
                diags.append(res.diagnostics)
    return {
        "ensemble_weights": pd.concat(weights, ignore_index=True) if weights else pd.DataFrame(),
        "stacked_outputs": pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(),
        "stacking_diagnostics": pd.concat(diags, ignore_index=True) if diags else pd.DataFrame(),
    }
