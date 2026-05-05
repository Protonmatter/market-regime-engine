# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 1e-9


def _clip_prob(p: np.ndarray | pd.Series) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def brier_score(y_true: Iterable[float], p_pred: Iterable[float]) -> float:
    """Mean squared probability error for binary outcomes."""
    y = np.asarray(list(y_true), dtype=float)
    p = _clip_prob(np.asarray(list(p_pred), dtype=float))
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean((y[mask] - p[mask]) ** 2))


def log_loss_score(y_true: Iterable[float], p_pred: Iterable[float]) -> float:
    """Binary log loss, clipped for numerical stability."""
    y = np.asarray(list(y_true), dtype=float)
    p = _clip_prob(np.asarray(list(p_pred), dtype=float))
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() == 0:
        return float("nan")
    return float(-np.mean(y[mask] * np.log(p[mask]) + (1.0 - y[mask]) * np.log(1.0 - p[mask])))


def calibration_table(y_true: Iterable[float], p_pred: Iterable[float], bins: int = 10) -> pd.DataFrame:
    """Reliability table: predicted probability bucket vs realized frequency."""
    y = np.asarray(list(y_true), dtype=float)
    p = _clip_prob(np.asarray(list(p_pred), dtype=float))
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() == 0:
        return pd.DataFrame(columns=["bin", "count", "pred_mean", "actual_rate", "calibration_error"])
    frame = pd.DataFrame({"y": y[mask], "p": p[mask]})
    frame["bin"] = pd.cut(frame["p"], np.linspace(0, 1, bins + 1), include_lowest=True, duplicates="drop")
    out = (
        frame.groupby("bin", observed=True)
        .agg(count=("y", "size"), pred_mean=("p", "mean"), actual_rate=("y", "mean"))
        .reset_index()
    )
    out["calibration_error"] = out["actual_rate"] - out["pred_mean"]
    out["bin"] = out["bin"].astype(str)
    return out


def expected_calibration_error(y_true: Iterable[float], p_pred: Iterable[float], bins: int = 10) -> float:
    table = calibration_table(y_true, p_pred, bins=bins)
    if table.empty or table["count"].sum() == 0:
        return float("nan")
    weights = table["count"] / table["count"].sum()
    return float(np.sum(weights * table["calibration_error"].abs()))


def pinball_loss(y_true: Iterable[float], q_pred: Iterable[float], tau: float) -> float:
    """Quantile/pinball loss for return quantile forecasts."""
    y = np.asarray(list(y_true), dtype=float)
    q = np.asarray(list(q_pred), dtype=float)
    mask = np.isfinite(y) & np.isfinite(q)
    if mask.sum() == 0:
        return float("nan")
    e = y[mask] - q[mask]
    return float(np.mean(np.maximum(tau * e, (tau - 1.0) * e)))


def quantile_coverage(y_true: Iterable[float], q_pred: Iterable[float]) -> float:
    """Observed share of outcomes below the predicted quantile."""
    y = np.asarray(list(y_true), dtype=float)
    q = np.asarray(list(q_pred), dtype=float)
    mask = np.isfinite(y) & np.isfinite(q)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(y[mask] <= q[mask]))


@dataclass(frozen=True)
class BinaryValidationResult:
    target: str
    horizon: str
    observations: int
    event_rate: float
    brier: float
    log_loss: float
    ece: float


def validate_binary_forecast(target: str, horizon: str, y_true: pd.Series, p_pred: pd.Series) -> BinaryValidationResult:
    aligned = pd.concat([y_true.rename("y"), p_pred.rename("p")], axis=1).dropna()
    if aligned.empty:
        return BinaryValidationResult(target, horizon, 0, float("nan"), float("nan"), float("nan"), float("nan"))
    return BinaryValidationResult(
        target=target,
        horizon=horizon,
        observations=len(aligned),
        event_rate=float(aligned["y"].mean()),
        brier=brier_score(aligned["y"], aligned["p"]),
        log_loss=log_loss_score(aligned["y"], aligned["p"]),
        ece=expected_calibration_error(aligned["y"], aligned["p"]),
    )


def validation_frame(results: list[BinaryValidationResult]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in results])
