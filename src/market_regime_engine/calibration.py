# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class PlattCalibrator:
    intercept: float = 0.0
    slope: float = 1.0
    fallback_rate: float | None = None

    def fit(self, y: pd.Series, p: pd.Series) -> PlattCalibrator:
        frame = pd.concat([y.rename("y"), p.rename("p")], axis=1).dropna()
        if frame.empty:
            self.fallback_rate = 0.0
            return self
        yv = frame["y"].astype(int).to_numpy()
        pv = np.clip(frame["p"].astype(float).to_numpy(), EPS, 1.0 - EPS)
        if len(np.unique(yv)) < 2 or len(frame) < 20:
            self.fallback_rate = float(np.mean(yv))
            return self
        model = LogisticRegression(solver="lbfgs")
        model.fit(_logit(pv).reshape(-1, 1), yv)
        self.intercept = float(model.intercept_[0])
        self.slope = float(model.coef_[0][0])
        self.fallback_rate = None
        return self

    def transform(self, p: pd.Series | np.ndarray) -> np.ndarray:
        arr = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
        if self.fallback_rate is not None:
            return np.full(len(arr), float(self.fallback_rate))
        return _sigmoid(self.intercept + self.slope * _logit(arr))

    def to_json(self) -> str:
        return json.dumps(
            {
                "method": "platt_logit",
                "intercept": self.intercept,
                "slope": self.slope,
                "fallback_rate": self.fallback_rate,
            },
            sort_keys=True,
        )


def fit_calibrators_from_validation(validation_dir: str = "data/validation") -> pd.DataFrame:
    pd.io.common.stringify_path(validation_dir) + "/binary_predictions_3m.csv"
    rows = []
    for horizon in ("3m", "6m", "12m"):
        f = f"{validation_dir}/binary_predictions_{horizon}.csv"
        try:
            preds = pd.read_csv(f)
        except FileNotFoundError:
            continue
        if preds.empty or not {"y", "p"}.issubset(preds.columns):
            continue
        cal = PlattCalibrator().fit(preds["y"], preds["p"])
        raw = np.clip(preds["p"].astype(float).to_numpy(), EPS, 1.0 - EPS)
        calibrated = cal.transform(raw)
        rows.append(
            {
                "horizon": horizon,
                "target": "drawdown_gt_10pct",
                "method": "platt_logit",
                "intercept": cal.intercept,
                "slope": cal.slope,
                "fallback_rate": cal.fallback_rate,
                "observations": len(preds.dropna(subset=["y", "p"])),
                "raw_mean": float(np.mean(raw)) if len(raw) else math.nan,
                "calibrated_mean": float(np.mean(calibrated)) if len(calibrated) else math.nan,
                "metadata_json": cal.to_json(),
            }
        )
    return pd.DataFrame(rows)


def apply_binary_calibration(model_outputs: pd.DataFrame, calibrators: pd.DataFrame) -> pd.DataFrame:
    if model_outputs.empty or calibrators.empty:
        return pd.DataFrame(
            columns=model_outputs.columns
            if not model_outputs.empty
            else ["model_name", "date", "horizon", "target", "value", "metadata_json"]
        )
    rows = []
    for _, row in model_outputs.iterrows():
        if str(row.get("target")) != "drawdown_gt_10pct":
            continue
        c = calibrators[
            (calibrators["horizon"].astype(str) == str(row["horizon"]))
            & (calibrators["target"].astype(str) == str(row["target"]))
        ]
        if c.empty:
            continue
        cr = c.iloc[0]
        cal = PlattCalibrator(
            intercept=float(cr["intercept"]),
            slope=float(cr["slope"]),
            fallback_rate=None if pd.isna(cr.get("fallback_rate")) else float(cr.get("fallback_rate")),
        )
        val = float(cal.transform(np.array([float(row["value"])]))[0])
        meta = {
            "calibrated_from": row.get("model_name"),
            "calibration_method": "platt_logit",
            "raw_value": float(row["value"]),
        }
        rows.append(
            {
                "model_name": f"{row['model_name']}_calibrated",
                "date": row["date"],
                "horizon": row["horizon"],
                "target": row["target"],
                "value": val,
                "metadata_json": json.dumps(meta, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)
