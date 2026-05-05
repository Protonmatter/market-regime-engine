# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from market_regime_engine.baselines import (
    expanding_event_rate_baseline,
    expanding_quantile_baseline,
    previous_event_baseline,
)
from market_regime_engine.models import ProbabilityModel, QuantileReturnModel
from market_regime_engine.promotion import PromotionGate, best_benchmark
from market_regime_engine.validation import (
    pinball_loss,
    quantile_coverage,
    validate_binary_forecast,
    validation_frame,
)
from market_regime_engine.walk_forward import PurgedWalkForward, evaluate_walk_forward

_HORIZON_RE = re.compile(r"(\d+)")


def _parse_horizon_months(horizon: str, fallback: int = 1) -> int:
    """Extract the integer horizon from a label like ``"3m"`` / ``"12m"``.

    The walk-forward purge uses ``H`` to drop training rows whose forward
    target window overlaps the test point. We default to ``1`` only when the
    label is unparsable; callers always pass a labelled horizon in practice.
    """
    if horizon is None:
        return fallback
    match = _HORIZON_RE.search(str(horizon))
    if match is None:
        return fallback
    try:
        h = int(match.group(1))
    except ValueError:
        return fallback
    return max(h, 1)


def expanding_window_binary_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    target: str,
    horizon: str,
    min_train: int = 96,
    step: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Purged walk-forward binary model backtest.

    Wraps :class:`walk_forward.PurgedWalkForward` so the training rows whose
    forward target window overlaps the test point are excluded (López de
    Prado 2018, Chapter 7), then funnels the OOS predictions through
    :func:`evaluate_walk_forward`. The public signature is preserved so the
    CLI does not need to change.
    """
    H = _parse_horizon_months(horizon)
    splitter = PurgedWalkForward(
        min_train=min_train,
        step=step,
        horizon=H,
        embargo=1,
        expanding=True,
        test_block=1,
    )

    feature_cols = list(X.columns)

    def predict_fn(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
        train_mask = y_train.notna()
        if int(train_mask.sum()) == 0:
            return np.full(len(X_test), float("nan"))
        model = ProbabilityModel().fit(X_train.loc[train_mask], y_train.loc[train_mask])
        return np.asarray(model.predict_proba(X_test), dtype=float)

    oos = evaluate_walk_forward(
        X[feature_cols],
        y,
        splitter=splitter,
        predict_fn=predict_fn,
        target=target,
        horizon=horizon,
        model_name="candidate_logistic",
    )
    if oos.empty:
        empty = pd.DataFrame(columns=["date", "target", "horizon", "y", "p", "model"])
        return empty, pd.DataFrame()

    pred_frame = oos.rename(columns={"model_name": "model"})[["date", "target", "horizon", "y", "p", "model"]]
    val = validation_frame([validate_binary_forecast(target, horizon, pred_frame["y"], pred_frame["p"])])
    val["model"] = "candidate_logistic"
    return pred_frame, val


def binary_benchmark_report(
    y: pd.Series,
    *,
    target: str,
    horizon: str,
    min_train: int = 96,
    step: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [
        expanding_event_rate_baseline(y, min_train=min_train, step=step),
        previous_event_baseline(y, min_train=min_train, step=step),
    ]
    preds = []
    vals = []
    for frame in frames:
        if frame.empty:
            continue
        bench_name = str(frame["benchmark"].iloc[0])
        frame = frame.copy()
        frame["target"] = target
        frame["horizon"] = horizon
        frame["model"] = bench_name
        preds.append(frame.reset_index())
        v = validation_frame([validate_binary_forecast(target, horizon, frame["y"], frame["p"])])
        v["model"] = bench_name
        vals.append(v)
    return (
        pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(),
        pd.concat(vals, ignore_index=True) if vals else pd.DataFrame(),
    )


def expanding_window_quantile_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    target: str,
    horizon: str,
    min_train: int = 120,
    step: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Purged walk-forward quantile-regression backtest.

    Mirrors :func:`expanding_window_binary_backtest`: build a purged walk-
    forward splitter for ``H`` periods (parsed from ``horizon``), train a
    fresh ``QuantileReturnModel`` on each fold's training slice, predict on
    the test slice, and emit the per-fold quantile frame.
    """
    H = _parse_horizon_months(horizon)
    splitter = PurgedWalkForward(
        min_train=min_train,
        step=step,
        horizon=H,
        embargo=1,
        expanding=True,
        test_block=1,
    )

    feature_cols = list(X.columns)
    quantile_cols = ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]

    aligned = X[feature_cols].join(y.rename("__y__"), how="inner").sort_index()
    n = len(aligned)
    rows: list[dict] = []
    for split in splitter.split(n):
        train = aligned.iloc[split.train_idx]
        test = aligned.iloc[split.test_idx]
        if train.empty or test.empty:
            continue
        train_mask = train["__y__"].notna()
        if int(train_mask.sum()) == 0:
            continue
        model = QuantileReturnModel().fit(train.loc[train_mask, feature_cols], train.loc[train_mask, "__y__"])
        preds = model.predict(test[feature_cols])
        for date, y_true in zip(test.index, test["__y__"].to_numpy(float), strict=False):
            row: dict[str, object] = {
                "fold": split.fold,
                "date": date,
                "target": target,
                "horizon": horizon,
                "model": "candidate_quantile_gbr",
                "y": float(y_true) if pd.notna(y_true) else float("nan"),
            }
            row.update({col: float(preds[col].loc[date]) for col in preds.columns if col in quantile_cols})
            rows.append(row)

    pred_frame = pd.DataFrame(rows)
    if pred_frame.empty:
        return pred_frame, pd.DataFrame()
    pred_frame = pred_frame.sort_values("date").reset_index(drop=True)
    pred_frame_indexed = pred_frame.set_index("date")
    metrics = _quantile_metrics(pred_frame_indexed, target, horizon, "candidate_quantile_gbr")
    return pred_frame, metrics


def quantile_benchmark_report(
    y: pd.Series,
    *,
    target: str,
    horizon: str,
    min_train: int = 120,
    step: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_frame = expanding_quantile_baseline(y, min_train=min_train, step=step)
    if pred_frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    pred_frame = pred_frame.copy()
    pred_frame["target"] = target
    pred_frame["horizon"] = horizon
    pred_frame["model"] = "expanding_return_quantiles"
    metrics = _quantile_metrics(pred_frame, target, horizon, "expanding_return_quantiles")
    return pred_frame.reset_index(), metrics


def _quantile_metrics(pred_frame: pd.DataFrame, target: str, horizon: str, model: str) -> pd.DataFrame:
    metrics = []
    quantile_map = {"q05": 0.05, "q25": 0.25, "q50": 0.50, "q75": 0.75, "q95": 0.95}
    for col, tau in quantile_map.items():
        if not pred_frame.empty and col in pred_frame:
            metrics.append(
                {
                    "model": model,
                    "target": target,
                    "horizon": horizon,
                    "quantile": tau,
                    "pinball_loss": pinball_loss(pred_frame["y"], pred_frame[col], tau),
                    "coverage": quantile_coverage(pred_frame["y"], pred_frame[col]),
                    "coverage_error": abs(quantile_coverage(pred_frame["y"], pred_frame[col]) - tau),
                    "observations": int(pred_frame[["y", col]].dropna().shape[0]),
                }
            )
    return pd.DataFrame(metrics)


def benchmark_report(
    X: pd.DataFrame,
    targets: pd.DataFrame,
    min_train: int = 120,
    step: int = 6,
) -> dict[str, pd.DataFrame]:
    """Run candidate and naive benchmark walk-forward checks for MVP targets."""
    reports: dict[str, pd.DataFrame] = {}
    binary_vals = []
    binary_bench_vals = []
    quant_vals = []
    quant_bench_vals = []

    for h in (3, 6, 12):
        pred, val = expanding_window_binary_backtest(
            X,
            targets[f"dd10_{h}m"],
            target="drawdown_gt_10pct",
            horizon=f"{h}m",
            min_train=min_train,
            step=step,
        )
        reports[f"binary_predictions_{h}m"] = pred
        binary_vals.append(val)

        bpred, bval = binary_benchmark_report(
            targets[f"dd10_{h}m"],
            target="drawdown_gt_10pct",
            horizon=f"{h}m",
            min_train=min_train,
            step=step,
        )
        reports[f"binary_benchmark_predictions_{h}m"] = bpred
        binary_bench_vals.append(bval)

        qpred, qval = expanding_window_quantile_backtest(
            X,
            targets[f"ret_{h}m"],
            target="forward_return_log",
            horizon=f"{h}m",
            min_train=min_train,
            step=step,
        )
        reports[f"quantile_predictions_{h}m"] = qpred
        quant_vals.append(qval)

        qbpred, qbval = quantile_benchmark_report(
            targets[f"ret_{h}m"],
            target="forward_return_log",
            horizon=f"{h}m",
            min_train=min_train,
            step=step,
        )
        reports[f"quantile_benchmark_predictions_{h}m"] = qbpred
        quant_bench_vals.append(qbval)

    candidate_binary = pd.concat(binary_vals, ignore_index=True) if binary_vals else pd.DataFrame()
    benchmark_binary = pd.concat(binary_bench_vals, ignore_index=True) if binary_bench_vals else pd.DataFrame()
    best_binary = best_benchmark(benchmark_binary) if not benchmark_binary.empty else benchmark_binary
    promotion = (
        PromotionGate().evaluate_binary(candidate_binary, best_binary)
        if not candidate_binary.empty and not best_binary.empty
        else pd.DataFrame()
    )

    reports["binary_validation"] = candidate_binary
    reports["binary_benchmark_validation"] = benchmark_binary
    reports["binary_best_benchmark"] = best_binary
    reports["model_promotion"] = promotion
    reports["quantile_validation"] = pd.concat(quant_vals, ignore_index=True) if quant_vals else pd.DataFrame()
    reports["quantile_benchmark_validation"] = (
        pd.concat(quant_bench_vals, ignore_index=True) if quant_bench_vals else pd.DataFrame()
    )
    return reports


def dataframe_to_json_records(df: pd.DataFrame) -> str:
    def clean(value: object) -> object:
        if isinstance(value, (np.floating, float)) and not np.isfinite(value):
            return None
        if isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m-%d")
        return value

    records = [{k: clean(v) for k, v in row.items()} for row in df.to_dict(orient="records")]
    return json.dumps(records, indent=2, sort_keys=True)


__all__ = [
    "benchmark_report",
    "binary_benchmark_report",
    "dataframe_to_json_records",
    "expanding_window_binary_backtest",
    "expanding_window_quantile_backtest",
    "quantile_benchmark_report",
]
