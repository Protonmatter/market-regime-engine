# SPDX-License-Identifier: Apache-2.0
"""Prediction-evidence harness for market-regime forecasts.

The engine already has serious probabilistic pieces: PIT lineage, purged
walk-forward validation, conformal prediction, release gates, and forecast
comparison statistics. This module is the missing "prove it" layer.

It turns out-of-sample prediction tables into an audit-friendly evidence report
with hard pass/fail rails. The goal is not to make another clever model. The goal
is to make a model survive contact with calibration, tails, regimes, and
baseline comparisons without hiding behind a chart.

Expected binary prediction columns
----------------------------------
Required:
    date, target, horizon, model_name, y, p

Optional:
    regime, benchmark_p, change_point_prob

Expected quantile prediction columns
------------------------------------
Required:
    date, target, horizon, model_name, y

Supported interval schemas:
    q_lo, q_hi
    q05, q95
    q10, q90

Optional:
    regime, benchmark_q_lo, benchmark_q_hi

All functions are pure pandas/numpy and intentionally dependency-light so the
harness can run in CI, change-management jobs, and stripped production shells.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from market_regime_engine.forecast_compare import diebold_mariano, murphy_decomposition
from market_regime_engine.validation import (
    brier_score,
    expected_calibration_error,
    log_loss_score,
    pinball_loss,
)

EPS = 1e-9


@dataclass(frozen=True)
class EvidenceThresholds:
    """Hard rails for prediction-evidence release checks.

    These defaults are intentionally conservative but not delusional. They are
    meant to be tightened after the first real validation baseline is measured.
    """

    min_observations: int = 60
    max_brier: float = 0.25
    max_log_loss: float = 0.75
    max_ece: float = 0.08
    max_regime_ece: float = 0.12
    min_interval_coverage: float = 0.85
    max_interval_width_to_baseline: float = 1.25
    min_dm_pvalue_for_challenger_claim: float = 0.05
    max_tail_miss_rate: float = 0.20


@dataclass(frozen=True)
class EvidenceCheck:
    name: str
    passed: bool
    value: float | str | None
    threshold: float | str | None
    severity: str = "info"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceReport:
    approved: bool
    decision: str
    summary: dict[str, Any]
    checks: list[EvidenceCheck]
    binary_metrics: list[dict[str, Any]]
    quantile_metrics: list[dict[str, Any]]
    regime_metrics: list[dict[str, Any]]
    tail_metrics: list[dict[str, Any]]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(c) for c in self.checks])

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": bool(self.approved),
            "decision": self.decision,
            "summary": self.summary,
            "checks": [asdict(c) for c in self.checks],
            "binary_metrics": self.binary_metrics,
            "quantile_metrics": self.quantile_metrics,
            "regime_metrics": self.regime_metrics,
            "tail_metrics": self.tail_metrics,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)

    def to_markdown(self) -> str:
        status = "APPROVED" if self.approved else "HOLD"
        lines = [
            f"# Prediction Evidence Report — {status}",
            "",
            "## Summary",
            "",
        ]
        for key, value in self.summary.items():
            lines.append(f"- **{key}:** {value}")
        lines.extend(["", "## Gate checks", ""])
        if self.checks:
            lines.append("| Check | Status | Value | Threshold | Severity |")
            lines.append("|---|---:|---:|---:|---|")
            for check in self.checks:
                icon = "PASS" if check.passed else "FAIL"
                lines.append(f"| {check.name} | {icon} | {check.value} | {check.threshold} | {check.severity} |")
        else:
            lines.append("_No checks generated._")
        lines.extend(["", "## Binary forecast metrics", ""])
        lines.extend(_markdown_table(self.binary_metrics))
        lines.extend(["", "## Quantile / interval metrics", ""])
        lines.extend(_markdown_table(self.quantile_metrics))
        lines.extend(["", "## Regime-sliced calibration", ""])
        lines.extend(_markdown_table(self.regime_metrics))
        lines.extend(["", "## Tail-risk diagnostics", ""])
        lines.extend(_markdown_table(self.tail_metrics))
        return "\n".join(lines).rstrip() + "\n"


def _read_frame(value: pd.DataFrame | str | Path | None) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _require_columns(frame: pd.DataFrame, columns: set[str], context: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")


def _finite_pair(a: pd.Series, b: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(a, errors="coerce").to_numpy(float)
    y = pd.to_numeric(b, errors="coerce").to_numpy(float)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _fmt_float(value: Any) -> float | str | None:
    val = _safe_float(value)
    if math.isnan(val):
        return "nan"
    return round(val, 6)


def _markdown_table(rows: list[dict[str, Any]], max_rows: int = 25) -> list[str]:
    if not rows:
        return ["_No rows._"]
    cols: list[str] = []
    for row in rows[:max_rows]:
        for key in row:
            if key not in cols:
                cols.append(key)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in rows[:max_rows]:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    if len(rows) > max_rows:
        lines.append(f"\n_Trimmed: showing {max_rows} of {len(rows)} rows._")
    return lines


def binary_forecast_evidence(
    predictions: pd.DataFrame | str | Path | None,
    *,
    thresholds: EvidenceThresholds | None = None,
) -> tuple[list[dict[str, Any]], list[EvidenceCheck], list[dict[str, Any]], list[dict[str, Any]]]:
    """Score binary probability forecasts.

    Returns ``(metrics, checks, regime_metrics, tail_metrics)``.
    """

    thresholds = thresholds or EvidenceThresholds()
    frame = _read_frame(predictions)
    if frame.empty:
        return [], [], [], []
    _require_columns(frame, {"target", "horizon", "model_name", "y", "p"}, "binary predictions")
    frame = frame.copy()
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["p"] = pd.to_numeric(frame["p"], errors="coerce").clip(EPS, 1 - EPS)
    frame["y"] = pd.to_numeric(frame["y"], errors="coerce")

    metrics: list[dict[str, Any]] = []
    checks: list[EvidenceCheck] = []
    regime_metrics: list[dict[str, Any]] = []
    tail_metrics: list[dict[str, Any]] = []

    group_cols = ["target", "horizon", "model_name"]
    for keys, group in frame.dropna(subset=["y", "p"]).groupby(group_cols, observed=True):
        target, horizon, model_name = (str(x) for x in keys)
        y, p = _finite_pair(group["y"], group["p"])
        n = len(y)
        if n == 0:
            continue
        brier = brier_score(y, p)
        logloss = log_loss_score(y, p)
        ece = expected_calibration_error(y, p, bins=10)
        murphy = murphy_decomposition(y, p, bins=10)
        event_rate = float(np.mean(y))
        row = {
            "target": target,
            "horizon": horizon,
            "model_name": model_name,
            "observations": n,
            "event_rate": _fmt_float(event_rate),
            "brier": _fmt_float(brier),
            "log_loss": _fmt_float(logloss),
            "ece": _fmt_float(ece),
            "murphy_reliability": _fmt_float(murphy.get("reliability")),
            "murphy_resolution": _fmt_float(murphy.get("resolution")),
            "murphy_uncertainty": _fmt_float(murphy.get("uncertainty")),
            "mean_probability": _fmt_float(np.mean(p)),
            "probability_std": _fmt_float(np.std(p)),
        }
        if "benchmark_p" in group:
            _, bench = _finite_pair(group["y"], group["benchmark_p"])
            if len(bench) == len(p):
                challenger_loss = (y - p) ** 2
                benchmark_loss = (y - bench) ** 2
                dm = diebold_mariano(challenger_loss, benchmark_loss, h=_horizon_to_periods(horizon))
                row["dm_pvalue_vs_benchmark"] = _fmt_float(dm.pvalue)
                row["dm_direction"] = dm.direction
        metrics.append(row)

        checks.extend(
            [
                EvidenceCheck(
                    f"{target}/{horizon}/{model_name}: min observations",
                    n >= thresholds.min_observations,
                    n,
                    thresholds.min_observations,
                    severity="blocker",
                ),
                EvidenceCheck(
                    f"{target}/{horizon}/{model_name}: Brier",
                    brier <= thresholds.max_brier,
                    _fmt_float(brier),
                    thresholds.max_brier,
                    severity="blocker",
                ),
                EvidenceCheck(
                    f"{target}/{horizon}/{model_name}: log loss",
                    logloss <= thresholds.max_log_loss,
                    _fmt_float(logloss),
                    thresholds.max_log_loss,
                    severity="blocker",
                ),
                EvidenceCheck(
                    f"{target}/{horizon}/{model_name}: ECE",
                    ece <= thresholds.max_ece,
                    _fmt_float(ece),
                    thresholds.max_ece,
                    severity="blocker",
                ),
            ]
        )

        if "regime" in group:
            for regime, rg in group.groupby("regime", observed=True):
                ry, rp = _finite_pair(rg["y"], rg["p"])
                if len(ry) == 0:
                    continue
                rg_ece = expected_calibration_error(ry, rp, bins=min(10, max(2, len(ry) // 10)))
                regime_metrics.append(
                    {
                        "target": target,
                        "horizon": horizon,
                        "model_name": model_name,
                        "regime": str(regime),
                        "observations": len(ry),
                        "event_rate": _fmt_float(np.mean(ry)),
                        "ece": _fmt_float(rg_ece),
                        "brier": _fmt_float(brier_score(ry, rp)),
                    }
                )
                if len(ry) >= max(20, thresholds.min_observations // 3):
                    checks.append(
                        EvidenceCheck(
                            f"{target}/{horizon}/{model_name}/regime={regime}: ECE",
                            rg_ece <= thresholds.max_regime_ece,
                            _fmt_float(rg_ece),
                            thresholds.max_regime_ece,
                            severity="major",
                        )
                    )

        tail_cut = np.quantile(p, 0.90) if len(p) >= 10 else 1.0
        high_risk = group[group["p"] >= tail_cut]
        if not high_risk.empty:
            hy = pd.to_numeric(high_risk["y"], errors="coerce").dropna().to_numpy(float)
            if hy.size:
                # For high-risk binary predictions, a "tail miss" means the model
                # issued a top-decile warning and the event did not occur.
                false_alarm_rate = float(np.mean(hy < 0.5))
                tail_metrics.append(
                    {
                        "target": target,
                        "horizon": horizon,
                        "model_name": model_name,
                        "slice": "top_decile_probability",
                        "observations": int(hy.size),
                        "mean_realized": _fmt_float(np.mean(hy)),
                        "false_alarm_rate": _fmt_float(false_alarm_rate),
                    }
                )

    return metrics, checks, regime_metrics, tail_metrics


def quantile_forecast_evidence(
    predictions: pd.DataFrame | str | Path | None,
    *,
    thresholds: EvidenceThresholds | None = None,
) -> tuple[list[dict[str, Any]], list[EvidenceCheck], list[dict[str, Any]]]:
    """Score interval / quantile forecasts."""

    thresholds = thresholds or EvidenceThresholds()
    frame = _read_frame(predictions)
    if frame.empty:
        return [], [], []
    _require_columns(frame, {"target", "horizon", "model_name", "y"}, "quantile predictions")
    frame = frame.copy()
    if "date" in frame:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")

    lo_col, hi_col = _detect_interval_columns(frame)
    if lo_col is None or hi_col is None:
        raise ValueError("quantile predictions need one interval pair: q_lo/q_hi, q05/q95, or q10/q90")

    metrics: list[dict[str, Any]] = []
    checks: list[EvidenceCheck] = []
    tail_metrics: list[dict[str, Any]] = []

    for keys, group in frame.dropna(subset=["y", lo_col, hi_col]).groupby(
        ["target", "horizon", "model_name"], observed=True
    ):
        target, horizon, model_name = (str(x) for x in keys)
        y = pd.to_numeric(group["y"], errors="coerce").to_numpy(float)
        lo = pd.to_numeric(group[lo_col], errors="coerce").to_numpy(float)
        hi = pd.to_numeric(group[hi_col], errors="coerce").to_numpy(float)
        mask = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
        y, lo, hi = y[mask], lo[mask], hi[mask]
        if len(y) == 0:
            continue
        lo2 = np.minimum(lo, hi)
        hi2 = np.maximum(lo, hi)
        covered = (y >= lo2) & (y <= hi2)
        coverage = float(np.mean(covered))
        width = float(np.mean(hi2 - lo2))
        row = {
            "target": target,
            "horizon": horizon,
            "model_name": model_name,
            "interval": f"{lo_col}/{hi_col}",
            "observations": len(y),
            "coverage": _fmt_float(coverage),
            "mean_width": _fmt_float(width),
            "tail_miss_rate": _fmt_float(1.0 - coverage),
        }
        if "q50" in group:
            q50 = pd.to_numeric(group.loc[mask, "q50"], errors="coerce").to_numpy(float)
            row["median_pinball"] = _fmt_float(pinball_loss(y, q50, 0.50))
        if {"benchmark_q_lo", "benchmark_q_hi"}.issubset(group.columns):
            blo = pd.to_numeric(group.loc[mask, "benchmark_q_lo"], errors="coerce").to_numpy(float)
            bhi = pd.to_numeric(group.loc[mask, "benchmark_q_hi"], errors="coerce").to_numpy(float)
            bwidth = float(np.nanmean(np.maximum(blo, bhi) - np.minimum(blo, bhi)))
            if math.isfinite(bwidth) and bwidth > EPS:
                row["width_to_benchmark"] = _fmt_float(width / bwidth)
                checks.append(
                    EvidenceCheck(
                        f"{target}/{horizon}/{model_name}: interval width vs benchmark",
                        width / bwidth <= thresholds.max_interval_width_to_baseline,
                        _fmt_float(width / bwidth),
                        thresholds.max_interval_width_to_baseline,
                        severity="major",
                    )
                )
        metrics.append(row)
        checks.extend(
            [
                EvidenceCheck(
                    f"{target}/{horizon}/{model_name}: interval observations",
                    len(y) >= thresholds.min_observations,
                    len(y),
                    thresholds.min_observations,
                    severity="blocker",
                ),
                EvidenceCheck(
                    f"{target}/{horizon}/{model_name}: interval coverage",
                    coverage >= thresholds.min_interval_coverage,
                    _fmt_float(coverage),
                    thresholds.min_interval_coverage,
                    severity="blocker",
                ),
                EvidenceCheck(
                    f"{target}/{horizon}/{model_name}: tail miss rate",
                    1.0 - coverage <= thresholds.max_tail_miss_rate,
                    _fmt_float(1.0 - coverage),
                    thresholds.max_tail_miss_rate,
                    severity="major",
                ),
            ]
        )
        lower_miss = float(np.mean(y < lo2))
        upper_miss = float(np.mean(y > hi2))
        tail_metrics.append(
            {
                "target": target,
                "horizon": horizon,
                "model_name": model_name,
                "slice": "interval_misses",
                "observations": len(y),
                "lower_tail_miss": _fmt_float(lower_miss),
                "upper_tail_miss": _fmt_float(upper_miss),
            }
        )

    return metrics, checks, tail_metrics


def build_prediction_evidence_report(
    *,
    binary_predictions: pd.DataFrame | str | Path | None = None,
    quantile_predictions: pd.DataFrame | str | Path | None = None,
    thresholds: EvidenceThresholds | None = None,
) -> EvidenceReport:
    """Build a full prediction-evidence report from OOS prediction tables."""

    thresholds = thresholds or EvidenceThresholds()
    binary_metrics, binary_checks, regime_metrics, binary_tail = binary_forecast_evidence(
        binary_predictions, thresholds=thresholds
    )
    quantile_metrics, quantile_checks, quantile_tail = quantile_forecast_evidence(
        quantile_predictions, thresholds=thresholds
    )
    checks = [*binary_checks, *quantile_checks]
    blockers = [c for c in checks if not c.passed and c.severity == "blocker"]
    majors = [c for c in checks if not c.passed and c.severity == "major"]
    approved = bool(checks) and not blockers
    summary = {
        "binary_model_groups": len(binary_metrics),
        "quantile_model_groups": len(quantile_metrics),
        "regime_slices": len(regime_metrics),
        "checks": len(checks),
        "failed_blockers": len(blockers),
        "failed_major": len(majors),
        "thresholds": asdict(thresholds),
    }
    return EvidenceReport(
        approved=approved,
        decision="release" if approved else "hold",
        summary=summary,
        checks=checks,
        binary_metrics=binary_metrics,
        quantile_metrics=quantile_metrics,
        regime_metrics=regime_metrics,
        tail_metrics=[*binary_tail, *quantile_tail],
    )


def _detect_interval_columns(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    for lo, hi in (("q_lo", "q_hi"), ("q05", "q95"), ("q10", "q90")):
        if {lo, hi}.issubset(frame.columns):
            return lo, hi
    return None, None


def _horizon_to_periods(horizon: str) -> int:
    text = str(horizon).strip().lower()
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return 1
    value = max(int(digits), 1)
    if text.endswith("m"):
        return value
    if text.endswith("q"):
        return value * 3
    if text.endswith("y"):
        return value * 12
    return value


__all__ = [
    "EvidenceCheck",
    "EvidenceReport",
    "EvidenceThresholds",
    "binary_forecast_evidence",
    "build_prediction_evidence_report",
    "quantile_forecast_evidence",
]
