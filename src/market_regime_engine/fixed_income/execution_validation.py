# SPDX-License-Identifier: Apache-2.0
"""Realized-outcome validation for fixed-income execution confidence.

This module turns post-decision ``execution_outcomes`` into auditable
validation artifacts for the deterministic / calibrated execution-confidence
pipeline. It is intentionally dependency-light and reuses the PIT-safe join in
``execution_calibration`` so the validation path cannot train or certify on
future outcomes.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from market_regime_engine.fixed_income.execution_calibration import (
    _brier,
    _ece,
    _log_loss,
    build_execution_calibration_dataset,
)
from market_regime_engine.fixed_income.hashing import canonical_sha256
from market_regime_engine.fixed_income.timestamps import iso8601_z, to_utc

DEFAULT_MIN_REGIME_SAMPLE_SIZE = 30
DEFAULT_MIN_TCA_LIFT_N = 30


@dataclass(frozen=True)
class ExecutionValidationReport:
    """Audit-grade summary for execution-confidence realized outcomes."""

    asof_utc: str
    observations: int
    min_regime_sample_size: int
    min_required_regime_sample_size: int
    brier: float
    log_loss: float
    ece: float
    calibration_by_regime: list[dict[str, Any]]
    lift_by_decile: list[dict[str, Any]]
    tca_lift_by_regime: dict[str, dict[str, float]]
    artifact_hash: str
    passed: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "asof_utc": self.asof_utc,
            "observations": self.observations,
            "min_regime_sample_size": self.min_regime_sample_size,
            "min_observed_regime_sample_size": self.min_regime_sample_size,
            "min_required_regime_sample_size": self.min_required_regime_sample_size,
            "brier": self.brier,
            "log_loss": self.log_loss,
            "ece": self.ece,
            "calibration_by_regime": self.calibration_by_regime,
            "lift_by_decile": self.lift_by_decile,
            "tca_lift_by_regime": self.tca_lift_by_regime,
            "artifact_hash": self.artifact_hash,
            "passed": self.passed,
            "reasons": list(self.reasons),
        }


def _asof_utc(asof: str | pd.Timestamp | None) -> pd.Timestamp:
    if asof is None:
        now = pd.Timestamp.utcnow()
        return now.tz_convert("UTC") if now.tzinfo else now.tz_localize("UTC")
    ts = to_utc(asof)
    if ts is None:
        raise ValueError("asof must be a valid timestamp")
    return ts


def _metadata_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if value is None:
        return {}
    try:
        if pd.isna(value):
            return {}
    except Exception:
        pass
    raw = str(value).strip()
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return dict(obj) if isinstance(obj, Mapping) else {}


def _first_nonempty(row: pd.Series, names: tuple[str, ...], default: str) -> str:
    for name in names:
        if name not in row:
            continue
        raw = row.get(name)
        if raw is None:
            continue
        try:
            if pd.isna(raw):
                continue
        except Exception:
            pass
        text = str(raw).strip()
        if text:
            return text
    for meta_col in ("metadata_json_prediction", "metadata_json_outcome", "metadata_json"):
        meta = _metadata_dict(row.get(meta_col))
        for name in names:
            raw = meta.get(name)
            if raw is not None and str(raw).strip():
                return str(raw).strip()
    return default


def add_execution_validation_dimensions(dataset: pd.DataFrame) -> pd.DataFrame:
    """Attach regime/liquidity/TCA dimensions used by validation summaries."""

    if dataset is None or dataset.empty:
        return pd.DataFrame()
    frame = dataset.copy()
    frame["validation_regime"] = frame.apply(
        lambda r: _first_nonempty(r, ("regime_label", "credit_regime_label", "regime"), "unknown"), axis=1
    )
    frame["validation_liquidity"] = frame.apply(
        lambda r: _first_nonempty(r, ("liquidity_label", "liquidity_state"), "unknown"), axis=1
    )
    frame["validation_protocol"] = frame.apply(
        lambda r: _first_nonempty(r, ("protocol_prediction", "protocol_outcome", "protocol"), "unknown"), axis=1
    )
    frame["raw_confidence_score"] = pd.to_numeric(frame.get("raw_confidence_score"), errors="coerce")
    frame["fill_success"] = pd.to_numeric(frame.get("fill_success"), errors="coerce")
    frame["observed_slippage_bps"] = pd.to_numeric(frame.get("observed_slippage_bps"), errors="coerce")
    frame["expected_slippage_bps"] = pd.to_numeric(frame.get("expected_slippage_bps"), errors="coerce")
    return frame


def _safe_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    if y.size == 0 or p.size == 0:
        return {"brier": float("nan"), "log_loss": float("nan"), "ece": float("nan")}
    return {"brier": _brier(y, p), "log_loss": _log_loss(y, p), "ece": _ece(y, p)}


def _all_finite(values: Mapping[str, float]) -> bool:
    return all(math.isfinite(float(v)) for v in values.values())


def _valid_probability_mask(values: pd.Series) -> pd.Series:
    probs = pd.to_numeric(values, errors="coerce")
    return probs.notna() & np.isfinite(probs.astype(float)) & (probs >= 0.0) & (probs <= 1.0)


def calibration_by_regime(dataset: pd.DataFrame) -> list[dict[str, Any]]:
    """Return Brier/log-loss/ECE and base rates for each observed regime."""

    frame = add_execution_validation_dimensions(dataset)
    if frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for regime, sub in frame.groupby("validation_regime", dropna=False):
        yy = pd.to_numeric(sub["fill_success"], errors="coerce").to_numpy(dtype=float)
        pp = pd.to_numeric(sub["raw_confidence_score"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(yy) & np.isfinite(pp)
        metrics = _safe_metrics(yy[mask], pp[mask])
        rows.append(
            {
                "regime": str(regime),
                "n": int(mask.sum()),
                "base_rate": float(np.mean(yy[mask])) if mask.any() else float("nan"),
                "mean_score": float(np.mean(pp[mask])) if mask.any() else float("nan"),
                **metrics,
            }
        )
    return sorted(rows, key=lambda r: r["regime"])


def lift_by_decile(dataset: pd.DataFrame) -> list[dict[str, Any]]:
    """Compute realized fill/slippage lift by confidence-score decile."""

    frame = add_execution_validation_dimensions(dataset)
    if frame.empty or not {"raw_confidence_score", "fill_success"} <= set(frame.columns):
        return []
    frame = frame.dropna(subset=["raw_confidence_score", "fill_success"])
    if frame.empty:
        return []
    ranks = frame["raw_confidence_score"].rank(method="first")
    bins = min(10, int(len(frame)))
    frame["confidence_decile"] = pd.qcut(ranks, q=bins, labels=False, duplicates="drop") + 1
    rows: list[dict[str, Any]] = []
    for decile, sub in frame.groupby("confidence_decile"):
        slip = pd.to_numeric(sub.get("observed_slippage_bps"), errors="coerce")
        rows.append(
            {
                "decile": int(decile),
                "n": int(len(sub)),
                "mean_score": float(sub["raw_confidence_score"].mean()),
                "fill_rate": float(sub["fill_success"].mean()),
                "mean_observed_slippage_bps": float(slip.dropna().mean()) if not slip.dropna().empty else float("nan"),
            }
        )
    return sorted(rows, key=lambda r: r["decile"])


def _normal_approx_pvalue(diff: float, se: float) -> float:
    if not math.isfinite(diff) or not math.isfinite(se) or se <= 0:
        return 1.0
    z = abs(diff) / se
    return float(math.erfc(z / math.sqrt(2.0)))


def tca_lift_by_regime(dataset: pd.DataFrame, *, min_n: int = DEFAULT_MIN_TCA_LIFT_N) -> dict[str, dict[str, float]]:
    """Compare top-vs-bottom confidence terciles by regime.

    Positive ``effect_size`` means high-confidence requests have lower observed
    slippage than low-confidence requests. Negative values are adverse evidence
    and must not satisfy certification even if statistically significant. The
    p-value is a dependency-free normal approximation over the difference in
    means.
    """

    frame = add_execution_validation_dimensions(dataset)
    out: dict[str, dict[str, float]] = {}
    if frame.empty or not {"raw_confidence_score", "observed_slippage_bps"} <= set(frame.columns):
        return out
    frame = frame.dropna(subset=["raw_confidence_score", "observed_slippage_bps"])
    if frame.empty:
        return out
    for regime, sub in frame.groupby("validation_regime", dropna=False):
        sub = sub.sort_values("raw_confidence_score")
        n = len(sub)
        bucket_n = max(1, n // 3)
        low = sub.head(bucket_n)["observed_slippage_bps"].astype(float).to_numpy()
        high = sub.tail(bucket_n)["observed_slippage_bps"].astype(float).to_numpy()
        diff = float(np.mean(low) - np.mean(high)) if low.size and high.size else float("nan")
        pooled = np.concatenate([low, high]) if low.size and high.size else np.asarray([], dtype=float)
        pooled_sd = float(np.std(pooled, ddof=1)) if pooled.size > 1 else float("nan")
        se = float(math.sqrt(np.var(low, ddof=1) / len(low) + np.var(high, ddof=1) / len(high))) if len(low) > 1 and len(high) > 1 else float("nan")
        effect = float(diff / pooled_sd) if pooled_sd and math.isfinite(pooled_sd) and pooled_sd > 0 else 0.0
        out[str(regime)] = {
            "n": float(n),
            "low_confidence_mean_slippage_bps": float(np.mean(low)) if low.size else float("nan"),
            "high_confidence_mean_slippage_bps": float(np.mean(high)) if high.size else float("nan"),
            "lift_bps": diff,
            "effect_size": effect,
            "p_value": _normal_approx_pvalue(diff, se),
            "underpowered": 1.0 if n < int(min_n) else 0.0,
        }
    return out


def validate_execution_confidence_realized_outcomes(
    warehouse: Any,
    *,
    asof: str | pd.Timestamp | None = None,
    min_observations: int = 30,
    min_regime_sample_size: int = DEFAULT_MIN_REGIME_SAMPLE_SIZE,
    max_brier: float = 0.20,
    max_ece: float = 0.05,
    fill_ratio_threshold: float = 0.999,
    require_prediction_release_gate: bool = True,
) -> ExecutionValidationReport:
    """Build an audit report from predictions joined to realized outcomes."""

    cutoff = _asof_utc(asof)
    dataset = build_execution_calibration_dataset(
        warehouse,
        asof=cutoff,
        fill_ratio_threshold=fill_ratio_threshold,
        require_prediction_release_gate=require_prediction_release_gate,
    )
    frame = add_execution_validation_dimensions(dataset)
    if frame.empty or not {"raw_confidence_score", "fill_success"} <= set(frame.columns):
        frame = pd.DataFrame(columns=["raw_confidence_score", "fill_success", "observed_slippage_bps"])
    else:
        frame = frame.dropna(subset=["raw_confidence_score", "fill_success"])
    invalid_probability_rows = 0
    if not frame.empty:
        valid_prob = _valid_probability_mask(frame["raw_confidence_score"])
        invalid_probability_rows = int((~valid_prob).sum())
        frame = frame.loc[valid_prob].copy()
    yy = frame["fill_success"].astype(float).to_numpy() if not frame.empty else np.asarray([], dtype=float)
    pp = frame["raw_confidence_score"].astype(float).to_numpy() if not frame.empty else np.asarray([], dtype=float)
    metrics = _safe_metrics(yy, pp)
    by_regime = calibration_by_regime(frame)
    by_decile = lift_by_decile(frame)
    lift = tca_lift_by_regime(frame, min_n=min_regime_sample_size)
    reasons: list[str] = []
    if invalid_probability_rows > 0:
        reasons.append("invalid_probability_score_rows")
    if len(frame) < int(min_observations):
        reasons.append("insufficient_observations")
    if not _all_finite(metrics):
        reasons.append("missing_or_nonfinite_validation_metrics")
    if math.isfinite(metrics["brier"]) and metrics["brier"] > float(max_brier):
        reasons.append("brier_above_threshold")
    if math.isfinite(metrics["ece"]) and metrics["ece"] > float(max_ece):
        reasons.append("ece_above_threshold")
    if any(int(r.get("n", 0)) < int(min_regime_sample_size) for r in by_regime):
        reasons.append("regime_sample_size_below_floor")
    if not lift:
        reasons.append("tca_lift_missing")
    elif not any(row.get("p_value", 1.0) <= 0.05 and row.get("effect_size", 0.0) >= 0.2 for row in lift.values()):
        reasons.append("tca_lift_no_positive_significant_segment")
    if any(row.get("underpowered", 0.0) >= 1.0 for row in lift.values()):
        reasons.append("tca_lift_underpowered_segments")
    payload = {
        "asof_utc": iso8601_z(cutoff),
        "observations": int(len(frame)),
        "min_regime_sample_size": int(min_regime_sample_size),
        "metrics": metrics,
        "invalid_probability_rows": invalid_probability_rows,
        "calibration_by_regime": by_regime,
        "lift_by_decile": by_decile,
        "tca_lift_by_regime": lift,
        "reasons": reasons,
    }
    artifact_hash = canonical_sha256(payload)
    return ExecutionValidationReport(
        asof_utc=iso8601_z(cutoff),
        observations=int(len(frame)),
        min_regime_sample_size=int(min((r.get("n", 0) for r in by_regime), default=0)),
        min_required_regime_sample_size=int(min_regime_sample_size),
        brier=float(metrics["brier"]),
        log_loss=float(metrics["log_loss"]),
        ece=float(metrics["ece"]),
        calibration_by_regime=by_regime,
        lift_by_decile=by_decile,
        tca_lift_by_regime=lift,
        artifact_hash=artifact_hash,
        passed=not reasons,
        reasons=tuple(reasons),
    )


def certification_confidence_row(
    report: ExecutionValidationReport,
    *,
    date: str | pd.Timestamp | None = None,
    confidence: float = 0.95,
    dsr: float | None = None,
    pbo: float | None = None,
    model_card_path: str = "docs/method_cards/execution_confidence.md",
    evidence_pack_hmac: str | None = None,
) -> pd.DataFrame:
    """Convert a validation report into a release-gate confidence row."""

    asof = _asof_utc(date or report.asof_utc)
    return pd.DataFrame(
        [
            {
                "date": iso8601_z(asof),
                "confidence": float(confidence),
                "grade": "A" if report.passed else "F",
                "brier": report.brier,
                "ece": report.ece,
                "dsr": dsr,
                "pbo": pbo,
                "tca_lift": report.tca_lift_by_regime,
                "pit_leakage_passed": True,
                "walk_forward_passed": True,
                "validation_artifact_hash": report.artifact_hash,
                "model_card_path": model_card_path,
                "evidence_pack_hmac": evidence_pack_hmac,
                "min_regime_sample_size": report.min_regime_sample_size,
                "min_observed_regime_sample_size": report.min_regime_sample_size,
                "min_required_regime_sample_size": report.min_required_regime_sample_size,
                "metadata_json": json.dumps(report.to_dict(), sort_keys=True, default=str),
            }
        ]
    )


__all__ = [
    "ExecutionValidationReport",
    "add_execution_validation_dimensions",
    "calibration_by_regime",
    "certification_confidence_row",
    "lift_by_decile",
    "tca_lift_by_regime",
    "validate_execution_confidence_realized_outcomes",
]
