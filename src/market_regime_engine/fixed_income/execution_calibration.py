# SPDX-License-Identifier: Apache-2.0
"""Empirical execution-confidence calibration from observed outcomes.

This module is the v1.7.0 bridge from the deterministic PR-5 execution
confidence baseline to an empirically calibrated decision-support score.  It
uses only rows that already exist in the fixed-income warehouse:

- ``execution_confidence_predictions``: the baseline score emitted at decision
  time and keyed by ``request_id``.
- ``execution_outcomes``: post-decision observed execution results keyed by
  the same ``request_id``.

The calibration target is deliberately narrow and auditable.  The default
``fill_success`` target is true when the observed filled quantity reaches the
configured fill-ratio threshold.  If an outcome metadata blob explicitly
provides ``execution_success`` or ``protocol_success``, that vendor/control-plane
label takes precedence.

Point-in-time contract:

- training rows must satisfy ``prediction.timestamp <= outcome.decision_timestamp``;
- outcomes must satisfy ``observed_at > decision_timestamp``;
- training rows are filtered to ``observed_at <= asof``;
- a stored calibrator is applied during scoring only when its
  ``training_cutoff_utc <= request.timestamp``.

The probability calibrator is a dependency-free Platt/logistic calibration on
``logit(raw_confidence_score)`` with L2 regularisation.  The slippage calibrator
is a dependency-free ridge linear correction of predicted expected slippage bps
to observed slippage bps when an observed slippage target is available.
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from market_regime_engine import __version__
from market_regime_engine.fixed_income.hashing import canonical_sha256
from market_regime_engine.fixed_income.timestamps import iso8601_z, to_utc

EXECUTION_CONFIDENCE_HORIZON = "execution_confidence"
FILL_SUCCESS_TARGET = "fill_success"
FILL_SUCCESS_METHOD = "platt_logistic"
SLIPPAGE_TARGET = "observed_slippage_bps"
SLIPPAGE_METHOD = "linear_ridge"
DEFAULT_MIN_OBSERVATIONS = 30
_DEFAULT_EPS = 1e-6


@dataclass(frozen=True)
class ExecutionCalibrationResult:
    """Summary of an empirical execution-confidence calibration fit."""

    run_id: str
    target: str
    method: str
    observations: int
    training_cutoff_utc: str
    intercept: float
    slope: float
    fallback_rate: float | None
    raw_mean: float
    calibrated_mean: float
    brier_raw: float | None = None
    brier_calibrated: float | None = None
    log_loss_raw: float | None = None
    log_loss_calibrated: float | None = None
    ece_raw: float | None = None
    ece_calibrated: float | None = None
    artifact_hash: str | None = None
    release_gate: bool = True
    metadata: Mapping[str, Any] | None = None


def _now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _coerce_asof(asof: str | pd.Timestamp | None) -> pd.Timestamp:
    if asof is None:
        return _now_utc()
    ts = to_utc(asof)
    if ts is None:
        raise ValueError("asof must be a valid timestamp")
    return ts


def _json_loads_maybe(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {}
    raw = str(value).strip()
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return dict(obj) if isinstance(obj, Mapping) else {}


def _truthy(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, (float, np.floating)):
        return bool(int(value)) if np.isfinite(value) else None
    raw = str(value).strip().lower()
    if raw in {"1", "true", "t", "yes", "y", "success", "succeeded", "filled"}:
        return True
    if raw in {"0", "false", "f", "no", "n", "failure", "failed", "unfilled"}:
        return False
    return None


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(p, dtype=float), _DEFAULT_EPS, 1.0 - _DEFAULT_EPS)
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(x, dtype=float)
    out = np.empty_like(arr, dtype=float)
    mask = arr >= 0
    out[mask] = 1.0 / (1.0 + np.exp(-arr[mask]))
    exp_x = np.exp(arr[~mask])
    out[~mask] = exp_x / (1.0 + exp_x)
    if np.ndim(x) == 0:
        return float(out.item())
    return out


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    yy = np.asarray(y, dtype=float)
    pp = np.clip(np.asarray(p, dtype=float), _DEFAULT_EPS, 1.0 - _DEFAULT_EPS)
    return float(-np.mean(yy * np.log(pp) + (1.0 - yy) * np.log(1.0 - pp)))


def _ece(y: np.ndarray, p: np.ndarray, *, bins: int = 10) -> float:
    yy = np.asarray(y, dtype=float)
    pp = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    if yy.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    total = float(yy.size)
    acc = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi >= 1.0:
            mask = (pp >= lo) & (pp <= hi)
        else:
            mask = (pp >= lo) & (pp < hi)
        if not mask.any():
            continue
        acc += float(mask.sum()) / total * abs(float(pp[mask].mean()) - float(yy[mask].mean()))
    return float(acc)


def _fit_platt(raw_score: np.ndarray, target: np.ndarray, *, l2: float = 1.0) -> tuple[float, float, np.ndarray]:
    """Fit intercept/slope for sigmoid(a + b * logit(raw_score)).

    Uses Newton/IRLS with a ridge penalty on the slope only.  The intercept is
    left unpenalised so the global base rate can be matched in small samples.
    Constant targets degrade to an intercept-only smoothed base-rate model.
    """

    raw = np.clip(np.asarray(raw_score, dtype=float), _DEFAULT_EPS, 1.0 - _DEFAULT_EPS)
    y = np.asarray(target, dtype=float)
    if raw.size != y.size or raw.size == 0:
        raise ValueError("raw_score and target must have the same non-zero length")
    if np.nanmin(y) == np.nanmax(y):
        # Jeffreys smoothing prevents infinite logits on all-0/all-1 samples.
        base = float((y.sum() + 0.5) / (y.size + 1.0))
        intercept = float(math.log(base / (1.0 - base)))
        calibrated = np.full_like(raw, base, dtype=float)
        return intercept, 0.0, calibrated

    x = _logit(raw)
    design = np.column_stack([np.ones_like(x), x])
    beta = np.array([0.0, 1.0], dtype=float)
    penalty = np.diag([0.0, float(l2)])
    for _ in range(100):
        eta = design @ beta
        p = np.asarray(_sigmoid(eta), dtype=float)
        w = np.clip(p * (1.0 - p), 1e-8, None)
        grad = design.T @ (p - y) + penalty @ beta
        hess = (design.T * w) @ design + penalty
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hess) @ grad
        beta_next = beta - step
        if np.max(np.abs(beta_next - beta)) < 1e-8:
            beta = beta_next
            break
        beta = beta_next
    calibrated = np.asarray(_sigmoid(design @ beta), dtype=float)
    return float(beta[0]), float(beta[1]), calibrated


def _fit_linear(raw_x: np.ndarray, target_y: np.ndarray, *, l2: float = 1.0) -> tuple[float, float, np.ndarray]:
    x = np.asarray(raw_x, dtype=float)
    y = np.asarray(target_y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        raise ValueError("linear calibration requires at least one finite row")
    if x.size == 1 or float(np.nanstd(x)) <= 1e-12:
        intercept = float(y.mean())
        slope = 0.0
        return intercept, slope, np.full_like(y, intercept, dtype=float)
    design = np.column_stack([np.ones_like(x), x])
    penalty = np.diag([0.0, float(l2)])
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    pred = design @ beta
    return float(beta[0]), float(beta[1]), np.asarray(pred, dtype=float)


def _observed_success(row: pd.Series, *, fill_ratio_threshold: float) -> float:
    metadata = _json_loads_maybe(
        row.get("outcome_metadata_json", row.get("metadata_json_outcome", row.get("metadata_json")))
    )
    for key in ("execution_success", "protocol_success", "fill_success"):
        explicit = _truthy(metadata.get(key))
        if explicit is not None:
            return 1.0 if explicit else 0.0
    filled = pd.to_numeric(pd.Series([row.get("filled_quantity")]), errors="coerce").iloc[0]
    notional = pd.to_numeric(
        pd.Series([row.get("notional_outcome", row.get("outcome_notional", row.get("notional")))], dtype="object"),
        errors="coerce",
    ).iloc[0]
    if pd.isna(filled) or pd.isna(notional) or float(notional) <= 0:
        return 0.0
    return 1.0 if float(filled) / float(notional) >= float(fill_ratio_threshold) else 0.0


def _observed_slippage_bps(row: pd.Series) -> float | None:
    metadata = _json_loads_maybe(
        row.get("outcome_metadata_json", row.get("metadata_json_outcome", row.get("metadata_json")))
    )
    for key in ("observed_slippage_bps", "slippage_bps", "market_impact_bps"):
        if key in metadata:
            value = pd.to_numeric(pd.Series([metadata.get(key)]), errors="coerce").iloc[0]
            if pd.notna(value) and np.isfinite(float(value)):
                return float(value)
    exec_price = pd.to_numeric(pd.Series([row.get("execution_price")], dtype="object"), errors="coerce").iloc[0]
    if pd.isna(exec_price) or float(exec_price) <= 0:
        return None
    reference_price = None
    for key in ("arrival_price", "benchmark_price", "reference_price", "mid_price"):
        if key in metadata:
            value = pd.to_numeric(pd.Series([metadata.get(key)]), errors="coerce").iloc[0]
            if pd.notna(value) and float(value) > 0:
                reference_price = float(value)
                break
    if reference_price is None:
        return None
    side = str(row.get("side_outcome", row.get("side")) or "").strip().lower()
    if side == "sell":
        return float((reference_price - float(exec_price)) / reference_price * 10_000.0)
    return float((float(exec_price) - reference_price) / reference_price * 10_000.0)


def build_execution_calibration_dataset(
    warehouse: Any,
    *,
    asof: str | pd.Timestamp | None = None,
    fill_ratio_threshold: float = 0.999,
    require_prediction_release_gate: bool = True,
) -> pd.DataFrame:
    """Join execution predictions to real observed outcomes for calibration.

    The returned frame has one row per request_id with PIT-safe training rows
    only.  Invalid rows are dropped rather than silently corrected because the
    purpose of this dataset is empirical calibration, not warehouse repair.
    """

    cutoff = _coerce_asof(asof)
    predictions = warehouse.read_execution_confidence_predictions()
    outcomes = warehouse.read_execution_outcomes()
    if predictions is None or outcomes is None or predictions.empty or outcomes.empty:
        return pd.DataFrame()

    pred = predictions.copy()
    out = outcomes.copy()
    pred["prediction_ts"] = pd.to_datetime(pred["timestamp"], utc=True, errors="coerce")
    out["decision_ts"] = pd.to_datetime(out["decision_timestamp"], utc=True, errors="coerce")
    out["observed_at_ts"] = pd.to_datetime(out["observed_at"], utc=True, errors="coerce")
    pred = pred.dropna(subset=["request_id", "prediction_ts", "confidence_score"])
    out = out.dropna(subset=["request_id", "decision_ts", "observed_at_ts"])
    if require_prediction_release_gate and "release_gate" in pred.columns:
        release = pd.to_numeric(pred["release_gate"], errors="coerce").fillna(0).astype(int)
        pred = pred.loc[release == 1].copy()

    joined = pred.merge(
        out,
        on="request_id",
        how="inner",
        suffixes=("_prediction", "_outcome"),
    )
    if joined.empty:
        return joined

    joined = joined.loc[
        (joined["prediction_ts"] <= joined["decision_ts"])
        & (joined["observed_at_ts"] > joined["decision_ts"])
        & (joined["observed_at_ts"] <= cutoff)
    ].copy()
    if joined.empty:
        return joined

    joined["raw_confidence_score"] = pd.to_numeric(joined["confidence_score"], errors="coerce")
    joined["fill_success"] = joined.apply(
        lambda row: _observed_success(row, fill_ratio_threshold=fill_ratio_threshold), axis=1
    )
    joined["observed_slippage_bps"] = joined.apply(_observed_slippage_bps, axis=1)
    joined["training_cutoff_utc"] = iso8601_z(cutoff)
    return joined.reset_index(drop=True)


def fit_execution_probability_calibrator(
    dataset: pd.DataFrame,
    *,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
    l2: float = 1.0,
    run_id: str | None = None,
    asof: str | pd.Timestamp | None = None,
) -> ExecutionCalibrationResult:
    """Fit the fill-success probability calibrator from a joined dataset."""

    if dataset is None or dataset.empty:
        raise ValueError("no joined prediction/outcome rows available for calibration")
    frame = dataset.copy()
    frame = frame.dropna(subset=["raw_confidence_score", FILL_SUCCESS_TARGET])
    frame = frame.loc[
        np.isfinite(pd.to_numeric(frame["raw_confidence_score"], errors="coerce"))
    ].copy()
    if len(frame) < int(min_observations):
        raise ValueError(
            f"fill-success calibration requires at least {int(min_observations)} rows; got {len(frame)}"
        )

    raw = np.clip(pd.to_numeric(frame["raw_confidence_score"], errors="coerce").to_numpy(dtype=float), _DEFAULT_EPS, 1.0 - _DEFAULT_EPS)
    y = pd.to_numeric(frame[FILL_SUCCESS_TARGET], errors="coerce").to_numpy(dtype=float)
    intercept, slope, calibrated = _fit_platt(raw, y, l2=l2)
    cutoff = _coerce_asof(asof or frame["training_cutoff_utc"].iloc[0])
    rid = run_id or f"execution-confidence-calibration-{uuid.uuid4().hex[:12]}"
    fallback = float(y.mean())
    metadata: dict[str, Any] = {
        "run_id": rid,
        "model_family": "execution_confidence_probability_calibration",
        "feature": "logit(raw_confidence_score)",
        "target_definition": "fill_success == filled_quantity / notional >= fill_ratio_threshold unless outcome metadata explicitly provides execution_success/protocol_success/fill_success",
        "training_cutoff_utc": iso8601_z(cutoff),
        "l2": float(l2),
        "raw_score_min": float(np.min(raw)),
        "raw_score_max": float(np.max(raw)),
        "target_positive_rate": fallback,
        "request_ids_hash": canonical_sha256(sorted(frame["request_id"].astype(str).tolist())),
    }
    payload = {
        "horizon": EXECUTION_CONFIDENCE_HORIZON,
        "target": FILL_SUCCESS_TARGET,
        "method": FILL_SUCCESS_METHOD,
        "intercept": intercept,
        "slope": slope,
        "fallback_rate": fallback,
        "observations": int(len(frame)),
        "raw_mean": float(raw.mean()),
        "calibrated_mean": float(calibrated.mean()),
        "training_cutoff_utc": iso8601_z(cutoff),
        "metrics": {
            "brier_raw": _brier(y, raw),
            "brier_calibrated": _brier(y, calibrated),
            "log_loss_raw": _log_loss(y, raw),
            "log_loss_calibrated": _log_loss(y, calibrated),
            "ece_raw": _ece(y, raw),
            "ece_calibrated": _ece(y, calibrated),
        },
        "request_ids_hash": metadata["request_ids_hash"],
    }
    artifact_hash = canonical_sha256(payload)
    metadata["artifact_hash"] = artifact_hash
    metadata["metrics"] = payload["metrics"]
    return ExecutionCalibrationResult(
        run_id=rid,
        target=FILL_SUCCESS_TARGET,
        method=FILL_SUCCESS_METHOD,
        observations=int(len(frame)),
        training_cutoff_utc=iso8601_z(cutoff),
        intercept=intercept,
        slope=slope,
        fallback_rate=fallback,
        raw_mean=float(raw.mean()),
        calibrated_mean=float(calibrated.mean()),
        brier_raw=payload["metrics"]["brier_raw"],
        brier_calibrated=payload["metrics"]["brier_calibrated"],
        log_loss_raw=payload["metrics"]["log_loss_raw"],
        log_loss_calibrated=payload["metrics"]["log_loss_calibrated"],
        ece_raw=payload["metrics"]["ece_raw"],
        ece_calibrated=payload["metrics"]["ece_calibrated"],
        artifact_hash=artifact_hash,
        release_gate=True,
        metadata=metadata,
    )


def fit_execution_slippage_calibrator(
    dataset: pd.DataFrame,
    *,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
    l2: float = 1.0,
    run_id: str | None = None,
    asof: str | pd.Timestamp | None = None,
) -> ExecutionCalibrationResult:
    """Fit expected-slippage bps correction from observed slippage bps."""

    if dataset is None or dataset.empty:
        raise ValueError("no joined prediction/outcome rows available for slippage calibration")
    frame = dataset.copy()
    frame["raw_expected_slippage_bps"] = pd.to_numeric(frame.get("expected_slippage_bps"), errors="coerce")
    frame["observed_slippage_bps"] = pd.to_numeric(frame.get("observed_slippage_bps"), errors="coerce")
    frame = frame.dropna(subset=["raw_expected_slippage_bps", "observed_slippage_bps"])
    if len(frame) < int(min_observations):
        raise ValueError(
            f"slippage calibration requires at least {int(min_observations)} rows; got {len(frame)}"
        )
    raw = frame["raw_expected_slippage_bps"].to_numpy(dtype=float)
    y = frame["observed_slippage_bps"].to_numpy(dtype=float)
    intercept, slope, calibrated = _fit_linear(raw, y, l2=l2)
    cutoff = _coerce_asof(asof or frame["training_cutoff_utc"].iloc[0])
    rid = run_id or f"execution-confidence-calibration-{uuid.uuid4().hex[:12]}"
    rmse_raw = float(np.sqrt(np.mean((raw - y) ** 2)))
    rmse_calibrated = float(np.sqrt(np.mean((calibrated - y) ** 2)))
    metadata: dict[str, Any] = {
        "run_id": rid,
        "model_family": "execution_confidence_slippage_calibration",
        "feature": "expected_slippage_bps",
        "target_definition": "observed_slippage_bps from outcome metadata or execution-vs-arrival/benchmark/reference/mid price",
        "training_cutoff_utc": iso8601_z(cutoff),
        "l2": float(l2),
        "rmse_raw": rmse_raw,
        "rmse_calibrated": rmse_calibrated,
        "request_ids_hash": canonical_sha256(sorted(frame["request_id"].astype(str).tolist())),
    }
    payload = {
        "horizon": EXECUTION_CONFIDENCE_HORIZON,
        "target": SLIPPAGE_TARGET,
        "method": SLIPPAGE_METHOD,
        "intercept": intercept,
        "slope": slope,
        "observations": int(len(frame)),
        "raw_mean": float(raw.mean()),
        "calibrated_mean": float(calibrated.mean()),
        "training_cutoff_utc": iso8601_z(cutoff),
        "metrics": {"rmse_raw": rmse_raw, "rmse_calibrated": rmse_calibrated},
        "request_ids_hash": metadata["request_ids_hash"],
    }
    artifact_hash = canonical_sha256(payload)
    metadata["artifact_hash"] = artifact_hash
    metadata["metrics"] = payload["metrics"]
    return ExecutionCalibrationResult(
        run_id=rid,
        target=SLIPPAGE_TARGET,
        method=SLIPPAGE_METHOD,
        observations=int(len(frame)),
        training_cutoff_utc=iso8601_z(cutoff),
        intercept=intercept,
        slope=slope,
        fallback_rate=None,
        raw_mean=float(raw.mean()),
        calibrated_mean=float(calibrated.mean()),
        artifact_hash=artifact_hash,
        release_gate=True,
        metadata=metadata,
    )


def _result_to_calibration_row(result: ExecutionCalibrationResult) -> dict[str, Any]:
    metadata = dict(result.metadata or {})
    metadata.setdefault("training_cutoff_utc", result.training_cutoff_utc)
    metadata.setdefault("artifact_hash", result.artifact_hash)
    metadata.setdefault("release_gate", bool(result.release_gate))
    return {
        "horizon": EXECUTION_CONFIDENCE_HORIZON,
        "target": result.target,
        "method": result.method,
        "intercept": float(result.intercept),
        "slope": float(result.slope),
        "fallback_rate": result.fallback_rate,
        "observations": int(result.observations),
        "raw_mean": float(result.raw_mean),
        "calibrated_mean": float(result.calibrated_mean),
        "metadata_json": json.dumps(metadata, sort_keys=True, default=str),
    }


def _write_model_run(warehouse: Any, result: ExecutionCalibrationResult) -> None:
    metadata = dict(result.metadata or {})
    warehouse.write_model_runs(
        pd.DataFrame(
            [
                {
                    "run_id": result.run_id,
                    "created_at_utc": iso8601_z(_now_utc()),
                    "engine_version": str(__version__),
                    "purpose": f"fixed_income_{result.target}_calibration",
                    "data_start": None,
                    "data_end": result.training_cutoff_utc,
                    "feature_count": 1,
                    "observation_count": int(result.observations),
                    "model_count": 1,
                    "code_version": str(__version__),
                    "artifact_hash": result.artifact_hash or canonical_sha256(metadata),
                    "metadata_json": json.dumps(metadata, sort_keys=True, default=str),
                }
            ]
        )
    )


def persist_execution_calibration(warehouse: Any, result: ExecutionCalibrationResult) -> None:
    """Persist a calibration row plus a model_run audit row."""

    warehouse.write_calibration_models(pd.DataFrame([_result_to_calibration_row(result)]))
    _write_model_run(warehouse, result)


def calibrate_execution_confidence_from_outcomes(
    warehouse: Any,
    *,
    asof: str | pd.Timestamp | None = None,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
    fill_ratio_threshold: float = 0.999,
    l2: float = 1.0,
    fit_slippage: bool = True,
    require_prediction_release_gate: bool = True,
    persist: bool = True,
    run_id: str | None = None,
) -> list[ExecutionCalibrationResult]:
    """Fit empirical execution-confidence calibrators from real outcomes.

    Returns the fitted result objects.  When ``persist=True`` (default), the
    fitted rows are written to ``calibration_models`` and their audit records
    are written to ``model_runs``.
    """

    cutoff = _coerce_asof(asof)
    rid = run_id or f"execution-confidence-calibration-{uuid.uuid4().hex[:12]}"
    dataset = build_execution_calibration_dataset(
        warehouse,
        asof=cutoff,
        fill_ratio_threshold=fill_ratio_threshold,
        require_prediction_release_gate=require_prediction_release_gate,
    )
    results: list[ExecutionCalibrationResult] = []
    probability = fit_execution_probability_calibrator(
        dataset,
        min_observations=min_observations,
        l2=l2,
        run_id=rid,
        asof=cutoff,
    )
    results.append(probability)
    if fit_slippage:
        try:
            slippage = fit_execution_slippage_calibrator(
                dataset,
                min_observations=min_observations,
                l2=l2,
                run_id=rid,
                asof=cutoff,
            )
        except ValueError:
            slippage = None
        if slippage is not None:
            results.append(slippage)
    if persist:
        for result in results:
            persist_execution_calibration(warehouse, result)
    return results


def _read_calibration_row(
    warehouse: Any,
    *,
    target: str,
    method: str,
) -> dict[str, Any] | None:
    try:
        df = warehouse.read_calibration_models()
    except Exception:
        return None
    if df is None or df.empty:
        return None
    sub = df.loc[
        (df["horizon"].astype(str) == EXECUTION_CONFIDENCE_HORIZON)
        & (df["target"].astype(str) == target)
        & (df["method"].astype(str) == method)
    ].copy()
    if sub.empty:
        return None
    row = sub.iloc[-1].to_dict()
    row["metadata"] = _json_loads_maybe(row.get("metadata_json"))
    return row


def load_execution_probability_calibrator(warehouse: Any) -> dict[str, Any] | None:
    """Load the persisted fill-success calibrator, if present."""

    return _read_calibration_row(
        warehouse, target=FILL_SUCCESS_TARGET, method=FILL_SUCCESS_METHOD
    )


def load_execution_slippage_calibrator(warehouse: Any) -> dict[str, Any] | None:
    """Load the persisted slippage calibrator, if present."""

    return _read_calibration_row(warehouse, target=SLIPPAGE_TARGET, method=SLIPPAGE_METHOD)


def calibrator_is_usable_asof(calibrator: Mapping[str, Any] | None, decision_ts: pd.Timestamp) -> bool:
    """Return true when a persisted calibrator is PIT-usable at decision_ts."""

    if not calibrator:
        return False
    metadata = dict(calibrator.get("metadata") or _json_loads_maybe(calibrator.get("metadata_json")))
    if _truthy(metadata.get("release_gate")) is False:
        return False
    training_cutoff = metadata.get("training_cutoff_utc")
    if training_cutoff is None:
        return False
    cutoff_ts = to_utc(training_cutoff)
    if cutoff_ts is None:
        return False
    return bool(cutoff_ts <= decision_ts)


def apply_probability_calibration(raw_score: float, calibrator: Mapping[str, Any]) -> float:
    """Apply persisted Platt/logistic calibration to a raw confidence score."""

    intercept = float(calibrator.get("intercept", 0.0))
    slope = float(calibrator.get("slope", 1.0))
    x = float(_logit(np.array([float(raw_score)]))[0])
    return float(np.clip(_sigmoid(intercept + slope * x), _DEFAULT_EPS, 1.0 - _DEFAULT_EPS))


def apply_slippage_calibration(raw_expected_slippage_bps: float | None, calibrator: Mapping[str, Any]) -> float | None:
    """Apply persisted linear slippage calibration."""

    if raw_expected_slippage_bps is None:
        return None
    raw = float(raw_expected_slippage_bps)
    if not np.isfinite(raw):
        return None
    intercept = float(calibrator.get("intercept", 0.0))
    slope = float(calibrator.get("slope", 1.0))
    return float(max(0.0, intercept + slope * raw))


def calibration_summary_payload(results: list[ExecutionCalibrationResult]) -> dict[str, Any]:
    """Return a JSON-serialisable CLI/report summary for fit results."""

    return {
        "status": "ok",
        "calibrations": [
            {
                "run_id": r.run_id,
                "target": r.target,
                "method": r.method,
                "observations": r.observations,
                "training_cutoff_utc": r.training_cutoff_utc,
                "intercept": r.intercept,
                "slope": r.slope,
                "fallback_rate": r.fallback_rate,
                "raw_mean": r.raw_mean,
                "calibrated_mean": r.calibrated_mean,
                "brier_raw": r.brier_raw,
                "brier_calibrated": r.brier_calibrated,
                "log_loss_raw": r.log_loss_raw,
                "log_loss_calibrated": r.log_loss_calibrated,
                "ece_raw": r.ece_raw,
                "ece_calibrated": r.ece_calibrated,
                "artifact_hash": r.artifact_hash,
                "release_gate": r.release_gate,
            }
            for r in results
        ],
    }
