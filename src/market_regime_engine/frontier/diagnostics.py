# SPDX-License-Identifier: Apache-2.0
"""Fail-closed diagnostics for experimental frontier models.

These helpers do not promote frontier models to the stable core. They produce
machine-readable diagnostics so an experimental frontier gate can refuse to use
posterior or smoothed nowcast paths when convergence, stability, or online-safety
contracts are not demonstrated.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from market_regime_engine.fixed_income.hashing import canonical_sha256


@dataclass(frozen=True)
class FrontierDiagnosticReport:
    component: str
    passed: bool
    reasons: tuple[str, ...]
    metrics: dict[str, Any]
    artifact_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "passed": self.passed,
            "reasons": list(self.reasons),
            "metrics": self.metrics,
            "artifact_hash": self.artifact_hash,
        }


def evaluate_bayesian_msvar_diagnostics(
    diagnostics: Mapping[str, Any],
    *,
    max_rhat: float = 1.05,
    min_ess: float = 100.0,
    max_divergences: int = 0,
    max_companion_radius: float = 1.0,
    min_min_state_mass: float = 0.02,
) -> FrontierDiagnosticReport:
    """Evaluate Bayesian MS-VAR posterior diagnostics.

    Required diagnostics: ``num_divergences``, ``max_rhat``, ``min_ess``,
    ``max_companion_radius``. Missing or non-finite required diagnostics fail
    closed. Optional ``min_state_mass`` checks label/support weakness across
    chains or posterior probabilities.
    """

    d = dict(diagnostics or {})
    reasons: list[str] = []

    def _finite(name: str) -> float | None:
        if name not in d:
            return None
        raw = d.get(name)
        if raw is None:
            return None
        try:
            val = float(raw)
        except Exception:
            return None
        return val if math.isfinite(val) else None

    def _nonnegative_int(name: str) -> int | None:
        val = _finite(name)
        if val is None or val < 0:
            return None
        return int(val)

    divergences = _nonnegative_int("num_divergences")
    rhat = _finite("max_rhat")
    ess = _finite("min_ess")
    radius = _finite("max_companion_radius")
    min_mass = _finite("min_state_mass")
    if divergences is None:
        reasons.append("posterior_divergences_missing_or_invalid")
    elif divergences > int(max_divergences):
        reasons.append("posterior_divergences")
    if rhat is None or rhat > float(max_rhat):
        reasons.append("posterior_rhat_above_threshold_or_missing")
    if ess is None or ess < float(min_ess):
        reasons.append("posterior_ess_below_threshold_or_missing")
    if radius is None or radius >= float(max_companion_radius):
        reasons.append("posterior_companion_radius_unstable_or_missing")
    if min_mass is not None and min_mass < float(min_min_state_mass):
        reasons.append("posterior_weak_regime_mass")
    thresholds = {
        "max_rhat": float(max_rhat),
        "min_ess": float(min_ess),
        "max_divergences": int(max_divergences),
        "max_companion_radius": float(max_companion_radius),
        "min_min_state_mass": float(min_min_state_mass),
    }
    metrics = {**d, "thresholds": thresholds}
    payload = {"component": "bayesian_msvar", "metrics": metrics, "reasons": reasons}
    return FrontierDiagnosticReport(
        component="bayesian_msvar",
        passed=not reasons,
        reasons=tuple(reasons),
        metrics=metrics,
        artifact_hash=canonical_sha256(payload),
    )


def evaluate_online_prefix_safety(
    full_panel: pd.DataFrame,
    score_fn: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    date_col: str = "as_of_date",
    value_col: str = "factor_value",
    checkpoints: int = 3,
    atol: float = 1e-12,
) -> FrontierDiagnosticReport:
    """Assert that prefix nowcasts do not change after future rows are added.

    ``score_fn`` must return a frame containing ``date_col`` and ``value_col``.
    For each checkpoint prefix, we compare the last prefix score to the same
    timestamp produced by the full-panel score. Any difference means the model
    is using a smoothed/retrospective value where an online filtered value was
    required.
    """

    if full_panel is None or len(full_panel) < 4:
        payload = {"component": "dfm_mq_online_safety", "reason": "insufficient_panel"}
        return FrontierDiagnosticReport(
            component="dfm_mq_online_safety",
            passed=False,
            reasons=("insufficient_panel",),
            metrics={"rows": 0 if full_panel is None else len(full_panel)},
            artifact_hash=canonical_sha256(payload),
        )
    try:
        full_scores = score_fn(full_panel.copy())
    except Exception as exc:
        payload = {"component": "dfm_mq_online_safety", "reason": "score_fn_failed", "error": str(exc)}
        return FrontierDiagnosticReport(
            component="dfm_mq_online_safety",
            passed=False,
            reasons=("score_fn_failed",),
            metrics={"error": str(exc)},
            artifact_hash=canonical_sha256(payload),
        )
    if full_scores is None or full_scores.empty or date_col not in full_scores or value_col not in full_scores:
        payload = {"component": "dfm_mq_online_safety", "reason": "missing_score_columns"}
        return FrontierDiagnosticReport(
            component="dfm_mq_online_safety",
            passed=False,
            reasons=("missing_score_columns",),
            metrics={},
            artifact_hash=canonical_sha256(payload),
        )
    if full_scores[date_col].duplicated().any():
        dupes = full_scores.loc[full_scores[date_col].duplicated(), date_col].astype(str).head(5).tolist()
        payload = {"component": "dfm_mq_online_safety", "reason": "duplicate_score_timestamps", "examples": dupes}
        return FrontierDiagnosticReport(
            component="dfm_mq_online_safety",
            passed=False,
            reasons=("duplicate_score_timestamps",),
            metrics={"duplicate_examples": dupes},
            artifact_hash=canonical_sha256(payload),
        )
    n = len(full_panel)
    idxs = np.linspace(max(2, n // 4), n - 2, num=min(int(checkpoints), max(1, n - 3)), dtype=int)
    violations: list[dict[str, Any]] = []
    comparisons = 0
    full_by_date = full_scores.set_index(date_col)
    for idx in idxs:
        prefix = full_panel.iloc[: idx + 1].copy()
        try:
            prefix_scores = score_fn(prefix)
        except Exception as exc:
            violations.append({"idx": int(idx), "reason": "prefix_score_fn_failed", "error": str(exc)})
            continue
        if prefix_scores is None or prefix_scores.empty:
            violations.append({"idx": int(idx), "reason": "empty_prefix_score"})
            continue
        if date_col not in prefix_scores or value_col not in prefix_scores:
            violations.append({"idx": int(idx), "reason": "missing_prefix_score_columns"})
            continue
        row = prefix_scores.iloc[-1]
        date = row[date_col]
        if date not in full_by_date.index:
            violations.append({"idx": int(idx), "reason": "missing_full_date", "date": str(date)})
            continue
        try:
            prefix_value = float(row[value_col])
            full_value = float(full_by_date.loc[date][value_col])
        except Exception as exc:
            violations.append({"idx": int(idx), "reason": "non_scalar_or_non_numeric_score", "error": str(exc)})
            continue
        comparisons += 1
        if not math.isfinite(prefix_value) or not math.isfinite(full_value):
            violations.append({"idx": int(idx), "date": str(date), "reason": "nonfinite_score"})
            continue
        if not math.isclose(prefix_value, full_value, rel_tol=0.0, abs_tol=float(atol)):
            violations.append({"idx": int(idx), "date": str(date), "prefix": prefix_value, "full": full_value})
    metrics = {"comparisons": comparisons, "violations": violations, "atol": float(atol)}
    reasons = ("prefix_safety_violation",) if violations else ()
    return FrontierDiagnosticReport(
        component="dfm_mq_online_safety",
        passed=not violations and comparisons > 0,
        reasons=reasons if comparisons > 0 else ("no_comparisons",),
        metrics=metrics,
        artifact_hash=canonical_sha256({"component": "dfm_mq_online_safety", "metrics": metrics}),
    )


__all__ = [
    "FrontierDiagnosticReport",
    "evaluate_bayesian_msvar_diagnostics",
    "evaluate_online_prefix_safety",
]
