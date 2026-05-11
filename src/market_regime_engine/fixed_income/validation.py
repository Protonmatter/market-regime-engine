# SPDX-License-Identifier: Apache-2.0
"""Fixed-Income validation surface — PR-1 skeleton.

Per ``MRE_FIXED_INCOME_INSTRUCTIONS.md §"File responsibilities"``:
``validation.py`` is the home for calibration, PIT checks, anti-overfit
checks, and model-quality tests. PR-1 ships the entry-point signatures
+ a bounded-score sanity check for each output type so consumers can
import the surface today; PR-3/4/5 fill in real checks (DSR/PBO,
expected-calibration-error, walk-forward purge correctness, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from market_regime_engine.fixed_income.schemas import (
    CreditRegimeOutput,
    ExecutionConfidenceResponse,
    LiquidityStressOutput,
)


@dataclass(frozen=True)
class ValidationReport:
    """Aggregated check results for an FI output.

    ``passed`` is computed from ``violations`` so callers do not need
    to manually keep the boolean and the list in sync. ``details``
    holds per-check diagnostics for the operator playbook.
    """

    component: str
    violations: tuple[str, ...] = field(default_factory=tuple)
    details: dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.violations


def _check_score_bounded(score: float, *, lo: float = 0.0, hi: float = 100.0) -> str | None:
    """Return a violation reason if ``score`` is outside ``[lo, hi]``."""
    if score < lo or score > hi:
        return f"score_out_of_bounds[{lo}, {hi}]:{score}"
    return None


def _check_probability(p: float) -> str | None:
    if p < 0.0 or p > 1.0:
        return f"probability_out_of_bounds[0, 1]:{p}"
    return None


def _check_governance_fields(model_run_id: str, artifact_hash: str) -> tuple[str, ...]:
    """Surface missing ``model_run_id`` / ``artifact_hash`` rails.

    Per non-negotiable constraint 7: no external signal without
    ``model_run_id``, ``release_gate``, ``artifact_hash``. We accept
    ``release_gate=False`` (that's the fail-closed signal) but the
    other two strings must be non-empty.
    """
    violations: list[str] = []
    if not model_run_id:
        violations.append("missing_model_run_id")
    if not artifact_hash:
        violations.append("missing_artifact_hash")
    return tuple(violations)


def validate_credit_regime_output(out: CreditRegimeOutput) -> ValidationReport:
    """Bounded-score + governance-rail check for a credit-regime signal."""
    violations: list[str] = []
    bound_violation = _check_score_bounded(out.regime_score)
    if bound_violation is not None:
        violations.append(bound_violation)
    conf_violation = _check_probability(out.confidence)
    if conf_violation is not None:
        violations.append(f"confidence_{conf_violation}")
    violations.extend(_check_governance_fields(out.model_run_id, out.artifact_hash))
    return ValidationReport(component="credit_regime", violations=tuple(violations))


def validate_liquidity_stress_output(out: LiquidityStressOutput) -> ValidationReport:
    """Bounded-score + governance-rail check for a liquidity-stress signal."""
    violations: list[str] = []
    bound_violation = _check_score_bounded(out.liquidity_index)
    if bound_violation is not None:
        violations.append(bound_violation)
    conf_violation = _check_probability(out.confidence)
    if conf_violation is not None:
        violations.append(f"confidence_{conf_violation}")
    if not out.scope_type:
        violations.append("missing_scope_type")
    violations.extend(_check_governance_fields(out.model_run_id, out.artifact_hash))
    return ValidationReport(component="liquidity_stress", violations=tuple(violations))


def validate_execution_confidence_response(resp: ExecutionConfidenceResponse) -> ValidationReport:
    """Probability-bound + fail-closed-rail check for an execution-confidence response."""
    violations: list[str] = []
    prob_violation = _check_probability(resp.confidence_score)
    if prob_violation is not None:
        violations.append(prob_violation)
    if (
        resp.confidence_interval_low is not None
        and resp.confidence_interval_high is not None
        and resp.confidence_interval_low > resp.confidence_interval_high
    ):
        violations.append("ci_low_above_high")
    if not resp.release_gate and not resp.human_review_required:
        violations.append("release_gate_false_but_human_review_not_required")
    violations.extend(_check_governance_fields(resp.model_run_id, resp.artifact_hash))
    return ValidationReport(component="execution_confidence", violations=tuple(violations))


__all__ = [
    "ValidationReport",
    "validate_credit_regime_output",
    "validate_execution_confidence_response",
    "validate_liquidity_stress_output",
]
