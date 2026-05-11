# SPDX-License-Identifier: Apache-2.0
"""Frozen data contracts for the v1.5 Fixed-Income RCIE layer.

Every externally consumed signal must carry ``model_run_id``,
``release_gate``, and ``artifact_hash`` per the non-negotiable
governance constraints in ``MRE_FIXED_INCOME_AGENT.md``. These frozen
dataclasses encode the contract so the API layer, CLI, and warehouse
all read from the same source of truth.

Label enums inherit from ``(str, Enum)`` per the v1.4.1 convention
(``pyproject.toml:128-131`` and ``training_data.TrainingMode``). The
``.label`` property returns the human-readable form documented in
``AGENT.md §"Recommended labels"`` so consumers can display the same
strings used in the report writer and Streamlit dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Label enums
# ---------------------------------------------------------------------------


class RegimeLabel(str, Enum):
    """Credit-regime label states (AGENT.md §"Credit regime labels").

    Bucket boundaries per ``regime_label_from_score``: ``[0, 20)`` →
    ``RISK_ON_COMPRESSION``; ``[20, 40)`` → ``NORMAL_LIQUIDITY``;
    ``[40, 60)`` → ``WATCH_TRANSITION``; ``[60, 80)`` →
    ``RISK_OFF_HIGH_RISK_AVERSION``; ``[80, 100]`` →
    ``CRISIS_SEVERE_DISLOCATION``.
    """

    RISK_ON_COMPRESSION = "risk_on_compression"
    NORMAL_LIQUIDITY = "normal_liquidity"
    WATCH_TRANSITION = "watch_transition"
    RISK_OFF_HIGH_RISK_AVERSION = "risk_off_high_risk_aversion"
    CRISIS_SEVERE_DISLOCATION = "crisis_severe_dislocation"

    @property
    def label(self) -> str:
        """Human-readable label matching AGENT.md §"Recommended labels"."""
        return _REGIME_HUMAN_LABELS[self]


_REGIME_HUMAN_LABELS: dict[RegimeLabel, str] = {
    RegimeLabel.RISK_ON_COMPRESSION: "Risk-On / Compression",
    RegimeLabel.NORMAL_LIQUIDITY: "Normal Liquidity",
    RegimeLabel.WATCH_TRANSITION: "Watch / Transition",
    RegimeLabel.RISK_OFF_HIGH_RISK_AVERSION: "Risk-Off / High Risk Aversion",
    RegimeLabel.CRISIS_SEVERE_DISLOCATION: "Crisis / Severe Dislocation",
}


class LiquidityLabel(str, Enum):
    """Liquidity-stress label states (AGENT.md §"Liquidity labels")."""

    NORMAL = "normal"
    MILD_STRESS = "mild_stress"
    ELEVATED_STRESS = "elevated_stress"
    SEVERE_STRESS = "severe_stress"
    CRISIS_LIQUIDITY = "crisis_liquidity"

    @property
    def label(self) -> str:
        return _LIQUIDITY_HUMAN_LABELS[self]


_LIQUIDITY_HUMAN_LABELS: dict[LiquidityLabel, str] = {
    LiquidityLabel.NORMAL: "Normal",
    LiquidityLabel.MILD_STRESS: "Mild Stress",
    LiquidityLabel.ELEVATED_STRESS: "Elevated Stress",
    LiquidityLabel.SEVERE_STRESS: "Severe Stress",
    LiquidityLabel.CRISIS_LIQUIDITY: "Crisis Liquidity",
}


class ExecutionRecommendation(str, Enum):
    """Execution-confidence recommendations (AGENT.md §"Execution recommendations").

    ``UNAVAILABLE_GOVERNANCE`` and ``UNAVAILABLE_STALE_SIGNAL`` are the
    fail-closed verdicts when ``release_gate`` is False or the upstream
    signals are too stale to drive Auto-X advisory (per non-negotiable
    constraint 8).
    """

    AUTO_X_ALLOWED = "auto_x_allowed"
    AUTO_X_CAUTION = "auto_x_caution"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"
    UNAVAILABLE_GOVERNANCE = "unavailable_governance"
    UNAVAILABLE_STALE_SIGNAL = "unavailable_stale_signal"

    @property
    def label(self) -> str:
        return _EXECUTION_HUMAN_LABELS[self]


_EXECUTION_HUMAN_LABELS: dict[ExecutionRecommendation, str] = {
    ExecutionRecommendation.AUTO_X_ALLOWED: "Auto-X allowed",
    ExecutionRecommendation.AUTO_X_CAUTION: "Auto-X caution / trader confirm",
    ExecutionRecommendation.MANUAL_REVIEW_REQUIRED: "Manual review required",
    ExecutionRecommendation.UNAVAILABLE_GOVERNANCE: "Unavailable — governance gate failed",
    ExecutionRecommendation.UNAVAILABLE_STALE_SIGNAL: "Unavailable — stale signal",
}


# ---------------------------------------------------------------------------
# Bucket helpers
# ---------------------------------------------------------------------------


def _bucket_label(score: float, mapping: tuple[tuple[float, Any], ...]) -> Any:
    """Map a numeric score to a label via half-open buckets ``[lo, hi)``."""
    if not (0.0 <= score <= 100.0):
        raise ValueError(f"score must be in [0, 100]; got {score!r}")
    for upper, label in mapping:
        if score < upper:
            return label
    return mapping[-1][1]


_REGIME_BUCKETS: tuple[tuple[float, RegimeLabel], ...] = (
    (20.0, RegimeLabel.RISK_ON_COMPRESSION),
    (40.0, RegimeLabel.NORMAL_LIQUIDITY),
    (60.0, RegimeLabel.WATCH_TRANSITION),
    (80.0, RegimeLabel.RISK_OFF_HIGH_RISK_AVERSION),
    (100.0 + 1e-9, RegimeLabel.CRISIS_SEVERE_DISLOCATION),
)

_LIQUIDITY_BUCKETS: tuple[tuple[float, LiquidityLabel], ...] = (
    (20.0, LiquidityLabel.NORMAL),
    (40.0, LiquidityLabel.MILD_STRESS),
    (60.0, LiquidityLabel.ELEVATED_STRESS),
    (80.0, LiquidityLabel.SEVERE_STRESS),
    (100.0 + 1e-9, LiquidityLabel.CRISIS_LIQUIDITY),
)


def regime_label_from_score(score: float) -> RegimeLabel:
    """Bucket a 0–100 credit-regime score into a :class:`RegimeLabel`.

    Boundary scores 20, 40, 60, 80 fall into the *upper* bucket
    (half-open ``[lo, hi)`` convention) so a score of exactly 20.0
    is ``NORMAL_LIQUIDITY``, not ``RISK_ON_COMPRESSION``.
    """
    return _bucket_label(float(score), _REGIME_BUCKETS)


def liquidity_label_from_score(score: float) -> LiquidityLabel:
    """Bucket a 0–100 liquidity-stress score into a :class:`LiquidityLabel`."""
    return _bucket_label(float(score), _LIQUIDITY_BUCKETS)


# ---------------------------------------------------------------------------
# Data contracts (frozen)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CreditRegimeOutput:
    """Output of ``score_credit_regime`` (AGENT.md §"CreditRegimeOutput").

    Required external-consumer fields: ``model_run_id``,
    ``release_gate``, ``artifact_hash`` (non-negotiable constraint 7).
    """

    timestamp: str
    regime_score: float
    regime_label: str
    confidence: float
    drivers: tuple[str, ...]
    component_scores: dict[str, float]
    model_run_id: str
    release_gate: bool
    artifact_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LiquidityStressOutput:
    """Output of ``score_liquidity_stress`` (AGENT.md §"LiquidityStressOutput")."""

    timestamp: str
    scope_type: str
    scope_id: str
    liquidity_index: float
    liquidity_label: str
    confidence: float
    drivers: tuple[str, ...]
    model_run_id: str
    release_gate: bool
    artifact_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionConfidenceRequest:
    """Inbound order payload for ``POST /v1/execution_confidence``.

    PR-5 wraps this dataclass in a Pydantic v2 model for request-body
    validation; PR-1 ships the typed shape so downstream code can
    refer to the contract. Optional fields default to ``None`` so the
    minimal order body in INSTRUCTIONS.md §6.3 round-trips cleanly.
    """

    timestamp: str
    cusip: str
    side: str
    notional: float
    protocol: str
    limit_price: float | None = None
    urgency: str | None = None
    sector: str | None = None
    rating: str | None = None
    maturity_bucket: str | None = None
    client_request_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionConfidenceResponse:
    """Output of ``score_execution_confidence`` (AGENT.md §"ExecutionConfidenceResponse")."""

    timestamp: str
    cusip: str
    side: str
    notional: float
    protocol: str
    confidence_score: float
    expected_slippage_bps: float | None
    confidence_interval_low: float | None
    confidence_interval_high: float | None
    recommended_action: str
    human_review_required: bool
    model_run_id: str
    release_gate: bool
    artifact_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FixedIncomeEvidencePack:
    """Tamper-evident reproducible record per signal.

    Per AGENT.md §"FixedIncomeEvidencePack" and non-negotiable
    constraint 6: every production output must be reproducible from one
    of these packs. PR-1 ships the dataclass + canonical SHA-256 hash;
    PR-7 adds HMAC signing/verification so ``hmac_signature`` may be
    ``None`` until then.
    """

    model_run_id: str
    component_name: str
    model_version: str
    timestamp: str
    code_sha: str | None
    model_hash: str
    input_features_hash: str
    output_hash: str
    data_vintages: dict[str, Any]
    validation_results: dict[str, Any]
    release_gate: bool
    random_seeds: dict[str, Any]
    python_version: str
    lockfile_hash: str | None
    hmac_signature: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "CreditRegimeOutput",
    "ExecutionConfidenceRequest",
    "ExecutionConfidenceResponse",
    "ExecutionRecommendation",
    "FixedIncomeEvidencePack",
    "LiquidityLabel",
    "LiquidityStressOutput",
    "RegimeLabel",
    "liquidity_label_from_score",
    "regime_label_from_score",
]
