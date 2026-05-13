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

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import pandas as pd


class _ReadOnlyMetadata(dict):
    """``dict`` subclass that raises on mutation.

    v1.6.0 (REVIEW_DEEP_V1_5_2.md F2 / Finding §3.15): the
    dataclasses below carry ``metadata: dict[str, Any]`` and are
    declared ``frozen=True``. ``frozen=True`` only prevents field
    *reassignment*; ``output.metadata['x'] = 1`` still mutated the
    underlying dict and could corrupt the audit trail of a
    canonical evidence pack (the metadata
    dict is part of the hashed canonical JSON).

    This subclass overrides every mutating method to raise
    :class:`TypeError`, while inheriting from :class:`dict` so
    :func:`dataclasses.asdict`, :func:`json.dumps`,
    :func:`copy.deepcopy`, and the rest of the standard-library
    tooling all keep working unchanged. The deep-review spec
    suggested :class:`types.MappingProxyType` — a dict-subclass
    is functionally equivalent for the immutability contract,
    composes cleanly with :func:`dataclasses.asdict` (which
    short-circuits on non-dict mappings), and avoids an extra
    coercion step in every ``output_to_dict`` call site.
    """

    _frozen_msg = (
        "metadata is read-only after construction "
        "(REVIEW_DEEP_V1_5_2.md F2). Build a new dataclass "
        "with the desired metadata instead."
    )

    def __setitem__(self, key: Any, value: Any) -> None:
        raise TypeError(self._frozen_msg)

    def __delitem__(self, key: Any) -> None:
        raise TypeError(self._frozen_msg)

    def update(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError(self._frozen_msg)

    def pop(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(self._frozen_msg)

    def popitem(self) -> Any:
        raise TypeError(self._frozen_msg)

    def clear(self) -> None:
        raise TypeError(self._frozen_msg)

    def setdefault(
        self, key: Any, default: Any = None
    ) -> Any:
        raise TypeError(self._frozen_msg)

    def __copy__(self) -> dict[Any, Any]:
        # Plain shallow-copy returns a regular mutable dict so the
        # API layer's ``out['metadata'].setdefault(...)`` and
        # similar post-``dataclasses.asdict`` operations stay
        # ergonomic. The immutability contract applies to the
        # dataclass instance only.
        return dict(self)

    def __deepcopy__(self, memo: dict[int, Any]) -> dict[Any, Any]:
        import copy as _copy

        return {k: _copy.deepcopy(v, memo) for k, v in self.items()}


def _freeze_metadata(obj: object) -> None:
    """Wrap ``obj.metadata`` in :class:`_ReadOnlyMetadata`."""
    raw = getattr(obj, "metadata", None)
    if raw is None:
        object.__setattr__(obj, "metadata", _ReadOnlyMetadata())
        return
    if isinstance(raw, _ReadOnlyMetadata):
        return
    object.__setattr__(obj, "metadata", _ReadOnlyMetadata(raw))

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


class CriticalFeature(str, Enum):
    """v1.5.1 (PR-9 FIX 8): hard-coded set of features that, when
    missing in the scorer input, force ``release_gate=False`` AND
    ``confidence_score <= 0.5`` AND a fail-closed label override,
    REGARDLESS of the active :class:`NanPolicy`.

    The legacy :func:`_apply_nan_policy` path silently re-weights the
    remaining components when ``nan_policy`` is not
    :attr:`NanPolicy.NAN_FAILS_PIT_AUDIT`. That behaviour is correct
    for *optional* inputs (e.g. the ETF dislocation proxy on illiquid
    sectors), but for the canonical credit / liquidity contracts
    listed below it is a silent under-reporting risk: a missing
    bid-ask snapshot or a torn CDS basis must not be reweighted away
    in production.

    The contract is enforced by
    :func:`fixed_income.critical_features.evaluate_critical_features`
    inside both the credit and liquidity scorers; the resulting
    audit log surfaces the offending features in
    ``metadata.critical_features_missing``.
    """

    CREDIT_BOND_SPREAD = "credit_bond_spread"
    CREDIT_CDS_BASIS = "credit_cds_basis"
    LIQUIDITY_BIDASK = "liquidity_bidask"
    LIQUIDITY_RFQ_RESPONSE = "liquidity_rfq_response"


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
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # v1.6.0 (REVIEW_DEEP_V1_5_2.md F2 / Finding §3.15):
        # wrap caller-supplied metadata in a read-only view so
        # the frozen-dataclass immutability contract extends
        # to the dict's contents, not just the field binding.
        _freeze_metadata(self)


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
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # v1.6.0 (REVIEW_DEEP_V1_5_2.md F2 / Finding §3.15):
        # wrap caller-supplied metadata in a read-only view so
        # the frozen-dataclass immutability contract extends
        # to the dict's contents, not just the field binding.
        _freeze_metadata(self)


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
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # v1.6.0 (REVIEW_DEEP_V1_5_2.md F2 / Finding §3.15):
        # wrap caller-supplied metadata in a read-only view so
        # the frozen-dataclass immutability contract extends
        # to the dict's contents, not just the field binding.
        _freeze_metadata(self)


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
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # v1.6.0 (REVIEW_DEEP_V1_5_2.md F2 / Finding §3.15):
        # wrap caller-supplied metadata in a read-only view so
        # the frozen-dataclass immutability contract extends
        # to the dict's contents, not just the field binding.
        _freeze_metadata(self)


@dataclass(frozen=True)
class TradeRecord:
    """A single trade record for TCA tagging (PR-6 §A.2).

    The minimal payload needed to tag a trade with prevailing regime /
    liquidity / execution-confidence context. ``timestamp`` is the
    *decision* timestamp; ``arrival_price`` and ``execution_price`` are
    optional so a trade can be tagged before the fill is observed
    (the per-trade TCA metrics gracefully degrade to ``None`` for
    fill-dependent fields). All numeric fields are ``float`` at the
    dataclass boundary; downstream :mod:`bps_precision` helpers coerce
    to :class:`decimal.Decimal` for the bps arithmetic.
    """

    request_id: str
    timestamp: pd.Timestamp
    cusip: str
    side: Literal["buy", "sell"]
    notional: float
    protocol: str
    arrival_price: float | None = None
    execution_price: float | None = None
    filled_quantity: float | None = None
    time_to_fill_seconds: float | None = None
    dealer_response_count: int | None = None
    sector: str | None = None
    rating: str | None = None
    maturity_years: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaggedTrade:
    """A :class:`TradeRecord` with regime / liquidity / execution context.

    Per PR-6 §A.2 and INSTRUCTIONS.md §6.4: every TCA segment row
    derives from a tagged trade. ``regime_soft_weights`` maps regime
    label → ``[0, 1]`` probability inferred from ``regime_score`` and
    the adjacent label boundaries via triangular weighting; the dict
    sums to 1.0. ``regime_label`` is the *hard* label after hysteresis
    (if enabled).
    """

    trade: TradeRecord
    regime_label: str
    regime_score: float
    regime_soft_weights: dict[str, float]
    liquidity_label: str
    liquidity_index: float
    execution_confidence_bucket: str  # "high" / "medium" / "low" / "unavailable"
    execution_confidence_score: float | None
    sector_bucket: str
    rating_bucket: str
    maturity_bucket: str  # "0-2y" / "2-5y" / "5-10y" / "10y+"
    notional_bucket: str  # "<1M" / "1-5M" / "5-25M" / "25M+"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TcaRegimeSegment:
    """A single TCA aggregation row.

    One row per ``(dimension-combo) × metric`` per PR-6 §A.2 and the
    ``tca_regime_segments`` warehouse table. The 9 dimension fields
    are nullable (``None``) when not used in the grouping; the
    warehouse persists ``None`` as a string sentinel (``"__all__"``)
    so the composite primary key remains stable across runs that
    aggregate over different dimension combinations.
    """

    timestamp: pd.Timestamp
    regime_label: str | None
    liquidity_label: str | None
    execution_confidence_bucket: str | None
    protocol: str | None
    side: str | None
    sector: str | None
    rating: str | None
    maturity_bucket: str | None
    notional_bucket: str | None
    metric_name: str
    metric_value: float
    sample_count: int
    model_run_id: str
    metadata_json: str


@dataclass(frozen=True)
class FixedIncomeEvidencePack:
    """Tamper-evident reproducible record per signal.

    Per AGENT.md §"FixedIncomeEvidencePack" and non-negotiable
    constraint 6: every production output must be reproducible from one
    of these packs. PR-1 ships the dataclass + canonical SHA-256 hash;
    PR-7 adds HMAC signing/verification so ``hmac_signature`` may be
    ``None`` until then.

    v1.5.1 (PR-9 FIX 3) adds the ``request_id`` field. When non-``None``
    the value rides through ``_pack_to_canonical_dict`` and therefore
    binds into the HMAC bytestream so a replay of the same
    ``(model_run_id, output_hash)`` under a different ``request_id`` no
    longer verifies. Legacy ``v1``-signed packs from v1.5.0 carry
    ``request_id=None`` and continue to verify under the same key
    version; production callers MUST set ``request_id`` and prefer the
    ``v2`` key version.
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
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # v1.6.0 (REVIEW_DEEP_V1_5_2.md F2 / Finding §3.15):
        # wrap caller-supplied metadata in a read-only view so
        # the frozen-dataclass immutability contract extends
        # to the dict's contents, not just the field binding.
        _freeze_metadata(self)
    request_id: str | None = None


__all__ = [
    "CreditRegimeOutput",
    "ExecutionConfidenceRequest",
    "ExecutionConfidenceResponse",
    "ExecutionRecommendation",
    "FixedIncomeEvidencePack",
    "LiquidityLabel",
    "LiquidityStressOutput",
    "RegimeLabel",
    "TaggedTrade",
    "TcaRegimeSegment",
    "TradeRecord",
    "liquidity_label_from_score",
    "regime_label_from_score",
]
