# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from market_regime_engine.fixed_income.execution_confidence import score_execution_confidence
from market_regime_engine.fixed_income.schemas import (
    ExecutionConfidenceRequest,
    ExecutionConfidenceResponse,
)

DEFAULT_CANDIDATE_PROTOCOLS: tuple[str, ...] = ("Auto-X", "RFQ", "Manual")
DEFAULT_TIE_BREAK_ORDER: tuple[str, ...] = ("RFQ", "Auto-X", "Manual")


@dataclass(frozen=True)
class ProtocolScore:
    protocol: str
    confidence_score: float
    release_gate: bool
    recommended_action: str
    artifact_hash: str
    human_review_required: bool
    expected_slippage_bps: float | None = None
    confidence_interval_low: float | None = None
    confidence_interval_high: float | None = None
    model_run_id: str | None = None
    reason: str | None = None


ProtocolCandidateScore = ProtocolScore


@dataclass(frozen=True)
class ProtocolRecommendation:
    request: ExecutionConfidenceRequest
    recommended_protocol: str
    candidate_scores: tuple[ProtocolScore, ...]
    best_response: ExecutionConfidenceResponse
    release_gate: bool
    human_review_required: bool
    reason_codes: tuple[str, ...]


def _candidate_run_id(base: str | None, protocol: str) -> str | None:
    if not base:
        return None
    suffix = protocol.lower().replace("-", "").replace(" ", "_")
    return f"{base}-{suffix}"


def _tie_break(protocol: str, candidate_protocols: tuple[str, ...], explicit_order: bool) -> int:
    order = candidate_protocols if explicit_order else DEFAULT_TIE_BREAK_ORDER
    try:
        return order.index(protocol)
    except ValueError:
        return len(order) + candidate_protocols.index(protocol)


def recommend_execution_protocol(
    request: ExecutionConfidenceRequest,
    *,
    warehouse: Any,
    candidate_protocols: tuple[str, ...] | list[str] | None = None,
    model_run_id: str | None = None,
    release_gate: bool = True,
    profile: str = "production",
    weights: dict[str, float] | None = None,
    coefficients: Any | None = None,
    use_empirical_calibration: bool = True,
) -> ProtocolRecommendation:
    """Rank candidate execution protocols using counterfactual scorer calls."""

    explicit_order = candidate_protocols is not None
    candidates = tuple(candidate_protocols or DEFAULT_CANDIDATE_PROTOCOLS)
    if not candidates:
        raise ValueError("candidate_protocols must contain at least one protocol")

    responses: list[ExecutionConfidenceResponse] = []
    scores: list[ProtocolScore] = []
    for protocol in candidates:
        protocol_request = replace(request, protocol=str(protocol))
        response = score_execution_confidence(
            protocol_request,
            warehouse=warehouse,
            model_run_id=_candidate_run_id(model_run_id, str(protocol)),
            release_gate=release_gate,
            profile=profile,
            weights=weights,
            coefficients=coefficients,
            use_empirical_calibration=use_empirical_calibration,
        )
        responses.append(response)
        scores.append(
            ProtocolScore(
                protocol=str(protocol),
                confidence_score=float(response.confidence_score),
                release_gate=bool(response.release_gate),
                recommended_action=response.recommended_action,
                artifact_hash=response.artifact_hash,
                human_review_required=bool(response.human_review_required),
                expected_slippage_bps=(
                    float(response.expected_slippage_bps)
                    if response.expected_slippage_bps is not None
                    else None
                ),
                confidence_interval_low=(
                    float(response.confidence_interval_low)
                    if response.confidence_interval_low is not None
                    else None
                ),
                confidence_interval_high=(
                    float(response.confidence_interval_high)
                    if response.confidence_interval_high is not None
                    else None
                ),
                model_run_id=response.model_run_id,
                reason=str(dict(response.metadata).get("reason") or ""),
            )
        )

    response_by_protocol = {response.protocol: response for response in responses}
    passing = [score for score in scores if score.release_gate]
    if passing:
        best_score = sorted(
            passing,
            key=lambda s: (
                -float(s.confidence_score),
                _tie_break(s.protocol, candidates, explicit_order),
            ),
        )[0]
        reason = f"{best_score.protocol.lower().replace('-', '').replace(' ', '_')}_ranked_best_counterfactual"
        return ProtocolRecommendation(
            request=request,
            recommended_protocol=best_score.protocol,
            candidate_scores=tuple(scores),
            best_response=response_by_protocol[best_score.protocol],
            release_gate=True,
            human_review_required=bool(best_score.human_review_required),
            reason_codes=(reason,),
        )

    fallback_protocol = "Manual" if "Manual" in candidates else candidates[-1]
    fallback_response = response_by_protocol[fallback_protocol]
    return ProtocolRecommendation(
        request=request,
        recommended_protocol=fallback_protocol,
        candidate_scores=tuple(scores),
        best_response=fallback_response,
        release_gate=False,
        human_review_required=True,
        reason_codes=("no_candidate_release_gate_passed", "manual_fail_closed"),
    )


__all__ = [
    "DEFAULT_CANDIDATE_PROTOCOLS",
    "DEFAULT_TIE_BREAK_ORDER",
    "ProtocolCandidateScore",
    "ProtocolScore",
    "ProtocolRecommendation",
    "recommend_execution_protocol",
]
