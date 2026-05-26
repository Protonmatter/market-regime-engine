# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy
import json
import uuid
from collections.abc import Mapping
from typing import Any

import pandas as pd

from market_regime_engine.evidence_common import canonical_json, canonical_sha256, hmac_sha256_hex
from market_regime_engine.fixed_income.evidence_pack import (
    get_hmac_keys,
    latest_hmac_version,
    require_production_hmac,
)
from market_regime_engine.fixed_income.numeric_contracts import (
    DEFAULT_NUMERIC_POLICY,
    assert_no_float_artifact,
    bps_to_q4,
    money_to_cents,
    price_to_q6,
    prob_to_ppm,
    timestamp_to_epoch_ns_str,
)
from market_regime_engine.fixed_income.protocol_recommendation import (
    ProtocolRecommendation,
    ProtocolScore,
    recommend_execution_protocol,
)
from market_regime_engine.fixed_income.schemas import ExecutionConfidenceRequest
from market_regime_engine.fixed_income.timestamps import iso8601_z, to_utc

ARTIFACT_VERSION = "xpro_decision_artifact_v1"


def _safe_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def _metadata_hash(metadata: Mapping[str, Any] | None) -> str:
    text_safe = {str(k): _safe_text(v) for k, v in dict(metadata or {}).items()}
    return canonical_sha256(text_safe, version="v2")


def _candidate_to_artifact(score: ProtocolScore) -> dict[str, Any]:
    return {
        "protocol": str(score.protocol),
        "execution_score_ppm": prob_to_ppm(score.confidence_score),
        "expected_slippage_bps_q4": (
            bps_to_q4(score.expected_slippage_bps) if score.expected_slippage_bps is not None else None
        ),
        "confidence_interval_low_ppm": (
            prob_to_ppm(score.confidence_interval_low) if score.confidence_interval_low is not None else None
        ),
        "confidence_interval_high_ppm": (
            prob_to_ppm(score.confidence_interval_high) if score.confidence_interval_high is not None else None
        ),
        "recommended_action": str(score.recommended_action),
        "human_review_required": bool(score.human_review_required),
        "release_gate": bool(score.release_gate),
        "model_run_id": _safe_text(score.model_run_id),
        "artifact_hash": str(score.artifact_hash),
        "reason": _safe_text(score.reason),
    }


def _hash_payload(artifact: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(artifact))
    evidence = dict(payload.get("evidence") or {})
    evidence.pop("artifact_hash", None)
    evidence.pop("hmac", None)
    payload["evidence"] = evidence
    return payload


def _hmac_payload(artifact: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(artifact))
    evidence = dict(payload.get("evidence") or {})
    evidence.pop("hmac", None)
    payload["evidence"] = evidence
    return payload


def _refresh_artifact_hash(artifact: dict[str, Any]) -> dict[str, Any]:
    artifact = copy.deepcopy(artifact)
    artifact.setdefault("evidence", {})
    artifact["evidence"].pop("hmac", None)
    artifact["evidence"]["artifact_hash"] = canonical_sha256(_hash_payload(artifact), version="v2")
    return artifact


def build_xpro_decision_artifact(
    request: ExecutionConfidenceRequest,
    *,
    warehouse: Any,
    request_id: str,
    decision_id: str | None = None,
    candidate_protocols: tuple[str, ...] | list[str] | None = None,
    recommendation: ProtocolRecommendation | None = None,
    sign: bool = True,
    **recommend_kwargs: Any,
) -> dict[str, Any]:
    """Build a deterministic fixed-point XPro execution decision artifact."""

    if not request_id:
        raise ValueError("request_id must be non-empty")
    if recommendation is None:
        recommendation = recommend_execution_protocol(
            request,
            warehouse=warehouse,
            candidate_protocols=candidate_protocols,
            **recommend_kwargs,
        )

    best = recommendation.best_response
    asof_epoch_ns = timestamp_to_epoch_ns_str(request.timestamp)
    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "decision_id": decision_id or uuid.uuid4().hex,
        "request_id": str(request_id),
        "asof_utc": iso8601_z(to_utc(request.timestamp)),
        "asof_epoch_ns": asof_epoch_ns,
        "model_run_id": str(best.model_run_id),
        "numeric_policy": DEFAULT_NUMERIC_POLICY.to_dict(),
        "input": {
            "asof_epoch_ns": asof_epoch_ns,
            "cusip": str(request.cusip),
            "side": str(request.side),
            "notional": {
                "value": money_to_cents(request.notional),
                "scale": "cents",
            },
            "protocol_requested": str(request.protocol),
            "limit_price": (
                {"value": price_to_q6(request.limit_price), "scale": "q6"} if request.limit_price is not None else None
            ),
            "urgency": _safe_text(request.urgency),
            "sector": _safe_text(request.sector),
            "rating": _safe_text(request.rating),
            "maturity_bucket": _safe_text(request.maturity_bucket),
            "client_request_id": _safe_text(request.client_request_id),
            "metadata_hash": _metadata_hash(request.metadata),
        },
        "candidate_protocol_scores": [_candidate_to_artifact(score) for score in recommendation.candidate_scores],
        "model_outputs": {
            "execution_confidence": {
                "protocol": str(best.protocol),
                "score_ppm": prob_to_ppm(best.confidence_score),
                "expected_slippage_bps_q4": (
                    bps_to_q4(best.expected_slippage_bps) if best.expected_slippage_bps is not None else None
                ),
                "confidence_interval_low_ppm": (
                    prob_to_ppm(best.confidence_interval_low) if best.confidence_interval_low is not None else None
                ),
                "confidence_interval_high_ppm": (
                    prob_to_ppm(best.confidence_interval_high) if best.confidence_interval_high is not None else None
                ),
                "recommended_action": str(best.recommended_action),
                "release_gate": bool(best.release_gate),
                "human_review_required": bool(best.human_review_required),
            }
        },
        "decision": {
            "recommended_protocol": str(recommendation.recommended_protocol),
            "release_gate": bool(recommendation.release_gate),
            "human_review_required": bool(recommendation.human_review_required),
            "reason_codes": [str(code) for code in recommendation.reason_codes],
        },
        "auto_x_gate": {
            "requested_protocol": str(request.protocol) == "Auto-X",
            "selected_auto_x": str(recommendation.recommended_protocol) == "Auto-X",
            "release_gate": bool(recommendation.release_gate),
            "permitted": bool(
                recommendation.release_gate
                and str(recommendation.recommended_protocol) == "Auto-X"
                and not recommendation.human_review_required
            ),
        },
        "lineage": {
            "selected_model_run_id": str(best.model_run_id),
            "selected_execution_confidence_artifact_hash": str(best.artifact_hash),
            "candidate_execution_confidence_artifact_hashes": [
                str(score.artifact_hash) for score in recommendation.candidate_scores
            ],
        },
        "evidence": {
            "canonical_json": "rfc8785-jcs-v2",
            "artifact_hash": "",
        },
    }
    artifact = _refresh_artifact_hash(artifact)
    assert_no_float_artifact(artifact)
    if sign:
        artifact = sign_xpro_decision_artifact(artifact)
    return artifact


def sign_xpro_decision_artifact(
    artifact: Mapping[str, Any],
    *,
    key_version: str | None = None,
) -> dict[str, Any]:
    """Return ``artifact`` with a current artifact hash and optional FI HMAC."""

    signed = _refresh_artifact_hash(dict(artifact))
    keys = get_hmac_keys()
    if not keys:
        assert_no_float_artifact(signed)
        return signed
    version = key_version or latest_hmac_version()
    if version is None or version not in keys:
        raise RuntimeError(f"unknown FI HMAC key version: {version!r}")
    payload = canonical_json(_hmac_payload(signed), version="v2").encode("utf-8")
    signed["evidence"]["hmac"] = {
        "algorithm": "HMAC-SHA256",
        "key_version": str(version),
        "digest_hex": hmac_sha256_hex(keys[version], payload),
        "canonical_json": "rfc8785-jcs-v2",
    }
    assert_no_float_artifact(signed)
    return signed


def verify_xpro_decision_artifact(
    artifact: Mapping[str, Any],
    *,
    require_hmac: bool | None = None,
) -> dict[str, Any]:
    """Verify artifact canonical hash and HMAC when present."""

    reasons: list[str] = []
    hmac_required = require_production_hmac() if require_hmac is None else bool(require_hmac)
    payload = copy.deepcopy(dict(artifact))
    evidence = dict(payload.get("evidence") or {})
    supplied_hash = str(evidence.get("artifact_hash") or "")
    computed_hash = canonical_sha256(_hash_payload(payload), version="v2")
    hash_valid = bool(supplied_hash and supplied_hash == computed_hash)
    if not hash_valid:
        reasons.append("artifact_hash_mismatch")

    hmac_valid: bool | None = None
    hmac_obj = evidence.get("hmac")
    if isinstance(hmac_obj, Mapping):
        version = str(hmac_obj.get("key_version") or "")
        digest = str(hmac_obj.get("digest_hex") or "")
        keys = get_hmac_keys()
        key = keys.get(version)
        if key is None:
            hmac_valid = False
            reasons.append("hmac_key_missing")
        else:
            expected = hmac_sha256_hex(
                key,
                canonical_json(_hmac_payload(payload), version="v2").encode("utf-8"),
            )
            hmac_valid = bool(digest and digest == expected)
            if not hmac_valid:
                reasons.append("hmac_mismatch")
    else:
        hmac_valid = None
        if hmac_required:
            reasons.append("hmac_missing")

    verified = hash_valid and (hmac_valid is True or (hmac_valid is None and not hmac_required))
    return {
        "verified": bool(verified),
        "artifact_hash_valid": bool(hash_valid),
        "hmac_required": bool(hmac_required),
        "hmac_valid": bool(hmac_valid) if hmac_valid is not None else None,
        "artifact_hash": computed_hash,
        "reasons": reasons,
    }


def xpro_decision_artifact_to_row(artifact: Mapping[str, Any]) -> dict[str, Any]:
    evidence = dict(artifact.get("evidence") or {})
    hmac = evidence.get("hmac")
    if isinstance(hmac, Mapping):
        hmac_signature = f"{hmac.get('key_version')}:{hmac.get('digest_hex')}"
    else:
        hmac_signature = None
    decision = dict(artifact.get("decision") or {})
    return {
        "decision_id": str(artifact.get("decision_id")),
        "request_id": str(artifact.get("request_id")),
        "timestamp": str(artifact.get("asof_utc")),
        "model_run_id": str(artifact.get("model_run_id")),
        "recommended_protocol": str(decision.get("recommended_protocol")),
        "release_gate": bool(decision.get("release_gate")),
        "artifact_hash": str(evidence.get("artifact_hash") or ""),
        "hmac_signature": hmac_signature,
        "payload_json": json.dumps(dict(artifact), sort_keys=True, separators=(",", ":")),
        "metadata_json": json.dumps(
            {
                "artifact_version": artifact.get("artifact_version"),
                "numeric_policy": artifact.get("numeric_policy"),
                "reason_codes": decision.get("reason_codes", []),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
    }


__all__ = [
    "ARTIFACT_VERSION",
    "build_xpro_decision_artifact",
    "sign_xpro_decision_artifact",
    "verify_xpro_decision_artifact",
    "xpro_decision_artifact_to_row",
]
