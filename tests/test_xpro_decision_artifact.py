# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import copy

from market_regime_engine.fixed_income import ExecutionConfidenceRequest
from market_regime_engine.fixed_income.xpro_decision import (
    build_xpro_decision_artifact,
    sign_xpro_decision_artifact,
    verify_xpro_decision_artifact,
)


class _Warehouse:
    pass


def _request(metadata: dict | None = None) -> ExecutionConfidenceRequest:
    return ExecutionConfidenceRequest(
        timestamp="2026-05-26T12:31:00Z",
        cusip="123456AB7",
        side="buy",
        notional=5_000_000.0,
        protocol="Auto-X",
        limit_price=101.25,
        urgency="normal",
        rating="BBB+",
        metadata={"mid_price": "101.00"} if metadata is None else metadata,
    )


def test_xpro_decision_artifact_is_fixed_point_and_hash_stable(monkeypatch) -> None:
    monkeypatch.setattr(
        "market_regime_engine.fixed_income.xpro_decision.recommend_execution_protocol",
        lambda request, warehouse, **kwargs: kwargs["recommendation"],
    )
    from market_regime_engine.fixed_income.protocol_recommendation import (
        ProtocolRecommendation,
        ProtocolScore,
    )
    from market_regime_engine.fixed_income.schemas import ExecutionConfidenceResponse

    response = ExecutionConfidenceResponse(
        timestamp="2026-05-26T12:31:00Z",
        cusip="123456AB7",
        side="buy",
        notional=5_000_000.0,
        protocol="RFQ",
        confidence_score=0.61244,
        expected_slippage_bps=12.125,
        confidence_interval_low=0.51244,
        confidence_interval_high=0.71244,
        recommended_action="Auto-X caution / trader confirm",
        human_review_required=False,
        model_run_id="run-1",
        release_gate=True,
        artifact_hash="sha256:exec",
        metadata={"regime_score": 35.0, "liquidity_index": 25.0},
    )
    recommendation = ProtocolRecommendation(
        request=_request(),
        recommended_protocol="RFQ",
        candidate_scores=(
            ProtocolScore("Auto-X", 0.478801, False, "Unavailable", "sha256:auto", True),
            ProtocolScore("RFQ", 0.61244, True, "Auto-X caution / trader confirm", "sha256:rfq", False),
        ),
        best_response=response,
        release_gate=True,
        human_review_required=False,
        reason_codes=("rfq_ranked_best_counterfactual",),
    )
    artifact = build_xpro_decision_artifact(
        _request(),
        warehouse=_Warehouse(),
        request_id="req-1",
        decision_id="dec-1",
        recommendation=recommendation,
    )
    assert artifact["asof_epoch_ns"] == "1779798660000000000"
    assert "created_epoch_ns" not in artifact
    assert artifact["input"]["notional"]["value"] == 500000000
    assert artifact["input"]["limit_price"]["value"] == 101250000
    assert artifact["model_outputs"]["execution_confidence"]["score_ppm"] == 612440
    assert artifact["evidence"]["artifact_hash"].startswith("sha256:")
    assert (
        build_xpro_decision_artifact(
            _request(),
            warehouse=_Warehouse(),
            request_id="req-1",
            decision_id="dec-1",
            recommendation=recommendation,
        )["evidence"]["artifact_hash"]
        == artifact["evidence"]["artifact_hash"]
    )


def test_xpro_metadata_hash_canonicalizes_nested_values(monkeypatch) -> None:
    from market_regime_engine.evidence_common import canonical_sha256
    from market_regime_engine.fixed_income.protocol_recommendation import (
        ProtocolRecommendation,
        ProtocolScore,
    )
    from market_regime_engine.fixed_income.schemas import ExecutionConfidenceResponse

    monkeypatch.setattr(
        "market_regime_engine.fixed_income.xpro_decision.recommend_execution_protocol",
        lambda request, warehouse, **kwargs: kwargs["recommendation"],
    )
    response = ExecutionConfidenceResponse(
        timestamp="2026-05-26T12:31:00Z",
        cusip="123456AB7",
        side="buy",
        notional=5_000_000.0,
        protocol="RFQ",
        confidence_score=0.61244,
        expected_slippage_bps=12.125,
        confidence_interval_low=0.51244,
        confidence_interval_high=0.71244,
        recommended_action="Auto-X caution / trader confirm",
        human_review_required=False,
        model_run_id="run-1",
        release_gate=True,
        artifact_hash="sha256:exec",
    )
    recommendation = ProtocolRecommendation(
        request=_request(),
        recommended_protocol="RFQ",
        candidate_scores=(ProtocolScore("RFQ", 0.61244, True, "Auto-X caution / trader confirm", "sha256:rfq", False),),
        best_response=response,
        release_gate=True,
        human_review_required=False,
        reason_codes=("rfq_ranked_best_counterfactual",),
    )
    metadata_a = {
        "nested": {"b": 2, "a": 1},
        "items": [{"z": "last", "a": "first"}],
        "missing": None,
    }
    metadata_b = {
        "missing": None,
        "items": [{"a": "first", "z": "last"}],
        "nested": {"a": 1, "b": 2},
    }

    artifact_a = build_xpro_decision_artifact(
        _request(metadata_a),
        warehouse=_Warehouse(),
        request_id="req-1",
        decision_id="dec-1",
        recommendation=recommendation,
    )
    artifact_b = build_xpro_decision_artifact(
        _request(metadata_b),
        warehouse=_Warehouse(),
        request_id="req-1",
        decision_id="dec-1",
        recommendation=recommendation,
    )

    expected_hash = canonical_sha256(
        {
            "items": [{"a": "first", "z": "last"}],
            "missing": None,
            "nested": {"a": 1, "b": 2},
        },
        version="v2",
    )
    legacy_repr_hash = canonical_sha256(
        {
            "items": "[{'z': 'last', 'a': 'first'}]",
            "missing": None,
            "nested": "{'b': 2, 'a': 1}",
        },
        version="v2",
    )

    assert artifact_a["input"]["metadata_hash"] == expected_hash
    assert artifact_b["input"]["metadata_hash"] == expected_hash
    assert artifact_a["input"]["metadata_hash"] != legacy_repr_hash


def test_xpro_signature_verification_fails_on_tamper(monkeypatch) -> None:
    monkeypatch.setenv("MRE_FI_HMAC_KEY_VERSIONS", '{"v1":"secret"}')
    artifact = {
        "artifact_version": "xpro_decision_artifact_v1",
        "decision_id": "dec-1",
        "request_id": "req-1",
        "asof_epoch_ns": "1779798660000000000",
        "numeric_policy": {"prob_scale": 1000000, "bps_scale": 10000},
        "input": {"cusip": "123456AB7"},
        "model_outputs": {},
        "decision": {"recommended_protocol": "RFQ", "release_gate": True},
        "lineage": {},
        "evidence": {"artifact_hash": "sha256:placeholder"},
    }
    signed = sign_xpro_decision_artifact(artifact)
    assert verify_xpro_decision_artifact(signed)["verified"] is True
    tampered = copy.deepcopy(signed)
    tampered["decision"]["recommended_protocol"] = "Manual"
    assert verify_xpro_decision_artifact(tampered)["verified"] is False


def test_unsigned_xpro_artifact_verifies_by_hash_in_dev(monkeypatch) -> None:
    monkeypatch.delenv("MRE_FI_HMAC_KEY_VERSIONS", raising=False)
    monkeypatch.delenv("MRE_FI_HMAC_KEY", raising=False)
    monkeypatch.delenv("MRE_FI_REQUIRE_HMAC", raising=False)
    monkeypatch.delenv("MRE_ENV", raising=False)
    artifact = {
        "artifact_version": "xpro_decision_artifact_v1",
        "decision_id": "dec-unsigned",
        "request_id": "req-unsigned",
        "asof_epoch_ns": "1779798660000000000",
        "numeric_policy": {"prob_scale": 1000000, "bps_scale": 10000},
        "input": {"cusip": "123456AB7"},
        "model_outputs": {},
        "decision": {"recommended_protocol": "RFQ", "release_gate": True},
        "lineage": {},
        "evidence": {"artifact_hash": "sha256:placeholder"},
    }
    unsigned = sign_xpro_decision_artifact(artifact)
    result = verify_xpro_decision_artifact(unsigned)
    assert result["verified"] is True
    assert result["hmac_required"] is False
    assert result["hmac_valid"] is None
    assert result["reasons"] == []

    tampered = copy.deepcopy(unsigned)
    tampered["decision"]["recommended_protocol"] = "Manual"
    assert verify_xpro_decision_artifact(tampered)["verified"] is False


def test_unsigned_xpro_artifact_fails_when_hmac_required(monkeypatch) -> None:
    monkeypatch.delenv("MRE_FI_HMAC_KEY_VERSIONS", raising=False)
    monkeypatch.delenv("MRE_FI_HMAC_KEY", raising=False)
    monkeypatch.delenv("MRE_FI_REQUIRE_HMAC", raising=False)
    monkeypatch.delenv("MRE_ENV", raising=False)
    artifact = sign_xpro_decision_artifact(
        {
            "artifact_version": "xpro_decision_artifact_v1",
            "decision_id": "dec-required",
            "request_id": "req-required",
            "asof_epoch_ns": "1779798660000000000",
            "numeric_policy": {"prob_scale": 1000000, "bps_scale": 10000},
            "input": {"cusip": "123456AB7"},
            "model_outputs": {},
            "decision": {"recommended_protocol": "RFQ", "release_gate": True},
            "lineage": {},
            "evidence": {"artifact_hash": "sha256:placeholder"},
        }
    )
    forced = verify_xpro_decision_artifact(artifact, require_hmac=True)
    assert forced["verified"] is False
    assert forced["hmac_required"] is True
    assert "hmac_missing" in forced["reasons"]

    monkeypatch.setenv("MRE_FI_REQUIRE_HMAC", "true")
    production_required = verify_xpro_decision_artifact(artifact)
    assert production_required["verified"] is False
    assert production_required["hmac_required"] is True
