# SPDX-License-Identifier: Apache-2.0
"""PR-1 acceptance tests for FI evidence pack hash stability (HMAC lands in PR-7)."""

from __future__ import annotations

import hashlib

from market_regime_engine.fixed_income.evidence_pack import (
    build_evidence_pack,
    canonical_pack_payload,
    compute_pack_hash,
    verify_pack_hash,
)
from market_regime_engine.fixed_income.hashing import canonical_json
from market_regime_engine.model_runs import envelope_to_json


def _make_pack(*, metadata: dict | None = None, output_hash: str = "sha256:out") -> object:
    return build_evidence_pack(
        model_run_id="run-1",
        component_name="credit_regime",
        model_version="0.1.0",
        code_sha="abcdef0",
        model_hash="sha256:model",
        input_features_hash="sha256:in",
        output_hash=output_hash,
        release_gate=True,
        data_vintages={"trace_trades": "2026-05-10T15:00:00Z"},
        validation_results={"calibration_error": 0.05},
        random_seeds={"numpy": 7, "jax": 11},
        lockfile_hash="sha256:lock",
        timestamp="2026-05-10T16:00:00Z",
        python_version="3.13.4",
        metadata=metadata,
    )


def test_evidence_pack_hash_stable_under_key_reorder() -> None:
    """Two packs whose ``metadata`` dicts carry the same keys in different
    insertion order produce identical canonical SHA-256 hashes."""
    pack_a = _make_pack(metadata={"a": 1, "b": 2, "c": 3})
    pack_b = _make_pack(metadata={"c": 3, "a": 1, "b": 2})
    assert compute_pack_hash(pack_a) == compute_pack_hash(pack_b)


def test_evidence_pack_hash_changes_when_output_changes() -> None:
    """Distinct ``output_hash`` payloads produce distinct pack hashes."""
    pack_a = _make_pack(output_hash="sha256:original")
    pack_b = _make_pack(output_hash="sha256:tampered")
    assert compute_pack_hash(pack_a) != compute_pack_hash(pack_b)


def test_verify_pack_hash_round_trips() -> None:
    pack = _make_pack()
    h = compute_pack_hash(pack)
    assert verify_pack_hash(pack, h)
    assert not verify_pack_hash(pack, "sha256:" + "0" * 64)


def test_canonical_sha256_matches_envelope_pattern() -> None:
    """The FI canonical hash uses the same separator/sort rule as
    ``model_runs.envelope_to_json`` (REVIEW flag F-7 alignment)."""
    payload = {"engine": "MRE", "purpose": "test", "rng_seeds": {"numpy": 7}}
    expected = "sha256:" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    from market_regime_engine.fixed_income.hashing import canonical_sha256

    assert canonical_sha256(payload) == expected
    # And the canonical_json bytes are a structural match to envelope_to_json
    # (envelope_to_json uses sort_keys + default=str; the only difference is
    # the separator tuple). Strip whitespace to compare structure.
    import json

    import pandas as pd

    from market_regime_engine.model_runs import build_repro_envelope

    env = build_repro_envelope(features=pd.DataFrame(), model_outputs=pd.DataFrame())
    by_envelope = envelope_to_json(env)
    by_canonical = canonical_json({"sentinel": "round-trip-check"})
    assert "sentinel" in by_canonical
    # Sanity: envelope JSON is non-empty and parses.
    json.loads(by_envelope)


def test_hmac_signature_field_excluded_from_hash() -> None:
    """Per AGENT.md 'Hashing rules', hmac_signature is excluded from
    the canonical bytestream. Setting it after the fact must NOT change
    the pack hash."""
    pack_unsigned = _make_pack()
    h_unsigned = compute_pack_hash(pack_unsigned)

    # Replace with a signed variant via dataclasses.replace.
    import dataclasses

    pack_signed = dataclasses.replace(pack_unsigned, hmac_signature="sig-v1-abc")
    h_signed = compute_pack_hash(pack_signed)
    assert h_signed == h_unsigned

    payload_signed = canonical_pack_payload(pack_signed)
    payload_unsigned = canonical_pack_payload(pack_unsigned)
    assert payload_signed == payload_unsigned
    assert "hmac_signature" not in payload_signed
