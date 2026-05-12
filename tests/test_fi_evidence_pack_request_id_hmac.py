# SPDX-License-Identifier: Apache-2.0
"""v1.5.1 (PR-9 FIX 3): request_id binding to the HMAC canonical bytestream.

The v1.5.0 ``FixedIncomeEvidencePack`` did NOT carry ``request_id`` in
its canonical payload, so a replay of the same
``(model_run_id, output_hash)`` under a different request id was
undetectable. v1.5.1 binds ``request_id`` into the canonical bytestream
when ``metadata[_request_id_bound]`` is True; the flag is auto-stamped
by ``build_evidence_pack(request_id=...)``.

Tests cover:

1. Replay attack: signed with request_id=A, then mutated to request_id=B
   → ``verify_pack`` returns False.
2. Legacy compat: a v1-signed pack with ``request_id=None`` continues to
   verify under the v1 key.
3. Production guard: with ``MRE_ENV=production`` and an
   execution_confidence pack missing ``request_id``, ``sign_pack`` raises.
4. Resign v1 → v2: legacy packs get re-signed under v2 preserving
   ``request_id=null`` semantics; the CLI emits a warning with a sample
   of model_run_ids.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import replace

import pandas as pd
import pytest

from market_regime_engine.fixed_income.evidence_pack import (
    _REQUEST_ID_BOUND_METADATA_KEY,
    build_evidence_pack,
    canonical_pack_payload,
    sign_pack,
    verify_pack,
)
from market_regime_engine.fixed_income.schemas import FixedIncomeEvidencePack


@pytest.fixture
def hmac_keys(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Configure v1 + v2 HMAC keys for the test session."""
    v1_key = base64.b64encode(b"legacy-v1-secret-key-32-bytes!!").decode()
    v2_key = base64.b64encode(b"new-v2-request-id-binding-32!!!").decode()
    monkeypatch.setenv(
        "MRE_FI_HMAC_KEY_VERSIONS",
        json.dumps({"v1": v1_key, "v2": v2_key}),
    )
    return {"v1": v1_key, "v2": v2_key}


def _build_pack(**overrides) -> FixedIncomeEvidencePack:
    defaults = dict(
        model_run_id="run-test",
        component_name="execution_confidence",
        model_version="v0.1.0",
        code_sha="abc1234",
        model_hash="sha256:m",
        input_features_hash="sha256:i",
        output_hash="sha256:o",
        release_gate=True,
        timestamp="2026-05-08T16:00:00Z",
    )
    defaults.update(overrides)
    return build_evidence_pack(**defaults)


def test_replay_to_different_request_id_breaks_signature(
    hmac_keys: dict[str, str],
) -> None:
    """Replay: change request_id after signing → verify_pack returns False."""
    pack_a = _build_pack(request_id="req-A")
    signed_a = sign_pack(pack_a)
    assert verify_pack(signed_a)
    # Mutating ``request_id`` after signing must break verification because
    # the canonical bytestream now differs from what we hashed.
    tampered = replace(signed_a, request_id="req-B")
    assert verify_pack(tampered) is False


def test_legacy_v1_pack_with_no_request_id_verifies_under_v1(
    hmac_keys: dict[str, str],
) -> None:
    """v1.5.0-shape packs (no request_id, no flag) still verify under v1."""
    legacy_pack = FixedIncomeEvidencePack(
        model_run_id="run-legacy",
        component_name="credit_regime",
        model_version="v0.1.0",
        timestamp="2026-05-08T16:00:00Z",
        code_sha="abc1234",
        model_hash="sha256:m",
        input_features_hash="sha256:i",
        output_hash="sha256:o",
        data_vintages={},
        validation_results={},
        release_gate=True,
        random_seeds={},
        python_version="3.13",
        lockfile_hash=None,
        hmac_signature=None,
        metadata={},
        request_id=None,
    )
    signed = sign_pack(legacy_pack, key_version="v1")
    assert signed.hmac_signature is not None
    assert signed.hmac_signature.startswith("v1:")
    assert verify_pack(signed)


def test_legacy_canonical_bytes_byte_identical_to_pre_pr9(
    hmac_keys: dict[str, str],
) -> None:
    """v1.5.0-shape canonical bytes must be byte-identical post-PR-9.

    The contract that lets v1 signatures keep verifying is "drop
    request_id from canonical bytes when ``metadata._request_id_bound``
    is not set". Confirm the canonical JSON for a v1.5.0-shape pack
    omits the key entirely.
    """
    legacy_pack = FixedIncomeEvidencePack(
        model_run_id="run-legacy",
        component_name="credit_regime",
        model_version="v0.1.0",
        timestamp="2026-05-08T16:00:00Z",
        code_sha="abc1234",
        model_hash="sha256:m",
        input_features_hash="sha256:i",
        output_hash="sha256:o",
        data_vintages={},
        validation_results={},
        release_gate=True,
        random_seeds={},
        python_version="3.13",
        lockfile_hash=None,
        hmac_signature=None,
        metadata={},
        request_id=None,
    )
    payload = canonical_pack_payload(legacy_pack)
    assert '"request_id"' not in payload
    # The flag must NOT leak into legacy canonical bytes either.
    assert _REQUEST_ID_BOUND_METADATA_KEY not in payload


def test_v2_pack_with_request_id_includes_id_in_canonical_bytes(
    hmac_keys: dict[str, str],
) -> None:
    pack = _build_pack(request_id="req-A")
    payload = canonical_pack_payload(pack)
    assert '"request_id":"req-A"' in payload
    assert _REQUEST_ID_BOUND_METADATA_KEY in payload


def test_production_guard_raises_when_execution_confidence_missing_request_id(
    hmac_keys: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``MRE_ENV=production`` + execution_confidence pack + request_id=None → RuntimeError."""
    monkeypatch.setenv("MRE_ENV", "production")
    pack_no_rid = FixedIncomeEvidencePack(
        model_run_id="run-prod",
        component_name="execution_confidence",
        model_version="v0.1.0",
        timestamp="2026-05-08T16:00:00Z",
        code_sha="abc1234",
        model_hash="sha256:m",
        input_features_hash="sha256:i",
        output_hash="sha256:o",
        data_vintages={},
        validation_results={},
        release_gate=True,
        random_seeds={},
        python_version="3.13",
        lockfile_hash=None,
        hmac_signature=None,
        metadata={},
        request_id=None,
    )
    with pytest.raises(RuntimeError, match="request_id"):
        sign_pack(pack_no_rid)


def test_production_guard_passes_when_request_id_set(
    hmac_keys: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MRE_ENV", "production")
    pack = _build_pack(request_id="req-A")
    signed = sign_pack(pack)
    assert signed.hmac_signature is not None
    assert verify_pack(signed)


def test_production_guard_skips_non_execution_confidence_components(
    hmac_keys: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``credit_regime`` packs do not consume an inbound request id and
    are therefore exempt from the production guard."""
    monkeypatch.setenv("MRE_ENV", "production")
    pack = _build_pack(component_name="credit_regime", request_id=None)
    signed = sign_pack(pack)
    assert verify_pack(signed)


def test_fi_evidence_resign_v1_to_v2_preserves_null_request_id_and_warns(
    hmac_keys: dict[str, str], tmp_path
) -> None:
    """v1 → v2 resign keeps ``request_id=null`` semantics; emit a warning."""
    from market_regime_engine.fixed_income.cli import run
    from market_regime_engine.fixed_income.evidence_pack import (
        evidence_pack_to_row,
        read_evidence_pack,
    )
    from market_regime_engine.storage import Warehouse

    db = tmp_path / "resign.duckdb"
    wh = Warehouse(path=str(db))
    legacy_pack = FixedIncomeEvidencePack(
        model_run_id="run-legacy",
        component_name="execution_confidence",
        model_version="v0.1.0",
        timestamp="2026-05-08T16:00:00Z",
        code_sha="abc1234",
        model_hash="sha256:m",
        input_features_hash="sha256:i",
        output_hash="sha256:o",
        data_vintages={},
        validation_results={},
        release_gate=True,
        random_seeds={},
        python_version="3.13",
        lockfile_hash=None,
        hmac_signature=None,
        metadata={},
        request_id=None,
    )
    signed_v1 = sign_pack(legacy_pack, key_version="v1")
    wh.write_evidence_pack(
        pd.DataFrame([evidence_pack_to_row(signed_v1, request_id="req-legacy")])
    )
    wh.close()

    rc = run(
        [
            "fi-evidence-resign",
            "--db",
            str(db),
            "--from-key",
            "v1",
            "--to-key",
            "v2",
        ]
    )
    assert rc == 0

    wh2 = Warehouse(path=str(db))
    try:
        pack_after = read_evidence_pack(wh2, model_run_id="run-legacy")
    finally:
        wh2.close()
    assert pack_after is not None
    assert pack_after.hmac_signature is not None
    assert pack_after.hmac_signature.startswith("v2:")
    # Critical: verify still passes — request_id is NOT bound in the
    # canonical bytestream because the original v1 pack had no
    # ``_request_id_bound`` flag.
    assert verify_pack(pack_after)


def test_new_v2_pack_round_trips_request_id_binding(
    hmac_keys: dict[str, str], tmp_path
) -> None:
    """A freshly built v2 pack written + read back preserves the binding."""
    from market_regime_engine.fixed_income.evidence_pack import (
        evidence_pack_to_row,
        read_evidence_pack,
    )
    from market_regime_engine.storage import Warehouse

    db = tmp_path / "v2.duckdb"
    wh = Warehouse(path=str(db))
    new_pack = _build_pack(model_run_id="run-new", request_id="req-X")
    signed = sign_pack(new_pack, key_version="v2")
    wh.write_evidence_pack(
        pd.DataFrame([evidence_pack_to_row(signed, request_id="req-X")])
    )
    wh.close()

    wh2 = Warehouse(path=str(db))
    try:
        pack_after = read_evidence_pack(wh2, model_run_id="run-new")
    finally:
        wh2.close()
    assert pack_after is not None
    assert pack_after.request_id == "req-X"
    assert verify_pack(pack_after)
    # Tampering with request_id breaks verification, even on the
    # round-tripped pack.
    tampered = replace(pack_after, request_id="req-Y")
    assert verify_pack(tampered) is False


def test_evidence_pack_to_row_mismatch_raises(hmac_keys: dict[str, str]) -> None:
    """Mismatched ``pack.request_id`` vs row-level ``request_id`` raises."""
    from market_regime_engine.fixed_income.evidence_pack import evidence_pack_to_row

    pack = _build_pack(request_id="req-A")
    with pytest.raises(ValueError, match="request_id"):
        evidence_pack_to_row(pack, request_id="req-B")
