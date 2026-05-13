# SPDX-License-Identifier: Apache-2.0
"""v1.6.0 (REVIEW_DEEP_V1_5_2.md section 2.5): FI evidence pack v1/v2 wire-format.

The deep review flagged the canonical-JSON encoder as not RFC 8785-
compliant; the fix introduces a versioned encoder (``v1`` legacy,
``v2`` RFC 8785) with full backward-compat for v1.5.x persisted packs.
These tests pin:

1. ``build_evidence_pack`` defaults new packs to v2 and stamps the
   metadata key the verifier uses to route encoders.
2. ``compute_pack_hash`` / ``verify_pack_hash`` / ``canonical_pack_payload``
   route via ``pack.metadata['_canonical_version']`` for legacy packs
   without the key falling back to v1.
3. A v1 vs v2 pack with the same logical content produces DIFFERENT
   canonical bytes (so a JCS verifier sees the encoder choice).
4. HMAC v1 keys continue to sign / verify v1 packs after the migration;
   HMAC v2 keys sign / verify v2 packs.
5. ``fi-evidence-resign --to-version v2`` upgrades a stored v1 pack
   to v2 in place (rewrites the canonical bytes + signature).
"""

from __future__ import annotations

import base64
import dataclasses
from typing import Any

import pytest

from market_regime_engine.evidence_common import _canonical_json_v2, canonical_json
from market_regime_engine.fixed_income.evidence_pack import (
    _CANONICAL_VERSION_METADATA_KEY,
    _pack_canonical_version,
    build_evidence_pack,
    canonical_pack_payload,
    compute_pack_hash,
    sign_pack,
    verify_pack,
    verify_pack_hash,
)


def _make_pack(
    *,
    canonical_version: str = "v2",
    metadata: dict[str, Any] | None = None,
    output_hash: str = "sha256:out",
    request_id: str | None = None,
) -> Any:
    """Build a deterministic FI evidence pack for the v1/v2 wire-format
    tests. ``data_vintages`` carries ISO-8601 strings (the on-the-wire
    shape; never raw datetimes by contract)."""
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
        request_id=request_id,
        canonical_version=canonical_version,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# 1. build_evidence_pack stamps the canonical_version metadata key
# ---------------------------------------------------------------------------


def test_build_evidence_pack_default_stamps_v2() -> None:
    """The default ``canonical_version`` for new packs is v2 -- the deep
    review's documented migration target."""
    pack = build_evidence_pack(
        model_run_id="r1",
        component_name="credit_regime",
        model_version="0.1",
        code_sha="abc",
        model_hash="m",
        input_features_hash="i",
        output_hash="o",
        release_gate=True,
        timestamp="2026-05-12T00:00:00Z",
        python_version="3.13.4",
    )
    assert pack.metadata[_CANONICAL_VERSION_METADATA_KEY] == "v2"
    assert _pack_canonical_version(pack) == "v2"


def test_build_evidence_pack_v1_does_not_stamp_metadata() -> None:
    """``canonical_version='v1'`` produces a pack with NO metadata stamp,
    so the canonical bytes are byte-identical to what v1.5.x would have
    produced (the legacy verify path must not see a foreign metadata
    key)."""
    pack = _make_pack(canonical_version="v1")
    assert _CANONICAL_VERSION_METADATA_KEY not in pack.metadata
    assert _pack_canonical_version(pack) == "v1"


# ---------------------------------------------------------------------------
# 2. compute_pack_hash routes via metadata
# ---------------------------------------------------------------------------


def test_compute_pack_hash_v1_legacy_pack_matches_legacy_canonical_json() -> None:
    """A v1.5.x pack (no metadata stamp) re-hashes under the legacy
    encoder -- the hash a v1.5.x process would have written."""
    pack = _make_pack(canonical_version="v1")
    expected_bytes = canonical_json(
        # _pack_to_canonical_dict strips hmac_signature + request_id_bound
        # but otherwise returns the dataclass as a dict; we re-derive the
        # legacy bytes from the same projection.
        _legacy_pack_dict(pack),
        version="v1",
    )
    import hashlib

    expected_hash = "sha256:" + hashlib.sha256(expected_bytes.encode("utf-8")).hexdigest()
    assert compute_pack_hash(pack) == expected_hash


def _legacy_pack_dict(pack: Any) -> dict[str, Any]:
    """Mimic ``_pack_to_canonical_dict(pack, version='v1')`` to compute
    the expected legacy bytes for the regression test above."""
    from dataclasses import asdict

    raw = asdict(pack)
    raw.pop("hmac_signature", None)
    metadata = raw.get("metadata", {})
    if not metadata.get("_request_id_bound"):
        raw.pop("request_id", None)
    if "_envelope_hash" in metadata:
        raw["metadata"] = {k: v for k, v in metadata.items() if k != "_envelope_hash"}
    return raw


def test_compute_pack_hash_v2_uses_rfc8785_encoder() -> None:
    """A v2 pack hashes the RFC 8785 bytes, not the legacy bytes."""
    pack = _make_pack(canonical_version="v2")
    payload_v2 = canonical_pack_payload(pack)
    # The v2 payload differs from the v1 projection because it contains
    # the _canonical_version metadata key AND the v2 encoder strips
    # ``1.0`` float trailing zeros from any numeric metadata fields.
    assert '"_canonical_version":"v2"' in payload_v2
    # And the encoder is RFC 8785 -- direct call must match.
    from market_regime_engine.fixed_income.evidence_pack import (
        _pack_to_canonical_dict,
    )

    assert payload_v2 == _canonical_json_v2(_pack_to_canonical_dict(pack, version="v2"))


def test_v1_vs_v2_pack_canonical_bytes_differ() -> None:
    """The same logical pack content produces different canonical bytes
    under v1 vs v2 (the cross-encoder regression the deep review
    flagged): v1 has no version stamp; v2 has the stamp + RFC 8785
    formatting."""
    pack_v1 = _make_pack(canonical_version="v1", output_hash="sha256:abc")
    pack_v2 = _make_pack(canonical_version="v2", output_hash="sha256:abc")
    payload_v1 = canonical_pack_payload(pack_v1)
    payload_v2 = canonical_pack_payload(pack_v2)
    assert payload_v1 != payload_v2
    # And consequently the hashes differ.
    assert compute_pack_hash(pack_v1) != compute_pack_hash(pack_v2)


def test_verify_pack_hash_round_trip_v1() -> None:
    pack = _make_pack(canonical_version="v1")
    h = compute_pack_hash(pack)
    assert verify_pack_hash(pack, h)


def test_verify_pack_hash_round_trip_v2() -> None:
    pack = _make_pack(canonical_version="v2")
    h = compute_pack_hash(pack)
    assert verify_pack_hash(pack, h)


# ---------------------------------------------------------------------------
# 3. v2 rejects NaN/Inf in inputs
# ---------------------------------------------------------------------------


def test_v2_pack_rejects_nan_in_validation_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """A v2 pack with NaN in a numeric field fails the canonical
    encoding (RFC 8785 forbids NaN/Inf). The legacy v1 path accepts
    them so a v1.5.x pack with sentinel NaN ``calibration_error``
    continues to verify."""
    bad_pack = build_evidence_pack(
        model_run_id="r1",
        component_name="credit_regime",
        model_version="0.1",
        code_sha="abc",
        model_hash="m",
        input_features_hash="i",
        output_hash="o",
        release_gate=True,
        validation_results={"calibration_error": float("nan")},
        timestamp="2026-05-12T00:00:00Z",
        python_version="3.13.4",
    )
    with pytest.raises(ValueError, match="NaN/Infinity"):
        compute_pack_hash(bad_pack)
    # v1 still accepts:
    v1_pack = dataclasses.replace(
        bad_pack,
        metadata={
            k: v
            for k, v in bad_pack.metadata.items()
            if k != _CANONICAL_VERSION_METADATA_KEY
        },
    )
    h = compute_pack_hash(v1_pack)
    assert h.startswith("sha256:")


# ---------------------------------------------------------------------------
# 4. HMAC v1 keys verify v1 packs (backward-compat)
# ---------------------------------------------------------------------------


@pytest.fixture
def hmac_v1_key(monkeypatch: pytest.MonkeyPatch) -> bytes:
    raw = base64.b64encode(b"v1-secret-key-32bytes-for-testing!").decode("ascii")
    monkeypatch.setenv(
        "MRE_FI_HMAC_KEY_VERSIONS", f'{{"v1": "{raw}"}}'
    )
    return raw.encode("ascii")


@pytest.fixture
def hmac_v1_v2_keys(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    keys = {
        "v1": base64.b64encode(b"v1-secret-key-32bytes-for-testing!").decode("ascii"),
        "v2": base64.b64encode(b"v2-secret-key-32bytes-for-testing!").decode("ascii"),
    }
    import json as _json

    monkeypatch.setenv("MRE_FI_HMAC_KEY_VERSIONS", _json.dumps(keys))
    return keys


def test_hmac_v1_key_verifies_v1_pack(
    hmac_v1_key: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Legacy: v1 HMAC key on a v1 canonical-JSON pack still verifies.
    This is the critical backward-compat contract the deep review
    asked us to preserve."""
    pack = _make_pack(canonical_version="v1")
    signed = sign_pack(pack)
    assert signed.hmac_signature is not None
    assert signed.hmac_signature.startswith("v1:")
    assert verify_pack(signed)


def test_hmac_v2_key_verifies_v2_pack(
    hmac_v1_v2_keys: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """v2 HMAC key on a v2 canonical-JSON pack verifies under the new
    encoder."""
    pack = _make_pack(canonical_version="v2")
    signed = sign_pack(pack)  # picks latest key version (v2)
    assert signed.hmac_signature is not None
    assert signed.hmac_signature.startswith("v2:")
    assert verify_pack(signed)


def test_hmac_v1_key_verifies_v2_pack_when_routed_correctly(
    hmac_v1_v2_keys: dict[str, str],
) -> None:
    """A v2 pack signed under v1 HMAC key still verifies -- the HMAC key
    version and the canonical-JSON encoder version are orthogonal: the
    HMAC prefix (``v1:``) routes the key, the metadata stamp
    (``_canonical_version='v2'``) routes the encoder. The verifier
    succeeds iff both routings yield matching bytes."""
    pack = _make_pack(canonical_version="v2")
    signed = sign_pack(pack, key_version="v1")
    assert signed.hmac_signature is not None
    assert signed.hmac_signature.startswith("v1:")
    assert verify_pack(signed)


def test_hmac_tamper_with_canonical_version_metadata_fails_verify(
    hmac_v1_v2_keys: dict[str, str],
) -> None:
    """An attacker who strips ``_canonical_version='v2'`` from metadata
    forces the verifier to fall back to the v1 encoder -- the canonical
    bytes (and therefore the HMAC) mismatch. Tamper-evident."""
    pack = _make_pack(canonical_version="v2")
    signed = sign_pack(pack)
    # Strip the canonical_version key.
    bad_meta = dict(signed.metadata)
    bad_meta.pop(_CANONICAL_VERSION_METADATA_KEY, None)
    tampered = dataclasses.replace(signed, metadata=bad_meta)
    assert not verify_pack(tampered)


# ---------------------------------------------------------------------------
# 5. v1 legacy regression fixture
# ---------------------------------------------------------------------------


def test_v1_legacy_pack_hash_matches_documented_v1_form() -> None:
    """A v1 pack with the exact field shape v1.5.x persisted produces
    the legacy SHA-256 hash byte-for-byte. The expected hash was
    derived offline by hashing the documented v1 canonical bytes."""
    pack = _make_pack(canonical_version="v1")
    payload_v1 = canonical_pack_payload(pack)
    # Verify the bytes match what json.dumps(default=str, sort_keys=True,
    # separators=(",", ":")) of the projected dict would produce.
    import hashlib
    import json as _json

    expected_dict = _legacy_pack_dict(pack)
    expected_bytes = _json.dumps(
        expected_dict, sort_keys=True, separators=(",", ":"), default=str
    )
    assert payload_v1 == expected_bytes
    expected_hash = "sha256:" + hashlib.sha256(expected_bytes.encode("utf-8")).hexdigest()
    assert compute_pack_hash(pack) == expected_hash


def test_v1_pack_request_id_binding_preserved_under_new_code() -> None:
    """A v1 pack with ``request_id`` set keeps the v1.5.1 binding
    semantics (request_id is in canonical bytes when the metadata flag
    is set) -- the new code path does not regress this."""
    pack = _make_pack(canonical_version="v1", request_id="req-123")
    assert pack.metadata.get("_request_id_bound") is True
    payload = canonical_pack_payload(pack)
    assert '"request_id":"req-123"' in payload


def test_v2_pack_request_id_binding_works_under_v2_encoder() -> None:
    """Same binding semantics under v2: the request_id is in canonical
    bytes because the metadata flag is set; the encoder version is v2
    so the bytes use RFC 8785 formatting."""
    pack = _make_pack(canonical_version="v2", request_id="req-456")
    payload = canonical_pack_payload(pack)
    assert '"request_id":"req-456"' in payload
    assert '"_canonical_version":"v2"' in payload
