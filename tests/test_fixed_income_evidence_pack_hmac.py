# SPDX-License-Identifier: Apache-2.0
"""PR-7 §A.1 — HMAC sign/verify acceptance tests for FI evidence packs.

Per AGENT.md non-negotiable 6 (every production output must be
reproducible from a tamper-evident pack) and INSTRUCTIONS.md §10
(governance rule 5: HMAC required in production).

The tests pin:

- ``sign_pack`` populates ``hmac_signature`` with the
  ``"v<ver>:<hex>"`` shape.
- ``verify_pack`` round-trips on an unmodified pack.
- Tampering with **any** field of the canonical bytestream fails
  verification (this is the canonical
  ``test_evidence_pack_hmac_rejects_tampering`` from AGENT.md).
- A signature signed under a key that is not in the configured map
  fails verification.
- A malformed signature (no version prefix, no separator) fails.
- Production mode without configured keys raises rather than silently
  returning an unsigned pack.
- Dev mode (no production env, no keys) passes through unsigned.
- ``hmac.compare_digest`` is used (constant-time) — verified by
  introspecting the call site source.
- Multi-version key resolution: signing under v2 verifies under v2
  even when v1 is also configured.
"""

from __future__ import annotations

import base64
import dataclasses
import inspect
import json
import secrets

import pytest

from market_regime_engine.fixed_income import evidence_pack as ep
from market_regime_engine.fixed_income.evidence_pack import (
    build_evidence_pack,
    canonical_pack_payload,
    get_hmac_keys,
    latest_hmac_version,
    require_production_hmac,
    sign_pack,
    verify_pack,
)


def _make_pack(**overrides):
    base = {
        "model_run_id": "run-1",
        "component_name": "credit_regime",
        "model_version": "0.1.0",
        "code_sha": "abcdef0",
        "model_hash": "sha256:model",
        "input_features_hash": "sha256:in",
        "output_hash": "sha256:out",
        "release_gate": True,
        "data_vintages": {"trace_trades": "2026-05-10T15:00:00Z"},
        "validation_results": {"calibration_error": 0.05},
        "random_seeds": {"numpy": 7},
        "lockfile_hash": "sha256:lock",
        "timestamp": "2026-05-10T16:00:00Z",
        "python_version": "3.13.4",
        "metadata": {"k": "v"},
    }
    base.update(overrides)
    return build_evidence_pack(**base)


def _b64key(n_bytes: int = 32) -> str:
    return base64.b64encode(secrets.token_bytes(n_bytes)).decode("ascii")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip every PR-7 HMAC env var before each test so tests can opt in.

    Without this the test order matters: a test that sets
    ``MRE_FI_HMAC_KEY_VERSIONS`` would leak into the next dev-mode
    test. ``monkeypatch.delenv`` with ``raising=False`` is a no-op
    when the var is unset.
    """
    for name in (
        "MRE_FI_HMAC_KEY_VERSIONS",
        "MRE_FI_HMAC_KEY",
        "MRE_FI_REQUIRE_HMAC",
        "MRE_ENV",
    ):
        monkeypatch.delenv(name, raising=False)
    yield


def test_sign_pack_populates_hmac_signature(monkeypatch):
    monkeypatch.setenv("MRE_FI_HMAC_KEY_VERSIONS", json.dumps({"v1": _b64key()}))
    pack = _make_pack()
    signed = sign_pack(pack)
    assert signed.hmac_signature is not None
    version, _, hex_digest = signed.hmac_signature.partition(":")
    assert version == "v1"
    assert len(hex_digest) == 64
    assert all(c in "0123456789abcdef" for c in hex_digest)


def test_verify_pack_returns_true_for_unmodified_pack(monkeypatch):
    monkeypatch.setenv("MRE_FI_HMAC_KEY_VERSIONS", json.dumps({"v1": _b64key()}))
    signed = sign_pack(_make_pack())
    assert verify_pack(signed) is True


def test_evidence_pack_hmac_rejects_tampering(monkeypatch):
    """Per AGENT.md test catalog — modifying any byte fails verification."""
    monkeypatch.setenv("MRE_FI_HMAC_KEY_VERSIONS", json.dumps({"v1": _b64key()}))
    signed = sign_pack(_make_pack())
    tampered = dataclasses.replace(signed, output_hash="sha256:tampered")
    assert verify_pack(tampered) is False
    tampered2 = dataclasses.replace(signed, model_version="9.9.9")
    assert verify_pack(tampered2) is False
    tampered3 = dataclasses.replace(signed, release_gate=False)
    assert verify_pack(tampered3) is False


def test_verify_pack_rejects_wrong_key(monkeypatch):
    monkeypatch.setenv("MRE_FI_HMAC_KEY_VERSIONS", json.dumps({"v1": _b64key()}))
    signed = sign_pack(_make_pack())
    monkeypatch.setenv("MRE_FI_HMAC_KEY_VERSIONS", json.dumps({"v1": _b64key()}))
    assert verify_pack(signed) is False


def test_verify_pack_rejects_malformed_signature(monkeypatch):
    monkeypatch.setenv("MRE_FI_HMAC_KEY_VERSIONS", json.dumps({"v1": _b64key()}))
    bad1 = dataclasses.replace(_make_pack(), hmac_signature="no-colon-here")
    bad2 = dataclasses.replace(_make_pack(), hmac_signature=":missing-version")
    bad3 = dataclasses.replace(_make_pack(), hmac_signature="v1:")
    bad4 = dataclasses.replace(_make_pack(), hmac_signature="vUNKNOWN:" + "a" * 64)
    for pack in (bad1, bad2, bad3, bad4):
        assert verify_pack(pack) is False


def test_production_mode_requires_hmac_or_raises(monkeypatch):
    monkeypatch.setenv("MRE_ENV", "production")
    assert require_production_hmac() is True
    with pytest.raises(RuntimeError, match="FI HMAC required"):
        sign_pack(_make_pack())


def test_dev_mode_allows_unsigned_passthrough(monkeypatch):
    assert require_production_hmac() is False
    pack = _make_pack()
    signed = sign_pack(pack)
    assert signed.hmac_signature is None
    assert verify_pack(signed) is True


def test_hmac_compare_digest_constant_time():
    """The verifier MUST use ``hmac.compare_digest``.

    Naive ``==`` comparison leaks the HMAC's matching prefix length via
    timing — a remote attacker that controls a forged-signature input
    can extract the digest one byte at a time. We assert the source
    of :func:`verify_pack` references ``compare_digest`` so a future
    refactor cannot regress.
    """
    src = inspect.getsource(verify_pack)
    assert "hmac.compare_digest" in src or "compare_digest" in src


def test_multiple_key_versions_resolve_correctly(monkeypatch):
    keys = {"v1": _b64key(), "v2": _b64key()}
    monkeypatch.setenv("MRE_FI_HMAC_KEY_VERSIONS", json.dumps(keys))
    assert latest_hmac_version() == "v2"
    parsed = get_hmac_keys()
    assert set(parsed.keys()) == {"v1", "v2"}

    signed_v2 = sign_pack(_make_pack())
    assert signed_v2.hmac_signature is not None
    assert signed_v2.hmac_signature.startswith("v2:")
    assert verify_pack(signed_v2) is True

    signed_v1 = sign_pack(_make_pack(), key_version="v1")
    assert signed_v1.hmac_signature is not None
    assert signed_v1.hmac_signature.startswith("v1:")
    assert verify_pack(signed_v1) is True


def test_hmac_version_natural_sort_orders_v10_after_v9(monkeypatch):
    keys = {"v9": _b64key(), "v10": _b64key()}
    monkeypatch.setenv("MRE_FI_HMAC_KEY_VERSIONS", json.dumps(keys))
    assert latest_hmac_version() == "v10"


def test_production_hmac_rejects_weak_key(monkeypatch):
    monkeypatch.setenv("MRE_ENV", "production")
    monkeypatch.setenv("MRE_FI_HMAC_KEY", "x")
    with pytest.raises(RuntimeError, match="at least 32 bytes"):
        get_hmac_keys()


def test_singleton_env_registers_as_v1(monkeypatch):
    raw = _b64key()
    monkeypatch.setenv("MRE_FI_HMAC_KEY", raw)
    keys = get_hmac_keys()
    assert list(keys.keys()) == ["v1"]
    signed = sign_pack(_make_pack())
    assert signed.hmac_signature is not None
    assert signed.hmac_signature.startswith("v1:")


def test_invalid_key_versions_env_raises(monkeypatch):
    monkeypatch.setenv("MRE_FI_HMAC_KEY_VERSIONS", "not-json")
    with pytest.raises(RuntimeError, match="JSON object"):
        get_hmac_keys()


def test_unknown_key_version_argument_raises(monkeypatch):
    monkeypatch.setenv("MRE_FI_HMAC_KEY_VERSIONS", json.dumps({"v1": _b64key()}))
    with pytest.raises(RuntimeError, match="not in the configured"):
        sign_pack(_make_pack(), key_version="vNOT_THERE")


def test_canonical_payload_excludes_signature():
    """Sanity: the bytestream signed must NOT include hmac_signature."""
    pack = _make_pack()
    payload = canonical_pack_payload(pack)
    assert "hmac_signature" not in payload
    pack_signed = dataclasses.replace(pack, hmac_signature="v1:abc")
    assert canonical_pack_payload(pack_signed) == payload


def test_require_hmac_flag_independent_of_mre_env(monkeypatch):
    monkeypatch.setenv("MRE_FI_REQUIRE_HMAC", "1")
    assert require_production_hmac() is True
    with pytest.raises(RuntimeError):
        sign_pack(_make_pack())


def test_module_uses_compare_digest_in_verify():
    """Defense-in-depth: ensure the module imports hmac and uses the
    constant-time digest comparator. ``hmac.compare_digest`` is the
    documented API; ``ep._hmac_hex`` produces the input."""
    assert hasattr(ep, "_hmac_hex")
    src = inspect.getsource(ep)
    assert "import hmac" in src
    assert "compare_digest" in src
