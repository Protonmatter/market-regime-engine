# SPDX-License-Identifier: Apache-2.0
"""Regression — ``fi_envelope_consistent`` actually compares hashes.

Pre-fix (REVIEW.md Tier-1 C-AUTO-1): ``_verify_fi_evidence_pack`` set
``fi_envelope_consistent = bool(recomputed.startswith("sha256:"))`` —
governance theatre that always returned True if the canonicalizer
succeeded. No authoritative hash was consulted.

Post-fix: ``write_evidence_pack`` stamps the canonical pack hash into
``pack.metadata["_envelope_hash"]`` (excluded from canonical hashing so
the stamping is idempotent and stable under HMAC re-signing).
``_verify_fi_evidence_pack`` recomputes :func:`compute_pack_hash` and
compares to the stamped value. Failure modes are explicit:
``envelope_hash_missing`` when no value was stamped at write time;
``envelope_hash_mismatch`` when the recomputed hash differs.
"""

from __future__ import annotations

import base64
import json
import secrets
from pathlib import Path

import pytest

import market_regime_engine.fixed_income  # noqa: F401  - register FI tables
from market_regime_engine.cli import _verify_fi_evidence_pack
from market_regime_engine.fixed_income.evidence_pack import (
    _ENVELOPE_HASH_METADATA_KEY,
    build_evidence_pack,
    evidence_pack_to_row,
    stored_envelope_hash,
    write_evidence_pack,
)
from market_regime_engine.storage import Warehouse


def _b64() -> str:
    return base64.b64encode(secrets.token_bytes(32)).decode("ascii")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    for name in (
        "MRE_FI_HMAC_KEY_VERSIONS",
        "MRE_FI_HMAC_KEY",
        "MRE_FI_REQUIRE_HMAC",
        "MRE_ENV",
    ):
        monkeypatch.delenv(name, raising=False)
    yield


def _persist(
    warehouse: Warehouse,
    *,
    model_run_id: str,
    request_id: str,
    sign: bool | None = None,
):
    pack = build_evidence_pack(
        model_run_id=model_run_id,
        component_name="credit_regime",
        model_version="0.1.0",
        code_sha="abcdef0",
        model_hash="sha256:m",
        input_features_hash="sha256:in",
        output_hash="sha256:out",
        release_gate=True,
        data_vintages={"trace_trades": "2026-05-08T16:00:00Z"},
        timestamp="2026-05-08T16:00:00Z",
    )
    return write_evidence_pack(warehouse, pack, request_id=request_id, sign=sign)


def test_envelope_consistent_true_when_hash_matches(tmp_path: Path) -> None:
    """Standard write → read → verify: stored envelope_hash matches
    the freshly recomputed compute_pack_hash; consistent=True with
    fi_envelope_reason=None."""
    wh = Warehouse(str(tmp_path / "envelope-match.duckdb"))
    try:
        _persist(wh, model_run_id="run-match", request_id="req-match")
        report = _verify_fi_evidence_pack(wh, "run-match")
    finally:
        wh.close()
    assert report["fi_evidence_pack_present"] is True
    assert report["fi_envelope_consistent"] is True
    assert report["fi_envelope_reason"] is None
    assert isinstance(report["fi_recomputed_hash"], str)
    assert report["fi_recomputed_hash"].startswith("sha256:")
    assert report["fi_expected_envelope_hash"] == report["fi_recomputed_hash"]


def test_envelope_consistent_false_when_hash_tampered(tmp_path: Path) -> None:
    """A tamper of any pack field (here: ``output_hash``) shifts the
    recomputed pack hash away from the stamped envelope_hash; verify
    returns consistent=False with reason ``envelope_hash_mismatch``."""
    wh = Warehouse(str(tmp_path / "envelope-mismatch.duckdb"))
    try:
        _persist(wh, model_run_id="run-tamper", request_id="req-tamper")
        # Tamper the row directly: rewrite output_hash to a different value.
        df = wh.read_evidence_packs()
        mask = df["model_run_id"] == "run-tamper"
        df.loc[mask, "output_hash"] = "sha256:tampered-after-write"
        wh.write_evidence_pack(df)
        report = _verify_fi_evidence_pack(wh, "run-tamper")
    finally:
        wh.close()
    assert report["fi_evidence_pack_present"] is True
    assert report["fi_envelope_consistent"] is False
    assert report["fi_envelope_reason"] == "envelope_hash_mismatch"
    # The recomputed and expected hashes are both surfaced so auditors
    # can see exactly what diverged.
    assert report["fi_recomputed_hash"] != report["fi_expected_envelope_hash"]


def test_envelope_consistent_false_when_authoritative_hash_missing(
    tmp_path: Path,
) -> None:
    """A pack written by a legacy code path (no envelope_hash stamped
    in metadata) fails closed with ``envelope_hash_missing`` rather
    than silently passing."""
    wh = Warehouse(str(tmp_path / "envelope-missing.duckdb"))
    try:
        pack = build_evidence_pack(
            model_run_id="run-legacy",
            component_name="credit_regime",
            model_version="0.1.0",
            code_sha="abcdef0",
            model_hash="sha256:m",
            input_features_hash="sha256:in",
            output_hash="sha256:out",
            release_gate=True,
            data_vintages={"trace_trades": "2026-05-08T16:00:00Z"},
            timestamp="2026-05-08T16:00:00Z",
        )
        # Bypass write_evidence_pack so the envelope_hash is NOT stamped
        # — simulates packs written by a legacy path or by an attacker
        # stripping the field.
        import pandas as pd

        row = evidence_pack_to_row(pack, request_id="req-legacy")
        # Replace metadata_json with one that has no envelope_hash key.
        row["metadata_json"] = "{}"
        wh.write_evidence_pack(pd.DataFrame([row]))
        report = _verify_fi_evidence_pack(wh, "run-legacy")
    finally:
        wh.close()
    assert report["fi_evidence_pack_present"] is True
    assert report["fi_envelope_consistent"] is False
    assert report["fi_envelope_reason"] == "envelope_hash_missing"


def test_envelope_reason_emitted_for_audit(tmp_path: Path) -> None:
    """All three reason paths surface ``fi_envelope_reason`` so an
    auditor can attribute consistency failures."""
    wh = Warehouse(str(tmp_path / "envelope-reasons.duckdb"))
    try:
        # Good pack -> reason is None
        _persist(wh, model_run_id="run-good", request_id="req-good")
        good = _verify_fi_evidence_pack(wh, "run-good")
        assert "fi_envelope_reason" in good
        assert good["fi_envelope_reason"] is None
        # Tampered pack -> reason is "envelope_hash_mismatch"
        _persist(wh, model_run_id="run-bad", request_id="req-bad")
        df = wh.read_evidence_packs()
        df.loc[df["model_run_id"] == "run-bad", "output_hash"] = "sha256:bad"
        wh.write_evidence_pack(df)
        bad = _verify_fi_evidence_pack(wh, "run-bad")
        assert "fi_envelope_reason" in bad
        assert bad["fi_envelope_reason"] == "envelope_hash_mismatch"
    finally:
        wh.close()


def test_envelope_hash_is_stamped_in_metadata_by_write_evidence_pack(
    tmp_path: Path,
) -> None:
    """White-box: ``write_evidence_pack`` populates
    ``metadata[_envelope_hash]`` with a sha256-prefixed string before
    persisting so the round-tripped pack carries the envelope hash."""
    wh = Warehouse(str(tmp_path / "envelope-stamp.duckdb"))
    try:
        persisted = _persist(wh, model_run_id="run-stamp", request_id="req-stamp")
    finally:
        wh.close()
    assert _ENVELOPE_HASH_METADATA_KEY in persisted.metadata
    assert persisted.metadata[_ENVELOPE_HASH_METADATA_KEY].startswith("sha256:")
    # ``stored_envelope_hash`` helper returns it.
    assert stored_envelope_hash(persisted) == persisted.metadata[_ENVELOPE_HASH_METADATA_KEY]


def test_envelope_hash_stable_under_hmac_signing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stamped envelope hash and the HMAC signature must be compatible:
    signing must NOT invalidate the envelope hash, and HMAC verification
    must continue to pass after stamping (i.e., the envelope_hash key is
    excluded from the canonical bytestream)."""
    monkeypatch.setenv("MRE_FI_HMAC_KEY_VERSIONS", json.dumps({"v1": _b64()}))
    wh = Warehouse(str(tmp_path / "envelope-signed.duckdb"))
    try:
        _persist(
            wh, model_run_id="run-signed", request_id="req-signed", sign=True
        )
        report = _verify_fi_evidence_pack(wh, "run-signed")
    finally:
        wh.close()
    assert report["fi_evidence_pack_present"] is True
    assert report["fi_envelope_consistent"] is True
    assert report["fi_envelope_reason"] is None
    assert report["fi_hmac_verified"] is True
