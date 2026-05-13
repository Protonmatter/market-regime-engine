# SPDX-License-Identifier: Apache-2.0
"""``mre fi-evidence-resign`` acceptance tests (PR-7 §F).

Per AGENT.md PR-7 + REVIEW.md §4.2 HMAC rotation: a quarterly rotation
playbook needs a tool that bulk re-signs every existing pack from the
old key version to the new one. The tool must:

- Process every pack signed under ``--from-key`` and re-sign it under
  ``--to-key``.
- Skip packs that are unsigned or signed under unrelated key versions
  so a half-rotation is reversible.
- Refuse to run when ``--to-key`` isn't in
  ``MRE_FI_HMAC_KEY_VERSIONS``.
- Support ``--dry-run`` for the on-call playbook smoke test.
"""

from __future__ import annotations

import base64
import json
import secrets
from pathlib import Path

import pandas as pd
import pytest

import market_regime_engine.fixed_income  # noqa: F401  - register FI tables
from market_regime_engine.fixed_income.cli import run as fi_cli_run
from market_regime_engine.fixed_income.evidence_pack import (
    build_evidence_pack,
    read_evidence_pack,
    verify_pack,
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


def _persist_signed_pack(
    warehouse: Warehouse,
    *,
    model_run_id: str,
    request_id: str,
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
    return write_evidence_pack(warehouse, pack, request_id=request_id, sign=True)


def test_fi_evidence_resign_updates_all_matching_packs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "resign.duckdb"
    keys_v1_only = json.dumps({"v1": _b64()})
    monkeypatch.setenv("MRE_FI_HMAC_KEY_VERSIONS", keys_v1_only)

    wh = Warehouse(str(db_path))
    try:
        _persist_signed_pack(wh, model_run_id="run-1", request_id="req-1")
        _persist_signed_pack(wh, model_run_id="run-2", request_id="req-2")
    finally:
        wh.close()

    keys_dict = json.loads(keys_v1_only)
    keys_dict["v2"] = _b64()
    monkeypatch.setenv("MRE_FI_HMAC_KEY_VERSIONS", json.dumps(keys_dict))

    rc = fi_cli_run(
        [
            "fi-evidence-resign",
            "--db",
            str(db_path),
            "--from-key",
            "v1",
            "--to-key",
            "v2",
        ]
    )
    assert rc == 0

    wh2 = Warehouse(str(db_path))
    try:
        df = wh2.read_evidence_packs()
        assert (df["hmac_signature"].astype(str).str.startswith("v2:")).all()
        for run_id in ("run-1", "run-2"):
            pack = read_evidence_pack(wh2, model_run_id=run_id)
            assert pack is not None
            assert verify_pack(pack) is True
    finally:
        wh2.close()


def test_fi_evidence_resign_dry_run_no_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    db_path = tmp_path / "resign-dry.duckdb"
    # Sign first under v1 only so the persisted pack ends up at v1 (max
    # version is lexicographic — having v2 in the env at sign time would
    # otherwise emit a v2 signature).
    monkeypatch.setenv("MRE_FI_HMAC_KEY_VERSIONS", json.dumps({"v1": _b64()}))

    wh = Warehouse(str(db_path))
    try:
        _persist_signed_pack(wh, model_run_id="run-1", request_id="req-1")
    finally:
        wh.close()

    keys_with_v2 = json.dumps({"v1": _b64(), "v2": _b64()})
    monkeypatch.setenv("MRE_FI_HMAC_KEY_VERSIONS", keys_with_v2)
    rc = fi_cli_run(
        [
            "fi-evidence-resign",
            "--db",
            str(db_path),
            "--from-key",
            "v1",
            "--to-key",
            "v2",
            "--dry-run",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr().out
    payload = json.loads(captured.splitlines()[-1])
    assert payload["dry_run"] is True
    assert payload["resigned"] == 0
    assert payload["matched"] >= 1

    wh2 = Warehouse(str(db_path))
    try:
        df = wh2.read_evidence_packs()
        assert df["hmac_signature"].astype(str).str.startswith("v1:").all()
    finally:
        wh2.close()


def test_fi_evidence_resign_skips_unsigned_packs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "resign-unsigned.duckdb"
    monkeypatch.delenv("MRE_FI_HMAC_KEY_VERSIONS", raising=False)
    wh = Warehouse(str(db_path))
    try:
        # Persist unsigned pack (no keys configured).
        pack = build_evidence_pack(
            model_run_id="run-unsigned",
            component_name="credit_regime",
            model_version="0.1.0",
            code_sha=None,
            model_hash="sha256:m",
            input_features_hash="sha256:in",
            output_hash="sha256:out",
            release_gate=True,
        )
        write_evidence_pack(wh, pack, request_id="req-unsigned")
    finally:
        wh.close()

    keys = json.dumps({"v1": _b64(), "v2": _b64()})
    monkeypatch.setenv("MRE_FI_HMAC_KEY_VERSIONS", keys)
    rc = fi_cli_run(
        [
            "fi-evidence-resign",
            "--db",
            str(db_path),
            "--from-key",
            "v1",
            "--to-key",
            "v2",
        ]
    )
    assert rc == 0

    wh2 = Warehouse(str(db_path))
    try:
        df = wh2.read_evidence_packs()
        # Unsigned pack untouched.
        sig = df.iloc[-1]["hmac_signature"]
        assert sig is None or pd.isna(sig) or sig == ""
    finally:
        wh2.close()


def test_fi_evidence_resign_raises_when_key_not_in_versions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = tmp_path / "resign-bad.duckdb"
    keys = json.dumps({"v1": _b64()})
    monkeypatch.setenv("MRE_FI_HMAC_KEY_VERSIONS", keys)
    Warehouse(str(db_path)).close()
    rc = fi_cli_run(
        [
            "fi-evidence-resign",
            "--db",
            str(db_path),
            "--from-key",
            "v1",
            "--to-key",
            "vNEW",
        ]
    )
    assert rc == 2
    payload = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert payload["status"] == "error"
    assert "vNEW" in payload["detail"]



def _persist_signed_v1_pack(
    warehouse: Warehouse,
    *,
    model_run_id: str,
    request_id: str,
):
    """Write a pack signed under v1 canonical-JSON encoding (no
    ``_canonical_version`` metadata stamp) so the resign migration
    has a legacy pack to operate on."""
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
        canonical_version="v1",
    )
    return write_evidence_pack(warehouse, pack, request_id=request_id, sign=True)


def test_fi_evidence_resign_to_version_v2_upgrades_legacy_packs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v1.6.0 (REVIEW_DEEP_V1_5_2.md section 2.5): ``--to-version v2``
    upgrades v1.5.x legacy packs to the RFC 8785 canonical-JSON
    encoder. After the resign:

    1. ``pack.metadata['_canonical_version']`` is stamped to ``"v2"``.
    2. The HMAC signature verifies under the new key + new encoder.
    3. The pack hash has changed (since the canonical bytes changed)
       -- this is the documented migration trade-off; auditors who
       need the historical v1 hash must keep the pre-migration row.
    """
    from market_regime_engine.fixed_income.evidence_pack import (
        _CANONICAL_VERSION_METADATA_KEY,
    )

    db_path = tmp_path / "resign-to-v2.duckdb"
    # Stage 1: only v1 HMAC key is configured so the legacy pack is
    # signed under v1: (the simulated v1.5.x deployment shape).
    v1_key = _b64()
    v2_key = _b64()
    monkeypatch.setenv("MRE_FI_HMAC_KEY_VERSIONS", json.dumps({"v1": v1_key}))

    wh = Warehouse(str(db_path))
    try:
        _persist_signed_v1_pack(wh, model_run_id="legacy-1", request_id="req-l1")
        # Sanity: the persisted pack has no _canonical_version stamp.
        pack_before = read_evidence_pack(wh, model_run_id="legacy-1")
        assert pack_before is not None
        assert _CANONICAL_VERSION_METADATA_KEY not in (pack_before.metadata or {})
        assert pack_before.hmac_signature is not None
        assert pack_before.hmac_signature.startswith("v1:")
        assert verify_pack(pack_before)
    finally:
        wh.close()

    # Stage 2: add the v2 key alongside v1 so the resign tool can pick
    # the new key version.
    monkeypatch.setenv(
        "MRE_FI_HMAC_KEY_VERSIONS", json.dumps({"v1": v1_key, "v2": v2_key})
    )

    rc = fi_cli_run(
        [
            "fi-evidence-resign",
            "--db",
            str(db_path),
            "--from-key",
            "v1",
            "--to-key",
            "v2",
            "--to-version",
            "v2",
        ]
    )
    assert rc == 0

    wh2 = Warehouse(str(db_path))
    try:
        pack_after = read_evidence_pack(wh2, model_run_id="legacy-1")
        assert pack_after is not None
        assert pack_after.metadata.get(_CANONICAL_VERSION_METADATA_KEY) == "v2"
        assert pack_after.hmac_signature is not None
        assert pack_after.hmac_signature.startswith("v2:")
        assert verify_pack(pack_after) is True
    finally:
        wh2.close()


def test_fi_evidence_resign_default_preserves_canonical_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``--to-version`` the resign tool only rotates the HMAC
    key; the canonical-JSON encoder version (and therefore the pack
    hash, ignoring the signature) is preserved."""
    from market_regime_engine.fixed_income.evidence_pack import (
        _CANONICAL_VERSION_METADATA_KEY,
    )

    db_path = tmp_path / "resign-preserve.duckdb"
    v1_key = _b64()
    v2_key = _b64()
    monkeypatch.setenv("MRE_FI_HMAC_KEY_VERSIONS", json.dumps({"v1": v1_key}))

    wh = Warehouse(str(db_path))
    try:
        _persist_signed_v1_pack(wh, model_run_id="legacy-2", request_id="req-l2")
    finally:
        wh.close()

    monkeypatch.setenv(
        "MRE_FI_HMAC_KEY_VERSIONS", json.dumps({"v1": v1_key, "v2": v2_key})
    )

    rc = fi_cli_run(
        [
            "fi-evidence-resign",
            "--db",
            str(db_path),
            "--from-key",
            "v1",
            "--to-key",
            "v2",
        ]
    )
    assert rc == 0

    wh2 = Warehouse(str(db_path))
    try:
        pack_after = read_evidence_pack(wh2, model_run_id="legacy-2")
        assert pack_after is not None
        # No --to-version supplied: pack stays under v1 canonical encoder.
        assert _CANONICAL_VERSION_METADATA_KEY not in (pack_after.metadata or {})
        assert pack_after.hmac_signature is not None
        assert pack_after.hmac_signature.startswith("v2:")
        # The new (v2-key-signed) v1-canonical pack still verifies.
        assert verify_pack(pack_after) is True
    finally:
        wh2.close()
