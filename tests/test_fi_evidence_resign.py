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
    verify_pack,
    write_evidence_pack,
    read_evidence_pack,
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


def test_fi_evidence_resign_updates_all_matching_packs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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


def test_fi_evidence_resign_skips_unsigned_packs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
