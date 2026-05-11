# SPDX-License-Identifier: Apache-2.0
"""PR-7 §G — ``mre verify-run`` FI extension acceptance tests.

The macro ``verify-run`` command must additionally verify the FI
evidence-pack envelope + HMAC when a pack matches the run_id.
Tampered packs flip ``approved`` to False so the operator command
exits non-zero.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import io
import json
import secrets
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd
import pytest

import market_regime_engine.fixed_income  # noqa: F401  - register FI tables
from market_regime_engine.cli import _verify_fi_evidence_pack
from market_regime_engine.fixed_income.evidence_pack import (
    build_evidence_pack,
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


def _persist_pack(
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


def test_verify_run_includes_fi_evidence_pack_verification_when_present(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "vr-fi-1.duckdb"
    wh = Warehouse(str(db_path))
    try:
        _persist_pack(wh, model_run_id="run-vr-1", request_id="req-vr-1")
        report = _verify_fi_evidence_pack(wh, "run-vr-1")
    finally:
        wh.close()
    assert report["fi_evidence_pack_present"] is True
    assert report["fi_envelope_consistent"] is True
    assert report["fi_hmac_verified"] is True
    assert report["fi_component_name"] == "credit_regime"
    assert report["fi_release_gate"] is True


def test_verify_run_passes_when_pack_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MRE_FI_HMAC_KEY_VERSIONS", json.dumps({"v1": _b64()}))
    db_path = tmp_path / "vr-fi-pass.duckdb"
    wh = Warehouse(str(db_path))
    try:
        _persist_pack(wh, model_run_id="run-vr-pass", request_id="req-pass", sign=True)
        report = _verify_fi_evidence_pack(wh, "run-vr-pass")
    finally:
        wh.close()
    assert report["fi_evidence_pack_present"] is True
    assert report["fi_hmac_verified"] is True


def test_verify_run_fails_when_pack_tampered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tampered pack must fail HMAC verification.

    We simulate tampering by overwriting the ``output_hash`` column
    after the pack has been signed: the signature was computed over
    the *original* canonical bytestream and will not validate against
    the new one.
    """
    monkeypatch.setenv("MRE_FI_HMAC_KEY_VERSIONS", json.dumps({"v1": _b64()}))
    db_path = tmp_path / "vr-fi-tamper.duckdb"
    wh = Warehouse(str(db_path))
    try:
        _persist_pack(wh, model_run_id="run-vr-tamper", request_id="req-t", sign=True)
        # Tamper: rewrite the output_hash on the persisted row.
        df = wh.read_evidence_packs()
        df.loc[df["model_run_id"] == "run-vr-tamper", "output_hash"] = (
            "sha256:tampered"
        )
        # Re-write the tampered row (composite PK ON CONFLICT replaces).
        wh.write_evidence_pack(df)
        report = _verify_fi_evidence_pack(wh, "run-vr-tamper")
    finally:
        wh.close()
    assert report["fi_evidence_pack_present"] is True
    assert report["fi_hmac_verified"] is False


def test_verify_run_skips_fi_when_no_pack(tmp_path: Path) -> None:
    db_path = tmp_path / "vr-fi-empty.duckdb"
    wh = Warehouse(str(db_path))
    try:
        report = _verify_fi_evidence_pack(wh, "run-not-here")
    finally:
        wh.close()
    assert report["fi_evidence_pack_present"] is False
    assert report["fi_envelope_consistent"] is None
    assert report["fi_hmac_verified"] is None
