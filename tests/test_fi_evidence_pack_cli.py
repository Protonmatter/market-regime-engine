# SPDX-License-Identifier: Apache-2.0
"""``mre fi-evidence-pack`` CLI acceptance tests (PR-7 §B)."""

from __future__ import annotations

import base64
import json
import secrets
from pathlib import Path

import pandas as pd
import pytest

import market_regime_engine.fixed_income  # noqa: F401  - register FI tables
from market_regime_engine.fixed_income.cli import run as fi_cli_run
from market_regime_engine.storage import Warehouse


def _b64(n: int = 32) -> str:
    return base64.b64encode(secrets.token_bytes(n)).decode("ascii")


def _seed_credit_regime_score(wh: Warehouse, *, model_run_id: str) -> None:
    """Plant one credit_regime_scores row so the pack can resolve."""
    df = pd.DataFrame(
        [
            {
                "model_run_id": model_run_id,
                "timestamp": "2026-05-08T16:00:00Z",
                "regime_score": 47.5,
                "regime_label": "Watch / Transition",
                "confidence": 0.88,
                "drivers_json": json.dumps(["spreads", "volatility"]),
                "component_scores_json": json.dumps({"spreads": 50.0, "volatility": 45.0}, sort_keys=True),
                "release_gate": 1,
                "artifact_hash": "sha256:" + "a" * 64,
                "metadata_json": "{}",
            }
        ]
    )
    wh.write_credit_regime_score(df)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        "MRE_FI_HMAC_KEY_VERSIONS",
        "MRE_FI_HMAC_KEY",
        "MRE_FI_REQUIRE_HMAC",
        "MRE_ENV",
    ):
        monkeypatch.delenv(name, raising=False)
    yield


def test_fi_evidence_pack_cli_builds_pack_for_model_run(tmp_path: Path) -> None:
    db_path = tmp_path / "ev-cli.duckdb"
    wh = Warehouse(str(db_path))
    try:
        _seed_credit_regime_score(wh, model_run_id="run-cli-1")
    finally:
        wh.close()

    out_json = tmp_path / "pack.json"
    rc = fi_cli_run(
        [
            "fi-evidence-pack",
            "--db",
            str(db_path),
            "--model-run-id",
            "run-cli-1",
            "--component",
            "credit_regime",
            "--request-id",
            "req-cli-1",
            "--sign",
            "auto",
            "--output-json",
            str(out_json),
        ]
    )
    assert rc == 0
    assert out_json.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["model_run_id"] == "run-cli-1"
    assert payload["component_name"] == "credit_regime"
    assert payload["request_id"] == "req-cli-1"
    assert payload["release_gate"] is True
    # Dev mode: no HMAC keys → pass-through unsigned.
    assert payload["hmac_signature"] is None
    assert payload["data_vintages"]["trace_trades"] == "1970-01-01T00:00:00Z"

    wh = Warehouse(str(db_path))
    try:
        df = wh.read_evidence_packs()
    finally:
        wh.close()
    assert not df.empty
    assert df.iloc[-1]["model_run_id"] == "run-cli-1"
    assert df.iloc[-1]["request_id"] == "req-cli-1"


def test_fi_evidence_pack_cli_signs_pack_in_production_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "ev-cli-prod.duckdb"
    wh = Warehouse(str(db_path))
    try:
        _seed_credit_regime_score(wh, model_run_id="run-prod-1")
    finally:
        wh.close()

    monkeypatch.setenv("MRE_ENV", "production")
    monkeypatch.setenv("MRE_FI_HMAC_KEY_VERSIONS", json.dumps({"v1": _b64()}))
    out_json = tmp_path / "pack.json"
    rc = fi_cli_run(
        [
            "fi-evidence-pack",
            "--db",
            str(db_path),
            "--model-run-id",
            "run-prod-1",
            "--component",
            "credit_regime",
            "--sign",
            "auto",
            "--output-json",
            str(out_json),
        ]
    )
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["hmac_signature"] is not None
    assert payload["hmac_signature"].startswith("v1:")


def test_fi_evidence_pack_cli_refuses_unsigned_in_production_mode_when_no_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    db_path = tmp_path / "ev-cli-prod-nokeys.duckdb"
    wh = Warehouse(str(db_path))
    try:
        _seed_credit_regime_score(wh, model_run_id="run-prod-2")
    finally:
        wh.close()

    monkeypatch.setenv("MRE_ENV", "production")
    rc = fi_cli_run(
        [
            "fi-evidence-pack",
            "--db",
            str(db_path),
            "--model-run-id",
            "run-prod-2",
            "--component",
            "credit_regime",
            "--sign",
            "auto",
        ]
    )
    assert rc == 3
    captured = capsys.readouterr().out
    payload = json.loads(captured.splitlines()[-1])
    assert payload["status"] == "error"
    assert payload["governance"] == "production_requires_hmac"


def test_fi_evidence_pack_cli_handles_unknown_run_id(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db_path = tmp_path / "ev-cli-missing.duckdb"
    Warehouse(str(db_path)).close()
    rc = fi_cli_run(
        [
            "fi-evidence-pack",
            "--db",
            str(db_path),
            "--model-run-id",
            "run-does-not-exist",
            "--component",
            "credit_regime",
        ]
    )
    assert rc == 2
    payload = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert payload["status"] == "not_found"


def test_fi_evidence_pack_cli_signs_when_sign_true(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--sign true`` with keys configured produces a signed pack."""
    db_path = tmp_path / "ev-sign-true.duckdb"
    wh = Warehouse(str(db_path))
    try:
        _seed_credit_regime_score(wh, model_run_id="run-sign-true")
    finally:
        wh.close()

    monkeypatch.setenv("MRE_FI_HMAC_KEY_VERSIONS", json.dumps({"v1": _b64()}))
    out_json = tmp_path / "pack.json"
    rc = fi_cli_run(
        [
            "fi-evidence-pack",
            "--db",
            str(db_path),
            "--model-run-id",
            "run-sign-true",
            "--component",
            "credit_regime",
            "--sign",
            "true",
            "--output-json",
            str(out_json),
        ]
    )
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["hmac_signature"] is not None
