# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from market_regime_engine.fixed_income.cli import run as fi_cli
from market_regime_engine.storage import Warehouse
from tests.test_protocol_recommendation import _seed


def _write_order(tmp_path: Path) -> Path:
    path = tmp_path / "order.json"
    path.write_text(
        json.dumps(
            {
                "timestamp": "2026-05-01T16:00:30Z",
                "cusip": "00206RGB6",
                "side": "buy",
                "notional": 1_000_000,
                "protocol": "Auto-X",
                "urgency": "normal",
                "request_id": "req-cli-xpro",
                "candidate_protocols": ["Auto-X", "RFQ", "Manual"],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_xpro_decision_cli_emits_and_persists_artifact(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.setenv("MRE_FI_HMAC_KEY_VERSIONS", '{"v1":"secret"}')
    db = tmp_path / "cli_xpro.duckdb"
    wh = Warehouse(db)
    _seed(wh, pd.Timestamp("2026-05-01T16:00:00Z"))
    wh.close()

    out_path = tmp_path / "decision.json"
    rc = fi_cli(
        [
            "fi-recommend-execution-protocol",
            "--db",
            str(db),
            "--input",
            str(_write_order(tmp_path)),
            "--output-json",
            str(out_path),
        ]
    )
    assert rc == 0
    body = json.loads(capsys.readouterr().out)
    assert body["artifact_version"] == "xpro_decision_artifact_v1"
    assert out_path.exists()
    wh2 = Warehouse(db)
    try:
        assert wh2.latest_xpro_decision_artifact(body["decision_id"]) is not None
    finally:
        wh2.close()


def test_xpro_decision_cli_verify_command(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.setenv("MRE_FI_HMAC_KEY_VERSIONS", '{"v1":"secret"}')
    db = tmp_path / "cli_verify.duckdb"
    wh = Warehouse(db)
    _seed(wh, pd.Timestamp("2026-05-01T16:00:00Z"))
    wh.close()
    fi_cli(["fi-recommend-execution-protocol", "--db", str(db), "--input", str(_write_order(tmp_path))])
    artifact = json.loads(capsys.readouterr().out)
    rc = fi_cli(["fi-verify-xpro-decision", "--db", str(db), "--decision-id", artifact["decision_id"]])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["verified"] is True


def test_xpro_decision_cli_verify_can_require_hmac(tmp_path: Path, capsys, monkeypatch) -> None:
    monkeypatch.delenv("MRE_FI_HMAC_KEY_VERSIONS", raising=False)
    monkeypatch.delenv("MRE_FI_HMAC_KEY", raising=False)
    monkeypatch.delenv("MRE_FI_REQUIRE_HMAC", raising=False)
    monkeypatch.delenv("MRE_ENV", raising=False)
    db = tmp_path / "cli_verify_require_hmac.duckdb"
    wh = Warehouse(db)
    _seed(wh, pd.Timestamp("2026-05-01T16:00:00Z"))
    wh.close()
    fi_cli(["fi-recommend-execution-protocol", "--db", str(db), "--input", str(_write_order(tmp_path))])
    artifact = json.loads(capsys.readouterr().out)

    rc = fi_cli(["fi-verify-xpro-decision", "--db", str(db), "--decision-id", artifact["decision_id"]])
    assert rc == 0
    relaxed = json.loads(capsys.readouterr().out)
    assert relaxed["verified"] is True
    assert relaxed["hmac_required"] is False
    assert relaxed["hmac_valid"] is None

    rc = fi_cli(
        [
            "fi-verify-xpro-decision",
            "--db",
            str(db),
            "--decision-id",
            artifact["decision_id"],
            "--require-hmac",
        ]
    )
    assert rc == 2
    strict = json.loads(capsys.readouterr().out)
    assert strict["verified"] is False
    assert strict["hmac_required"] is True
