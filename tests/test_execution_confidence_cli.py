# SPDX-License-Identifier: Apache-2.0
"""PR-5 §K: ``mre fi-score-execution-confidence`` CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import market_regime_engine.fixed_income  # noqa: F401  registers FI schema
from market_regime_engine.fixed_income import (
    score_credit_regime,
    score_liquidity_stress,
    write_credit_regime_score,
    write_liquidity_stress_score,
)
from market_regime_engine.fixed_income.cli import run as fi_cli
from market_regime_engine.storage import Warehouse


def _seed(wh: Warehouse, ts: pd.Timestamp) -> None:
    rows = [
        {
            "date": ts - pd.Timedelta(days=i),
            "feature_name": "cdx_ig_5y",
            "value": float(i),
            "source_timestamp": ts - pd.Timedelta(days=i),
            "vintage_date": None,
        }
        for i in range(100, -1, -1)
    ]
    features = pd.DataFrame(rows)
    features.attrs["nan_policy"] = "NAN_TO_LAST_VALID"
    write_credit_regime_score(wh, score_credit_regime(features, asof=ts, release_gate=True))

    rows = [
        {
            "date": ts - pd.Timedelta(days=i),
            "feature_name": "bid_ask_width",
            "value": float(i),
            "source_timestamp": ts - pd.Timedelta(days=i),
            "vintage_date": None,
        }
        for i in range(100, -1, -1)
    ]
    features = pd.DataFrame(rows)
    features.attrs["nan_policy"] = "NAN_TO_LAST_VALID"
    write_liquidity_stress_score(
        wh,
        score_liquidity_stress(
            features,
            scope_type="cusip",
            scope_id="00206RGB6",
            asof=ts,
            release_gate=True,
        ),
    )


def _write_order(tmp_path: Path, request_id: str = "req-cli-1") -> Path:
    path = tmp_path / "order.json"
    payload = {
        "timestamp": "2026-05-01T16:00:30Z",
        "cusip": "00206RGB6",
        "side": "buy",
        "notional": 1_000_000,
        "protocol": "Auto-X",
        "urgency": "normal",
        "request_id": request_id,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_cli_runs_and_emits_envelope(tmp_path: Path, capsys) -> None:
    db = tmp_path / "cli.duckdb"
    wh = Warehouse(db)
    _seed(wh, pd.Timestamp("2026-05-01T16:00:00Z"))
    wh.close()

    order = _write_order(tmp_path)
    rc = fi_cli(["fi-score-execution-confidence", "--db", str(db), "--input", str(order)])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    envelope = json.loads(out)
    assert envelope["cusip"] == "00206RGB6"
    assert envelope["release_gate"] is True
    assert envelope["request_id"] == "req-cli-1"
    assert "confidence_score" in envelope


def test_cli_persists_prediction_row(tmp_path: Path) -> None:
    db = tmp_path / "cli_persist.duckdb"
    wh = Warehouse(db)
    _seed(wh, pd.Timestamp("2026-05-01T16:00:00Z"))
    wh.close()

    order = _write_order(tmp_path, request_id="req-persisted")
    rc = fi_cli(["fi-score-execution-confidence", "--db", str(db), "--input", str(order)])
    assert rc == 0
    wh2 = Warehouse(db)
    try:
        df = wh2.read_execution_confidence_predictions()
        assert "req-persisted" in df["request_id"].astype(str).tolist()
    finally:
        wh2.close()


def test_cli_auto_generates_request_id_when_not_supplied(tmp_path: Path, capsys) -> None:
    db = tmp_path / "cli_uuid.duckdb"
    wh = Warehouse(db)
    _seed(wh, pd.Timestamp("2026-05-01T16:00:00Z"))
    wh.close()

    # Order JSON without request_id; CLI must inject a UUID4.
    order = tmp_path / "order_no_id.json"
    payload = {
        "timestamp": "2026-05-01T16:00:30Z",
        "cusip": "00206RGB6",
        "side": "buy",
        "notional": 1_000_000,
        "protocol": "Auto-X",
        "urgency": "normal",
    }
    order.write_text(json.dumps(payload), encoding="utf-8")
    rc = fi_cli(["fi-score-execution-confidence", "--db", str(db), "--input", str(order)])
    assert rc == 0
    envelope = json.loads(capsys.readouterr().out.strip())
    assert len(envelope["request_id"]) >= 16


def test_cli_writes_output_json_when_requested(tmp_path: Path) -> None:
    db = tmp_path / "cli_out.duckdb"
    wh = Warehouse(db)
    _seed(wh, pd.Timestamp("2026-05-01T16:00:00Z"))
    wh.close()

    order = _write_order(tmp_path, request_id="req-with-out")
    out_path = tmp_path / "envelope.json"
    rc = fi_cli(
        [
            "fi-score-execution-confidence",
            "--db",
            str(db),
            "--input",
            str(order),
            "--output-json",
            str(out_path),
        ]
    )
    assert rc == 0
    assert out_path.exists()
    envelope = json.loads(out_path.read_text(encoding="utf-8"))
    assert envelope["request_id"] == "req-with-out"


def test_cli_rejects_invalid_input_with_exit_code_2(tmp_path: Path, capsys) -> None:
    db = tmp_path / "cli_bad.duckdb"
    wh = Warehouse(db)
    _seed(wh, pd.Timestamp("2026-05-01T16:00:00Z"))
    wh.close()

    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps({"timestamp": "2026-05-01T16:00:30"}),  # naive ts, missing fields
        encoding="utf-8",
    )
    rc = fi_cli(["fi-score-execution-confidence", "--db", str(db), "--input", str(bad)])
    assert rc == 2
    envelope = json.loads(capsys.readouterr().out.strip())
    assert envelope["status"] == "validation_error"


def test_cli_release_gate_false_propagates_to_envelope(tmp_path: Path, capsys) -> None:
    db = tmp_path / "cli_gate.duckdb"
    wh = Warehouse(db)
    _seed(wh, pd.Timestamp("2026-05-01T16:00:00Z"))
    wh.close()

    order = _write_order(tmp_path, request_id="req-gate-false")
    rc = fi_cli(
        [
            "fi-score-execution-confidence",
            "--db",
            str(db),
            "--input",
            str(order),
            "--release-gate",
            "false",
        ]
    )
    assert rc == 0
    envelope = json.loads(capsys.readouterr().out.strip())
    assert envelope["release_gate"] is False
    assert envelope["recommended_action"] == "Manual review required"
    assert envelope["human_review_required"] is True
