# SPDX-License-Identifier: Apache-2.0
"""``mre fi-report`` + ``generate_fi_report`` acceptance tests (PR-7 §C)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import market_regime_engine.fixed_income  # noqa: F401  - register FI tables
from market_regime_engine.fixed_income.cli import run as fi_cli_run
from market_regime_engine.fixed_income.evidence_pack import (
    build_evidence_pack,
    write_evidence_pack,
)
from market_regime_engine.fixed_income.report import generate_fi_report
from market_regime_engine.storage import Warehouse


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


def _seed_full_warehouse(wh: Warehouse) -> None:
    wh.write_credit_regime_score(
        pd.DataFrame(
            [
                {
                    "model_run_id": "run-r1",
                    "timestamp": "2026-05-08T16:00:00Z",
                    "regime_score": 47.0,
                    "regime_label": "Watch / Transition",
                    "confidence": 0.85,
                    "drivers_json": json.dumps(["spreads", "vol"]),
                    "component_scores_json": json.dumps({"spreads": 50.0}),
                    "release_gate": 1,
                    "artifact_hash": "sha256:" + "a" * 64,
                    "metadata_json": "{}",
                }
            ]
        )
    )
    wh.write_liquidity_stress_score(
        pd.DataFrame(
            [
                {
                    "model_run_id": "run-l1",
                    "scope_type": "market",
                    "scope_id": "ALL",
                    "timestamp": "2026-05-08T16:00:00Z",
                    "liquidity_score": 30.0,
                    "liquidity_label": "Mild Stress",
                    "confidence": 0.9,
                    "drivers_json": json.dumps(["bid_ask"]),
                    "release_gate": 1,
                    "artifact_hash": "sha256:" + "b" * 64,
                    "metadata_json": "{}",
                }
            ]
        )
    )
    wh.write_execution_confidence_prediction(
        pd.DataFrame(
            [
                {
                    "request_id": "req-1",
                    "timestamp": "2026-05-08T16:30:00Z",
                    "model_run_id": "run-e1",
                    "cusip": "AAA111111",
                    "side": "buy",
                    "notional": 1_000_000.0,
                    "protocol": "Auto-X",
                    "confidence_score": 0.85,
                    "expected_slippage_bps": 5.0,
                    "confidence_interval_low": 0.75,
                    "confidence_interval_high": 0.95,
                    "recommended_action": "Auto-X allowed",
                    "human_review_required": 0,
                    "release_gate": 1,
                    "artifact_hash": "sha256:" + "c" * 64,
                    "metadata_json": "{}",
                }
            ]
        )
    )
    pack = build_evidence_pack(
        model_run_id="run-r1",
        component_name="credit_regime",
        model_version="0.1.0",
        code_sha=None,
        model_hash="sha256:m",
        input_features_hash="sha256:in",
        output_hash="sha256:out",
        release_gate=True,
        data_vintages={"trace_trades": "2026-05-08T16:00:00Z"},
        timestamp="2026-05-08T16:00:00Z",
    )
    write_evidence_pack(wh, pack, request_id="req-1")


def test_generate_fi_report_includes_all_required_sections(tmp_path: Path) -> None:
    db_path = tmp_path / "rep-full.duckdb"
    wh = Warehouse(str(db_path))
    try:
        _seed_full_warehouse(wh)
        body = generate_fi_report(wh, asof=pd.Timestamp("2026-05-08T17:00:00Z"))
    finally:
        wh.close()
    assert "# Fixed-Income RCIE Report" in body
    assert "## Credit Regime Index" in body
    assert "## Liquidity Stress Index" in body
    assert "## Execution Confidence" in body
    assert "## TCA By Regime" in body
    assert "## Release Gate Status" in body
    assert "## Evidence Packs" in body
    # Specific values from the seed.
    assert "47.00" in body
    assert "Mild Stress" in body
    assert "Auto-X allowed" in body
    assert "credit_regime" in body


def test_generate_fi_report_handles_missing_data_gracefully(tmp_path: Path) -> None:
    db_path = tmp_path / "rep-empty.duckdb"
    wh = Warehouse(str(db_path))
    try:
        body = generate_fi_report(wh, asof=pd.Timestamp("2026-05-08T17:00:00Z"))
    finally:
        wh.close()
    assert "no data" in body.lower() or "no credit regime" in body.lower()
    # Every section heading is still present.
    assert "## Credit Regime Index" in body
    assert "## Liquidity Stress Index" in body
    assert "## Execution Confidence" in body


def test_generate_fi_report_markdown_format(tmp_path: Path) -> None:
    db_path = tmp_path / "rep-md.duckdb"
    wh = Warehouse(str(db_path))
    try:
        body = generate_fi_report(
            wh,
            asof=pd.Timestamp("2026-05-08T17:00:00Z"),
            output_format="markdown",
        )
    finally:
        wh.close()
    assert body.startswith("# Fixed-Income RCIE Report")
    assert not body.startswith("<!DOCTYPE html>")


def test_generate_fi_report_html_format(tmp_path: Path) -> None:
    db_path = tmp_path / "rep-html.duckdb"
    wh = Warehouse(str(db_path))
    try:
        _seed_full_warehouse(wh)
        body = generate_fi_report(
            wh,
            asof=pd.Timestamp("2026-05-08T17:00:00Z"),
            output_format="html",
        )
    finally:
        wh.close()
    assert body.startswith("<!DOCTYPE html>") or body.startswith("<html")
    assert "Fixed-Income RCIE Report" in body


def test_fi_report_cli_writes_to_specified_path(tmp_path: Path) -> None:
    db_path = tmp_path / "rep-cli.duckdb"
    wh = Warehouse(str(db_path))
    try:
        _seed_full_warehouse(wh)
    finally:
        wh.close()

    out_path = tmp_path / "report.md"
    rc = fi_cli_run(
        [
            "fi-report",
            "--db",
            str(db_path),
            "--out",
            str(out_path),
            "--format",
            "markdown",
            "--asof",
            "2026-05-08T17:00:00Z",
        ]
    )
    assert rc == 0
    assert out_path.exists()
    body = out_path.read_text(encoding="utf-8")
    assert body.startswith("# Fixed-Income RCIE Report")
    assert "Credit Regime Index" in body


def test_fi_report_cli_html_format(tmp_path: Path) -> None:
    db_path = tmp_path / "rep-cli-html.duckdb"
    wh = Warehouse(str(db_path))
    try:
        _seed_full_warehouse(wh)
    finally:
        wh.close()
    out_path = tmp_path / "report.html"
    rc = fi_cli_run(
        [
            "fi-report",
            "--db",
            str(db_path),
            "--out",
            str(out_path),
            "--format",
            "html",
        ]
    )
    assert rc == 0
    assert out_path.exists()
    body = out_path.read_text(encoding="utf-8")
    assert body.startswith("<!DOCTYPE html>")
