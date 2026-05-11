# SPDX-License-Identifier: Apache-2.0
"""PR-6 §G.4 — ``mre fi-tca-segment`` CLI tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import market_regime_engine.fixed_income  # noqa: F401 — register FI schema
from market_regime_engine.fixed_income import (
    ExecutionConfidenceRequest,
    score_credit_regime,
    score_execution_confidence,
    score_liquidity_stress,
    write_credit_regime_score,
    write_execution_confidence_prediction,
    write_execution_outcome,
    write_liquidity_stress_score,
)
from market_regime_engine.fixed_income.cli import run
from market_regime_engine.storage import Warehouse


def _seed_warehouse_for_materialise(db: Path, *, asof: pd.Timestamp) -> None:
    wh = Warehouse(db)
    try:
        # Seed credit regime
        rows = [
            {
                "date": asof - pd.Timedelta(days=100 - i),
                "feature_name": "cdx_ig_5y",
                "value": float(i),
                "source_timestamp": asof - pd.Timedelta(days=100 - i),
                "vintage_date": None,
            }
            for i in range(100)
        ]
        feats = pd.DataFrame(rows)
        feats.attrs["nan_policy"] = "NAN_TO_LAST_VALID"
        write_credit_regime_score(wh, score_credit_regime(feats, asof=asof, release_gate=True))

        # Seed liquidity stress
        rows = [
            {
                "date": asof - pd.Timedelta(days=100 - i),
                "feature_name": "bid_ask_width",
                "value": float(i),
                "source_timestamp": asof - pd.Timedelta(days=100 - i),
                "vintage_date": None,
            }
            for i in range(100)
        ]
        feats = pd.DataFrame(rows)
        feats.attrs["nan_policy"] = "NAN_TO_LAST_VALID"
        write_liquidity_stress_score(
            wh,
            score_liquidity_stress(
                feats,
                scope_type="cusip",
                scope_id="00206RGB6",
                asof=asof,
                release_gate=True,
            ),
        )

        # Seed one decision + outcome
        request = ExecutionConfidenceRequest(
            timestamp=(asof + pd.Timedelta(seconds=10)).isoformat(),
            cusip="00206RGB6",
            side="buy",
            notional=1_000_000.0,
            protocol="Auto-X",
            sector="industrials",
            rating="BBB+",
        )
        response = score_execution_confidence(request, warehouse=wh, release_gate=True)
        write_execution_confidence_prediction(wh, response, request_id="req-cli-1")
        write_execution_outcome(
            wh,
            request_id="req-cli-1",
            observed={
                "cusip": "00206RGB6",
                "side": "buy",
                "notional": 1_000_000.0,
                "filled_quantity": 1_000_000.0,
                "execution_price": 100.05,
                "observed_at": (asof + pd.Timedelta(minutes=30)).isoformat(),
                "outcome_observation_lag": 1800.0,
                "decision_timestamp": (asof + pd.Timedelta(seconds=10)).isoformat(),
                "arrival_price": 100.0,
                "vwap_price": 100.04,
                "mid_price_at_arrival": 100.005,
                "best_bid_at_arrival": 99.99,
                "best_ask_at_arrival": 100.02,
                "time_to_fill_seconds": 120.0,
                "dealer_response_count": 5,
                "markout_price_1d": 100.10,
                "markout_price_5d": 100.20,
            },
        )
    finally:
        wh.close()


def test_fi_tca_segment_cli_runs_on_synthetic_data(tmp_path: Path, capsys) -> None:
    db = tmp_path / "tca_cli.duckdb"
    asof = pd.Timestamp("2026-05-01T16:00:00Z")
    _seed_warehouse_for_materialise(db, asof=asof)

    rc = run(
        [
            "fi-tca-segment",
            "--db",
            str(db),
            "--date",
            "2026-05-01",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out.strip().splitlines()[-1])
    assert payload["status"] == "ok"
    assert payload["date"] == "2026-05-01"
    assert payload["rows_written"] > 0


def test_fi_tca_segment_cli_writes_expected_segment_count(
    tmp_path: Path, capsys
) -> None:
    db = tmp_path / "tca_cli.duckdb"
    asof = pd.Timestamp("2026-05-01T16:00:00Z")
    _seed_warehouse_for_materialise(db, asof=asof)
    output_json = tmp_path / "summary.json"

    rc = run(
        [
            "fi-tca-segment",
            "--db",
            str(db),
            "--date",
            "2026-05-01",
            "--output-json",
            str(output_json),
        ]
    )
    assert rc == 0
    assert output_json.exists()
    parsed = json.loads(output_json.read_text(encoding="utf-8"))
    assert parsed["rows_written"] > 0


def test_fi_tca_segment_cli_supports_soft_weighting_flag(
    tmp_path: Path, capsys
) -> None:
    db = tmp_path / "tca_cli.duckdb"
    asof = pd.Timestamp("2026-05-01T16:00:00Z")
    _seed_warehouse_for_materialise(db, asof=asof)

    rc = run(
        [
            "fi-tca-segment",
            "--db",
            str(db),
            "--date",
            "2026-05-01",
            "--soft-weighting",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out.strip().splitlines()[-1])
    assert payload["soft_weighting"] is True


def test_fi_tca_segment_cli_rejects_unknown_dimension(tmp_path: Path, capsys) -> None:
    db = tmp_path / "tca_cli.duckdb"
    asof = pd.Timestamp("2026-05-01T16:00:00Z")
    _seed_warehouse_for_materialise(db, asof=asof)

    rc = run(
        [
            "fi-tca-segment",
            "--db",
            str(db),
            "--date",
            "2026-05-01",
            "--dimensions",
            "bogus_dim",
        ]
    )
    assert rc == 2
    out = capsys.readouterr().out
    payload = json.loads(out.strip().splitlines()[-1])
    assert payload["status"] == "error"


def test_fi_tca_segment_cli_with_empty_warehouse_emits_zero_row_count(
    tmp_path: Path, capsys
) -> None:
    db = tmp_path / "tca_cli_empty.duckdb"
    wh = Warehouse(db)
    wh.close()
    rc = run(
        [
            "fi-tca-segment",
            "--db",
            str(db),
            "--date",
            "2026-05-01",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out.strip().splitlines()[-1])
    assert payload["rows_written"] == 0
