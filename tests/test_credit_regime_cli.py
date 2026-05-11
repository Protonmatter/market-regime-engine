# SPDX-License-Identifier: Apache-2.0
"""``mre fi-score-credit-regime`` CLI tests (PR-3 task G)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import market_regime_engine.fixed_income  # noqa: F401 - register FI schema
from market_regime_engine.fixed_income.calendars import is_trading_day
from market_regime_engine.fixed_income.cli import run as fi_cli_run
from market_regime_engine.storage import Warehouse

_ASOF_STR = "2026-05-08T16:00:00Z"


def _seed_warehouse(wh: Warehouse) -> None:
    """Plant curve + CDS + vintage data spanning the lookback window."""
    # 30 SIFMA trading days ending Friday 2026-05-08. We over-sample
    # 60 calendar days and then filter against the SIFMA calendar so
    # the seed never lands on Good Friday / Memorial Day / etc.
    raw = pd.date_range(end=pd.Timestamp(_ASOF_STR), periods=60, freq="D", tz="UTC")
    dates = [d for d in raw if is_trading_day(d)][-30:]

    curve_rows: list[dict] = []
    cds_rows: list[dict] = []
    vintage_rows: list[dict] = []

    for i, d in enumerate(dates):
        ts = d.isoformat().replace("+00:00", "Z")
        # Treasury curve 2Y / 5Y / 10Y — drift slightly to produce realistic features.
        curve_rows.append(
            {
                "timestamp": ts,
                "curve_type": "ust",
                "tenor": "2Y",
                "rate": 4.50 + 0.005 * i,
                "source": "fed",
                "metadata_json": "{}",
            }
        )
        curve_rows.append(
            {
                "timestamp": ts,
                "curve_type": "ust",
                "tenor": "5Y",
                "rate": 4.20 + 0.003 * i,
                "source": "fed",
                "metadata_json": "{}",
            }
        )
        curve_rows.append(
            {
                "timestamp": ts,
                "curve_type": "ust",
                "tenor": "10Y",
                "rate": 4.10 + 0.002 * i,
                "source": "fed",
                "metadata_json": "{}",
            }
        )
        # CDX IG / HY 5Y.
        cds_rows.append(
            {
                "timestamp": ts,
                "reference_entity": "CDX.IG",
                "tenor": "5Y",
                "spread_bps": 60.0 + 0.4 * i,
                "source": "markit",
                "metadata_json": "{}",
            }
        )
        cds_rows.append(
            {
                "timestamp": ts,
                "reference_entity": "CDX.HY",
                "tenor": "5Y",
                "spread_bps": 320.0 + 2.0 * i,
                "source": "markit",
                "metadata_json": "{}",
            }
        )
        # VIX / MOVE / ETF prem/disc via vintage observations.
        for series_id, value in (
            ("VIX", 18.0 + 0.1 * i),
            ("MOVE", 95.0 + 0.5 * i),
            ("ETF_PREM_DISC", 0.10 + 0.001 * i),
        ):
            vintage_rows.append(
                {
                    "series_id": series_id,
                    "observation_date": ts,
                    "value": value,
                    "realtime_start": ts,
                    "realtime_end": None,
                    "vintage_date": ts,
                    "source": "synthetic",
                    "ingested_at_utc": ts,
                    "metadata_json": "{}",
                }
            )

    wh.write_curve_snapshots(pd.DataFrame(curve_rows))
    wh.write_cds_curve_snapshots(pd.DataFrame(cds_rows))
    wh.write_vintage_observations(pd.DataFrame(vintage_rows))


def test_fi_score_credit_regime_cli_runs_on_synthetic_data(tmp_path: Path) -> None:
    """In-process CLI invocation writes a credit_regime_scores row."""
    db_path = tmp_path / "fi-cli.duckdb"
    wh = Warehouse(str(db_path))
    try:
        _seed_warehouse(wh)
    finally:
        wh.close()

    out_json = tmp_path / "regime.json"
    rc = fi_cli_run(
        [
            "fi-score-credit-regime",
            "--db",
            str(db_path),
            "--asof",
            _ASOF_STR,
            "--profile",
            "production",
            "--release-gate",
            "true",
            "--output-json",
            str(out_json),
        ]
    )
    assert rc == 0
    assert out_json.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert "regime_score" in payload
    assert "model_run_id" in payload
    assert "release_gate" in payload
    assert "artifact_hash" in payload
    assert payload["timestamp"].endswith("Z")

    wh = Warehouse(str(db_path))
    try:
        df = wh.read_credit_regime_scores()
        assert not df.empty
        assert df.iloc[-1]["model_run_id"] == payload["model_run_id"]
    finally:
        wh.close()


def test_fi_score_credit_regime_cli_handles_release_gate_false(tmp_path: Path) -> None:
    """``--release-gate false`` flips the persisted row's gate and caps confidence."""
    db_path = tmp_path / "fi-cli-rg.duckdb"
    wh = Warehouse(str(db_path))
    try:
        _seed_warehouse(wh)
    finally:
        wh.close()

    rc = fi_cli_run(
        [
            "fi-score-credit-regime",
            "--db",
            str(db_path),
            "--asof",
            _ASOF_STR,
            "--release-gate",
            "false",
            "--profile",
            "production",
            "--model-run-id",
            "cli-rg-false",
        ]
    )
    assert rc == 0

    wh = Warehouse(str(db_path))
    try:
        df = wh.read_credit_regime_scores()
        row = df.iloc[-1]
        assert int(row["release_gate"]) == 0
        assert float(row["confidence"]) <= 0.5 + 1e-9
        assert row["model_run_id"] == "cli-rg-false"
    finally:
        wh.close()


@pytest.mark.slow
def test_fi_score_credit_regime_cli_subprocess_smoke(tmp_path: Path) -> None:
    """Out-of-process smoke through the ``mre`` entry point — slow path."""
    db_path = tmp_path / "fi-cli-sub.duckdb"
    wh = Warehouse(str(db_path))
    try:
        _seed_warehouse(wh)
    finally:
        wh.close()
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "market_regime_engine.cli_dispatch",
            "fi-score-credit-regime",
            "--db",
            str(db_path),
            "--asof",
            _ASOF_STR,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stderr={proc.stderr}"
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert "regime_score" in payload
    assert payload["artifact_hash"].startswith("sha256:")
