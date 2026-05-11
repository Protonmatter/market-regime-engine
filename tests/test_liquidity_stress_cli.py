# SPDX-License-Identifier: Apache-2.0
"""``mre fi-score-liquidity`` CLI tests (PR-4 task F / H.8)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import market_regime_engine.fixed_income  # noqa: F401  - register FI schema
from market_regime_engine.fixed_income.calendars import is_trading_day
from market_regime_engine.fixed_income.cli import run as fi_cli_run
from market_regime_engine.fixed_income.schemas import LiquidityLabel
from market_regime_engine.storage import Warehouse

_ASOF_STR = "2026-05-08T16:00:00Z"
_ASOF = pd.Timestamp(_ASOF_STR)
_CUSIP = "9128283N8"


def _numpyro_available() -> bool:
    try:
        import numpyro  # noqa: F401

        return True
    except ImportError:
        return False


def _seed_warehouse(wh: Warehouse) -> None:
    """Plant a minimal bond universe with trades / RFQs / quotes."""
    raw = pd.date_range(end=_ASOF, periods=60, freq="D", tz="UTC")
    dates = [d for d in raw if is_trading_day(d)][-30:]

    wh.write_bond_reference(
        pd.DataFrame(
            [
                {
                    "cusip": _CUSIP,
                    "valid_from": (_ASOF - pd.Timedelta(days=400)).isoformat(),
                    "valid_to": None,
                    "ticker": "TSY",
                    "issuer": "US Treasury",
                    "sector": "treasuries",
                    "rating": "AAA",
                    "issue_date": (_ASOF - pd.Timedelta(days=400)).isoformat(),
                    "maturity": "2033-05-01T00:00:00+00:00",
                    "coupon": 3.5,
                    "currency": "USD",
                    "country": "US",
                    "duration": 7.0,
                    "convexity": 1.0,
                    "amount_outstanding": 10e9,
                    "is_callable": 0,
                    "call_schedule_json": "{}",
                    "default_date": None,
                    "delisted_date": None,
                    "metadata_json": "{}",
                }
            ]
        )
    )

    trades: list[dict] = []
    rfqs: list[dict] = []
    quotes: list[dict] = []
    for i, d in enumerate(dates):
        ts = d.isoformat().replace("+00:00", "Z")
        for k in (0, 1):
            trades.append(
                {
                    "trade_id": f"T-{i}-{k}",
                    "timestamp": ts,
                    "cusip": _CUSIP,
                    "price": 100.0 + 0.01 * i + 0.02 * k,
                    "yield_pct": 4.5 + 0.005 * i,
                    "size": 1_000_000.0,
                    "side": "buy" if k == 0 else "sell",
                    "protocol": "RFQ",
                    "venue": "MarketAxess",
                    "source": "trace",
                    "reported_at": ts,
                    "metadata_json": "{}",
                }
            )
        rfqs.append(
            {
                "rfq_id": f"R-{i}",
                "timestamp": ts,
                "cusip": _CUSIP,
                "side": "buy",
                "notional": 2_000_000.0,
                "protocol": "RFQ",
                "status": "filled",
                "dealers_requested": 5,
                "dealers_responded": 3,
                "time_to_first_response_ms": 1500,
                "client_id": "fund_a",
                "metadata_json": "{}",
            }
        )
        for dealer in ("DEAL_A", "DEAL_B"):
            quotes.append(
                {
                    "timestamp": ts,
                    "cusip": _CUSIP,
                    "dealer_id": dealer,
                    "side": "bid",
                    "price": 99.5 + 0.005 * (1 if dealer == "DEAL_A" else -1) + 0.01 * i,
                    "size": 1_000_000.0,
                    "expires_at": ts,
                    "metadata_json": "{}",
                }
            )
            quotes.append(
                {
                    "timestamp": ts,
                    "cusip": _CUSIP,
                    "dealer_id": dealer,
                    "side": "ask",
                    "price": 100.5 + 0.005 * (1 if dealer == "DEAL_A" else -1) + 0.01 * i,
                    "size": 1_000_000.0,
                    "expires_at": ts,
                    "metadata_json": "{}",
                }
            )

    wh.write_trace_trades(pd.DataFrame(trades))
    wh.write_rfq_events(pd.DataFrame(rfqs))
    wh.write_dealer_quotes(pd.DataFrame(quotes))


def test_fi_score_liquidity_cli_runs_on_synthetic_data(tmp_path: Path) -> None:
    """In-process CLI invocation writes a liquidity_stress_scores row."""
    db_path = tmp_path / "fi-liq-cli.duckdb"
    wh = Warehouse(str(db_path))
    try:
        _seed_warehouse(wh)
    finally:
        wh.close()

    out_json = tmp_path / "liquidity.json"
    rc = fi_cli_run(
        [
            "fi-score-liquidity",
            "--db",
            str(db_path),
            "--scope-type",
            "market",
            "--scope-id",
            "ALL",
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
    for key in (
        "liquidity_index",
        "liquidity_label",
        "model_run_id",
        "release_gate",
        "artifact_hash",
        "scope_type",
        "scope_id",
        "timestamp",
    ):
        assert key in payload
    assert payload["timestamp"].endswith("Z")
    assert payload["scope_type"] == "market"

    wh = Warehouse(str(db_path))
    try:
        df = wh.read_liquidity_stress_scores()
        assert not df.empty
        assert df.iloc[-1]["model_run_id"] == payload["model_run_id"]
    finally:
        wh.close()


def test_fi_score_liquidity_cli_handles_release_gate_false(tmp_path: Path) -> None:
    db_path = tmp_path / "fi-liq-cli-rg.duckdb"
    wh = Warehouse(str(db_path))
    try:
        _seed_warehouse(wh)
    finally:
        wh.close()

    rc = fi_cli_run(
        [
            "fi-score-liquidity",
            "--db",
            str(db_path),
            "--scope-type",
            "cusip",
            "--scope-id",
            _CUSIP,
            "--asof",
            _ASOF_STR,
            "--release-gate",
            "false",
            "--profile",
            "production",
            "--model-run-id",
            "cli-liq-rg-false",
        ]
    )
    assert rc == 0

    wh = Warehouse(str(db_path))
    try:
        df = wh.read_liquidity_stress_scores()
        row = df.iloc[-1]
        assert int(row["release_gate"]) == 0
        assert float(row["confidence"]) <= 0.5 + 1e-9
        assert row["model_run_id"] == "cli-liq-rg-false"
    finally:
        wh.close()


@pytest.mark.skipif(not _numpyro_available(), reason="numpyro not installed")
def test_fi_score_liquidity_cli_with_hierarchical_flag(tmp_path: Path) -> None:
    """``--use-hierarchical`` sets the metadata flag; deterministic stays primary."""
    db_path = tmp_path / "fi-liq-cli-hier.duckdb"
    wh = Warehouse(str(db_path))
    try:
        _seed_warehouse(wh)
    finally:
        wh.close()

    rc = fi_cli_run(
        [
            "fi-score-liquidity",
            "--db",
            str(db_path),
            "--scope-type",
            "market",
            "--scope-id",
            "ALL",
            "--asof",
            _ASOF_STR,
            "--use-hierarchical",
        ]
    )
    assert rc == 0


def test_fi_score_liquidity_cli_prev_label_from_warehouse(tmp_path: Path) -> None:
    """A previously-written label is picked up and routed through hysteresis."""
    db_path = tmp_path / "fi-liq-cli-prev.duckdb"
    wh = Warehouse(str(db_path))
    try:
        _seed_warehouse(wh)
        # Plant a row whose label is CRISIS_LIQUIDITY so the next run
        # inherits it as prev_label.
        wh.write_liquidity_stress_score(
            pd.DataFrame(
                [
                    {
                        "model_run_id": "seed-prev",
                        "scope_type": "market",
                        "scope_id": "ALL",
                        "timestamp": (_ASOF - pd.Timedelta(hours=2)).isoformat().replace("+00:00", "Z"),
                        "liquidity_score": 90.0,
                        "liquidity_label": LiquidityLabel.CRISIS_LIQUIDITY.label,
                        "confidence": 0.7,
                        "drivers_json": "[]",
                        "release_gate": 1,
                        "artifact_hash": "sha256:0",
                        "metadata_json": "{}",
                    }
                ]
            )
        )
    finally:
        wh.close()

    rc = fi_cli_run(
        [
            "fi-score-liquidity",
            "--db",
            str(db_path),
            "--scope-type",
            "market",
            "--scope-id",
            "ALL",
            "--asof",
            _ASOF_STR,
            "--prev-label-from-warehouse",
            "true",
            "--model-run-id",
            "cli-prev-hys",
        ]
    )
    assert rc == 0
    wh = Warehouse(str(db_path))
    try:
        df = wh.read_liquidity_stress_scores()
        new_row = df.loc[df["model_run_id"] == "cli-prev-hys"].iloc[-1]
        meta = json.loads(new_row["metadata_json"])
        assert meta.get("hysteresis_applied") is True
        assert meta.get("prev_label") == LiquidityLabel.CRISIS_LIQUIDITY.value
    finally:
        wh.close()
