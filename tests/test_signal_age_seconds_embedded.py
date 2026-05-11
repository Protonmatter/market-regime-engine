# SPDX-License-Identifier: Apache-2.0
"""PR-5 review §3.6 PR-13: signal_age_seconds embedded in every response."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import market_regime_engine.fixed_income  # noqa: F401  registers FI schema
from market_regime_engine.fixed_income import (
    ExecutionConfidenceRequest,
    score_credit_regime,
    score_execution_confidence,
    score_liquidity_stress,
    write_credit_regime_score,
    write_liquidity_stress_score,
)
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


def test_signal_age_seconds_keys_present_on_every_response(tmp_path: Path) -> None:
    wh = Warehouse(tmp_path / "age.duckdb")
    signal_ts = pd.Timestamp("2026-05-01T16:00:00Z")
    _seed(wh, signal_ts)
    decision_ts = signal_ts + pd.Timedelta(seconds=45)
    request = ExecutionConfidenceRequest(
        timestamp=decision_ts.isoformat(),
        cusip="00206RGB6",
        side="buy",
        notional=1_000_000,
        protocol="Auto-X",
        urgency="normal",
    )
    out = score_execution_confidence(request, warehouse=wh, release_gate=True)
    assert "signal_age_seconds_credit_regime" in out.metadata
    assert "signal_age_seconds_liquidity" in out.metadata
    assert "max_signal_age_seconds" in out.metadata
    # 45-second delta should round to ~45 (the regime/liquidity rows were
    # stamped exactly at signal_ts).
    assert 30 <= out.metadata["max_signal_age_seconds"] <= 60


def test_signal_age_seconds_present_even_on_stale_signal_response(tmp_path: Path) -> None:
    wh = Warehouse(tmp_path / "age_stale.duckdb")
    signal_ts = pd.Timestamp("2026-05-01T16:00:00Z")
    _seed(wh, signal_ts)
    decision_ts = signal_ts + pd.Timedelta(hours=5)  # well beyond the 15-min threshold
    request = ExecutionConfidenceRequest(
        timestamp=decision_ts.isoformat(),
        cusip="00206RGB6",
        side="buy",
        notional=1_000_000,
        protocol="Auto-X",
        urgency="normal",
    )
    out = score_execution_confidence(request, warehouse=wh, release_gate=True)
    assert out.recommended_action == "Unavailable — stale signal"
    # The stale-response path still populates the age keys for telemetry.
    assert "signal_age_seconds_credit_regime" in out.metadata
    assert "signal_age_seconds_liquidity" in out.metadata
    assert out.metadata["max_signal_age_seconds"] > 60 * 60
