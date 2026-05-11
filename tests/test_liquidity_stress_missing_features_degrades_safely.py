# SPDX-License-Identifier: Apache-2.0
"""Liquidity stress: missing-features degradation (AGENT.md test catalog).

Per ``MRE_FIXED_INCOME_AGENT.md §"Acceptance tests"``: a sparse-data
cusip — one whose trade / quote / RFQ tables have only partial
coverage of the lookback window — must yield
``release_gate=false`` (and confidence capped at 0.5) instead of a
silent "Normal" verdict. This pins the
``NanPolicy.NAN_FAILS_PIT_AUDIT`` contract end-to-end.
"""

from __future__ import annotations

import pandas as pd

import market_regime_engine.fixed_income  # noqa: F401  - register FI schema
from market_regime_engine.fixed_income.liquidity_stress import score_liquidity_stress
from market_regime_engine.frontier.data_cleaning import NanPolicy

_ASOF = pd.Timestamp("2026-05-08T16:00:00+00:00")


def _sparse_features() -> pd.DataFrame:
    """Build a feature frame with ONLY bid-ask present (other components missing)."""
    dates = pd.date_range(end=_ASOF, periods=10, freq="D", tz="UTC")
    rows = []
    for ts in dates:
        rows.append(
            {
                "date": ts,
                "feature_name": "bid_ask_width",
                "value": 1.0,
                "source_timestamp": ts,
                "vintage_date": None,
            }
        )
    frame = pd.DataFrame(rows)
    frame.attrs["nan_policy"] = NanPolicy.NAN_FAILS_PIT_AUDIT.value
    return frame


def test_liquidity_stress_missing_features_degrades_safely() -> None:
    """Sparse cusip → ``release_gate=False`` and capped confidence (no silent Normal)."""
    out = score_liquidity_stress(
        _sparse_features(),
        scope_type="cusip",
        scope_id="9128283N8",
        asof=_ASOF,
        model_run_id="run-sparse",
        release_gate=True,  # caller asked for production, the gate must self-flip.
    )
    assert out.release_gate is False, (
        "missing required features must flip release_gate=False; "
        f"got release_gate={out.release_gate}"
    )
    assert out.confidence <= 0.5 + 1e-9, (
        f"confidence must be capped at 0.5 when release_gate flips; got {out.confidence}"
    )
    assert out.metadata.get("pit_audit_failed") is True
    # The scorer must surface *which* components were missing for the
    # operator runbook — five of the six (only bid_ask present).
    missing = set(out.metadata.get("missing_features", []))
    assert "quotes_dispersion" in missing
    assert "trade_velocity" in missing
    assert "rfq_fill_rate" in missing
    assert "amihud" in missing
    assert "time_gap" in missing
