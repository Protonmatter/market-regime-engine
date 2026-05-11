# SPDX-License-Identifier: Apache-2.0
"""PR-5 acceptance gate: ``release_gate=False`` MUST fail closed.

Per ``MRE_FIXED_INCOME_AGENT.md`` non-negotiable 8 and the PR-5 plan
acceptance criteria: when ``release_gate=False`` is passed in, the
response MUST land at ``recommended_action="Manual review required"``
with ``human_review_required=True`` — regardless of how high the
underlying logit climbs."""

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
    # Strong inputs so the underlying score would normally be high.
    rows = [
        {
            "date": ts - pd.Timedelta(days=i),
            "feature_name": "cdx_ig_5y",
            "value": float(i),
            "source_timestamp": ts - pd.Timedelta(days=i),
            "vintage_date": None,
        }
        for i in range(100, 0, -1)
    ]
    rows.append(
        {
            "date": ts,
            "feature_name": "cdx_ig_5y",
            "value": 10.0,
            "source_timestamp": ts,
            "vintage_date": None,
        }
    )
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
        for i in range(100, 0, -1)
    ]
    rows.append(
        {
            "date": ts,
            "feature_name": "bid_ask_width",
            "value": 10.0,
            "source_timestamp": ts,
            "vintage_date": None,
        }
    )
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


def test_execution_confidence_release_gate_false_fails_closed(tmp_path: Path) -> None:
    wh = Warehouse(tmp_path / "fail_closed.duckdb")
    asof = pd.Timestamp("2026-05-01T16:00:00Z")
    _seed(wh, asof)
    request = ExecutionConfidenceRequest(
        timestamp=(asof + pd.Timedelta(seconds=30)).isoformat(),
        cusip="00206RGB6",
        side="buy",
        notional=100_000,
        protocol="Auto-X",
        urgency="low",
        rating="AAA",
    )
    out = score_execution_confidence(
        request,
        warehouse=wh,
        release_gate=False,
        # Even with a pumped intercept the gate MUST flip the decision.
        weights={"base_intercept": 5.0},
    )
    assert out.recommended_action == "Manual review required"
    assert out.human_review_required is True
    assert out.release_gate is False
