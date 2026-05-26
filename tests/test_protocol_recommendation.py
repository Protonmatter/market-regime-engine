# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pandas as pd

from market_regime_engine.fixed_income import (
    ExecutionConfidenceRequest,
    LiquidityLabel,
    score_credit_regime,
    score_liquidity_stress,
    write_credit_regime_score,
    write_liquidity_stress_score,
)
from market_regime_engine.fixed_income.protocol_recommendation import (
    recommend_execution_protocol,
)
from market_regime_engine.storage import Warehouse


def _seed(wh: Warehouse, ts: pd.Timestamp, *, liquidity_index: float = 10.0) -> None:
    credit_rows = []
    for i in range(100):
        row_ts = ts - pd.Timedelta(days=100 - i)
        for feature_name in ("cdx_ig_5y", "cdx_hy_5y"):
            credit_rows.append(
                {
                    "date": row_ts,
                    "feature_name": feature_name,
                    "value": float(i),
                    "source_timestamp": row_ts,
                    "vintage_date": None,
                }
            )
    credit = pd.DataFrame(credit_rows)
    credit.attrs["nan_policy"] = "NAN_TO_LAST_VALID"
    credit_out = score_credit_regime(credit, asof=ts, release_gate=True)
    write_credit_regime_score(
        wh,
        type(credit_out)(
            timestamp=credit_out.timestamp,
            regime_score=credit_out.regime_score,
            regime_label=credit_out.regime_label,
            confidence=credit_out.confidence,
            drivers=credit_out.drivers,
            component_scores=credit_out.component_scores,
            model_run_id=credit_out.model_run_id,
            release_gate=True,
            artifact_hash=credit_out.artifact_hash,
            metadata=dict(credit_out.metadata),
        ),
    )

    liquidity_rows = []
    for i in range(100):
        row_ts = ts - pd.Timedelta(days=100 - i)
        for feature_name in ("bid_ask_width", "quotes_received"):
            liquidity_rows.append(
                {
                    "date": row_ts,
                    "feature_name": feature_name,
                    "value": float(liquidity_index),
                    "source_timestamp": row_ts,
                    "vintage_date": None,
                }
            )
    liquidity = pd.DataFrame(liquidity_rows)
    liquidity.attrs["nan_policy"] = "NAN_TO_LAST_VALID"
    liq_out = score_liquidity_stress(
        liquidity,
        scope_type="cusip",
        scope_id="00206RGB6",
        asof=ts,
        release_gate=True,
    )
    write_liquidity_stress_score(
        wh,
        type(liq_out)(
            timestamp=liq_out.timestamp,
            scope_type=liq_out.scope_type,
            scope_id=liq_out.scope_id,
            liquidity_index=liq_out.liquidity_index,
            liquidity_label=LiquidityLabel.NORMAL.label,
            confidence=liq_out.confidence,
            drivers=liq_out.drivers,
            model_run_id=liq_out.model_run_id,
            release_gate=True,
            artifact_hash=liq_out.artifact_hash,
            metadata=dict(liq_out.metadata),
        ),
    )


def _request(ts: pd.Timestamp) -> ExecutionConfidenceRequest:
    return ExecutionConfidenceRequest(
        timestamp=(ts + pd.Timedelta(seconds=30)).isoformat(),
        cusip="00206RGB6",
        side="buy",
        notional=1_000_000.0,
        protocol="Auto-X",
        urgency="normal",
        rating="BBB+",
    )


def test_recommendation_scores_all_candidates_without_persisting(tmp_path) -> None:
    wh = Warehouse(tmp_path / "protocol.duckdb")
    try:
        ts = pd.Timestamp("2026-05-01T16:00:00Z")
        _seed(wh, ts)
        out = recommend_execution_protocol(_request(ts), warehouse=wh)
        assert [score.protocol for score in out.candidate_scores] == ["Auto-X", "RFQ", "Manual"]
        assert out.recommended_protocol == "Auto-X"
        assert out.best_response.protocol == "Auto-X"
        assert wh.read_execution_confidence_predictions().empty
    finally:
        wh.close()


def test_recommendation_fails_closed_when_all_candidates_unavailable(tmp_path) -> None:
    wh = Warehouse(tmp_path / "protocol_stale.duckdb")
    try:
        ts = pd.Timestamp("2026-05-01T16:00:00Z")
        _seed(wh, ts)
        stale = _request(pd.Timestamp("2030-01-01T16:00:00Z"))
        out = recommend_execution_protocol(stale, warehouse=wh)
        assert out.recommended_protocol == "Manual"
        assert out.release_gate is False
        assert out.human_review_required is True
        assert "no_candidate_release_gate_passed" in out.reason_codes
    finally:
        wh.close()
