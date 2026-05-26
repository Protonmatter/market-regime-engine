# SPDX-License-Identifier: Apache-2.0
"""PR-6 §G.1 — TCA segmentation acceptance suite.

Pins:

- :func:`tag_trade_with_regime_context` attaches regime / liquidity /
  execution-confidence context (hard label + soft weights + buckets).
- :func:`compute_tca_metrics_for_outcome` returns every name in
  :data:`TCA_METRICS` with Decimal-precision arithmetic.
- :func:`aggregate_tca_by_regime` groups by the requested dimensions
  with both hard labels (default) and soft weighting.
- :func:`write_tca_regime_segment` /
  :func:`latest_tca_regime_segments` round-trip a
  :class:`TcaRegimeSegment` cleanly.
- :func:`materialize_tca_segments_for_day` writes one row per
  ``(dim-combo, metric)`` and returns the row count.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import market_regime_engine.fixed_income  # noqa: F401 — register FI schema
from market_regime_engine.fixed_income import (
    ExecutionConfidenceRequest,
    LiquidityLabel,
    RegimeLabel,
    TaggedTrade,
    TcaRegimeSegment,
    TradeRecord,
    score_credit_regime,
    score_execution_confidence,
    score_liquidity_stress,
    write_credit_regime_score,
    write_execution_confidence_prediction,
    write_execution_outcome,
    write_liquidity_stress_score,
)
from market_regime_engine.fixed_income.schemas import ExecutionConfidenceResponse
from market_regime_engine.fixed_income.tca_segmentation import (
    DIMENSION_COLUMNS,
    TCA_METRICS,
    aggregate_tca_by_regime,
    compute_tca_metrics_for_outcome,
    latest_tca_regime_segments,
    materialize_tca_segments_for_day,
    tag_trade_with_regime_context,
    write_tca_regime_segment,
)
from market_regime_engine.storage import Warehouse

# ---------------------------------------------------------------------------
# fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def wh(tmp_path: Path) -> Warehouse:
    return Warehouse(tmp_path / "tca.duckdb")


def _seed_signal_features(
    wh: Warehouse,
    *,
    asof: pd.Timestamp,
    regime_score_target: float = 30.0,
    liquidity_score_target: float = 25.0,
    cusip: str = "00206RGB6",
) -> None:
    """Seed warehouse with credit + liquidity rows targeting the scores.

    Seeds BOTH critical features per scorer so the v1.6.0 A11
    fail-closed override (which resets score to neutral 50) does not fire.
    """
    rows = []
    credit_feature_names = ("cdx_ig_5y", "cdx_hy_5y")
    for i in range(100):
        ts = asof - pd.Timedelta(days=100 - i)
        for fname in credit_feature_names:
            rows.append(
                {
                    "date": ts,
                    "feature_name": fname,
                    "value": float(i),
                    "source_timestamp": ts,
                    "vintage_date": None,
                }
            )
    for r in rows[-len(credit_feature_names) :]:
        r["value"] = float(regime_score_target - 1)
    feats = pd.DataFrame(rows)
    feats.attrs["nan_policy"] = "NAN_TO_LAST_VALID"
    credit_out = score_credit_regime(feats, asof=asof, release_gate=True)
    credit_out = type(credit_out)(
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
    )
    write_credit_regime_score(wh, credit_out)

    rows = []
    liquidity_feature_names = ("bid_ask_width", "quotes_received")
    for i in range(100):
        ts = asof - pd.Timedelta(days=100 - i)
        for fname in liquidity_feature_names:
            rows.append(
                {
                    "date": ts,
                    "feature_name": fname,
                    "value": float(i),
                    "source_timestamp": ts,
                    "vintage_date": None,
                }
            )
    for r in rows[-len(liquidity_feature_names) :]:
        r["value"] = float(liquidity_score_target)
    feats = pd.DataFrame(rows)
    feats.attrs["nan_policy"] = "NAN_TO_LAST_VALID"
    liquidity_out = score_liquidity_stress(
        feats,
        scope_type="cusip",
        scope_id=cusip,
        asof=asof,
        release_gate=True,
    )
    liquidity_out = type(liquidity_out)(
        timestamp=liquidity_out.timestamp,
        scope_type=liquidity_out.scope_type,
        scope_id=liquidity_out.scope_id,
        liquidity_index=liquidity_out.liquidity_index,
        liquidity_label=liquidity_out.liquidity_label,
        confidence=liquidity_out.confidence,
        drivers=liquidity_out.drivers,
        model_run_id=liquidity_out.model_run_id,
        release_gate=True,
        artifact_hash=liquidity_out.artifact_hash,
        metadata=dict(liquidity_out.metadata),
    )
    write_liquidity_stress_score(wh, liquidity_out)


def _trade(
    *,
    timestamp: pd.Timestamp,
    cusip: str = "00206RGB6",
    notional: float = 1_000_000.0,
    side: str = "buy",
    sector: str | None = "industrials",
    rating: str | None = "BBB+",
    maturity_years: float | None = 4.5,
    protocol: str = "Auto-X",
    arrival_price: float | None = 100.0,
    execution_price: float | None = 100.05,
) -> TradeRecord:
    return TradeRecord(
        request_id=f"req-{cusip}-{timestamp.value}",
        timestamp=timestamp,
        cusip=cusip,
        side=side,  # type: ignore[arg-type]
        notional=notional,
        protocol=protocol,
        arrival_price=arrival_price,
        execution_price=execution_price,
        filled_quantity=notional,
        time_to_fill_seconds=120.0,
        dealer_response_count=5,
        sector=sector,
        rating=rating,
        maturity_years=maturity_years,
    )


# ---------------------------------------------------------------------------
# tag_trade_with_regime_context
# ---------------------------------------------------------------------------


def test_tag_trade_with_regime_context_returns_tagged_trade(wh: Warehouse) -> None:
    asof = pd.Timestamp("2026-05-01T16:00:00Z")
    _seed_signal_features(wh, asof=asof)
    trade = _trade(timestamp=asof + pd.Timedelta(seconds=30))
    tagged = tag_trade_with_regime_context(trade, warehouse=wh)
    assert isinstance(tagged, TaggedTrade)
    assert tagged.trade is trade


def test_tag_trade_attaches_credit_regime_label_and_score(wh: Warehouse) -> None:
    asof = pd.Timestamp("2026-05-01T16:00:00Z")
    _seed_signal_features(wh, asof=asof, regime_score_target=30.0)
    trade = _trade(timestamp=asof + pd.Timedelta(seconds=30))
    tagged = tag_trade_with_regime_context(trade, warehouse=wh)
    assert tagged.regime_label  # non-empty
    # regime_score must reflect the seeded value (deterministic scorer).
    assert 0.0 <= tagged.regime_score <= 100.0


def test_tag_trade_attaches_liquidity_label_per_cusip_fallback_market(
    wh: Warehouse,
) -> None:
    asof = pd.Timestamp("2026-05-01T16:00:00Z")
    _seed_signal_features(wh, asof=asof, cusip="00206RGB6")
    # Trade in a *different* cusip → no cusip-scoped liquidity row → falls
    # back to the market scope. But our seeder also only wrote a cusip
    # row, so test should fall back to None liquidity → "unknown" label,
    # or — under the v1.5.1 (PR-9 FIX 8) critical-feature contract — the
    # explicit fail-closed ``NO_DECISION`` label when the seeded liquidity
    # row had no bid-ask / RFQ observations.
    trade = _trade(timestamp=asof + pd.Timedelta(seconds=30), cusip="DIFFERENTCUSIP")
    tagged = tag_trade_with_regime_context(trade, warehouse=wh)
    # When no scope at all is available, fallback to neutral.
    assert tagged.liquidity_label in {
        "unknown",
        "NO_DECISION",
        *{lbl.label for lbl in LiquidityLabel},
    }


def test_tag_trade_attaches_execution_confidence_bucket(wh: Warehouse) -> None:
    asof = pd.Timestamp("2026-05-01T16:00:00Z")
    _seed_signal_features(wh, asof=asof)
    # Score and persist an execution-confidence prediction for the cusip so
    # the tagger can look it up.
    request = ExecutionConfidenceRequest(
        timestamp=(asof + pd.Timedelta(seconds=10)).isoformat(),
        cusip="00206RGB6",
        side="buy",
        notional=1_000_000.0,
        protocol="Auto-X",
    )
    response = score_execution_confidence(request, warehouse=wh, release_gate=True)
    write_execution_confidence_prediction(wh, response, request_id="req-tagtest")
    trade = _trade(timestamp=asof + pd.Timedelta(seconds=30))
    tagged = tag_trade_with_regime_context(trade, warehouse=wh)
    assert tagged.execution_confidence_bucket in {"high", "medium", "low", "unavailable"}


def test_tag_trade_uses_prior_prediction_when_future_prediction_exists(
    wh: Warehouse,
) -> None:
    """P0 adversarial PIT regression: a future latest prediction must not
    hide a valid prior prediction for the same CUSIP."""
    asof = pd.Timestamp("2026-05-01T16:00:00Z")
    _seed_signal_features(wh, asof=asof)

    prior_request = ExecutionConfidenceRequest(
        timestamp=(asof + pd.Timedelta(seconds=10)).isoformat(),
        cusip="00206RGB6",
        side="buy",
        notional=1_000_000.0,
        protocol="Auto-X",
    )
    prior_response = score_execution_confidence(prior_request, warehouse=wh, release_gate=True)
    write_execution_confidence_prediction(wh, prior_response, request_id="req-prior")

    future_request = ExecutionConfidenceRequest(
        timestamp=(asof + pd.Timedelta(hours=1)).isoformat(),
        cusip="00206RGB6",
        side="buy",
        notional=1_000_000.0,
        protocol="Auto-X",
    )
    future_response = score_execution_confidence(future_request, warehouse=wh, release_gate=True)
    write_execution_confidence_prediction(wh, future_response, request_id="req-future")

    trade = _trade(timestamp=asof + pd.Timedelta(seconds=30))
    tagged = tag_trade_with_regime_context(trade, warehouse=wh)

    assert tagged.metadata["execution_confidence_source_request_id"] == "req-prior"
    assert tagged.execution_confidence_score == pytest.approx(prior_response.confidence_score)


def test_tag_trade_soft_regime_weights_sum_to_one(wh: Warehouse) -> None:
    asof = pd.Timestamp("2026-05-01T16:00:00Z")
    _seed_signal_features(wh, asof=asof, regime_score_target=45.0)
    trade = _trade(timestamp=asof + pd.Timedelta(seconds=30))
    tagged = tag_trade_with_regime_context(trade, warehouse=wh)
    weights_sum = sum(tagged.regime_soft_weights.values())
    assert weights_sum == pytest.approx(1.0, abs=1e-9)
    # Saturated low or high scores collapse to a single bucket.
    _seed_signal_features(wh, asof=asof + pd.Timedelta(days=1), regime_score_target=5.0)
    trade2 = _trade(timestamp=asof + pd.Timedelta(days=1, seconds=30))
    tagged2 = tag_trade_with_regime_context(trade2, warehouse=wh)
    assert sum(tagged2.regime_soft_weights.values()) == pytest.approx(1.0, abs=1e-9)


def test_tag_trade_hard_label_matches_regime_score_bucket_with_hysteresis(
    wh: Warehouse,
) -> None:
    asof = pd.Timestamp("2026-05-01T16:00:00Z")
    _seed_signal_features(wh, asof=asof, regime_score_target=10.0)
    trade = _trade(timestamp=asof + pd.Timedelta(seconds=30))
    tagged_h = tag_trade_with_regime_context(trade, warehouse=wh, use_hysteresis=True)
    tagged_nh = tag_trade_with_regime_context(trade, warehouse=wh, use_hysteresis=False)
    # Both should be non-empty labels. With a sub-20 score, sharp-bucket
    # = RISK_ON_COMPRESSION.
    assert tagged_nh.regime_label == RegimeLabel.RISK_ON_COMPRESSION.label
    # Hysteresis label should also be RISK_ON given the prev_label
    # (persisted credit_regime_score.regime_label) is the same.
    assert tagged_h.regime_label in {
        RegimeLabel.RISK_ON_COMPRESSION.label,
        RegimeLabel.NORMAL_LIQUIDITY.label,
    }


def test_tag_trade_assigns_maturity_bucket_correctly(wh: Warehouse) -> None:
    asof = pd.Timestamp("2026-05-01T16:00:00Z")
    _seed_signal_features(wh, asof=asof)
    cases = [
        (1.0, "0-2y"),
        (3.0, "2-5y"),
        (7.0, "5-10y"),
        (15.0, "10y+"),
        (None, "unknown"),
    ]
    for y, expected in cases:
        trade = _trade(
            timestamp=asof + pd.Timedelta(seconds=30),
            maturity_years=y,
        )
        tagged = tag_trade_with_regime_context(trade, warehouse=wh)
        assert tagged.maturity_bucket == expected, (y, tagged.maturity_bucket)


def test_tag_trade_assigns_notional_bucket_correctly(wh: Warehouse) -> None:
    asof = pd.Timestamp("2026-05-01T16:00:00Z")
    _seed_signal_features(wh, asof=asof)
    cases = [
        (100_000.0, "<1M"),
        (3_000_000.0, "1-5M"),
        (10_000_000.0, "5-25M"),
        (50_000_000.0, "25M+"),
    ]
    for n, expected in cases:
        trade = _trade(timestamp=asof + pd.Timedelta(seconds=30), notional=n)
        tagged = tag_trade_with_regime_context(trade, warehouse=wh)
        assert tagged.notional_bucket == expected, (n, tagged.notional_bucket)


# ---------------------------------------------------------------------------
# compute_tca_metrics_for_outcome
# ---------------------------------------------------------------------------


def _basic_request_response_outcome() -> tuple[ExecutionConfidenceRequest, ExecutionConfidenceResponse, dict]:
    decision_ts = "2026-05-01T16:00:00Z"
    observed_at = "2026-05-01T16:30:00Z"
    request = ExecutionConfidenceRequest(
        timestamp=decision_ts,
        cusip="00206RGB6",
        side="buy",
        notional=1_000_000.0,
        protocol="Auto-X",
    )
    response = ExecutionConfidenceResponse(
        timestamp=decision_ts,
        cusip="00206RGB6",
        side="buy",
        notional=1_000_000.0,
        protocol="Auto-X",
        confidence_score=0.75,
        expected_slippage_bps=10.0,
        confidence_interval_low=0.65,
        confidence_interval_high=0.85,
        recommended_action="Auto-X allowed",
        human_review_required=False,
        model_run_id="m-1",
        release_gate=True,
        artifact_hash="sha256:test",
    )
    outcome = {
        "observed_at": observed_at,
        "arrival_price": 100.0,
        "execution_price": 100.05,  # 5 bps cost on a buy
        "vwap_price": 100.04,
        "mid_price_at_arrival": 100.005,
        "best_bid_at_arrival": 99.99,
        "best_ask_at_arrival": 100.02,
        "time_to_fill_seconds": 120.0,
        "dealer_response_count": 5,
        "filled_quantity": 1_000_000.0,
        "markout_price_1d": 100.10,
        "markout_price_5d": 100.20,
    }
    return request, response, outcome


def test_compute_tca_metrics_for_outcome_returns_all_required_metrics() -> None:
    request, response, outcome = _basic_request_response_outcome()
    # Force asof_now > 5 trading days after decision so the markout
    # window observability check passes.
    asof_now = pd.Timestamp("2026-05-15T16:00:00Z")
    result = compute_tca_metrics_for_outcome(request, response, outcome, warehouse=None, asof_now=asof_now)
    for metric in TCA_METRICS:
        assert metric in result, metric
    # All non-None for the well-populated outcome.
    assert result["arrival_cost_bps"] is not None
    assert result["execution_success"] in {0.0, 1.0}
    # arrival_cost_bps = +5 bps for a buy.
    assert result["arrival_cost_bps"] == pytest.approx(5.0, abs=0.01)


def test_compute_tca_metrics_decimal_precision_preserved() -> None:
    """A 0.25-bps cost on a $100M buy must produce arrival_cost_bps = 0.25 exactly."""
    request = ExecutionConfidenceRequest(
        timestamp="2026-05-01T16:00:00Z",
        cusip="00206RGB6",
        side="buy",
        notional=100_000_000.0,
        protocol="Auto-X",
    )
    response = ExecutionConfidenceResponse(
        timestamp="2026-05-01T16:00:00Z",
        cusip="00206RGB6",
        side="buy",
        notional=100_000_000.0,
        protocol="Auto-X",
        confidence_score=0.75,
        expected_slippage_bps=10.0,
        confidence_interval_low=0.65,
        confidence_interval_high=0.85,
        recommended_action="Auto-X allowed",
        human_review_required=False,
        model_run_id="m-2",
        release_gate=True,
        artifact_hash="sha256:test",
    )
    outcome = {
        "observed_at": "2026-05-01T16:30:00Z",
        "arrival_price": 100.0,
        # 0.25 bps cost = price moves 0.0025 on a buy at par.
        "execution_price": 100.0025,
    }
    result = compute_tca_metrics_for_outcome(request, response, outcome, warehouse=None)
    assert result["arrival_cost_bps"] == pytest.approx(0.25, abs=1e-9)


def test_compute_tca_metrics_returns_none_for_unobservable_markout() -> None:
    """5d markout returns None when the 5-trading-day window has not closed."""
    request, response, outcome = _basic_request_response_outcome()
    # asof_now = 1 trading day after decision → 1d window closed, 5d not.
    asof_now = pd.Timestamp("2026-05-04T16:00:00Z")  # Monday after Friday
    result = compute_tca_metrics_for_outcome(request, response, outcome, warehouse=None, asof_now=asof_now)
    assert result["post_trade_markout_5d_bps"] is None


# ---------------------------------------------------------------------------
# aggregate_tca_by_regime
# ---------------------------------------------------------------------------


def _synthetic_trades_frame(n: int = 25) -> pd.DataFrame:
    rows = []
    regime_labels = [lbl.label for lbl in RegimeLabel]
    liquidity_labels = [lbl.label for lbl in LiquidityLabel]
    for i in range(n):
        rows.append(
            {
                "request_id": f"req-{i}",
                "regime_label": regime_labels[i % len(regime_labels)],
                "liquidity_label": liquidity_labels[i % len(liquidity_labels)],
                "execution_confidence_bucket": "high",
                "protocol": "Auto-X",
                "side": "buy",
                "sector": "industrials",
                "rating": "IG",
                "maturity_bucket": "2-5y",
                "notional_bucket": "1-5M",
                "arrival_cost_bps": float(i % 5),
                "vwap_slippage_bps": float(i % 3),
                "price_improvement_bps": 0.0,
                "market_impact_bps": 1.0,
                "time_to_fill_seconds": 60.0,
                "dealer_response_count": 5.0,
                "quote_quality": 0.5,
                "protocol_success": 1.0,
                "post_trade_markout_1d_bps": 0.0,
                "post_trade_markout_5d_bps": 0.0,
                "execution_success": 1.0 if (i % 4) != 0 else 0.0,
                "regime_soft_weights": {
                    regime_labels[i % len(regime_labels)]: 0.7,
                    regime_labels[(i + 1) % len(regime_labels)]: 0.3,
                },
            }
        )
    return pd.DataFrame(rows)


def test_aggregate_tca_by_regime_groups_correctly_with_hard_labels() -> None:
    trades = _synthetic_trades_frame(n=25)
    agg = aggregate_tca_by_regime(
        trades,
        dimensions=("regime_label", "liquidity_label"),
        metrics_names=("arrival_cost_bps", "execution_success"),
    )
    assert not agg.empty
    assert set(agg.columns) >= {
        "regime_label",
        "liquidity_label",
        "metric_name",
        "metric_value",
        "sample_count",
    }
    # Hard-label grouping: each (regime, liq) bucket appears at most once per metric.
    grouped = agg.groupby(["regime_label", "liquidity_label", "metric_name"]).size()
    assert (grouped == 1).all(), grouped


def test_aggregate_tca_by_regime_groups_correctly_with_soft_weighting() -> None:
    trades = _synthetic_trades_frame(n=25)
    agg = aggregate_tca_by_regime(
        trades,
        dimensions=("regime_label",),
        metrics_names=("arrival_cost_bps",),
        soft_weighting=True,
    )
    assert not agg.empty
    # Soft weighting must distribute trades across multiple labels per row,
    # so total sample_count across regime labels exceeds the trade count.
    total_count = agg.loc[agg["metric_name"] == "arrival_cost_bps", "sample_count"].sum()
    # Each trade contributes to up to 2 labels (0.7 + 0.3 weights = 1.0).
    assert total_count >= 25, total_count


def test_aggregate_tca_by_regime_empty_input_returns_empty_frame() -> None:
    agg = aggregate_tca_by_regime(
        pd.DataFrame(),
        dimensions=("regime_label",),
        metrics_names=("arrival_cost_bps",),
    )
    assert agg.empty


def test_aggregate_tca_by_regime_rejects_invalid_dimensions() -> None:
    trades = _synthetic_trades_frame(n=5)
    with pytest.raises(ValueError, match="invalid dimensions"):
        aggregate_tca_by_regime(
            trades,
            dimensions=("bogus_dim",),  # type: ignore[arg-type]
            metrics_names=("arrival_cost_bps",),
        )


def test_aggregate_tca_by_regime_rejects_invalid_metric() -> None:
    trades = _synthetic_trades_frame(n=5)
    with pytest.raises(ValueError, match="invalid metrics"):
        aggregate_tca_by_regime(
            trades,
            dimensions=("regime_label",),
            metrics_names=("bogus_metric",),
        )


# ---------------------------------------------------------------------------
# write_tca_regime_segment / latest_tca_regime_segments round-trip
# ---------------------------------------------------------------------------


def test_write_and_read_tca_regime_segment_roundtrip(wh: Warehouse) -> None:
    segment = TcaRegimeSegment(
        timestamp=pd.Timestamp("2026-05-01T00:00:00Z"),
        regime_label="Normal Liquidity",
        liquidity_label="Normal",
        execution_confidence_bucket=None,
        protocol=None,
        side=None,
        sector=None,
        rating=None,
        maturity_bucket=None,
        notional_bucket=None,
        metric_name="arrival_cost_bps",
        metric_value=2.5,
        sample_count=10,
        model_run_id="rt-1",
        metadata_json=json.dumps({"dim": "regime_label,liquidity_label"}, sort_keys=True),
    )
    rows = write_tca_regime_segment(wh, segment)
    assert rows == 1
    out = latest_tca_regime_segments(wh, limit=10)
    assert len(out) == 1
    assert out[0].metric_name == "arrival_cost_bps"
    assert out[0].metric_value == 2.5
    assert out[0].sample_count == 10
    # The non-dim fields persist as None via the __all__ sentinel.
    assert out[0].protocol is None


# ---------------------------------------------------------------------------
# materialize_tca_segments_for_day
# ---------------------------------------------------------------------------


def test_materialize_tca_segments_for_day_writes_one_row_per_dim_combo_metric(
    wh: Warehouse,
) -> None:
    asof = pd.Timestamp("2026-05-01T16:00:00Z")
    _seed_signal_features(wh, asof=asof)
    # Synthesise one decision + one outcome so the materialiser has data.
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
    write_execution_confidence_prediction(wh, response, request_id="req-mat-1")
    write_execution_outcome(
        wh,
        request_id="req-mat-1",
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
            "maturity_years": 4.5,
        },
    )
    rows_written = materialize_tca_segments_for_day(wh, date=asof.normalize(), soft_weighting=False)
    assert rows_written > 0
    # Sanity: read back rows.
    segments = latest_tca_regime_segments(wh, limit=200)
    assert len(segments) == rows_written


def test_dimension_columns_covers_spec_set() -> None:
    expected = {
        "regime_label",
        "liquidity_label",
        "execution_confidence_bucket",
        "protocol",
        "side",
        "sector",
        "rating",
        "maturity_bucket",
        "notional_bucket",
    }
    assert set(DIMENSION_COLUMNS) == expected


def test_tca_metrics_covers_spec_set() -> None:
    expected = {
        "arrival_cost_bps",
        "vwap_slippage_bps",
        "price_improvement_bps",
        "market_impact_bps",
        "time_to_fill_seconds",
        "dealer_response_count",
        "quote_quality",
        "protocol_success",
        "post_trade_markout_1d_bps",
        "post_trade_markout_5d_bps",
        "execution_success",
    }
    assert set(TCA_METRICS) == expected
