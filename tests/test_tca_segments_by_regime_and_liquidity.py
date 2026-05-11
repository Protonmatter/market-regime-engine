# SPDX-License-Identifier: Apache-2.0
"""PR-6 §G.2 — ``test_tca_segments_by_regime_and_liquidity`` (AGENT.md test catalog).

Synthesises a 100-trade dataset spanning the cartesian product of the
5 regime labels × 5 liquidity labels and asserts that
:func:`aggregate_tca_by_regime` produces the full 25-bucket × N-metric
grid.
"""

from __future__ import annotations

import math
from itertools import product

import pandas as pd

import market_regime_engine.fixed_income  # noqa: F401 — register FI schema
from market_regime_engine.fixed_income import LiquidityLabel, RegimeLabel
from market_regime_engine.fixed_income.tca_segmentation import (
    TCA_METRICS,
    aggregate_tca_by_regime,
)


def _build_synthetic_trades() -> pd.DataFrame:
    """100 trades: 4 trades per (regime × liquidity) bucket = 25 × 4."""
    rows = []
    pairs = list(
        product(
            [lbl.label for lbl in RegimeLabel],
            [lbl.label for lbl in LiquidityLabel],
        )
    )
    assert len(pairs) == 25
    request_id = 0
    for regime_label, liquidity_label in pairs:
        for k in range(4):
            rows.append(
                {
                    "request_id": f"req-{request_id}",
                    "regime_label": regime_label,
                    "liquidity_label": liquidity_label,
                    "execution_confidence_bucket": "high",
                    "protocol": "Auto-X",
                    "side": "buy",
                    "sector": "industrials",
                    "rating": "IG",
                    "maturity_bucket": "2-5y",
                    "notional_bucket": "1-5M",
                    "arrival_cost_bps": 2.0 + k * 0.5,
                    "vwap_slippage_bps": 1.5 + k * 0.5,
                    "price_improvement_bps": 0.5,
                    "market_impact_bps": 1.0,
                    "time_to_fill_seconds": 90.0,
                    "dealer_response_count": 5.0,
                    "quote_quality": 0.5,
                    "protocol_success": 1.0,
                    "post_trade_markout_1d_bps": 0.0,
                    "post_trade_markout_5d_bps": 0.0,
                    "execution_success": 1.0 if k < 3 else 0.0,
                    "regime_soft_weights": {regime_label: 1.0},
                }
            )
            request_id += 1
    return pd.DataFrame(rows)


def test_tca_segments_by_regime_and_liquidity_produces_25_buckets() -> None:
    trades = _build_synthetic_trades()
    assert len(trades) == 100

    agg = aggregate_tca_by_regime(
        trades,
        dimensions=("regime_label", "liquidity_label"),
        metrics_names=TCA_METRICS,
    )
    # 25 buckets x 11 metrics = 275 rows.
    assert not agg.empty
    by_metric_bucket = agg.groupby(["regime_label", "liquidity_label", "metric_name"]).size()
    # Each bucket x metric should appear exactly once.
    assert (by_metric_bucket == 1).all()
    # Every (regime, liquidity) pair populated.
    populated_pairs = agg[["regime_label", "liquidity_label"]].drop_duplicates()
    assert len(populated_pairs) == 25
    # All metrics populated.
    assert set(agg["metric_name"].unique()) == set(TCA_METRICS)


def test_tca_segments_by_regime_and_liquidity_each_bucket_has_four_trades() -> None:
    trades = _build_synthetic_trades()
    agg = aggregate_tca_by_regime(
        trades,
        dimensions=("regime_label", "liquidity_label"),
        metrics_names=("arrival_cost_bps",),
    )
    # Sample count per bucket = 4.
    assert (agg["sample_count"] == 4).all()


def test_tca_segments_by_regime_and_liquidity_metric_means_are_deterministic() -> None:
    """Bucket mean of arrival_cost_bps is (2.0 + 2.5 + 3.0 + 3.5) / 4 = 2.75."""
    trades = _build_synthetic_trades()
    agg = aggregate_tca_by_regime(
        trades,
        dimensions=("regime_label", "liquidity_label"),
        metrics_names=("arrival_cost_bps",),
    )
    assert all(agg["metric_value"].apply(lambda v: math.isclose(v, 2.75, abs_tol=1e-9)))
