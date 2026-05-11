# SPDX-License-Identifier: Apache-2.0
"""Per-scope liquidity-stress feature-builder + warehouse tests (task H.2)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import market_regime_engine.fixed_income  # noqa: F401  - register FI schema
from market_regime_engine.fixed_income.calendars import is_trading_day
from market_regime_engine.fixed_income.feature_builders import build_liquidity_features
from market_regime_engine.fixed_income.liquidity_stress import (
    latest_liquidity_stress_score,
    list_recent_liquidity_stress_scores,
    score_liquidity_stress,
    write_liquidity_stress_score,
)
from market_regime_engine.storage import Warehouse

_ASOF = pd.Timestamp("2026-05-08T16:00:00+00:00")
_CUSIP_A = "9128283N8"
_CUSIP_B = "912828K58"


def _trading_dates(asof: pd.Timestamp, n: int) -> list[pd.Timestamp]:
    """Pick the last ``n`` SIFMA trading days ending at ``asof``."""
    raw = pd.date_range(end=asof, periods=n * 2, freq="D", tz="UTC")
    return [d for d in raw if is_trading_day(d)][-n:]


def _seed_warehouse(wh: Warehouse) -> None:
    """Plant a per-cusip universe with trades / RFQs / quotes / bond reference."""
    dates = _trading_dates(_ASOF, 30)

    # Bond reference: two sectors × two ratings.
    valid_from = (_ASOF - pd.Timedelta(days=400)).isoformat()
    wh.write_bond_reference(
        pd.DataFrame(
            [
                {
                    "cusip": _CUSIP_A,
                    "valid_from": valid_from,
                    "valid_to": None,
                    "ticker": "GS-A",
                    "issuer": "GS",
                    "sector": "financials",
                    "rating": "AA",
                    "issue_date": valid_from,
                    "maturity": "2031-05-01T00:00:00+00:00",
                    "coupon": 4.5,
                    "currency": "USD",
                    "country": "US",
                    "duration": 6.5,
                    "convexity": 0.8,
                    "amount_outstanding": 5e9,
                    "is_callable": 0,
                    "call_schedule_json": "{}",
                    "default_date": None,
                    "delisted_date": None,
                    "metadata_json": "{}",
                },
                {
                    "cusip": _CUSIP_B,
                    "valid_from": valid_from,
                    "valid_to": None,
                    "ticker": "XOM-B",
                    "issuer": "XOM",
                    "sector": "energy",
                    "rating": "BBB",
                    "issue_date": valid_from,
                    "maturity": "2034-05-01T00:00:00+00:00",
                    "coupon": 5.0,
                    "currency": "USD",
                    "country": "US",
                    "duration": 8.0,
                    "convexity": 1.0,
                    "amount_outstanding": 3e9,
                    "is_callable": 0,
                    "call_schedule_json": "{}",
                    "default_date": None,
                    "delisted_date": None,
                    "metadata_json": "{}",
                },
            ]
        )
    )

    trades: list[dict] = []
    rfqs: list[dict] = []
    quotes: list[dict] = []
    for i, d in enumerate(dates):
        ts = d.isoformat().replace("+00:00", "Z")
        for j, cusip in enumerate((_CUSIP_A, _CUSIP_B)):
            # Two trades per day at slightly different prices and sizes.
            trades.append(
                {
                    "trade_id": f"T-{i}-{j}-0",
                    "timestamp": ts,
                    "cusip": cusip,
                    "price": 100.0 + 0.01 * i + 0.05 * j,
                    "yield_pct": 4.50 + 0.005 * i,
                    "size": 1_000_000.0,
                    "side": "buy" if j == 0 else "sell",
                    "protocol": "RFQ",
                    "venue": "MarketAxess",
                    "source": "trace",
                    "reported_at": ts,
                    "metadata_json": "{}",
                }
            )
            trades.append(
                {
                    "trade_id": f"T-{i}-{j}-1",
                    "timestamp": ts,
                    "cusip": cusip,
                    "price": 100.05 + 0.01 * i + 0.05 * j,
                    "yield_pct": 4.51 + 0.005 * i,
                    "size": 500_000.0,
                    "side": "sell" if j == 0 else "buy",
                    "protocol": "RFQ",
                    "venue": "Tradeweb",
                    "source": "trace",
                    "reported_at": ts,
                    "metadata_json": "{}",
                }
            )
            # One RFQ per cusip per day.
            rfqs.append(
                {
                    "rfq_id": f"R-{i}-{j}",
                    "timestamp": ts,
                    "cusip": cusip,
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
            # Two dealer quotes per cusip per day.
            for dealer in ("DEAL_A", "DEAL_B"):
                quotes.append(
                    {
                        "timestamp": ts,
                        "cusip": cusip,
                        "dealer_id": dealer,
                        "side": "bid",
                        "price": 99.5 + 0.01 * i + 0.05 * j,
                        "size": 1_000_000.0,
                        "expires_at": ts,
                        "metadata_json": "{}",
                    }
                )
                quotes.append(
                    {
                        "timestamp": ts,
                        "cusip": cusip,
                        "dealer_id": dealer,
                        "side": "ask",
                        "price": 100.5 + 0.01 * i + 0.05 * j,
                        "size": 1_000_000.0,
                        "expires_at": ts,
                        "metadata_json": "{}",
                    }
                )

    wh.write_trace_trades(pd.DataFrame(trades))
    wh.write_rfq_events(pd.DataFrame(rfqs))
    wh.write_dealer_quotes(pd.DataFrame(quotes))


# ---------------------------------------------------------------------------
# Scope filtering
# ---------------------------------------------------------------------------


@pytest.fixture
def seeded_warehouse(tmp_path: Path) -> Warehouse:
    wh = Warehouse(str(tmp_path / "fi-scope.duckdb"))
    _seed_warehouse(wh)
    yield wh
    wh.close()


def test_market_scope_aggregates_across_all_cusips(seeded_warehouse: Warehouse) -> None:
    """``scope_type='market'`` produces features across the entire bond universe."""
    features = build_liquidity_features(
        seeded_warehouse,
        asof=_ASOF,
        scope_type="market",
        scope_id="ALL",
        lookback_days=30,
    )
    assert not features.empty
    # Every emitted feature_name must be one of the eleven liquidity inputs.
    valid_names = {
        "bid_ask_width",
        "trade_count_velocity",
        "volume_over_adv",
        "time_since_last_trade",
        "dealers_requested",
        "quotes_received",
        "quote_dispersion",
        "amihud_illiquidity",
        "dealer_response_count",
        "axe_freshness_proxy",
        "order_imbalance",
    }
    assert set(features["feature_name"].unique()) <= valid_names


def test_sector_scope_filters_via_bond_reference(seeded_warehouse: Warehouse) -> None:
    """``scope_type='sector'`` filters the trade universe to one sector."""
    fin = build_liquidity_features(
        seeded_warehouse,
        asof=_ASOF,
        scope_type="sector",
        scope_id="financials",
        lookback_days=30,
    )
    ene = build_liquidity_features(
        seeded_warehouse,
        asof=_ASOF,
        scope_type="sector",
        scope_id="energy",
        lookback_days=30,
    )
    assert not fin.empty
    assert not ene.empty
    # The two sectors live in different CUSIPs, so trade counts diverge.
    fin_velo = fin.loc[fin["feature_name"] == "trade_count_velocity", "value"].sum()
    ene_velo = ene.loc[ene["feature_name"] == "trade_count_velocity", "value"].sum()
    assert fin_velo > 0
    assert ene_velo > 0


def test_rating_scope_filters_via_bond_reference(seeded_warehouse: Warehouse) -> None:
    """``scope_type='rating'`` filters by rating bucket."""
    aa = build_liquidity_features(
        seeded_warehouse,
        asof=_ASOF,
        scope_type="rating",
        scope_id="AA",
        lookback_days=30,
    )
    bbb = build_liquidity_features(
        seeded_warehouse,
        asof=_ASOF,
        scope_type="rating",
        scope_id="BBB",
        lookback_days=30,
    )
    assert not aa.empty
    assert not bbb.empty


def test_cusip_scope_returns_single_bond_score(seeded_warehouse: Warehouse) -> None:
    """``scope_type='cusip'`` reads exactly one bond."""
    features = build_liquidity_features(
        seeded_warehouse,
        asof=_ASOF,
        scope_type="cusip",
        scope_id=_CUSIP_A,
        lookback_days=30,
    )
    assert not features.empty
    out = score_liquidity_stress(
        features,
        scope_type="cusip",
        scope_id=_CUSIP_A,
        asof=_ASOF,
        model_run_id="run-cusip",
    )
    assert out.scope_type == "cusip"
    assert out.scope_id == _CUSIP_A
    assert 0.0 <= out.liquidity_index <= 100.0


# ---------------------------------------------------------------------------
# Warehouse round-trip
# ---------------------------------------------------------------------------


def test_write_and_read_per_scope_roundtrip(tmp_path: Path) -> None:
    """A market-scope row and a cusip-scope row coexist; each is fetched by filter."""
    wh = Warehouse(str(tmp_path / "fi-rt.duckdb"))
    try:
        _seed_warehouse(wh)
        market_features = build_liquidity_features(
            wh,
            asof=_ASOF,
            scope_type="market",
            scope_id="ALL",
            lookback_days=30,
        )
        cusip_features = build_liquidity_features(
            wh,
            asof=_ASOF,
            scope_type="cusip",
            scope_id=_CUSIP_A,
            lookback_days=30,
        )
        market_out = score_liquidity_stress(
            market_features,
            scope_type="market",
            scope_id="ALL",
            asof=_ASOF,
            model_run_id="rt-market",
        )
        cusip_out = score_liquidity_stress(
            cusip_features,
            scope_type="cusip",
            scope_id=_CUSIP_A,
            asof=_ASOF,
            model_run_id="rt-cusip",
        )
        assert write_liquidity_stress_score(wh, market_out) == 1
        assert write_liquidity_stress_score(wh, cusip_out) == 1

        latest_any = latest_liquidity_stress_score(wh)
        assert latest_any is not None

        latest_market = latest_liquidity_stress_score(
            wh, scope_type="market", scope_id="ALL"
        )
        assert latest_market is not None
        assert latest_market.scope_type == "market"
        assert latest_market.model_run_id == "rt-market"

        latest_cusip = latest_liquidity_stress_score(
            wh, scope_type="cusip", scope_id=_CUSIP_A
        )
        assert latest_cusip is not None
        assert latest_cusip.scope_type == "cusip"
        assert latest_cusip.scope_id == _CUSIP_A

        # No row for the missing scope.
        assert latest_liquidity_stress_score(wh, scope_type="cusip", scope_id="DOESNT-EXIST") is None

        recent = list_recent_liquidity_stress_scores(wh, limit=10)
        assert len(recent) == 2
        recent_market = list_recent_liquidity_stress_scores(wh, scope_type="market", limit=10)
        assert len(recent_market) == 1
        assert recent_market[0].scope_type == "market"
    finally:
        wh.close()
