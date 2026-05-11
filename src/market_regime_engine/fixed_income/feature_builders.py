# SPDX-License-Identifier: Apache-2.0
"""Fixed-Income feature builders — PR-1 skeleton.

Full implementations land in later PRs:

- **PR-3** (credit spread regime): :func:`build_credit_features`
  consumes Treasury/swap/rating/sector curves, OAS/Z-spread,
  CDS/CDX, MOVE/VIX, ETF prem/disc, and macro surprise proxies per
  ``MRE_FIXED_INCOME_INSTRUCTIONS.md §6.1``.
- **PR-4** (liquidity stress): :func:`build_liquidity_features`
  consumes TRACE / RFQ / dealer-quote / order-book proxies per
  INSTRUCTIONS.md §6.2.
- **PR-5** (execution confidence): :func:`build_execution_features`
  composes the order body with the prevailing regime/liquidity
  indices plus order-book / time-of-day / historical-performance
  features per INSTRUCTIONS.md §6.3.

All FI feature builders must:

1. Pass every produced row through :func:`pit_guard.assert_pit_safe`
   so a stale or future-dated feature trips ``PitViolationError``
   rather than silently feeding the scorer.
2. Default the per-column NaN policy to ``NAN_FAILS_PIT_AUDIT``
   (per PR-3 plan §3) so missing inputs trigger ``release_gate=False``
   instead of fake "Normal" regime scores.
3. Reject naive datetimes at the FI boundary (PR-3 ``timestamps.py``).

The PR-1 stubs raise :class:`NotImplementedError` so any caller that
imports the API surface today gets a clear pointer to the PR that
fills the gap.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_credit_features(
    *,
    asof: pd.Timestamp,
    curves: pd.DataFrame | None = None,
    spreads: pd.DataFrame | None = None,
    cds: pd.DataFrame | None = None,
    volatility: pd.DataFrame | None = None,
    macro_surprises: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build PR-3 credit-spread-regime features (skeleton).

    Will return a wide frame keyed by ``feature_name`` carrying
    Treasury/swap curve level/slope/curvature, OAS / Z-spread /
    G-spread, rating curve spreads, sector curve spreads, CDS term
    structure, CDS-bond basis, CDX IG/HY, MOVE / VIX / credit-vol,
    ETF prem/disc, and macro-surprise proxies. PR-1 is intentionally
    not-yet-implemented; the typed signature is fixed from day 1.
    """
    raise NotImplementedError("build_credit_features lands in PR-3 (credit spread regime model)")


def build_liquidity_features(
    *,
    asof: pd.Timestamp,
    scope_type: str,
    scope_id: str,
    trace: pd.DataFrame | None = None,
    rfq: pd.DataFrame | None = None,
    quotes: pd.DataFrame | None = None,
    bond_reference: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build PR-4 liquidity-stress features (skeleton).

    Will return per-scope features: bid-ask, trade-count velocity,
    time since last trade, volume / trailing ADV, RFQ dealers
    requested, quotes received, quote dispersion, Amihud illiquidity,
    dealer response count, axe freshness. Four scope levels:
    ``market`` / ``sector`` / ``rating`` / ``cusip``. PR-1 stub.
    """
    raise NotImplementedError("build_liquidity_features lands in PR-4 (liquidity stress model)")


def build_execution_features(
    *,
    asof: pd.Timestamp,
    request: Any,
    bond_reference: pd.DataFrame | None = None,
    regime_index: dict[str, Any] | None = None,
    liquidity_index: dict[str, Any] | None = None,
    market_state: pd.DataFrame | None = None,
    rfq_stats: pd.DataFrame | None = None,
    historical_performance: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build PR-5 execution-confidence features (skeleton).

    Will combine the order body (``ExecutionConfidenceRequest``) with
    the prevailing regime/liquidity indices, top-of-book / depth /
    intraday-vol / recent-volume, RFQ stats, time-of-day, and the
    historical-performance prior. PR-1 stub.
    """
    raise NotImplementedError(
        "build_execution_features lands in PR-5 (execution confidence model)"
    )


__all__ = [
    "build_credit_features",
    "build_execution_features",
    "build_liquidity_features",
]
