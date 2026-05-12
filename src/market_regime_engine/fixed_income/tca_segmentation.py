# SPDX-License-Identifier: Apache-2.0
"""PR-6 TCA segmentation — tag + aggregate.

Per ``MRE_FIXED_INCOME_AGENT.md §"PR 6 — TCA segmentation"`` and
``MRE_FIXED_INCOME_INSTRUCTIONS.md §6.4``: tag every trade with the
prevailing regime / liquidity / execution-confidence context, aggregate
TCA metrics over the documented segmentation dimensions, and persist
the result as one row per ``(dimension-combo) × metric`` to the
``tca_regime_segments`` warehouse table.

Public surface
--------------

- :func:`tag_trade_with_regime_context` — single-trade tagging.
- :func:`aggregate_tca_by_regime` — grouped aggregation with optional
  soft-weighting and the deep-research §4 metric catalog.
- :func:`compute_tca_metrics_for_outcome` — per-trade TCA metrics from
  the deterministic baseline; Decimal-precision throughout.
- :func:`compute_execution_success_label` — PR-10 label-construction
  guard with strict ``observed_at > decision_timestamp`` enforcement.
- :func:`write_tca_regime_segment` / :func:`latest_tca_regime_segments`
  — warehouse plumbing.
- :func:`materialize_tca_segments_for_day` — end-of-day materialisation
  driver.

Governance discipline
---------------------

- *Decimal precision*: TCA metrics accumulate in :class:`decimal.Decimal`
  via :mod:`market_regime_engine.fixed_income.bps_precision`; conversion
  to ``float`` happens only at the report boundary
  (:func:`decimal_to_float_for_report`).
- *PIT safety*: every ``latest_*`` read uses ``asof <= trade.timestamp``
  semantics so the regime/liquidity/exec-confidence context never leaks
  the future. The :class:`PitViolationError` rail fires on a post-decision
  context row.
- *Outcome lag*: :func:`compute_tca_metrics_for_outcome` and
  :func:`compute_execution_success_label` enforce the strict inequality
  ``outcome.observed_at > request.timestamp`` (PR-10 in REVIEW.md §3.6;
  ``write_execution_outcome`` already enforces it at write time).
- *Markout windows*: ``post_trade_markout_1d_bps`` / ``post_trade_markout_5d_bps``
  use the SIFMA bond trading-day calendar from PR-3; when the window
  has not yet closed the metric returns ``None`` so the segment row
  records ``sample_count=0`` for that metric.
- *NaN propagation*: :func:`aggregate_tca_by_regime` drops NaN rows at
  the aggregation boundary and emits the ``fi_tca_dropped_rows_total``
  counter labelled by ``metric`` / ``regime_label`` / ``liquidity_label``
  per REVIEW.md §3.6 PR-11.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from market_regime_engine.fixed_income.bps_precision import (
    bps_arithmetic_mean,
    decimal_to_float_for_report,
    to_bps,
    to_decimal,
)
from market_regime_engine.fixed_income.calendars import (
    TradingCalendar,
    next_trading_day,
)
from market_regime_engine.fixed_income.credit_spread_regime import (
    classify_with_hysteresis as classify_credit_with_hysteresis,
    latest_credit_regime_score,
)
from market_regime_engine.fixed_income.execution_confidence import (
    latest_execution_confidence_prediction,
)
from market_regime_engine.fixed_income.liquidity_stress import (
    latest_liquidity_stress_score,
)
from market_regime_engine.fixed_income.pit_guard import PitViolationError, assert_pit_safe
from market_regime_engine.fixed_income.schemas import (
    ExecutionConfidenceRequest,
    ExecutionConfidenceResponse,
    RegimeLabel,
    TaggedTrade,
    TcaRegimeSegment,
    TradeRecord,
    regime_label_from_score,
)
from market_regime_engine.fixed_income.tca_outcome_lag import (
    EXECUTION_SUCCESS_DEFAULT_THRESHOLD_BPS,
    assert_outcome_after_decision,
    compute_execution_success_label,
)
from market_regime_engine.fixed_income.timestamps import iso8601_z
from market_regime_engine.observability import incr

log = logging.getLogger(__name__)


__all__ = [
    "DIMENSION_COLUMNS",
    "DROPPED_ROWS_COUNTER",
    "EXECUTION_SUCCESS_DEFAULT_THRESHOLD_BPS",
    "TCA_METRICS",
    "aggregate_tca_by_regime",
    "compute_execution_success_label",
    "compute_tca_metrics_for_outcome",
    "latest_tca_regime_segments",
    "materialize_tca_segments_for_day",
    "tag_trade_with_regime_context",
    "write_tca_regime_segment",
]


# ---------------------------------------------------------------------------
# constants / catalogues
# ---------------------------------------------------------------------------


TCA_METRICS: tuple[str, ...] = (
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
)
"""Canonical TCA metric list (AGENT.md §"PR 6" + INSTRUCTIONS.md §6.4)."""


DIMENSION_COLUMNS: tuple[str, ...] = (
    "regime_label",
    "liquidity_label",
    "execution_confidence_bucket",
    "protocol",
    "side",
    "sector",
    "rating",
    "maturity_bucket",
    "notional_bucket",
)
"""Supported segmentation dimensions (INSTRUCTIONS.md §6.4)."""


DROPPED_ROWS_COUNTER: str = "fi_tca_dropped_rows_total"
"""Observability counter for NaN-dropped TCA rows.

Per ``REVIEW.md §3.6 PR-11``: aggregating with NaN poisons the mean
(``NaN + 1 = NaN`` → bucket reports NaN even though most trades were
clean). :func:`aggregate_tca_by_regime` drops at the aggregation
boundary and emits this counter labelled by ``metric`` + the active
grouping dimensions so dashboards can correlate drops with regime /
liquidity buckets.
"""


def _register_counter() -> None:
    """Idempotent registration of :data:`DROPPED_ROWS_COUNTER`.

    The in-process registry registers on first ``incr``; the no-op
    ``+0`` here pre-creates the counter family so a Prometheus scrape
    immediately after module import returns the family with no samples
    rather than 404. Re-imports are safe — the underlying
    ``defaultdict(float)`` is idempotent on the (name, labels) key.

    v1.5 PR-8 (Tier-2 fix C-AUTO-2): routes through the module-level
    :func:`incr` so the registration also mirrors to the OTel meter
    when ``configure_otel(enabled=True)`` was called at boot.
    """
    incr(DROPPED_ROWS_COUNTER, 0.0)


_register_counter()


# ---------------------------------------------------------------------------
# helpers — bucketing
# ---------------------------------------------------------------------------


_MATURITY_BUCKETS: tuple[tuple[float, str], ...] = (
    (2.0, "0-2y"),
    (5.0, "2-5y"),
    (10.0, "5-10y"),
    (float("inf"), "10y+"),
)


_NOTIONAL_BUCKETS: tuple[tuple[float, str], ...] = (
    (1_000_000.0, "<1M"),
    (5_000_000.0, "1-5M"),
    (25_000_000.0, "5-25M"),
    (float("inf"), "25M+"),
)


# v1.5 PR-6: execution-confidence buckets follow the AGENT.md scoring
# bands. Boundary 0.60 → "low"/"medium" so the bucket aligns with the
# AUTO_X_CAUTION decision-rule threshold; 0.80 → "high" so AUTO_X_ALLOWED
# trades land in the top bucket.
_EXECUTION_CONFIDENCE_BUCKETS: tuple[tuple[float, str], ...] = (
    (0.60, "low"),
    (0.80, "medium"),
    (1.01, "high"),  # 1.01 upper bound so confidence_score=1.0 lands in "high"
)


def _maturity_bucket_for(maturity_years: float | None) -> str:
    if maturity_years is None or not math.isfinite(float(maturity_years)):
        return "unknown"
    y = float(maturity_years)
    for upper, label in _MATURITY_BUCKETS:
        if y < upper:
            return label
    return _MATURITY_BUCKETS[-1][1]


def _notional_bucket_for(notional: float) -> str:
    n = float(notional)
    for upper, label in _NOTIONAL_BUCKETS:
        if n < upper:
            return label
    return _NOTIONAL_BUCKETS[-1][1]


def _execution_confidence_bucket_for(confidence_score: float | None) -> str:
    if confidence_score is None or not math.isfinite(float(confidence_score)):
        return "unavailable"
    s = float(confidence_score)
    for upper, label in _EXECUTION_CONFIDENCE_BUCKETS:
        if s < upper:
            return label
    return _EXECUTION_CONFIDENCE_BUCKETS[-1][1]


def _sector_bucket_for(sector: str | None) -> str:
    if sector is None or not str(sector).strip():
        return "unknown"
    return str(sector).strip().lower()


def _rating_bucket_for(rating: str | None) -> str:
    """IG / HY / Unrated bucket from a raw rating string.

    Mirrors the execution_confidence._rating_class helper but exposes
    "unrated" rather than ``None`` for the segment-key stability.
    """
    if rating is None:
        return "unrated"
    raw = str(rating).strip().upper()
    if not raw:
        return "unrated"
    if raw in {"IG", "HY"}:
        return raw
    if raw[0].isdigit():
        try:
            num = float(raw)
        except ValueError:
            return "unrated"
        return "IG" if num <= 10.0 else "HY"
    core = raw.replace("+", "").replace("-", "")
    if core.startswith(("AAA", "AA", "A")) and not core.startswith("BBB"):
        return "IG"
    if core.startswith("BBB"):
        return "IG"
    if core.startswith(("BB", "B", "CCC", "CC", "C", "D")):
        return "HY"
    return "unrated"


# ---------------------------------------------------------------------------
# helpers — timestamp / soft-weight math
# ---------------------------------------------------------------------------


def _coerce_utc(ts: pd.Timestamp | str) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        return out.tz_localize("UTC")
    return out.tz_convert("UTC")


_REGIME_BOUNDARIES: tuple[float, ...] = (10.0, 30.0, 50.0, 70.0, 90.0)
"""Bucket centers for the credit-regime score (midpoints of [0,20), [20,40),
[40,60), [60,80), [80,100]). Used by the triangular soft-weighting.
"""

_REGIME_LABELS_ORDERED: tuple[RegimeLabel, ...] = (
    RegimeLabel.RISK_ON_COMPRESSION,
    RegimeLabel.NORMAL_LIQUIDITY,
    RegimeLabel.WATCH_TRANSITION,
    RegimeLabel.RISK_OFF_HIGH_RISK_AVERSION,
    RegimeLabel.CRISIS_SEVERE_DISLOCATION,
)


def _triangular_soft_weights(regime_score: float) -> dict[str, float]:
    """Triangular weighting of a 0-100 score across the 5 regime labels.

    The score is mapped to two adjacent bucket centers and split
    linearly: ``w_left = (center_right - score) / 20``, ``w_right = 1 -
    w_left``. Scores below the first center or above the last center
    saturate at that endpoint (so a score of 0 → 100% RISK_ON;
    score 100 → 100% CRISIS).

    Sums to 1.0 within Decimal-level precision; the returned dict only
    includes labels with strictly-positive weight so downstream code can
    iterate without filtering.
    """
    score = max(0.0, min(100.0, float(regime_score)))
    centers = _REGIME_BOUNDARIES
    labels = _REGIME_LABELS_ORDERED
    if score <= centers[0]:
        return {labels[0].label: 1.0}
    if score >= centers[-1]:
        return {labels[-1].label: 1.0}
    for i in range(len(centers) - 1):
        left_c, right_c = centers[i], centers[i + 1]
        if left_c <= score <= right_c:
            span = right_c - left_c
            w_right = (score - left_c) / span
            w_left = 1.0 - w_right
            out: dict[str, float] = {}
            if w_left > 0:
                out[labels[i].label] = float(w_left)
            if w_right > 0:
                out[labels[i + 1].label] = float(w_right)
            return out
    # Defensive fallback (should not reach here given the saturation guards).
    return {regime_label_from_score(score).label: 1.0}


# ---------------------------------------------------------------------------
# A.1 — tag_trade_with_regime_context
# ---------------------------------------------------------------------------


def tag_trade_with_regime_context(
    trade: TradeRecord,
    *,
    warehouse: Any,
    use_hysteresis: bool = True,
    tolerance: pd.Timedelta = pd.Timedelta("5min"),  # noqa: B008  module-level constant
) -> TaggedTrade:
    """Attach prevailing regime / liquidity / execution-confidence context to a trade.

    Reads:

    - Latest ``credit_regime_score`` (asof ``trade.timestamp``).
    - Latest ``liquidity_stress_score`` (cusip scope, fallback to market scope).
    - Latest ``execution_confidence_prediction`` for the same cusip
      (or ``None`` if no prediction was logged).

    PIT-safe: every read is bounded ``asof <= trade.timestamp``. The
    :class:`PitViolationError` rail fires when any context row's
    timestamp exceeds ``trade.timestamp``.

    ``use_hysteresis`` re-classifies the regime score under the
    asymmetric hysteresis bands from PR-4 (using the *latest* regime
    label as the prev label) so the tagged regime is "sticky" near a
    bucket boundary. ``False`` falls back to the sharp-bucket label.

    ``tolerance`` is documented for future merge-asof style joins per
    REVIEW.md §3.4 Q-9; the PR-6 deterministic baseline pulls
    ``latest_*`` snapshots and does not need a tolerance window yet.

    Returns a :class:`TaggedTrade` with hard label, soft-weight dict,
    liquidity index, execution-confidence bucket, and the four bucket
    fields (sector / rating / maturity / notional).
    """
    trade_ts = _coerce_utc(trade.timestamp)

    regime = latest_credit_regime_score(warehouse)
    liquidity = latest_liquidity_stress_score(
        warehouse, scope_type="cusip", scope_id=trade.cusip
    )
    if liquidity is None:
        liquidity = latest_liquidity_stress_score(warehouse)

    if regime is not None:
        assert_pit_safe(
            feature_timestamp=regime.timestamp,
            decision_timestamp=trade_ts,
            label="credit_regime_tag",
        )
    if liquidity is not None:
        assert_pit_safe(
            feature_timestamp=liquidity.timestamp,
            decision_timestamp=trade_ts,
            label="liquidity_stress_tag",
        )

    regime_score = float(regime.regime_score) if regime is not None else 50.0
    if regime is None:
        regime_label_str = "unknown"
        soft_weights: dict[str, float] = {}
    else:
        if use_hysteresis:
            # Use the persisted label as prev_label so hysteresis is
            # applied around the score; the hysteresis function does
            # the right thing when prev_label is None (sharp fallback).
            try:
                prev_label = next(
                    lbl for lbl in RegimeLabel if lbl.label == regime.regime_label
                )
            except StopIteration:
                prev_label = None
            regime_label_str = classify_credit_with_hysteresis(
                regime_score, prev_label
            ).label
        else:
            regime_label_str = regime_label_from_score(regime_score).label
        soft_weights = _triangular_soft_weights(regime_score)

    liquidity_label_str = (
        liquidity.liquidity_label if liquidity is not None else "unknown"
    )
    liquidity_index = (
        float(liquidity.liquidity_index) if liquidity is not None else 50.0
    )

    exec_pred = latest_execution_confidence_prediction(warehouse, cusip=trade.cusip)
    if exec_pred is not None:
        try:
            pred_ts = _coerce_utc(exec_pred.timestamp)
        except Exception:
            pred_ts = None
        if pred_ts is not None and pred_ts > trade_ts:
            # Per PR-6 §A.1 PIT contract: a stored prediction that
            # post-dates the trade is not a valid tag-time context.
            # Treat as "unavailable" rather than raising — the trade
            # was made without that prediction.
            exec_pred = None

    execution_confidence_score = (
        float(exec_pred.confidence_score) if exec_pred is not None else None
    )
    execution_confidence_bucket = _execution_confidence_bucket_for(
        execution_confidence_score
    )

    sector_bucket = _sector_bucket_for(trade.sector)
    rating_bucket = _rating_bucket_for(trade.rating)
    maturity_bucket = _maturity_bucket_for(trade.maturity_years)
    notional_bucket = _notional_bucket_for(trade.notional)

    metadata: dict[str, Any] = {
        "use_hysteresis": bool(use_hysteresis),
        "tolerance_seconds": float(tolerance.total_seconds()),
        "regime_score_source_timestamp": (
            regime.timestamp if regime is not None else None
        ),
        "liquidity_index_source_timestamp": (
            liquidity.timestamp if liquidity is not None else None
        ),
        "liquidity_scope_type": (
            liquidity.scope_type if liquidity is not None else None
        ),
        "liquidity_scope_id": (
            liquidity.scope_id if liquidity is not None else None
        ),
        "execution_confidence_source_request_id": (
            exec_pred.metadata.get("request_id")
            if exec_pred is not None and isinstance(exec_pred.metadata, dict)
            else None
        ),
        "execution_confidence_release_gate": (
            bool(exec_pred.release_gate) if exec_pred is not None else None
        ),
        "regime_release_gate": (bool(regime.release_gate) if regime is not None else None),
        "liquidity_release_gate": (
            bool(liquidity.release_gate) if liquidity is not None else None
        ),
    }

    return TaggedTrade(
        trade=trade,
        regime_label=regime_label_str,
        regime_score=regime_score,
        regime_soft_weights=soft_weights,
        liquidity_label=liquidity_label_str,
        liquidity_index=liquidity_index,
        execution_confidence_bucket=execution_confidence_bucket,
        execution_confidence_score=execution_confidence_score,
        sector_bucket=sector_bucket,
        rating_bucket=rating_bucket,
        maturity_bucket=maturity_bucket,
        notional_bucket=notional_bucket,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# A.1 — compute_tca_metrics_for_outcome
# ---------------------------------------------------------------------------


def _side_sign(side: str) -> int:
    """+1 for buy, -1 for sell — implementation-shortfall sign convention.

    The arrival-cost / market-impact bps are computed as
    ``side_sign * (execution - benchmark) / benchmark * 10_000`` so a
    *positive* number is a *cost* (higher exec than benchmark on a buy,
    lower exec than benchmark on a sell).
    """
    s = (side or "").strip().lower()
    if s == "buy":
        return 1
    if s == "sell":
        return -1
    raise ValueError(f"side must be 'buy' or 'sell'; got {side!r}")


def _markout_window_observable(
    decision_ts: pd.Timestamp,
    *,
    days_forward: int,
    asof_now: pd.Timestamp | None,
    calendar: TradingCalendar = TradingCalendar.SIFMA_BOND,
) -> tuple[bool, pd.Timestamp]:
    """Return ``(observable, window_close_ts)`` for a markout window.

    The window close is computed as ``days_forward`` *trading days*
    forward from ``decision_ts`` on the SIFMA bond calendar. The window
    is observable when ``asof_now >= window_close_ts``. When
    ``asof_now`` is ``None`` the helper conservatively assumes the
    window is observable (used by historical re-aggregation where the
    caller knows the window already closed).
    """
    ts = decision_ts
    for _ in range(int(days_forward)):
        ts = _coerce_utc(next_trading_day(ts, calendar))
    if asof_now is None:
        return True, ts
    asof_utc = _coerce_utc(asof_now)
    return asof_utc >= ts, ts


def compute_tca_metrics_for_outcome(
    request: ExecutionConfidenceRequest,
    response: ExecutionConfidenceResponse,
    outcome: Mapping[str, Any],
    *,
    warehouse: Any,
    asof_now: pd.Timestamp | None = None,
) -> dict[str, float | None]:
    """Compute per-trade TCA metrics from a request + response + observed outcome.

    Decimal-precision arithmetic throughout; only the public dict
    returns ``float`` (or ``None`` for unobservable metrics). The
    governance rails:

    - Strict inequality ``outcome.observed_at > request.timestamp`` —
      raises :class:`PitViolationError` on violation (PR-10 / PR-6 §C.1).
    - Markout-window observability — when the 1d/5d trading-day window
      has not yet closed, the metric returns ``None`` (PR-6 §C.2).

    Parameters
    ----------
    request:
        The decision-time ``ExecutionConfidenceRequest``;
        ``request.timestamp`` is the decision timestamp.
    response:
        The ``ExecutionConfidenceResponse`` from the scorer; carries
        ``expected_slippage_bps`` used for the ``quote_quality`` proxy.
    outcome:
        Observed-outcome mapping. Required key ``observed_at``;
        optional ``arrival_price``, ``execution_price``,
        ``vwap_price``, ``mid_price_at_arrival``, ``best_bid_at_arrival``,
        ``best_ask_at_arrival``, ``time_to_fill_seconds``,
        ``dealer_response_count``, ``markout_price_1d``,
        ``markout_price_5d``.
    warehouse:
        The v1.5 Warehouse — currently unused inside the deterministic
        metric set but threaded through so future per-cusip markout
        joins can pull TRACE prints without changing the signature.
    asof_now:
        Optional "now" timestamp for the markout-window observability
        check; defaults to ``pd.Timestamp.utcnow()``.

    Returns
    -------
    dict mapping every name in :data:`TCA_METRICS` to a ``float`` (or
    ``None`` when the metric cannot be observed for this trade).
    """
    del warehouse  # reserved for future per-cusip markout joins
    observed_at_raw = outcome.get("observed_at")
    if observed_at_raw is None:
        raise ValueError(
            "compute_tca_metrics_for_outcome: outcome must include 'observed_at'"
        )
    decision_ts, _ = assert_outcome_after_decision(
        decision_timestamp=request.timestamp,
        observed_at=observed_at_raw,
        label="compute_tca_metrics_for_outcome",
    )

    asof = asof_now if asof_now is not None else pd.Timestamp.now(tz="UTC")
    asof = _coerce_utc(asof)

    sign = _side_sign(request.side)
    execution = outcome.get("execution_price")
    arrival = outcome.get("arrival_price")
    vwap = outcome.get("vwap_price")
    mid = outcome.get("mid_price_at_arrival")
    best_bid = outcome.get("best_bid_at_arrival")
    best_ask = outcome.get("best_ask_at_arrival")

    results: dict[str, float | None] = {m: None for m in TCA_METRICS}

    # arrival_cost_bps = sign * (execution - arrival) / arrival * 10_000
    if execution is not None and arrival is not None:
        try:
            arrival_cost = to_bps(
                to_decimal(sign) * (to_decimal(execution) - to_decimal(arrival)),
                to_decimal(arrival),
            )
            results["arrival_cost_bps"] = decimal_to_float_for_report(arrival_cost)
        except ZeroDivisionError:
            results["arrival_cost_bps"] = None

    # vwap_slippage_bps = sign * (execution - vwap) / vwap * 10_000
    if execution is not None and vwap is not None:
        try:
            vwap_slip = to_bps(
                to_decimal(sign) * (to_decimal(execution) - to_decimal(vwap)),
                to_decimal(vwap),
            )
            results["vwap_slippage_bps"] = decimal_to_float_for_report(vwap_slip)
        except ZeroDivisionError:
            results["vwap_slippage_bps"] = None

    # price_improvement_bps: for buys, (best_ask - execution); for sells,
    # (execution - best_bid). Divided by mid (or by best_ask/best_bid if
    # mid is unavailable) and scaled to bps. Positive = improvement.
    if execution is not None and (best_bid is not None or best_ask is not None):
        try:
            if sign == 1 and best_ask is not None:
                ref = to_decimal(mid) if mid is not None else to_decimal(best_ask)
                pi = to_bps(to_decimal(best_ask) - to_decimal(execution), ref)
            elif sign == -1 and best_bid is not None:
                ref = to_decimal(mid) if mid is not None else to_decimal(best_bid)
                pi = to_bps(to_decimal(execution) - to_decimal(best_bid), ref)
            else:
                pi = None
            results["price_improvement_bps"] = (
                decimal_to_float_for_report(pi) if pi is not None else None
            )
        except ZeroDivisionError:
            results["price_improvement_bps"] = None

    # market_impact_bps = sign * (execution - mid) / mid * 10_000
    if execution is not None and mid is not None:
        try:
            mi = to_bps(
                to_decimal(sign) * (to_decimal(execution) - to_decimal(mid)),
                to_decimal(mid),
            )
            results["market_impact_bps"] = decimal_to_float_for_report(mi)
        except ZeroDivisionError:
            results["market_impact_bps"] = None

    if outcome.get("time_to_fill_seconds") is not None:
        results["time_to_fill_seconds"] = float(outcome["time_to_fill_seconds"])
    if outcome.get("dealer_response_count") is not None:
        results["dealer_response_count"] = float(outcome["dealer_response_count"])

    # quote_quality: per-trade ratio of dealer_response_count to the
    # response's expected_slippage_bps clipped at 1.0. A higher value
    # means the dealer pool delivered more competitive quotes than the
    # baseline expected. The metric is bounded in [0, 1] so it can be
    # cleanly averaged across regimes.
    if outcome.get("dealer_response_count") is not None and response.expected_slippage_bps is not None:
        try:
            slip = max(1.0, float(response.expected_slippage_bps))
            dealer_n = float(outcome["dealer_response_count"])
            # Normalise: dealer_n / (dealer_n + slip) is bounded in [0, 1).
            results["quote_quality"] = float(dealer_n / (dealer_n + slip))
        except (TypeError, ValueError, ZeroDivisionError):
            results["quote_quality"] = None

    # protocol_success: 1.0 when filled_quantity / notional >= 0.95,
    # else 0.0. ``None`` when the fill data is absent (e.g. an RFQ that
    # never responded).
    filled = outcome.get("filled_quantity")
    if filled is not None:
        try:
            n = float(request.notional)
            f = float(filled)
            if n > 0:
                results["protocol_success"] = float(1.0 if (f / n) >= 0.95 else 0.0)
        except (TypeError, ValueError, ZeroDivisionError):
            results["protocol_success"] = None

    # execution_success — strict-inequality guard already fired above.
    es = compute_execution_success_label(request, outcome)
    if es is not None:
        results["execution_success"] = float(1.0 if bool(es) else 0.0)

    # post-trade markouts — trading-day calendar window observability.
    markout_1d_observable, _ = _markout_window_observable(
        decision_ts, days_forward=1, asof_now=asof
    )
    markout_5d_observable, _ = _markout_window_observable(
        decision_ts, days_forward=5, asof_now=asof
    )
    if markout_1d_observable and outcome.get("markout_price_1d") is not None and execution is not None:
        try:
            mk = to_bps(
                to_decimal(sign)
                * (to_decimal(outcome["markout_price_1d"]) - to_decimal(execution)),
                to_decimal(execution),
            )
            results["post_trade_markout_1d_bps"] = decimal_to_float_for_report(mk)
        except ZeroDivisionError:
            results["post_trade_markout_1d_bps"] = None
    if markout_5d_observable and outcome.get("markout_price_5d") is not None and execution is not None:
        try:
            mk = to_bps(
                to_decimal(sign)
                * (to_decimal(outcome["markout_price_5d"]) - to_decimal(execution)),
                to_decimal(execution),
            )
            results["post_trade_markout_5d_bps"] = decimal_to_float_for_report(mk)
        except ZeroDivisionError:
            results["post_trade_markout_5d_bps"] = None

    return results


# ---------------------------------------------------------------------------
# A.1 — aggregate_tca_by_regime
# ---------------------------------------------------------------------------


def _is_metric_numeric(metric: str) -> bool:
    """Whether a metric averages as a number (vs counts as a sum)."""
    return metric in {
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


def _drop_nan_rows(
    trades: pd.DataFrame,
    *,
    metric: str,
    dimensions: Sequence[str] = (),
) -> tuple[pd.DataFrame, int]:
    """Drop rows where ``metric`` is NaN at the aggregation boundary.

    Returns ``(cleaned_frame, dropped_count)``. PR-6 task D /
    REVIEW.md §3.6 PR-11: aggregating with NaN poisons the mean. This
    helper drops at the boundary and emits :data:`DROPPED_ROWS_COUNTER`
    labelled by ``metric`` + the active grouping ``dimensions`` so
    dashboards can pivot drops by regime / liquidity bucket. The label
    values use ``"__all__"`` because at this point the drop is
    pre-grouping (we have not yet split by bucket); a future enhancement
    can split increments per group when the join can prove it is cheap.
    """
    if metric not in trades.columns:
        return trades.iloc[0:0], 0
    nan_mask = trades[metric].isna()
    n_dropped = int(nan_mask.sum())
    if n_dropped == 0:
        return trades, 0
    cleaned = trades.loc[~nan_mask].copy()
    label_kwargs: dict[str, str] = {"metric": metric}
    if "regime_label" in dimensions:
        label_kwargs["regime_label"] = "__all__"
    if "liquidity_label" in dimensions:
        label_kwargs["liquidity_label"] = "__all__"
    # v1.5 PR-8 (Tier-2 fix C-AUTO-2): route through the module-level
    # ``incr`` so this counter mirrors to OTel (observability.py:393-409)
    # alongside the legacy ``MetricsRegistry`` snapshot. Pre-fix this
    # call went through ``metrics().incr(...)`` which only writes to
    # the legacy ``_GLOBAL`` and silently bypasses the OTel meter.
    incr(DROPPED_ROWS_COUNTER, float(n_dropped), **label_kwargs)
    return cleaned, n_dropped


def _group_aggregate(
    trades: pd.DataFrame,
    *,
    dimensions: Sequence[str],
    metric: str,
    soft_weighting: bool,
) -> pd.DataFrame:
    """One row per dim-combo for the given metric.

    Decimal mean: pivot-then-reduce by accumulating each row's value
    in :class:`Decimal` via :func:`bps_arithmetic_mean`; weights come
    from ``regime_soft_weights`` when ``soft_weighting=True`` and
    ``"regime_label"`` is a dimension, else from constant 1.0 weights.
    """
    if trades.empty:
        return pd.DataFrame(
            columns=[*dimensions, "metric_value", "sample_count"]
        )
    groups: dict[tuple, list[tuple[float, float]]] = {}
    soft_weight_dicts = trades.get("regime_soft_weights")
    has_soft = soft_weighting and "regime_label" in dimensions and soft_weight_dicts is not None
    for _, row in trades.iterrows():
        value = row.get(metric)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        # Soft weighting: expand a single row into multiple (label, weight)
        # contributions; one per regime label with positive weight.
        if has_soft:
            sw = row.get("regime_soft_weights") or {}
            if not isinstance(sw, dict):
                sw = {}
            if not sw:
                # Fall back to hard label with weight 1.0.
                sw = {str(row.get("regime_label", "unknown")): 1.0}
            for label, weight in sw.items():
                key = tuple(
                    label if dim == "regime_label" else row.get(dim) for dim in dimensions
                )
                groups.setdefault(key, []).append((float(value), float(weight)))
        else:
            key = tuple(row.get(dim) for dim in dimensions)
            groups.setdefault(key, []).append((float(value), 1.0))
    rows: list[dict[str, Any]] = []
    for key, value_weight_pairs in groups.items():
        values = [v for v, _ in value_weight_pairs]
        weights = [w for _, w in value_weight_pairs]
        if not values:
            continue
        if all(w == 1.0 for w in weights):
            agg = bps_arithmetic_mean(values)
        else:
            agg = bps_arithmetic_mean(values, weights=weights)
        out: dict[str, Any] = dict(zip(dimensions, key, strict=True))
        out["metric_value"] = decimal_to_float_for_report(agg)
        # sample_count is the number of trade contributions to the bucket,
        # rounded down on soft-weighted aggregates so partial weights do
        # not inflate counts.
        out["sample_count"] = int(
            sum(weights) if not all(w == 1.0 for w in weights) else len(values)
        )
        rows.append(out)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[*dimensions, "metric_value", "sample_count"]
        )
    return df


def aggregate_tca_by_regime(
    trades: pd.DataFrame,
    *,
    dimensions: Sequence[str] = ("regime_label", "liquidity_label"),
    metrics_names: Sequence[str] = TCA_METRICS,
    soft_weighting: bool = False,
) -> pd.DataFrame:
    """Aggregate TCA metrics grouped by the given dimensions.

    Parameters
    ----------
    trades:
        Long-form trades frame. Columns required: every name in
        ``dimensions`` plus every name in ``metrics_names``. Optional
        ``regime_soft_weights`` column (dict[label, weight]) — populated
        from :class:`TaggedTrade` for soft weighting.
    dimensions:
        Subset of :data:`DIMENSION_COLUMNS`. Default is
        ``("regime_label", "liquidity_label")`` per the AGENT.md
        catalogue.
    metrics_names:
        Subset of :data:`TCA_METRICS`. The returned frame has one row
        per ``(dim-combo, metric)``.
    soft_weighting:
        When ``True`` and ``"regime_label"`` is in ``dimensions``,
        :class:`TaggedTrade.regime_soft_weights` weighs each trade
        across regime labels (one trade contributes to multiple labels
        fractionally). Defaults to ``False`` (hard label grouping).

    Returns a long-form DataFrame with columns ``dimensions``,
    ``metric_name``, ``metric_value``, ``sample_count``.
    """
    if trades is None or len(trades) == 0:
        out_cols = [*dimensions, "metric_name", "metric_value", "sample_count"]
        return pd.DataFrame(columns=out_cols)

    invalid_dims = [d for d in dimensions if d not in DIMENSION_COLUMNS]
    if invalid_dims:
        raise ValueError(
            f"aggregate_tca_by_regime: invalid dimensions {invalid_dims!r}; "
            f"valid set is {DIMENSION_COLUMNS!r}"
        )
    invalid_metrics = [m for m in metrics_names if m not in TCA_METRICS]
    if invalid_metrics:
        raise ValueError(
            f"aggregate_tca_by_regime: invalid metrics {invalid_metrics!r}; "
            f"valid set is {TCA_METRICS!r}"
        )

    parts: list[pd.DataFrame] = []
    for metric in metrics_names:
        cleaned, _ = _drop_nan_rows(
            trades, metric=metric, dimensions=dimensions
        )
        if cleaned.empty:
            continue
        agg = _group_aggregate(
            cleaned,
            dimensions=dimensions,
            metric=metric,
            soft_weighting=soft_weighting,
        )
        if agg.empty:
            continue
        agg["metric_name"] = metric
        parts.append(agg)
    if not parts:
        out_cols = [*dimensions, "metric_name", "metric_value", "sample_count"]
        return pd.DataFrame(columns=out_cols)
    out = pd.concat(parts, ignore_index=True)
    final_cols = [*dimensions, "metric_name", "metric_value", "sample_count"]
    return out.reindex(columns=final_cols)


# ---------------------------------------------------------------------------
# A.1 — write_tca_regime_segment / latest_tca_regime_segments
# ---------------------------------------------------------------------------


_DIMENSION_SENTINEL: str = "__all__"
"""Sentinel string used for ``None`` dimension values when persisting.

The warehouse PK on ``tca_regime_segments`` requires every dimension
column to be ``NOT NULL`` (each is part of the composite PK). When a
segmentation aggregates over a dimension (e.g. ``("regime_label",
"liquidity_label")`` does not group by ``protocol``), the missing
dimensions are persisted as ``"__all__"`` so the composite PK remains
stable across runs that aggregate over different dimension combinations.
The :func:`latest_tca_regime_segments` reader translates back to
``None`` when the sentinel is present.
"""


def _segment_to_row(segment: TcaRegimeSegment) -> dict[str, Any]:
    """Convert a :class:`TcaRegimeSegment` to a warehouse row dict."""
    return {
        "model_run_id": str(segment.model_run_id),
        "timestamp": iso8601_z(_coerce_utc(segment.timestamp)),
        "regime_label": str(segment.regime_label) if segment.regime_label is not None else _DIMENSION_SENTINEL,
        "liquidity_label": str(segment.liquidity_label) if segment.liquidity_label is not None else _DIMENSION_SENTINEL,
        "execution_confidence_bucket": (
            str(segment.execution_confidence_bucket)
            if segment.execution_confidence_bucket is not None
            else _DIMENSION_SENTINEL
        ),
        "protocol": str(segment.protocol) if segment.protocol is not None else _DIMENSION_SENTINEL,
        "side": str(segment.side) if segment.side is not None else _DIMENSION_SENTINEL,
        "sector": str(segment.sector) if segment.sector is not None else _DIMENSION_SENTINEL,
        "rating": str(segment.rating) if segment.rating is not None else _DIMENSION_SENTINEL,
        "maturity_bucket": (
            str(segment.maturity_bucket) if segment.maturity_bucket is not None else _DIMENSION_SENTINEL
        ),
        "notional_bucket": (
            str(segment.notional_bucket) if segment.notional_bucket is not None else _DIMENSION_SENTINEL
        ),
        "metric_name": str(segment.metric_name),
        "metric_value": float(segment.metric_value),
        "sample_count": int(segment.sample_count),
        "metadata_json": str(segment.metadata_json),
    }


def _row_to_segment(row: pd.Series) -> TcaRegimeSegment:
    def _read(col: str) -> str | None:
        value = row.get(col)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        s = str(value)
        return None if s == _DIMENSION_SENTINEL else s

    ts_raw = row["timestamp"]
    ts = pd.Timestamp(ts_raw)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return TcaRegimeSegment(
        timestamp=ts,
        regime_label=_read("regime_label"),
        liquidity_label=_read("liquidity_label"),
        execution_confidence_bucket=_read("execution_confidence_bucket"),
        protocol=_read("protocol"),
        side=_read("side"),
        sector=_read("sector"),
        rating=_read("rating"),
        maturity_bucket=_read("maturity_bucket"),
        notional_bucket=_read("notional_bucket"),
        metric_name=str(row["metric_name"]),
        metric_value=float(row["metric_value"]),
        sample_count=int(row["sample_count"]),
        model_run_id=str(row["model_run_id"]),
        metadata_json=(
            str(row.get("metadata_json")) if row.get("metadata_json") is not None else "{}"
        ),
    )


def write_tca_regime_segment(warehouse: Any, segment: TcaRegimeSegment) -> int:
    """Persist a :class:`TcaRegimeSegment` row.

    Returns the number of rows written (always 1 on success). The
    warehouse uses an INSERT-OR-REPLACE policy keyed by the composite
    PK, so re-running a materialisation on the same day idempotently
    overwrites the prior segment row.
    """
    row = _segment_to_row(segment)
    return int(warehouse.write_tca_regime_segment(pd.DataFrame([row])))


def latest_tca_regime_segments(
    warehouse: Any,
    *,
    dimensions: Sequence[str] | None = None,
    limit: int = 100,
) -> list[TcaRegimeSegment]:
    """Read the most recent ``tca_regime_segments`` rows.

    Optionally filtered by ``dimensions``: a row qualifies when every
    listed dimension column is non-sentinel for that row. The default
    (``None``) returns the most recent ``limit`` rows across every
    segment regardless of which dimensions are grouped.
    """
    if limit <= 0:
        return []
    df = warehouse.read_tca_regime_segments()
    if df is None or df.empty:
        return []
    if dimensions:
        for dim in dimensions:
            if dim not in df.columns:
                continue
            df = df.loc[df[dim].astype(str) != _DIMENSION_SENTINEL]
            if df.empty:
                return []
    df = df.sort_values("timestamp", ascending=False).head(int(limit))
    return [_row_to_segment(r) for _, r in df.iterrows()]


# ---------------------------------------------------------------------------
# A.1 — materialize_tca_segments_for_day
# ---------------------------------------------------------------------------


def _dimension_combinations() -> tuple[tuple[str, ...], ...]:
    """Canonical combinations supported by :func:`materialize_tca_segments_for_day`.

    Keeps the materialisation deterministic: every run writes the same
    set of dim-combos so the warehouse table is comparable across runs.
    """
    return (
        ("regime_label", "liquidity_label"),
        ("regime_label",),
        ("liquidity_label",),
        ("regime_label", "liquidity_label", "execution_confidence_bucket"),
        ("regime_label", "liquidity_label", "protocol"),
        ("regime_label", "liquidity_label", "side"),
        ("regime_label", "sector"),
        ("regime_label", "rating"),
        ("regime_label", "maturity_bucket"),
        ("regime_label", "notional_bucket"),
    )


def _build_trades_frame_from_outcomes(
    warehouse: Any,
    *,
    date: pd.Timestamp,
) -> pd.DataFrame:
    """Build a long-form trades frame for the given date.

    Workflow:

    1. Read ``execution_outcomes`` for the date.
    2. Join the matching ``execution_confidence_predictions`` row by
       ``request_id`` for the decision-time request + response context.
    3. Tag each trade with regime / liquidity / execution-confidence
       context via :func:`tag_trade_with_regime_context`.
    4. Compute per-trade TCA metrics via
       :func:`compute_tca_metrics_for_outcome`.

    Each trade contributes one row with all dimension labels and metric
    columns populated. Trades that fail PIT (post-decision outcome
    timestamp) are silently dropped from the materialisation
    (``write_execution_outcome`` already prevented their entry but a
    historical dump may carry pre-strict-check rows).
    """
    outcomes = warehouse.read_execution_outcomes()
    if outcomes is None or outcomes.empty:
        return pd.DataFrame()
    outcomes = outcomes.copy()
    outcomes["observed_at_ts"] = pd.to_datetime(
        outcomes["observed_at"], utc=True, errors="coerce"
    )
    outcomes["decision_ts"] = pd.to_datetime(
        outcomes["decision_timestamp"], utc=True, errors="coerce"
    )
    day = _coerce_utc(date).normalize()
    day_end = day + pd.Timedelta(days=1)
    outcomes = outcomes.loc[
        (outcomes["decision_ts"] >= day) & (outcomes["decision_ts"] < day_end)
    ]
    if outcomes.empty:
        return pd.DataFrame()

    predictions = warehouse.read_execution_confidence_predictions()
    if predictions is None or predictions.empty:
        return pd.DataFrame()
    pred_by_request: dict[str, pd.Series] = {
        str(row["request_id"]): row for _, row in predictions.iterrows()
    }

    trade_rows: list[dict[str, Any]] = []
    for _, outcome_row in outcomes.iterrows():
        request_id = str(outcome_row["request_id"])
        pred_row = pred_by_request.get(request_id)
        if pred_row is None:
            continue
        request = _reconstruct_request_from_prediction(pred_row)
        response = _reconstruct_response_from_prediction(pred_row)
        outcome_dict = _outcome_row_to_dict(outcome_row)
        trade = TradeRecord(
            request_id=request_id,
            timestamp=request.timestamp,
            cusip=request.cusip,
            side=request.side,  # type: ignore[arg-type]
            notional=float(request.notional),
            protocol=request.protocol,
            arrival_price=outcome_dict.get("arrival_price"),
            execution_price=outcome_dict.get("execution_price"),
            filled_quantity=outcome_dict.get("filled_quantity"),
            time_to_fill_seconds=outcome_dict.get("time_to_fill_seconds"),
            dealer_response_count=outcome_dict.get("dealer_response_count"),
            sector=request.sector,
            rating=request.rating,
            maturity_years=outcome_dict.get("maturity_years"),
        )
        tagged = tag_trade_with_regime_context(trade, warehouse=warehouse)
        try:
            tca = compute_tca_metrics_for_outcome(
                request, response, outcome_dict, warehouse=warehouse
            )
        except PitViolationError:
            log.warning(
                "materialize_tca_segments_for_day: dropping request_id=%s due to "
                "outcome-observation-lag violation",
                request_id,
            )
            continue
        row_out: dict[str, Any] = {
            "request_id": request_id,
            "trade_timestamp": iso8601_z(_coerce_utc(request.timestamp)),
            "regime_label": tagged.regime_label,
            "liquidity_label": tagged.liquidity_label,
            "execution_confidence_bucket": tagged.execution_confidence_bucket,
            "protocol": trade.protocol,
            "side": trade.side,
            "sector": tagged.sector_bucket,
            "rating": tagged.rating_bucket,
            "maturity_bucket": tagged.maturity_bucket,
            "notional_bucket": tagged.notional_bucket,
            "regime_soft_weights": dict(tagged.regime_soft_weights),
        }
        row_out.update(tca)
        trade_rows.append(row_out)
    return pd.DataFrame(trade_rows)


def _outcome_row_to_dict(row: pd.Series) -> dict[str, Any]:
    """Materialise an ``execution_outcomes`` row plus its metadata blob."""
    metadata_json = row.get("metadata_json")
    metadata: dict[str, Any] = {}
    if metadata_json is not None and not (
        isinstance(metadata_json, float) and math.isnan(metadata_json)
    ):
        try:
            metadata = json.loads(str(metadata_json))
        except json.JSONDecodeError:
            metadata = {}
    out: dict[str, Any] = dict(metadata)
    for col in (
        "observed_at",
        "execution_price",
        "filled_quantity",
        "decision_timestamp",
    ):
        if col in row.index and row.get(col) is not None and not (
            isinstance(row.get(col), float) and math.isnan(row.get(col))
        ):
            out[col] = row.get(col)
    # ``arrival_price`` etc. ride in the metadata blob; surface them at
    # top-level for the TCA helpers.
    return out


def _reconstruct_request_from_prediction(row: pd.Series) -> ExecutionConfidenceRequest:
    metadata_json = row.get("metadata_json")
    metadata: dict[str, Any] = {}
    if metadata_json is not None and not (
        isinstance(metadata_json, float) and math.isnan(metadata_json)
    ):
        try:
            metadata = json.loads(str(metadata_json))
        except json.JSONDecodeError:
            metadata = {}
    return ExecutionConfidenceRequest(
        timestamp=str(row["timestamp"]),
        cusip=str(row["cusip"]),
        side=str(row["side"]),
        notional=float(row["notional"]),
        protocol=str(row["protocol"]),
        sector=metadata.get("sector"),
        rating=metadata.get("rating"),
        maturity_bucket=metadata.get("maturity_bucket"),
        urgency=metadata.get("urgency"),
        metadata=metadata,
    )


def _reconstruct_response_from_prediction(row: pd.Series) -> ExecutionConfidenceResponse:
    expected_slippage = row.get("expected_slippage_bps")
    if expected_slippage is not None and isinstance(expected_slippage, float) and math.isnan(expected_slippage):
        expected_slippage = None
    return ExecutionConfidenceResponse(
        timestamp=str(row["timestamp"]),
        cusip=str(row["cusip"]),
        side=str(row["side"]),
        notional=float(row["notional"]),
        protocol=str(row["protocol"]),
        confidence_score=float(row["confidence_score"]),
        expected_slippage_bps=(
            float(expected_slippage) if expected_slippage is not None else None
        ),
        confidence_interval_low=None,
        confidence_interval_high=None,
        recommended_action=str(row.get("recommended_action") or ""),
        human_review_required=bool(int(row.get("human_review_required") or 0)),
        model_run_id=str(row["model_run_id"]),
        release_gate=bool(int(row.get("release_gate") or 0)),
        artifact_hash=str(row.get("artifact_hash") or ""),
        metadata={},
    )


def materialize_tca_segments_for_day(
    warehouse: Any,
    *,
    date: pd.Timestamp,
    soft_weighting: bool = False,
    use_hysteresis: bool = True,  # noqa: ARG001  threaded for API compatibility
    model_run_id: str | None = None,
) -> int:
    """Materialise tca_regime_segments rows for the given date.

    Reads ``execution_outcomes`` for the day, builds a trades frame
    with every dim-label populated, aggregates over the canonical
    dim-combos in :func:`_dimension_combinations`, and writes one row
    per ``(combo, metric)`` to ``tca_regime_segments``.

    Returns the number of segment rows written.
    """
    trades = _build_trades_frame_from_outcomes(warehouse, date=date)
    if trades.empty:
        return 0
    timestamp_utc = _coerce_utc(date)
    resolved_run_id = (
        model_run_id
        if model_run_id and model_run_id.strip()
        else f"tca_segmentation-{uuid.uuid4().hex[:12]}"
    )
    rows_written = 0
    for dim_combo in _dimension_combinations():
        agg = aggregate_tca_by_regime(
            trades,
            dimensions=dim_combo,
            metrics_names=TCA_METRICS,
            soft_weighting=soft_weighting,
        )
        if agg.empty:
            continue
        for _, agg_row in agg.iterrows():
            segment = TcaRegimeSegment(
                timestamp=timestamp_utc,
                regime_label=agg_row.get("regime_label") if "regime_label" in dim_combo else None,
                liquidity_label=(
                    agg_row.get("liquidity_label") if "liquidity_label" in dim_combo else None
                ),
                execution_confidence_bucket=(
                    agg_row.get("execution_confidence_bucket")
                    if "execution_confidence_bucket" in dim_combo
                    else None
                ),
                protocol=agg_row.get("protocol") if "protocol" in dim_combo else None,
                side=agg_row.get("side") if "side" in dim_combo else None,
                sector=agg_row.get("sector") if "sector" in dim_combo else None,
                rating=agg_row.get("rating") if "rating" in dim_combo else None,
                maturity_bucket=(
                    agg_row.get("maturity_bucket") if "maturity_bucket" in dim_combo else None
                ),
                notional_bucket=(
                    agg_row.get("notional_bucket") if "notional_bucket" in dim_combo else None
                ),
                metric_name=str(agg_row["metric_name"]),
                metric_value=float(agg_row["metric_value"]),
                sample_count=int(agg_row["sample_count"]),
                model_run_id=resolved_run_id,
                metadata_json=json.dumps(
                    {
                        "soft_weighting": bool(soft_weighting),
                        "dimensions": list(dim_combo),
                    },
                    sort_keys=True,
                ),
            )
            write_tca_regime_segment(warehouse, segment)
            rows_written += 1
    return rows_written
