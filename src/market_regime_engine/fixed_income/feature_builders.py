# SPDX-License-Identifier: Apache-2.0
"""Fixed-Income feature builders.

PR-3 (this commit) lands :func:`build_credit_features` consuming
``curve_snapshots``, ``cds_curve_snapshots`` and (when present)
``vintage_observations`` for VIX / MOVE. PR-4 and PR-5 fill in
:func:`build_liquidity_features` and :func:`build_execution_features`.

All FI feature builders share the same contracts:

1. Every produced row passes through :func:`pit_guard.assert_pit_safe`
   (and, where the warehouse carries vintage info, the vintage rail
   too) so a stale or future-dated feature trips ``PitViolationError``
   rather than silently feeding the scorer.
2. The per-column NaN policy defaults to
   :attr:`NanPolicy.NAN_FAILS_PIT_AUDIT` so missing inputs trigger
   ``release_gate=False`` instead of silently zero-filling.
3. Naive datetimes are rejected via :func:`timestamps.to_utc`.
4. Trade timestamps must fall on a SIFMA trading day (closed-day rows
   raise :class:`PitViolationError`).

Output shape: a *long* DataFrame with columns
``["date", "feature_name", "value", "source_timestamp", "vintage_date"]``
so the same plumbing serves both the scorer (pivots to wide) and the
audit / evidence-pack path (groups by ``feature_name``).

``merge_asof`` joins inside the builders use the
:data:`DEFAULT_INTRADAY_MERGE_TOLERANCE` constant — see REVIEW.md
§3.4 Q-9. The tolerance is documented per join type in the helper
docstrings.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from typing import Any

import pandas as pd

from market_regime_engine.fixed_income.calendars import (
    TradingCalendar,
    is_trading_day,
)
from market_regime_engine.fixed_income.pit_guard import (
    PitViolationError,
    assert_pit_safe,
)
from market_regime_engine.fixed_income.timestamps import to_utc
from market_regime_engine.frontier.data_cleaning import NanPolicy

log = logging.getLogger(__name__)

# REVIEW.md §3.4 Q-9: explicit tolerance for intraday merge_asof joins.
# 5 minutes covers the typical FI quote/trade alignment without
# letting an out-of-sync feed leak hours-stale data into the join.
DEFAULT_INTRADAY_MERGE_TOLERANCE: pd.Timedelta = pd.Timedelta("5min")

# Daily tolerance for end-of-day curve / volatility joins; a same-day
# observation off by minutes is fine but a join across days is not.
DEFAULT_EOD_MERGE_TOLERANCE: pd.Timedelta = pd.Timedelta("1D")

_FI_OUTPUT_COLUMNS: tuple[str, ...] = (
    "date",
    "feature_name",
    "value",
    "source_timestamp",
    "vintage_date",
)

__all__ = [
    "DEFAULT_EOD_MERGE_TOLERANCE",
    "DEFAULT_INTRADAY_MERGE_TOLERANCE",
    "build_credit_features",
    "build_execution_features",
    "build_liquidity_features",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _to_utc_index(series: pd.Series) -> pd.DatetimeIndex:
    """Coerce a timestamp series to a UTC ``DatetimeIndex``.

    Per REVIEW.md §3.4 Q-7 every FI ingest call site routes its
    incoming timestamp through :func:`timestamps.to_utc`. Builder
    inputs from the warehouse may carry tz-naive strings (the storage
    convention is ISO-8601 with ``Z``); we coerce here and accept both
    aware and naive (interpreting naive as UTC for warehouse-stored
    values, since the writer stamped them with the Z suffix).
    """
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return pd.DatetimeIndex(parsed)


def _curve_metric(
    snap: pd.DataFrame,
    *,
    curve_type: str,
    tenor: str,
) -> pd.Series:
    """Return the per-date rate for ``(curve_type, tenor)`` indexed by date.

    Used by :func:`_build_curve_features` to extract the 2Y / 5Y / 10Y
    rates required for slope / curvature. Returns an empty series with
    a UTC index when no matching rows exist.
    """
    if snap.empty:
        return pd.Series(dtype=float, name=f"{curve_type}_{tenor}")
    mask = (snap["curve_type"].astype(str) == curve_type) & (snap["tenor"].astype(str) == tenor)
    rows = snap.loc[mask].copy()
    if rows.empty:
        return pd.Series(dtype=float, name=f"{curve_type}_{tenor}")
    rows["timestamp"] = _to_utc_index(rows["timestamp"])
    rows = rows.dropna(subset=["timestamp"]).sort_values("timestamp")
    series = rows.groupby("timestamp")["rate"].last().astype(float)
    series.name = f"{curve_type}_{tenor}"
    return series


def _cds_metric(
    snap: pd.DataFrame,
    *,
    reference_entity: str,
    tenor: str,
) -> pd.Series:
    """Return per-date spread (bps) for ``(reference_entity, tenor)``."""
    if snap.empty:
        return pd.Series(dtype=float, name=f"{reference_entity}_{tenor}")
    mask = (snap["reference_entity"].astype(str) == reference_entity) & (snap["tenor"].astype(str) == tenor)
    rows = snap.loc[mask].copy()
    if rows.empty:
        return pd.Series(dtype=float, name=f"{reference_entity}_{tenor}")
    rows["timestamp"] = _to_utc_index(rows["timestamp"])
    rows = rows.dropna(subset=["timestamp"]).sort_values("timestamp")
    series = rows.groupby("timestamp")["spread_bps"].last().astype(float)
    series.name = f"{reference_entity}_{tenor}"
    return series


def _vintage_metric(
    obs: pd.DataFrame,
    series_id: str,
) -> pd.DataFrame:
    """Return ``vintage_observations`` rows for ``series_id`` (latest vintage per obs date).

    Each row carries ``date`` (observation_date), ``value``, and
    ``vintage_date``. The "latest vintage per observation date" rule
    matches the PIT contract that an FI scoring run uses the freshest
    realtime-available print of each indicator.
    """
    if obs.empty or "series_id" not in obs.columns:
        return pd.DataFrame(columns=["date", "value", "vintage_date"])
    rows = obs.loc[obs["series_id"].astype(str) == series_id].copy()
    if rows.empty:
        return pd.DataFrame(columns=["date", "value", "vintage_date"])
    rows["date"] = _to_utc_index(rows["observation_date"])
    rows["vintage_date"] = pd.to_datetime(rows["vintage_date"], errors="coerce", utc=True)
    rows = rows.dropna(subset=["date"]).sort_values(["date", "vintage_date"])
    # Pick latest vintage per observation date.
    return rows.groupby("date").tail(1)[["date", "value", "vintage_date"]].reset_index(drop=True)


def _filter_lookback(
    series: pd.Series,
    *,
    asof: pd.Timestamp,
    lookback_days: int,
) -> pd.Series:
    if series.empty:
        return series
    lower = asof - pd.Timedelta(days=int(lookback_days))
    return series[(series.index > lower) & (series.index <= asof)]


def _emit_row(
    *,
    date: pd.Timestamp,
    feature_name: str,
    value: float | None,
    source_timestamp: pd.Timestamp,
    vintage_date: pd.Timestamp | None = None,
) -> dict[str, Any]:
    return {
        "date": date,
        "feature_name": feature_name,
        "value": float("nan") if value is None else float(value),
        "source_timestamp": source_timestamp,
        "vintage_date": vintage_date,
    }


def _safe_read(warehouse: Any, attr: str) -> pd.DataFrame:
    """Call ``warehouse.<attr>()`` defensively; return empty on absence."""
    method = getattr(warehouse, attr, None)
    if method is None:
        return pd.DataFrame()
    try:
        result = method()
    except Exception as exc:
        log.warning("read failure on %s: %s; treating as empty", attr, exc)
        return pd.DataFrame()
    return result if result is not None else pd.DataFrame()


# ---------------------------------------------------------------------------
# public builders
# ---------------------------------------------------------------------------


def build_credit_features(
    warehouse: Any,
    asof: pd.Timestamp | str,
    *,
    lookback_days: int = 504,
    freq: str = "D",
    nan_policy: NanPolicy = NanPolicy.NAN_FAILS_PIT_AUDIT,
    calendar: TradingCalendar = TradingCalendar.SIFMA_BOND,
) -> pd.DataFrame:
    """Build PIT-safe credit features from the FI warehouse.

    Reads:

    - ``curve_snapshots`` — Treasury (``ust``) and swap (``swap``)
      curves; computes ``level`` (10Y rate), ``slope`` (10Y - 2Y),
      ``curvature`` (2 × 5Y - 2Y - 10Y).
    - ``cds_curve_snapshots`` — CDX.IG 5Y and CDX.HY 5Y spreads.
    - ``vintage_observations`` — VIX, MOVE, and (optionally) the
      ETF premium/discount proxy.

    Returns
    -------
    DataFrame
        Long-form with columns
        ``["date", "feature_name", "value", "source_timestamp", "vintage_date"]``.
        Rows are within ``(asof - lookback_days, asof]``.

    Notes
    -----
    - Every emitted row passes through :func:`pit_guard.assert_pit_safe`.
      A row whose ``source_timestamp > asof`` raises
      :class:`PitViolationError` — silent drop is not an option.
    - Rows whose ``source_timestamp`` falls on a closed bond-market
      day (per the SIFMA calendar) raise
      :class:`PitViolationError`; weekends are tolerated for the
      vintage_observations path (VIX/MOVE settle daily and an
      end-of-week aggregator is fine).
    - ``merge_asof`` is NOT used inside this builder because the
      curve / CDS / vintage tables already carry one-row-per-date
      semantics; the merge tolerance constants
      (:data:`DEFAULT_INTRADAY_MERGE_TOLERANCE`,
      :data:`DEFAULT_EOD_MERGE_TOLERANCE`) are exported for downstream
      builders that DO join across cadences.
    - ``freq`` is reserved for future cadences (``"H"`` /``"15min"``);
      PR-3 ships daily features only.
    - ``nan_policy`` is forwarded to the scorer that consumes this
      frame; the builder itself does not apply it (cleaning happens
      after pivot).

    Raises
    ------
    PitViolationError
        If any feature row's ``source_timestamp`` is after ``asof`` or
        falls on a closed trading day.
    """
    asof_utc = (
        to_utc(asof)
        if isinstance(asof, str)
        else pd.Timestamp(asof, tz="UTC")
        if pd.Timestamp(asof).tzinfo is None
        else pd.Timestamp(asof).tz_convert("UTC")
    )
    if asof_utc is None:
        raise ValueError("asof is required and must not be None")

    rows: list[dict[str, Any]] = []

    # ----- Treasury & swap curves -----
    curve_snap = _safe_read(warehouse, "read_curve_snapshots")
    if not curve_snap.empty:
        for curve_type in ("ust", "swap"):
            ten_2y = _curve_metric(curve_snap, curve_type=curve_type, tenor="2Y")
            ten_5y = _curve_metric(curve_snap, curve_type=curve_type, tenor="5Y")
            ten_10y = _curve_metric(curve_snap, curve_type=curve_type, tenor="10Y")
            ten_2y = _filter_lookback(ten_2y, asof=asof_utc, lookback_days=lookback_days)
            ten_5y = _filter_lookback(ten_5y, asof=asof_utc, lookback_days=lookback_days)
            ten_10y = _filter_lookback(ten_10y, asof=asof_utc, lookback_days=lookback_days)
            rows.extend(_emit_curve_rows(ten_2y, ten_5y, ten_10y, curve_type=curve_type))

    # ----- CDX IG / HY 5Y -----
    cds_snap = _safe_read(warehouse, "read_cds_curve_snapshots")
    if not cds_snap.empty:
        for entity in ("CDX.IG", "CDX.HY"):
            series = _cds_metric(cds_snap, reference_entity=entity, tenor="5Y")
            series = _filter_lookback(series, asof=asof_utc, lookback_days=lookback_days)
            feat = entity.lower().replace(".", "_") + "_5y"
            for ts, val in series.items():
                rows.append(
                    _emit_row(
                        date=ts,
                        feature_name=feat,
                        value=val,
                        source_timestamp=ts,
                        vintage_date=None,
                    )
                )

    # ----- VIX / MOVE / ETF premium-discount from vintage_observations -----
    obs = _safe_read(warehouse, "read_vintage_observations")
    if not obs.empty:
        for series_id, feat in (("VIX", "vix"), ("MOVE", "move"), ("ETF_PREM_DISC", "etf_prem_disc")):
            df = _vintage_metric(obs, series_id)
            if df.empty:
                continue
            mask = (df["date"] > asof_utc - pd.Timedelta(days=int(lookback_days))) & (df["date"] <= asof_utc)
            for _, r in df.loc[mask].iterrows():
                rows.append(
                    _emit_row(
                        date=r["date"],
                        feature_name=feat,
                        value=r["value"],
                        source_timestamp=r["date"],
                        vintage_date=r["vintage_date"],
                    )
                )

    if not rows:
        return pd.DataFrame(columns=list(_FI_OUTPUT_COLUMNS))

    frame = pd.DataFrame(rows, columns=list(_FI_OUTPUT_COLUMNS))
    _enforce_pit_and_calendar(frame, asof=asof_utc, calendar=calendar)
    # Forward the active NaN policy on the metadata so the scorer can
    # mirror the caller's intent without parameter threading.
    frame.attrs["nan_policy"] = nan_policy.value
    frame.attrs["asof"] = asof_utc.isoformat()
    frame.attrs["lookback_days"] = int(lookback_days)
    frame.attrs["freq"] = freq
    return frame


def _emit_curve_rows(
    two: pd.Series,
    five: pd.Series,
    ten: pd.Series,
    *,
    curve_type: str,
) -> Iterable[dict[str, Any]]:
    """Emit (level, slope, curvature) rows for a single curve type.

    Each date that has *all three* tenors contributes three rows; we
    intentionally skip dates with partial coverage rather than
    forward-fill, deferring the policy decision to the scorer's NaN
    policy via :func:`clean_with_policy`.
    """
    if two.empty or five.empty or ten.empty:
        return []
    joined = pd.concat([two, five, ten], axis=1, join="inner").dropna()
    out: list[dict[str, Any]] = []
    for ts, row in joined.iterrows():
        two_v, five_v, ten_v = float(row.iloc[0]), float(row.iloc[1]), float(row.iloc[2])
        out.append(
            _emit_row(
                date=ts,
                feature_name=f"{curve_type}_level",
                value=ten_v,
                source_timestamp=ts,
            )
        )
        out.append(
            _emit_row(
                date=ts,
                feature_name=f"{curve_type}_slope",
                value=ten_v - two_v,
                source_timestamp=ts,
            )
        )
        out.append(
            _emit_row(
                date=ts,
                feature_name=f"{curve_type}_curvature",
                value=2.0 * five_v - two_v - ten_v,
                source_timestamp=ts,
            )
        )
    return out


def _enforce_pit_and_calendar(
    frame: pd.DataFrame,
    *,
    asof: pd.Timestamp,
    calendar: TradingCalendar,
) -> None:
    """Row-by-row PIT + trading-day enforcement.

    Mirrors the AGENT.md PIT contract: any source_timestamp > asof
    raises :class:`PitViolationError`. We deliberately do NOT enforce
    the trading-day rail on vintage_observations (VIX/MOVE may carry
    a settlement timestamp on a weekend in some vendor feeds);
    curve/CDS rows on closed trading days do raise.

    v1.5 PR-8 (Tier-2 fix A2, REVIEW.md): the PIT rails are now
    enforced via the vectorised :func:`audit_pit_dataframe` rather
    than a per-row ``iterrows()`` + :func:`assert_pit_safe` loop. The
    trading-day rail still loops because it depends on the per-row
    feature_name prefix; that loop is bounded by the curve/CDS subset
    and so is acceptable.
    """
    if frame.empty:
        return
    from market_regime_engine.fixed_income.pit_guard import audit_pit_dataframe

    df = frame.copy()
    df["__decision_ts"] = asof
    vintage_col = "vintage_date" if "vintage_date" in df.columns else None
    if vintage_col is None:
        log.warning("feature frame missing vintage_date column; PIT vintage rail skipped for all rows")
    report = audit_pit_dataframe(
        df,
        decision_timestamp_col="__decision_ts",
        feature_timestamp_col="source_timestamp",
        vintage_timestamp_col=vintage_col,
    )
    if report.violation_count > 0:
        first = report.violations.iloc[0]
        label = str(first.get("feature_name", "feature"))
        reason = str(first.get("pit_violation_reason", ""))
        raise PitViolationError(
            f"PIT audit failed: {report.violation_count} row(s) violate PIT "
            f"(asof={asof}, first violator label={label!r} "
            f"reason={reason!r})"
        )
    # Calendar check: only relevant for curve/CDS feature_name prefixes,
    # which is a tiny subset of the frame on the hot path. Vectorising
    # would require a calendar-aware vectorised is_trading_day; staying
    # on the per-row loop here keeps the cost bounded by the curve/CDS
    # subset rather than the full feature frame.
    prefixes = ("ust_", "swap_", "cdx_ig_", "cdx_hy_")
    name_mask = frame["feature_name"].astype(str).str.startswith(prefixes)
    curves = frame.loc[name_mask]
    if curves.empty:
        return
    for _, row in curves.iterrows():
        source_ts = pd.Timestamp(row["source_timestamp"])
        if source_ts.tzinfo is None:
            source_ts = source_ts.tz_localize("UTC")
        if not is_trading_day(source_ts, calendar):
            raise PitViolationError(
                f"feature {row['feature_name']!r} reports on closed trading day "
                f"{source_ts.isoformat()} per calendar {calendar.value}"
            )


def build_liquidity_features(
    warehouse: Any,
    asof: pd.Timestamp | str,
    *,
    scope_type: str,
    scope_id: str,
    lookback_days: int = 30,
    nan_policy: NanPolicy = NanPolicy.NAN_FAILS_PIT_AUDIT,
    calendar: TradingCalendar = TradingCalendar.SIFMA_BOND,
) -> pd.DataFrame:
    """Build PIT-safe liquidity-stress features from the FI warehouse.

    Per ``MRE_FIXED_INCOME_AGENT.md §"PR 4"`` and ``INSTRUCTIONS.md §6.2``,
    the eleven feature names emitted are::

        bid_ask_width, trade_count_velocity, volume_over_adv,
        time_since_last_trade, dealers_requested, quotes_received,
        quote_dispersion, amihud_illiquidity, dealer_response_count,
        axe_freshness_proxy, order_imbalance

    Scope filtering (REVIEW.md §3.4 Q-4 survivorship-safe via
    :func:`read_bond_reference_asof`):

    * ``market`` — every cusip seen in the trade/quote/RFQ tables.
    * ``sector`` — filter the bond reference at ``asof`` to
      ``sector == scope_id``, then restrict trades/quotes/RFQs to the
      surviving cusip set.
    * ``rating`` — same shape with ``rating == scope_id``.
    * ``cusip`` — single bond keyed by ``scope_id``.

    Output: long-form DataFrame with the standard PR-3 columns
    ``["date", "feature_name", "value", "source_timestamp",
    "vintage_date"]``. Per-feature semantics:

    * ``bid_ask_width`` — per-day median(ask) − median(bid) across
      cusips/dealers.
    * ``trade_count_velocity`` — count of trades per UTC date.
    * ``volume_over_adv`` — daily volume / mean daily volume in the
      lookback (the trailing average daily volume proxy).
    * ``time_since_last_trade`` — minutes between ``asof`` and the
      most recent trade in the window (one row at ``asof``).
    * ``dealers_requested`` / ``quotes_received`` /
      ``dealer_response_count`` — RFQ aggregates per day.
    * ``quote_dispersion`` — per-day std of dealer quote prices
      across (cusip, side); aggregated across cusips by mean.
    * ``amihud_illiquidity`` — per-day |VWAP return| / daily volume.
    * ``axe_freshness_proxy`` — seconds between ``asof`` and the
      most recent dealer quote (one row at ``asof``).
    * ``order_imbalance`` — per-day (buy_volume − sell_volume) /
      total_volume.

    PIT contract: every emitted row passes
    :func:`pit_guard.assert_pit_safe`. Per-source-trading-day
    enforcement is applied only to ``trade_count_velocity`` /
    ``volume_over_adv`` / ``quote_dispersion`` / ``bid_ask_width``
    (market-microstructure features that are only meaningful on open
    trading days); the ``time_since_last_trade`` and
    ``axe_freshness_proxy`` rows are stamped at ``asof`` so they
    always pass the trading-day rail.

    Raises
    ------
    PitViolationError
        If a market-microstructure row falls on a closed trading day.
    ValueError
        If ``scope_type`` is not in
        ``{"market", "sector", "rating", "cusip"}``.
    """
    if scope_type not in {"market", "sector", "rating", "cusip"}:
        raise ValueError(f"scope_type must be one of {{'market', 'sector', 'rating', 'cusip'}}; got {scope_type!r}")

    asof_utc = _coerce_asof_utc(asof)
    lower = asof_utc - pd.Timedelta(days=int(lookback_days))

    cusip_filter: set[str] | None = _resolve_scope_cusips(warehouse, asof_utc, scope_type=scope_type, scope_id=scope_id)

    trades = _filter_microstructure(
        _safe_read(warehouse, "read_trace_trades"),
        column="timestamp",
        lower=lower,
        upper=asof_utc,
        cusip_filter=cusip_filter,
    )
    quotes = _filter_microstructure(
        _safe_read(warehouse, "read_dealer_quotes"),
        column="timestamp",
        lower=lower,
        upper=asof_utc,
        cusip_filter=cusip_filter,
    )
    rfqs = _filter_microstructure(
        _safe_read(warehouse, "read_rfq_events"),
        column="timestamp",
        lower=lower,
        upper=asof_utc,
        cusip_filter=cusip_filter,
    )

    rows: list[dict[str, Any]] = []
    rows.extend(_emit_bid_ask_rows(quotes))
    rows.extend(_emit_trade_velocity_rows(trades))
    rows.extend(_emit_volume_over_adv_rows(trades))
    rows.extend(_emit_time_since_last_trade_rows(trades, asof=asof_utc))
    rows.extend(_emit_rfq_rows(rfqs))
    rows.extend(_emit_quote_dispersion_rows(quotes))
    rows.extend(_emit_amihud_rows(trades))
    rows.extend(_emit_axe_freshness_rows(quotes, asof=asof_utc))
    rows.extend(_emit_order_imbalance_rows(trades))

    if not rows:
        return pd.DataFrame(columns=list(_FI_OUTPUT_COLUMNS))

    frame = pd.DataFrame(rows, columns=list(_FI_OUTPUT_COLUMNS))
    _enforce_pit_liquidity(frame, asof=asof_utc, calendar=calendar)
    frame.attrs["nan_policy"] = nan_policy.value
    frame.attrs["asof"] = asof_utc.isoformat()
    frame.attrs["lookback_days"] = int(lookback_days)
    frame.attrs["scope_type"] = scope_type
    frame.attrs["scope_id"] = scope_id
    return frame


# ---------------------------------------------------------------------------
# liquidity-feature helpers
# ---------------------------------------------------------------------------


# Features that *must* land on a SIFMA trading day (the market
# microstructure feeds). ``time_since_last_trade`` and
# ``axe_freshness_proxy`` rows are stamped at ``asof`` and skip the
# trading-day rail (they describe the gap to the decision instant, not
# the print clock).
_LIQUIDITY_TRADING_DAY_FEATURES: frozenset[str] = frozenset(
    {
        "bid_ask_width",
        "trade_count_velocity",
        "volume_over_adv",
        "quote_dispersion",
        "amihud_illiquidity",
        "dealers_requested",
        "quotes_received",
        "dealer_response_count",
        "order_imbalance",
    }
)


def _coerce_asof_utc(asof: pd.Timestamp | str) -> pd.Timestamp:
    if isinstance(asof, str):
        out = to_utc(asof)
        if out is None:
            raise ValueError("asof must not be None")
        return out
    ts = pd.Timestamp(asof)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _resolve_scope_cusips(
    warehouse: Any,
    asof: pd.Timestamp,
    *,
    scope_type: str,
    scope_id: str,
) -> set[str] | None:
    """Return the cusip set to filter trades/quotes/RFQs by, or ``None`` for market scope.

    Sector / rating scopes route through :func:`read_bond_reference_asof`
    so the filter respects survivorship (defaulted / delisted cusips
    are excluded automatically). Cusip scope returns the single
    requested cusip wrapped in a set. Market scope returns ``None`` so
    no filter is applied.
    """
    if scope_type == "market":
        return None
    if scope_type == "cusip":
        if not scope_id:
            raise ValueError("cusip scope requires a non-empty scope_id")
        return {str(scope_id)}
    # sector / rating: read bond reference at asof and filter the
    # column matching the scope.
    from market_regime_engine.storage import read_bond_reference_asof  # local import to avoid cycle

    if not scope_id:
        raise ValueError(f"{scope_type} scope requires a non-empty scope_id")
    bond_ref = read_bond_reference_asof(warehouse, asof)
    if bond_ref is None or bond_ref.empty or scope_type not in bond_ref.columns:
        return set()
    matches = bond_ref.loc[bond_ref[scope_type].astype(str) == str(scope_id), "cusip"]
    return set(matches.astype(str).tolist())


def _filter_microstructure(
    frame: pd.DataFrame,
    *,
    column: str,
    lower: pd.Timestamp,
    upper: pd.Timestamp,
    cusip_filter: set[str] | None,
) -> pd.DataFrame:
    """Time-window + cusip filter for a microstructure table.

    The merge tolerance constant
    :data:`DEFAULT_INTRADAY_MERGE_TOLERANCE` is referenced by downstream
    cusip-level joins (e.g. trade↔quote slippage) which v1.5 does not
    yet compute; the constant remains the documented contract per
    REVIEW.md §3.4 Q-9.
    """
    if frame is None or frame.empty or column not in frame.columns:
        return pd.DataFrame()
    f = frame.copy()
    f[column] = pd.to_datetime(f[column], utc=True, errors="coerce")
    f = f.dropna(subset=[column])
    f = f.loc[(f[column] > lower) & (f[column] <= upper)]
    if cusip_filter is not None and "cusip" in f.columns:
        if not cusip_filter:
            return f.iloc[0:0]
        f = f.loc[f["cusip"].astype(str).isin(cusip_filter)]
    return f.reset_index(drop=True)


def _floor_date(ts_series: pd.Series) -> pd.Series:
    return pd.to_datetime(ts_series, utc=True, errors="coerce").dt.floor("D")


def _emit_bid_ask_rows(quotes: pd.DataFrame) -> list[dict[str, Any]]:
    if quotes.empty:
        return []
    q = quotes.copy()
    q["date"] = _floor_date(q["timestamp"])
    q = q.dropna(subset=["date"])
    if "side" not in q.columns:
        return []
    bid = q.loc[q["side"].astype(str).str.lower() == "bid"].groupby(["date", "cusip"])["price"].median()
    ask = q.loc[q["side"].astype(str).str.lower() == "ask"].groupby(["date", "cusip"])["price"].median()
    width = (ask - bid).dropna()
    if width.empty:
        return []
    daily = width.groupby(level="date").mean()
    out: list[dict[str, Any]] = []
    for ts, v in daily.items():
        out.append(_emit_row(date=ts, feature_name="bid_ask_width", value=float(v), source_timestamp=ts))
    return out


def _emit_trade_velocity_rows(trades: pd.DataFrame) -> list[dict[str, Any]]:
    if trades.empty:
        return []
    t = trades.copy()
    t["date"] = _floor_date(t["timestamp"])
    t = t.dropna(subset=["date"])
    if t.empty:
        return []
    counts = t.groupby("date").size().astype(float)
    return [
        _emit_row(date=ts, feature_name="trade_count_velocity", value=float(v), source_timestamp=ts)
        for ts, v in counts.items()
    ]


def _emit_volume_over_adv_rows(trades: pd.DataFrame) -> list[dict[str, Any]]:
    if trades.empty or "size" not in trades.columns:
        return []
    t = trades.copy()
    t["date"] = _floor_date(t["timestamp"])
    t = t.dropna(subset=["date"])
    if t.empty:
        return []
    daily_volume = t.groupby("date")["size"].sum().astype(float)
    if daily_volume.empty:
        return []
    adv = float(daily_volume.mean())
    if adv <= 0:
        return []
    return [
        _emit_row(
            date=ts,
            feature_name="volume_over_adv",
            value=float(v) / adv,
            source_timestamp=ts,
        )
        for ts, v in daily_volume.items()
    ]


def _emit_time_since_last_trade_rows(trades: pd.DataFrame, *, asof: pd.Timestamp) -> list[dict[str, Any]]:
    """Emit one row at ``asof`` carrying minutes since the most recent trade.

    The scorer's ``_latest`` reads only the final non-NaN value, so a
    single ``asof``-stamped row is the cleanest representation.
    """
    if trades.empty or "timestamp" not in trades.columns:
        return []
    t = trades.copy()
    t["timestamp"] = pd.to_datetime(t["timestamp"], utc=True, errors="coerce")
    t = t.dropna(subset=["timestamp"])
    if t.empty:
        return []
    last_ts = t["timestamp"].max()
    delta = asof - last_ts
    minutes = max(0.0, float(delta.total_seconds()) / 60.0)
    return [
        _emit_row(
            date=asof,
            feature_name="time_since_last_trade",
            value=minutes,
            source_timestamp=asof,
        )
    ]


def _emit_rfq_rows(rfqs: pd.DataFrame) -> list[dict[str, Any]]:
    if rfqs.empty:
        return []
    r = rfqs.copy()
    r["date"] = _floor_date(r["timestamp"])
    r = r.dropna(subset=["date"])
    if r.empty:
        return []
    out: list[dict[str, Any]] = []
    if "dealers_requested" in r.columns:
        daily_req = r.groupby("date")["dealers_requested"].sum().astype(float)
        for ts, v in daily_req.items():
            out.append(_emit_row(date=ts, feature_name="dealers_requested", value=float(v), source_timestamp=ts))
    if "dealers_responded" in r.columns:
        daily_resp = r.groupby("date")["dealers_responded"].sum().astype(float)
        for ts, v in daily_resp.items():
            out.append(_emit_row(date=ts, feature_name="quotes_received", value=float(v), source_timestamp=ts))
            out.append(
                _emit_row(
                    date=ts,
                    feature_name="dealer_response_count",
                    value=float(v),
                    source_timestamp=ts,
                )
            )
    return out


def _emit_quote_dispersion_rows(quotes: pd.DataFrame) -> list[dict[str, Any]]:
    if quotes.empty or "price" not in quotes.columns:
        return []
    q = quotes.copy()
    q["date"] = _floor_date(q["timestamp"])
    q = q.dropna(subset=["date"])
    if q.empty:
        return []
    std = q.groupby(["date", "cusip"])["price"].std(ddof=0)
    daily = std.groupby(level="date").mean().dropna()
    return [
        _emit_row(date=ts, feature_name="quote_dispersion", value=float(v), source_timestamp=ts)
        for ts, v in daily.items()
    ]


def _emit_amihud_rows(trades: pd.DataFrame) -> list[dict[str, Any]]:
    """Amihud (2002) illiquidity ratio: ``|return| / volume`` per day.

    Volume-weighted average price (VWAP) is computed per day across
    all surviving cusips in the scope; the day-over-day VWAP return
    is divided by daily volume. Tiny denominators are squashed to
    ``NaN`` rather than producing a near-infinite stress signal.

    v1.5.1 (PR-9 FIX 5): the per-day VWAP / volume aggregation now
    runs via vectorised :meth:`groupby` reductions on the (px*size,
    size) columns rather than ``groupby.apply(lambda g: ...)``. The
    output is byte-identical (zero-volume days yield NaN VWAP and
    are dropped at the ``dropna`` boundary). The
    ``MRE_FI_LEGACY_VECTORIZE=1`` env var routes through the legacy
    ``apply`` path so operators can A/B parity on a suspected
    regression. The legacy branch is slated for deletion in v1.5.2.
    """
    if trades.empty or "size" not in trades.columns:
        return []
    t = trades.copy()
    t["timestamp"] = pd.to_datetime(t["timestamp"], utc=True, errors="coerce")
    t = t.dropna(subset=["timestamp"])
    if t.empty:
        return []
    t["date"] = t["timestamp"].dt.floor("D")
    if os.getenv("MRE_FI_LEGACY_VECTORIZE", "").strip() in {"1", "true", "yes", "on"}:
        grouped = t.groupby("date").apply(
            lambda g: pd.Series(
                {
                    "vwap": (g["price"] * g["size"]).sum() / g["size"].sum() if g["size"].sum() > 0 else float("nan"),
                    "volume": float(g["size"].sum()),
                }
            ),
            include_groups=False,
        )
    else:
        t["__notional"] = t["price"] * t["size"]
        agg = t.groupby("date").agg(
            notional=("__notional", "sum"),
            volume=("size", "sum"),
        )
        # Zero-volume days → NaN VWAP (matches legacy ``apply`` branch).
        volume_safe = agg["volume"].where(agg["volume"] > 0)
        agg["vwap"] = agg["notional"] / volume_safe
        grouped = agg[["vwap", "volume"]].astype(float)
    if grouped.empty:
        return []
    grouped = grouped.sort_index()
    grouped["return"] = grouped["vwap"].pct_change().abs()
    grouped["amihud"] = (grouped["return"] / grouped["volume"]).replace([float("inf"), -float("inf")], float("nan"))
    grouped = grouped.dropna(subset=["amihud"])
    return [
        _emit_row(date=ts, feature_name="amihud_illiquidity", value=float(v), source_timestamp=ts)
        for ts, v in grouped["amihud"].items()
    ]


def _emit_axe_freshness_rows(quotes: pd.DataFrame, *, asof: pd.Timestamp) -> list[dict[str, Any]]:
    """Seconds between ``asof`` and the most recent dealer quote (proxy for axe staleness)."""
    if quotes.empty or "timestamp" not in quotes.columns:
        return []
    q = quotes.copy()
    q["timestamp"] = pd.to_datetime(q["timestamp"], utc=True, errors="coerce")
    q = q.dropna(subset=["timestamp"])
    if q.empty:
        return []
    most_recent = q["timestamp"].max()
    seconds = max(0.0, float((asof - most_recent).total_seconds()))
    return [
        _emit_row(
            date=asof,
            feature_name="axe_freshness_proxy",
            value=seconds,
            source_timestamp=asof,
        )
    ]


def _emit_order_imbalance_rows(trades: pd.DataFrame) -> list[dict[str, Any]]:
    """Per-day (buy_volume − sell_volume) / total_volume."""
    if trades.empty or "size" not in trades.columns or "side" not in trades.columns:
        return []
    t = trades.copy()
    t["date"] = _floor_date(t["timestamp"])
    t = t.dropna(subset=["date"])
    if t.empty:
        return []
    t["side_norm"] = t["side"].astype(str).str.lower()
    pivot = t.groupby(["date", "side_norm"])["size"].sum().unstack(fill_value=0.0)
    if "buy" not in pivot.columns and "sell" not in pivot.columns:
        return []
    buy = pivot.get("buy", 0.0)
    sell = pivot.get("sell", 0.0)
    total = (buy + sell).replace(0.0, float("nan"))
    imbalance = ((buy - sell) / total).dropna()
    return [
        _emit_row(date=ts, feature_name="order_imbalance", value=float(v), source_timestamp=ts)
        for ts, v in imbalance.items()
    ]


def _enforce_pit_liquidity(
    frame: pd.DataFrame,
    *,
    asof: pd.Timestamp,
    calendar: TradingCalendar,
) -> None:
    """Row-by-row PIT + trading-day enforcement for liquidity features.

    v1.5.1 (PR-9 FIX 5): vectorised PIT path mirrors the credit
    builder's :func:`_enforce_pit_and_calendar` and the liquidity
    scorer's :func:`liquidity_stress._audit_pit`. The PIT rails go
    through :func:`audit_pit_dataframe` once; the trading-day rail
    still loops, but only over the subset of feature_names that
    require a SIFMA trading calendar check (e.g. trade-velocity,
    not VIX/MOVE). The ``MRE_FI_LEGACY_VECTORIZE=1`` env var routes
    through the pre-PR-9 iterrows loop for one release cycle.
    """
    if frame.empty:
        return
    if os.getenv("MRE_FI_LEGACY_VECTORIZE", "").strip() in {"1", "true", "yes", "on"}:
        _enforce_pit_liquidity_legacy_iterrows(frame, asof=asof, calendar=calendar)
        return

    from market_regime_engine.fixed_income.pit_guard import audit_pit_dataframe

    df = frame.copy()
    df["__decision_ts"] = asof
    vintage_col = "vintage_date" if "vintage_date" in df.columns else None
    report = audit_pit_dataframe(
        df,
        decision_timestamp_col="__decision_ts",
        feature_timestamp_col="source_timestamp",
        vintage_timestamp_col=vintage_col,
    )
    if report.violation_count > 0:
        first = report.violations.iloc[0]
        label = str(first.get("feature_name", "feature"))
        reason = str(first.get("pit_violation_reason", ""))
        raise PitViolationError(
            f"liquidity PIT audit failed: {report.violation_count} row(s) violate PIT "
            f"(asof={asof}, first violator label={label!r} reason={reason!r})"
        )

    name_series = frame["feature_name"].astype(str)
    cal_mask = name_series.isin(_LIQUIDITY_TRADING_DAY_FEATURES)
    if not cal_mask.any():
        return
    cal_subset = frame.loc[cal_mask]
    for _, row in cal_subset.iterrows():
        source_ts = pd.Timestamp(row["source_timestamp"])
        if source_ts.tzinfo is None:
            source_ts = source_ts.tz_localize("UTC")
        if not is_trading_day(source_ts, calendar):
            raise PitViolationError(
                f"liquidity feature {row['feature_name']!r} reports on closed trading day "
                f"{source_ts.isoformat()} per calendar {calendar.value}"
            )


def _enforce_pit_liquidity_legacy_iterrows(
    frame: pd.DataFrame,
    *,
    asof: pd.Timestamp,
    calendar: TradingCalendar,
) -> None:
    """Pre-v1.5.1 iterrows path, gated behind ``MRE_FI_LEGACY_VECTORIZE=1``.

    Slated for deletion in v1.5.2 once the vectorised path has burned in.
    """
    for _, row in frame.iterrows():
        source_ts = pd.Timestamp(row["source_timestamp"])
        if source_ts.tzinfo is None:
            source_ts = source_ts.tz_localize("UTC")
        vintage = row.get("vintage_date")
        if vintage is not None and not pd.isna(vintage):
            vintage_ts = pd.Timestamp(vintage)
            if vintage_ts.tzinfo is None:
                vintage_ts = vintage_ts.tz_localize("UTC")
        else:
            vintage_ts = None
        assert_pit_safe(
            feature_timestamp=source_ts,
            decision_timestamp=asof,
            vintage_timestamp=vintage_ts,
            label=str(row["feature_name"]),
        )
        if str(row["feature_name"]) in _LIQUIDITY_TRADING_DAY_FEATURES and not is_trading_day(source_ts, calendar):
            raise PitViolationError(
                f"liquidity feature {row['feature_name']!r} reports on closed trading day "
                f"{source_ts.isoformat()} per calendar {calendar.value}"
            )


def build_execution_features(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """v1.5 PR-5 shim: the real :func:`build_execution_features` now lives
    in :mod:`fixed_income.execution_confidence`. Imported lazily so the
    feature-builder module stays free of the PR-5 dependency cycle.
    """
    from market_regime_engine.fixed_income.execution_confidence import (
        build_execution_features as _impl,
    )

    return _impl(*args, **kwargs)
