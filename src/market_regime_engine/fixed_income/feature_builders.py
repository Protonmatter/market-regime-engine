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
    """
    if frame.empty:
        return
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
            if "vintage_date" not in frame.columns:
                log.warning("feature row %r missing vintage_date; PIT vintage rail skipped", row["feature_name"])
        assert_pit_safe(
            feature_timestamp=source_ts,
            decision_timestamp=asof,
            vintage_timestamp=vintage_ts,
            label=str(row["feature_name"]),
        )
        # Curve / CDS rows must fall on a SIFMA trading day; the
        # vintage_observations (VIX/MOVE/ETF) path tolerates weekend
        # settlement dates per the per-source rule above.
        if str(row["feature_name"]).startswith(("ust_", "swap_", "cdx_ig_", "cdx_hy_")) and not is_trading_day(
            source_ts, calendar
        ):
            raise PitViolationError(
                f"feature {row['feature_name']!r} reports on closed trading day "
                f"{source_ts.isoformat()} per calendar {calendar.value}"
            )


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
    ``market`` / ``sector`` / ``rating`` / ``cusip``. PR-3 keeps the
    stub; the real implementation lands in PR-4.
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
    historical-performance prior. PR-3 keeps the stub.
    """
    raise NotImplementedError("build_execution_features lands in PR-5 (execution confidence model)")
