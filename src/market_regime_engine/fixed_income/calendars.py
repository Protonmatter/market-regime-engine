# SPDX-License-Identifier: Apache-2.0
"""Fixed-Income trading-day calendars (REVIEW.md §3.4 Q-8).

The macro-side ``release_calendar.py`` family tracks BLS / BEA / Fed
*release* dates — when an economic series prints. FI needs the
complementary trading-day calendar — when bond markets are open for
trades. Federal holidays differ from market closures (e.g. Veterans
Day is a federal closure but historically NYSE bond markets were
open).

This module loads a hand-curated YAML at
``data/calendars/sifma_bond.yaml`` (2020-2030) into a frozen
:class:`_CalendarCache`. Optional :mod:`pandas_market_calendars` is
runtime-detected: when installed and the calendar name resolves, the
loader prefers ``pmc`` as the source of truth and falls back to the
YAML cache otherwise. The YAML cache is the single source of truth
on a vanilla install (``pmc`` is in the ``[fixed_income]`` extra).

PIT integration: :func:`assert_trading_day` raises
:class:`market_regime_engine.fixed_income.pit_guard.PitViolationError`
on closed days. FI feature builders call this helper before scoring
so a trade timestamp on Christmas does not silently feed the model.

Refresh: set ``MRE_FI_CALENDAR_REFRESH=1`` before importing this
module (or calling :func:`reset_calendar_cache`) to force a reload of
the YAML. The cache is otherwise loaded once per process.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

log = logging.getLogger(__name__)

__all__ = [
    "TradingCalendar",
    "assert_trading_day",
    "is_trading_day",
    "next_trading_day",
    "previous_trading_day",
    "reset_calendar_cache",
    "trading_days_between",
]


class TradingCalendar(str, Enum):
    """Enumerated FI trading calendars.

    - :attr:`SIFMA_BOND` — SIFMA US Treasury / corporate bond
      recommended close schedule. Default for every FI feature builder.
    - :attr:`NYSE_BOND` — NYSE bond market schedule. Currently aliased
      to the SIFMA YAML (the two diverge only on Veterans Day; we
      treat that divergence as a v1.5.1 concern).
    - :attr:`FEDERAL` — US federal holiday schedule (existing macro
      calendar). Use for Treasury-auction interactions where the
      federal observance matters more than the market schedule.
    """

    SIFMA_BOND = "sifma_bond"
    NYSE_BOND = "nyse_bond"
    FEDERAL = "federal"


_CALENDAR_DIR = Path(__file__).resolve().parents[3] / "data" / "calendars"


@dataclass(frozen=True)
class _CalendarSnapshot:
    """Materialised closures for a single calendar.

    Stored as a frozen set of ``pd.Timestamp`` (date-normalised to
    midnight) so membership lookups are O(1). ``early_closes`` is kept
    separately because SIFMA treats those days as trading days with a
    shortened session — feature timestamps on early-close days must
    pass PIT enforcement.
    """

    name: str
    closures: frozenset[pd.Timestamp]
    early_closes: frozenset[pd.Timestamp]
    metadata: dict[str, Any] = field(default_factory=dict)


_CACHE: dict[TradingCalendar, _CalendarSnapshot] = {}
_CACHE_LOCK = threading.Lock()


def _normalize_date(value: Any) -> pd.Timestamp:
    """Coerce ``value`` to a midnight-aligned naive ``pd.Timestamp``.

    Calendar membership is date-based, so we strip the time component
    after parsing. Timezone-aware inputs are converted to UTC first so
    a late-evening ET timestamp doesn't roll the date forward on
    storage.
    """
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return pd.Timestamp(year=ts.year, month=ts.month, day=ts.day)


def _yaml_path(calendar: TradingCalendar) -> Path:
    if calendar in (TradingCalendar.SIFMA_BOND, TradingCalendar.NYSE_BOND):
        return _CALENDAR_DIR / "sifma_bond.yaml"
    if calendar is TradingCalendar.FEDERAL:
        return _CALENDAR_DIR / "federal.yaml"
    raise ValueError(f"unknown calendar: {calendar!r}")


def _load_from_yaml(calendar: TradingCalendar) -> _CalendarSnapshot:
    path = _yaml_path(calendar)
    if not path.exists():
        log.warning("calendar YAML missing for %s at %s; empty calendar loaded", calendar, path)
        return _CalendarSnapshot(name=calendar.value, closures=frozenset(), early_closes=frozenset())
    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    years = payload.get("years", [])
    closures: set[pd.Timestamp] = set()
    early: set[pd.Timestamp] = set()
    for entry in years:
        for c in entry.get("closures", []) or []:
            closures.add(_normalize_date(c["date"]))
        for c in entry.get("early_closes", []) or []:
            early.add(_normalize_date(c["date"]))
    return _CalendarSnapshot(
        name=calendar.value,
        closures=frozenset(closures),
        early_closes=frozenset(early),
        metadata={"source": "yaml", "path": str(path)},
    )


def _load_from_pmc(calendar: TradingCalendar) -> _CalendarSnapshot | None:
    """Optional :mod:`pandas_market_calendars` adapter.

    Returns ``None`` when the package isn't installed, the calendar
    name does not resolve, or the lookup fails for any reason. The
    YAML cache always wins as the *baseline*; ``pmc`` is consulted only
    when explicitly opted-in via the ``MRE_FI_USE_PMC=1`` env var so a
    stale ``pmc`` install on a developer laptop can never override the
    audit-grade YAML reference.
    """
    if os.environ.get("MRE_FI_USE_PMC", "0") not in {"1", "true", "TRUE"}:
        return None
    try:
        import pandas_market_calendars as pmc
    except ImportError:
        return None
    pmc_name = (
        "SIFMA_US" if calendar in (TradingCalendar.SIFMA_BOND, TradingCalendar.NYSE_BOND) else "us_federal_government"
    )
    try:
        cal = pmc.get_calendar(pmc_name)
        holidays = pd.DatetimeIndex(cal.holidays().holidays)
    except Exception as exc:  # pragma: no cover - third-party errors
        log.warning("pandas_market_calendars load failed for %s: %s", pmc_name, exc)
        return None
    closures = frozenset(_normalize_date(d) for d in holidays)
    return _CalendarSnapshot(
        name=calendar.value,
        closures=closures,
        early_closes=frozenset(),
        metadata={"source": "pandas_market_calendars", "calendar": pmc_name},
    )


def _ensure_loaded(calendar: TradingCalendar) -> _CalendarSnapshot:
    """Lazy load + cache the calendar snapshot.

    Refresh via :func:`reset_calendar_cache` or by setting the env
    ``MRE_FI_CALENDAR_REFRESH=1`` before first use. Loader prefers
    ``pmc`` when opted in (and available), else the YAML cache.
    """
    if os.environ.get("MRE_FI_CALENDAR_REFRESH", "0") in {"1", "true", "TRUE"}:
        reset_calendar_cache()
    with _CACHE_LOCK:
        cached = _CACHE.get(calendar)
        if cached is not None:
            return cached
        snapshot = _load_from_pmc(calendar) or _load_from_yaml(calendar)
        _CACHE[calendar] = snapshot
        return snapshot


def reset_calendar_cache() -> None:
    """Drop the in-process snapshot cache; next read reloads from disk.

    Intended for tests that mutate the YAML or for the
    ``MRE_FI_CALENDAR_REFRESH=1`` operator path. Thread-safe.
    """
    with _CACHE_LOCK:
        _CACHE.clear()


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def is_trading_day(
    date: pd.Timestamp | str,
    calendar: TradingCalendar = TradingCalendar.SIFMA_BOND,
) -> bool:
    """Return ``True`` if ``date`` is a trading day on ``calendar``.

    Weekends are never trading days; early-close days *are* trading
    days (the session is shortened, not closed). Date-only semantics:
    a non-midnight timestamp is truncated to the date before the
    lookup.
    """
    snap = _ensure_loaded(calendar)
    ts = _normalize_date(date)
    if ts.weekday() >= 5:  # Sat/Sun
        return False
    return ts not in snap.closures


def next_trading_day(
    date: pd.Timestamp | str,
    calendar: TradingCalendar = TradingCalendar.SIFMA_BOND,
) -> pd.Timestamp:
    """Smallest trading day strictly *after* ``date``.

    Walks one calendar day at a time and skips weekends + closures.
    Returns a midnight-aligned naive timestamp matching the on-disk
    YAML format. Defensive upper bound of 30 days so a corrupt YAML
    cannot lock the caller in an infinite loop; raises
    :class:`RuntimeError` if exceeded.
    """
    ts = _normalize_date(date)
    for _ in range(30):
        ts = ts + pd.Timedelta(days=1)
        if is_trading_day(ts, calendar):
            return ts
    raise RuntimeError(f"no trading day found within 30 days of {date!r} on {calendar.value}")


def previous_trading_day(
    date: pd.Timestamp | str,
    calendar: TradingCalendar = TradingCalendar.SIFMA_BOND,
) -> pd.Timestamp:
    """Largest trading day strictly *before* ``date``.

    Symmetric to :func:`next_trading_day`.
    """
    ts = _normalize_date(date)
    for _ in range(30):
        ts = ts - pd.Timedelta(days=1)
        if is_trading_day(ts, calendar):
            return ts
    raise RuntimeError(f"no trading day found within 30 days before {date!r} on {calendar.value}")


def trading_days_between(
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    calendar: TradingCalendar = TradingCalendar.SIFMA_BOND,
) -> pd.DatetimeIndex:
    """Inclusive trading-day index on ``[start, end]``.

    Useful for materialising rolling lookback windows in FI feature
    builders. Returns an empty index when ``end < start``.
    """
    start_ts = _normalize_date(start)
    end_ts = _normalize_date(end)
    if end_ts < start_ts:
        return pd.DatetimeIndex([])
    snap = _ensure_loaded(calendar)
    weekdays = pd.bdate_range(start_ts, end_ts, freq="C", weekmask="Mon Tue Wed Thu Fri")
    return pd.DatetimeIndex([d for d in weekdays if d not in snap.closures])


def assert_trading_day(
    timestamp: pd.Timestamp | str,
    calendar: TradingCalendar = TradingCalendar.SIFMA_BOND,
    *,
    label: str = "timestamp",
) -> None:
    """Raise :class:`PitViolationError` if ``timestamp`` is closed.

    Used inside FI feature builders to reject rows that report on a
    closed-market day; per AGENT.md non-negotiable constraint 5 these
    are always operational errors and must surface release_gate=false
    rather than be silently aggregated.
    """
    # Late import keeps the pit_guard <-> calendars relationship one-directional
    # at import time (pit_guard does not import calendars).
    from market_regime_engine.fixed_income.pit_guard import PitViolationError

    if not is_trading_day(timestamp, calendar):
        raise PitViolationError(
            f"PIT violation for {label}: {pd.Timestamp(timestamp).isoformat()} is not a "
            f"trading day on calendar {calendar.value}"
        )


def closed_days(
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    calendar: TradingCalendar = TradingCalendar.SIFMA_BOND,
) -> Iterable[pd.Timestamp]:
    """Iterator over closed days (closures + weekends) on ``[start, end]``.

    Returned in ascending order. Convenience for the report writer.
    """
    snap = _ensure_loaded(calendar)
    s = _normalize_date(start)
    e = _normalize_date(end)
    cur = s
    while cur <= e:
        if cur.weekday() >= 5 or cur in snap.closures:
            yield cur
        cur = cur + pd.Timedelta(days=1)
