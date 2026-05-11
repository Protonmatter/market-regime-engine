# SPDX-License-Identifier: Apache-2.0
"""SIFMA bond-market calendar tests (PR-3 task D, REVIEW.md §3.4 Q-8)."""

from __future__ import annotations

import pandas as pd
import pytest

from market_regime_engine.fixed_income.calendars import (
    TradingCalendar,
    assert_trading_day,
    closed_days,
    is_trading_day,
    next_trading_day,
    previous_trading_day,
    reset_calendar_cache,
    trading_days_between,
)
from market_regime_engine.fixed_income.pit_guard import PitViolationError


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    reset_calendar_cache()
    yield
    reset_calendar_cache()


def test_is_trading_day_excludes_federal_holidays() -> None:
    """SIFMA bond market is closed on the federal holidays from the YAML."""
    # Per the 2026 YAML.
    assert not is_trading_day("2026-01-01")  # New Year's Day
    assert not is_trading_day("2026-02-16")  # Presidents' Day
    assert not is_trading_day("2026-05-25")  # Memorial Day
    assert not is_trading_day("2026-07-03")  # Independence Day observed (Jul 4 Sat)
    assert not is_trading_day("2026-12-25")  # Christmas
    assert not is_trading_day("2025-04-18")  # Good Friday 2025


def test_is_trading_day_handles_weekends() -> None:
    """Saturdays and Sundays are never trading days."""
    # 2026-05-09 is a Saturday; 2026-05-10 is a Sunday.
    assert not is_trading_day("2026-05-09")
    assert not is_trading_day("2026-05-10")
    # 2026-05-11 is a Monday with no closures — trading day.
    assert is_trading_day("2026-05-11")


def test_is_trading_day_sifma_early_closes_treated_as_trading_days() -> None:
    """Early-close days have shortened sessions but are open for trading."""
    # 2026-04-02 = day before Good Friday early close per the YAML.
    assert is_trading_day("2026-04-02")
    # 2026-07-02 = day before Independence Day early close per the YAML.
    assert is_trading_day("2026-07-02")
    # 2026-12-24 = Christmas Eve early close per the YAML.
    assert is_trading_day("2026-12-24")


def test_next_trading_day_skips_holidays() -> None:
    """``next_trading_day`` skips weekend + closure runs."""
    # Friday before Memorial Day weekend 2026 → next is Tue 2026-05-26 (Mon
    # 5/25 is Memorial Day, Sat/Sun are weekends).
    assert next_trading_day("2026-05-22") == pd.Timestamp("2026-05-26")
    # Thursday before Christmas weekend 2026 → next trading day after 12/24
    # is Mon 12/28 because Fri 12/25 is Christmas, Sat/Sun follow.
    assert next_trading_day("2026-12-24") == pd.Timestamp("2026-12-28")


def test_previous_trading_day_skips_holidays() -> None:
    """``previous_trading_day`` symmetry."""
    # Day after Memorial Day weekend 2026 → previous is Fri 2026-05-22.
    assert previous_trading_day("2026-05-26") == pd.Timestamp("2026-05-22")
    # Day after Christmas observance 2026 → previous is Thu 12/24.
    assert previous_trading_day("2026-12-28") == pd.Timestamp("2026-12-24")


def test_trading_days_between_2023_q1_count() -> None:
    """Q1 2023 has 62 SIFMA trading days (NY observed + MLK + Presidents)."""
    idx = trading_days_between("2023-01-01", "2023-03-31")
    assert len(idx) == 62
    # Sanity checks on edges.
    assert pd.Timestamp("2023-01-02") not in idx  # NY Day observed
    assert pd.Timestamp("2023-01-03") in idx  # First open day
    assert pd.Timestamp("2023-01-16") not in idx  # MLK
    assert pd.Timestamp("2023-02-20") not in idx  # Presidents'
    assert pd.Timestamp("2023-03-31") in idx  # Fri end-of-quarter


def test_assert_trading_day_raises_on_closed_day() -> None:
    """Closed-day PIT enforcement surfaces ``PitViolationError``."""
    with pytest.raises(PitViolationError) as exc:
        assert_trading_day("2026-12-25", label="OAS_observation")
    assert "OAS_observation" in str(exc.value)
    assert "sifma_bond" in str(exc.value)


def test_assert_trading_day_no_raise_on_open_day() -> None:
    """Open day returns without error."""
    assert_trading_day("2026-05-11", label="OAS_observation")


def test_assert_trading_day_pit_guard_re_export() -> None:
    """The :mod:`pit_guard` re-export keeps the FI scoring imports tidy."""
    from market_regime_engine.fixed_income.pit_guard import assert_trading_day as guard_assert

    with pytest.raises(PitViolationError):
        guard_assert("2026-01-01", calendar=TradingCalendar.SIFMA_BOND)
    guard_assert("2026-05-11")


def test_closed_days_yields_in_order() -> None:
    """Iterator yields closures + weekends in ascending date order."""
    days = list(closed_days("2026-05-22", "2026-05-26"))
    assert days == [
        pd.Timestamp("2026-05-23"),  # Sat
        pd.Timestamp("2026-05-24"),  # Sun
        pd.Timestamp("2026-05-25"),  # Memorial Day
    ]


def test_trading_days_between_empty_when_end_before_start() -> None:
    assert len(trading_days_between("2026-05-15", "2026-05-10")) == 0
