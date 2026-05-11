# SPDX-License-Identifier: Apache-2.0
"""PR-1 acceptance tests for sub-daily as-of grid support (REVIEW.md ASK-2)."""

from __future__ import annotations

import pandas as pd
import pytest

from market_regime_engine.asof import Cadence, _resolve_asof_grid


@pytest.fixture
def vintage_frame() -> pd.DataFrame:
    """Synthetic vintage panel covering ~3 months at daily resolution."""
    obs = pd.date_range("2024-01-01", "2024-03-31", freq="D")
    return pd.DataFrame(
        {
            "vintage_date": obs,
            "observation_date": obs,
        }
    )


def test_monthly_default_preserves_existing_behavior(vintage_frame: pd.DataFrame) -> None:
    """Calling without ``freq`` returns the v1.4 month-start grid."""
    grid_default = _resolve_asof_grid(vintage_frame, min_history_months=0)
    grid_explicit = _resolve_asof_grid(vintage_frame, min_history_months=0, freq="MS")
    assert grid_default == grid_explicit
    # 2024-01-01, 2024-02-01, 2024-03-01 are all month-starts inside the range.
    expected = {pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01"), pd.Timestamp("2024-03-01")}
    assert expected.issubset(set(grid_default))
    # Every entry must be midnight-normalised (legacy v1.4 invariant).
    assert all(ts.hour == 0 and ts.minute == 0 and ts.second == 0 for ts in grid_default)


def test_daily_grid_generates_business_day_count(vintage_frame: pd.DataFrame) -> None:
    """``freq="D"`` produces ~91 calendar-day entries over Jan-Mar 2024."""
    grid = _resolve_asof_grid(vintage_frame, min_history_months=0, freq="D")
    # 2024 is a leap year; Jan(31) + Feb(29) + Mar(31) = 91 days total.
    assert len(grid) == 91
    assert grid[0] == pd.Timestamp("2024-01-01")
    assert grid[-1] == pd.Timestamp("2024-03-31")
    # Daily cadence is day-grained so .normalize() still applies.
    assert all(ts.hour == 0 for ts in grid)


def test_intraday_15min_grid_works(vintage_frame: pd.DataFrame) -> None:
    """``freq="15min"`` produces a high-resolution grid, time component preserved."""
    grid = _resolve_asof_grid(vintage_frame, min_history_months=0, freq="15min")
    # 91 days × 96 quarter-hours/day = 8736; the first 15min slot is 00:00,
    # so the grid endpoints are 2024-01-01 00:00 and 2024-03-31 00:00.
    assert len(grid) == 8641
    assert grid[0] == pd.Timestamp("2024-01-01 00:00:00")
    # The grid must include at least one mid-day entry (proves time is
    # preserved rather than collapsed to midnight).
    times = {ts.time() for ts in grid[:10]}
    assert len(times) > 1, "15-min grid collapsed to a single time; preservation broke"


def test_cadence_enum_round_trip() -> None:
    """`Cadence.from_pandas_freq` round-trips canonical and legacy aliases."""
    assert Cadence.from_pandas_freq("MS") is Cadence.MONTHLY
    assert Cadence.from_pandas_freq("D") is Cadence.DAILY
    assert Cadence.from_pandas_freq("h") is Cadence.HOURLY
    assert Cadence.from_pandas_freq("H") is Cadence.HOURLY  # legacy upper-case
    assert Cadence.from_pandas_freq("15min") is Cadence.MIN_15
    assert Cadence.from_pandas_freq("15T") is Cadence.MIN_15
    assert Cadence.from_pandas_freq("1min") is Cadence.MIN_1
    assert Cadence.from_pandas_freq("min") is Cadence.MIN_1
    assert Cadence.from_pandas_freq("T") is Cadence.MIN_1
    with pytest.raises(ValueError):
        Cadence.from_pandas_freq("frobnicate")


def test_empty_vintage_returns_empty_grid() -> None:
    df = pd.DataFrame({"vintage_date": pd.Series(dtype="datetime64[ns]"), "observation_date": pd.Series(dtype="datetime64[ns]")})
    assert _resolve_asof_grid(df, min_history_months=0) == []
    assert _resolve_asof_grid(df, min_history_months=0, freq="D") == []
