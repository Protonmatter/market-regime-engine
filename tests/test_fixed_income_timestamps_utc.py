# SPDX-License-Identifier: Apache-2.0
"""UTC enforcement at the FI boundary (PR-3 task E, REVIEW.md §3.4 Q-7)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pandas as pd
import pytest

from market_regime_engine.fixed_income.timestamps import assert_utc, iso8601_z, to_utc


def test_to_utc_naive_raises() -> None:
    """Naive datetimes (and naive strings) trigger ValueError at the boundary."""
    with pytest.raises(ValueError):
        to_utc(datetime(2026, 5, 11, 12, 0, 0))
    with pytest.raises(ValueError):
        to_utc("2026-05-11T12:00:00")
    with pytest.raises(ValueError):
        to_utc(pd.Timestamp("2026-05-11T12:00:00"))


def test_to_utc_aware_converts() -> None:
    """Aware ET / aware UTC inputs both produce a UTC ``pd.Timestamp``."""
    et = timezone(timedelta(hours=-4))
    aware_et = datetime(2026, 5, 11, 14, 0, 0, tzinfo=et)
    out = to_utc(aware_et)
    assert isinstance(out, pd.Timestamp)
    assert out.tzinfo is not None
    assert out.utcoffset() == pd.Timedelta(0)
    # 14:00 ET == 18:00 UTC
    assert out == pd.Timestamp("2026-05-11T18:00:00+00:00")


def test_to_utc_none_passthrough() -> None:
    assert to_utc(None) is None


def test_to_utc_string_parsed() -> None:
    """ISO-8601 'Z' and ``+HH:MM`` both round-trip to UTC."""
    out_z = to_utc("2026-05-11T18:00:00Z")
    out_off = to_utc("2026-05-11T14:00:00-04:00")
    assert out_z == pd.Timestamp("2026-05-11T18:00:00+00:00")
    assert out_off == pd.Timestamp("2026-05-11T18:00:00+00:00")


def test_assert_utc_raises_on_naive() -> None:
    """``assert_utc`` is strict: naive AND non-UTC aware both raise."""
    with pytest.raises(ValueError):
        assert_utc(pd.Timestamp("2026-05-11T18:00:00"), label="OAS_ts")
    et = timezone(timedelta(hours=-4))
    with pytest.raises(ValueError):
        assert_utc(pd.Timestamp("2026-05-11T14:00:00", tz=et), label="OAS_ts")
    # UTC passes.
    assert_utc(pd.Timestamp("2026-05-11T18:00:00+00:00"), label="OAS_ts")


def test_iso8601_z_format() -> None:
    """ISO-8601 emit uses explicit ``Z`` suffix and accepts UTC only."""
    ts = pd.Timestamp("2026-05-11T18:00:00+00:00")
    assert iso8601_z(ts) == "2026-05-11T18:00:00Z"
    # Microseconds are preserved.
    ts_us = pd.Timestamp("2026-05-11T18:00:00.123456+00:00")
    assert iso8601_z(ts_us).endswith("Z")
    assert "123456" in iso8601_z(ts_us)
    # Naive input raises.
    with pytest.raises(ValueError):
        iso8601_z(pd.Timestamp("2026-05-11T18:00:00"))


def test_to_utc_roundtrip_with_datetime_utc() -> None:
    """``datetime.UTC`` inputs round-trip."""
    dt = datetime(2026, 5, 11, 18, 0, 0, tzinfo=UTC)
    out = to_utc(dt)
    assert out == pd.Timestamp("2026-05-11T18:00:00+00:00")
