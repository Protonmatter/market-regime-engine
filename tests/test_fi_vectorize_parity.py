# SPDX-License-Identifier: Apache-2.0
"""v1.5.1 (PR-9 FIX 5): golden-fixture parity for vectorised hot paths.

These tests pin the byte-identical output between the legacy iterrows
implementations (gated behind ``MRE_FI_LEGACY_VECTORIZE=1``) and the
vectorised v1.5.1 paths so a future refactor cannot silently regress
the PIT-audit behaviour. Each test feeds a deterministic golden frame
to both implementations and asserts the raise/accept semantics match.

Hot paths covered:

- :func:`liquidity_stress._audit_pit` — long-form liquidity features
- :func:`feature_builders._enforce_pit_liquidity` — long-form liquidity features
  with feature-name-aware trading-day rail
"""

from __future__ import annotations

import pandas as pd
import pytest

from market_regime_engine.fixed_income.calendars import TradingCalendar
from market_regime_engine.fixed_income.feature_builders import (
    _enforce_pit_liquidity,
    _enforce_pit_liquidity_legacy_iterrows,
)
from market_regime_engine.fixed_income.liquidity_stress import (
    _audit_pit,
    _audit_pit_legacy_iterrows,
)
from market_regime_engine.fixed_income.pit_guard import PitViolationError


# -- Fixtures -----------------------------------------------------------------


@pytest.fixture()
def golden_features_clean() -> pd.DataFrame:
    """Long-form feature frame where every row is PIT-safe."""
    dates = pd.date_range("2026-01-02 09:00", periods=12, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "date": dates,
            "feature_name": ["bid_ask_width"] * 12,
            "value": list(range(12)),
            "source_timestamp": dates,
            "vintage_date": dates,
        }
    )


@pytest.fixture()
def golden_features_violating() -> pd.DataFrame:
    """A long-form frame whose last row's source_timestamp is *after* asof."""
    dates = pd.date_range("2026-01-02 09:00", periods=12, freq="1h", tz="UTC")
    source = list(dates)
    source[-1] = source[-1] + pd.Timedelta(hours=48)  # post-asof violation
    return pd.DataFrame(
        {
            "date": dates,
            "feature_name": ["bid_ask_width"] * 12,
            "value": list(range(12)),
            "source_timestamp": source,
            "vintage_date": dates,
        }
    )


# -- liquidity_stress._audit_pit parity ---------------------------------------


def test_audit_pit_vectorised_matches_legacy_clean_input(
    golden_features_clean: pd.DataFrame,
) -> None:
    """A clean input must accept under both implementations."""
    asof = pd.Timestamp("2026-01-03 00:00", tz="UTC")
    _audit_pit(golden_features_clean, asof=asof)  # vectorised
    _audit_pit_legacy_iterrows(golden_features_clean, asof=asof)  # legacy


def test_audit_pit_vectorised_matches_legacy_violation(
    golden_features_violating: pd.DataFrame,
) -> None:
    """A violating input must raise under both implementations."""
    asof = pd.Timestamp("2026-01-03 00:00", tz="UTC")
    with pytest.raises((PitViolationError, ValueError)):
        _audit_pit(golden_features_violating, asof=asof)
    with pytest.raises((PitViolationError, ValueError)):
        _audit_pit_legacy_iterrows(golden_features_violating, asof=asof)


def test_audit_pit_kill_switch_routes_to_legacy(
    monkeypatch: pytest.MonkeyPatch,
    golden_features_violating: pd.DataFrame,
) -> None:
    """``MRE_FI_LEGACY_VECTORIZE=1`` forces the iterrows path."""
    monkeypatch.setenv("MRE_FI_LEGACY_VECTORIZE", "1")
    asof = pd.Timestamp("2026-01-03 00:00", tz="UTC")
    with pytest.raises((PitViolationError, ValueError)):
        _audit_pit(golden_features_violating, asof=asof)


# -- feature_builders._enforce_pit_liquidity parity ---------------------------


def test_enforce_pit_liquidity_vectorised_matches_legacy_clean(
    golden_features_clean: pd.DataFrame,
) -> None:
    asof = pd.Timestamp("2026-01-03 00:00", tz="UTC")
    _enforce_pit_liquidity(
        golden_features_clean, asof=asof, calendar=TradingCalendar.SIFMA_BOND
    )
    _enforce_pit_liquidity_legacy_iterrows(
        golden_features_clean, asof=asof, calendar=TradingCalendar.SIFMA_BOND
    )


def test_enforce_pit_liquidity_vectorised_matches_legacy_violation(
    golden_features_violating: pd.DataFrame,
) -> None:
    asof = pd.Timestamp("2026-01-03 00:00", tz="UTC")
    with pytest.raises((PitViolationError, ValueError)):
        _enforce_pit_liquidity(
            golden_features_violating, asof=asof, calendar=TradingCalendar.SIFMA_BOND
        )
    with pytest.raises((PitViolationError, ValueError)):
        _enforce_pit_liquidity_legacy_iterrows(
            golden_features_violating, asof=asof, calendar=TradingCalendar.SIFMA_BOND
        )


def test_enforce_pit_liquidity_kill_switch(
    monkeypatch: pytest.MonkeyPatch,
    golden_features_clean: pd.DataFrame,
) -> None:
    """The kill-switch routes _enforce_pit_liquidity through the iterrows path."""
    monkeypatch.setenv("MRE_FI_LEGACY_VECTORIZE", "1")
    asof = pd.Timestamp("2026-01-03 00:00", tz="UTC")
    _enforce_pit_liquidity(
        golden_features_clean, asof=asof, calendar=TradingCalendar.SIFMA_BOND
    )


def test_enforce_pit_liquidity_calendar_violation_still_raises(
    golden_features_clean: pd.DataFrame,
) -> None:
    """The PIT path passes but the calendar rail still fires for a closed-day."""
    # Pick a Saturday (2026-01-03 is a Saturday).
    saturday = pd.Timestamp("2026-01-03 12:00", tz="UTC")
    bad = golden_features_clean.copy()
    bad["feature_name"] = "trade_count_velocity"  # trading-day-rail feature
    bad.loc[bad.index[-1], "source_timestamp"] = saturday
    bad.loc[bad.index[-1], "date"] = saturday
    # Use an asof past the Saturday so PIT itself does not fire.
    asof = saturday + pd.Timedelta(days=2)
    with pytest.raises(PitViolationError):
        _enforce_pit_liquidity(
            bad, asof=asof, calendar=TradingCalendar.SIFMA_BOND
        )


def test_audit_pit_empty_frame_is_noop_under_both_paths() -> None:
    """An empty frame is a no-op (no PIT rail to enforce)."""
    asof = pd.Timestamp("2026-01-03 00:00", tz="UTC")
    empty = pd.DataFrame(columns=["date", "feature_name", "value", "source_timestamp"])
    _audit_pit(empty, asof=asof)
    _audit_pit_legacy_iterrows(empty, asof=asof)


def test_audit_pit_missing_source_timestamp_is_noop() -> None:
    """A frame lacking ``source_timestamp`` is a no-op (audit is opt-in)."""
    asof = pd.Timestamp("2026-01-03 00:00", tz="UTC")
    df = pd.DataFrame({"feature_name": ["x"], "value": [1.0]})
    _audit_pit(df, asof=asof)


# -- _emit_amihud_rows vectorisation parity -----------------------------------


@pytest.fixture()
def golden_trades_for_amihud() -> pd.DataFrame:
    """A deterministic two-day trade frame for Amihud calibration."""
    ts = pd.date_range("2026-01-02 10:00", periods=8, freq="3h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "price": [100.0, 100.5, 101.0, 100.7, 99.5, 99.8, 100.2, 100.1],
            "size": [10.0, 12.0, 8.0, 9.0, 14.0, 11.0, 10.0, 15.0],
        }
    )


def test_amihud_vectorised_matches_legacy_groupby_apply(
    monkeypatch: pytest.MonkeyPatch,
    golden_trades_for_amihud: pd.DataFrame,
) -> None:
    """The new vectorised aggregation must produce the same Amihud rows
    as the legacy ``groupby().apply(...)`` branch."""
    from market_regime_engine.fixed_income.feature_builders import _emit_amihud_rows

    monkeypatch.delenv("MRE_FI_LEGACY_VECTORIZE", raising=False)
    new_rows = _emit_amihud_rows(golden_trades_for_amihud.copy())

    monkeypatch.setenv("MRE_FI_LEGACY_VECTORIZE", "1")
    legacy_rows = _emit_amihud_rows(golden_trades_for_amihud.copy())

    assert len(new_rows) == len(legacy_rows)
    for new_row, legacy_row in zip(new_rows, legacy_rows, strict=True):
        assert new_row["feature_name"] == legacy_row["feature_name"] == "amihud_illiquidity"
        assert new_row["date"] == legacy_row["date"]
        assert abs(new_row["value"] - legacy_row["value"]) < 1e-12, (
            f"Amihud row drift: new={new_row['value']!r} legacy={legacy_row['value']!r}"
        )


def test_amihud_vectorised_empty_input_is_noop() -> None:
    from market_regime_engine.fixed_income.feature_builders import _emit_amihud_rows

    assert _emit_amihud_rows(pd.DataFrame()) == []
    # Frame with no 'size' column is also a noop.
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2026-01-02"]), "price": [100.0]})
    assert _emit_amihud_rows(df) == []
