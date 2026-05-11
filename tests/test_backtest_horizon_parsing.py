# SPDX-License-Identifier: Apache-2.0
"""PR-5 AF-10: cadence-aware horizon parsing.

The pre-PR-5 ``_parse_horizon_months`` accepted any ``"N<unit>"`` string and
silently coerced the integer ``N`` into months — so ``"15min"`` was parsed
as ``15`` months and ``"1d"`` was parsed as ``1`` month. That blew up the
FI execution-confidence intraday cadence and any future daily-cadence
backtest. PR-5 introduces :func:`_parse_horizon_periods` that requires an
explicit ``cadence=`` parameter and rejects mismatched suffixes.

The legacy ``_parse_horizon_months`` is kept as a deprecation shim so the
macro backtest call sites compile unchanged.
"""

from __future__ import annotations

import warnings

import pytest

from market_regime_engine.backtest import (
    _parse_horizon_months,
    _parse_horizon_periods,
)


# --------------------------------------------------------------------------
# new cadence-aware parser
# --------------------------------------------------------------------------


def test_parse_horizon_periods_monthly() -> None:
    assert _parse_horizon_periods("12m", cadence="monthly") == 12
    assert _parse_horizon_periods("3mo", cadence="monthly") == 3
    assert _parse_horizon_periods("6month", cadence="monthly") == 6
    assert _parse_horizon_periods("9months", cadence="monthly") == 9


def test_parse_horizon_periods_daily() -> None:
    assert _parse_horizon_periods("1d", cadence="daily") == 1
    assert _parse_horizon_periods("5day", cadence="daily") == 5
    assert _parse_horizon_periods("10days", cadence="daily") == 10
    # 1 week = 5 trading days (documented conversion via _HORIZON_UNITS).
    assert _parse_horizon_periods("1w", cadence="daily") == 5
    assert _parse_horizon_periods("2week", cadence="daily") == 10


def test_parse_horizon_periods_intraday() -> None:
    assert _parse_horizon_periods("15min", cadence="intraday") == 15
    assert _parse_horizon_periods("60min", cadence="intraday") == 60
    # 1h = 60 minutes (documented conversion).
    assert _parse_horizon_periods("1h", cadence="intraday") == 60
    assert _parse_horizon_periods("2hour", cadence="intraday") == 120
    # Bare numeric: interpreted in the cadence's natural unit.
    assert _parse_horizon_periods("30", cadence="intraday") == 30


def test_parse_horizon_periods_rejects_cadence_mismatch() -> None:
    # ``"12m"`` is months — invalid under daily cadence.
    with pytest.raises(ValueError, match="not valid for cadence"):
        _parse_horizon_periods("12m", cadence="daily")
    # ``"1d"`` is days — invalid under monthly cadence.
    with pytest.raises(ValueError, match="not valid for cadence"):
        _parse_horizon_periods("1d", cadence="monthly")
    # ``"15min"`` is minutes — invalid under monthly cadence.
    with pytest.raises(ValueError, match="not valid for cadence"):
        _parse_horizon_periods("15min", cadence="monthly")


def test_parse_horizon_periods_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _parse_horizon_periods("", cadence="monthly")
    with pytest.raises(ValueError, match="positive integer"):
        _parse_horizon_periods("0d", cadence="daily")
    with pytest.raises(ValueError, match="not of the form"):
        _parse_horizon_periods("foo", cadence="monthly")
    with pytest.raises(ValueError, match="unknown cadence"):
        _parse_horizon_periods("12m", cadence="unknown")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# legacy shim — back-compat with v1.4 macro backtest
# --------------------------------------------------------------------------


def test_parse_horizon_months_legacy_shim_emits_deprecation_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = _parse_horizon_months("12m")
    assert out == 12
    assert any(
        isinstance(w.message, DeprecationWarning)
        and "_parse_horizon_months is deprecated" in str(w.message)
        for w in caught
    )


def test_parse_horizon_months_legacy_shim_preserves_fallback_on_garbage_input() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert _parse_horizon_months(None) == 1  # type: ignore[arg-type]
        assert _parse_horizon_months("foo", fallback=7) == 7
