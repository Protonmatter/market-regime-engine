# SPDX-License-Identifier: Apache-2.0
"""Per-column NaN policy unit tests (PR-3 ASK-5 / AF-8).

The default :attr:`NanPolicy.NAN_TO_ZERO` must be bit-for-bit
identical to the legacy ``frame.replace([inf, -inf], NaN).ffill().fillna(0.0)``
cleaner so the bocpd / MSVAR / GP-CPD numerical fixtures keep
passing untouched. The three other policies cover the FI use cases:
forward-fill only, drop-rows, fail-closed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.frontier.data_cleaning import (
    NanPolicy,
    PitAuditFailure,
    clean_with_policy,
)


def _ref_legacy_cleaner(frame: pd.DataFrame) -> pd.DataFrame:
    """The pre-PR-3 cleaner — must remain bit-equal to ``NAN_TO_ZERO``."""
    return frame.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)


def test_clean_with_policy_default_nan_to_zero_preserves_monthly_behavior() -> None:
    """Default policy matches the legacy ``ffill().fillna(0.0)`` cleaner."""
    frame = pd.DataFrame(
        {
            "labor": [np.nan, -0.5, 0.1, np.nan, 0.3],
            "rates": [0.2, np.inf, -0.1, 0.5, np.nan],
            "credit": [np.nan, np.nan, 0.0, -np.inf, 0.4],
        }
    )
    cleaned = clean_with_policy(frame)
    ref = _ref_legacy_cleaner(frame)
    pd.testing.assert_frame_equal(cleaned, ref)


def test_clean_with_policy_nan_to_last_valid() -> None:
    """``NAN_TO_LAST_VALID`` forward-fills but does NOT seed the first row with 0."""
    frame = pd.DataFrame({"oas": [np.nan, 100.0, np.nan, 120.0, np.nan]})
    cleaned = clean_with_policy(frame, default_policy=NanPolicy.NAN_TO_LAST_VALID)
    assert pd.isna(cleaned["oas"].iloc[0])
    assert cleaned["oas"].iloc[1] == 100.0
    assert cleaned["oas"].iloc[2] == 100.0
    assert cleaned["oas"].iloc[3] == 120.0
    assert cleaned["oas"].iloc[4] == 120.0


def test_clean_with_policy_nan_drops_row() -> None:
    """``NAN_DROPS_ROW`` removes the row but keeps the other columns intact."""
    frame = pd.DataFrame(
        {
            "oas": [100.0, np.nan, 102.0, 103.0],
            "vix": [12.0, 13.0, 14.0, 15.0],
        }
    )
    cleaned = clean_with_policy(frame, default_policy=NanPolicy.NAN_DROPS_ROW)
    assert len(cleaned) == 3
    assert list(cleaned["oas"].to_numpy()) == [100.0, 102.0, 103.0]
    assert list(cleaned["vix"].to_numpy()) == [12.0, 14.0, 15.0]


def test_clean_with_policy_nan_fails_pit_audit_raises() -> None:
    """``NAN_FAILS_PIT_AUDIT`` raises with the offending column name."""
    frame = pd.DataFrame(
        {
            "oas": [100.0, 101.0, 102.0],
            "vix": [np.nan, 13.0, 14.0],
            "move": [70.0, 71.0, np.nan],
        }
    )
    with pytest.raises(PitAuditFailure) as excinfo:
        clean_with_policy(frame, default_policy=NanPolicy.NAN_FAILS_PIT_AUDIT)
    msg = str(excinfo.value)
    assert "vix" in msg
    assert "move" in msg
    assert "oas" not in msg


def test_clean_with_policy_per_column_overrides() -> None:
    """``column_policies`` overrides the default per-column.

    Set ``vix`` to ``NAN_TO_LAST_VALID`` and ``oas`` to ``NAN_DROPS_ROW``;
    the other columns inherit the default ``NAN_TO_ZERO``.
    """
    frame = pd.DataFrame(
        {
            "oas": [100.0, np.nan, 102.0],
            "vix": [np.nan, 13.0, 14.0],
            "ust_slope": [0.5, np.nan, 0.7],
        }
    )
    cleaned = clean_with_policy(
        frame,
        default_policy=NanPolicy.NAN_TO_ZERO,
        column_policies={
            "vix": NanPolicy.NAN_TO_LAST_VALID,
            "oas": NanPolicy.NAN_DROPS_ROW,
        },
    )
    # Row 1 dropped because ``oas`` is NaN; only rows 0 and 2 survive.
    assert len(cleaned) == 2
    assert cleaned["oas"].tolist() == [100.0, 102.0]
    # vix used NAN_TO_LAST_VALID — row 0 has no prior value so stays NaN.
    assert pd.isna(cleaned["vix"].iloc[0])
    assert cleaned["vix"].iloc[1] == 14.0
    # ust_slope used the default zero policy — row 1 was dropped so we see
    # only rows 0 and 2 (no NaN remains).
    assert cleaned["ust_slope"].tolist() == [0.5, 0.7]


def test_clean_with_policy_inf_coerced_to_nan_first() -> None:
    """``+/-inf`` are coerced to NaN before any policy fires.

    Under ``NAN_FAILS_PIT_AUDIT`` an ``inf`` in any column raises;
    under ``NAN_TO_ZERO`` it becomes 0.0 just like the legacy cleaner.
    """
    inf_frame = pd.DataFrame({"x": [1.0, np.inf, 3.0], "y": [-np.inf, 2.0, 3.0]})

    with pytest.raises(PitAuditFailure):
        clean_with_policy(inf_frame, default_policy=NanPolicy.NAN_FAILS_PIT_AUDIT)

    cleaned = clean_with_policy(inf_frame, default_policy=NanPolicy.NAN_TO_ZERO)
    # Default zero policy seeds the leading -inf row with 0 and ffills the inf row.
    assert cleaned["x"].tolist() == [1.0, 1.0, 3.0]
    assert cleaned["y"].tolist() == [0.0, 2.0, 3.0]


def test_clean_with_policy_empty_frame_round_trips() -> None:
    """Empty frame returns empty without raising."""
    out = clean_with_policy(pd.DataFrame(), default_policy=NanPolicy.NAN_FAILS_PIT_AUDIT)
    assert out.empty


def test_clean_with_policy_no_nan_inputs_round_trip() -> None:
    """Inputs without NaN/Inf are returned bit-equal under every policy."""
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    for policy in NanPolicy:
        cleaned = clean_with_policy(frame, default_policy=policy)
        pd.testing.assert_frame_equal(cleaned, frame)
