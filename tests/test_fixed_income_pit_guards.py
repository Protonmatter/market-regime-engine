# SPDX-License-Identifier: Apache-2.0
"""PR-1 acceptance tests for the FI PIT guard helpers (AGENT.md "PIT guard rules")."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pandas as pd
import pytest

from market_regime_engine.fixed_income.pit_guard import (
    PitViolationError,
    assert_pit_safe,
    audit_pit_dataframe,
)


def test_assert_pit_safe_accepts_feature_before_decision() -> None:
    decision = datetime(2026, 5, 10, 16, 0, 0, tzinfo=UTC)
    feature = decision - timedelta(hours=1)
    assert_pit_safe(feature, decision)


def test_assert_pit_safe_rejects_feature_after_decision() -> None:
    decision = datetime(2026, 5, 10, 16, 0, 0, tzinfo=UTC)
    feature = decision + timedelta(seconds=1)
    with pytest.raises(PitViolationError) as exc:
        assert_pit_safe(feature, decision, label="OAS")
    assert "OAS" in str(exc.value)
    assert "after" in str(exc.value)


def test_assert_pit_safe_rejects_vintage_after_decision() -> None:
    decision = datetime(2026, 5, 10, 16, 0, 0, tzinfo=UTC)
    feature = decision - timedelta(hours=2)
    vintage = decision + timedelta(seconds=1)
    with pytest.raises(PitViolationError):
        assert_pit_safe(feature, decision, vintage_timestamp=vintage)


def test_assert_pit_safe_handles_naive_and_aware_timestamps() -> None:
    """Mixed tz-naive (interpreted as UTC) + tz-aware inputs both work."""
    decision_naive = datetime(2026, 5, 10, 16, 0, 0)
    feature_aware = datetime(2026, 5, 10, 15, 0, 0, tzinfo=UTC)
    assert_pit_safe(feature_aware, decision_naive)

    feature_eastern = datetime(2026, 5, 10, 11, 0, 0, tzinfo=timezone(timedelta(hours=-5)))
    assert_pit_safe(feature_eastern, datetime(2026, 5, 10, 16, 0, 0, tzinfo=UTC))

    feature_late_eastern = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone(timedelta(hours=-5)))
    decision_utc = datetime(2026, 5, 10, 16, 0, 0, tzinfo=UTC)
    with pytest.raises(PitViolationError):
        assert_pit_safe(feature_late_eastern + timedelta(minutes=1), decision_utc)


def test_audit_pit_dataframe_returns_offenders() -> None:
    """5 rows; 2 violate (one feature-after-decision, one vintage-after); 3 clean."""
    rows = [
        {"decision_ts": "2026-05-10 16:00", "feature_ts": "2026-05-10 14:00", "vintage_ts": "2026-05-10 13:00"},
        {"decision_ts": "2026-05-10 16:00", "feature_ts": "2026-05-10 17:00", "vintage_ts": "2026-05-10 13:00"},
        {"decision_ts": "2026-05-10 16:00", "feature_ts": "2026-05-10 13:00", "vintage_ts": "2026-05-10 18:00"},
        {"decision_ts": "2026-05-10 16:00", "feature_ts": "2026-05-10 12:00", "vintage_ts": "2026-05-10 12:00"},
        {"decision_ts": "2026-05-10 16:00", "feature_ts": "2026-05-10 15:30", "vintage_ts": "2026-05-10 15:00"},
    ]
    df = pd.DataFrame(rows)
    report = audit_pit_dataframe(
        df,
        decision_timestamp_col="decision_ts",
        feature_timestamp_col="feature_ts",
        vintage_timestamp_col="vintage_ts",
    )
    assert report.rows == 5
    assert report.violation_count == 2
    assert not report.passed
    assert report.status == "FAIL"
    reasons = report.violations["pit_violation_reason"].tolist()
    assert any("feature_after_decision" in r for r in reasons)
    assert any("vintage_after_decision" in r for r in reasons)


def test_audit_pit_dataframe_passes_on_clean_input() -> None:
    df = pd.DataFrame(
        [
            {"decision_ts": "2026-05-10 16:00", "feature_ts": "2026-05-10 12:00"},
            {"decision_ts": "2026-05-10 16:00", "feature_ts": "2026-05-10 14:00"},
        ]
    )
    report = audit_pit_dataframe(
        df,
        decision_timestamp_col="decision_ts",
        feature_timestamp_col="feature_ts",
    )
    assert report.violation_count == 0
    assert report.passed
    assert report.status == "PASS"


def test_audit_pit_dataframe_handles_empty_input() -> None:
    report = audit_pit_dataframe(
        pd.DataFrame(),
        decision_timestamp_col="decision_ts",
        feature_timestamp_col="feature_ts",
    )
    assert report.rows == 0
    assert report.violation_count == 0
    assert report.passed
