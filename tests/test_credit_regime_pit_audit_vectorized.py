# SPDX-License-Identifier: Apache-2.0
"""Regression — vectorised ``_audit_pit`` / ``_enforce_pit_and_calendar``.

Pre-fix (REVIEW.md Tier-2 A2): both helpers ran a per-row
``features.iterrows()`` loop calling :func:`assert_pit_safe`. At
200k rows the loop spent hundreds of ms; the vectorised
:func:`audit_pit_dataframe` runs in O(few ms) per column comparison.

Post-fix: ``_audit_pit`` (credit_spread_regime.py) and
``_enforce_pit_and_calendar`` (feature_builders.py) route through
``audit_pit_dataframe`` and raise :class:`PitViolationError` on any
non-zero violation count. We verify accept/reject semantics match
the legacy iterrows path on 50 seeded synthetic frames, and (under
``@pytest.mark.slow``) that the 200k-row path stays under 500 ms.
"""

from __future__ import annotations

import random
import time

import pandas as pd
import pytest

import market_regime_engine.fixed_income  # noqa: F401  - register FI tables
from market_regime_engine.fixed_income.credit_spread_regime import _audit_pit
from market_regime_engine.fixed_income.pit_guard import (
    PitViolationError,
    assert_pit_safe,
)


def _legacy_audit_pit(features: pd.DataFrame, *, asof: pd.Timestamp) -> bool:
    """The pre-fix per-row loop. Returns True iff the audit raises."""
    if "source_timestamp" not in features.columns:
        return False
    try:
        for _, row in features.iterrows():
            source = pd.Timestamp(row["source_timestamp"])
            if source.tzinfo is None:
                source = source.tz_localize("UTC")
            vintage = row.get("vintage_date")
            if vintage is not None and not pd.isna(vintage):
                vintage_ts = pd.Timestamp(vintage)
                if vintage_ts.tzinfo is None:
                    vintage_ts = vintage_ts.tz_localize("UTC")
            else:
                vintage_ts = None
            assert_pit_safe(
                feature_timestamp=source,
                decision_timestamp=asof,
                vintage_timestamp=vintage_ts,
                label=str(row.get("feature_name", "feature")),
            )
    except PitViolationError:
        return True
    return False


def _new_audit_pit(features: pd.DataFrame, *, asof: pd.Timestamp) -> bool:
    """The post-fix vectorised helper. Returns True iff it raises."""
    try:
        _audit_pit(features, asof=asof)
    except PitViolationError:
        return True
    return False


def _build_seeded_frame(seed: int, asof: pd.Timestamp) -> pd.DataFrame:
    """Build a small, structurally-varied feature frame for parity tests."""
    rng = random.Random(seed)
    rows: list[dict] = []
    n = rng.randint(0, 20)
    for i in range(n):
        # Mix tz-aware and tz-naive timestamps; cover both sides of asof.
        days_offset = rng.randint(-5, 5)
        source = asof + pd.Timedelta(days=days_offset, hours=rng.randint(-12, 12))
        if rng.random() < 0.5:
            source_value: pd.Timestamp | str = source
        else:
            # tz-naive form — exercises the localise-on-the-fly path.
            source_value = source.tz_convert(None)
        has_vintage = rng.random() < 0.7
        if has_vintage:
            vintage = asof + pd.Timedelta(days=rng.randint(-30, 5))
            vintage_value: pd.Timestamp | None = vintage
        else:
            vintage_value = None
        rows.append(
            {
                "feature_name": f"feature_{i}",
                "source_timestamp": source_value,
                "vintage_date": vintage_value,
            }
        )
    return pd.DataFrame(rows)


def test_vectorized_audit_matches_legacy_iterrows_on_seeded_inputs() -> None:
    """Property test: 50 seeded synthetic frames; the legacy iterrows
    path and the vectorised path agree on accept / reject for every
    one. ``_audit_pit`` is allowed to ALSO raise on rows the legacy
    helper would have raised on; what we forbid is a divergence in
    the raise vs. don't-raise outcome."""
    asof = pd.Timestamp("2026-05-08T16:00:00+00:00")
    for seed in range(50):
        frame = _build_seeded_frame(seed, asof)
        legacy = _legacy_audit_pit(frame, asof=asof)
        new = _new_audit_pit(frame, asof=asof)
        assert legacy == new, (
            f"seed={seed} legacy_raises={legacy} vectorised_raises={new}; "
            f"divergence in accept/reject semantics"
        )


def test_vectorized_audit_empty_frame_is_a_pass() -> None:
    """Empty input is a clean PASS — no rows means no violations to
    report."""
    asof = pd.Timestamp("2026-05-08T16:00:00+00:00")
    _audit_pit(pd.DataFrame(), asof=asof)  # must not raise
    _audit_pit(
        pd.DataFrame({"source_timestamp": pd.Series(dtype="datetime64[ns, UTC]")}),
        asof=asof,
    )


def test_vectorized_audit_missing_source_timestamp_column_is_a_pass() -> None:
    """Frames without ``source_timestamp`` are skipped per the legacy
    contract — the FI scoring layer constructs the column upstream;
    missing it means the caller is using a different feature shape."""
    asof = pd.Timestamp("2026-05-08T16:00:00+00:00")
    frame = pd.DataFrame({"feature_name": ["x", "y"], "value": [1.0, 2.0]})
    _audit_pit(frame, asof=asof)  # must not raise


def test_vectorized_audit_raises_with_helpful_label() -> None:
    """The raise message must surface the first violating row's
    feature_name + reason so the operator can attribute the failure."""
    asof = pd.Timestamp("2026-05-08T16:00:00+00:00")
    frame = pd.DataFrame(
        [
            {
                "feature_name": "future_leak",
                "source_timestamp": asof + pd.Timedelta(days=1),
                "vintage_date": asof,
            }
        ]
    )
    with pytest.raises(PitViolationError) as exc_info:
        _audit_pit(frame, asof=asof)
    msg = str(exc_info.value)
    assert "future_leak" in msg
    assert "feature_after_decision" in msg


@pytest.mark.slow
def test_vectorized_audit_perf_at_200k_rows_under_500ms() -> None:
    """At 200k rows the vectorised path must complete in well under
    500 ms — the iterrows path historically took hundreds of ms even
    on the happy path."""
    asof = pd.Timestamp("2026-05-08T16:00:00+00:00")
    n = 200_000
    source_ts = asof - pd.Timedelta(days=1)
    frame = pd.DataFrame(
        {
            "feature_name": [f"feat_{i}" for i in range(n)],
            "source_timestamp": [source_ts] * n,
            "vintage_date": [source_ts] * n,
        }
    )
    start = time.perf_counter()
    _audit_pit(frame, asof=asof)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert elapsed_ms < 500.0, (
        f"vectorised _audit_pit took {elapsed_ms:.1f} ms on 200k rows; "
        f"target is < 500 ms"
    )
