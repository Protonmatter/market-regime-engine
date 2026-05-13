# SPDX-License-Identifier: Apache-2.0
"""v1.6.0 — A7 / Finding §3.2 regression tests.

Pin the contract that ``Warehouse.read_dealer_response_stats(window_start,
window_end)`` is the canonical accessor and that

1. its result matches the legacy ``read_dealer_response_stats()`` filtered
   by hand in pandas, and
2. it accepts datetime / pandas timestamp / ISO-string bounds
   interchangeably, and
3. ``build_execution_features`` consumes the new accessor instead of
   reaching through ``warehouse._backend`` to issue ``SELECT *``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import market_regime_engine.fixed_income  # noqa: F401  (table registration)
from market_regime_engine.storage import Warehouse


@pytest.fixture
def wh(tmp_path: Path) -> Warehouse:
    return Warehouse(tmp_path / "drs.duckdb")


def _seed(wh: Warehouse) -> pd.DataFrame:
    """Seed 100 dealer_response_stats rows spread across two years."""
    rows = []
    base = pd.Timestamp("2024-01-15T00:00:00Z")
    for i in range(100):
        window_end = base + pd.Timedelta(days=i * 7)  # weekly cadence
        window_start = window_end - pd.Timedelta(days=1)
        rows.append(
            {
                "dealer_id": f"DLR_{i % 5}",
                "window_start": window_start.isoformat(),
                "window_end": window_end.isoformat(),
                "requests": 100 + i,
                "responses": 80 + (i % 20),
                "avg_response_ms": float(250 + i),
                "metadata_json": "{}",
            }
        )
    frame = pd.DataFrame(rows)
    wh.write_dealer_response_stats(frame)
    return frame


def test_read_dealer_response_stats_full_table_unchanged(wh: Warehouse) -> None:
    """No-arg call still returns every row (back-compat with v1.5.x)."""
    seeded = _seed(wh)
    out = wh.read_dealer_response_stats()
    assert len(out) == len(seeded)


def test_read_dealer_response_stats_windowed_matches_pandas_filter(
    wh: Warehouse,
) -> None:
    """The indexed SQL path returns the same rows as the legacy full-
    table read + pandas filter — pinned per REVIEW_DEEP_V1_5_2.md A7."""
    _seed(wh)
    window_start = pd.Timestamp("2024-06-01T00:00:00Z")
    window_end = pd.Timestamp("2024-12-31T23:59:59Z")

    indexed = wh.read_dealer_response_stats(
        window_start=window_start, window_end=window_end
    )

    legacy_full = wh.read_dealer_response_stats()
    legacy_full = legacy_full.copy()
    legacy_full["window_end_ts"] = pd.to_datetime(
        legacy_full["window_end"], utc=True, errors="coerce"
    )
    legacy_filtered = legacy_full.loc[
        (legacy_full["window_end_ts"] >= window_start)
        & (legacy_full["window_end_ts"] <= window_end)
    ].drop(columns=["window_end_ts"])

    indexed_sorted = indexed.reset_index(drop=True).sort_values(
        ["dealer_id", "window_start"]
    ).reset_index(drop=True)
    legacy_sorted = legacy_filtered.reset_index(drop=True).sort_values(
        ["dealer_id", "window_start"]
    ).reset_index(drop=True)

    assert len(indexed_sorted) == len(legacy_sorted)
    assert indexed_sorted["dealer_id"].tolist() == legacy_sorted["dealer_id"].tolist()
    assert (
        indexed_sorted["window_end"].tolist()
        == legacy_sorted["window_end"].tolist()
    )


def test_read_dealer_response_stats_accepts_string_and_datetime_bounds(
    wh: Warehouse,
) -> None:
    """ISO strings, pandas Timestamps and datetimes are all valid bounds."""
    from datetime import UTC, datetime

    _seed(wh)
    ts_iso = "2024-06-01T00:00:00+00:00"
    ts_pd = pd.Timestamp(ts_iso)
    ts_dt = datetime(2024, 6, 1, tzinfo=UTC)

    via_iso = wh.read_dealer_response_stats(window_start=ts_iso)
    via_pd = wh.read_dealer_response_stats(window_start=ts_pd)
    via_dt = wh.read_dealer_response_stats(window_start=ts_dt)
    assert len(via_iso) == len(via_pd) == len(via_dt)


def test_build_execution_features_no_longer_pierces_backend_private_attr(
    wh: Warehouse,
) -> None:
    """REVIEW_DEEP_V1_5_2.md A7: the build_execution_features call site
    must NOT call ``warehouse._backend.read_sql``. The public
    ``warehouse.read_dealer_response_stats`` accessor is the contract."""
    import inspect

    from market_regime_engine.fixed_income.execution_confidence import (
        build_execution_features,
    )

    src = inspect.getsource(build_execution_features)
    assert "warehouse._backend.read_sql" not in src
    assert "read_dealer_response_stats" in src
