# SPDX-License-Identifier: Apache-2.0
"""Acceptance tests for bond_reference temporal versioning (PR-2 Q-4).

Q-4 in REVIEW.md §3.4: bonds that defaulted or were called are removed
from the current-universe snapshot but must remain available for any
as-of training query that ran before the failure. The PR-2 schema adds
``valid_from`` / ``valid_to`` plus nullable ``default_date`` /
``delisted_date`` columns so the survivorship cut is data-driven; the
``read_bond_reference_asof`` and ``read_bond_reference_history``
helpers in :mod:`market_regime_engine.storage` materialise the
read-time queries.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import market_regime_engine.fixed_income  # noqa: F401 - register FI tables
from market_regime_engine.storage import (
    Warehouse,
    read_bond_reference_asof,
    read_bond_reference_history,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _bond_cols() -> list[str]:
    return [
        "cusip",
        "valid_from",
        "valid_to",
        "ticker",
        "issuer",
        "sector",
        "rating",
        "issue_date",
        "maturity",
        "coupon",
        "currency",
        "country",
        "duration",
        "convexity",
        "amount_outstanding",
        "is_callable",
        "call_schedule_json",
        "default_date",
        "delisted_date",
        "metadata_json",
    ]


def _make_bond_rows() -> pd.DataFrame:
    """Three CUSIPs with overlapping temporal versions and
    survivorship-failure cases.

    - ``AAA`` — single version valid from 2026-01-01 (still active).
    - ``BBB`` — two versions: v1 valid 2026-01-01→2026-03-01, v2 valid
      2026-03-01→present.
    - ``CCC`` — one version valid 2026-01-01→present, but defaulted
      on 2026-04-01.
    - ``DDD`` — one version, currently active.
    """

    rows: list[dict] = []

    def _row(
        cusip: str,
        valid_from: str,
        valid_to: str | None,
        rating: str,
        default_date: str | None = None,
        delisted_date: str | None = None,
    ) -> dict:
        return {
            "cusip": cusip,
            "valid_from": valid_from,
            "valid_to": valid_to,
            "ticker": cusip,
            "issuer": f"{cusip} Inc.",
            "sector": "Financials",
            "rating": rating,
            "issue_date": "2020-01-01T00:00:00+00:00",
            "maturity": "2030-01-01T00:00:00+00:00",
            "coupon": 1.0,
            "currency": "USD",
            "country": "US",
            "duration": 7.0,
            "convexity": 50.0,
            "amount_outstanding": 1.0e9,
            "is_callable": 0,
            "call_schedule_json": "[]",
            "default_date": default_date,
            "delisted_date": delisted_date,
            "metadata_json": "{}",
        }

    rows.append(_row("AAA", "2026-01-01T00:00:00+00:00", None, "AA"))
    rows.append(_row("BBB", "2026-01-01T00:00:00+00:00", "2026-03-01T00:00:00+00:00", "BBB"))
    rows.append(_row("BBB", "2026-03-01T00:00:00+00:00", None, "BB+"))
    rows.append(
        _row(
            "CCC",
            "2026-01-01T00:00:00+00:00",
            None,
            "CCC",
            default_date="2026-04-01T00:00:00+00:00",
        )
    )
    rows.append(_row("DDD", "2026-02-01T00:00:00+00:00", None, "A"))

    return pd.DataFrame(rows)


@pytest.fixture(params=["duckdb", "sqlite"])
def warehouse_with_bonds(request, tmp_path: Path) -> Warehouse:
    backend = request.param
    if backend == "duckdb":
        pytest.importorskip("duckdb")
    suffix = ".duckdb" if backend == "duckdb" else ".db"
    wh = Warehouse(str(tmp_path / f"bonds{suffix}"), backend=backend)
    df = _make_bond_rows()
    wh._backend.upsert_frame("bond_reference", df, _bond_cols(), mode="REPLACE")
    wh._backend.commit()
    yield wh
    wh.close()


def test_read_bond_reference_asof_uses_valid_window(warehouse_with_bonds: Warehouse) -> None:
    """At 2026-02-15 the active (non-defaulted) universe is
    {AAA (v1), BBB (v1), DDD}; BBB v2 starts after the asof, CCC is
    omitted by the survivorship filter because default_date is non-null.

    The valid_window slicing picks v1 of BBB over v2 because v2's
    valid_from is *after* the asof.
    """

    df = read_bond_reference_asof(warehouse_with_bonds, "2026-02-15T00:00:00+00:00")
    cusips = sorted(df["cusip"].tolist())
    assert cusips == ["AAA", "BBB", "DDD"]
    bbb = df[df["cusip"] == "BBB"].iloc[0]
    assert str(bbb["rating"]) == "BBB"

    # At 2026-03-15 BBB v2 is in force; v1 has been superseded.
    df_later = read_bond_reference_asof(warehouse_with_bonds, "2026-03-15T00:00:00+00:00")
    bbb_later = df_later[df_later["cusip"] == "BBB"].iloc[0]
    assert str(bbb_later["rating"]) == "BB+"
    # DDD is active by 2026-03-15 too.
    assert "DDD" in df_later["cusip"].tolist()


def test_read_bond_reference_asof_excludes_post_default(warehouse_with_bonds: Warehouse) -> None:
    """The survivorship filter is sticky: CCC's default_date=2026-04-01
    is a *structural* survivorship-failure flag, so the default-mode
    read drops CCC at every asof regardless of whether the asof is
    before or after the default event. This matches the user-facing
    spec (PR-2 task C / Q-4): bonds whose default_date or delisted_date
    is non-null are excluded unless include_survivorship_failures=True.
    """

    df_before = read_bond_reference_asof(warehouse_with_bonds, "2026-02-15T00:00:00+00:00")
    df_after = read_bond_reference_asof(warehouse_with_bonds, "2026-05-01T00:00:00+00:00")
    assert "CCC" not in df_before["cusip"].tolist()
    assert "CCC" not in df_after["cusip"].tolist()
    # AAA, DDD remain in both windows; BBB transitions v1→v2 between
    # the two snapshots.
    assert sorted(df_after["cusip"].tolist()) == ["AAA", "BBB", "DDD"]


def test_read_bond_reference_asof_includes_post_default_when_flagged(
    warehouse_with_bonds: Warehouse,
) -> None:
    """``include_survivorship_failures=True`` is the audit-mode path
    used by ingestion validators that need to see the dropped rows."""

    df = read_bond_reference_asof(
        warehouse_with_bonds,
        "2026-05-01T00:00:00+00:00",
        include_survivorship_failures=True,
    )
    assert "CCC" in df["cusip"].tolist()
    assert sorted(df["cusip"].tolist()) == ["AAA", "BBB", "CCC", "DDD"]


def test_read_bond_reference_history_returns_temporal_order(
    warehouse_with_bonds: Warehouse,
) -> None:
    """BBB has two versions; history returns both in valid_from order."""

    df = read_bond_reference_history(warehouse_with_bonds, "BBB")
    assert len(df) == 2
    valid_from = pd.to_datetime(df["valid_from"], utc=True).tolist()
    assert valid_from[0] <= valid_from[1]
    ratings = df["rating"].tolist()
    assert ratings == ["BBB", "BB+"]


def test_read_bond_reference_asof_marks_is_active(warehouse_with_bonds: Warehouse) -> None:
    """``is_active`` is computed at read time (True for every row the
    asof query returned)."""

    df = read_bond_reference_asof(warehouse_with_bonds, "2026-02-15T00:00:00+00:00")
    assert "is_active" in df.columns
    assert df["is_active"].all()


def test_read_bond_reference_history_returns_empty_for_unknown_cusip(
    warehouse_with_bonds: Warehouse,
) -> None:
    """Unknown CUSIP -> empty frame (not an error)."""

    df = read_bond_reference_history(warehouse_with_bonds, "XXXXXXXXX")
    assert df.empty
