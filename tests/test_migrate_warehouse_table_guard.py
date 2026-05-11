# SPDX-License-Identifier: Apache-2.0
"""Acceptance test for PR-2 task G / AF-12 — migrate_warehouse must
refuse table names that are not in the registry, closing the
SQL-injection foot-gun flagged at storage.py:1588 in the pre-PR-2 code.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

import market_regime_engine.fixed_income  # noqa: F401 - register FI tables
from market_regime_engine.storage import (
    Warehouse,
    migrate_warehouse,
    registered_tables,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def test_migrate_warehouse_rejects_unregistered_table(tmp_path: Path) -> None:
    """Passing a ``tables=`` argument with a name that is not in the
    registry raises ValueError BEFORE any f-string interpolation."""

    src = tmp_path / "src.duckdb"
    dst = tmp_path / "dst.duckdb"
    Warehouse(str(src), backend="duckdb").close()
    Warehouse(str(dst), backend="duckdb").close()

    with pytest.raises(ValueError, match="unregistered tables"):
        migrate_warehouse(
            src,
            dst,
            tables=["sqlite_master; DROP TABLE observations--"],
        )


def test_migrate_warehouse_accepts_registered_subset(tmp_path: Path) -> None:
    """A subset of registered table names is accepted and round-trips
    without raising; this guards against an over-strict allow-list
    that would also block legitimate selective migrations."""

    src = tmp_path / "src.duckdb"
    dst = tmp_path / "dst.duckdb"
    Warehouse(str(src), backend="duckdb").close()
    Warehouse(str(dst), backend="duckdb").close()

    # Subset of known-registered tables.
    counts = migrate_warehouse(
        src,
        dst,
        tables=["observations", "regimes", "credit_regime_scores"],
    )
    # No rows in the empty source, but the call returns a count dict.
    assert set(counts) == {"observations", "regimes", "credit_regime_scores"}
    for v in counts.values():
        assert v == 0


def test_migrate_warehouse_default_covers_registry(tmp_path: Path) -> None:
    """``tables=None`` defaults to the full registry; the returned
    count dict covers every registered table name."""

    src = tmp_path / "src.duckdb"
    dst = tmp_path / "dst.duckdb"
    Warehouse(str(src), backend="duckdb").close()
    Warehouse(str(dst), backend="duckdb").close()

    counts = migrate_warehouse(src, dst)
    expected = {spec.name for spec in registered_tables()}
    assert set(counts) == expected


def test_migrate_warehouse_rejects_partially_unregistered_list(tmp_path: Path) -> None:
    """A list containing one valid and one unregistered name is still
    rejected at the function entry — the guard is conservative."""

    src = tmp_path / "src.duckdb"
    dst = tmp_path / "dst.duckdb"
    Warehouse(str(src), backend="duckdb").close()
    Warehouse(str(dst), backend="duckdb").close()

    with pytest.raises(ValueError, match="unregistered tables"):
        migrate_warehouse(
            src,
            dst,
            tables=["observations", "__not_a_real_table__"],
        )
