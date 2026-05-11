# SPDX-License-Identifier: Apache-2.0
"""Acceptance tests for the v1.5 PR-2 ASK-3 register_tables registry.

The four contracts verified here mirror the user-facing API surface
introduced by PR-2: idempotent registration, snapshot reads, and the
back-compat ``SCHEMA_STATEMENTS`` / ``_TABLE_PKS`` aggregates that
downstream callers may continue to import.
"""

from __future__ import annotations

import pytest

from market_regime_engine import storage
from market_regime_engine.storage import (
    TableSpec,
    register_tables,
    registered_tables,
)

# Importing the FI package eagerly so the post-FI-registration view is
# the one the legacy-aggregate tests inspect. PEP 562 module
# __getattr__ resolves ``storage.SCHEMA_STATEMENTS`` and
# ``storage._TABLE_PKS`` dynamically; the tests therefore reference
# the names via the ``storage`` module rather than rebinding them at
# import time (which would freeze the pre-FI snapshot).
import market_regime_engine.fixed_income  # noqa: E402, F401


def test_register_tables_idempotent_on_name() -> None:
    """Re-registering the same TableSpec is a no-op; a *different* spec
    under an already-claimed name raises ValueError."""

    name = "_pr2_idempotent_probe"
    spec = TableSpec(
        name=name,
        create_sql=f"CREATE TABLE IF NOT EXISTS {name} (x INTEGER PRIMARY KEY)",
        primary_key=("x",),
    )
    initial_count = len(registered_tables())
    try:
        register_tables([spec])
        assert any(s.name == name for s in registered_tables())
        # Re-register the exact same spec: idempotent.
        register_tables([spec])
        register_tables([spec])
        assert sum(1 for s in registered_tables() if s.name == name) == 1
        assert len(registered_tables()) == initial_count + 1

        # Register a *different* spec under the same name: must raise.
        conflicting = TableSpec(
            name=name,
            create_sql=f"CREATE TABLE IF NOT EXISTS {name} (x INTEGER PRIMARY KEY, y INTEGER)",
            primary_key=("x",),
        )
        with pytest.raises(ValueError, match="conflicting content"):
            register_tables([conflicting])
    finally:
        storage._REGISTRY[:] = [s for s in storage._REGISTRY if s.name != name]


def test_registered_tables_returns_all_specs() -> None:
    """registered_tables() returns the full registry as a tuple."""

    specs = registered_tables()
    names = {s.name for s in specs}
    # Pre-PR-2 the warehouse shipped 34 core tables; PR-2 adds 13 FI
    # tables once ``market_regime_engine.fixed_income`` is imported.
    # The full universe is at least the 34 cores; FI is registered by
    # importing the package in the conftest's autouse fixture below.
    assert "observations" in names
    assert "release_gates" in names
    assert "model_runs" in names
    assert isinstance(specs, tuple)
    # Snapshot — callers cannot mutate the registry through the result.
    with pytest.raises((TypeError, AttributeError)):
        specs.append(  # type: ignore[attr-defined]
            TableSpec(name="x", create_sql="", primary_key=())
        )


def test_legacy_schema_statements_aggregate_matches_registry() -> None:
    """``storage.SCHEMA_STATEMENTS`` is a PEP 562 dynamic aggregate
    over the registry; every spec's create_sql appears exactly once
    and the insertion order is preserved.

    Reading through ``storage.SCHEMA_STATEMENTS`` rather than a
    top-level ``from storage import SCHEMA_STATEMENTS`` is intentional:
    the attribute is dynamically resolved on each lookup, so out-of-tree
    callers that follow the same pattern see the registry as it stands
    at access time (including any FI / future product line
    contributions registered after their first import of ``storage``).
    """

    aggregate = list(storage.SCHEMA_STATEMENTS)
    direct = [spec.create_sql for spec in registered_tables()]
    assert aggregate == direct


def test_legacy_table_pks_aggregate_matches_registry() -> None:
    """``storage._TABLE_PKS`` is a PEP 562 dynamic aggregate; every
    spec's PK appears under its name and the values are tuples."""

    aggregate = dict(storage._TABLE_PKS)
    direct = {spec.name: spec.primary_key for spec in registered_tables()}
    assert aggregate == direct
    for pk in aggregate.values():
        assert isinstance(pk, tuple)


def test_fi_tables_register_on_fixed_income_import() -> None:
    """Importing the FI package registers its 13 tables idempotently."""

    import market_regime_engine.fixed_income as fi

    names = {s.name for s in registered_tables()}
    for fi_table in fi.FI_TABLE_NAMES:
        assert fi_table in names, f"FI table {fi_table!r} missing from registry"

    # Re-importing is a no-op (idempotent register_tables).
    from importlib import reload

    reload(fi)
    names_after_reload = {s.name for s in registered_tables()}
    assert names == names_after_reload


def test_get_table_pk_returns_explicit_primary_key() -> None:
    """The DuckDB upsert path uses ``_get_table_pk`` instead of the
    regex parser; verify it returns the explicit ``primary_key`` from
    the TableSpec."""

    assert storage._get_table_pk("observations") == ("series_id", "date", "vintage_date")
    assert storage._get_table_pk("regimes") == ("date",)
    # Unregistered table -> empty tuple, which the DuckDB builder
    # interprets as "no ON CONFLICT clause".
    assert storage._get_table_pk("__definitely_not_a_real_table__") == ()


def test_extract_pk_emits_deprecation_warning() -> None:
    """The pre-PR-2 regex PK parser is kept for legacy callers but
    emits a DeprecationWarning when invoked."""

    sql = "CREATE TABLE foo (a INTEGER, b INTEGER, PRIMARY KEY(a, b))"
    with pytest.warns(DeprecationWarning, match="TableSpec.primary_key"):
        result = storage._extract_pk(sql)
    assert result == ("a", "b")


def test_extract_table_name_emits_deprecation_warning() -> None:
    """The pre-PR-2 regex name parser is kept for legacy callers but
    emits a DeprecationWarning when invoked."""

    sql = "CREATE TABLE IF NOT EXISTS foo (a INTEGER PRIMARY KEY)"
    with pytest.warns(DeprecationWarning, match="TableSpec.name"):
        result = storage._extract_table_name(sql)
    assert result == "foo"
