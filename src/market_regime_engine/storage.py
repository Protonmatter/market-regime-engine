# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

"""Backward-compatible storage facade.

The v1.6 refactor splits storage internals across focused modules:

- :mod:`storage_registry` owns DDL/table registration and legacy schema aggregates.
- :mod:`storage_backends` owns backend selection plus SQLite/DuckDB adapters.
- :mod:`storage_repositories` owns the Warehouse read/write repository API.
- :mod:`storage_pool` owns per-process pooling and write-concurrency utilities.

This module intentionally remains import-compatible for existing callers.
"""

from typing import Any

from market_regime_engine.storage_registry import (
    BackendName,
    TableSpec,
    _REGISTRY,
    _extract_pk,
    _extract_table_name,
    _get_table_pk,
    legacy_aggregate,
    register_tables,
    registered_tables,
)
from market_regime_engine.storage_backends import (
    _Backend,
    _DuckDBBackend,
    _SqliteBackend,
    _quote_columns,
    _select_backend,
)
from market_regime_engine.storage_repositories import (
    Warehouse,
    _is_duckdb_backend,
    _normalise_asof_for_sql,
    _read_with_params,
    migrate_warehouse,
    read_bond_reference_asof,
    read_bond_reference_history,
)
from market_regime_engine.storage_pool import (
    close_pooled_warehouses,
    get_pooled_warehouse,
    is_pooled_warehouse,
    pooled_warehouse_paths,
    pooled_warehouse_write_lock,
)


def __getattr__(name: str) -> Any:
    if name in {"SCHEMA_STATEMENTS", "_TABLE_PKS", "_TABLE_NAMES"}:
        return legacy_aggregate(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BackendName",
    "TableSpec",
    "Warehouse",
    "close_pooled_warehouses",
    "get_pooled_warehouse",
    "is_pooled_warehouse",
    "migrate_warehouse",
    "pooled_warehouse_paths",
    "pooled_warehouse_write_lock",
    "read_bond_reference_asof",
    "read_bond_reference_history",
    "register_tables",
    "registered_tables",
]
