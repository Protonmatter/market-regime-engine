# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import threading
from pathlib import Path

from market_regime_engine.storage_repositories import Warehouse

_POOLED_WAREHOUSES: dict[str, Warehouse] = {}
_POOLED_LOCKS: dict[str, threading.RLock] = {}
_POOL_LOCK = threading.RLock()


def _resolve_pool_key(path: str | Path) -> str:
    return str(Path(path).resolve())


def get_pooled_warehouse(path: str | Path) -> Warehouse:
    """Return the per-process :class:`Warehouse` for ``path``.

    Construction is serialised through ``_POOL_LOCK`` (re-entrant); after
    that the same instance is returned on every call. Pooled instances are
    keyed by the resolved absolute path so two callers passing
    ``"./data/mre.duckdb"`` and ``"data/mre.duckdb"`` from the same cwd get
    the same instance.

    **DuckDB threading note:** the underlying DuckDB Python connection is
    not safe for concurrent ``execute`` calls from multiple threads. Wrap
    writes in :func:`pooled_warehouse_write_lock` (or hold the
    :class:`threading.RLock` returned by it) when sharing the pooled
    warehouse across threads — the lock is re-entrant so nested writers
    inside the same thread do not deadlock.
    """
    path_str = _resolve_pool_key(path)
    with _POOL_LOCK:
        existing = _POOLED_WAREHOUSES.get(path_str)
        if existing is None:
            existing = Warehouse(path_str)
            _POOLED_WAREHOUSES[path_str] = existing
            _POOLED_LOCKS[path_str] = threading.RLock()
        return existing


@contextlib.contextmanager
def pooled_warehouse_write_lock(path: str | Path):
    """Context manager that holds the per-warehouse write lock.

    Recommended usage::

        wh = get_pooled_warehouse(path)
        with pooled_warehouse_write_lock(path):
            wh.write_credit_regime_score(df)

    The lock is :class:`threading.RLock` so nested ``with`` blocks inside
    the same thread do not deadlock; concurrent writers from different
    threads serialise around the lock.
    """
    path_str = _resolve_pool_key(path)
    with _POOL_LOCK:
        lock = _POOLED_LOCKS.get(path_str)
        if lock is None:
            # Ensure the warehouse + lock pair are minted together so a
            # caller can grab the lock before opening the warehouse.
            get_pooled_warehouse(path)
            lock = _POOLED_LOCKS[path_str]
    with lock:
        yield


def close_pooled_warehouses() -> None:
    """Close every pooled warehouse and clear the pool.

    Intended for FastAPI shutdown handlers and test teardown. Idempotent;
    on individual close failure the function still clears the pool so a
    partial failure does not leak references, then re-raises the
    aggregated error.
    """
    errors: list[Exception] = []
    with _POOL_LOCK:
        for wh in list(_POOLED_WAREHOUSES.values()):
            try:
                wh.close()
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(exc)
        _POOLED_WAREHOUSES.clear()
        _POOLED_LOCKS.clear()
    if errors:
        raise RuntimeError(f"close_pooled_warehouses encountered {len(errors)} errors: {errors!r}")


def pooled_warehouse_paths() -> tuple[str, ...]:
    """Inspect the current pool — used by tests for the singleton rail."""
    with _POOL_LOCK:
        return tuple(_POOLED_WAREHOUSES)


def is_pooled_warehouse(warehouse: Warehouse) -> bool:
    """Return True iff ``warehouse`` is currently owned by the process pool.

    Used by FI GET handlers (``fixed_income/api.py``) to guard against the
    legacy ``finally: wh.close()`` pattern poisoning the pool: a pooled
    :class:`Warehouse` lives for the lifetime of the FastAPI process and
    is released via the ``_lifespan`` shutdown hook
    (``close_pooled_warehouses``). Closing it from inside a request
    handler leaves a dead instance in the registry and the next request
    hits a closed DuckDB connection.

    Identity-based: compares by ``is`` so a non-pooled :class:`Warehouse`
    instance constructed against the same path (e.g. test factories that
    bypass the pool) is correctly classified as not-pool-owned.
    """
    with _POOL_LOCK:
        return any(warehouse is wh for wh in _POOLED_WAREHOUSES.values())



__all__ = [
    "close_pooled_warehouses",
    "get_pooled_warehouse",
    "is_pooled_warehouse",
    "pooled_warehouse_paths",
    "pooled_warehouse_write_lock",
]
