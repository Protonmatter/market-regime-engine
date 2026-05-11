# SPDX-License-Identifier: Apache-2.0
"""PR-5 ASK-8: per-process Warehouse singleton.

Pre-PR-5 every FastAPI request opened/closed a fresh Warehouse, paying
DuckDB catalog + WAL teardown on the hot path. PR-5 caches the Warehouse
per-process keyed by absolute path. Reads run concurrently under DuckDB
MVCC; writes serialise via a re-entrant lock.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pandas as pd
import pytest

# Importing the FI package registers the 13 FI tables; required for
# ``write_credit_regime_score`` below.
import market_regime_engine.fixed_income  # noqa: F401
from market_regime_engine.storage import (
    Warehouse,
    close_pooled_warehouses,
    get_pooled_warehouse,
    pooled_warehouse_paths,
    pooled_warehouse_write_lock,
)


@pytest.fixture(autouse=True)
def _teardown_pool() -> None:
    """Empty the pool around every test so state never leaks."""
    close_pooled_warehouses()
    yield
    close_pooled_warehouses()


def test_get_pooled_warehouse_returns_same_instance_for_same_path(tmp_path: Path) -> None:
    db = tmp_path / "pool.duckdb"
    a = get_pooled_warehouse(db)
    b = get_pooled_warehouse(db)
    assert a is b


def test_get_pooled_warehouse_different_path_different_instance(tmp_path: Path) -> None:
    a = get_pooled_warehouse(tmp_path / "a.duckdb")
    b = get_pooled_warehouse(tmp_path / "b.duckdb")
    assert a is not b
    assert isinstance(a, Warehouse)
    assert isinstance(b, Warehouse)


def test_get_pooled_warehouse_normalises_path(tmp_path: Path) -> None:
    db = tmp_path / "norm.duckdb"
    a = get_pooled_warehouse(db)
    b = get_pooled_warehouse(str(db))
    # ``Path.resolve()`` should normalise both inputs to the same absolute
    # path, so the pool returns the same instance.
    assert a is b


def test_close_pooled_warehouses_clears_pool(tmp_path: Path) -> None:
    get_pooled_warehouse(tmp_path / "x.duckdb")
    get_pooled_warehouse(tmp_path / "y.duckdb")
    assert len(pooled_warehouse_paths()) == 2
    close_pooled_warehouses()
    assert pooled_warehouse_paths() == ()


def test_pooled_warehouse_concurrent_write_serializes(tmp_path: Path) -> None:
    """20 threads writing 5 rows each through ``pooled_warehouse_write_lock``
    must produce 100 rows without raising. The lock serialises the DuckDB
    Python connection (which is not safe for concurrent ``execute``
    calls)."""
    db = tmp_path / "concurrent.duckdb"
    n_threads = 20
    rows_per_thread = 5

    def worker(idx: int) -> None:
        wh = get_pooled_warehouse(db)
        df = pd.DataFrame(
            [
                {
                    "model_run_id": f"run-{idx}-{i}",
                    "timestamp": pd.Timestamp("2026-05-01", tz="UTC").isoformat(),
                    "regime_score": 50.0,
                    "regime_label": "Normal Liquidity",
                    "confidence": 0.7,
                    "drivers_json": "[]",
                    "component_scores_json": "{}",
                    "release_gate": 1,
                    "artifact_hash": f"hash-{idx}-{i}",
                    "metadata_json": "{}",
                }
                for i in range(rows_per_thread)
            ]
        )
        with pooled_warehouse_write_lock(db):
            wh.write_credit_regime_score(df)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    wh = get_pooled_warehouse(db)
    with pooled_warehouse_write_lock(db):
        out = wh.read_credit_regime_scores()
    assert len(out) == n_threads * rows_per_thread


def test_pooled_warehouse_write_lock_is_reentrant(tmp_path: Path) -> None:
    """The lock must be re-entrant so a writer that grabs the lock then
    calls into a helper that also grabs it does not deadlock."""
    db = tmp_path / "reentrant.duckdb"
    get_pooled_warehouse(db)  # mint the lock
    with pooled_warehouse_write_lock(db), pooled_warehouse_write_lock(db):
        pass  # would deadlock with a non-re-entrant lock


def test_pooled_warehouse_write_lock_mints_warehouse_lazily(tmp_path: Path) -> None:
    """``pooled_warehouse_write_lock`` must mint the warehouse + lock on
    first access so a caller can grab the lock before the warehouse."""
    db = tmp_path / "lazy.duckdb"
    assert pooled_warehouse_paths() == ()
    with pooled_warehouse_write_lock(db):
        assert _resolve_path(db) in pooled_warehouse_paths()


def _resolve_path(p: Path) -> str:
    return str(Path(p).resolve())


def test_pooled_warehouse_singleton_under_first_access_race(tmp_path: Path) -> None:
    """Two threads racing the very first ``get_pooled_warehouse`` must end
    up with the *same* instance — verifies the RLock guards the
    constructor."""
    db = tmp_path / "race.duckdb"
    instances: list[Warehouse] = []
    barrier = threading.Barrier(8)

    def open_pool() -> None:
        barrier.wait()
        instances.append(get_pooled_warehouse(db))

    threads = [threading.Thread(target=open_pool) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    first = instances[0]
    assert all(inst is first for inst in instances)
