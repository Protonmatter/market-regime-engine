# SPDX-License-Identifier: Apache-2.0
"""PR-5 ASK-8: ``api_v1`` reads through the pooled :class:`Warehouse`.

Pre-PR-5 every ``/v1/...`` request opened a fresh ``Warehouse(_db_path())``
and closed it in a ``try/finally`` — the DuckDB catalog + WAL teardown
dominated latency on a busy worker. PR-5 wires every read through
:func:`get_pooled_warehouse`, and ``_on_shutdown_close_pool`` releases the
pool on graceful FastAPI shutdown.
"""

from __future__ import annotations

from pathlib import Path

import market_regime_engine.fixed_income  # noqa: F401  registers FI schema
from market_regime_engine.storage import (
    close_pooled_warehouses,
    get_pooled_warehouse,
    pooled_warehouse_paths,
)


def _seed_warehouse(path: Path) -> None:
    wh = get_pooled_warehouse(path)
    # Touch read_release_gates so the table exists; init_schema in the
    # backend creates it but the read smoke-tests the warehouse handle.
    wh.read_release_gates()


def test_api_v1_health_endpoint_uses_pooled_warehouse(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / "pool_api.duckdb"
    monkeypatch.setenv("MRE_DB_PATH", str(db))
    monkeypatch.delenv("MRE_API_KEY", raising=False)
    close_pooled_warehouses()
    _seed_warehouse(db)

    # Re-import so the module re-reads ``MRE_DB_PATH``.
    import importlib

    from fastapi.testclient import TestClient

    import market_regime_engine.api_v1 as api_v1

    importlib.reload(api_v1)

    client = TestClient(api_v1.app)
    resp = client.get("/v1/health")
    assert resp.status_code == 200

    # After the request the pool contains the resolved path.
    paths = pooled_warehouse_paths()
    assert any(p.endswith("pool_api.duckdb") for p in paths)
    close_pooled_warehouses()


def test_api_v1_repeated_reads_reuse_pooled_warehouse(monkeypatch, tmp_path: Path) -> None:
    """100 sequential reads must populate the pool with exactly one
    Warehouse — proving DuckDB is not being torn down + rebuilt per
    request."""
    db = tmp_path / "reuse.duckdb"
    monkeypatch.setenv("MRE_DB_PATH", str(db))
    monkeypatch.delenv("MRE_API_KEY", raising=False)
    close_pooled_warehouses()
    _seed_warehouse(db)

    import importlib

    from fastapi.testclient import TestClient

    import market_regime_engine.api_v1 as api_v1

    importlib.reload(api_v1)

    client = TestClient(api_v1.app)
    for _ in range(100):
        resp = client.get("/v1/health")
        assert resp.status_code == 200

    # Pool size is 1 — the same warehouse served every request.
    assert len(pooled_warehouse_paths()) == 1
    close_pooled_warehouses()


def test_api_v1_shutdown_event_drains_pool(monkeypatch, tmp_path: Path) -> None:
    """The shutdown event closes the pool so no DuckDB file handles leak
    across uvicorn reload cycles."""
    db = tmp_path / "shutdown.duckdb"
    monkeypatch.setenv("MRE_DB_PATH", str(db))
    monkeypatch.delenv("MRE_API_KEY", raising=False)
    close_pooled_warehouses()
    _seed_warehouse(db)

    import importlib

    from fastapi.testclient import TestClient

    import market_regime_engine.api_v1 as api_v1

    importlib.reload(api_v1)

    with TestClient(api_v1.app) as client:
        client.get("/v1/health")
        assert len(pooled_warehouse_paths()) >= 1
    # TestClient's __exit__ triggers the shutdown event which drains the
    # pool.
    assert pooled_warehouse_paths() == ()
