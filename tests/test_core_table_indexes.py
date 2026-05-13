# SPDX-License-Identifier: Apache-2.0
"""v1.6.0 — A13 / Finding §3.7 regression tests.

Pin the contract that the new secondary indexes on the hot core-table
read paths (observations, features, model_outputs) are

1. registered after the FI schema boots, and
2. picked up by the SQLite planner for the canonical ``ORDER BY`` reads.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from market_regime_engine.storage import Warehouse


@pytest.fixture
def wh_sqlite(tmp_path: Path) -> Warehouse:
    return Warehouse(tmp_path / "core.sqlite")


def _seed_observations(wh: Warehouse, n: int) -> None:
    base = pd.Timestamp("2020-01-01")
    rows = [
        {
            "series_id": f"s{i % 5}",
            "date": (base + pd.Timedelta(days=i)).isoformat(),
            "value": float(i),
            "vintage_date": None,
            "source": "fred",
            "metadata_json": "{}",
        }
        for i in range(n)
    ]
    wh.write_observations(pd.DataFrame(rows))


def _seed_features(wh: Warehouse, n: int) -> None:
    base = pd.Timestamp("2020-01-01")
    rows = [
        {
            "feature_name": f"f{i % 3}",
            "date": (base + pd.Timedelta(days=i)).isoformat(),
            "value": float(i),
            "domain": "macro",
            "metadata_json": "{}",
        }
        for i in range(n)
    ]
    wh.write_features(pd.DataFrame(rows))


def test_idx_observations_date_series_exists(wh_sqlite: Warehouse) -> None:
    _seed_observations(wh_sqlite, n=10)
    backend = wh_sqlite._backend
    rows = backend.read_sql(
        "SELECT name FROM sqlite_master WHERE type='index' "
        "AND tbl_name='observations'"
    )
    names = set(rows["name"].astype(str).tolist())
    assert "idx_observations_date_series" in names, names


def test_idx_features_date_name_exists(wh_sqlite: Warehouse) -> None:
    _seed_features(wh_sqlite, n=10)
    backend = wh_sqlite._backend
    rows = backend.read_sql(
        "SELECT name FROM sqlite_master WHERE type='index' "
        "AND tbl_name='features'"
    )
    names = set(rows["name"].astype(str).tolist())
    assert "idx_features_date_name" in names, names


def test_idx_model_outputs_date_exists(wh_sqlite: Warehouse) -> None:
    # Seed a single model output so the table is created with its index.
    wh_sqlite.write_model_outputs(
        pd.DataFrame(
            [
                {
                    "model_name": "m1",
                    "date": "2026-01-01",
                    "horizon": "1m",
                    "target": "regime",
                    "value": 0.5,
                    "metadata_json": "{}",
                }
            ]
        )
    )
    backend = wh_sqlite._backend
    rows = backend.read_sql(
        "SELECT name FROM sqlite_master WHERE type='index' "
        "AND tbl_name='model_outputs'"
    )
    names = set(rows["name"].astype(str).tolist())
    assert "idx_model_outputs_date" in names, names


def test_observations_read_plan_hits_secondary_index(wh_sqlite: Warehouse) -> None:
    """EXPLAIN QUERY PLAN should reference idx_observations_date_series
    for the canonical read pattern."""
    _seed_observations(wh_sqlite, n=200)
    backend = wh_sqlite._backend
    plan = backend.read_sql(
        "EXPLAIN QUERY PLAN "
        "SELECT * FROM observations ORDER BY date, series_id"
    )
    plan_text = " ".join(plan["detail"].astype(str).tolist())
    assert "idx_observations_date_series" in plan_text, plan_text
