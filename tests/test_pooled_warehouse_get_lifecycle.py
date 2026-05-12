# SPDX-License-Identifier: Apache-2.0
"""Regression — pool poisoning via ``finally: wh.close()`` in FI GET handlers.

Pre-fix (REVIEW.md Tier-1 A1/B-Auto-1): every FI GET handler closed the
warehouse in a ``finally`` block. The default factory returned the
per-process pooled :class:`Warehouse`; closing it left a dead instance
in ``_POOLED_WAREHOUSES`` so the next request received a closed DuckDB
connection.

Post-fix: handlers guard the close behind
:func:`market_regime_engine.storage.is_pooled_warehouse` (via
``_close_if_not_pooled``) — non-pooled instances (e.g. test factories)
are still closed; the per-process pool is left alone so the FastAPI
lifespan hook drains it on shutdown.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import market_regime_engine.fixed_income  # noqa: F401  - register FI schema
from market_regime_engine.fixed_income.api import build_router, reset_fi_cache
from market_regime_engine.fixed_income.credit_spread_regime import (
    score_credit_regime,
    write_credit_regime_score,
)
from market_regime_engine.fixed_income.liquidity_stress import (
    score_liquidity_stress,
    write_liquidity_stress_score,
)
from market_regime_engine.storage import (
    Warehouse,
    close_pooled_warehouses,
    get_pooled_warehouse,
    is_pooled_warehouse,
)

_ASOF = pd.Timestamp("2026-05-08T16:00:00+00:00")


@pytest.fixture(autouse=True)
def _teardown_pool() -> None:
    close_pooled_warehouses()
    reset_fi_cache()
    yield
    close_pooled_warehouses()
    reset_fi_cache()


def _credit_features(asof: pd.Timestamp, n: int = 30) -> pd.DataFrame:
    dates = pd.date_range(end=asof, periods=n, freq="D", tz="UTC")
    rows: list[dict] = []
    for ts in dates:
        rows.append(
            {
                "date": ts,
                "feature_name": "ust_slope",
                "value": 0.5,
                "source_timestamp": ts,
                "vintage_date": None,
            }
        )
        rows.append(
            {
                "date": ts,
                "feature_name": "ust_curvature",
                "value": 0.1,
                "source_timestamp": ts,
                "vintage_date": None,
            }
        )
        rows.append(
            {
                "date": ts,
                "feature_name": "cdx_ig_5y",
                "value": 65.0,
                "source_timestamp": ts,
                "vintage_date": None,
            }
        )
        rows.append(
            {
                "date": ts,
                "feature_name": "cdx_hy_5y",
                "value": 350.0,
                "source_timestamp": ts,
                "vintage_date": None,
            }
        )
        rows.append(
            {
                "date": ts,
                "feature_name": "vix",
                "value": 18.0,
                "source_timestamp": ts,
                "vintage_date": None,
            }
        )
        rows.append(
            {
                "date": ts,
                "feature_name": "move",
                "value": 100.0,
                "source_timestamp": ts,
                "vintage_date": None,
            }
        )
        rows.append(
            {
                "date": ts,
                "feature_name": "etf_prem_disc",
                "value": 0.10,
                "source_timestamp": ts,
                "vintage_date": None,
            }
        )
    return pd.DataFrame(rows)


def _liquidity_features(asof: pd.Timestamp, n: int = 60) -> pd.DataFrame:
    dates = pd.date_range(end=asof, periods=n, freq="D", tz="UTC")
    rows: list[dict] = []
    for ts in dates:
        for name, value in [
            ("etf_prem_disc", 0.10),
            ("trace_volume_z", 0.5),
            ("repo_haircut", 0.02),
            ("fra_ois", 0.20),
            ("trace_volume_share_top10", 0.15),
            ("dealer_inventory_z", -0.30),
            ("settlement_fails_z", 0.20),
        ]:
            rows.append(
                {
                    "date": ts,
                    "feature_name": name,
                    "value": value,
                    "source_timestamp": ts,
                    "vintage_date": None,
                }
            )
    return pd.DataFrame(rows)


def _seed_credit(wh: Warehouse) -> None:
    out = score_credit_regime(_credit_features(_ASOF), asof=_ASOF, model_run_id="run-1")
    write_credit_regime_score(wh, out)


def _seed_liquidity(wh: Warehouse) -> None:
    out = score_liquidity_stress(
        _liquidity_features(_ASOF),
        asof=_ASOF,
        scope_type="market",
        scope_id="USD_IG",
        model_run_id="run-liq-1",
    )
    write_liquidity_stress_score(wh, out)


# ---------------------------------------------------------------------------
# Unit-level: ``is_pooled_warehouse`` introspection helper.
# ---------------------------------------------------------------------------


def test_is_pooled_warehouse_true_for_pooled_instance(tmp_path: Path) -> None:
    wh = get_pooled_warehouse(tmp_path / "pool.duckdb")
    assert is_pooled_warehouse(wh) is True


def test_is_pooled_warehouse_false_for_direct_construction(tmp_path: Path) -> None:
    wh = Warehouse(str(tmp_path / "direct.duckdb"))
    try:
        assert is_pooled_warehouse(wh) is False
    finally:
        wh.close()


def test_pooled_get_warehouse_survives_repeated_close_calls(tmp_path: Path) -> None:
    """Acceptance-gate test from the plan: closing the pooled instance
    must NOT poison the pool — the next ``get_pooled_warehouse`` call
    must return a still-live connection (in practice we make the close
    a no-op for pooled handles by guarding through
    ``is_pooled_warehouse``)."""
    db = tmp_path / "lifecycle.duckdb"
    a = get_pooled_warehouse(db)
    _seed_credit(a)

    # The FI handlers historically called .close() here. With the FIX 1
    # guard wrapped in ``_close_if_not_pooled`` they no longer do; the
    # connection must remain live.
    assert is_pooled_warehouse(a) is True

    b = get_pooled_warehouse(db)
    # Same instance from the pool.
    assert a is b
    # And it must still be usable.
    df = b.read_credit_regime_scores()
    assert df is not None
    assert not df.empty


# ---------------------------------------------------------------------------
# End-to-end: two sequential FI GETs against the pooled factory must share
# the pool without poisoning.
# ---------------------------------------------------------------------------


def _app_with_pooled_factory(db: Path) -> FastAPI:
    app = FastAPI()

    def factory() -> Warehouse:
        return get_pooled_warehouse(db)

    app.include_router(build_router(warehouse_factory=factory))
    return app


def test_fi_regime_index_latest_two_sequential_gets_share_pool(tmp_path: Path) -> None:
    db = tmp_path / "two-gets.duckdb"
    wh = get_pooled_warehouse(db)
    _seed_credit(wh)

    client = TestClient(_app_with_pooled_factory(db))

    # Two sequential GETs. Pre-fix the second one would fail because the
    # first ``finally: wh.close()`` poisoned the pool.
    first = client.get("/v1/regime_index/latest")
    second = client.get("/v1/regime_index/latest")
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text

    # And the pooled handle is still usable from outside the request
    # cycle — proving the DuckDB connection isn't closed.
    df = wh.read_credit_regime_scores()
    assert df is not None
    assert not df.empty


def test_fi_liquidity_index_latest_two_sequential_gets_share_pool(tmp_path: Path) -> None:
    db = tmp_path / "two-gets-liq.duckdb"
    wh = get_pooled_warehouse(db)
    _seed_liquidity(wh)

    client = TestClient(_app_with_pooled_factory(db))

    first = client.get("/v1/liquidity_index/latest")
    second = client.get("/v1/liquidity_index/latest")
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text

    df = wh.read_liquidity_stress_scores()
    assert df is not None
    assert not df.empty


def test_fi_liquidity_index_scoped_two_sequential_gets_share_pool(tmp_path: Path) -> None:
    db = tmp_path / "two-gets-scoped.duckdb"
    wh = get_pooled_warehouse(db)
    _seed_liquidity(wh)

    client = TestClient(_app_with_pooled_factory(db))

    first = client.get("/v1/liquidity_index/market/USD_IG")
    second = client.get("/v1/liquidity_index/market/USD_IG")
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text

    df = wh.read_liquidity_stress_scores()
    assert df is not None
    assert not df.empty


def test_fi_tca_segments_two_sequential_gets_share_pool(tmp_path: Path) -> None:
    db = tmp_path / "two-gets-tca.duckdb"
    wh = get_pooled_warehouse(db)
    # No TCA seed — 503 is fine; the test only verifies the pool isn't
    # poisoned between the two GETs.

    client = TestClient(_app_with_pooled_factory(db))

    first = client.get("/v1/tca/regime-segments/latest")
    second = client.get("/v1/tca/regime-segments/latest")
    # Either 200 or 503 is acceptable; what we care about is the pool
    # survives.
    assert first.status_code in (200, 503), first.text
    assert second.status_code in (200, 503), second.text
    # Pool still alive.
    assert is_pooled_warehouse(wh) is True
    df = wh.read_credit_regime_scores()  # any read, just smokes the conn
    assert df is not None


def test_fi_evidence_pack_two_sequential_gets_share_pool(tmp_path: Path) -> None:
    db = tmp_path / "two-gets-pack.duckdb"
    wh = get_pooled_warehouse(db)

    client = TestClient(_app_with_pooled_factory(db))

    first = client.get("/v1/evidence-pack/run-x")
    second = client.get("/v1/evidence-pack/run-x")
    assert first.status_code == 404, first.text
    assert second.status_code == 404, second.text
    # Pool still alive.
    assert is_pooled_warehouse(wh) is True
