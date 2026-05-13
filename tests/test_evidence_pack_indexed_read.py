# SPDX-License-Identifier: Apache-2.0
"""v1.6.0 — A8 / Finding §3.3 regression tests.

Pin the contract that

1. ``Warehouse.latest_evidence_pack(model_run_id)`` returns the most
   recent pack on a fixture with 100+ packs (i.e. SQL fast path works
   correctly, not a partial read).
2. ``Warehouse.latest_evidence_pack(model_run_id, request_id=...)``
   returns the exact match.
3. ``read_evidence_pack`` (FI module accessor) consumes the new method
   when available and falls back to legacy ``read_evidence_packs()``
   when the warehouse object is a v1.5.x mock.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import market_regime_engine.fixed_income  # noqa: F401  (table registration)
from market_regime_engine.fixed_income.evidence_pack import read_evidence_pack
from market_regime_engine.storage import Warehouse


@pytest.fixture
def wh(tmp_path: Path) -> Warehouse:
    return Warehouse(tmp_path / "ep.duckdb")


def _seed_packs(wh: Warehouse, n: int = 150) -> list[dict]:
    """Seed N evidence-pack rows under 3 model_run_ids."""
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    rows = []
    for i in range(n):
        rows.append(
            {
                "model_run_id": f"credit_spread_regime-production-{i % 3}",
                "request_id": f"req-{i:04d}",
                "component_name": "credit_spread_regime",
                "model_version": "v1.6.0",
                "timestamp": (base + pd.Timedelta(seconds=i * 10)).isoformat(),
                "code_sha": "abc123",
                "model_hash": f"model-hash-{i}",
                "input_features_hash": f"feat-hash-{i}",
                "output_hash": f"out-hash-{i}",
                "data_vintages_json": "{}",
                "validation_results_json": "{}",
                "release_gate": 1,
                "random_seeds_json": "{}",
                "python_version": "3.13.4",
                "lockfile_hash": None,
                "hmac_signature": None,
                "metadata_json": "{}",
            }
        )
    wh.write_evidence_pack(pd.DataFrame(rows))
    return rows


def test_latest_evidence_pack_returns_most_recent_for_model_run_id(
    wh: Warehouse,
) -> None:
    """A8: with 150 seeded packs, the indexed lookup returns the row
    with the maximum timestamp for the requested model_run_id."""
    rows = _seed_packs(wh, n=150)
    target_run_id = "credit_spread_regime-production-1"

    # The expected row is the one with the latest timestamp under that
    # run_id; deterministic from the seeding above.
    expected = max(
        (r for r in rows if r["model_run_id"] == target_run_id),
        key=lambda r: r["timestamp"],
    )

    out = wh.latest_evidence_pack(target_run_id)
    assert out is not None
    assert len(out) == 1
    assert out.iloc[0]["model_run_id"] == target_run_id
    assert out.iloc[0]["request_id"] == expected["request_id"]
    assert out.iloc[0]["timestamp"] == expected["timestamp"]


def test_latest_evidence_pack_with_request_id_returns_exact_match(
    wh: Warehouse,
) -> None:
    """A8: ``request_id`` narrows the result to that exact pack."""
    _seed_packs(wh, n=150)
    out = wh.latest_evidence_pack(
        "credit_spread_regime-production-2",
        request_id="req-0050",
    )
    assert out is not None
    assert len(out) == 1
    assert out.iloc[0]["request_id"] == "req-0050"


def test_latest_evidence_pack_returns_none_for_unknown_model_run_id(
    wh: Warehouse,
) -> None:
    """A8: an unknown model_run_id yields None (caller chooses 503 vs
    fallback)."""
    _seed_packs(wh, n=10)
    out = wh.latest_evidence_pack("nonexistent-run-id")
    assert out is None


def test_read_evidence_pack_uses_indexed_path_when_available(
    wh: Warehouse,
) -> None:
    """A8: the public ``read_evidence_pack`` accessor consumes the
    indexed path and returns the expected pack."""
    rows = _seed_packs(wh, n=150)
    target_run_id = "credit_spread_regime-production-0"
    expected = max(
        (r for r in rows if r["model_run_id"] == target_run_id),
        key=lambda r: r["timestamp"],
    )
    pack = read_evidence_pack(wh, model_run_id=target_run_id)
    assert pack is not None
    assert pack.model_run_id == target_run_id
    assert pack.request_id == expected["request_id"]


def test_read_evidence_pack_falls_back_to_legacy_for_v1_5_mocks(
    wh: Warehouse,
) -> None:
    """A8: callers / test mocks that only implement
    ``read_evidence_packs()`` (no ``latest_evidence_pack``) continue to
    work."""
    rows = _seed_packs(wh, n=50)

    class _LegacyMock:
        def read_evidence_packs(self) -> pd.DataFrame:
            return wh.read_evidence_packs()

    target_run_id = "credit_spread_regime-production-0"
    expected = max(
        (r for r in rows if r["model_run_id"] == target_run_id),
        key=lambda r: r["timestamp"],
    )
    pack = read_evidence_pack(_LegacyMock(), model_run_id=target_run_id)
    assert pack is not None
    assert pack.model_run_id == target_run_id
    assert pack.request_id == expected["request_id"]
