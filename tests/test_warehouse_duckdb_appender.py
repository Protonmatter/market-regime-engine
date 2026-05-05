# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the v1.4 DuckDB appender (item C).

Three contracts:

1. **Throughput** — ``test_duckdb_bulk_write_10k_rows_under_2s`` exercises
   the new ``register`` + ``INSERT ... SELECT ... ON CONFLICT`` path on
   the largest-PK table (``vintage_observations``) and asserts the
   wall-clock for a 10k-row write stays under 2 seconds. The v1.3
   executemany loop took ~427s on the same payload; the bulk-load
   target is 25-50x faster.
2. **Default routing** — ``test_warehouse_default_routes_to_duckdb_for_new_path``
   pins the v1.4 default-backend flip: a path with no recognised suffix
   (e.g. ``/tmp/x.warehouse``) routes to DuckDB out-of-the-box, and
   ``data/mre.duckdb`` resolves to the DuckDB backend via ``backend="auto"``.
3. **End-to-end smoke** — ``test_warehouse_smoke_against_duckdb_under_60s``
   writes a representative cross-section of every table and asserts the
   total wall-clock stays well under 60s. The v1.3 smoke at the same
   payload size was 427s; we expect <30s on the bulk-load path.

Plus:

- The existing v1.3 parity tests in ``test_v1_3_fixes.py`` still pass
  byte-for-byte (no change to those assertions in v1.4).
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _require_duckdb() -> None:
    pytest.importorskip("duckdb")


def _make_vintage_frame(n: int) -> pd.DataFrame:
    """Build an n-row vintage_observations frame whose PK is unique.

    PK is ``(series_id, observation_date, vintage_date)``. We walk
    ``series_id`` over a 200-bucket cycle and shift the vintage by the
    row index so every row gets a distinct (date, vintage) tuple.
    """
    base = pd.Timestamp("2000-01-01")
    rows = []
    for i in range(n):
        obs_offset = i  # unique per row
        vintage_offset = i + 1
        rows.append(
            {
                "series_id": f"S{i % 200:03d}",
                "observation_date": (base + pd.Timedelta(days=obs_offset)).strftime("%Y-%m-%d"),
                "value": float(i),
                "realtime_start": (base + pd.Timedelta(days=vintage_offset)).strftime("%Y-%m-%d"),
                "realtime_end": None,
                "vintage_date": (base + pd.Timedelta(days=vintage_offset)).strftime("%Y-%m-%d"),
                "source": "perf",
                "ingested_at_utc": "2026-05-01T00:00:00+00:00",
                "metadata_json": "{}",
            }
        )
    return pd.DataFrame(rows)


def test_duckdb_bulk_write_10k_rows_under_2s() -> None:
    """v1.4 (criterion 12): the bulk-load path writes 10k rows in <2s."""
    _require_duckdb()
    from market_regime_engine.storage import Warehouse

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "perf.duckdb"
        wh = Warehouse(str(path), backend="duckdb")
        try:
            df = _make_vintage_frame(10_000)
            t0 = time.perf_counter()
            n = wh.write_vintage_observations(df)
            elapsed = time.perf_counter() - t0
            assert n == 10_000, f"expected 10_000 rows, wrote {n}"
            assert elapsed < 2.0, f"bulk-write took {elapsed:.2f}s, exceeds 2s budget"
            # Sanity: rows are queryable without a roundtrip.
            read = wh.read_vintage_observations()
            assert len(read) == 10_000
        finally:
            wh.close()


def test_warehouse_default_routes_to_duckdb_for_new_path() -> None:
    """v1.4 default-backend flip: unrecognised suffixes pick DuckDB."""
    _require_duckdb()
    from market_regime_engine.storage import Warehouse

    with tempfile.TemporaryDirectory() as tmp:
        # Path with no recognised suffix -> DuckDB.
        path_unknown = Path(tmp) / "wh.warehouse"
        wh = Warehouse(str(path_unknown))
        try:
            assert wh.backend_name == "duckdb"
        finally:
            wh.close()
        # Explicit .duckdb suffix also routes to DuckDB.
        path_duck = Path(tmp) / "wh.duckdb"
        wh = Warehouse(str(path_duck))
        try:
            assert wh.backend_name == "duckdb"
        finally:
            wh.close()
        # Existing .db / .sqlite suffix preserves SQLite back-compat.
        path_sqlite = Path(tmp) / "wh.db"
        wh = Warehouse(str(path_sqlite))
        try:
            assert wh.backend_name == "sqlite"
        finally:
            wh.close()


def test_warehouse_default_dataclass_field_is_auto() -> None:
    """The Warehouse dataclass declares ``backend="auto"`` as the default."""
    from market_regime_engine.storage import Warehouse

    fields = Warehouse.__dataclass_fields__  # type: ignore[attr-defined]
    assert fields["backend"].default == "auto"


def test_warehouse_smoke_against_duckdb_under_60s() -> None:
    """v1.4 (criterion 5): the end-to-end smoke flow stays <60s on DuckDB.

    Skipped on CI runners that opt out via ``MRE_SKIP_PERF=1``. The
    payload is intentionally small (50 obs/series × 8 series + a few
    per-table writes) so the budget is dominated by the appender path,
    not by feature engineering.
    """
    if os.environ.get("MRE_SKIP_PERF") == "1":
        pytest.skip("MRE_SKIP_PERF=1 — perf budget gate disabled")
    _require_duckdb()
    from market_regime_engine.storage import Warehouse

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "smoke.duckdb"
        t0 = time.perf_counter()
        wh = Warehouse(str(path))
        try:
            assert wh.backend_name == "duckdb"
            # Observations: 50 per series * 8 series = 400 rows.
            obs_rows = []
            for i, s in enumerate(["UNRATE", "DGS10", "PAYEMS", "CPIAUCSL", "BAA10Y", "PERMIT", "HOUST", "DCOILWTICO"]):
                for k in range(50):
                    obs_rows.append(
                        {
                            "series_id": s,
                            "date": f"2020-{(k % 12) + 1:02d}-01",
                            "value": float(i * 100 + k),
                            "vintage_date": f"2020-{(k % 12) + 1:02d}-15",
                            "source": "smoke",
                        }
                    )
            wh.write_observations(pd.DataFrame(obs_rows))

            feats = pd.DataFrame(
                [
                    {"feature_name": f"f{i}", "date": "2020-01-01", "value": float(i), "domain": "labor"}
                    for i in range(60)
                ]
            )
            wh.write_features(feats)

            wh.write_regimes(
                pd.DataFrame(
                    [
                        {
                            "date": "2020-01-01",
                            "regime": "expansion",
                            "decoded_regime": "risk_on_expansion",
                            "score": 0.75,
                            "change_point_prob": 0.05,
                            "metadata_json": "{}",
                        }
                    ]
                )
            )

            wh.write_model_outputs(
                pd.DataFrame(
                    [
                        {
                            "model_name": "logreg",
                            "date": "2020-01-01",
                            "horizon": "3m",
                            "target": "recession",
                            "value": 0.12,
                            "metadata_json": "{}",
                        }
                    ]
                )
            )

            wh.write_recession_labels(
                pd.DataFrame(
                    [
                        {
                            "date": "2020-01-01",
                            "recession": 0.0,
                            "source": "USREC",
                            "metadata_json": "{}",
                        }
                    ]
                )
            )

            wh.write_invalidation_triggers(
                pd.DataFrame(
                    [
                        {
                            "date": "2020-01-01",
                            "trigger": "psi_breach",
                            "severity": "low",
                            "status": "ok",
                            "value": 0.05,
                            "threshold": 0.20,
                            "metadata_json": "{}",
                        }
                    ]
                )
            )

            wh.write_release_gates(
                pd.DataFrame(
                    [
                        {
                            "date": "2020-01-01",
                            "approved": True,
                            "decision": "go",
                            "confidence": 0.7,
                            "confidence_grade": "B",
                            "severe_drift": 0,
                            "major_drift": 0,
                            "max_psi": 0.05,
                            "high_invalidation_triggers": 0,
                            "active_trigger_names": "",
                            "reasons": "ok",
                            "metadata_json": "{}",
                        }
                    ]
                )
            )
            elapsed = time.perf_counter() - t0
            assert elapsed < 60.0, f"end-to-end smoke {elapsed:.2f}s exceeds 60s budget"
        finally:
            wh.close()


def test_warehouse_v14_new_tables_are_writable() -> None:
    """v1.4 schema additions: the two new tables are writable + readable."""
    _require_duckdb()
    from market_regime_engine.storage import Warehouse

    with tempfile.TemporaryDirectory() as tmp:
        wh = Warehouse(str(Path(tmp) / "v14.duckdb"))
        try:
            n1 = wh.write_bayesian_msvar_diagnostics(
                pd.DataFrame(
                    [
                        {
                            "run_id": "test_v14",
                            "method": "nuts",
                            "num_chains": 2,
                            "num_divergences": 0,
                            "max_rhat": 1.04,
                            "min_ess": 250.0,
                            "runtime_seconds": 12.5,
                            "metadata_json": "{}",
                        }
                    ]
                )
            )
            assert n1 == 1
            assert not wh.read_bayesian_msvar_diagnostics().empty

            n2 = wh.write_release_calendar_refreshes(
                pd.DataFrame(
                    [
                        {
                            "agency": "bls",
                            "fetched_at_utc": "2026-05-01T00:00:00Z",
                            "entries_count": 12,
                            "status": "ok",
                            "error": None,
                            "source_hash": "abc123",
                            "metadata_json": "{}",
                        }
                    ]
                )
            )
            assert n2 == 1
            assert not wh.read_release_calendar_refreshes().empty
        finally:
            wh.close()


def test_warehouse_duckdb_upsert_does_not_duplicate() -> None:
    """The bulk-load path retains ON CONFLICT semantics — re-write upserts in place."""
    _require_duckdb()
    from market_regime_engine.storage import Warehouse

    with tempfile.TemporaryDirectory() as tmp:
        wh = Warehouse(str(Path(tmp) / "u.duckdb"))
        try:
            df1 = pd.DataFrame(
                [
                    {"series_id": "X", "date": "2020-01-01", "value": 1.0, "vintage_date": "2020-02-01", "source": "a"},
                    {"series_id": "X", "date": "2020-02-01", "value": 2.0, "vintage_date": "2020-03-01", "source": "a"},
                ]
            )
            wh.write_observations(df1)
            # Same PK, different value → must replace, not duplicate.
            df2 = pd.DataFrame(
                [
                    {"series_id": "X", "date": "2020-01-01", "value": 9.9, "vintage_date": "2020-02-01", "source": "b"},
                ]
            )
            wh.write_observations(df2)
            read = wh.read_observations()
            assert len(read) == 2
            jan = read[read["date"] == "2020-01-01"].iloc[0]
            assert float(jan["value"]) == pytest.approx(9.9)
            assert str(jan["source"]) == "b"
        finally:
            wh.close()
