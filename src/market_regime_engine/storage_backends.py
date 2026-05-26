# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from market_regime_engine.storage_registry import (
    BackendName,
    _get_table_pk,
    registered_tables,
)

class _Backend(Protocol):
    """Minimal backend interface used by the Warehouse facade."""

    def init_schema(self) -> None: ...
    def upsert(
        self,
        table: str,
        rows: list[tuple],
        cols: list[str],
        *,
        mode: str = "REPLACE",
    ) -> None: ...
    # v1.4 (item C): bulk-load entry point. When the caller already has
    # a DataFrame (the Warehouse facade always does) the DuckDB backend
    # uses ``register`` + ``INSERT ... SELECT ... ON CONFLICT`` instead
    # of executemany. The default implementation falls back to the
    # row-tuple ``upsert`` so the protocol stays back-compat.
    def upsert_frame(
        self,
        table: str,
        frame: pd.DataFrame,
        cols: list[str],
        *,
        mode: str = "REPLACE",
    ) -> None: ...
    def execute(self, sql: str, params: tuple = ()) -> None: ...
    def commit(self) -> None: ...
    def read_sql(self, sql: str, params: tuple | list | None = None) -> pd.DataFrame: ...
    def close(self) -> None: ...
    def column_names(self, table: str) -> set[str]: ...



def _sqlite_scalar(value: Any) -> Any:
    """Coerce pandas/numpy scalar values to sqlite3-bindable values."""
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if hasattr(value, "isoformat") and value.__class__.__module__.startswith(("datetime", "pandas")):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    return value


def _sqlite_rows(rows: list[tuple]) -> list[tuple]:
    return [tuple(_sqlite_scalar(v) for v in row) for row in rows]


def _quote_columns(cols: Iterable[str]) -> str:
    """Quote SQLite/DuckDB reserved words like ``group``."""
    out = []
    for c in cols:
        if c.lower() in {"group", "order", "table", "select", "from", "where"}:
            out.append(f'"{c}"')
        else:
            out.append(c)
    return ", ".join(out)


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------


@dataclass
class _SqliteBackend:
    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        # Concurrency hardening (second-opinion #3): WAL allows concurrent
        # readers + a single writer, ``synchronous=NORMAL`` is the documented
        # WAL companion, ``busy_timeout`` blocks competing writers up to 10s
        # before raising "database is locked", and ``foreign_keys=ON`` enforces
        # the implicit FK relationships in the schema.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA busy_timeout=10000")
        self.conn.execute("PRAGMA foreign_keys=ON")

    def init_schema(self) -> None:
        # v1.5 (PR-2 ASK-3): iterate the explicit registry instead of the
        # module-level SCHEMA_STATEMENTS tuple. Per-table indexes (ASK-11)
        # are applied after the CREATE TABLE statement so the table
        # exists when ``CREATE INDEX`` runs.
        for spec in registered_tables():
            ddl = spec.sqlite_create_sql or spec.create_sql
            self.conn.execute(ddl)
            for idx_sql in spec.index_sql:
                self.conn.execute(idx_sql)
        self.conn.commit()
        # Idempotent backfill for the v1.1 release_gates.severe_drift column.
        cols = pd.read_sql_query("PRAGMA table_info(release_gates)", self.conn)
        col_names = set(cols.get("name", []))
        if "severe_drift" not in col_names:
            with contextlib.suppress(Exception):
                self.conn.execute("ALTER TABLE release_gates ADD COLUMN severe_drift INTEGER")
                self.conn.commit()
        # v1.5 (PR-1 ASK-7) idempotent backfill for resolved_profile.
        if "resolved_profile" not in col_names:
            with contextlib.suppress(Exception):
                self.conn.execute("ALTER TABLE release_gates ADD COLUMN resolved_profile TEXT")
                self.conn.commit()
        exec_cols = pd.read_sql_query("PRAGMA table_info(execution_confidence_predictions)", self.conn)
        exec_col_names = set(exec_cols.get("name", []))
        for name in (
            "notional_cents",
            "confidence_score_ppm",
            "expected_slippage_bps_q4",
            "confidence_interval_low_ppm",
            "confidence_interval_high_ppm",
        ):
            if name not in exec_col_names:
                with contextlib.suppress(Exception):
                    self.conn.execute(f"ALTER TABLE execution_confidence_predictions ADD COLUMN {name} INTEGER")
                    self.conn.commit()

    def upsert(
        self,
        table: str,
        rows: list[tuple],
        cols: list[str],
        *,
        mode: str = "REPLACE",
    ) -> None:
        if not rows:
            return
        col_sql = _quote_columns(cols)
        placeholders = ", ".join(["?"] * len(cols))
        # SQLite supports INSERT OR REPLACE and INSERT OR IGNORE natively.
        self.conn.executemany(
            f"INSERT OR {mode} INTO {table} ({col_sql}) VALUES ({placeholders})",
            _sqlite_rows(rows),
        )

    def upsert_frame(
        self,
        table: str,
        frame: pd.DataFrame,
        cols: list[str],
        *,
        mode: str = "REPLACE",
    ) -> None:
        # SQLite's executemany is already the fastest available path; the
        # frame entry point just delegates so the Warehouse facade can
        # always call ``upsert_frame`` regardless of backend.
        if frame is None or frame.empty:
            return
        rows = list(frame[cols].itertuples(index=False, name=None))
        self.upsert(table, rows, cols, mode=mode)

    def execute(self, sql: str, params: tuple = ()) -> None:
        self.conn.execute(sql, params)

    def commit(self) -> None:
        self.conn.commit()

    def read_sql(self, sql: str, params: tuple | list | None = None) -> pd.DataFrame:
        if params is None:
            return pd.read_sql_query(sql, self.conn)
        return pd.read_sql_query(sql, self.conn, params=tuple(params))

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.conn.close()

    def column_names(self, table: str) -> set[str]:
        try:
            cols = pd.read_sql_query(f"PRAGMA table_info({table})", self.conn)
        except Exception:
            return set()
        return set(cols.get("name", []))


# ---------------------------------------------------------------------------
# DuckDB backend
# ---------------------------------------------------------------------------


@dataclass
class _DuckDBBackend:
    path: Path

    def __post_init__(self) -> None:
        try:
            import duckdb
        except Exception as exc:  # pragma: no cover - import path
            raise RuntimeError(
                "duckdb backend requested but the duckdb package is not installed; "
                "install with `pip install market-regime-engine[analytics]`."
            ) from exc
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.path))

    def init_schema(self) -> None:
        # v1.5 (PR-2 ASK-3 / ASK-11): walk the explicit registry, run
        # the CREATE TABLE then any CREATE INDEX statements for each
        # spec. DuckDB accepts TEXT/REAL/INTEGER as aliases for the v1.4
        # core tables, plus native TIMESTAMP/DECIMAL/JSON for the FI
        # tables registered by the fixed_income package.
        for spec in registered_tables():
            self.conn.execute(spec.create_sql)
            for idx_sql in spec.index_sql:
                self.conn.execute(idx_sql)
        # release_gates.severe_drift back-compat column. DuckDB's
        # information_schema is the canonical way to inspect a table.
        cols = self.conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'release_gates'"
        ).fetchall()
        col_names = {c[0] for c in cols}
        if "severe_drift" not in col_names:
            with contextlib.suppress(Exception):
                self.conn.execute("ALTER TABLE release_gates ADD COLUMN severe_drift INTEGER")
        # v1.5 (PR-1 ASK-7) idempotent backfill for resolved_profile.
        if "resolved_profile" not in col_names:
            with contextlib.suppress(Exception):
                self.conn.execute("ALTER TABLE release_gates ADD COLUMN resolved_profile TEXT")
        exec_cols = self.conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'execution_confidence_predictions'"
        ).fetchall()
        exec_col_names = {c[0] for c in exec_cols}
        for name in (
            "notional_cents",
            "confidence_score_ppm",
            "expected_slippage_bps_q4",
            "confidence_interval_low_ppm",
            "confidence_interval_high_ppm",
        ):
            if name not in exec_col_names:
                with contextlib.suppress(Exception):
                    self.conn.execute(f"ALTER TABLE execution_confidence_predictions ADD COLUMN {name} BIGINT")

    @staticmethod
    def _build_upsert_sql(
        table: str,
        cols: list[str],
        pk: tuple[str, ...],
        *,
        mode: str,
        source: str,
    ) -> str:
        """Construct the INSERT...ON CONFLICT statement for ``table``.

        ``source`` is either a VALUES placeholder list (``executemany``
        path) or a ``SELECT ... FROM __staging`` clause (bulk-load path).
        """
        col_sql = _quote_columns(cols)
        if mode == "REPLACE" and pk:
            non_pk = [c for c in cols if c not in pk]
            conflict_cols = _quote_columns(pk)
            if non_pk:
                set_clause = ", ".join(f"{_quote_columns([c])} = EXCLUDED.{_quote_columns([c])}" for c in non_pk)
                tail = f"ON CONFLICT ({conflict_cols}) DO UPDATE SET {set_clause}"
            else:
                tail = f"ON CONFLICT ({conflict_cols}) DO NOTHING"
        elif mode == "IGNORE" and pk:
            conflict_cols = _quote_columns(pk)
            tail = f"ON CONFLICT ({conflict_cols}) DO NOTHING"
        else:
            tail = ""
        return f"INSERT INTO {table} ({col_sql}) {source} {tail}".rstrip()

    def upsert(
        self,
        table: str,
        rows: list[tuple],
        cols: list[str],
        *,
        mode: str = "REPLACE",
    ) -> None:
        # v1.4 (item C): the row-tuple path is preserved for callers that
        # cannot construct a DataFrame (e.g. ``warehouse-migrate`` over
        # heterogeneous tables). It still uses ON CONFLICT, just via
        # executemany rather than the bulk-load INSERT...SELECT pattern.
        if not rows:
            return
        placeholders = "VALUES (" + ", ".join(["?"] * len(cols)) + ")"
        pk = _get_table_pk(table)
        sql = self._build_upsert_sql(table, cols, pk, mode=mode, source=placeholders)
        # Wrap the executemany batch in an explicit transaction so a
        # crash mid-batch leaves the table consistent (DuckDB autocommits
        # per statement otherwise).
        self.conn.execute("BEGIN TRANSACTION")
        try:
            self.conn.executemany(sql, rows)
        except Exception:
            with contextlib.suppress(Exception):
                self.conn.execute("ROLLBACK")
            raise
        self.conn.execute("COMMIT")

    def upsert_frame(
        self,
        table: str,
        frame: pd.DataFrame,
        cols: list[str],
        *,
        mode: str = "REPLACE",
    ) -> None:
        """Bulk-load via ``register`` + ``INSERT ... SELECT ... ON CONFLICT``.

        This is the v1.4 fast path. It avoids the per-row Python ↔ C
        round-trip that made executemany dominate the v1.3 smoke runs
        (7.6s SQLite vs 427s DuckDB on the same payload). DuckDB
        ``register`` zero-copies the pandas frame as a virtual table so
        the INSERT becomes a single columnar plan.
        """
        if frame is None or frame.empty:
            return
        # The DuckDB ``register`` API requires explicit, named columns
        # that match the destination table. We project a defensive copy
        # so caller-side mutations cannot corrupt the staging frame.
        staging = frame[cols].copy()
        # Stable, table-scoped staging name so concurrent writers do not
        # collide on the same registered view name within a connection
        # cursor.
        staging_name = f"__mre_staging_{table}"
        select_sql = f"SELECT {_quote_columns(cols)} FROM {staging_name}"
        pk = _get_table_pk(table)
        insert_sql = self._build_upsert_sql(table, cols, pk, mode=mode, source=select_sql)

        # Wrap in an explicit transaction so the bulk insert is atomic.
        self.conn.execute("BEGIN TRANSACTION")
        try:
            self.conn.register(staging_name, staging)
            try:
                self.conn.execute(insert_sql)
            finally:
                with contextlib.suppress(Exception):
                    self.conn.unregister(staging_name)
        except Exception:
            with contextlib.suppress(Exception):
                self.conn.execute("ROLLBACK")
            raise
        self.conn.execute("COMMIT")

    def execute(self, sql: str, params: tuple = ()) -> None:
        self.conn.execute(sql, params)

    def commit(self) -> None:
        # DuckDB autocommits each statement; this is a no-op so callers
        # can use a single Warehouse facade across both backends.
        with contextlib.suppress(Exception):
            self.conn.commit()

    def read_sql(self, sql: str, params: tuple | list | None = None) -> pd.DataFrame:
        if params is None:
            return self.conn.execute(sql).fetchdf()
        return self.conn.execute(sql, list(params)).fetchdf()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.conn.close()

    def column_names(self, table: str) -> set[str]:
        try:
            cols = self.conn.execute(
                f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}'"
            ).fetchall()
        except Exception:
            return set()
        return {c[0] for c in cols}


# ---------------------------------------------------------------------------
# Warehouse facade
# ---------------------------------------------------------------------------
def _select_backend(path: Path, backend: BackendName) -> _Backend:
    if backend == "duckdb":
        return _DuckDBBackend(path=path)
    if backend == "sqlite":
        return _SqliteBackend(path=path)
    if backend == "auto":
        # v1.4 (item C step 2): the auto-detect rule is suffix-driven:
        #   ``.db`` / ``.sqlite`` / ``.sqlite3`` -> SQLite (back-compat
        #     for every v1.3 deployment),
        #   ``.duckdb`` -> DuckDB,
        #   no recognised suffix -> DuckDB. The CLI default flipped from
        #     ``data/mre.db`` to ``data/mre.duckdb`` so a bare ``mre``
        #     run picks DuckDB out-of-the-box; existing ``--db data/mre.db``
        #     callers keep using SQLite.
        suffix = path.suffix.lower()
        if suffix in {".db", ".sqlite", ".sqlite3"}:
            return _SqliteBackend(path=path)
        if suffix == ".duckdb":
            try:
                import duckdb

                return _DuckDBBackend(path=path)
            except Exception:
                # Fall back to SQLite if the optional duckdb package
                # is missing. Surfaces a clear suffix→backend mismatch
                # instead of crashing on import.
                return _SqliteBackend(path=path)
        # No recognised suffix: prefer DuckDB when the optional
        # package is available, otherwise SQLite.
        try:
            import duckdb  # noqa: F401  type: ignore[import-not-found]

            return _DuckDBBackend(path=path)
        except Exception:
            return _SqliteBackend(path=path)
    raise ValueError(f"Unknown backend: {backend!r}")


__all__ = [
    "_Backend",
    "_SqliteBackend",
    "_DuckDBBackend",
    "_select_backend",
    "_quote_columns",
]
