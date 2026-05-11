from __future__ import annotations

import contextlib
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path
from typing import Literal, Protocol

import pandas as pd

# v1.3 (item D): the warehouse becomes a thin facade over a backend
# protocol so DuckDB can be used as the primary store without breaking
# SQLite callers. ``backend="sqlite"`` is the default for back-compat;
# ``backend="auto"`` picks DuckDB when ``path.suffix == ".duckdb"`` and
# the optional ``duckdb`` package is importable. Schema parity is
# verified end-to-end by ``tests/test_warehouse_duckdb_parity.py``.

BackendName = Literal["sqlite", "duckdb", "auto"]


# ---------------------------------------------------------------------------
# Schema definitions (shared across backends)
# ---------------------------------------------------------------------------


SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS observations (
        series_id TEXT NOT NULL,
        date TEXT NOT NULL,
        value REAL NOT NULL,
        vintage_date TEXT,
        source TEXT NOT NULL,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(series_id, date, vintage_date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS features (
        feature_name TEXT NOT NULL,
        date TEXT NOT NULL,
        value REAL NOT NULL,
        domain TEXT,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(feature_name, date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS regimes (
        date TEXT PRIMARY KEY,
        regime TEXT NOT NULL,
        decoded_regime TEXT NOT NULL,
        score REAL NOT NULL,
        change_point_prob REAL,
        metadata_json TEXT DEFAULT '{}'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_outputs (
        model_name TEXT NOT NULL,
        date TEXT NOT NULL,
        horizon TEXT NOT NULL,
        target TEXT NOT NULL,
        value REAL NOT NULL,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(model_name, date, horizon, target)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS recession_labels (
        date TEXT PRIMARY KEY,
        recession REAL NOT NULL,
        source TEXT NOT NULL,
        metadata_json TEXT DEFAULT '{}'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS historical_analogs (
        as_of_date TEXT NOT NULL,
        analog_date TEXT NOT NULL,
        rank INTEGER NOT NULL,
        distance REAL NOT NULL,
        similarity REAL NOT NULL,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(as_of_date, analog_date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS driver_attribution (
        date TEXT NOT NULL,
        attribution_type TEXT NOT NULL,
        rank INTEGER NOT NULL,
        name TEXT NOT NULL,
        domain TEXT,
        value REAL,
        zscore REAL,
        change_3m REAL,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(date, attribution_type, rank, name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_runs (
        run_id TEXT PRIMARY KEY,
        created_at_utc TEXT NOT NULL,
        engine_version TEXT NOT NULL,
        purpose TEXT NOT NULL,
        data_start TEXT,
        data_end TEXT,
        feature_count INTEGER,
        observation_count INTEGER,
        model_count INTEGER,
        code_version TEXT,
        artifact_hash TEXT NOT NULL,
        metadata_json TEXT DEFAULT '{}'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS calibrated_outputs (
        model_name TEXT NOT NULL,
        date TEXT NOT NULL,
        horizon TEXT NOT NULL,
        target TEXT NOT NULL,
        value REAL NOT NULL,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(model_name, date, horizon, target)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS calibration_models (
        horizon TEXT NOT NULL,
        target TEXT NOT NULL,
        method TEXT NOT NULL,
        intercept REAL,
        slope REAL,
        fallback_rate REAL,
        observations INTEGER,
        raw_mean REAL,
        calibrated_mean REAL,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(horizon, target, method)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS release_calendar_audit (
        series_id TEXT PRIMARY KEY,
        rows INTEGER,
        violations INTEGER,
        coverage INTEGER,
        release_family TEXT,
        domain TEXT,
        min_required_release TEXT,
        max_actual_vintage TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS invalidation_triggers (
        date TEXT NOT NULL,
        trigger TEXT NOT NULL,
        severity TEXT NOT NULL,
        status TEXT NOT NULL,
        value REAL,
        threshold REAL,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(date, trigger)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS confidence_scores (
        date TEXT PRIMARY KEY,
        confidence REAL NOT NULL,
        grade TEXT NOT NULL,
        metadata_json TEXT DEFAULT '{}'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS exact_release_calendar (
        series_id TEXT NOT NULL,
        observation_date TEXT NOT NULL,
        release_timestamp_utc TEXT NOT NULL,
        domain TEXT,
        lag_days INTEGER,
        source TEXT,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(series_id, observation_date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ensemble_weights (
        horizon TEXT NOT NULL,
        target TEXT NOT NULL,
        model_name TEXT NOT NULL,
        weight REAL NOT NULL,
        method TEXT NOT NULL,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(horizon, target, model_name, method)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS stacking_diagnostics (
        horizon TEXT NOT NULL,
        target TEXT NOT NULL,
        observations INTEGER,
        log_loss REAL,
        brier REAL,
        model_count INTEGER,
        method TEXT,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(horizon, target, method)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_drift (
        date TEXT NOT NULL,
        feature_name TEXT NOT NULL,
        psi REAL NOT NULL,
        mean_shift REAL,
        status TEXT NOT NULL,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(date, feature_name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS release_gates (
        date TEXT PRIMARY KEY,
        approved INTEGER NOT NULL,
        decision TEXT NOT NULL,
        confidence REAL,
        confidence_grade TEXT,
        severe_drift INTEGER,
        major_drift INTEGER,
        max_psi REAL,
        high_invalidation_triggers INTEGER,
        active_trigger_names TEXT,
        reasons TEXT,
        metadata_json TEXT DEFAULT '{}',
        resolved_profile TEXT
    )
    """,
    # v1.5 PR-1 ASK-7 / P2: nullable resolved_profile column added above so
    # existing v1.4 rows round-trip without migration. The column carries
    # whichever profile (production / default) drove the threshold
    # selection per release_gates._resolve_profile. Note: the SQL comment
    # is placed *outside* the CREATE TABLE statement because
    # storage._extract_pk uses a naive parse that would mis-detect a
    # parenthesised comment as the PRIMARY KEY column list
    # (FLAG F-4 in REVIEW.md; the planned PR-2 register_tables refactor
    # replaces the regex parse with an explicit pk_map).
    """
    CREATE TABLE IF NOT EXISTS alfred_ingestion_manifest (
        series_id TEXT NOT NULL,
        realtime_start TEXT NOT NULL,
        realtime_end TEXT NOT NULL,
        observation_start TEXT,
        observation_end TEXT,
        rows INTEGER,
        status TEXT NOT NULL,
        error TEXT,
        ingested_at_utc TEXT NOT NULL,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(series_id, realtime_start, realtime_end)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS hazard_diagnostics (
        date TEXT NOT NULL,
        model_name TEXT NOT NULL,
        observations INTEGER,
        events INTEGER,
        feature_count INTEGER,
        constant_fallback INTEGER,
        latest_monthly_hazard REAL,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(date, model_name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS oos_predictions (
        date TEXT NOT NULL,
        model_name TEXT NOT NULL,
        horizon TEXT NOT NULL,
        target TEXT NOT NULL,
        value REAL NOT NULL,
        y REAL,
        regime_bucket TEXT,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(date, model_name, horizon, target)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS routed_alerts (
        date TEXT NOT NULL,
        alert_type TEXT NOT NULL,
        severity TEXT NOT NULL,
        channel TEXT NOT NULL,
        message TEXT NOT NULL,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(date, alert_type, channel)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS promotion_workflow (
        date TEXT NOT NULL,
        workflow TEXT NOT NULL,
        approved INTEGER NOT NULL,
        decision TEXT NOT NULL,
        confidence REAL,
        promoted_challenger_present INTEGER,
        release_gate_approved INTEGER,
        drift_ok INTEGER,
        reasons TEXT,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(date, workflow)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS series_vintages (
        series_id TEXT NOT NULL,
        vintage_date TEXT NOT NULL,
        source TEXT NOT NULL,
        ingested_at_utc TEXT NOT NULL,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(series_id, vintage_date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS vintage_observations (
        series_id TEXT NOT NULL,
        observation_date TEXT NOT NULL,
        value REAL NOT NULL,
        realtime_start TEXT NOT NULL,
        realtime_end TEXT,
        vintage_date TEXT NOT NULL,
        source TEXT NOT NULL,
        ingested_at_utc TEXT NOT NULL,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(series_id, observation_date, vintage_date)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS feature_asof_values (
        as_of_date TEXT NOT NULL,
        feature_name TEXT NOT NULL,
        source_series_id TEXT NOT NULL,
        observation_date TEXT NOT NULL,
        vintage_date TEXT NOT NULL,
        value REAL NOT NULL,
        transform_name TEXT NOT NULL,
        created_at_utc TEXT NOT NULL,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(as_of_date, feature_name)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS vintage_audits (
        audit TEXT NOT NULL,
        run_at_utc TEXT NOT NULL,
        rows INTEGER,
        violations INTEGER,
        status TEXT NOT NULL,
        details TEXT,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(audit, run_at_utc)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS conformal_coverage (
        as_of_date TEXT NOT NULL,
        target TEXT NOT NULL,
        horizon TEXT NOT NULL,
        bucket TEXT NOT NULL,
        n INTEGER NOT NULL,
        realized_coverage REAL NOT NULL,
        target_coverage REAL NOT NULL,
        threshold REAL,
        method TEXT NOT NULL,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(as_of_date, target, horizon, bucket, method)
    )
    """,
    # ----- v1.2 frontier tables -----
    """
    CREATE TABLE IF NOT EXISTS e_value_log (
        date TEXT NOT NULL,
        target TEXT NOT NULL,
        horizon TEXT NOT NULL,
        challenger TEXT NOT NULL,
        champion TEXT,
        e_value REAL NOT NULL,
        level REAL NOT NULL,
        decision TEXT NOT NULL,
        n INTEGER,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(date, target, horizon, challenger)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS nowcast_factors (
        as_of_date TEXT NOT NULL,
        domain TEXT NOT NULL,
        factor_value REAL NOT NULL,
        factor_se REAL,
        frequency_mix TEXT,
        backend TEXT,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(as_of_date, domain)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS conditional_coverage_report (
        as_of_date TEXT NOT NULL,
        target TEXT NOT NULL,
        horizon TEXT NOT NULL,
        "group" TEXT NOT NULL,
        coverage REAL NOT NULL,
        n INTEGER NOT NULL,
        alpha REAL NOT NULL,
        method TEXT NOT NULL,
        worst_violation REAL,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(as_of_date, target, horizon, "group", method)
    )
    """,
    # ----- v1.3 alert dispatch table -----
    """
    CREATE TABLE IF NOT EXISTS alert_dispatches (
        date TEXT NOT NULL,
        alert_type TEXT NOT NULL,
        sink TEXT NOT NULL,
        status TEXT NOT NULL,
        detail TEXT,
        dispatched_at_utc TEXT NOT NULL,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(date, alert_type, sink, dispatched_at_utc)
    )
    """,
    # ----- v1.4 Bayesian MS-VAR diagnostics (item A) -----
    """
    CREATE TABLE IF NOT EXISTS bayesian_msvar_diagnostics (
        run_id TEXT NOT NULL,
        method TEXT NOT NULL,
        num_chains INTEGER,
        num_divergences INTEGER,
        max_rhat REAL,
        min_ess REAL,
        runtime_seconds REAL,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(run_id, method)
    )
    """,
    # ----- v1.4 release calendar refresh outcomes (item D) -----
    """
    CREATE TABLE IF NOT EXISTS release_calendar_refreshes (
        agency TEXT NOT NULL,
        fetched_at_utc TEXT NOT NULL,
        entries_count INTEGER,
        status TEXT NOT NULL,
        error TEXT,
        source_hash TEXT,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(agency, fetched_at_utc)
    )
    """,
)


# Tables whose primary keys are required for the DuckDB ON CONFLICT
# clause. SQLite gets these for free via INSERT OR REPLACE, but
# DuckDB needs an explicit conflict target. We map the canonical
# column list at runtime; PK columns are extracted via simple parsing
# of the schema strings above (single source of truth).
def _extract_pk(schema_sql: str) -> tuple[str, ...]:
    """Parse the PRIMARY KEY tuple from a CREATE TABLE statement."""
    text = schema_sql.upper()
    idx = text.find("PRIMARY KEY")
    if idx < 0:
        return ()
    # Slice from PRIMARY KEY to the next closing paren.
    start = schema_sql.find("(", idx + len("PRIMARY KEY"))
    if start < 0:
        return ()
    depth = 1
    end = start + 1
    while end < len(schema_sql) and depth > 0:
        if schema_sql[end] == "(":
            depth += 1
        elif schema_sql[end] == ")":
            depth -= 1
        end += 1
    inner = schema_sql[start + 1 : end - 1]
    cols = [c.strip().strip('"') for c in inner.split(",") if c.strip()]
    return tuple(cols)


def _extract_table_name(schema_sql: str) -> str:
    text = schema_sql.upper()
    idx = text.find("CREATE TABLE")
    if idx < 0:
        return ""
    rest = schema_sql[idx + len("CREATE TABLE") :].strip()
    # Strip optional "IF NOT EXISTS"
    if rest.upper().startswith("IF NOT EXISTS"):
        rest = rest[len("IF NOT EXISTS") :].strip()
    name_end = rest.find("(")
    if name_end < 0:
        return ""
    return rest[:name_end].strip()


_TABLE_PKS: dict[str, tuple[str, ...]] = {_extract_table_name(sql): _extract_pk(sql) for sql in SCHEMA_STATEMENTS}


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


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
    def read_sql(self, sql: str) -> pd.DataFrame: ...
    def close(self) -> None: ...
    def column_names(self, table: str) -> set[str]: ...


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
        for sql in SCHEMA_STATEMENTS:
            self.conn.execute(sql)
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
            rows,
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

    def read_sql(self, sql: str) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.conn)

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
        # DuckDB SQL is mostly SQLite-compatible. The CREATE TABLE
        # IF NOT EXISTS statements above all parse cleanly (DuckDB
        # accepts TEXT, REAL, INTEGER as type aliases).
        for sql in SCHEMA_STATEMENTS:
            self.conn.execute(sql)
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
        pk = _TABLE_PKS.get(table, ())
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
        pk = _TABLE_PKS.get(table, ())
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

    def read_sql(self, sql: str) -> pd.DataFrame:
        return self.conn.execute(sql).fetchdf()

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


@dataclass
class Warehouse:
    """Storage facade for the Market Regime Engine.

    v1.3 (item D): the historical SQLite-only implementation has been
    refactored into a thin facade over a backend protocol with
    :class:`_SqliteBackend` and :class:`_DuckDBBackend` implementations.

    v1.4 (item C): the default is now ``backend="auto"`` and the
    backend is selected from the path suffix:
      - ``.db`` / ``.sqlite`` / ``.sqlite3``  → SQLite
        (every existing v1.3 deployment continues to work).
      - ``.duckdb`` (or any unrecognised suffix when the optional
        ``duckdb`` package is installed) → DuckDB.
    Pass ``backend="sqlite"`` explicitly to opt back into the v1.3
    SQLite-first behaviour.

    The DuckDB write path was rewritten in v1.4 to use
    ``register`` + ``INSERT ... SELECT ... ON CONFLICT``: the v1.3
    executemany loop dominated wall-clock during the smoke run
    (~427 s), and the new bulk-load brings the same end-to-end smoke
    under one minute.

    The public method surface (write_observations, read_regimes, ...)
    is unchanged. Schema parity is end-to-end tested in
    ``tests/test_v1_3_fixes.py`` and ``tests/test_warehouse_duckdb_appender.py``.
    """

    path: str | Path
    backend: BackendName = "auto"

    _backend: _Backend = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self._backend = _select_backend(self.path, self.backend)
        self.init_schema()

    # ----- forwarded protocol methods -----

    @property
    def conn(self):  # back-compat property used by ad-hoc callers
        # Some legacy callers (e.g. analytics_warehouse.export_sqlite_to_lake)
        # reach into ``warehouse.conn`` directly. Surface the underlying
        # connection from whichever backend is in use.
        return getattr(self._backend, "conn", None)

    @property
    def backend_name(self) -> str:
        return type(self._backend).__name__.removeprefix("_").removesuffix("Backend").lower()

    def init_schema(self) -> None:
        self._backend.init_schema()

    def init_release_gates_severe_column(self) -> None:
        # v1.1 back-compat method. The backend init_schema already
        # handles this idempotently; the public name is preserved for
        # any external callers that depended on it pre-v1.3.
        self._backend.init_schema()

    def close(self) -> None:
        self._backend.close()

    # ----- low-level write helpers used by the per-table writers -----

    def _write(self, table: str, df: pd.DataFrame, cols: list[str], mode: str = "REPLACE") -> int:
        if df is None or df.empty:
            return 0
        frame = df.copy()
        if "metadata_json" in cols and "metadata_json" not in frame:
            frame["metadata_json"] = "{}"
        for c in ["date", "vintage_date", "as_of_date", "analog_date"]:
            if c in frame:
                frame[c] = pd.to_datetime(frame[c]).dt.strftime("%Y-%m-%d")
        # v1.4 (item C): route through the bulk-load entry point. SQLite
        # delegates to executemany; DuckDB uses ``register`` +
        # ``INSERT ... SELECT ... ON CONFLICT`` for the columnar fast path.
        self._backend.upsert_frame(table, frame, cols, mode=mode)
        self._backend.commit()
        return len(frame)

    # ----- per-table writers -----

    def write_observations(self, df: pd.DataFrame) -> int:
        frame = df.copy()
        if "vintage_date" not in frame:
            frame["vintage_date"] = frame["date"]
        if "metadata_json" not in frame:
            frame["metadata_json"] = "{}"
        return self._write(
            "observations", frame, ["series_id", "date", "value", "vintage_date", "source", "metadata_json"]
        )

    def write_features(self, df: pd.DataFrame) -> int:
        frame = df.copy()
        if frame.empty:
            return 0
        frame = frame.dropna(subset=["value"])
        return self._write("features", frame, ["feature_name", "date", "value", "domain", "metadata_json"])

    def write_regimes(self, df: pd.DataFrame) -> int:
        return self._write(
            "regimes", df, ["date", "regime", "decoded_regime", "score", "change_point_prob", "metadata_json"]
        )

    def write_model_outputs(self, df: pd.DataFrame) -> int:
        return self._write("model_outputs", df, ["model_name", "date", "horizon", "target", "value", "metadata_json"])

    def write_recession_labels(self, df: pd.DataFrame) -> int:
        frame = df.copy()
        if "source" not in frame:
            frame["source"] = "unknown"
        return self._write("recession_labels", frame, ["date", "recession", "source", "metadata_json"])

    def write_historical_analogs(self, df: pd.DataFrame) -> int:
        return self._write(
            "historical_analogs", df, ["as_of_date", "analog_date", "rank", "distance", "similarity", "metadata_json"]
        )

    def write_driver_attribution(self, df: pd.DataFrame, attribution_type: str) -> int:
        if df is None or df.empty:
            return 0
        frame = df.copy()
        if attribution_type == "domain":
            frame["name"] = frame["domain"]
            frame["value"] = frame["score"]
        else:
            frame["name"] = frame["feature_name"]
        frame["attribution_type"] = attribution_type
        if "domain" not in frame:
            frame["domain"] = None
        if "change_3m" not in frame:
            frame["change_3m"] = None
        return self._write(
            "driver_attribution",
            frame,
            ["date", "attribution_type", "rank", "name", "domain", "value", "zscore", "change_3m", "metadata_json"],
        )

    def write_model_runs(self, df: pd.DataFrame) -> int:
        return self._write(
            "model_runs",
            df,
            [
                "run_id",
                "created_at_utc",
                "engine_version",
                "purpose",
                "data_start",
                "data_end",
                "feature_count",
                "observation_count",
                "model_count",
                "code_version",
                "artifact_hash",
                "metadata_json",
            ],
            mode="IGNORE",
        )

    def write_calibrated_outputs(self, df: pd.DataFrame) -> int:
        return self._write(
            "calibrated_outputs", df, ["model_name", "date", "horizon", "target", "value", "metadata_json"]
        )

    def write_calibration_models(self, df: pd.DataFrame) -> int:
        return self._write(
            "calibration_models",
            df,
            [
                "horizon",
                "target",
                "method",
                "intercept",
                "slope",
                "fallback_rate",
                "observations",
                "raw_mean",
                "calibrated_mean",
                "metadata_json",
            ],
        )

    def write_release_calendar_audit(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        frame = df.copy()
        frame["coverage"] = frame["coverage"].astype(int)
        return self._write(
            "release_calendar_audit",
            frame,
            [
                "series_id",
                "rows",
                "violations",
                "coverage",
                "release_family",
                "domain",
                "min_required_release",
                "max_actual_vintage",
            ],
        )

    def write_invalidation_triggers(self, df: pd.DataFrame) -> int:
        return self._write(
            "invalidation_triggers",
            df,
            ["date", "trigger", "severity", "status", "value", "threshold", "metadata_json"],
        )

    def write_confidence_scores(self, df: pd.DataFrame) -> int:
        return self._write("confidence_scores", df, ["date", "confidence", "grade", "metadata_json"])

    def write_exact_release_calendar(self, df: pd.DataFrame) -> int:
        return self._write(
            "exact_release_calendar",
            df,
            ["series_id", "observation_date", "release_timestamp_utc", "domain", "lag_days", "source", "metadata_json"],
        )

    def write_ensemble_weights(self, df: pd.DataFrame) -> int:
        return self._write(
            "ensemble_weights", df, ["horizon", "target", "model_name", "weight", "method", "metadata_json"]
        )

    def write_stacking_diagnostics(self, df: pd.DataFrame) -> int:
        return self._write(
            "stacking_diagnostics",
            df,
            ["horizon", "target", "observations", "log_loss", "brier", "model_count", "method", "metadata_json"],
        )

    def write_model_drift(self, df: pd.DataFrame) -> int:
        return self._write("model_drift", df, ["date", "feature_name", "psi", "mean_shift", "status", "metadata_json"])

    def write_release_gates(self, df: pd.DataFrame) -> int:
        frame = df.copy()
        if not frame.empty:
            frame["approved"] = frame["approved"].astype(int)
            if "severe_drift" not in frame.columns:
                frame["severe_drift"] = 0
            # v1.5 (PR-1 ASK-7): resolved_profile is optional on the
            # write path so v1.4 callers that have not been refreshed
            # to emit the column continue to work.
            if "resolved_profile" not in frame.columns:
                frame["resolved_profile"] = None
        return self._write(
            "release_gates",
            frame,
            [
                "date",
                "approved",
                "decision",
                "confidence",
                "confidence_grade",
                "severe_drift",
                "major_drift",
                "max_psi",
                "high_invalidation_triggers",
                "active_trigger_names",
                "reasons",
                "metadata_json",
                "resolved_profile",
            ],
        )

    def write_alfred_ingestion_manifest(self, df: pd.DataFrame) -> int:
        return self._write(
            "alfred_ingestion_manifest",
            df,
            [
                "series_id",
                "realtime_start",
                "realtime_end",
                "observation_start",
                "observation_end",
                "rows",
                "status",
                "error",
                "ingested_at_utc",
                "metadata_json",
            ],
        )

    def write_hazard_diagnostics(self, df: pd.DataFrame) -> int:
        frame = df.copy()
        if not frame.empty:
            frame["constant_fallback"] = frame["constant_fallback"].astype(int)
        return self._write(
            "hazard_diagnostics",
            frame,
            [
                "date",
                "model_name",
                "observations",
                "events",
                "feature_count",
                "constant_fallback",
                "latest_monthly_hazard",
                "metadata_json",
            ],
        )

    def write_oos_predictions(self, df: pd.DataFrame) -> int:
        frame = df.copy()
        if frame.empty:
            return 0
        if "regime_bucket" not in frame:
            frame["regime_bucket"] = None
        if "y" not in frame:
            frame["y"] = None
        return self._write(
            "oos_predictions",
            frame,
            ["date", "model_name", "horizon", "target", "value", "y", "regime_bucket", "metadata_json"],
        )

    def write_routed_alerts(self, df: pd.DataFrame) -> int:
        return self._write(
            "routed_alerts", df, ["date", "alert_type", "severity", "channel", "message", "metadata_json"]
        )

    def write_promotion_workflow(self, df: pd.DataFrame) -> int:
        frame = df.copy()
        if not frame.empty:
            for c in ["approved", "promoted_challenger_present", "release_gate_approved", "drift_ok"]:
                frame[c] = frame[c].astype(int)
        return self._write(
            "promotion_workflow",
            frame,
            [
                "date",
                "workflow",
                "approved",
                "decision",
                "confidence",
                "promoted_challenger_present",
                "release_gate_approved",
                "drift_ok",
                "reasons",
                "metadata_json",
            ],
        )

    def write_series_vintages(self, df: pd.DataFrame) -> int:
        return self._write(
            "series_vintages", df, ["series_id", "vintage_date", "source", "ingested_at_utc", "metadata_json"]
        )

    def write_vintage_observations(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        frame = df.copy()
        if "observation_date" not in frame and "date" in frame:
            frame["observation_date"] = frame["date"]
        if "realtime_start" not in frame:
            frame["realtime_start"] = frame.get("vintage_date")
        if "realtime_end" not in frame:
            frame["realtime_end"] = None
        if "ingested_at_utc" not in frame:
            from datetime import datetime

            frame["ingested_at_utc"] = datetime.now(UTC).isoformat()
        if "metadata_json" not in frame:
            frame["metadata_json"] = "{}"
        for c in ["observation_date", "realtime_start", "realtime_end", "vintage_date"]:
            if c in frame:
                frame[c] = pd.to_datetime(frame[c], errors="coerce").dt.strftime("%Y-%m-%d")
        return self._write(
            "vintage_observations",
            frame,
            [
                "series_id",
                "observation_date",
                "value",
                "realtime_start",
                "realtime_end",
                "vintage_date",
                "source",
                "ingested_at_utc",
                "metadata_json",
            ],
        )

    def write_feature_asof_values(self, df: pd.DataFrame) -> int:
        return self._write(
            "feature_asof_values",
            df,
            [
                "as_of_date",
                "feature_name",
                "source_series_id",
                "observation_date",
                "vintage_date",
                "value",
                "transform_name",
                "created_at_utc",
                "metadata_json",
            ],
        )

    def write_vintage_audits(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        from datetime import datetime

        frame = df.copy()
        frame["run_at_utc"] = datetime.now(UTC).isoformat()
        if "metadata_json" not in frame:
            frame["metadata_json"] = "{}"
        return self._write(
            "vintage_audits",
            frame,
            ["audit", "run_at_utc", "rows", "violations", "status", "details", "metadata_json"],
            mode="IGNORE",
        )

    # ----- per-table readers -----

    def read_observations(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM observations ORDER BY date, series_id")

    def read_features(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM features ORDER BY date, feature_name")

    def read_regimes(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM regimes ORDER BY date")

    def read_model_outputs(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM model_outputs ORDER BY date")

    def read_recession_labels(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM recession_labels ORDER BY date")

    def read_historical_analogs(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM historical_analogs ORDER BY as_of_date DESC, rank")

    def read_driver_attribution(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM driver_attribution ORDER BY date DESC, attribution_type, rank")

    def read_model_runs(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM model_runs ORDER BY created_at_utc")

    def read_calibrated_outputs(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM calibrated_outputs ORDER BY date")

    def read_calibration_models(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM calibration_models ORDER BY horizon, target")

    def read_release_calendar_audit(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM release_calendar_audit ORDER BY series_id")

    def read_invalidation_triggers(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM invalidation_triggers ORDER BY date DESC, severity, trigger")

    def read_confidence_scores(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM confidence_scores ORDER BY date")

    def read_exact_release_calendar(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM exact_release_calendar ORDER BY series_id, observation_date")

    def read_ensemble_weights(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM ensemble_weights ORDER BY target, horizon, weight DESC")

    def read_stacking_diagnostics(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM stacking_diagnostics ORDER BY target, horizon")

    def read_model_drift(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM model_drift ORDER BY date DESC, psi DESC")

    def read_release_gates(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM release_gates ORDER BY date")

    def read_alfred_ingestion_manifest(self) -> pd.DataFrame:
        return self._backend.read_sql(
            "SELECT * FROM alfred_ingestion_manifest ORDER BY ingested_at_utc DESC, series_id, realtime_start"
        )

    def read_hazard_diagnostics(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM hazard_diagnostics ORDER BY date")

    def read_oos_predictions(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM oos_predictions ORDER BY date, target, horizon, model_name")

    def read_routed_alerts(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM routed_alerts ORDER BY date DESC, severity, alert_type")

    def read_promotion_workflow(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM promotion_workflow ORDER BY date")

    def read_series_vintages(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM series_vintages ORDER BY series_id, vintage_date")

    def read_vintage_observations(self) -> pd.DataFrame:
        return self._backend.read_sql(
            "SELECT * FROM vintage_observations ORDER BY observation_date, series_id, vintage_date"
        )

    def read_feature_asof_values(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM feature_asof_values ORDER BY as_of_date, feature_name")

    def read_vintage_audits(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM vintage_audits ORDER BY run_at_utc, audit")

    # ----- conformal coverage / e-value / nowcast -----

    def write_conformal_coverage(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        frame = df.copy()
        if "method" not in frame.columns:
            frame["method"] = "mondrian_binary"
        if "threshold" not in frame.columns:
            frame["threshold"] = float("nan")
        if "metadata_json" not in frame.columns:
            frame["metadata_json"] = "{}"
        frame["n"] = frame["n"].astype(int)
        return self._write(
            "conformal_coverage",
            frame,
            [
                "as_of_date",
                "target",
                "horizon",
                "bucket",
                "n",
                "realized_coverage",
                "target_coverage",
                "threshold",
                "method",
                "metadata_json",
            ],
        )

    def read_conformal_coverage(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM conformal_coverage ORDER BY as_of_date, target, horizon, bucket")

    def write_e_value_log(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        frame = df.copy()
        for c in ["champion", "metadata_json", "n"]:
            if c not in frame.columns:
                frame[c] = None if c != "metadata_json" else "{}"
        frame["e_value"] = frame["e_value"].astype(float)
        frame["level"] = frame["level"].astype(float)
        return self._write(
            "e_value_log",
            frame,
            [
                "date",
                "target",
                "horizon",
                "challenger",
                "champion",
                "e_value",
                "level",
                "decision",
                "n",
                "metadata_json",
            ],
        )

    def read_e_value_log(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM e_value_log ORDER BY date, target, horizon, challenger")

    def write_nowcast_factors(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        frame = df.copy()
        for c in ["factor_se", "frequency_mix", "backend", "metadata_json"]:
            if c not in frame.columns:
                frame[c] = None if c != "metadata_json" else "{}"
        frame["factor_value"] = frame["factor_value"].astype(float)
        return self._write(
            "nowcast_factors",
            frame,
            [
                "as_of_date",
                "domain",
                "factor_value",
                "factor_se",
                "frequency_mix",
                "backend",
                "metadata_json",
            ],
        )

    def read_nowcast_factors(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM nowcast_factors ORDER BY as_of_date, domain")

    def write_conditional_coverage_report(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        frame = df.copy()
        for c in ["worst_violation", "metadata_json"]:
            if c not in frame.columns:
                frame[c] = None if c != "metadata_json" else "{}"
        if "method" not in frame.columns:
            frame["method"] = "conditional_conformal"
        frame["coverage"] = frame["coverage"].astype(float)
        frame["alpha"] = frame["alpha"].astype(float)
        frame["n"] = frame["n"].astype(int)
        # Use the shared upsert path; the column quoting helper handles
        # the SQLite/DuckDB ``"group"`` reserved word.
        cols = [
            "as_of_date",
            "target",
            "horizon",
            "group",
            "coverage",
            "n",
            "alpha",
            "method",
            "worst_violation",
            "metadata_json",
        ]
        frame = frame.copy()
        frame["as_of_date"] = pd.to_datetime(frame["as_of_date"]).dt.strftime("%Y-%m-%d")
        self._backend.upsert_frame("conditional_coverage_report", frame, cols, mode="REPLACE")
        self._backend.commit()
        return len(frame)

    def read_conditional_coverage_report(self) -> pd.DataFrame:
        return self._backend.read_sql(
            'SELECT * FROM conditional_coverage_report ORDER BY as_of_date, target, horizon, "group"'
        )

    # ----- v1.3 alert dispatch -----

    def write_alert_dispatches(self, df: pd.DataFrame) -> int:
        """Persist live-sink dispatch outcomes (v1.3 item E)."""
        if df is None or df.empty:
            return 0
        frame = df.copy()
        if "metadata_json" not in frame.columns:
            frame["metadata_json"] = "{}"
        if "detail" not in frame.columns:
            frame["detail"] = ""
        return self._write(
            "alert_dispatches",
            frame,
            [
                "date",
                "alert_type",
                "sink",
                "status",
                "detail",
                "dispatched_at_utc",
                "metadata_json",
            ],
        )

    def read_alert_dispatches(self) -> pd.DataFrame:
        return self._backend.read_sql(
            "SELECT * FROM alert_dispatches ORDER BY dispatched_at_utc DESC, alert_type, sink"
        )

    # ----- v1.4 Bayesian MS-VAR diagnostics (item A) -----

    def write_bayesian_msvar_diagnostics(self, df: pd.DataFrame) -> int:
        """Persist NumPyro NUTS / SVI fit diagnostics (v1.4 item A)."""
        if df is None or df.empty:
            return 0
        frame = df.copy()
        if "metadata_json" not in frame.columns:
            frame["metadata_json"] = "{}"
        for c in ("num_chains", "num_divergences"):
            if c in frame.columns:
                frame[c] = pd.to_numeric(frame[c], errors="coerce").astype("Int64")
        return self._write(
            "bayesian_msvar_diagnostics",
            frame,
            [
                "run_id",
                "method",
                "num_chains",
                "num_divergences",
                "max_rhat",
                "min_ess",
                "runtime_seconds",
                "metadata_json",
            ],
        )

    def read_bayesian_msvar_diagnostics(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM bayesian_msvar_diagnostics ORDER BY run_id, method")

    # ----- v1.4 release calendar refresh outcomes (item D) -----

    def write_release_calendar_refreshes(self, df: pd.DataFrame) -> int:
        """Record one row per ``mre refresh-release-calendars`` run (v1.4 item D)."""
        if df is None or df.empty:
            return 0
        frame = df.copy()
        if "metadata_json" not in frame.columns:
            frame["metadata_json"] = "{}"
        if "error" not in frame.columns:
            frame["error"] = None
        if "source_hash" not in frame.columns:
            frame["source_hash"] = None
        return self._write(
            "release_calendar_refreshes",
            frame,
            [
                "agency",
                "fetched_at_utc",
                "entries_count",
                "status",
                "error",
                "source_hash",
                "metadata_json",
            ],
        )

    def read_release_calendar_refreshes(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM release_calendar_refreshes ORDER BY fetched_at_utc DESC, agency")


# ---------------------------------------------------------------------------
# Migration helper
# ---------------------------------------------------------------------------


_TABLE_NAMES: tuple[str, ...] = tuple(_TABLE_PKS.keys())


def migrate_warehouse(
    src: str | Path,
    dst: str | Path,
    *,
    src_backend: BackendName = "auto",
    dst_backend: BackendName = "auto",
) -> dict[str, int]:
    """Copy every warehouse table from ``src`` to ``dst``.

    Returns a dict mapping table name to row count copied. Used by the
    ``mre warehouse-migrate`` CLI command (v1.3 item D).
    """
    src_wh = Warehouse(src, backend=src_backend)
    dst_wh = Warehouse(dst, backend=dst_backend)
    counts: dict[str, int] = {}
    try:
        for table in _TABLE_NAMES:
            try:
                df = src_wh._backend.read_sql(f"SELECT * FROM {table}")
            except Exception:
                counts[table] = 0
                continue
            if df is None or df.empty:
                counts[table] = 0
                continue
            cols = list(df.columns)
            rows = list(df[cols].itertuples(index=False, name=None))
            dst_wh._backend.upsert(table, rows, cols, mode="REPLACE")
            dst_wh._backend.commit()
            counts[table] = len(rows)
    finally:
        src_wh.close()
        dst_wh.close()
    return counts


__all__ = ["SCHEMA_STATEMENTS", "Warehouse", "migrate_warehouse"]
