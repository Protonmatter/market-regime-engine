from __future__ import annotations

import contextlib
import sqlite3
import threading
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path
from typing import Any, Literal, Protocol

import pandas as pd

# v1.3 (item D): the warehouse becomes a thin facade over a backend
# protocol so DuckDB can be used as the primary store without breaking
# SQLite callers. ``backend="sqlite"`` is the default for back-compat;
# ``backend="auto"`` picks DuckDB when ``path.suffix == ".duckdb"`` and
# the optional ``duckdb`` package is importable. Schema parity is
# verified end-to-end by ``tests/test_warehouse_duckdb_parity.py``.
#
# v1.5 (PR-2, ASK-3): the SCHEMA_STATEMENTS tuple has been replaced by
# an explicit :class:`TableSpec` + :func:`register_tables` registry so
# downstream packages (fixed_income, future equities, future FX) can
# extend the schema without mutating a module-level tuple. PR-1's regex
# PK parser (``_extract_pk``) is preserved with a DeprecationWarning so
# any out-of-tree caller continues to work; the in-tree warehouse uses
# the explicit ``TableSpec.primary_key`` field instead. Per-table
# indexes (ASK-11) ride along on ``TableSpec.index_sql``. SQLite/DuckDB
# DDL parity (PR-2 task D) is supported via the optional
# ``TableSpec.sqlite_create_sql`` override.

BackendName = Literal["sqlite", "duckdb", "auto"]


# ---------------------------------------------------------------------------
# Table registry (v1.5 PR-2 ASK-3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TableSpec:
    """Declarative description of one warehouse table.

    ``name`` is the canonical table identifier used by the upsert path
    when constructing ``ON CONFLICT`` clauses on DuckDB. ``create_sql``
    is the DuckDB-flavoured DDL (which doubles as the SQLite DDL for
    every existing v1.4 table because they all use SQLite-compatible
    type aliases). ``primary_key`` is explicit so the v1.4-era regex
    parser (``_extract_pk``) is no longer load-bearing.

    Optional fields:

    - ``index_sql``: additional ``CREATE INDEX IF NOT EXISTS`` statements
      run after ``create_sql`` in registration order. Both SQLite and
      DuckDB accept the same ``CREATE INDEX IF NOT EXISTS`` grammar so a
      single string works on both backends in practice.
    - ``sqlite_create_sql``: optional SQLite-only DDL substituted for
      ``create_sql`` when the backend is SQLite (used by the FI tables
      to keep DuckDB-native ``TIMESTAMP`` / ``DECIMAL(18,6)`` / ``JSON``
      types while emitting ``TEXT`` / ``REAL`` / ``TEXT`` on SQLite).
    """

    name: str
    create_sql: str
    primary_key: tuple[str, ...]
    index_sql: tuple[str, ...] = ()
    sqlite_create_sql: str | None = None


_REGISTRY: list[TableSpec] = []


def register_tables(specs: Sequence[TableSpec]) -> None:
    """Register tables for warehouse initialization.

    Idempotent on the ``name`` column: re-registering the same
    :class:`TableSpec` is a no-op; registering a *different* spec under
    a name that already exists raises :class:`ValueError`. This protects
    against silent schema drift when two packages try to claim the same
    table.
    """

    by_name = {spec.name: spec for spec in _REGISTRY}
    for spec in specs:
        existing = by_name.get(spec.name)
        if existing is None:
            _REGISTRY.append(spec)
            by_name[spec.name] = spec
            continue
        if existing == spec:
            continue
        raise ValueError(
            "Table "
            f"{spec.name!r} already registered with conflicting content. "
            f"existing.primary_key={existing.primary_key!r} new.primary_key={spec.primary_key!r}; "
            "register_tables is idempotent only when the TableSpec is byte-for-byte identical."
        )


def registered_tables() -> tuple[TableSpec, ...]:
    """Read-only snapshot of the current registry, in insertion order."""

    return tuple(_REGISTRY)


def _get_table_pk(table: str) -> tuple[str, ...]:
    """Internal helper: look up the primary-key tuple for ``table``.

    Returns the empty tuple when the table is not in the registry; the
    DuckDB upsert path interprets an empty PK as "no ON CONFLICT clause"
    which is the historically correct behaviour for ad-hoc tables.
    """

    for spec in _REGISTRY:
        if spec.name == table:
            return spec.primary_key
    return ()


def _register_core_tables() -> None:
    """Register the 34 core (macro / regime / governance) tables.

    Called once on module import. Tables are listed in the same order
    they appeared in the pre-PR-2 ``SCHEMA_STATEMENTS`` tuple so the
    init-schema sequence is byte-for-byte unchanged.
    """

    register_tables(_CORE_TABLES)


_CORE_TABLES: tuple[TableSpec, ...] = (
    TableSpec(
        name="observations",
        create_sql="""
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
        primary_key=("series_id", "date", "vintage_date"),
    ),
    TableSpec(
        name="features",
        create_sql="""
    CREATE TABLE IF NOT EXISTS features (
        feature_name TEXT NOT NULL,
        date TEXT NOT NULL,
        value REAL NOT NULL,
        domain TEXT,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(feature_name, date)
    )
    """,
        primary_key=("feature_name", "date"),
    ),
    TableSpec(
        name="regimes",
        create_sql="""
    CREATE TABLE IF NOT EXISTS regimes (
        date TEXT PRIMARY KEY,
        regime TEXT NOT NULL,
        decoded_regime TEXT NOT NULL,
        score REAL NOT NULL,
        change_point_prob REAL,
        metadata_json TEXT DEFAULT '{}'
    )
    """,
        primary_key=("date",),
    ),
    TableSpec(
        name="model_outputs",
        create_sql="""
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
        primary_key=("model_name", "date", "horizon", "target"),
    ),
    TableSpec(
        name="recession_labels",
        create_sql="""
    CREATE TABLE IF NOT EXISTS recession_labels (
        date TEXT PRIMARY KEY,
        recession REAL NOT NULL,
        source TEXT NOT NULL,
        metadata_json TEXT DEFAULT '{}'
    )
    """,
        primary_key=("date",),
    ),
    TableSpec(
        name="historical_analogs",
        create_sql="""
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
        primary_key=("as_of_date", "analog_date"),
    ),
    TableSpec(
        name="driver_attribution",
        create_sql="""
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
        primary_key=("date", "attribution_type", "rank", "name"),
    ),
    TableSpec(
        name="model_runs",
        create_sql="""
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
        primary_key=("run_id",),
    ),
    TableSpec(
        name="calibrated_outputs",
        create_sql="""
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
        primary_key=("model_name", "date", "horizon", "target"),
    ),
    TableSpec(
        name="calibration_models",
        create_sql="""
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
        primary_key=("horizon", "target", "method"),
    ),
    TableSpec(
        name="release_calendar_audit",
        create_sql="""
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
        primary_key=("series_id",),
    ),
    TableSpec(
        name="invalidation_triggers",
        create_sql="""
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
        primary_key=("date", "trigger"),
    ),
    TableSpec(
        name="confidence_scores",
        create_sql="""
    CREATE TABLE IF NOT EXISTS confidence_scores (
        date TEXT PRIMARY KEY,
        confidence REAL NOT NULL,
        grade TEXT NOT NULL,
        metadata_json TEXT DEFAULT '{}'
    )
    """,
        primary_key=("date",),
    ),
    TableSpec(
        name="exact_release_calendar",
        create_sql="""
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
        primary_key=("series_id", "observation_date"),
    ),
    TableSpec(
        name="ensemble_weights",
        create_sql="""
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
        primary_key=("horizon", "target", "model_name", "method"),
    ),
    TableSpec(
        name="stacking_diagnostics",
        create_sql="""
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
        primary_key=("horizon", "target", "method"),
    ),
    TableSpec(
        name="model_drift",
        create_sql="""
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
        primary_key=("date", "feature_name"),
    ),
    # v1.5 PR-1 ASK-7 / P2: nullable resolved_profile column on release_gates.
    # Pre-PR-2 the trailing comment was placed *outside* the CREATE TABLE
    # statement because the regex PK extractor would mis-detect a
    # parenthesised comment as the PRIMARY KEY column list (FLAG F-4 in
    # REVIEW.md). PR-2 retires the regex parse (PK is explicit on
    # TableSpec) so the comment can now live wherever it reads naturally.
    TableSpec(
        name="release_gates",
        create_sql="""
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
        primary_key=("date",),
    ),
    TableSpec(
        name="alfred_ingestion_manifest",
        create_sql="""
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
        primary_key=("series_id", "realtime_start", "realtime_end"),
    ),
    TableSpec(
        name="hazard_diagnostics",
        create_sql="""
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
        primary_key=("date", "model_name"),
    ),
    TableSpec(
        name="oos_predictions",
        create_sql="""
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
        primary_key=("date", "model_name", "horizon", "target"),
    ),
    TableSpec(
        name="routed_alerts",
        create_sql="""
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
        primary_key=("date", "alert_type", "channel"),
    ),
    TableSpec(
        name="promotion_workflow",
        create_sql="""
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
        primary_key=("date", "workflow"),
    ),
    TableSpec(
        name="series_vintages",
        create_sql="""
    CREATE TABLE IF NOT EXISTS series_vintages (
        series_id TEXT NOT NULL,
        vintage_date TEXT NOT NULL,
        source TEXT NOT NULL,
        ingested_at_utc TEXT NOT NULL,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(series_id, vintage_date)
    )
    """,
        primary_key=("series_id", "vintage_date"),
    ),
    TableSpec(
        name="vintage_observations",
        create_sql="""
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
        primary_key=("series_id", "observation_date", "vintage_date"),
    ),
    TableSpec(
        name="feature_asof_values",
        create_sql="""
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
        primary_key=("as_of_date", "feature_name"),
    ),
    TableSpec(
        name="vintage_audits",
        create_sql="""
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
        primary_key=("audit", "run_at_utc"),
    ),
    TableSpec(
        name="conformal_coverage",
        create_sql="""
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
        primary_key=("as_of_date", "target", "horizon", "bucket", "method"),
    ),
    # ----- v1.2 frontier tables -----
    TableSpec(
        name="e_value_log",
        create_sql="""
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
        primary_key=("date", "target", "horizon", "challenger"),
    ),
    TableSpec(
        name="nowcast_factors",
        create_sql="""
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
        primary_key=("as_of_date", "domain"),
    ),
    TableSpec(
        name="conditional_coverage_report",
        create_sql="""
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
        primary_key=("as_of_date", "target", "horizon", "group", "method"),
    ),
    # ----- v1.3 alert dispatch table -----
    TableSpec(
        name="alert_dispatches",
        create_sql="""
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
        primary_key=("date", "alert_type", "sink", "dispatched_at_utc"),
    ),
    # ----- v1.4 Bayesian MS-VAR diagnostics (item A) -----
    TableSpec(
        name="bayesian_msvar_diagnostics",
        create_sql="""
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
        primary_key=("run_id", "method"),
    ),
    # ----- v1.4 release calendar refresh outcomes (item D) -----
    TableSpec(
        name="release_calendar_refreshes",
        create_sql="""
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
        primary_key=("agency", "fetched_at_utc"),
    ),
)


# Register the core tables once at import time. Downstream packages
# (fixed_income, future product lines) call register_tables(...) in
# their own __init__.py to extend the registry.
_register_core_tables()


# legacy: TODO remove after PR-3+ migrations.
# Pre-PR-2 the warehouse extracted the PRIMARY KEY tuple from each
# CREATE TABLE statement via a naive parse. PR-2 makes the PK an
# explicit field on :class:`TableSpec`, so any caller passing a raw
# SQL string here is on a deprecated path. Kept for back-compat with
# out-of-tree extensions that have not yet ported to the registry.
def _extract_pk(schema_sql: str) -> tuple[str, ...]:
    """Parse the PRIMARY KEY tuple from a raw CREATE TABLE statement.

    .. deprecated:: 1.5 (PR-2)
        Use :attr:`TableSpec.primary_key` instead. This regex-style
        parser is retained only for legacy callers that still feed raw
        SQL strings to ``migrate_warehouse`` or similar.
    """

    warnings.warn(
        "storage._extract_pk is deprecated; use TableSpec.primary_key. "
        "This shim will be removed after the PR-3+ migration cycle.",
        DeprecationWarning,
        stacklevel=2,
    )
    text = schema_sql.upper()
    idx = text.find("PRIMARY KEY")
    if idx < 0:
        return ()
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


# legacy: TODO remove after PR-3+ migrations.
def _extract_table_name(schema_sql: str) -> str:
    """Parse the table name from a raw CREATE TABLE statement.

    .. deprecated:: 1.5 (PR-2)
        Use :attr:`TableSpec.name` instead.
    """

    warnings.warn(
        "storage._extract_table_name is deprecated; use TableSpec.name. "
        "This shim will be removed after the PR-3+ migration cycle.",
        DeprecationWarning,
        stacklevel=2,
    )
    text = schema_sql.upper()
    idx = text.find("CREATE TABLE")
    if idx < 0:
        return ""
    rest = schema_sql[idx + len("CREATE TABLE") :].strip()
    if rest.upper().startswith("IF NOT EXISTS"):
        rest = rest[len("IF NOT EXISTS") :].strip()
    name_end = rest.find("(")
    if name_end < 0:
        return ""
    return rest[:name_end].strip()


def __getattr__(name: str) -> Any:
    """PEP 562 module-level attribute lookup for legacy aggregates.

    Pre-PR-2 callers reference ``SCHEMA_STATEMENTS`` / ``_TABLE_PKS`` /
    ``_TABLE_NAMES`` directly. Those names continue to resolve, but now
    they are dynamically derived from :func:`registered_tables` so
    downstream registrations (e.g. ``fixed_income.schema.register``)
    are reflected without re-importing this module.
    """

    if name == "SCHEMA_STATEMENTS":
        return tuple(spec.create_sql for spec in _REGISTRY)
    if name == "_TABLE_PKS":
        return {spec.name: spec.primary_key for spec in _REGISTRY}
    if name == "_TABLE_NAMES":
        return tuple(spec.name for spec in _REGISTRY)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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

    def bulk_load_chunked(
        self,
        table: str,
        df: pd.DataFrame,
        chunk_rows: int = 1_000_000,
        *,
        cols: Sequence[str] | None = None,
        mode: str = "REPLACE",
    ) -> int:
        """Chunked bulk insert via the per-backend bulk-load path.

        v1.5 (PR-2 PR-14): a 100M-row TRACE import OOMs the worker when
        the entire frame is registered to DuckDB at once. This helper
        feeds the underlying ``upsert_frame`` one slice at a time so at
        most ``chunk_rows`` of ``df`` sit in DuckDB's internal staging
        buffer simultaneously. Each chunk runs inside its own
        ``BEGIN/COMMIT`` (provided by ``upsert_frame``) so a crash
        mid-chunk leaves the table consistent up to the last committed
        chunk.

        DuckDB callers keep the v1.4 ``register`` + ``INSERT ... SELECT
        ... ON CONFLICT`` fast path (no regression on the 6600× speedup
        the v1.4 transcript locked in). SQLite callers fall back to the
        same ``executemany`` path they already use, just chunked.

        Returns the total row count successfully committed across all
        chunks. ``cols`` defaults to ``df.columns``; pass an explicit
        list when the destination table expects a subset.
        """

        if df is None or df.empty:
            return 0
        if chunk_rows <= 0:
            raise ValueError(f"chunk_rows must be positive; got {chunk_rows!r}")

        column_list = list(cols) if cols is not None else list(df.columns)
        total = 0
        n = len(df)
        for start in range(0, n, chunk_rows):
            stop = min(start + chunk_rows, n)
            chunk = df.iloc[start:stop]
            # ``upsert_frame`` wraps each call in its own BEGIN/COMMIT;
            # committing per chunk is exactly the partial-failure
            # semantics PR-14 asks for.
            self._backend.upsert_frame(table, chunk, column_list, mode=mode)
            self._backend.commit()
            total += len(chunk)
        return total

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

    # ----- v1.5 PR-2: Fixed-Income write helpers -----
    #
    # These thin shims accept a DataFrame, fill in any missing optional
    # columns with safe defaults, and route through the standard
    # ``_backend.upsert_frame`` path so the v1.4 DuckDB bulk-load fast
    # path is preserved. Per the AGENT.md non-negotiable constraint 7,
    # the governance writers (credit_regime_scores,
    # liquidity_stress_scores, execution_confidence_predictions,
    # fixed_income_evidence_packs) all require model_run_id / release_gate
    # / artifact_hash on the input frame.

    def _write_fi(
        self,
        table: str,
        df: pd.DataFrame,
        cols: list[str],
        mode: str = "REPLACE",
    ) -> int:
        """Shared FI write path: defensive copy + bulk insert.

        Distinct from the macro-side ``_write`` because the FI tables
        do not need the ``YYYY-MM-DD`` ISO date coercion the v1.0 macro
        path performs on ``date`` / ``vintage_date`` / ``as_of_date``:
        FI timestamps are full ISO-8601 and survive on both backends
        without normalisation.
        """

        if df is None or df.empty:
            return 0
        frame = df.copy()
        if "metadata_json" in cols and "metadata_json" not in frame:
            frame["metadata_json"] = "{}"
        self._backend.upsert_frame(table, frame, cols, mode=mode)
        self._backend.commit()
        return len(frame)

    def write_trace_trades(self, df: pd.DataFrame) -> int:
        return self._write_fi(
            "trace_trades",
            df,
            [
                "trade_id",
                "timestamp",
                "cusip",
                "price",
                "yield_pct",
                "size",
                "side",
                "protocol",
                "venue",
                "source",
                "reported_at",
                "metadata_json",
            ],
        )

    def read_trace_trades(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM trace_trades ORDER BY timestamp, trade_id")

    def write_rfq_events(self, df: pd.DataFrame) -> int:
        return self._write_fi(
            "rfq_events",
            df,
            [
                "rfq_id",
                "timestamp",
                "cusip",
                "side",
                "notional",
                "protocol",
                "status",
                "dealers_requested",
                "dealers_responded",
                "time_to_first_response_ms",
                "client_id",
                "metadata_json",
            ],
        )

    def read_rfq_events(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM rfq_events ORDER BY timestamp, rfq_id")

    def write_dealer_quotes(self, df: pd.DataFrame) -> int:
        return self._write_fi(
            "dealer_quotes",
            df,
            ["timestamp", "cusip", "dealer_id", "side", "price", "size", "expires_at", "metadata_json"],
        )

    def read_dealer_quotes(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM dealer_quotes ORDER BY timestamp, cusip, dealer_id")

    def write_dealer_response_stats(self, df: pd.DataFrame) -> int:
        return self._write_fi(
            "dealer_response_stats",
            df,
            ["dealer_id", "window_start", "window_end", "requests", "responses", "avg_response_ms", "metadata_json"],
        )

    def read_dealer_response_stats(self) -> pd.DataFrame:
        return self._backend.read_sql(
            "SELECT * FROM dealer_response_stats ORDER BY dealer_id, window_start"
        )

    def write_curve_snapshots(self, df: pd.DataFrame) -> int:
        return self._write_fi(
            "curve_snapshots",
            df,
            ["timestamp", "curve_type", "tenor", "rate", "source", "metadata_json"],
        )

    def read_curve_snapshots(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM curve_snapshots ORDER BY timestamp, curve_type, tenor")

    def write_cds_curve_snapshots(self, df: pd.DataFrame) -> int:
        return self._write_fi(
            "cds_curve_snapshots",
            df,
            ["timestamp", "reference_entity", "tenor", "spread_bps", "source", "metadata_json"],
        )

    def read_cds_curve_snapshots(self) -> pd.DataFrame:
        return self._backend.read_sql(
            "SELECT * FROM cds_curve_snapshots ORDER BY timestamp, reference_entity, tenor"
        )

    def write_credit_regime_score(self, df: pd.DataFrame) -> int:
        return self._write_fi(
            "credit_regime_scores",
            df,
            [
                "model_run_id",
                "timestamp",
                "regime_score",
                "regime_label",
                "confidence",
                "drivers_json",
                "component_scores_json",
                "release_gate",
                "artifact_hash",
                "metadata_json",
            ],
        )

    def read_credit_regime_scores(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM credit_regime_scores ORDER BY timestamp, model_run_id")

    def write_liquidity_stress_score(self, df: pd.DataFrame) -> int:
        return self._write_fi(
            "liquidity_stress_scores",
            df,
            [
                "model_run_id",
                "scope_type",
                "scope_id",
                "timestamp",
                "liquidity_score",
                "liquidity_label",
                "confidence",
                "drivers_json",
                "release_gate",
                "artifact_hash",
                "metadata_json",
            ],
        )

    def read_liquidity_stress_scores(self) -> pd.DataFrame:
        return self._backend.read_sql(
            "SELECT * FROM liquidity_stress_scores ORDER BY timestamp, scope_type, scope_id"
        )

    def write_execution_confidence_prediction(self, df: pd.DataFrame) -> int:
        return self._write_fi(
            "execution_confidence_predictions",
            df,
            [
                "request_id",
                "timestamp",
                "model_run_id",
                "cusip",
                "side",
                "notional",
                "protocol",
                "confidence_score",
                "expected_slippage_bps",
                "confidence_interval_low",
                "confidence_interval_high",
                "recommended_action",
                "human_review_required",
                "release_gate",
                "artifact_hash",
                "metadata_json",
            ],
        )

    def read_execution_confidence_predictions(self) -> pd.DataFrame:
        return self._backend.read_sql(
            "SELECT * FROM execution_confidence_predictions ORDER BY timestamp, request_id"
        )

    def write_execution_outcome(self, df: pd.DataFrame) -> int:
        """Persist execution outcomes; enforces ``observed_at > decision_timestamp``.

        Q-2 (REVIEW.md §3.4): the inequality is enforced here in the
        writer rather than as a DB CHECK constraint because DuckDB +
        SQLite differ on CHECK semantics. Any row that violates the
        constraint raises ``ValueError`` before the bulk insert runs.
        """

        if df is None or df.empty:
            return 0
        observed_at = pd.to_datetime(df["observed_at"], utc=True, errors="coerce")
        decision_ts = pd.to_datetime(df["decision_timestamp"], utc=True, errors="coerce")
        bad = (observed_at.isna() | decision_ts.isna()) | (observed_at <= decision_ts)
        if bad.any():
            offenders = df.loc[bad, "request_id"].tolist()
            raise ValueError(
                "execution_outcomes requires observed_at > decision_timestamp; "
                f"offending request_ids: {offenders!r}"
            )
        return self._write_fi(
            "execution_outcomes",
            df,
            [
                "request_id",
                "cusip",
                "side",
                "notional",
                "filled_quantity",
                "execution_price",
                "observed_at",
                "outcome_observation_lag",
                "decision_timestamp",
                "metadata_json",
            ],
        )

    def read_execution_outcomes(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM execution_outcomes ORDER BY request_id")

    def write_tca_regime_segment(self, df: pd.DataFrame) -> int:
        return self._write_fi(
            "tca_regime_segments",
            df,
            [
                "model_run_id",
                "timestamp",
                "regime_label",
                "liquidity_label",
                "execution_confidence_bucket",
                "protocol",
                "side",
                "sector",
                "rating",
                "maturity_bucket",
                "notional_bucket",
                "metric_name",
                "metric_value",
                "sample_count",
                "metadata_json",
            ],
        )

    def read_tca_regime_segments(self) -> pd.DataFrame:
        return self._backend.read_sql(
            "SELECT * FROM tca_regime_segments ORDER BY timestamp, model_run_id, metric_name"
        )

    def write_evidence_pack(self, df: pd.DataFrame) -> int:
        return self._write_fi(
            "fixed_income_evidence_packs",
            df,
            [
                "model_run_id",
                "request_id",
                "component_name",
                "model_version",
                "timestamp",
                "code_sha",
                "model_hash",
                "input_features_hash",
                "output_hash",
                "data_vintages_json",
                "validation_results_json",
                "release_gate",
                "random_seeds_json",
                "python_version",
                "lockfile_hash",
                "hmac_signature",
                "metadata_json",
            ],
        )

    def read_evidence_packs(self) -> pd.DataFrame:
        return self._backend.read_sql(
            "SELECT * FROM fixed_income_evidence_packs ORDER BY timestamp, model_run_id, request_id"
        )

    def write_bond_reference(self, df: pd.DataFrame) -> int:
        return self._write_fi(
            "bond_reference",
            df,
            [
                "cusip",
                "valid_from",
                "valid_to",
                "ticker",
                "issuer",
                "sector",
                "rating",
                "issue_date",
                "maturity",
                "coupon",
                "currency",
                "country",
                "duration",
                "convexity",
                "amount_outstanding",
                "is_callable",
                "call_schedule_json",
                "default_date",
                "delisted_date",
                "metadata_json",
            ],
        )


# ---------------------------------------------------------------------------
# bond_reference temporal versioning helpers (PR-2 task C / Q-4)
# ---------------------------------------------------------------------------


def read_bond_reference_asof(
    warehouse: Warehouse,
    asof: pd.Timestamp | str,
    *,
    include_survivorship_failures: bool = False,
) -> pd.DataFrame:
    """Return the ``bond_reference`` snapshot effective at ``asof``.

    A row is "effective at ``asof``" when ``valid_from <= asof`` and
    (``valid_to`` is ``NULL`` OR ``valid_to > asof``). The function
    additionally filters out CUSIPs that have a ``default_date`` or
    ``delisted_date`` set (the survivorship rule from REVIEW.md §3.4
    Q-4) unless the caller passes ``include_survivorship_failures=True``,
    which is the audit-mode path used by ingestion validators.

    Returns an empty frame when ``bond_reference`` is empty or missing.
    The bool ``is_active`` is computed at read time and never stored,
    so the warehouse cannot drift out of sync with the survivorship
    rule.
    """

    asof_ts = pd.Timestamp(asof)
    # ISO-8601 with explicit microseconds keeps both backends happy:
    # DuckDB parses it as TIMESTAMP, SQLite compares it as TEXT against
    # the same ISO-8601 strings written by the writer.
    asof_str = asof_ts.isoformat()
    sql_filter = (
        "valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)"
    )
    params: tuple[Any, ...] = (asof_str, asof_str)
    if not include_survivorship_failures:
        sql_filter += " AND default_date IS NULL AND delisted_date IS NULL"

    backend = warehouse._backend
    try:
        df = _read_with_params(backend, f"SELECT * FROM bond_reference WHERE {sql_filter}", params)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns if df is not None else [])
    df = df.copy()
    df["is_active"] = True
    return df


def read_bond_reference_history(
    warehouse: Warehouse,
    cusip: str,
) -> pd.DataFrame:
    """Return every ``bond_reference`` row for ``cusip`` in temporal order.

    Rows are ordered by ``valid_from`` ascending; the most recent
    snapshot is the last row. Returns an empty frame when ``cusip`` has
    never been written.
    """

    backend = warehouse._backend
    try:
        df = _read_with_params(
            backend,
            "SELECT * FROM bond_reference WHERE cusip = ? ORDER BY valid_from ASC",
            (cusip,),
        )
    except Exception:
        return pd.DataFrame()
    if df is None:
        return pd.DataFrame()
    return df


def _read_with_params(backend: _Backend, sql: str, params: tuple[Any, ...]) -> pd.DataFrame:
    """Backend-agnostic parameterised SELECT helper.

    The :class:`_Backend` protocol only exposes ``read_sql(sql)`` so we
    reach into the underlying connection to bind parameters. Both
    SQLite's ``pd.read_sql_query`` and DuckDB's ``execute(sql, params)
    .fetchdf()`` accept positional parameters via ``?`` placeholders.
    """

    conn = getattr(backend, "conn", None)
    if conn is None:
        return backend.read_sql(sql)
    # SQLite3 connection has ``execute`` returning a cursor; DuckDB
    # connection has ``execute`` returning a result object with
    # ``fetchdf``. We try the DuckDB-shaped path first because that is
    # the new default backend.
    try:
        result = conn.execute(sql, params)
        if hasattr(result, "fetchdf"):
            return result.fetchdf()
    except TypeError:
        # SQLite execute does not accept tuple params via positional
        # placeholders on read_sql — fall through to pandas helper.
        pass
    return pd.read_sql_query(sql, conn, params=params)


# ---------------------------------------------------------------------------
# Migration helper
# ---------------------------------------------------------------------------


def migrate_warehouse(
    src: str | Path,
    dst: str | Path,
    *,
    src_backend: BackendName = "auto",
    dst_backend: BackendName = "auto",
    tables: Sequence[str] | None = None,
) -> dict[str, int]:
    """Copy every warehouse table from ``src`` to ``dst``.

    Returns a dict mapping table name to row count copied. Used by the
    ``mre warehouse-migrate`` CLI command (v1.3 item D).

    v1.5 (PR-2 AF-12): every table name is validated against
    :func:`registered_tables` before it is interpolated into the
    ``SELECT * FROM {table}`` statement. Pre-PR-2 the same f-string was
    a documented internal-only SQL-injection foot-gun; the guard makes
    the assumption explicit and refuses any unregistered name.
    """

    allowed = {spec.name for spec in registered_tables()}
    if tables is None:
        target_tables: tuple[str, ...] = tuple(allowed)
    else:
        target_tables = tuple(tables)
        unknown = [t for t in target_tables if t not in allowed]
        if unknown:
            raise ValueError(
                f"migrate_warehouse refusing unregistered tables: {unknown!r}. "
                "Register the table via storage.register_tables(...) before invoking migrate."
            )

    src_wh = Warehouse(src, backend=src_backend)
    dst_wh = Warehouse(dst, backend=dst_backend)
    counts: dict[str, int] = {}
    try:
        for table in target_tables:
            # Defensive re-check: even when ``tables is None`` the name
            # comes from the registry, but assert anyway so future
            # callers cannot bypass the guard by mutating the iterable
            # mid-loop.
            if table not in allowed:
                raise ValueError(
                    f"migrate_warehouse refusing unregistered table {table!r}. "
                    "Register the table via storage.register_tables(...) before invoking migrate."
                )
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


# ---------------------------------------------------------------------------
# v1.5 PR-5 (ASK-8): Per-process Warehouse singleton
# ---------------------------------------------------------------------------

# Pre-PR-5 every FastAPI request opened a fresh ``Warehouse(path)`` and closed
# it in a ``try/finally`` (see ``api_v1._read`` / ``fixed_income.api``). On a
# busy FI deployment that meant DuckDB tore down and rebuilt its catalog +
# WAL file ~100 times/second, dominating wall-clock latency. PR-5 caches the
# Warehouse per-process keyed by resolved path; reads run concurrently via
# DuckDB's MVCC, and writes serialise via a per-path :class:`threading.RLock`
# exposed through :func:`pooled_warehouse_write_lock` (DuckDB's Python
# connection is **not** thread-safe for concurrent ``execute`` calls; the
# pool provides a recommended write-serialization helper so the caller does
# not need to roll their own).

_POOLED_WAREHOUSES: dict[str, "Warehouse"] = {}
_POOLED_LOCKS: dict[str, threading.RLock] = {}
_POOL_LOCK = threading.RLock()


def _resolve_pool_key(path: str | Path) -> str:
    return str(Path(path).resolve())


def get_pooled_warehouse(path: str | Path) -> "Warehouse":
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
def pooled_warehouse_write_lock(path: str | Path):  # type: ignore[no-untyped-def]
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
        raise RuntimeError(
            f"close_pooled_warehouses encountered {len(errors)} errors: {errors!r}"
        )


def pooled_warehouse_paths() -> tuple[str, ...]:
    """Inspect the current pool — used by tests for the singleton rail."""
    with _POOL_LOCK:
        return tuple(_POOLED_WAREHOUSES)


__all__ = [
    "TableSpec",
    "Warehouse",
    "close_pooled_warehouses",
    "get_pooled_warehouse",
    "migrate_warehouse",
    "pooled_warehouse_paths",
    "pooled_warehouse_write_lock",
    "read_bond_reference_asof",
    "read_bond_reference_history",
    "register_tables",
    "registered_tables",
]
# Note: ``SCHEMA_STATEMENTS``, ``_TABLE_PKS``, and ``_TABLE_NAMES`` are
# resolved dynamically via PEP 562 module __getattr__ (defined above)
# so they continue to import from this module by name (e.g.
# ``from market_regime_engine.storage import SCHEMA_STATEMENTS``) even
# though they are not listed in ``__all__``. Listing them here would
# trip Ruff's F822 because there is no top-level binding.
