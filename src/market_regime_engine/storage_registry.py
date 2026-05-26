# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

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
        # v1.6.0 (REVIEW_DEEP_V1_5_2.md A13 / Finding §3.7): the
        # PK leads with ``series_id`` but ``read_observations``
        # issues ``ORDER BY date, series_id``. Add a secondary
        # index whose leading column matches the read pattern so
        # the planner can serve the sort from an index scan
        # instead of a full table sort.
        index_sql=("CREATE INDEX IF NOT EXISTS idx_observations_date_series ON observations(date, series_id)",),
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
        # v1.6.0 (A13): leading column on the hot read path is
        # ``date`` not ``feature_name``. Add a secondary index
        # to match.
        index_sql=("CREATE INDEX IF NOT EXISTS idx_features_date_name ON features(date, feature_name)",),
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
        # v1.6.0 (A13): the PK leads with ``model_name`` but the
        # read path orders by ``date`` only. Add a secondary index
        # so the planner can index-scan the date column.
        index_sql=("CREATE INDEX IF NOT EXISTS idx_model_outputs_date ON model_outputs(date)",),
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


def legacy_aggregate(name: str) -> Any:
    """Return legacy aggregate values formerly materialized in storage.py."""
    if name == "SCHEMA_STATEMENTS":
        return tuple(spec.create_sql for spec in _REGISTRY)
    if name == "_TABLE_PKS":
        return {spec.name: spec.primary_key for spec in _REGISTRY}
    if name == "_TABLE_NAMES":
        return tuple(spec.name for spec in _REGISTRY)
    raise AttributeError(name)


__all__ = [
    "_REGISTRY",
    "BackendName",
    "TableSpec",
    "_extract_pk",
    "_extract_table_name",
    "_get_table_pk",
    "legacy_aggregate",
    "register_tables",
    "registered_tables",
]
