# SPDX-License-Identifier: Apache-2.0
"""Fixed-Income warehouse schema declarations.

v1.5 (PR-2 task B): the FI adapter contributes 13 tables to the
warehouse — bond reference, market data (TRACE / RFQ / dealer quotes /
curves / CDS), governance outputs (credit regime / liquidity stress /
execution confidence / execution outcomes / TCA segments / evidence
packs). Each is declared as a :class:`market_regime_engine.storage.TableSpec`
in :data:`_FI_TABLES` below, with:

- A DuckDB-flavoured ``create_sql`` using native ``TIMESTAMP`` /
  ``DECIMAL(18,6)`` for monetary fields and ``JSON`` for ``*_json``
  columns where DuckDB can index the payload natively.
- A ``sqlite_create_sql`` override that swaps the DuckDB-only types
  for the SQLite-compatible ``TEXT`` / ``REAL`` / ``TEXT`` (ISO-8601
  for timestamps, JSON-as-text for json columns).
- An explicit ``primary_key`` tuple.
- Per-table ``index_sql`` for the hot reads called out in
  REVIEW.md §3.2 ASK-11 / plan §2.

Two governance tables — :data:`execution_confidence_predictions` and
:data:`fixed_income_evidence_packs` — carry a mandatory ``request_id``
column on the composite primary key (PR-15) to prevent the cross-worker
race where two API workers write predictions or packs for the same
client request under different ``model_run_id`` values.

:func:`register` is called from :mod:`market_regime_engine.fixed_income`
``__init__.py`` so importing the FI package is enough to add the FI
tables to the warehouse on the next ``Warehouse(...)`` instantiation.

:data:`FI_TABLE_NAMES` is exported as the canonical list of FI table
names for tests, telemetry, and migration tooling.
"""

from __future__ import annotations

from market_regime_engine.storage import TableSpec, register_tables

# ---------------------------------------------------------------------------
# Table specs
# ---------------------------------------------------------------------------

# ----- bond_reference (with temporal versioning per PR-2 task C, Q-4) -----
# valid_from / valid_to model the as-of snapshot the FI feature builders must
# consume so they do not silently drop survivorship-failed bonds. The
# default_date and delisted_date columns are nullable; ``is_active`` is a
# **read-time** boolean computed by storage.read_bond_reference_asof and is
# deliberately NOT stored, so the warehouse cannot drift out of sync with the
# survivorship rule per Q-4 in REVIEW.md §3.4.
_BOND_REFERENCE_DUCKDB = """
    CREATE TABLE IF NOT EXISTS bond_reference (
        cusip TEXT NOT NULL,
        valid_from TIMESTAMP NOT NULL,
        valid_to TIMESTAMP NULL,
        ticker TEXT,
        issuer TEXT,
        sector TEXT,
        rating TEXT,
        issue_date TIMESTAMP,
        maturity TIMESTAMP,
        coupon DECIMAL(18,6),
        currency TEXT,
        country TEXT,
        duration DECIMAL(18,6),
        convexity DECIMAL(18,6),
        amount_outstanding DECIMAL(18,2),
        is_callable INTEGER,
        call_schedule_json JSON,
        default_date TIMESTAMP NULL,
        delisted_date TIMESTAMP NULL,
        metadata_json JSON,
        PRIMARY KEY(cusip, valid_from)
    )
    """

_BOND_REFERENCE_SQLITE = """
    CREATE TABLE IF NOT EXISTS bond_reference (
        cusip TEXT NOT NULL,
        valid_from TEXT NOT NULL,
        valid_to TEXT,
        ticker TEXT,
        issuer TEXT,
        sector TEXT,
        rating TEXT,
        issue_date TEXT,
        maturity TEXT,
        coupon REAL,
        currency TEXT,
        country TEXT,
        duration REAL,
        convexity REAL,
        amount_outstanding REAL,
        is_callable INTEGER,
        call_schedule_json TEXT,
        default_date TEXT,
        delisted_date TEXT,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(cusip, valid_from)
    )
    """


# ----- trace_trades -----
_TRACE_TRADES_DUCKDB = """
    CREATE TABLE IF NOT EXISTS trace_trades (
        trade_id TEXT NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        cusip TEXT NOT NULL,
        price DECIMAL(18,6) NOT NULL,
        yield_pct DECIMAL(18,6),
        size DECIMAL(18,2) NOT NULL,
        side TEXT,
        protocol TEXT,
        venue TEXT,
        source TEXT,
        reported_at TIMESTAMP,
        metadata_json JSON,
        PRIMARY KEY(trade_id, cusip, timestamp)
    )
    """

_TRACE_TRADES_SQLITE = """
    CREATE TABLE IF NOT EXISTS trace_trades (
        trade_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        cusip TEXT NOT NULL,
        price REAL NOT NULL,
        yield_pct REAL,
        size REAL NOT NULL,
        side TEXT,
        protocol TEXT,
        venue TEXT,
        source TEXT,
        reported_at TEXT,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(trade_id, cusip, timestamp)
    )
    """


# ----- rfq_events -----
_RFQ_EVENTS_DUCKDB = """
    CREATE TABLE IF NOT EXISTS rfq_events (
        rfq_id TEXT NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        cusip TEXT NOT NULL,
        side TEXT,
        notional DECIMAL(18,2),
        protocol TEXT,
        status TEXT,
        dealers_requested INTEGER,
        dealers_responded INTEGER,
        time_to_first_response_ms INTEGER,
        client_id TEXT,
        metadata_json JSON,
        PRIMARY KEY(rfq_id, timestamp)
    )
    """

_RFQ_EVENTS_SQLITE = """
    CREATE TABLE IF NOT EXISTS rfq_events (
        rfq_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        cusip TEXT NOT NULL,
        side TEXT,
        notional REAL,
        protocol TEXT,
        status TEXT,
        dealers_requested INTEGER,
        dealers_responded INTEGER,
        time_to_first_response_ms INTEGER,
        client_id TEXT,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(rfq_id, timestamp)
    )
    """


# ----- dealer_quotes -----
_DEALER_QUOTES_DUCKDB = """
    CREATE TABLE IF NOT EXISTS dealer_quotes (
        timestamp TIMESTAMP NOT NULL,
        cusip TEXT NOT NULL,
        dealer_id TEXT NOT NULL,
        side TEXT,
        price DECIMAL(18,6),
        size DECIMAL(18,2),
        expires_at TIMESTAMP,
        metadata_json JSON,
        PRIMARY KEY(cusip, dealer_id, timestamp, side)
    )
    """

_DEALER_QUOTES_SQLITE = """
    CREATE TABLE IF NOT EXISTS dealer_quotes (
        timestamp TEXT NOT NULL,
        cusip TEXT NOT NULL,
        dealer_id TEXT NOT NULL,
        side TEXT,
        price REAL,
        size REAL,
        expires_at TEXT,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(cusip, dealer_id, timestamp, side)
    )
    """


# ----- dealer_response_stats -----
_DEALER_RESPONSE_STATS_DUCKDB = """
    CREATE TABLE IF NOT EXISTS dealer_response_stats (
        dealer_id TEXT NOT NULL,
        window_start TIMESTAMP NOT NULL,
        window_end TIMESTAMP NOT NULL,
        requests INTEGER,
        responses INTEGER,
        avg_response_ms DECIMAL(18,3),
        metadata_json JSON,
        PRIMARY KEY(dealer_id, window_start, window_end)
    )
    """

_DEALER_RESPONSE_STATS_SQLITE = """
    CREATE TABLE IF NOT EXISTS dealer_response_stats (
        dealer_id TEXT NOT NULL,
        window_start TEXT NOT NULL,
        window_end TEXT NOT NULL,
        requests INTEGER,
        responses INTEGER,
        avg_response_ms REAL,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(dealer_id, window_start, window_end)
    )
    """


# ----- curve_snapshots -----
_CURVE_SNAPSHOTS_DUCKDB = """
    CREATE TABLE IF NOT EXISTS curve_snapshots (
        timestamp TIMESTAMP NOT NULL,
        curve_type TEXT NOT NULL,
        tenor TEXT NOT NULL,
        rate DECIMAL(18,6) NOT NULL,
        source TEXT,
        metadata_json JSON,
        PRIMARY KEY(timestamp, curve_type, tenor)
    )
    """

_CURVE_SNAPSHOTS_SQLITE = """
    CREATE TABLE IF NOT EXISTS curve_snapshots (
        timestamp TEXT NOT NULL,
        curve_type TEXT NOT NULL,
        tenor TEXT NOT NULL,
        rate REAL NOT NULL,
        source TEXT,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(timestamp, curve_type, tenor)
    )
    """


# ----- cds_curve_snapshots -----
_CDS_CURVE_SNAPSHOTS_DUCKDB = """
    CREATE TABLE IF NOT EXISTS cds_curve_snapshots (
        timestamp TIMESTAMP NOT NULL,
        reference_entity TEXT NOT NULL,
        tenor TEXT NOT NULL,
        spread_bps DECIMAL(18,4) NOT NULL,
        source TEXT,
        metadata_json JSON,
        PRIMARY KEY(timestamp, reference_entity, tenor)
    )
    """

_CDS_CURVE_SNAPSHOTS_SQLITE = """
    CREATE TABLE IF NOT EXISTS cds_curve_snapshots (
        timestamp TEXT NOT NULL,
        reference_entity TEXT NOT NULL,
        tenor TEXT NOT NULL,
        spread_bps REAL NOT NULL,
        source TEXT,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(timestamp, reference_entity, tenor)
    )
    """


# ----- credit_regime_scores (governance output) -----
# Mirrors CreditRegimeOutput data contract in schemas.py with the
# required model_run_id / release_gate / artifact_hash fields per
# non-negotiable constraint 7.
_CREDIT_REGIME_SCORES_DUCKDB = """
    CREATE TABLE IF NOT EXISTS credit_regime_scores (
        model_run_id TEXT NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        regime_score DECIMAL(8,4) NOT NULL,
        regime_label TEXT NOT NULL,
        confidence DECIMAL(8,6) NOT NULL,
        drivers_json JSON,
        component_scores_json JSON,
        release_gate INTEGER NOT NULL,
        artifact_hash TEXT NOT NULL,
        metadata_json JSON,
        PRIMARY KEY(model_run_id, timestamp)
    )
    """

_CREDIT_REGIME_SCORES_SQLITE = """
    CREATE TABLE IF NOT EXISTS credit_regime_scores (
        model_run_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        regime_score REAL NOT NULL,
        regime_label TEXT NOT NULL,
        confidence REAL NOT NULL,
        drivers_json TEXT,
        component_scores_json TEXT,
        release_gate INTEGER NOT NULL,
        artifact_hash TEXT NOT NULL,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(model_run_id, timestamp)
    )
    """


# ----- liquidity_stress_scores (governance output) -----
# Scope-aware: scope_type ∈ {market, sector, rating, cusip}.
_LIQUIDITY_STRESS_SCORES_DUCKDB = """
    CREATE TABLE IF NOT EXISTS liquidity_stress_scores (
        model_run_id TEXT NOT NULL,
        scope_type TEXT NOT NULL,
        scope_id TEXT NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        liquidity_score DECIMAL(8,4) NOT NULL,
        liquidity_label TEXT NOT NULL,
        confidence DECIMAL(8,6) NOT NULL,
        drivers_json JSON,
        release_gate INTEGER NOT NULL,
        artifact_hash TEXT NOT NULL,
        metadata_json JSON,
        PRIMARY KEY(model_run_id, scope_type, scope_id, timestamp)
    )
    """

_LIQUIDITY_STRESS_SCORES_SQLITE = """
    CREATE TABLE IF NOT EXISTS liquidity_stress_scores (
        model_run_id TEXT NOT NULL,
        scope_type TEXT NOT NULL,
        scope_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        liquidity_score REAL NOT NULL,
        liquidity_label TEXT NOT NULL,
        confidence REAL NOT NULL,
        drivers_json TEXT,
        release_gate INTEGER NOT NULL,
        artifact_hash TEXT NOT NULL,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(model_run_id, scope_type, scope_id, timestamp)
    )
    """


# ----- execution_confidence_predictions (governance output, with request_id) -----
# PR-15 (REVIEW.md §3.6) — request_id is required on the composite
# primary key so two API workers cannot write predictions for the same
# client request under different model_run_ids. The request_id is
# generated upstream by the API handler (PR-5).
_EXECUTION_CONFIDENCE_PREDICTIONS_DUCKDB = """
    CREATE TABLE IF NOT EXISTS execution_confidence_predictions (
        request_id TEXT NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        model_run_id TEXT NOT NULL,
        cusip TEXT NOT NULL,
        side TEXT NOT NULL,
        notional DECIMAL(18,2) NOT NULL,
        protocol TEXT NOT NULL,
        confidence_score DECIMAL(8,6) NOT NULL,
        expected_slippage_bps DECIMAL(18,4),
        confidence_interval_low DECIMAL(8,6),
        confidence_interval_high DECIMAL(8,6),
        recommended_action TEXT NOT NULL,
        human_review_required INTEGER NOT NULL,
        release_gate INTEGER NOT NULL,
        artifact_hash TEXT NOT NULL,
        metadata_json JSON,
        PRIMARY KEY(request_id, timestamp)
    )
    """

_EXECUTION_CONFIDENCE_PREDICTIONS_SQLITE = """
    CREATE TABLE IF NOT EXISTS execution_confidence_predictions (
        request_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        model_run_id TEXT NOT NULL,
        cusip TEXT NOT NULL,
        side TEXT NOT NULL,
        notional REAL NOT NULL,
        protocol TEXT NOT NULL,
        confidence_score REAL NOT NULL,
        expected_slippage_bps REAL,
        confidence_interval_low REAL,
        confidence_interval_high REAL,
        recommended_action TEXT NOT NULL,
        human_review_required INTEGER NOT NULL,
        release_gate INTEGER NOT NULL,
        artifact_hash TEXT NOT NULL,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(request_id, timestamp)
    )
    """


# ----- execution_outcomes -----
# Q-2 (REVIEW.md §3.4): observed_at > decision_timestamp is enforced
# by the writer; not encoded as a CHECK constraint because DuckDB +
# SQLite differ on CHECK clause support / semantics.
_EXECUTION_OUTCOMES_DUCKDB = """
    CREATE TABLE IF NOT EXISTS execution_outcomes (
        request_id TEXT NOT NULL,
        cusip TEXT NOT NULL,
        side TEXT NOT NULL,
        notional DECIMAL(18,2) NOT NULL,
        filled_quantity DECIMAL(18,2),
        execution_price DECIMAL(18,6),
        observed_at TIMESTAMP NOT NULL,
        outcome_observation_lag DECIMAL(18,3),
        decision_timestamp TIMESTAMP NOT NULL,
        metadata_json JSON,
        PRIMARY KEY(request_id)
    )
    """

_EXECUTION_OUTCOMES_SQLITE = """
    CREATE TABLE IF NOT EXISTS execution_outcomes (
        request_id TEXT NOT NULL,
        cusip TEXT NOT NULL,
        side TEXT NOT NULL,
        notional REAL NOT NULL,
        filled_quantity REAL,
        execution_price REAL,
        observed_at TEXT NOT NULL,
        outcome_observation_lag REAL,
        decision_timestamp TEXT NOT NULL,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(request_id)
    )
    """


# ----- tca_regime_segments -----
_TCA_REGIME_SEGMENTS_DUCKDB = """
    CREATE TABLE IF NOT EXISTS tca_regime_segments (
        model_run_id TEXT NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        regime_label TEXT NOT NULL,
        liquidity_label TEXT NOT NULL,
        execution_confidence_bucket TEXT NOT NULL,
        protocol TEXT NOT NULL,
        side TEXT NOT NULL,
        sector TEXT NOT NULL,
        rating TEXT NOT NULL,
        maturity_bucket TEXT NOT NULL,
        notional_bucket TEXT NOT NULL,
        metric_name TEXT NOT NULL,
        metric_value DECIMAL(18,6) NOT NULL,
        sample_count INTEGER NOT NULL,
        metadata_json JSON,
        PRIMARY KEY(
            model_run_id, timestamp, regime_label, liquidity_label,
            execution_confidence_bucket, protocol, side, sector,
            rating, maturity_bucket, notional_bucket, metric_name
        )
    )
    """

_TCA_REGIME_SEGMENTS_SQLITE = """
    CREATE TABLE IF NOT EXISTS tca_regime_segments (
        model_run_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        regime_label TEXT NOT NULL,
        liquidity_label TEXT NOT NULL,
        execution_confidence_bucket TEXT NOT NULL,
        protocol TEXT NOT NULL,
        side TEXT NOT NULL,
        sector TEXT NOT NULL,
        rating TEXT NOT NULL,
        maturity_bucket TEXT NOT NULL,
        notional_bucket TEXT NOT NULL,
        metric_name TEXT NOT NULL,
        metric_value REAL NOT NULL,
        sample_count INTEGER NOT NULL,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(
            model_run_id, timestamp, regime_label, liquidity_label,
            execution_confidence_bucket, protocol, side, sector,
            rating, maturity_bucket, notional_bucket, metric_name
        )
    )
    """


# ----- fixed_income_evidence_packs (governance output, with request_id) -----
# PR-15 (REVIEW.md §3.6) — composite PK on (model_run_id, request_id)
# so two API workers cannot land conflicting packs for the same client
# request under different model_run_ids.
_EVIDENCE_PACKS_DUCKDB = """
    CREATE TABLE IF NOT EXISTS fixed_income_evidence_packs (
        model_run_id TEXT NOT NULL,
        request_id TEXT NOT NULL,
        component_name TEXT NOT NULL,
        model_version TEXT NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        code_sha TEXT,
        model_hash TEXT NOT NULL,
        input_features_hash TEXT NOT NULL,
        output_hash TEXT NOT NULL,
        data_vintages_json JSON,
        validation_results_json JSON,
        release_gate INTEGER NOT NULL,
        random_seeds_json JSON,
        python_version TEXT,
        lockfile_hash TEXT,
        hmac_signature TEXT,
        metadata_json JSON,
        PRIMARY KEY(model_run_id, request_id)
    )
    """

_EVIDENCE_PACKS_SQLITE = """
    CREATE TABLE IF NOT EXISTS fixed_income_evidence_packs (
        model_run_id TEXT NOT NULL,
        request_id TEXT NOT NULL,
        component_name TEXT NOT NULL,
        model_version TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        code_sha TEXT,
        model_hash TEXT NOT NULL,
        input_features_hash TEXT NOT NULL,
        output_hash TEXT NOT NULL,
        data_vintages_json TEXT,
        validation_results_json TEXT,
        release_gate INTEGER NOT NULL,
        random_seeds_json TEXT,
        python_version TEXT,
        lockfile_hash TEXT,
        hmac_signature TEXT,
        metadata_json TEXT DEFAULT '{}',
        PRIMARY KEY(model_run_id, request_id)
    )
    """


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


# Per-table indexes (REVIEW.md §3.2 ASK-11). CREATE INDEX IF NOT EXISTS
# parses cleanly on both DuckDB 1.x and SQLite ≥3.30; no per-backend
# branching is needed.
_FI_TABLES: tuple[TableSpec, ...] = (
    TableSpec(
        name="bond_reference",
        create_sql=_BOND_REFERENCE_DUCKDB,
        sqlite_create_sql=_BOND_REFERENCE_SQLITE,
        primary_key=("cusip", "valid_from"),
        index_sql=(
            "CREATE INDEX IF NOT EXISTS idx_bond_reference_valid_window ON bond_reference(valid_from, valid_to)",
        ),
    ),
    TableSpec(
        name="trace_trades",
        create_sql=_TRACE_TRADES_DUCKDB,
        sqlite_create_sql=_TRACE_TRADES_SQLITE,
        primary_key=("trade_id", "cusip", "timestamp"),
        index_sql=("CREATE INDEX IF NOT EXISTS idx_trace_trades_cusip_ts ON trace_trades(cusip, timestamp)",),
    ),
    TableSpec(
        name="rfq_events",
        create_sql=_RFQ_EVENTS_DUCKDB,
        sqlite_create_sql=_RFQ_EVENTS_SQLITE,
        primary_key=("rfq_id", "timestamp"),
        index_sql=("CREATE INDEX IF NOT EXISTS idx_rfq_events_cusip_ts ON rfq_events(cusip, timestamp)",),
    ),
    TableSpec(
        name="dealer_quotes",
        create_sql=_DEALER_QUOTES_DUCKDB,
        sqlite_create_sql=_DEALER_QUOTES_SQLITE,
        primary_key=("cusip", "dealer_id", "timestamp", "side"),
        index_sql=("CREATE INDEX IF NOT EXISTS idx_dealer_quotes_cusip_ts ON dealer_quotes(cusip, timestamp)",),
    ),
    TableSpec(
        name="dealer_response_stats",
        create_sql=_DEALER_RESPONSE_STATS_DUCKDB,
        sqlite_create_sql=_DEALER_RESPONSE_STATS_SQLITE,
        primary_key=("dealer_id", "window_start", "window_end"),
    ),
    TableSpec(
        name="curve_snapshots",
        create_sql=_CURVE_SNAPSHOTS_DUCKDB,
        sqlite_create_sql=_CURVE_SNAPSHOTS_SQLITE,
        primary_key=("timestamp", "curve_type", "tenor"),
    ),
    TableSpec(
        name="cds_curve_snapshots",
        create_sql=_CDS_CURVE_SNAPSHOTS_DUCKDB,
        sqlite_create_sql=_CDS_CURVE_SNAPSHOTS_SQLITE,
        primary_key=("timestamp", "reference_entity", "tenor"),
    ),
    TableSpec(
        name="credit_regime_scores",
        create_sql=_CREDIT_REGIME_SCORES_DUCKDB,
        sqlite_create_sql=_CREDIT_REGIME_SCORES_SQLITE,
        primary_key=("model_run_id", "timestamp"),
    ),
    TableSpec(
        name="liquidity_stress_scores",
        create_sql=_LIQUIDITY_STRESS_SCORES_DUCKDB,
        sqlite_create_sql=_LIQUIDITY_STRESS_SCORES_SQLITE,
        primary_key=("model_run_id", "scope_type", "scope_id", "timestamp"),
        index_sql=(
            "CREATE INDEX IF NOT EXISTS idx_liquidity_scope_ts "
            "ON liquidity_stress_scores(scope_type, scope_id, timestamp)",
        ),
    ),
    TableSpec(
        name="execution_confidence_predictions",
        create_sql=_EXECUTION_CONFIDENCE_PREDICTIONS_DUCKDB,
        sqlite_create_sql=_EXECUTION_CONFIDENCE_PREDICTIONS_SQLITE,
        primary_key=("request_id", "timestamp"),
        index_sql=(
            "CREATE INDEX IF NOT EXISTS idx_exec_conf_action_ts "
            "ON execution_confidence_predictions(timestamp, recommended_action)",
        ),
    ),
    TableSpec(
        name="execution_outcomes",
        create_sql=_EXECUTION_OUTCOMES_DUCKDB,
        sqlite_create_sql=_EXECUTION_OUTCOMES_SQLITE,
        primary_key=("request_id",),
        # v1.5 PR-8 (Tier-4 FLAG F-A1, REVIEW.md): drop the explicit
        # ``idx_exec_outcomes_request_id`` index — DuckDB and SQLite
        # both auto-create an index on the PRIMARY KEY
        # (``request_id``), so the additional ``CREATE INDEX IF NOT
        # EXISTS idx_exec_outcomes_request_id ON
        # execution_outcomes(request_id)`` was redundant and only
        # consumed extra space.
        index_sql=(),
    ),
    TableSpec(
        name="tca_regime_segments",
        create_sql=_TCA_REGIME_SEGMENTS_DUCKDB,
        sqlite_create_sql=_TCA_REGIME_SEGMENTS_SQLITE,
        primary_key=(
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
        ),
    ),
    TableSpec(
        name="fixed_income_evidence_packs",
        create_sql=_EVIDENCE_PACKS_DUCKDB,
        sqlite_create_sql=_EVIDENCE_PACKS_SQLITE,
        primary_key=("model_run_id", "request_id"),
        index_sql=(
            "CREATE INDEX IF NOT EXISTS idx_evidence_packs_run_id ON fixed_income_evidence_packs(model_run_id)",
        ),
    ),
)


FI_TABLE_NAMES: tuple[str, ...] = tuple(spec.name for spec in _FI_TABLES)


def register() -> None:
    """Idempotently register the 13 FI tables with the warehouse.

    Called once by :mod:`market_regime_engine.fixed_income`'s
    ``__init__.py`` so any consumer that imports the FI package picks
    up the schema. :func:`register_tables` deduplicates by name so
    repeated imports are safe.
    """

    register_tables(_FI_TABLES)


__all__ = ["FI_TABLE_NAMES", "register"]
