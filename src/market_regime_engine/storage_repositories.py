# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from market_regime_engine.storage_backends import _Backend, _select_backend
from market_regime_engine.storage_registry import BackendName, registered_tables


def _normalise_asof_for_sql(value: Any) -> str | None:
    """Coerce ``value`` (``None``, ``datetime``, ``pd.Timestamp``, str) → ISO-8601.

    v1.5.1 (PR-9 FIX 2): the new indexed-SQL ``latest_*`` reads on the
    :class:`Warehouse` use parameter binding for the ``asof`` cap. SQLite
    and DuckDB both accept an ISO-8601 string compared against the
    ``timestamp`` column lexicographically (matching the table writers
    which emit ISO-8601 ``Z`` form), so we route every input through a
    single normaliser to keep the SQL deterministic.

    Returns ``None`` when ``value`` is ``None`` (meaning "no upper bound").
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return str(value)
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _isoformat_utc(value: Any) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()


def _is_duckdb_backend(backend: Any) -> bool:
    """True when the backend is the DuckDB facade.

    v1.5.1 (PR-9 FIX 2): used by
    :meth:`Warehouse.enrich_execution_requests_asof` to gate the
    DuckDB-specific ``ASOF LEFT JOIN`` path; SQLite callers stay on
    the existing pandas merge_asof.
    """
    return backend.__class__.__name__ == "_DuckDBBackend"


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
        with contextlib.suppress(Exception):
            self.backfill_execution_confidence_prediction_quantized_columns()

    def table_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in registered_tables())

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

    def read_dealer_response_stats(
        self,
        window_start: datetime | pd.Timestamp | str | None = None,
        window_end: datetime | pd.Timestamp | str | None = None,
    ) -> pd.DataFrame:
        """Read ``dealer_response_stats`` rows in the requested window.

        v1.6.0 (REVIEW_DEEP_V1_5_2.md A7 / Finding §3.2): pushes the
        ``window_end`` time-range filter into the SQL layer instead of
        reading the whole table and filtering in pandas. Removes the
        ``warehouse._backend.read_sql("SELECT *")`` private-attribute
        access in
        :func:`execution_confidence.build_execution_features` and lets
        the read scale with the requested lookback window instead of
        with the deployment age.

        ``window_start`` and ``window_end`` are inclusive bounds
        against ``dealer_response_stats.window_end`` (the right edge
        of the statistics window — a row is in scope iff its
        ``window_end`` lies in ``[window_start, window_end]``). Pass
        ``None`` for either bound to disable that side of the
        filter.

        Backed by ``idx_dealer_response_stats_window_end`` (see
        :mod:`fixed_income.schema`) so the ``WHERE window_end <= ?``
        leading filter is index-satisfied on both DuckDB and SQLite.
        """
        clauses: list[str] = []
        params: list[object] = []
        if window_start is not None:
            clauses.append("window_end >= ?")
            params.append(pd.Timestamp(window_start).isoformat())
        if window_end is not None:
            clauses.append("window_end <= ?")
            params.append(pd.Timestamp(window_end).isoformat())
        sql = "SELECT * FROM dealer_response_stats"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY dealer_id, window_start"
        return self._backend.read_sql(sql, params if params else None)

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
        return self._backend.read_sql("SELECT * FROM cds_curve_snapshots ORDER BY timestamp, reference_entity, tenor")

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

    def latest_credit_regime_score(self, asof: datetime | pd.Timestamp | str | None = None) -> pd.DataFrame | None:
        """Return at most one row from ``credit_regime_scores`` as of ``asof``.

        v1.5.1 (PR-9 FIX 2): mirrors the indexed SQL fast path that
        :func:`latest_credit_regime_score_identity` already uses, but
        returns the full row so the FI scorer can build a
        :class:`CreditRegimeOutput` without a follow-up read. Falls back
        to ``None`` (caller decides 503 vs fail-closed) when the table
        is empty for the ``asof`` slice.

        Implementation uses a parameterised ``SELECT ... WHERE
        timestamp <= ? ORDER BY timestamp DESC, model_run_id DESC LIMIT
        1`` against the backend.

        v1.6.0 (REVIEW_DEEP_V1_5_2.md A12 / Finding §3.6): SQLite +
        DuckDB honour the dedicated ``idx_credit_regime_ts_run``
        secondary index defined in :mod:`fixed_income.schema`
        whose leading column matches the ORDER BY — the v1.5.x
        docstring incorrectly claimed the PK index served the
        sort, but the PK on ``(model_run_id, timestamp)`` cannot
        satisfy a leading ``timestamp DESC`` sort.
        """
        asof_iso = _normalise_asof_for_sql(asof)
        if asof_iso is None:
            df = self._backend.read_sql(
                "SELECT * FROM credit_regime_scores ORDER BY timestamp DESC, model_run_id DESC LIMIT 1"
            )
        else:
            df = self._backend.read_sql(
                "SELECT * FROM credit_regime_scores "
                "WHERE timestamp <= ? "
                "ORDER BY timestamp DESC, model_run_id DESC "
                "LIMIT 1",
                params=(asof_iso,),
            )
        if df is None or df.empty:
            return None
        return df

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
        return self._backend.read_sql("SELECT * FROM liquidity_stress_scores ORDER BY timestamp, scope_type, scope_id")

    def latest_liquidity_stress_score(
        self,
        *,
        scope_type: str | None = None,
        scope_id: str | None = None,
        asof: datetime | pd.Timestamp | str | None = None,
    ) -> pd.DataFrame | None:
        """Return at most one row from ``liquidity_stress_scores``.

        v1.5.1 (PR-9 FIX 2): the existing
        :func:`fixed_income.liquidity_stress.latest_liquidity_stress_score`
        loaded the entire ``liquidity_stress_scores`` table and filtered
        in pandas; this path issues a parameterised SQL ``LIMIT 1``
        instead, hitting ``idx_liquidity_scope_ts`` when ``scope_type``
        / ``scope_id`` are pinned and the primary key otherwise. Falls
        back to ``None`` when no matching row exists.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if scope_type is not None:
            clauses.append("scope_type = ?")
            params.append(str(scope_type))
        if scope_id is not None:
            clauses.append("scope_id = ?")
            params.append(str(scope_id))
        asof_iso = _normalise_asof_for_sql(asof)
        if asof_iso is not None:
            clauses.append("timestamp <= ?")
            params.append(asof_iso)
        where_sql = ""
        if clauses:
            where_sql = " WHERE " + " AND ".join(clauses)
        sql = (
            "SELECT * FROM liquidity_stress_scores"
            + where_sql
            + " ORDER BY timestamp DESC, scope_type DESC, scope_id DESC LIMIT 1"
        )
        df = self._backend.read_sql(sql, params=tuple(params) if params else None)
        if df is None or df.empty:
            return None
        return df

    def latest_execution_confidence_prediction(
        self,
        *,
        request_id: str | None = None,
        cusip: str | None = None,
        asof: datetime | pd.Timestamp | str | None = None,
    ) -> pd.DataFrame | None:
        """Return at most one row from ``execution_confidence_predictions``.

        v1.5.1 (PR-9 FIX 2): bond-level execution-confidence reads
        (``bond_level_execution_predictions`` workflow) previously
        scanned the full table. Parameterised SQL with the indexed
        ``timestamp`` column gives O(log N) latency per request.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if request_id is not None:
            clauses.append("request_id = ?")
            params.append(str(request_id))
        if cusip is not None:
            clauses.append("cusip = ?")
            params.append(str(cusip))
        asof_iso = _normalise_asof_for_sql(asof)
        if asof_iso is not None:
            clauses.append("timestamp <= ?")
            params.append(asof_iso)
        where_sql = ""
        if clauses:
            where_sql = " WHERE " + " AND ".join(clauses)
        sql = (
            "SELECT * FROM execution_confidence_predictions"
            + where_sql
            + " ORDER BY timestamp DESC, request_id DESC LIMIT 1"
        )
        df = self._backend.read_sql(sql, params=tuple(params) if params else None)
        if df is None or df.empty:
            return None
        return df

    def latest_tca_regime_segment(
        self,
        *,
        regime_label: str | None = None,
        liquidity_label: str | None = None,
        asof: datetime | pd.Timestamp | str | None = None,
    ) -> pd.DataFrame | None:
        """Return at most one row from ``tca_regime_segments``.

        v1.5.1 (PR-9 FIX 2): mirrors the credit / liquidity fast paths
        so any consumer that only needs the latest row pays O(log N)
        rather than O(N).
        """
        clauses: list[str] = []
        params: list[Any] = []
        if regime_label is not None:
            clauses.append("regime_label = ?")
            params.append(str(regime_label))
        if liquidity_label is not None:
            clauses.append("liquidity_label = ?")
            params.append(str(liquidity_label))
        asof_iso = _normalise_asof_for_sql(asof)
        if asof_iso is not None:
            clauses.append("timestamp <= ?")
            params.append(asof_iso)
        where_sql = ""
        if clauses:
            where_sql = " WHERE " + " AND ".join(clauses)
        sql = "SELECT * FROM tca_regime_segments" + where_sql + " ORDER BY timestamp DESC, model_run_id DESC LIMIT 1"
        df = self._backend.read_sql(sql, params=tuple(params) if params else None)
        if df is None or df.empty:
            return None
        return df

    def enrich_execution_requests_asof(self, execution_requests: pd.DataFrame) -> pd.DataFrame:
        """ASOF-join ``execution_requests`` to credit + liquidity scores.

        v1.5.1 (PR-9 FIX 2): on a DuckDB backend this rewrites the
        traditional pandas merge_asof into a single SQL statement that
        uses DuckDB's native ``ASOF LEFT JOIN``. The execution-request
        frame is registered as a temporary view; the join produces one
        row per input request annotated with the latest
        ``credit_regime_scores.regime_label`` and per-CUSIP
        ``liquidity_stress_scores.liquidity_label`` available at the
        request timestamp.

        When DuckDB is unavailable or the warehouse is backed by SQLite,
        the helper now performs the same release-gated point-in-time join
        with ``pandas.merge_asof`` instead of returning the input frame
        unchanged. This keeps the public method behavior complete across
        both backends while preserving DuckDB's native ASOF JOIN fast path
        where available.

        ``execution_requests`` MUST carry ``timestamp`` (ISO-8601 string
        or :class:`pandas.Timestamp`) and ``cusip`` columns. Any
        additional columns are passed through verbatim.
        """
        if execution_requests is None or execution_requests.empty:
            return execution_requests.copy() if execution_requests is not None else execution_requests

        def _release_gated(df: pd.DataFrame) -> pd.DataFrame:
            if df is None or df.empty or "release_gate" not in df.columns:
                return df.copy() if df is not None else pd.DataFrame()
            release_gate = pd.to_numeric(df["release_gate"], errors="coerce").fillna(0).astype(int)
            return df.loc[release_gate == 1].copy()

        def _pandas_asof_fallback() -> pd.DataFrame:
            helper_ts = "_mre_asof_timestamp"
            out = execution_requests.copy()
            out[helper_ts] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
            out["_mre_input_order"] = range(len(out))

            # Credit-regime join: latest released global regime row as-of
            # each request timestamp.
            credit = _release_gated(self.read_credit_regime_scores())
            if credit is not None and not credit.empty and "timestamp" in credit.columns:
                credit = credit.copy()
                credit[helper_ts] = pd.to_datetime(credit["timestamp"], utc=True, errors="coerce")
                credit = credit.dropna(subset=[helper_ts]).sort_values(helper_ts)
                out_sorted = out.sort_values(helper_ts)
                if not credit.empty:
                    out = pd.merge_asof(
                        out_sorted,
                        credit[[helper_ts, "regime_label"]],
                        on=helper_ts,
                        direction="backward",
                    )
                else:
                    out = out_sorted.copy()
                    out["regime_label"] = pd.NA
            else:
                out = out.sort_values(helper_ts).copy()
                out["regime_label"] = pd.NA

            # Liquidity join: latest released CUSIP-scoped liquidity row
            # as-of the request timestamp. ``merge_asof(..., by=...)`` is
            # used rather than post-filtering so a future row for the same
            # CUSIP cannot mask a valid prior row.
            liquidity = _release_gated(self.read_liquidity_stress_scores())
            if (
                liquidity is not None
                and not liquidity.empty
                and {"timestamp", "scope_type", "scope_id"}.issubset(liquidity.columns)
            ):
                liquidity = liquidity.loc[liquidity["scope_type"].astype(str) == "cusip"].copy()
                liquidity[helper_ts] = pd.to_datetime(liquidity["timestamp"], utc=True, errors="coerce")
                liquidity = liquidity.dropna(subset=[helper_ts]).rename(columns={"scope_id": "cusip"})
                if not liquidity.empty:
                    out = pd.merge_asof(
                        out.sort_values(helper_ts),
                        liquidity[[helper_ts, "cusip", "liquidity_label"]].sort_values(helper_ts),
                        on=helper_ts,
                        by="cusip",
                        direction="backward",
                    )
                else:
                    out["liquidity_label"] = pd.NA
            else:
                out["liquidity_label"] = pd.NA

            out = out.sort_values("_mre_input_order").drop(columns=[helper_ts, "_mre_input_order"], errors="ignore")
            return out.reset_index(drop=True)

        if not _is_duckdb_backend(self._backend):
            return _pandas_asof_fallback()

        # DuckDB ASOF LEFT JOIN path. Register the input frame as a view
        # so the SQL statement is self-contained.
        try:  # pragma: no cover - exercised when duckdb is installed
            backend = self._backend
            conn = backend.conn  # type: ignore[attr-defined]
            frame = execution_requests.copy()
            helper_ts = "_mre_asof_timestamp"
            # DuckDB FI tables store timestamps as TIMESTAMP; register a
            # timezone-naive UTC helper column for the ASOF predicate while
            # preserving the caller's original ``timestamp`` column.
            parsed_ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            frame[helper_ts] = parsed_ts.dt.tz_convert("UTC").dt.tz_localize(None)

            def _quote_ident(name: object) -> str:
                return '"' + str(name).replace('"', '""') + '"'

            passthrough_cols = [col for col in frame.columns if col != helper_ts]
            select_cols = ",\n                        ".join(f"e.{_quote_ident(col)}" for col in passthrough_cols)
            conn.register("execution_requests_asof_input", frame)
            try:
                # v1.6.0 (REVIEW_DEEP_V1_5_2.md A14 / Finding
                # §3.8): the ASOF JOIN MUST restrict to
                # release-gated rows so a not-yet-promoted
                # candidate score never colours an execution
                # decision. Previously every credit / liquidity
                # row qualified, including release_gate=0
                # candidates — a governance contract violation
                # because downstream consumers (TCA segmentation,
                # execution-confidence dashboards) treat the
                # joined label as if it had cleared the gate.
                joined = conn.execute(
                    f"""
                    SELECT
                        {select_cols},
                        c.regime_label  AS regime_label,
                        l.liquidity_label AS liquidity_label
                    FROM execution_requests_asof_input AS e
                    ASOF LEFT JOIN credit_regime_scores AS c
                        ON e.{_quote_ident(helper_ts)} >= c.timestamp
                       AND c.release_gate = 1
                    ASOF LEFT JOIN liquidity_stress_scores AS l
                        ON e.{_quote_ident(helper_ts)} >= l.timestamp
                       AND e.cusip = l.scope_id
                       AND l.scope_type = 'cusip'
                       AND l.release_gate = 1
                    """
                ).fetchdf()
            finally:
                with contextlib.suppress(Exception):
                    conn.unregister("execution_requests_asof_input")
        except Exception:
            # Defensive fallback: any DuckDB error degrades to the
            # pandas path so a malformed input or unexpected schema does
            # not bring down execution-confidence scoring.
            return _pandas_asof_fallback()
        return joined

    def write_execution_confidence_prediction(self, df: pd.DataFrame) -> int:
        frame = df.copy() if df is not None else df
        frame = self._populate_execution_confidence_quantized_columns(frame)
        return self._write_fi(
            "execution_confidence_predictions",
            frame,
            [
                "request_id",
                "timestamp",
                "model_run_id",
                "cusip",
                "side",
                "notional",
                "notional_cents",
                "protocol",
                "confidence_score",
                "confidence_score_ppm",
                "expected_slippage_bps",
                "expected_slippage_bps_q4",
                "confidence_interval_low",
                "confidence_interval_low_ppm",
                "confidence_interval_high",
                "confidence_interval_high_ppm",
                "recommended_action",
                "human_review_required",
                "release_gate",
                "artifact_hash",
                "metadata_json",
            ],
        )

    def read_execution_confidence_predictions(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM execution_confidence_predictions ORDER BY timestamp, request_id")

    def _populate_execution_confidence_quantized_columns(
        self,
        frame: pd.DataFrame | None,
    ) -> pd.DataFrame | None:
        if frame is None or frame.empty:
            return frame
        from market_regime_engine.fixed_income.numeric_contracts import (
            bps_to_q4,
            money_to_cents,
            prob_to_ppm,
        )

        out = frame.copy()
        spec: tuple[tuple[str, str, Any], ...] = (
            ("notional_cents", "notional", money_to_cents),
            ("confidence_score_ppm", "confidence_score", prob_to_ppm),
            ("expected_slippage_bps_q4", "expected_slippage_bps", bps_to_q4),
            ("confidence_interval_low_ppm", "confidence_interval_low", prob_to_ppm),
            ("confidence_interval_high_ppm", "confidence_interval_high", prob_to_ppm),
        )
        for target, source, fn in spec:
            existing = list(out[target]) if target in out else [None] * len(out)
            source_values = list(out[source]) if source in out else [None] * len(out)
            values: list[int | None] = []
            for current, source_value in zip(existing, source_values, strict=False):
                try:
                    current_missing = current is None or pd.isna(current)
                except Exception:
                    current_missing = False
                if not current_missing:
                    values.append(int(str(current)))
                    continue
                try:
                    if source_value is None or pd.isna(source_value):
                        values.append(None)
                    else:
                        values.append(fn(source_value))
                except Exception:
                    values.append(None)
            out[target] = values
        return out

    def backfill_execution_confidence_prediction_quantized_columns(self) -> int:
        required = {
            "notional",
            "confidence_score",
            "expected_slippage_bps",
            "confidence_interval_low",
            "confidence_interval_high",
            "notional_cents",
            "confidence_score_ppm",
            "expected_slippage_bps_q4",
            "confidence_interval_low_ppm",
            "confidence_interval_high_ppm",
        }
        if not required <= self._backend.column_names("execution_confidence_predictions"):
            return 0
        frame = self.read_execution_confidence_predictions()
        if frame is None or frame.empty:
            return 0
        targets = [
            "notional_cents",
            "confidence_score_ppm",
            "expected_slippage_bps_q4",
            "confidence_interval_low_ppm",
            "confidence_interval_high_ppm",
        ]
        missing = frame[targets].isna().any(axis=1)
        if not bool(missing.any()):
            return 0
        subset = frame.loc[missing].copy()
        populated = self._populate_execution_confidence_quantized_columns(subset)
        if populated is None or populated.empty:
            return 0
        changed = pd.Series(False, index=populated.index)
        for target in targets:
            changed = changed | (subset[target].isna() & populated[target].notna())
        changed_frame = populated.loc[changed].copy()
        if changed_frame.empty:
            return 0
        return self.write_execution_confidence_prediction(changed_frame)

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
                f"execution_outcomes requires observed_at > decision_timestamp; offending request_ids: {offenders!r}"
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

    def write_execution_outcomes(self, df: pd.DataFrame) -> int:
        return self.write_execution_outcome(df)

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
        return self._backend.read_sql("SELECT * FROM tca_regime_segments ORDER BY timestamp, model_run_id, metric_name")

    def write_xpro_decision_artifact(self, artifact: Any) -> int:
        from market_regime_engine.fixed_income.xpro_decision import xpro_decision_artifact_to_row

        row = artifact
        if isinstance(artifact, Mapping):
            row = xpro_decision_artifact_to_row(artifact)
        frame = pd.DataFrame([row]) if isinstance(row, Mapping) else pd.DataFrame(row)
        if not frame.empty:
            frame["release_gate"] = frame["release_gate"].astype(int)
            if "metadata_json" not in frame:
                frame["metadata_json"] = "{}"
        return self._write_fi(
            "xpro_decision_artifacts",
            frame,
            [
                "decision_id",
                "request_id",
                "timestamp",
                "model_run_id",
                "recommended_protocol",
                "release_gate",
                "artifact_hash",
                "hmac_signature",
                "payload_json",
                "metadata_json",
            ],
        )

    def read_xpro_decision_artifacts(self) -> pd.DataFrame:
        return self._backend.read_sql("SELECT * FROM xpro_decision_artifacts ORDER BY timestamp, decision_id")

    def latest_xpro_decision_artifact(self, decision_id: str) -> pd.DataFrame | None:
        frame = self._backend.read_sql(
            "SELECT * FROM xpro_decision_artifacts WHERE decision_id = ? ORDER BY timestamp DESC LIMIT 1",
            [str(decision_id)],
        )
        if frame is None or frame.empty:
            return None
        return frame

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

    def latest_evidence_pack(
        self,
        model_run_id: str,
        *,
        request_id: str | None = None,
        asof: datetime | pd.Timestamp | str | None = None,
    ) -> pd.DataFrame | None:
        """Return at most one ``fixed_income_evidence_packs`` row.

        v1.6.0 (REVIEW_DEEP_V1_5_2.md A8 / Finding §3.3): mirrors
        :meth:`latest_credit_regime_score` and pushes filtering into
        the SQL layer instead of reading the whole table and
        filtering in pandas. Backed by the existing
        ``idx_evidence_packs_run_id`` index on
        ``fixed_income_evidence_packs(model_run_id)`` defined in
        :mod:`fixed_income.schema`.

        Filters:
        - ``model_run_id`` — required exact match.
        - ``request_id`` — optional exact match.
        - ``asof`` — when present, restrict to ``timestamp <= asof``.

        Returns the most-recent matching row (by
        ``timestamp DESC, request_id DESC``) as a single-row
        DataFrame, or ``None`` when no row matches.
        """
        clauses: list[str] = ["model_run_id = ?"]
        params: list[object] = [str(model_run_id)]
        if request_id is not None:
            clauses.append("request_id = ?")
            params.append(str(request_id))
        if asof is not None:
            clauses.append("timestamp <= ?")
            params.append(pd.Timestamp(asof).isoformat())
        sql = (
            "SELECT * FROM fixed_income_evidence_packs WHERE "
            + " AND ".join(clauses)
            + " ORDER BY timestamp DESC, request_id DESC LIMIT 1"
        )
        out = self._backend.read_sql(sql, params)
        if out is None or out.empty:
            return None
        if "timestamp" in out:
            out = out.copy()
            out["timestamp"] = out["timestamp"].map(_isoformat_utc)
        return out

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
    sql_filter = "valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)"
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


__all__ = [
    "Warehouse",
    "_is_duckdb_backend",
    "_normalise_asof_for_sql",
    "_read_with_params",
    "migrate_warehouse",
    "read_bond_reference_asof",
    "read_bond_reference_history",
]
