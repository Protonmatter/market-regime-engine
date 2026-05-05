# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

EXPORT_TABLES = [
    "observations",
    "features",
    "regimes",
    "model_outputs",
    "calibrated_outputs",
    "recession_labels",
    "historical_analogs",
    "driver_attribution",
    "confidence_scores",
    "invalidation_triggers",
    "model_runs",
    "release_calendar_audit",
    "calibration_models",
    "exact_release_calendar",
    "ensemble_weights",
    "stacking_diagnostics",
    "model_drift",
    "release_gates",
    "alfred_ingestion_manifest",
    "hazard_diagnostics",
    "oos_predictions",
    "routed_alerts",
    "promotion_workflow",
    "series_vintages",
    "vintage_observations",
    "feature_asof_values",
    "vintage_audits",
]


@dataclass
class WarehouseExportResult:
    table: str
    rows: int
    path: str
    format: str
    duckdb_registered: bool = False

    def as_dict(self) -> dict:
        return {
            "table": self.table,
            "rows": self.rows,
            "path": self.path,
            "format": self.format,
            "duckdb_registered": self.duckdb_registered,
        }


def _has_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def export_sqlite_to_lake(
    sqlite_path: str | Path,
    out_dir: str | Path = "data/lake",
    tables: list[str] | None = None,
    prefer_parquet: bool = True,
) -> pd.DataFrame:
    """Export SQLite tables to a simple analytical lake.

    Parquet is used when a parquet engine is installed. Otherwise CSV is used.
    The function intentionally avoids making DuckDB mandatory so the MVP remains
    runnable on locked-down hosts.
    """
    sqlite_path = Path(sqlite_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = tables or EXPORT_TABLES
    can_parquet = prefer_parquet and (_has_module("pyarrow") or _has_module("fastparquet"))
    fmt = "parquet" if can_parquet else "csv"

    results: list[WarehouseExportResult] = []
    with sqlite3.connect(str(sqlite_path)) as conn:
        existing = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].tolist()
        for table in tables:
            if table not in existing:
                continue
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            if df.empty:
                continue
            path = out_dir / f"{table}.{fmt}"
            if fmt == "parquet":
                df.to_parquet(path, index=False)
            else:
                df.to_csv(path, index=False)
            results.append(WarehouseExportResult(table=table, rows=len(df), path=str(path), format=fmt))

    manifest = pd.DataFrame([r.as_dict() for r in results])
    if not manifest.empty:
        manifest.to_csv(out_dir / "manifest.csv", index=False)
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest.to_dict(orient="records"), indent=2), encoding="utf-8"
        )
    return manifest


def build_duckdb_database(
    lake_dir: str | Path = "data/lake", duckdb_path: str | Path = "data/mre.duckdb"
) -> pd.DataFrame:
    """Build a DuckDB database from exported CSV/Parquet artifacts when duckdb exists.

    If duckdb is unavailable, returns a manifest with status=skipped instead of
    failing. That keeps v0.6 deployable before optional analytics deps are added.
    """
    lake_dir = Path(lake_dir)
    duckdb_path = Path(duckdb_path)
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = lake_dir / "manifest.csv"
    if not manifest_path.exists():
        return pd.DataFrame([{"status": "missing_manifest", "duckdb_path": str(duckdb_path)}])
    manifest = pd.read_csv(manifest_path)
    if not _has_module("duckdb"):
        manifest["status"] = "duckdb_not_installed"
        manifest["duckdb_path"] = str(duckdb_path)
        return manifest

    import duckdb  # type: ignore

    con = duckdb.connect(str(duckdb_path))
    created = []
    try:
        for row in manifest.to_dict(orient="records"):
            table = row["table"]
            path = row["path"]
            fmt = row["format"]
            if fmt == "parquet":
                sql = f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_parquet('{path}')"
            else:
                sql = f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_csv_auto('{path}', HEADER=TRUE)"
            con.execute(sql)
            created.append(
                {"table": table, "rows": int(row["rows"]), "status": "created", "duckdb_path": str(duckdb_path)}
            )
    finally:
        con.close()
    return pd.DataFrame(created)


def warehouse_health(lake_dir: str | Path = "data/lake") -> pd.DataFrame:
    lake_dir = Path(lake_dir)
    manifest_path = lake_dir / "manifest.csv"
    if not manifest_path.exists():
        return pd.DataFrame([{"status": "no_lake_export", "path": str(lake_dir)}])
    manifest = pd.read_csv(manifest_path)
    manifest["exists"] = manifest["path"].map(lambda p: Path(str(p)).exists())
    manifest["size_bytes"] = manifest["path"].map(lambda p: Path(str(p)).stat().st_size if Path(str(p)).exists() else 0)
    return manifest
