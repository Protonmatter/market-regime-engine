# SPDX-License-Identifier: Apache-2.0
"""Data-contract primitives for point-in-time market data.

This module is intentionally boring. Boring contracts are useful contracts.
Clever contracts are how leakage sneaks in wearing a fake mustache.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_FEATURE_COLUMNS: tuple[str, ...] = (
    "series_id",
    "entity_id",
    "forecast_origin",
    "observation_date",
    "observed_at",
    "available_at",
    "as_of",
    "value",
    "source",
    "source_revision_id",
    "snapshot_id",
)

REQUIRED_LABEL_COLUMNS: tuple[str, ...] = (
    "entity_id",
    "forecast_origin",
    "label_time",
    "horizon",
    "target",
    "label_value",
    "label_available_at",
)

FEATURE_DATETIME_COLUMNS: tuple[str, ...] = (
    "forecast_origin",
    "observation_date",
    "observed_at",
    "available_at",
    "as_of",
    "source_revision_available_at",
    "revision_available_at",
)

LABEL_DATETIME_COLUMNS: tuple[str, ...] = (
    "forecast_origin",
    "label_time",
    "label_available_at",
    "joined_at",
    "label_joined_at",
    "as_of",
)


@dataclass(frozen=True)
class ContractIssue:
    """One schema or contract violation."""

    severity: str
    table: str
    check: str
    message: str
    row: int | None = None
    column: str | None = None
    value: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContractReport:
    """Structured report for schema validation."""

    table: str
    rows: int
    required_columns: tuple[str, ...]
    missing_columns: tuple[str, ...] = ()
    issues: tuple[ContractIssue, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        return not self.missing_columns and not any(issue.severity == "blocker" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "rows": self.rows,
            "required_columns": list(self.required_columns),
            "missing_columns": list(self.missing_columns),
            "passed": self.passed,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def read_table(path: str | Path) -> pd.DataFrame:
    """Read CSV, JSON/JSONL, or Parquet into a pandas DataFrame."""

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    suffix = p.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if suffix == ".jsonl":
        return pd.read_json(p, lines=True)
    if suffix == ".json":
        return pd.read_json(p)
    return pd.read_csv(p)


def missing_columns(frame: pd.DataFrame, required: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(col for col in required if col not in frame.columns)


def require_columns(frame: pd.DataFrame, required: tuple[str, ...], *, table: str) -> ContractReport:
    """Validate required columns without mutating the frame."""

    missing = missing_columns(frame, required)
    issues = tuple(
        ContractIssue(
            severity="blocker",
            table=table,
            check="required_column",
            message=f"Missing required column: {col}",
            column=col,
        )
        for col in missing
    )
    return ContractReport(table=table, rows=int(len(frame)), required_columns=required, missing_columns=missing, issues=issues)


def coerce_datetime_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Return a copy with known datetime columns coerced to pandas timestamps."""

    out = frame.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce", utc=True)
    return out


def null_datetime_issues(frame: pd.DataFrame, columns: tuple[str, ...], *, table: str) -> list[ContractIssue]:
    """Report datetime columns that could not be parsed."""

    issues: list[ContractIssue] = []
    for col in columns:
        if col not in frame.columns:
            continue
        mask = frame[col].isna()
        for idx in frame.index[mask].tolist():
            issues.append(
                ContractIssue(
                    severity="blocker",
                    table=table,
                    check="parse_datetime",
                    message=f"Unparseable or null datetime in {col}",
                    row=int(idx) if isinstance(idx, int) else None,
                    column=col,
                    value=None,
                )
            )
    return issues


__all__ = [
    "ContractIssue",
    "ContractReport",
    "FEATURE_DATETIME_COLUMNS",
    "LABEL_DATETIME_COLUMNS",
    "REQUIRED_FEATURE_COLUMNS",
    "REQUIRED_LABEL_COLUMNS",
    "coerce_datetime_columns",
    "missing_columns",
    "null_datetime_issues",
    "read_table",
    "require_columns",
]
