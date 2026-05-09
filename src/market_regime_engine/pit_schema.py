# SPDX-License-Identifier: Apache-2.0
"""Point-in-time schema validation.

The schema checks here validate table shape and row-local time invariants. Join
leakage checks live in :mod:`market_regime_engine.leakage_checks`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from market_regime_engine.data_contracts import (
    ContractIssue,
    ContractReport,
    FEATURE_DATETIME_COLUMNS,
    LABEL_DATETIME_COLUMNS,
    REQUIRED_FEATURE_COLUMNS,
    REQUIRED_LABEL_COLUMNS,
    coerce_datetime_columns,
    null_datetime_issues,
    require_columns,
)


@dataclass(frozen=True)
class PITSchemaReport:
    """Combined feature + label schema report."""

    feature_report: ContractReport
    label_report: ContractReport

    @property
    def passed(self) -> bool:
        return self.feature_report.passed and self.label_report.passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "features": self.feature_report.to_dict(),
            "labels": self.label_report.to_dict(),
        }


def validate_feature_schema(features: pd.DataFrame) -> tuple[pd.DataFrame, ContractReport]:
    """Validate required feature columns and row-local PIT invariants."""

    base = require_columns(features, REQUIRED_FEATURE_COLUMNS, table="features")
    if base.missing_columns:
        return features.copy(), base

    frame = coerce_datetime_columns(features, FEATURE_DATETIME_COLUMNS)
    issues = list(base.issues)
    issues.extend(null_datetime_issues(frame, ("forecast_origin", "observed_at", "available_at", "as_of"), table="features"))

    issues.extend(
        _pairwise_time_check(
            frame,
            left="observed_at",
            op="<=",
            right="as_of",
            table="features",
            check="observed_at_lte_as_of",
            message="observed_at must be <= as_of",
        )
    )
    issues.extend(
        _pairwise_time_check(
            frame,
            left="available_at",
            op="<=",
            right="as_of",
            table="features",
            check="available_at_lte_as_of",
            message="available_at must be <= as_of",
        )
    )
    issues.extend(
        _pairwise_time_check(
            frame,
            left="as_of",
            op="<=",
            right="forecast_origin",
            table="features",
            check="feature_as_of_lte_forecast_origin",
            message="feature.as_of must be <= forecast_origin",
        )
    )

    revision_col = _revision_available_column(frame)
    if revision_col:
        issues.extend(
            _pairwise_time_check(
                frame,
                left=revision_col,
                op="<=",
                right="as_of",
                table="features",
                check="revision_available_lte_as_of",
                message=f"{revision_col} must be <= as_of",
            )
        )

    return frame, ContractReport(
        table="features",
        rows=int(len(frame)),
        required_columns=REQUIRED_FEATURE_COLUMNS,
        missing_columns=base.missing_columns,
        issues=tuple(issues),
    )


def validate_label_schema(labels: pd.DataFrame) -> tuple[pd.DataFrame, ContractReport]:
    """Validate required label columns and row-local label invariants."""

    base = require_columns(labels, REQUIRED_LABEL_COLUMNS, table="labels")
    if base.missing_columns:
        return labels.copy(), base

    frame = coerce_datetime_columns(labels, LABEL_DATETIME_COLUMNS)
    issues = list(base.issues)
    issues.extend(null_datetime_issues(frame, ("forecast_origin", "label_time", "label_available_at"), table="labels"))

    issues.extend(
        _pairwise_time_check(
            frame,
            left="forecast_origin",
            op="<=",
            right="label_time",
            table="labels",
            check="forecast_origin_lte_label_time",
            message="forecast_origin must be <= label_time",
        )
    )
    issues.extend(
        _pairwise_time_check(
            frame,
            left="label_time",
            op="<=",
            right="label_available_at",
            table="labels",
            check="label_time_lte_label_available_at",
            message="label_available_at must be >= label_time",
        )
    )

    for joined_col in ("joined_at", "label_joined_at", "as_of"):
        if joined_col in frame.columns:
            issues.extend(
                _pairwise_time_check(
                    frame,
                    left="label_available_at",
                    op="<=",
                    right=joined_col,
                    table="labels",
                    check="label_available_lte_join_time",
                    message=f"label_available_at must be <= {joined_col}",
                )
            )

    return frame, ContractReport(
        table="labels",
        rows=int(len(frame)),
        required_columns=REQUIRED_LABEL_COLUMNS,
        missing_columns=base.missing_columns,
        issues=tuple(issues),
    )


def validate_pit_schema(features: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, PITSchemaReport]:
    """Validate both feature and label PIT contracts."""

    fframe, freport = validate_feature_schema(features)
    lframe, lreport = validate_label_schema(labels)
    return fframe, lframe, PITSchemaReport(feature_report=freport, label_report=lreport)


def _revision_available_column(frame: pd.DataFrame) -> str | None:
    for col in ("source_revision_available_at", "revision_available_at"):
        if col in frame.columns:
            return col
    return None


def _pairwise_time_check(
    frame: pd.DataFrame,
    *,
    left: str,
    op: str,
    right: str,
    table: str,
    check: str,
    message: str,
) -> list[ContractIssue]:
    if left not in frame.columns or right not in frame.columns:
        return []
    if op != "<=":
        raise ValueError(f"Unsupported operator: {op}")
    mask = frame[left].notna() & frame[right].notna() & (frame[left] > frame[right])
    issues: list[ContractIssue] = []
    for idx in frame.index[mask].tolist():
        issues.append(
            ContractIssue(
                severity="blocker",
                table=table,
                check=check,
                message=message,
                row=int(idx) if isinstance(idx, int) else None,
                column=left,
                value=str(frame.loc[idx, left]),
            )
        )
    return issues


__all__ = [
    "PITSchemaReport",
    "validate_feature_schema",
    "validate_label_schema",
    "validate_pit_schema",
]
