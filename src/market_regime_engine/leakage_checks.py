# SPDX-License-Identifier: Apache-2.0
"""Point-in-time leakage checks.

A backtest that consumes future features or unavailable labels is not a model
validation exercise. It is a spreadsheet séance. This module makes those ghosts
show up as blocker findings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from market_regime_engine.data_contracts import ContractIssue, read_table
from market_regime_engine.pit_schema import PITSchemaReport, validate_pit_schema


@dataclass(frozen=True)
class LeakageAuditReport:
    """Combined PIT schema + leakage report."""

    schema: PITSchemaReport
    issues: tuple[ContractIssue, ...]
    feature_rows: int
    label_rows: int
    matched_pairs: int

    @property
    def passed(self) -> bool:
        return not any(issue.severity == "blocker" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "feature_rows": self.feature_rows,
            "label_rows": self.label_rows,
            "matched_pairs": self.matched_pairs,
            "schema": self.schema.to_dict(),
            "issues": [issue.to_dict() for issue in self.issues],
            "summary": self.summary(),
        }

    def summary(self) -> dict[str, Any]:
        by_check: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        for issue in self.issues:
            by_check[issue.check] = by_check.get(issue.check, 0) + 1
            by_severity[issue.severity] = by_severity.get(issue.severity, 0) + 1
        return {
            "issue_count": len(self.issues),
            "blockers": by_severity.get("blocker", 0),
            "by_check": by_check,
            "by_severity": by_severity,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)

    def to_markdown(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"# Point-in-Time Leakage Audit — {status}",
            "",
            "## Summary",
            "",
            f"- **feature_rows:** {self.feature_rows}",
            f"- **label_rows:** {self.label_rows}",
            f"- **matched_pairs:** {self.matched_pairs}",
            f"- **issue_count:** {len(self.issues)}",
            f"- **blockers:** {self.summary()['blockers']}",
            "",
            "## Issues",
            "",
        ]
        if not self.issues:
            lines.append("_No PIT leakage issues detected._")
        else:
            lines.append("| Severity | Table | Check | Row | Column | Message | Value |")
            lines.append("|---|---|---|---:|---|---|---|")
            for issue in self.issues[:200]:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(issue.severity),
                            str(issue.table),
                            str(issue.check),
                            "" if issue.row is None else str(issue.row),
                            "" if issue.column is None else str(issue.column),
                            str(issue.message).replace("|", "\\|"),
                            "" if issue.value is None else str(issue.value),
                        ]
                    )
                    + " |"
                )
            if len(self.issues) > 200:
                lines.append(f"\n_Trimmed: showing 200 of {len(self.issues)} issues._")
        return "\n".join(lines).rstrip() + "\n"


def audit_pit_paths(features: str | Path, labels: str | Path) -> LeakageAuditReport:
    """Read feature/label tables and run the full PIT leakage audit."""

    return audit_pit_frames(read_table(features), read_table(labels))


def audit_pit_frames(features: pd.DataFrame, labels: pd.DataFrame) -> LeakageAuditReport:
    """Run schema and cross-table point-in-time leakage checks."""

    fframe, lframe, schema = validate_pit_schema(features, labels)
    issues: list[ContractIssue] = [*schema.feature_report.issues, *schema.label_report.issues]
    matched_pairs = 0

    if not schema.feature_report.missing_columns and not schema.label_report.missing_columns:
        cross_issues, matched_pairs = _cross_table_issues(fframe, lframe)
        issues.extend(cross_issues)

    return LeakageAuditReport(
        schema=schema,
        issues=tuple(issues),
        feature_rows=len(features),
        label_rows=len(labels),
        matched_pairs=int(matched_pairs),
    )


def _cross_table_issues(features: pd.DataFrame, labels: pd.DataFrame) -> tuple[list[ContractIssue], int]:
    join_keys = ["entity_id", "forecast_origin"]
    if not all(k in features.columns for k in join_keys) or not all(k in labels.columns for k in join_keys):
        return [], 0

    merged = features.reset_index(names="_feature_row").merge(
        labels.reset_index(names="_label_row"),
        on=join_keys,
        how="inner",
        suffixes=("_feature", "_label"),
    )
    issues: list[ContractIssue] = []
    if merged.empty:
        return issues, 0

    feature_as_of_col = "as_of_feature" if "as_of_feature" in merged.columns else "as_of"
    if feature_as_of_col in merged.columns:
        mask = (
            merged[feature_as_of_col].notna()
            & merged["forecast_origin"].notna()
            & (merged[feature_as_of_col] > merged["forecast_origin"])
        )
        for _, row in merged[mask].iterrows():
            issues.append(
                ContractIssue(
                    severity="blocker",
                    table="features+labels",
                    check="feature_as_of_lte_label_forecast_origin",
                    message="feature.as_of must be <= label.forecast_origin for joined training rows",
                    row=_safe_int(row.get("_feature_row")),
                    column="as_of",
                    value=str(row.get(feature_as_of_col)),
                )
            )

    for joined_col in ("joined_at", "label_joined_at", "as_of_label"):
        if joined_col not in merged.columns or "label_available_at" not in merged.columns:
            continue
        mask = (
            merged["label_available_at"].notna()
            & merged[joined_col].notna()
            & (merged["label_available_at"] > merged[joined_col])
        )
        for _, row in merged[mask].iterrows():
            issues.append(
                ContractIssue(
                    severity="blocker",
                    table="labels",
                    check="label_joined_before_available",
                    message=f"label_available_at must be <= {joined_col}",
                    row=_safe_int(row.get("_label_row")),
                    column="label_available_at",
                    value=str(row.get("label_available_at")),
                )
            )

    revision_col = _merged_revision_available_col(merged)
    if revision_col and feature_as_of_col in merged.columns:
        mask = (
            merged[revision_col].notna()
            & merged[feature_as_of_col].notna()
            & (merged[revision_col] > merged[feature_as_of_col])
        )
        for _, row in merged[mask].iterrows():
            issues.append(
                ContractIssue(
                    severity="blocker",
                    table="features",
                    check="vintage_revision_used_before_available",
                    message=f"{revision_col} must be <= feature.as_of",
                    row=_safe_int(row.get("_feature_row")),
                    column=revision_col.replace("_feature", ""),
                    value=str(row.get(revision_col)),
                )
            )

    return issues, len(merged)


def _merged_revision_available_col(frame: pd.DataFrame) -> str | None:
    for col in (
        "source_revision_available_at_feature",
        "revision_available_at_feature",
        "source_revision_available_at",
        "revision_available_at",
    ):
        if col in frame.columns:
            return col
    return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


__all__ = ["LeakageAuditReport", "audit_pit_frames", "audit_pit_paths"]
