# SPDX-License-Identifier: Apache-2.0
"""Per-vendor :class:`IngestContract` for FI feeds (PR-7 §K).

Contract surface per plan §7 §K + AGENT.md PR-7 §"Ingestion contracts"
+ REVIEW.md §3.6 PR-9 (schema-drift assertion). The contract is
per-vendor, immutable, and validates a :class:`pd.DataFrame` against:

- Required columns (every column must be present; missing → error).
- Optional columns (informational only; absence does not fail
  validation).
- Per-column callable validators that reject individual rows.
- Notional bounds (lo, hi) — rows outside the band are dropped with a
  ``notional_out_of_bounds`` error report entry.
- Monotonic-timestamp policy — when ``True`` (the default), reject
  feeds whose ``timestamp`` column is not non-decreasing within the
  same key.
- Schema-drift policy via :meth:`assert_no_unknown_columns` — WARN by
  default (passes), promotable to "error" for strict deployments.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

log = logging.getLogger(__name__)

__all__ = ["IngestContract", "IngestReport"]


@dataclass(frozen=True)
class IngestReport:
    """Validation outcome for a vendor DataFrame."""

    passed: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    dropped_count: int = 0
    rows_in: int = 0
    rows_out: int = 0
    vendor_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "dropped_count": int(self.dropped_count),
            "rows_in": int(self.rows_in),
            "rows_out": int(self.rows_out),
            "vendor_name": str(self.vendor_name),
        }


@dataclass(frozen=True)
class IngestContract:
    """Immutable per-vendor ingestion validator.

    Attributes
    ----------
    vendor_name
        Operator-friendly vendor identifier; used in log messages and
        :class:`IngestReport.vendor_name`.
    required_columns
        Tuple of column names that MUST be present.
    optional_columns
        Tuple of column names allowed but not required. Other columns
        trigger :meth:`assert_no_unknown_columns`.
    column_validators
        Mapping of ``column → callable`` returning ``True`` for valid
        cell values. Rows where any validator returns ``False`` are
        dropped.
    notional_bounds
        ``(min, max)`` tuple in the natural notional unit of the
        vendor; rows outside the band are dropped. ``None`` disables
        the check.
    timestamp_monotonic
        When ``True``, reject the feed if the ``timestamp`` column is
        not non-decreasing.
    timestamp_column
        Name of the timestamp column; defaults to ``"timestamp"``.
    """

    vendor_name: str
    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...] = ()
    column_validators: Mapping[str, Callable[[Any], bool]] = field(default_factory=dict)
    notional_bounds: tuple[float, float] | None = None
    timestamp_monotonic: bool = True
    timestamp_column: str = "timestamp"

    @property
    def known_columns(self) -> frozenset[str]:
        return frozenset(self.required_columns) | frozenset(self.optional_columns)

    def validate(
        self,
        df: pd.DataFrame,
        *,
        strict_unknown: bool = False,
    ) -> tuple[pd.DataFrame, IngestReport]:
        """Validate ``df`` against the contract.

        Returns ``(filtered_df, report)`` where ``filtered_df`` has
        rows that failed the validators dropped. Raises
        :class:`ValueError` only when a required column is missing or
        ``strict_unknown=True`` triggers an unknown-column error.
        """
        if df is None:
            df = pd.DataFrame()
        rows_in = int(len(df))
        errors: list[str] = []
        warnings: list[str] = []
        dropped_count = 0

        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            errors.append(f"missing required columns: {sorted(missing)!r}")
            return df.iloc[0:0], IngestReport(
                passed=False,
                errors=tuple(errors),
                warnings=(),
                dropped_count=int(rows_in),
                rows_in=rows_in,
                rows_out=0,
                vendor_name=self.vendor_name,
            )

        unknown = [c for c in df.columns if c not in self.known_columns]
        if unknown:
            msg = f"unknown columns ignored: {sorted(unknown)!r}"
            if strict_unknown:
                errors.append(msg)
                return df.iloc[0:0], IngestReport(
                    passed=False,
                    errors=tuple(errors),
                    warnings=(),
                    dropped_count=int(rows_in),
                    rows_in=rows_in,
                    rows_out=0,
                    vendor_name=self.vendor_name,
                )
            warnings.append(msg)

        # Timestamp monotonicity check (against the entire frame).
        if (
            self.timestamp_monotonic
            and self.timestamp_column in df.columns
            and rows_in > 1
        ):
            ts = pd.to_datetime(df[self.timestamp_column], errors="coerce")
            if ts.is_monotonic_increasing is False and (
                ts.dropna().diff().dropna() < pd.Timedelta(0)
            ).any():
                errors.append("timestamp column is not monotonic non-decreasing")
                return df.iloc[0:0], IngestReport(
                    passed=False,
                    errors=tuple(errors),
                    warnings=tuple(warnings),
                    dropped_count=int(rows_in),
                    rows_in=rows_in,
                    rows_out=0,
                    vendor_name=self.vendor_name,
                )

        mask = pd.Series(True, index=df.index)

        # Column-level validators (drop offending rows).
        for column, validator in self.column_validators.items():
            if column not in df.columns:
                continue
            col_mask = df[column].map(validator).astype(bool)
            invalid = (~col_mask).sum()
            if invalid:
                warnings.append(
                    f"column {column!r}: {int(invalid)} rows failed validator"
                )
            mask &= col_mask

        # Notional bounds (drop offending rows).
        if self.notional_bounds is not None and "notional" in df.columns:
            lo, hi = self.notional_bounds
            within = (df["notional"] >= float(lo)) & (df["notional"] <= float(hi))
            invalid = (~within).sum()
            if invalid:
                warnings.append(
                    f"notional out of bounds [{lo}, {hi}]: {int(invalid)} rows dropped"
                )
            mask &= within

        out_df = df[mask].copy()
        dropped_count = int(rows_in - len(out_df))
        report = IngestReport(
            passed=len(errors) == 0,
            errors=tuple(errors),
            warnings=tuple(warnings),
            dropped_count=dropped_count,
            rows_in=rows_in,
            rows_out=int(len(out_df)),
            vendor_name=self.vendor_name,
        )
        return out_df, report

    def assert_no_unknown_columns(
        self,
        df: pd.DataFrame,
        *,
        level: Literal["error", "warn"] = "warn",
    ) -> None:
        """Assert no unknown columns; level=warn (default) or error.

        Per REVIEW.md §3.6 PR-9: a new vendor column should not silently
        slip through. WARN mode emits a logger warning so production
        operators see the drift; strict deployments pass
        ``level="error"`` to raise.
        """
        if df is None:
            return
        unknown = [c for c in df.columns if c not in self.known_columns]
        if not unknown:
            return
        msg = (
            f"vendor={self.vendor_name!r}: unknown columns {sorted(unknown)!r} "
            f"(known: {sorted(self.known_columns)!r})"
        )
        if level == "warn":
            log.warning("%s", msg)
            return
        raise ValueError(msg)
