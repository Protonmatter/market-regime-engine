# SPDX-License-Identifier: Apache-2.0
"""TRACE ingest contract + bulk loader (PR-7 §K.2)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from market_regime_engine.fixed_income.ingest.contract import (
    IngestContract,
    IngestReport,
)

__all__ = ["TRACE_CONTRACT", "ingest_trace"]


TRACE_CONTRACT = IngestContract(
    vendor_name="TRACE",
    required_columns=(
        "timestamp",
        "cusip",
        "price",
        "size",
        "side",
        "trade_id",
    ),
    optional_columns=(
        "yield_pct",
        "spread_bps",
        "source_snapshot_id",
        "venue",
        "protocol",
        "source",
        "reported_at",
        "metadata_json",
    ),
    notional_bounds=(0.0, 500_000_000.0),
    timestamp_monotonic=True,
)


def ingest_trace(
    warehouse: Any,
    df: pd.DataFrame,
    *,
    strict_unknown: bool = False,
) -> IngestReport:
    """Validate + bulk-load TRACE trades into the ``trace_trades`` table.

    Per REVIEW.md §3.6 PR-9 the contract WARN-logs any unknown columns
    by default and only raises if ``strict_unknown=True``.

    Returns the :class:`IngestReport` so callers can surface
    drop-counts to dashboards / CI.
    """
    TRACE_CONTRACT.assert_no_unknown_columns(
        df, level="error" if strict_unknown else "warn"
    )
    out_df, report = TRACE_CONTRACT.validate(df, strict_unknown=strict_unknown)
    if report.passed and not out_df.empty:
        # Map TRACE 'size' to the warehouse 'size' (already aligned).
        write_df = out_df.copy()
        if "metadata_json" not in write_df.columns:
            write_df["metadata_json"] = "{}"
        for opt in ("yield_pct", "venue", "protocol", "source", "reported_at"):
            if opt not in write_df.columns:
                write_df[opt] = None
        warehouse.write_trace_trades(write_df)
    return report
