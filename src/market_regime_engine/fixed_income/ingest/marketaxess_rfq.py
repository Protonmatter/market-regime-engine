# SPDX-License-Identifier: Apache-2.0
"""MarketAxess RFQ ingest contract + bulk loader (PR-7 §K.2)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from market_regime_engine.fixed_income.ingest.contract import (
    IngestContract,
    IngestReport,
)

__all__ = ["MARKETAXESS_RFQ_CONTRACT", "ingest_marketaxess_rfq"]


_VALID_STATUS = {"open", "filled", "cancelled", "expired"}
_VALID_SIDE = {"buy", "sell"}


MARKETAXESS_RFQ_CONTRACT = IngestContract(
    vendor_name="MarketAxess RFQ",
    required_columns=(
        "rfq_id",
        "timestamp",
        "cusip",
        "side",
        "notional",
        "protocol",
        "dealers_requested",
        "quotes_received",
        "status",
    ),
    optional_columns=(
        "best_bid",
        "best_ask",
        "mid_price",
        "execution_price",
        "filled_quantity",
        "client_id",
        "metadata_json",
        "time_to_first_response_ms",
        "dealers_responded",
    ),
    column_validators={
        "status": lambda v: str(v) in _VALID_STATUS,
        "side": lambda v: str(v) in _VALID_SIDE,
    },
    notional_bounds=(0.0, 500_000_000.0),
    timestamp_monotonic=False,
)


def ingest_marketaxess_rfq(
    warehouse: Any,
    df: pd.DataFrame,
    *,
    strict_unknown: bool = False,
) -> IngestReport:
    """Validate + bulk-load MarketAxess RFQ events into ``rfq_events``.

    Returns the :class:`IngestReport` so callers can react to drop
    counts / warnings (REVIEW.md §3.6 PR-9 schema-drift assertion).
    """
    MARKETAXESS_RFQ_CONTRACT.assert_no_unknown_columns(df, level="error" if strict_unknown else "warn")
    out_df, report = MARKETAXESS_RFQ_CONTRACT.validate(df, strict_unknown=strict_unknown)
    if report.passed and not out_df.empty:
        write_df = out_df.copy()
        if "metadata_json" not in write_df.columns:
            write_df["metadata_json"] = "{}"
        # Map quotes_received → dealers_responded for the warehouse
        # column convention; if the operator already provided
        # dealers_responded, keep it.
        if "dealers_responded" not in write_df.columns:
            write_df["dealers_responded"] = write_df.get("quotes_received")
        if "time_to_first_response_ms" not in write_df.columns:
            write_df["time_to_first_response_ms"] = None
        if "client_id" not in write_df.columns:
            write_df["client_id"] = None
        warehouse.write_rfq_events(write_df)
    return report
