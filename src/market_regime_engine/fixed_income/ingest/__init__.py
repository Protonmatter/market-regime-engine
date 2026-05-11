# SPDX-License-Identifier: Apache-2.0
"""Per-vendor FI ingest contracts (PR-7 §K).

Per plan §7 §K + REVIEW.md §3.6 PR-9 (schema-drift assertion): every
vendor feed gets an :class:`IngestContract` with required / optional
columns, optional column validators, monotonic-timestamp policy, and
notional-bound checks. ``assert_no_unknown_columns`` defaults to WARN
mode so a new vendor column does not break ingestion immediately;
strict deployments can promote the warning to an error.

PR-7 ships the contract surface plus two scaffold contracts (TRACE
and MarketAxess RFQ); full vendor implementations are v1.5.1+.
"""

from market_regime_engine.fixed_income.ingest.contract import (
    IngestContract,
    IngestReport,
)
from market_regime_engine.fixed_income.ingest.marketaxess_rfq import (
    MARKETAXESS_RFQ_CONTRACT,
    ingest_marketaxess_rfq,
)
from market_regime_engine.fixed_income.ingest.trace import (
    TRACE_CONTRACT,
    ingest_trace,
)

__all__ = [
    "MARKETAXESS_RFQ_CONTRACT",
    "TRACE_CONTRACT",
    "IngestContract",
    "IngestReport",
    "ingest_marketaxess_rfq",
    "ingest_trace",
]
