# SPDX-License-Identifier: Apache-2.0
"""PR-6 §C — outcome-observation-lag guard + label-construction rail.

Per ``REVIEW.md §3.4 Q-2`` and ``REVIEW.md §3.6 PR-10``: every FI label
that depends on an observed-after-decision outcome must enforce a
strict inequality ``outcome.observed_at > request.timestamp``. A weak
inequality (``>=``) silently admits clock-drift leaks where a trade
fill reports at the same nanosecond as the decision and the label
construction picks up the future-info row.

This module is the single source of truth for the guard. Both the
storage writer (``Warehouse.write_execution_outcome`` — PR-5) and the
TCA segmentation aggregator (``tca_segmentation.compute_tca_metrics_for_outcome``
— PR-6 task A) route through it, so a future label
(``post_trade_directional_pnl_1d``, etc.) only has to call
:func:`assert_outcome_after_decision` to inherit the same rail.

Public surface:

- :func:`assert_outcome_after_decision` — raises
  :class:`PitViolationError` on a weak-or-reverse outcome lag.
- :func:`compute_execution_success_label` — the canonical binary
  execution-success label (slippage within threshold) with strict-lag
  + missing-data semantics documented in INSTRUCTIONS.md §6.4.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any

import pandas as pd

from market_regime_engine.fixed_income.bps_precision import to_bps, to_decimal
from market_regime_engine.fixed_income.pit_guard import PitViolationError
from market_regime_engine.fixed_income.schemas import ExecutionConfidenceRequest

log = logging.getLogger(__name__)


__all__ = [
    "EXECUTION_SUCCESS_DEFAULT_THRESHOLD_BPS",
    "SUCCESS_THRESHOLD_ENV",
    "assert_outcome_after_decision",
    "compute_execution_success_label",
]


EXECUTION_SUCCESS_DEFAULT_THRESHOLD_BPS: float = 25.0
"""Default slippage threshold (bps) for execution_success.

A trade counts as a successful execution when the absolute
arrival-price slippage is *strictly less than* ``threshold_bps``.
25 bps is the INSTRUCTIONS.md §6.4 default. Override via the explicit
``success_threshold_bps`` kwarg or the
``MRE_FI_TCA_SUCCESS_THRESHOLD_BPS`` environment variable.
"""

SUCCESS_THRESHOLD_ENV: str = "MRE_FI_TCA_SUCCESS_THRESHOLD_BPS"


def _coerce_utc(ts: Any) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        return out.tz_localize("UTC")
    return out.tz_convert("UTC")


def assert_outcome_after_decision(
    *,
    decision_timestamp: pd.Timestamp | str,
    observed_at: pd.Timestamp | str,
    label: str = "outcome",
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Raise :class:`PitViolationError` unless ``observed_at > decision_timestamp``.

    The check is *strict*: ``observed_at == decision_timestamp`` is a
    PIT violation (REVIEW.md §3.6 PR-10 — a same-nanosecond report is
    a clock-drift artefact, not a real outcome). The returned tuple
    has the two timestamps normalised to UTC so downstream callers
    don't need to re-coerce.
    """
    decision_utc = _coerce_utc(decision_timestamp)
    observed_utc = _coerce_utc(observed_at)
    if observed_utc <= decision_utc:
        raise PitViolationError(
            f"PIT violation for {label}: outcome.observed_at "
            f"({observed_utc.isoformat()}) must be strictly greater than "
            f"decision.timestamp ({decision_utc.isoformat()})"
        )
    return decision_utc, observed_utc


def _resolve_success_threshold(default: float) -> float:
    raw = os.getenv(SUCCESS_THRESHOLD_ENV, "").strip()
    if not raw:
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        log.warning(
            "invalid %s=%r; falling back to default %s",
            SUCCESS_THRESHOLD_ENV,
            raw,
            default,
        )
        return default


def compute_execution_success_label(
    request: ExecutionConfidenceRequest,
    outcome: Mapping[str, Any],
    *,
    success_threshold_bps: float | None = None,
) -> bool | None:
    """Binary execution-success label per PR-10 / INSTRUCTIONS.md §6.4.

    Returns:

    - ``True`` when ``|execution_price - arrival_price| / arrival_price * 10_000
      < success_threshold_bps`` (strict).
    - ``False`` when the slippage exceeds the threshold.
    - ``None`` when the outcome is not yet observable (``observed_at``
      absent) OR when the arrival/execution price is absent.

    Raises :class:`PitViolationError` when ``observed_at`` is present
    but does not satisfy the strict ``observed_at > request.timestamp``
    inequality (PR-10).
    """
    decision_ts_raw = request.timestamp
    observed_at_raw = outcome.get("observed_at")
    if observed_at_raw is None:
        return None
    # Strict inequality rail — raises before any label arithmetic.
    assert_outcome_after_decision(
        decision_timestamp=decision_ts_raw,
        observed_at=observed_at_raw,
        label="execution_success",
    )

    arrival = outcome.get("arrival_price")
    execution = outcome.get("execution_price")
    if arrival is None or execution is None:
        return None

    threshold = (
        float(success_threshold_bps)
        if success_threshold_bps is not None
        else _resolve_success_threshold(EXECUTION_SUCCESS_DEFAULT_THRESHOLD_BPS)
    )

    try:
        bps = abs(
            to_bps(to_decimal(execution) - to_decimal(arrival), to_decimal(arrival))
        )
    except ZeroDivisionError:
        return None
    return bps < to_decimal(threshold)
