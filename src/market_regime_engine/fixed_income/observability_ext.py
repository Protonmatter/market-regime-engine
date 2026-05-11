# SPDX-License-Identifier: Apache-2.0
"""PR-7 §E.3 — Pre-registered FI counters / histograms for OTel + legacy.

Per plan §7 §4.1 / AGENT.md PR-7 §"Observability": the v1.5 dashboard
contract requires the following metric names to exist at FI worker
boot, even before any signal has been computed, so a fresh deployment
shows the canonical zero-baseline rather than 404 / "metric not
found":

Counters
--------
- ``fi_credit_regime_score_total``
- ``fi_liquidity_stress_score_total``
- ``fi_execution_confidence_request_total{recommended_action}``
- ``fi_release_gate_blocks_total{reason}``
- ``fi_evidence_pack_verify_fail_total``
- ``fi_tca_dropped_rows_total{metric}``
- ``fi_hmac_signature_failures_total``

Histograms
----------
- ``fi_execution_confidence_latency_seconds``
- ``fi_credit_regime_score_latency_seconds``
- ``fi_liquidity_stress_score_latency_seconds``
- ``fi_tca_aggregation_latency_seconds``

This module exposes thin helpers (``incr_*`` / ``record_*``) so the FI
scorers do not have to import ``observability.incr`` / ``record_histogram``
directly; production callers stay one import away from the OTel-or-
legacy adapter while keeping the call sites readable.
"""

from __future__ import annotations

from typing import Any

from market_regime_engine.observability import (
    incr,
    metrics,
    record_histogram,
)

# ---------------------------------------------------------------------------
# Counter / histogram name constants (single source of truth)
# ---------------------------------------------------------------------------

COUNTER_CREDIT_REGIME_SCORE = "fi_credit_regime_score_total"
COUNTER_LIQUIDITY_STRESS_SCORE = "fi_liquidity_stress_score_total"
COUNTER_EXECUTION_CONFIDENCE_REQUEST = "fi_execution_confidence_request_total"
COUNTER_RELEASE_GATE_BLOCKS = "fi_release_gate_blocks_total"
COUNTER_EVIDENCE_PACK_VERIFY_FAIL = "fi_evidence_pack_verify_fail_total"
COUNTER_TCA_DROPPED_ROWS = "fi_tca_dropped_rows_total"
COUNTER_HMAC_SIGNATURE_FAILURES = "fi_hmac_signature_failures_total"

HIST_EXECUTION_CONFIDENCE_LATENCY = "fi_execution_confidence_latency_seconds"
HIST_CREDIT_REGIME_SCORE_LATENCY = "fi_credit_regime_score_latency_seconds"
HIST_LIQUIDITY_STRESS_SCORE_LATENCY = "fi_liquidity_stress_score_latency_seconds"
HIST_TCA_AGGREGATION_LATENCY = "fi_tca_aggregation_latency_seconds"

FI_COUNTER_NAMES: tuple[str, ...] = (
    COUNTER_CREDIT_REGIME_SCORE,
    COUNTER_LIQUIDITY_STRESS_SCORE,
    COUNTER_EXECUTION_CONFIDENCE_REQUEST,
    COUNTER_RELEASE_GATE_BLOCKS,
    COUNTER_EVIDENCE_PACK_VERIFY_FAIL,
    COUNTER_TCA_DROPPED_ROWS,
    COUNTER_HMAC_SIGNATURE_FAILURES,
)

FI_HISTOGRAM_NAMES: tuple[str, ...] = (
    HIST_EXECUTION_CONFIDENCE_LATENCY,
    HIST_CREDIT_REGIME_SCORE_LATENCY,
    HIST_LIQUIDITY_STRESS_SCORE_LATENCY,
    HIST_TCA_AGGREGATION_LATENCY,
)


# ---------------------------------------------------------------------------
# Pre-registration at module load
# ---------------------------------------------------------------------------


def _pre_register() -> None:
    """Touch every counter / histogram so it shows up in the registry.

    A 0-value increment on a fresh ``MetricsRegistry`` is sufficient to
    make the metric appear in :func:`prometheus_text` even before any
    real call site exercises it. For OTel-mode this also primes the
    instrument cache so the first real emit doesn't pay the
    ``create_counter`` round-trip.
    """
    for name in FI_COUNTER_NAMES:
        incr(name, 0.0)
    for name in FI_HISTOGRAM_NAMES:
        # Recording 0.0 is the minimal initialisation that makes the
        # metric appear; the BoundedHistogram tolerates the value.
        record_histogram(name, 0.0)


_pre_register()


# ---------------------------------------------------------------------------
# Public emit helpers
# ---------------------------------------------------------------------------


def incr_credit_regime_score(**labels: str) -> None:
    incr(COUNTER_CREDIT_REGIME_SCORE, 1.0, **labels)


def incr_liquidity_stress_score(**labels: str) -> None:
    incr(COUNTER_LIQUIDITY_STRESS_SCORE, 1.0, **labels)


def incr_execution_confidence_request(*, recommended_action: str) -> None:
    incr(
        COUNTER_EXECUTION_CONFIDENCE_REQUEST,
        1.0,
        recommended_action=str(recommended_action),
    )


def incr_release_gate_block(*, reason: str) -> None:
    incr(COUNTER_RELEASE_GATE_BLOCKS, 1.0, reason=str(reason))


def incr_evidence_pack_verify_fail(**labels: str) -> None:
    incr(COUNTER_EVIDENCE_PACK_VERIFY_FAIL, 1.0, **labels)


def incr_tca_dropped_rows(*, metric: str, count: float = 1.0) -> None:
    incr(COUNTER_TCA_DROPPED_ROWS, float(count), metric=str(metric))


def incr_hmac_signature_failures(**labels: str) -> None:
    incr(COUNTER_HMAC_SIGNATURE_FAILURES, 1.0, **labels)


def record_credit_regime_latency(seconds: float, **labels: str) -> None:
    record_histogram(HIST_CREDIT_REGIME_SCORE_LATENCY, float(seconds), **labels)


def record_liquidity_stress_latency(seconds: float, **labels: str) -> None:
    record_histogram(HIST_LIQUIDITY_STRESS_SCORE_LATENCY, float(seconds), **labels)


def record_execution_confidence_latency(seconds: float, **labels: str) -> None:
    record_histogram(HIST_EXECUTION_CONFIDENCE_LATENCY, float(seconds), **labels)


def record_tca_aggregation_latency(seconds: float, **labels: str) -> None:
    record_histogram(HIST_TCA_AGGREGATION_LATENCY, float(seconds), **labels)


def fi_metric_snapshot() -> dict[str, Any]:
    """Return a snapshot of FI counters + histogram counts.

    Useful for the Streamlit dashboard tab + acceptance tests so a
    fresh deployment exposes the same canonical metric names without
    shaping the legacy ``MetricsRegistry`` snapshot manually.
    """
    snap = metrics().snapshot()
    fi_counters: dict[str, float] = {}
    fi_histograms: dict[str, dict[str, float]] = {}
    for key, value in snap["counters"].items():
        base = key.split("{", 1)[0]
        if base in FI_COUNTER_NAMES:
            fi_counters[key] = float(value)
    for key, stats in snap["histograms"].items():
        base = key.split("{", 1)[0]
        if base in FI_HISTOGRAM_NAMES:
            fi_histograms[key] = dict(stats)
    return {"counters": fi_counters, "histograms": fi_histograms}


__all__ = [
    "COUNTER_CREDIT_REGIME_SCORE",
    "COUNTER_EVIDENCE_PACK_VERIFY_FAIL",
    "COUNTER_EXECUTION_CONFIDENCE_REQUEST",
    "COUNTER_HMAC_SIGNATURE_FAILURES",
    "COUNTER_LIQUIDITY_STRESS_SCORE",
    "COUNTER_RELEASE_GATE_BLOCKS",
    "COUNTER_TCA_DROPPED_ROWS",
    "FI_COUNTER_NAMES",
    "FI_HISTOGRAM_NAMES",
    "HIST_CREDIT_REGIME_SCORE_LATENCY",
    "HIST_EXECUTION_CONFIDENCE_LATENCY",
    "HIST_LIQUIDITY_STRESS_SCORE_LATENCY",
    "HIST_TCA_AGGREGATION_LATENCY",
    "fi_metric_snapshot",
    "incr_credit_regime_score",
    "incr_evidence_pack_verify_fail",
    "incr_execution_confidence_request",
    "incr_hmac_signature_failures",
    "incr_liquidity_stress_score",
    "incr_release_gate_block",
    "incr_tca_dropped_rows",
    "record_credit_regime_latency",
    "record_execution_confidence_latency",
    "record_liquidity_stress_latency",
    "record_tca_aggregation_latency",
]
