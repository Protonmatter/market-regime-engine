# SPDX-License-Identifier: Apache-2.0
"""PR-5 execution-confidence scorer (deterministic logistic baseline).

Per ``MRE_FIXED_INCOME_AGENT.md §"PR 5"`` and
``MRE_FIXED_INCOME_INSTRUCTIONS.md §6.3``: ship the explainable
deterministic baseline first; model-based scorers (calibrated logistic
on historical execution_outcomes, gradient-boosted alternatives) land in
v1.5.1 behind validation. Every output carries the v1.5 governance triple
(``model_run_id``, ``release_gate``, ``artifact_hash``) and embeds
``signal_age_seconds`` for the credit-regime + liquidity signals so
operators can detect stale-signal degradation.

Decision rule (INSTRUCTIONS.md §6.3)::

    if release_gate is False:
        recommended_action = MANUAL_REVIEW_REQUIRED
        human_review_required = True
    elif confidence_score >= 0.80 and liquidity_label NOT IN
            {"Severe Stress", "Crisis Liquidity"}:
        recommended_action = AUTO_X_ALLOWED
    elif confidence_score >= 0.60:
        recommended_action = AUTO_X_CAUTION
    else:
        recommended_action = MANUAL_REVIEW_REQUIRED

Logit components::

    base_intercept       = +0.5      (50% prior)
    liquidity_penalty    = -0.01 * liquidity_index
    notional_penalty     = -0.15 * max(0, log10(notional) - 6)
    regime_penalty       = -0.008 * regime_score
    protocol_bonus       = {Auto-X: +0.10, RFQ: +0.05, Manual: -0.10}
    urgency_penalty      = {low: 0.0, normal: -0.05, high: -0.15}
    rating_bonus         = {IG: +0.10, HY: -0.10}
    limit_distance_penalty  ≈ -0.05 * max(0, |limit - mid| - 10)   (bps)

``confidence_score = sigmoid(sum(logit_components))`` clipped to
``[0.05, 0.95]``.

Expected slippage::

    expected_slippage_bps = 5.0 + 30.0 * (1 - confidence_score)
                           + 0.5 * liquidity_index
    floor 1 bps, ceiling 200 bps

CI: ``[confidence_score - 0.10, confidence_score + 0.10]`` clipped to
``[0, 1]``. v1.5.1 will swap this heuristic interval for calibrated
quantile output.

Stale-signal policy: when either credit-regime or liquidity feed is older
than ``MRE_FI_MAX_SIGNAL_STALENESS_SEC`` (default 900 = 15 minutes), the
scorer soft-fails with ``recommended_action=UNAVAILABLE_STALE_SIGNAL``
and ``release_gate=False`` rather than raising.
"""

from __future__ import annotations

import json
import logging
import math
import os
import uuid
from collections.abc import Mapping
from typing import Any

import pandas as pd

from market_regime_engine.fixed_income.bps_precision import (
    decimal_to_float_for_report,
    to_bps,
    to_decimal,
)
from market_regime_engine.fixed_income.credit_spread_regime import (
    latest_credit_regime_score,
)
from market_regime_engine.fixed_income.hashing import canonical_sha256
from market_regime_engine.fixed_income.liquidity_stress import (
    latest_liquidity_stress_score,
)
from market_regime_engine.fixed_income.pit_guard import (
    assert_pit_safe,
)
from market_regime_engine.fixed_income.schemas import (
    ExecutionConfidenceRequest,
    ExecutionConfidenceResponse,
    ExecutionRecommendation,
)
from market_regime_engine.fixed_income.timestamps import iso8601_z, to_utc

log = logging.getLogger(__name__)


__all__ = [
    "DEFAULT_LIMIT_TOLERANCE_BPS",
    "DEFAULT_WEIGHTS",
    "MAX_SIGNAL_STALENESS_ENV",
    "build_execution_features",
    "latest_execution_confidence_prediction",
    "score_execution_confidence",
    "write_execution_confidence_prediction",
    "write_execution_outcome",
]


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


DEFAULT_WEIGHTS: dict[str, float] = {
    "base_intercept": 0.5,
    "liquidity_coef": -0.01,
    "notional_coef": -0.15,
    "regime_coef": -0.008,
    "protocol_auto_x": 0.10,
    "protocol_rfq": 0.05,
    "protocol_manual": -0.10,
    "urgency_low": 0.0,
    "urgency_normal": -0.05,
    "urgency_high": -0.15,
    "rating_ig": 0.10,
    "rating_hy": -0.10,
    "limit_distance_coef": -0.05,
}

DEFAULT_LIMIT_TOLERANCE_BPS: float = 10.0
"""Limit-price tolerance (bps from mid) below which no penalty applies."""

MAX_SIGNAL_STALENESS_ENV: str = "MRE_FI_MAX_SIGNAL_STALENESS_SEC"
_DEFAULT_MAX_STALENESS_SEC: float = 900.0  # 15 minutes

_CONFIDENCE_FLOOR: float = 0.05
_CONFIDENCE_CEIL: float = 0.95
_EXPECTED_SLIPPAGE_FLOOR_BPS: float = 1.0
_EXPECTED_SLIPPAGE_CEIL_BPS: float = 200.0

_SEVERE_OR_CRISIS_LIQUIDITY: frozenset[str] = frozenset({"Severe Stress", "Crisis Liquidity"})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _resolve_max_staleness() -> float:
    raw = os.getenv(MAX_SIGNAL_STALENESS_ENV, "").strip()
    if not raw:
        return _DEFAULT_MAX_STALENESS_SEC
    try:
        return max(0.0, float(raw))
    except ValueError:
        log.warning(
            "invalid %s=%r; falling back to default %ss",
            MAX_SIGNAL_STALENESS_ENV,
            raw,
            _DEFAULT_MAX_STALENESS_SEC,
        )
        return _DEFAULT_MAX_STALENESS_SEC


def _coerce_decision_ts(timestamp: str | pd.Timestamp) -> pd.Timestamp:
    if isinstance(timestamp, pd.Timestamp):
        if timestamp.tzinfo is None:
            return timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC")
    ts = to_utc(timestamp)
    if ts is None:
        raise ValueError("request.timestamp must not be None")
    return ts


def _signal_age_seconds(signal_ts_iso: str | None, decision_ts: pd.Timestamp) -> float:
    if signal_ts_iso is None:
        return float("inf")
    signal_ts = pd.Timestamp(signal_ts_iso)
    if signal_ts.tzinfo is None:
        signal_ts = signal_ts.tz_localize("UTC")
    delta = (decision_ts - signal_ts).total_seconds()
    # Negative deltas (signal *after* decision) are PIT violations and
    # surface in the assert_pit_safe rail upstream; here we clamp to 0
    # so the metadata float never confuses downstream tooling.
    return float(max(0.0, delta))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _rating_class(rating: str | None) -> str | None:
    """Map a textual rating (``AAA``/``BB+``/etc.) to ``"IG"`` / ``"HY"``.

    Recognises the standard letter buckets — ``AAA``..``BBB`` are IG,
    ``BB``..``D`` are HY. Numeric "rating_numeric" inputs (1–22 scale,
    common in vendor data) are also accepted; 10 and below is IG.
    """
    if rating is None:
        return None
    raw = str(rating).strip().upper()
    if not raw:
        return None
    if raw in {"IG", "HY"}:
        return raw
    if raw[0].isdigit():
        try:
            num = float(raw)
        except ValueError:
            return None
        return "IG" if num <= 10.0 else "HY"
    # Strip modifiers (+/-).
    core = raw.replace("+", "").replace("-", "")
    if core.startswith(("AAA", "AA", "A")) and core[:3] != "BBB":
        # A and above are IG
        return "IG"
    if core.startswith("BBB"):
        return "IG"
    if core.startswith(("BB", "B", "CCC", "CC", "C", "D")):
        return "HY"
    return None


def _logit_components(
    *,
    request: ExecutionConfidenceRequest,
    liquidity_index: float,
    regime_score: float,
    rating_class: str | None,
    limit_distance_bps: float | None,
    weights: Mapping[str, float],
) -> dict[str, float]:
    components: dict[str, float] = {}
    components["base_intercept"] = float(weights["base_intercept"])
    components["liquidity_penalty"] = float(weights["liquidity_coef"]) * float(liquidity_index)
    components["regime_penalty"] = float(weights["regime_coef"]) * float(regime_score)
    notional_log10 = math.log10(max(float(request.notional), 1.0))
    components["notional_penalty"] = float(weights["notional_coef"]) * max(0.0, notional_log10 - 6.0)
    protocol = (request.protocol or "").strip()
    if protocol == "Auto-X":
        components["protocol_bonus"] = float(weights["protocol_auto_x"])
    elif protocol == "RFQ":
        components["protocol_bonus"] = float(weights["protocol_rfq"])
    elif protocol == "Manual":
        components["protocol_bonus"] = float(weights["protocol_manual"])
    else:
        components["protocol_bonus"] = 0.0
    urgency = (request.urgency or "normal").strip().lower()
    if urgency == "low":
        components["urgency_penalty"] = float(weights["urgency_low"])
    elif urgency == "high":
        components["urgency_penalty"] = float(weights["urgency_high"])
    else:
        components["urgency_penalty"] = float(weights["urgency_normal"])
    if rating_class == "IG":
        components["rating_bonus"] = float(weights["rating_ig"])
    elif rating_class == "HY":
        components["rating_bonus"] = float(weights["rating_hy"])
    else:
        components["rating_bonus"] = 0.0
    if limit_distance_bps is not None and limit_distance_bps > DEFAULT_LIMIT_TOLERANCE_BPS:
        components["limit_distance_penalty"] = float(weights["limit_distance_coef"]) * (
            float(limit_distance_bps) - DEFAULT_LIMIT_TOLERANCE_BPS
        )
    else:
        components["limit_distance_penalty"] = 0.0
    return components


def _drivers_from_components(components: Mapping[str, float]) -> tuple[str, ...]:
    """Top-3 logit components by absolute magnitude (excluding intercept)."""
    ranked = sorted(
        ((name, abs(value)) for name, value in components.items() if name != "base_intercept"),
        key=lambda kv: -kv[1],
    )
    return tuple(name for name, _ in ranked[:3])


def _expected_slippage_bps(confidence_score: float, liquidity_index: float) -> float:
    raw = 5.0 + 30.0 * (1.0 - float(confidence_score)) + 0.5 * float(liquidity_index)
    return float(max(_EXPECTED_SLIPPAGE_FLOOR_BPS, min(_EXPECTED_SLIPPAGE_CEIL_BPS, raw)))


def _confidence_interval(score: float) -> tuple[float, float]:
    low = max(0.0, float(score) - 0.10)
    high = min(1.0, float(score) + 0.10)
    return low, high


def _decision_rule(
    *,
    confidence_score: float,
    liquidity_label: str,
    release_gate: bool,
) -> tuple[ExecutionRecommendation, bool]:
    """Apply the INSTRUCTIONS.md §6.3 decision rule. Returns
    ``(recommended_action, human_review_required)``."""
    if not release_gate:
        return ExecutionRecommendation.MANUAL_REVIEW_REQUIRED, True
    if confidence_score >= 0.80 and liquidity_label not in _SEVERE_OR_CRISIS_LIQUIDITY:
        return ExecutionRecommendation.AUTO_X_ALLOWED, False
    if confidence_score >= 0.60:
        return ExecutionRecommendation.AUTO_X_CAUTION, False
    return ExecutionRecommendation.MANUAL_REVIEW_REQUIRED, True


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def score_execution_confidence(
    request: ExecutionConfidenceRequest,
    *,
    warehouse: Any,
    model_run_id: str | None = None,
    release_gate: bool = True,
    profile: str = "production",
    weights: Mapping[str, float] | None = None,
) -> ExecutionConfidenceResponse:
    """Score a single execution-confidence request.

    Reads the latest credit-regime + cusip-scoped liquidity-stress rows
    from the warehouse (falls back to the market-scope liquidity when no
    cusip-specific row exists), blends those signals with order
    attributes through a deterministic logistic baseline, and returns a
    governance-stamped :class:`ExecutionConfidenceResponse`.

    Parameters
    ----------
    request:
        Inbound order body. ``request.timestamp`` is the decision
        timestamp; the regime + liquidity signals are required to satisfy
        ``signal.timestamp <= request.timestamp``.
    warehouse:
        Storage facade with ``read_credit_regime_scores`` and
        ``read_liquidity_stress_scores`` methods (the v1.5 PR-2
        :class:`Warehouse`).
    model_run_id:
        Reproducibility id; profile-stamped UUID minted when omitted.
    release_gate:
        Inbound governance flag. ``False`` short-circuits the decision
        rule to MANUAL_REVIEW_REQUIRED + ``human_review_required=True``.
        Stale-signal detection independently flips ``release_gate=False``
        on the *output* even when the input was ``True``.
    profile:
        Operating profile tag for the metadata blob; downstream tooling
        differentiates production runs from dev runs.
    weights:
        Optional override of the logit weights; missing keys fall back to
        :data:`DEFAULT_WEIGHTS`.

    Returns
    -------
    ExecutionConfidenceResponse with ``confidence_score``,
    ``expected_slippage_bps``, ``confidence_interval_*``,
    ``recommended_action``, ``human_review_required``, ``model_run_id``,
    ``release_gate``, ``artifact_hash``, and metadata including the top-3
    drivers and ``signal_age_seconds_*`` keys.

    Raises
    ------
    PitViolationError
        If the regime or liquidity signal is post-decision.
    ValueError
        On naive / missing request.timestamp.
    """
    decision_ts = _coerce_decision_ts(request.timestamp)
    decision_iso = iso8601_z(decision_ts)
    merged_weights: dict[str, float] = dict(DEFAULT_WEIGHTS)
    if weights:
        merged_weights.update({k: float(v) for k, v in weights.items()})

    resolved_run_id = (
        model_run_id
        if model_run_id and model_run_id.strip()
        else f"execution_confidence-{profile}-{uuid.uuid4().hex[:12]}"
    )

    regime = latest_credit_regime_score(warehouse)
    liquidity = latest_liquidity_stress_score(warehouse, scope_type="cusip", scope_id=request.cusip)
    if liquidity is None:
        # Fallback per AGENT.md: cusip scope missing → use market scope.
        liquidity = latest_liquidity_stress_score(warehouse)

    max_staleness = _resolve_max_staleness()

    if regime is None or liquidity is None:
        log.warning(
            "execution_confidence: missing signal regime=%s liquidity=%s",
            regime is not None,
            liquidity is not None,
        )
        return _stale_response(
            request=request,
            decision_iso=decision_iso,
            resolved_run_id=resolved_run_id,
            profile=profile,
            regime=regime,
            liquidity=liquidity,
            decision_ts=decision_ts,
            max_staleness=max_staleness,
            release_gate_input=release_gate,
            reason="missing_signal",
        )

    # PIT rails — raise on lookahead leak.
    assert_pit_safe(
        feature_timestamp=regime.timestamp,
        decision_timestamp=decision_ts,
        label="credit_regime",
    )
    assert_pit_safe(
        feature_timestamp=liquidity.timestamp,
        decision_timestamp=decision_ts,
        label="liquidity_stress",
    )

    regime_age = _signal_age_seconds(regime.timestamp, decision_ts)
    liquidity_age = _signal_age_seconds(liquidity.timestamp, decision_ts)
    max_age = max(regime_age, liquidity_age)

    if max_age > max_staleness:
        return _stale_response(
            request=request,
            decision_iso=decision_iso,
            resolved_run_id=resolved_run_id,
            profile=profile,
            regime=regime,
            liquidity=liquidity,
            decision_ts=decision_ts,
            max_staleness=max_staleness,
            release_gate_input=release_gate,
            reason="stale_signal",
        )

    # Limit-distance in bps (None when the caller did not pass a limit).
    # v1.6.0 (REVIEW_DEEP_V1_5_2.md A5 / Finding §3.1): route through the
    # Decimal-precision bps_precision helpers instead of raw float math.
    # Per-request error is negligible (~1e-13 relative) but accumulates
    # over millions of evaluations into a coefficient drift on the
    # decision boundary; using Decimal eliminates that drift and brings
    # this call site into agreement with the rest of the FI TCA stack.
    limit_distance_bps: float | None = None
    if request.limit_price is not None:
        # The exec-confidence dataclass intentionally does not carry a
        # mid-market quote; the request body's ``metadata.mid_price`` is
        # the canonical caller-supplied reference (informational input
        # only — the deterministic baseline can score without it).
        mid_price = (
            request.metadata.get("mid_price")
            if isinstance(request.metadata, Mapping)
            else None
        )
        if mid_price is not None:
            try:
                mid_dec = to_decimal(mid_price)
                if mid_dec > 0:
                    price_diff = to_decimal(request.limit_price) - mid_dec
                    if price_diff < 0:
                        price_diff = -price_diff
                    bps_dec = to_bps(price_diff, mid_dec)
                    limit_distance_bps = decimal_to_float_for_report(bps_dec)
            except (TypeError, ValueError, ZeroDivisionError):
                limit_distance_bps = None

    rating_class = _rating_class(request.rating)
    components = _logit_components(
        request=request,
        liquidity_index=liquidity.liquidity_index,
        regime_score=regime.regime_score,
        rating_class=rating_class,
        limit_distance_bps=limit_distance_bps,
        weights=merged_weights,
    )

    logit = sum(components.values())
    confidence_score = max(_CONFIDENCE_FLOOR, min(_CONFIDENCE_CEIL, _sigmoid(logit)))
    expected_slippage_bps = _expected_slippage_bps(confidence_score, liquidity.liquidity_index)
    ci_low, ci_high = _confidence_interval(confidence_score)

    drivers = _drivers_from_components(components)

    recommended, human_review = _decision_rule(
        confidence_score=confidence_score,
        liquidity_label=liquidity.liquidity_label,
        release_gate=release_gate,
    )

    metadata: dict[str, Any] = {
        "profile": profile,
        "regime_score": float(regime.regime_score),
        "regime_label": regime.regime_label,
        "liquidity_index": float(liquidity.liquidity_index),
        "liquidity_label": liquidity.liquidity_label,
        "liquidity_scope_type": liquidity.scope_type,
        "liquidity_scope_id": liquidity.scope_id,
        "drivers": list(drivers),
        "logit_components": {k: float(v) for k, v in components.items()},
        "rating_class": rating_class,
        "limit_distance_bps": (float(limit_distance_bps) if limit_distance_bps is not None else None),
        "signal_age_seconds_credit_regime": float(regime_age),
        "signal_age_seconds_liquidity": float(liquidity_age),
        "max_signal_age_seconds": float(max_age),
        "max_signal_staleness_threshold_seconds": float(max_staleness),
        "release_gate_input": bool(release_gate),
        "reason": "scored" if release_gate else "release_gate_false",
        "weights_used": {k: float(v) for k, v in merged_weights.items()},
    }

    artifact_payload = {
        "timestamp": decision_iso,
        "cusip": request.cusip,
        "side": request.side,
        "notional": float(request.notional),
        "protocol": request.protocol,
        "confidence_score": float(confidence_score),
        "expected_slippage_bps": float(expected_slippage_bps),
        "recommended_action": recommended.value,
        "release_gate": bool(release_gate),
        "regime_score": float(regime.regime_score),
        "liquidity_index": float(liquidity.liquidity_index),
        "drivers": list(drivers),
    }
    artifact_hash = canonical_sha256(artifact_payload)

    return ExecutionConfidenceResponse(
        timestamp=decision_iso,
        cusip=str(request.cusip),
        side=str(request.side),
        notional=float(request.notional),
        protocol=str(request.protocol),
        confidence_score=float(confidence_score),
        expected_slippage_bps=float(expected_slippage_bps),
        confidence_interval_low=float(ci_low),
        confidence_interval_high=float(ci_high),
        recommended_action=recommended.label,
        human_review_required=bool(human_review),
        model_run_id=resolved_run_id,
        release_gate=bool(release_gate),
        artifact_hash=artifact_hash,
        metadata=metadata,
    )


def _stale_response(
    *,
    request: ExecutionConfidenceRequest,
    decision_iso: str,
    resolved_run_id: str,
    profile: str,
    regime: Any,
    liquidity: Any,
    decision_ts: pd.Timestamp,
    max_staleness: float,
    release_gate_input: bool,
    reason: str,
) -> ExecutionConfidenceResponse:
    """Build the soft-fail stale-signal response per the PR-5 spec.

    Returns ``recommended_action=UNAVAILABLE_STALE_SIGNAL`` with
    ``release_gate=False`` so downstream consumers fail closed. The
    ``signal_age_seconds_*`` keys remain populated (NaN when the
    corresponding signal was absent entirely) for telemetry."""
    regime_age = _signal_age_seconds(regime.timestamp if regime is not None else None, decision_ts)
    liquidity_age = _signal_age_seconds(liquidity.timestamp if liquidity is not None else None, decision_ts)
    max_age = max(regime_age, liquidity_age)
    metadata: dict[str, Any] = {
        "profile": profile,
        "reason": reason,
        "regime_score": float(regime.regime_score) if regime is not None else None,
        "regime_label": regime.regime_label if regime is not None else None,
        "liquidity_index": (float(liquidity.liquidity_index) if liquidity is not None else None),
        "liquidity_label": (liquidity.liquidity_label if liquidity is not None else None),
        "liquidity_scope_type": (liquidity.scope_type if liquidity is not None else None),
        "liquidity_scope_id": (liquidity.scope_id if liquidity is not None else None),
        "drivers": [],
        "logit_components": {},
        "signal_age_seconds_credit_regime": float(regime_age),
        "signal_age_seconds_liquidity": float(liquidity_age),
        "max_signal_age_seconds": float(max_age),
        "max_signal_staleness_threshold_seconds": float(max_staleness),
        "release_gate_input": bool(release_gate_input),
    }
    artifact_payload = {
        "timestamp": decision_iso,
        "cusip": request.cusip,
        "side": request.side,
        "notional": float(request.notional),
        "protocol": request.protocol,
        "recommended_action": ExecutionRecommendation.UNAVAILABLE_STALE_SIGNAL.value,
        "release_gate": False,
        "reason": reason,
        "max_signal_age_seconds": float(max_age) if math.isfinite(max_age) else None,
    }
    return ExecutionConfidenceResponse(
        timestamp=decision_iso,
        cusip=str(request.cusip),
        side=str(request.side),
        notional=float(request.notional),
        protocol=str(request.protocol),
        confidence_score=0.0,
        expected_slippage_bps=None,
        confidence_interval_low=None,
        confidence_interval_high=None,
        recommended_action=ExecutionRecommendation.UNAVAILABLE_STALE_SIGNAL.label,
        human_review_required=True,
        model_run_id=resolved_run_id,
        release_gate=False,
        artifact_hash=canonical_sha256(artifact_payload),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# warehouse plumbing
# ---------------------------------------------------------------------------


def write_execution_confidence_prediction(
    warehouse: Any,
    response: ExecutionConfidenceResponse,
    *,
    request_id: str,
) -> int:
    """Persist a :class:`ExecutionConfidenceResponse` row.

    ``request_id`` is the PR-15 composite-PK column on
    ``execution_confidence_predictions``; it must be supplied by the
    caller so two API workers serving the same client request cannot
    write conflicting rows.
    """
    if not request_id:
        raise ValueError("request_id must be non-empty")
    row = {
        "request_id": str(request_id),
        "timestamp": response.timestamp,
        "model_run_id": response.model_run_id,
        "cusip": response.cusip,
        "side": response.side,
        "notional": float(response.notional),
        "protocol": response.protocol,
        "confidence_score": float(response.confidence_score),
        "expected_slippage_bps": (
            float(response.expected_slippage_bps) if response.expected_slippage_bps is not None else None
        ),
        "confidence_interval_low": (
            float(response.confidence_interval_low) if response.confidence_interval_low is not None else None
        ),
        "confidence_interval_high": (
            float(response.confidence_interval_high) if response.confidence_interval_high is not None else None
        ),
        "recommended_action": response.recommended_action,
        "human_review_required": 1 if response.human_review_required else 0,
        "release_gate": 1 if response.release_gate else 0,
        "artifact_hash": response.artifact_hash,
        "metadata_json": json.dumps(response.metadata, sort_keys=True, default=str),
    }
    return int(warehouse.write_execution_confidence_prediction(pd.DataFrame([row])))


def write_execution_outcome(
    warehouse: Any,
    *,
    request_id: str,
    observed: Mapping[str, Any],
) -> int:
    """Persist an observed execution outcome.

    The warehouse writer enforces ``observed_at > decision_timestamp``
    (PR-6 Q-2 / REVIEW.md §3.6 PR-10); a failing inequality raises
    :class:`ValueError` before any rows hit DuckDB.
    """
    if not request_id:
        raise ValueError("request_id must be non-empty")
    required = {"observed_at", "decision_timestamp"}
    missing = required - set(observed)
    if missing:
        raise ValueError(f"observed payload missing required keys: {sorted(missing)!r}")
    consumed_keys: frozenset[str] = frozenset(
        {
            "cusip",
            "side",
            "notional",
            "filled_quantity",
            "execution_price",
            "observed_at",
            "outcome_observation_lag",
            "decision_timestamp",
        }
    )
    metadata = {k: observed[k] for k in observed if k not in consumed_keys}
    row: dict[str, Any] = {
        "request_id": str(request_id),
        "cusip": str(observed.get("cusip", "")),
        "side": str(observed.get("side", "")),
        "notional": float(observed.get("notional", 0.0)),
        "filled_quantity": (float(observed["filled_quantity"]) if "filled_quantity" in observed else None),
        "execution_price": (float(observed["execution_price"]) if "execution_price" in observed else None),
        "observed_at": str(observed["observed_at"]),
        "outcome_observation_lag": (
            float(observed["outcome_observation_lag"]) if "outcome_observation_lag" in observed else None
        ),
        "decision_timestamp": str(observed["decision_timestamp"]),
        "metadata_json": json.dumps(metadata, sort_keys=True, default=str),
    }
    return int(warehouse.write_execution_outcome(pd.DataFrame([row])))


def latest_execution_confidence_prediction(
    warehouse: Any,
    *,
    cusip: str | None = None,
) -> ExecutionConfidenceResponse | None:
    """Return the most recent ``execution_confidence_predictions`` row.

    Optionally filtered by cusip; without a filter the most recent row
    across every cusip is returned. The ordering mirrors the table's
    natural sort (``timestamp, request_id``).
    """
    df = warehouse.read_execution_confidence_predictions()
    if df is None or df.empty:
        return None
    if cusip is not None:
        df = df.loc[df["cusip"].astype(str) == str(cusip)]
        if df.empty:
            return None
    df = df.sort_values("timestamp")
    row = df.iloc[-1]
    metadata_json = row.get("metadata_json")
    metadata = json.loads(metadata_json) if metadata_json else {}
    return ExecutionConfidenceResponse(
        timestamp=str(row["timestamp"]),
        cusip=str(row["cusip"]),
        side=str(row["side"]),
        notional=float(row["notional"]),
        protocol=str(row["protocol"]),
        confidence_score=float(row["confidence_score"]),
        expected_slippage_bps=(
            float(row["expected_slippage_bps"]) if pd.notna(row.get("expected_slippage_bps")) else None
        ),
        confidence_interval_low=(
            float(row["confidence_interval_low"]) if pd.notna(row.get("confidence_interval_low")) else None
        ),
        confidence_interval_high=(
            float(row["confidence_interval_high"]) if pd.notna(row.get("confidence_interval_high")) else None
        ),
        recommended_action=str(row["recommended_action"]),
        human_review_required=bool(int(row["human_review_required"])),
        model_run_id=str(row["model_run_id"]),
        release_gate=bool(int(row["release_gate"])),
        artifact_hash=str(row["artifact_hash"]),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# B: build_execution_features
# ---------------------------------------------------------------------------


def build_execution_features(
    warehouse: Any,
    request: ExecutionConfidenceRequest,
    *,
    lookback_days: int = 30,
) -> pd.DataFrame:
    """Build a flat features frame for execution-confidence scoring.

    Pulls:

    - ``bond_reference`` snapshot at ``request.timestamp`` (survivorship-safe),
    - latest ``credit_regime_scores``,
    - latest cusip-scoped ``liquidity_stress_scores`` (fallback to market scope),
    - recent ``dealer_response_stats`` aggregated over ``lookback_days``,
    - time-of-day decomposition,
    - historical ``execution_outcomes`` for the same cusip/protocol pair.

    Returns a single-row :class:`pandas.DataFrame` whose columns are
    scalar features keyed by name. PR-5 ships this as the **input
    materializer** for the deterministic baseline (which still consumes
    the request + the warehouse-latest signals directly); v1.5.1 will
    swap the deterministic baseline for a calibrated logistic that
    consumes this frame directly.

    PIT-safety: every emitted feature carries the warehouse value at or
    before ``request.timestamp``. The downstream :func:`score_execution_confidence`
    re-asserts PIT on the signal timestamps; this builder is a
    convenience layer.
    """
    decision_ts = _coerce_decision_ts(request.timestamp)
    out: dict[str, Any] = {
        "request_timestamp": iso8601_z(decision_ts),
        "cusip": str(request.cusip),
        "side": str(request.side),
        "notional": float(request.notional),
        "notional_log10": math.log10(max(float(request.notional), 1.0)),
        "protocol": str(request.protocol),
        "urgency": str(request.urgency or "normal"),
        "limit_price": (float(request.limit_price) if request.limit_price is not None else None),
        "sector": request.sector,
        "rating": request.rating,
        "rating_class": _rating_class(request.rating),
        "maturity_bucket": request.maturity_bucket,
        "hour_of_day_utc": int(decision_ts.hour),
        "minute_of_hour": int(decision_ts.minute),
        "day_of_week": int(decision_ts.day_of_week),
    }

    regime = latest_credit_regime_score(warehouse)
    if regime is not None:
        # v1.6.0 PIT rail (REVIEW_DEEP_V1_5_2.md A6 / Finding #15): the
        # CLI / batch builder now mirrors the hot-path PIT enforcement in
        # ``score_execution_confidence`` so future-dated rows cannot leak
        # into offline training data via ``build_execution_features``.
        assert_pit_safe(
            feature_timestamp=regime.timestamp,
            decision_timestamp=decision_ts,
            label="credit_regime",
        )
        out["regime_score"] = float(regime.regime_score)
        out["regime_label"] = regime.regime_label
        out["regime_release_gate"] = bool(regime.release_gate)
        out["signal_age_seconds_credit_regime"] = _signal_age_seconds(regime.timestamp, decision_ts)

    liquidity = latest_liquidity_stress_score(warehouse, scope_type="cusip", scope_id=request.cusip)
    if liquidity is None:
        liquidity = latest_liquidity_stress_score(warehouse)
    if liquidity is not None:
        # v1.6.0 PIT rail (REVIEW_DEEP_V1_5_2.md A6 / Finding #15).
        assert_pit_safe(
            feature_timestamp=liquidity.timestamp,
            decision_timestamp=decision_ts,
            label="liquidity_stress",
        )
        out["liquidity_index"] = float(liquidity.liquidity_index)
        out["liquidity_label"] = liquidity.liquidity_label
        out["liquidity_scope_type"] = liquidity.scope_type
        out["liquidity_scope_id"] = liquidity.scope_id
        out["liquidity_release_gate"] = bool(liquidity.release_gate)
        out["signal_age_seconds_liquidity"] = _signal_age_seconds(liquidity.timestamp, decision_ts)

    # bond_reference asof — best-effort; survivorship-safe via the
    # storage helper.
    try:
        from market_regime_engine.storage import read_bond_reference_asof

        ref = read_bond_reference_asof(warehouse, decision_ts)
        if ref is not None and not ref.empty:
            sub = ref.loc[ref["cusip"].astype(str) == str(request.cusip)]
            if not sub.empty:
                row = sub.iloc[0]
                out["bond_ref_sector"] = str(row["sector"]) if pd.notna(row.get("sector")) else None
                out["bond_ref_rating"] = str(row["rating"]) if pd.notna(row.get("rating")) else None
                out["bond_ref_duration"] = float(row["duration"]) if pd.notna(row.get("duration")) else None
                out["bond_ref_amount_outstanding"] = (
                    float(row["amount_outstanding"]) if pd.notna(row.get("amount_outstanding")) else None
                )
                if pd.notna(row.get("amount_outstanding")):
                    out["amount_outstanding_log10"] = math.log10(max(float(row["amount_outstanding"]), 1.0))
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("bond_reference_asof lookup failed: %s", exc)

    # dealer_response_stats summary over the lookback window.
    try:
        dealer_stats = warehouse._backend.read_sql("SELECT * FROM dealer_response_stats")
    except Exception:
        dealer_stats = None
    if dealer_stats is not None and not dealer_stats.empty:
        window_start = decision_ts - pd.Timedelta(days=int(lookback_days))
        dealer_stats = dealer_stats.copy()
        dealer_stats["window_end_ts"] = pd.to_datetime(dealer_stats["window_end"], utc=True, errors="coerce")
        recent = dealer_stats.loc[
            (dealer_stats["window_end_ts"] >= window_start) & (dealer_stats["window_end_ts"] <= decision_ts)
        ]
        if not recent.empty:
            requests_total = float(recent["requests"].fillna(0).sum())
            responses_total = float(recent["responses"].fillna(0).sum())
            out["dealer_response_count"] = responses_total
            out["dealer_fill_rate"] = responses_total / requests_total if requests_total > 0 else None
            avg_ms = recent["avg_response_ms"].dropna()
            out["dealer_avg_response_ms"] = float(avg_ms.mean()) if not avg_ms.empty else None

    # historical execution_outcomes for this cusip — observed slippage
    # mean / count as a deterministic prior.
    try:
        outcomes = warehouse.read_execution_outcomes()
    except Exception:
        outcomes = None
    if outcomes is not None and not outcomes.empty:
        outcomes = outcomes.copy()
        outcomes["observed_at_ts"] = pd.to_datetime(outcomes["observed_at"], utc=True, errors="coerce")
        window_start = decision_ts - pd.Timedelta(days=int(lookback_days))
        sub = outcomes.loc[
            (outcomes["cusip"].astype(str) == str(request.cusip))
            & (outcomes["observed_at_ts"] >= window_start)
            & (outcomes["observed_at_ts"] < decision_ts)
        ]
        out["historical_outcome_count"] = len(sub)
        if not sub.empty and "outcome_observation_lag" in sub.columns:
            lags = sub["outcome_observation_lag"].dropna()
            if not lags.empty:
                out["historical_outcome_lag_mean"] = float(lags.mean())

    return pd.DataFrame([out])
