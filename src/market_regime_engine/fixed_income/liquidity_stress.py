# SPDX-License-Identifier: Apache-2.0
"""PR-4 liquidity-stress scorer (deterministic composite + hysteresis).

Per ``MRE_FIXED_INCOME_AGENT.md §"PR 4 — liquidity stress model"`` and
``MRE_FIXED_INCOME_INSTRUCTIONS.md §6.2``: build a scope-aware
deterministic composite from the eleven liquidity features
(bid-ask, trade-count velocity, volume / trailing ADV, time since last
trade, RFQ dealers requested, quotes received, quote dispersion,
Amihud illiquidity, dealer response count, axe freshness proxy, order
imbalance) and emit a 0-100 ``liquidity_index`` where higher means
*more* stress.

Component scores (each normalised to 0-100, higher = more stress):

    score_quotes_dispersion  z-score sigmoid of ``quote_dispersion``
                             vs the trailing window
    score_bid_ask            percentile rank of ``bid_ask_width``
    score_trade_velocity     inverse percentile of ``trade_count_velocity``
                             (low velocity → high stress)
    score_rfq_fill_rate      ``(1 - quotes_received/dealers_requested) * 100``
    score_amihud             percentile rank of ``amihud_illiquidity``
    score_time_gap           ``min(100, time_since_last_trade_minutes * 2)``

Composite: weighted average over the components with present data;
default weights ``{quotes_dispersion: 0.20, bid_ask: 0.20,
trade_velocity: 0.15, rfq_fill_rate: 0.20, amihud: 0.15,
time_gap: 0.10}`` (sum = 1.0). Custom weights are normalised to sum
to 1.0 internally.

Confidence: ``1.0 - missing_component_fraction`` (same as PR-3),
capped at 0.5 when ``release_gate=False`` per AGENT.md non-negotiable 8.

Drivers: top-2 component names by ``|score - 50.0|`` (most-extreme
deviation from the neutral midline). Ties broken in component
declaration order.

Hysteresis: when ``prev_label`` is supplied, the new label is the
output of :func:`classify_with_hysteresis`; otherwise the sharp
:func:`liquidity_label_from_score` bucket applies.

Warehouse contract: :func:`write_liquidity_stress_score`,
:func:`latest_liquidity_stress_score`, and
:func:`list_recent_liquidity_stress_scores` map ``LiquidityStressOutput``
to / from the ``liquidity_stress_scores`` table declared in
``fixed_income/schema.py``. The DB column ``liquidity_score`` stores
the same numeric quantity exposed as the ``liquidity_index`` field on
:class:`LiquidityStressOutput` (the AGENT.md naming).
"""

from __future__ import annotations

import json
import logging
import math
import os
import uuid
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any, Literal

import pandas as pd

from market_regime_engine.fixed_income.critical_features import (
    CRITICAL_LABEL_LIQUIDITY,
    LIQUIDITY_CRITICAL_COLUMNS,
    evaluate_critical_features,
)
from market_regime_engine.fixed_income.hashing import canonical_sha256
from market_regime_engine.fixed_income.hysteresis import apply_hysteresis
from market_regime_engine.fixed_income.pit_guard import (
    PitViolationError,
    assert_pit_safe,
)
from market_regime_engine.fixed_income.schemas import (
    LiquidityLabel,
    LiquidityStressOutput,
    liquidity_label_from_score,
)
from market_regime_engine.fixed_income.timestamps import iso8601_z, to_utc
from market_regime_engine.frontier.data_cleaning import NanPolicy, PitAuditFailure

log = logging.getLogger(__name__)

ScopeType = Literal["market", "sector", "rating", "cusip"]
_VALID_SCOPE_TYPES: frozenset[str] = frozenset({"market", "sector", "rating", "cusip"})


__all__ = [
    "COMPONENT_FEATURES",
    "DEFAULT_WEIGHTS",
    "HYSTERESIS_BANDS_LIQUIDITY",
    "ScopeType",
    "classify_with_hysteresis",
    "latest_liquidity_stress_score",
    "list_recent_liquidity_stress_scores",
    "output_to_dict",
    "score_liquidity_stress",
    "write_liquidity_stress_score",
]


# ---------------------------------------------------------------------------
# component / feature configuration
# ---------------------------------------------------------------------------


COMPONENT_FEATURES: dict[str, tuple[str, ...]] = {
    "quotes_dispersion": ("quote_dispersion",),
    "bid_ask": ("bid_ask_width",),
    "trade_velocity": ("trade_count_velocity",),
    "rfq_fill_rate": ("dealers_requested", "quotes_received"),
    "amihud": ("amihud_illiquidity",),
    "time_gap": ("time_since_last_trade",),
}

DEFAULT_WEIGHTS: dict[str, float] = {
    "quotes_dispersion": 0.20,
    "bid_ask": 0.20,
    "trade_velocity": 0.15,
    "rfq_fill_rate": 0.20,
    "amihud": 0.15,
    "time_gap": 0.10,
}

_NEUTRAL_SCORE: float = 50.0


# v1.5 (PR-4 task C): asymmetric (enter, exit) hysteresis bands per
# liquidity label so the bucket is "sticky" once entered. The
# convention mirrors the credit module:
#
#     NORMAL: (None, 25)             — exit upward at 25.
#     MILD_STRESS: (20, 45)          — enter at 20+, exit upward at 45.
#     ELEVATED_STRESS: (40, 65)
#     SEVERE_STRESS: (60, 85)
#     CRISIS_LIQUIDITY: (80, None)   — terminal upper edge.
HYSTERESIS_BANDS_LIQUIDITY: dict[LiquidityLabel, tuple[float | None, float | None]] = {
    LiquidityLabel.NORMAL: (None, 25.0),
    LiquidityLabel.MILD_STRESS: (20.0, 45.0),
    LiquidityLabel.ELEVATED_STRESS: (40.0, 65.0),
    LiquidityLabel.SEVERE_STRESS: (60.0, 85.0),
    LiquidityLabel.CRISIS_LIQUIDITY: (80.0, None),
}


def classify_with_hysteresis(score: float, prev_label: LiquidityLabel | None) -> LiquidityLabel:
    """Map ``score`` to a :class:`LiquidityLabel` with asymmetric hysteresis.

    ``prev_label is None`` → sharp-bucket fallback via
    :func:`liquidity_label_from_score`. ``prev_label`` is sticky inside
    its band; outside the band the score re-classifies via the sharp
    bucket mapping.
    """
    return apply_hysteresis(
        float(score),
        prev_label=prev_label,
        bands=HYSTERESIS_BANDS_LIQUIDITY,
        sharp_fallback=liquidity_label_from_score,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _series(wide: pd.DataFrame, column: str) -> pd.Series:
    if column not in wide.columns:
        return pd.Series(dtype=float, name=column)
    return wide[column].astype(float)


def _latest(series: pd.Series) -> float | None:
    """Return the last non-NaN value in ``series`` or ``None``."""
    if series is None or series.empty:
        return None
    finite = series.dropna()
    if finite.empty:
        return None
    return float(finite.iloc[-1])


def _percentile_score(series: pd.Series, latest: float, *, direction: int = 1) -> float:
    """Empirical percentile of ``latest`` in ``series`` → 0-100.

    ``direction=+1``: higher value → higher stress (bid-ask width, Amihud).
    ``direction=-1``: higher value → lower stress (trade velocity).
    """
    finite = series.dropna()
    if finite.empty:
        return _NEUTRAL_SCORE
    pct = float((finite <= latest).mean() * 100.0)
    if direction < 0:
        pct = 100.0 - pct
    return max(0.0, min(100.0, pct))


def _zscore_sigmoid(value: float, mean: float, std: float) -> float:
    if std is None or not math.isfinite(std) or std == 0:
        return _NEUTRAL_SCORE
    z = (value - mean) / std
    return float(100.0 / (1.0 + math.exp(-z)))


def _pivot_features(features: pd.DataFrame) -> pd.DataFrame:
    """Long → wide pivot keyed by ``date`` × ``feature_name``.

    Multiple rows for the same (date, feature_name) collapse via
    ``last`` (the builder aggregates per timestamp; this only fires on
    adversarial inputs). ``aggfunc="last"`` is deterministic given
    the upstream sort order.
    """
    if features is None or features.empty:
        return pd.DataFrame()
    wide = features.pivot_table(
        index="date",
        columns="feature_name",
        values="value",
        aggfunc="last",
    ).sort_index()
    wide.columns.name = None
    return wide


def _resolve_nan_policy(features: pd.DataFrame | None) -> NanPolicy:
    if features is None:
        return NanPolicy.NAN_FAILS_PIT_AUDIT
    name = features.attrs.get("nan_policy") if hasattr(features, "attrs") else None
    if isinstance(name, NanPolicy):
        return name
    if isinstance(name, str):
        try:
            return NanPolicy(name)
        except ValueError:
            return NanPolicy.NAN_FAILS_PIT_AUDIT
    return NanPolicy.NAN_FAILS_PIT_AUDIT


def _apply_nan_policy(
    wide: pd.DataFrame,
    *,
    nan_policy: NanPolicy,
    overrides: Mapping[str, NanPolicy] | None,
) -> None:
    """Audit-only column policy: under ``NAN_FAILS_PIT_AUDIT`` a column
    with zero non-NaN observations in the lookback window means an
    input is missing, so the audit must fire (mirrors the PR-3 fix)."""
    if wide is None or wide.empty:
        if nan_policy is NanPolicy.NAN_FAILS_PIT_AUDIT:
            raise PitAuditFailure("liquidity stress features empty; cannot satisfy NAN_FAILS_PIT_AUDIT")
        return
    if nan_policy is not NanPolicy.NAN_FAILS_PIT_AUDIT and not overrides:
        return
    missing_cols: list[str] = []
    for col in wide.columns:
        policy = (overrides or {}).get(col, nan_policy)
        if policy is not NanPolicy.NAN_FAILS_PIT_AUDIT:
            continue
        if wide[col].dropna().empty:
            missing_cols.append(str(col))
    if missing_cols:
        raise PitAuditFailure(
            "NAN_FAILS_PIT_AUDIT triggered for liquidity feature(s) "
            f"with no non-NaN observation in the lookback window: {sorted(missing_cols)!r}"
        )


# ---------------------------------------------------------------------------
# component scorers
# ---------------------------------------------------------------------------


def _quotes_dispersion_component(wide: pd.DataFrame) -> float | None:
    series = _series(wide, "quote_dispersion")
    finite = series.dropna()
    if finite.empty:
        return None
    latest = float(finite.iloc[-1])
    mean = float(finite.mean())
    std = float(finite.std(ddof=0))
    return _zscore_sigmoid(latest, mean, std)


def _bid_ask_component(wide: pd.DataFrame) -> float | None:
    series = _series(wide, "bid_ask_width")
    latest = _latest(series)
    if latest is None:
        return None
    return _percentile_score(series, latest, direction=+1)


def _trade_velocity_component(wide: pd.DataFrame) -> float | None:
    series = _series(wide, "trade_count_velocity")
    latest = _latest(series)
    if latest is None:
        return None
    # Inverse rank: low velocity (low percentile) → high stress.
    return _percentile_score(series, latest, direction=-1)


def _rfq_fill_rate_component(wide: pd.DataFrame) -> float | None:
    requested = _latest(_series(wide, "dealers_requested"))
    received = _latest(_series(wide, "quotes_received"))
    if requested is None or received is None:
        return None
    if requested <= 0:
        # Zero requests carries no signal; mark missing so the
        # confidence drop matches operator intuition.
        return None
    fill_rate = max(0.0, min(1.0, received / requested))
    return float(max(0.0, min(100.0, (1.0 - fill_rate) * 100.0)))


def _amihud_component(wide: pd.DataFrame) -> float | None:
    series = _series(wide, "amihud_illiquidity")
    latest = _latest(series)
    if latest is None:
        return None
    return _percentile_score(series, latest, direction=+1)


def _time_gap_component(wide: pd.DataFrame) -> float | None:
    """``min(100, minutes_since_last_trade * 2)`` — 50 minutes saturates the stress.

    The feature is expected in *minutes*; the builder is responsible
    for the units, mirroring the AGENT.md feature catalog.
    """
    latest = _latest(_series(wide, "time_since_last_trade"))
    if latest is None:
        return None
    return float(max(0.0, min(100.0, float(latest) * 2.0)))


_COMPONENT_FUNCS: dict[str, Any] = {
    "quotes_dispersion": _quotes_dispersion_component,
    "bid_ask": _bid_ask_component,
    "trade_velocity": _trade_velocity_component,
    "rfq_fill_rate": _rfq_fill_rate_component,
    "amihud": _amihud_component,
    "time_gap": _time_gap_component,
}


# ---------------------------------------------------------------------------
# weights / drivers helpers
# ---------------------------------------------------------------------------


def _normalise_weights(weights: Mapping[str, float] | None) -> dict[str, float]:
    """Return a copy of ``weights`` summing to 1.0 (defaults preserved)."""
    src = dict(weights) if weights else dict(DEFAULT_WEIGHTS)
    total = float(sum(src.values()))
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {name: float(v) / total for name, v in src.items()}


def _drivers(scores: Mapping[str, float]) -> tuple[str, ...]:
    """Top-2 components by absolute deviation from the neutral midline."""
    ranked = sorted(
        scores.items(),
        key=lambda kv: (-abs(float(kv[1]) - _NEUTRAL_SCORE), list(DEFAULT_WEIGHTS).index(kv[0])),
    )
    return tuple(name for name, _ in ranked[:2])


def _compose(scores: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """Weighted average over the components that actually have scores."""
    total_weight = sum(weights.get(name, 0.0) for name in scores)
    if total_weight <= 0:
        return float(_NEUTRAL_SCORE)
    weighted = sum(scores[name] * weights.get(name, 0.0) for name in scores)
    return float(max(0.0, min(100.0, weighted / total_weight)))


def _confidence(
    *,
    missing_components: list[str],
    all_components: tuple[str, ...],
    release_gate: bool,
) -> float:
    if not all_components:
        return 0.0
    fraction_missing = float(len(missing_components)) / float(len(all_components))
    conf = max(0.0, min(1.0, 1.0 - fraction_missing))
    if not release_gate:
        conf = min(conf, 0.5)
    return conf


# ---------------------------------------------------------------------------
# PIT audit
# ---------------------------------------------------------------------------


def _coerce_asof(asof: pd.Timestamp | str) -> pd.Timestamp:
    if isinstance(asof, str):
        out = to_utc(asof)
        if out is None:
            raise ValueError("asof must not be None")
        return out
    ts = pd.Timestamp(asof)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _resolve_model_run_id(model_run_id: str | None, scope_type: str, profile: str) -> str:
    if model_run_id and model_run_id.strip():
        return model_run_id
    return f"liquidity-{scope_type}-{profile}-{uuid.uuid4().hex[:12]}"


def _audit_pit(features: pd.DataFrame, *, asof: pd.Timestamp) -> None:
    """Row-level PIT enforcement on the long-form feature frame.

    v1.5.1 (PR-9 FIX 5): vectorised replacement of the legacy
    ``features.iterrows()`` + per-row :func:`assert_pit_safe` loop.
    Mirrors the pattern from
    :func:`credit_spread_regime._audit_pit` (PR-8 Tier-2 fix A2):
    we batch-normalise the timestamp columns once and let
    :func:`audit_pit_dataframe` produce the same accept / reject
    semantics in O(few ms) per column comparison.

    The ``MRE_FI_LEGACY_VECTORIZE=1`` env var routes through the
    legacy iterrows loop so operators can compare row-counts on a
    suspected regression. The legacy path will be removed in
    v1.5.2.
    """
    if features.empty or "source_timestamp" not in features.columns:
        return
    if os.getenv("MRE_FI_LEGACY_VECTORIZE", "").strip() in {"1", "true", "yes", "on"}:
        _audit_pit_legacy_iterrows(features, asof=asof)
        return

    from market_regime_engine.fixed_income.pit_guard import audit_pit_dataframe

    df = features.copy()
    df["__decision_ts"] = asof
    vintage_col = "vintage_date" if "vintage_date" in df.columns else None
    report = audit_pit_dataframe(
        df,
        decision_timestamp_col="__decision_ts",
        feature_timestamp_col="source_timestamp",
        vintage_timestamp_col=vintage_col,
    )
    if report.violation_count > 0:
        first = report.violations.iloc[0]
        label = str(first.get("feature_name", "feature"))
        reason = str(first.get("pit_violation_reason", ""))
        raise PitViolationError(
            f"liquidity PIT audit failed: {report.violation_count} row(s) violate PIT "
            f"(asof={asof}, first violator label={label!r} reason={reason!r})"
        )


def _audit_pit_legacy_iterrows(features: pd.DataFrame, *, asof: pd.Timestamp) -> None:
    """Pre-v1.5.1 iterrows loop, gated behind ``MRE_FI_LEGACY_VECTORIZE=1``.

    Kept for one release cycle so operators can A/B the parity of the
    new vectorised path on suspected regressions. Slated for deletion
    in v1.5.2 once the vectorised path has burned in.
    """
    for _, row in features.iterrows():
        source = pd.Timestamp(row["source_timestamp"])
        if source.tzinfo is None:
            source = source.tz_localize("UTC")
        vintage = row.get("vintage_date")
        if vintage is not None and not pd.isna(vintage):
            vintage_ts = pd.Timestamp(vintage)
            if vintage_ts.tzinfo is None:
                vintage_ts = vintage_ts.tz_localize("UTC")
        else:
            vintage_ts = None
        assert_pit_safe(
            feature_timestamp=source,
            decision_timestamp=asof,
            vintage_timestamp=vintage_ts,
            label=str(row.get("feature_name", "feature")),
        )


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------


def score_liquidity_stress(
    features: pd.DataFrame,
    *,
    scope_type: ScopeType,
    scope_id: str,
    asof: pd.Timestamp | str,
    model_run_id: str | None = None,
    release_gate: bool = True,
    profile: str = "production",
    prev_label: LiquidityLabel | None = None,
    weights: Mapping[str, float] | None = None,
    nan_policy_overrides: Mapping[str, NanPolicy] | None = None,
) -> LiquidityStressOutput:
    """Compute the liquidity-stress score for the given scope.

    Parameters
    ----------
    features:
        Long-form DataFrame with columns
        ``["date", "feature_name", "value", "source_timestamp", "vintage_date"]``
        from :func:`build_liquidity_features`. Empty input returns a
        neutral 50.0 score with ``confidence=0.0`` and
        ``release_gate=False``.
    scope_type:
        One of ``"market"``, ``"sector"``, ``"rating"``, ``"cusip"``.
    scope_id:
        Scope identifier — ``"ALL"`` (or any opaque tag) for market
        scope; a sector / rating / cusip value otherwise.
    asof:
        Decision timestamp. Must be UTC (string inputs route through
        :func:`to_utc`).
    model_run_id:
        Reproducibility id; profile-stamped UUID minted when omitted.
    release_gate:
        Governance gate. ``False`` caps confidence at 0.5; the output
        ``release_gate`` flips to ``False`` automatically when the
        NaN-audit fires on a required feature.
    profile:
        Operating profile tag stored in ``metadata.profile``.
    prev_label:
        Previous run's :class:`LiquidityLabel` for asymmetric
        hysteresis (PR-4 task C). ``None`` → sharp-bucket label.
    weights:
        Optional component weights override; normalised to sum to 1.0.
    nan_policy_overrides:
        Optional per-feature NaN-policy overrides forwarded to the
        column audit.

    Returns
    -------
    LiquidityStressOutput
        Frozen dataclass per PR-1 schemas. The ``liquidity_index``
        field carries the 0-100 score (higher = more stress).

    Raises
    ------
    PitViolationError
        If any feature row's ``source_timestamp`` exceeds ``asof``.
    """
    if scope_type not in _VALID_SCOPE_TYPES:
        raise ValueError(f"scope_type must be one of {sorted(_VALID_SCOPE_TYPES)!r}; got {scope_type!r}")
    if not scope_id:
        raise ValueError("scope_id must not be empty")

    asof_utc = _coerce_asof(asof)
    weights_norm = _normalise_weights(weights)

    if features is not None and not features.empty and "source_timestamp" in features.columns:
        _audit_pit(features, asof=asof_utc)

    wide = _pivot_features(features)

    component_scores: dict[str, float] = {}
    missing_components: list[str] = []
    pit_audit_failed = False
    for component in DEFAULT_WEIGHTS:
        score = _COMPONENT_FUNCS[component](wide)
        if score is None:
            missing_components.append(component)
        else:
            component_scores[component] = float(score)

    nan_policy = _resolve_nan_policy(features)
    try:
        _apply_nan_policy(wide, nan_policy=nan_policy, overrides=nan_policy_overrides)
    except PitAuditFailure:
        pit_audit_failed = True
        log.warning("liquidity stress PIT audit failed (column-level); flipping release_gate=False")
    if nan_policy is NanPolicy.NAN_FAILS_PIT_AUDIT and missing_components:
        pit_audit_failed = True
        log.warning(
            "liquidity stress PIT audit failed: missing components %r; flipping release_gate=False",
            missing_components,
        )

    # v1.5.1 (PR-9 FIX 8): critical-feature contract overrides
    # nan_policy re-weighting. Missing bid-ask or RFQ-response
    # observations force release_gate=False and the ``NO_DECISION``
    # fail-closed label regardless of nan_policy.
    critical_audit = evaluate_critical_features(wide, contract=LIQUIDITY_CRITICAL_COLUMNS)

    if not component_scores:
        liquidity_index = float(_NEUTRAL_SCORE)
        confidence = 0.0
        gate = False
        drivers: tuple[str, ...] = ()
        component_scores = {}
    else:
        liquidity_index = _compose(component_scores, weights_norm)
        confidence = _confidence(
            missing_components=missing_components,
            all_components=tuple(DEFAULT_WEIGHTS),
            release_gate=release_gate,
        )
        gate = bool(release_gate) and not pit_audit_failed
        drivers = _drivers(component_scores)
        if not gate:
            confidence = min(confidence, 0.5)

    hysteresis_applied = prev_label is not None
    liquidity_label_enum = classify_with_hysteresis(liquidity_index, prev_label)
    liquidity_label = liquidity_label_enum.label
    if critical_audit.fail_closed:
        gate = False
        confidence = min(confidence, 0.5)
        liquidity_label = CRITICAL_LABEL_LIQUIDITY
        # v1.6.0 fail-closed consistency fix (REVIEW_DEEP_V1_5_2.md A11 /
        # Finding #11): also reset the numeric ``liquidity_index`` to the
        # neutral midpoint so downstream consumers cannot present an
        # internally inconsistent state of e.g. index=85 paired with
        # label="UNCERTAIN".
        liquidity_index = float(_NEUTRAL_SCORE)
        log.warning(
            "liquidity stress critical-feature contract violated: missing=%r; "
            "flipping release_gate=False, label=%r, liquidity_index=%.1f (neutral)",
            [feature.value for feature in critical_audit.missing],
            CRITICAL_LABEL_LIQUIDITY,
            liquidity_index,
        )

    metadata: dict[str, Any] = {
        "weights_used": weights_norm,
        "feature_count": int(len(features) if features is not None else 0),
        "missing_features": missing_components,
        "score_components": component_scores,
        "nan_policy": nan_policy.value,
        "profile": profile,
        "pit_audit_failed": pit_audit_failed,
        "hysteresis_applied": hysteresis_applied,
        "prev_label": prev_label.value if prev_label is not None else None,
        # v1.5.1 (PR-9 FIX 8): surface the critical-feature audit so
        # operators can pivot dashboards by which canonical input
        # tripped the fail-closed gate.
        "critical_features_missing": [feature.value for feature in critical_audit.missing],
        "critical_features_fail_closed": critical_audit.fail_closed,
    }

    timestamp_iso = iso8601_z(asof_utc)
    resolved_run_id = _resolve_model_run_id(model_run_id, scope_type, profile)
    artifact_payload = {
        "timestamp": timestamp_iso,
        "scope_type": scope_type,
        "scope_id": scope_id,
        "liquidity_index": liquidity_index,
        "liquidity_label": liquidity_label,
        "confidence": confidence,
        "drivers": list(drivers),
        "component_scores": component_scores,
    }
    artifact_hash = canonical_sha256(artifact_payload)

    return LiquidityStressOutput(
        timestamp=timestamp_iso,
        scope_type=str(scope_type),
        scope_id=str(scope_id),
        liquidity_index=float(liquidity_index),
        liquidity_label=liquidity_label,
        confidence=float(confidence),
        drivers=drivers,
        model_run_id=resolved_run_id,
        release_gate=bool(gate),
        artifact_hash=artifact_hash,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# warehouse plumbing
# ---------------------------------------------------------------------------


def write_liquidity_stress_score(warehouse: Any, output: LiquidityStressOutput) -> int:
    """Persist a :class:`LiquidityStressOutput` row to ``liquidity_stress_scores``.

    Returns the number of rows written (always 1 on success). The DB
    column ``liquidity_score`` stores ``output.liquidity_index``
    (the AGENT.md field name); the metadata blob carries the full
    ``score_components`` mapping so the API / evidence-pack code can
    rehydrate every component without a JOIN.
    """
    row = {
        "model_run_id": output.model_run_id,
        "scope_type": output.scope_type,
        "scope_id": output.scope_id,
        "timestamp": output.timestamp,
        "liquidity_score": float(output.liquidity_index),
        "liquidity_label": output.liquidity_label,
        "confidence": float(output.confidence),
        "drivers_json": json.dumps(list(output.drivers)),
        "release_gate": 1 if output.release_gate else 0,
        "artifact_hash": output.artifact_hash,
        "metadata_json": json.dumps(output.metadata, sort_keys=True, default=str),
    }
    return int(warehouse.write_liquidity_stress_score(pd.DataFrame([row])))


def latest_liquidity_stress_score(
    warehouse: Any,
    *,
    scope_type: str | None = None,
    scope_id: str | None = None,
    asof: pd.Timestamp | str | None = None,
) -> LiquidityStressOutput | None:
    """Read the most recent ``liquidity_stress_scores`` row.

    When ``scope_type`` and ``scope_id`` are both provided, the result
    is filtered to that scope; otherwise the most recent row across
    every scope is returned. When ``asof`` is supplied, the selected row
    is bounded by ``timestamp <= asof`` before the latest row is chosen.

    v1.5.1 (PR-9 FIX 2): prefer the indexed SQL fast path that hits
    ``idx_liquidity_scope_ts`` when ``scope_type`` / ``scope_id`` are
    pinned. Falls back to the legacy full-table read when the
    warehouse does not expose the new method (in-memory / external
    test doubles).
    """
    latest_fast = getattr(warehouse, "latest_liquidity_stress_score", None)
    if callable(latest_fast):
        used_legacy_fast = False
        try:
            fast_df = latest_fast(scope_type=scope_type, scope_id=scope_id, asof=asof)
        except TypeError:  # pragma: no cover - legacy external test double
            fast_df = latest_fast(scope_type=scope_type, scope_id=scope_id)
            used_legacy_fast = True
        except Exception:  # pragma: no cover - fall back on backend miss
            fast_df = None
        if used_legacy_fast and asof is not None:
            # A legacy fast path may have ignored ``asof``. Let the fallback
            # table scan choose the latest valid historical row.
            fast_df = None
        if fast_df is not None and not fast_df.empty:
            return _row_to_output(fast_df.iloc[0])
        if fast_df is not None and fast_df.empty:
            return None
    df = warehouse.read_liquidity_stress_scores()
    if df is None or df.empty:
        return None
    if scope_type is not None and scope_id is not None:
        mask = (df["scope_type"].astype(str) == str(scope_type)) & (df["scope_id"].astype(str) == str(scope_id))
        df = df.loc[mask]
        if df.empty:
            return None
    elif scope_type is not None:
        df = df.loc[df["scope_type"].astype(str) == str(scope_type)]
        if df.empty:
            return None
    if asof is not None:
        asof_ts = to_utc(asof)
        timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.loc[timestamps <= asof_ts]
        if df.empty:
            return None
    df = df.sort_values("timestamp")
    row = df.iloc[-1]
    return _row_to_output(row)


def list_recent_liquidity_stress_scores(
    warehouse: Any,
    *,
    scope_type: str | None = None,
    limit: int = 100,
) -> list[LiquidityStressOutput]:
    """List recent liquidity-stress scores, optionally filtered by ``scope_type``."""
    if limit <= 0:
        return []
    df = warehouse.read_liquidity_stress_scores()
    if df is None or df.empty:
        return []
    if scope_type is not None:
        df = df.loc[df["scope_type"].astype(str) == str(scope_type)]
        if df.empty:
            return []
    df = df.sort_values("timestamp", ascending=False).head(int(limit))
    return [_row_to_output(r) for _, r in df.iterrows()]


def _row_to_output(row: pd.Series) -> LiquidityStressOutput:
    drivers_json = row.get("drivers_json")
    metadata_json = row.get("metadata_json")
    drivers: tuple[str, ...] = tuple(json.loads(drivers_json)) if drivers_json else ()
    metadata = json.loads(metadata_json) if metadata_json else {}
    ts_raw = row["timestamp"]
    ts = pd.Timestamp(ts_raw)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return LiquidityStressOutput(
        timestamp=iso8601_z(ts),
        scope_type=str(row["scope_type"]),
        scope_id=str(row["scope_id"]),
        liquidity_index=float(row["liquidity_score"]),
        liquidity_label=str(row["liquidity_label"]),
        confidence=float(row["confidence"]),
        drivers=drivers,
        model_run_id=str(row["model_run_id"]),
        release_gate=bool(int(row["release_gate"])),
        artifact_hash=str(row["artifact_hash"]),
        metadata=metadata,
    )


def output_to_dict(output: LiquidityStressOutput) -> dict[str, Any]:
    """JSON-serialisable dict form of :class:`LiquidityStressOutput`.

    Drivers become a list (not a tuple) so :func:`json.dumps` does not
    require ``default=str``.
    """
    out = asdict(output)
    out["drivers"] = list(output.drivers)
    # v1.6.0 (REVIEW_DEEP_V1_5_2.md F2): coerce metadata to plain
    # dict so callers can mutate downstream without tripping the
    # read-only ``_ReadOnlyMetadata`` (dict subclass) guard.
    out["metadata"] = dict(out.get("metadata", {}) or {})
    return out
