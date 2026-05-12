# SPDX-License-Identifier: Apache-2.0
"""PR-3 credit-spread regime scorer (deterministic composite).

Per ``MRE_FIXED_INCOME_AGENT.md §"PR 3"``: ship the explainable
deterministic composite first, model-based scorers later. The
deterministic scorer computes five component scores (each on 0-100
where higher = more risk-off) from FI features, then a weighted
composite. Every output carries the v1.5 governance triple
(``model_run_id``, ``release_gate``, ``artifact_hash``) so the
downstream API / report writer / evidence-pack tooling reads from one
contract.

Component design::

    score_treasury_curve   slope inversion (10Y-2Y) + curvature
    score_spreads          OAS percentile vs 2y rolling history
    score_cds              CDX HY 5Y percentile vs 2y rolling history
    score_volatility       VIX + MOVE z-scores
    score_etf_dislocation  |ETF premium/discount| percentile

Default weights {treasury_curve: 0.15, spreads: 0.30, cds: 0.25,
volatility: 0.20, etf_dislocation: 0.10} sum to 1.0 and match
INSTRUCTIONS.md §6.1's emphasis on credit spreads. Callers can pass
``weights={...}``; weights are normalised to sum to 1.0 internally.

Confidence rule: ``1.0 - (fraction_of_components_with_missing_input_data)``,
capped at ``0.5`` when ``release_gate=False`` per AGENT.md
non-negotiable 8.

Drivers: top-2 component names by ``|score - 50.0|`` (most-extreme
deviation from the neutral midline). Ties are broken in component-
declaration order.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

import pandas as pd

from market_regime_engine.fixed_income.hashing import canonical_sha256
from market_regime_engine.fixed_income.hysteresis import apply_hysteresis
from market_regime_engine.fixed_income.pit_guard import (
    PitViolationError,
)
from market_regime_engine.fixed_income.critical_features import (
    CREDIT_CRITICAL_COLUMNS,
    CRITICAL_LABEL_CREDIT,
    evaluate_critical_features,
)
from market_regime_engine.fixed_income.schemas import (
    CreditRegimeOutput,
    RegimeLabel,
    regime_label_from_score,
)
from market_regime_engine.fixed_income.timestamps import iso8601_z, to_utc
from market_regime_engine.frontier.data_cleaning import NanPolicy, PitAuditFailure

log = logging.getLogger(__name__)

__all__ = [
    "COMPONENT_FEATURES",
    "DEFAULT_WEIGHTS",
    "HYSTERESIS_BANDS_CREDIT",
    "classify_with_hysteresis",
    "latest_credit_regime_score",
    "latest_credit_regime_score_identity",
    "score_credit_regime",
    "write_credit_regime_score",
]


# ---------------------------------------------------------------------------
# composite configuration
# ---------------------------------------------------------------------------


COMPONENT_FEATURES: dict[str, tuple[str, ...]] = {
    # 10Y-2Y slope + 2*5Y-2Y-10Y curvature on the UST curve.
    "treasury_curve": ("ust_slope", "ust_curvature"),
    # AGENT.md OAS / Z-spread proxy via CDX.IG 5Y (PR-4 will fan out per-rating).
    "spreads": ("cdx_ig_5y",),
    # AGENT.md credit-vol proxy via CDX.HY 5Y.
    "cds": ("cdx_hy_5y",),
    # AGENT.md MOVE + VIX volatility composite.
    "volatility": ("move", "vix"),
    # AGENT.md ETF premium/discount proxy.
    "etf_dislocation": ("etf_prem_disc",),
}

DEFAULT_WEIGHTS: dict[str, float] = {
    "treasury_curve": 0.15,
    "spreads": 0.30,
    "cds": 0.25,
    "volatility": 0.20,
    "etf_dislocation": 0.10,
}

# v1.5 (PR-4 task C): asymmetric (enter, exit) hysteresis bands per
# label so the regime label is "sticky" near a bucket boundary rather
# than flipping on every tick. The first slot is the score *minimum*
# required to keep the label from below; the second is the score at
# which the label exits to the next-higher tier. ``None`` means an
# unbounded edge.
#
# Cold start (no prev label) falls through to ``regime_label_from_score``
# so PR-3 callers without ``prev_label=`` keep the v1.5 PR-3 contract
# bit-for-bit.
HYSTERESIS_BANDS_CREDIT: dict[RegimeLabel, tuple[float | None, float | None]] = {
    RegimeLabel.RISK_ON_COMPRESSION: (None, 25.0),
    RegimeLabel.NORMAL_LIQUIDITY: (20.0, 45.0),
    RegimeLabel.WATCH_TRANSITION: (40.0, 65.0),
    RegimeLabel.RISK_OFF_HIGH_RISK_AVERSION: (60.0, 85.0),
    RegimeLabel.CRISIS_SEVERE_DISLOCATION: (80.0, None),
}

_NEUTRAL_SCORE: float = 50.0
_DEFAULT_PERCENTILE: float = 50.0


def classify_with_hysteresis(score: float, prev_label: RegimeLabel | None) -> RegimeLabel:
    """Map ``score`` to a :class:`RegimeLabel` with asymmetric hysteresis.

    ``prev_label is None`` → sharp-bucket fallback via
    :func:`regime_label_from_score` (preserves the PR-3 contract for
    cold-start callers).

    ``prev_label`` is sticky inside its hysteresis band; outside the
    band the score re-classifies via the sharp bucket mapping (so a
    multi-tier move — e.g. CRISIS → NORMAL — does not require
    intermediate steps).
    """
    return apply_hysteresis(
        float(score),
        prev_label=prev_label,
        bands=HYSTERESIS_BANDS_CREDIT,
        sharp_fallback=regime_label_from_score,
    )


# ---------------------------------------------------------------------------
# normalisation helpers (deterministic, no random state)
# ---------------------------------------------------------------------------


def _percentile_score(series: pd.Series, latest: float, *, direction: int = 1) -> float:
    """Empirical percentile of ``latest`` in ``series`` → 0-100.

    ``direction=+1``: higher value → higher score (risk-off direction
    for spreads / CDS / VIX).
    ``direction=-1``: lower value → higher score (risk-off direction
    for slope: a more-inverted curve is risk-off).
    """
    if series is None or len(series) == 0:
        return _DEFAULT_PERCENTILE
    finite = series.dropna()
    if finite.empty:
        return _DEFAULT_PERCENTILE
    # Strict "less than or equal" so the latest value sits at its own quantile.
    pct = float((finite <= latest).mean() * 100.0)
    if direction < 0:
        pct = 100.0 - pct
    return max(0.0, min(100.0, pct))


def _zscore_sigmoid(value: float, mean: float, std: float) -> float:
    """Map a value to 0-100 via z-score sigmoid (50 = z=0).

    Sigmoid keeps the score bounded without ad-hoc clipping; the
    inflection at z=0 maps to 50 so the neutral case is the midline.
    """
    if std is None or not math.isfinite(std) or std == 0:
        return _NEUTRAL_SCORE
    z = (value - mean) / std
    # ``100 / (1 + exp(-z))``: z=0 → 50, z=2 → ~88, z=-2 → ~12.
    return float(100.0 / (1.0 + math.exp(-z)))


def _slope_score(slope_series: pd.Series, latest_slope: float | None) -> float | None:
    """Lower slope → higher score (inverted curve = risk-off)."""
    if latest_slope is None or pd.isna(latest_slope):
        return None
    return _percentile_score(slope_series, float(latest_slope), direction=-1)


def _curvature_score(series: pd.Series, latest: float | None) -> float | None:
    """Curvature is bounded both sides; use z-sigmoid of absolute deviation."""
    if latest is None or pd.isna(latest):
        return None
    finite = series.dropna()
    if finite.empty:
        return _NEUTRAL_SCORE
    mean = float(finite.mean())
    std = float(finite.std(ddof=0))
    # Absolute deviation is the risk-off signal: extreme positive or
    # extreme negative curvature both reflect dislocation.
    return _zscore_sigmoid(abs(float(latest) - mean), 0.0, std if std > 0 else 1.0)


def _spreads_score(series: pd.Series, latest: float | None) -> float | None:
    if latest is None or pd.isna(latest):
        return None
    return _percentile_score(series, float(latest), direction=+1)


def _cds_score(series: pd.Series, latest: float | None) -> float | None:
    if latest is None or pd.isna(latest):
        return None
    return _percentile_score(series, float(latest), direction=+1)


def _vol_score(series: pd.Series, latest: float | None) -> float | None:
    if latest is None or pd.isna(latest):
        return None
    finite = series.dropna()
    if finite.empty:
        return _NEUTRAL_SCORE
    return _zscore_sigmoid(float(latest), float(finite.mean()), float(finite.std(ddof=0)))


def _etf_score(series: pd.Series, latest: float | None) -> float | None:
    if latest is None or pd.isna(latest):
        return None
    abs_series = series.abs()
    return _percentile_score(abs_series, abs(float(latest)), direction=+1)


# Component → (latest_extractor_name, scorer_function).
def _treasury_curve_component(wide: pd.DataFrame) -> float | None:
    slope = _series(wide, "ust_slope")
    curv = _series(wide, "ust_curvature")
    if slope.dropna().empty and curv.dropna().empty:
        return None
    slope_score = _slope_score(slope, _latest(slope))
    curv_score = _curvature_score(curv, _latest(curv))
    parts = [s for s in (slope_score, curv_score) if s is not None]
    if not parts:
        return None
    return float(sum(parts) / len(parts))


def _spreads_component(wide: pd.DataFrame) -> float | None:
    series = _series(wide, "cdx_ig_5y")
    if series.dropna().empty:
        return None
    return _spreads_score(series, _latest(series))


def _cds_component(wide: pd.DataFrame) -> float | None:
    series = _series(wide, "cdx_hy_5y")
    if series.dropna().empty:
        return None
    return _cds_score(series, _latest(series))


def _volatility_component(wide: pd.DataFrame) -> float | None:
    move = _series(wide, "move")
    vix = _series(wide, "vix")
    if move.dropna().empty and vix.dropna().empty:
        return None
    move_score = _vol_score(move, _latest(move))
    vix_score = _vol_score(vix, _latest(vix))
    parts = [s for s in (move_score, vix_score) if s is not None]
    if not parts:
        return None
    return float(sum(parts) / len(parts))


def _etf_component(wide: pd.DataFrame) -> float | None:
    series = _series(wide, "etf_prem_disc")
    if series.dropna().empty:
        return None
    return _etf_score(series, _latest(series))


_COMPONENT_FUNCS: dict[str, Any] = {
    "treasury_curve": _treasury_curve_component,
    "spreads": _spreads_component,
    "cds": _cds_component,
    "volatility": _volatility_component,
    "etf_dislocation": _etf_component,
}


# ---------------------------------------------------------------------------
# pivot helpers
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


def _pivot_features(features: pd.DataFrame) -> pd.DataFrame:
    """Long → wide pivot keyed by ``date`` × ``feature_name``.

    Multiple observations for the same (date, feature_name) collapse
    via ``last`` (a deterministic policy: PR-3 feeders aggregate per
    day, so this only fires on adversarial inputs).
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


# ---------------------------------------------------------------------------
# weights / drivers helpers
# ---------------------------------------------------------------------------


def _normalise_weights(weights: Mapping[str, float] | None) -> dict[str, float]:
    """Return a copy of ``weights`` summing to 1.0 (defaults preserved).

    ``None`` or all-zero falls back to :data:`DEFAULT_WEIGHTS`. Missing
    components default to 0.0 so a caller passing
    ``weights={"spreads": 1.0}`` puts all weight on credit spreads.
    """
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


def _resolve_model_run_id(model_run_id: str | None, profile: str) -> str:
    if model_run_id and model_run_id.strip():
        return model_run_id
    # Deterministic-only callers can pass an explicit id; otherwise we
    # mint a profile-stamped UUID so the warehouse PK is unique.
    return f"credit_regime-{profile}-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# public entry points
# ---------------------------------------------------------------------------


def score_credit_regime(
    features: pd.DataFrame,
    *,
    asof: pd.Timestamp | str,
    model_run_id: str | None = None,
    release_gate: bool = True,
    profile: str = "production",
    weights: Mapping[str, float] | None = None,
    nan_policy_overrides: Mapping[str, NanPolicy] | None = None,
    prev_label: RegimeLabel | None = None,
) -> CreditRegimeOutput:
    """Compute the credit-regime score from a long-form feature frame.

    Parameters
    ----------
    features:
        Output of :func:`build_credit_features` — long DataFrame with
        ``["date", "feature_name", "value", "source_timestamp", "vintage_date"]``.
        Empty input returns a neutral 50.0 score with
        ``confidence=0.0`` and ``release_gate=False``.
    asof:
        Decision timestamp. Required and must be in UTC (string inputs
        are routed through :func:`to_utc`).
    model_run_id:
        Reproducibility id propagated into the output and warehouse
        row. When omitted, a profile-stamped UUID is minted.
    release_gate:
        Inbound governance flag. ``False`` caps ``confidence`` at 0.5
        (per AGENT.md non-negotiable 8). The output ``release_gate``
        also flips to ``False`` automatically when any feature row
        violates the NaN audit (i.e. fails the PIT-audit policy on a
        required feature).
    profile:
        Operating profile tag stored in ``metadata.profile``. The
        deterministic scorer treats every profile identically; future
        model-based scorers will branch here.
    weights:
        Optional component weights override. Missing components
        default to 0.0; the dict is normalised to sum to 1.0.
    nan_policy_overrides:
        Optional per-feature NaN-policy overrides forwarded to the
        cleaner. The default is taken from
        ``features.attrs["nan_policy"]`` (set by the builder) or
        :attr:`NanPolicy.NAN_FAILS_PIT_AUDIT` when missing.
    prev_label:
        Previous run's :class:`RegimeLabel` to apply asymmetric
        hysteresis to the new label (PR-4 task C). ``None`` (default)
        falls back to sharp-bucket mapping so PR-3 callers that did not
        pass ``prev_label=`` keep the original behaviour bit-for-bit.

    Returns
    -------
    CreditRegimeOutput
        Frozen dataclass per PR-1 schemas.

    Raises
    ------
    PitViolationError
        If any feature row's ``source_timestamp`` exceeds ``asof`` or
        the vintage rail fires.
    """
    asof_utc = _coerce_asof(asof)
    weights_norm = _normalise_weights(weights)

    # Per-row PIT enforcement (extra rail beyond the builder's check;
    # protects callers that synthesise feature frames bypassing the
    # builder).
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

    # Two-step NaN-policy enforcement:
    # 1. Column-level: every column present in the wide pivot must have
    #    at least one non-NaN observation in the lookback window.
    # 2. Component-level: under NAN_FAILS_PIT_AUDIT, any missing
    #    component is an audit failure (pivot_table silently drops
    #    all-NaN columns so step 1 alone cannot catch the case where
    #    every observation of a required feature was NaN).
    nan_policy = _resolve_nan_policy(features)
    try:
        _apply_nan_policy(wide, nan_policy=nan_policy, overrides=nan_policy_overrides)
    except PitAuditFailure:
        pit_audit_failed = True
        log.warning("credit regime PIT audit failed (column-level); flipping release_gate=False")
    if nan_policy is NanPolicy.NAN_FAILS_PIT_AUDIT and missing_components:
        pit_audit_failed = True
        log.warning(
            "credit regime PIT audit failed: missing components %r; flipping release_gate=False",
            missing_components,
        )

    # v1.5.1 (PR-9 FIX 8): critical-feature contract overrides the
    # NaN-policy re-weighting. A missing canonical credit spread or
    # CDS basis input forces release_gate=False and the
    # ``UNCERTAIN`` fail-closed label regardless of nan_policy.
    critical_audit = evaluate_critical_features(
        wide, contract=CREDIT_CRITICAL_COLUMNS
    )

    if not component_scores:
        # No features at all — neutral score, zero confidence, fail closed.
        regime_score = float(_NEUTRAL_SCORE)
        confidence = 0.0
        gate = False
        drivers: tuple[str, ...] = ()
        component_scores = {}
    else:
        regime_score = _compose(component_scores, weights_norm)
        confidence = _confidence(
            missing_components=missing_components, all_components=tuple(DEFAULT_WEIGHTS), release_gate=release_gate
        )
        gate = bool(release_gate) and not pit_audit_failed
        drivers = _drivers(component_scores)
        if not gate:
            confidence = min(confidence, 0.5)

    # v1.5.1 (PR-9 FIX 8): apply the fail-closed override AFTER the
    # legacy NaN-policy path so the regime label is overridden to
    # the explicit "UNCERTAIN" string and the gate is unconditionally
    # closed. The label override is intentionally a plain string and
    # bypasses :class:`RegimeLabel` so consumers can recognise the
    # critical-feature audit verdict at sight rather than mistaking
    # it for a normal "watch_transition" mid-band score.
    hysteresis_applied = prev_label is not None
    regime_label_enum = classify_with_hysteresis(regime_score, prev_label)
    regime_label = regime_label_enum.label
    if critical_audit.fail_closed:
        gate = False
        confidence = min(confidence, 0.5)
        regime_label = CRITICAL_LABEL_CREDIT
        log.warning(
            "credit regime critical-feature contract violated: missing=%r; "
            "flipping release_gate=False, label=%r",
            [feature.value for feature in critical_audit.missing],
            CRITICAL_LABEL_CREDIT,
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
        "critical_features_missing": [
            feature.value for feature in critical_audit.missing
        ],
        "critical_features_fail_closed": critical_audit.fail_closed,
    }

    timestamp_iso = iso8601_z(asof_utc)
    resolved_run_id = _resolve_model_run_id(model_run_id, profile)
    artifact_payload = {
        "timestamp": timestamp_iso,
        "regime_score": regime_score,
        "regime_label": regime_label,
        "confidence": confidence,
        "drivers": list(drivers),
        "component_scores": component_scores,
    }
    artifact_hash = canonical_sha256(artifact_payload)

    return CreditRegimeOutput(
        timestamp=timestamp_iso,
        regime_score=float(regime_score),
        regime_label=regime_label,
        confidence=float(confidence),
        drivers=drivers,
        component_scores=component_scores,
        model_run_id=resolved_run_id,
        release_gate=bool(gate),
        artifact_hash=artifact_hash,
        metadata=metadata,
    )


def _compose(scores: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """Weighted average over the components that actually have scores.

    Missing components do NOT count toward the denominator; the
    weights of the present components are re-normalised so the
    composite stays bounded in 0-100 even when some inputs are
    unavailable.
    """
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
    """Validate that the *latest non-NaN value per column* is present.

    The deterministic scorer uses ``_latest(series)`` which returns the
    last non-NaN value, so the audit must mirror that semantic: an
    "input missing" means *no observation anywhere in the lookback
    window*, not "the literal final timestamp doesn't carry this
    feature". Curve / CDS / vintage cadences rarely align at the exact
    same instant (EOD curve at 21:00 UTC vs. vintage observation at
    midnight) so the strict-last-row check would falsely fire on every
    real-data run.
    """
    if wide is None or wide.empty:
        if nan_policy is NanPolicy.NAN_FAILS_PIT_AUDIT:
            raise PitAuditFailure("credit regime features empty; cannot satisfy NAN_FAILS_PIT_AUDIT")
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
            f"NAN_FAILS_PIT_AUDIT triggered: feature(s) have no non-NaN observation in the lookback window: "
            f"{sorted(missing_cols)!r}"
        )


def _audit_pit(features: pd.DataFrame, *, asof: pd.Timestamp) -> None:
    """Row-level PIT enforcement on the long-form feature frame.

    v1.5 PR-8 (Tier-2 fix A2, REVIEW.md): vectorised replacement of
    the legacy ``features.iterrows()`` + per-row :func:`assert_pit_safe`
    loop. At 200k rows the old path spent hundreds of ms; the
    vectorised :func:`audit_pit_dataframe` runs in O(few ms) per
    column comparison and produces the same accept/reject semantics
    with a richer report. On any violation we raise
    :class:`PitViolationError` with the first offending row's label so
    operators see the same error shape as before.
    """
    if features.empty or "source_timestamp" not in features.columns:
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
            f"PIT audit failed: {report.violation_count} row(s) violate PIT "
            f"(asof={asof}, first violator label={label!r} "
            f"reason={reason!r})"
        )


# ---------------------------------------------------------------------------
# warehouse plumbing
# ---------------------------------------------------------------------------


def write_credit_regime_score(warehouse: Any, output: CreditRegimeOutput) -> int:
    """Persist a :class:`CreditRegimeOutput` row to ``credit_regime_scores``.

    Returns the number of rows written (always 1 on success). Uses
    :meth:`Warehouse.write_credit_regime_score` from PR-2.
    """
    row = {
        "model_run_id": output.model_run_id,
        "timestamp": output.timestamp,
        "regime_score": float(output.regime_score),
        "regime_label": output.regime_label,
        "confidence": float(output.confidence),
        "drivers_json": json.dumps(list(output.drivers)),
        "component_scores_json": json.dumps(output.component_scores, sort_keys=True),
        "release_gate": 1 if output.release_gate else 0,
        "artifact_hash": output.artifact_hash,
        "metadata_json": json.dumps(output.metadata, sort_keys=True, default=str),
    }
    return int(warehouse.write_credit_regime_score(pd.DataFrame([row])))


def latest_credit_regime_score(warehouse: Any) -> CreditRegimeOutput | None:
    """Read the most recent ``credit_regime_scores`` row → :class:`CreditRegimeOutput`.

    Returns ``None`` when the table is empty (caller decides whether to
    surface 503 / fail-closed).

    v1.5.1 (PR-9 FIX 2): prefer the indexed-SQL fast path on the
    :class:`Warehouse` so a 100k-row table reads in ≤ 5 ms (p99) rather
    than ≈ 80 ms full-table sweep. Falls back to the legacy
    ``read_credit_regime_scores()`` → ``.iloc[-1]`` path when the
    warehouse does not expose the new method (e.g. external test
    doubles), mirroring the
    :func:`latest_credit_regime_score_identity` pattern.
    """
    latest_fast = getattr(warehouse, "latest_credit_regime_score", None)
    if callable(latest_fast):
        try:
            fast_df = latest_fast()
        except Exception:  # pragma: no cover - fall back on backend miss
            fast_df = None
        if fast_df is not None and not fast_df.empty:
            return _row_to_output(fast_df.iloc[0])
        if fast_df is not None and fast_df.empty:
            return None
    df = warehouse.read_credit_regime_scores()
    if df is None or df.empty:
        return None
    row = df.iloc[-1]
    return _row_to_output(row)


def latest_credit_regime_score_identity(
    warehouse: Any,
) -> tuple[str, str, str] | None:
    """Return the ``(timestamp, model_run_id, artifact_hash)`` triple of
    the most recent ``credit_regime_scores`` row, or ``None`` when the
    table is empty.

    v1.5 PR-8 (Tier-2 fix A-Q1): two writes with the same canonical
    ``timestamp`` but different ``(model_run_id, artifact_hash)`` are
    legal — two backfills, two retraining runs at the same close, etc.
    Pre-fix the FastAPI per-process cache keyed on ``timestamp`` only,
    so the second read returned the FIRST run's artifact silently.
    The full triple is now the cache key so a new run with the same
    timestamp invalidates the prior entry.

    Implementation: prefer a parameterless SQL ``ORDER BY timestamp
    DESC, model_run_id DESC LIMIT 1`` against the backend so we don't
    pay a full-table read on every request (REVIEW.md §3.6 Tier-3 #9).
    Falls back to the legacy ``read_credit_regime_scores()`` →
    ``.iloc[-1]`` path when the backend doesn't expose ``read_sql``
    (e.g. test doubles).
    """
    backend = getattr(warehouse, "_backend", None)
    read_sql = getattr(backend, "read_sql", None) if backend is not None else None
    if callable(read_sql):
        try:
            df = read_sql(
                "SELECT timestamp, model_run_id, artifact_hash "
                "FROM credit_regime_scores "
                "ORDER BY timestamp DESC, model_run_id DESC "
                "LIMIT 1"
            )
        except Exception:  # pragma: no cover - SQL-side failure
            df = None
        if df is not None and not df.empty:
            row = df.iloc[0]
            return (
                _iso_z_or_str(row["timestamp"]),
                str(row["model_run_id"]),
                str(row["artifact_hash"]),
            )
        if df is not None and df.empty:
            return None
    # Fallback for in-memory / test-double warehouses.
    df = warehouse.read_credit_regime_scores()
    if df is None or df.empty:
        return None
    row = df.iloc[-1]
    return (
        _iso_z_or_str(row["timestamp"]),
        str(row["model_run_id"]),
        str(row["artifact_hash"]),
    )


def _iso_z_or_str(value: Any) -> str:
    """Normalise a DuckDB / SQLite timestamp to its string form.

    DuckDB returns ``Timestamp`` objects from ``read_sql``; SQLite
    returns the underlying ``str``. We coerce to ``str(...)`` so the
    cache key is canonical regardless of backend.
    """
    return str(value)


def _row_to_output(row: pd.Series) -> CreditRegimeOutput:
    drivers_json = row.get("drivers_json")
    component_json = row.get("component_scores_json")
    metadata_json = row.get("metadata_json")
    drivers = tuple(json.loads(drivers_json)) if drivers_json else ()
    component_scores = json.loads(component_json) if component_json else {}
    metadata = json.loads(metadata_json) if metadata_json else {}
    ts_raw = row["timestamp"]
    ts = pd.Timestamp(ts_raw)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return CreditRegimeOutput(
        timestamp=iso8601_z(ts),
        regime_score=float(row["regime_score"]),
        regime_label=str(row["regime_label"]),
        confidence=float(row["confidence"]),
        drivers=drivers,
        component_scores=component_scores,
        model_run_id=str(row["model_run_id"]),
        release_gate=bool(int(row["release_gate"])),
        artifact_hash=str(row["artifact_hash"]),
        metadata=metadata,
    )


def output_to_dict(output: CreditRegimeOutput) -> dict[str, Any]:
    """JSON-serialisable dict form of :class:`CreditRegimeOutput`.

    Convenience for the FastAPI handler and CLI ``--out-json`` writer.
    Drivers stay as a list (not a tuple) to round-trip through
    ``json.dumps`` without ``default=str``.
    """
    out = asdict(output)
    out["drivers"] = list(output.drivers)
    return out
