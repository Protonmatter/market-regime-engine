from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from typing import Any, Literal

import pandas as pd

from market_regime_engine.frontier.experimental import require_frontier_experimental
from market_regime_engine.package_boundary import BoundaryName, resolve_boundary

# v1.4.1 (item F): the release-gate profile defaults flipped from
# permissive (v1.2.1 baseline) to production. Sentinel objects below
# let ``evaluate_release_gate`` detect "the caller passed nothing"
# vs. "the caller deliberately passed the v1.2.1 looser value" so the
# resolution priority can apply the strict profile defaults without
# clobbering an explicit operator override. Use ``is`` comparison on
# these sentinels — equality would defeat the point.

_UNSET: Any = object()


def production_profile() -> dict[str, Any]:
    """Production-grade release-gate kwargs (v1.3 item G).

    Returns the kwargs dictionary that turns on every governance gate
    by default. Use via ``evaluate_release_gate(..., profile="production")``
    or ``mre release-gate --profile production`` from the CLI.

    v1.4.1 (item F) makes ``profile="production"`` the *default*
    when no explicit profile is passed and ``MRE_ENV`` is unset (or
    set to ``production``); see :func:`evaluate_release_gate` for the
    full resolution order.

    The defaults below are intentionally strict:

    - ``min_confidence=0.75`` (vs the v1.2.1 baseline 0.55)
    - ``max_major_drift=0`` and ``max_severe_drift=0`` — any drift
      blocks release.
    - ``require_mcs_membership=True`` — Hansen MCS evidence is mandatory.
    - ``min_coverage=0.85`` and ``coverage_drop_pp=0.02`` — conformal
      coverage drift > 2pp blocks release.
    - ``promotion_method="mcs"`` — Hansen MCS is the only accepted
      promotion criterion in this profile.

    v1.5 PR-5 (Q-5, deep research): DSR / PBO rails:

    - ``min_dsr=0.5`` — Deflated Sharpe ≥ 0.5 (Bailey–López de Prado);
      blocks release when ``confidence`` frame carries ``dsr`` column.
    - ``max_pbo=0.05`` — Probability of Backtest Overfitting ≤ 5%
      (BBLZ); blocks release when ``confidence`` frame carries ``pbo``.

    Both columns are **optional** in the confidence frame; absence
    skips the rail entirely so legacy callers without DSR/PBO emit the
    same release decision as v1.4.1.
    """
    return {
        "min_confidence": 0.75,
        "max_major_drift": 0,
        "max_severe_drift": 0,
        "require_mcs_membership": True,
        "min_coverage": 0.85,
        "coverage_drop_pp": 0.02,
        "promotion_method": "mcs",
        "min_dsr": 0.5,
        "max_pbo": 0.05,
        # v1.5.1 (PR-9 FIX 4b): Brier / ECE rails on the production
        # profile per Naeini et al. (2015). Both are optional columns
        # on the confidence frame; absence skips the rail so legacy
        # callers stay green.
        "max_brier": 0.20,
        "max_ece": 0.05,
        # v1.5.1 (PR-9 FIX 4c): TCA-lift rail. When set, the gate
        # consumes a ``tca_lift`` column on the confidence frame
        # (per-row dict {regime: {p_value, effect_size, n}}) and
        # requires at least one regime where p_value <= max_tca_p AND
        # |effect_size| >= min_tca_effect.
        "max_tca_p": 0.05,
        "min_tca_effect": 0.2,
    }


def certification_profile() -> dict[str, Any]:
    """Audit/certification release-gate kwargs.

    This profile is stricter than ``production``: validation artifacts are
    mandatory rather than opportunistic. It is intended for external review,
    model-risk sign-off, and stable-core promotion packs.
    """
    out = production_profile()
    out.update(
        {
            "require_validation_artifacts": True,
            "require_model_card": True,
            "require_evidence_hmac": True,
            "min_regime_sample_size": 30,
            "min_tca_lift_n": 30,
        }
    )
    return out


def default_profile() -> dict[str, Any]:
    """v1.2.1-baseline release-gate kwargs (the looser defaults).

    Returns the kwargs that the v1.4.0 CLI applied when ``mre
    release-gate`` was invoked with no flags. Available as an explicit
    opt-back-in for legitimate dev / staging environments via
    ``profile="default"`` or ``MRE_ENV=dev``.

    v1.5 PR-5: ``min_dsr=None`` and ``max_pbo=None`` so the dev profile
    does not gate on validation primitives that the macro engine does
    not emit.
    """
    return {
        "min_confidence": 0.55,
        "max_major_drift": 0,
        "max_severe_drift": 0,
        "require_mcs_membership": False,
        "min_coverage": None,
        "coverage_drop_pp": 0.05,
        "promotion_method": "mcs",
        "min_dsr": None,
        "max_pbo": None,
        # PR-9 FIX 4: dev / staging skip the calibration + TCA-lift
        # rails so a partial-input integration test does not have to
        # plumb every column.
        "max_brier": None,
        "max_ece": None,
        "max_tca_p": None,
        "min_tca_effect": None,
    }


_PROFILE_FACTORIES: dict[str, Any] = {
    "production": production_profile,
    "certification": certification_profile,
    "default": default_profile,
}


def _finite_float(value: Any) -> float | None:
    """Return a finite float or ``None`` without raising on bad payloads."""

    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _expand_confidence_metadata(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    if frame is None or frame.empty or "metadata_json" not in frame.columns:
        return frame
    out = frame.copy()
    metadata_rows: list[dict[str, Any]] = []
    keys: set[str] = set()
    for raw in out["metadata_json"].tolist():
        payload: dict[str, Any] = {}
        if isinstance(raw, Mapping):
            payload = dict(raw)
        else:
            try:
                if raw is not None and not pd.isna(raw):
                    parsed = json.loads(str(raw))
                    if isinstance(parsed, Mapping):
                        payload = dict(parsed)
            except Exception:
                payload = {}
        metadata_rows.append(payload)
        keys.update(str(k) for k in payload)
    for key in sorted(keys):
        if key not in out.columns:
            out[key] = [payload.get(key) for payload in metadata_rows]
    return out


def _coerce_tca_lift_payload(value: Any) -> dict[str, dict[str, float]]:
    """Decode a ``tca_lift`` confidence-frame cell to the canonical dict.

    The release gate must fail closed on malformed TCA evidence. Older code
    accepted ``dict(v)`` directly, which could raise on non-mapping nested
    values and turn a gate decision into an unhandled exception. This helper
    now drops invalid regime rows and only returns rows with finite ``p_value``,
    finite ``effect_size``, and finite ``n`` values. An empty return value is
    interpreted by the caller as missing/invalid TCA evidence.
    """

    parsed: Any = value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
    if not isinstance(parsed, dict):
        return {}

    out: dict[str, dict[str, float]] = {}
    for regime, row in parsed.items():
        if not isinstance(row, dict):
            continue
        p_value = _finite_float(row.get("p_value"))
        effect_size = _finite_float(row.get("effect_size"))
        n_value = _finite_float(row.get("n"))
        if p_value is None or effect_size is None or n_value is None:
            continue
        out[str(regime)] = {
            "p_value": p_value,
            "effect_size": effect_size,
            "n": n_value,
            **{str(k): v for k, v in row.items() if k not in {"p_value", "effect_size", "n"}},
        }
    return out


def _latest_row(frame: pd.DataFrame, *, date_col: str = "date") -> pd.Series:
    """Return the latest row by date when available, otherwise final row.

    Release-gate callers sometimes pass single-row synthetic frames without a
    date column. Certification hardening should not turn those frames into a
    KeyError; it should evaluate the available evidence deterministically.
    """

    if frame is None or frame.empty:
        raise ValueError("_latest_row requires a non-empty frame")
    if date_col not in frame.columns:
        return frame.iloc[-1]
    tmp = frame.copy()
    parsed = pd.to_datetime(tmp[date_col], utc=True, errors="coerce")
    if parsed.notna().any():
        return tmp.loc[parsed.sort_values(kind="mergesort").index].iloc[-1]
    return tmp.sort_values(date_col, kind="mergesort").iloc[-1]


def _resolve_profile(profile: str | None) -> str:
    """v1.4.1 (item F) profile resolution priority.

    1. Explicit ``profile=`` arg wins (when set to one of the known
       profiles).
    2. Else ``MRE_ENV`` env var: ``MRE_ENV=production`` →
       ``"production"``; ``MRE_ENV=dev`` → ``"default"``.
    3. Else fall back to ``"production"``.

    Returns the resolved profile name (always one of the keys of
    :data:`_PROFILE_FACTORIES`). The function does not raise on an
    unknown explicit profile — that is the caller's job; raise from
    :func:`evaluate_release_gate` so the error surfaces with full
    context.
    """
    if profile and profile != "":
        return profile
    env = os.environ.get("MRE_ENV", "").strip().lower()
    if env == "production":
        return "production"
    if env in {"dev", "development", "staging", "test"}:
        return "default"
    return "production"


def evaluate_release_gate(
    confidence: pd.DataFrame | None = None,
    drift: pd.DataFrame | None = None,
    invalidation: pd.DataFrame | None = None,
    promotion: pd.DataFrame | None = None,
    min_confidence: Any = _UNSET,
    max_major_drift: Any = _UNSET,
    max_severe_drift: Any = _UNSET,
    *,
    require_mcs_membership: Any = _UNSET,
    min_coverage: Any = _UNSET,
    coverage_report: pd.DataFrame | None = None,
    coverage_alpha: float = 0.10,
    coverage_drop_pp: Any = _UNSET,
    promotion_method: Any = _UNSET,
    e_value_log: pd.DataFrame | None = None,
    e_value_alpha: float = 0.05,
    profile: Literal["default", "production", "certification"] | None = None,
    gate_boundary: BoundaryName = "stable_core",
    min_dsr: Any = _UNSET,
    max_pbo: Any = _UNSET,
    max_brier: Any = _UNSET,
    max_ece: Any = _UNSET,
    max_tca_p: Any = _UNSET,
    min_tca_effect: Any = _UNSET,
    require_validation_artifacts: Any = _UNSET,
    require_model_card: Any = _UNSET,
    require_evidence_hmac: Any = _UNSET,
    min_regime_sample_size: Any = _UNSET,
    min_tca_lift_n: Any = _UNSET,
) -> pd.DataFrame:
    """Evaluate the release gate.

    ``require_mcs_membership`` enforces that the latest promotion row
    carries ``mcs_evidence == "in_set"``; the v1.2.1 default behaviour
    was ``False`` (no enforcement).

    ``min_coverage`` gates on per-bucket realized coverage drift. When
    set, ``coverage_report`` must be a frame with a ``coverage`` column
    (one row per ``(target, horizon, bucket)``); the gate blocks if the
    worst realized coverage drops more than ``coverage_drop_pp`` below
    ``1 - coverage_alpha``.

    ``promotion_method="e_values"`` (v1.2 frontier path) replaces the
    Hansen MCS evidence requirement with a sequential e-value safe-test
    gate. The caller passes the latest ``e_value_log`` frame (rows
    with ``e_value`` and ``decision`` columns); the gate fires when at
    least one row has ``e_value >= 1 / e_value_alpha`` AND
    ``decision == "promote"``.

    v1.3 (item G) introduced ``profile="production"`` to merge in the
    :func:`production_profile` defaults. v1.4.1 (item F) **flips the
    default profile to ``production``** with the following resolution
    priority:

    1. Explicit ``profile=`` argument wins.
    2. Else ``MRE_ENV`` env var: ``MRE_ENV=production`` →
       ``"production"`` profile; ``MRE_ENV=dev`` (or ``development``,
       ``staging``, ``test``) → ``"default"`` profile.
    3. Else fall back to ``"production"``.

    Explicit ``min_confidence`` / ``require_mcs_membership`` /
    ``min_coverage`` / etc. kwargs always override the
    profile-resolved defaults so a caller that needs to relax a
    single rail in production can do so explicitly.

    **Breaking change in v1.4.1:** v1.4.0-and-earlier callers who
    invoked ``evaluate_release_gate()`` with no kwargs and no env
    vars got the v1.2.1 looser baseline (``min_confidence=0.55``,
    ``require_mcs_membership=False``, ``min_coverage=None``); v1.4.1
    callers in the same configuration get the production profile.
    Use ``profile="default"`` or ``MRE_ENV=dev`` to opt back into the
    looser thresholds.

    ``gate_boundary`` separates the production-certifiable stable core
    from explicit-opt-in frontier research. The stable core defaults to
    the production profile; ``experimental_frontier`` requires
    ``MRE_ENABLE_EXPERIMENTAL_FRONTIER=1`` and records a non-production
    boundary marker in ``metadata_json``.
    """
    boundary = resolve_boundary(gate_boundary)
    if boundary.requires_experimental_flag:
        require_frontier_experimental(
            f"release gate boundary {boundary.name!r} is experimental and not production-eligible"
        )
    profile_for_resolution = profile
    if profile_for_resolution is None and boundary.name == "experimental_frontier":
        profile_for_resolution = boundary.default_gate_profile
    resolved_profile = _resolve_profile(profile_for_resolution)
    try:
        factory = _PROFILE_FACTORIES[resolved_profile]
    except KeyError as exc:
        raise ValueError(
            f"Unknown release-gate profile: {resolved_profile!r}; valid profiles are {sorted(_PROFILE_FACTORIES)}."
        ) from exc
    prof_kwargs: dict[str, Any] = factory()

    # v1.4.1 (item F): apply profile-resolved defaults only when the
    # caller passed _UNSET. An explicit kwarg always wins. We
    # deliberately use ``is _UNSET`` so callers can pass legitimate
    # falsy values (``min_coverage=None``, ``max_major_drift=0``,
    # ``require_mcs_membership=False``) and have those preserved.
    if min_confidence is _UNSET:
        min_confidence = float(prof_kwargs.get("min_confidence", 0.55))
    else:
        min_confidence = float(min_confidence)
    if max_major_drift is _UNSET:
        max_major_drift = int(prof_kwargs.get("max_major_drift", 0))
    else:
        max_major_drift = int(max_major_drift)
    if max_severe_drift is _UNSET:
        max_severe_drift = int(prof_kwargs.get("max_severe_drift", 0))
    else:
        max_severe_drift = int(max_severe_drift)
    if require_mcs_membership is _UNSET:
        require_mcs_membership = bool(prof_kwargs.get("require_mcs_membership", False))
    else:
        require_mcs_membership = bool(require_mcs_membership)
    if min_coverage is _UNSET:
        cov = prof_kwargs.get("min_coverage")
        min_coverage = float(cov) if cov is not None else None
    else:
        min_coverage = float(min_coverage) if min_coverage is not None else None
    if coverage_drop_pp is _UNSET:
        coverage_drop_pp = float(prof_kwargs.get("coverage_drop_pp", 0.05))
    else:
        coverage_drop_pp = float(coverage_drop_pp)
    if promotion_method is _UNSET:
        promotion_method = str(prof_kwargs.get("promotion_method", "mcs"))
    else:
        promotion_method = str(promotion_method)
    # v1.5 PR-5: optional DSR / PBO thresholds. ``None`` opts out of the
    # rail; the production profile defaults to ``0.5`` / ``0.05``.
    if min_dsr is _UNSET:
        prof_dsr = prof_kwargs.get("min_dsr")
        min_dsr = float(prof_dsr) if prof_dsr is not None else None
    else:
        min_dsr = float(min_dsr) if min_dsr is not None else None
    if max_pbo is _UNSET:
        prof_pbo = prof_kwargs.get("max_pbo")
        max_pbo = float(prof_pbo) if prof_pbo is not None else None
    else:
        max_pbo = float(max_pbo) if max_pbo is not None else None
    # v1.5.1 (PR-9 FIX 4): Brier / ECE / TCA-lift rails. Each is
    # optional; absence on the confidence frame skips the rail.
    if max_brier is _UNSET:
        prof_brier = prof_kwargs.get("max_brier")
        max_brier = float(prof_brier) if prof_brier is not None else None
    else:
        max_brier = float(max_brier) if max_brier is not None else None
    if max_ece is _UNSET:
        prof_ece = prof_kwargs.get("max_ece")
        max_ece = float(prof_ece) if prof_ece is not None else None
    else:
        max_ece = float(max_ece) if max_ece is not None else None
    if max_tca_p is _UNSET:
        prof_tca_p = prof_kwargs.get("max_tca_p")
        max_tca_p = float(prof_tca_p) if prof_tca_p is not None else None
    else:
        max_tca_p = float(max_tca_p) if max_tca_p is not None else None
    if min_tca_effect is _UNSET:
        prof_tca_eff = prof_kwargs.get("min_tca_effect")
        min_tca_effect = float(prof_tca_eff) if prof_tca_eff is not None else None
    else:
        min_tca_effect = float(min_tca_effect) if min_tca_effect is not None else None
    if require_validation_artifacts is _UNSET:
        require_validation_artifacts = bool(prof_kwargs.get("require_validation_artifacts", False))
    else:
        require_validation_artifacts = bool(require_validation_artifacts)
    if require_model_card is _UNSET:
        require_model_card = bool(prof_kwargs.get("require_model_card", False))
    else:
        require_model_card = bool(require_model_card)
    if require_evidence_hmac is _UNSET:
        require_evidence_hmac = bool(prof_kwargs.get("require_evidence_hmac", False))
    else:
        require_evidence_hmac = bool(require_evidence_hmac)
    if min_regime_sample_size is _UNSET:
        prof_regime_n = prof_kwargs.get("min_regime_sample_size")
        min_regime_sample_size = int(prof_regime_n) if prof_regime_n is not None else None
    else:
        min_regime_sample_size = int(min_regime_sample_size) if min_regime_sample_size is not None else None
    if min_tca_lift_n is _UNSET:
        prof_tca_n = prof_kwargs.get("min_tca_lift_n")
        min_tca_lift_n = int(prof_tca_n) if prof_tca_n is not None else None
    else:
        min_tca_lift_n = int(min_tca_lift_n) if min_tca_lift_n is not None else None
    conf_val = 0.0
    conf_grade = "F"
    date = None
    dsr_val: float | None = None
    pbo_val: float | None = None
    brier_val: float | None = None
    ece_val: float | None = None
    tca_lift_raw: Any = None
    certification_fields: dict[str, Any] = {}
    confidence = _expand_confidence_metadata(confidence)
    if confidence is not None and not confidence.empty:
        latest = _latest_row(confidence)
        conf_val = float(latest.get("confidence", 0.0))
        conf_grade = str(latest.get("grade", "unknown"))
        date = latest.get("date")
        # DSR / PBO are optional columns on the confidence frame; coerce
        # to None when absent or NaN so the rail can decide whether to
        # fire below.
        if "dsr" in confidence.columns:
            raw = latest.get("dsr")
            dsr_val = float(raw) if raw is not None and not pd.isna(raw) else None
        if "pbo" in confidence.columns:
            raw = latest.get("pbo")
            pbo_val = float(raw) if raw is not None and not pd.isna(raw) else None
        # v1.5.1 (PR-9 FIX 4b): Brier / ECE columns are optional.
        if "brier" in confidence.columns:
            raw = latest.get("brier")
            brier_val = float(raw) if raw is not None and not pd.isna(raw) else None
        if "ece" in confidence.columns:
            raw = latest.get("ece")
            ece_val = float(raw) if raw is not None and not pd.isna(raw) else None
        # v1.5.1 (PR-9 FIX 4c): TCA-lift column carries the
        # ``tca_lift_test`` output dict. We accept either a dict or a
        # JSON string for back-compat with warehouse round-trips.
        if "tca_lift" in confidence.columns:
            tca_lift_raw = latest.get("tca_lift")
        certification_fields = {str(k): latest.get(k) for k in confidence.columns}
    severe_drift = 0
    major_drift = 0
    max_psi = 0.0
    if drift is not None and not drift.empty:
        latest_date = drift["date"].max()
        d = drift[drift["date"] == latest_date]
        if "status" in d:
            severe_drift = int((d["status"] == "severe").sum())
            major_drift = int((d["status"] == "major").sum())
        max_psi = float(d["psi"].max()) if "psi" in d and not d.empty else 0.0
        date = date or latest_date
    high_triggers = 0
    trigger_names = []
    if invalidation is not None and not invalidation.empty:
        inv = invalidation[invalidation["date"] == invalidation["date"].max()]
        high = inv[
            (inv["status"].astype(str).str.lower() == "active")
            & (inv["severity"].astype(str).str.lower().isin(["high", "critical"]))
        ]
        high_triggers = len(high)
        trigger_names = list(high["trigger"].astype(str).head(10))
        date = date or inv["date"].max()
    promoted = True
    mcs_evidence = "absent"
    if promotion is not None and not promotion.empty and "promoted" in promotion.columns:
        # v1.5 (PR-1 AF-7 / P2): filter to the latest date before
        # checking ``.any()`` so a stale promoted=True row does not
        # keep satisfying the gate forever. When a ``date`` column is
        # absent we fall back to the legacy ``.any()`` behaviour so
        # tests with no date column stay green.
        promotion_latest = promotion
        if "date" in promotion.columns and not promotion["date"].isna().all():
            latest_date = promotion["date"].max()
            promotion_latest = promotion.loc[promotion["date"] == latest_date]
        promoted = bool(promotion_latest["promoted"].astype(bool).any())
        if "mcs_evidence" in promotion_latest.columns:
            evidence_values = promotion_latest["mcs_evidence"].astype(str).tolist()
            if "in_set" in evidence_values:
                mcs_evidence = "in_set"
            elif "out_of_set" in evidence_values:
                mcs_evidence = "out_of_set"
            else:
                mcs_evidence = "absent"

    # v1.5 (PR-1 AF-6 / P0): empty / all-NaN coverage_report no longer
    # silently passes the gate. The pre-v1.5 code only set
    # ``worst_coverage`` when the dropna() series was non-empty,
    # which meant the gate skipped the coverage rail entirely. Now
    # we append ``coverage_data_missing`` to reasons and emit
    # worst_coverage=NaN so the operator can see the missing rail.
    worst_coverage: float | None = None
    coverage_missing = False
    if coverage_report is not None and not coverage_report.empty and "coverage" in coverage_report.columns:
        cov_series = coverage_report["coverage"].astype(float).dropna()
        if cov_series.empty:
            coverage_missing = True
        else:
            worst_coverage = float(cov_series.min())
    elif min_coverage is not None:
        # An explicit ``min_coverage`` (or the production default) was
        # requested but the caller supplied no coverage frame at all
        # — treat as missing data so the rail fires.
        coverage_missing = True

    def _field_truthy(name: str) -> bool:
        raw = certification_fields.get(name)
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            return False
        if isinstance(raw, bool):
            return raw
        text = str(raw).strip().lower()
        return text in {"1", "true", "yes", "y", "pass", "passed", "ok"}

    def _field_present(name: str) -> bool:
        raw = certification_fields.get(name)
        if raw is None:
            return False
        try:
            if pd.isna(raw):
                return False
        except Exception:
            pass
        return str(raw).strip() != ""

    reasons = []
    if conf_val < min_confidence:
        reasons.append(f"confidence_below_{min_confidence:.2f}")
    if severe_drift > max_severe_drift:
        reasons.append("severe_drift_detected")
    if major_drift > max_major_drift:
        reasons.append("major_drift_detected")
    if high_triggers > 0:
        reasons.append("high_invalidation_trigger_active")
    if not promoted:
        reasons.append("no_promoted_model")
    if promotion_method == "mcs":
        if require_mcs_membership and mcs_evidence != "in_set":
            reasons.append(f"mcs_evidence_{mcs_evidence}")
    elif promotion_method == "e_values":
        require_frontier_experimental(
            "promotion_method='e_values' is an experimental frontier promotion path; production defaults use MCS"
        )
        # v1.2 SafeTestPromotion gate. Require at least one row in
        # ``e_value_log`` whose ``e_value >= 1/alpha`` AND ``decision``
        # signals "promote". Anything else blocks the gate.
        e_threshold = 1.0 / max(float(e_value_alpha), 1e-12)
        passed = False
        if e_value_log is not None and not e_value_log.empty and "e_value" in e_value_log.columns:
            evs = e_value_log.copy()
            evs["e_value"] = evs["e_value"].astype(float)
            # v1.5 (PR-1 AF-14 / P0): when promotion_method=="e_values"
            # the ``decision`` column is mandatory. Pre-v1.5 silently
            # defaulted to all "promote", which was permissive in a
            # way that operators could not detect. Raise so the
            # missing column surfaces immediately.
            if "decision" not in evs.columns:
                raise ValueError("e_value_log missing 'decision' column")
            decisions = evs["decision"]
            passed = bool(((evs["e_value"] >= e_threshold) & (decisions.astype(str).str.lower() == "promote")).any())
        if not passed:
            reasons.append("e_value_gate_not_fired")
    if coverage_missing:
        reasons.append("coverage_data_missing")
    if min_coverage is not None and worst_coverage is not None:
        coverage_floor = (1.0 - coverage_alpha) - coverage_drop_pp
        if worst_coverage < min(min_coverage, coverage_floor):
            reasons.append("conformal_coverage_below_floor")
    # v1.5 PR-5 (Q-5 / deep research): DSR / PBO rails. Optional —
    # absence of the column on the confidence frame skips the rail so
    # legacy callers without DSR/PBO emit the same decision as v1.4.1.
    if min_dsr is not None and dsr_val is not None and dsr_val < min_dsr:
        reasons.append(f"deflated_sharpe_below_{min_dsr:.2f}")
    if max_pbo is not None and pbo_val is not None and pbo_val > max_pbo:
        reasons.append(f"probability_of_overfit_above_{max_pbo:.2f}")
    # v1.5.1 (PR-9 FIX 4b): Brier / ECE rails. Optional. When the
    # threshold is configured AND the confidence frame carries a finite
    # value we fire the rail.
    if max_brier is not None and brier_val is not None and brier_val > max_brier:
        reasons.append(f"brier_above_{max_brier:.2f}")
    if max_ece is not None and ece_val is not None and ece_val > max_ece:
        reasons.append(f"ece_above_{max_ece:.2f}")
    # v1.5.1 (PR-9 FIX 4c): TCA-lift rail. The gate FAILS unless at
    # least one regime in ``tca_lift`` reports
    # ``p_value <= max_tca_p`` AND ``|effect_size| >= min_tca_effect``.
    #
    # v1.6.0 fail-closed fix (REVIEW_DEEP_V1_5_2.md A4 / Finding #8): when
    # ``_coerce_tca_lift_payload`` returns an empty dict (bad JSON, wrong
    # type, no rows), the prior code skipped the rail entirely so the
    # release gate silently passed. The contract is that an enabled TCA
    # rail with a missing-or-invalid payload must FAIL the gate; only a
    # well-formed payload with no significant segment may emit the
    # ``tca_lift_no_significant_segment`` rejection reason.
    lift: dict[str, dict[str, float]] = {}
    if (
        max_tca_p is not None
        and min_tca_effect is not None
        and tca_lift_raw is not None
    ):
        lift = _coerce_tca_lift_payload(tca_lift_raw)
        if not lift:
            reasons.append("tca_lift_missing_or_invalid")
        else:
            # Positive lift is required: high-confidence executions must have
            # lower observed slippage than low-confidence executions. A large
            # negative effect is adverse evidence, not a reason to certify.
            best_passes = any(
                float(row["p_value"]) <= max_tca_p
                and float(row["effect_size"]) >= min_tca_effect
                for row in lift.values()
            )
            if not best_passes:
                reasons.append("tca_lift_no_significant_segment")
                reasons.append("tca_lift_no_positive_significant_segment")
            if min_tca_lift_n is not None:
                underpowered = [
                    name
                    for name, row in lift.items()
                    if int(float(row.get("n", 0.0))) < min_tca_lift_n
                ]
                if underpowered:
                    reasons.append("tca_lift_underpowered_segments")

    if require_validation_artifacts:
        required_numeric = {
            "dsr": dsr_val,
            "pbo": pbo_val,
            "brier": brier_val,
            "ece": ece_val,
        }
        for name, value in required_numeric.items():
            if value is None or not pd.notna(value) or not math.isfinite(float(value)):
                # Preserve the original missing-artifact reason for downstream
                # dashboards while adding the stricter non-finite reason.
                reasons.append(f"certification_missing_{name}")
                reasons.append(f"certification_missing_or_nonfinite_{name}")
        for flag in ("pit_leakage_passed", "walk_forward_passed"):
            if not _field_truthy(flag):
                reasons.append(f"certification_{flag}_false_or_missing")
        for artifact in ("validation_artifact_hash",):
            if not _field_present(artifact):
                reasons.append(f"certification_missing_{artifact}")
        if min_regime_sample_size is not None:
            raw_n = certification_fields.get("min_regime_sample_size")
            if raw_n is None:
                raw_n = certification_fields.get("min_regime_n")
            try:
                regime_n = int(raw_n)
            except Exception:
                regime_n = 0
            if regime_n < min_regime_sample_size:
                reasons.append("certification_regime_sample_size_below_floor")
        if tca_lift_raw is None:
            reasons.append("certification_missing_tca_lift")
    if require_model_card and not _field_present("model_card_path"):
        reasons.append("certification_missing_model_card")
    if require_evidence_hmac and not _field_present("evidence_pack_hmac"):
        reasons.append("certification_missing_evidence_pack_hmac")
    approved = len(reasons) == 0
    return pd.DataFrame(
        [
            {
                "date": date,
                "approved": bool(approved),
                "decision": "release" if approved else "hold",
                "confidence": conf_val,
                "confidence_grade": conf_grade,
                "severe_drift": severe_drift,
                "major_drift": major_drift,
                "max_psi": max_psi,
                "high_invalidation_triggers": high_triggers,
                "active_trigger_names": ",".join(trigger_names),
                "reasons": ",".join(reasons) if reasons else "passed",
                "metadata_json": json.dumps(
                    {
                        "package_boundary": boundary.name,
                        "production_eligible": boundary.production_eligible,
                        "requires_experimental_flag": boundary.requires_experimental_flag,
                        "certification_profile": resolved_profile == "certification",
                        "validation_artifacts_required": bool(require_validation_artifacts),
                        "min_regime_sample_size": min_regime_sample_size,
                        "min_tca_lift_n": min_tca_lift_n,
                        "max_brier": max_brier,
                        "max_ece": max_ece,
                        "max_tca_p": max_tca_p,
                        "min_tca_effect": min_tca_effect,
                    },
                    sort_keys=True,
                ),
                "mcs_evidence": mcs_evidence,
                "worst_coverage": worst_coverage if worst_coverage is not None else float("nan"),
                # v1.5 (PR-1 ASK-7 / P2): surface which profile drove
                # the threshold selection. Persisted into the
                # release_gates warehouse table by storage.write_release_gates.
                "resolved_profile": resolved_profile,
                # v1.5 PR-5: surface the DSR / PBO values used (NaN when
                # the rail was skipped). The release_gates warehouse
                # writer ignores unknown columns so this is back-compat
                # safe for callers that have not migrated yet.
                "dsr": dsr_val if dsr_val is not None else float("nan"),
                "pbo": pbo_val if pbo_val is not None else float("nan"),
                "brier": brier_val if brier_val is not None else float("nan"),
                "ece": ece_val if ece_val is not None else float("nan"),
            }
        ]
    )
