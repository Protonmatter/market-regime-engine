from __future__ import annotations

import os
from typing import Any, Literal

import pandas as pd

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
    """
    return {
        "min_confidence": 0.75,
        "max_major_drift": 0,
        "max_severe_drift": 0,
        "require_mcs_membership": True,
        "min_coverage": 0.85,
        "coverage_drop_pp": 0.02,
        "promotion_method": "mcs",
    }


def default_profile() -> dict[str, Any]:
    """v1.2.1-baseline release-gate kwargs (the looser defaults).

    Returns the kwargs that the v1.4.0 CLI applied when ``mre
    release-gate`` was invoked with no flags. Available as an explicit
    opt-back-in for legitimate dev / staging environments via
    ``profile="default"`` or ``MRE_ENV=dev``.
    """
    return {
        "min_confidence": 0.55,
        "max_major_drift": 0,
        "max_severe_drift": 0,
        "require_mcs_membership": False,
        "min_coverage": None,
        "coverage_drop_pp": 0.05,
        "promotion_method": "mcs",
    }


_PROFILE_FACTORIES: dict[str, Any] = {
    "production": production_profile,
    "default": default_profile,
}


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
    profile: Literal["default", "production"] | None = None,
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
    """
    resolved_profile = _resolve_profile(profile)
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
    conf_val = 0.0
    conf_grade = "F"
    date = None
    if confidence is not None and not confidence.empty:
        latest = confidence.sort_values("date").iloc[-1]
        conf_val = float(latest.get("confidence", 0.0))
        conf_grade = str(latest.get("grade", "unknown"))
        date = latest.get("date")
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
        promoted = bool(promotion["promoted"].astype(bool).any())
        if "mcs_evidence" in promotion.columns:
            evidence_values = promotion["mcs_evidence"].astype(str).tolist()
            if "in_set" in evidence_values:
                mcs_evidence = "in_set"
            elif "out_of_set" in evidence_values:
                mcs_evidence = "out_of_set"
            else:
                mcs_evidence = "absent"

    worst_coverage: float | None = None
    if coverage_report is not None and not coverage_report.empty and "coverage" in coverage_report.columns:
        cov_series = coverage_report["coverage"].astype(float).dropna()
        if not cov_series.empty:
            worst_coverage = float(cov_series.min())

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
        # v1.2 SafeTestPromotion gate. Require at least one row in
        # ``e_value_log`` whose ``e_value >= 1/alpha`` AND ``decision``
        # signals "promote". Anything else blocks the gate.
        e_threshold = 1.0 / max(float(e_value_alpha), 1e-12)
        passed = False
        if e_value_log is not None and not e_value_log.empty and "e_value" in e_value_log.columns:
            evs = e_value_log.copy()
            evs["e_value"] = evs["e_value"].astype(float)
            decisions = evs.get("decision", pd.Series(["promote"] * len(evs)))
            passed = bool(((evs["e_value"] >= e_threshold) & (decisions.astype(str).str.lower() == "promote")).any())
        if not passed:
            reasons.append("e_value_gate_not_fired")
    if min_coverage is not None and worst_coverage is not None:
        coverage_floor = (1.0 - coverage_alpha) - coverage_drop_pp
        if worst_coverage < min(min_coverage, coverage_floor):
            reasons.append("conformal_coverage_below_floor")
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
                "metadata_json": "{}",
                "mcs_evidence": mcs_evidence,
                "worst_coverage": worst_coverage if worst_coverage is not None else float("nan"),
            }
        ]
    )
