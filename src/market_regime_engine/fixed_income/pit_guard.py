# SPDX-License-Identifier: Apache-2.0
"""Point-in-time guards for Fixed-Income feature pipelines.

Per ``MRE_FIXED_INCOME_AGENT.md §"PIT guard rules"``, any feature
builder must reject (or flag) rows where:

- ``source_timestamp > decision_timestamp``;
- ``vintage_timestamp > decision_timestamp``;
- ``feature_computed_from_outcome_after_decision == True`` (this last
  case is enforced at the feature-construction layer; the helper below
  covers the two timestamp rails).

The scalar :func:`assert_pit_safe` is the canonical row-level helper
used by FI scoring entry points. :func:`audit_pit_dataframe` is the
vectorised batch variant for use in label-construction and warehouse
integrity audits. Both helpers accept timezone-naive *and*
timezone-aware timestamps and normalise to UTC for the comparison so
mixed-tz inputs from upstream vendors do not silently fail open.

This module is intentionally minimal — the rich audit report shape
lives in :mod:`market_regime_engine.leakage_checks`; this module only
exposes the FI-facing surface (``PitViolationError``, ``assert_pit_safe``,
``audit_pit_dataframe``) so FI callers can import without dragging the
full contract-issues module into the scoring hot path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd


class PitViolationError(RuntimeError):
    """Raised when a feature/decision/vintage timestamp tuple breaks PIT.

    Subclasses :class:`RuntimeError` rather than :class:`ValueError`
    because PIT violations are operational defects (an upstream
    contract was broken), not value-domain validation errors. The FI
    scoring entry points catch this at the boundary and flip
    ``release_gate=False`` rather than returning a degraded score.
    """


@dataclass(frozen=True)
class PitAuditReport:
    """Result of :func:`audit_pit_dataframe`.

    ``violations`` is the subset of input rows that broke at least one
    PIT rail. ``violation_count`` is precomputed for cheap status
    checks; ``status`` is ``"PASS"`` when ``violation_count == 0``.
    """

    rows: int
    violation_count: int
    violations: pd.DataFrame
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return "PASS" if self.violation_count == 0 else "FAIL"

    @property
    def passed(self) -> bool:
        return self.violation_count == 0


def _to_utc(value: Any) -> pd.Timestamp:
    """Coerce ``value`` to a UTC-normalised ``pd.Timestamp``.

    Timezone-naive inputs are interpreted as UTC; timezone-aware
    inputs are converted to UTC. This mirrors the AGENT.md PIT-guard
    contract: callers may pass either form, and the comparison is
    always performed in UTC.
    """
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def assert_pit_safe(
    feature_timestamp: Any,
    decision_timestamp: Any,
    vintage_timestamp: Any = None,
    *,
    label: str = "feature",
) -> None:
    """Raise :class:`PitViolationError` if PIT rails are broken.

    Two rails are enforced:

    1. ``feature_timestamp <= decision_timestamp`` — using a feature
       observed after the decision is the canonical lookahead leak.
    2. ``vintage_timestamp <= decision_timestamp`` when the caller
       supplies a vintage — using a feature derived from a vintage
       that did not yet exist at decision time is the
       revision-leakage failure mode (``vintage_date > as_of`` in the
       macro PIT machinery).

    ``label`` is interpolated into the error message so callers can
    surface which feature or row triggered the rail without wrapping
    the call in their own ``try/except``.
    """
    feat_utc = _to_utc(feature_timestamp)
    decision_utc = _to_utc(decision_timestamp)
    if feat_utc > decision_utc:
        raise PitViolationError(
            f"PIT violation for {label}: feature_timestamp={feat_utc.isoformat()} "
            f"is after decision_timestamp={decision_utc.isoformat()}"
        )
    if vintage_timestamp is not None:
        vintage_utc = _to_utc(vintage_timestamp)
        if vintage_utc > decision_utc:
            raise PitViolationError(
                f"PIT violation for {label}: vintage_timestamp={vintage_utc.isoformat()} "
                f"is after decision_timestamp={decision_utc.isoformat()}"
            )


def _normalise_ts_column(series: pd.Series) -> pd.Series:
    """Vectorised UTC-normalisation for a timestamp column.

    Mixed tz-aware / tz-naive rows are bridged by localising the
    naive subset to UTC before converting the aware subset. The
    resulting series is uniformly tz-aware UTC, suitable for direct
    ``Series > Series`` comparison.
    """
    return pd.to_datetime(series, errors="coerce", utc=True)


def audit_pit_dataframe(
    df: pd.DataFrame,
    decision_timestamp_col: str,
    feature_timestamp_col: str,
    vintage_timestamp_col: str | None = None,
) -> PitAuditReport:
    """Vectorised batch PIT audit; returns offending rows.

    Mirrors the rails enforced by :func:`assert_pit_safe`. The returned
    :class:`PitAuditReport` carries:

    - ``rows`` — total rows audited;
    - ``violation_count`` — number of rows that broke at least one
      rail;
    - ``violations`` — a DataFrame of the offending rows with a
      ``pit_violation_reason`` column describing which rail(s) fired
      (``"feature_after_decision"`` and/or ``"vintage_after_decision"``);
    - ``details`` — per-rail counts for the operator playbook.

    The audit is read-only; the input ``df`` is not mutated. Empty
    inputs return a clean ``PASS`` report so the helper composes
    cleanly with empty-frame upstream filters.
    """
    if df is None or df.empty:
        empty = pd.DataFrame()
        return PitAuditReport(
            rows=0,
            violation_count=0,
            violations=empty,
            details={"feature_after_decision": 0, "vintage_after_decision": 0},
        )

    if decision_timestamp_col not in df.columns:
        raise KeyError(f"decision_timestamp column {decision_timestamp_col!r} missing from frame")
    if feature_timestamp_col not in df.columns:
        raise KeyError(f"feature_timestamp column {feature_timestamp_col!r} missing from frame")

    decision = _normalise_ts_column(df[decision_timestamp_col])
    feature = _normalise_ts_column(df[feature_timestamp_col])
    feat_after = feature > decision
    feat_after = feat_after.fillna(False)

    if vintage_timestamp_col is not None and vintage_timestamp_col in df.columns:
        vintage = _normalise_ts_column(df[vintage_timestamp_col])
        vintage_after = (vintage > decision).fillna(False)
    else:
        vintage_after = pd.Series(False, index=df.index)

    violators = feat_after | vintage_after
    offenders = df.loc[violators].copy()
    if not offenders.empty:
        reasons = []
        for is_feat_after, is_vint_after in zip(
            feat_after[violators].tolist(),
            vintage_after[violators].tolist(),
            strict=False,
        ):
            parts: list[str] = []
            if is_feat_after:
                parts.append("feature_after_decision")
            if is_vint_after:
                parts.append("vintage_after_decision")
            reasons.append(",".join(parts))
        offenders["pit_violation_reason"] = reasons

    details = {
        "feature_after_decision": int(feat_after.sum()),
        "vintage_after_decision": int(vintage_after.sum()),
    }
    return PitAuditReport(
        rows=len(df),
        violation_count=len(offenders),
        violations=offenders,
        details=details,
    )


def utc_now() -> datetime:
    """Convenience helper for FI ingest/scoring code paths.

    Kept here (rather than in a separate ``timestamps.py``) so the FI
    PIT layer has one canonical place to mint ``decision_timestamp``
    values when the caller is not provided one explicitly. PR-3 may
    extract this into ``fixed_income/timestamps.py`` once UTC ingest
    enforcement lands.
    """
    return pd.Timestamp.now(tz="UTC").to_pydatetime()


__all__ = [
    "PitAuditReport",
    "PitViolationError",
    "assert_pit_safe",
    "audit_pit_dataframe",
    "utc_now",
]
