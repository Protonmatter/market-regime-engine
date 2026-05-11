# SPDX-License-Identifier: Apache-2.0
"""Per-column NaN cleaning policy (REVIEW.md §3.2 ASK-5 / §3.1 AF-8).

The historical v1.0-v1.4 cleaner used at :mod:`bocpd`,
:mod:`frontier.bayesian_msvar`, and :mod:`frontier.gp_cpd` was the
single-line expression::

    frame.replace([inf, -inf], NaN).ffill().fillna(0.0)

That rule is correct for monthly macro: a missing inflation print
should not collapse a forward fill, and the seed 0.0 only fires on
the very first warmup row. The same rule is *wrong* for Fixed-Income
features: a CUSIP that did not trade yesterday, or a curve tenor that
did not print at the asof, must NOT silently report "zero spread" —
that pretends data exists when it does not, and the downstream
release-gate then promotes a fake "Normal" regime.

The PR-3 fix is a per-column policy with four modes and one default::

    NAN_TO_ZERO            — back-compat default; matches old cleaner.
    NAN_TO_LAST_VALID      — forward-fill only; no zero seed.
    NAN_DROPS_ROW          — drop rows whose required cols stay NaN.
    NAN_FAILS_PIT_AUDIT    — raise PitAuditFailure so release_gate=False.

Inf / -Inf are always coerced to NaN first, identical to the legacy
cleaner. The default ``NAN_TO_ZERO`` preserves the bocpd / MSVAR /
GP-CPD numerics bit-for-bit so existing tests keep passing
(``test_bocpd*`` / ``test_bayesian_msvar*`` / ``test_v1_2_frontier``
all pin against the old numbers).
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_FI_POLICY",
    "DEFAULT_MACRO_POLICY",
    "NanPolicy",
    "PitAuditFailure",
    "clean_with_policy",
]


class PitAuditFailure(RuntimeError):
    """Raised by :func:`clean_with_policy` under :attr:`NanPolicy.NAN_FAILS_PIT_AUDIT`.

    Subclasses :class:`RuntimeError` so callers that want fail-closed
    semantics can ``except PitAuditFailure`` cleanly without
    accidentally catching value-domain errors. FI scoring entry points
    catch this at the boundary and flip ``release_gate=False`` on the
    governance output rather than returning a fake-but-bounded score.
    """


class NanPolicy(str, Enum):
    """Per-column NaN policy.

    Values are simple string slugs so a ``column_policies`` mapping can
    be loaded from JSON / YAML config without an enum import.
    """

    NAN_TO_ZERO = "nan_to_zero"
    NAN_TO_LAST_VALID = "nan_to_last_valid"
    NAN_DROPS_ROW = "nan_drops_row"
    NAN_FAILS_PIT_AUDIT = "nan_fails_pit_audit"


# Documented presets so callers don't repeat the literal.
DEFAULT_MACRO_POLICY: NanPolicy = NanPolicy.NAN_TO_ZERO
DEFAULT_FI_POLICY: NanPolicy = NanPolicy.NAN_FAILS_PIT_AUDIT


def _coerce_inf(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce ``+/-inf`` to ``NaN`` on every column, matching the legacy cleaner."""
    return frame.replace([np.inf, -np.inf], np.nan)


def _apply_column(series: pd.Series, policy: NanPolicy) -> tuple[pd.Series, bool, list[int]]:
    """Apply ``policy`` to ``series``; return (cleaned, audit_failure, drop_indices).

    ``drop_indices`` is the list of positional indices the caller must
    drop when the policy is :attr:`NanPolicy.NAN_DROPS_ROW`. The caller
    aggregates per-column drop sets to produce a single row-drop mask.
    ``audit_failure`` is True for :attr:`NanPolicy.NAN_FAILS_PIT_AUDIT`
    when at least one NaN remains after inf coercion.
    """
    if policy is NanPolicy.NAN_TO_ZERO:
        # Legacy behaviour: ffill the early-window NaNs, then seed the
        # very first NaN streak with 0. This is bit-for-bit identical
        # to ``s.ffill().fillna(0.0)`` and pins the existing bocpd /
        # MSVAR / GP-CPD numerical traces.
        return series.ffill().fillna(0.0), False, []
    if policy is NanPolicy.NAN_TO_LAST_VALID:
        return series.ffill(), False, []
    if policy is NanPolicy.NAN_DROPS_ROW:
        mask = series.isna()
        return series, False, list(np.flatnonzero(mask.to_numpy()))
    if policy is NanPolicy.NAN_FAILS_PIT_AUDIT:
        has_nan = bool(series.isna().any())
        return series, has_nan, []
    raise ValueError(f"unknown NanPolicy: {policy!r}")


def clean_with_policy(
    frame: pd.DataFrame,
    *,
    default_policy: NanPolicy = NanPolicy.NAN_TO_ZERO,
    column_policies: Mapping[str, NanPolicy] | None = None,
) -> pd.DataFrame:
    """Apply per-column NaN policy after coercing ``+/-inf`` to NaN.

    ``+/-inf`` are always coerced to NaN first (matching the legacy
    cleaner) so any downstream NaN-policy branch fires uniformly
    regardless of whether the upstream column contains floating-point
    overflow.

    Per-column policies are resolved as::

        column_policies.get(col, default_policy)

    so callers can override only the columns that need a different
    rail; everything else inherits the default. When ``default_policy``
    is :attr:`NanPolicy.NAN_TO_ZERO` and ``column_policies`` is empty
    or ``None``, the output is bit-for-bit identical to::

        frame.replace([inf, -inf], NaN).ffill().fillna(0.0)

    which is what the existing bocpd / MSVAR / GP-CPD tests pin
    against.

    :attr:`NanPolicy.NAN_FAILS_PIT_AUDIT` raises :class:`PitAuditFailure`
    listing every offending column. The message includes column names
    and (truncated) row counts so the operator can wire the failure
    straight into the FI evidence-pack audit log.

    :attr:`NanPolicy.NAN_DROPS_ROW` removes a row when *any* of its
    drop-policy columns contain NaN; columns under a different policy
    contribute their cleaned value to the surviving rows.
    """
    if frame is None or frame.empty:
        return frame if frame is not None else pd.DataFrame()

    cleaned = _coerce_inf(frame).copy()
    audit_failures: list[str] = []
    drop_positions: set[int] = set()

    overrides = dict(column_policies or {})
    for col in cleaned.columns:
        policy = overrides.get(col, default_policy)
        result, audit_fail, drop_indices = _apply_column(cleaned[col], policy)
        cleaned[col] = result
        if audit_fail:
            audit_failures.append(col)
        if drop_indices:
            drop_positions.update(drop_indices)

    if audit_failures:
        raise PitAuditFailure(
            "NAN_FAILS_PIT_AUDIT triggered; missing inputs in columns: "
            f"{sorted(audit_failures)!r} (rows={len(cleaned)})"
        )

    if drop_positions:
        keep_mask = np.ones(len(cleaned), dtype=bool)
        for pos in drop_positions:
            if 0 <= pos < len(cleaned):
                keep_mask[pos] = False
        cleaned = cleaned.iloc[keep_mask]

    return cleaned
