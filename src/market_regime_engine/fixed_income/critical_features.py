# SPDX-License-Identifier: Apache-2.0
"""v1.5.1 (PR-9 FIX 8): strict missing-data fail-closed contract.

Each fixed-income scorer maintains a hardcoded set of
:class:`CriticalFeature` values that, when missing from the input,
force:

- ``release_gate = False``
- ``confidence_score <= 0.5``
- a fail-closed label override (``"UNCERTAIN"`` for credit,
  ``"NO_DECISION"`` for liquidity)

REGARDLESS of the active :class:`NanPolicy`. This overrides the
:func:`_apply_nan_policy` re-weighting behaviour so a missing canonical
input cannot be silently re-weighted away in production.

The set of *optional* features that may legitimately be missing
(e.g. the ETF dislocation proxy on illiquid sectors) is unaffected
and keeps the legacy behaviour.

Contract reference: ``docs/V1_5_FIXED_INCOME_RCIE.md`` §"Fail-Closed
Contract".
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import pandas as pd

from market_regime_engine.fixed_income.schemas import CriticalFeature

__all__ = [
    "CREDIT_CRITICAL_COLUMNS",
    "CRITICAL_LABEL_CREDIT",
    "CRITICAL_LABEL_LIQUIDITY",
    "LIQUIDITY_CRITICAL_COLUMNS",
    "CriticalFeatureAudit",
    "evaluate_critical_features",
]


# v1.5.1 (PR-9 FIX 8): per-scorer hard-coded contracts.
#
# The keys are pivoted-feature-name strings (the columns produced by
# the wide-pivot used inside the scorer). The values are the
# corresponding :class:`CriticalFeature` enum member so we surface a
# stable identifier in ``metadata.critical_features_missing``.
CREDIT_CRITICAL_COLUMNS: dict[str, CriticalFeature] = {
    # cdx_ig_5y is the canonical credit-bond-spread proxy in the
    # current build (per ``COMPONENT_FEATURES["spreads"]``).
    "cdx_ig_5y": CriticalFeature.CREDIT_BOND_SPREAD,
    # cdx_hy_5y is the canonical CDS / OAS proxy.
    "cdx_hy_5y": CriticalFeature.CREDIT_CDS_BASIS,
}

LIQUIDITY_CRITICAL_COLUMNS: dict[str, CriticalFeature] = {
    "bid_ask_width": CriticalFeature.LIQUIDITY_BIDASK,
    "quotes_received": CriticalFeature.LIQUIDITY_RFQ_RESPONSE,
}

# Fail-closed label overrides (str literals so we don't fight the
# scorer enums; the credit and liquidity scorers already accept any
# string for their ``regime_label`` / ``liquidity_label`` fields).
CRITICAL_LABEL_CREDIT: str = "UNCERTAIN"
CRITICAL_LABEL_LIQUIDITY: str = "NO_DECISION"


@dataclass(frozen=True)
class CriticalFeatureAudit:
    """Outcome of :func:`evaluate_critical_features`.

    Attributes:
        missing: the :class:`CriticalFeature` members that were absent
            or all-NaN in the input.
        missing_columns: the offending column names (intentionally
            kept alongside the enum so the audit log surfaces both
            the canonical identifier and the implementation-specific
            column name).
        fail_closed: True iff at least one critical feature was
            missing and the caller must therefore force
            ``release_gate=False``, ``confidence<=0.5``, and the
            fail-closed label.
    """

    missing: tuple[CriticalFeature, ...]
    missing_columns: tuple[str, ...]

    @property
    def fail_closed(self) -> bool:
        return bool(self.missing)


def _column_is_present_and_observed(wide: pd.DataFrame, column: str) -> bool:
    """Return True iff ``column`` is in the frame AND has at least one non-NaN observation.

    Mirrors the semantic of :func:`_apply_nan_policy`'s "no observation
    anywhere in the lookback window" check so a column that is
    technically present in the schema but entirely NaN is still
    treated as missing.
    """
    if column not in wide.columns:
        return False
    series = wide[column]
    if series.empty:
        return False
    return bool(series.dropna().size > 0)


def evaluate_critical_features(
    wide: pd.DataFrame | None,
    *,
    contract: dict[str, CriticalFeature] | Iterable[tuple[str, CriticalFeature]],
) -> CriticalFeatureAudit:
    """Audit ``wide`` against the per-scorer critical-feature contract.

    ``wide`` is the post-pivot wide frame the scorer feeds into the
    component reducers. ``contract`` is a mapping of column name to
    :class:`CriticalFeature` (or an iterable of (column, feature)
    pairs).

    The audit walks each contract entry and flags a missing critical
    feature when the column is absent from the frame OR every
    observation in the lookback is NaN. The caller MUST treat a
    fail-closed audit as binding: re-weighting around the missing
    feature is explicitly forbidden by the v1.5.1 fail-closed
    contract.
    """
    items: Sequence[tuple[str, CriticalFeature]]
    if isinstance(contract, dict):
        items = list(contract.items())
    else:
        items = list(contract)

    if wide is None or wide.empty:
        # Empty input: every critical feature is missing.
        return CriticalFeatureAudit(
            missing=tuple(feature for _, feature in items),
            missing_columns=tuple(column for column, _ in items),
        )

    missing_pairs: list[tuple[str, CriticalFeature]] = []
    for column, feature in items:
        if not _column_is_present_and_observed(wide, column):
            missing_pairs.append((column, feature))

    return CriticalFeatureAudit(
        missing=tuple(feature for _, feature in missing_pairs),
        missing_columns=tuple(column for column, _ in missing_pairs),
    )
