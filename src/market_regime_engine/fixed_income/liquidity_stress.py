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

This commit ships:

- :data:`HYSTERESIS_BANDS_LIQUIDITY` and :func:`classify_with_hysteresis`
  so a previous :class:`LiquidityLabel` can be applied as a Schmitt
  trigger on the new score (task C).

The full deterministic scorer, warehouse round-trip, and CLI/API
hooks land in subsequent commits within PR-4 (tasks A/B/F/G).
"""

from __future__ import annotations

from market_regime_engine.fixed_income.hysteresis import apply_hysteresis
from market_regime_engine.fixed_income.schemas import (
    LiquidityLabel,
    liquidity_label_from_score,
)

__all__ = [
    "HYSTERESIS_BANDS_LIQUIDITY",
    "classify_with_hysteresis",
]


# v1.5 (PR-4 task C): asymmetric (enter, exit) hysteresis bands per
# liquidity label so the bucket is "sticky" once entered. The
# convention mirrors the credit module (see ``credit_spread_regime``):
#
#     NORMAL: (None, 25)             — exit upward at 25.
#     MILD_STRESS: (20, 45)          — enter at 20+, exit upward at 45.
#     ELEVATED_STRESS: (40, 65)
#     SEVERE_STRESS: (60, 85)
#     CRISIS_LIQUIDITY: (80, None)   — terminal upper edge.
#
# Cold start (``prev_label is None``) falls through to
# ``liquidity_label_from_score`` for full back-compat with consumers
# that have no priors.
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
    :func:`liquidity_label_from_score`.

    ``prev_label`` is sticky inside its band; outside the band the
    score re-classifies via the sharp bucket mapping.
    """
    return apply_hysteresis(
        float(score),
        prev_label=prev_label,
        bands=HYSTERESIS_BANDS_LIQUIDITY,
        sharp_fallback=liquidity_label_from_score,
    )
