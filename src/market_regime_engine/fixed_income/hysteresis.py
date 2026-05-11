# SPDX-License-Identifier: Apache-2.0
"""Generic hysteresis ("Schmitt trigger") label classifier for FI signals.

Per ``MRE_FIXED_INCOME_INSTRUCTIONS.md §6 retro`` and the v1.5 plan §4,
the credit-regime and liquidity-stress labels must not flip on every
tick when the score oscillates near a bucket boundary. The asymmetric
``(enter_threshold, exit_threshold)`` band design — also known as a
Schmitt trigger — makes each label sticky inside its band while still
respecting the sharp-bucket mapping for cold-start (``prev_label is
None``).

The helper here is policy-free: callers supply

    * ``bands`` — a mapping of label → (enter_lower, exit_upper),
      where ``enter_lower`` is the minimum score required to retain
      the label from below and ``exit_upper`` is the minimum score
      that pushes the label out to a higher tier. ``None`` on either
      bound means unbounded (terminal labels).
    * ``sharp_fallback`` — a callable mapping ``float -> Label`` used
      when ``prev_label is None`` or when the score has clearly moved
      out of ``prev_label``'s band.

The shared algorithm guarantees:

    1. ``prev_label is None`` → sharp-bucket fallback (preserves the
       existing PR-3 contract for cold-start consumers).
    2. ``prev_label`` is sticky inside its band (Schmitt trigger).
    3. Outside the band, the sharp bucket re-classifies the score
       (which lets us collapse multiple bucket transitions in one
       step — e.g. a CRISIS-to-NORMAL move skipping the intermediate
       labels).

The implementation is intentionally minimal so the credit-regime and
liquidity-stress modules can re-export ``classify_with_hysteresis``
with their own typed bands without keeping two algorithms in sync.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeVar

L = TypeVar("L")


def apply_hysteresis(
    score: float,
    *,
    prev_label: L | None,
    bands: Mapping[L, tuple[float | None, float | None]],
    sharp_fallback: Callable[[float], L],
) -> L:
    """Return the hysteresis-aware label for ``score``.

    Parameters
    ----------
    score:
        Composite score in ``[0, 100]``.
    prev_label:
        Previous label, or ``None`` to short-circuit to the sharp
        bucket mapping (cold-start path).
    bands:
        ``{label: (enter_lower, exit_upper)}`` per the module
        docstring.
    sharp_fallback:
        ``score -> label`` used when ``prev_label`` is ``None`` or the
        score has clearly moved out of ``prev_label``'s band.

    Notes
    -----
    The "stay" predicate is::

        (enter is None or score >= enter) and (exit is None or score < exit)

    Both bounds are evaluated as half-open ``[enter, exit)`` so the
    boundary scores (e.g. score=45 when MILD_STRESS exit is 45.0)
    fall *out* of the band — they trigger a re-classification rather
    than ambiguously sticking to the source label.
    """
    if prev_label is None:
        return sharp_fallback(float(score))
    band = bands.get(prev_label)
    if band is None:
        # Unknown prev_label — defensive fallback to sharp buckets so
        # a stale enum value cannot pin the result to an invalid state.
        return sharp_fallback(float(score))
    enter, exit_ = band
    s = float(score)
    in_lower = enter is None or s >= float(enter)
    in_upper = exit_ is None or s < float(exit_)
    if in_lower and in_upper:
        return prev_label
    return sharp_fallback(s)


__all__ = ["apply_hysteresis"]
