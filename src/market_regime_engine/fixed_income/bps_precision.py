# SPDX-License-Identifier: Apache-2.0
"""Decimal-precision basis-point arithmetic for FI TCA aggregations.

Per ``REVIEW.md §3.4 Q-6`` and PR-6 task B: FI TCA aggregations are
precision-sensitive at the daily-volume scale. A $25M trade at 0.5 bps
is exactly $1,250 of cost; under naive ``float64`` accumulation, summing
100k trades' worth of bps across a $1B day drifts by ~1e-9 bps per
trade due to representation error in 0.0001-magnitude numbers. The
drift is invisible per-trade and visible at scale.

This module provides the canonical Decimal helpers + module-level
context. Callers accumulate :class:`Decimal` throughout the
aggregation pipeline (``compute_tca_metrics_for_outcome``,
``aggregate_tca_by_regime``) and convert to ``float`` only at the
report boundary (warehouse write, JSON serialisation, API response).

Precision context (28 digits, ROUND_HALF_EVEN banker's rounding) is
chosen to match the v1.5 governance discipline:

- 28 digits comfortably covers $1B notional × 0.5 bps × 10 years of
  daily aggregation with headroom for numerator/denominator scaling.
- ROUND_HALF_EVEN avoids the upward bias of ROUND_HALF_UP that would
  otherwise compound across millions of TCA rows.

The context is *thread-local* by Python convention (``decimal.getcontext()``
returns the thread-local context) so concurrent FI scoring workers do
not collide. The :func:`with_tca_context` context manager scopes the
precision change to a block so other parts of the codebase that rely on
:class:`Decimal` defaults are not perturbed.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from decimal import (
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    localcontext,
)

__all__ = [
    "BPS_SCALE",
    "TCA_PRECISION_CONTEXT",
    "bps_aggregate_sum",
    "bps_arithmetic_mean",
    "decimal_to_float_for_report",
    "to_bps",
    "to_decimal",
    "with_tca_context",
]


# Module-level precision context for FI TCA arithmetic. Callers route
# through :func:`with_tca_context` to scope it; the value is exported as a
# constant so test code can introspect it without monkey-patching.
TCA_PRECISION_CONTEXT: Context = Context(prec=28, rounding=ROUND_HALF_EVEN)

# Pre-computed scale factor for price-diff → bps. 10_000 = 1 bps as a
# fraction of the reference price (``bps = diff / ref * 10_000``).
BPS_SCALE: Decimal = Decimal("10000")


@contextlib.contextmanager
def with_tca_context() -> Iterator[Context]:
    """Activate the TCA precision context for the lifetime of the block.

    Wraps :func:`decimal.localcontext` so the caller's enclosing
    Decimal context is restored on exit. Idempotent / nestable.
    """
    with localcontext(TCA_PRECISION_CONTEXT) as ctx:
        yield ctx


def to_decimal(value: Decimal | float | int | str | None) -> Decimal:
    """Coerce ``value`` to a :class:`Decimal` under the TCA context.

    ``None`` raises :class:`ValueError`. Floats route through ``str``
    first so the Decimal carries the *displayed* magnitude rather than
    the binary expansion (e.g. ``Decimal(0.1) != Decimal("0.1")``).
    """
    if value is None:
        raise ValueError("to_decimal: value must not be None")
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    # float / str go through str() so the Decimal reflects the literal,
    # not the float binary expansion (which is the whole point of
    # routing through Decimal in the first place).
    return Decimal(str(value))


def to_bps(
    price_diff: Decimal | float | int,
    reference: Decimal | float | int,
) -> Decimal:
    """Convert a price difference to basis points using Decimal arithmetic.

    ``bps = (price_diff / reference) * 10_000``

    Raises :class:`ZeroDivisionError` when the reference is zero (the
    caller must validate the reference price before computing bps; the
    deterministic baseline never silently substitutes a default).
    """
    pd = to_decimal(price_diff)
    ref = to_decimal(reference)
    if ref == 0:
        raise ZeroDivisionError("to_bps: reference price must not be zero")
    with with_tca_context():
        return (pd / ref) * BPS_SCALE


def bps_arithmetic_mean(
    values: Sequence[Decimal | float | int],
    weights: Sequence[Decimal | float | int] | None = None,
) -> Decimal:
    """Weighted arithmetic mean of bps values in Decimal precision.

    Empty input returns ``Decimal("0")`` (consistent with the v1.1
    histogram convention for empty samples). When ``weights`` is
    supplied, it must match ``values`` in length and have a strictly
    positive sum.
    """
    if not values:
        return Decimal(0)
    if weights is None:
        with with_tca_context():
            total = sum((to_decimal(v) for v in values), start=Decimal(0))
            return total / Decimal(len(values))
    if len(weights) != len(values):
        raise ValueError(
            f"bps_arithmetic_mean: weights length ({len(weights)}) does not match values length ({len(values)})"
        )
    with with_tca_context():
        weighted_sum = sum(
            (to_decimal(v) * to_decimal(w) for v, w in zip(values, weights, strict=True)),
            start=Decimal(0),
        )
        weight_sum = sum((to_decimal(w) for w in weights), start=Decimal(0))
        if weight_sum == 0:
            raise ValueError("bps_arithmetic_mean: weights must have a positive sum")
        return weighted_sum / weight_sum


def bps_aggregate_sum(
    values: Sequence[Decimal | float | int],
) -> Decimal:
    """Sum bps values in Decimal precision.

    Useful for daily volume aggregates where the per-trade bps cost
    accumulates across the day. Empty input returns ``Decimal("0")``.
    """
    if not values:
        return Decimal(0)
    with with_tca_context():
        return sum((to_decimal(v) for v in values), start=Decimal(0))


def decimal_to_float_for_report(d: Decimal | float | int | None) -> float | None:
    """Final conversion at the report boundary only.

    Used at the warehouse-write / JSON-serialise / API-response edge.
    ``None`` passes through so the helper composes with optional
    metrics like ``post_trade_markout_*_bps`` whose window may not yet
    have closed.
    """
    if d is None:
        return None
    if isinstance(d, float):
        return d
    if isinstance(d, int):
        return float(d)
    return float(d)
