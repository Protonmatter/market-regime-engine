# SPDX-License-Identifier: Apache-2.0
"""Filtered-vs-smoothed posterior enforcement for FI real-time decisions.

Non-negotiable constraint 6 (``MRE_FIXED_INCOME_AGENT.md``) forbids the
use of smoothed posteriors for real-time decisioning. Smoothed
posteriors are computed with information from *after* the decision
time (the smoother sweeps backwards through the chain), so consuming
one for an Auto-X advisory is a covert lookahead leak.

This module defines:

- :class:`PosteriorMode` — string Enum with ``FILTERED`` / ``SMOOTHED``.
- :class:`FilteredPosterior` / :class:`SmoothedPosterior` — frozen
  wrappers that carry the mode tag inline so the type system can
  enforce the rail at the call site.
- :func:`require_filtered` — runtime guard that raises ``TypeError``
  when an FI real-time entry point is handed a :class:`SmoothedPosterior`.

The wrappers carry the raw posterior payload (``data: np.ndarray``,
``timestamps: pd.DatetimeIndex``) verbatim so PR-3 / PR-4 scorers can
unwrap and consume without copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Union

import numpy as np
import pandas as pd


class PosteriorMode(str, Enum):
    """Whether a posterior was computed by a filter (forward-only) or smoother."""

    FILTERED = "filtered"
    SMOOTHED = "smoothed"


@dataclass(frozen=True)
class FilteredPosterior:
    """Forward-only (filtered) posterior — safe for real-time decisioning.

    The ``mode`` field is fixed to :attr:`PosteriorMode.FILTERED` at
    construction time so :func:`require_filtered` can type-check via
    ``isinstance`` rather than relying on the operator setting the
    field correctly.
    """

    data: np.ndarray
    timestamps: pd.DatetimeIndex
    mode: PosteriorMode = PosteriorMode.FILTERED


@dataclass(frozen=True)
class SmoothedPosterior:
    """Two-pass (smoothed) posterior — forbidden for real-time decisioning.

    Smoothed posteriors incorporate information from observations
    after ``timestamps[t]`` and are therefore lookahead-leaky for any
    decision dated at or before ``timestamps[t]``. Allowed for
    retrospective analytics only.
    """

    data: np.ndarray
    timestamps: pd.DatetimeIndex
    mode: PosteriorMode = PosteriorMode.SMOOTHED


PosteriorLike = Union[FilteredPosterior, SmoothedPosterior]


def require_filtered(post: PosteriorLike) -> FilteredPosterior:
    """Return ``post`` if filtered; raise ``TypeError`` otherwise.

    Per AGENT.md non-negotiable constraint 6: "Do not use smoothed
    posteriors for real-time decisioning." Any FI scoring entry point
    that consumes posterior probabilities (credit-regime, liquidity-stress,
    execution-confidence) routes its input through this guard so the
    rail fires before the score is computed.
    """
    if isinstance(post, SmoothedPosterior) or getattr(post, "mode", None) == PosteriorMode.SMOOTHED:
        raise TypeError(
            "smoothed posteriors must not be used for real-time decisioning "
            "per AGENT.md non-negotiable constraint 6"
        )
    if not isinstance(post, FilteredPosterior):
        raise TypeError(
            f"expected FilteredPosterior, got {type(post).__name__}; "
            f"wrap your data in FilteredPosterior(...) before passing to FI scoring"
        )
    return post


__all__ = [
    "FilteredPosterior",
    "PosteriorLike",
    "PosteriorMode",
    "SmoothedPosterior",
    "require_filtered",
]
