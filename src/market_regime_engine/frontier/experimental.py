# SPDX-License-Identifier: Apache-2.0
"""Explicit opt-in fence for retrospective-only frontier paths.

Set ``MRE_ENABLE_EXPERIMENTAL_FRONTIER=1`` (or pass an equivalent test
fixture environment) before invoking paths that are mathematically useful for
research but unsafe for real-time production decisioning because they can use
future information, retrospective smoothing, or still-experimental promotion
logic.
"""

from __future__ import annotations

import os

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}


def frontier_experimental_enabled() -> bool:
    """Return whether experimental frontier paths are explicitly enabled."""
    return os.environ.get("MRE_ENABLE_EXPERIMENTAL_FRONTIER", "").strip().lower() in _TRUE_VALUES


def require_frontier_experimental(reason: str) -> None:
    """Raise unless the operator explicitly enabled experimental frontier paths."""
    if frontier_experimental_enabled():
        return
    raise RuntimeError(
        "experimental frontier path disabled: "
        f"{reason}. Set MRE_ENABLE_EXPERIMENTAL_FRONTIER=1 only for retrospective research, "
        "offline validation, or controlled experiments; do not enable for real-time release gates."
    )


__all__ = ["frontier_experimental_enabled", "require_frontier_experimental"]
