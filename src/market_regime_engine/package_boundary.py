# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BoundaryName = Literal["stable_core", "experimental_frontier"]


@dataclass(frozen=True)
class PackageBoundary:
    """Release-governance boundary for model components.

    ``stable_core`` is the mature, certifiable subset used by the default
    production release gate. ``experimental_frontier`` is the research subset
    that may use optional dependencies, retrospective diagnostics, or methods
    whose production contract is still being hardened. Frontier paths must be
    explicitly enabled before they can participate in a gate.
    """

    name: BoundaryName
    package_prefix: str
    production_eligible: bool
    requires_experimental_flag: bool
    default_gate_profile: Literal["production", "default"]
    description: str


STABLE_CORE = PackageBoundary(
    name="stable_core",
    package_prefix="market_regime_engine",
    production_eligible=True,
    requires_experimental_flag=False,
    default_gate_profile="production",
    description="Mature macro, storage, validation, fixed-income, and governance components.",
)

EXPERIMENTAL_FRONTIER = PackageBoundary(
    name="experimental_frontier",
    package_prefix="market_regime_engine.frontier",
    production_eligible=False,
    requires_experimental_flag=True,
    default_gate_profile="default",
    description="Research/frontier models behind explicit opt-in and separate review gates.",
)

_BOUNDARIES: dict[str, PackageBoundary] = {
    STABLE_CORE.name: STABLE_CORE,
    EXPERIMENTAL_FRONTIER.name: EXPERIMENTAL_FRONTIER,
}


def resolve_boundary(name: BoundaryName | str | None = None) -> PackageBoundary:
    """Return the configured package boundary or raise on unknown names."""

    key = (name or "stable_core").strip()
    try:
        return _BOUNDARIES[key]
    except KeyError as exc:
        raise ValueError(f"unknown package boundary {key!r}; valid boundaries are {sorted(_BOUNDARIES)}") from exc


__all__ = [
    "EXPERIMENTAL_FRONTIER",
    "STABLE_CORE",
    "BoundaryName",
    "PackageBoundary",
    "resolve_boundary",
]
