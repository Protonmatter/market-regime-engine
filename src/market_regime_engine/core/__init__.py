# SPDX-License-Identifier: Apache-2.0
"""Stable-core package boundary.

The stable core is the production-certifiable subset of the Market Regime
Engine. It deliberately excludes :mod:`market_regime_engine.frontier`, whose
modules remain optional, research-oriented, or explicitly experimental until a
separate production promotion record exists.
"""

STABLE_CORE_COMPONENTS: tuple[str, ...] = (
    "storage",
    "release_gates",
    "validation",
    "walk_forward",
    "forecast_compare",
    "fixed_income",
    "models",
)

__all__ = ["STABLE_CORE_COMPONENTS"]
