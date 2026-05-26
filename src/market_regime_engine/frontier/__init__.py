"""2026-2027 SOTA frontier modeling layer.

This subpackage hosts the v1.2 modeling additions (frontier conformal,
mixed-frequency nowcasting, distributional regression, neural sequence
baselines, sequential anytime-valid testing, GP change-point) plus the
v1.5 / v1.6 additions (hierarchical liquidity, online conformal,
overfit-control) that live behind optional ``[frontier]`` / ``[nowcast]``
extras.

Each submodule is independently importable and degrades gracefully when its
soft dependency (statsmodels, ngboost, torch) isn't installed. The legacy
public surface (``MondrianBinaryConformal``, ``OnlineBMA``, ...) is unchanged
— the new primitives plug in via their existing ``backend=`` /
``predictions: dict[str, float]`` contracts.

v1.6.0 (REVIEW_DEEP_V1_5_2.md §6 / Phase 5.5): the ``__all__`` list now
mirrors the actual on-disk module set (every ``frontier/<name>.py`` is
listed). The Phase 5.5 spec called out re-adding ``data_cleaning`` —
that entry was already restored in PR-3 (commit 39130a2 "feat(frontier):
NaN policy enum and per-column cleaning"); the actually-missing entries
were the three v1.5 / v1.6 additions ``hierarchical_liquidity``,
``online_conformal``, ``overfit_control``. Phase 5.5 adds them so the
``from market_regime_engine.frontier import *`` surface and
``__init__.py`` introspection both report the canonical module set.
"""

from __future__ import annotations

__all__ = [
    "bayesian_msvar",
    "conformal_ts",
    "data_cleaning",
    "deep_kernel",
    "dfm_mq",
    "diagnostics",
    "distributional",
    "gp_cpd",
    "hierarchical_liquidity",
    "midas",
    "neural_seq",
    "online_conformal",
    "overfit_control",
    "release_calendars",
    "sequential_testing",
]


EXPERIMENTAL_FRONTIER_COMPONENTS: tuple[str, ...] = (
    "bayesian_msvar",
    "dfm_mq",
    "diagnostics",
    "deep_kernel",
    "gp_cpd",
    "neural_seq",
    "sequential_testing",
)

__all__ = tuple(sorted(set(globals().get("__all__", ())) | {"EXPERIMENTAL_FRONTIER_COMPONENTS"}))
