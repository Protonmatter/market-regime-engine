"""2026-2027 SOTA frontier modeling layer.

This subpackage hosts the v1.2 modeling additions (frontier conformal,
mixed-frequency nowcasting, distributional regression, neural sequence
baselines, sequential anytime-valid testing, GP change-point) that live
behind optional ``[frontier]`` / ``[nowcast]`` extras.

Each submodule is independently importable and degrades gracefully when its
soft dependency (statsmodels, ngboost, torch) isn't installed. The legacy
public surface (``MondrianBinaryConformal``, ``OnlineBMA``, ...) is unchanged
— the new primitives plug in via their existing ``backend=`` /
``predictions: dict[str, float]`` contracts.
"""

from __future__ import annotations

__all__ = [
    "bayesian_msvar",
    "conformal_ts",
    "deep_kernel",
    "dfm_mq",
    "distributional",
    "gp_cpd",
    "midas",
    "neural_seq",
    "release_calendars",
    "sequential_testing",
]
