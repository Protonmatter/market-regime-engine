# SPDX-License-Identifier: Apache-2.0
"""PyPortfolioOpt adapter for regime-conditioned allocation inputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from market_regime_engine.adapters.core import assert_governed_signal_contract, normalize_governed_signals


@dataclass(frozen=True)
class RegimeConditionedInputs:
    expected_returns: pd.Series
    covariance: pd.DataFrame
    latest_regime: str
    confidence: float
    release_gate_approved: bool
    allocation_allowed: bool
    block_reason: str | None = None


def allocation_permission(signals: pd.DataFrame, *, confidence_floor: float = 0.50) -> tuple[bool, str | None, pd.Series]:
    governed = normalize_governed_signals(signals)
    assert_governed_signal_contract(governed)
    latest = governed.iloc[-1]
    if not bool(latest["release_gate_approved"]):
        return False, "release_gate_not_approved", latest
    if float(latest["confidence_score"]) < confidence_floor:
        return False, "confidence_below_floor", latest
    return True, None, latest


def regime_condition_expected_returns(
    base_expected_returns: pd.Series,
    signals: pd.DataFrame,
    *,
    regime_tilts: Mapping[str, Mapping[str, float]] | None = None,
    confidence_floor: float = 0.50,
    blocked_policy: Literal["zero", "base", "raise"] = "zero",
) -> pd.Series:
    """Apply transparent regime tilts to an expected-return vector.

    When allocation is blocked, the policy is explicit instead of smuggling a
    governance decision into optimizer inputs.
    """

    allowed, reason, latest = allocation_permission(signals, confidence_floor=confidence_floor)
    mu = base_expected_returns.astype(float).copy()
    if not allowed:
        if blocked_policy == "raise":
            raise RuntimeError(f"allocation blocked: {reason}")
        if blocked_policy == "base":
            return mu
        return mu * 0.0

    regime = str(latest["regime_state"]).lower()
    tilts = regime_tilts or {}
    for ticker, delta in tilts.get(regime, {}).items():
        if ticker in mu.index:
            mu.loc[ticker] = float(mu.loc[ticker]) + float(delta)
    return mu


def build_regime_conditioned_inputs(
    base_expected_returns: pd.Series,
    covariance: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    regime_tilts: Mapping[str, Mapping[str, float]] | None = None,
    confidence_floor: float = 0.50,
    blocked_policy: Literal["zero", "base", "raise"] = "zero",
) -> RegimeConditionedInputs:
    allowed, reason, latest = allocation_permission(signals, confidence_floor=confidence_floor)
    mu = regime_condition_expected_returns(
        base_expected_returns,
        signals,
        regime_tilts=regime_tilts,
        confidence_floor=confidence_floor,
        blocked_policy=blocked_policy,
    )
    missing = [idx for idx in mu.index if idx not in covariance.index or idx not in covariance.columns]
    if missing:
        raise ValueError(f"covariance matrix missing assets: {missing}")
    return RegimeConditionedInputs(
        expected_returns=mu,
        covariance=covariance.loc[mu.index, mu.index],
        latest_regime=str(latest["regime_state"]),
        confidence=float(latest["confidence_score"]),
        release_gate_approved=bool(latest["release_gate_approved"]),
        allocation_allowed=allowed,
        block_reason=reason,
    )


def build_efficient_frontier(inputs: RegimeConditionedInputs, **kwargs):
    """Instantiate PyPortfolioOpt EfficientFrontier when installed."""

    if not inputs.allocation_allowed:
        raise RuntimeError(f"allocation blocked: {inputs.block_reason}")
    try:
        from pypfopt import EfficientFrontier
    except ImportError as exc:  # pragma: no cover - optional adapter path
        raise ImportError("Install PyPortfolioOpt to build an EfficientFrontier from governed inputs.") from exc
    return EfficientFrontier(inputs.expected_returns, inputs.covariance, **kwargs)


__all__ = [
    "RegimeConditionedInputs",
    "allocation_permission",
    "build_efficient_frontier",
    "build_regime_conditioned_inputs",
    "regime_condition_expected_returns",
]
