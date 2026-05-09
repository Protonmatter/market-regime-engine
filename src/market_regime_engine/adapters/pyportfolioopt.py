# SPDX-License-Identifier: Apache-2.0
"""PyPortfolioOpt adapter for regime-conditioned allocation inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from market_regime_engine.adapters.core import assert_governed_signal_contract, normalize_governed_signals


@dataclass(frozen=True)
class RegimeConditionedInputs:
    expected_returns: pd.Series
    covariance: pd.DataFrame
    latest_regime: str
    confidence: float
    release_gate_approved: bool


def regime_condition_expected_returns(
    base_expected_returns: pd.Series,
    signals: pd.DataFrame,
    *,
    regime_tilts: Mapping[str, Mapping[str, float]] | None = None,
    confidence_floor: float = 0.50,
) -> pd.Series:
    """Apply transparent regime tilts to an expected-return vector.

    ``regime_tilts`` maps regime name -> ticker -> additive expected-return
    delta. This keeps the optimizer adapter auditable instead of hiding regime
    assumptions inside black-box weights.
    """

    governed = normalize_governed_signals(signals)
    assert_governed_signal_contract(governed)
    latest = governed.iloc[-1]
    mu = base_expected_returns.astype(float).copy()
    if not bool(latest["release_gate_approved"]) or float(latest["confidence_score"]) < confidence_floor:
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
) -> RegimeConditionedInputs:
    governed = normalize_governed_signals(signals)
    assert_governed_signal_contract(governed)
    latest = governed.iloc[-1]
    mu = regime_condition_expected_returns(
        base_expected_returns,
        governed,
        regime_tilts=regime_tilts,
        confidence_floor=confidence_floor,
    )
    return RegimeConditionedInputs(
        expected_returns=mu,
        covariance=covariance.loc[mu.index, mu.index],
        latest_regime=str(latest["regime_state"]),
        confidence=float(latest["confidence_score"]),
        release_gate_approved=bool(latest["release_gate_approved"]),
    )


def build_efficient_frontier(inputs: RegimeConditionedInputs, **kwargs):
    """Instantiate PyPortfolioOpt EfficientFrontier when installed."""

    try:
        from pypfopt import EfficientFrontier  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional adapter path
        raise ImportError("Install PyPortfolioOpt to build an EfficientFrontier from governed inputs.") from exc
    return EfficientFrontier(inputs.expected_returns, inputs.covariance, **kwargs)


__all__ = [
    "RegimeConditionedInputs",
    "build_efficient_frontier",
    "build_regime_conditioned_inputs",
    "regime_condition_expected_returns",
]
