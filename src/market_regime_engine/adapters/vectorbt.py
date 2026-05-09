# SPDX-License-Identifier: Apache-2.0
"""vectorbt adapter for governed macro regime signals."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from market_regime_engine.adapters.core import assert_governed_signal_contract, normalize_governed_signals


@dataclass(frozen=True)
class VectorBTSignals:
    entries: pd.Series
    exits: pd.Series
    risk_off: pd.Series
    governed: pd.DataFrame


def to_vectorbt_signals(
    frame: pd.DataFrame,
    *,
    price_index: pd.Index | None = None,
    long_regimes: tuple[str, ...] = ("expansion", "recovery", "bull", "risk_on"),
    risk_off_regimes: tuple[str, ...] = ("contraction", "crisis", "recession", "bear", "risk_off"),
    min_confidence: float = 0.60,
    max_change_point_prob: float = 0.50,
    require_release_gate: bool = True,
) -> VectorBTSignals:
    """Convert governed regime states into vectorbt entry/exit boolean series."""

    governed = normalize_governed_signals(frame)
    assert_governed_signal_contract(governed)
    idx = pd.to_datetime(governed["date"])
    governed = governed.set_index(idx)

    states = governed["regime_state"].str.lower()
    gate_ok = governed["release_gate_approved"].astype(bool) if require_release_gate else pd.Series(True, index=governed.index)
    confidence_ok = governed["confidence_score"].astype(float) >= float(min_confidence)
    cp_ok = governed["change_point_prob"].astype(float) <= float(max_change_point_prob)

    entries = states.isin({s.lower() for s in long_regimes}) & gate_ok & confidence_ok & cp_ok
    risk_off = states.isin({s.lower() for s in risk_off_regimes}) | ~gate_ok | ~cp_ok
    exits = risk_off | ~confidence_ok

    if price_index is not None:
        entries = entries.reindex(price_index, method="ffill").fillna(False).astype(bool)
        exits = exits.reindex(price_index, method="ffill").fillna(False).astype(bool)
        risk_off = risk_off.reindex(price_index, method="ffill").fillna(True).astype(bool)

    return VectorBTSignals(entries=entries, exits=exits, risk_off=risk_off, governed=governed)


def build_vectorbt_portfolio(price: pd.Series | pd.DataFrame, signals: VectorBTSignals, **kwargs):
    """Build a vectorbt Portfolio if vectorbt is installed."""

    try:
        import vectorbt as vbt  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional adapter path
        raise ImportError("Install vectorbt to build a Portfolio from governed signals.") from exc
    return vbt.Portfolio.from_signals(price, signals.entries, signals.exits, **kwargs)


__all__ = ["VectorBTSignals", "build_vectorbt_portfolio", "to_vectorbt_signals"]
