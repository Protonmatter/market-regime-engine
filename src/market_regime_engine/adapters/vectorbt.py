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
    entry_score: pd.Series
    governed: pd.DataFrame


def to_vectorbt_signals(
    frame: pd.DataFrame,
    *,
    price_index: pd.Index | None = None,
    long_regimes: tuple[str, ...] = ("expansion", "recovery", "bull", "risk_on"),
    risk_off_regimes: tuple[str, ...] = ("contraction", "crisis", "recession", "bear", "risk_off"),
    min_confidence: float = 0.60,
    max_change_point_prob: float = 0.50,
    entry_threshold: float = 0.65,
    exit_threshold: float = 0.40,
    max_drawdown_prob: float = 0.60,
    require_release_gate: bool = True,
) -> VectorBTSignals:
    """Convert governed regime states into vectorbt signal series.

    The adapter preserves uncertainty through ``entry_score`` instead of using
    only a hard regime label. Downstream strategy code can tune thresholds.
    """

    governed = normalize_governed_signals(frame)
    assert_governed_signal_contract(governed)
    idx = pd.to_datetime(governed["date"])
    governed = governed.set_index(idx)

    states = governed["regime_state"].str.lower()
    long_state = states.isin({s.lower() for s in long_regimes}).astype(float)
    risk_state = states.isin({s.lower() for s in risk_off_regimes})
    gate_ok = governed["release_gate_approved"].astype(bool) if require_release_gate else pd.Series(True, index=governed.index)
    confidence = governed["confidence_score"].astype(float).clip(0.0, 1.0)
    cp_ok = (1.0 - governed["change_point_prob"].astype(float).clip(0.0, 1.0)).clip(0.0, 1.0)
    drawdown_ok = (1.0 - governed["drawdown_prob"].astype(float).clip(0.0, 1.0)).clip(0.0, 1.0)

    entry_score = long_state * confidence * cp_ok * drawdown_ok * gate_ok.astype(float)
    confidence_ok = confidence >= float(min_confidence)
    entries = entry_score >= float(entry_threshold)
    risk_off = risk_state | ~gate_ok | (governed["change_point_prob"].astype(float) > float(max_change_point_prob)) | (
        governed["drawdown_prob"].astype(float) > float(max_drawdown_prob)
    )
    exits = risk_off | ~confidence_ok | (entry_score <= float(exit_threshold))

    if price_index is not None:
        entries = entries.reindex(price_index, method="ffill").fillna(False).astype(bool)
        exits = exits.reindex(price_index, method="ffill").fillna(False).astype(bool)
        risk_off = risk_off.reindex(price_index, method="ffill").fillna(True).astype(bool)
        entry_score = entry_score.reindex(price_index, method="ffill").fillna(0.0).astype(float)

    return VectorBTSignals(entries=entries, exits=exits, risk_off=risk_off, entry_score=entry_score, governed=governed)


def build_vectorbt_portfolio(price: pd.Series | pd.DataFrame, signals: VectorBTSignals, **kwargs):
    """Build a vectorbt Portfolio if vectorbt is installed."""

    try:
        import vectorbt as vbt  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional adapter path
        raise ImportError("Install vectorbt to build a Portfolio from governed signals.") from exc
    return vbt.Portfolio.from_signals(price, signals.entries, signals.exits, **kwargs)


__all__ = ["VectorBTSignals", "build_vectorbt_portfolio", "to_vectorbt_signals"]
