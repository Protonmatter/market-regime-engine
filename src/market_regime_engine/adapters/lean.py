# SPDX-License-Identifier: Apache-2.0
"""QuantConnect LEAN adapter for governed macro regime signals."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from market_regime_engine.adapters.core import GovernedSignalExport, assert_governed_signal_contract, normalize_governed_signals


def to_lean_custom_data_csv(
    frame: pd.DataFrame,
    out_path: str | Path,
    *,
    symbol: str = "MRE_REGIME",
) -> GovernedSignalExport:
    """Export governed signals as a LEAN-friendly custom-data CSV.

    LEAN algorithms can consume this as custom data by implementing a BaseData
    class that parses the columns emitted here. The adapter intentionally emits
    signal state and confidence, not orders.
    """

    signals = normalize_governed_signals(frame)
    assert_governed_signal_contract(signals)
    out = signals.copy()
    out.insert(1, "symbol", symbol)
    out = out.rename(columns={"date": "time"})
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return GovernedSignalExport(path=str(path), rows=len(out), format="csv", columns=tuple(out.columns))


def lean_python_custom_data_stub(
    *,
    class_name: str = "MarketRegimeSignal",
    remote_url_placeholder: str = "https://example.internal/mre/governed_signals.csv",
) -> str:
    """Return a minimal Python BaseData stub for LEAN custom-data ingestion."""

    return f'''from AlgorithmImports import *


class {class_name}(PythonData):
    """Governed macro-regime signal emitted by Market Regime Engine."""

    def GetSource(self, config, date, isLiveMode):
        return SubscriptionDataSource(
            "{remote_url_placeholder}",
            SubscriptionTransportMedium.RemoteFile,
            FileFormat.Csv,
        )

    def Reader(self, config, line, date, isLiveMode):
        if not line or line.startswith("time,"):
            return None
        parts = line.split(",")
        if len(parts) < 13:
            return None
        item = {class_name}()
        item.Symbol = config.Symbol
        item.Time = datetime.strptime(parts[0], "%Y-%m-%d")
        item.Value = float(parts[3] or 0.0)
        item["regime_state"] = parts[2]
        item["regime_confidence"] = float(parts[3] or 0.0)
        item["change_point_prob"] = float(parts[4] or 0.0)
        item["drawdown_prob"] = float(parts[5] or 0.0)
        item["recession_prob"] = float(parts[6] or 0.0)
        item["release_gate_decision"] = parts[8]
        item["release_gate_approved"] = parts[9].lower() == "true"
        item["model_run_id"] = parts[10]
        item["artifact_hash"] = parts[11]
        return item
'''


__all__ = ["lean_python_custom_data_stub", "to_lean_custom_data_csv"]
