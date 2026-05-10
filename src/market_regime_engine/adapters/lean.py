# SPDX-License-Identifier: Apache-2.0
"""QuantConnect LEAN adapter for governed macro regime signals."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from market_regime_engine.adapters.core import (
    GovernedSignalExport,
    assert_governed_signal_contract,
    normalize_governed_signals,
)


def to_lean_custom_data_csv(
    frame: pd.DataFrame,
    out_path: str | Path,
    *,
    symbol: str = "MRE_REGIME",
) -> GovernedSignalExport:
    """Export governed signals as a LEAN-friendly custom-data CSV."""

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
    """Return a minimal PythonData stub that parses quoted CSV correctly."""

    return f'''from AlgorithmImports import *
from datetime import datetime
import csv


class {class_name}(PythonData):
    """Governed macro-regime signal emitted by Market Regime Engine."""

    Columns = [
        "time", "symbol", "regime_state", "regime_confidence",
        "change_point_prob", "drawdown_prob", "recession_prob",
        "confidence_score", "release_gate_decision", "release_gate_approved",
        "model_run_id", "artifact_hash", "metadata_json",
    ]

    def GetSource(self, config, date, isLiveMode):
        return SubscriptionDataSource(
            "{remote_url_placeholder}",
            SubscriptionTransportMedium.RemoteFile,
            FileFormat.Csv,
        )

    def Reader(self, config, line, date, isLiveMode):
        if not line or line.startswith("time,"):
            return None
        row = next(csv.reader([line]))
        if len(row) < len(self.Columns):
            return None
        rec = dict(zip(self.Columns, row))
        item = {class_name}()
        item.Symbol = config.Symbol
        item.Time = datetime.strptime(rec["time"], "%Y-%m-%d")
        item.Value = float(rec.get("regime_confidence") or 0.0)
        item["regime_state"] = rec.get("regime_state", "unknown")
        item["regime_confidence"] = float(rec.get("regime_confidence") or 0.0)
        item["change_point_prob"] = float(rec.get("change_point_prob") or 0.0)
        item["drawdown_prob"] = float(rec.get("drawdown_prob") or 0.0)
        item["recession_prob"] = float(rec.get("recession_prob") or 0.0)
        item["confidence_score"] = float(rec.get("confidence_score") or 0.0)
        item["release_gate_decision"] = rec.get("release_gate_decision", "unknown")
        item["release_gate_approved"] = str(rec.get("release_gate_approved", "false")).strip().lower() in ("true", "1", "yes", "y")
        item["model_run_id"] = rec.get("model_run_id", "")
        item["artifact_hash"] = rec.get("artifact_hash", "")
        item["metadata_json"] = rec.get("metadata_json", "")
        return item
'''


__all__ = ["lean_python_custom_data_stub", "to_lean_custom_data_csv"]
