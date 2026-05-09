from __future__ import annotations

import pandas as pd

from market_regime_engine.adapters.core import export_governed_signals, normalize_governed_signals
from market_regime_engine.adapters.lean import lean_python_custom_data_stub, to_lean_custom_data_csv
from market_regime_engine.adapters.openbb import to_openbb_obbject_like
from market_regime_engine.adapters.pyportfolioopt import regime_condition_expected_returns
from market_regime_engine.adapters.vectorbt import to_vectorbt_signals


def _sample() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": "2024-01-31",
                "decoded_regime": "expansion",
                "score": 0.82,
                "change_point_prob": 0.12,
                "drawdown_prob": 0.20,
                "recession_prob": 0.10,
                "confidence": 0.81,
                "decision": "release",
                "approved": True,
                "run_id": "abc123",
                "artifact_hash": "deadbeef",
            },
            {
                "date": "2024-02-29",
                "decoded_regime": "crisis",
                "score": 0.74,
                "change_point_prob": 0.71,
                "drawdown_prob": 0.65,
                "recession_prob": 0.42,
                "confidence": 0.73,
                "decision": "hold",
                "approved": False,
                "run_id": "abc124",
                "artifact_hash": "feedface",
            },
        ]
    )


def test_normalize_governed_signals_contract() -> None:
    out = normalize_governed_signals(_sample())
    assert list(out["regime_state"]) == ["expansion", "crisis"]
    assert out["release_gate_approved"].tolist() == [True, False]
    assert set(["date", "artifact_hash", "metadata_json"]).issubset(out.columns)


def test_export_governed_signals_csv(tmp_path) -> None:
    result = export_governed_signals(_sample(), tmp_path / "signals.csv")
    assert result.rows == 2
    assert (tmp_path / "signals.csv").exists()


def test_lean_export_and_stub(tmp_path) -> None:
    result = to_lean_custom_data_csv(_sample(), tmp_path / "lean.csv")
    assert result.rows == 2
    stub = lean_python_custom_data_stub(class_name="MRESignal")
    assert "class MRESignal" in stub
    assert "PythonData" in stub


def test_vectorbt_signals_gate_and_changepoint_logic() -> None:
    signals = to_vectorbt_signals(_sample())
    assert signals.entries.tolist() == [True, False]
    assert signals.exits.tolist() == [False, True]
    assert signals.risk_off.tolist() == [False, True]


def test_pyportfolioopt_expected_returns_zeroed_when_gate_fails() -> None:
    base = pd.Series({"SPY": 0.07, "TLT": 0.03})
    last_hold = _sample()
    mu = regime_condition_expected_returns(base, last_hold)
    assert mu.eq(0.0).all()


def test_openbb_object_like_shape() -> None:
    obj = to_openbb_obbject_like(_sample())
    assert obj["provider"] == "market_regime_engine"
    assert obj["extra"]["metadata"]["contract"] == "governed_macro_regime_signal_v1"
    assert len(obj["results"]) == 2
