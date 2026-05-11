# SPDX-License-Identifier: Apache-2.0
"""Release-gate propagation tests for the credit regime scorer (PR-3 task J.6)."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

import market_regime_engine.fixed_income  # noqa: F401  - register FI schema
from market_regime_engine.fixed_income.credit_spread_regime import (
    latest_credit_regime_score,
    score_credit_regime,
    write_credit_regime_score,
)
from market_regime_engine.storage import Warehouse

_ASOF = pd.Timestamp("2026-05-08T16:00:00+00:00")


def _row(date: pd.Timestamp, feature_name: str, value: float) -> dict:
    return {
        "date": date,
        "feature_name": feature_name,
        "value": float(value),
        "source_timestamp": date,
        "vintage_date": None,
    }


def _features(asof: pd.Timestamp, n: int = 30) -> pd.DataFrame:
    dates = pd.date_range(end=asof, periods=n, freq="D", tz="UTC")
    rows: list[dict] = []
    for ts in dates:
        rows.append(_row(ts, "ust_slope", 1.0))
        rows.append(_row(ts, "ust_curvature", 0.0))
        rows.append(_row(ts, "cdx_ig_5y", 60.0))
        rows.append(_row(ts, "cdx_hy_5y", 320.0))
        rows.append(_row(ts, "vix", 18.0))
        rows.append(_row(ts, "move", 95.0))
        rows.append(_row(ts, "etf_prem_disc", 0.10))
    return pd.DataFrame(rows)


def test_credit_regime_release_gate_false_caps_confidence_and_propagates() -> None:
    """``release_gate=False`` caps confidence at 0.5 and persists onto the output."""
    out_true = score_credit_regime(_features(_ASOF), asof=_ASOF, model_run_id="run-rg-t", release_gate=True)
    out_false = score_credit_regime(_features(_ASOF), asof=_ASOF, model_run_id="run-rg-f", release_gate=False)
    assert out_true.release_gate is True
    assert out_false.release_gate is False
    assert out_false.confidence <= 0.5 + 1e-9
    assert out_false.confidence <= out_true.confidence + 1e-9
    # Score itself does NOT change — only the governance gate flips.
    assert math.isclose(out_true.regime_score, out_false.regime_score, rel_tol=1e-9)


def test_credit_regime_release_gate_false_writes_to_warehouse(tmp_path: Path) -> None:
    """A ``release_gate=False`` row still lands in the warehouse and round-trips intact."""
    wh = Warehouse(str(tmp_path / "fi-gate.duckdb"))
    try:
        out = score_credit_regime(_features(_ASOF), asof=_ASOF, model_run_id="run-rg-write", release_gate=False)
        assert out.release_gate is False
        rows = write_credit_regime_score(wh, out)
        assert rows == 1
        readback = latest_credit_regime_score(wh)
        assert readback is not None
        assert readback.release_gate is False
        assert readback.model_run_id == out.model_run_id
        assert readback.confidence == out.confidence
    finally:
        wh.close()
