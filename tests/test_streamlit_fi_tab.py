# SPDX-License-Identifier: Apache-2.0
"""PR-7 §H — FI Streamlit tab smoke + summary acceptance tests.

We test the underlying ``fi_dashboard_summary`` / ``load_fi_tables``
helpers (no Streamlit invocation needed) plus an import-level smoke
on ``render_fi_tab`` to catch syntax / import regressions. The full
Streamlit page render is not covered (Streamlit ships with its own
test harness; PR-7 ships the helper-level contract).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import market_regime_engine.fixed_income  # noqa: F401  - register FI tables
from market_regime_engine.fixed_income.dashboard_tab import (
    fi_dashboard_summary,
    load_fi_tables,
)
from market_regime_engine.storage import Warehouse


def test_fi_dashboard_summary_handles_empty_data() -> None:
    summary = fi_dashboard_summary({})
    assert summary["credit_regime_rows"] == 0
    assert summary["liquidity_scopes"] == 0
    assert summary["execution_predictions"] == 0
    assert summary["release_gate_rows"] == 0
    assert summary["evidence_pack_rows"] == 0


def test_fi_dashboard_summary_includes_latest_score(tmp_path: Path) -> None:
    db_path = tmp_path / "fi-tab.duckdb"
    wh = Warehouse(str(db_path))
    try:
        wh.write_credit_regime_score(
            pd.DataFrame(
                [
                    {
                        "model_run_id": "run-1",
                        "timestamp": "2026-05-08T16:00:00Z",
                        "regime_score": 47.5,
                        "regime_label": "Watch / Transition",
                        "confidence": 0.85,
                        "drivers_json": "[]",
                        "component_scores_json": "{}",
                        "release_gate": 1,
                        "artifact_hash": "sha256:" + "a" * 64,
                        "metadata_json": "{}",
                    }
                ]
            )
        )
        wh.write_liquidity_stress_score(
            pd.DataFrame(
                [
                    {
                        "model_run_id": "run-l1",
                        "scope_type": "market",
                        "scope_id": "ALL",
                        "timestamp": "2026-05-08T16:00:00Z",
                        "liquidity_score": 30.0,
                        "liquidity_label": "Mild Stress",
                        "confidence": 0.9,
                        "drivers_json": "[]",
                        "release_gate": 1,
                        "artifact_hash": "sha256:" + "b" * 64,
                        "metadata_json": "{}",
                    },
                    {
                        "model_run_id": "run-l2",
                        "scope_type": "sector",
                        "scope_id": "TECH",
                        "timestamp": "2026-05-08T16:00:00Z",
                        "liquidity_score": 35.0,
                        "liquidity_label": "Mild Stress",
                        "confidence": 0.9,
                        "drivers_json": "[]",
                        "release_gate": 1,
                        "artifact_hash": "sha256:" + "c" * 64,
                        "metadata_json": "{}",
                    },
                ]
            )
        )
    finally:
        wh.close()
    fi = load_fi_tables(str(db_path))
    summary = fi_dashboard_summary(fi)
    assert summary["credit_regime_rows"] == 1
    assert summary["latest_regime_score"] == pytest.approx(47.5)
    assert summary["latest_regime_label"] == "Watch / Transition"
    assert summary["liquidity_scopes"] == 2


def test_fi_dashboard_render_function_imports() -> None:
    """Smoke: importing the helper module does not raise."""
    from market_regime_engine.fixed_income import dashboard_tab as mod

    assert callable(mod.render_fi_tab)
    assert callable(mod.fi_dashboard_summary)
    assert callable(mod.load_fi_tables)


def test_load_fi_tables_returns_empty_dict_for_missing_warehouse(
    tmp_path: Path,
) -> None:
    """A missing DuckDB path produces a dict of empty frames."""
    out = load_fi_tables(str(tmp_path / "does-not-exist.duckdb"))
    assert isinstance(out, dict)
    for key in (
        "credit_regime_scores",
        "liquidity_stress_scores",
        "execution_confidence_predictions",
        "release_gates",
    ):
        assert key in out
        assert out[key].empty
