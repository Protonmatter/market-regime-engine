# SPDX-License-Identifier: Apache-2.0
"""PR-1 acceptance tests for the new resolved_profile column (REVIEW.md ASK-7)."""

from __future__ import annotations

import pandas as pd
import pytest

from market_regime_engine.release_gates import evaluate_release_gate


def _passing_inputs() -> dict[str, pd.DataFrame]:
    return {
        "confidence": pd.DataFrame([{"date": "2026-05-01", "confidence": 0.90, "grade": "A"}]),
        "drift": pd.DataFrame(),
        "invalidation": pd.DataFrame(),
        "promotion": pd.DataFrame(
            [
                {
                    "date": "2026-05-01",
                    "promoted": True,
                    "mcs_evidence": "in_set",
                }
            ]
        ),
    }


def test_resolved_profile_column_emitted() -> None:
    """``profile="production"`` round-trips into the output frame column."""
    inputs = _passing_inputs()
    gate = evaluate_release_gate(
        **inputs,
        coverage_report=pd.DataFrame([{"coverage": 0.92, "bucket": "x", "n": 100}]),
        profile="production",
    )
    assert "resolved_profile" in gate.columns
    assert gate.iloc[0]["resolved_profile"] == "production"


def test_resolved_profile_explicit_default() -> None:
    inputs = _passing_inputs()
    gate = evaluate_release_gate(
        **inputs,
        coverage_report=pd.DataFrame([{"coverage": 0.92, "bucket": "x", "n": 100}]),
        profile="default",
    )
    assert gate.iloc[0]["resolved_profile"] == "default"


def test_resolved_profile_env_var_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    """``MRE_ENV=staging`` (no explicit profile) resolves to "default"."""
    monkeypatch.setenv("MRE_ENV", "staging")
    inputs = _passing_inputs()
    gate = evaluate_release_gate(
        **inputs,
        coverage_report=pd.DataFrame([{"coverage": 0.92, "bucket": "x", "n": 100}]),
    )
    assert gate.iloc[0]["resolved_profile"] == "default"


def test_resolved_profile_fallback_production(monkeypatch: pytest.MonkeyPatch) -> None:
    """No explicit profile and ``MRE_ENV`` unset → production."""
    monkeypatch.delenv("MRE_ENV", raising=False)
    inputs = _passing_inputs()
    gate = evaluate_release_gate(
        **inputs,
        coverage_report=pd.DataFrame([{"coverage": 0.92, "bucket": "x", "n": 100}]),
    )
    assert gate.iloc[0]["resolved_profile"] == "production"
