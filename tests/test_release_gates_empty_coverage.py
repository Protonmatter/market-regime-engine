# SPDX-License-Identifier: Apache-2.0
"""PR-1 acceptance tests for empty/NaN coverage handling (REVIEW.md AF-6)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from market_regime_engine.release_gates import evaluate_release_gate


def _passing_inputs() -> dict[str, pd.DataFrame]:
    """Inputs that satisfy every non-coverage rail of the production profile."""
    return {
        "confidence": pd.DataFrame([{"date": "2026-05-01", "confidence": 0.80, "grade": "A"}]),
        "drift": pd.DataFrame(),
        "invalidation": pd.DataFrame(),
        "promotion": pd.DataFrame(
            [
                {
                    "date": "2026-05-01",
                    "target": "drawdown_gt_10pct",
                    "horizon": "3m",
                    "promoted": True,
                    "mcs_evidence": "in_set",
                }
            ]
        ),
    }


def test_empty_coverage_appends_missing_reason_and_fails_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``coverage_report=pd.DataFrame()`` under the production profile fires
    the new ``coverage_data_missing`` rail and the gate does not approve."""
    monkeypatch.delenv("MRE_ENV", raising=False)
    inputs = _passing_inputs()
    gate = evaluate_release_gate(**inputs, coverage_report=pd.DataFrame())
    row = gate.iloc[0]
    reasons = str(row["reasons"])
    assert "coverage_data_missing" in reasons, reasons
    assert bool(row["approved"]) is False, row.to_dict()
    assert math.isnan(float(row["worst_coverage"]))


def test_all_nan_coverage_treated_same(monkeypatch: pytest.MonkeyPatch) -> None:
    """A coverage frame whose rows are all NaN is equivalent to "missing"."""
    monkeypatch.delenv("MRE_ENV", raising=False)
    inputs = _passing_inputs()
    bad_coverage = pd.DataFrame(
        [
            {"coverage": float("nan"), "bucket": "x", "n": 10},
            {"coverage": float("nan"), "bucket": "y", "n": 10},
        ]
    )
    gate = evaluate_release_gate(**inputs, coverage_report=bad_coverage)
    row = gate.iloc[0]
    reasons = str(row["reasons"])
    assert "coverage_data_missing" in reasons, reasons
    assert bool(row["approved"]) is False


def test_valid_coverage_does_not_trigger_missing_rail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-empty, non-NaN coverage that clears the floor still passes."""
    monkeypatch.delenv("MRE_ENV", raising=False)
    inputs = _passing_inputs()
    good_coverage = pd.DataFrame(
        [
            {"coverage": 0.91, "bucket": "x", "n": 100},
            {"coverage": 0.92, "bucket": "y", "n": 100},
        ]
    )
    gate = evaluate_release_gate(**inputs, coverage_report=good_coverage)
    row = gate.iloc[0]
    reasons = str(row["reasons"])
    assert "coverage_data_missing" not in reasons, reasons
    assert bool(row["approved"]) is True, row.to_dict()


def test_missing_coverage_no_floor_still_silent_under_default_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``profile="default"`` (min_coverage=None) the missing-rail does
    not fire — preserves v1.4 behaviour for dev/staging environments."""
    monkeypatch.setenv("MRE_ENV", "dev")
    inputs = _passing_inputs()
    gate = evaluate_release_gate(**inputs, coverage_report=pd.DataFrame())
    row = gate.iloc[0]
    assert "coverage_data_missing" not in str(row["reasons"])
