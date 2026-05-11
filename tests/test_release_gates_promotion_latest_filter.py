# SPDX-License-Identifier: Apache-2.0
"""PR-1 acceptance test for the stale-promotion filter (REVIEW.md AF-7)."""

from __future__ import annotations

import pandas as pd
import pytest

from market_regime_engine.release_gates import evaluate_release_gate


def test_stale_promotion_row_does_not_satisfy_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 6-month-old promoted=True row + current promoted=False row must
    NOT satisfy the promotion rail. Pre-v1.5 the gate did ``.any()`` over
    the entire frame and a stale True would keep the rail green forever."""
    monkeypatch.delenv("MRE_ENV", raising=False)
    inputs = {
        "confidence": pd.DataFrame([{"date": "2026-05-01", "confidence": 0.90, "grade": "A"}]),
        "drift": pd.DataFrame(),
        "invalidation": pd.DataFrame(),
        "promotion": pd.DataFrame(
            [
                {
                    "date": "2025-11-01",
                    "target": "drawdown_gt_10pct",
                    "horizon": "3m",
                    "promoted": True,
                    "mcs_evidence": "in_set",
                },
                {
                    "date": "2026-05-01",
                    "target": "drawdown_gt_10pct",
                    "horizon": "3m",
                    "promoted": False,
                    "mcs_evidence": "absent",
                },
            ]
        ),
    }
    good_coverage = pd.DataFrame(
        [
            {"coverage": 0.92, "bucket": "x", "n": 100},
        ]
    )
    gate = evaluate_release_gate(**inputs, coverage_report=good_coverage)
    row = gate.iloc[0]
    reasons = str(row["reasons"])
    assert "no_promoted_model" in reasons, reasons
    assert bool(row["approved"]) is False


def test_latest_promotion_row_promoted_true_satisfies_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative control: latest=True + older=False must satisfy the rail."""
    monkeypatch.delenv("MRE_ENV", raising=False)
    inputs = {
        "confidence": pd.DataFrame([{"date": "2026-05-01", "confidence": 0.90, "grade": "A"}]),
        "drift": pd.DataFrame(),
        "invalidation": pd.DataFrame(),
        "promotion": pd.DataFrame(
            [
                {"date": "2025-11-01", "promoted": False, "mcs_evidence": "absent"},
                {"date": "2026-05-01", "promoted": True, "mcs_evidence": "in_set"},
            ]
        ),
    }
    gate = evaluate_release_gate(
        **inputs,
        coverage_report=pd.DataFrame([{"coverage": 0.92, "bucket": "x", "n": 100}]),
    )
    row = gate.iloc[0]
    reasons = str(row["reasons"])
    assert "no_promoted_model" not in reasons, reasons


def test_promotion_without_date_column_falls_back_to_any(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Back-compat: a promotion frame without a date column still uses ``.any()``."""
    monkeypatch.delenv("MRE_ENV", raising=False)
    inputs = {
        "confidence": pd.DataFrame([{"date": "2026-05-01", "confidence": 0.90, "grade": "A"}]),
        "drift": pd.DataFrame(),
        "invalidation": pd.DataFrame(),
        "promotion": pd.DataFrame(
            [
                {"promoted": True, "mcs_evidence": "in_set"},
            ]
        ),
    }
    gate = evaluate_release_gate(
        **inputs,
        coverage_report=pd.DataFrame([{"coverage": 0.92, "bucket": "x", "n": 100}]),
    )
    row = gate.iloc[0]
    assert "no_promoted_model" not in str(row["reasons"])
