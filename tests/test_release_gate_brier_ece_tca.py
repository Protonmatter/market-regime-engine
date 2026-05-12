# SPDX-License-Identifier: Apache-2.0
"""v1.5.1 (PR-9 FIX 4b/4c): release-gate rails for Brier / ECE / TCA-lift."""

from __future__ import annotations

import json

import pandas as pd

from market_regime_engine.release_gates import evaluate_release_gate


def _baseline_confidence(**columns: object) -> pd.DataFrame:
    """Build a minimal confidence frame that passes existing rails."""
    base = {
        "date": ["2026-05-08"],
        "confidence": [0.95],
        "grade": ["A"],
    }
    base.update({k: [v] for k, v in columns.items()})
    return pd.DataFrame(base)


def _baseline_promotion() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-05-08"],
            "promoted": [True],
            "mcs_evidence": ["in_set"],
        }
    )


def _baseline_coverage() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-05-08"],
            "coverage": [0.92],
        }
    )


def test_release_gate_blocks_when_brier_above_threshold() -> None:
    conf = _baseline_confidence(brier=0.25)  # above 0.20 default
    out = evaluate_release_gate(
        confidence=conf,
        promotion=_baseline_promotion(),
        coverage_report=_baseline_coverage(),
    )
    assert bool(out.iloc[0]["approved"]) is False
    assert "brier_above_0.20" in str(out.iloc[0]["reasons"])


def test_release_gate_passes_when_brier_below_threshold() -> None:
    conf = _baseline_confidence(brier=0.15)
    out = evaluate_release_gate(
        confidence=conf,
        promotion=_baseline_promotion(),
        coverage_report=_baseline_coverage(),
    )
    # Brier is below threshold; other rails were arranged to pass.
    assert "brier_above_" not in str(out.iloc[0]["reasons"])


def test_release_gate_blocks_when_ece_above_threshold() -> None:
    conf = _baseline_confidence(ece=0.12)  # above 0.05 default
    out = evaluate_release_gate(
        confidence=conf,
        promotion=_baseline_promotion(),
        coverage_report=_baseline_coverage(),
    )
    assert bool(out.iloc[0]["approved"]) is False
    assert "ece_above_0.05" in str(out.iloc[0]["reasons"])


def test_release_gate_blocks_when_no_significant_tca_segment() -> None:
    """All segments have p > 0.05 OR |d| < 0.2 → rail fires."""
    tca_lift = {
        "calm": {"p_value": 0.20, "effect_size": 0.05, "n": 100},
        "stressed": {"p_value": 0.10, "effect_size": 0.15, "n": 100},
    }
    conf = _baseline_confidence(tca_lift=tca_lift)
    out = evaluate_release_gate(
        confidence=conf,
        promotion=_baseline_promotion(),
        coverage_report=_baseline_coverage(),
    )
    assert bool(out.iloc[0]["approved"]) is False
    assert "tca_lift_no_significant_segment" in str(out.iloc[0]["reasons"])


def test_release_gate_passes_when_some_tca_segment_significant() -> None:
    """At least one segment with p ≤ 0.05 AND |d| ≥ 0.2 → rail does not fire."""
    tca_lift = {
        "calm": {"p_value": 0.50, "effect_size": 0.05, "n": 200},
        "stressed": {"p_value": 0.01, "effect_size": 0.40, "n": 200},
    }
    conf = _baseline_confidence(tca_lift=tca_lift)
    out = evaluate_release_gate(
        confidence=conf,
        promotion=_baseline_promotion(),
        coverage_report=_baseline_coverage(),
    )
    assert "tca_lift_no_significant_segment" not in str(out.iloc[0]["reasons"])


def test_release_gate_accepts_tca_lift_as_json_string() -> None:
    """The ``tca_lift`` cell may be a JSON string after a warehouse round-trip."""
    tca_lift_json = json.dumps({"stressed": {"p_value": 0.001, "effect_size": 0.5, "n": 200}})
    conf = _baseline_confidence(tca_lift=tca_lift_json)
    out = evaluate_release_gate(
        confidence=conf,
        promotion=_baseline_promotion(),
        coverage_report=_baseline_coverage(),
    )
    assert "tca_lift_no_significant_segment" not in str(out.iloc[0]["reasons"])


def test_release_gate_skips_brier_when_column_absent() -> None:
    """Legacy callers that don't carry ``brier`` keep their pre-PR-9 decision."""
    conf = _baseline_confidence()  # no brier column
    out = evaluate_release_gate(
        confidence=conf,
        promotion=_baseline_promotion(),
        coverage_report=_baseline_coverage(),
    )
    assert "brier_above_" not in str(out.iloc[0]["reasons"])


def test_release_gate_skips_tca_lift_when_threshold_unset() -> None:
    """``profile='default'`` opts out of the TCA rail entirely."""
    conf = _baseline_confidence(tca_lift={"alone": {"p_value": 1.0, "effect_size": 0.0, "n": 10}})
    out = evaluate_release_gate(
        confidence=conf,
        promotion=_baseline_promotion(),
        profile="default",
    )
    assert "tca_lift_no_significant_segment" not in str(out.iloc[0]["reasons"])
