# SPDX-License-Identifier: Apache-2.0
"""PR-1 acceptance tests for the e-value missing-decision rail (REVIEW.md AF-14)."""

from __future__ import annotations

import pandas as pd
import pytest

from market_regime_engine.release_gates import evaluate_release_gate


def _e_value_inputs() -> dict[str, pd.DataFrame]:
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


def test_evalues_missing_decision_column_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``promotion_method="e_values"`` + log without ``decision`` raises."""
    monkeypatch.delenv("MRE_ENV", raising=False)
    inputs = _e_value_inputs()
    log_without_decision = pd.DataFrame(
        [
            {"date": "2026-05-01", "challenger": "m1", "e_value": 50.0},
            {"date": "2026-05-01", "challenger": "m2", "e_value": 25.0},
        ]
    )
    with pytest.raises(ValueError, match="e_value_log missing 'decision' column"):
        evaluate_release_gate(
            **inputs,
            promotion_method="e_values",
            e_value_log=log_without_decision,
            min_coverage=None,
            require_mcs_membership=False,
        )


def test_evalues_with_decision_column_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """Correct log with ``decision`` column does not raise."""
    monkeypatch.delenv("MRE_ENV", raising=False)
    inputs = _e_value_inputs()
    log = pd.DataFrame(
        [
            {"date": "2026-05-01", "challenger": "m1", "e_value": 50.0, "decision": "promote"},
        ]
    )
    gate = evaluate_release_gate(
        **inputs,
        promotion_method="e_values",
        e_value_log=log,
        e_value_alpha=0.05,
        min_coverage=None,
        require_mcs_membership=False,
    )
    row = gate.iloc[0]
    reasons = str(row["reasons"])
    assert "e_value_gate_not_fired" not in reasons, reasons
