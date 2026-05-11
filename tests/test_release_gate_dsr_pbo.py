# SPDX-License-Identifier: Apache-2.0
"""PR-5 Q-5: production-profile release gate enforces DSR / PBO rails."""

from __future__ import annotations

import pandas as pd

from market_regime_engine.release_gates import (
    default_profile,
    evaluate_release_gate,
    production_profile,
)


def _base_confidence(dsr: float | None = None, pbo: float | None = None) -> pd.DataFrame:
    row: dict = {
        "date": pd.Timestamp("2026-05-01"),
        "confidence": 0.9,
        "grade": "A",
    }
    if dsr is not None:
        row["dsr"] = dsr
    if pbo is not None:
        row["pbo"] = pbo
    return pd.DataFrame([row])


def _approving_promotion() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-05-01"),
                "promoted": True,
                "mcs_evidence": "in_set",
            }
        ]
    )


def _approving_coverage() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "target": "drawdown_gt_10pct",
                "horizon": "3m",
                "bucket": "all",
                "coverage": 0.92,
            }
        ]
    )


def test_production_profile_carries_dsr_pbo_thresholds() -> None:
    prof = production_profile()
    assert prof["min_dsr"] == 0.5
    assert prof["max_pbo"] == 0.05


def test_default_profile_has_dsr_pbo_thresholds_disabled() -> None:
    prof = default_profile()
    assert prof["min_dsr"] is None
    assert prof["max_pbo"] is None


def test_production_profile_blocks_low_dsr() -> None:
    out = evaluate_release_gate(
        confidence=_base_confidence(dsr=0.30, pbo=0.01),
        promotion=_approving_promotion(),
        coverage_report=_approving_coverage(),
        profile="production",
    )
    assert not bool(out["approved"].iloc[0])
    assert "deflated_sharpe_below_0.50" in out["reasons"].iloc[0]


def test_production_profile_blocks_high_pbo() -> None:
    out = evaluate_release_gate(
        confidence=_base_confidence(dsr=0.80, pbo=0.30),
        promotion=_approving_promotion(),
        coverage_report=_approving_coverage(),
        profile="production",
    )
    assert not bool(out["approved"].iloc[0])
    assert "probability_of_overfit_above_0.05" in out["reasons"].iloc[0]


def test_production_profile_passes_with_strong_dsr_low_pbo() -> None:
    out = evaluate_release_gate(
        confidence=_base_confidence(dsr=0.80, pbo=0.02),
        promotion=_approving_promotion(),
        coverage_report=_approving_coverage(),
        profile="production",
    )
    assert bool(out["approved"].iloc[0]), out["reasons"].iloc[0]
    # Surface DSR/PBO so the operator can see the realised values.
    assert out["dsr"].iloc[0] == 0.80
    assert out["pbo"].iloc[0] == 0.02


def test_back_compat_no_dsr_pbo_columns_skips_rail() -> None:
    """Legacy callers without DSR/PBO columns must not see a new rail
    failure (back-compat preserved)."""
    out = evaluate_release_gate(
        confidence=_base_confidence(),  # no dsr/pbo columns
        promotion=_approving_promotion(),
        coverage_report=_approving_coverage(),
        profile="production",
    )
    assert bool(out["approved"].iloc[0]), out["reasons"].iloc[0]


def test_explicit_min_dsr_kwarg_overrides_profile_default() -> None:
    """Callers can dial DSR up via the kwarg even in production."""
    out = evaluate_release_gate(
        confidence=_base_confidence(dsr=0.60, pbo=0.02),
        promotion=_approving_promotion(),
        coverage_report=_approving_coverage(),
        profile="production",
        min_dsr=0.90,
    )
    assert "deflated_sharpe_below_0.90" in out["reasons"].iloc[0]


def test_explicit_max_pbo_none_disables_rail_in_production() -> None:
    out = evaluate_release_gate(
        confidence=_base_confidence(dsr=0.80, pbo=0.40),
        promotion=_approving_promotion(),
        coverage_report=_approving_coverage(),
        profile="production",
        max_pbo=None,
    )
    assert bool(out["approved"].iloc[0]), out["reasons"].iloc[0]
