# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pandas as pd

from market_regime_engine.frontier.diagnostics import (
    evaluate_bayesian_msvar_diagnostics,
    evaluate_online_prefix_safety,
)


def test_bayesian_msvar_diagnostics_fail_closed_on_missing_or_bad_values() -> None:
    bad = evaluate_bayesian_msvar_diagnostics(
        {"num_divergences": 2, "max_rhat": 1.2, "min_ess": 20, "max_companion_radius": 1.1}
    )
    assert bad.passed is False
    assert "posterior_divergences" in bad.reasons
    assert "posterior_companion_radius_unstable_or_missing" in bad.reasons

    good = evaluate_bayesian_msvar_diagnostics(
        {"num_divergences": 0, "max_rhat": 1.01, "min_ess": 250, "max_companion_radius": 0.92, "min_state_mass": 0.10}
    )
    assert good.passed is True
    assert good.artifact_hash


def test_online_prefix_safety_detects_retrospective_smoothing_leakage() -> None:
    panel = pd.DataFrame({"x": range(8)})

    def online_score(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"as_of_date": df.index, "factor_value": df["x"].astype(float)})

    assert evaluate_online_prefix_safety(panel, online_score).passed is True

    def smoothed_score(df: pd.DataFrame) -> pd.DataFrame:
        # Every historical value changes when future rows are appended.
        return pd.DataFrame({"as_of_date": df.index, "factor_value": float(df["x"].mean())})

    report = evaluate_online_prefix_safety(panel, smoothed_score)
    assert report.passed is False
    assert "prefix_safety_violation" in report.reasons


def test_bayesian_msvar_diagnostics_fail_closed_when_required_metrics_missing() -> None:
    report = evaluate_bayesian_msvar_diagnostics({"max_rhat": 1.01, "min_ess": 250})
    assert report.passed is False
    assert "posterior_divergences_missing_or_invalid" in report.reasons
    assert "posterior_companion_radius_unstable_or_missing" in report.reasons


def test_online_prefix_safety_fails_closed_on_duplicate_timestamps_and_score_errors() -> None:
    panel = pd.DataFrame({"x": range(8)})

    def duplicate_score(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"as_of_date": [0 for _ in range(len(df))], "factor_value": df["x"].astype(float)})

    duplicate_report = evaluate_online_prefix_safety(panel, duplicate_score)
    assert duplicate_report.passed is False
    assert "duplicate_score_timestamps" in duplicate_report.reasons

    def broken_score(df: pd.DataFrame) -> pd.DataFrame:
        raise RuntimeError("boom")

    broken_report = evaluate_online_prefix_safety(panel, broken_score)
    assert broken_report.passed is False
    assert "score_fn_failed" in broken_report.reasons
