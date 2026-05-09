# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

from market_regime_engine.prediction_evidence import EvidenceThresholds, build_prediction_evidence_report

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "prediction_evidence"


def test_good_prediction_evidence_fixtures_pass_release_rails() -> None:
    report = build_prediction_evidence_report(
        binary_predictions=FIXTURE_DIR / "binary_oos_good.csv",
        quantile_predictions=FIXTURE_DIR / "quantile_oos_good.csv",
        thresholds=EvidenceThresholds(),
    )
    assert report.approved is True, report.to_json()
    assert report.decision == "release"
    assert report.summary["failed_blockers"] == 0


def test_bad_binary_fixture_fails_release_rails() -> None:
    report = build_prediction_evidence_report(
        binary_predictions=FIXTURE_DIR / "binary_oos_bad_calibration.csv",
        thresholds=EvidenceThresholds(),
    )
    assert report.approved is False
    assert report.decision == "hold"
    assert report.summary["failed_blockers"] > 0
    failed_names = {check.name for check in report.checks if not check.passed}
    assert any("Brier" in name for name in failed_names)
    assert any("log loss" in name for name in failed_names)


def test_bad_quantile_fixture_fails_release_rails() -> None:
    report = build_prediction_evidence_report(
        quantile_predictions=FIXTURE_DIR / "quantile_oos_bad_coverage.csv",
        thresholds=EvidenceThresholds(),
    )
    assert report.approved is False
    assert report.decision == "hold"
    assert report.summary["failed_blockers"] > 0
    failed_names = {check.name for check in report.checks if not check.passed}
    assert any("interval coverage" in name for name in failed_names)
