# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pandas as pd

from market_regime_engine.prediction_evidence import (
    EvidenceThresholds,
    binary_forecast_evidence,
    build_prediction_evidence_report,
    quantile_forecast_evidence,
)


def _good_binary_frame(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    p = rng.uniform(0.05, 0.95, size=n)
    y = rng.binomial(1, p)
    return pd.DataFrame(
        {
            "date": pd.date_range("2010-01-01", periods=n, freq="ME"),
            "target": "drawdown_gt_10pct",
            "horizon": "3m",
            "model_name": "candidate",
            "y": y,
            "p": p,
            "regime": np.where(np.arange(n) % 2 == 0, "expansion", "stress"),
        }
    )


def test_binary_evidence_passes_calibrated_probabilities() -> None:
    metrics, checks, regimes, tails = binary_forecast_evidence(
        _good_binary_frame(),
        thresholds=EvidenceThresholds(max_brier=0.35, max_log_loss=1.0, max_ece=0.20, max_regime_ece=0.30),
    )
    assert metrics
    assert regimes
    assert tails
    assert all(c.passed for c in checks if c.severity == "blocker"), [c for c in checks if not c.passed]


def test_binary_evidence_fails_bad_calibration() -> None:
    frame = _good_binary_frame()
    frame["p"] = 1.0 - frame["p"]
    _metrics, checks, _regimes, _tails = binary_forecast_evidence(
        frame,
        thresholds=EvidenceThresholds(max_brier=0.20, max_log_loss=0.60, max_ece=0.05),
    )
    assert any(not c.passed and "Brier" in c.name for c in checks)
    assert any(not c.passed and "log loss" in c.name for c in checks)


def test_quantile_evidence_scores_interval_coverage() -> None:
    n = 120
    y = np.linspace(-1.0, 1.0, n)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2010-01-01", periods=n, freq="ME"),
            "target": "forward_return",
            "horizon": "3m",
            "model_name": "candidate_quantile",
            "y": y,
            "q_lo": y - 0.5,
            "q_hi": y + 0.5,
            "q50": y,
        }
    )
    metrics, checks, tails = quantile_forecast_evidence(
        frame,
        thresholds=EvidenceThresholds(min_interval_coverage=0.90),
    )
    assert metrics
    assert tails
    assert all(c.passed for c in checks if c.severity == "blocker"), [c for c in checks if not c.passed]


def test_full_prediction_evidence_report_holds_without_any_inputs() -> None:
    report = build_prediction_evidence_report()
    assert report.approved is False
    assert report.decision == "hold"
    assert report.summary["checks"] == 0


def test_full_prediction_evidence_report_serializes_markdown_and_json() -> None:
    report = build_prediction_evidence_report(
        binary_predictions=_good_binary_frame(),
        thresholds=EvidenceThresholds(max_brier=0.35, max_log_loss=1.0, max_ece=0.20, max_regime_ece=0.30),
    )
    as_json = report.to_json()
    as_md = report.to_markdown()
    assert "Prediction Evidence Report" in as_md
    assert '"decision"' in as_json
