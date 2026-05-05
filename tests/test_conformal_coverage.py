"""Tests for the conformal_coverage warehouse table + the min_coverage gate
wired into ``release_gates.evaluate_release_gate``."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_regime_engine.orchestration import compute_conformal_coverage
from market_regime_engine.release_gates import evaluate_release_gate
from market_regime_engine.storage import Warehouse


@pytest.fixture()
def warehouse_path() -> Path:
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp) / "mre_test.db"


def _binary_validation_frame(seed: int = 0, n: int = 240, alpha_cal: float = 0.10) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    bucket = rng.choice(["risk_on", "risk_off"], size=n)
    p = np.clip(rng.beta(2.0, 5.0, size=n), 1e-3, 1 - 1e-3)
    y = (rng.uniform(size=n) < p).astype(int)
    dates = pd.date_range("2018-01-01", periods=n, freq="MS")
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "target": "drawdown_gt_10pct",
            "horizon": "3m",
            "model": "candidate_logistic",
            "y": y.astype(float),
            "p": p,
            "regime_bucket": bucket,
        }
    )


def test_conformal_coverage_round_trip(warehouse_path: Path) -> None:
    """Two warehouse instances against the same path can write distinct tables
    sequentially without ``database is locked`` (covers item E too)."""
    db1 = Warehouse(warehouse_path)
    try:
        rows = pd.DataFrame(
            [
                {
                    "as_of_date": "2024-01-31",
                    "target": "drawdown_gt_10pct",
                    "horizon": "3m",
                    "bucket": "risk_on",
                    "n": 50,
                    "realized_coverage": 0.92,
                    "target_coverage": 0.90,
                    "threshold": 0.45,
                    "method": "mondrian_binary",
                    "metadata_json": "{}",
                },
                {
                    "as_of_date": "2024-01-31",
                    "target": "drawdown_gt_10pct",
                    "horizon": "3m",
                    "bucket": "risk_off",
                    "n": 30,
                    "realized_coverage": 0.88,
                    "target_coverage": 0.90,
                    "threshold": 0.50,
                    "method": "mondrian_binary",
                    "metadata_json": "{}",
                },
            ]
        )
        n = db1.write_conformal_coverage(rows)
        assert n == 2
    finally:
        db1.close()

    # A second connection must be able to read what the first wrote and write
    # to a different table without contention. WAL+busy_timeout makes this
    # work even on Windows.
    db2 = Warehouse(warehouse_path)
    try:
        out = db2.read_conformal_coverage()
        assert len(out) == 2
        assert {"risk_on", "risk_off"}.issubset(set(out["bucket"]))
        # Sequential write to a sibling table to exercise WAL.
        labels = pd.DataFrame(
            [
                {
                    "date": "2024-01-31",
                    "recession": 0.0,
                    "source": "built_in_nber_windows",
                    "metadata_json": "{}",
                },
            ]
        )
        db2.write_recession_labels(labels)
        assert not db2.read_recession_labels().empty
    finally:
        db2.close()


def test_compute_conformal_coverage_emits_per_bucket_rows(tmp_path: Path) -> None:
    val_dir = tmp_path / "validation"
    val_dir.mkdir()
    frame = _binary_validation_frame(seed=42, n=400)
    frame.to_csv(val_dir / "binary_predictions_3m.csv", index=False)
    out = compute_conformal_coverage(val_dir, alpha=0.10)
    assert not out.empty
    expected_cols = {
        "as_of_date",
        "target",
        "horizon",
        "bucket",
        "n",
        "realized_coverage",
        "target_coverage",
        "threshold",
        "method",
    }
    assert expected_cols.issubset(out.columns)
    assert (out["target_coverage"] == 0.90).all()
    assert set(out["bucket"]) == {"risk_on", "risk_off"}


def test_release_gate_blocks_when_realized_coverage_drops_below_floor() -> None:
    """``min_coverage`` blocks the gate if any bucket's realized coverage is
    more than 5pp below ``1 - alpha``."""
    coverage_report = pd.DataFrame(
        [
            {"realized_coverage": 0.92, "bucket": "risk_on", "n": 80},
            {"realized_coverage": 0.55, "bucket": "risk_off", "n": 40},  # severe drop
        ]
    ).rename(columns={"realized_coverage": "coverage"})
    confidence = pd.DataFrame([{"date": "2024-01-31", "confidence": 0.7, "grade": "B"}])
    promotion = pd.DataFrame(
        [
            {
                "target": "drawdown_gt_10pct",
                "horizon": "3m",
                "promoted": True,
                "mcs_evidence": "absent",
            }
        ]
    )
    # v1.4.1 (item F): profile="default" preserves v1.2.1 confidence
    # threshold (0.55) so we isolate the coverage rail.
    gate = evaluate_release_gate(
        confidence=confidence,
        drift=pd.DataFrame(),
        invalidation=pd.DataFrame(),
        promotion=promotion,
        coverage_report=coverage_report,
        min_coverage=0.85,
        coverage_alpha=0.10,
        coverage_drop_pp=0.05,
        profile="default",
    )
    row = gate.iloc[0]
    assert bool(row["approved"]) is False
    assert "conformal_coverage_below_floor" in str(row["reasons"])
    assert float(row["worst_coverage"]) == pytest.approx(0.55)


def test_release_gate_passes_when_coverage_at_or_above_floor() -> None:
    coverage_report = pd.DataFrame(
        [
            {"coverage": 0.91, "bucket": "risk_on", "n": 80},
            {"coverage": 0.87, "bucket": "risk_off", "n": 40},
        ]
    )
    confidence = pd.DataFrame([{"date": "2024-01-31", "confidence": 0.7, "grade": "B"}])
    promotion = pd.DataFrame(
        [
            {
                "target": "drawdown_gt_10pct",
                "horizon": "3m",
                "promoted": True,
                "mcs_evidence": "absent",
            }
        ]
    )
    # v1.4.1 (item F): profile="default" preserves v1.2.1 baseline.
    gate = evaluate_release_gate(
        confidence=confidence,
        drift=pd.DataFrame(),
        invalidation=pd.DataFrame(),
        promotion=promotion,
        coverage_report=coverage_report,
        min_coverage=0.85,
        profile="default",
    )
    assert bool(gate.iloc[0]["approved"]) is True


def test_release_gate_min_coverage_default_preserves_back_compat() -> None:
    """With the v1.2.1 ``profile="default"`` and ``min_coverage=None``
    the gate must not consult coverage.

    v1.4.1 (item F) flipped the no-flags default to ``production``;
    this test exercises the explicit-default opt-in path so the
    v1.2.1 back-compat semantic is preserved verbatim.
    """
    bad_coverage = pd.DataFrame([{"coverage": 0.10, "bucket": "x", "n": 10}])
    confidence = pd.DataFrame([{"date": "2024-01-31", "confidence": 0.7, "grade": "B"}])
    promotion = pd.DataFrame(
        [
            {
                "target": "drawdown_gt_10pct",
                "horizon": "3m",
                "promoted": True,
                "mcs_evidence": "absent",
            }
        ]
    )
    gate = evaluate_release_gate(
        confidence=confidence,
        drift=pd.DataFrame(),
        invalidation=pd.DataFrame(),
        promotion=promotion,
        coverage_report=bad_coverage,
        profile="default",
    )
    assert bool(gate.iloc[0]["approved"]) is True
