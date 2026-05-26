# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from market_regime_engine.fixed_income.cli import run as fi_cli
from market_regime_engine.release_gates import evaluate_release_gate
from market_regime_engine.storage import Warehouse


def _seed_validation_rows(wh: Warehouse, n: int = 90) -> None:
    base = pd.Timestamp("2026-01-01T14:30:00Z")
    predictions = []
    outcomes = []
    for i in range(n):
        request_id = f"req-{i:03d}"
        decision = base + pd.Timedelta(minutes=i)
        score = 0.25 + 0.70 * (i / max(1, n - 1))
        success = score >= 0.50
        predictions.append(
            {
                "request_id": request_id,
                "timestamp": decision.isoformat(),
                "model_run_id": "run-1",
                "cusip": "000000AA0",
                "side": "buy",
                "notional": 1_000_000.0,
                "protocol": "Auto-X",
                "confidence_score": score,
                "expected_slippage_bps": 40.0 - 25.0 * score,
                "confidence_interval_low": max(0.0, score - 0.1),
                "confidence_interval_high": min(1.0, score + 0.1),
                "recommended_action": "Auto-X caution / trader confirm",
                "human_review_required": 0,
                "release_gate": 1,
                "artifact_hash": f"hash-{i}",
                "metadata_json": json.dumps({"regime_label": "calm", "liquidity_label": "Normal"}),
            }
        )
        outcomes.append(
            {
                "request_id": request_id,
                "cusip": "000000AA0",
                "side": "buy",
                "notional": 1_000_000.0,
                "filled_quantity": 1_000_000.0 if success else 0.0,
                "execution_price": 100.0,
                "observed_at": (decision + pd.Timedelta(minutes=10)).isoformat(),
                "outcome_observation_lag": 600.0,
                "decision_timestamp": decision.isoformat(),
                "metadata_json": json.dumps({"observed_slippage_bps": 35.0 - 28.0 * score}),
            }
        )
    wh.write_execution_confidence_prediction(pd.DataFrame(predictions))
    wh.write_execution_outcomes(pd.DataFrame(outcomes))


def test_validation_cli_writes_release_gate_ready_confidence_row(tmp_path: Path, capsys) -> None:
    db = tmp_path / "validation.duckdb"
    wh = Warehouse(db)
    _seed_validation_rows(wh)
    wh.close()
    out_path = tmp_path / "xpro_certification_report.json"
    rc = fi_cli(
        [
            "fi-validate-execution-confidence",
            "--db",
            str(db),
            "--asof",
            "2026-01-02T00:00:00Z",
            "--out-json",
            str(out_path),
            "--dsr",
            "0.75",
            "--pbo",
            "0.01",
            "--evidence-pack-hmac",
            "v1:hmac",
        ]
    )
    assert rc == 0
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["execution_confidence"]["artifact_hash"].startswith("sha256:")
    wh2 = Warehouse(db)
    try:
        conf = wh2.read_confidence_scores()
    finally:
        wh2.close()
    gate = evaluate_release_gate(
        confidence=conf,
        profile="certification",
        drift=pd.DataFrame([{"date": "2026-01-01", "feature_name": "x", "psi": 0.0, "status": "ok"}]),
        invalidation=pd.DataFrame([{"date": "2026-01-01", "trigger": "none", "severity": "low", "status": "inactive"}]),
        promotion=pd.DataFrame([{"date": "2026-01-01", "promoted": True, "mcs_evidence": "in_set"}]),
        coverage_report=pd.DataFrame([{"coverage": 0.95, "bucket": "all", "n": 90}]),
    )
    assert "certification_missing_validation_artifact_hash" not in str(gate.iloc[0]["reasons"])
