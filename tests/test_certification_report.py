# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from market_regime_engine.cli_dispatch import main as cli_main
from market_regime_engine.storage import Warehouse


def _seed_execution_validation_rows(wh: Warehouse, n: int = 90) -> None:
    base = pd.Timestamp("2026-01-01T14:30:00Z")
    predictions = []
    outcomes = []
    for i in range(n):
        request_id = f"req-cert-{i:03d}"
        decision = base + pd.Timedelta(minutes=i)
        score = 0.25 + 0.70 * (i / max(1, n - 1))
        success = ((i * 23) % 100) / 100 < score
        predictions.append(
            {
                "request_id": request_id,
                "timestamp": decision.isoformat(),
                "model_run_id": "run-cert",
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
                "artifact_hash": f"hash-cert-{i}",
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


def _seed_certification_inputs(tmp_path: Path) -> tuple[Path, Path]:
    import market_regime_engine.fixed_income  # noqa: F401

    db = tmp_path / "certification.duckdb"
    validation_dir = tmp_path / "validation"
    validation_dir.mkdir()
    wh = Warehouse(db)
    try:
        _seed_execution_validation_rows(wh)
        wh.write_model_drift(
            pd.DataFrame(
                [
                    {
                        "date": "2026-01-01",
                        "feature_name": "x",
                        "psi": 0.0,
                        "mean_shift": 0.0,
                        "status": "ok",
                        "metadata_json": "{}",
                    }
                ]
            )
        )
        wh.write_invalidation_triggers(
            pd.DataFrame(
                [
                    {
                        "date": "2026-01-01",
                        "trigger": "none",
                        "severity": "low",
                        "status": "inactive",
                        "value": 0.0,
                        "threshold": 1.0,
                        "metadata_json": "{}",
                    }
                ]
            )
        )
        wh.write_conditional_coverage_report(
            pd.DataFrame(
                [
                    {
                        "as_of_date": "2026-01-01",
                        "target": "execution_fill",
                        "horizon": "intraday",
                        "group": "all",
                        "coverage": 0.95,
                        "n": 90,
                        "alpha": 0.10,
                        "method": "mondrian_binary",
                        "metadata_json": "{}",
                    }
                ]
            )
        )
    finally:
        wh.close()
    pd.DataFrame([{"date": "2026-01-01", "promoted": True, "mcs_evidence": "in_set"}]).to_csv(
        validation_dir / "model_promotion.csv",
        index=False,
    )
    return db, validation_dir


def test_build_certification_report_approves_and_hash_verifies(tmp_path: Path) -> None:
    from market_regime_engine.certification_report import build_certification_report, verify_certification_report

    db, validation_dir = _seed_certification_inputs(tmp_path)
    wh = Warehouse(db)
    try:
        report = build_certification_report(
            wh,
            validation_dir=validation_dir,
            asof="2026-01-02T00:00:00Z",
            dsr=0.75,
            pbo=0.01,
            evidence_pack_hmac="v1:hmac",
        )
    finally:
        wh.close()

    assert report["artifact_version"] == "xpro_certification_report_v1"
    assert report["approved"] is True
    assert report["release_gate"]["decision"] == "release"
    assert report["checks"]["execution_confidence"]["passed"] is True
    assert report["checks"]["method_cards"]["passed"] is True
    assert report["checks"]["frontier"]["status"] == "disabled"
    assert report["artifact_hash"].startswith("sha256:")
    assert verify_certification_report(report)["verified"] is True


def test_certification_report_fails_closed_without_required_hmac(tmp_path: Path) -> None:
    from market_regime_engine.certification_report import build_certification_report

    db, validation_dir = _seed_certification_inputs(tmp_path)
    wh = Warehouse(db)
    try:
        report = build_certification_report(
            wh,
            validation_dir=validation_dir,
            asof="2026-01-02T00:00:00Z",
            dsr=0.75,
            pbo=0.01,
            evidence_pack_hmac=None,
        )
    finally:
        wh.close()

    assert report["approved"] is False
    assert "certification_missing_evidence_pack_hmac" in report["release_gate"]["reasons"]


def test_certification_report_cli_writes_json(tmp_path: Path) -> None:
    db, validation_dir = _seed_certification_inputs(tmp_path)
    out = tmp_path / "certification_report.json"

    rc = cli_main(
        [
            "certification-report",
            "--db",
            str(db),
            "--validation-dir",
            str(validation_dir),
            "--asof",
            "2026-01-02T00:00:00Z",
            "--out-json",
            str(out),
            "--dsr",
            "0.75",
            "--pbo",
            "0.01",
            "--evidence-pack-hmac",
            "v1:hmac",
            "--fail-on-hold",
        ]
    )

    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["approved"] is True
    assert payload["artifact_hash"].startswith("sha256:")


def test_ci_fixture_script_seeds_certification_inputs(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "build_xpro_certification_fixture.py"
    db = tmp_path / "ci-certification.duckdb"
    validation_dir = tmp_path / "ci-validation"
    out = tmp_path / "ci-certification-report.json"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--db",
            str(db),
            "--validation-dir",
            str(validation_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert db.exists()
    assert (validation_dir / "model_promotion.csv").exists()

    rc = cli_main(
        [
            "certification-report",
            "--db",
            str(db),
            "--validation-dir",
            str(validation_dir),
            "--asof",
            "2026-01-02T00:00:00Z",
            "--out-json",
            str(out),
            "--dsr",
            "0.75",
            "--pbo",
            "0.01",
            "--evidence-pack-hmac",
            "v1:ci-certification-fixture",
            "--fail-on-hold",
        ]
    )

    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["approved"] is True
