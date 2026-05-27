# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""Build a deterministic warehouse fixture for XPro certification CI.

The fixture intentionally uses synthetic rows. Its purpose is to exercise the
certification report's release-gate wiring in CI, not to certify live model
quality or market-data ingestion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

import market_regime_engine.fixed_income  # noqa: F401 - registers FI warehouse tables
from market_regime_engine.storage import Warehouse


def _prediction_and_outcome_rows(n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.Timestamp("2026-01-01T14:30:00Z")
    predictions: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    for i in range(n):
        request_id = f"req-ci-cert-{i:03d}"
        decision = base + pd.Timedelta(minutes=i)
        score = 0.25 + 0.70 * (i / max(1, n - 1))
        success = ((i * 23) % 100) / 100 < score
        predictions.append(
            {
                "request_id": request_id,
                "timestamp": decision.isoformat(),
                "model_run_id": "ci-certification-fixture",
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
                "artifact_hash": "sha256:" + hashlib.sha256(request_id.encode("utf-8")).hexdigest(),
                "metadata_json": json.dumps(
                    {
                        "regime_label": "calm",
                        "liquidity_label": "Normal",
                        "fixture": "xpro_certification_report_ci",
                    },
                    sort_keys=True,
                ),
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
                "metadata_json": json.dumps({"observed_slippage_bps": 35.0 - 28.0 * score}, sort_keys=True),
            }
        )
    return pd.DataFrame(predictions), pd.DataFrame(outcomes)


def build_fixture(db_path: Path, validation_dir: Path, *, rows: int = 90, force: bool = False) -> dict[str, Any]:
    if db_path.exists():
        if not force:
            raise SystemExit(f"{db_path} already exists; pass --force to replace the fixture database")
        db_path.unlink()
    validation_dir.mkdir(parents=True, exist_ok=True)
    predictions, outcomes = _prediction_and_outcome_rows(rows)
    drift = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "feature_name": "ci_fixture",
                "psi": 0.0,
                "mean_shift": 0.0,
                "status": "ok",
                "metadata_json": "{}",
            }
        ]
    )
    invalidation = pd.DataFrame(
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
    coverage = pd.DataFrame(
        [
            {
                "as_of_date": "2026-01-01",
                "target": "execution_fill",
                "horizon": "intraday",
                "group": "all",
                "coverage": 0.95,
                "n": rows,
                "alpha": 0.10,
                "method": "mondrian_binary",
                "metadata_json": "{}",
            }
        ]
    )
    promotion = pd.DataFrame([{"date": "2026-01-01", "promoted": True, "mcs_evidence": "in_set"}])

    db_path.parent.mkdir(parents=True, exist_ok=True)
    wh = Warehouse(db_path)
    try:
        wh.write_execution_confidence_prediction(predictions)
        wh.write_execution_outcomes(outcomes)
        wh.write_model_drift(drift)
        wh.write_invalidation_triggers(invalidation)
        wh.write_conditional_coverage_report(coverage)
    finally:
        wh.close()
    promotion.to_csv(validation_dir / "model_promotion.csv", index=False)
    coverage.to_csv(validation_dir / "conditional_coverage_report.csv", index=False)
    return {
        "db": db_path.as_posix(),
        "validation_dir": validation_dir.as_posix(),
        "prediction_rows": len(predictions),
        "outcome_rows": len(outcomes),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="Output DuckDB fixture path.")
    parser.add_argument("--validation-dir", required=True, help="Output validation directory.")
    parser.add_argument("--rows", type=int, default=90)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = build_fixture(
        Path(args.db),
        Path(args.validation_dir),
        rows=int(args.rows),
        force=bool(args.force),
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
