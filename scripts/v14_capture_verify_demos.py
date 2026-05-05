# SPDX-License-Identifier: Apache-2.0
"""v1.4 — capture verbatim JSON for criteria 7 and 8.

Criterion 7: ``verify-run`` fail-closed PIT demo (regression from v1.2.1).
Criterion 8: ``verify-data`` drift demo (regression from v1.3).

Both are produced against an isolated DuckDB warehouse so the script
can be re-run without touching the live ``data/mre.duckdb``. Outputs:

- ``docs/v14_demo_verify_run.json`` — JSON returned by
  :func:`market_regime_engine.model_runs.verify_run`.
- ``docs/v14_demo_verify_data.json`` — JSON returned by
  :func:`market_regime_engine.verify_data.verify_warehouse_state`.

Run with::

    .venv\\Scripts\\python.exe scripts/v14_capture_verify_demos.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from market_regime_engine.model_runs import (
    build_repro_envelope,
    create_model_run,
    model_run_frame,
    verify_run,
)
from market_regime_engine.storage import Warehouse
from market_regime_engine.verify_data import verify_warehouse_state


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def capture_verify_run_demo() -> dict:
    """Reproduce the v1.2.1 ``training_mode_drift`` fail-closed demo on v1.4."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "demo.duckdb")
        db = Warehouse(db_path)
        try:
            feats = pd.DataFrame(
                [
                    {"feature_name": "f1", "date": "2020-01-01", "value": 0.5, "domain": "labor"},
                ]
            )
            outputs = pd.DataFrame(
                [
                    {
                        "model_name": "logreg",
                        "date": "2020-01-01",
                        "horizon": "3m",
                        "target": "rec",
                        "value": 0.1,
                    }
                ]
            )
            db.write_features(feats)
            db.write_model_outputs(outputs)
            run = create_model_run(
                engine_version="1.4.0-demo",
                purpose="v1.4 fail-closed verify-run demo",
                features=feats,
                model_outputs=outputs,
                training_audit={
                    "mode": "point_in_time",
                    "mode_used": "fail_closed",
                    "fallback_authorized": False,
                    "fallback_reason": "feature_asof_values empty",
                },
            )
            db.write_model_runs(model_run_frame(run))
            runs = db.read_model_runs()
            envelope = build_repro_envelope(features=feats, model_outputs=outputs)
            report = verify_run(
                str(runs.iloc[-1]["run_id"]), runs.iloc[-1], current_envelope=envelope
            )
        finally:
            db.close()
    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / "v14_demo_verify_run.json"
    out.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out}: approved={report.get('approved')}")
    return report


def capture_verify_data_demo() -> dict:
    """Reproduce the v1.3 verify-data drift demo against a DuckDB warehouse."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "drift.duckdb")
        db = Warehouse(db_path)
        try:
            feats = pd.DataFrame(
                [
                    {
                        "feature_name": "f1",
                        "date": "2020-01-01",
                        "value": 0.5,
                        "domain": "labor",
                    }
                ]
            )
            outputs = pd.DataFrame(
                [
                    {
                        "model_name": "logreg",
                        "date": "2020-01-01",
                        "horizon": "3m",
                        "target": "rec",
                        "value": 0.1,
                    }
                ]
            )
            db.write_features(feats)
            db.write_model_outputs(outputs)
            run = create_model_run(
                engine_version="1.4.0-demo",
                purpose="v1.4 verify-data drift demo",
                features=feats,
                model_outputs=outputs,
            )
            db.write_model_runs(model_run_frame(run))

            # Drift the warehouse: rewrite features under the same PK
            # with a different value so the recomputed payload hash
            # diverges from the stored envelope.
            drifted = pd.DataFrame(
                [
                    {
                        "feature_name": "f1",
                        "date": "2020-01-01",
                        "value": 999.99,
                        "domain": "labor",
                    }
                ]
            )
            db.write_features(drifted)
            run_id = str(run.run_id)
        finally:
            db.close()

        report = verify_warehouse_state(run_id=run_id, db_path=db_path)
    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / "v14_demo_verify_data.json"
    out.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out}: approved={report.get('approved')}")
    return report


def main() -> None:
    run_report = capture_verify_run_demo()
    data_report = capture_verify_data_demo()
    if run_report.get("approved") is True:
        raise SystemExit("expected fail-closed verify-run to be rejected")
    if data_report.get("approved") is True:
        raise SystemExit("expected verify-data drift to be rejected")
    print("OK: both demos rejected as expected (exit code = 2 on the CLI path)")


if __name__ == "__main__":
    main()
