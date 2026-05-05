# SPDX-License-Identifier: Apache-2.0
"""v1.4.1 — capture verbatim JSON for acceptance criteria 17 and 18.

Criterion 17: ``verify_run`` rejects arbitrary ``extra`` envelope drift.
Criterion 18: ``mre release-gate`` default = production blocks a v1.2.1
              permissive synthetic run.

Both demos are produced in isolation so the script can be re-run
without touching the live ``data/mre.duckdb``. Outputs:

- ``docs/v141_demo_verify_run_extra_drift.json`` — JSON returned by
  :func:`market_regime_engine.model_runs.verify_run` when the stored
  envelope has ``extra={"foo": "bar"}`` and the current envelope has
  ``extra={"foo": "baz"}``. Must show ``approved=false`` with
  ``differences["extra:foo"]`` populated.
- ``docs/v141_demo_release_gate_default_production.json`` — JSON
  representation of the release-gate row when called with no flags
  AND no ``MRE_ENV`` env var. The synthetic input passes the v1.2.1
  permissive defaults but fails the production defaults
  (``min_confidence=0.65 < 0.75``, ``mcs_evidence="absent"``,
  ``require_mcs_membership=True``). Must show ``approved=false``.

Run with::

    .venv\\Scripts\\python.exe scripts/v141_capture_verify_demos.py
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from market_regime_engine.model_runs import (
    build_repro_envelope,
    create_model_run,
    model_run_frame,
    verify_run,
)
from market_regime_engine.release_gates import evaluate_release_gate
from market_regime_engine.storage import Warehouse

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def capture_verify_run_extra_drift_demo() -> dict[str, Any]:
    """Reproduce the v1.4.1 arbitrary-`extra`-drift fail-closed demo.

    Stores a model_run whose ``extra={"foo": "bar"}``, then constructs a
    fresh ``current_envelope`` with ``extra={"foo": "baz"}`` and runs
    :func:`verify_run`. Expected: ``approved=False`` with
    ``differences["extra:foo"]`` populated.
    """
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
                engine_version="1.4.1-demo",
                purpose="v1.4.1 arbitrary-extra-drift demo",
                features=feats,
                model_outputs=outputs,
            )
            # Inject the operator-supplied ``extra={"foo": "bar"}`` into
            # the stored metadata to simulate a real run that stamps an
            # arbitrary compliance / tenant / run-tag field.
            meta = json.loads(run.metadata_json)
            meta["repro_envelope"]["extra"]["foo"] = "bar"
            patched = pd.DataFrame(
                [
                    {
                        **{k: getattr(run, k) for k in run.__dataclass_fields__},
                        "metadata_json": json.dumps(meta, sort_keys=True),
                    }
                ]
            )
            db.write_model_runs(patched)
            stored_row = db.read_model_runs().iloc[-1]
        finally:
            db.close()
        # Build the *current* envelope with the drifted extra value.
        current = build_repro_envelope(
            features=feats,
            model_outputs=outputs,
            extra={
                "engine_version": "1.4.1-demo",
                "purpose": "v1.4.1 arbitrary-extra-drift demo",
                "foo": "baz",
            },
        )
        report = verify_run(str(stored_row["run_id"]), stored_row, current_envelope=current)
    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / "v141_demo_verify_run_extra_drift.json"
    out.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out}: approved={report.get('approved')}")
    return report


def capture_release_gate_default_production_demo() -> dict[str, Any]:
    """Reproduce the v1.4.1 release-gate-default-production fail-closed demo.

    Builds a synthetic input that passes the v1.2.1 looser defaults
    (``min_confidence=0.55``, ``require_mcs_membership=False``) but
    fails the production defaults
    (``min_confidence=0.75``, ``require_mcs_membership=True``). Calls
    :func:`evaluate_release_gate` with no profile / kwargs and no
    ``MRE_ENV`` env var so the resolution priority falls back to
    ``"production"``. Expected: ``approved=False`` with reasons
    populated.
    """
    # Defensive: defeat any inherited ``MRE_ENV`` so the resolution
    # priority surfaces the production fallback.
    os.environ.pop("MRE_ENV", None)
    confidence = pd.DataFrame(
        [{"date": "2026-05-01", "confidence": 0.65, "grade": "B", "metadata_json": "{}"}]
    )
    drift = pd.DataFrame(columns=["date", "feature_name", "psi", "status"])
    invalidation = pd.DataFrame(columns=["date", "trigger", "severity", "status"])
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
    gate_no_flags = evaluate_release_gate(
        confidence=confidence,
        drift=drift,
        invalidation=invalidation,
        promotion=promotion,
    )
    # Also capture the v1.2.1-permissive control so reviewers can see
    # the same input passes under ``profile="default"``.
    gate_default = evaluate_release_gate(
        confidence=confidence,
        drift=drift,
        invalidation=invalidation,
        promotion=promotion,
        profile="default",
    )
    def _row_to_json_safe(row: pd.Series) -> dict[str, Any]:
        """Coerce pandas/numpy NaN into JSON null so the output is RFC8259-clean."""
        import math

        out: dict[str, Any] = {}
        for k, v in row.to_dict().items():
            if isinstance(v, float) and math.isnan(v):
                out[k] = None
            else:
                out[k] = v
        return out

    record = {
        "scenario": (
            "Synthetic input: confidence=0.65 (passes v1.2.1 0.55 floor, fails "
            "production 0.75 floor) AND mcs_evidence=absent (passes v1.2.1 default "
            "require_mcs_membership=False, fails production require_mcs_membership=True)."
        ),
        "no_flags_no_env": _row_to_json_safe(gate_no_flags.iloc[0]),
        "explicit_profile_default": _row_to_json_safe(gate_default.iloc[0]),
    }
    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / "v141_demo_release_gate_default_production.json"
    out.write_text(
        json.dumps(record, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {out}: no_flags approved={record['no_flags_no_env']['approved']}, "
        f"explicit_default approved={record['explicit_profile_default']['approved']}"
    )
    return record


def main() -> None:
    extra_report = capture_verify_run_extra_drift_demo()
    gate_report = capture_release_gate_default_production_demo()
    if extra_report.get("approved") is True:
        raise SystemExit(
            "expected fail-closed verify-run on arbitrary extra drift to be rejected"
        )
    if extra_report.get("differences", {}).get("extra:foo") is None:
        raise SystemExit(
            "expected differences['extra:foo'] in verify_run report; got "
            f"{extra_report.get('differences')!r}"
        )
    no_flags_row = gate_report["no_flags_no_env"]
    if bool(no_flags_row.get("approved")) is True:
        raise SystemExit(
            "expected release-gate with no flags to block under the production "
            "default; got "
            f"{no_flags_row!r}"
        )
    explicit_default_row = gate_report["explicit_profile_default"]
    if bool(explicit_default_row.get("approved")) is False:
        raise SystemExit(
            "expected release-gate with profile='default' to APPROVE the same "
            "synthetic input (this is the v1.2.1 looser baseline); got "
            f"{explicit_default_row!r}"
        )
    print("OK: both v1.4.1 demos behave as the acceptance criteria require.")


if __name__ == "__main__":
    main()
