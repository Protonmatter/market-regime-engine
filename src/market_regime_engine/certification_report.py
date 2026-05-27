# SPDX-License-Identifier: Apache-2.0
"""Machine-readable XPro certification report builder.

The certification report is a release-level envelope. It does not replace the
underlying release gate, evidence packs, method cards, or realized-outcome
validation; it ties those controls together into one canonical JSON object that
CI and model-risk review can archive.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from market_regime_engine import __version__ as ENGINE_VERSION
from market_regime_engine.evidence_common import canonical_sha256
from market_regime_engine.fixed_income.execution_validation import (
    certification_confidence_row,
    validate_execution_confidence_realized_outcomes,
)
from market_regime_engine.fixed_income.timestamps import iso8601_z, to_utc
from market_regime_engine.fixed_income.xpro_decision import verify_xpro_decision_artifact
from market_regime_engine.model_runs import _git_dirty, _git_revision, _lockfile_hashes_dict
from market_regime_engine.release_gates import evaluate_release_gate

ARTIFACT_VERSION = "xpro_certification_report_v1"

REQUIRED_METHOD_CARDS = {
    "hmm.md",
    "msvar.md",
    "bayesian_msvar.md",
    "dfm_mq.md",
    "gw.md",
    "mcs.md",
    "conformal.md",
    "execution_confidence.md",
    "protocol_recommendation.md",
    "xpro_decision_artifact.md",
    "liquidity_stress.md",
    "credit_spread_regime.md",
}

REQUIRED_METHOD_CARD_SECTIONS = (
    "## Production status",
    "## Module path",
    "## Mathematical equation",
    "## Inputs",
    "## Outputs",
    "## Assumptions",
    "## Failure modes",
    "## Diagnostics",
    "## Release-gate requirements",
    "## Tests that validate it",
    "## Known limitations",
)


@dataclass(frozen=True)
class CertificationReportOptions:
    validation_dir: Path = Path("data/validation")
    profile: str = "certification"
    asof: str | pd.Timestamp | None = None
    run_execution_validation: bool = True
    dsr: float | None = None
    pbo: float | None = None
    evidence_pack_hmac: str | None = None
    model_card_path: str = "docs/method_cards/execution_confidence.md"
    require_hmac: bool | None = None
    xpro_decision_id: str | None = None
    frontier_diagnostics_json: Path | None = None
    repo_root: Path | None = None


def build_certification_report(
    warehouse: Any,
    *,
    validation_dir: str | Path = "data/validation",
    profile: str = "certification",
    asof: str | pd.Timestamp | None = None,
    run_execution_validation: bool = True,
    dsr: float | None = None,
    pbo: float | None = None,
    evidence_pack_hmac: str | None = None,
    model_card_path: str = "docs/method_cards/execution_confidence.md",
    require_hmac: bool | None = None,
    xpro_decision_id: str | None = None,
    frontier_diagnostics_json: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build and hash an audit-grade certification report.

    The builder is fail-closed: missing realized-outcome validation,
    certification release-gate evidence, method cards, or required HMAC
    material appears as an explicit failed check and makes the top-level
    ``approved`` value false.
    """

    options = CertificationReportOptions(
        validation_dir=Path(validation_dir),
        profile=profile,
        asof=asof,
        run_execution_validation=run_execution_validation,
        dsr=dsr,
        pbo=pbo,
        evidence_pack_hmac=evidence_pack_hmac,
        model_card_path=model_card_path,
        require_hmac=require_hmac,
        xpro_decision_id=xpro_decision_id,
        frontier_diagnostics_json=Path(frontier_diagnostics_json) if frontier_diagnostics_json else None,
        repo_root=Path(repo_root) if repo_root else None,
    )
    asof_utc = _asof_text(options.asof)
    execution_check = _execution_confidence_check(warehouse, options)
    method_cards = audit_method_cards(options.repo_root)
    frontier = _frontier_diagnostics_check(options.frontier_diagnostics_json)
    xpro = _xpro_decision_check(warehouse, options.xpro_decision_id, require_hmac=options.require_hmac)
    release_gate = _release_gate_check(warehouse, options)
    checks = {
        "execution_confidence": execution_check,
        "release_gate": release_gate,
        "method_cards": method_cards,
        "frontier": frontier,
        "xpro_decision": xpro,
    }
    reasons = _collect_reasons(checks)
    approved = bool(
        execution_check.get("passed")
        and release_gate.get("approved")
        and method_cards.get("passed")
        and frontier.get("passed")
        and xpro.get("passed")
    )
    report = {
        "artifact_version": ARTIFACT_VERSION,
        "asof_utc": asof_utc,
        "approved": approved,
        "decision": "release" if approved else "hold",
        "profile": options.profile,
        "reasons": reasons if reasons else ["passed"],
        "build": _build_metadata(options.repo_root),
        "inputs": {
            "validation_dir": options.validation_dir.as_posix(),
            "model_card_path": options.model_card_path,
            "xpro_decision_id": options.xpro_decision_id,
            "frontier_diagnostics_json": options.frontier_diagnostics_json.as_posix()
            if options.frontier_diagnostics_json
            else None,
        },
        "checks": checks,
        "release_gate": release_gate,
    }
    return sign_certification_report(report)


def sign_certification_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``report`` with a deterministic ``artifact_hash`` field."""

    payload = dict(report)
    payload.pop("artifact_hash", None)
    normalized = _json_safe(payload)
    normalized["artifact_hash"] = canonical_sha256(normalized, version="v2")
    return normalized


def verify_certification_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the report hash and return a verifier envelope."""

    expected = str(report.get("artifact_hash") or "")
    payload = dict(report)
    payload.pop("artifact_hash", None)
    actual = canonical_sha256(_json_safe(payload), version="v2")
    return {
        "verified": bool(expected and expected == actual),
        "artifact_hash": expected,
        "computed_hash": actual,
        "reasons": [] if expected and expected == actual else ["artifact_hash_mismatch"],
    }


def write_certification_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_json_safe(dict(report)), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return out


def audit_method_cards(repo_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    cards_dir = root / "docs" / "method_cards"
    missing_files: list[str] = []
    section_failures: dict[str, list[str]] = {}
    missing_tests: list[str] = []
    for name in sorted(REQUIRED_METHOD_CARDS):
        path = cards_dir / name
        if not path.exists():
            missing_files.append(name)
            continue
        text = path.read_text(encoding="utf-8")
        missing_sections = [section for section in REQUIRED_METHOD_CARD_SECTIONS if section not in text]
        if missing_sections:
            section_failures[name] = missing_sections
        if "tests/" not in text:
            missing_tests.append(name)
    passed = not missing_files and not section_failures and not missing_tests
    reasons: list[str] = []
    if missing_files:
        reasons.append("method_card_files_missing")
    if section_failures:
        reasons.append("method_card_sections_missing")
    if missing_tests:
        reasons.append("method_card_test_references_missing")
    return {
        "passed": passed,
        "checked": sorted(REQUIRED_METHOD_CARDS),
        "missing_files": missing_files,
        "missing_sections": section_failures,
        "missing_test_references": missing_tests,
        "reasons": reasons,
    }


def _execution_confidence_check(warehouse: Any, options: CertificationReportOptions) -> dict[str, Any]:
    if not options.run_execution_validation:
        return {
            "passed": False,
            "status": "skipped",
            "reasons": ["execution_validation_skipped"],
        }
    hmac_value = options.evidence_pack_hmac or _latest_evidence_hmac(warehouse)
    try:
        report = validate_execution_confidence_realized_outcomes(warehouse, asof=options.asof)
        row = certification_confidence_row(
            report,
            date=options.asof,
            dsr=options.dsr,
            pbo=options.pbo,
            model_card_path=options.model_card_path,
            evidence_pack_hmac=hmac_value,
        )
        row_payload = row.iloc[0].to_dict()
        metadata = json.loads(str(row_payload.get("metadata_json") or "{}"))
        for key, value in row_payload.items():
            if key in {"date", "confidence", "grade", "metadata_json"}:
                continue
            metadata[key] = value
        row.loc[:, "metadata_json"] = json.dumps(_json_safe(metadata), sort_keys=True)
        warehouse.write_confidence_scores(row)
        return {
            "passed": bool(report.passed),
            "status": "validated" if report.passed else "hold",
            "report": report.to_dict(),
            "confidence_row": {
                "date": row_payload.get("date"),
                "confidence": row_payload.get("confidence"),
                "grade": row_payload.get("grade"),
                "evidence_pack_hmac": hmac_value,
            },
            "reasons": list(report.reasons),
        }
    except Exception as exc:
        return {
            "passed": False,
            "status": "failed",
            "reasons": ["execution_validation_failed"],
            "error": str(exc),
        }


def _release_gate_check(warehouse: Any, options: CertificationReportOptions) -> dict[str, Any]:
    try:
        gate = evaluate_release_gate(
            confidence=warehouse.read_confidence_scores(),
            drift=warehouse.read_model_drift(),
            invalidation=warehouse.read_invalidation_triggers(),
            promotion=_read_promotion(options.validation_dir),
            coverage_report=_read_coverage_report(warehouse, options.validation_dir),
            profile=options.profile,  # type: ignore[arg-type]
        )
        row = gate.iloc[0].to_dict()
    except Exception as exc:
        return {
            "approved": False,
            "decision": "hold",
            "reasons": ["release_gate_failed"],
            "error": str(exc),
        }
    reasons = _split_reasons(row.get("reasons"))
    return {
        "approved": bool(row.get("approved")),
        "decision": str(row.get("decision", "hold")),
        "reasons": reasons,
        "row": _json_safe(row),
    }


def _xpro_decision_check(warehouse: Any, decision_id: str | None, *, require_hmac: bool | None) -> dict[str, Any]:
    if not decision_id:
        return {
            "passed": True,
            "status": "not_checked",
            "reasons": [],
        }
    try:
        latest = warehouse.latest_xpro_decision_artifact(decision_id)
    except Exception as exc:
        return {
            "passed": False,
            "status": "read_failed",
            "reasons": ["xpro_decision_read_failed"],
            "error": str(exc),
        }
    if latest is None or latest.empty:
        return {
            "passed": False,
            "status": "not_found",
            "reasons": ["xpro_decision_not_found"],
        }
    try:
        artifact = json.loads(str(latest.iloc[0]["payload_json"]))
        verification = verify_xpro_decision_artifact(artifact, require_hmac=require_hmac)
    except Exception as exc:
        return {
            "passed": False,
            "status": "verify_failed",
            "reasons": ["xpro_decision_verify_failed"],
            "error": str(exc),
        }
    return {
        "passed": bool(verification.get("verified")),
        "status": "verified" if verification.get("verified") else "failed",
        "verification": verification,
        "reasons": list(verification.get("reasons") or []),
    }


def _frontier_diagnostics_check(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "passed": True,
            "status": "disabled",
            "reasons": [],
            "diagnostics": [],
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "passed": False,
            "status": "failed",
            "reasons": ["frontier_diagnostics_unreadable"],
            "error": str(exc),
            "diagnostics": [],
        }
    diagnostics = payload if isinstance(payload, list) else payload.get("diagnostics", [payload])
    if not isinstance(diagnostics, list):
        diagnostics = [diagnostics]
    failed = [d for d in diagnostics if not bool(isinstance(d, Mapping) and d.get("passed"))]
    return {
        "passed": not failed,
        "status": "passed" if not failed else "failed",
        "reasons": [] if not failed else ["frontier_diagnostics_failed"],
        "diagnostics": _json_safe(diagnostics),
    }


def _read_promotion(validation_dir: Path) -> pd.DataFrame:
    path = validation_dir / "model_promotion.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_coverage_report(warehouse: Any, validation_dir: Path) -> pd.DataFrame:
    for name in ("coverage_report.csv", "conditional_coverage_report.csv"):
        path = validation_dir / name
        if path.exists():
            return pd.read_csv(path)
    try:
        frame = warehouse.read_conditional_coverage_report()
    except Exception:
        return pd.DataFrame()
    return frame


def _latest_evidence_hmac(warehouse: Any) -> str | None:
    try:
        packs = warehouse.read_evidence_packs()
    except Exception:
        return None
    if packs is None or packs.empty or "hmac_signature" not in packs:
        return None
    series = packs["hmac_signature"].dropna().astype(str)
    series = series[series.str.strip() != ""]
    if series.empty:
        return None
    return str(series.iloc[-1])


def _build_metadata(repo_root: Path | None) -> dict[str, Any]:
    return {
        "engine_version": ENGINE_VERSION,
        "git_sha": _git_revision(short=False),
        "git_sha_short": _git_revision(short=True),
        "git_dirty": _git_dirty(),
        "lockfile_hashes": _lockfile_hashes_dict(repo_root),
    }


def _asof_text(asof: str | pd.Timestamp | None) -> str:
    ts = to_utc(asof) if asof is not None else pd.Timestamp.utcnow()
    if ts is None:
        ts = pd.Timestamp.utcnow()
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return iso8601_z(ts)


def _split_reasons(value: Any) -> list[str]:
    if value is None:
        return []
    text = str(value)
    if not text or text == "passed":
        return []
    return [part for part in text.split(",") if part]


def _collect_reasons(checks: Mapping[str, Mapping[str, Any]]) -> list[str]:
    reasons: list[str] = []
    for name, check in checks.items():
        raw = check.get("reasons") or []
        if isinstance(raw, str):
            raw = [raw]
        for reason in raw:
            text = str(reason)
            if text and text != "passed":
                reasons.append(f"{name}:{text}")
    return reasons


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return [_json_safe(v) for v in sorted(value, key=repr)]
    if isinstance(value, pd.Timestamp):
        ts = value.tz_convert("UTC") if value.tzinfo else value.tz_localize("UTC")
        return iso8601_z(ts)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, Path):
        return value.as_posix()
    return value


__all__ = [
    "ARTIFACT_VERSION",
    "CertificationReportOptions",
    "audit_method_cards",
    "build_certification_report",
    "sign_certification_report",
    "verify_certification_report",
    "write_certification_report",
]
