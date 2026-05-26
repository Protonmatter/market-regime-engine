# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from market_regime_engine.logging_setup import get_logger
from market_regime_engine.training_data import TrainingMode

log = get_logger("mre.cli")


def _resolve_training_mode(args: argparse.Namespace) -> TrainingMode:
    if getattr(args, "legacy_features", False):
        return TrainingMode.LEGACY
    return TrainingMode.POINT_IN_TIME


def _resolve_allow_legacy_fallback(args: argparse.Namespace) -> bool:
    """Surface ``--allow-legacy-fallback`` as a single source of truth.

    When the flag is set without ``--legacy-features`` we emit an explicit
    WARNING — the operator wants the PIT path but is leaving a safety net
    in place. The audit dict on the resulting model run will carry
    ``fallback_authorized = True`` so ``mre verify-run`` can surface the
    deliberate downgrade.
    """
    allow = bool(getattr(args, "allow_legacy_fallback", False))
    if allow and not getattr(args, "legacy_features", False):
        log.warning(
            "PIT path active but legacy fallback authorized as a safety net "
            "(--allow-legacy-fallback set without --legacy-features).",
        )
    return allow


def _training_audit_path(db_path: str) -> Path:
    """Sidecar file next to ``--db`` where the training audit is stashed.

    Persisting the audit on disk lets ``mre model-run`` pick it up later and
    embed it in the reproducibility envelope, even though it runs in a
    separate process from ``train-baseline`` / ``validate``.
    """
    return Path(db_path).parent / "training_audit.json"


def _persist_training_audit(db_path: str, audit: dict) -> Path:
    out = _training_audit_path(db_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, sort_keys=True, default=str), encoding="utf-8")
    return out


def _load_training_audit(db_path: str) -> dict | None:
    path = _training_audit_path(db_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _verify_fi_evidence_pack(db: Any, model_run_id: str) -> dict[str, Any]:
    """v1.5 PR-7 §G — Verify the FI evidence pack for ``model_run_id``.

    Returns a JSON-friendly dict reporting:

    - ``fi_evidence_pack_present``: True when at least one row matches.
    - ``fi_envelope_consistent``: True when the recomputed canonical
      pack hash matches the envelope hash stamped at write time by
      :func:`write_evidence_pack` (v1.5 PR-8 Tier-1 fix C-AUTO-1).
      False when the stored envelope hash is missing
      (``fi_envelope_reason="envelope_hash_missing"``) or when the
      recomputed value differs from the stored one
      (``fi_envelope_reason="envelope_hash_mismatch"``). ``None`` when
      no pack is present.
    - ``fi_envelope_reason``: ``"envelope_hash_missing"`` /
      ``"envelope_hash_mismatch"`` / ``None`` so auditors can tell why
      consistency failed.
    - ``fi_hmac_verified``: True when ``verify_pack`` accepts the
      signature (or no keys are configured and signature is None).

    Adds the quartet to the macro ``verify_run`` report so a single
    operator command surfaces both layers of governance.
    """
    from market_regime_engine.fixed_income.evidence_pack import (
        compute_pack_hash,
        read_evidence_pack,
        stored_envelope_hash,
        verify_pack,
    )
    from market_regime_engine.fixed_income.observability_ext import (
        incr_evidence_pack_verify_fail,
    )

    out: dict[str, Any] = {
        "fi_evidence_pack_present": False,
        "fi_envelope_consistent": None,
        "fi_envelope_reason": None,
        "fi_hmac_verified": None,
    }
    try:
        pack = read_evidence_pack(db, model_run_id=model_run_id)
    except Exception as exc:
        out["fi_envelope_consistent"] = False
        out["fi_envelope_reason"] = "envelope_read_failed"
        out["fi_hmac_verified"] = False
        out["fi_error"] = str(exc)
        # v1.5 PR-8 Tier-1 C-AUTO-3: surface pack-read failures on the
        # operator dashboard.
        incr_evidence_pack_verify_fail(component="unknown", surface="cli_verify_run")
        return out
    if pack is None:
        return out
    out["fi_evidence_pack_present"] = True
    try:
        recomputed = compute_pack_hash(pack)
        out["fi_recomputed_hash"] = recomputed
        expected = stored_envelope_hash(pack)
        out["fi_expected_envelope_hash"] = expected
        if not expected:
            out["fi_envelope_consistent"] = False
            out["fi_envelope_reason"] = "envelope_hash_missing"
        elif recomputed != expected:
            out["fi_envelope_consistent"] = False
            out["fi_envelope_reason"] = "envelope_hash_mismatch"
        else:
            out["fi_envelope_consistent"] = True
            out["fi_envelope_reason"] = None
    except Exception as exc:
        out["fi_envelope_consistent"] = False
        out["fi_envelope_reason"] = "envelope_compare_failed"
        out["fi_envelope_error"] = str(exc)
    try:
        out["fi_hmac_verified"] = bool(verify_pack(pack))
    except Exception as exc:
        out["fi_hmac_verified"] = False
        out["fi_hmac_error"] = str(exc)
    out["fi_component_name"] = pack.component_name
    out["fi_release_gate"] = bool(pack.release_gate)
    # v1.5 PR-8 Tier-1 C-AUTO-3: increment the verify-fail counter once
    # per call where either the envelope check or the HMAC verify
    # failed. The HMAC failure also auto-increments
    # ``fi_hmac_signature_failures_total{reason=...}`` from within
    # ``verify_pack``; this counter is the surface-aware aggregate so
    # the runbook in ``docs/V1_5_HMAC_OPERATIONS.md`` can alert on
    # CLI vs API regressions independently.
    if out["fi_envelope_consistent"] is False or out["fi_hmac_verified"] is False:
        incr_evidence_pack_verify_fail(
            component=str(pack.component_name),
            surface="cli_verify_run",
        )
    return out


__all__ = [
    "_load_training_audit",
    "_persist_training_audit",
    "_resolve_allow_legacy_fallback",
    "_resolve_training_mode",
    "_training_audit_path",
    "_verify_fi_evidence_pack",
]
