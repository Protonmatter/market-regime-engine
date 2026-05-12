# SPDX-License-Identifier: Apache-2.0
"""Regression — HMAC + evidence-pack failure counters are actually emitted.

Pre-fix (REVIEW.md Tier-1 C-AUTO-3): the FI observability module
pre-registered ``fi_hmac_signature_failures_total`` and
``fi_evidence_pack_verify_fail_total`` but neither counter had a real
call site. The HMAC runbook (``docs/V1_5_HMAC_OPERATIONS.md``) tells
on-call to alert when these go up — pre-fix, they never moved.

Post-fix:
- :func:`verify_pack` increments
  ``fi_hmac_signature_failures_total{reason=...}`` on every False
  return path (missing_signature, malformed_signature, key_not_found,
  compare_digest_mismatch).
- :func:`cli._verify_fi_evidence_pack` increments
  ``fi_evidence_pack_verify_fail_total{component, surface}`` once per
  call where either the envelope check or the HMAC verify failed.
"""

from __future__ import annotations

import base64
import json
import secrets
from pathlib import Path

import pandas as pd
import pytest

import market_regime_engine.fixed_income  # noqa: F401  - register FI schema + metrics
from market_regime_engine import observability
from market_regime_engine.cli import _verify_fi_evidence_pack
from market_regime_engine.fixed_income.evidence_pack import (
    build_evidence_pack,
    sign_pack,
    verify_pack,
    write_evidence_pack,
)
from market_regime_engine.storage import Warehouse


def _b64() -> str:
    return base64.b64encode(secrets.token_bytes(32)).decode("ascii")


def _hmac_failures_count(reason: str) -> float:
    snap = observability.metrics().snapshot()
    key = f"fi_hmac_signature_failures_total{{reason={reason}}}"
    return float(snap["counters"].get(key, 0.0))


def _evidence_pack_verify_fail_count(*, component: str, surface: str) -> float:
    snap = observability.metrics().snapshot()
    key = (
        "fi_evidence_pack_verify_fail_total"
        f"{{component={component},surface={surface}}}"
    )
    return float(snap["counters"].get(key, 0.0))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    for name in (
        "MRE_FI_HMAC_KEY_VERSIONS",
        "MRE_FI_HMAC_KEY",
        "MRE_FI_REQUIRE_HMAC",
        "MRE_ENV",
    ):
        monkeypatch.delenv(name, raising=False)
    yield


def _persist_pack(
    warehouse: Warehouse,
    *,
    model_run_id: str,
    request_id: str,
    sign: bool | None = None,
):
    pack = build_evidence_pack(
        model_run_id=model_run_id,
        component_name="credit_regime",
        model_version="0.1.0",
        code_sha="abcdef0",
        model_hash="sha256:m",
        input_features_hash="sha256:in",
        output_hash="sha256:out",
        release_gate=True,
        data_vintages={"trace_trades": "2026-05-08T16:00:00Z"},
        timestamp="2026-05-08T16:00:00Z",
    )
    return write_evidence_pack(warehouse, pack, request_id=request_id, sign=sign)


def test_verify_pack_increments_hmac_failure_counter_on_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The HMAC compare-digest mismatch path must drive the
    ``compare_digest_mismatch`` reason on the metric."""
    monkeypatch.setenv("MRE_FI_HMAC_KEY_VERSIONS", json.dumps({"v1": _b64()}))
    pack = build_evidence_pack(
        model_run_id="tamper",
        component_name="credit_regime",
        model_version="0.1.0",
        code_sha="abc",
        model_hash="sha256:m",
        input_features_hash="sha256:in",
        output_hash="sha256:out",
        release_gate=True,
        timestamp="2026-05-08T16:00:00Z",
    )
    signed = sign_pack(pack)
    # Tamper: keep version prefix but flip a digit in the digest.
    bad_sig = signed.hmac_signature
    assert bad_sig is not None
    version, _, hexd = bad_sig.partition(":")
    flipped_hex = ("a" if hexd[0] != "a" else "b") + hexd[1:]
    import dataclasses

    tampered = dataclasses.replace(signed, hmac_signature=f"{version}:{flipped_hex}")
    before = _hmac_failures_count("compare_digest_mismatch")
    assert verify_pack(tampered) is False
    after = _hmac_failures_count("compare_digest_mismatch")
    assert after == pytest.approx(before + 1.0)


def test_verify_pack_increments_with_correct_reason_label_for_each_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each False return branch in verify_pack uses a distinct reason
    label so the runbook can attribute failures."""
    monkeypatch.setenv("MRE_FI_HMAC_KEY_VERSIONS", json.dumps({"v1": _b64()}))

    base = build_evidence_pack(
        model_run_id="reason",
        component_name="credit_regime",
        model_version="0.1.0",
        code_sha="abc",
        model_hash="sha256:m",
        input_features_hash="sha256:in",
        output_hash="sha256:out",
        release_gate=True,
        timestamp="2026-05-08T16:00:00Z",
    )

    import dataclasses

    # missing_signature: signature is None but keys are configured.
    before_missing = _hmac_failures_count("missing_signature")
    no_sig = dataclasses.replace(base, hmac_signature=None)
    assert verify_pack(no_sig) is False
    assert _hmac_failures_count("missing_signature") == pytest.approx(
        before_missing + 1.0
    )

    # malformed_signature: missing ":" separator.
    before_malformed = _hmac_failures_count("malformed_signature")
    bad_format = dataclasses.replace(base, hmac_signature="garbage-no-colon")
    assert verify_pack(bad_format) is False
    assert _hmac_failures_count("malformed_signature") == pytest.approx(
        before_malformed + 1.0
    )

    # key_not_found: version prefix is unknown.
    before_key_missing = _hmac_failures_count("key_not_found")
    unknown_key = dataclasses.replace(base, hmac_signature="v99:abcdef")
    assert verify_pack(unknown_key) is False
    assert _hmac_failures_count("key_not_found") == pytest.approx(
        before_key_missing + 1.0
    )


def test_verify_pack_does_not_increment_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy-path verify_pack must NOT increment any HMAC failure
    counter — false-positive alerts erode the runbook's signal."""
    monkeypatch.setenv("MRE_FI_HMAC_KEY_VERSIONS", json.dumps({"v1": _b64()}))
    pack = build_evidence_pack(
        model_run_id="ok",
        component_name="credit_regime",
        model_version="0.1.0",
        code_sha="abc",
        model_hash="sha256:m",
        input_features_hash="sha256:in",
        output_hash="sha256:out",
        release_gate=True,
        timestamp="2026-05-08T16:00:00Z",
    )
    signed = sign_pack(pack)
    before_each = {
        reason: _hmac_failures_count(reason)
        for reason in (
            "missing_signature",
            "malformed_signature",
            "key_not_found",
            "compare_digest_mismatch",
            "compare_digest_error",
        )
    }
    assert verify_pack(signed) is True
    for reason, before in before_each.items():
        assert _hmac_failures_count(reason) == pytest.approx(before), reason


def test_verify_run_increments_evidence_pack_verify_fail_counter_on_envelope_inconsistent(
    tmp_path: Path,
) -> None:
    """The CLI verify-run path must surface envelope inconsistency on
    the per-surface evidence-pack-verify-fail counter so the runbook
    can alert on CLI vs API regressions independently."""
    wh = Warehouse(str(tmp_path / "counter-cli.duckdb"))
    try:
        _persist_pack(wh, model_run_id="run-counter", request_id="req-counter")
        df = wh.read_evidence_packs()
        df.loc[df["model_run_id"] == "run-counter", "output_hash"] = (
            "sha256:tampered"
        )
        wh.write_evidence_pack(df)
        before = _evidence_pack_verify_fail_count(
            component="credit_regime", surface="cli_verify_run"
        )
        report = _verify_fi_evidence_pack(wh, "run-counter")
    finally:
        wh.close()
    assert report["fi_envelope_consistent"] is False
    after = _evidence_pack_verify_fail_count(
        component="credit_regime", surface="cli_verify_run"
    )
    assert after == pytest.approx(before + 1.0)


def test_verify_run_does_not_increment_evidence_pack_counter_on_clean_pack(
    tmp_path: Path,
) -> None:
    """A clean pack must not increment the evidence-pack-verify-fail
    counter for the CLI surface — false positives erode signal."""
    wh = Warehouse(str(tmp_path / "counter-clean.duckdb"))
    try:
        _persist_pack(wh, model_run_id="run-clean", request_id="req-clean")
        before = _evidence_pack_verify_fail_count(
            component="credit_regime", surface="cli_verify_run"
        )
        report = _verify_fi_evidence_pack(wh, "run-clean")
    finally:
        wh.close()
    assert report["fi_envelope_consistent"] is True
    after = _evidence_pack_verify_fail_count(
        component="credit_regime", surface="cli_verify_run"
    )
    assert after == pytest.approx(before)
