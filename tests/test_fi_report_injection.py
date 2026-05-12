# SPDX-License-Identifier: Apache-2.0
"""Regression — FI report Markdown / HTML injection sanitizer.

Pre-fix (REVIEW.md Tier-1 C-AUTO-5): the FI report interpolated
operator-untrusted DB strings (scope_id, regime_label, model_run_id,
release-gate reasons, ...) raw into Markdown table cells. The HTML
render path then passed those cells through ``markdown.markdown(...)``
which expands HTML in cells — so a CUSIP-shaped attacker input like
``9128283N8\\n<script>alert(1)</script>`` historically broke out into
actual ``<script>`` tags.

Post-fix: every DB-sourced cell in the four affected sections
(liquidity, TCA, release-gate reasons, evidence packs) routes through
:func:`_safe_md_cell` which:

- escapes pipes / backticks / backslashes (Markdown table syntax),
- collapses newlines + carriage returns to a single space (row
  injection),
- HTML-escapes the result so ``markdown.markdown(...)`` cannot expand
  HTML entities in the rendered output.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

import market_regime_engine.fixed_income  # noqa: F401  - register FI tables
from market_regime_engine.fixed_income.report import (
    _safe_md_cell,
    generate_fi_report,
)
from market_regime_engine.storage import Warehouse


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


# ---------------------------------------------------------------------------
# Unit tests for the _safe_md_cell helper.
# ---------------------------------------------------------------------------


def test_md_cell_escapes_pipe() -> None:
    """A raw pipe must be escaped so it doesn't break the column count
    of a Markdown table row."""
    out = _safe_md_cell("a|b|c")
    assert "|" not in out.replace("\\|", "")


def test_md_cell_escapes_backtick() -> None:
    """A raw backtick must be escaped so it cannot break out of an
    inline-code wrapper."""
    out = _safe_md_cell("`escape`")
    assert "`" not in out.replace("\\`", "")


def test_md_cell_escapes_backslash() -> None:
    """Backslash must be escaped first so subsequent escapes don't get
    double-escaped accidentally."""
    out = _safe_md_cell("\\path")
    assert out.startswith("\\\\")


def test_md_cell_escapes_html_tags() -> None:
    """HTML metacharacters must be escaped so ``markdown.markdown(...)``
    cannot expand them on the HTML render path."""
    out = _safe_md_cell("<script>alert(1)</script>")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out
    assert "&lt;/script&gt;" in out


def test_md_cell_escapes_newlines() -> None:
    """Any newline / CR sequence must collapse to a space so an
    attacker cannot terminate the current row and inject a new one."""
    assert _safe_md_cell("a\nb") == "a b"
    assert _safe_md_cell("a\rb") == "a b"
    assert _safe_md_cell("a\r\nb") == "a b"


def test_md_cell_none_returns_empty_string() -> None:
    assert _safe_md_cell(None) == ""


def test_md_cell_numeric_value_renders_as_string() -> None:
    assert _safe_md_cell(3.14) == "3.14"
    assert _safe_md_cell(42) == "42"


# ---------------------------------------------------------------------------
# Integration test — end-to-end report generation with a malicious row.
# ---------------------------------------------------------------------------


def _persist_malicious_evidence_pack(wh: Warehouse) -> None:
    from market_regime_engine.fixed_income.evidence_pack import (
        build_evidence_pack,
        write_evidence_pack,
    )

    pack = build_evidence_pack(
        model_run_id="run-mal\n<script>alert(1)</script>",
        component_name="credit_regime",
        model_version="0.1.0",
        code_sha="abc",
        model_hash="sha256:m",
        input_features_hash="sha256:in",
        output_hash="sha256:out",
        release_gate=True,
        timestamp="2026-05-08T16:00:00Z",
    )
    write_evidence_pack(wh, pack, request_id="req|injection|attempt")


def test_full_report_with_malicious_scope_id_renders_literal_text(
    tmp_path,
) -> None:
    """Populate the warehouse with a liquidity row whose scope_id
    embeds a CUSIP-style attacker payload, render the report in BOTH
    Markdown and HTML, and assert no ``<script>`` survives in either
    output."""
    wh = Warehouse(str(tmp_path / "injection.duckdb"))
    try:
        wh.write_liquidity_stress_score(
            pd.DataFrame(
                [
                    {
                        "model_run_id": "run-mal",
                        "scope_type": "cusip",
                        "scope_id": "9128283N8\n<script>alert(1)</script>",
                        "timestamp": "2026-05-08T16:00:00Z",
                        "liquidity_score": 30.0,
                        "liquidity_label": "Mild|Stress`backtick`",
                        "confidence": 0.9,
                        "drivers_json": json.dumps([]),
                        "release_gate": 1,
                        "artifact_hash": "sha256:" + "b" * 64,
                        "metadata_json": "{}",
                    }
                ]
            )
        )
        md_body = generate_fi_report(wh, output_format="markdown")
        html_body = generate_fi_report(wh, output_format="html")
    finally:
        wh.close()

    # Markdown body: the raw attacker payload must NOT survive intact.
    assert "<script>" not in md_body
    assert "</script>" not in md_body
    # Newline embedded by the attacker must NOT terminate the row;
    # the literal character is collapsed to a space.
    assert "9128283N8 " in md_body or "9128283N8\\" in md_body  # safe forms
    # Pipe inside the label must be escaped.
    assert "Mild\\|Stress" in md_body
    # HTML body: scripts must be inert. Both render paths (the optional
    # ``markdown`` lib and the no-deps ``<pre>`` fallback) must
    # neutralise the script tag.
    assert "<script>" not in html_body
    assert "<script>alert(1)</script>" not in html_body


def test_full_report_with_malicious_release_gate_reasons(tmp_path) -> None:
    """The release_gates ``reasons`` field is operator-controlled in
    some test rigs; assert injection there doesn't survive either."""
    wh = Warehouse(str(tmp_path / "release-gate-injection.duckdb"))
    try:
        wh.write_release_gates(
            pd.DataFrame(
                [
                    {
                        "date": "2026-05-08",
                        "approved": True,
                        "decision": "release",
                        "confidence": 0.8,
                        "confidence_grade": "high",
                        "severe_drift": 0,
                        "major_drift": 0,
                        "max_psi": 0.05,
                        "high_invalidation_triggers": 0,
                        "active_trigger_names": "[]",
                        "reasons": "ok\n<script>alert(1)</script>",
                        "metadata_json": "{}",
                        "resolved_profile": "production<img>",
                    }
                ]
            )
        )
        md_body = generate_fi_report(wh, output_format="markdown")
        html_body = generate_fi_report(wh, output_format="html")
    finally:
        wh.close()

    assert "<script>" not in md_body
    assert "<img>" not in md_body
    assert "<script>" not in html_body
    assert "<img>" not in html_body


def test_full_report_with_malicious_evidence_pack(tmp_path) -> None:
    """Evidence pack rows ALSO route through ``_safe_md_cell`` —
    a malicious model_run_id / request_id must render as literal
    text, not executable HTML."""
    wh = Warehouse(str(tmp_path / "evidence-injection.duckdb"))
    try:
        _persist_malicious_evidence_pack(wh)
        md_body = generate_fi_report(wh, output_format="markdown")
        html_body = generate_fi_report(wh, output_format="html")
    finally:
        wh.close()

    assert "<script>" not in md_body
    assert "<script>" not in html_body
    # The injected pipe in request_id must be escaped.
    assert "req\\|injection\\|attempt" in md_body
