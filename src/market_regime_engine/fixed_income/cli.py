# SPDX-License-Identifier: Apache-2.0
"""Fixed-Income CLI entry points — PR-1 stubs.

Per ``MRE_FIXED_INCOME_INSTRUCTIONS.md §8``: the FI feature ships 7
``fi-*`` subcommands on the ``mre`` CLI. PR-1 lands the full argparse
surface (so flags don't shift across PRs and downstream automation can
hard-code the call sites today) plus stub handlers that emit the
canonical ``not_yet_implemented`` JSON payload and return exit code 0.

Subsequent PRs swap each stub for the real implementation:

- PR-3 ``fi-build-features``, ``fi-score-credit-regime``
- PR-4 ``fi-score-liquidity``
- PR-5 ``fi-score-execution-confidence``
- PR-6 ``fi-tca-segment``
- PR-7 ``fi-evidence-pack``, ``fi-report``

The dispatcher lives in :func:`run`; ``cli_dispatch.main`` routes any
argv whose first token starts with ``fi-`` here.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

CLI_COMMANDS: tuple[str, ...] = (
    "fi-build-features",
    "fi-score-credit-regime",
    "fi-score-liquidity",
    "fi-score-execution-confidence",
    "fi-tca-segment",
    "fi-evidence-pack",
    "fi-report",
)

# Commands fully implemented in this PR. Stub-emitting commands fall
# back to ``_emit_stub`` below.
_LIVE_COMMANDS: frozenset[str] = frozenset({"fi-score-credit-regime"})


def _emit_stub(command: str) -> int:
    """Print the canonical ``not_yet_implemented`` JSON envelope.

    Newline-terminated so shell pipelines can consume it without
    needing ``--no-pretty``. Exit code 0 mirrors the ``snapshot-build``
    stub pattern in v1.4 ``cli_dispatch._cmd_snapshot_build``.
    """
    payload = {"status": "not_yet_implemented", "command": command}
    print(json.dumps(payload, sort_keys=True))
    return 0


def _build_fi_build_features(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "fi-build-features",
        help="Build PIT-safe FI features from TRACE/RFQ/quotes/curves (PR-3+).",
    )
    parser.add_argument("--db", help="DuckDB warehouse path.", default="data/mre.duckdb")
    parser.add_argument("--asof", help="ISO-8601 as-of timestamp; defaults to now (UTC).")
    parser.add_argument("--out", help="Optional output path for materialised features.")


def _build_fi_score_credit_regime(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "fi-score-credit-regime",
        help="Score the latest credit-spread-regime index (PR-3).",
    )
    parser.add_argument("--db", help="DuckDB warehouse path.", default="data/mre.duckdb")
    parser.add_argument("--asof", help="ISO-8601 as-of timestamp; defaults to now (UTC).")
    parser.add_argument(
        "--profile",
        help="Operating profile (production | default).",
        default="production",
    )
    parser.add_argument(
        "--release-gate",
        help="Inbound governance flag (default: true).",
        default="true",
        choices=["true", "false"],
    )
    parser.add_argument("--model-run-id", help="Explicit model_run_id (auto-generated if omitted).")
    parser.add_argument(
        "--lookback-days",
        help="Rolling window length (default 504 = ~2y).",
        type=int,
        default=504,
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to write the scoring envelope JSON.",
        dest="output_json",
    )
    # PR-1 alias kept for back-compat with the original stub flag name.
    parser.add_argument(
        "--out-json",
        help=argparse.SUPPRESS,
        dest="out_json_legacy",
    )


def _build_fi_score_liquidity(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "fi-score-liquidity",
        help="Score liquidity-stress indices per scope (PR-4).",
    )
    parser.add_argument("--db", help="DuckDB warehouse path.", default="data/mre.duckdb")
    parser.add_argument("--asof", help="ISO-8601 as-of timestamp.")
    parser.add_argument("--scope-type", help="Scope: market / sector / rating / cusip.")
    parser.add_argument("--scope-id", help="Specific scope_id (required for non-market scopes).")
    parser.add_argument("--out-json", help="Optional path to write the scoring envelope.")


def _build_fi_score_execution_confidence(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "fi-score-execution-confidence",
        help="Score execution-confidence for a candidate order (PR-5).",
    )
    parser.add_argument("--db", help="DuckDB warehouse path.", default="data/mre.duckdb")
    parser.add_argument(
        "--input",
        help="Path to the ExecutionConfidenceRequest JSON payload.",
        required=False,
    )
    parser.add_argument("--out-json", help="Optional path to write the scoring envelope.")


def _build_fi_tca_segment(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "fi-tca-segment",
        help="Tag and aggregate TCA segments by regime/liquidity/confidence (PR-6).",
    )
    parser.add_argument("--db", help="DuckDB warehouse path.", default="data/mre.duckdb")
    parser.add_argument("--start", help="ISO-8601 segment start timestamp.")
    parser.add_argument("--end", help="ISO-8601 segment end timestamp.")
    parser.add_argument("--out-json", help="Optional path to write the aggregated metrics.")


def _build_fi_evidence_pack(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "fi-evidence-pack",
        help="Generate or fetch a Fixed-Income evidence pack (PR-7).",
    )
    parser.add_argument("--db", help="DuckDB warehouse path.", default="data/mre.duckdb")
    parser.add_argument("--model-run-id", help="Model run id to render the pack for.", required=False)
    parser.add_argument("--out", help="Optional output path for the pack JSON.")


def _build_fi_report(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "fi-report",
        help="Generate the Fixed-Income RCIE Markdown/HTML report (PR-7).",
    )
    parser.add_argument("--db", help="DuckDB warehouse path.", default="data/mre.duckdb")
    parser.add_argument("--out", help="Output report path.", default="data/reports/fixed_income_rcie.md")
    parser.add_argument("--format", help="Report format: 'markdown' or 'html'.", default="markdown")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mre", description="Fixed-Income RCIE / X-Pro Auto-X CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    _build_fi_build_features(sub)
    _build_fi_score_credit_regime(sub)
    _build_fi_score_liquidity(sub)
    _build_fi_score_execution_confidence(sub)
    _build_fi_tca_segment(sub)
    _build_fi_evidence_pack(sub)
    _build_fi_report(sub)
    return parser


def _cmd_fi_score_credit_regime(ns: argparse.Namespace) -> int:
    """Run :func:`score_credit_regime` end-to-end and persist the row.

    Workflow:

    1. Open the DuckDB warehouse at ``--db``.
    2. Build PIT-safe credit features for the asof timestamp.
    3. Score via :func:`score_credit_regime` with the explicit profile,
       release_gate, and (optional) model_run_id.
    4. Persist via :func:`write_credit_regime_score` so the API and
       evidence-pack consumers see the new row.
    5. Print the canonical scoring envelope to stdout.
    6. (Optional) write the same envelope to ``--output-json``.

    Returns 0 on success, 2 on PIT or input validation failure. The
    exit code maps to the operator runbook: 0 = clean run (including
    release_gate=false), 2 = upstream contract broken (PIT, missing
    feature with audit policy, etc).
    """
    # Lazy imports keep ``fi-* stub commands`` cheap (they don't pay
    # the pandas / DuckDB import cost just to print ``not_yet_implemented``).
    from market_regime_engine.fixed_income.credit_spread_regime import (
        score_credit_regime,
        write_credit_regime_score,
    )
    from market_regime_engine.fixed_income.feature_builders import (
        build_credit_features,
    )
    from market_regime_engine.fixed_income.pit_guard import PitViolationError
    from market_regime_engine.fixed_income.timestamps import to_utc
    from market_regime_engine.frontier.data_cleaning import PitAuditFailure
    from market_regime_engine.storage import Warehouse

    asof_arg = getattr(ns, "asof", None)
    if asof_arg:
        try:
            asof_ts = to_utc(asof_arg)
        except ValueError as exc:
            print(json.dumps({"status": "error", "detail": str(exc)}, sort_keys=True))
            return 2
    else:
        asof_ts = pd.Timestamp.now(tz="UTC")
    if asof_ts is None:
        print(json.dumps({"status": "error", "detail": "asof must not be None"}, sort_keys=True))
        return 2

    release_gate = (getattr(ns, "release_gate", "true") or "true").lower() != "false"
    profile = getattr(ns, "profile", "production")

    wh = Warehouse(ns.db)
    try:
        try:
            features = build_credit_features(wh, asof_ts, lookback_days=int(getattr(ns, "lookback_days", 504)))
        except PitViolationError as exc:
            print(json.dumps({"status": "pit_violation", "detail": str(exc)}, sort_keys=True))
            return 2
        try:
            output = score_credit_regime(
                features,
                asof=asof_ts,
                model_run_id=getattr(ns, "model_run_id", None),
                release_gate=release_gate,
                profile=profile,
            )
        except PitViolationError as exc:
            print(json.dumps({"status": "pit_violation", "detail": str(exc)}, sort_keys=True))
            return 2
        except PitAuditFailure as exc:
            print(json.dumps({"status": "pit_audit_failed", "detail": str(exc)}, sort_keys=True))
            return 2
        write_credit_regime_score(wh, output)
    finally:
        wh.close()

    envelope = _envelope_from_output(output)
    print(json.dumps(envelope, sort_keys=True))

    output_path = getattr(ns, "output_json", None) or getattr(ns, "out_json_legacy", None)
    if output_path:
        _write_optional_json(output_path, envelope)
    return 0


def _envelope_from_output(output: Any) -> dict[str, Any]:
    """Stdout summary envelope per the scope spec.

    Includes the AGENT.md §6.1 fields plus the v1.5 governance triple.
    Drivers are emitted as a list (JSON-friendly).
    """
    return {
        "timestamp": output.timestamp,
        "regime_score": float(output.regime_score),
        "regime_label": output.regime_label,
        "confidence": float(output.confidence),
        "drivers": list(output.drivers),
        "component_scores": dict(output.component_scores),
        "model_run_id": output.model_run_id,
        "release_gate": bool(output.release_gate),
        "artifact_hash": output.artifact_hash,
    }


def _write_optional_json(path: str, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")


def run(args: Sequence[str]) -> int:
    """Dispatch an ``fi-*`` subcommand.

    Returns the subprocess exit code. Stub commands (PR-4..PR-7)
    return 0 + a ``not_yet_implemented`` JSON payload so downstream
    automation can detect the placeholder state via ``status`` rather
    than a non-zero exit. The PR-3 ``fi-score-credit-regime`` command
    runs the real workflow and returns 0 on success, 2 on PIT or
    audit failure.
    """
    parser = _build_parser()
    ns = parser.parse_args(list(args))
    command = ns.command
    if command == "fi-score-credit-regime":
        return _cmd_fi_score_credit_regime(ns)
    if command in CLI_COMMANDS:
        return _emit_stub(command)
    parser.error(f"unsupported fi-* command: {command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run(sys.argv[1:]))


__all__ = ["CLI_COMMANDS", "run"]
