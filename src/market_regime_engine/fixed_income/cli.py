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
import sys
from collections.abc import Sequence

CLI_COMMANDS: tuple[str, ...] = (
    "fi-build-features",
    "fi-score-credit-regime",
    "fi-score-liquidity",
    "fi-score-execution-confidence",
    "fi-tca-segment",
    "fi-evidence-pack",
    "fi-report",
)


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
    parser.add_argument("--asof", help="ISO-8601 as-of timestamp; defaults to latest available.")
    parser.add_argument("--out-json", help="Optional path to write the scoring envelope.")


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


def run(args: Sequence[str]) -> int:
    """Dispatch an ``fi-*`` subcommand.

    Returns the subprocess exit code. PR-1 stubs always return 0 plus
    a canonical ``not_yet_implemented`` JSON payload so downstream
    automation can detect the placeholder state via ``status`` rather
    than a non-zero exit (which would break the existing CI
    integration smoke tests).
    """
    parser = _build_parser()
    ns = parser.parse_args(list(args))
    command = ns.command
    if command in CLI_COMMANDS:
        return _emit_stub(command)
    parser.error(f"unsupported fi-* command: {command}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run(sys.argv[1:]))


__all__ = ["CLI_COMMANDS", "run"]
