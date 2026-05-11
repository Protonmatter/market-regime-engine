# SPDX-License-Identifier: Apache-2.0
"""CLI dispatch wrapper for focused validation commands.

This module intercepts point-in-time validation commands and delegates all
existing commands to the legacy CLI. Yes, it is a tiny router. No, it should not
become a second command framework wearing a trench coat.

v1.5 (PR-1, section B.1): the fast-path command set is extended with seven
``fi-*`` Fixed-Income commands (``fi-build-features``, ``fi-score-credit-regime``,
``fi-score-liquidity``, ``fi-score-execution-confidence``, ``fi-tca-segment``,
``fi-evidence-pack``, ``fi-report``). Each routes to
:mod:`market_regime_engine.fixed_income.cli` via lazy import so the legacy
CLI users do not pay the FI-import cost. PR-1 ships stubs that emit
``not_yet_implemented`` JSON; PR-3..PR-7 swap in the real handlers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from market_regime_engine.leakage_checks import audit_pit_paths
from market_regime_engine.snapshot_manifest import (
    build_snapshot_manifest,
    verify_snapshot_manifest,
    write_snapshot_manifest,
)

_FI_COMMANDS: frozenset[str] = frozenset(
    {
        "fi-build-features",
        "fi-score-credit-regime",
        "fi-score-liquidity",
        "fi-score-execution-confidence",
        "fi-tca-segment",
        "fi-evidence-pack",
        "fi-report",
    }
)

CUSTOM_COMMANDS = frozenset(
    {"pit-audit", "snapshot-build", "snapshot-verify"} | _FI_COMMANDS
)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch new PIT commands or delegate to the existing CLI.

    Routing priority:

    1. ``fi-*`` Fixed-Income commands → :func:`_run_fi` (lazy import of
       :mod:`market_regime_engine.fixed_income.cli`).
    2. ``pit-audit`` / ``snapshot-*`` commands → :func:`_run_custom`.
    3. Anything else → legacy :func:`market_regime_engine.cli.main`.
    """

    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in _FI_COMMANDS:
        return _run_fi(args)
    if args and args[0] in CUSTOM_COMMANDS:
        return _run_custom(args)
    return _delegate_to_legacy_cli(args, argv_was_none=argv is None)


def _run_fi(args: Sequence[str]) -> int:
    """Lazy-load the FI CLI dispatcher.

    Imported inside the function so a vanilla ``mre pit-audit ...`` call
    does not pay the cost of importing the FI package (which pulls
    pandas, FastAPI, and the schemas module at parse-time).
    """
    from market_regime_engine.fixed_income.cli import run as fi_run

    return int(fi_run(list(args)))


def _run_custom(args: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="mre")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pit = subparsers.add_parser("pit-audit", help="Audit feature/label tables for point-in-time leakage.")
    pit.add_argument("--features", required=True, help="Feature table path: CSV, JSON, JSONL, or Parquet.")
    pit.add_argument("--labels", required=True, help="Label table path: CSV, JSON, JSONL, or Parquet.")
    pit.add_argument("--out-json", help="Optional JSON report output path.")
    pit.add_argument("--out-md", help="Optional Markdown report output path.")
    pit.add_argument("--fail-on-leakage", action="store_true", help="Exit non-zero when blocker issues are found.")

    build = subparsers.add_parser("snapshot-build", help="Build a deterministic SHA-256 snapshot manifest.")
    build.add_argument("--input", required=True, help="Input file or directory to hash.")
    build.add_argument("--out", required=True, help="Output manifest JSON path.")
    build.add_argument("--snapshot-id", help="Optional explicit snapshot identifier.")

    verify = subparsers.add_parser("snapshot-verify", help="Verify a snapshot manifest against current files.")
    verify.add_argument("--manifest", required=True, help="Snapshot manifest JSON path.")
    verify.add_argument("--out-json", help="Optional JSON verification report path.")
    verify.add_argument("--out-md", help="Optional Markdown verification report path.")
    verify.add_argument("--fail-on-mismatch", action="store_true", help="Exit non-zero when mismatches are found.")

    ns = parser.parse_args(list(args))
    if ns.command == "pit-audit":
        return _cmd_pit_audit(ns)
    if ns.command == "snapshot-build":
        return _cmd_snapshot_build(ns)
    if ns.command == "snapshot-verify":
        return _cmd_snapshot_verify(ns)
    parser.error(f"unsupported command: {ns.command}")
    return 2


def _cmd_pit_audit(ns: argparse.Namespace) -> int:
    report = audit_pit_paths(ns.features, ns.labels)
    json_text = report.to_json()
    md_text = report.to_markdown()
    _write_optional(ns.out_json, json_text)
    _write_optional(ns.out_md, md_text)
    if not ns.out_json and not ns.out_md:
        print(md_text, end="")
    elif ns.out_json:
        print(f"Wrote PIT leakage JSON report: {ns.out_json}")
    elif ns.out_md:
        print(f"Wrote PIT leakage Markdown report: {ns.out_md}")
    return 2 if ns.fail_on_leakage and not report.passed else 0


def _cmd_snapshot_build(ns: argparse.Namespace) -> int:
    manifest = build_snapshot_manifest(ns.input, snapshot_id=ns.snapshot_id)
    out = write_snapshot_manifest(manifest, ns.out)
    print(f"Wrote snapshot manifest: {out}")
    print(f"snapshot_id={manifest.snapshot_id}")
    print(f"manifest_sha256={manifest.manifest_sha256}")
    return 0


def _cmd_snapshot_verify(ns: argparse.Namespace) -> int:
    report = verify_snapshot_manifest(ns.manifest)
    json_text = report.to_json()
    md_text = report.to_markdown()
    _write_optional(ns.out_json, json_text)
    _write_optional(ns.out_md, md_text)
    if not ns.out_json and not ns.out_md:
        print(md_text, end="")
    elif ns.out_json:
        print(f"Wrote snapshot verification JSON report: {ns.out_json}")
    elif ns.out_md:
        print(f"Wrote snapshot verification Markdown report: {ns.out_md}")
    return 2 if ns.fail_on_mismatch and not report.passed else 0


def _delegate_to_legacy_cli(args: Sequence[str], *, argv_was_none: bool) -> int:
    from market_regime_engine.cli import main as legacy_main

    if argv_was_none:
        return int(legacy_main() or 0)

    old_argv = sys.argv
    sys.argv = [old_argv[0], *args]
    try:
        return int(legacy_main() or 0)
    finally:
        sys.argv = old_argv


def _write_optional(path: str | None, text: str) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
