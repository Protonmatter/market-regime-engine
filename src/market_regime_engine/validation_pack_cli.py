# SPDX-License-Identifier: Apache-2.0
"""CLI for tamper-evident empirical validation evidence packs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from market_regime_engine.validation_pack import build_evidence_pack, verify_evidence_pack


def _load_metadata(path: str | None) -> dict:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mre-validation-pack")
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="Build a tamper-evident validation evidence pack.")
    b.add_argument("--out", required=True, help="Output evidence-pack directory.")
    b.add_argument("--include", action="append", required=True, help="File or directory to copy. Repeat as needed.")
    b.add_argument("--metadata-json", help="Optional JSON file with run/model metadata to embed.")
    b.add_argument("--force", action="store_true", help="Replace an existing evidence-pack directory.")
    b.add_argument("--hmac-key", help="Optional HMAC key. If omitted, MRE_EVIDENCE_HMAC_KEY is used when set.")
    b.add_argument("--require-signed", action="store_true", default=None, help="Fail if no HMAC key/signature is available.")
    b.add_argument("--absolute-source-map", action="store_true", help="Store absolute source paths in source_map.")
    b.add_argument("--lockfile", action="append", help="Specific lockfile to hash. Repeat as needed.")

    v = sub.add_parser("verify", help="Verify a validation evidence pack.")
    v.add_argument("path", help="Evidence-pack directory.")
    v.add_argument("--hmac-key", help="Optional HMAC key. If omitted, MRE_EVIDENCE_HMAC_KEY is used when set.")
    v.add_argument("--require-signed", action="store_true", help="Fail verification if the pack is unsigned.")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        result = build_evidence_pack(
            includes=args.include,
            out_dir=args.out,
            metadata=_load_metadata(args.metadata_json),
            force=args.force,
            hmac_key=args.hmac_key,
            require_signed=args.require_signed,
            absolute_source_map=args.absolute_source_map,
            lockfiles=args.lockfile,
            command_line=["mre-validation-pack", *(argv or sys.argv[1:])],
        )
        print(json.dumps(result.__dict__, indent=2, sort_keys=True))
        return 0
    if args.command == "verify":
        report = verify_evidence_pack(args.path, hmac_key=args.hmac_key, require_signed=args.require_signed)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["approved"] else 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
