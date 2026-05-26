"""Backward-compatible CLI facade.

The CLI is decomposed into:

- :mod:`cli_parser` for argparse construction.
- :mod:`cli_handlers` for command implementation.
- :mod:`cli_helpers` for shared helper/state utilities.

This module preserves the historical public import surface.
"""

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os

from market_regime_engine import __version__ as ENGINE_VERSION
from market_regime_engine.cli_handlers import *  # noqa: F403
from market_regime_engine.cli_helpers import _verify_fi_evidence_pack
from market_regime_engine.cli_parser import parser
from market_regime_engine.logging_setup import configure_logging


def main(argv: list[str] | None = None) -> None:
    # v1.3 version sanity (item H). ``mre --version`` short-circuits the
    # subcommand requirement so a one-shot ``--version`` smoke check from
    # CI matches the version_sanity job exactly.
    if argv is None:
        import sys as _sys

        argv_list = list(_sys.argv[1:])
    else:
        argv_list = list(argv)
    if argv_list and argv_list[0] in {"--version", "-V"}:
        print(ENGINE_VERSION)
        return
    cli_parser = parser()
    cli_parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=f"market-regime-engine {ENGINE_VERSION}",
    )
    cli_parser.add_argument("--json-logs", action="store_true", help="Emit logs as one JSON object per line.")
    cli_parser.add_argument("--log-level", default=None, help="Override log level (DEBUG|INFO|WARNING|ERROR).")
    args = cli_parser.parse_args(argv)
    if getattr(args, "json_logs", False) or os.getenv("MRE_LOG_FORMAT") == "json":
        configure_logging(level=args.log_level or "INFO", fmt="json")
    elif args.log_level:
        configure_logging(level=args.log_level, fmt="human")
    else:
        configure_logging()
    args.func(args)


if __name__ == "__main__":
    main()


__all__ = [
    "main",
    "parser",
    "_verify_fi_evidence_pack",
    *[name for name in globals() if name.endswith("_cmd") or name == "bootstrap_sample"],
]
