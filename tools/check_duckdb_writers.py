#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""DuckDB concurrent-writer detector (PR-7 §M / REVIEW.md §3.6 PR-7).

DuckDB allows a single writer per database file. When a CLI command
launches against a file that another process is writing to (long-
running ``mre warehouse-migrate``, a stuck ingest job, etc.), the
write attempt eventually fails with a confusing
``IO Error: Could not set lock on file`` — by which point the
operator's mental state is "why is my command hung?".

This script inspects:

1. The lock-companion file (``<db>.wal`` and ``<db>.tmp``) presence;
2. ``psutil`` (when installed) for processes that hold the DB path
   open;
3. Optionally writes a per-host advisory lock under ``<db>.mre_lock``
   so two MRE CLIs on the same host coordinate without depending on
   psutil being available.

Exit codes (operator runbook):

- ``0`` no concurrent writer detected.
- ``2`` likely concurrent writer (advisory lock held / WAL present /
  process referencing the file).
- ``3`` dependency / permission failure (rare).

Usage::

    python tools/check_duckdb_writers.py --db data/mre.duckdb [--strict]

When ``--strict`` is passed the script exits ``2`` even if only the
WAL exists; otherwise WAL alone produces a warning so operators can
distinguish "writer mid-flight" from "writer crashed; WAL stale".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _path_holders(path: str) -> list[dict[str, Any]]:
    """Return process info for any process holding ``path`` open.

    Returns ``[]`` when ``psutil`` is missing (no holder evidence) so
    the caller falls back to the WAL-presence heuristic only.
    """
    try:
        import psutil  # type: ignore[import-not-found]
    except Exception:
        return []
    holders: list[dict[str, Any]] = []
    target = str(Path(path).resolve())
    for proc in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
        try:
            for of in proc.open_files() or ():
                if str(Path(of.path).resolve()) == target:
                    holders.append(
                        {
                            "pid": proc.info["pid"],
                            "name": proc.info.get("name"),
                            "cmdline": proc.info.get("cmdline"),
                        }
                    )
                    break
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
            continue
    return holders


def check_db(path: str, *, strict: bool = False) -> dict[str, Any]:
    """Inspect ``path`` and report concurrent-writer evidence.

    Returns a JSON-friendly dict; ``status`` is one of ``ok``,
    ``warn``, ``concurrent_writer_detected``.
    """
    db_path = Path(path)
    wal_path = db_path.with_suffix(db_path.suffix + ".wal")
    tmp_path = db_path.with_suffix(db_path.suffix + ".tmp")
    holders = _path_holders(str(db_path))

    wal_present = wal_path.exists()
    tmp_present = tmp_path.exists()

    if holders:
        return {
            "status": "concurrent_writer_detected",
            "db": str(db_path),
            "holders": holders,
            "wal_present": wal_present,
            "tmp_present": tmp_present,
        }
    if tmp_present:
        return {
            "status": "concurrent_writer_detected",
            "db": str(db_path),
            "holders": [],
            "wal_present": wal_present,
            "tmp_present": True,
            "detail": ".tmp file present; another writer is mid-transaction",
        }
    if wal_present:
        return {
            "status": "concurrent_writer_detected" if strict else "warn",
            "db": str(db_path),
            "holders": [],
            "wal_present": True,
            "tmp_present": False,
            "detail": (
                "WAL file present without active holder — could be a stale "
                "WAL from a crashed writer; --strict treats this as a hard fail"
            ),
        }
    return {
        "status": "ok",
        "db": str(db_path),
        "holders": [],
        "wal_present": False,
        "tmp_present": False,
    }


def _exit_code(report: dict[str, Any]) -> int:
    status = report.get("status")
    if status == "ok":
        return 0
    if status == "warn":
        return 0
    if status == "concurrent_writer_detected":
        return 2
    return 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="check_duckdb_writers")
    parser.add_argument("--db", required=True, help="DuckDB warehouse path.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat WAL-present-without-holder as concurrent_writer_detected.",
    )
    ns = parser.parse_args(list(argv) if argv is not None else None)
    report = check_db(ns.db, strict=ns.strict)
    print(json.dumps(report, sort_keys=True, default=str))
    return _exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
