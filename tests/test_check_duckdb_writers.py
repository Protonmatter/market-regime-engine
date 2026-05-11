# SPDX-License-Identifier: Apache-2.0
"""PR-7 §M — DuckDB writer contention detector acceptance tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


_TOOL = Path(__file__).resolve().parents[1] / "tools" / "check_duckdb_writers.py"


def _run(args: list[str]) -> tuple[int, dict]:
    proc = subprocess.run(
        [sys.executable, str(_TOOL), *args],
        capture_output=True,
        text=True,
    )
    payload = {}
    for line in proc.stdout.splitlines():
        try:
            payload = json.loads(line)
        except Exception:
            continue
    return proc.returncode, payload


def test_check_writers_returns_zero_when_no_concurrent_writer(tmp_path: Path) -> None:
    db_path = tmp_path / "no-writer.duckdb"
    db_path.write_bytes(b"")
    rc, payload = _run(["--db", str(db_path)])
    assert rc == 0
    assert payload["status"] == "ok"
    assert payload["wal_present"] is False
    assert payload["tmp_present"] is False


def test_check_writers_returns_nonzero_when_tmp_present(tmp_path: Path) -> None:
    """A ``.tmp`` companion file means another writer is mid-transaction."""
    db_path = tmp_path / "with-tmp.duckdb"
    db_path.write_bytes(b"")
    tmp_companion = db_path.with_suffix(db_path.suffix + ".tmp")
    tmp_companion.write_bytes(b"")
    rc, payload = _run(["--db", str(db_path)])
    assert rc == 2
    assert payload["status"] == "concurrent_writer_detected"
    assert payload["tmp_present"] is True


def test_check_writers_warns_on_wal_only(tmp_path: Path) -> None:
    """WAL present without an active holder → warn (exit 0) by default."""
    db_path = tmp_path / "with-wal.duckdb"
    db_path.write_bytes(b"")
    wal_companion = db_path.with_suffix(db_path.suffix + ".wal")
    wal_companion.write_bytes(b"")
    rc, payload = _run(["--db", str(db_path)])
    assert rc == 0
    assert payload["status"] == "warn"
    assert payload["wal_present"] is True


def test_check_writers_strict_promotes_wal_to_failure(tmp_path: Path) -> None:
    db_path = tmp_path / "wal-strict.duckdb"
    db_path.write_bytes(b"")
    db_path.with_suffix(db_path.suffix + ".wal").write_bytes(b"")
    rc, payload = _run(["--db", str(db_path), "--strict"])
    assert rc == 2
    assert payload["status"] == "concurrent_writer_detected"
