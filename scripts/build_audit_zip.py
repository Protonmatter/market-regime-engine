# SPDX-License-Identifier: Apache-2.0
"""Build the audit-grade source archive for the Market Regime Engine.

This is the v1.2.1 successor to the top-level ``build_zip.py`` shim. It
emits a single zip suitable for forensic reproducibility:

- Includes ``.git/`` so ``mre verify-run`` can resolve the SHA after
  extraction (regulators do not care about your editable install; they
  care that the run-pinning hash is auditable).
- Excludes runtime caches, virtualenvs, the SQLite warehouse, and any
  test output dumps — these would either bloat the zip or leak state.
- Honors the v1.2.1 release naming: ``dist/market-regime-engine-1.2.1-source.zip``.

Run from the project root::

    .venv\\Scripts\\python.exe scripts/build_audit_zip.py

The resulting archive is the "audit-grade source archive" half of the
v1.2.1 dual-artifact release strategy. The other half is the wheel
produced by ``python -m build`` (emitted into the same ``dist/``
directory). See ``docs/V1_2_1_FIXES.md`` for the rationale.

v1.3 changes (item A): the v1.2.1 audit zip ballooned to 41 MB because
it bundled the smoke-run warehouse (``data/``), build caches, and the
unpacked ``.git/pack/`` objects. The exclude list below now mirrors the
build_zip.py contract, ``--with-runtime-data`` exists for forensic
deep-dives that genuinely need the warehouse, and the script runs
``git gc --aggressive --prune=now`` before zipping so ``.git/pack/``
is compacted.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"

# v1.3: directories that are categorically excluded from the audit zip.
# ``data`` is excluded by default (re-included by ``--with-runtime-data``).
EXCLUDE_DIRS = {
    ".venv",
    ".smoke-venv",
    ".hypothesis",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".ci-artifacts",
    ".ci-artifacts-raw",
    "__pycache__",
    "data",
    "target",
    "build",
    "dist",
    "node_modules",
}

EXCLUDE_PATH_PREFIXES = ("rust_ext/target/",)

EXCLUDE_FILE_GLOBS = (
    "*.pyc",
    "*.pyo",
    "*.so",
    "*.pyd",
    "*.db",
    "*.duckdb",
    "*.parquet",
    "*.sqlite",
    "*.sqlite3",
    "*.zip",
    "*.whl",
    "*.tar.gz",
)

EXCLUDE_ROOT_FILES = {
    ".coverage",
    "cov_out.txt",
    "mypy_out.txt",
    "pytest_out.txt",
    "r.txt",
    "ruff_check_out.txt",
    "ruff_fmt_out.txt",
    "ruff_stats.txt",
}


def _read_pyproject_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    in_project = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_project = stripped == "[project]"
            continue
        if not in_project:
            continue
        m = re.match(r'^version\s*=\s*"([^"]+)"\s*$', stripped)
        if m:
            return m.group(1)
    raise SystemExit("[project] version not found in pyproject.toml")


def is_excluded(rel: str, *, exclude_dirs: frozenset[str]) -> bool:
    rel_fwd = rel.replace("\\", "/")
    parts = rel_fwd.split("/")
    for part in parts[:-1]:
        if part in exclude_dirs:
            return True
        if part.endswith(".egg-info"):
            return True
    for prefix in EXCLUDE_PATH_PREFIXES:
        if rel_fwd.startswith(prefix):
            return True
    if "/" not in rel_fwd and rel_fwd in EXCLUDE_ROOT_FILES:
        return True
    name = parts[-1]
    for glob in EXCLUDE_FILE_GLOBS:
        if fnmatch.fnmatch(name, glob):
            return True
    return False


def _git_gc(quiet: bool = False) -> None:
    """Compact ``.git/pack/`` so the audit zip ships a small object store.

    Failing here is non-fatal: the audit zip should still build even on
    machines without git installed (e.g. a CI runner where the checkout
    happened via an action that copied the working tree).
    """
    if not (ROOT / ".git").exists():
        return
    try:
        subprocess.run(
            ["git", "gc", "--aggressive", "--prune=now"],
            cwd=str(ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
        if not quiet:
            print("git gc --aggressive --prune=now: OK")
    except FileNotFoundError:
        if not quiet:
            print("git not on PATH; skipping git gc compaction")
    except subprocess.CalledProcessError as exc:
        if not quiet:
            print(f"git gc failed (non-fatal): {exc.stderr.strip()}")


def build_audit_zip(
    version: str | None = None,
    *,
    with_runtime_data: bool = False,
    skip_git_gc: bool = False,
) -> Path:
    """Build the audit zip and return its path.

    ``with_runtime_data`` re-includes the ``data/`` directory and any
    runtime-state files (``*.db``, ``*.parquet``, ...). This is meant
    for forensic deep-dives where the warehouse contents are part of
    the audit material; the default is to ship a clean source archive.
    """
    DIST.mkdir(parents=True, exist_ok=True)
    version = version or _read_pyproject_version()
    out = DIST / f"market-regime-engine-{version}-source.zip"
    if out.exists():
        out.unlink()

    if not skip_git_gc:
        _git_gc()

    exclude_dirs = set(EXCLUDE_DIRS)
    exclude_globs: tuple[str, ...] = EXCLUDE_FILE_GLOBS
    if with_runtime_data:
        exclude_dirs.discard("data")
        exclude_globs = tuple(g for g in EXCLUDE_FILE_GLOBS if g not in ("*.db", "*.duckdb", "*.parquet", "*.sqlite", "*.sqlite3"))

    written = 0
    bytes_total = 0
    arc_root = f"market-regime-engine-{version}/"
    excluded_dirs_frozen = frozenset(exclude_dirs)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for dirpath, dirnames, filenames in os.walk(ROOT, topdown=True):
            dirnames[:] = [
                d for d in dirnames if d not in exclude_dirs and not d.endswith(".egg-info")
            ]
            for fn in filenames:
                full = Path(dirpath) / fn
                rel = str(full.relative_to(ROOT))
                if is_excluded(rel, exclude_dirs=excluded_dirs_frozen):
                    continue
                # Honor the runtime-data flag for file-level globs.
                rel_fwd = rel.replace("\\", "/")
                if any(fnmatch.fnmatch(Path(rel_fwd).name, g) for g in exclude_globs):
                    continue
                arc = arc_root + rel.replace("\\", "/")
                zf.write(full, arcname=arc)
                written += 1
                bytes_total += full.stat().st_size

    size_mb = out.stat().st_size / 1024 / 1024
    raw_mb = bytes_total / 1024 / 1024
    ratio = (1 - out.stat().st_size / max(bytes_total, 1)) * 100
    print(f"Wrote {written} files to {out}")
    print(f"  raw bytes:        {raw_mb:.2f} MB")
    print(f"  compressed bytes: {size_mb:.2f} MB")
    print(f"  compression:      {ratio:.1f}%")
    if not with_runtime_data and size_mb > 5.0:
        # Surface the regression loudly in the build log so the
        # package-sanity CI gate has something concrete to point at when
        # it trips.
        print(
            f"WARNING: audit zip is {size_mb:.2f} MB, exceeds the 5 MB v1.3 budget. "
            "Check the exclude list for newly-added build artefacts.",
            file=sys.stderr,
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        default=None,
        help="Override version (defaults to pyproject [project] version).",
    )
    parser.add_argument(
        "--with-runtime-data",
        action="store_true",
        help=(
            "Include the runtime warehouse (data/) and lake exports for "
            "forensic deep-dives. Adds ~30+ MB to the archive."
        ),
    )
    parser.add_argument(
        "--skip-git-gc",
        action="store_true",
        help="Skip the git gc --aggressive --prune=now compaction step.",
    )
    args = parser.parse_args(argv)
    out = build_audit_zip(
        version=args.version,
        with_runtime_data=args.with_runtime_data,
        skip_git_gc=args.skip_git_gc,
    )
    sha256 = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"sha256({out.name}) = {sha256}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
