# SPDX-License-Identifier: Apache-2.0
"""Build the v1.2.1 dual-artifact release.

Three artifacts land in ``dist/``:

1. ``dist/market_regime_engine-<version>-py3-none-any.whl`` — built by
   ``python -m build``. The pip-installable wheel for downstream
   consumers.
2. ``dist/market-regime-engine-<version>.tar.gz`` — sdist for PyPI
   submission and source-tree mirrors.
3. ``dist/market-regime-engine-<version>-source.zip`` — audit-grade
   archive with ``.git/`` preserved so ``mre verify-run`` works after
   extraction. Produced by ``scripts/build_audit_zip.py``.

After each artifact is built the script prints its SHA-256 so the
release-notes can pin the canonical hashes. Run from the project root::

    .venv\\Scripts\\python.exe scripts/build_release.py

Use ``--skip-wheel`` or ``--skip-audit-zip`` to build only one half of
the dual artifact (handy when iterating on a single output).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"


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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _print_artifact(path: Path) -> None:
    if not path.exists():
        print(f"  - missing: {path.name}")
        return
    sha = _sha256(path)
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"  - {path.name}  ({size_mb:.2f} MB)  sha256={sha}")


def build_wheel(python: str = sys.executable) -> None:
    """Build wheel + sdist via the standard ``python -m build`` invocation."""
    print("[build_release] building wheel + sdist via `python -m build` ...")
    cmd = [python, "-m", "build", str(ROOT)]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"`python -m build` exited {result.returncode}")


def build_audit_zip(python: str = sys.executable) -> None:
    print("[build_release] building audit-grade source zip ...")
    cmd = [python, str(ROOT / "scripts" / "build_audit_zip.py")]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"build_audit_zip.py exited {result.returncode}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-wheel", action="store_true", help="Skip the wheel + sdist build.")
    parser.add_argument("--skip-audit-zip", action="store_true", help="Skip the audit-grade zip build.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove dist/ before building so stale artifacts cannot leak into the release.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to invoke for child processes (defaults to the current one).",
    )
    args = parser.parse_args(argv)

    version = _read_pyproject_version()
    print(f"[build_release] version = {version}")
    if args.clean and DIST.exists():
        print(f"[build_release] cleaning {DIST}")
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True, exist_ok=True)

    if not args.skip_wheel:
        build_wheel(python=args.python)
    if not args.skip_audit_zip:
        build_audit_zip(python=args.python)

    pkg_us = "market_regime_engine"
    pkg_dash = "market-regime-engine"
    # Setuptools normalizes the sdist filename to use underscores, while
    # the audit zip is hand-named with dashes. Both are listed
    # explicitly so the printed report is unambiguous.
    artifacts = [
        DIST / f"{pkg_us}-{version}-py3-none-any.whl",
        DIST / f"{pkg_us}-{version}.tar.gz",
        DIST / f"{pkg_dash}-{version}-source.zip",
    ]
    print()
    print("Release artifacts:")
    for artifact in artifacts:
        _print_artifact(artifact)
    return 0


if __name__ == "__main__":
    sys.exit(main())
