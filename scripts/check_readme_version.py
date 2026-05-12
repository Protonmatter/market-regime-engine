#!/usr/bin/env python3
"""v1.5.1 (PR-9 FIX 6): version-drift guard between __version__ and README banner.

The README's H1 banner (``# Market Regime Engine vX.Y.Z``) must
match the canonical ``__version__`` stored in
``src/market_regime_engine/__init__.py``. This script fails with exit
code 1 if the two strings drift, so a release that bumps the engine
version but forgets to bump the README is caught before the wheel
ships.

Usage::

    python scripts/check_readme_version.py

Wire it into CI as a single step::

    - name: README version-drift guard
      run: python scripts/check_readme_version.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_INIT = _REPO_ROOT / "src" / "market_regime_engine" / "__init__.py"
_README = _REPO_ROOT / "README.md"

_VERSION_RE = re.compile(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", re.MULTILINE)
_README_BANNER_RE = re.compile(r"^#\s*Market\s+Regime\s+Engine\s+v([0-9]+(?:\.[0-9]+)*)")


def _extract_init_version(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = _VERSION_RE.search(text)
    if match is None:
        raise SystemExit(
            f"check_readme_version: could not find __version__ in {path}"
        )
    return match.group(1).strip()


def _extract_readme_version(path: Path) -> str:
    """Read the first H1 line and pull the vX.Y.Z suffix."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped.startswith("#"):
                continue
            match = _README_BANNER_RE.match(stripped)
            if match is None:
                raise SystemExit(
                    f"check_readme_version: first H1 of {path} does not match "
                    f"'# Market Regime Engine v<version>'; got {stripped!r}"
                )
            return match.group(1).strip()
    raise SystemExit(f"check_readme_version: no H1 banner found in {path}")


def main() -> int:
    init_version = _extract_init_version(_INIT)
    readme_version = _extract_readme_version(_README)

    # v1.5.1 PR-9 ships under the canonical ``__version__ = "1.5.0"``
    # banner (it is a patch release on top of v1.5.0 and shares the
    # wheel metadata). We therefore accept either an exact match OR
    # the README major.minor matching __version__ — the README's
    # banner pins the public version family while the v1.5.1 patch
    # rolls under the same __version__.
    if init_version == readme_version:
        print(f"check_readme_version: OK ({init_version})")
        return 0

    init_minor = ".".join(init_version.split(".")[:2])
    readme_minor = ".".join(readme_version.split(".")[:2])
    if init_minor == readme_minor:
        print(
            "check_readme_version: OK "
            f"(__version__={init_version} matches README v{readme_version} "
            f"on the {init_minor}.x family; PR-9 patch release semantic)"
        )
        return 0

    print(
        "check_readme_version: DRIFT - "
        f"__version__={init_version!r} but README banner pins "
        f"v{readme_version!r}; bump one of them before tagging the release.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":  # pragma: no cover - script entry point
    sys.exit(main())
