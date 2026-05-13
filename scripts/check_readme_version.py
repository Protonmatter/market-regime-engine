#!/usr/bin/env python3
"""v1.5.1 (PR-9 FIX 6) + v1.6.1: version-drift guard across pyproject.toml,
``__version__``, and the README banner.

The release identity is pinned in three places that must agree exactly:

* ``[project] version`` in ``pyproject.toml`` (drives wheel METADATA)
* ``__version__`` in ``src/market_regime_engine/__init__.py`` (drives
  runtime ``import market_regime_engine; print(__version__)``)
* the README's first H1 banner ``# Market Regime Engine vX.Y.Z`` (drives
  the wheel ``Description`` first line, which setuptools sources from
  the README)

This script fails with exit code 1 if any of the three drift, so a
release that bumps one but forgets the others is caught before the
wheel ships.

History:

* v1.5.1 (PR-9 FIX 6) introduced the ``__version__`` vs README banner
  comparison.
* v1.6.1 hardens the guard to also check ``pyproject.toml``. The v1.6.0
  release would have caught the source-vs-tag identity mismatch in CI
  if (a) GitHub Actions weren't billing-blocked and (b) this guard had
  also been checking the pyproject version. Both gaps are closed here.

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

try:  # Python 3.11+ ships tomllib in the stdlib; the project pins >=3.11.
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - defensive fallback
    tomllib = None  # type: ignore[assignment]

_REPO_ROOT = Path(__file__).resolve().parent.parent
_INIT = _REPO_ROOT / "src" / "market_regime_engine" / "__init__.py"
_README = _REPO_ROOT / "README.md"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

_VERSION_RE = re.compile(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", re.MULTILINE)
_README_BANNER_RE = re.compile(r"^#\s*Market\s+Regime\s+Engine\s+v([0-9]+(?:\.[0-9]+)*)")
# Narrow regex deliberately: only matches ``version = "X.Y.Z"`` lines that
# are on their own line. Used as a fallback when ``tomllib`` is unavailable.
_PYPROJECT_VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"\s*$', re.MULTILINE)


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


def _extract_pyproject_version(path: Path) -> str:
    """Return the ``[project] version`` string from ``pyproject.toml``.

    Prefer ``tomllib`` (Python 3.11+, guaranteed by the project's
    ``requires-python = ">=3.11"`` floor) so we read the canonical
    parsed table value. Fall back to a deliberately narrow regex if
    ``tomllib`` is somehow unavailable.
    """
    if not path.exists():
        raise SystemExit(
            f"check_readme_version: pyproject.toml not found at {path}"
        )
    if tomllib is not None:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
        try:
            version = data["project"]["version"]
        except KeyError as exc:
            raise SystemExit(
                f"check_readme_version: [project] version missing in {path}"
            ) from exc
        if not isinstance(version, str):
            raise SystemExit(
                f"check_readme_version: [project] version in {path} is not a string: {version!r}"
            )
        return version.strip()
    # Fallback regex path: only walk lines inside [project].
    text = path.read_text(encoding="utf-8")
    in_project = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_project = stripped == "[project]"
            continue
        if not in_project:
            continue
        match = _PYPROJECT_VERSION_RE.match(stripped)
        if match:
            return match.group(1).strip()
    raise SystemExit(
        f"check_readme_version: [project] version not found in {path}"
    )


def main() -> int:
    init_version = _extract_init_version(_INIT)
    readme_version = _extract_readme_version(_README)
    pyproject_version = _extract_pyproject_version(_PYPROJECT)

    # v1.6.1 contract: strict three-way equality. The earlier v1.5.1
    # patch-family fallback (README banner allowed to lag __version__
    # within the same X.Y series) was deliberately tightened in v1.6.1
    # because the v1.6.0 source-vs-tag identity mismatch demonstrated
    # that any laxity here turns into a published wheel reporting a
    # stale version. All three identities must agree exactly.
    if init_version == readme_version == pyproject_version:
        print(f"check_readme_version: OK ({init_version})")
        return 0

    print(
        "check_readme_version: DRIFT - the three release-identity "
        "sources disagree.\n"
        f"  pyproject.toml [project] version : {pyproject_version!r}\n"
        f"  src/market_regime_engine/__init__.py __version__ : {init_version!r}\n"
        f"  README.md H1 banner version      : {readme_version!r}\n"
        "Bump all three before tagging the release; a published wheel "
        "must report a single, consistent version string.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":  # pragma: no cover - script entry point
    sys.exit(main())
