# SPDX-License-Identifier: Apache-2.0
"""README ↔ pyproject version sanity (v1.4.1 item B).

The v1.4.0 reviewer caught that ``README.md`` still carried the
``# Market Regime Engine v1.2.1`` H1 even though the wheel METADATA
had ``Version: 1.4.0``. Because ``pyproject.toml`` declares
``readme = "README.md"``, the long-description Description first line
is whatever the README starts with — so the wheel shipped with two
identities.

This test pins the contract: the README's first-H1 version must agree
with ``[project] version`` in ``pyproject.toml`` (and therefore with
``__version__`` in ``src/market_regime_engine/__init__.py``, which the
existing :mod:`tests.test_version_sanity` already pins to pyproject).

The CI ``version-sanity`` job runs this test alongside the existing
pyproject ↔ ``__init__`` ↔ ``mre --version`` checks so a future bump
that forgets the README H1 fails fast.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_pyproject_version() -> str:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
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
    raise AssertionError("[project] version not found in pyproject.toml")


def _read_readme_h1_version() -> str:
    """Parse the README's first H1 heading and extract the version string.

    The contract is that the H1 reads exactly ``# Market Regime Engine
    vX.Y.Z`` (the ``v`` prefix is mandatory, the Z patch component is
    mandatory) so the regex is intentionally narrow.
    """
    readme = REPO_ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    pattern = re.compile(r"^# Market Regime Engine v(?P<version>\d+\.\d+\.\d+)\b", re.MULTILINE)
    m = pattern.search(text)
    if not m:
        raise AssertionError(
            "README.md must start with '# Market Regime Engine vX.Y.Z' as the first H1; no matching heading was found."
        )
    return m.group("version")


def test_readme_h1_matches_pyproject_version() -> None:
    """The README H1 version must equal ``[project] version`` in pyproject.

    v1.4.0 shipped with this contract violated (README H1 stayed at
    ``v1.2.1`` while pyproject and the wheel METADATA were on
    ``1.4.0``). v1.4.1 closes the gap.
    """
    readme_version = _read_readme_h1_version()
    pyproject_version = _read_pyproject_version()
    assert readme_version == pyproject_version, (
        f"README.md H1 declares version {readme_version!r} but "
        f"pyproject.toml [project] version is {pyproject_version!r}. "
        "Bump both — every release ships with the README and the "
        "wheel METADATA agreeing on the same version, otherwise the "
        "wheel's Description first line and the on-disk README "
        "diverge."
    )


def test_readme_h1_is_first_heading() -> None:
    """The H1 must be the very first non-blank line of the README.

    ``setuptools`` uses the README contents as the long-description /
    PKG-INFO Description; if the H1 is preceded by a different heading
    or by stray prose, the wheel METADATA Description first line
    diverges from what the test above pins.
    """
    readme = REPO_ROOT / "README.md"
    for line in readme.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        assert stripped.startswith("# "), f"README.md must start with an H1 line; first non-blank line was {line!r}."
        assert re.match(r"^# Market Regime Engine v\d+\.\d+\.\d+\b", stripped), (
            f"README.md first H1 must match '# Market Regime Engine vX.Y.Z'; got {stripped!r}."
        )
        return
    raise AssertionError("README.md is empty")
