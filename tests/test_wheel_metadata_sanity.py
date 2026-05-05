# SPDX-License-Identifier: Apache-2.0
"""Wheel METADATA sanity (v1.4.1 item C).

The v1.4.0 wheel shipped with ``Version: 1.4.0`` in METADATA but the
first non-blank line of the long-description Description body still
read ``Market Regime Engine v1.2.1`` (because ``pyproject.toml``
declares ``readme = "README.md"`` and the README H1 was stale). v1.4.1
closes the gap and pins it with this regression test.

The test builds the wheel via ``python -m build --wheel``, opens the
resulting ``*.whl`` as a zip archive, reads the
``*.dist-info/METADATA`` file, and asserts:

1. The ``Version:`` header equals the source ``[project] version``.
2. The first non-blank line of the long-description body (the
   ``Description`` section after the ``\\n\\n`` separator) matches the
   on-disk README's first H1.

Because building a wheel takes a non-trivial amount of time on a
single test run, the test is fenced behind ``MRE_WHEEL_METADATA_TEST``
so the default ``pytest -q -m "not slow"`` run does not pay the
``python -m build`` cost. The CI ``package-sanity`` job sets the env
var so the assertion runs there.

The CI workflow already reuses ``tests/test_package_metadata.py``
inside a fresh venv to verify the installed wheel's runtime
``__version__``; this module adds the on-disk wheel METADATA inspection
so the static text in the wheel itself agrees with the source.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

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


def _read_readme_h1() -> str:
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    pattern = re.compile(r"^# Market Regime Engine v(\d+\.\d+\.\d+)\b", re.MULTILINE)
    m = pattern.search(text)
    if not m:
        raise AssertionError("README.md H1 unparseable")
    return f"Market Regime Engine v{m.group(1)}"


def _wheel_for_current_version(version: str) -> Path | None:
    """Return the path to a pre-built wheel, if one exists in dist/."""
    candidate = REPO_ROOT / "dist" / f"market_regime_engine-{version}-py3-none-any.whl"
    return candidate if candidate.exists() else None


def _build_wheel(version: str) -> Path:
    """Invoke ``python -m build --wheel`` and return the resulting wheel path."""
    cmd = [sys.executable, "-m", "build", "--wheel", "--outdir", str(REPO_ROOT / "dist"), str(REPO_ROOT)]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"`python -m build --wheel` exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    wheel = _wheel_for_current_version(version)
    if wheel is None:
        raise RuntimeError(
            f"Expected dist/market_regime_engine-{version}-py3-none-any.whl after build; "
            f"found: {[p.name for p in (REPO_ROOT / 'dist').iterdir()]}"
        )
    return wheel


def _read_wheel_metadata(wheel_path: Path) -> str:
    with zipfile.ZipFile(wheel_path) as zf:
        metadata_names = [n for n in zf.namelist() if n.endswith(".dist-info/METADATA")]
        if not metadata_names:
            raise AssertionError(f"No *.dist-info/METADATA in {wheel_path.name}")
        if len(metadata_names) > 1:
            raise AssertionError(f"Expected exactly one METADATA in wheel, got {metadata_names!r}")
        return zf.read(metadata_names[0]).decode("utf-8")


def _split_metadata(text: str) -> tuple[dict[str, str], str]:
    """Split the METADATA file into (headers_dict, description_body).

    The PKG-INFO format separates RFC822-style headers from the
    long-description body with a single blank line. ``email`` parsers
    handle this but we want a stable split for the assertion below
    without pulling in the ``email`` machinery.

    Wheels built on Windows use ``\r\n`` line endings; wheels built on
    POSIX use ``\n``. We try both separators so the assertion is
    cross-platform.
    """
    head: str
    body: str
    if "\r\n\r\n" in text:
        head, body = text.split("\r\n\r\n", 1)
    elif "\n\n" in text:
        head, body = text.split("\n\n", 1)
    else:
        head, body = text, ""
    headers: dict[str, str] = {}
    for line in head.splitlines():
        if not line or line[0] in (" ", "\t"):
            continue
        if ": " not in line:
            continue
        key, _, value = line.partition(": ")
        headers[key] = value
    return headers, body


def _first_nonblank_line(body: str) -> str:
    for line in body.splitlines():
        if line.strip():
            return line.rstrip()
    return ""


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def _wheel_for_test() -> Path:
    """Locate or build the wheel for the current version.

    Honours the ``MRE_WHEEL_METADATA_TEST`` env-var: if the env-var is
    unset, the test tries to use a pre-built wheel from ``dist/``; if
    none exists, the test is skipped (avoiding the multi-second build
    cost in the default test run). When the env-var is set, the test
    builds the wheel from scratch.
    """
    version = _read_pyproject_version()
    wheel = _wheel_for_current_version(version)
    force_build = os.environ.get("MRE_WHEEL_METADATA_TEST") in ("1", "true", "yes")
    if wheel is None:
        if not force_build:
            pytest.skip(
                f"No dist/market_regime_engine-{version}-py3-none-any.whl found and "
                "MRE_WHEEL_METADATA_TEST is not set. Run `python -m build --wheel` "
                "or set the env-var to enable on-the-fly build in this test."
            )
        wheel = _build_wheel(version)
    return wheel


def test_wheel_metadata_version_matches_pyproject() -> None:
    wheel = _wheel_for_test()
    metadata = _read_wheel_metadata(wheel)
    headers, _ = _split_metadata(metadata)
    pyproject_version = _read_pyproject_version()
    wheel_version = headers.get("Version", "")
    assert wheel_version == pyproject_version, (
        f"wheel METADATA Version: {wheel_version!r} != pyproject [project] version "
        f"{pyproject_version!r}. The wheel was built against a different source tree."
    )


def test_wheel_metadata_description_first_line_matches_readme_h1() -> None:
    """The wheel's long-description body first line equals the README H1.

    The v1.4.0 reviewer caught that the wheel METADATA had
    ``Version: 1.4.0`` but the Description first line still read
    ``Market Regime Engine v1.2.1``. This pins the alignment.
    """
    wheel = _wheel_for_test()
    metadata = _read_wheel_metadata(wheel)
    headers, body = _split_metadata(metadata)
    expected = _read_readme_h1()
    first = _first_nonblank_line(body).lstrip("#").strip()
    assert first == expected, (
        f"wheel METADATA Description first non-blank line {first!r} != README H1 "
        f"{expected!r}. Either the README H1 is stale (regression of the v1.4.0 "
        "bug) or pyproject is using a different long-description source."
    )
    # Defensive: every release should at least surface ``Market Regime
    # Engine`` and the pyproject version somewhere in the first line.
    pyproject_version = _read_pyproject_version()
    assert pyproject_version in first, (
        f"wheel METADATA Description first line must contain the pyproject version "
        f"{pyproject_version!r}; got {first!r}."
    )
    assert "Market Regime Engine" in first, (
        f"wheel METADATA Description first line must contain 'Market Regime Engine'; got {first!r}."
    )
    # Sanity: the headers dict is consistent (so the split was honest).
    assert headers.get("Name", "").replace("_", "-") in {"market-regime-engine", "Market Regime Engine"}, headers
