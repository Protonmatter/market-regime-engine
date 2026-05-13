# SPDX-License-Identifier: Apache-2.0
"""v1.5.1 (PR-9 FIX 6) + v1.6.1: coverage for ``scripts/check_readme_version.py``.

The v1.6.1 patch tightens this script to also verify
``pyproject.toml``'s ``[project] version`` against ``__version__`` and
the README banner, after the v1.6.0 source-vs-tag identity mismatch
demonstrated that two-way checks (``__version__`` vs README) leave a
gap where the wheel METADATA can drift independently. The new strict
three-way contract is pinned by the tests below.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_readme_version.py"


def _run_script() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
    )


def _write_pyproject(repo: Path, version: str) -> None:
    """Write a minimal ``pyproject.toml`` with ``[project] version = "<version>"``.

    The script only reads the ``[project] version`` key, so a minimal
    table is enough to exercise the parser.
    """
    (repo / "pyproject.toml").write_text(
        '[build-system]\n'
        'requires = ["setuptools>=68", "wheel"]\n'
        'build-backend = "setuptools.build_meta"\n'
        '\n'
        '[project]\n'
        'name = "market-regime-engine"\n'
        f'version = "{version}"\n',
        encoding="utf-8",
    )


def _write_init(repo: Path, version: str) -> None:
    pkg = repo / "src" / "market_regime_engine"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text(
        f'__version__ = "{version}"\n', encoding="utf-8"
    )


def _write_readme(repo: Path, version: str) -> None:
    (repo / "README.md").write_text(
        f"# Market Regime Engine v{version}\n\nbody\n", encoding="utf-8"
    )


def _copy_script(repo: Path) -> Path:
    scripts = repo / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    target = scripts / "check_readme_version.py"
    target.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def _build_fake_repo(
    tmp_path: Path,
    *,
    pyproject: str,
    init: str,
    readme: str,
) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    _write_pyproject(repo, pyproject)
    _write_init(repo, init)
    _write_readme(repo, readme)
    _copy_script(repo)
    return repo


def _run_in(repo: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(repo / "scripts" / "check_readme_version.py")],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(repo),
    )


def test_check_readme_version_passes_on_current_repo() -> None:
    """The current pyproject, ``__version__`` and README banner must agree."""
    result = _run_script()
    assert result.returncode == 0, (
        f"check_readme_version exited {result.returncode}; stdout={result.stdout!r}; stderr={result.stderr!r}"
    )
    assert "OK" in result.stdout


def test_check_readme_version_detects_readme_drift(tmp_path: Path) -> None:
    """If the README banner diverges from the other two, exit non-zero."""
    repo = _build_fake_repo(
        tmp_path, pyproject="2.0.0", init="2.0.0", readme="1.5.0"
    )
    result = _run_in(repo)
    assert result.returncode == 1
    assert "DRIFT" in result.stderr
    # All three values must appear in the side-by-side diagnostic.
    assert "'2.0.0'" in result.stderr
    assert "'1.5.0'" in result.stderr


def test_check_readme_version_detects_init_drift(tmp_path: Path) -> None:
    """If ``__version__`` diverges from pyproject + README, exit non-zero."""
    repo = _build_fake_repo(
        tmp_path, pyproject="1.6.1", init="1.5.2", readme="1.6.1"
    )
    result = _run_in(repo)
    assert result.returncode == 1
    assert "DRIFT" in result.stderr
    assert "'1.6.1'" in result.stderr
    assert "'1.5.2'" in result.stderr


def test_check_readme_version_detects_pyproject_drift(tmp_path: Path) -> None:
    """v1.6.1 regression test: if ``pyproject.toml`` diverges from
    ``__version__`` and the README, the script must exit non-zero.

    This is the case the v1.6.0 release would have caught in CI if the
    guard had also been checking pyproject (it didn't, until v1.6.1).
    """
    repo = _build_fake_repo(
        tmp_path, pyproject="1.5.2", init="1.6.1", readme="1.6.1"
    )
    result = _run_in(repo)
    assert result.returncode == 1
    assert "DRIFT" in result.stderr
    # Pyproject mismatch must be visible in the side-by-side diagnostic.
    assert "pyproject.toml" in result.stderr
    assert "'1.5.2'" in result.stderr
    assert "'1.6.1'" in result.stderr


def test_check_readme_version_passes_on_three_way_match(tmp_path: Path) -> None:
    """All three sources at the same version must yield exit code 0."""
    repo = _build_fake_repo(
        tmp_path, pyproject="3.1.4", init="3.1.4", readme="3.1.4"
    )
    result = _run_in(repo)
    assert result.returncode == 0, (
        f"check_readme_version exited {result.returncode}; stdout={result.stdout!r}; stderr={result.stderr!r}"
    )
    assert "OK (3.1.4)" in result.stdout


def test_check_readme_version_rejects_old_patch_family_loophole(tmp_path: Path) -> None:
    """v1.6.1 contract change: the v1.5.1 patch-family loophole (README
    banner allowed to lag ``__version__`` within the same X.Y series)
    is gone. A README pinned at ``v1.5.0`` while ``__version__`` /
    pyproject are at ``1.5.1`` must now fail.
    """
    repo = _build_fake_repo(
        tmp_path, pyproject="1.5.1", init="1.5.1", readme="1.5.0"
    )
    result = _run_in(repo)
    assert result.returncode == 1
    assert "DRIFT" in result.stderr
