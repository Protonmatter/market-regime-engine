# SPDX-License-Identifier: Apache-2.0
"""v1.5.1 (PR-9 FIX 6): coverage for ``scripts/check_readme_version.py``."""

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


def test_check_readme_version_passes_on_current_repo() -> None:
    """The current ``__version__`` and README banner must agree."""
    result = _run_script()
    assert result.returncode == 0, (
        f"check_readme_version exited {result.returncode}; stdout={result.stdout!r}; stderr={result.stderr!r}"
    )
    assert "OK" in result.stdout


def test_check_readme_version_detects_drift(tmp_path: Path) -> None:
    """If the README banner and __version__ diverge across major.minor,
    the script must exit non-zero and explain the drift on stderr.
    """
    # Build a fake "repo" with mismatched README and __init__.
    fake_repo = tmp_path / "repo"
    (fake_repo / "src" / "market_regime_engine").mkdir(parents=True)
    (fake_repo / "src" / "market_regime_engine" / "__init__.py").write_text('__version__ = "2.0.0"\n', encoding="utf-8")
    (fake_repo / "README.md").write_text("# Market Regime Engine v1.5.0\n\nbody\n", encoding="utf-8")
    (fake_repo / "scripts").mkdir(parents=True)
    # Copy the real script over so it resolves _REPO_ROOT from its
    # own location.
    (fake_repo / "scripts" / "check_readme_version.py").write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(fake_repo / "scripts" / "check_readme_version.py")],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(fake_repo),
    )
    assert result.returncode == 1
    assert "DRIFT" in result.stderr


def test_check_readme_version_accepts_patch_release_family(tmp_path: Path) -> None:
    """``__version__ = "1.5.0"`` paired with a banner of "1.5.0"
    must pass; the script also accepts patch-family agreement so a
    v1.5.1 patch release sharing the v1.5.0 banner does not block
    CI.
    """
    fake_repo = tmp_path / "repo"
    (fake_repo / "src" / "market_regime_engine").mkdir(parents=True)
    (fake_repo / "src" / "market_regime_engine" / "__init__.py").write_text('__version__ = "1.5.1"\n', encoding="utf-8")
    (fake_repo / "README.md").write_text("# Market Regime Engine v1.5.0\n\nbody\n", encoding="utf-8")
    (fake_repo / "scripts").mkdir(parents=True)
    (fake_repo / "scripts" / "check_readme_version.py").write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(fake_repo / "scripts" / "check_readme_version.py")],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(fake_repo),
    )
    assert result.returncode == 0
    assert "OK" in result.stdout
