"""Identity-drift guard for the v1.2.1 release.

Three things must agree, otherwise CI lies and downstream tooling
(``importlib.metadata.version("market-regime-engine")`` consumers, the
release-zip script, ``api`` / ``api_v1`` ``version`` payload) silently
diverges:

- ``[project] version`` in ``pyproject.toml``
- ``__version__`` in ``src/market_regime_engine/__init__.py``
- the installed-package metadata version (when the package is importable)

The corresponding CI job (``version-sanity``) extends this with
``${{ github.ref_name }}`` agreement on tag pushes.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_pyproject_version() -> str:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    # Match the [project] table version (deliberately narrow regex so the
    # accidental "version" inside [build-system] requires never matches).
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


def _read_init_version() -> str:
    init = REPO_ROOT / "src" / "market_regime_engine" / "__init__.py"
    text = init.read_text(encoding="utf-8")
    m = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    if not m:
        raise AssertionError("__version__ not found in src/market_regime_engine/__init__.py")
    return m.group(1)


def test_pyproject_and_init_versions_agree() -> None:
    assert _read_pyproject_version() == _read_init_version(), (
        "pyproject [project] version and src/market_regime_engine/__init__.py "
        "__version__ have drifted apart. Bump both."
    )


def test_version_string_is_pep440_release_or_pre_release() -> None:
    """Catch typos like ``1.2.1-alpha`` (PEP 440 wants ``1.2.1a0``)."""
    pep440 = re.compile(
        r"^([1-9][0-9]*!)?"
        r"(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*"
        r"((a|b|rc)(0|[1-9][0-9]*))?"
        r"(\.post(0|[1-9][0-9]*))?"
        r"(\.dev(0|[1-9][0-9]*))?"
        r"(\+[a-z0-9]+(\.[a-z0-9]+)*)?$",
        re.IGNORECASE,
    )
    v = _read_pyproject_version()
    assert pep440.match(v), f"version {v!r} is not PEP 440 compliant"


def test_runtime_version_matches_pyproject() -> None:
    """The package's runtime ``__version__`` must match the file constant."""
    import market_regime_engine

    assert market_regime_engine.__version__ == _read_init_version()


def test_installed_metadata_version_matches_when_installed() -> None:
    """When the package is installed (editable or wheel), the metadata
    version must agree with the source ``__version__``. Skip when the
    package is not installed (source-only test runs)."""
    from importlib import metadata

    try:
        installed = metadata.version("market-regime-engine")
    except metadata.PackageNotFoundError:
        pytest.skip("market-regime-engine not installed; skipping metadata check")
    assert installed == _read_init_version(), (
        f"installed metadata version {installed!r} != source __version__ "
        f"{_read_init_version()!r}. Reinstall with `pip install -e .`."
    )


def test_api_modules_report_runtime_version(monkeypatch) -> None:
    """Smoke: both API surfaces must surface the runtime ``__version__``
    on the ``/health`` route, otherwise the v1.2.1 string never reaches
    callers. The legacy ``api`` mount requires the v1.2.1 unauth gate to
    be acknowledged."""
    import importlib
    import sys

    monkeypatch.setenv("MRE_LEGACY_API_ALLOW_UNAUTH", "1")
    sys.modules.pop("market_regime_engine.api", None)
    legacy_api = importlib.import_module("market_regime_engine.api")
    legacy_app = legacy_api.app

    from fastapi.testclient import TestClient

    from market_regime_engine import __version__ as runtime_version
    from market_regime_engine.api_v1 import app as v1_app

    legacy_resp = TestClient(legacy_app).get("/health").json()
    v1_resp = TestClient(v1_app).get("/health").json()
    assert legacy_resp["version"] == runtime_version
    assert v1_resp["version"] == runtime_version
