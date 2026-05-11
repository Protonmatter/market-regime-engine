# SPDX-License-Identifier: Apache-2.0
"""PR-1 acceptance tests for MRE_BUILD_SHA / MRE_BUILD_DIRTY env overrides
(REVIEW.md AF-13 / ASK-12)."""

from __future__ import annotations

import pytest

from market_regime_engine.model_runs import _git_dirty, _git_revision


def test_mre_build_sha_env_var_overrides_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``MRE_BUILD_SHA`` is set, both short and long forms use the env."""
    monkeypatch.setenv("MRE_BUILD_SHA", "abcdef1234567890aaaaaaaa")
    assert _git_revision(short=True) == "abcdef1"
    assert _git_revision(short=False) == "abcdef1234567890aaaaaaaa"


def test_mre_build_sha_unset_falls_back_to_git_or_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MRE_BUILD_SHA", raising=False)
    out = _git_revision(short=True)
    # In a CI without git this is "unknown"; in dev it's a 7-char SHA.
    # Either way, the env-set branch was not used.
    assert out != "abcdef1"


def test_mre_build_dirty_env_var_overrides_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Truthy/falsy values for ``MRE_BUILD_DIRTY`` map to True/False."""
    monkeypatch.setenv("MRE_BUILD_DIRTY", "1")
    assert _git_dirty() is True
    monkeypatch.setenv("MRE_BUILD_DIRTY", "true")
    assert _git_dirty() is True
    monkeypatch.setenv("MRE_BUILD_DIRTY", "yes")
    assert _git_dirty() is True

    monkeypatch.setenv("MRE_BUILD_DIRTY", "0")
    assert _git_dirty() is False
    monkeypatch.setenv("MRE_BUILD_DIRTY", "false")
    assert _git_dirty() is False
    monkeypatch.setenv("MRE_BUILD_DIRTY", "no")
    assert _git_dirty() is False


def test_mre_build_dirty_unset_falls_back_to_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No env-set → subprocess path or False on failure (boolean result)."""
    monkeypatch.delenv("MRE_BUILD_DIRTY", raising=False)
    out = _git_dirty()
    assert isinstance(out, bool)


def test_mre_build_sha_empty_string_treated_as_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MRE_BUILD_SHA", "")
    # Empty string should NOT short-circuit the env path; we want the
    # subprocess fallback (or "unknown") so a misconfigured operator
    # does not silently emit empty SHAs.
    out = _git_revision(short=True)
    assert out != ""
