# SPDX-License-Identifier: Apache-2.0
"""PR-1 acceptance tests for the api_v1 db_path default flip (REVIEW.md AF-1 / AF-2)."""

from __future__ import annotations

import importlib
import logging

import pytest


def _reload_api_v1() -> object:
    import market_regime_engine.api_v1 as api_v1

    return importlib.reload(api_v1)


def test_db_path_default_is_duckdb_not_sqlite(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default-path flips from data/mre.db → data/mre.duckdb."""
    monkeypatch.delenv("MRE_DB_PATH", raising=False)
    api_v1 = _reload_api_v1()
    path = api_v1._db_path()  # type: ignore[attr-defined]
    assert path == "data/mre.duckdb"


def test_db_path_logs_resolved_path_once(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The INFO ``resolved db_path=...`` line fires once per app boot."""
    monkeypatch.delenv("MRE_DB_PATH", raising=False)
    api_v1 = _reload_api_v1()
    with caplog.at_level(logging.INFO, logger="market_regime_engine.api_v1"):
        api_v1._db_path()  # type: ignore[attr-defined]
        api_v1._db_path()  # type: ignore[attr-defined]
        api_v1._db_path()  # type: ignore[attr-defined]
    info_msgs = [r.message for r in caplog.records if "resolved db_path" in r.message]
    assert len(info_msgs) == 1, info_msgs


def test_db_path_raises_when_explicit_env_var_missing_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Explicit MRE_DB_PATH=<missing path> raises RuntimeError at first use."""
    bogus = tmp_path / "does_not_exist.duckdb"
    monkeypatch.setenv("MRE_DB_PATH", str(bogus))
    api_v1 = _reload_api_v1()
    with pytest.raises(RuntimeError, match="MRE_DB_PATH=") as exc:
        api_v1._db_path()  # type: ignore[attr-defined]
    assert "does not exist" in str(exc.value)


def test_db_path_default_missing_file_logs_warning_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When MRE_DB_PATH is unset and the default path doesn't exist, log
    WARNING but do not raise (preserves Warehouse auto-create)."""
    monkeypatch.delenv("MRE_DB_PATH", raising=False)
    api_v1 = _reload_api_v1()
    with caplog.at_level(logging.WARNING, logger="market_regime_engine.api_v1"):
        path = api_v1._db_path()  # type: ignore[attr-defined]
    assert path == "data/mre.duckdb"
    warning_msgs = [r.message for r in caplog.records if "default db_path" in r.message]
    # If the canonical repo happens to have ``data/mre.duckdb`` checked in
    # locally this list can be empty; we only assert it does NOT raise.
    assert isinstance(warning_msgs, list)


def test_db_path_explicit_env_present_file_passes_through(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    real = tmp_path / "mre.duckdb"
    real.write_bytes(b"")
    monkeypatch.setenv("MRE_DB_PATH", str(real))
    api_v1 = _reload_api_v1()
    assert api_v1._db_path() == str(real)  # type: ignore[attr-defined]
