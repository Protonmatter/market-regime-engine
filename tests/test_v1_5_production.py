from __future__ import annotations

import pytest

from market_regime_engine.production import select_api_db_path, validate_production_settings


def test_dev_api_db_defaults_to_duckdb() -> None:
    assert select_api_db_path({}, default="data/mre.duckdb") == "data/mre.duckdb"


def test_production_requires_api_key_and_db_path() -> None:
    report = validate_production_settings({"MRE_ENV": "production"})
    assert report.production is True
    assert report.ok is False
    assert "MRE_API_KEY is required in production" in report.errors
    assert "MRE_DB_PATH is required in production" in report.errors


def test_production_accepts_explicit_key_and_db_path() -> None:
    report = validate_production_settings(
        {
            "MRE_ENV": "production",
            "MRE_API_KEY": "secret",
            "MRE_DB_PATH": "data/mre.duckdb",
        }
    )
    assert report.ok is True
    assert report.errors == ()


def test_production_rejects_legacy_unauth_api() -> None:
    report = validate_production_settings(
        {
            "MRE_ENV": "production",
            "MRE_API_KEY": "secret",
            "MRE_DB_PATH": "data/mre.duckdb",
            "MRE_LEGACY_API_ALLOW_UNAUTH": "1",
        }
    )
    assert report.ok is False
    assert "legacy unauthenticated API is not allowed in production" in report.errors


def test_production_db_path_must_be_explicit() -> None:
    with pytest.raises(RuntimeError):
        select_api_db_path({"MRE_ENV": "production"})
