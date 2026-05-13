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


def _full_production_env() -> dict[str, str]:
    """Return a minimal production-mode env that satisfies the v1.6.0
    expanded production guard (REVIEW_DEEP_V1_5_2.md §3.14)."""
    return {
        "MRE_ENV": "production",
        "MRE_API_KEY": "secret",
        "MRE_DB_PATH": "data/mre.duckdb",
        # v1.6.0 production guard additions:
        "MRE_FI_HMAC_KEY_VERSIONS": "v1=" + ("a" * 64),
        "MRE_FI_RATE_LIMIT_ENABLED": "1",
    }


def test_production_accepts_explicit_key_and_db_path() -> None:
    # v1.6.0 (REVIEW_DEEP_V1_5_2.md §3.14): production guard now also
    # requires slowapi to be importable when MRE_FI_RATE_LIMIT_ENABLED
    # is truthy. Skip on dev boxes without the [security] extra.
    pytest.importorskip("slowapi")
    report = validate_production_settings(_full_production_env())
    assert report.ok is True, report.errors
    assert report.errors == ()


def test_production_rejects_legacy_unauth_api() -> None:
    env = _full_production_env()
    env["MRE_LEGACY_API_ALLOW_UNAUTH"] = "1"
    report = validate_production_settings(env)
    assert report.ok is False
    assert "legacy unauthenticated API is not allowed in production" in report.errors


def test_production_requires_hmac_key() -> None:
    """v1.6.0 (REVIEW_DEEP_V1_5_2.md §3.14): production must have an HMAC key."""
    env = _full_production_env()
    del env["MRE_FI_HMAC_KEY_VERSIONS"]
    report = validate_production_settings(env)
    assert report.ok is False
    assert any("HMAC" in e for e in report.errors)


def test_production_accepts_legacy_hmac_single_key() -> None:
    """v1.6.0 §3.14: the legacy MRE_FI_HMAC_KEY single-version env var
    is still accepted (back-compat with v1.5.x deployments that have
    not migrated to the versioned-keys envelope)."""
    pytest.importorskip("slowapi")
    env = _full_production_env()
    del env["MRE_FI_HMAC_KEY_VERSIONS"]
    env["MRE_FI_HMAC_KEY"] = "a" * 64
    report = validate_production_settings(env)
    assert report.ok is True, report.errors


def test_production_requires_rate_limit_enabled() -> None:
    """v1.6.0 (REVIEW_DEEP_V1_5_2.md §3.14): production must enable rate limiting."""
    env = _full_production_env()
    del env["MRE_FI_RATE_LIMIT_ENABLED"]
    report = validate_production_settings(env)
    assert report.ok is False
    assert any("MRE_FI_RATE_LIMIT_ENABLED" in e for e in report.errors)


def test_production_db_path_must_be_explicit() -> None:
    with pytest.raises(RuntimeError):
        select_api_db_path({"MRE_ENV": "production"})
