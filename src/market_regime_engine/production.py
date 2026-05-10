# SPDX-License-Identifier: Apache-2.0
"""Production-mode guard rails for Market Regime Engine."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

_PRODUCTION_VALUES = {"prod", "production"}
_DEV_VALUES = {"", "dev", "development", "local", "test", "staging"}


@dataclass(frozen=True)
class ProductionCheckResult:
    """Structured result for runtime posture checks."""

    production: bool
    ok: bool
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    env: str = ""

    def raise_for_errors(self) -> None:
        if not self.ok:
            raise RuntimeError("Production readiness failed: " + "; ".join(self.errors))


def env_name(env: Mapping[str, str] | None = None) -> str:
    values = os.environ if env is None else env
    return values.get("MRE_ENV", "").strip().lower()


def is_production_env(env: Mapping[str, str] | None = None) -> bool:
    """Return True when the process is running in production mode."""

    return env_name(env) in _PRODUCTION_VALUES


def select_api_db_path(env: Mapping[str, str] | None = None, *, default: str = "data/mre.duckdb") -> str:
    """Resolve the API database path."""

    values = os.environ if env is None else env
    value = values.get("MRE_DB_PATH", "").strip()
    if value:
        return value
    if is_production_env(values):
        raise RuntimeError("MRE_DB_PATH is required when MRE_ENV=production")
    return default


def validate_production_settings(env: Mapping[str, str] | None = None) -> ProductionCheckResult:
    """Validate fail-closed production settings without mutating process state."""

    values = os.environ if env is None else env
    mre_env = env_name(values)
    production = mre_env in _PRODUCTION_VALUES
    errors: list[str] = []
    warnings: list[str] = []

    if mre_env not in _PRODUCTION_VALUES | _DEV_VALUES:
        warnings.append(f"unknown MRE_ENV={mre_env!r}; treating as non-production")

    if production:
        if not values.get("MRE_API_KEY", "").strip():
            errors.append("MRE_API_KEY is required in production")
        if not values.get("MRE_DB_PATH", "").strip():
            errors.append("MRE_DB_PATH is required in production")
        db_path = values.get("MRE_DB_PATH", "").strip()
        if db_path and Path(db_path).suffix.lower() == ".db":
            warnings.append("MRE_DB_PATH uses a .db suffix; DuckDB production deployments should prefer .duckdb")
        if values.get("MRE_LEGACY_API_ALLOW_UNAUTH", "").strip() == "1" and values.get(
            "MRE_ALLOW_LEGACY_API_IN_PRODUCTION", ""
        ).strip() != "1":
            errors.append("legacy unauthenticated API is not allowed in production")
        if values.get("MRE_CACHE_BACKEND", "").strip().lower() == "redis" and not values.get(
            "MRE_REDIS_URL", ""
        ).strip():
            errors.append("MRE_REDIS_URL is required when MRE_CACHE_BACKEND=redis in production")

    return ProductionCheckResult(
        production=production,
        ok=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        env=mre_env,
    )


def assert_production_ready(env: Mapping[str, str] | None = None) -> ProductionCheckResult:
    """Raise ``RuntimeError`` when production posture is unsafe."""

    result = validate_production_settings(env)
    result.raise_for_errors()
    return result


__all__ = [
    "ProductionCheckResult",
    "assert_production_ready",
    "env_name",
    "is_production_env",
    "select_api_db_path",
    "validate_production_settings",
]
