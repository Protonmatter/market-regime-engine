# SPDX-License-Identifier: Apache-2.0
"""Production-mode guard rails for Market Regime Engine."""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

_PRODUCTION_VALUES = {"prod", "production"}
_DEV_VALUES = {"", "dev", "development", "local", "test", "staging"}
_MIN_PRODUCTION_HMAC_KEY_BYTES = 32


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _decode_hmac_key_material(value: str) -> bytes:
    raw = value.strip()
    if not raw:
        return b""
    try:
        return base64.b64decode(raw, validate=True)
    except Exception:
        pass
    try:
        padded = raw + "=" * (-len(raw) % 4)
        return base64.b64decode(padded, validate=True)
    except Exception:
        return raw.encode("utf-8")


def _parse_hmac_key_versions(raw: str) -> dict[str, str]:
    """Parse production HMAC key env in JSON or legacy ``v1=...`` form."""
    text = raw.strip()
    if not text:
        return {}
    if text.startswith("{"):
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("MRE_FI_HMAC_KEY_VERSIONS must decode to an object")
        return {str(k): str(v) for k, v in parsed.items()}

    out: dict[str, str] = {}
    for part in text.replace(";", ",").split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("legacy HMAC key version entries must use version=value")
        version, value = item.split("=", 1)
        version = version.strip()
        if not version:
            raise ValueError("HMAC key version must be non-empty")
        out[version] = value.strip()
    return out


def _hmac_key_strength_errors(values: Mapping[str, str]) -> list[str]:
    errors: list[str] = []
    versions_raw = values.get("MRE_FI_HMAC_KEY_VERSIONS", "").strip()
    singleton_raw = values.get("MRE_FI_HMAC_KEY", "").strip()
    try:
        key_values = _parse_hmac_key_versions(versions_raw) if versions_raw else {}
    except Exception as exc:
        return [f"MRE_FI_HMAC_KEY_VERSIONS is invalid: {exc}"]

    if not key_values and singleton_raw:
        key_values = {"v1": singleton_raw}

    for version, key_value in key_values.items():
        decoded = _decode_hmac_key_material(key_value)
        if len(decoded) < _MIN_PRODUCTION_HMAC_KEY_BYTES:
            errors.append(
                f"HMAC key {version!r} decodes to {len(decoded)} bytes; "
                f"production requires at least {_MIN_PRODUCTION_HMAC_KEY_BYTES} bytes"
            )
    return errors


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
        if (
            values.get("MRE_LEGACY_API_ALLOW_UNAUTH", "").strip() == "1"
            and values.get("MRE_ALLOW_LEGACY_API_IN_PRODUCTION", "").strip() != "1"
        ):
            errors.append("legacy unauthenticated API is not allowed in production")
        if (
            values.get("MRE_CACHE_BACKEND", "").strip().lower() == "redis"
            and not values.get("MRE_REDIS_URL", "").strip()
        ):
            errors.append("MRE_REDIS_URL is required when MRE_CACHE_BACKEND=redis in production")
        if _truthy(values.get("MRE_CACHE_ALLOW_PICKLE")):
            errors.append("MRE_CACHE_ALLOW_PICKLE is forbidden in production")
        # v1.6.0 (REVIEW_DEEP_V1_5_2.md §3.2 / Finding §3.14):
        # production MUST require an HMAC key for the FI evidence
        # pack path. ``sign_pack`` cannot fall back to "unsigned"
        # in production without making the evidence layer non-
        # tamper-evident. Either ``MRE_FI_HMAC_KEY_VERSIONS``
        # (preferred) or the legacy ``MRE_FI_HMAC_KEY`` single-
        # version env var satisfies the contract.
        has_hmac_versions = bool(values.get("MRE_FI_HMAC_KEY_VERSIONS", "").strip())
        has_hmac_legacy = bool(values.get("MRE_FI_HMAC_KEY", "").strip())
        if not (has_hmac_versions or has_hmac_legacy):
            errors.append(
                "MRE_FI_HMAC_KEY_VERSIONS (or legacy MRE_FI_HMAC_KEY) "
                "is required in production for FI evidence-pack signing"
            )
        else:
            errors.extend(_hmac_key_strength_errors(values))
        # v1.6.0 (Finding §3.14): production must rate-limit the
        # FI endpoint AND the slowapi backend must be importable.
        # The FastAPI startup guard already raises when the env
        # var is set but slowapi is missing; this check ensures
        # the env var is actually set in the first place AND that
        # slowapi imports cleanly so the rate-limit fail-closed
        # contract holds end-to-end.
        rate_limit_raw = values.get("MRE_FI_RATE_LIMIT_ENABLED", "").strip().lower()
        if rate_limit_raw not in {"1", "true", "yes", "on"}:
            errors.append("MRE_FI_RATE_LIMIT_ENABLED must be set (1/true/yes/on) in production")
        else:
            try:
                import importlib

                importlib.import_module("slowapi")
            except ImportError:
                errors.append(
                    "slowapi must be importable when "
                    "MRE_FI_RATE_LIMIT_ENABLED is truthy in production; "
                    "install with `pip install "
                    "market-regime-engine[security]`"
                )

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
