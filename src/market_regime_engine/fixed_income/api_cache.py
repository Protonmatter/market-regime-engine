# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from market_regime_engine.fixed_income.credit_spread_regime import (
    latest_credit_regime_score_identity,
)

# a fresh score automatically invalidates the cache without waiting
# for TTL. Cache is per-process (no Redis fan-out) — the cross-worker
# OTel emit handles aggregation; this cache is purely a read-path
# accelerator inside one worker.

_FI_CACHE_LOCK = threading.RLock()
_FI_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}
# (endpoint, warehouse_id) -> (version_key, payload)
# v1.5 PR-8 (Tier-2 fix A-Q1): ``version_key`` is now opaque — for
# ``regime_index/latest`` it's the
# ``(timestamp, model_run_id, artifact_hash)`` triple so two writes
# with the same canonical timestamp but different runs invalidate the
# cache. Other endpoints continue to use a single timestamp string.


def _warehouse_identity(warehouse: Any) -> str:
    """Return a stable identity string for the cache key.

    Production: uses the resolved DuckDB / SQLite path so two FastAPI
    workers pointing at the same DB share the cache key. Tests that
    spawn ephemeral ``tmp_path`` warehouses get distinct keys
    automatically; the per-test fixture isolation is preserved.
    """
    path = getattr(warehouse, "path", None)
    if path is not None:
        return str(path)
    return f"id:{id(warehouse)}"


def _fi_cache_get_or_compute(
    *,
    endpoint: str,
    warehouse: Any,
    latest_ts: Any | None,
    compute: Callable[[], Any],
) -> Any:
    """Return cached payload when ``latest_ts`` matches; else compute + cache.

    ``latest_ts`` is the canonical version key. For
    ``regime_index/latest`` it's the
    ``(timestamp, model_run_id, artifact_hash)`` triple (v1.5 PR-8
    Tier-2 A-Q1: two writes with the same canonical timestamp but
    different runs MUST invalidate the cache, which a timestamp-only
    key fails to do). For the other FI endpoints it's still the
    ISO-8601 timestamp string. When the key advances, the previous
    cached entry is dropped — a fresh score invalidates the cache
    instantly per REVIEW.md §3.6 PR-8.
    """
    if latest_ts is None:
        # No data: always recompute (cheap for empty-warehouse path).
        return compute()
    cache_key = (endpoint, _warehouse_identity(warehouse))
    with _FI_CACHE_LOCK:
        cached = _FI_CACHE.get(cache_key)
        if cached is not None and cached[0] == latest_ts:
            return cached[1]
    value = compute()
    with _FI_CACHE_LOCK:
        _FI_CACHE[cache_key] = (latest_ts, value)
    return value


def reset_fi_cache() -> None:
    """Drop every FI cache entry (test helper / operator handle)."""
    with _FI_CACHE_LOCK:
        _FI_CACHE.clear()


def _latest_credit_regime_timestamp(warehouse: Any) -> str | None:
    """Legacy helper kept for back-compat with any external callers.

    v1.5 PR-8 (Tier-2 fix A-Q1): the FastAPI handler now uses
    :func:`latest_credit_regime_score_identity` so the cache key
    includes ``model_run_id`` and ``artifact_hash``; this helper is no
    longer called from the hot path but is preserved so the public
    module surface does not regress.
    """
    triple = latest_credit_regime_score_identity(warehouse)
    if triple is None:
        return None
    return triple[0]


def _latest_liquidity_timestamp(
    warehouse: Any,
    *,
    scope_type: str | None = None,
    scope_id: str | None = None,
) -> str | None:
    df = warehouse.read_liquidity_stress_scores()
    if df is None or df.empty:
        return None
    if scope_type is not None:
        df = df[df["scope_type"] == scope_type]
    if scope_id is not None:
        df = df[df["scope_id"] == scope_id]
    if df.empty:
        return None
    return str(df.iloc[-1]["timestamp"])


__all__ = [
    "_fi_cache_get_or_compute",
    "_latest_credit_regime_timestamp",
    "_latest_liquidity_timestamp",
    "_warehouse_identity",
    "reset_fi_cache",
]
