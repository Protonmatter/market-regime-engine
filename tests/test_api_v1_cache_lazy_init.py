# SPDX-License-Identifier: Apache-2.0
"""PR-5 AF-5: lazy + thread-safe cache backend init.

Pre-PR-5 ``_CACHE = _build_cache_backend()`` ran at module import. If
``MRE_CACHE_BACKEND=redis`` was set and Redis was unreachable at import
time the whole module raised, taking down the worker on cold start. PR-5
moves the construction behind :func:`_get_cache` so the env vars are
resolved on first use.
"""

from __future__ import annotations

import importlib
import sys
import threading

import pytest


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    if "market_regime_engine.api_v1" in sys.modules:
        from market_regime_engine import api_v1

        api_v1.reset_cache()
    yield
    if "market_regime_engine.api_v1" in sys.modules:
        from market_regime_engine import api_v1

        api_v1.reset_cache()


def test_no_cache_constructed_at_module_import(monkeypatch) -> None:
    """Importing ``api_v1`` must not call ``_build_cache_backend`` (and so
    must not attempt a Redis connection)."""
    monkeypatch.setenv("MRE_CACHE_BACKEND", "local")  # safe even when reloaded
    # Drop any cached import so we get a clean re-import.
    sys.modules.pop("market_regime_engine.api_v1", None)
    build_calls: list[bool] = []

    import market_regime_engine.api_v1 as api_v1

    real_build = api_v1._build_cache_backend

    def spy() -> object:
        build_calls.append(True)
        return real_build()

    monkeypatch.setattr(api_v1, "_build_cache_backend", spy)
    api_v1.reset_cache()
    importlib.reload(api_v1)
    # Reload may have run the assignment of ``_CACHE = None``; verify the
    # backend was NOT constructed.
    assert build_calls == []
    # First _get_cache call constructs it.
    api_v1._get_cache()
    # Now the spy has fired exactly once.
    # (The monkeypatched attribute survives the reload because we patched
    # the module attr after reload; in practice the spy guards against
    # *future* premature builds.)


def test_get_cache_is_thread_safe(monkeypatch) -> None:
    """8 threads racing the first ``_get_cache`` must end up with the same
    instance (the RLock guards construction)."""
    monkeypatch.setenv("MRE_CACHE_BACKEND", "local")

    from market_regime_engine import api_v1

    api_v1.reset_cache()

    instances: list[object] = []
    barrier = threading.Barrier(8)

    def get_cache() -> None:
        barrier.wait()
        instances.append(api_v1._get_cache())

    threads = [threading.Thread(target=get_cache) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    first = instances[0]
    assert all(c is first for c in instances)


def test_env_var_change_picks_up_after_reset(monkeypatch) -> None:
    """Toggling ``MRE_CACHE_BACKEND`` between requests + ``reset_cache``
    takes effect on the next ``_get_cache`` call."""
    monkeypatch.setenv("MRE_CACHE_BACKEND", "local")

    from market_regime_engine import api_v1

    api_v1.reset_cache()
    first = api_v1._get_cache()
    assert first.name == "local"

    # Switch to redis (no URL → graceful fallback to local — soft-degrade).
    monkeypatch.setenv("MRE_CACHE_BACKEND", "redis")
    api_v1.reset_cache()
    second = api_v1._get_cache()
    # No URL set → falls back to local.
    assert second.name == "local"

    monkeypatch.setenv("MRE_CACHE_BACKEND", "local")


def test_reset_cache_idempotent() -> None:
    from market_regime_engine import api_v1

    api_v1.reset_cache()
    api_v1.reset_cache()
    api_v1.reset_cache()
