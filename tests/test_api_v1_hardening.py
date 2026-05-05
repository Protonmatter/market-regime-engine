"""Hardening regressions for the v1 API.

Covers:
- ``hmac.compare_digest`` is used for the API-key check.
- ``/v1/metrics`` requires the API key when ``MRE_API_KEY`` is set.
- ``/v1/health`` is **not** gated so load-balancer probes still work.
- ``_TTLCache`` is concurrency-safe under a thread storm.
"""

from __future__ import annotations

import threading


def _client(monkeypatch, *, api_key: str | None):
    """Build a fresh FastAPI ``TestClient`` with the env-var override applied."""
    if api_key is None:
        monkeypatch.delenv("MRE_API_KEY", raising=False)
    else:
        monkeypatch.setenv("MRE_API_KEY", api_key)
    from fastapi.testclient import TestClient

    from market_regime_engine.api_v1 import app

    return TestClient(app)


def test_metrics_requires_api_key_when_configured(monkeypatch) -> None:
    client = _client(monkeypatch, api_key="topsecret")
    bad = client.get("/v1/metrics")
    assert bad.status_code == 401, bad.text

    bad_header = client.get("/v1/metrics", headers={"X-API-Key": "wrong"})
    assert bad_header.status_code == 401

    good = client.get("/v1/metrics", headers={"X-API-Key": "topsecret"})
    assert good.status_code == 200


def test_health_endpoint_is_public_even_when_api_key_set(monkeypatch) -> None:
    client = _client(monkeypatch, api_key="topsecret")
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"


def test_require_api_key_uses_constant_time_compare() -> None:
    """Static-source guard against regressing the ``==`` compare. The check
    uses ``hmac.compare_digest`` so the function source must contain the
    sentinel call."""
    import inspect

    from market_regime_engine import api_v1

    src = inspect.getsource(api_v1.require_api_key)
    assert "hmac.compare_digest" in src, "require_api_key must use hmac.compare_digest, not == compare"


def test_ttl_cache_is_lock_protected_under_concurrent_writes() -> None:
    """Hammer ``set`` and ``get`` from many threads; the OrderedDict must not
    raise or lose its eviction invariant."""
    from market_regime_engine.api_v1 import _TTLCache

    cache = _TTLCache(max_entries=8)
    errors: list[BaseException] = []

    def worker(start: int) -> None:
        try:
            for i in range(200):
                cache.set(f"k{(start + i) % 32}", i)
                _ = cache.get(f"k{(start + i) % 32}")
        except BaseException as exc:  # pragma: no cover - defensive
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i * 17,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent access raised: {errors}"
    # The cache must respect its size cap even after the storm.
    assert len(cache._store) <= 8


def test_ttl_cache_has_threading_lock_attr() -> None:
    """Source-level guard that the lock attribute exists on the cache."""
    from market_regime_engine.api_v1 import _TTLCache

    cache = _TTLCache()
    assert hasattr(cache, "_lock")
    # Don't assert exact type; ``threading.Lock()`` returns a private wrapper.
    cache._lock.acquire()
    cache._lock.release()
