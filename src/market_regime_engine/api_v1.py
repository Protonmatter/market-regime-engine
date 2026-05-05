"""v1 hardened API for the Market Regime Engine.

Improvements over the read-only :mod:`api` module:

- Versioned ``/v1/...`` routes so future incompatible changes can land
  without breaking clients.
- Optional API-key authentication via the ``MRE_API_KEY`` env var. When set,
  every endpoint *except* ``/v1/health`` requires a matching ``X-API-Key``
  header. The constant-time ``hmac.compare_digest`` is used so the auth
  check does not leak the expected key by timing side-channel.
- ``/v1/metrics`` returns Prometheus scrape text or the in-process snapshot
  when ``prometheus_client`` is missing. It honours the same ``MRE_API_KEY``
  dependency; ``/v1/health`` stays public so a load balancer can probe the
  engine even when keys are rotated.
- Read-through caching with a short TTL on the latest-* endpoints. Default 30
  seconds; tunable via ``MRE_CACHE_TTL`` seconds.
- Health endpoint now reports the version *and* the most recent successful
  release-gate decision, so a load balancer can take the engine out of
  rotation when the gate is held.

v1.3 (item M): the cache is now polymorphic. ``MRE_CACHE_BACKEND``
selects ``local`` (default; the historical process-local TTL cache) or
``redis`` (shared cache via ``MRE_REDIS_URL``). The Redis backend
soft-degrades to the local backend with a warning when ``redis`` isn't
installed or ``MRE_REDIS_URL`` is unreachable, so a misconfigured
deployment never serves stale data through a half-broken cache.

The legacy :mod:`api` module is kept intact for back-compat. New deployments
should mount ``api_v1.app``.
"""

from __future__ import annotations

import hmac
import logging
import os
import pickle  # nosec B403 - cache values are produced by this module only
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Protocol

from fastapi import Depends, FastAPI, Header, HTTPException, Response

from market_regime_engine import __version__
from market_regime_engine.analogs import analog_summary
from market_regime_engine.explain import latest_explanation
from market_regime_engine.observability import metrics, prometheus_text, time_block
from market_regime_engine.storage import Warehouse

log = logging.getLogger(__name__)


def _api_key_required() -> str | None:
    return os.getenv("MRE_API_KEY")


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = _api_key_required()
    if expected is None:
        return
    if not x_api_key:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")
    # ``hmac.compare_digest`` is constant-time; the naive ``==`` compare leaks
    # the prefix length of the expected key via timing.
    if not hmac.compare_digest(str(x_api_key), str(expected)):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def _db_path() -> str:
    return os.getenv("MRE_DB_PATH", "data/mre.db")


def _ttl_seconds() -> float:
    try:
        return float(os.getenv("MRE_CACHE_TTL", "30"))
    except ValueError:
        return 30.0


class _CacheBackend(Protocol):
    name: str

    def get(self, key: str) -> object | None: ...
    def set(self, key: str, value: object) -> None: ...


class _LocalTTLCache:
    """Tiny FIFO TTL cache (no external dependency).

    The cache is lock-protected so concurrent ``set`` and ``get`` calls under
    a thread-pool FastAPI deployment cannot corrupt the underlying
    ``OrderedDict``.
    """

    name = "local"

    def __init__(self, max_entries: int = 64) -> None:
        self.max_entries = max_entries
        self._store: OrderedDict[str, tuple[float, object]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> object | None:
        now = time.time()
        with self._lock:
            if key not in self._store:
                return None
            ts, value = self._store[key]
            if now - ts > _ttl_seconds():
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: object) -> None:
        with self._lock:
            self._store[key] = (time.time(), value)
            if len(self._store) > self.max_entries:
                self._store.popitem(last=False)


class _RedisTTLCache:
    """Shared cache backend for multi-worker uvicorn deployments.

    Keys are scoped under ``mre:cache:`` so a Redis instance can be
    safely shared across services. Values are pickled (the cache only
    stores objects produced by this module — never user-supplied
    payloads — so the pickle attack surface is zero).
    """

    name = "redis"

    _KEY_PREFIX = "mre:cache:"

    def __init__(self, url: str) -> None:
        try:
            import redis  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - import path
            raise RuntimeError(
                "redis backend requested but the redis package is not installed; "
                "install with `pip install market-regime-engine[redis]`."
            ) from exc
        self.client = redis.Redis.from_url(url, socket_timeout=2.0, socket_connect_timeout=2.0)
        # Probe once at construction time so a typo in MRE_REDIS_URL
        # surfaces immediately rather than silently swallowing every
        # request.
        self.client.ping()

    def _scoped(self, key: str) -> str:
        return f"{self._KEY_PREFIX}{key}"

    def get(self, key: str) -> object | None:
        try:
            raw = self.client.get(self._scoped(key))
        except Exception as exc:  # pragma: no cover - transport failure
            log.warning("redis cache get failed: %s", exc)
            return None
        if raw is None:
            return None
        try:
            return pickle.loads(raw)  # nosec B301 - trusted producer (this module)
        except Exception:  # pragma: no cover - corrupt entry
            return None

    def set(self, key: str, value: object) -> None:
        try:
            self.client.set(
                self._scoped(key),
                pickle.dumps(value),
                ex=int(_ttl_seconds()),
            )
        except Exception as exc:  # pragma: no cover - transport failure
            log.warning("redis cache set failed: %s", exc)


def _build_cache_backend() -> _CacheBackend:
    """Choose a cache backend from ``MRE_CACHE_BACKEND`` / ``MRE_REDIS_URL``.

    Soft-degrades to the local cache (with a warning) when:

    - ``MRE_CACHE_BACKEND`` is set to ``redis`` but the ``redis``
      package isn't installed.
    - ``MRE_REDIS_URL`` is unset or unreachable.

    This lets a deployment toggle the env var on/off without risking a
    cold-start crash if the Redis cluster is being rebooted.
    """
    backend = os.getenv("MRE_CACHE_BACKEND", "local").lower()
    if backend != "redis":
        return _LocalTTLCache()
    url = os.getenv("MRE_REDIS_URL", "")
    if not url:
        log.warning("MRE_CACHE_BACKEND=redis but MRE_REDIS_URL is empty; falling back to local cache")
        return _LocalTTLCache()
    try:
        return _RedisTTLCache(url=url)
    except Exception as exc:
        log.warning("redis cache unavailable (%s); falling back to local cache", exc)
        return _LocalTTLCache()


# Back-compat alias so the v1.1 ``_TTLCache`` import path still works
# (the public name was always private, but a few internal tests depend
# on it). The Redis backend uses the same protocol so a test that
# instantiates ``_TTLCache`` directly always gets the local backend.
_TTLCache = _LocalTTLCache

_CACHE: _CacheBackend = _build_cache_backend()


def _read(name: str, fn: Callable[[Warehouse], object]) -> object:
    cached = _CACHE.get(name)
    if cached is not None:
        metrics().incr("mre_api_cache_hits_total", endpoint=name)
        return cached
    metrics().incr("mre_api_cache_misses_total", endpoint=name)
    db = Warehouse(_db_path())
    try:
        with time_block("mre_api_read_seconds", endpoint=name):
            value = fn(db)
        _CACHE.set(name, value)
        return value
    finally:
        db.close()


app = FastAPI(title="Market Regime Engine v1", version=__version__)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/v1/health")
def v1_health() -> dict:
    """Public liveness probe.

    Intentionally *not* gated by ``require_api_key`` so load balancers and
    Kubernetes liveness probes can hit it without provisioning a key. Only
    the latest release-gate decision is exposed; no row-level data leaks.
    """
    db = Warehouse(_db_path())
    try:
        gates = db.read_release_gates()
        latest_decision = "unknown"
        if not gates.empty:
            latest_decision = str(gates.iloc[-1]["decision"])
    finally:
        db.close()
    return {"status": "ok", "version": __version__, "release_gate": latest_decision}


@app.get("/v1/metrics", dependencies=[Depends(require_api_key)])
def v1_metrics() -> Response:
    """Prometheus scrape endpoint.

    Gated behind the same ``MRE_API_KEY`` as the data routes. Operators that
    want to scrape from an unauthenticated Prometheus should expose this
    behind a sidecar that injects the header, or run with ``MRE_API_KEY``
    unset (the default) which leaves the endpoint open.
    """
    return Response(content=prometheus_text(), media_type="text/plain; version=0.0.4")


@app.get("/v1/regime/latest", dependencies=[Depends(require_api_key)])
def v1_regime_latest() -> dict:
    def _fn(db: Warehouse) -> dict:
        out = latest_explanation(db.read_regimes())
        if out.get("status"):
            raise HTTPException(status_code=404, detail="No regime scores found")
        return out

    return _read("regime_latest", _fn)  # type: ignore[return-value]


@app.get("/v1/model-outputs/latest", dependencies=[Depends(require_api_key)])
def v1_model_outputs_latest() -> dict:
    def _fn(db: Warehouse) -> dict:
        df = db.read_model_outputs()
        if df.empty:
            raise HTTPException(status_code=404, detail="No model outputs found")
        latest = df[df["date"] == df["date"].max()]
        return {"date": str(latest["date"].iloc[0]), "outputs": latest.to_dict(orient="records")}

    return _read("model_outputs_latest", _fn)  # type: ignore[return-value]


@app.get("/v1/calibrated-outputs/latest", dependencies=[Depends(require_api_key)])
def v1_calibrated_outputs_latest() -> dict:
    def _fn(db: Warehouse) -> dict:
        df = db.read_calibrated_outputs()
        if df.empty:
            raise HTTPException(status_code=404, detail="No calibrated outputs found")
        latest = df[df["date"] == df["date"].max()]
        return {"date": str(latest["date"].iloc[0]), "outputs": latest.to_dict(orient="records")}

    return _read("calibrated_outputs_latest", _fn)  # type: ignore[return-value]


@app.get("/v1/release-gate/latest", dependencies=[Depends(require_api_key)])
def v1_release_gate_latest() -> dict:
    def _fn(db: Warehouse) -> dict:
        df = db.read_release_gates()
        if df.empty:
            raise HTTPException(status_code=404, detail="No release gate found")
        return df.iloc[-1].to_dict()

    return _read("release_gate_latest", _fn)  # type: ignore[return-value]


@app.get("/v1/analogs/latest", dependencies=[Depends(require_api_key)])
def v1_analogs_latest() -> dict:
    def _fn(db: Warehouse) -> dict:
        df = db.read_historical_analogs()
        if df.empty:
            raise HTTPException(status_code=404, detail="No analogs found")
        latest_date = df["as_of_date"].max()
        latest = df[df["as_of_date"] == latest_date]
        return {
            "as_of_date": latest_date,
            "summary": analog_summary(latest),
            "analogs": latest.to_dict(orient="records"),
        }

    return _read("analogs_latest", _fn)  # type: ignore[return-value]


__all__ = [
    "_LocalTTLCache",
    "_RedisTTLCache",
    "_TTLCache",
    "_build_cache_backend",
    "app",
    "require_api_key",
]
