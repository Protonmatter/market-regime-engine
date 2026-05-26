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

import contextlib
import hmac
import json
import logging
import os
import pickle  # nosec B403 - cache values are produced by this module only
import threading
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol

from fastapi import Depends, FastAPI, Header, HTTPException, Response

from market_regime_engine import __version__
from market_regime_engine.analogs import analog_summary
from market_regime_engine.explain import latest_explanation
from market_regime_engine.observability import (
    incr,
    prometheus_text,
    time_block,
)
from market_regime_engine.production import (
    assert_production_ready,
    is_production_env,
    select_api_db_path,
)
from market_regime_engine.storage import (
    Warehouse,
    close_pooled_warehouses,
    get_pooled_warehouse,
)

log = logging.getLogger(__name__)


# v1.6 PR-22: fail-closed production posture check at import time. When
# ``MRE_ENV=production`` is set but ``MRE_API_KEY`` / ``MRE_DB_PATH`` are
# missing (or the cache backend is misconfigured), ``assert_production_ready``
# raises ``RuntimeError`` so a misconfigured deploy never reaches the
# ``app = FastAPI(...)`` line. Outside production this is a no-op and
# returns a ``ProductionCheckResult`` with ``ok=True``.
_PRODUCTION_CHECK = assert_production_ready()


def _api_key_required() -> str | None:
    return os.getenv("MRE_API_KEY")


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = _api_key_required()
    if expected is None:
        # v1.6 PR-22: in production, fail-closed when no key is set so a
        # misconfigured deploy cannot accidentally serve unauthenticated
        # endpoints (the import-time ``assert_production_ready`` already
        # guards this at startup, but a key rotation that wipes the env
        # var without restarting the worker would otherwise leak open).
        if is_production_env():
            raise HTTPException(
                status_code=503,
                detail="MRE_API_KEY is required in production",
            )
        return
    if not x_api_key:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")
    # ``hmac.compare_digest`` is constant-time; the naive ``==`` compare leaks
    # the prefix length of the expected key via timing.
    if not hmac.compare_digest(str(x_api_key), str(expected)):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


_DB_PATH_LOGGED: bool = False


def _db_path() -> str:
    """Resolve the warehouse DB path with v1.5-aligned defaults.

    v1.5 (PR-1 AF-1 / P0 REVIEW.md section 3.1):

    - Default flips from ``data/mre.db`` (SQLite) to ``data/mre.duckdb``
      (DuckDB) to match the v1.4 CLI ``mre`` default. Pre-v1.5 deploys
      that left ``MRE_DB_PATH`` unset were serving the API from a stale
      SQLite while the CLI wrote to DuckDB.
    - First call logs INFO with the resolved path + whether the file
      exists (module-level ``_DB_PATH_LOGGED`` guard prevents spam).
    - If ``MRE_DB_PATH`` is *explicitly* set BUT the file is missing,
      raise ``RuntimeError`` at first use so a misconfigured deployment
      fails fast (per AF-2). Default-path (env unset) absence still
      degrades to a warning to preserve the existing auto-create
      behaviour of the underlying Warehouse.

    v1.6 PR-22: the default-resolution now routes through
    :func:`market_regime_engine.production.select_api_db_path`, which
    raises when ``MRE_ENV=production`` and ``MRE_DB_PATH`` is unset.
    That extra gate fires at import time via
    :func:`assert_production_ready` so a production deploy without
    ``MRE_DB_PATH`` is rejected long before the first request lands.
    """
    global _DB_PATH_LOGGED
    explicit = os.environ.get("MRE_DB_PATH")
    path = select_api_db_path(default="data/mre.duckdb")
    exists = os.path.exists(path)
    if not _DB_PATH_LOGGED:
        log.info("resolved db_path=%s exists=%s", path, exists)
        _DB_PATH_LOGGED = True
    if explicit and not exists:
        raise RuntimeError(f"MRE_DB_PATH={path} but file does not exist")
    if not explicit and not exists:
        log.warning(
            "default db_path=%s does not exist; warehouse will be auto-created on first write",
            path,
        )
    return path


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


# v1.5 PR-5 (AF-5 / ASK-10): the Redis cache historically used ``pickle``
# for serialization. Pickle gives an attacker who controls the Redis
# instance (or who can inject keys into a shared Redis) arbitrary-code-
# execution on the FastAPI worker. PR-5 switches the default to JSON; the
# pickle path is gated behind the explicit opt-in env var
# ``MRE_CACHE_ALLOW_PICKLE=1`` for callers who need to store non-JSON
# native objects.

_PICKLE_OPT_IN_ENV = "MRE_CACHE_ALLOW_PICKLE"


def _pickle_opt_in() -> bool:
    enabled = os.getenv(_PICKLE_OPT_IN_ENV) == "1"
    if enabled and is_production_env():
        raise RuntimeError(f"{_PICKLE_OPT_IN_ENV}=1 is forbidden in production")
    return enabled


def _serialize_cache_value(value: Any) -> bytes:
    """Encode ``value`` for storage in Redis.

    Default: ``json.dumps(value, default=str)``. ``default=str`` keeps the
    encoder forgiving for pandas Timestamp / numpy scalars without
    requiring callers to pre-coerce.

    Opt-in (``MRE_CACHE_ALLOW_PICKLE=1``): pickled. Use only when the
    cache value is not JSON-serialisable and the operator accepts the
    attack-surface trade-off.
    """
    if _pickle_opt_in():
        return pickle.dumps(value)
    return json.dumps(value, default=str, separators=(",", ":")).encode("utf-8")


def _deserialize_cache_value(blob: bytes) -> Any:
    """Decode the cache value written by :func:`_serialize_cache_value`."""
    if _pickle_opt_in():
        try:
            return pickle.loads(blob)  # nosec B301 - opt-in only
        except Exception:
            # Allow opt-in callers to read JSON entries written before the
            # env-var was set, so an in-place toggle does not invalidate
            # the cache.
            return json.loads(blob.decode("utf-8"))
    return json.loads(blob.decode("utf-8"))


class _RedisTTLCache:
    """Shared cache backend for multi-worker uvicorn deployments.

    Keys are scoped under ``mre:cache:`` so a Redis instance can be safely
    shared across services. v1.5 PR-5 (AF-5 / ASK-10): values are
    serialised through :func:`_serialize_cache_value` which defaults to
    JSON; pickle is gated behind ``MRE_CACHE_ALLOW_PICKLE=1``.
    """

    name = "redis"

    _KEY_PREFIX = "mre:cache:"

    def __init__(self, url: str) -> None:
        try:
            import redis
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
        # v1.6.0 (REVIEW_DEEP_V1_5_2.md §4.2): redis-py's stubs declare
        # ``Redis.get`` as returning ``Awaitable[Any] | Any`` because the
        # same client class supports both sync and async modes. We
        # constructed a sync client (``Redis.from_url`` without an
        # async pool) so ``raw`` is always ``bytes`` at runtime, but
        # narrow with an explicit ``isinstance`` so the type-checker
        # can prove it and so a future regression to an async client
        # surfaces here as a returned ``None`` rather than an obscure
        # downstream ``TypeError`` inside ``_deserialize_cache_value``.
        if not isinstance(raw, bytes):
            log.warning(
                "redis cache get returned non-bytes payload (type=%s); "
                "ignoring entry. Did the client switch to async mode?",
                type(raw).__name__,
            )
            return None
        try:
            return _deserialize_cache_value(raw)
        except Exception:  # pragma: no cover - corrupt entry
            return None

    def set(self, key: str, value: object) -> None:
        try:
            self.client.set(
                self._scoped(key),
                _serialize_cache_value(value),
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

    This lets a dev / staging deployment toggle the env var on/off
    without risking a cold-start crash if the Redis cluster is being
    rebooted.

    v1.6 PR-22: under ``MRE_ENV=production`` the soft-degrade path is
    disabled. A production worker that declared ``MRE_CACHE_BACKEND=redis``
    with no reachable URL must fail at startup rather than silently
    serve from the local in-process cache (which would mean per-worker
    cache divergence under uvicorn ``--workers``). The shared-cache
    invariant is part of the production contract.
    """
    backend = os.getenv("MRE_CACHE_BACKEND", "local").lower()
    if backend != "redis":
        return _LocalTTLCache()
    url = os.getenv("MRE_REDIS_URL", "")
    if not url:
        msg = "MRE_CACHE_BACKEND=redis but MRE_REDIS_URL is empty"
        if is_production_env():
            raise RuntimeError(msg)
        log.warning("%s; falling back to local cache", msg)
        return _LocalTTLCache()
    try:
        return _RedisTTLCache(url=url)
    except Exception as exc:
        if is_production_env():
            raise RuntimeError(f"redis cache unavailable in production: {exc}") from exc
        log.warning("redis cache unavailable (%s); falling back to local cache", exc)
        return _LocalTTLCache()


# Back-compat alias so the v1.1 ``_TTLCache`` import path still works
# (the public name was always private, but a few internal tests depend
# on it). The Redis backend uses the same protocol so a test that
# instantiates ``_TTLCache`` directly always gets the local backend.
_TTLCache = _LocalTTLCache

# v1.5 PR-5 (AF-5): the cache backend is lazily constructed on first use
# so that ``import market_regime_engine.api_v1`` does not attempt a Redis
# connection at module load. ``_CACHE`` is reset to ``None`` by
# ``reset_cache()`` (test helper) so env-var changes between requests
# take effect.
_CACHE: _CacheBackend | None = None
_CACHE_LOCK = threading.RLock()


def _get_cache() -> _CacheBackend:
    """Lazy, thread-safe accessor for the API cache backend.

    First call constructs via :func:`_build_cache_backend`; subsequent
    calls return the same instance. ``reset_cache()`` clears the slot so
    operators (and tests) can pick up an updated ``MRE_CACHE_BACKEND`` /
    ``MRE_REDIS_URL`` env var without restarting the worker.
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    with _CACHE_LOCK:
        if _CACHE is None:
            _CACHE = _build_cache_backend()
        return _CACHE


def reset_cache() -> None:
    """Drop the cache backend so the next ``_get_cache()`` re-resolves env vars."""
    global _CACHE
    with _CACHE_LOCK:
        _CACHE = None


def _read(name: str, fn: Callable[[Warehouse], object]) -> object:
    cache = _get_cache()
    cached = cache.get(name)
    if cached is not None:
        # v1.5 PR-8 (Tier-2 fix C-AUTO-4): route through module-level
        # ``incr`` so cache hits/misses mirror to OTel alongside the
        # legacy registry. ``metrics().incr(...)`` only writes to
        # _GLOBAL and silently bypasses the OTel meter.
        incr("mre_api_cache_hits_total", endpoint=name)
        return cached
    incr("mre_api_cache_misses_total", endpoint=name)
    # v1.5 PR-5 (ASK-8): pooled per-process Warehouse so the FastAPI hot
    # path no longer pays DuckDB catalog + WAL teardown per request. The
    # pool is closed on shutdown (see ``_lifespan`` below).
    db = get_pooled_warehouse(_db_path())
    with time_block("mre_api_read_seconds", endpoint=name):
        value = fn(db)
    cache.set(name, value)
    return value


@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Release every pooled :class:`Warehouse` on FastAPI shutdown.

    Avoids leaking DuckDB file handles when the app is replaced (e.g. a
    ``uvicorn --reload`` cycle). FastAPI 0.110+ deprecates
    ``on_event("shutdown")`` so PR-5 uses the lifespan context manager
    pattern.
    """
    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            close_pooled_warehouses()


app = FastAPI(title="Market Regime Engine v1", version=__version__, lifespan=_lifespan)


# v1.5 PR-7 §I: install correlation-id middleware + log filter so every
# request_id (X-Request-ID or generated UUID4) propagates through every
# log line emitted in the request handler. Log lines pick up the id
# automatically via the contextvars-backed CorrelationIdLogFilter.
try:
    from market_regime_engine.fixed_income.correlation import (
        CorrelationIdMiddleware,
        install_correlation_id_log_filter,
    )

    app.add_middleware(CorrelationIdMiddleware)
    install_correlation_id_log_filter()
except Exception as exc:  # pragma: no cover - defensive
    log.warning("could not install correlation-id middleware: %s", exc)


# v1.5 PR-8 (Tier-2 fix B-Ask-1, REVIEW.md): install the body-size
# ASGI middleware for ``/v1/execution_confidence`` so the 32 KB cap is
# enforced BEFORE the route handler runs. Pre-fix the cap lived inside
# the handler and was bypassed by chunked Transfer-Encoding requests
# (no Content-Length means the header check is a no-op). The ASGI
# middleware accumulates body bytes on ``receive`` and emits HTTP 413
# directly once the running total exceeds the cap.
try:
    from market_regime_engine.fixed_income.middleware import (
        install_max_body_size_middleware,
    )

    install_max_body_size_middleware(app)
except Exception as exc:  # pragma: no cover - defensive
    log.warning("could not install body-size middleware: %s", exc)


def _mount_fixed_income_router() -> None:
    """Mount the FI router on the v1 app (v1.5 PR-3).

    The router from PR-1 was deliberately not mounted while every
    handler still returned ``501 not_yet_implemented``. PR-3 ships
    the first real handler (``GET /v1/regime_index/latest``); PR-4
    lights up the liquidity endpoints; PR-5 lights up the
    ``POST /v1/execution_confidence`` endpoint with slowapi rate
    limiting + 32 KB body cap. Mounted via factory + late import so
    the FI subpackage is not eagerly loaded when an operator only
    needs the macro routes.
    """
    # v1.5.1 (PR-9 FIX 1): if the operator opts-in to the rate
    # limiter via ``MRE_FI_RATE_LIMIT_ENABLED=1`` we MUST raise a
    # RuntimeError BEFORE the FastAPI app finalises route binding when
    # slowapi is not installed. This block runs before the
    # ``try/except`` below so the guard is not swallowed by the
    # defensive logger.
    from market_regime_engine.fixed_income.api import assert_slowapi_available

    assert_slowapi_available()

    try:
        from market_regime_engine.fixed_income.api import (
            _build_rate_limiter,
        )
        from market_regime_engine.fixed_income.api import (
            build_router as _fi_build_router,
        )

        limiter = _build_rate_limiter()
        if limiter is not None and getattr(limiter, "uses_slowapi", True):
            # slowapi requires the limiter to be attached to ``app.state``
            # and its ``RateLimitExceeded`` exception handler to be
            # registered before the routes use it. PR-5 also enforces the
            # ``Retry-After: 1`` header per the plan spec.
            try:
                from fastapi import Request as _RLRequest
                from fastapi.responses import JSONResponse as _RLJSONResponse
                from slowapi.errors import RateLimitExceeded

                app.state.limiter = limiter

                async def _rate_limit_handler(
                    request: _RLRequest, exc: RateLimitExceeded
                ) -> _RLJSONResponse:
                    return _RLJSONResponse(
                        {"detail": f"rate limit exceeded: {exc.detail}"},
                        status_code=429,
                        headers={"Retry-After": "1"},
                    )

                app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
            except Exception as exc:  # pragma: no cover - defensive
                log.warning("slowapi exception handler setup failed: %s", exc)
                limiter = None
        app.include_router(_fi_build_router(limiter=limiter))
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("could not mount fixed_income router: %s", exc)


_mount_fixed_income_router()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/v1/health")
def v1_health() -> dict:
    """Public liveness probe.

    Intentionally *not* gated by ``require_api_key`` so load balancers and
    Kubernetes liveness probes can hit it without provisioning a key. Only
    the latest release-gate decision is exposed; no row-level data leaks.

    v1.5 PR-5 (ASK-8): reads from the per-process pooled Warehouse so
    liveness probes do not flap DuckDB catalog handles on the hot path.
    """
    db = get_pooled_warehouse(_db_path())
    gates = db.read_release_gates()
    latest_decision = "unknown"
    if not gates.empty:
        latest_decision = str(gates.iloc[-1]["decision"])
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
    "_deserialize_cache_value",
    "_get_cache",
    "_serialize_cache_value",
    "app",
    "require_api_key",
    "reset_cache",
]
