"""v1 hardened API for the Market Regime Engine.

v1.5 product-boundary hardening:

- Non-production defaults to ``data/mre.duckdb`` so the API and CLI share the
  DuckDB-primary posture.
- ``MRE_ENV=production`` fails closed at import time unless ``MRE_API_KEY`` and
  ``MRE_DB_PATH`` are explicitly set.
- Redis cache soft-degrade is kept for local/dev, but production refuses a
  half-configured Redis backend.
- API-key auth remains optional only outside production. In production it is a
  required runtime contract, not a suggestion taped to the side of the server.
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
from market_regime_engine.production import assert_production_ready, is_production_env, select_api_db_path
from market_regime_engine.storage import Warehouse

log = logging.getLogger(__name__)

# Fail fast before uvicorn starts serving. Local/dev imports remain unchanged.
_PRODUCTION_CHECK = assert_production_ready()


def _api_key_required() -> str | None:
    return os.getenv("MRE_API_KEY")


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = _api_key_required()
    if expected is None:
        if is_production_env():
            raise HTTPException(status_code=503, detail="MRE_API_KEY is required in production")
        return
    if not x_api_key:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")
    # ``hmac.compare_digest`` is constant-time; the naive ``==`` compare leaks
    # the prefix length of the expected key via timing.
    if not hmac.compare_digest(str(x_api_key), str(expected)):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def _db_path() -> str:
    return select_api_db_path(default="data/mre.duckdb")


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
    """Tiny FIFO TTL cache (no external dependency)."""

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
    """Shared cache backend for multi-worker uvicorn deployments."""

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
    """Choose a cache backend from ``MRE_CACHE_BACKEND`` / ``MRE_REDIS_URL``."""

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
    return {
        "status": "ok",
        "version": __version__,
        "production": _PRODUCTION_CHECK.production,
        "db_path": _db_path() if not _PRODUCTION_CHECK.production else "explicit",
    }


@app.get("/v1/health")
def v1_health() -> dict:
    """Public liveness probe.

    Only the latest release-gate decision is exposed; no row-level data leaks.
    """

    db = Warehouse(_db_path())
    try:
        gates = db.read_release_gates()
        latest_decision = "unknown"
        if not gates.empty:
            latest_decision = str(gates.iloc[-1]["decision"])
    finally:
        db.close()
    return {
        "status": "ok",
        "version": __version__,
        "release_gate": latest_decision,
        "production": _PRODUCTION_CHECK.production,
    }


@app.get("/v1/metrics", dependencies=[Depends(require_api_key)])
def v1_metrics() -> Response:
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
