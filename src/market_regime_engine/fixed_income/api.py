# SPDX-License-Identifier: Apache-2.0
"""FastAPI router for the Fixed-Income v1.5 endpoints.

PR-1 shipped the router as 6 placeholder ``501 not_yet_implemented``
endpoints (deliberately not mounted on ``api_v1.app``). PR-3 lands the
first real handler — ``GET /v1/regime_index/latest`` — and mounts the
router on the existing FastAPI app so the new path is live alongside
the macro routes.

PR-4: ``GET /v1/liquidity_index/latest`` and
``GET /v1/liquidity_index/{scope_type}/{scope_id}``.

PR-5 (this commit):

- ``POST /v1/execution_confidence`` is live with Pydantic v2 request
  validation, a 32 KB body cap (413 on oversize), and per-API-key rate
  limiting via slowapi (100 req/s default; configurable via
  ``MRE_FI_EXEC_CONF_RATE_LIMIT``). 429 carries ``Retry-After: 1``.
- Reads via the per-process pooled Warehouse from PR-5 task F so the
  hot path no longer pays DuckDB catalog teardown per request.

Remaining stubs are replaced in PR-6 / PR-7.

API contract for the regime endpoint:

- 200 with the full :class:`CreditRegimeOutput` JSON payload when a
  row exists (including rows where ``release_gate=False`` — consumers
  fail closed downstream so they MUST see the row).
- 503 ``{"detail": "no_data", "release_gate": false}`` when the
  ``credit_regime_scores`` table is empty.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from dataclasses import asdict
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from market_regime_engine.fixed_income.correlation import log_safe
from market_regime_engine.fixed_income.credit_spread_regime import (
    latest_credit_regime_score,
    latest_credit_regime_score_identity,
)
from market_regime_engine.fixed_income.execution_confidence import (
    score_execution_confidence,
    write_execution_confidence_prediction,
)
from market_regime_engine.fixed_income.liquidity_stress import (
    latest_liquidity_stress_score,
)
from market_regime_engine.fixed_income.liquidity_stress import (
    output_to_dict as liquidity_output_to_dict,
)
from market_regime_engine.fixed_income.pit_guard import PitViolationError
from market_regime_engine.fixed_income.schemas import (
    CreditRegimeOutput,
    ExecutionConfidenceRequest,
    ExecutionConfidenceResponse,
    LiquidityStressOutput,
    TcaRegimeSegment,
)
from market_regime_engine.fixed_income.tca_segmentation import (
    DIMENSION_COLUMNS,
    latest_tca_regime_segments,
)

log = logging.getLogger(__name__)

_NOT_YET = "not_yet_implemented"
_HTTP_NOT_IMPLEMENTED = 501
_HTTP_NOT_FOUND = 404
_HTTP_SERVICE_UNAVAILABLE = 503
_HTTP_BAD_REQUEST = 400
_HTTP_PAYLOAD_TOO_LARGE = 413
_HTTP_RATE_LIMITED = 429

_VALID_SCOPE_TYPES: frozenset[str] = frozenset({"market", "sector", "rating", "cusip"})

# v1.5 PR-5: cap the request body at 32 KB. A typical
# ``ExecutionConfidenceRequest`` JSON is < 1 KB, so 32 KB gives 30x
# headroom for metadata payloads while preventing a memory-exhaustion
# probe.
_BODY_SIZE_CAP_BYTES: int = 32 * 1024

_DEFAULT_RATE_LIMIT: str = "100/second"
_RATE_LIMIT_ENV: str = "MRE_FI_EXEC_CONF_RATE_LIMIT"

# v1.5.1 (PR-9 FIX 1): when the operator opts-in to the rate limiter
# via ``MRE_FI_RATE_LIMIT_ENABLED=1`` we MUST fail closed at startup
# if slowapi is not importable, rather than silently mounting an
# unlimited handler. The check runs before the FastAPI app binds the
# port; see :func:`assert_slowapi_available`.
_RATE_LIMIT_ENABLED_ENV: str = "MRE_FI_RATE_LIMIT_ENABLED"

__all__ = [
    "ExecutionConfidenceRequestModel",
    "assert_slowapi_available",
    "build_router",
    "credit_regime_output_to_dict",
    "execution_confidence_response_to_dict",
    "liquidity_stress_output_to_dict",
    "rate_limit_enabled",
    "tca_regime_segment_to_dict",
]


def rate_limit_enabled() -> bool:
    """Return True iff the operator opted-in to the slowapi rate limiter.

    v1.5.1 (PR-9 FIX 1): the rate limiter is gated on the
    ``MRE_FI_RATE_LIMIT_ENABLED`` env var. Accepted truthy values are
    ``"1"``, ``"true"``, ``"yes"`` (case-insensitive); any other value
    (including unset) leaves the limiter off. This is intentionally
    distinct from :data:`_RATE_LIMIT_ENV` which sets the limit *spec*
    (e.g. ``"100/second"``).
    """
    raw = os.getenv(_RATE_LIMIT_ENABLED_ENV, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def assert_slowapi_available() -> None:
    """Raise ``RuntimeError`` when the limiter is opted-in but slowapi is missing.

    v1.5.1 (PR-9 FIX 1): production callers set
    ``MRE_FI_RATE_LIMIT_ENABLED=1`` to require rate limiting on
    POST /v1/execution_confidence. If the import fails (e.g. the
    deployment forgot the ``[security]`` extra) we MUST raise at
    startup rather than silently mount an unlimited handler. Tests
    can monkeypatch ``sys.modules["slowapi"] = None`` to simulate the
    missing-dependency path.

    No-op when ``MRE_FI_RATE_LIMIT_ENABLED`` is unset / false.
    """
    if not rate_limit_enabled():
        return
    import importlib
    import sys

    cached = sys.modules.get("slowapi")
    if cached is None and "slowapi" in sys.modules:
        # Tests inject ``sys.modules["slowapi"] = None`` to force the
        # missing-dependency branch without uninstalling slowapi.
        raise RuntimeError(
            "slowapi required when MRE_FI_RATE_LIMIT_ENABLED=1; "
            "install with: pip install market-regime-engine[security]"
        )
    try:
        importlib.import_module("slowapi")
    except Exception as exc:
        raise RuntimeError(
            "slowapi required when MRE_FI_RATE_LIMIT_ENABLED=1; "
            "install with: pip install market-regime-engine[security]"
        ) from exc


# ---------------------------------------------------------------------------
# Pydantic v2 request model (PR-5 §C.1)
# ---------------------------------------------------------------------------


class ExecutionConfidenceRequestModel(BaseModel):
    """Pydantic v2 validation model for POST /v1/execution_confidence body.

    The dataclass :class:`ExecutionConfidenceRequest` is the internal
    contract; this Pydantic shim wraps it for the FastAPI request body so
    type errors at the boundary surface as 422 rather than slipping into
    the scorer as ``TypeError``."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    timestamp: str = Field(..., description="ISO-8601 UTC timestamp with explicit tz info")
    cusip: str = Field(..., min_length=8, max_length=12)
    side: Literal["buy", "sell"]
    notional: float = Field(..., gt=0, le=500_000_000.0)
    protocol: Literal["Auto-X", "RFQ", "Manual"]
    limit_price: float | None = Field(default=None, gt=0)
    urgency: Literal["low", "normal", "high"] = "normal"
    request_id: str = Field(..., min_length=1, max_length=128)
    sector: str | None = None
    rating: str | None = None
    maturity_bucket: str | None = None
    client_request_id: str | None = None
    metadata: dict[str, Any] | None = None

    @field_validator("timestamp")
    @classmethod
    def _ts_must_be_utc_iso8601(cls, v: str) -> str:
        """Coerce inbound timestamp to canonical UTC ``...Z`` form.

        v1.6.0 (REVIEW_DEEP_V1_5_2.md A9 / Finding §3.4): the
        v1.5.x validator accepted any tz-aware ISO-8601 string
        (``+05:30``, ``-08:00``, ``Z``) without normalisation, so
        the same logical instant submitted from different
        operator timezones produced different canonical bytes and
        therefore different artifact hashes. The validator now
        rewrites every accepted timestamp to ``YYYY-MM-DDTHH:MM:SS[.ffffff]Z`` (UTC, microseconds-when-present), so two requests
        for the same instant under different offsets produce
        byte-identical canonical payloads and therefore identical
        ``artifact_hash`` values.
        """
        import pandas as pd

        try:
            parsed = pd.Timestamp(v)
        except Exception as exc:
            raise ValueError(f"timestamp must be ISO-8601: {v!r}") from exc
        if parsed.tzinfo is None:
            raise ValueError(
                f"timestamp must carry explicit tz info (e.g. 'Z' suffix): {v!r}"
            )
        utc_ts = parsed.tz_convert("UTC")
        canonical = utc_ts.strftime("%Y-%m-%dT%H:%M:%S")
        if utc_ts.microsecond:
            canonical += f".{utc_ts.microsecond:06d}"
        canonical += "Z"
        return canonical

    @field_validator("cusip")
    @classmethod
    def _cusip_must_be_alphanumeric(cls, v: str) -> str:
        if not v.isalnum():
            raise ValueError(f"cusip must be alphanumeric: {v!r}")
        return v.upper()

    def to_dataclass(self) -> ExecutionConfidenceRequest:
        """Project the Pydantic model onto the internal dataclass."""
        return ExecutionConfidenceRequest(
            timestamp=self.timestamp,
            cusip=self.cusip,
            side=self.side,
            notional=float(self.notional),
            protocol=self.protocol,
            limit_price=float(self.limit_price) if self.limit_price is not None else None,
            urgency=self.urgency,
            sector=self.sector,
            rating=self.rating,
            maturity_bucket=self.maturity_bucket,
            client_request_id=self.client_request_id or self.request_id,
            metadata=dict(self.metadata or {}),
        )


def _stub_response(endpoint: str) -> JSONResponse:
    """Return the canonical PR-1 ``not_yet_implemented`` JSON response."""
    return JSONResponse(
        {"status": _NOT_YET, "endpoint": endpoint},
        status_code=_HTTP_NOT_IMPLEMENTED,
    )


def credit_regime_output_to_dict(output: CreditRegimeOutput) -> dict[str, Any]:
    """JSON-serialisable dict form of :class:`CreditRegimeOutput`.

    Drivers are exposed as a list (not a tuple) so ``json.dumps`` does
    not need ``default=str``. Mirrors the AGENT.md §6.1 output example
    exactly.

    PR-7 §N (PR-13): the response also exposes
    ``metadata.signal_age_seconds`` (computed against the current UTC
    clock) so Auto-X consumers can check the SLA without parsing the
    timestamp twice.
    """
    out = asdict(output)
    out["drivers"] = list(output.drivers)
    out.setdefault("metadata", {})
    out["metadata"].setdefault("signal_age_seconds", _signal_age_seconds_now(output.timestamp))
    return out


def _signal_age_seconds_now(ts: str | None) -> float:
    """Return seconds between ``ts`` (ISO-8601) and now (UTC).

    Returns ``float('inf')`` when ``ts`` is ``None`` so consumers that
    rely on the SLA gate (≤ MRE_FI_MAX_SIGNAL_STALENESS_SEC) trip
    automatically on a missing timestamp.
    """
    if ts is None:
        return float("inf")
    try:
        import pandas as pd

        parsed = pd.Timestamp(ts)
    except Exception:
        return float("inf")
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    now = pd.Timestamp.now(tz="UTC")
    return float((now - parsed).total_seconds())


def liquidity_stress_output_to_dict(output: LiquidityStressOutput) -> dict[str, Any]:
    """JSON-serialisable dict form of :class:`LiquidityStressOutput`.

    Re-export of :func:`fixed_income.liquidity_stress.output_to_dict`
    on the API namespace; PR-7 §N enriches the dict with
    ``metadata.signal_age_seconds`` so Auto-X consumers see the same
    staleness signal across all FI endpoints.
    """
    out = liquidity_output_to_dict(output)
    out.setdefault("metadata", {})
    out["metadata"].setdefault("signal_age_seconds", _signal_age_seconds_now(output.timestamp))
    return out


def _resolve_db_path() -> str:
    """Mirror ``api_v1._db_path`` defaulting so a vanilla install works.

    AF-1 fix from PR-1: defaults to ``data/mre.duckdb`` to match the
    CLI default. The FI router does NOT enforce
    ``MRE_DB_PATH``-must-exist (the macro endpoint does); for the FI
    path a missing DB returns 503 ``no_data`` rather than 500, so a
    fresh deployment can spin up before any FI run has landed.
    """
    explicit = os.environ.get("MRE_DB_PATH")
    return explicit if explicit else "data/mre.duckdb"


# ---------------------------------------------------------------------------
# v1.5 PR-7 §L — Versioned cache key (REVIEW.md §3.6 PR-8)
# ---------------------------------------------------------------------------
#
# The FI endpoints originally read fresh from DuckDB on every request.
# Per plan §7 §L: include the latest data timestamp in the cache key so
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


def _warehouse_factory_default() -> Any:
    """Default :class:`Warehouse` constructor.

    Lazy import keeps the FastAPI cold-start path free of the DuckDB
    + storage cost when no FI endpoint is mounted. v1.5 PR-5 (ASK-8)
    routes through the per-process pool so the hot path no longer pays
    DuckDB catalog teardown per request.
    """
    from market_regime_engine.storage import get_pooled_warehouse

    return get_pooled_warehouse(_resolve_db_path())


def _close_if_not_pooled(warehouse: Any) -> None:
    """Close ``warehouse`` only when it is not pool-owned.

    The default :func:`_warehouse_factory_default` returns the per-process
    pooled warehouse; closing it from a request handler leaves a dead
    instance in ``_POOLED_WAREHOUSES`` so the next request returns a
    closed DuckDB connection (the pool-poisoning bug surfaced in two
    independent audits, Tier-1 A1/B-Auto-1 in REVIEW.md).

    Pooled warehouses are released via the FastAPI lifespan shutdown
    hook (:func:`close_pooled_warehouses`); we only close in this code
    path for test factories that hand back a fresh, non-pooled instance.
    """
    close = getattr(warehouse, "close", None)
    if not callable(close):
        return
    try:
        from market_regime_engine.storage import is_pooled_warehouse
    except Exception:
        # Defensive: if storage cannot be imported (shouldn't happen
        # under normal operation), fall back to closing — preserves the
        # pre-fix behaviour and avoids leaking handles for test
        # factories.
        close()
        return
    if is_pooled_warehouse(warehouse):
        return
    close()


def execution_confidence_response_to_dict(
    response: ExecutionConfidenceResponse,
) -> dict[str, Any]:
    """JSON-serialisable dict form of :class:`ExecutionConfidenceResponse`."""
    return asdict(response)


def tca_regime_segment_to_dict(segment: TcaRegimeSegment) -> dict[str, Any]:
    """JSON-serialisable dict form of :class:`TcaRegimeSegment`.

    The dataclass already round-trips through ``asdict``; this wrapper
    coerces the ``timestamp`` to an ISO-8601 string with the Z suffix
    (mirrors the other FI output converters).
    """
    out = asdict(segment)
    ts = segment.timestamp
    out["timestamp"] = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
    return out


def _resolve_rate_limit_spec() -> str:
    raw = os.getenv(_RATE_LIMIT_ENV, "").strip()
    return raw if raw else _DEFAULT_RATE_LIMIT


def _build_rate_limiter() -> Any | None:
    """Construct a slowapi :class:`Limiter` keyed by API key.

    Returns ``None`` when slowapi is not installed; the POST handler
    then runs without rate limiting (the body cap + Pydantic validation
    still apply). This keeps the FI router optional-dependency-light
    so a vanilla install does not have to install slowapi.
    """
    try:
        from slowapi import Limiter
    except Exception:
        log.warning(
            "slowapi not installed; POST /v1/execution_confidence will not be rate-limited. "
            "Install with `pip install market-regime-engine[security]` to enable."
        )
        return None

    def _key_func(request: Request) -> str:
        # API-key-scoped: callers without an API key get one shared bucket.
        return request.headers.get("X-API-Key") or "anonymous"

    return Limiter(key_func=_key_func, default_limits=[_resolve_rate_limit_spec()])


def build_router(
    warehouse_factory: Callable[[], Any] | None = None,
    *,
    limiter: Any | None = None,
) -> APIRouter:
    """Return the FI ``APIRouter``.

    ``warehouse_factory`` is intentionally injectable so tests can pass
    a temp-DuckDB-backed :class:`Warehouse`; production callers can
    rely on the default which resolves through ``MRE_DB_PATH``.
    ``limiter`` is the slowapi rate limiter instance; ``None`` (default)
    leaves the POST handler unlimited which is intended for tests + dev
    environments. Production should pass the limiter constructed via
    :func:`_build_rate_limiter`.
    """

    router = APIRouter(prefix="/v1", tags=["fixed_income"])
    factory = warehouse_factory or _warehouse_factory_default

    @router.get("/regime_index/latest")
    async def regime_index_latest() -> JSONResponse:
        """Return the most recent credit-regime score.

        - **200** with the full :class:`CreditRegimeOutput` JSON
          (including ``model_run_id`` / ``release_gate`` /
          ``artifact_hash``) when at least one row exists.
        - **503** ``{"detail": "no_data", "release_gate": false}``
          when the warehouse has no credit regime rows yet.
        - Rows with ``release_gate=false`` are returned with that flag
          set so consumers can fail closed downstream (AGENT.md
          non-negotiable 8).

        v1.5 PR-7 §L: the response is cached per-process keyed on the
        latest ``credit_regime_scores.timestamp`` so a fresh score
        automatically invalidates the cache.
        """
        wh = factory()
        try:
            try:
                # v1.5 PR-8 (Tier-2 fix A-Q1): use the
                # ``(timestamp, model_run_id, artifact_hash)`` triple as
                # the cache version key so two writes with the same
                # canonical timestamp but different runs invalidate the
                # cache. Pre-fix the cache returned the FIRST run's
                # artifact silently on the second read.
                identity = latest_credit_regime_score_identity(wh)
                if identity is None:
                    return JSONResponse(
                        {"detail": "no_data", "release_gate": False},
                        status_code=_HTTP_SERVICE_UNAVAILABLE,
                    )

                def _compute() -> dict[str, Any]:
                    out = latest_credit_regime_score(wh)
                    if out is None:
                        return {"detail": "no_data", "release_gate": False, "_status": 503}
                    payload = credit_regime_output_to_dict(out)
                    payload["_status"] = 200
                    return payload

                payload = _fi_cache_get_or_compute(
                    endpoint="regime_index/latest",
                    warehouse=wh,
                    latest_ts=identity,
                    compute=_compute,
                )
            except Exception as exc:
                log.exception("regime_index/latest read failed: %s", exc)
                raise HTTPException(
                    status_code=_HTTP_SERVICE_UNAVAILABLE,
                    detail={"detail": "no_data", "release_gate": False},
                ) from exc
        finally:
            _close_if_not_pooled(wh)
        status = int(payload.pop("_status", 200))
        return JSONResponse(payload, status_code=status)

    @router.get("/liquidity_index/latest")
    async def liquidity_index_latest() -> JSONResponse:
        """Return the most recent liquidity_stress_scores row across ALL scopes.

        - **200** with the full :class:`LiquidityStressOutput` JSON
          (including ``release_gate=False`` rows so consumers can fail
          closed downstream per AGENT.md non-negotiable 8).
        - **503** ``{"detail": "no_data", "release_gate": false}`` when
          the warehouse has no liquidity rows yet.

        v1.5 PR-7 §L: response cached per-process keyed on the latest
        ``liquidity_stress_scores.timestamp``.
        """
        wh = factory()
        try:
            try:
                latest_ts = _latest_liquidity_timestamp(wh)
                if latest_ts is None:
                    return JSONResponse(
                        {"detail": "no_data", "release_gate": False},
                        status_code=_HTTP_SERVICE_UNAVAILABLE,
                    )

                def _compute() -> dict[str, Any]:
                    out = latest_liquidity_stress_score(wh)
                    if out is None:
                        return {"detail": "no_data", "release_gate": False, "_status": 503}
                    payload = liquidity_stress_output_to_dict(out)
                    payload["_status"] = 200
                    return payload

                payload = _fi_cache_get_or_compute(
                    endpoint="liquidity_index/latest",
                    warehouse=wh,
                    latest_ts=latest_ts,
                    compute=_compute,
                )
            except Exception as exc:
                log.exception("liquidity_index/latest read failed: %s", exc)
                raise HTTPException(
                    status_code=_HTTP_SERVICE_UNAVAILABLE,
                    detail={"detail": "no_data", "release_gate": False},
                ) from exc
        finally:
            _close_if_not_pooled(wh)
        status = int(payload.pop("_status", 200))
        return JSONResponse(payload, status_code=status)

    @router.get("/liquidity_index/{scope_type}/{scope_id}")
    async def liquidity_index_scoped(scope_type: str, scope_id: str) -> JSONResponse:
        """Return the most recent liquidity row for ``(scope_type, scope_id)``.

        - **404** when ``scope_type`` is not in
          ``{"market", "sector", "rating", "cusip"}``.
        - **503** ``{"detail": "no_data", "release_gate": false}`` when
          no row exists for the requested scope.
        - **200** with the full output payload otherwise. Rows where
          ``release_gate=false`` are still returned with the flag set
          so consumers can fail closed downstream.
        """
        if scope_type not in _VALID_SCOPE_TYPES:
            return JSONResponse(
                {
                    "detail": "invalid_scope_type",
                    "valid_scope_types": sorted(_VALID_SCOPE_TYPES),
                },
                status_code=_HTTP_NOT_FOUND,
            )
        wh = factory()
        try:
            try:
                latest = latest_liquidity_stress_score(wh, scope_type=scope_type, scope_id=scope_id)
            except Exception as exc:
                log.exception("liquidity_index/%s/%s read failed: %s", scope_type, scope_id, exc)
                raise HTTPException(
                    status_code=_HTTP_SERVICE_UNAVAILABLE,
                    detail={"detail": "no_data", "release_gate": False},
                ) from exc
        finally:
            _close_if_not_pooled(wh)
        if latest is None:
            return JSONResponse(
                {"detail": "no_data", "release_gate": False},
                status_code=_HTTP_SERVICE_UNAVAILABLE,
            )
        return JSONResponse(liquidity_stress_output_to_dict(latest), status_code=200)

    async def _execution_confidence_handler(
        request: Request,
        body: ExecutionConfidenceRequestModel,
    ) -> JSONResponse:
        """Score a single execution-confidence request.

        - 413 Payload Too Large when the body exceeds ``_BODY_SIZE_CAP_BYTES``
          (32 KB).
        - 422 (FastAPI auto) on Pydantic validation failure.
        - 503 ``{"detail": "no_data", "release_gate": false}`` when neither
          credit-regime nor liquidity signals exist yet.
        - 200 with the full :class:`ExecutionConfidenceResponse` JSON
          otherwise (including stale / fail-closed payloads — the consumer
          fails closed downstream on ``release_gate=false``).
        """
        # v1.5 PR-8 (Tier-2 fix B-Ask-1): the authoritative body-size
        # cap is enforced by ``MaxBodySizeMiddleware`` in
        # ``api_v1.app`` BEFORE this handler runs, so chunked
        # transfer-encoding cannot bypass it. The in-handler
        # Content-Length re-check below is kept as defense in depth
        # for direct router mounts (test rigs that mount build_router
        # on a fresh FastAPI without our middleware).
        cl_header = request.headers.get("content-length")
        if cl_header is not None:
            try:
                content_length = int(cl_header)
            except ValueError:
                content_length = 0
            if content_length > _BODY_SIZE_CAP_BYTES:
                raise HTTPException(
                    status_code=_HTTP_PAYLOAD_TOO_LARGE,
                    detail={
                        "detail": "request body exceeds 32 KB cap",
                        "limit_bytes": _BODY_SIZE_CAP_BYTES,
                    },
                )

        wh = factory()
        try:
            dataclass_req = body.to_dataclass()
            try:
                response = score_execution_confidence(
                    dataclass_req,
                    warehouse=wh,
                    release_gate=True,
                )
            except PitViolationError as exc:
                log.warning("execution_confidence PIT violation: %s", exc)
                raise HTTPException(
                    status_code=_HTTP_BAD_REQUEST,
                    detail={
                        "detail": "pit_violation",
                        "message": str(exc),
                        "release_gate": False,
                    },
                ) from exc
            except Exception as exc:
                # v1.6.0 (REVIEW_DEEP_V1_5_2.md A10 / Finding
                # §3.5): the v1.5.x handler caught only
                # PitViolationError, so any other exception inside
                # score_execution_confidence left ``response``
                # unbound and the subsequent JSONResponse(...)
                # raised UnboundLocalError, leaking a 500 with no
                # governance envelope. Map every unexpected scorer
                # failure to a stable 503 fail-closed shape so the
                # release-gate contract holds even on scorer bugs.
                log.exception(
                    "execution_confidence score failed (request_id=%s)",
                    log_safe(body.request_id),
                )
                raise HTTPException(
                    status_code=_HTTP_SERVICE_UNAVAILABLE,
                    detail={
                        "detail": "score_failed",
                        "release_gate": False,
                    },
                ) from exc
            try:
                write_execution_confidence_prediction(
                    wh, response, request_id=body.request_id
                )
            except Exception as exc:
                log.warning(
                    "execution_confidence write failed (request_id=%s): %s",
                    log_safe(body.request_id),
                    exc,
                )
        finally:
            # Pooled warehouse — do NOT close. Pre-PR-5 the per-request
            # ``Warehouse(...)`` had to be closed; the pool owns the
            # handle now.
            pass

        return JSONResponse(execution_confidence_response_to_dict(response), status_code=200)

    if limiter is not None:
        # Apply the slowapi limiter via the decorator pattern; the
        # ``limiter.limit(spec)`` decorator wraps the handler at
        # registration time so the throttler is invoked per request.
        rate_spec = _resolve_rate_limit_spec()
        decorated_handler = limiter.limit(rate_spec)(_execution_confidence_handler)
        router.post("/execution_confidence", status_code=200)(decorated_handler)
    else:
        router.post("/execution_confidence", status_code=200)(_execution_confidence_handler)

    @router.get("/tca/regime-segments/latest")
    async def tca_regime_segments_latest(
        dimensions: str | None = None,
        limit: int = 100,
    ) -> JSONResponse:
        """Return the most recent N ``tca_regime_segments`` rows.

        Query params:

        - ``dimensions`` — optional comma-separated list of segmentation
          dimensions (subset of ``DIMENSION_COLUMNS``). When supplied,
          a row qualifies only when every listed dimension column is
          non-sentinel for that row. When omitted, all rows qualify.
        - ``limit`` — max rows to return (default 100, clamped to
          ``[1, 1000]``).

        Returns 200 with ``{"segments": [...], "count": N}``; 503 with
        ``{"detail": "no_data"}`` when no rows match.
        """
        clamped_limit = max(1, min(1000, int(limit)))

        dim_list: list[str] | None = None
        if dimensions:
            dim_list = [d.strip() for d in dimensions.split(",") if d.strip()]
            invalid = [d for d in dim_list if d not in DIMENSION_COLUMNS]
            if invalid:
                return JSONResponse(
                    {
                        "detail": "invalid_dimensions",
                        "invalid_dimensions": invalid,
                        "valid_dimensions": sorted(DIMENSION_COLUMNS),
                    },
                    status_code=_HTTP_BAD_REQUEST,
                )

        wh = factory()
        try:
            try:
                segments = latest_tca_regime_segments(wh, dimensions=dim_list, limit=clamped_limit)
            except Exception as exc:
                log.exception("tca/regime-segments/latest read failed: %s", exc)
                raise HTTPException(
                    status_code=_HTTP_SERVICE_UNAVAILABLE,
                    detail={"detail": "no_data"},
                ) from exc
        finally:
            _close_if_not_pooled(wh)
        if not segments:
            return JSONResponse({"detail": "no_data"}, status_code=_HTTP_SERVICE_UNAVAILABLE)
        return JSONResponse(
            {
                "segments": [tca_regime_segment_to_dict(s) for s in segments],
                "count": len(segments),
            },
            status_code=200,
        )

    @router.get("/evidence-pack/{model_run_id}")
    async def evidence_pack_get(model_run_id: str) -> JSONResponse:
        """Return the most recent evidence pack for ``model_run_id``.

        - **200** with the full :class:`FixedIncomeEvidencePack` JSON
          (including ``hmac_signature`` so a downstream verifier can
          authenticate the pack independently).
        - **404** ``{"detail": "evidence_pack_not_found"}`` when no
          row matches.
        - **503** when the warehouse read fails.
        """
        from market_regime_engine.fixed_income.evidence_pack import (
            evidence_pack_to_dict,
            read_evidence_pack,
        )

        wh = factory()
        try:
            try:
                pack = read_evidence_pack(wh, model_run_id=model_run_id)
            except Exception as exc:
                log.exception("evidence-pack read failed: %s", exc)
                raise HTTPException(
                    status_code=_HTTP_SERVICE_UNAVAILABLE,
                    detail={"detail": "evidence_pack_read_failed"},
                ) from exc
        finally:
            _close_if_not_pooled(wh)
        if pack is None:
            return JSONResponse(
                {"detail": "evidence_pack_not_found", "model_run_id": model_run_id},
                status_code=_HTTP_NOT_FOUND,
            )
        return JSONResponse(evidence_pack_to_dict(pack), status_code=200)

    return router
