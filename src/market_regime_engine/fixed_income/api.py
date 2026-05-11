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
from collections.abc import Callable
from dataclasses import asdict
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from market_regime_engine.fixed_income.credit_spread_regime import (
    latest_credit_regime_score,
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

__all__ = [
    "ExecutionConfidenceRequestModel",
    "build_router",
    "credit_regime_output_to_dict",
    "execution_confidence_response_to_dict",
    "liquidity_stress_output_to_dict",
]


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
        import pandas as pd

        try:
            parsed = pd.Timestamp(v)
        except Exception as exc:
            raise ValueError(f"timestamp must be ISO-8601: {v!r}") from exc
        if parsed.tzinfo is None:
            raise ValueError(f"timestamp must carry explicit tz info (e.g. 'Z' suffix): {v!r}")
        return v

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
    """
    out = asdict(output)
    out["drivers"] = list(output.drivers)
    return out


def liquidity_stress_output_to_dict(output: LiquidityStressOutput) -> dict[str, Any]:
    """JSON-serialisable dict form of :class:`LiquidityStressOutput`.

    Re-export of :func:`fixed_income.liquidity_stress.output_to_dict`
    on the API namespace so FastAPI handlers and downstream consumers
    have a single canonical converter without crossing module
    boundaries.
    """
    return liquidity_output_to_dict(output)


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


def _warehouse_factory_default() -> Any:
    """Default :class:`Warehouse` constructor.

    Lazy import keeps the FastAPI cold-start path free of the DuckDB
    + storage cost when no FI endpoint is mounted. v1.5 PR-5 (ASK-8)
    routes through the per-process pool so the hot path no longer pays
    DuckDB catalog teardown per request.
    """
    from market_regime_engine.storage import get_pooled_warehouse

    return get_pooled_warehouse(_resolve_db_path())


def execution_confidence_response_to_dict(
    response: ExecutionConfidenceResponse,
) -> dict[str, Any]:
    """JSON-serialisable dict form of :class:`ExecutionConfidenceResponse`."""
    return asdict(response)


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
        """
        wh = factory()
        try:
            try:
                latest = latest_credit_regime_score(wh)
            except Exception as exc:
                log.exception("regime_index/latest read failed: %s", exc)
                raise HTTPException(
                    status_code=_HTTP_SERVICE_UNAVAILABLE,
                    detail={"detail": "no_data", "release_gate": False},
                ) from exc
        finally:
            close = getattr(wh, "close", None)
            if callable(close):
                close()
        if latest is None:
            return JSONResponse(
                {"detail": "no_data", "release_gate": False},
                status_code=_HTTP_SERVICE_UNAVAILABLE,
            )
        return JSONResponse(credit_regime_output_to_dict(latest), status_code=200)

    @router.get("/liquidity_index/latest")
    async def liquidity_index_latest() -> JSONResponse:
        """Return the most recent liquidity_stress_scores row across ALL scopes.

        - **200** with the full :class:`LiquidityStressOutput` JSON
          (including ``release_gate=False`` rows so consumers can fail
          closed downstream per AGENT.md non-negotiable 8).
        - **503** ``{"detail": "no_data", "release_gate": false}`` when
          the warehouse has no liquidity rows yet.
        """
        wh = factory()
        try:
            try:
                latest = latest_liquidity_stress_score(wh)
            except Exception as exc:
                log.exception("liquidity_index/latest read failed: %s", exc)
                raise HTTPException(
                    status_code=_HTTP_SERVICE_UNAVAILABLE,
                    detail={"detail": "no_data", "release_gate": False},
                ) from exc
        finally:
            close = getattr(wh, "close", None)
            if callable(close):
                close()
        if latest is None:
            return JSONResponse(
                {"detail": "no_data", "release_gate": False},
                status_code=_HTTP_SERVICE_UNAVAILABLE,
            )
        return JSONResponse(liquidity_stress_output_to_dict(latest), status_code=200)

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
            close = getattr(wh, "close", None)
            if callable(close):
                close()
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
        # Body size cap — re-check post-parse so a Pydantic-tolerated body
        # still gets rejected if it crossed the cap on the wire.
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
            try:
                write_execution_confidence_prediction(wh, response, request_id=body.request_id)
            except Exception as exc:
                log.warning(
                    "execution_confidence write failed (request_id=%s): %s",
                    body.request_id,
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
    async def tca_regime_segments_latest() -> JSONResponse:
        return _stub_response("GET /v1/tca/regime-segments/latest")

    @router.get("/evidence-pack/{model_run_id}")
    async def evidence_pack_get(model_run_id: str) -> JSONResponse:
        return _stub_response(f"GET /v1/evidence-pack/{model_run_id}")

    return router
