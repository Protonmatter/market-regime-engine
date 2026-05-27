# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from market_regime_engine.fixed_income.api_cache import (
    _fi_cache_get_or_compute,
    _latest_liquidity_timestamp,
)
from market_regime_engine.fixed_income.api_middleware import _BODY_SIZE_CAP_BYTES, _resolve_rate_limit_spec
from market_regime_engine.fixed_income.api_schemas import (
    ExecutionConfidenceRequestModel,
    XProDecisionRequestModel,
    credit_regime_output_to_dict,
    execution_confidence_response_to_dict,
    liquidity_stress_output_to_dict,
    tca_regime_segment_to_dict,
)
from market_regime_engine.fixed_income.correlation import log_safe
from market_regime_engine.fixed_income.credit_spread_regime import (
    latest_credit_regime_score_identity,
)
from market_regime_engine.fixed_income.liquidity_stress import latest_liquidity_stress_score
from market_regime_engine.fixed_income.pit_guard import PitViolationError
from market_regime_engine.fixed_income.tca_segmentation import DIMENSION_COLUMNS, latest_tca_regime_segments

log = logging.getLogger(__name__)

_NOT_YET = "not_yet_implemented"
_HTTP_NOT_IMPLEMENTED = 501
_HTTP_NOT_FOUND = 404
_HTTP_SERVICE_UNAVAILABLE = 503
_HTTP_BAD_REQUEST = 400
_HTTP_PAYLOAD_TOO_LARGE = 413
_HTTP_RATE_LIMITED = 429
_VALID_SCOPE_TYPES: frozenset[str] = frozenset({"market", "sector", "rating", "cusip"})


def _stub_response(endpoint: str) -> JSONResponse:
    """Return the canonical PR-1 ``not_yet_implemented`` JSON response."""
    return JSONResponse(
        {"status": _NOT_YET, "endpoint": endpoint},
        status_code=_HTTP_NOT_IMPLEMENTED,
    )


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


def _xpro_write_context(warehouse: Any):
    """Return the write guard required for pooled DuckDB-backed warehouses."""

    try:
        from market_regime_engine.storage import is_pooled_warehouse, pooled_warehouse_write_lock
    except Exception:
        return nullcontext()
    try:
        if not is_pooled_warehouse(warehouse):
            return nullcontext()
    except Exception:
        return nullcontext()
    path = getattr(warehouse, "path", None)
    if path is None:
        return nullcontext()
    return pooled_warehouse_write_lock(path)


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
                    from market_regime_engine.fixed_income import api as api_surface

                    out = api_surface.latest_credit_regime_score(wh)
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
                from market_regime_engine.fixed_income import api as api_surface

                response = api_surface.score_execution_confidence(
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
                api_surface.write_execution_confidence_prediction(wh, response, request_id=body.request_id)
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

    @router.post("/xpro/decision", status_code=200)
    async def xpro_decision(body: XProDecisionRequestModel) -> JSONResponse:
        wh = factory()
        try:
            try:
                from market_regime_engine.fixed_income.xpro_decision import (
                    build_xpro_decision_artifact,
                )

                artifact = build_xpro_decision_artifact(
                    body.to_dataclass(),
                    warehouse=wh,
                    request_id=body.request_id,
                    decision_id=body.decision_id,
                    candidate_protocols=tuple(body.candidate_protocols),
                )
                with _xpro_write_context(wh):
                    wh.write_xpro_decision_artifact(artifact)
            except PitViolationError as exc:
                raise HTTPException(
                    status_code=_HTTP_BAD_REQUEST,
                    detail={
                        "detail": "pit_violation",
                        "message": str(exc),
                        "release_gate": False,
                    },
                ) from exc
            except Exception as exc:
                log.exception(
                    "xpro decision failed (request_id=%s)",
                    log_safe(body.request_id),
                )
                raise HTTPException(
                    status_code=_HTTP_SERVICE_UNAVAILABLE,
                    detail={"detail": "xpro_decision_failed", "release_gate": False},
                ) from exc
        finally:
            _close_if_not_pooled(wh)
        return JSONResponse(artifact, status_code=200)

    @router.get("/xpro/decision/{decision_id}", status_code=200)
    async def xpro_decision_get(decision_id: str) -> JSONResponse:
        wh = factory()
        try:
            try:
                latest = wh.latest_xpro_decision_artifact(decision_id)
            except Exception as exc:
                log.exception(
                    "xpro decision read failed (decision_id=%s): %s",
                    log_safe(decision_id),
                    exc,
                )
                return JSONResponse(
                    {"detail": "xpro_decision_read_failed", "release_gate": False},
                    status_code=_HTTP_SERVICE_UNAVAILABLE,
                )
        finally:
            _close_if_not_pooled(wh)
        if latest is None or latest.empty:
            return JSONResponse({"detail": "not_found"}, status_code=_HTTP_NOT_FOUND)
        try:
            artifact = json.loads(str(latest.iloc[0]["payload_json"]))
        except Exception as exc:
            raise HTTPException(
                status_code=_HTTP_SERVICE_UNAVAILABLE,
                detail={"detail": "xpro_decision_payload_invalid"},
            ) from exc
        return JSONResponse(artifact, status_code=200)

    @router.post("/xpro/decision/verify", status_code=200)
    async def xpro_decision_verify(artifact: dict[str, Any]) -> JSONResponse:
        from market_regime_engine.fixed_income.xpro_decision import verify_xpro_decision_artifact

        return JSONResponse(verify_xpro_decision_artifact(artifact), status_code=200)

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


__all__ = [
    "_close_if_not_pooled",
    "_resolve_db_path",
    "_stub_response",
    "_warehouse_factory_default",
    "build_router",
]
