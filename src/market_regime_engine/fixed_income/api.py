# SPDX-License-Identifier: Apache-2.0
"""FastAPI router for the Fixed-Income v1.5 endpoints.

PR-1 shipped the router as 6 placeholder ``501 not_yet_implemented``
endpoints (deliberately not mounted on ``api_v1.app``). PR-3 lands the
first real handler — ``GET /v1/regime_index/latest`` — and mounts the
router on the existing FastAPI app so the new path is live alongside
the macro routes.

The other 5 endpoints remain ``501 not_yet_implemented``; each one is
replaced in the PR that ships its handler:

- PR-4: ``GET /v1/liquidity_index/latest`` and
  ``GET /v1/liquidity_index/{scope_type}/{scope_id}``
- PR-5: ``POST /v1/execution_confidence``
- PR-6: ``GET /v1/tca/regime-segments/latest``
- PR-7: ``GET /v1/evidence-pack/{model_run_id}``

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
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from market_regime_engine.fixed_income.credit_spread_regime import (
    latest_credit_regime_score,
)
from market_regime_engine.fixed_income.liquidity_stress import (
    latest_liquidity_stress_score,
)
from market_regime_engine.fixed_income.liquidity_stress import (
    output_to_dict as liquidity_output_to_dict,
)
from market_regime_engine.fixed_income.schemas import (
    CreditRegimeOutput,
    LiquidityStressOutput,
)

log = logging.getLogger(__name__)

_NOT_YET = "not_yet_implemented"
_HTTP_NOT_IMPLEMENTED = 501
_HTTP_NOT_FOUND = 404
_HTTP_SERVICE_UNAVAILABLE = 503

_VALID_SCOPE_TYPES: frozenset[str] = frozenset({"market", "sector", "rating", "cusip"})

__all__ = [
    "build_router",
    "credit_regime_output_to_dict",
    "liquidity_stress_output_to_dict",
]


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
    + storage cost when no FI endpoint is mounted.
    """
    from market_regime_engine.storage import Warehouse

    return Warehouse(_resolve_db_path())


def build_router(
    warehouse_factory: Callable[[], Any] | None = None,
) -> APIRouter:
    """Return the FI ``APIRouter``.

    ``warehouse_factory`` is intentionally injectable so tests can pass
    a temp-DuckDB-backed :class:`Warehouse`; production callers can
    rely on the default which resolves through ``MRE_DB_PATH``.
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
                latest = latest_liquidity_stress_score(
                    wh, scope_type=scope_type, scope_id=scope_id
                )
            except Exception as exc:
                log.exception(
                    "liquidity_index/%s/%s read failed: %s", scope_type, scope_id, exc
                )
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

    @router.post("/execution_confidence")
    async def execution_confidence() -> JSONResponse:
        return _stub_response("POST /v1/execution_confidence")

    @router.get("/tca/regime-segments/latest")
    async def tca_regime_segments_latest() -> JSONResponse:
        return _stub_response("GET /v1/tca/regime-segments/latest")

    @router.get("/evidence-pack/{model_run_id}")
    async def evidence_pack_get(model_run_id: str) -> JSONResponse:
        return _stub_response(f"GET /v1/evidence-pack/{model_run_id}")

    return router
