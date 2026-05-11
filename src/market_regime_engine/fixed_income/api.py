# SPDX-License-Identifier: Apache-2.0
"""Placeholder FastAPI router for the Fixed-Income v1.5 endpoints.

Per ``MRE_FIXED_INCOME_INSTRUCTIONS.md §7``: register the 6 FI routes
under ``/v1/...``. PR-1 lands the router with all 6 endpoints stubbed
to return HTTP 501 ``"not_yet_implemented"`` JSON; subsequent PRs
replace the stubs with real handlers and mount the router onto
``api_v1.app`` in the order:

- PR-3: ``GET /v1/regime_index/latest``
- PR-4: ``GET /v1/liquidity_index/latest`` and
  ``GET /v1/liquidity_index/{scope_type}/{scope_id}``
- PR-5: ``POST /v1/execution_confidence``
- PR-6: ``GET /v1/tca/regime-segments/latest``
- PR-7: ``GET /v1/evidence-pack/{model_run_id}``

The PR-1 router is importable without side-effects (no DB
connection, no metric registration, no warehouse touch). It is
intentionally *not* mounted on ``api_v1.app`` in PR-1; mounting
happens per-endpoint at the PR that ships the real handler so the
fail-closed behaviour does not regress (the 501 stubs would not
respect the X-API-Key dependency, for example).
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

_NOT_YET = "not_yet_implemented"
_HTTP_NOT_IMPLEMENTED = 501


def _stub_response(endpoint: str) -> JSONResponse:
    """Return the canonical PR-1 ``not_yet_implemented`` JSON response."""
    return JSONResponse(
        {"status": _NOT_YET, "endpoint": endpoint},
        status_code=_HTTP_NOT_IMPLEMENTED,
    )


def build_router() -> APIRouter:
    """Return the FI ``APIRouter`` carrying the 6 placeholder endpoints.

    Factory pattern (rather than a module-level ``router = APIRouter(...)``)
    so tests that need a fresh router per case do not collide on the
    cached object. PR-3+ replace each route's handler with the real
    implementation while keeping the same path + method.
    """
    router = APIRouter(prefix="/v1", tags=["fixed_income"])

    @router.get("/regime_index/latest")
    async def regime_index_latest() -> JSONResponse:
        return _stub_response("GET /v1/regime_index/latest")

    @router.get("/liquidity_index/latest")
    async def liquidity_index_latest() -> JSONResponse:
        return _stub_response("GET /v1/liquidity_index/latest")

    @router.get("/liquidity_index/{scope_type}/{scope_id}")
    async def liquidity_index_scoped(scope_type: str, scope_id: str) -> JSONResponse:
        return _stub_response(f"GET /v1/liquidity_index/{scope_type}/{scope_id}")

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


__all__ = ["build_router"]
