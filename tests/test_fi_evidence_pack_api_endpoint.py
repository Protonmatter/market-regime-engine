# SPDX-License-Identifier: Apache-2.0
"""``GET /v1/evidence-pack/{model_run_id}`` acceptance tests (PR-7 §D)."""

from __future__ import annotations

import base64
import json
import secrets
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import market_regime_engine.fixed_income  # noqa: F401  - register FI tables
from market_regime_engine.fixed_income.api import build_router
from market_regime_engine.fixed_income.evidence_pack import (
    build_evidence_pack,
    write_evidence_pack,
)
from market_regime_engine.storage import Warehouse


def _b64(n: int = 32) -> str:
    return base64.b64encode(secrets.token_bytes(n)).decode("ascii")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    for name in (
        "MRE_FI_HMAC_KEY_VERSIONS",
        "MRE_FI_HMAC_KEY",
        "MRE_FI_REQUIRE_HMAC",
        "MRE_ENV",
    ):
        monkeypatch.delenv(name, raising=False)
    yield


def _build_app(warehouse: Warehouse) -> FastAPI:
    app = FastAPI()
    app.include_router(build_router(lambda: warehouse))
    return app


def _persist_pack(
    warehouse: Warehouse,
    *,
    model_run_id: str,
    request_id: str,
    sign: bool | None = None,
):
    pack = build_evidence_pack(
        model_run_id=model_run_id,
        component_name="credit_regime",
        model_version="0.1.0",
        code_sha="abcdef0",
        model_hash="sha256:m",
        input_features_hash="sha256:in",
        output_hash="sha256:out",
        release_gate=True,
        data_vintages={"trace_trades": "2026-05-08T16:00:00Z"},
        timestamp="2026-05-08T16:00:00Z",
        python_version="3.13.4",
    )
    return write_evidence_pack(warehouse, pack, request_id=request_id, sign=sign)


def test_get_evidence_pack_returns_200_for_existing_run_id(tmp_path: Path) -> None:
    db_path = tmp_path / "ep-api-200.duckdb"
    wh = Warehouse(str(db_path))
    try:
        _persist_pack(wh, model_run_id="run-api-1", request_id="req-1")
    finally:
        wh.close()

    wh2 = Warehouse(str(db_path))
    try:
        client = TestClient(_build_app(wh2))
        resp = client.get("/v1/evidence-pack/run-api-1")
    finally:
        wh2.close()
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["model_run_id"] == "run-api-1"
    assert payload["component_name"] == "credit_regime"
    assert "data_vintages" in payload
    # Dev mode: HMAC signature is None.
    assert payload["hmac_signature"] is None


def test_get_evidence_pack_returns_404_for_unknown_run_id(tmp_path: Path) -> None:
    db_path = tmp_path / "ep-api-404.duckdb"
    Warehouse(str(db_path)).close()
    wh = Warehouse(str(db_path))
    try:
        client = TestClient(_build_app(wh))
        resp = client.get("/v1/evidence-pack/no-such-run")
    finally:
        wh.close()
    assert resp.status_code == 404
    payload = resp.json()
    assert payload["detail"] == "evidence_pack_not_found"


def test_get_evidence_pack_includes_hmac_signature_when_signed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MRE_FI_HMAC_KEY_VERSIONS", json.dumps({"v1": _b64()}))
    db_path = tmp_path / "ep-api-signed.duckdb"
    wh = Warehouse(str(db_path))
    try:
        _persist_pack(wh, model_run_id="run-signed", request_id="req-signed", sign=True)
    finally:
        wh.close()

    wh2 = Warehouse(str(db_path))
    try:
        client = TestClient(_build_app(wh2))
        resp = client.get("/v1/evidence-pack/run-signed")
    finally:
        wh2.close()
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["hmac_signature"] is not None
    assert payload["hmac_signature"].startswith("v1:")


def test_get_evidence_pack_returns_pack_metadata(tmp_path: Path) -> None:
    """Smoke: full pack JSON includes the canonical fields the contract
    requires (model_hash, output_hash, data_vintages, timestamp,
    python_version, release_gate)."""
    db_path = tmp_path / "ep-api-fields.duckdb"
    wh = Warehouse(str(db_path))
    try:
        _persist_pack(wh, model_run_id="run-fields", request_id="req-fields")
    finally:
        wh.close()
    wh2 = Warehouse(str(db_path))
    try:
        client = TestClient(_build_app(wh2))
        resp = client.get("/v1/evidence-pack/run-fields")
    finally:
        wh2.close()
    payload = resp.json()
    for key in (
        "model_run_id",
        "component_name",
        "model_version",
        "model_hash",
        "input_features_hash",
        "output_hash",
        "data_vintages",
        "release_gate",
        "python_version",
        "timestamp",
    ):
        assert key in payload, f"missing field: {key}"
