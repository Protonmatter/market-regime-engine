# SPDX-License-Identifier: Apache-2.0
"""PR-7 §N (PR-13) — signal_age_seconds in every FI API response."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import market_regime_engine.fixed_income  # noqa: F401  - register FI tables
from market_regime_engine.fixed_income.api import build_router
from market_regime_engine.storage import Warehouse


def _seed_credit(wh: Warehouse) -> None:
    wh.write_credit_regime_score(
        pd.DataFrame(
            [
                {
                    "model_run_id": "run-r1",
                    "timestamp": "2026-05-08T16:00:00Z",
                    "regime_score": 47.0,
                    "regime_label": "Watch / Transition",
                    "confidence": 0.85,
                    "drivers_json": "[]",
                    "component_scores_json": "{}",
                    "release_gate": 1,
                    "artifact_hash": "sha256:" + "a" * 64,
                    "metadata_json": "{}",
                }
            ]
        )
    )


def _seed_liquidity(wh: Warehouse) -> None:
    wh.write_liquidity_stress_score(
        pd.DataFrame(
            [
                {
                    "model_run_id": "run-l1",
                    "scope_type": "market",
                    "scope_id": "ALL",
                    "timestamp": "2026-05-08T16:00:00Z",
                    "liquidity_score": 30.0,
                    "liquidity_label": "Mild Stress",
                    "confidence": 0.9,
                    "drivers_json": "[]",
                    "release_gate": 1,
                    "artifact_hash": "sha256:" + "b" * 64,
                    "metadata_json": "{}",
                }
            ]
        )
    )


def _build_app(wh: Warehouse) -> FastAPI:
    app = FastAPI()
    app.include_router(build_router(lambda: wh))
    return app


def test_credit_regime_response_includes_signal_age(tmp_path: Path) -> None:
    db_path = tmp_path / "sig-age-cr.duckdb"
    wh = Warehouse(str(db_path))
    try:
        _seed_credit(wh)
    finally:
        wh.close()
    wh2 = Warehouse(str(db_path))
    try:
        client = TestClient(_build_app(wh2))
        resp = client.get("/v1/regime_index/latest")
    finally:
        try:
            wh2.close()
        except Exception:
            pass
    assert resp.status_code == 200
    body = resp.json()
    assert "metadata" in body
    assert "signal_age_seconds" in body["metadata"]
    assert isinstance(body["metadata"]["signal_age_seconds"], (int, float))


def test_liquidity_stress_response_includes_signal_age(tmp_path: Path) -> None:
    db_path = tmp_path / "sig-age-liq.duckdb"
    wh = Warehouse(str(db_path))
    try:
        _seed_liquidity(wh)
    finally:
        wh.close()
    wh2 = Warehouse(str(db_path))
    try:
        client = TestClient(_build_app(wh2))
        resp = client.get("/v1/liquidity_index/latest")
    finally:
        try:
            wh2.close()
        except Exception:
            pass
    assert resp.status_code == 200
    body = resp.json()
    assert "metadata" in body
    assert "signal_age_seconds" in body["metadata"]


def test_execution_confidence_response_includes_signal_age() -> None:
    """PR-5 already embeds ``signal_age_seconds_*`` in the execution-
    confidence response metadata. Smoke-test that the canonical
    response converter still surfaces the staleness keys."""
    from market_regime_engine.fixed_income.api import (
        execution_confidence_response_to_dict,
    )
    from market_regime_engine.fixed_income.schemas import (
        ExecutionConfidenceResponse,
    )

    response = ExecutionConfidenceResponse(
        timestamp="2026-05-08T16:00:00Z",
        cusip="AAA111111",
        side="buy",
        notional=1_000_000.0,
        protocol="Auto-X",
        confidence_score=0.85,
        expected_slippage_bps=5.0,
        confidence_interval_low=0.75,
        confidence_interval_high=0.95,
        recommended_action="Auto-X allowed",
        human_review_required=False,
        model_run_id="run-e1",
        release_gate=True,
        artifact_hash="sha256:" + "c" * 64,
        metadata={"signal_age_seconds_credit_regime": 30.0},
    )
    payload = execution_confidence_response_to_dict(response)
    assert "metadata" in payload
    assert "signal_age_seconds_credit_regime" in payload["metadata"]


def test_signal_age_seconds_helper_returns_inf_for_none() -> None:
    from market_regime_engine.fixed_income.api import _signal_age_seconds_now

    assert _signal_age_seconds_now(None) == float("inf")


def test_signal_age_seconds_helper_uses_utc() -> None:
    from market_regime_engine.fixed_income.api import _signal_age_seconds_now

    # A timestamp 60 seconds in the past should give a positive age.
    now = pd.Timestamp.now(tz="UTC")
    sixty_s_ago = now - pd.Timedelta(seconds=60)
    age = _signal_age_seconds_now(
        sixty_s_ago.strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    assert age >= 50.0  # tolerance for clock skew between assertion and call
