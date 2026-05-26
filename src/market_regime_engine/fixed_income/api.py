# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

"""Backward-compatible Fixed-Income API facade.

The router has been split into focused modules:

- :mod:`api_schemas` — Pydantic request models and response serializers.
- :mod:`api_handlers` — FastAPI route construction and endpoint handlers.
- :mod:`api_middleware` — body-size/rate-limit startup guards.
- :mod:`api_cache` — versioned per-process FI response cache.

This module preserves historical imports such as
``from market_regime_engine.fixed_income.api import build_router`` and keeps
monkeypatch-compatible scorer bindings for existing tests.
"""

from market_regime_engine.fixed_income.api_cache import (
    _fi_cache_get_or_compute,
    _latest_credit_regime_timestamp,
    _latest_liquidity_timestamp,
    _warehouse_identity,
    reset_fi_cache,
)
from market_regime_engine.fixed_income.api_handlers import (
    _close_if_not_pooled,
    _resolve_db_path,
    _stub_response,
    _warehouse_factory_default,
    build_router,
)
from market_regime_engine.fixed_income.api_middleware import (
    _BODY_SIZE_CAP_BYTES,
    _build_rate_limiter,
    _resolve_rate_limit_spec,
    assert_slowapi_available,
    rate_limit_enabled,
)
from market_regime_engine.fixed_income.api_schemas import (
    ExecutionConfidenceRequestModel,
    XProDecisionRequestModel,
    _signal_age_seconds_now,
    credit_regime_output_to_dict,
    execution_confidence_response_to_dict,
    liquidity_stress_output_to_dict,
    tca_regime_segment_to_dict,
)
from market_regime_engine.fixed_income.credit_spread_regime import (
    latest_credit_regime_score,
    latest_credit_regime_score_identity,
)
from market_regime_engine.fixed_income.execution_confidence import (
    score_execution_confidence,
    write_execution_confidence_prediction,
)

__all__ = [
    "ExecutionConfidenceRequestModel",
    "XProDecisionRequestModel",
    "assert_slowapi_available",
    "build_router",
    "credit_regime_output_to_dict",
    "execution_confidence_response_to_dict",
    "latest_credit_regime_score",
    "latest_credit_regime_score_identity",
    "liquidity_stress_output_to_dict",
    "rate_limit_enabled",
    "reset_fi_cache",
    "score_execution_confidence",
    "tca_regime_segment_to_dict",
    "write_execution_confidence_prediction",
]
