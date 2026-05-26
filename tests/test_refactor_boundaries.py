# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json

import pandas as pd
import pytest


def test_storage_facade_preserves_public_imports() -> None:
    from market_regime_engine.storage import TableSpec, Warehouse, registered_tables
    from market_regime_engine.storage_backends import _select_backend
    from market_regime_engine.storage_pool import close_pooled_warehouses
    from market_regime_engine.storage_registry import TableSpec as RegistryTableSpec

    assert TableSpec is RegistryTableSpec
    assert Warehouse.__module__.endswith("storage_repositories")
    assert callable(_select_backend)
    assert callable(close_pooled_warehouses)
    assert any(spec.name == "observations" for spec in registered_tables())


def test_cli_facade_preserves_parser_and_handlers() -> None:
    from market_regime_engine.cli import label_recessions_cmd, parser
    from market_regime_engine.cli_handlers import label_recessions_cmd as handler
    from market_regime_engine.cli_parser import parser as parser_builder

    assert label_recessions_cmd is handler
    assert parser is parser_builder
    ns = parser().parse_args(["release-gate", "--gate-boundary", "stable_core"])
    assert ns.gate_boundary == "stable_core"


def test_fixed_income_api_facade_preserves_router_and_schema() -> None:
    from market_regime_engine.fixed_income.api import ExecutionConfidenceRequestModel, build_router, reset_fi_cache
    from market_regime_engine.fixed_income.api_handlers import build_router as handler_router
    from market_regime_engine.fixed_income.api_schemas import ExecutionConfidenceRequestModel as SchemaModel

    assert build_router is handler_router
    assert ExecutionConfidenceRequestModel is SchemaModel
    assert callable(reset_fi_cache)


def _release_gate_inputs() -> dict:
    return {
        "confidence": pd.DataFrame(
            [{"date": "2026-01-01", "confidence": 0.90, "grade": "A"}]
        ),
        "drift": pd.DataFrame(
            [{"date": "2026-01-01", "feature_name": "x", "psi": 0.0, "status": "ok"}]
        ),
        "invalidation": pd.DataFrame(
            [{"date": "2026-01-01", "trigger": "none", "severity": "low", "status": "inactive"}]
        ),
        "promotion": pd.DataFrame(
            [{"date": "2026-01-01", "promoted": True, "mcs_evidence": "in_set"}]
        ),
        "profile": "default",
    }


def test_release_gate_stable_core_metadata_boundary() -> None:
    from market_regime_engine.release_gates import evaluate_release_gate

    out = evaluate_release_gate(**_release_gate_inputs(), gate_boundary="stable_core")
    metadata = json.loads(str(out.iloc[0]["metadata_json"]))
    assert metadata["package_boundary"] == "stable_core"
    assert metadata["production_eligible"] is True
    assert bool(out.iloc[0]["approved"]) is True


def test_release_gate_experimental_frontier_requires_explicit_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    from market_regime_engine.release_gates import evaluate_release_gate

    monkeypatch.delenv("MRE_ENABLE_EXPERIMENTAL_FRONTIER", raising=False)
    with pytest.raises(RuntimeError, match="experimental frontier path disabled"):
        evaluate_release_gate(**_release_gate_inputs(), gate_boundary="experimental_frontier")

    monkeypatch.setenv("MRE_ENABLE_EXPERIMENTAL_FRONTIER", "1")
    out = evaluate_release_gate(**_release_gate_inputs(), gate_boundary="experimental_frontier")
    metadata = json.loads(str(out.iloc[0]["metadata_json"]))
    assert metadata["package_boundary"] == "experimental_frontier"
    assert metadata["production_eligible"] is False
