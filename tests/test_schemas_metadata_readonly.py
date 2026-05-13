# SPDX-License-Identifier: Apache-2.0
"""v1.6.0 — F2 / Finding §3.15 regression tests.

Pin the contract that the frozen FI dataclasses expose ``metadata`` as
a read-only view. Mutating the metadata mapping on a constructed
dataclass instance must raise :class:`TypeError` so the audit trail of
the canonical evidence pack cannot be silently rewritten.

The deep-review spec suggested ``types.MappingProxyType``. The shipped
implementation uses a ``dict`` subclass (``_ReadOnlyMetadata``) that
raises on mutation; this preserves compatibility with
``dataclasses.asdict``, ``json.dumps``, and other standard-library
tooling that expects a real ``dict`` instance. Both designs deliver
the same end-user contract — every mutation raises ``TypeError``.
"""

from __future__ import annotations

import copy

import pytest

from market_regime_engine.fixed_income.schemas import (
    CreditRegimeOutput,
    ExecutionConfidenceRequest,
    FixedIncomeEvidencePack,
    LiquidityStressOutput,
)


def _credit_output() -> CreditRegimeOutput:
    return CreditRegimeOutput(
        timestamp="2026-05-01T16:00:00Z",
        regime_score=50.0,
        regime_label="NORMAL_LIQUIDITY",
        confidence=0.8,
        drivers=(),
        component_scores={},
        model_run_id="m1",
        release_gate=True,
        artifact_hash="h",
        metadata={"a": 1, "nested": {"b": 2}},
    )


def _liquidity_output() -> LiquidityStressOutput:
    return LiquidityStressOutput(
        timestamp="2026-05-01T16:00:00Z",
        scope_type="cusip",
        scope_id="CUSIP1",
        liquidity_index=30.0,
        liquidity_label="NORMAL",
        confidence=0.8,
        drivers=(),
        model_run_id="m1",
        release_gate=True,
        artifact_hash="h",
        metadata={"x": 9},
    )


def _exec_request() -> ExecutionConfidenceRequest:
    return ExecutionConfidenceRequest(
        timestamp="2026-05-01T16:00:00Z",
        cusip="CUSIP1",
        side="buy",
        notional=1_000_000.0,
        protocol="Auto-X",
        metadata={"mid_price": 99.5},
    )


@pytest.mark.parametrize(
    "factory",
    [_credit_output, _liquidity_output, _exec_request],
)
def test_metadata_setitem_raises_type_error(factory) -> None:
    instance = factory()
    with pytest.raises(TypeError, match="read-only"):
        instance.metadata["x"] = 99  # type: ignore[index]


@pytest.mark.parametrize(
    "factory",
    [_credit_output, _liquidity_output, _exec_request],
)
def test_metadata_setdefault_raises_type_error(factory) -> None:
    instance = factory()
    with pytest.raises(TypeError, match="read-only"):
        instance.metadata.setdefault("k", "v")  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "factory",
    [_credit_output, _liquidity_output, _exec_request],
)
def test_metadata_delitem_raises_type_error(factory) -> None:
    instance = factory()
    with pytest.raises(TypeError, match="read-only"):
        del instance.metadata["a"]  # type: ignore[attr-defined]


def test_metadata_update_raises_type_error() -> None:
    out = _credit_output()
    with pytest.raises(TypeError, match="read-only"):
        out.metadata.update({"new": 1})  # type: ignore[union-attr]


def test_metadata_clear_raises_type_error() -> None:
    out = _credit_output()
    with pytest.raises(TypeError, match="read-only"):
        out.metadata.clear()  # type: ignore[union-attr]


def test_metadata_supports_reads() -> None:
    """Reads continue to work (it IS a Mapping, after all)."""
    out = _credit_output()
    assert out.metadata["a"] == 1
    assert "a" in out.metadata
    assert sorted(out.metadata.keys()) == ["a", "nested"]


def test_metadata_deepcopy_returns_plain_dict() -> None:
    """A deepcopy of the metadata is a *plain* dict so callers can
    mutate the copy freely without bumping into the read-only guard."""
    out = _credit_output()
    cloned = copy.deepcopy(out.metadata)
    assert isinstance(cloned, dict)
    # Plain dict — mutation must succeed.
    cloned["new"] = 42
    assert cloned["new"] == 42


def test_metadata_dict_constructor_unwraps_to_plain_dict() -> None:
    out = _credit_output()
    plain = dict(out.metadata)
    assert isinstance(plain, dict)
    plain["new"] = 42  # no TypeError


def test_evidence_pack_metadata_is_also_read_only() -> None:
    pack = FixedIncomeEvidencePack(
        model_run_id="m1",
        component_name="credit_spread_regime",
        model_version="v1.6.0",
        timestamp="2026-05-01T16:00:00Z",
        code_sha="abc",
        model_hash="mh",
        input_features_hash="ih",
        output_hash="oh",
        data_vintages={},
        validation_results={},
        release_gate=True,
        random_seeds={},
        python_version="3.13.4",
        lockfile_hash=None,
        hmac_signature=None,
        metadata={"audit": True},
    )
    with pytest.raises(TypeError, match="read-only"):
        pack.metadata["tampered"] = True  # type: ignore[index]


def test_response_to_dict_helpers_return_mutable_metadata() -> None:
    """The ``output_to_dict`` helpers in api.py / credit_spread_regime
    / liquidity_stress coerce the read-only view back to a plain dict
    so downstream callers can attach derived fields (signal_age_seconds
    etc.) without tripping the guard."""
    from market_regime_engine.fixed_income.api import (
        credit_regime_output_to_dict,
        liquidity_stress_output_to_dict,
    )

    cred = credit_regime_output_to_dict(_credit_output())
    assert isinstance(cred["metadata"], dict)
    # Mutation succeeds on the helper-coerced dict.
    cred["metadata"]["extra"] = "ok"

    liq = liquidity_stress_output_to_dict(_liquidity_output())
    assert isinstance(liq["metadata"], dict)
    liq["metadata"]["extra"] = "ok"
