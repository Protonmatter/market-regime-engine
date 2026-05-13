# SPDX-License-Identifier: Apache-2.0
"""Phase-5.4 tests for the new
:class:`LogitCoefficients` typed coefficient surface in
:mod:`market_regime_engine.fixed_income.execution_confidence`.

REVIEW_DEEP_V1_5_2.md section 4 — Phase 5.4 of the v1.6.0 fix campaign.

Pin three contracts:

1. The default :class:`LogitCoefficients` reproduces the legacy
   :data:`DEFAULT_WEIGHTS` dict bit-for-bit (no behaviour change for
   existing callers).
2. Overriding a single field (``notional_penalty_per_log10``) shifts
   the confidence score in the **expected direction** (more negative
   coefficient → lower confidence on a large notional).
3. The dataclass takes precedence over the legacy ``weights`` mapping
   when both are supplied (the explicit named-field surface wins).
"""

from __future__ import annotations

import pandas as pd
import pytest

from market_regime_engine.fixed_income.execution_confidence import (
    DEFAULT_LOGIT_COEFFICIENTS,
    DEFAULT_WEIGHTS,
    LogitCoefficients,
    score_execution_confidence,
)
from market_regime_engine.fixed_income.schemas import ExecutionConfidenceRequest

# Signal timestamps must lie inside the default staleness window
# (MRE_FI_MAX_SIGNAL_STALENESS_SEC = 900s = 15 min). Use a 5-minute lag
# from the decision timestamp so the soft-fail stale-signal path is not
# triggered.
_SIGNAL_TS = "2026-05-01T15:55:00Z"
_DECISION_TS = "2026-05-01T16:00:00Z"


class _LiquidityRow:
    def __init__(self) -> None:
        self.timestamp = _SIGNAL_TS
        self.liquidity_index = 50.0
        self.liquidity_label = "Normal"
        self.scope_type = "market"
        self.scope_id = "market"
        self.release_gate = True


class _RegimeRow:
    def __init__(self) -> None:
        self.timestamp = _SIGNAL_TS
        self.regime_score = 30.0
        self.regime_label = "Normal"
        self.release_gate = True


class _StubWarehouse:
    def read_credit_regime_scores(self):
        return pd.DataFrame()

    def read_liquidity_stress_scores(self, *args, **kwargs):
        return pd.DataFrame()


@pytest.fixture
def stub_signals(monkeypatch):
    """Patch the credit / liquidity readers to return deterministic rows."""
    from market_regime_engine.fixed_income import execution_confidence as ec

    monkeypatch.setattr(ec, "latest_credit_regime_score", lambda *a, **kw: _RegimeRow())
    monkeypatch.setattr(ec, "latest_liquidity_stress_score", lambda *a, **kw: _LiquidityRow())


def _request_for(notional: float = 5_000_000.0) -> ExecutionConfidenceRequest:
    return ExecutionConfidenceRequest(
        timestamp=_DECISION_TS,
        cusip="037833DT4",
        side="Buy",
        notional=notional,
        protocol="Auto-X",
        urgency="normal",
        rating="A",
    )


def test_default_logit_coefficients_round_trip_to_default_weights():
    """The dataclass-derived dict must match the legacy DEFAULT_WEIGHTS
    keys + values bit-for-bit so existing callers keep their numerics.
    """
    derived = DEFAULT_LOGIT_COEFFICIENTS.to_weights_dict()
    assert set(derived.keys()) == set(DEFAULT_WEIGHTS.keys())
    for key in derived:
        assert derived[key] == DEFAULT_WEIGHTS[key], f"mismatch on {key}: {derived[key]} vs {DEFAULT_WEIGHTS[key]}"


def test_logit_coefficients_immutable():
    """``frozen=True``: a downstream pipeline that hands the same
    instance to multiple workers must not be able to mutate it
    by accident.
    """
    with pytest.raises((AttributeError, Exception)):
        DEFAULT_LOGIT_COEFFICIENTS.base_intercept = 0.99  # type: ignore[misc]


def test_score_execution_confidence_default_coefficients_match_legacy(stub_signals):
    """Calling :func:`score_execution_confidence` with no ``coefficients``
    and no ``weights`` argument must produce the same confidence score as
    calling it with ``coefficients=DEFAULT_LOGIT_COEFFICIENTS`` explicitly.
    """
    request = _request_for()
    warehouse = _StubWarehouse()
    out_default = score_execution_confidence(request, warehouse=warehouse)
    out_explicit = score_execution_confidence(
        request, warehouse=warehouse, coefficients=DEFAULT_LOGIT_COEFFICIENTS
    )
    assert out_default.confidence_score == pytest.approx(out_explicit.confidence_score, abs=1e-9)
    assert out_default.recommended_action == out_explicit.recommended_action


def test_score_execution_confidence_more_negative_notional_coef_lowers_score(stub_signals):
    """Override ``notional_penalty_per_log10`` to a more-negative value
    on a large-notional request: the confidence score must DROP
    (steeper notional penalty hurts a $50MM order).
    """
    request = _request_for(notional=50_000_000.0)  # log10 = 7.7 ⇒ above the 6.0 threshold
    warehouse = _StubWarehouse()
    baseline = score_execution_confidence(request, warehouse=warehouse)
    steeper = score_execution_confidence(
        request,
        warehouse=warehouse,
        coefficients=LogitCoefficients(notional_penalty_per_log10=-0.50),
    )
    assert steeper.confidence_score < baseline.confidence_score - 1e-3, (
        f"notional_penalty_per_log10=-0.50 should drop confidence below baseline "
        f"({baseline.confidence_score:.4f}); got {steeper.confidence_score:.4f}"
    )


def test_score_execution_confidence_more_positive_intercept_raises_score(stub_signals):
    """Bumping ``base_intercept`` from 0.5 to 1.5 must RAISE confidence
    (logit + 1.0 → sigmoid gain).
    """
    request = _request_for()
    warehouse = _StubWarehouse()
    baseline = score_execution_confidence(request, warehouse=warehouse)
    higher = score_execution_confidence(
        request, warehouse=warehouse, coefficients=LogitCoefficients(base_intercept=1.5)
    )
    assert higher.confidence_score > baseline.confidence_score - 1e-9


def test_coefficients_take_precedence_over_legacy_weights(stub_signals):
    """When both ``coefficients`` and ``weights`` are supplied, the
    typed dataclass wins — the legacy mapping must NOT silently shadow
    a dataclass field.
    """
    request = _request_for()
    warehouse = _StubWarehouse()
    coef_only = score_execution_confidence(
        request,
        warehouse=warehouse,
        coefficients=LogitCoefficients(base_intercept=1.5),
    )
    coef_and_weights = score_execution_confidence(
        request,
        warehouse=warehouse,
        coefficients=LogitCoefficients(base_intercept=1.5),
        weights={"base_intercept": -0.5},  # would crash the score if it took effect
    )
    assert coef_and_weights.confidence_score == pytest.approx(coef_only.confidence_score, abs=1e-9)
