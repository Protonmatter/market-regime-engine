# SPDX-License-Identifier: Apache-2.0
"""PR-7 §J / FLAG F-20 — FI baselines registered in models/registry.py."""

from __future__ import annotations

import pytest

from market_regime_engine.models.fi_baselines import (
    FiCreditRegimeBaseline,
    FiExecutionConfidenceBaseline,
    FiLiquidityStressBaseline,
)
from market_regime_engine.models.registry import (
    available_models,
    get_model_class,
    make_model,
    model_cards,
)


def test_fi_components_registered_in_models_registry() -> None:
    """The three FI baselines must appear in available_models()."""
    names = available_models()
    assert "fi_credit_regime_baseline" in names
    assert "fi_liquidity_stress_baseline" in names
    assert "fi_execution_confidence_baseline" in names


def test_fi_baselines_classes_resolve_via_registry() -> None:
    assert get_model_class("fi_credit_regime_baseline") is FiCreditRegimeBaseline
    assert get_model_class("fi_liquidity_stress_baseline") is FiLiquidityStressBaseline
    assert get_model_class("fi_execution_confidence_baseline") is FiExecutionConfidenceBaseline


def test_fi_baseline_model_cards_describe_each_component() -> None:
    cards = {card["model_name"]: card for card in model_cards()}
    for name in (
        "fi_credit_regime_baseline",
        "fi_liquidity_stress_baseline",
        "fi_execution_confidence_baseline",
    ):
        assert name in cards
        card = cards[name]
        assert card["family"] == "fixed_income"
        assert card["component_name"] in {
            "credit_regime",
            "liquidity_stress",
            "execution_confidence",
        }
        assert card["scorer_path"].startswith("market_regime_engine.fixed_income")


def test_fi_baseline_fit_redirects_to_live_scorer() -> None:
    """The registry entries are metadata-only — fit/predict raise."""
    model = make_model("fi_credit_regime_baseline")
    with pytest.raises(NotImplementedError, match="metadata-only"):
        model.fit(X=[[0.0]], y=[0.0])
    with pytest.raises(NotImplementedError, match="metadata-only"):
        model.predict([[0.0]])


def test_fi_baseline_aliases_normalise_via_normalize_model_name() -> None:
    from market_regime_engine.models.registry import normalize_model_name

    # Hyphen and underscore should both resolve to the canonical name
    # via the existing normaliser.
    assert (
        normalize_model_name("FI-credit-regime-baseline")
        == "fi_credit_regime_baseline"
    )
