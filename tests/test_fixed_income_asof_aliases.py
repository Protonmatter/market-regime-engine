# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import market_regime_engine.fixed_income as fi
import market_regime_engine.fixed_income.credit_spread_regime as credit_spread_regime
import market_regime_engine.fixed_income.execution_confidence as execution_confidence
import market_regime_engine.fixed_income.liquidity_stress as liquidity_stress


def test_fixed_income_package_exports_explicit_asof_aliases() -> None:
    assert fi.latest_credit_regime_score_asof is fi.latest_credit_regime_score
    assert fi.latest_liquidity_stress_score_asof is fi.latest_liquidity_stress_score
    assert fi.latest_execution_confidence_prediction_asof is fi.latest_execution_confidence_prediction

    assert "latest_credit_regime_score_asof" in fi.__all__
    assert "latest_liquidity_stress_score_asof" in fi.__all__
    assert "latest_execution_confidence_prediction_asof" in fi.__all__


def test_fixed_income_submodules_expose_explicit_asof_aliases() -> None:
    assert credit_spread_regime.latest_credit_regime_score_asof is credit_spread_regime.latest_credit_regime_score
    assert liquidity_stress.latest_liquidity_stress_score_asof is liquidity_stress.latest_liquidity_stress_score
    assert (
        execution_confidence.latest_execution_confidence_prediction_asof
        is execution_confidence.latest_execution_confidence_prediction
    )

    assert "latest_credit_regime_score_asof" in credit_spread_regime.__all__
    assert "latest_liquidity_stress_score_asof" in liquidity_stress.__all__
    assert "latest_execution_confidence_prediction_asof" in execution_confidence.__all__
