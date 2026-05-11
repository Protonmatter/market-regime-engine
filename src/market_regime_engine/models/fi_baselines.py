# SPDX-License-Identifier: Apache-2.0
"""Fixed-Income baseline component shims for the model registry (PR-7 §J).

REVIEW.md §3.3 FLAG F-20 / plan §7 §J — the baseline-model registry
introduced by PR #9 (`models/registry.py`) was originally scoped to
the macro classifier zoo. The Fixed-Income RCIE adapter ships three
deterministic baseline scorers (credit regime, liquidity stress,
execution confidence) that need to be discoverable through the same
registry so the existing promotion / release-gate machinery can list
them by name.

These classes conform to the :class:`ForecastModel` Protocol surface
just enough to integrate with :func:`models.registry.model_cards`:

- ``model_name`` / ``output_type`` / ``model_card()`` return the
  governance-required metadata (component name, version, dependencies,
  reference to the live scorer module).
- ``fit`` / ``predict`` / ``get_params`` raise
  :class:`NotImplementedError` with a redirect message — production
  callers must use the live scorers in
  :mod:`market_regime_engine.fixed_income` directly. The model
  registry is metadata-only for FI components.

The v1.5.1+ roadmap can swap these stubs for fittable variants once
the FI training pipeline lands, without breaking any consumer that
queries ``available_models()`` today.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "FiCreditRegimeBaseline",
    "FiExecutionConfidenceBaseline",
    "FiLiquidityStressBaseline",
]


def _redirect_msg(component: str, scorer: str) -> str:
    return (
        f"FI baseline {component!r} is metadata-only in the model registry. "
        f"Use {scorer!r} in market_regime_engine.fixed_income directly to "
        "fit / score / persist."
    )


class _FiBaselineBase:
    """Common scaffold for the three FI baseline shims."""

    model_name: str = ""
    output_type: str = "fi_governance_score"
    family: str = "fixed_income"
    component_name: str = ""
    scorer_path: str = ""
    description: str = ""
    dependencies: tuple[str, ...] = ()

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
        **kwargs: Any,
    ) -> _FiBaselineBase:
        raise NotImplementedError(_redirect_msg(self.component_name, self.scorer_path))

    def predict(self, X: pd.DataFrame | np.ndarray, **kwargs: Any) -> pd.DataFrame:
        raise NotImplementedError(_redirect_msg(self.component_name, self.scorer_path))

    def get_params(self) -> dict[str, Any]:
        return {
            "component_name": self.component_name,
            "scorer_path": self.scorer_path,
            "family": self.family,
        }

    def model_card(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "output_type": self.output_type,
            "family": self.family,
            "description": self.description,
            "params": self.get_params(),
            "dependencies": list(self.dependencies),
            "scorer_path": self.scorer_path,
            "component_name": self.component_name,
        }


class FiCreditRegimeBaseline(_FiBaselineBase):
    """Deterministic composite credit-regime scorer (PR-3)."""

    model_name = "fi_credit_regime_baseline"
    component_name = "credit_regime"
    scorer_path = "market_regime_engine.fixed_income.score_credit_regime"
    description = (
        "Explainable composite credit-spread regime scorer (treasury curve + "
        "spreads + CDS + volatility + ETF dislocation) per AGENT.md PR-3."
    )


class FiLiquidityStressBaseline(_FiBaselineBase):
    """Deterministic per-scope liquidity-stress scorer (PR-4)."""

    model_name = "fi_liquidity_stress_baseline"
    component_name = "liquidity_stress"
    scorer_path = "market_regime_engine.fixed_income.score_liquidity_stress"
    description = (
        "Explainable liquidity-stress scorer with bid-ask, RFQ depth, "
        "Amihud illiquidity, and dealer response features per AGENT.md PR-4."
    )


class FiExecutionConfidenceBaseline(_FiBaselineBase):
    """Deterministic logistic execution-confidence scorer (PR-5)."""

    model_name = "fi_execution_confidence_baseline"
    component_name = "execution_confidence"
    scorer_path = "market_regime_engine.fixed_income.score_execution_confidence"
    description = (
        "Deterministic logistic execution-confidence baseline with "
        "fail-closed gating per AGENT.md PR-5 + INSTRUCTIONS.md §6.3."
    )
