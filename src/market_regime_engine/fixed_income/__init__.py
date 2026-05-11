# SPDX-License-Identifier: Apache-2.0
"""Fixed-Income RCIE / X-Pro Auto-X Adapter — v1.5 scaffolding.

Mission (per ``MRE_FIXED_INCOME_AGENT.md §"Mission"``): "Add fixed-income
RCIE / X-Pro Auto-X integration layer without destabilizing the existing
macro/regime engine. The repo already contains the core quantitative and
governance engine. Add fixed-income-specific modules, schemas, API
contracts, CLI commands, warehouse tables, tests, and documentation."

PR-1 (this commit) lands the package skeleton, data contracts, hashing,
PIT guard, posterior-mode enforcement, evidence-pack scaffolding, and
placeholder API/CLI surfaces. Subsequent PRs (per the v1.5 implementation
plan) fill in warehouse, scorers, execution-confidence, TCA segmentation,
HMAC signing, and reports.

Public surface (PR-1):

- Data contracts: :class:`CreditRegimeOutput`, :class:`LiquidityStressOutput`,
  :class:`ExecutionConfidenceRequest`, :class:`ExecutionConfidenceResponse`,
  :class:`FixedIncomeEvidencePack`.
- Label enums: :class:`RegimeLabel`, :class:`LiquidityLabel`,
  :class:`ExecutionRecommendation`.
- Posterior wrappers: :class:`PosteriorMode`, :class:`FilteredPosterior`,
  :class:`SmoothedPosterior`.
- Helpers: :func:`assert_pit_safe`, :func:`canonical_sha256`.
"""

from market_regime_engine.fixed_income.hashing import canonical_sha256
from market_regime_engine.fixed_income.pit_guard import assert_pit_safe
from market_regime_engine.fixed_income.posterior_mode import (
    FilteredPosterior,
    PosteriorMode,
    SmoothedPosterior,
)
from market_regime_engine.fixed_income.schemas import (
    CreditRegimeOutput,
    ExecutionConfidenceRequest,
    ExecutionConfidenceResponse,
    ExecutionRecommendation,
    FixedIncomeEvidencePack,
    LiquidityLabel,
    LiquidityStressOutput,
    RegimeLabel,
)

__all__ = [
    "CreditRegimeOutput",
    "ExecutionConfidenceRequest",
    "ExecutionConfidenceResponse",
    "ExecutionRecommendation",
    "FilteredPosterior",
    "FixedIncomeEvidencePack",
    "LiquidityLabel",
    "LiquidityStressOutput",
    "PosteriorMode",
    "RegimeLabel",
    "SmoothedPosterior",
    "assert_pit_safe",
    "canonical_sha256",
]
