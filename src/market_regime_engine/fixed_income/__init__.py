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
from market_regime_engine.fixed_income.schema import FI_TABLE_NAMES, register as _register_fi_schema
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

# v1.5 (PR-2 task B): register the 13 FI warehouse tables with the
# storage registry on package import. ``register_tables`` is idempotent
# on name so a re-import is a no-op; the resulting warehouse therefore
# carries all 34 core + 13 FI tables (47 total) whenever
# ``market_regime_engine.fixed_income`` is imported before
# ``Warehouse(...)`` is instantiated.
_register_fi_schema()

__all__ = [
    "FI_TABLE_NAMES",
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
